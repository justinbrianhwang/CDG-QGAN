"""Two candidate fixes for the finding that WGAN-GP learns no dependency at all.

The problem (RESULTS_lr.md)
---------------------------
Across a 1000x sweep of the quantum learning rate, the true-edge error never leaves the
floor. At lr_q=5e-2 the angles move 2.0 rad — they are not frozen — and the model still
learns nothing. The WGAN-GP critic cannot identify which pairs ought to be dependent; the
gradient it hands `gamma` is noise.

Meanwhile the same circuit, optimized directly on the dependency pattern, reaches 0.0103.
So the solution exists. The objective cannot find it.

The two fixes tested here
-------------------------
(A) BATCH-AWARE CRITIC.  A per-sample MLP critic has to infer a correlation structure from
    single rows. It converges to matching marginals — which the ~2000 head parameters can do
    unaided — and ignores the joint. Standard remedy: minibatch discrimination (Salimans et
    al. 2016). Each sample is given a summary of how it relates to the rest of its batch, so
    the critic can compare a *batch's* structure against a real batch's.
    This is a generic architectural technique, not the evaluation metric. **v2 §8.10 survives.**

(B) DEPENDENCY LOSS ON A FIT SPLIT, EVALUATED ON A HELD-OUT SPLIT.  The 120 pairs are split
    50/50 into FIT and EVAL, **identically for every variant**. A dependency term is applied
    only to FIT. The score is computed only on EVAL.
    **Non-circular by construction: no model ever optimized a pair it is scored on.**
    The dependency term acts on `q = <Z>`, i.e. on the quantum core, which is exactly the
    decomposition the model already asserts — the core makes dependence, the heads make
    marginals.

How to read the result
----------------------
    EVAL-pair error drops below the floor  -> the fix makes the model learn dependency
    aligned < permuted on EVAL pairs       -> the confirmatory experiment is alive again
    neither happens                        -> fall back to (C): the ceiling becomes the
                                              primary result and the GAN failure is reported
                                              honestly as a finding about tabular GANs
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_SYN, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from model import CDGQGAN  # noqa: E402
from train import DEVICE, _npn, gradient_penalty, partial_corr_matrix  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731
SEEDS = [0, 1, 2]
RIDGE = 1e-3


@dataclass
class Cfg:
    steps: int = 3000
    batch: int = 256
    critic_steps: int = 5
    lr_g: float = 5e-5
    lr_q: float = 5e-3          # quantum angles are in radians; they need their own scale
    lr_d: float = 1e-4
    lambda_gp: float = 10.0
    head_width: int = 8
    seed: int = 0
    batch_critic: bool = False  # fix (A)
    lambda_dep: float = 0.0     # fix (B); 0 disables
    dep_batch: int = 2048       # a 16x16 partial correlation needs more than 256 rows


# ---------------------------------------------------------------------------
# (A) batch-aware critic
# ---------------------------------------------------------------------------
class MinibatchDiscrimination(nn.Module):
    """Salimans et al. 2016. Gives each sample a summary of its batch.

    Deliberately generic: it never computes a correlation, and it has no idea which pairs the
    evaluation cares about. It only lets the critic notice that a *batch* of fakes is
    distributed differently from a *batch* of reals — which is precisely the signal a
    per-sample critic throws away.
    """

    def __init__(self, in_dim: int, n_kernels: int = 32, kernel_dim: int = 16):
        super().__init__()
        self.n, self.k = n_kernels, kernel_dim
        self.T = nn.Parameter(torch.randn(in_dim, n_kernels * kernel_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = (x @ self.T).view(-1, self.n, self.k)          # (B, n, k)
        d = (M.unsqueeze(0) - M.unsqueeze(1)).abs().sum(3)  # (B, B, n)
        return torch.exp(-d).sum(1) - 1.0                   # (B, n), excluding self


class Critic(nn.Module):
    def __init__(self, n_features: int, cond_dim: int, batch_aware: bool = False):
        super().__init__()
        self.mb = MinibatchDiscrimination(n_features + cond_dim) if batch_aware else None
        extra = self.mb.n if self.mb else 0
        self.net = nn.Sequential(
            nn.Linear(n_features + cond_dim + extra, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, c], dim=-1)
        if self.mb is not None:
            h = torch.cat([h, self.mb(h)], dim=-1)
        return self.net(h).squeeze(-1)


# ---------------------------------------------------------------------------
# (B) dependency term on the FIT pairs only
# ---------------------------------------------------------------------------
def partial_corr_torch(q: torch.Tensor) -> torch.Tensor:
    x = q - q.mean(0, keepdim=True)
    x = x / (x.std(0, keepdim=True) + 1e-6)
    S = (x.T @ x) / (x.shape[0] - 1)
    P = torch.linalg.inv(S + RIDGE * torch.eye(S.shape[0], device=S.device))
    d = torch.sqrt(torch.diag(P))
    R = -P / torch.outer(d, d)
    return R - torch.diag(torch.diag(R)) + torch.eye(S.shape[0], device=S.device)


def split_pairs(seed: int = 11):
    """Fixed 50/50 FIT/EVAL split of the 120 pairs. Identical for every variant."""
    allp = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
    idx = np.random.default_rng(seed).permutation(len(allp))
    half = len(allp) // 2
    return [allp[i] for i in idx[:half]], [allp[i] for i in idx[half:]]


FIT, EVAL = split_pairs()


def train_fix(X, C, edges, cfg: Cfg, target: torch.Tensor):
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = CDGQGAN(n, edges, 1, C.shape[1], cfg.head_width, seed=cfg.seed).to(DEVICE)
    D = Critic(n, C.shape[1], batch_aware=cfg.batch_critic).to(DEVICE)

    og = torch.optim.Adam([{"params": list(G.quantum.parameters()), "lr": cfg.lr_q},
                           {"params": list(G.heads.parameters()), "lr": cfg.lr_g}],
                          betas=(0.0, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9))

    fu = torch.tensor([p[0] for p in FIT], device=DEVICE)
    fv = torch.tensor([p[1] for p in FIT], device=DEVICE)

    def real(b):
        i = torch.randint(0, N, (b,), device=DEVICE)
        return Xt[i], Ct[i]

    def zz(b):
        return 2 * torch.rand(b, n, device=DEVICE) - 1

    for _ in range(cfg.steps):
        for _ in range(cfg.critic_steps):
            xr, c = real(cfg.batch)
            with torch.no_grad():
                xf = G(zz(cfg.batch), c)
            ld = (D(xf, c).mean() - D(xr, c).mean()
                  + cfg.lambda_gp * gradient_penalty(D, xr, xf, c))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()

        _, c = real(cfg.batch)
        lg = -D(G(zz(cfg.batch), c), c).mean()

        if cfg.lambda_dep > 0:
            # acts on q = <Z>, i.e. on the quantum core. FIT pairs only.
            _, cd = real(cfg.dep_batch)
            q = G.core(zz(cfg.dep_batch), cd[:, 0])
            R = partial_corr_torch(q)
            lg = lg + cfg.lambda_dep * (R[fu, fv] - target[fu, fv]).abs().mean()

        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()

    return G


@torch.no_grad()
def gen(G, C, n_samples, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n_samples)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n_samples, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy()


def main() -> None:
    rng = np.random.default_rng(20260711)
    Gt = teacher_graph(rng)
    X, C, _ = teacher_data(Gt, N_TRAIN, rng)
    Gp = isomorphic_permuted(Gt, np.random.default_rng(7))

    Xn = _npn(X)
    S = np.corrcoef(Xn, rowvar=False)
    P = np.linalg.inv(S + RIDGE * np.eye(N_FEAT))
    d = np.sqrt(np.diag(P))
    Rstar = -P / np.outer(d, d)
    np.fill_diagonal(Rstar, 1.0)
    target = torch.tensor(Rstar, dtype=torch.float32, device=DEVICE)
    zr = Z(partial_corr_matrix(X))

    def score(Xs):
        zs = Z(partial_corr_matrix(Xs))
        ev = float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in EVAL]))
        ft = float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in FIT]))
        eg = float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in Gt.edges()]))
        return ev, ft, eg

    # floor: a model with zero dependency, scored the same way
    fl = []
    for s in range(3):
        r = np.random.default_rng(100 + s)
        fl.append(score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])))
    fl = np.mean(fl, 0)

    print("=" * 96)
    print("Two fixes for 'WGAN-GP learns no dependency'")
    print("=" * 96)
    print(f"  120 pairs split 50/50, identically for every variant: "
          f"{len(FIT)} FIT (in the loss) · {len(EVAL)} EVAL (never in any loss)")
    print()
    print(f"  {'config':<34} {'graph':<10} {'EVAL60':>9} {'FIT60':>9} {'true edges':>12}")
    print("  " + "-" * 80)
    print(f"  {'floor (zero dependency)':<34} {'—':<10} {fl[0]:>9.4f} {fl[1]:>9.4f} "
          f"{fl[2]:>12.4f}")
    print()

    configs = [
        ("baseline (WGAN-GP, lr_q=5e-3)", Cfg(lr_q=5e-3)),
        ("(A) batch-aware critic",        Cfg(lr_q=5e-3, batch_critic=True)),
        ("(B) dependency loss on FIT",    Cfg(lr_q=5e-3, lambda_dep=1.0)),
        ("(A)+(B)",                       Cfg(lr_q=5e-3, batch_critic=True, lambda_dep=1.0)),
    ]

    for name, base in configs:
        for gname, Gv in (("aligned", Gt), ("permuted", Gp)):
            res, t0 = [], time.time()
            for s in SEEDS:
                cfg = Cfg(**{**base.__dict__, "seed": s})
                Gm = train_fix(X, C, list(Gv.edges()), cfg, target)
                res.append(score(gen(Gm, C, N_SYN, s)))
            r = np.mean(res, 0)
            print(f"  {name:<34} {gname:<10} {r[0]:>9.4f} {r[1]:>9.4f} {r[2]:>12.4f}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
        print()

    print("=" * 96)
    print("  EVAL60 is the primary number: no model ever optimized those pairs.")
    print(f"    below the floor ({fl[0]:.4f})   -> the fix makes the model learn dependency")
    print("    aligned < permuted             -> the confirmatory experiment is alive again")
    print("    neither                        -> fall back to (C), the ceiling as primary")


if __name__ == "__main__":
    main()

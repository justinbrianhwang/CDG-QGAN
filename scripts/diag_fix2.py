"""Second round of fixes: make the critic blind to the marginals.

Why
---
The failure mode is that WGAN-GP converges to matching marginals and ignores the joint. But
matching the marginals is exactly what the ~2000 classical head parameters can already do on
their own, unaided by the quantum core. So the critic's capacity is being spent on the one
part of the problem the entangling angles cannot help with, and `gamma` gets nothing.

Fix (E): the COPULA CRITIC.  Rank-transform each feature *within the batch* before the critic
sees it (a differentiable soft rank, then the inverse normal CDF). This is the nonparanormal
transform, made differentiable. It destroys all marginal information — every feature is
standard normal by construction — so the ONLY thing left for the critic to discriminate on is
the dependency structure. There is nowhere else for the gradient to go but `gamma`.

Note what this is and is not. It does **not** compute a correlation, a partial correlation, or
anything the evaluation metric computes; it is an input transform on the critic. It is the
same monotone-invariance argument that already justifies measuring the metric in nonparanormal
space (review A-4) — applied to the critic instead of to the metric. v2 §8.10 survives.

Fix (D): a bigger critic batch.  A 16-dimensional joint structure estimated from 256 rows is
a noisy thing to take a gradient through. Cheap to test, uncontroversial.

Both are combined with the batch-aware critic (A) from diag_fix.py.

Primary metric is unchanged: EVAL60 — the 60 pairs that never enter any loss.
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
from diag_fix import EVAL, FIT, MinibatchDiscrimination, gen  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from model import CDGQGAN  # noqa: E402
from train import DEVICE, partial_corr_matrix  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731
SQRT2 = float(np.sqrt(2.0))

# SCREENING MODE. Two seeds, 2000 steps — enough to see whether a fix breaks the floor at all,
# not enough to report. Whatever survives here gets re-run at the full 3 seeds x 3000 steps
# under the same conditions as every other result, and only that number is reported.
SEEDS = [0, 1]
STEPS = 2000


@dataclass
class Cfg:
    steps: int = STEPS
    batch: int = 256
    critic_steps: int = 5
    lr_g: float = 5e-5
    lr_q: float = 5e-3
    lr_d: float = 1e-4
    lambda_gp: float = 10.0
    head_width: int = 8
    seed: int = 0
    copula: bool = False      # fix (E)
    batch_critic: bool = False
    tau: float = 0.05         # soft-rank temperature


def soft_npn(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Differentiable nonparanormal transform, computed within the batch.

    rank_u(i) = mean_j sigmoid((x_u(i) - x_u(j)) / tau)   -> a soft empirical CDF
    then map through the inverse normal CDF.

    Every output feature is (approximately) standard normal, so the marginals carry no
    information at all and only the copula survives.
    """
    d = (x.unsqueeze(1) - x.unsqueeze(0)) / tau      # (B, B, n)
    r = torch.sigmoid(d).mean(dim=1)                 # (B, n) in (0,1)
    B = x.shape[0]
    r = r.clamp(1.0 / (2 * B), 1.0 - 1.0 / (2 * B))
    return SQRT2 * torch.erfinv(2 * r - 1)           # inverse normal CDF


class Critic(nn.Module):
    def __init__(self, n: int, cond_dim: int, copula: bool, batch_aware: bool, tau: float):
        super().__init__()
        self.copula, self.tau = copula, tau
        self.mb = MinibatchDiscrimination(n + cond_dim) if batch_aware else None
        extra = self.mb.n if self.mb else 0
        self.net = nn.Sequential(
            nn.Linear(n + cond_dim + extra, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.copula:
            x = soft_npn(x, self.tau)
        h = torch.cat([x, c], dim=-1)
        if self.mb is not None:
            h = torch.cat([h, self.mb(h)], dim=-1)
        return self.net(h).squeeze(-1)


def gradient_penalty(critic, xr, xf, c) -> torch.Tensor:
    """Local copy: the critic here transforms its input, so it cannot reuse train.py's."""
    a = torch.rand(xr.size(0), 1, device=xr.device)
    xh = (a * xr + (1 - a) * xf).requires_grad_(True)
    g, = torch.autograd.grad(critic(xh, c).sum(), xh, create_graph=True)
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def train_fix2(X, C, edges, cfg: Cfg):
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = CDGQGAN(n, edges, 1, C.shape[1], cfg.head_width, seed=cfg.seed).to(DEVICE)
    D = Critic(n, C.shape[1], cfg.copula, cfg.batch_critic, cfg.tau).to(DEVICE)

    og = torch.optim.Adam([{"params": list(G.quantum.parameters()), "lr": cfg.lr_q},
                           {"params": list(G.heads.parameters()), "lr": cfg.lr_g}],
                          betas=(0.0, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9))

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
        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()

    return G


def main() -> None:
    rng = np.random.default_rng(20260711)
    Gt = teacher_graph(rng)
    X, C, _ = teacher_data(Gt, N_TRAIN, rng)
    Gp = isomorphic_permuted(Gt, np.random.default_rng(7))
    zr = Z(partial_corr_matrix(X))

    def score(Xs):
        zs = Z(partial_corr_matrix(Xs))
        ev = float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in EVAL]))
        ft = float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in FIT]))
        eg = float(np.mean([abs(zs[i, j] - zr[i, j]) for i, j in Gt.edges()]))
        return ev, ft, eg

    fl = []
    for s in range(3):
        r = np.random.default_rng(100 + s)
        fl.append(score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])))
    fl = np.mean(fl, 0)

    print("=" * 96)
    print("Round 2: make the critic blind to the marginals   [SCREENING — not reportable]")
    print("=" * 96)
    print(f"  {len(SEEDS)} seeds x {STEPS} steps. Enough to see whether a fix breaks the floor;")
    print("  not enough to report. Survivors are re-run at 3 seeds x 3000 steps.")
    print("  No dependency term anywhere in these runs. The loss is pure WGAN-GP.")
    print("  All 120 pairs are out-of-loss; EVAL60 is reported for comparability with round 1.")
    print()
    print(f"  {'config':<36} {'graph':<10} {'EVAL60':>9} {'FIT60':>9} {'true edges':>12}")
    print("  " + "-" * 82)
    print(f"  {'floor (zero dependency)':<36} {'—':<10} {fl[0]:>9.4f} {fl[1]:>9.4f} "
          f"{fl[2]:>12.4f}")
    print()

    configs = [
        ("(E) copula critic",              Cfg(copula=True)),
        ("(E)+(A) copula + batch-aware",   Cfg(copula=True, batch_critic=True)),
        ("(D) big batch (1024)",           Cfg(batch=1024)),
        ("(E)+(D) copula + big batch",     Cfg(copula=True, batch=1024)),
    ]

    for name, base in configs:
        for gname, Gv in (("aligned", Gt), ("permuted", Gp)):
            res, t0 = [], time.time()
            for s in SEEDS:
                cfg = Cfg(**{**base.__dict__, "seed": s})
                Gm = train_fix2(X, C, list(Gv.edges()), cfg)
                res.append(score(gen(Gm, C, N_SYN, s)))
            r = np.mean(res, 0)
            print(f"  {name:<36} {gname:<10} {r[0]:>9.4f} {r[1]:>9.4f} {r[2]:>12.4f}"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
        print()

    print("=" * 96)
    print("  'true edges' is the sharpest number here: the floor is 0.3676 and the circuit can")
    print("  reach ~0. Anything that pulls it meaningfully below the floor means the model has")
    print("  finally started to learn dependency at all.")


if __name__ == "__main__":
    main()

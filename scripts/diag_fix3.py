"""Round 3: put the critic in the same space as the metric.

The bug
-------
The copula critic (diag_fix2, fix E) soft-rank-transforms x before the critic sees it, which is
what forced the gradient into the entangling angles and rescued the synthetic run. But it
rank-transforms x **pooled over all c**, and never residualizes on c.

The metric (`eval_dep.partial_corr_c`, which is also the estimator the CDG is *defined* by) does
both: nonparanormal, THEN residualize on a nonlinear basis of c, THEN partial correlation.

So the critic judges the POOLED copula and the metric scores the C-CONDITIONAL copula. They are
different objects. The generator faithfully optimizes the one it is shown, and it is not the one
we score. This is the same estimator-mismatch that already bit us twice — between `build_cdg` and
`eval_dep` (REVISIONS A-4/E-3), and between the metric and the training loss (E-3) — surfacing a
third time, now between the metric and the *critic*.

Evidence (`scripts/diag_head.py`, real MIMIC, 3000 steps):

    head       graph   error on q   error on x   fold %
    free MLP   cdg        0.1092       0.0943     57.9%
    monotone   cdg        0.1030       0.1455      0.0%

Two things fall out of that table. The heads fold on 58% of the q-grid when nothing constrains
them (the copula critic is blind to marginals, so nothing does). And forcing monotonicity does
NOT fix the output: h(q_u, c) monotone in q_u still reshapes the POOLED distribution through its
c-dependence, so copula(x) != copula(q) once you pool. Monotonicity in q at fixed c is not enough;
the estimator pools.

The fix
-------
Give the critic the same view the metric has:

    critic input  =  residualize( soft_npn(x) , cond_basis(c) )   ++  c

Now real and fake are compared in exactly the space the CDG lives in. Marginals still carry no
information (soft_npn destroys them), so the gradient still has nowhere to go but gamma — but
now the structure it is pushed toward is the c-conditional one we actually score.

On circularity (v2 §8.10)
-------------------------
This is an INPUT TRANSFORM on the critic, not a term in the loss. The loss remains pure WGAN-GP.
The critic never computes a partial correlation, never sees the target matrix, and never sees the
CDG. It is the same monotone-invariance argument that already justified soft_npn (review A-4),
extended to the conditioning variables. What would be circular is putting the evaluation metric
in the objective; we are not doing that, and `no_entangle` remains the control that proves the
dependency has to come from the entangling angles.

Tested here: the residualized critic, with the free head and with the monotone head, on the CDG
and on an isomorphic permutation.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from diag_fix import MinibatchDiscrimination  # noqa: E402
from diag_fix2 import soft_npn  # noqa: E402
from diag_head import Gen  # noqa: E402
from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_SYN = 20000


@dataclass
class Cfg3:
    steps: int = 8000
    batch: int = 512
    critic_steps: int = 5
    lr_g: float = 5e-5
    lr_q: float = 5e-3
    lr_d: float = 1e-4
    lambda_gp: float = 10.0
    head_width: int = 8
    seed: int = 0
    tau: float = 0.05
    residualize: bool = True     # the fix
    monotone: bool = False


def cond_basis_t(C: torch.Tensor) -> torch.Tensor:
    """Torch mirror of eval_dep.cond_basis — same columns, same order."""
    n, k = C.shape
    cols = [torch.ones(n, 1, device=C.device)]
    cols += [C[:, j:j + 1] for j in range(k)]
    cols += [C[:, j:j + 1] ** 2 for j in range(k)]
    for a in range(k):
        for b in range(a + 1, k):
            cols.append(C[:, a:a + 1] * C[:, b:b + 1])
    return torch.cat(cols, dim=1)


class CriticC(nn.Module):
    """Copula critic that lives in the metric's space: soft_npn -> residualize on c -> MLP."""

    def __init__(self, n: int, cond_dim: int, tau: float, residualize: bool):
        super().__init__()
        self.tau, self.residualize = tau, residualize
        self.mb = MinibatchDiscrimination(n + cond_dim)
        self.net = nn.Sequential(
            nn.Linear(n + cond_dim + self.mb.n, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        u = soft_npn(x, self.tau)
        if self.residualize:
            D = cond_basis_t(c)
            beta = torch.linalg.lstsq(D, u).solution
            u = u - D @ beta                       # the space eval_dep scores in
        h = torch.cat([u, c], dim=-1)
        h = torch.cat([h, self.mb(h)], dim=-1)
        return self.net(h).squeeze(-1)


def gp(critic, xr, xf, c):
    a = torch.rand(xr.size(0), 1, device=xr.device)
    xh = (a * xr + (1 - a) * xf).requires_grad_(True)
    g, = torch.autograd.grad(critic(xh, c).sum(), xh, create_graph=True)
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def train3(X, C, edges, cfg: Cfg3):
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = Gen(n, edges, C.shape[1], cfg.head_width, cfg.seed, cfg.monotone).to(DEVICE)
    D = CriticC(n, C.shape[1], cfg.tau, cfg.residualize).to(DEVICE)
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
            ld = (D(xf, c).mean() - D(xr, c).mean() + cfg.lambda_gp * gp(D, xr, xf, c))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()
        _, c = real(cfg.batch)
        lg = -D(G(zz(cfg.batch), c), c).mean()
        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()
    return G


@torch.no_grad()
def sample(G, C, n, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy(), G.q(z, c).cpu().numpy(), C[idx]


def main() -> None:
    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    zz_ = np.load(PROCESSED / "cdg.npz", allow_pickle=True)
    G_cdg = nx.Graph()
    G_cdg.add_nodes_from(range(N_FEAT))
    G_cdg.add_edges_from([tuple(e) for e in zz_["E_fit"]])

    zr = fisher_z(partial_corr_c(X, C))

    def score(A, Cs):
        return dep_error(zr, fisher_z(partial_corr_c(A, Cs)), ALL_PAIRS)

    fl = []
    for s in range(5):
        r = np.random.default_rng(1000 + s)
        fl.append(score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)]),
                        C[r.permutation(len(C))]))
    floor = float(np.mean(fl))

    variants = {"cdg": G_cdg,
                "permuted": isomorphic_permuted(G_cdg, np.random.default_rng(20260711))}

    print("=" * 98)
    print("Round 3 — put the critic in the metric's space (residualize on c)")
    print("=" * 98)
    print(f"  floor {floor:.4f}   ·   irreducible (L=1 cone) 0.0331   ·   "
          f"ceiling without a GAN 0.0437")
    print("  Reference (round 2, pooled copula critic, 3000 steps): cdg x-error 0.0943")
    print()
    print(f"  {'critic':<22} {'head':<10} {'graph':<10} {'err on q':>9} {'err on x':>9} "
          f"{'vs floor':>9}")
    print("  " + "-" * 78)

    out = {"floor": floor}
    configs = [
        ("pooled copula (old)",   Cfg3(residualize=False, monotone=False)),
        ("residualized copula",   Cfg3(residualize=True,  monotone=False)),
        ("residualized + monot.", Cfg3(residualize=True,  monotone=True)),
    ]
    for cname, base in configs:
        for gname, Gv in variants.items():
            t0 = time.time()
            cfg = Cfg3(**{**base.__dict__})
            Gm = train3(X, C, list(Gv.edges()), cfg)
            Xs, Qs, Cs = sample(Gm, C, N_SYN, 0)
            eq, ex = score(Qs, Cs), score(Xs, Cs)
            out[f"{cname}/{gname}"] = {"q": eq, "x": ex}
            hn = "monotone" if base.monotone else "free"
            print(f"  {cname:<22} {hn:<10} {gname:<10} {eq:>9.4f} {ex:>9.4f} "
                  f"{(ex - floor) / floor * 100:>+8.1f}%   ({time.time()-t0:.0f}s)", flush=True)
        print()

    print("=" * 98)
    print("  What to look for: 'err on x' well below the floor, and cdg well below permuted.")
    print("  The ceiling says 0.0437 is reachable. Anything near it means training is fixed.")

    RESULTS.mkdir(exist_ok=True)
    import json
    (RESULTS / "diag_fix3.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

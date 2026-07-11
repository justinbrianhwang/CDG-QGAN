"""Are the local heads destroying the dependency the quantum core creates?

The hypothesis
--------------
`ceiling_real.py` proved the circuit CAN represent the real conditional dependency pattern
(aligned 0.0437 against a floor of 0.0986 and an irreducible bound of 0.0331). So the WGAN-GP
failure is a training failure. This script tests the most likely mechanism.

The copula critic (diag_fix2, fix E) rank-transforms its input, which is exactly what forces the
gradient into the entangling angles instead of the marginals. But it has a side effect nobody
asked for: **nothing in the loss constrains the local head h_u to be monotone in q_u any more.**

That matters, because the whole argument rests on monotonicity:

    h_u monotone in q_u  =>  npn(x_u) = npn(q_u)  =>  copula(x) = copula(q)

i.e. the dependency the quantum core creates is the dependency the output has. If a head instead
learns a FOLD — a U-shape in q_u — then two different values of q_u map to the same x_u, the
copula is scrambled, and a perfectly good dependency in q becomes noise in x. That would produce
precisely the symptom we see: a score WORSE than the floor. A model that creates no dependency
scores 0.0986; a model that creates the right dependency and then folds it can score worse,
because it is actively manufacturing structure that is not there.

The synthetic teacher hid this: its marginals are near-Gaussian, so a monotone head fits them and
there is no pressure to fold. Real MIMIC labs are skewed and heavy-tailed.

The measurement
---------------
Train as in WP-2, then score the SAME model twice:

    error on q  — the core's own output, before the heads
    error on x  — after the heads.  This is what WP-2 scores.

  error_q good, error_x bad  -> the heads are scrambling it. Confirmed. Fix the head.
  both bad                   -> the core never learned it. The critic is still the problem.

Also reported: the fraction of the q-grid on which dh_u/dq_u < 0 (a direct read of folding), and
a MONOTONE head — d h/d q >= 0 by construction — as the candidate fix.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from diag_fix2 import Cfg, Critic, gradient_penalty  # noqa: E402
from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from model import CDGQGAN  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from qsim import GraphLocalQuantumGenerator  # noqa: E402
from qsim_lightcone import LightconeGenerator  # noqa: E402
from train import DEVICE  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
STEPS = 3000
N_SYN = 20000


class MonotoneHead(nn.Module):
    """1-D map (q_u, c) -> x~_u that is NON-DECREASING in q_u by construction.

    x~_u = b2(c) + sum_k softplus(v_k) * tanh( softplus(w_k) * q_u + a_k(c) )

    dh/dq = sum_k softplus(v_k) * softplus(w_k) * sech^2(...) >= 0, always.

    c still enters freely — it shifts each unit's offset and the output bias — so the head keeps
    every bit of its power to shape the conditional marginal. It just cannot fold q_u back on
    itself. Proposition D-2 is untouched: the head still reads only (q_u, c), never q_v.
    """

    def __init__(self, cond_dim: int, width: int = 8):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(width))          # softplus(0)=0.69 -> slope starts ~1
        self.v = nn.Parameter(torch.zeros(width))
        self.a = nn.Linear(cond_dim, width)                # offsets, free in c
        self.b = nn.Linear(cond_dim, 1)                    # output bias, free in c

    def forward(self, q_u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(F.softplus(self.w) * q_u.unsqueeze(-1) + self.a(c))
        return (h * F.softplus(self.v)).sum(-1) + self.b(c).squeeze(-1)


class Gen(nn.Module):
    """CDGQGAN with a switchable head. Everything else identical."""

    def __init__(self, n, edges, cond_dim, head_width, seed, monotone: bool):
        super().__init__()
        self.n = n
        self.quantum = GraphLocalQuantumGenerator(n, edges, 1, seed=seed)
        self.core = LightconeGenerator(self.quantum, edges, n, 1)
        H = MonotoneHead if monotone else _FreeHead
        self.heads = nn.ModuleList([H(cond_dim, head_width) for _ in range(n)])

    def q(self, z, c):
        return self.core(z, c[:, 0])

    def forward(self, z, c):
        q = self.q(z, c)
        return torch.stack([h(q[:, u], c) for u, h in enumerate(self.heads)], dim=1)


class _FreeHead(nn.Module):
    """The current head — a free MLP. Nothing stops it from folding."""

    def __init__(self, cond_dim: int, width: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1 + cond_dim, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, q_u, c):
        return self.net(torch.cat([q_u.unsqueeze(-1), c], dim=-1)).squeeze(-1)


@torch.no_grad()
def fold_fraction(G, C) -> float:
    """Fraction of the (q, c) grid where dh_u/dq_u < 0 — i.e. how much the heads fold."""
    qs = torch.linspace(-1, 1, 201, device=DEVICE)
    idx = np.random.default_rng(0).integers(0, len(C), 64)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    neg = tot = 0
    for h in G.heads:
        for ci in range(len(c)):
            cc = c[ci:ci + 1].expand(len(qs), -1)
            y = h(qs, cc)
            d = y[1:] - y[:-1]
            neg += int((d < 0).sum())
            tot += d.numel()
    return neg / tot


@torch.no_grad()
def sample(G, C, n, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy(), G.q(z, c).cpu().numpy(), C[idx]


def train(X, C, edges, cfg: Cfg, monotone: bool):
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = Gen(n, edges, C.shape[1], cfg.head_width, cfg.seed, monotone).to(DEVICE)
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

    print("=" * 94)
    print("Are the heads scrambling the dependency the core creates?")
    print("=" * 94)
    print(f"  floor = {floor:.4f}   ·   irreducible (L=1 light cone) = 0.0331   ·   "
          f"ceiling reached without a GAN = 0.0437")
    print(f"  {STEPS} steps · copula + batch-aware critic · pure WGAN-GP")
    print()
    print("  error on q = the quantum core's own output, BEFORE the heads")
    print("  error on x = after the heads. This is what WP-2 scores.")
    print()
    print(f"  {'head':<12} {'graph':<10} {'error on q':>11} {'error on x':>11} "
          f"{'fold %':>8}   verdict")
    print("  " + "-" * 74)

    out = {"floor": floor}
    for monotone in (False, True):
        hname = "monotone" if monotone else "free MLP"
        for gname, Gv in variants.items():
            t0 = time.time()
            cfg = Cfg(steps=STEPS, seed=0, copula=True, batch_critic=True, lr_q=5e-3)
            Gm = train(X, C, list(Gv.edges()), cfg, monotone)
            Xs, Qs, Cs = sample(Gm, C, N_SYN, 0)
            eq, ex = score(Qs, Cs), score(Xs, Cs)
            fold = fold_fraction(Gm, C)
            v = ("heads scramble it" if eq < floor and ex >= floor
                 else "both fine" if ex < floor else "core never learned it")
            out[f"{hname}/{gname}"] = {"q": eq, "x": ex, "fold": fold}
            print(f"  {hname:<12} {gname:<10} {eq:>11.4f} {ex:>11.4f} {fold*100:>7.1f}%   "
                  f"{v}   ({time.time()-t0:.0f}s)", flush=True)
        print()

    print("=" * 94)
    print("  If 'free MLP' shows a good error on q and a bad error on x with a nonzero fold %,")
    print("  the copula critic bought us gradient into gamma at the cost of letting the heads")
    print("  fold — and the monotone head gives us both.")

    RESULTS.mkdir(exist_ok=True)
    import json
    (RESULTS / "diag_head.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

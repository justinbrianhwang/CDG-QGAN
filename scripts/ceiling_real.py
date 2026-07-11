"""Joint expressivity ceiling on the REAL cohort. GAN removed.

This is `ceiling_joint.py` moved from the synthetic teacher to MIMIC-IV, and it answers the one
question that decides whether WP-2 is worth running at all:

    Can a depth-1 circuit wired on the CDG reproduce the real conditional dependency pattern —
    21 edges at a mean |z| of 0.21, the other 99 pairs near zero — at the same time?

The WP-2 smoke run put every trained variant 33% ABOVE the floor. There are two readings, and
they demand opposite responses:

    (a) the circuit cannot represent the real pattern     -> the design is wrong. WP-2 is moot.
    (b) it can, but WGAN-GP does not find it              -> the training is wrong. Fix that.

On the synthetic teacher this same script settled the question (aligned 0.0103 vs floor 0.0609)
and told us to go fix the critic. Nothing guarantees the answer is the same on real data: the
real signal is far weaker (floor 0.0986 = mean |z| over all pairs) and 72 of 120 pairs sit
outside the L=1 light cone, where Corollary 1 forces the model to exactly zero.

The best score any L=1 CDG circuit can possibly achieve is therefore NOT zero. It is the error
it must pay on those 72 unreachable pairs:

    irreducible = (1/120) * sum_{d_G(u,v) > 2} |z_true(u,v)|

which this script computes and prints. Read the ceiling against THAT, not against zero.

Method: gradient-descend the circuit parameters directly on the c-conditional dependency error,
scored on a fixed held-out draw. The score uses the same estimator as `eval_dep.partial_corr_c`
(soft nonparanormal -> residualize on the c basis -> partial correlation), made differentiable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from diag_fix2 import soft_npn  # noqa: E402
from eval_dep import RIDGE, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import degree_preserving_rewire, distance_matched_permuted, isomorphic_permuted  # noqa: E402
from model import CDGQGAN  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402

N_FEAT = 16
STEPS = 1200
BATCH = 2048          # soft_npn is O(B^2 n); 2048 keeps it under ~300 MB
RESTARTS = 2
EVAL_EVERY = 50
TAU = 0.05


def cond_basis_t(C: torch.Tensor) -> torch.Tensor:
    """Torch mirror of eval_dep.cond_basis. Same columns, same order."""
    n, k = C.shape
    cols = [torch.ones(n, 1, device=C.device)]
    cols += [C[:, j:j + 1] for j in range(k)]
    cols += [C[:, j:j + 1] ** 2 for j in range(k)]
    for a in range(k):
        for b in range(a + 1, k):
            cols.append(C[:, a:a + 1] * C[:, b:b + 1])
    return torch.cat(cols, dim=1)


def partial_corr_c_t(q: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Differentiable eval_dep.partial_corr_c: soft-npn -> residualize on c -> partial corr."""
    Xn = soft_npn(q, TAU)
    D = cond_basis_t(C)
    beta = torch.linalg.lstsq(D, Xn).solution
    R = Xn - D @ beta
    R = R - R.mean(0, keepdim=True)
    R = R / (R.std(0, keepdim=True) + 1e-6)
    S = (R.T @ R) / (R.shape[0] - 1)
    P = torch.linalg.inv(S + RIDGE * torch.eye(S.shape[0], device=S.device))
    d = torch.sqrt(torch.diag(P))
    M = -P / torch.outer(d, d)
    return M - torch.diag(torch.diag(M)) + torch.eye(S.shape[0], device=S.device)


def fit(edges, target_z: torch.Tensor, Ct: torch.Tensor, seed: int, iu) -> float:
    torch.manual_seed(seed)
    m = CDGQGAN(N_FEAT, list(edges), depth=1, cond_dim=Ct.shape[1], seed=seed).to(DEVICE)
    core = m.core
    opt = torch.optim.Adam(list(m.quantum.parameters()), lr=0.05)

    g = torch.Generator(device=DEVICE).manual_seed(12345)
    z_ev = 2 * torch.rand(BATCH, N_FEAT, device=DEVICE, generator=g) - 1
    c_ev = Ct[torch.randint(0, len(Ct), (BATCH,), device=DEVICE, generator=g)]

    @torch.no_grad()
    def held_out() -> float:
        Z = torch.arctanh(partial_corr_c_t(core(z_ev, c_ev[:, 0]), c_ev).clamp(-0.999, 0.999))
        return (Z[iu[0], iu[1]] - target_z[iu[0], iu[1]]).abs().mean().item()

    best = held_out()
    for step in range(STEPS):
        idx = torch.randint(0, len(Ct), (BATCH,), device=DEVICE)
        c = Ct[idx]
        z = 2 * torch.rand(BATCH, N_FEAT, device=DEVICE) - 1
        Z = torch.arctanh(partial_corr_c_t(core(z, c[:, 0]), c).clamp(-0.999, 0.999))
        loss = (Z[iu[0], iu[1]] - target_z[iu[0], iu[1]]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            best = min(best, held_out())
    return best


def main() -> None:
    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    zz = np.load(PROCESSED / "cdg.npz", allow_pickle=True)
    G = nx.Graph()
    G.add_nodes_from(range(N_FEAT))
    G.add_edges_from([tuple(e) for e in zz["E_fit"]])
    E_hold = [tuple(e) for e in zz["E_holdout"]]

    zr = fisher_z(partial_corr_c(X, C))
    target_z = torch.tensor(zr, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)
    iu = torch.triu_indices(N_FEAT, N_FEAT, offset=1)

    ALL = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
    out_cone = [p for p in ALL if not nx.has_path(G, *p)
                or nx.shortest_path_length(G, *p) > 2]
    irreducible = float(np.mean([abs(zr[i, j]) if (i, j) in out_cone else 0.0 for i, j in ALL]))
    floor = float(np.mean([abs(zr[i, j]) for i, j in ALL]))

    gr = np.random.default_rng(20260711)
    variants = {
        "aligned (CDG)": G,
        "permuted": isomorphic_permuted(G, gr),
        "distmatched": distance_matched_permuted(G, E_hold, gr)[0],
        "rewired": degree_preserving_rewire(G, gr),
        "no_entangle": nx.empty_graph(N_FEAT),
    }

    print("=" * 86)
    print("Joint expressivity ceiling on REAL MIMIC-IV — direct optimization, no GAN")
    print("=" * 86)
    print(f"  n = {len(df):,}   ·   CDG {G.number_of_edges()} edges   ·   L=1")
    print(f"  floor       = {floor:.4f}   a model that creates no dependency at all")
    print(f"  irreducible = {irreducible:.4f}   the error ANY L=1 CDG circuit must pay on the")
    print(f"                         {len(out_cone)} pairs outside its light cone (Corollary 1)")
    print()
    print(f"  {'model':<16} {'120-pair error':>15} {'vs floor':>10} {'vs irreducible':>16}")
    print("  " + "-" * 62)

    res = {}
    for name, Gv in variants.items():
        t0 = time.time()
        e = min(fit(Gv.edges(), target_z, Ct, s, iu) for s in range(RESTARTS))
        res[name] = e
        print(f"  {name:<16} {e:>15.4f} {(floor - e) / floor * 100:>9.1f}% "
              f"{e - irreducible:>+15.4f}   ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 86)
    a = res["aligned (CDG)"]
    for ref in list(variants)[1:]:
        d = a - res[ref]
        print(f"  aligned − {ref:<14} = {d:+.4f}   "
              f"{'aligned better' if d < 0 else 'ALIGNED WORSE'}")
    print()
    if a >= floor * 0.95:
        print("  >> The circuit CANNOT represent the real pattern. Design flaw — WP-2 is moot.")
    elif a < res["permuted"] - 0.002:
        print("  >> Representable, and alignment helps. The hypothesis holds on real data;")
        print("     the WP-2 smoke failure is a TRAINING problem, not a design problem.")
    else:
        print("  >> Representable, but alignment does not help. The premise of WP-2 collapses.")
    print("=" * 86)

    RESULTS.mkdir(exist_ok=True)
    import json
    (RESULTS / "ceiling_real.json").write_text(
        json.dumps({"floor": floor, "irreducible": irreducible, "results": res}, indent=2))


if __name__ == "__main__":
    main()

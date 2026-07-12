"""How close to a classical baseline can this circuit actually get?

The story so far
----------------
    Gaussian copula (classical)   0.0060      <- the number to beat
    CDG-QGAN Δ=3, trained         0.0694      11.6x worse
    CDG-QGAN Δ=3, ceiling         0.0436      fully converged: 4x more optimization moves it 0.0001
    CDG-QGAN Δ=4, ceiling         0.0300      and NOT converged — still +0.0160 above its own bound

Raising the degree cap from 3 to 4 (v2 §7.6 set it at 3; the data never asked for that) cut the
Corollary 1 bound from 0.0331 to 0.0140 and the achieved ceiling from 0.0436 to 0.0300 — but only
once the optimizer was given 4x the budget. A denser graph is a harder optimization problem, and
at the original budget it looked like the extra edges bought nothing (0.0418). They do.

Crucially the light cone is UNCHANGED: L=1 throughout. Corollary 1 and Proposition D-2 hold exactly
as before. What moved is a hyperparameter of the graph estimator, not the theory.

This sweep asks the obvious next question: how far does that go?

For each Δ ∈ {3,4,5,6} it optimizes ONLY the aligned circuit, hard, and reports:
    bound       what Corollary 1 forbids (a property of the graph)
    ceiling     what the circuit reaches (a property of the graph AND the circuit)
    gap         ceiling - bound = what the CIRCUIT cannot express
    vs copula   ceiling / 0.0060

A gap that stays flat while the bound falls means density is still paying. A gap that grows means
we are hitting the circuit's own capacity — the point where the 16-qubit, depth-1, one-qubit-per-
feature architecture runs out, and the next lever would have to be qubits per feature, not edges.

This is a PERFORMANCE sweep. It says nothing about whether the CDG beats its controls — density
also makes a random permutation reach more pairs by luck, and that trade-off is measured separately
(`design_sweep.py` for the alignment z, `ceiling_real.py` for the trained contrasts). Both have to
hold for the design point to be worth adopting.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import ceiling_real as CR  # noqa: E402
from eval_dep import fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
COPULA = 0.0060
FLOOR = 0.0986

STEPS = 6000
RESTARTS = 5


def main() -> None:
    CR.STEPS, CR.RESTARTS = STEPS, RESTARTS

    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    zr = fisher_z(partial_corr_c(X, C))
    az = np.abs(zr)
    target_z = torch.tensor(zr, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)
    iu = torch.triu_indices(N_FEAT, N_FEAT, offset=1)

    print("=" * 96)
    print("Performance sweep — how close to the classical baseline can the circuit get?")
    print("=" * 96)
    print(f"  Gaussian copula {COPULA:.4f}   ·   floor {FLOOR:.4f}   ·   L = 1 throughout")
    print(f"  optimization: {STEPS} steps x {RESTARTS} restarts (Δ=3 is converged at this budget)")
    print()
    print(f"  {'Δ':>2} {'|E|':>4} {'maxdeg':>7} {'out of cone':>12} {'bound':>8} "
          f"{'ceiling':>9} {'gap':>8} {'vs copula':>10}")
    print("  " + "-" * 74)

    rows = {}
    for delta, fname in ((3, "cdg.npz"), (4, "cdg_d4.npz"), (5, "cdg_d5.npz"), (6, "cdg_d6.npz")):
        p = PROCESSED / fname
        if not p.exists():
            print(f"  {delta:>2}  (missing {fname} — run build_cdg.py --max-degree {delta})")
            continue
        z = np.load(p, allow_pickle=True)
        G = nx.Graph()
        G.add_nodes_from(range(N_FEAT))
        G.add_edges_from([tuple(e) for e in z["E_fit"]])

        out = [q for q in ALL_PAIRS
               if not nx.has_path(G, *q) or nx.shortest_path_length(G, *q) > 2]
        bound = float(np.sum([az[i, j] for i, j in out]) / len(ALL_PAIRS))

        t0 = time.time()
        e = min(CR.fit(G.edges(), target_z, Ct, s, iu) for s in range(RESTARTS))
        md = max(dict(G.degree()).values())
        rows[delta] = {"edges": G.number_of_edges(), "maxdeg": md, "out_of_cone": len(out),
                       "bound": bound, "ceiling": e, "gap": e - bound}
        print(f"  {delta:>2} {G.number_of_edges():>4} {md:>7} {len(out):>9}/120 {bound:>8.4f} "
              f"{e:>9.4f} {e - bound:>8.4f} {e / COPULA:>9.1f}x   ({time.time()-t0:.0f}s)",
              flush=True)

    print()
    print("=" * 96)
    print("  gap = ceiling − bound. It is what the CIRCUIT cannot express, as opposed to what the")
    print("  GRAPH forbids. If the gap grows as Δ rises, the bottleneck has moved off the light")
    print("  cone and onto the circuit itself, and more edges will not help — the next lever would")
    print("  be qubits per feature (L=1 and Corollary 1 would still hold; the cone is defined on")
    print("  the feature graph, not on the qubit count).")
    print()
    print("  Density is NOT free: it also lets a random permutation reach more pairs by luck, so")
    print("  the alignment contrast shrinks. A design point is only worth adopting if BOTH this")
    print("  table and the contrast tables hold. Do not read this one alone.")
    print("=" * 96)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "perf_sweep.json").write_text(
        json.dumps({"copula": COPULA, "floor": FLOOR, "steps": STEPS,
                    "restarts": RESTARTS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

"""The trade-off that is the paper: expressivity vs. alignment.

A Gaussian copula scores 0.0060 on our dependency metric. Our trained CDG circuit scores 0.0694,
and no depth-1 CDG circuit can ever score below 0.0331 (Corollary 1: 72 of 120 pairs are outside
the light cone and are forced to exactly zero). So on THIS metric, the quantum model loses to a
textbook classical baseline by construction.

The obvious response is "make the circuit deeper". This script measures exactly what that buys and
exactly what it costs, because the two move in opposite directions:

  deeper  ->  the reach radius 2L grows  ->  more pairs become representable  ->  LOWER bound
  deeper  ->  the reach radius 2L grows  ->  the topology forbids less        ->  the CDG stops
                                                                                 differing from a
                                                                                 permutation

At L=3 every pair is within reach of every graph, and the CDG is *literally identical* to a random
permutation as far as expressible dependency mass is concerned (z = 0.00, RESULTS_precheck.md).
The circuit could then compete on the metric — and the entire scientific claim would be gone.

This prints both columns side by side. It is the honest statement of what the design can and
cannot be, and it belongs in the paper as a table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from eval_dep import fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]

# measured elsewhere, quoted here so the trade-off can be read in one place
Z_ALIGN = {1: 3.79, 2: 1.41, 3: 0.00}          # RESULTS_precheck.md
CLASSICAL = 0.0060                              # scripts/baseline_classical.py
ORACLE = 0.0055


def main() -> None:
    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    z = np.load(PROCESSED / "cdg.npz", allow_pickle=True)
    G = nx.Graph()
    G.add_nodes_from(range(N_FEAT))
    G.add_edges_from([tuple(e) for e in z["E_fit"]])

    zr = fisher_z(partial_corr_c(X, C))
    az = np.abs(zr)
    floor = float(np.mean([az[i, j] for i, j in ALL_PAIRS]))

    print("=" * 96)
    print("The trade-off: every step of depth buys expressivity and sells the hypothesis")
    print("=" * 96)
    print(f"  floor (no dependency at all)      {floor:.4f}")
    print(f"  Gaussian copula (classical)       {CLASSICAL:.4f}")
    print(f"  empirical resample (oracle)       {ORACLE:.4f}")
    print()
    print(f"  {'L':>2} {'reach':>6} {'pairs in cone':>14} {'Corollary 1 bound':>18} "
          f"{'alignment z':>12}   verdict")
    print("  " + "-" * 84)

    for L in (1, 2, 3):
        reach = 2 * L
        inc = [p for p in ALL_PAIRS
               if nx.has_path(G, *p) and nx.shortest_path_length(G, *p) <= reach]
        out = [p for p in ALL_PAIRS if p not in inc]
        # the error the circuit MUST pay: the true |z| of every pair it cannot reach
        bound = float(np.sum([az[i, j] for i, j in out]) / len(ALL_PAIRS))
        zz = Z_ALIGN[L]
        if L == 1:
            v = "hypothesis testable, cannot compete on the metric"
        elif L == 2:
            v = "competitive-ish, effect already 63% gone"
        else:
            v = "CDG == permuted. Nothing left to test."
        print(f"  {L:>2} {reach:>6} {len(inc):>10}/120 {bound:>18.4f} {zz:>+12.2f}   {v}")

    print()
    print("=" * 96)
    print("  Read the two middle columns together. They move in opposite directions, and that is")
    print("  not an artefact — it is the same fact stated twice. The light cone is what makes the")
    print("  CDG hypothesis TESTABLE (a wrong graph is provably unable to express a dependency it")
    print("  should have) and it is also what caps the model's accuracy. You cannot keep one and")
    print("  drop the other.")
    print()
    print("  So the paper cannot be 'a better tabular generator'. On the dependency metric a")
    print("  Gaussian copula beats us 11x, and it will keep beating us at L=1 no matter how good")
    print("  the optimizer gets. The paper is: WHERE you put a small, fixed budget of entanglement")
    print("  decides what the circuit can express, a clinically-derived graph beats every")
    print("  resource-matched control at deciding that, and the effect vanishes exactly as the")
    print("  light-cone theory predicts when you make the circuit deep enough not to care.")
    print("=" * 96)


if __name__ == "__main__":
    main()

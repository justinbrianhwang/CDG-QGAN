"""How much dependency signal does the real cohort actually contain, and where is it?

The floor is 0.0985 on real data — and the floor is just the mean |Fisher-z| of the true
conditional dependency structure, because a zero-dependency model's error on a pair IS |z_true|.
So the floor is not an arbitrary bar: it is a direct readout of how much signal exists.

If the signal is concentrated on a few pairs and near zero on the other hundred, then the way to
LOSE to the floor is not to miss the strong pairs — it is to manufacture correlation on the weak
ones. A circuit whose RZZ angles sit at a moderate value produces a dependency on every edge in
its cone, whether or not the data wants one there. This script measures the target the model has
to hit, before we ask whether it hits it.

Also reports the sampling-noise level (the score of the real data against itself, resampled),
which is the ceiling: no model can do better than this on a finite sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]


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
    E = [tuple(sorted(e)) for e in G.edges()]
    NON = [p for p in ALL_PAIRS if p not in E]
    # pairs a depth-1 circuit on the CDG can touch at all: d_G(u,v) <= 2
    IN_CONE = [p for p in ALL_PAIRS
               if nx.has_path(G, *p) and nx.shortest_path_length(G, *p) <= 2]
    OUT_CONE = [p for p in ALL_PAIRS if p not in IN_CONE]

    zr = fisher_z(partial_corr_c(X, C))
    az = np.abs(zr)

    def m(pairs):
        return float(np.mean([az[i, j] for i, j in pairs]))

    print("=" * 88)
    print("How much conditional dependency is actually in the cohort?")
    print("=" * 88)
    print(f"  n = {len(df):,}   ·   CDG: {len(E)} edges, {len(IN_CONE)} pairs inside the L=1 cone")
    print()
    print("  mean |Fisher-z| of the TRUE conditional dependency  (= the error a zero-dependency")
    print("  model makes on that group, i.e. its floor):")
    print(f"    all 120 pairs        {m(ALL_PAIRS):.4f}     <- this IS the floor")
    print(f"    {len(E):>3} CDG edges         {m(E):.4f}")
    print(f"    {len(NON):>3} non-edges         {m(NON):.4f}")
    print(f"    {len(IN_CONE):>3} inside the cone   {m(IN_CONE):.4f}")
    print(f"    {len(OUT_CONE):>3} outside the cone  {m(OUT_CONE):.4f}   <- forced to 0 by Corollary 1;")
    print("                                     the model CANNOT err here, and CANNOT help here")
    print()

    # the strongest pairs, and whether the circuit can reach them
    tri = [(az[i, j], i, j) for i, j in ALL_PAIRS]
    tri.sort(reverse=True)
    print("  the 12 strongest conditional dependencies, and their CDG distance:")
    print(f"    {'pair':<28} {'|z|':>7} {'d_G':>5}  reachable at L=1?")
    for v, i, j in tri[:12]:
        d = nx.shortest_path_length(G, i, j) if nx.has_path(G, i, j) else 99
        ok = "yes" if d <= 2 else "NO"
        print(f"    {names[i] + '—' + names[j]:<28} {v:>7.4f} {d:>5}  {ok}")
    print()

    # sampling noise: the real data against a resample of itself. No model can beat this.
    noise = []
    for s in range(5):
        r = np.random.default_rng(s)
        idx = r.integers(0, len(X), 20000)
        noise.append(dep_error(zr, fisher_z(partial_corr_c(X[idx], C[idx])), ALL_PAIRS))
    print(f"  sampling noise at n=20,000 (real data vs itself)  : {np.mean(noise):.4f}")
    print(f"  floor  (a model that creates no dependency)       : {m(ALL_PAIRS):.4f}")
    print()
    print("  The gap between those two lines is the ENTIRE room a model has to work in.")
    print("=" * 88)


if __name__ == "__main__":
    main()

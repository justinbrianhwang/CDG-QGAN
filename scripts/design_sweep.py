"""Redesign: can the circuit be made competitive WITHOUT killing the hypothesis?

The problem
-----------
A Gaussian copula scores 0.0060 on our dependency metric. A depth-1 CDG circuit can never score
below 0.0331, because 72 of the 120 pairs lie outside its light cone and Corollary 1 forces them to
exactly zero. So at the current design we lose to a textbook baseline by construction.

The obvious lever — DEPTH — does not work. It buys reach and sells the hypothesis at the same rate:

    L=1: bound 0.0331, alignment z = +3.79     <- testable, uncompetitive
    L=2: bound 0.0068, alignment z = +1.41     <- competitive-ish, effect 63% gone
    L=3: bound 0.0000, alignment z =  0.00     <- CDG is literally identical to a permutation

At L=3 every graph reaches every pair, so there is nothing left to test.

The lever this script tests: DENSITY
------------------------------------
The light cone at L=1 has reach radius 2. For a 16-node graph, EVERY pair can be within distance 2
of a graph with maximum degree 4 (Moore bound: 1 + d + d(d-1) = d^2+1 >= 16 needs d >= 4). Our CDG
has max degree 3 — a hard cap set in v2 §7.6, not by the data.

Density is a fundamentally different lever from depth:

  depth   widens the cone for EVERY graph equally, so the CDG and a random permutation converge.
          The control stops being a control.
  density adds edges at CLINICALLY CHOSEN positions. A permutation of a denser graph still has to
          put those edges somewhere, and the question "are they in the right place?" survives.

So density may lower the bound while keeping the alignment effect. Or it may not — a denser graph
also has a smaller diameter, and a permutation of it reaches more pairs too. That is an empirical
question, and it is the one that decides whether this project can have both.

What is swept
-------------
`build_cdg.MAX_DEGREE`. Raising it lets the degree-constrained Kruskal keep adding edges IN WEIGHT
ORDER — i.e. the next strongest partial correlations. The graph stays the clinical one; it just
stops being truncated by an arbitrary cap.

For each Delta we report, at L=1:
  |E|, max degree, diameter
  pairs inside the cone, and the Corollary 1 bound (the error the circuit MUST pay)
  the alignment z against a 2,000-draw isomorphic-permutation null   <- the hypothesis
  NISQ cost: max degree drives the number of RZZ layers a hardware schedule needs

We are looking for a Delta where the bound approaches the classical baseline AND z stays large.
If no such Delta exists, that is a real finding and the paper says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import build_cdg  # noqa: E402
from build_cdg import npn_residualize, partial_corr, stability  # noqa: E402
from eval_dep import fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_NULL = 2000
ALPHA = 0.05

CLASSICAL = 0.0060      # scripts/baseline_classical.py — the number to beat
FLOOR = 0.0986


def mass(G: nx.Graph, az: np.ndarray, L: int) -> tuple[float, int]:
    """Expressible dependency mass and the bound: what the circuit CANNOT reach, it pays for."""
    inc, out = 0.0, 0.0
    n_in = 0
    for i, j in ALL_PAIRS:
        d = nx.shortest_path_length(G, i, j) if nx.has_path(G, i, j) else 99
        if d <= 2 * L:
            inc += az[i, j]
            n_in += 1
        else:
            out += az[i, j]
    return inc, n_in, out / len(ALL_PAIRS)


def alignment_z(G: nx.Graph, az: np.ndarray, L: int, rng) -> tuple[float, float]:
    """CDG's expressible mass against an isomorphic-permutation null (relabel the nodes only)."""
    m_cdg, _, _ = mass(G, az, L)
    null = []
    nodes = list(G.nodes())
    for _ in range(N_NULL):
        perm = rng.permutation(nodes)
        H = nx.relabel_nodes(G, {nodes[k]: perm[k] for k in range(len(nodes))}, copy=True)
        m, _, _ = mass(H, az, L)
        null.append(m)
    null = np.array(null)
    z = (m_cdg - null.mean()) / (null.std() + 1e-12)
    pct = float((null < m_cdg).mean() * 100)
    return z, pct


def main() -> None:
    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)

    # the target the metric scores against
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    az = np.abs(fisher_z(partial_corr_c(Xs, C)))

    # the graph is estimated in the same space (build_cdg)
    Xr = npn_residualize(X, C)
    rho = partial_corr(Xr, ALPHA)
    stab = stability(Xr, ALPHA, build_cdg.N_BOOTSTRAP, np.random.default_rng(20260711))

    print("=" * 104)
    print("Design sweep — can DENSITY buy expressivity without selling the hypothesis?")
    print("=" * 104)
    print(f"  floor {FLOOR:.4f}   ·   Gaussian copula (classical) {CLASSICAL:.4f}   "
          f"·   L = 1 throughout")
    print()
    print(f"  {'Δ':>2} {'|E|':>4} {'maxdeg':>7} {'diam':>5} {'in cone':>9} "
          f"{'bound':>8} {'vs classical':>13} {'z':>7} {'pct':>6}")
    print("  " + "-" * 84)

    rows = []
    for delta in (3, 4, 5, 6):
        build_cdg.MAX_DEGREE = delta
        rng = np.random.default_rng(20260711)
        try:
            G, E_fit, E_hold = build_cdg.build_graph(rho, stab, names, rng)
        except RuntimeError as e:
            print(f"  {delta:>2}  build failed: {e}")
            continue

        _, n_in, bound = mass(G, az, 1)
        z, pct = alignment_z(G, az, 1, np.random.default_rng(7))
        md = max(dict(G.degree()).values())
        diam = nx.diameter(G) if nx.is_connected(G) else 99
        rows.append((delta, G.number_of_edges(), md, diam, n_in, bound, z, pct))
        print(f"  {delta:>2} {G.number_of_edges():>4} {md:>7} {diam:>5} {n_in:>6}/120 "
              f"{bound:>8.4f} {bound / CLASSICAL:>12.1f}x {z:>+7.2f} {pct:>5.1f}%")

    print()
    print("=" * 104)
    print("  What to look for:")
    print("    bound  -> must approach 0.0060 for the model to be competitive at all")
    print("    z      -> must stay well above ~2 for the CDG hypothesis to remain testable")
    print()
    print("  If both hold at some Δ, the redesign works and that Δ is the new design point.")
    print("  If z collapses as soon as the bound falls, then expressivity and alignment are the")
    print("  same quantity in this architecture, and NO choice of graph can give us both — which")
    print("  is itself the paper's central result, and it must be reported as such rather than")
    print("  tuned around.")
    print("=" * 104)


if __name__ == "__main__":
    main()

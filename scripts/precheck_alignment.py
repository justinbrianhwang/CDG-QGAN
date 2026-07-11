"""[Decisive precheck] Is the CDG alignment even structurally capable of producing an effect?

The verdict is reached with zero training, from the graph and the true partial correlations
alone. This is the gate that must be passed before spending any GPU time.

--------------------------------------------------------------------------
The key statistic: the expressible dependency mass

    M(G, L) = sum_{(u,v)} |rho_true(u,v)| * 1[ d_G(u,v) <= 2L ]

By Corollary 1, any pair with d_G(u,v) > 2L has a conditional covariance of exactly 0.
The total dependency a depth-L circuit can express is therefore given by the expression
above. If the CDG's claim is correct, M(CDG, L) must sit above the distribution obtained
from random permutations.

A permutation is isomorphic, so the number of nodes, the number of edges, the degree
sequence, the number of triangles and the clustering coefficient are all preserved. The only
thing that changes is *which clinical pair sits inside which triangle*. This test therefore
measures exactly one thing: alignment with clinical meaning.

--------------------------------------------------------------------------
Verdict

  CDG's M lies in the upper tail of the permutation null  -> proceed with the experiment
  CDG's M sits in the middle of the null                  -> no effect. The design must change.

If we lose this test, no amount of GPU burn will ever produce Delta_HDE < 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import paths  # noqa: E402

N_PERM = 5000


def dist_matrix(G: nx.Graph, p: int) -> np.ndarray:
    D = np.full((p, p), 99)
    for u, lengths in nx.all_pairs_shortest_path_length(G):
        for v, d in lengths.items():
            D[u, v] = d
    return D


def mass(D: np.ndarray, R: np.ndarray, L: int) -> float:
    """The dependency mass expressible at depth L."""
    reach = (D <= 2 * L) & (D > 0)
    return float((np.abs(R) * reach).sum() / 2.0)  # symmetric, hence /2


def main() -> None:
    d = np.load(paths.PROCESSED / "cdg.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    rho = d["rho"]
    E_fit = [tuple(e) for e in d["E_fit"]]
    p = len(names)

    G = nx.Graph()
    G.add_nodes_from(range(p))
    G.add_edges_from(E_fit)

    R = np.abs(rho.copy())
    np.fill_diagonal(R, 0.0)
    total = R.sum() / 2.0

    print("=" * 78)
    print("Precheck: does the CDG alignment increase the expressible dependency mass?")
    print("=" * 78)
    print(f"  nodes {p}  edges {G.number_of_edges()}  max degree {max(dict(G.degree()).values())}  "
          f"diameter {nx.diameter(G)}  clustering {nx.average_clustering(G):.3f}")
    print(f"  total dependency mass sum|rho| = {total:.2f}")
    print()

    rng = np.random.default_rng(20260711)
    D_cdg = dist_matrix(G, p)

    for L in (1, 2, 3):
        m_cdg = mass(D_cdg, R, L)

        null = np.empty(N_PERM)
        for k in range(N_PERM):
            perm = rng.permutation(p)
            H = nx.Graph()
            H.add_nodes_from(range(p))
            H.add_edges_from((int(perm[u]), int(perm[v])) for u, v in G.edges())
            null[k] = mass(dist_matrix(H, p), R, L)

        pct = float((null < m_cdg).mean() * 100)
        pval = float((null >= m_cdg).mean())
        z = (m_cdg - null.mean()) / (null.std() + 1e-12)

        n_reach = int(((D_cdg <= 2 * L) & (D_cdg > 0)).sum() / 2)
        n_pairs = p * (p - 1) // 2

        print(f"  L={L}  (reach radius {2*L})")
        print(f"    reachable pairs      : {n_reach}/{n_pairs}")
        print(f"    CDG mass             : {m_cdg:.3f}  ({m_cdg/total*100:.1f}% of total)")
        print(f"    permutation null     : {null.mean():.3f} +- {null.std():.3f}   "
              f"[{null.min():.3f}, {null.max():.3f}]")
        print(f"    percentile of the CDG: {pct:.1f}%   z={z:+.2f}   p={pval:.4f}")

        if n_reach == n_pairs:
            verdict = "MEANINGLESS - every pair is reachable. The topology imposes no constraint at all."
        elif pval < 0.05:
            verdict = "PASS - the CDG is in the upper tail of the null. An alignment effect can exist."
        else:
            verdict = "FAIL - the CDG is indistinguishable from a random permutation."
        print(f"    verdict: {verdict}")
        print()

    print("=" * 78)
    print("  Interpretation")
    print("=" * 78)
    print("  This test never runs the circuit once. From the graph structure and the true")
    print("  partial correlations alone, it decides whether the confirmatory experiment has")
    print("  any chance of succeeding.")
    print()
    print("  On FAIL, no amount of GPU burn will ever produce Delta_HDE < 0.")
    print("  In that case the following must be considered:")
    print("    - Lower the circuit depth L (L=1). The narrower the reach radius, the more")
    print("      discriminative the topology.")
    print("    - Reduce the number of edges to grow the diameter (right now the graph is")
    print("      near-3-regular, so the diameter is small).")
    print("    - Abandon the held-out scheme and evaluate on all 120 pairs (false positives")
    print("      included).")
    print("    - Check whether the CDG really does contain clinical clusters (triangles).")


if __name__ == "__main__":
    main()

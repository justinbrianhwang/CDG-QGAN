"""Control-graph construction (CDG-QGAN v2 §7.8, §9) + extra controls added per review.

Confirmatory controls:
  - isomorphic permuted-CDG : permutes the adjacency only; the feature-qubit
    assignment stays fixed (v2 §5.3)
  - distance-matched permuted-CDG : see [review B] below

Mechanism controls:
  - degree-preserving rewired
  - ring / ring-with-chords
  - no-entanglement

[review B] Why a distance-matched permutation is necessary
----------------------------------------------------------
By Corollary 1, if d_G(u,v) > 2L the conditional covariance is exactly 0.
But the held-out clinical pairs sit at roughly distance 2 in the CDG (they share a
common neighbor), while a random permutation scatters them out to distance 3-5.
So at L=1, Delta_HDE < 0 is not an "experimental result" — it is already fixed by
the combinatorics of the graph. I.e. a test that cannot fail.

The distance-matched permutation forces the distance distribution of the held-out
pairs to be the same as in the CDG. That cancels out the "being close" advantage,
and the only difference left is whether the exact edges the CDG picked really carry
the clinical structure. That is an informative test whose outcome is not known in
advance.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def _dist_profile(G: nx.Graph, pairs: list[tuple[int, int]]) -> tuple[int, ...]:
    """Graph-distance distribution of the given pairs (as a sorted tuple)."""
    out = []
    for u, v in pairs:
        out.append(nx.shortest_path_length(G, u, v) if nx.has_path(G, u, v) else 99)
    return tuple(sorted(out))


def isomorphic_permuted(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Permute the adjacency only (v2 §5.3).

    Preserves the node count, edge count, degree sequence, diameter, isomorphism
    class, gate count and parameter count. The only thing that changes is which
    pairs of clinical features are close to each other.
    Qubit u still generates feature x_u (the feature-qubit assignment is fixed).
    """
    n = G.number_of_nodes()
    perm = rng.permutation(n)
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from((int(perm[u]), int(perm[v])) for u, v in G.edges())
    return H


def distance_matched_permuted(
    G: nx.Graph,
    holdout: list[tuple[int, int]],
    rng: np.random.Generator,
    n_tries: int = 20000,
) -> tuple[nx.Graph, bool]:
    """[review B] Find a permutation whose held-out pair distance distribution matches the CDG.

    Returns: (graph, whether the match was exact). If no exact match is found, the
    closest one.
    """
    target = _dist_profile(G, holdout)
    best, best_cost = None, np.inf

    for _ in range(n_tries):
        H = isomorphic_permuted(G, rng)
        prof = _dist_profile(H, holdout)
        if prof == target:
            return H, True
        cost = sum(abs(a - b) for a, b in zip(prof, target))
        if cost < best_cost:
            best, best_cost = H, cost

    return best, False


def degree_preserving_rewire(G: nx.Graph, rng: np.random.Generator, n_swaps: int = 200) -> nx.Graph:
    """double-edge swap. Keeps the degree sequence and edge count, changes the path structure (v2 §7.8)."""
    H = G.copy()
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 50, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass  # if there are too few swappable combinations, take as many as possible
    return H


def ring_with_chords(n: int, n_edges: int, rng: np.random.Generator) -> nx.Graph:
    """Uniform-topology baseline with the resources (edge count) matched (v2 §9.2)."""
    H = nx.cycle_graph(n)
    extra = n_edges - H.number_of_edges()
    if extra > 0:
        cand = [(i, j) for i in range(n) for j in range(i + 2, n) if not H.has_edge(i, j)]
        rng.shuffle(cand)
        H.add_edges_from(cand[:extra])
    return H


def no_entanglement(n: int) -> nx.Graph:
    """E = empty set. A quantum model that generates only local marginals (v2 §7.8)."""
    H = nx.Graph()
    H.add_nodes_from(range(n))
    return H


def summarize(name: str, G: nx.Graph, holdout: list[tuple[int, int]], L_list=(1, 2)) -> str:
    """Check that a control really preserves the resources + whether the held-out pairs are reachable."""
    deg = dict(G.degree())
    prof = _dist_profile(G, holdout) if G.number_of_edges() else tuple([99] * len(holdout))
    arr = np.array(prof)
    reach = "  ".join(f"L={L}:{int((arr <= 2*L).sum())}/{len(arr)}" for L in L_list)
    conn = nx.is_connected(G) if G.number_of_edges() else False
    return (f"  {name:<26} |E|={G.number_of_edges():>2}  maxdeg={max(deg.values()) if deg else 0}  "
            f"connected={str(conn):<5}  held-out reach {reach}")


def build_all(G_cdg: nx.Graph, holdout: list[tuple[int, int]], seed: int = 20260711,
              n_perms: int = 3) -> dict[str, nx.Graph]:
    """Build all confirmatory / mechanism controls at once."""
    rng = np.random.default_rng(seed)
    n, m = G_cdg.number_of_nodes(), G_cdg.number_of_edges()

    out: dict[str, nx.Graph] = {"cdg": G_cdg}
    for k in range(n_perms):
        out[f"permuted_{k}"] = isomorphic_permuted(G_cdg, rng)
    for k in range(n_perms):
        H, exact = distance_matched_permuted(G_cdg, holdout, rng)
        out[f"distmatched_{k}"] = H
        if not exact:
            print(f"  [warning] distance-matched permutation {k}: no exact match, using the closest one")
    out["rewired"] = degree_preserving_rewire(G_cdg, rng)
    out["ring"] = ring_with_chords(n, m, rng)
    out["no_entangle"] = no_entanglement(n)
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    d = np.load(Path("data/processed/cdg.npz"), allow_pickle=True)
    names = list(d["names"])
    E_fit = [tuple(e) for e in d["E_fit"]]
    holdout = [tuple(e) for e in d["E_holdout"]]

    G = nx.Graph()
    G.add_nodes_from(range(len(names)))
    G.add_edges_from(E_fit)

    print("=" * 78)
    print("Control graphs  (same resources, only the clinical alignment is destroyed)")
    print("=" * 78)
    graphs = build_all(G, holdout)
    print()
    for k, H in graphs.items():
        print(summarize(k, H, holdout))
    print()
    print("  Key point: permuted must reach fewer held-out pairs than the CDG, while")
    print("             distmatched must reach the same number. If the CDG still wins")
    print("             at parity, that is the clinical structure, not the distance")
    print("             layout. [review B]")

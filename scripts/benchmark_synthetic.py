"""Controlled synthetic-graph benchmark (CDG-QGAN v2 §10).

With real clinical data we do not know the true graph. Here we start out knowing it.
-> We can verify, without MIMIC, that the theory matches the implementation and that
   the confirmatory design actually works.

Teacher:
  1. A known sparse precision matrix Omega* (modular graph — mimics clinical clusters)
  2. Gaussian latent ~ N(0, Omega*^-1)
  3. A different nonlinear monotone transform per dimension -> non-normal marginals
     (nonparanormal)

Comparison (all with identical resources: edge count, depth, parameters, critic, loss,
number of steps):
  aligned      : uses the true graph as the circuit topology
  permuted     : isomorphic permutation — destroys only the clinical alignment
  distmatched  : [review B] permutation matched on the distance distribution as well
  rewired      : degree-preserving rewire
  no_entangle  : RZZ removed

Evaluation [review B-2]:
  primary = conditional dependency error over all 120 pairs (false negatives + false
  positives)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from graphs import degree_preserving_rewire, distance_matched_permuted, isomorphic_permuted  # noqa: E402
from train import Cfg, dependency_error, generate, train  # noqa: E402

N_FEAT = 16
N_TRAIN = 20000
N_SYN = 20000
SEEDS = [0, 1, 2]
DEPTH = 1  # [findings] L=1 is the only operating point. At L=3 every pair becomes reachable, so it is meaningless.


def teacher_graph(rng) -> nx.Graph:
    """Modular graph: 4 clusters (4 nodes each) + bridges between clusters.

    Mimics real clinical structure — local clusters such as circulatory / electrolyte /
    renal / hematologic.
    """
    G = nx.Graph()
    G.add_nodes_from(range(N_FEAT))
    for b in range(4):  # build a triangle inside each block (guarantees distance-2 paths)
        o = 4 * b
        G.add_edges_from([(o, o + 1), (o + 1, o + 2), (o, o + 2), (o + 2, o + 3)])
    G.add_edges_from([(3, 4), (7, 8), (11, 12)])  # bridges between blocks
    return G


def teacher_data(G: nx.Graph, n: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate nonparanormal data from the true graph. Returns: (X, C, Omega)."""
    p = N_FEAT
    Om = np.eye(p) * 1.0
    for u, v in G.edges():
        w = rng.uniform(0.35, 0.55) * rng.choice([-1, 1])
        Om[u, v] = Om[v, u] = w
    Om += np.eye(p) * (abs(np.linalg.eigvalsh(Om).min()) + 0.15)  # ensure positive definiteness

    Sigma = np.linalg.inv(Om)
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)

    g = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    # Per-dimension nonlinear monotone transform -> non-normal marginals
    X = np.column_stack([
        np.sign(g[:, j]) * np.abs(g[:, j]) ** (1.0 + 0.5 * (j % 3)) if j % 3 else np.exp(0.5 * g[:, j])
        for j in range(p)
    ])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    y = (rng.random(n) < 0.1).astype(float)  # condition (mimics the mortality label)
    C = np.column_stack([y, rng.normal(0, 1, n), rng.random(n) < 0.5, rng.integers(0, 3, n)]).astype(float)
    return X, C, Om


def main() -> None:
    rng = np.random.default_rng(20260711)
    G_true = teacher_graph(rng)
    X, C, _ = teacher_data(G_true, N_TRAIN, rng)

    print("=" * 78)
    print("Controlled synthetic-graph benchmark (v2 §10)")
    print("=" * 78)
    print(f"  teacher: {N_FEAT} nodes {G_true.number_of_edges()} edges  "
          f"diameter={nx.diameter(G_true)}  clustering={nx.average_clustering(G_true):.3f}")
    print(f"  depth L={DEPTH} (reach radius {2*DEPTH})   n_train={N_TRAIN:,}  seeds={SEEDS}")

    n_reach = sum(1 for u in range(N_FEAT) for v in range(u + 1, N_FEAT)
                  if nx.shortest_path_length(G_true, u, v) <= 2 * DEPTH)
    print(f"  pairs reachable under aligned: {n_reach}/120")
    print()

    gr = np.random.default_rng(7)
    E_true = list(G_true.edges())
    holdout = [(u, v) for u in range(N_FEAT) for v in range(u + 1, N_FEAT)
               if not G_true.has_edge(u, v)
               and nx.shortest_path_length(G_true, u, v) == 2][:10]

    variants = {
        "aligned": G_true,
        "permuted": isomorphic_permuted(G_true, gr),
        "distmatched": distance_matched_permuted(G_true, holdout, gr)[0],
        "rewired": degree_preserving_rewire(G_true, gr),
        "no_entangle": nx.empty_graph(N_FEAT),
    }

    print(f"  {'model':<14} {'|E|':>4} {'120-pair error':>16}  (mean ± std, 3 seeds)")
    print("  " + "-" * 60)

    results = {}
    for name, Gv in variants.items():
        errs = []
        t0 = time.time()
        for s in SEEDS:
            cfg = Cfg(depth=DEPTH, seed=s)
            Gm, _ = train(X, C, list(Gv.edges()), cfg)
            Xs = generate(Gm, C, N_SYN, seed=s)
            errs.append(dependency_error(X, Xs))  # all 120 pairs [review B-2]
        results[name] = np.array(errs)
        print(f"  {name:<14} {Gv.number_of_edges():>4} "
              f"{errs and np.mean(errs):>10.4f} ± {np.std(errs):.4f}   ({time.time()-t0:.0f}s)",
              flush=True)

    print()
    print("=" * 78)
    print("Confirmatory contrasts")
    print("=" * 78)
    a = results["aligned"]
    for ref in ("permuted", "distmatched", "rewired", "no_entangle"):
        b = results[ref]
        delta = a.mean() - b.mean()
        print(f"  aligned - {ref:<12} = {delta:+.4f}   "
              f"{'aligned better' if delta < 0 else 'aligned worse'}")
    print()
    print("  Expected: aligned < permuted (the clinical-alignment effect)")
    print("            aligned <= distmatched means the effect goes beyond mere distance layout [review B]")
    print("            no_entangle being worst means entanglement really is what creates dependency")

    np.savez("results_benchmark.npz", **results)


if __name__ == "__main__":
    main()

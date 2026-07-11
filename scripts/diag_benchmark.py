"""Diagnosing the null result of the benchmark (no training required).

benchmark_synthetic.py produced 0.135 +- 0.009 for all four variants.
But we **do not know whether 0.135 is a good value or a bad one.** There is no baseline.

Here we compute two baselines without any training:

  [floor]  A model that creates no dependency whatsoever (independent per-column shuffle).
           The 120-pair error of a model that reproduces the true marginals perfectly but
           has exactly zero dependency.
           This is the score of "learned nothing at all".

  [ceil]   A fresh draw of the true data (same Omega, different seed).
           The best attainable score, containing only finite-sample noise.

And we split the error:
  - the 19 true-edge pairs   (false negatives: failing to create a dependency that should exist)
  - the 101 non-edge pairs   (false positives: inventing a dependency that should not exist)

If the trained model (0.135) is **worse** than the floor, then the model is not learning
dependency but spewing out spurious dependency, and the 120-pair mean metric is dominated
by that noise.
-> This must be resolved before running the confirmatory experiment (90 runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from train import partial_corr_matrix  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731


def split_error(zr: np.ndarray, zs: np.ndarray, G: nx.Graph):
    edge, non = [], []
    for i in range(N_FEAT):
        for j in range(i + 1, N_FEAT):
            (edge if G.has_edge(i, j) else non).append(abs(zs[i, j] - zr[i, j]))
    allp = edge + non
    return np.mean(allp), np.mean(edge), np.mean(non), len(edge), len(non)


def main() -> None:
    rng = np.random.default_rng(20260711)
    G = teacher_graph(rng)
    X, C, Om = teacher_data(G, N_TRAIN, rng)

    Rr = partial_corr_matrix(X)
    zr = Z(Rr)

    print("=" * 78)
    print("Benchmark diagnosis — is 0.135 a good value or a bad one?")
    print("=" * 78)

    # Magnitude of the true partial correlations — how much signal is there to capture at all
    e_true = [abs(Rr[i, j]) for i, j in G.edges()]
    n_true = [abs(Rr[i, j]) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)
              if not G.has_edge(i, j)]
    print(f"\n  Partial correlation |rho| of the true data")
    print(f"    19 edge pairs    : mean {np.mean(e_true):.3f}  (range {min(e_true):.3f}~{max(e_true):.3f})")
    print(f"    101 non-edge pairs: mean {np.mean(n_true):.3f}  (range {min(n_true):.3f}~{max(n_true):.3f})")
    print(f"    -> mean edge magnitude in Fisher-z = {np.mean(np.abs(Z(np.array(e_true)))):.3f}")

    print(f"\n  {'baseline':<34} {'120pair':>8} {'edge19':>8} {'nonedge101':>10}")
    print("  " + "-" * 64)

    # [floor] zero-dependency model: shuffle each column independently -> perfect marginals, zero dependency
    errs = []
    for s in range(5):
        r = np.random.default_rng(100 + s)
        Xi = np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])
        errs.append(split_error(zr, Z(partial_corr_matrix(Xi)), G))
    f = np.mean([e[0] for e in errs]), np.mean([e[1] for e in errs]), np.mean([e[2] for e in errs])
    print(f"  {'[floor] zero dependency (indep. shuffle)':<34} {f[0]:>8.4f} {f[1]:>8.4f} {f[2]:>10.4f}")

    # [ceil] resample the true distribution — finite-sample noise only
    errs = []
    for s in range(5):
        Xn, _, _ = teacher_data(G, N_TRAIN, np.random.default_rng(200 + s))
        errs.append(split_error(zr, Z(partial_corr_matrix(Xn)), G))
    c = np.mean([e[0] for e in errs]), np.mean([e[1] for e in errs]), np.mean([e[2] for e in errs])
    print(f"  {'[ceil]  true dist. resampled (noise only)':<34} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>10.4f}")

    print(f"  {'[obs]   trained model (benchmark)':<34} {0.1359:>8.4f} {'?':>8} {'?':>10}")

    print()
    print("=" * 78)
    print("Interpretation")
    print("=" * 78)
    print(f"  score of a model that creates no dependency at all = {f[0]:.4f}")
    print(f"  score of the trained model                         = 0.1359")
    if 0.1359 > f[0]:
        print()
        print("  >> The trained model is **worse** than the 'do-nothing model'.")
        print("     The model is not learning the true dependency — it is **spewing out spurious dependency**.")
        print("     The 120-pair mean is dominated by that noise (false positives), and the alignment effect is buried.")
        print("     -> Running the confirmatory experiment would yield nothing. Removing the cause comes first.")
    else:
        print()
        print("  >> The trained model is at least better than the floor. In that case the problem is signal dilution, not the metric.")
    print()
    print(f"  best attainable (noise floor) = {c[0]:.4f}")
    print(f"  floor - ceil = {f[0]-c[0]:.4f}  <- this interval is 'everything that can be gained by learning'.")
    print("     If this interval is narrow, no contrast will be statistically detectable.")


if __name__ == "__main__":
    main()

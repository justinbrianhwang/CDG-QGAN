"""What is the trained model actually producing (stage 2 of the benchmark null diagnosis).

Results from diag_benchmark.py:
    zero-dependency model = 0.0648
    trained model         = 0.1359   <- 2.1x worse

=> The model is not failing to learn the true dependency; it is **inventing dependency
   that does not exist**.

Hypothesis: it is because of the condition vector c.
    The generator is x~_u = h_u(q_u, c), and c is shared across all features.
    -> An unconditional partial correlation, measured without controlling for c, mixes in
       all of the correlation that c creates.
    -> The ground truth (the synthetic teacher) has no such component (since C was drawn
       independently of X).
    The CDG is defined as "conditional on c", yet the evaluation is unconditional
    = estimator mismatch.

Verification:
    [A] unconditional partial correlation  (the current train.dependency_error)
    [B] partial correlation after residualizing on c (matches the CDG definition)
    If the hypothesis is right, then under [B] the error drops below the floor and
    aligned/permuted separate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_SYN, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from train import Cfg, _npn, generate, partial_corr_matrix, train  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731
SEEDS = [0, 1, 2]


def partial_corr_resid(X: np.ndarray, C: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Partial correlation controlling for c — the same space as the CDG definition.

    After the nonparanormal transform, linearly regress on c (plus an intercept) and
    measure the partial correlation of the residuals.
    The same procedure build_cdg.py uses when it constructs the CDG.
    """
    Xn = _npn(X)
    D = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(D, Xn, rcond=None)
    R = Xn - D @ beta
    S = np.corrcoef(R, rowvar=False)
    P = np.linalg.inv(S + ridge * np.eye(S.shape[0]))
    d = np.sqrt(np.diag(P))
    M = -P / np.outer(d, d)
    np.fill_diagonal(M, 1.0)
    return M


def split(zr, zs, G):
    e = [abs(zs[i, j] - zr[i, j]) for i, j in G.edges()]
    nn = [abs(zs[i, j] - zr[i, j]) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)
          if not G.has_edge(i, j)]
    return float(np.mean(e + nn)), float(np.mean(e)), float(np.mean(nn))


def main() -> None:
    rng = np.random.default_rng(20260711)
    G = teacher_graph(rng)
    X, C, _ = teacher_data(G, N_TRAIN, rng)
    Gp = isomorphic_permuted(G, np.random.default_rng(7))

    zr_u = Z(partial_corr_matrix(X))          # unconditional ground truth
    zr_c = Z(partial_corr_resid(X, C))        # c-residualized ground truth

    # floor: zero-dependency model
    fl_u, fl_c = [], []
    for s in range(3):
        r = np.random.default_rng(100 + s)
        Xi = np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])
        Ci = C[r.permutation(len(C))]
        fl_u.append(split(zr_u, Z(partial_corr_matrix(Xi)), G))
        fl_c.append(split(zr_c, Z(partial_corr_resid(Xi, Ci)), G))

    print("=" * 88)
    print("What is the trained model producing?")
    print("=" * 88)
    print(f"  {'model':<22} {'[A] unconditional (current)':>30}   {'[B] c-residualized (CDG def.)':>30}")
    print(f"  {'':<22} {'120pair':>9}{'edge19':>9}{'nonedge101':>11}   "
          f"{'120pair':>9}{'edge19':>9}{'nonedge101':>11}")
    print("  " + "-" * 84)

    def row(name, a, b):
        print(f"  {name:<22} {a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>11.4f}   "
              f"{b[0]:>9.4f}{b[1]:>9.4f}{b[2]:>11.4f}", flush=True)

    row("floor (zero dependency)", np.mean(fl_u, 0), np.mean(fl_c, 0))

    for name, Gv in [("aligned", G), ("permuted", Gp), ("no_entangle", nx.empty_graph(N_FEAT))]:
        au, ac = [], []
        t0 = time.time()
        for s in SEEDS:
            Gm, _ = train(X, C, list(Gv.edges()), Cfg(depth=1, seed=s))
            Xs = generate(Gm, C, N_SYN, seed=s)
            Cs = C[np.random.default_rng(s).integers(0, len(C), N_SYN)]  # same seed rule as generate
            au.append(split(zr_u, Z(partial_corr_matrix(Xs)), G))
            ac.append(split(zr_c, Z(partial_corr_resid(Xs, Cs)), G))
        row(f"{name} ({time.time()-t0:.0f}s)", np.mean(au, 0), np.mean(ac, 0))

    print()
    print("  How to read this:")
    print("    If the 'nonedge101' column is large under [A] -> spurious dependency is being spewed out (false positives).")
    print("    If it shrinks under [B]      -> the cause is the shared condition c. Align the metric with the CDG definition.")
    print("    If it does not shrink under [B] -> the cause is not c but the training/circuit. Dig further.")
    print("    If aligned vs permuted separate under [B], the confirmatory design is back in business.")


if __name__ == "__main__":
    main()

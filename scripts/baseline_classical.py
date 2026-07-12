"""How do trivial classical baselines score on our own dependency metric?

Read this before writing any sentence that claims the quantum model is "better".

The internal contrasts (CDG vs permuted vs distmatched vs no_entangle) all compare the SAME
circuit with the edges moved around. They establish that *alignment matters*. They say nothing
about whether the model is competitive with a classical generator, because no classical generator
has ever been run against this metric.

This script runs the cheapest possible ones. It exists to make the comparison honest before a
reviewer does it for us.

The structural problem, stated up front
---------------------------------------
A depth-1 CDG circuit CANNOT beat 0.0331 on this metric — 72 of the 120 pairs lie outside its
light cone and Corollary 1 forces their conditional covariance to exactly zero, at every parameter
setting. A classical model has no light cone. It can fit all 120 pairs. The sampling-noise level
at n=20,000 is 0.0058.

So on THIS metric the quantum model is bounded away from the classical optimum by construction.
That is not a bug and it is not a result to hide: it is the price of the structural guarantee
(Prop. D-2 — the dependency can only come from 21 entangling angles, and only along the CDG).

The paper cannot claim "lower dependency error than classical baselines". What it can claim has to
be about structure, parameter count, and interpretability — and those claims must be made against
these numbers, in the open.

Baselines
---------
  gaussian_copula  : the metric's own model. npn -> residualize on c -> fit the residual
                     covariance -> sample from it -> invert the ranks. This is close to an oracle:
                     it fits exactly the object the metric measures. Expect it near sampling noise.
  independent      : the floor. Marginals right, zero dependency.
  empirical_resample: draw real rows with replacement. The ceiling — pure sampling noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from eval_dep import cond_basis, dep_error, fisher_z, npn, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_SYN = 20000

TRAINED_CDG = 0.0694      # WGAN-GP, 8000 steps, batch 512 (diag_fix3 / wp2 probe)
CEILING_CDG = 0.0437      # RESULTS_ceiling_real.md — GAN removed
BOUND = 0.0331            # Corollary 1 — no L=1 CDG circuit can go below this


def gaussian_copula_sample(X, C, n, rng):
    """Fit the metric's own model and sample from it. Near-oracle by construction."""
    Xn = npn(X)
    D = cond_basis(C)
    beta, *_ = np.linalg.lstsq(D, Xn, rcond=None)
    R = Xn - D @ beta
    S = np.cov(R, rowvar=False)

    idx = rng.integers(0, len(C), n)
    Cs = C[idx]
    Rs = rng.multivariate_normal(np.zeros(N_FEAT), S, size=n)
    Zs = cond_basis(Cs) @ beta + Rs          # back to nonparanormal space

    # invert the nonparanormal transform: map each column's ranks back onto the real marginal
    Xs = np.empty_like(Zs)
    for j in range(N_FEAT):
        u = norm.cdf((Zs[:, j] - Zs[:, j].mean()) / (Zs[:, j].std() + 1e-12))
        q = np.clip((u * len(X)).astype(int), 0, len(X) - 1)
        Xs[:, j] = np.sort(X[:, j])[q]
    return Xs, Cs


def main() -> None:
    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    zr = fisher_z(partial_corr_c(X, C))

    def score(Xs, Cs):
        return dep_error(zr, fisher_z(partial_corr_c(Xs, Cs)), ALL_PAIRS)

    rows = {}

    v = []
    for s in range(5):
        r = np.random.default_rng(200 + s)
        idx = r.integers(0, len(X), N_SYN)
        v.append(score(X[idx], C[idx]))
    rows["empirical resample (oracle)"] = (np.mean(v), np.std(v), "—")

    v = []
    for s in range(5):
        r = np.random.default_rng(300 + s)
        v.append(score(*gaussian_copula_sample(X, C, N_SYN, r)))
    rows["gaussian copula (classical)"] = (np.mean(v), np.std(v), f"{N_FEAT*(N_FEAT-1)//2} cov. params")

    v = []
    for s in range(5):
        r = np.random.default_rng(1000 + s)
        v.append(score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)]),
                       C[r.permutation(len(C))]))
    floor = float(np.mean(v))
    rows["independent (floor)"] = (floor, np.std(v), "0")

    print("=" * 92)
    print("Classical baselines on our own dependency metric — real MIMIC-IV")
    print("=" * 92)
    print(f"  n = {len(df):,}   ·   metric: conditional partial correlation, all 120 pairs")
    print()
    print(f"  {'model':<32} {'120-pair error':>16} {'dependency params':>20}")
    print("  " + "-" * 74)
    for k, (m, sd, p) in rows.items():
        print(f"  {k:<32} {m:>10.4f} ± {sd:.4f} {p:>20}")
    print()
    print(f"  {'CDG-QGAN, trained (WGAN-GP)':<32} {TRAINED_CDG:>10.4f}          "
          f"{'21 RZZ angles':>20}")
    print(f"  {'CDG-QGAN, ceiling (no GAN)':<32} {CEILING_CDG:>10.4f}          "
          f"{'21 RZZ angles':>20}")
    print(f"  {'CDG-QGAN, Corollary 1 bound':<32} {BOUND:>10.4f}          "
          f"{'unreachable by ANY L=1':>20}")
    print()
    print("=" * 92)
    gc = rows["gaussian copula (classical)"][0]
    print(f"  A Gaussian copula scores {gc:.4f}. The best a depth-1 CDG circuit could EVER score")
    print(f"  is {BOUND:.4f}, and our trained model scores {TRAINED_CDG:.4f}.")
    print()
    print("  So on this metric the quantum model LOSES to a textbook classical baseline, and it")
    print("  loses by construction, not by bad luck: 72 of the 120 pairs are outside the L=1 light")
    print("  cone and Corollary 1 forces them to exactly zero. The classical model has no cone.")
    print()
    print("  This must be stated plainly in the paper. The contribution is NOT 'lower dependency")
    print("  error'. It is that ALL of the dependency is produced by 21 entangling angles, confined")
    print("  to a clinically-specified graph, with a provable guarantee (Prop. D-2) that the ~2,000")
    print("  classical head parameters cannot create any of it — and that WHERE those 21 angles go")
    print("  is what decides the outcome (CDG beats distance-matched by 0.0262).")
    print("=" * 92)


if __name__ == "__main__":
    main()

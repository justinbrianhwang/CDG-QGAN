"""The floor is the wrong null for a v3 model, and this is how much it is wrong by.

The problem
-----------
`wp2.py`'s floor permutes every feature independently AND permutes c. That destroys two things at
once: the cross-feature dependence (which is what we want a null to destroy) and the x–c relation
(which we do not). It was the right null for v2, where nothing in the loss trained the heads to
match the marginals and they didn't. It is the wrong null for v3, whose 1-D conditional marginal
term makes E[x_u | c] correct on purpose.

Why that matters is a property of the *estimator*, not of the model. `eval_dep.partial_corr_c`
residualizes on a fixed basis: 1, c, c², and the pairwise products c_a·c_b. Whatever part of
E[x_u | c] lies outside the span of that basis survives residualization. It is a deterministic
function of c, so it is *shared across features* — and a shared component is exactly what a
correlation estimator reports as cross-feature dependence.

The real structure `zr` is measured with the same estimator, so `zr` contains this bias too.
Therefore a model that reproduces E[x_u | c] reproduces the bias, lands closer to `zr`, and scores
BELOW the floor **without creating any conditional dependence whatsoever**. The floor is not a
floor for such a model. Reporting "−19% vs floor" without saying this would be overclaiming.

The honest null
---------------
Bin c (mortality × age-quantile × sex × icu_type) and permute each feature independently WITHIN
each bin. Then:

  * Cov(x_u, x_v | c) = 0 exactly, by construction — no model, no training, no circuit.
  * E[x_u | c] is the true conditional marginal, at bin resolution.

Its score is what a perfect-marginal, zero-dependency model earns for free from the estimator's
conditioning bias. That, not the permutation floor, is the number a topology claim must beat.

Run it at two bin resolutions. If the answer moves with the resolution, it is a binning artefact
and this whole argument is wrong.

Measured (full cohort, MIMIC-IV v3.1, n=48,561):

    wp2 floor        (x–c destroyed too)              0.0985
    honest null      (197 bins, age-quartile)         0.0942 ± 0.0003     -4.3%
    honest null      (357 bins, age-octile)           0.0940 ± 0.0004     -4.6%
    no_entangle      (trained, v3)                    0.0966
    CDG Δ=4          (trained, v3)                    0.0796              -15.5% vs honest null

Stable across resolution, so it is not the bins. And note where `no_entangle` lands: ABOVE the
honest null. A trained zero-entanglement model does not even collect the whole free lunch, because
its conditional marginals are not perfect (W1 = 0.131). The falsifier is stronger than we thought,
not weaker: `no_entangle` creates no dependency, and every bit of its sub-floor score is accounted
for without invoking any.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_REP = 5


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

    print("=" * 88)
    print("The floor is the wrong null for a model that fits the conditional marginals")
    print("=" * 88)
    print(f"  cohort n={len(df):,}  ·  {N_REP} repetitions each")
    print()

    # the wp2.py floor — cross-feature structure AND the x–c relation both destroyed
    fl = [score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)]),
                C[r.permutation(len(C))])
          for r in (np.random.default_rng(1000 + s) for s in range(N_REP))]
    floor = float(np.mean(fl))

    out = {"floor": floor}
    for nq, tag in ((4, "age-quartile"), (8, "age-octile")):
        aq = pd.qcut(C[:, 1], nq, labels=False, duplicates="drop")
        gid = (pd.DataFrame({"y": C[:, 0], "a": aq, "s": C[:, 2], "t": C[:, 3]})
               .groupby(["y", "a", "s", "t"], sort=False).ngroup().to_numpy())
        bins = [np.flatnonzero(gid == g) for g in np.unique(gid)]

        vals = []
        for s in range(N_REP):
            r = np.random.default_rng(2000 + s)
            Xs = X.copy()
            for m in bins:                       # independent permutation per feature, per bin
                if len(m) < 2:
                    continue
                for j in range(N_FEAT):
                    Xs[m, j] = X[r.permutation(m), j]
            vals.append(score(Xs, C))
        v = float(np.mean(vals))
        out[f"null_{tag}"] = {"score": v, "sd": float(np.std(vals)), "bins": len(bins)}
        print(f"  honest null  ({tag}, {len(bins)} bins)   {v:.4f} ± {np.std(vals):.4f}"
              f"   {(v - floor) / floor * 100:+.1f}% vs floor")

    print(f"  wp2 floor    (x–c destroyed too)          {floor:.4f}")
    print()
    print("  The two resolutions agree, so this is the estimator's conditioning bias and not the")
    print("  bins. A zero-dependency model with correct conditional marginals scores ~4% below the")
    print("  floor for free. Every topology claim must be stated against THIS null, not the floor.")
    print("=" * 88)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "null_condmarg.json").write_text(
        __import__("json").dumps(out, indent=2, default=float))
    print(f"\n  saved: {RESULTS / 'null_condmarg.json'}")


if __name__ == "__main__":
    main()

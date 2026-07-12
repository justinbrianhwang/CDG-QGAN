"""WP-3 — the classical baselines, on the same metric, plus downstream utility.

Why this had to happen before any "our model is good" sentence
--------------------------------------------------------------
Every comparison so far has been INTERNAL: CDG vs permuted vs distance-matched vs no_entangle.
Those are all the same circuit with the edges moved. They establish that *where* the entanglement
goes matters. They say nothing about whether the model is any good, because no classical generator
had ever been run against this metric.

The first one we ran (`baseline_classical.py`) was brutal: a Gaussian copula scores 0.0060 and our
trained circuit scores 0.0694. But that comparison is also unfair in a way that flatters the
copula, and the unfairness has to be said out loud:

    **our dependency metric IS a Gaussian-copula quantity** — a partial correlation in
    nonparanormal space. A Gaussian copula fits exactly that object and nothing else. It is
    effectively the ORACLE for this metric (0.0060 vs the empirical resample's 0.0055), and no
    generative model can beat it here. Reporting "we lose to a Gaussian copula on partial
    correlations" is true, and about as informative as "we lose to the sample covariance".

So this script runs the baselines people actually publish against — CTGAN and TVAE — on the same
metric, and adds the metric that decides whether synthetic data is USEFUL:

  dependency error : the 120-pair conditional partial correlation (as everywhere else)
  TSTR             : Train on Synthetic, Test on Real. Fit a mortality classifier on synthetic
                     rows, score it on a held-out slice of REAL patients. This is what a synthetic
                     ICU dataset is FOR. It weights a dependency by how much it matters clinically,
                     which the unweighted 120-pair average does not.
  marginals        : mean 1-D Wasserstein distance, so a model cannot win TSTR by ignoring them.

Read the three together. A model that nails the partial correlations and produces clinically
implausible rows is not better than one that gets the strong dependencies right and drops the weak
ones — and the 120-pair average cannot tell those two apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from eval_dep import cond_basis, dep_error, fisher_z, npn, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_SYN = 20000
COND = ["y", "age", "sex", "icu_type"]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def gaussian_copula(Xtr, Ctr, n, rng):
    """The metric's own model. Near-oracle here — reported for calibration, not as a rival."""
    Xn = npn(Xtr)
    D = cond_basis(Ctr)
    beta, *_ = np.linalg.lstsq(D, Xn, rcond=None)
    R = Xn - D @ beta
    S = np.cov(R, rowvar=False)

    idx = rng.integers(0, len(Ctr), n)
    Cs = Ctr[idx]
    Zs = cond_basis(Cs) @ beta + rng.multivariate_normal(np.zeros(N_FEAT), S, size=n)
    Xs = np.empty_like(Zs)
    for j in range(N_FEAT):
        u = norm.cdf((Zs[:, j] - Zs[:, j].mean()) / (Zs[:, j].std() + 1e-12))
        q = np.clip((u * len(Xtr)).astype(int), 0, len(Xtr) - 1)
        Xs[:, j] = np.sort(Xtr[:, j])[q]
    return Xs, Cs


def _sdv_fit_sample(model_cls, Xtr, Ctr, n, epochs, seed):
    """CTGAN / TVAE. Conditioning is handled by giving them c as ordinary columns."""
    import torch
    torch.manual_seed(seed)
    names = [f.name for f in CORE16]
    df = pd.DataFrame(np.column_stack([Xtr, Ctr]), columns=names + COND)
    m = model_cls(epochs=epochs, verbose=False, cuda=True)
    m.fit(df, discrete_columns=["y", "sex", "icu_type"])
    s = m.sample(n)
    return s[names].to_numpy(float), s[COND].to_numpy(float)


def independent(Xtr, Ctr, n, rng):
    """The floor: marginals exact, dependency zero."""
    idx = rng.integers(0, len(Ctr), n)
    return np.column_stack([rng.permutation(Xtr[:, j])[:n] if n <= len(Xtr)
                            else Xtr[rng.integers(0, len(Xtr), n), j]
                            for j in range(N_FEAT)]), Ctr[idx]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def tstr(Xs, Cs, Xte, Cte, seed=0):
    """Train on synthetic, test on real. Predicts in-hospital mortality (c[:,0] = y)."""
    from xgboost import XGBClassifier
    ys = Cs[:, 0].astype(int)
    if ys.sum() < 20 or (1 - ys).sum() < 20:      # degenerate synthetic label
        return float("nan"), float("nan")
    # the classifier sees the 16 features plus the non-label conditions
    Fs = np.column_stack([Xs, Cs[:, 1:]])
    Fte = np.column_stack([Xte, Cte[:, 1:]])
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        random_state=seed, n_jobs=8, tree_method="hist")
    clf.fit(Fs, ys)
    p = clf.predict_proba(Fte)[:, 1]
    yte = Cte[:, 0].astype(int)
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def marginal_w1(Xs, Xr):
    return float(np.mean([wasserstein_distance(Xs[:, j], Xr[:, j]) for j in range(N_FEAT)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300, help="CTGAN/TVAE epochs")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(subset=names + COND[1:])
    X = df[names].to_numpy(float)
    C = df[COND].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # a real held-out slice — TSTR must be scored on patients no generator ever saw
    Xtr, Xte, Ctr, Cte = train_test_split(X, C, test_size=0.25, random_state=0,
                                          stratify=C[:, 0])

    zr = fisher_z(partial_corr_c(Xtr, Ctr))

    def dep(Xs, Cs):
        return dep_error(zr, fisher_z(partial_corr_c(Xs, Cs)), ALL_PAIRS)

    # the real data's own TSTR — the ceiling for utility (train on real, test on real)
    trtr = tstr(Xtr, Ctr, Xte, Cte)

    from ctgan import CTGAN, TVAE

    runners = {
        "gaussian copula (≈oracle here)": lambda s: gaussian_copula(
            Xtr, Ctr, N_SYN, np.random.default_rng(s)),
        "CTGAN": lambda s: _sdv_fit_sample(CTGAN, Xtr, Ctr, N_SYN, args.epochs, s),
        "TVAE": lambda s: _sdv_fit_sample(TVAE, Xtr, Ctr, N_SYN, args.epochs, s),
        "independent (floor)": lambda s: independent(
            Xtr, Ctr, N_SYN, np.random.default_rng(1000 + s)),
    }

    print("=" * 104)
    print("WP-3 — classical baselines, real MIMIC-IV")
    print("=" * 104)
    print(f"  train n={len(Xtr):,}   ·   held-out real test n={len(Xte):,}   ·   "
          f"{args.seeds} seeds   ·   CTGAN/TVAE {args.epochs} epochs")
    print(f"  TSTR ceiling (train on REAL, test on real): AUROC {trtr[0]:.4f}  AUPRC {trtr[1]:.4f}")
    print()
    print(f"  {'model':<32} {'dep. error':>11} {'TSTR AUROC':>11} {'TSTR AUPRC':>11} "
          f"{'marginal W1':>12}")
    print("  " + "-" * 84)

    out = {"tstr_ceiling": trtr}
    for name, run in runners.items():
        de, au, ap_, w1 = [], [], [], []
        t0 = time.time()
        for s in range(args.seeds):
            Xs, Cs = run(s)
            de.append(dep(Xs, Cs))
            a, p = tstr(Xs, Cs, Xte, Cte, s)
            au.append(a); ap_.append(p)
            w1.append(marginal_w1(Xs, Xtr))
        out[name] = {"dep": float(np.mean(de)), "auroc": float(np.nanmean(au)),
                     "auprc": float(np.nanmean(ap_)), "w1": float(np.mean(w1))}
        print(f"  {name:<32} {np.mean(de):>11.4f} {np.nanmean(au):>11.4f} "
              f"{np.nanmean(ap_):>11.4f} {np.mean(w1):>12.4f}   ({time.time()-t0:.0f}s)",
              flush=True)

    print()
    print("  CDG-QGAN (Δ=3, L=1, trained)          0.0694   <- to be filled from the WP-2 run")
    print("  CDG-QGAN (Δ=3, L=1, ceiling, no GAN)  0.0437")
    print("  CDG-QGAN Corollary 1 bound (Δ=3)      0.0331   <- unreachable by ANY L=1 Δ=3 circuit")
    print("  CDG-QGAN Corollary 1 bound (Δ=4)      0.0125   <- the redesign (design_sweep.py)")
    print()
    print("=" * 104)
    print("  How to read this table, honestly:")
    print("   - The Gaussian copula is the ORACLE for the dependency column, not a rival. The")
    print("     metric is a copula quantity; of course a copula wins it. It is here for calibration.")
    print("   - CTGAN/TVAE are the real baselines. If they beat us on dependency error too, then")
    print("     the quantum model has no performance argument at L=1 and the paper must rest on")
    print("     the structural result (Prop. D-2) and the alignment effect — and say so plainly.")
    print("   - TSTR is what a synthetic ICU dataset is FOR. A model can lose the 120-pair average")
    print("     (which weights a |z|=0.02 pair the same as a |z|=1.06 pair) and still be the more")
    print("     useful generator, because the CDG puts the STRONG dependencies inside the cone by")
    print("     construction. If that is what happens, it is a finding, not an excuse — but it only")
    print("     counts if it was predicted in advance, and it is being written down here, first.")
    print("=" * 104)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "wp3_baselines.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  saved: {RESULTS / 'wp3_baselines.json'}")


if __name__ == "__main__":
    main()

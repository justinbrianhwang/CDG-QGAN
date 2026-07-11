"""Diagnosis: is an arithmetic identity hiding inside the feature set?

The problem:
    MAP ~= (SBP + 2*DBP)/3 is not a clinical dependency but an **arithmetic identity**.
    If we keep it as a generated variable, a substantial part of the CDG's advantage comes
    from "it recovered the identity", and a reviewer will ask "isn't that arithmetic rather
    than clinical structure discovery?"

    Hematocrit ~= 3 * Hemoglobin is the same story (it is on the substitute-variable list).

This script measures the actual explanatory power (R^2) of each candidate identity, so that
the data decides which variable must be dropped from the generation set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def main() -> None:
    df = pd.read_parquet(paths.PROCESSED / "cohort_v31.parquet")
    print("=" * 74)
    print(f"Identity diagnosis  (MIMIC-IV v3.1, n={len(df):,})")
    print("=" * 74)
    print()

    # --- 1. MAP = (SBP + 2 DBP) / 3 ---
    d = df.dropna(subset=["sbp", "dbp", "map"])
    pred = (d["sbp"] + 2 * d["dbp"]) / 3
    print("  [1] MAP ≈ (SBP + 2·DBP)/3")
    print(f"      n={len(d):,}")
    print(f"      R²            = {r2(d['map'].to_numpy(), pred.to_numpy()):.4f}")
    print(f"      Pearson r     = {np.corrcoef(d['map'], pred)[0,1]:.4f}")
    print(f"      mean abs err  = {np.abs(d['map'] - pred).mean():.2f} mmHg")
    print(f"      MAP std       = {d['map'].std():.2f} mmHg")
    print()

    # --- 2. Hematocrit = 3 * Hemoglobin (candidate substitute variable) ---
    if "hematocrit" in df.columns:
        d2 = df.dropna(subset=["hemoglobin", "hematocrit"])
        if len(d2) > 100:
            r = np.corrcoef(d2["hemoglobin"], d2["hematocrit"])[0, 1]
            ratio = (d2["hematocrit"] / d2["hemoglobin"]).median()
            print("  [2] Hematocrit ≈ 3 × Hemoglobin  (on the substitute-variable list)")
            print(f"      n={len(d2):,}")
            print(f"      Pearson r     = {r:.4f}")
            print(f"      Hct/Hb median = {ratio:.2f}")
            print()

    # --- 3. What happens to the SBP-DBP relation when MAP is removed ---
    print("  [3] What happens to the SBP–DBP partial correlation when MAP is dropped")
    from scipy.stats import norm, rankdata
    from sklearn.linear_model import LinearRegression

    names_with = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
                  "glucose", "sodium", "potassium", "chloride", "bicarbonate",
                  "creatinine", "bun", "hemoglobin", "platelet"]
    names_wo = [n for n in names_with if n != "map"]

    dd = df.dropna(subset=names_with + ["y", "age", "sex", "icu_type"])
    C = dd[["y", "age", "sex", "icu_type"]].to_numpy(float)

    def pcorr(names):
        X = dd[names].to_numpy(float)
        X = X - LinearRegression().fit(C, X).predict(C)  # residualize
        N = X.shape[0]
        X = np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / N) for j in range(X.shape[1])])
        S = np.corrcoef(X, rowvar=False)
        P = np.linalg.inv(S + 1e-3 * np.eye(len(names)))
        dg = np.sqrt(np.diag(P))
        R = -P / np.outer(dg, dg)
        np.fill_diagonal(R, 1.0)
        return R

    Rw, Ro = pcorr(names_with), pcorr(names_wo)
    iw, jw = names_with.index("sbp"), names_with.index("dbp")
    io, jo = names_wo.index("sbp"), names_wo.index("dbp")
    print(f"      MAP included: ρ(sbp,dbp) = {Rw[iw,jw]:+.3f}   (suppressed because MAP mediates it)")
    print(f"      MAP excluded: ρ(sbp,dbp) = {Ro[io,jo]:+.3f}   (the real physiological relation emerges)")
    print()

    # --- 4. Contribution of each pair in Fisher-z space ---
    print("  [4] Fisher-z dominance structure (with MAP included)")
    z = np.arctanh(np.clip(np.abs(Rw), 0, 0.999))
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    contrib = sorted(((z[i, j], names_with[i], names_with[j]) for i, j in pairs), reverse=True)
    total = sum(c[0] for c in contrib)
    print(f"      total Σ|z| = {total:.2f}")
    for k, (zz, a, b) in enumerate(contrib[:6], 1):
        print(f"      {k}. {a:<12}--{b:<12} |z|={zz:.3f}  ({zz/total*100:4.1f}%)")
    top3 = sum(c[0] for c in contrib[:3])
    print(f"      the top 3 pairs account for {top3/total*100:.1f}% of the total")
    print()

    print("=" * 74)
    print("  Verdict")
    print("=" * 74)


if __name__ == "__main__":
    main()

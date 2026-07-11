"""진단: 특징 집합에 산술 항등식이 숨어 있는가?

문제:
    MAP ~= (SBP + 2*DBP)/3 은 임상 의존관계가 아니라 **산술 항등식**이다.
    이걸 생성 변수로 두면, CDG의 우위 상당 부분이 "항등식을 복원했다"에서 나오고
    리뷰어는 "임상 구조 발견이 아니라 산수 아니냐"고 묻는다.

    Hematocrit ~= 3 * Hemoglobin 도 마찬가지다 (대체 변수 목록에 있음).

이 스크립트는 각 후보 항등식의 실제 설명력(R^2)을 측정해서,
어떤 변수를 생성 집합에서 빼야 하는지 데이터로 판정한다.
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
    print(f"항등식 진단  (MIMIC-IV v3.1, n={len(df):,})")
    print("=" * 74)
    print()

    # --- 1. MAP = (SBP + 2 DBP) / 3 ---
    d = df.dropna(subset=["sbp", "dbp", "map"])
    pred = (d["sbp"] + 2 * d["dbp"]) / 3
    print("  [1] MAP ≈ (SBP + 2·DBP)/3")
    print(f"      n={len(d):,}")
    print(f"      R²            = {r2(d['map'].to_numpy(), pred.to_numpy()):.4f}")
    print(f"      Pearson r     = {np.corrcoef(d['map'], pred)[0,1]:.4f}")
    print(f"      평균 절대오차 = {np.abs(d['map'] - pred).mean():.2f} mmHg")
    print(f"      MAP 표준편차  = {d['map'].std():.2f} mmHg")
    print()

    # --- 2. Hematocrit = 3 * Hemoglobin (대체 변수 후보) ---
    if "hematocrit" in df.columns:
        d2 = df.dropna(subset=["hemoglobin", "hematocrit"])
        if len(d2) > 100:
            r = np.corrcoef(d2["hemoglobin"], d2["hematocrit"])[0, 1]
            ratio = (d2["hematocrit"] / d2["hemoglobin"]).median()
            print("  [2] Hematocrit ≈ 3 × Hemoglobin  (대체 변수 목록에 있음)")
            print(f"      n={len(d2):,}")
            print(f"      Pearson r     = {r:.4f}")
            print(f"      Hct/Hb 중앙값 = {ratio:.2f}")
            print()

    # --- 3. MAP 제거 시 SBP–DBP 관계가 어떻게 되는가 ---
    print("  [3] MAP을 빼면 SBP–DBP 부분상관이 어떻게 되는가")
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
    print(f"      MAP 포함:  ρ(sbp,dbp) = {Rw[iw,jw]:+.3f}   (MAP이 매개해서 눌림)")
    print(f"      MAP 제외:  ρ(sbp,dbp) = {Ro[io,jo]:+.3f}   (진짜 생리 관계로 드러남)")
    print()

    # --- 4. Fisher-z 공간에서 각 쌍의 기여도 ---
    print("  [4] Fisher-z 지배 구조 (MAP 포함 시)")
    z = np.arctanh(np.clip(np.abs(Rw), 0, 0.999))
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    contrib = sorted(((z[i, j], names_with[i], names_with[j]) for i, j in pairs), reverse=True)
    total = sum(c[0] for c in contrib)
    print(f"      전체 Σ|z| = {total:.2f}")
    for k, (zz, a, b) in enumerate(contrib[:6], 1):
        print(f"      {k}. {a:<12}--{b:<12} |z|={zz:.3f}  ({zz/total*100:4.1f}%)")
    top3 = sum(c[0] for c in contrib[:3])
    print(f"      상위 3쌍이 전체의 {top3/total*100:.1f}%를 차지")
    print()

    print("=" * 74)
    print("  판정")
    print("=" * 74)


if __name__ == "__main__":
    main()

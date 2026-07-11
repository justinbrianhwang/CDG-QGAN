"""MIMIC-IV 24시간 landmark 코호트 + 16특징 추출 (CDG-QGAN v2 §6).

demo(v2.2, 환자 100명)와 전체(v3.1) 양쪽에서 동일하게 동작한다.
CSV.gz를 직접 읽으므로 DB 셋업이 필요 없다.

코호트 (v2 §6.2):
  - 만 18세 이상
  - 환자별 최초 ICU 입실
  - ICU 입실 후 24시간까지 생존 + ICU 재원 (LOS >= 24h)
  -> 24시간 이전 사망/조기퇴실이 결측 마스크를 통해 사망 라벨을 누설하는 문제를 차단

결과 라벨:
  y = 24시간 landmark 이후, 병원 퇴원 전 사망

사용법:
  python scripts/extract_cohort.py              # MIMIC-IV v3.1 (본 실험)
  python scripts/extract_cohort.py --demo       # demo 100명 (파이프라인 디버그)

데이터 경로는 scripts/paths.py 에서 단일 관리한다 (공용 아카이브).
파생물도 아카이브의 _derived/ 로 나간다 — 코드 프로젝트에 환자 데이터를 두지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from features import CORE16, EVAL_ONLY, FALLBACKS, MIN_OBSERVED_RATE, TEMP_F_ITEMID  # noqa: E402

WINDOW_H = 24.0


def _read(root: Path, module: str, table: str, **kw) -> pd.DataFrame:
    for ext in (".csv.gz", ".csv", ".parquet"):
        p = root / module / f"{table}{ext}"
        if p.exists():
            if ext == ".parquet":
                return pd.read_parquet(p, **kw)
            return pd.read_csv(p, **kw)
    raise FileNotFoundError(f"{root/module/table} (.csv.gz/.csv/.parquet 중 없음)")


def build_cohort(root: Path) -> pd.DataFrame:
    """24시간 landmark 코호트 + 조건 변수."""
    icustays = _read(root, "icu", "icustays", parse_dates=["intime", "outtime"])
    patients = _read(root, "hosp", "patients")
    admissions = _read(root, "hosp", "admissions", parse_dates=["admittime", "dischtime", "deathtime"])

    # 환자별 최초 ICU 입실
    first = icustays.sort_values("intime").groupby("subject_id", as_index=False).first()
    n0 = len(first)

    # LOS >= 24h (= 24시간까지 ICU 재원 & 생존)
    first = first[first["los"] >= WINDOW_H / 24.0]
    n1 = len(first)

    df = first.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    df = df.merge(admissions[["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"]],
                  on="hadm_id", how="left")

    # 성인
    df = df[df["anchor_age"] >= 18]
    n2 = len(df)

    df["landmark"] = df["intime"] + pd.Timedelta(hours=WINDOW_H)

    # y = landmark 이후 병원 내 사망
    died = df["hospital_expire_flag"].fillna(0).astype(int) == 1
    death_after = df["deathtime"].notna() & (df["deathtime"] > df["landmark"])
    # deathtime이 없는데 expire_flag=1이면 dischtime을 사망시각으로 간주
    fallback = died & df["deathtime"].isna() & (df["dischtime"] > df["landmark"])
    df["y"] = ((died & death_after) | fallback).astype(int)

    # 24시간 이전 사망자는 LOS>=24h 필터로 대부분 제거되지만, 남으면 명시 제외
    pre = died & df["deathtime"].notna() & (df["deathtime"] <= df["landmark"])
    df = df[~pre]
    n3 = len(df)

    df["age"] = df["anchor_age"]
    df["sex"] = (df["gender"] == "F").astype(int)
    df["icu_type"] = df["first_careunit"].astype("category").cat.codes

    print("  코호트 구성")
    print(f"    환자별 최초 ICU 입실           : {n0:>7,}")
    print(f"    LOS >= 24h (landmark 생존)     : {n1:>7,}")
    print(f"    성인 (>=18세)                  : {n2:>7,}")
    print(f"    24h 이전 사망 제외             : {n3:>7,}")
    if n3:
        print(f"    y=1 (24h 이후 병원내 사망)     : {int(df['y'].sum()):>7,}  ({df['y'].mean()*100:.1f}%)")

    keep = ["subject_id", "hadm_id", "stay_id", "intime", "landmark",
            "age", "sex", "icu_type", "y"]
    return df[keep].reset_index(drop=True)


def _summarize(ev: pd.DataFrame, cohort: pd.DataFrame, feats, time_col: str) -> pd.DataFrame:
    """24시간 창 안의 값을 특징별로 요약."""
    ev = ev.merge(cohort[["subject_id", "stay_id", "intime", "landmark"]], on="subject_id", how="inner")
    ev = ev[(ev[time_col] >= ev["intime"]) & (ev[time_col] < ev["landmark"])]

    out = cohort[["stay_id"]].copy()
    for f in feats:
        sub = ev[ev["itemid"].isin(f.itemids)][["stay_id", "itemid", "valuenum"]].copy()
        if f.name == "temperature" and len(sub):  # 화씨 -> 섭씨
            m = sub["itemid"] == TEMP_F_ITEMID
            sub.loc[m, "valuenum"] = (sub.loc[m, "valuenum"] - 32.0) * 5.0 / 9.0
        # 물리적으로 불가능한 값은 결측 처리 (제거 아님)
        sub = sub[(sub["valuenum"] > f.lo) & (sub["valuenum"] <= f.hi)]
        agg = sub.groupby("stay_id")["valuenum"].agg(f.agg).rename(f.name) if len(sub) \
            else pd.Series(dtype=float, name=f.name)
        out = out.merge(agg, on="stay_id", how="left")
    return out


def extract(root: Path, out_path: Path) -> pd.DataFrame:
    print("=" * 70)
    print(f"MIMIC-IV 추출: {root}")
    print("=" * 70)

    cohort = build_cohort(root)
    if cohort.empty:
        raise SystemExit("코호트가 비었습니다.")

    # EVAL_ONLY(MAP)도 추출한다 — 생성하지는 않지만 임상 개연성 평가에 필요하다.
    feats = list(CORE16) + list(EVAL_ONLY) + list(FALLBACKS)
    chart_f = [f for f in feats if f.source == "chart"]
    lab_f = [f for f in feats if f.source == "lab"]

    print("\n  chartevents 읽는 중...")
    ce = _read(root, "icu", "chartevents",
               usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
               parse_dates=["charttime"])
    ce = ce[ce["itemid"].isin({i for f in chart_f for i in f.itemids})]
    ce = ce.drop(columns=["stay_id"])
    print(f"    관련 이벤트 {len(ce):,}건")

    print("  labevents 읽는 중...")
    le = _read(root, "hosp", "labevents",
               usecols=["subject_id", "charttime", "itemid", "valuenum"],
               parse_dates=["charttime"])
    le = le[le["itemid"].isin({i for f in lab_f for i in f.itemids})]
    print(f"    관련 이벤트 {len(le):,}건")

    print("\n  24시간 창 요약 중...")
    v = _summarize(ce, cohort, chart_f, "charttime")
    lab = _summarize(le, cohort, lab_f, "charttime")
    df = cohort.merge(v, on="stay_id").merge(lab, on="stay_id")

    # --- 관측률 점검 및 대체 변수 교체 (v2 §6.3) ---
    print("\n  특징별 관측률")
    names = [f.name for f in CORE16]
    rates = {n: df[n].notna().mean() for n in names}
    for n in names:
        flag = "" if rates[n] >= MIN_OBSERVED_RATE else "  <- 기준 미달"
        print(f"    {n:<14} {rates[n]*100:5.1f}%{flag}")

    low = [n for n in names if rates[n] < MIN_OBSERVED_RATE]
    if low:
        print(f"\n  기준({MIN_OBSERVED_RATE:.0%}) 미달 {len(low)}개: {', '.join(low)}")
        print("    -> 대체 변수 관측률:")
        for f in FALLBACKS:
            print(f"       {f.name:<14} {df[f.name].notna().mean()*100:5.1f}%")
        print("    (교체는 본 데이터에서 실험 전 1회 확정. demo에서는 확정하지 않음)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n  저장: {out_path}  ({len(df):,} x {df.shape[1]})")

    n_complete = df[names].notna().all(axis=1).sum()
    print(f"  16특징 전부 관측된 환자: {n_complete:,} / {len(df):,} ({n_complete/len(df)*100:.1f}%)")
    return df


if __name__ == "__main__":
    import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="demo 100명으로 실행 (파이프라인 디버그)")
    ap.add_argument("--root", type=Path, default=None, help="직접 지정 시")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    root = a.root or (paths.MIMIC_IV_DEMO if a.demo else paths.MIMIC_IV)
    out = a.out or paths.PROCESSED / ("cohort_demo.parquet" if a.demo else "cohort_v31.parquet")
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    extract(root, out)

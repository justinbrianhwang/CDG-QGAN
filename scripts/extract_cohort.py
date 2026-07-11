"""MIMIC-IV 24h landmark cohort + extraction of the 16 features (CDG-QGAN v2 §6).

Works identically on the demo (v2.2, 100 patients) and on the full release (v3.1).
It reads the CSV.gz files directly, so no database setup is required.

Cohort (v2 §6.2):
  - age 18 or over
  - each patient's first ICU stay
  - alive and still in the ICU 24h after admission (LOS >= 24h)
  -> blocks the leakage in which death or early discharge before 24h leaks the mortality
     label through the missingness mask

Outcome label:
  y = death after the 24h landmark and before hospital discharge

Usage:
  python scripts/extract_cohort.py              # MIMIC-IV v3.1 (the actual experiment)
  python scripts/extract_cohort.py --demo       # 100-patient demo (pipeline debugging)

Data paths are managed in one place, scripts/paths.py (the shared archive).
Derivatives are written out to _derived/ in the archive as well — no patient data ever
lives in the code project. MIMIC-IV is DUA-protected and requires credentialed access;
never commit or redistribute these outputs.
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
    raise FileNotFoundError(f"{root/module/table} (none of .csv.gz/.csv/.parquet found)")


def build_cohort(root: Path) -> pd.DataFrame:
    """24h landmark cohort + condition variables."""
    icustays = _read(root, "icu", "icustays", parse_dates=["intime", "outtime"])
    patients = _read(root, "hosp", "patients")
    admissions = _read(root, "hosp", "admissions", parse_dates=["admittime", "dischtime", "deathtime"])

    # each patient's first ICU stay
    first = icustays.sort_values("intime").groupby("subject_id", as_index=False).first()
    n0 = len(first)

    # LOS >= 24h (= still in the ICU and alive at 24h)
    first = first[first["los"] >= WINDOW_H / 24.0]
    n1 = len(first)

    df = first.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
    df = df.merge(admissions[["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"]],
                  on="hadm_id", how="left")

    # adults
    df = df[df["anchor_age"] >= 18]
    n2 = len(df)

    df["landmark"] = df["intime"] + pd.Timedelta(hours=WINDOW_H)

    # y = in-hospital death after the landmark
    died = df["hospital_expire_flag"].fillna(0).astype(int) == 1
    death_after = df["deathtime"].notna() & (df["deathtime"] > df["landmark"])
    # if deathtime is missing but expire_flag=1, take dischtime as the time of death
    fallback = died & df["deathtime"].isna() & (df["dischtime"] > df["landmark"])
    df["y"] = ((died & death_after) | fallback).astype(int)

    # the LOS>=24h filter removes most deaths before 24h, but exclude any survivors of it explicitly
    pre = died & df["deathtime"].notna() & (df["deathtime"] <= df["landmark"])
    df = df[~pre]
    n3 = len(df)

    df["age"] = df["anchor_age"]
    df["sex"] = (df["gender"] == "F").astype(int)
    df["icu_type"] = df["first_careunit"].astype("category").cat.codes

    print("  Cohort construction")
    print(f"    first ICU stay per patient        : {n0:>7,}")
    print(f"    LOS >= 24h (alive at landmark)    : {n1:>7,}")
    print(f"    adults (>=18y)                    : {n2:>7,}")
    print(f"    deaths before 24h excluded        : {n3:>7,}")
    if n3:
        print(f"    y=1 (in-hospital death after 24h) : {int(df['y'].sum()):>7,}  ({df['y'].mean()*100:.1f}%)")

    keep = ["subject_id", "hadm_id", "stay_id", "intime", "landmark",
            "age", "sex", "icu_type", "y"]
    return df[keep].reset_index(drop=True)


def _summarize(ev: pd.DataFrame, cohort: pd.DataFrame, feats, time_col: str) -> pd.DataFrame:
    """Summarize the values inside the 24h window, feature by feature."""
    ev = ev.merge(cohort[["subject_id", "stay_id", "intime", "landmark"]], on="subject_id", how="inner")
    ev = ev[(ev[time_col] >= ev["intime"]) & (ev[time_col] < ev["landmark"])]

    out = cohort[["stay_id"]].copy()
    for f in feats:
        sub = ev[ev["itemid"].isin(f.itemids)][["stay_id", "itemid", "valuenum"]].copy()
        if f.name == "temperature" and len(sub):  # Fahrenheit -> Celsius
            m = sub["itemid"] == TEMP_F_ITEMID
            sub.loc[m, "valuenum"] = (sub.loc[m, "valuenum"] - 32.0) * 5.0 / 9.0
        # physiologically impossible values are treated as missing (not dropped)
        sub = sub[(sub["valuenum"] > f.lo) & (sub["valuenum"] <= f.hi)]
        agg = sub.groupby("stay_id")["valuenum"].agg(f.agg).rename(f.name) if len(sub) \
            else pd.Series(dtype=float, name=f.name)
        out = out.merge(agg, on="stay_id", how="left")
    return out


def extract(root: Path, out_path: Path) -> pd.DataFrame:
    print("=" * 70)
    print(f"MIMIC-IV extraction: {root}")
    print("=" * 70)

    cohort = build_cohort(root)
    if cohort.empty:
        raise SystemExit("Cohort is empty.")

    # EVAL_ONLY (MAP) is extracted too — it is never generated, but it is needed for the
    # clinical-plausibility evaluation.
    feats = list(CORE16) + list(EVAL_ONLY) + list(FALLBACKS)
    chart_f = [f for f in feats if f.source == "chart"]
    lab_f = [f for f in feats if f.source == "lab"]

    print("\n  reading chartevents...")
    ce = _read(root, "icu", "chartevents",
               usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
               parse_dates=["charttime"])
    ce = ce[ce["itemid"].isin({i for f in chart_f for i in f.itemids})]
    ce = ce.drop(columns=["stay_id"])
    print(f"    {len(ce):,} relevant events")

    print("  reading labevents...")
    le = _read(root, "hosp", "labevents",
               usecols=["subject_id", "charttime", "itemid", "valuenum"],
               parse_dates=["charttime"])
    le = le[le["itemid"].isin({i for f in lab_f for i in f.itemids})]
    print(f"    {len(le):,} relevant events")

    print("\n  summarizing the 24h window...")
    v = _summarize(ce, cohort, chart_f, "charttime")
    lab = _summarize(le, cohort, lab_f, "charttime")
    df = cohort.merge(v, on="stay_id").merge(lab, on="stay_id")

    # --- Check observation rates and substitute replacement variables (v2 §6.3) ---
    print("\n  Observation rate per feature")
    names = [f.name for f in CORE16]
    rates = {n: df[n].notna().mean() for n in names}
    for n in names:
        flag = "" if rates[n] >= MIN_OBSERVED_RATE else "  <- below threshold"
        print(f"    {n:<14} {rates[n]*100:5.1f}%{flag}")

    low = [n for n in names if rates[n] < MIN_OBSERVED_RATE]
    if low:
        print(f"\n  {len(low)} feature(s) below the {MIN_OBSERVED_RATE:.0%} threshold: {', '.join(low)}")
        print("    -> observation rate of the replacement variables:")
        for f in FALLBACKS:
            print(f"       {f.name:<14} {df[f.name].notna().mean()*100:5.1f}%")
        print("    (the substitution is fixed once, on the real data, before the experiment. "
              "It is not fixed on the demo.)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n  saved: {out_path}  ({len(df):,} x {df.shape[1]})")

    n_complete = df[names].notna().all(axis=1).sum()
    print(f"  patients with all 16 features observed: {n_complete:,} / {len(df):,} ({n_complete/len(df)*100:.1f}%)")
    return df


if __name__ == "__main__":
    import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run on the 100-patient demo (pipeline debugging)")
    ap.add_argument("--root", type=Path, default=None, help="to specify the root explicitly")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    root = a.root or (paths.MIMIC_IV_DEMO if a.demo else paths.MIMIC_IV)
    out = a.out or paths.PROCESSED / ("cohort_demo.parquet" if a.demo else "cohort_v31.parquet")
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    extract(root, out)

"""Definition of the 16 core features (CDG-QGAN v2 §6.3).

The itemids and the outlier bounds are taken from the official concept SQL of the
MIMIC Code Repository:
  data/external/mimic-code/mimic-iv/concepts/measurement/{vitalsign,chemistry,complete_blood_count}.sql

Values outside the bounds are set to missing, not dropped (v2 §6.6: "physiologically
impossible values are treated as missing. Clinically plausible extreme values are not
removed automatically").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str
    source: str  # "chart" (icu/chartevents) | "lab" (hosp/labevents)
    itemids: tuple[int, ...]
    agg: str  # how the 24h window is summarized
    lo: float  # physiological lower bound (below it -> missing)
    hi: float  # physiological upper bound (above it -> missing)
    domain: str
    unit: str = ""
    note: str = ""


# --- Vital signs: mean over the time window (v2 §6.3) ---
#
# MAP is excluded from the generated variables.  <-- change relative to the plan.
# Evidence: scripts/diag_collinearity.py
#
#   MAP ≈ (SBP + 2·DBP)/3 is not a clinical dependency but an **arithmetic identity**.
#   Measured: R² = 0.860, r = 0.962 (n=51,587). 86% of MAP is explained by arithmetic on
#   SBP and DBP.
#
#   Worse: because MAP is a deterministic function of SBP and DBP, conditioning on MAP in
#   the precision matrix induces a **spurious negative correlation** between SBP and DBP
#   (a collider/suppression artifact).
#       MAP included: ρ(sbp,dbp) = -0.508   <- physiologically wrong
#       MAP excluded: ρ(sbp,dbp) = +0.499   <- the real relationship
#   In other words, including MAP puts a **physiologically incorrect edge** into the CDG.
#   That is not a metric problem, it is a scientific error.
#
#   There is also the risk that the CDG's advantage collapses into "it recovered an
#   arithmetic identity". Reviewer: "Isn't this arithmetic rather than the discovery of
#   clinical structure?"
#
# MAP is still extracted (EVAL_ONLY), but used only as a clinical-plausibility **evaluation
# metric**:
#   from the generated SBP~ and DBP~ we compute MAP~ = (SBP~ + 2·DBP~)/3 and compare it
#   against the real MAP. This is a stricter check than generating MAP directly and fitting
#   it. (v2 §6.4 and §12.7 likewise define MAP as a diagnostic metric, not a training loss.)
VITALS = [
    Feature("heart_rate", "chart", (220045,), "mean", 0, 300, "circulation", "bpm"),
    Feature("sbp", "chart", (220179, 220050, 225309), "mean", 0, 400, "circulation", "mmHg"),
    Feature("dbp", "chart", (220180, 220051, 225310), "mean", 0, 300, "circulation", "mmHg"),
    Feature("resp_rate", "chart", (220210, 224690), "mean", 0, 70, "respiration", "insp/min"),
    Feature("spo2", "chart", (220277,), "mean", 0, 100, "respiration", "%"),
    Feature("temperature", "chart", (223762, 223761), "mean", 25, 45, "temperature", "degC",
            note="223761=F -> needs conversion to C"),
]

# --- Labs: median over the time window (v2 §6.3) ---
LABS = [
    Feature("wbc", "lab", (51301,), "median", 0, 1000, "inflammation", "K/uL",
            note="Put in where MAP was. This is critical-care data and the inflammation axis "
                 "was missing entirely."),
    Feature("glucose", "lab", (50931,), "median", 0, 10000, "metabolism", "mg/dL"),
    Feature("sodium", "lab", (50983,), "median", 0, 200, "electrolyte", "mEq/L"),
    Feature("potassium", "lab", (50971,), "median", 0, 30, "electrolyte", "mEq/L"),
    Feature("chloride", "lab", (50902,), "median", 0, 10000, "electrolyte", "mEq/L"),
    Feature("bicarbonate", "lab", (50882,), "median", 0, 10000, "acid-base", "mEq/L"),
    Feature("creatinine", "lab", (50912,), "median", 0, 150, "renal", "mg/dL"),
    Feature("bun", "lab", (51006,), "median", 0, 300, "renal", "mg/dL"),
    Feature("hemoglobin", "lab", (51222,), "median", 0, 30, "hematology", "g/dL"),
    Feature("platelet", "lab", (51265,), "median", 0, 2000, "hematology", "K/uL"),
]

CORE16 = VITALS + LABS
assert len(CORE16) == 16

# --- Evaluation only (never generated) ---
EVAL_ONLY = [
    Feature("map", "chart", (220052, 220181, 225312), "mean", 0, 300, "circulation", "mmHg",
            note="Not generated. Used for clinical-plausibility evaluation: compare against "
                 "MAP~ computed from SBP~/DBP~."),
]

# --- Predefined replacement variables (v2 §6.3). If the observation rate falls short,
#     substitute in this order ---
#
# hematocrit was excluded. Measured r(Hb, Hct) = 0.962, median Hct/Hb = 3.02
# -> Hct ≈ 3 × Hb is exactly the same kind of **arithmetic identity** as MAP. If it were
#    substituted in automatically, we would walk straight back into the same trap (spurious
#    partial correlation + the result collapsing into "it recovered an identity").
FALLBACKS = [
    Feature("calcium", "lab", (50893,), "median", 0, 10000, "electrolyte", "mg/dL"),
    Feature("magnesium", "lab", (50960,), "median", 0, 100, "electrolyte", "mg/dL"),
    Feature("phosphate", "lab", (50970,), "median", 0, 100, "electrolyte", "mg/dL"),
]

# Fahrenheit temperature itemid (needs conversion)
TEMP_F_ITEMID = 223761

# Lower bound on the observation rate. Below it, substitute from FALLBACKS (v2 §6.3, §6.7).
MIN_OBSERVED_RATE = 0.70

# Condition vector c. v2 used y alone, but since the CDG is residualized on
# (y, age, sex, ICU type), these three must also enter the condition for the conditional
# estimation on synthetic data to be the same estimator.
# (review A-1: fixes the estimator mismatch)
CONDITION_VARS = ["mortality_after_24h", "age", "sex", "icu_type"]


def all_features() -> list[Feature]:
    return list(CORE16)


def feature_table() -> str:
    rows = ["| # | domain | feature | source | itemid | agg | bounds |",
            "|---:|---|---|---|---|---|---|"]
    for i, f in enumerate(CORE16, 1):
        ids = ", ".join(str(x) for x in f.itemids)
        rows.append(f"| {i} | {f.domain} | {f.name} | {f.source} | {ids} | {f.agg} | [{f.lo}, {f.hi}] {f.unit} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{len(CORE16)} core features ({len(VITALS)} vital signs + {len(LABS)} labs)\n")
    print(feature_table())
    print(f"\nReplacement variables: {', '.join(f.name for f in FALLBACKS)}")
    print(f"Condition vector c: {', '.join(CONDITION_VARS)}")

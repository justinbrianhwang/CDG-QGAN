"""Single source of truth for data paths.

Medical data — **raw or derived, no exception** — lives in the shared archive, not in
the project:
    D:\\pythondata\\Medical Data

Principle: **not a single patient-level record lives in the code project.**

  - Do not copy the 10GB of MIMIC-IV into every project
  - The cohort table and the CDG are patient-level derivatives too, so they fall under
    the Data Use Agreement (DUA). Keeping them in the project is an accident waiting to
    happen the moment it is published to GitHub -> push them out to _derived/ in the archive
  - The project keeps only code and **aggregate results** (per-seed metrics and the like,
    nothing patient-level)

Override with the MEDICAL_DATA_ROOT environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Shared archive (raw sources) ---
ARCHIVE = Path(os.environ.get("MEDICAL_DATA_ROOT", r"D:\pythondata\Medical Data"))

EHR = ARCHIVE / "Electronic Health Records"
MIMIC_IV = EHR / "MIMIC-IV"                 # full v3.1 (364,627 patients / 94,458 ICU stays)
MIMIC_IV_DEMO = EHR / "MIMIC-IV-demo-2.2"   # public demo (100 patients) — for pipeline debugging
EICU = EHR / "eicu-crd"                     # for external validation (out of scope for the first paper)

# --- Derivatives: patient-level -> archive (DUA-protected, do not distribute) ---
PROCESSED = ARCHIVE / "_derived" / "CDG-QGAN"

# --- Project: code and aggregate results only (no patient-level data) ---
PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"                  # aggregates such as per-seed metrics
MIMIC_CODE = PROJECT / "tools" / "mimic-code"  # official concept SQL


def check() -> None:
    """Verify that the paths actually exist."""
    for name, p, need in [
        ("MIMIC-IV v3.1", MIMIC_IV, True),
        ("MIMIC-IV demo", MIMIC_IV_DEMO, False),
        ("eICU-CRD", EICU, False),
        ("mimic-code", MIMIC_CODE, True),
    ]:
        ok = p.exists()
        mark = "OK " if ok else ("MISSING" if need else "absent(optional)")
        print(f"  [{mark:>10}] {name:<16} {p}")
        if need and not ok:
            raise FileNotFoundError(f"{name} not found: {p}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("Data paths")
    check()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"\n  derivative output: {PROCESSED}")
    print(f"  results output   : {RESULTS}")

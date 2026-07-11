"""데이터 경로 단일 관리.

의료 데이터는 **원본이든 파생물이든** 프로젝트가 아니라 공용 아카이브에 둔다:
    D:\\pythondata\\Medical Data

원칙: **코드 프로젝트에는 환자 단위 데이터가 한 톨도 없다.**

  - MIMIC-IV 10GB를 프로젝트마다 복제하지 않는다
  - 코호트 테이블과 CDG도 환자 단위 파생물이므로 DUA 대상이다.
    프로젝트에 두면 GitHub 공개 시 사고가 난다 -> 아카이브의 _derived/ 로 뺀다
  - 프로젝트에는 코드와 **집계 결과**(seed별 지표 등, 환자 단위 아님)만 남긴다

환경변수 MEDICAL_DATA_ROOT 로 덮어쓸 수 있다.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- 공용 아카이브 (원본) ---
ARCHIVE = Path(os.environ.get("MEDICAL_DATA_ROOT", r"D:\pythondata\Medical Data"))

EHR = ARCHIVE / "Electronic Health Records"
MIMIC_IV = EHR / "MIMIC-IV"                 # v3.1 전체 (환자 364,627 / ICU 94,458)
MIMIC_IV_DEMO = EHR / "MIMIC-IV-demo-2.2"   # 공개 demo (환자 100) — 파이프라인 디버그용
EICU = EHR / "eicu-crd"                     # 외부 검증용 (첫 논문 범위 밖)

# --- 파생물: 환자 단위 -> 아카이브 (DUA 대상, 배포 금지) ---
PROCESSED = ARCHIVE / "_derived" / "CDG-QGAN"

# --- 프로젝트: 코드와 집계 결과만 (환자 단위 데이터 없음) ---
PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"                  # seed별 지표 등 집계값
MIMIC_CODE = PROJECT / "tools" / "mimic-code"  # 공식 concept SQL


def check() -> None:
    """경로가 실제로 존재하는지 확인."""
    for name, p, need in [
        ("MIMIC-IV v3.1", MIMIC_IV, True),
        ("MIMIC-IV demo", MIMIC_IV_DEMO, False),
        ("eICU-CRD", EICU, False),
        ("mimic-code", MIMIC_CODE, True),
    ]:
        ok = p.exists()
        mark = "OK " if ok else ("MISSING" if need else "없음(선택)")
        print(f"  [{mark:>10}] {name:<16} {p}")
        if need and not ok:
            raise FileNotFoundError(f"{name} 가 없습니다: {p}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("데이터 경로")
    check()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"\n  파생물 출력: {PROCESSED}")
    print(f"  결과 출력  : {RESULTS}")

"""16개 핵심 특징 정의 (CDG-QGAN v2 §6.3).

itemid와 이상치 경계는 MIMIC Code Repository 공식 concept SQL에서 가져왔다:
  data/external/mimic-code/mimic-iv/concepts/measurement/{vitalsign,chemistry,complete_blood_count}.sql

경계를 벗어난 값은 제거가 아니라 결측 처리한다 (v2 §6.6: "물리적으로 불가능한 값은
결측으로 처리한다. 임상적으로 가능한 극단값은 자동 제거하지 않는다").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str
    source: str  # "chart" (icu/chartevents) | "lab" (hosp/labevents)
    itemids: tuple[int, ...]
    agg: str  # 24시간 창 요약 방식
    lo: float  # 물리적 하한 (미만이면 결측)
    hi: float  # 물리적 상한 (초과면 결측)
    domain: str
    unit: str = ""
    note: str = ""


# --- 활력징후: 시간창 평균 (v2 §6.3) ---
#
# MAP은 생성 변수에서 제외한다.  <-- 계획서 대비 변경. 근거: scripts/diag_collinearity.py
#
#   MAP ≈ (SBP + 2·DBP)/3 은 임상 의존관계가 아니라 **산술 항등식**이다.
#   실측: R² = 0.860, r = 0.962 (n=51,587). MAP의 86%가 SBP·DBP의 산술로 설명된다.
#
#   더 심각한 것: MAP은 SBP·DBP의 결정론적 함수이므로, 정밀도 행렬에서 MAP을 조건부로
#   고정하면 SBP–DBP 사이에 **가짜 음의 상관**이 유도된다 (collider/suppression 아티팩트).
#       MAP 포함: ρ(sbp,dbp) = -0.508   <- 생리학적으로 틀림
#       MAP 제외: ρ(sbp,dbp) = +0.499   <- 진짜 관계
#   즉 MAP을 넣으면 CDG에 **생리학적으로 잘못된 간선**이 들어간다. 지표 문제가 아니라
#   과학적 오류다.
#
#   또한 CDG의 우위가 "산술 항등식을 복원했다"로 환원되어 보이는 위험이 있다.
#   리뷰어: "임상 구조 발견이 아니라 산수 아닌가?"
#
# MAP은 여전히 추출하되(EVAL_ONLY), 임상 개연성 **평가 지표**로만 쓴다:
#   생성된 SBP~, DBP~ 로부터 MAP~ = (SBP~ + 2·DBP~)/3 을 계산해 실제 MAP과 비교한다.
#   이는 MAP을 직접 생성해 맞추는 것보다 엄격한 검증이다. (v2 §6.4, §12.7도 MAP을
#   학습 손실이 아니라 진단 지표로 규정하고 있다.)
VITALS = [
    Feature("heart_rate", "chart", (220045,), "mean", 0, 300, "순환", "bpm"),
    Feature("sbp", "chart", (220179, 220050, 225309), "mean", 0, 400, "순환", "mmHg"),
    Feature("dbp", "chart", (220180, 220051, 225310), "mean", 0, 300, "순환", "mmHg"),
    Feature("resp_rate", "chart", (220210, 224690), "mean", 0, 70, "호흡", "insp/min"),
    Feature("spo2", "chart", (220277,), "mean", 0, 100, "호흡", "%"),
    Feature("temperature", "chart", (223762, 223761), "mean", 25, 45, "체온", "degC",
            note="223761=F -> C 변환 필요"),
]

# --- 검사: 시간창 중앙값 (v2 §6.3) ---
LABS = [
    Feature("wbc", "lab", (51301,), "median", 0, 1000, "염증", "K/uL",
            note="MAP 자리에 투입. 중환자 데이터인데 염증 축이 통째로 빠져 있었다."),
    Feature("glucose", "lab", (50931,), "median", 0, 10000, "대사", "mg/dL"),
    Feature("sodium", "lab", (50983,), "median", 0, 200, "전해질", "mEq/L"),
    Feature("potassium", "lab", (50971,), "median", 0, 30, "전해질", "mEq/L"),
    Feature("chloride", "lab", (50902,), "median", 0, 10000, "전해질", "mEq/L"),
    Feature("bicarbonate", "lab", (50882,), "median", 0, 10000, "산염기", "mEq/L"),
    Feature("creatinine", "lab", (50912,), "median", 0, 150, "신장", "mg/dL"),
    Feature("bun", "lab", (51006,), "median", 0, 300, "신장", "mg/dL"),
    Feature("hemoglobin", "lab", (51222,), "median", 0, 30, "혈액", "g/dL"),
    Feature("platelet", "lab", (51265,), "median", 0, 2000, "혈액", "K/uL"),
]

CORE16 = VITALS + LABS
assert len(CORE16) == 16

# --- 평가 전용 (생성하지 않음) ---
EVAL_ONLY = [
    Feature("map", "chart", (220052, 220181, 225312), "mean", 0, 300, "순환", "mmHg",
            note="생성 안 함. SBP~/DBP~로부터 계산한 MAP~과 비교하는 임상 개연성 평가용."),
]

# --- 사전 정의 대체 변수 (v2 §6.3). 관측률 미달 시 이 순서로 교체 ---
#
# hematocrit을 제외했다. 실측 r(Hb, Hct) = 0.962, Hct/Hb 중앙값 = 3.02
# -> Hct ≈ 3 × Hb 는 MAP과 똑같은 **산술 항등식**이다. 대체 변수로 자동 투입되면
#    같은 함정(가짜 부분상관 + 항등식 복원으로 환원)을 다시 밟는다.
FALLBACKS = [
    Feature("calcium", "lab", (50893,), "median", 0, 10000, "전해질", "mg/dL"),
    Feature("magnesium", "lab", (50960,), "median", 0, 100, "전해질", "mg/dL"),
    Feature("phosphate", "lab", (50970,), "median", 0, 100, "전해질", "mg/dL"),
]

# 온도 화씨 itemid (변환 필요)
TEMP_F_ITEMID = 223761

# 관측률 하한. 미달하면 FALLBACKS에서 교체 (v2 §6.3, §6.7).
MIN_OBSERVED_RATE = 0.70

# 조건 벡터 c. v2는 y만 썼으나, CDG를 (y, age, sex, ICU type)에 residualize하므로
# 합성 데이터에서도 동일하게 조건부 추정을 하려면 이 셋이 조건에 들어가야 한다.
# (리뷰 지적 A-1: 추정량 불일치 수정)
CONDITION_VARS = ["mortality_after_24h", "age", "sex", "icu_type"]


def all_features() -> list[Feature]:
    return list(CORE16)


def feature_table() -> str:
    rows = ["| # | 영역 | 특징 | 소스 | itemid | 요약 | 경계 |",
            "|---:|---|---|---|---|---|---|"]
    for i, f in enumerate(CORE16, 1):
        ids = ", ".join(str(x) for x in f.itemids)
        rows.append(f"| {i} | {f.domain} | {f.name} | {f.source} | {ids} | {f.agg} | [{f.lo}, {f.hi}] {f.unit} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"핵심 특징 {len(CORE16)}개 (활력징후 {len(VITALS)} + 검사 {len(LABS)})\n")
    print(feature_table())
    print(f"\n대체 변수: {', '.join(f.name for f in FALLBACKS)}")
    print(f"조건 벡터 c: {', '.join(CONDITION_VARS)}")

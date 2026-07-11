# CDG-QGAN 구현 작업 명세 (Codex용)

최종 갱신: 2026-07-11 · 설계·리뷰·해석은 PM, 구현은 Codex

이 문서는 **무엇을 왜 만들어야 하는가**와 **무엇으로 합격 판정하는가**만 적는다.
구현 방식은 Codex 재량.

---

## 0. 먼저 읽을 것

| 문서 | 내용 |
|---|---|
| `CDG-QGAN_research_plan_v2_ko.md` | 연구계획서 v2 (기준 문서) |
| **`REVISIONS.md`** | **v2에서 실험으로 뒤집힌 것. 계획서와 충돌하면 이게 우선한다.** |
| `RESULTS_precheck.md` | CDG 정렬 효과 검증 (실제 데이터) |
| `RESULTS_ceiling.md` | 회로 표현력 천장 · light-cone 검증 |
| `DATA.md` | 데이터 위치, 환경 함정 |
| `scripts/*` | 검증 완료된 참조 구현. 재작성보다 확장을 권한다. |

### 환경 (필독)

conda env **`cdg-qgan`**. **numpy는 2.2.6에 고정. 절대 올리지 말 것.**
torch 2.11+cu128이 numpy 2.2 ABI 빌드라 2.4로 올리면 `shm.dll` 로드가 깨진다
(`OSError: [WinError 127]`). 실제로 base conda가 이렇게 망가졌다.

```bash
python -m pip install -c .backup/constraints.txt <package>   # 항상 constraint를 걸 것
```

데이터 경로는 `scripts/paths.py`가 단일 관리한다.
**원본도 파생물도 공용 아카이브(`D:\pythondata\Medical Data`)에 있다.
코드 프로젝트에 환자 데이터를 절대 두지 말 것** (DUA).

---

## 1. 이미 완료된 것 (재작업 불필요)

| 항목 | 결과 |
|---|---|
| MIMIC-IV v3.1 코호트 추출 | **51,668명 / 사망 5,350명(10.4%)**. 16특징 완전관측 48,561명 |
| CDG 추정 | 16노드 · 23간선 · Δ≤3 · 지름 5 |
| **정렬 효과 사전 검사** | **`L=1`: z=+3.18, p=0.0004 PASS** (5,000 순열 중 2개만 CDG를 이김) |
| 회로 표현력 천장 | `L=1` 인접 쌍 **0.991** → 강한 임상 관계 표현 가능. `d≥3`은 ~0.01 |
| light-cone 따름정리 | **수치 입증됨.** 절벽이 정확히 `d=2L` |
| head 비혼합성 | Jacobian이 정확히 대각행렬 → 고전 파라미터는 의존성을 만들 수 없음 |
| light-cone 시뮬레이터 | n=16, L=1에서 **4096배 절감**, 전체와 1e-6 일치 |
| MAP/Hct 항등식 제거 | `REVISIONS.md` C-3 |

**미결 항목 없음. 설계는 데이터로 전부 확정됐다.**

---

## 2. 계획서 v2와 충돌 시 이쪽을 따를 것

실험으로 뒤집힌 5가지. 근거는 `REVISIONS.md`.

1. **`L=1` 단독.** `L=2`는 실제 데이터에서 FAIL(z=+0.66, p=0.287),
   `L=3`은 모든 쌍이 도달 가능해져 **정의상 null**. → **`L=2`, `L=3` 확증 실험 금지.**
2. **Primary endpoint는 held-out 간선이 아니라 120쌍 전체** 의존성 오차.
   held-out만으로 재면 CDG가 무작위 permutation보다 못 나온다.
3. **조건 벡터는 `c = (사망, 연령, 성별, ICU유형)`** — `y` 하나가 아니다.
4. **의존성 지표는 nonparanormal 공간**에서 계산. 원 단위 Pearson 금지.
5. **MAP은 생성 변수가 아니다** (산술 항등식). 평가 전용.

---

## 3. 작업 패키지

> **순서: WP-6 → WP-2 → WP-3 → WP-4 → WP-5**
> WP-1(데이터 파이프라인)은 완료됨.

### WP-6. 시뮬레이터 성능 ← **먼저**

`qsim_lightcone.py`는 **정확하지만 느리다.** 병목은 연산이 아니라 **커널 런치
오버헤드**다. `L=1, Δ≤3`이면 cone이 4큐비트(16차원)라 텐서가 너무 작은데,
큐비트 16개 × 게이트 ~10개를 파이썬 루프로 돌려 스텝당 수백 번의 마이크로 커널이
발생한다. (실측: GPU 83%가 계산이 아니라 런치 대기)

**실측 근거**: `benchmark_synthetic.py`에서 변형 1개(seed 3개)에 **1.5시간+**.
5개 변형 완주에 7시간+ 필요 → 중단함. 확증 실험 90회 학습으로는 감당 불가.

**최적화 방향** (수치는 바뀌면 안 됨)
1. **16개 cone을 하나의 배치 축으로 묶는다.** 모두 ≤4큐비트이므로 `(B, 16, 16)`
   텐서 하나로 동시 처리 가능. cone 크기가 다르면 최대 크기로 패딩.
2. 단일 큐비트 게이트를 cone 축에 대해 vectorize.
3. `torch.compile` 또는 CUDA graph로 런치 오버헤드 제거.

**합격 기준**
- `qsim_lightcone.verify_against_full()` 통과 (기존 구현과 `<1e-5` 일치)
- `L=1` 3000스텝 학습이 **seed당 1분 이내**
- 최적화 후 **`benchmark_synthetic.py` 재실행** → 참 그래프를 아는 상태에서
  `aligned < permuted` 가 나오는지 확인 (확증 설계의 학습 단계 검증)

---

### WP-2. 확증 실험

**깊이 `L=1` 고정.** seed 10개.

| 모델 | 목적 |
|---|---|
| CDG-QGAN | 제안 모델 |
| **isomorphic permuted-CDG × 3** | **확증 대조군.** 임상 정렬만 파괴 |
| **distance-matched permuted × 3** | held-out 거리 분포까지 맞춤. 효과가 단순 거리 배치를 넘는지 |
| degree-preserving rewired | 추상 topology 효과 |
| ring (자원 맞춤) | 균일 토폴로지 기준선 |
| no-entanglement | 얽힘 필요성 |

전부 동일 자원: 간선 수, 깊이, 파라미터, critic, loss, step 예산.

**Primary endpoint**: 120쌍 전체 조건부 의존성 오차 (nonparanormal 공간, Fisher-z).
**위음성**(못 만든 의존)과 **위양성**(없는 의존을 만들어냄)을 모두 잡아야 한다.
held-out 간선 오차는 secondary.

**음성 대조로만** `L=2`를 돌려 정렬 효과가 소멸하는지 확인한다
(예상: `z: +3.18 → +0.66 → 0.00`). **이 감쇠 곡선이 논문의 그림이다.**

**합격 기준**
- `Δ = CDG − permuted 평균 < 0`, hierarchical bootstrap 95% CI가 0을 포함하지 않을 것
- 특정 permutation 하나에 의존하지 않을 것
- 주변분포 fidelity가 현저히 악화된 상태에서 얻은 결과가 아닐 것

---

### WP-3. 고전 대조군 (양자 기여를 분리)

| 모델 | 목적 |
|---|---|
| CDG-local classical message-passing (GNN) | **가장 중요.** 동일 receptive field·유사 파라미터 |
| Random Fourier Feature 생성기 | 단순 비선형 특징맵으로 설명되는지 |
| Global MLP-WGAN | 절대 품질 맥락 (양자 기여의 증거로 쓰지 말 것) |
| TabDDPM | 현대적 고전 기준선 (맥락 수치) |

**해석 규칙 (v2 §18.2)**: CDG-QGAN이 GNN보다 낮은 오차일 때만 "양자 코어의 추가 이점"을
제한적으로 주장한다. **동등하면 "graph-local inductive bias는 유효했으나 양자 회로의
추가 이점 증거는 없음"으로 결론을 제한한다.** 이 결과도 논문이 된다.

> **PM 판단**: GNN이 이기거나 비길 확률이 높다. 그 경우 논문의 축은
> **파라미터–성능 Pareto**가 된다 (양자 코어는 얽힘 각도 ~23개로 동일 receptive field 달성).
> 이 그림을 Discussion 곁가지가 아니라 **Results 주요 그림**으로 준비할 것.

---

### WP-4. 평가 스위트

- **의존성**: 120쌍 전체 오차(primary), 거리별 층화, held-out(secondary),
  precision matrix edge recovery, **위양성률**(참 non-edge에서 만들어낸 가짜 의존)
- **주변분포**: Wasserstein-1, KS, quantile 오차
- **다변량**: MMD, energy distance, real-vs-synthetic AUROC
- **임상 개연성**:
  - `DBP ≤ MAP ≤ SBP` 순서 관계
  - **`MAP~ = (SBP~ + 2·DBP~)/3` 을 계산**해 실제 MAP 분포와 비교
    (MAP은 생성하지 않는다. 계산해서 맞추는 것이 더 엄격한 검증이다.)
  - 위반율을 0으로 만드는 게 목표가 아니다. **`|VR_syn − VR_real|` 최소화.**
- **utility**: TRTR / TSTR (LR, RF, XGBoost, MLP) — AUROC, AUPRC, **Brier, ECE**
  - **`TSTR > TRTR`은 성공이 아니라 적신호로 처리할 것** (조건부 생성이 클래스를
    과분리한 것). calibration을 반드시 함께 볼 것.
- **프라이버시**: 단일 MIA AUC로 비교 금지. **privacy–fidelity Pareto frontier**로 제시.
  (못 학습한 모델이 자동으로 안전해 보이는 교락을 피하기 위함)

---

### WP-5. NISQ

- finite-shot `S ∈ {256, 1024, 4096}` — **최종 모델에만** 적용
- **모든 Z, ZZ가 계산기저에서 대각**이므로 **Z-basis 비트스트링 한 세트**에서
  전부 동시 추정 가능. 측정 basis 수가 간선 수만큼 늘지 않는다. 논문에 명시할 것.
- shot-noise-aware fine-tuning: 독립 가우시안이 아니라 **상관된** surrogate
  `ε ~ N(0, Σ_O/S)` 사용
  > **PM 관찰**: `Cov(Ẑ_u, Ẑ_v) = (⟨Z_uZ_v⟩ − μ_uμ_v)/S` 는 **참 상관과 같은 방향**이다.
  > 즉 shot 노이즈가 출력 의존성을 **체계적으로 부풀린다.** 낮은 S에서 지표가
  > 좋아 보이는 아티팩트가 나올 수 있다. 편향 항을 해석적으로 유도해 보고할 것.
  > 작지만 진짜 기여가 된다.
- transpiled 2큐비트 게이트 수·깊이, SWAP 수, CDG edge 보존율 보고
- `barren plateau`라는 용어는 큐비트 수에 따른 지수적 scaling을 검증한 경우에만 사용

---

## 4. 하지 말 것

- **`L=2` / `L=3` 확증 실험** — `L=3`은 120쌍 전부가 light cone 안에 들어와 CDG와
  permuted가 문자 그대로 동일해진다. `L=2`도 실제 데이터에서 FAIL.
  (`L=2`는 **음성 대조**로만 돌린다)
- **held-out 간선만으로 primary 판정** — CDG가 무작위 permutation보다 못 나온다
- **원 단위 Pearson으로 의존성 측정** — 주변분포 오차가 지표를 오염시킨다
- **MAP을 생성 변수에 다시 넣기** — 산술 항등식(R²=0.860)이고 SBP–DBP에 가짜 음의
  상관을 유도한다 (ρ 부호가 −0.508 ↔ +0.499로 뒤집힘)
- **Hematocrit을 대체 변수로 투입** — `Hct ≈ 3×Hb` (r=0.962). 같은 항등식
- **`r_Q`(양자 파라미터 비율)를 방어 근거로 사용** — 3.5%라 오히려 역효과.
  대신 **"의존성을 만들 수 있는 파라미터의 100%가 양자"** 를 쓸 것
- **CTGAN·TabDDPM 대비 우위를 양자 기여의 증거로 해석** — 손실 구성이 다르므로 맥락 수치일 뿐
- **전체 상태벡터로 훈련** — light-cone 부분회로로 4096배 싸게 정확히 같은 값을 얻는다

---

## 5. PM에게 즉시 보고할 것

1. **WP-6 최적화 후 `verify_against_full()`이 실패** → 수치가 바뀌었다는 뜻. 즉시 중단.
2. `benchmark_synthetic.py`에서 **`aligned`가 `permuted`보다 나쁘게 나옴** → 확증 설계에
   문제가 있다는 뜻. GPU를 더 태우지 말고 보고.
3. **`TSTR > TRTR`** 관측 시 → 조건부 생성이 클래스를 과분리했을 가능성.
4. **CDG-QGAN이 graph-local GNN에 짐** → 논문 축을 파라미터–성능 Pareto로 전환.
5. 새로운 **산술 항등식** 발견 시 (변수 간 R² > 0.8) → 변수 집합을 다시 봐야 한다.

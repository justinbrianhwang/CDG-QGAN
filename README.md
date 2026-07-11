# CDG-QGAN

Clinical Dependency Graph 를 얕은 파라미터화 양자회로의 **얽힘 토폴로지**로 삼는
하이브리드 양자-고전 GAN. MIMIC-IV ICU 데이터의 합성 생성을 목표로 한다.

핵심 주장: 임상 변수 간 조건부 의존성 그래프(CDG)를 회로의 RZZ 배치에 그대로 심으면,
같은 자원을 쓰는 **동형 순열 그래프**보다 의존 구조를 더 잘 복원한다.
즉 "Clinical"이 단순한 수식어가 아니라 실제로 비트를 기여함을 보인다.

## 설계의 뼈대

- **1 특징 = 1 큐비트.** 각 특징에 국소 잠재변수 `z_u`를 하나씩 붙이고 각도로 인코딩한다.
- **RZZ는 CDG 간선에만.** 얽힘 각도 `gamma`가 유일하게 교차특징 의존성을 만들 수 있는 파라미터다.
- **특징별 1차원 local head** `h_u(q_u, c)`. 다른 큐비트의 `q_v`를 못 본다.
  따라서 고전 파라미터는 조건부 교차특징 의존성을 **구조적으로 만들 수 없다**
  (`model.assert_no_cross_feature_mixing`이 Jacobian이 대각임을 검증).
- **명제 1 (light cone).** product 초기상태 + 국소 각도 인코딩이면 `<Z_u>`는 그래프 거리 `L`
  이내의 큐비트에만 의존한다. 따름정리: `d_G(u,v) > 2L` 이면 `Cov(x_u, x_v | c) = 0` (정확히).
  이 항등식을 계산에도 써서 2^16 상태벡터 대신 큐비트당 2^|N_L(u)| 부분회로만 시뮬레이션한다.

`L=1`이 유일한 작동점이다. `L=3`이면 120쌍 전부가 light cone 안에 들어와 CDG와 순열 그래프가
정의상 동일해진다. 자세한 근거는 `REVISIONS.md`.

## 지금까지 확인된 것

| | 결과 | 문서 |
|---|---|---|
| light cone 절벽이 `d = 2L`에서 정확히 생김 | 밖에서 최대 \|rho\|=0.012 | `RESULTS_ceiling.md` |
| 인접 쌍 표현력 천장 (L=1) | \|rho\| = 0.991 | `RESULTS_ceiling.md` |
| 실제 CDG의 정렬 사전검사 `M(G,L)` | L=1: z=+3.18 (p=0.0004) / L=2: z=+0.66 (기각) | `RESULTS_precheck.md` |
| light-cone 시뮬레이터 최적화 | ~70배 가속 (>1800s → 25.7s / seed) | `WP6_REPORT.md` |

## 미해결 — 현재 최우선 과제

통제된 합성 그래프 벤치마크에서 **확증 대조가 작동하지 않는다.**
`aligned`, `permuted`, `rewired`, `no_entangle` 네 변형이 120쌍 의존성 오차 0.132~0.136으로
전부 겹치며, 얽힘이 **아예 없는** `no_entangle`이 가장 좋다.

진단 결과 (`scripts/diag_benchmark.py`, `scripts/diag_trained.py`):

- 의존성을 전혀 만들지 않는 모델의 오차 = **0.065**. 학습된 모델(0.136)이 그보다 **2배 나쁘다**.
- 참 간선 19쌍 오차: floor 0.368 → aligned 0.366. **얽힘 게이트가 기여한 것이 사실상 0.**
- 비간선 101쌍에 가짜 의존성 0.093을 만든다. 얽힘이 0개인 모델도 0.086을 만든다
  → 그 정체는 **모든 특징이 공유하는 조건 벡터 `c`**이고, 평가 지표가 그걸 통제하지 않는다
  (CDG는 `c` 조건부로 정의되는데 지표는 무조건부 부분상관을 잰다 = 추정량 불일치).

즉 **WGAN-GP critic이 `gamma`에 쓸모 있는 경사를 주지 못하고 있다.**
표현력 문제인지 학습 문제인지는 `scripts/ceiling_joint.py`가 판정한다
(GAN을 빼고 120쌍 패턴에 직접 경사하강).

## 데이터

MIMIC-IV / eICU 는 PhysioNet **credentialed access + DUA** 대상이다.
원본도, 그로부터 파생된 **환자 수준** 데이터도 이 저장소에 절대 들어오지 않는다.
전부 로컬 아카이브에만 둔다. 경로와 획득 방법은 `DATA.md`.

## 환경

```
conda env create -f environment.yml   # env 이름: cdg-qgan
```

**numpy 는 2.2.6 에 고정되어 있다. 올리지 말 것** — torch 2.11+cu128 이 numpy 2.2 ABI 로
빌드되어 있어 `shm.dll` 로딩이 깨진다. 추가 설치는 반드시 제약파일과 함께:

```
python -m pip install -c .backup/constraints.txt <pkg>
```

`tools/mimic-code` 는 별도로 클론한다:

```
git clone https://github.com/MIT-LCP/mimic-code tools/mimic-code
```

## 문서

- `CDG-QGAN_research_plan_v2_ko.md` — 연구 계획 (기준 문서)
- `REVISIONS.md` — **v2와 충돌하면 이쪽이 우선한다.** 리뷰 반영 사항
- `HANDOFF.md` — 작업 패키지 명세 (WP-2 ~ WP-6)
- `DATA.md` — 데이터 위치와 환경 함정

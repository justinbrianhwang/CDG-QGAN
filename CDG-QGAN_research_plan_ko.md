---
title: "CDG-QGAN: 임상 의존성 그래프 기반 하이브리드 양자 GAN을 이용한 합성 중환자 데이터 생성"
subtitle: "단일 저자 연구계획서"
lang: ko-KR
version: "1.0"
date: "2026-07-11"
---

# CDG-QGAN 연구계획서

## 정식 연구명

### 국문

**CDG-QGAN: 임상 의존성 그래프 기반 하이브리드 양자–고전 GAN을 이용한 프라이버시 감사형 합성 중환자 데이터 생성**

### 영문

**CDG-QGAN: A Clinical Dependency Graph-Guided Hybrid Quantum–Classical Generative Adversarial Network for Privacy-Audited Synthetic ICU Data**

### 약어

**CDG-QGAN** = **C**linical **D**ependency **G**raph-guided **Q**uantum **G**enerative **A**dversarial **N**etwork

> **명칭 선택 이유**  
> `CDG-QGAN`은 본 연구의 핵심 신규성인 “임상 변수 의존성 그래프를 양자 회로의 얽힘 구조에 반영한다”는 점을 이름만으로 전달한다. `ClinQGAN`보다 방법론적 기여가 선명하며, 의료 응용과 회로 설계의 연결 고리를 강조하기에 적합하다. 다만 CDG는 인과 그래프가 아니라 **통계적·임상적 의존성 그래프**이므로, 논문 전반에서 causal graph 또는 causal discovery라는 표현을 사용하지 않는다.

---

# 초록형 연구 요약

전자건강기록(Electronic Health Record, EHR)은 의료 인공지능 연구에 필수적이지만, 환자 프라이버시와 데이터 사용 제약으로 인해 접근과 공유가 제한된다. 합성 의료 데이터는 실제 환자 데이터의 분포와 임상적 관계를 보존하면서 데이터 접근성을 높일 수 있는 대안이지만, 기존 생성모델은 혼합형 변수, 결측 구조, 희귀 환자군, 변수 간 임상적 의존성, 프라이버시 위험을 동시에 다루기 어렵다.

본 연구는 임상 변수 사이의 의존 관계를 **Clinical Dependency Graph(CDG)**로 구성하고, 해당 그래프를 매개변수화 양자 회로(Parameterized Quantum Circuit, PQC)의 얽힘 토폴로지로 변환하는 하이브리드 양자–고전 생성모델 **CDG-QGAN**을 제안한다. 제안 모델은 소수의 큐비트로 양자 잠재표현을 생성하고, 고전형 type-specific decoder를 통해 연속형·범주형·이진형·결측 마스크를 포함한 합성 중환자 레코드로 복원한다. 생성 학습에는 conditional WGAN-GP 목적함수, 임상 제약 손실, 변수 의존 구조 보존 손실, 조건 일관성 손실을 결합한다.

연구의 핵심 검증 대상은 무조건적인 “양자 우위”가 아니다. 동일한 decoder, critic, 손실함수, 출력 차원 및 유사한 학습 파라미터 수를 갖는 고전 MLP 생성기와 비교하여, CDG 기반 PQC가 의료 표형 데이터 생성에 유용한 귀납 편향(inductive bias) 또는 파라미터 효율성을 제공하는지 분석한다. 또한 큐비트 수, 회로 깊이, shot 수, noise model, transpiled two-qubit gate 수에 따른 성능 변화를 평가한다.

합성 데이터는 통계적 유사성만으로 프라이버시가 보장되지 않으므로, 본 연구의 기본 모델은 **privacy-preserving**이 아니라 **privacy-audited**로 표현한다. Membership inference, singling-out, linkability, attribute inference, nearest-neighbor memorization을 종합 평가하며, 별도 확장 모델에서는 DP-SGD를 적용하여 \((\varepsilon,\delta)\)-differential privacy와 utility 간 trade-off를 분석한다.

---

# 1. 연구 배경과 필요성

## 1.1 의료 데이터 접근성과 프라이버시 문제

EHR에는 인구학적 정보, 활력징후, 검사 결과, 진단, 처치 및 임상 결과가 포함되어 의료 인공지능 모델 개발에 높은 가치를 갖는다. 그러나 의료 데이터는 다음 이유로 연구 활용이 제한된다.

- 환자 식별 및 재식별 위험
- 기관별 데이터 사용 계약과 윤리 심의
- 희귀 질환 및 소수 환자군의 표본 부족
- 기관 간 분포 차이와 데이터 표준화 문제
- 결측, 불균형, 혼합형 자료형이 결합된 복잡한 구조

단순한 비식별화만으로는 고차원 속성 조합을 통한 재식별 위험을 제거하기 어렵다. 합성 데이터는 원본 레코드를 직접 배포하지 않고 통계적 패턴을 학습하여 새로운 표본을 생성한다는 장점이 있지만, 생성모델이 훈련 샘플을 암기하면 실제 환자 정보가 노출될 수 있다. 따라서 합성 데이터의 품질과 프라이버시를 동시에 평가해야 한다.

## 1.2 합성 EHR 생성 연구의 발전

medGAN은 autoencoder와 GAN을 결합하여 고차원 이산형 환자 기록을 생성하였다 [4]. CTGAN은 다중 모드 연속형 변수와 불균형 범주형 변수를 처리하는 조건부 표형 GAN을 제안하였다 [5]. EHR-M-GAN은 연속형과 이산형의 혼합형 종단 EHR을 동시에 생성하고, 실제 ICU 데이터에서 fidelity, utility 및 privacy를 평가하였다 [6]. CTAB-GAN+는 혼합형 표형 데이터 처리, WGAN-GP 및 DP-SGD를 결합하여 privacy–utility trade-off를 다루었다 [7]. TabDDPM은 mixed-type tabular data에 diffusion model을 적용하여 GAN 및 VAE 계열과 비교되는 강력한 고전 기준선을 제공한다 [8].

이러한 연구는 합성 EHR이 단순한 변수별 평균 재현을 넘어 다변량 관계, downstream utility, 결측 구조 및 프라이버시를 함께 평가해야 함을 보여준다.

## 1.3 양자 생성모델과 표형 데이터

QGAN은 양자 생성기와 고전 또는 양자 판별자의 적대적 학습을 통해 분포를 학습하는 양자 생성모델 계열이다 [9,10]. 실제 초전도 양자 프로세서에서 이미지 분포 생성을 시연한 연구도 존재하지만, 현재 NISQ 장치에서는 제한된 큐비트 수, noise, shot 비용, gradient estimation 비용 및 barren plateau가 주요 제약이다 [11,12].

최근 표형 데이터에 특화된 연구로 TabularQGAN은 MIMIC-III 및 Adult Census에서 양자 생성모델을 CTGAN 및 CopulaGAN과 비교하였고 [13], QTabGAN은 PQC가 생성한 \(2^n\) 차원 확률 벡터를 고전 신경망으로 표형 변수에 매핑하는 구조를 제안하였다 [14]. 따라서 “QGAN을 의료 표형 데이터에 적용했다”는 사실만으로는 충분한 신규성이 되기 어렵다.

## 1.4 기존 연구의 공백

| 연구 공백 | 기존 접근의 한계 | 본 연구의 대응 |
|---|---|---|
| 의료 변수 구조와 양자 회로의 불일치 | ring, linear, all-to-all 등 의미 비의존적 얽힘 사용 | CDG를 구축하고 이를 entanglement topology로 변환 |
| 양자 효과의 불명확성 | decoder 크기나 전체 파라미터가 달라 공정한 비교가 어려움 | parameter-matched 및 output-matched classical generator 구성 |
| 혼합형·결측 EHR 처리 | 작은 수의 연속형 특징 또는 단순 binary 데이터에 집중 | 연속형, 범주형, 이진형, 결측 마스크를 공동 생성 |
| 임상적 관계 평가 부족 | 변수별 유사도 중심 | 생리학적 제약과 임상 의존 구조 보존을 손실 및 평가에 반영 |
| 프라이버시 과장 | 합성이라는 이유로 자동적인 익명성을 가정 | 공격 기반 privacy audit와 formal DP를 분리 |
| NISQ 실용성 분석 부족 | ideal simulator 결과만 보고 | shot, noise, depth, gate count, gradient variance 분석 |

---

# 2. 연구 목표

## 2.1 주요 목표

임상 변수 간 의존성을 양자 회로의 얽힘 구조에 반영한 하이브리드 생성모델 **CDG-QGAN**을 설계하고, 실제 중환자 데이터의 통계적 충실도, 임상적 개연성, 머신러닝 활용성, 다양성, 프라이버시 위험 및 NISQ 자원 효율성을 종합 평가한다.

## 2.2 세부 목표

1. MIMIC-IV 첫 ICU 입실 후 첫 24시간 데이터를 기반으로 혼합형 정적 환자 테이블을 구축한다.
2. 자료형별 연관도와 공개된 생리학적 관계를 결합하여 Clinical Dependency Graph를 구축한다.
3. CDG의 커뮤니티 구조를 큐비트에 매핑하고, 그래프 간선을 PQC의 entangling gate로 변환한다.
4. expectation value 기반 quantum latent representation과 type-specific decoder를 결합한다.
5. 임상 제약 및 의존 구조를 보존하는 학습 목적함수를 설계한다.
6. 강력한 고전 생성모델 및 자원 통제형 MLP generator와 비교한다.
7. 공격 기반 프라이버시 평가와 선택적 differential privacy 확장을 수행한다.
8. 큐비트 수, 회로 깊이, shot 수, noise 및 hardware topology에 따른 강건성을 분석한다.

---

# 3. 연구 질문과 가설

| 연구 질문 | 가설 |
|---|---|
| **RQ1.** CDG 기반 양자 잠재 생성기가 동일 규모의 고전 잠재 생성기보다 유용한가? | **H1.** 동일 decoder·critic·loss 조건에서 CDG-QGAN은 parameter-matched MLP generator보다 CDG edge-weighted dependency error가 작거나, TSTR utility에서 비열등하다. |
| **RQ2.** 임상 의존성 기반 얽힘 구조가 일반적인 회로 연결보다 효과적인가? | **H2.** 동일한 two-qubit edge 수를 갖는 ring 및 random sparse ansatz보다 CDG ansatz가 상관 및 조건부 의존 구조를 더 정확히 보존한다. |
| **RQ3.** 임상 제약 손실이 생성 레코드의 개연성을 높이는가? | **H3.** Clinical Constraint Loss는 생리학적 위반율을 감소시키며, utility 저하는 사전 허용 범위 이내에 머문다. |
| **RQ4.** 생성 품질 향상은 프라이버시 위험과 어떤 관계를 갖는가? | **H4.** fidelity가 높아질수록 일부 희귀 표본에서 membership 및 attribute inference 위험이 증가할 수 있으며, DP 적용은 공격 우위를 낮추는 대신 utility를 감소시킨다. |
| **RQ5.** 제안 모델은 NISQ 조건에서 얼마나 강건한가? | **H5.** sparse CDG ansatz는 gate 수가 많은 회로보다 calibrated noise에서 작은 성능 저하와 안정적인 gradient를 보인다. |
| **RQ6.** 양자 회로의 개선이 단순한 모델 크기 차이로 설명되는가? | **H6.** parameter-matched, output-matched, compute-reported 비교 후에도 CDG-QGAN의 일부 구조 보존 이점이 유지된다. |

---

# 4. 연구의 핵심 기여

## 4.1 CDG 기반 양자 회로 설계

의료 변수 간 안정적인 의존 관계를 추정하고, 이를 큐비트 커뮤니티 및 entanglement topology로 변환하는 재현 가능한 절차를 제안한다.

\[
\text{Clinical variables}
\rightarrow
\text{Dependency graph}
\rightarrow
\text{Feature communities}
\rightarrow
\text{Qubit mapping}
\rightarrow
\text{Entanglement topology}
\]

## 4.2 파라미터 및 출력 차원이 통제된 비교

양자 회로를 동일한 latent output dimension과 유사한 trainable parameter 수를 갖는 고전 MLP로 교체한다. decoder, critic, loss 및 training protocol을 동일하게 유지하여 양자 코어의 효과를 분리한다.

## 4.3 임상 구조 보존 목적함수

WGAN-GP에 혈압 관계, 생리학적 범위, 변수 의존 행렬, 조건 일관성 및 결측 구조를 반영하는 손실을 결합한다.

## 4.4 종합적 privacy audit

Membership inference뿐 아니라 singling-out, linkability, attribute inference 및 memorization을 평가하고, 희귀 환자 subgroup의 위험을 별도로 분석한다.

## 4.5 NISQ 실용성 분석

모델 성능만이 아니라 큐비트, 깊이, shot, noise, transpiled gate count, gradient variance, 회로 호출 수 및 wall-clock cost를 보고한다.

---

# 5. 연구 범위와 주장 경계

## 5.1 본 연구에서 주장할 수 있는 내용

- CDG 기반 회로 설계가 특정 의료 표형 데이터 생성에서 유용한 inductive bias인지 여부
- 동일 자원 조건의 고전 모델 대비 fidelity, utility, clinical consistency 및 parameter efficiency
- NISQ 조건에서 성능이 어떻게 저하되는지
- 공격 기반 privacy risk와 DP 적용 시의 privacy–utility trade-off

## 5.2 본 연구에서 자동으로 주장할 수 없는 내용

- 계산복잡도 관점의 보편적 quantum advantage
- 합성 데이터의 완전한 익명성
- 임상적 안전성 또는 실제 진료 적용 가능성
- CDG 간선의 인과관계
- simulator 결과만으로 실제 양자 하드웨어 우위 입증
- 적은 파라미터 수만을 근거로 한 전체 계산 효율성 우위

논문의 중심 문장은 다음과 같이 설정한다.

> **We do not claim unconditional quantum advantage. We investigate whether a clinical-dependency-guided parameterized quantum circuit provides a useful and resource-efficient inductive bias for mixed-type ICU data synthesis under matched classical baselines and NISQ constraints.**

---

# 6. 데이터 설계

## 6.1 주 데이터셋

주 데이터셋은 **MIMIC-IV v3.1**을 사용한다 [15,16]. MIMIC-IV는 Beth Israel Deaconess Medical Center의 비식별화된 응급실 및 중환자실 데이터를 포함한다. 데이터 접근에는 PhysioNet credentialing, 요구 교육 이수 및 데이터 사용 동의가 필요하다.

원본 데이터와 환자 단위 파생 데이터는 공개하지 않으며, 다음 항목을 공개 가능한 재현성 산출물로 설정한다.

- 코호트 정의
- SQL 추출 코드
- 변수 사전
- 전처리 코드
- 모델 및 평가 코드
- 하이퍼파라미터 설정
- random seed
- 회로 및 transpilation 설정

## 6.2 분석 단위

첫 논문의 범위는 종단 시계열 전체가 아니라 다음 형태의 **환자별 정적 혼합형 테이블**로 제한한다.

- 환자별 최초 ICU 입실
- ICU 입실 시점부터 첫 24시간
- 활력징후의 최소값, 평균값, 최대값 또는 임상적으로 선택된 요약값
- 검사 결과의 첫값, 최솟값, 최댓값 또는 대표값
- 치료 여부 및 입원 정보
- 변수별 결측 마스크
- 병원 내 사망 여부

이 범위는 전체 longitudinal EHR 생성보다 단일 저자가 구현·검증하기 현실적이며, 양자 생성기의 방법론적 기여를 분명히 보여주기 적합하다.

## 6.3 포함 기준

- 만 18세 이상 성인
- 환자별 최초 ICU 입실
- ICU 입실 후 관측 가능한 임상 기록이 존재하는 환자
- 주 outcome 및 핵심 인구학적 변수 식별 가능

## 6.4 제외 및 편향 관리

24시간 이전에 사망, 전실 또는 퇴실한 환자를 일괄 제외하면 생존 시간이 긴 환자만 남는 selection bias가 발생할 수 있다. 따라서 관측 가능한 기간의 값을 사용하고, 미관측 변수는 missing mask로 표현한다.

명백한 데이터 오류와 단위 오류는 사전 정의한 규칙에 따라 결측으로 처리한다. 극단값을 무조건 제거하지 않고, 임상적으로 가능한 희귀값과 물리적으로 불가능한 값을 구분한다.

## 6.5 변수 구성

목표 특징 수는 약 30~45개로 설정한다.

| 변수군 | 후보 변수 |
|---|---|
| 인구학 | 나이, 성별, 인종 범주 |
| 입원 정보 | 입원 유형, ICU 유형, 응급 입원 여부 |
| 활력징후 | 심박수, 수축기 혈압, 이완기 혈압, 평균동맥압, 호흡수, 산소포화도, 체온 |
| 신장·전해질 | 크레아티닌, BUN, 나트륨, 칼륨, 염소 |
| 대사·산염기 | 혈당, 젖산, 중탄산염 |
| 혈액학 | 백혈구, 헤모글로빈, 헤마토크릿, 혈소판 |
| 치료 | 기계환기 여부, 혈압상승제 사용 여부 |
| 결측 정보 | 변수별 관측 여부 또는 변수군별 mask |
| 조건/결과 | 병원 내 사망 여부 |

변수 선택은 결측률, 임상적 중요도, 데이터 추출 안정성 및 중복성을 기준으로 training set에서 사전 고정한다.

## 6.6 조건 변수

주 조건은 병원 내 사망 여부로 설정한다.

\[
y \in \{0,1\}, \qquad y=1:\text{in-hospital mortality}
\]

조건부 생성은 단순한 전체 분포 복제뿐 아니라 소수 사망군의 생성 성능을 별도로 평가할 수 있게 한다. 추가 조건으로 ICU 유형을 고려할 수 있으나, 첫 논문의 핵심 조건은 하나로 제한한다.

## 6.7 데이터 분할

동일 환자의 여러 입원이 서로 다른 분할에 들어가지 않도록 `subject_id` 단위로 분할한다.

\[
\mathcal D
=
\mathcal D_{train}
\cup
\mathcal D_{validation}
\cup
\mathcal D_{test}
\]

권장 비율은 다음과 같다.

\[
70\% : 15\% : 15\%
\]

- 사망 여부를 기준으로 층화
- 전처리 통계량은 training set에서만 추정
- CDG는 training set에서만 구축
- validation set은 하이퍼파라미터 및 early stopping에 사용
- test set은 최종 평가에만 사용

---

# 7. 전처리 방법

## 7.1 데이터 품질 정리

1. 동일 임상 항목의 item identifier 통합
2. 단위 변환 및 단위 불일치 탐지
3. 중복 측정 처리 규칙 고정
4. 물리적으로 불가능한 값은 결측 처리
5. 임상적으로 가능한 극단값은 보존하되 robust scaling 적용
6. 모든 규칙을 코드와 변수 사전에 기록

## 7.2 연속형 변수 변환

### 기본안: robust scaling

연속형 변수 \(x_j\)는 training set의 중앙값과 IQR을 이용하여 변환한다.

\[
\tilde{x}_j
=
\frac{x_j-\operatorname{median}_{train}(x_j)}
{\operatorname{IQR}_{train}(x_j)+\epsilon}
\]

필요한 경우 \([-c,c]\) 범위로 clipping한 뒤 \([-1,1]\)로 재조정한다. 역변환은 training set 통계량만 사용한다.

### 확장안: mode-specific normalization

다중 모드 연속형 변수에는 CTGAN 방식의 mode-specific normalization을 적용할 수 있다 [5]. 변수 \(x_j\)에 variational Gaussian mixture를 적합하고 선택된 component \(k\)에 대해

\[
\alpha_j
=
\frac{x_j-\mu_{jk}}{4\sigma_{jk}}
\]

와 component one-hot vector \(\beta_j\)를 함께 표현한다. 생성기는 \(\alpha_j\)와 \(\beta_j\)를 출력하며, \(\beta_j\)는 Gumbel-Softmax를 이용한다.

주 분석에서는 robust scaling과 mode-specific normalization 중 validation 성능이 안정적인 방식을 선택하되, 두 전처리가 모델 순위에 미치는 영향을 보조 실험으로 확인한다.

## 7.3 범주형 변수

- 낮은 cardinality: one-hot representation
- 높은 cardinality: 희귀 범주를 training set 기준으로 `Other`에 통합
- 생성기 출력: Gumbel-Softmax
- 최종 생성: argmax 또는 categorical sampling

Gumbel-Softmax는 학습 중 미분 가능한 근사치를 제공한다.

\[
\tilde c_{jk}
=
\frac{
\exp((\log \pi_{jk}+g_k)/\tau)
}{
\sum_r \exp((\log \pi_{jr}+g_r)/\tau)
}
\]

## 7.4 결측 구조

의료 데이터의 결측은 단순한 무작위 손실이 아니라 검사 선택과 환자 상태를 반영할 수 있으므로, 결측을 모두 KNN imputation으로 제거하지 않는다.

각 변수의 관측 여부를

\[
m_j=
\begin{cases}
1,&x_j\text{ observed}\\
0,&x_j\text{ missing}
\end{cases}
\]

로 정의한다.

훈련 입력은 임시 대체값 \(x_j^{imp}\)와 mask \(m_j\)를 함께 사용한다.

\[
x_j^{model}=m_jx_j^{imp}+(1-m_j)c_j
\]

여기서 \(c_j\)는 scaling 이후의 중립값이다. 생성기는 값과 mask를 공동 출력하며, 최종 합성 레코드에서 \(\tilde m_j=0\)인 위치를 결측으로 복원한다.

## 7.5 클래스 불균형

사망군이 적은 경우 conditional sampling을 적용한다. 각 minibatch에서 조건 \(y\)의 비율을 조정할 수 있지만, 최종 데이터 생성 시에는 다음 두 설정을 구분한다.

1. **Distribution-preserving generation**: 실제 training prevalence를 재현
2. **Balanced augmentation generation**: downstream 증강을 위해 조건별 동일 수 생성

두 결과를 혼합하지 않고 별도로 보고한다.

---

# 8. 제안 방법: CDG-QGAN

## 8.1 전체 구조

```text
Random noise z + clinical condition y
                    │
                    ▼
          Classical angle encoder
                    │
                    ▼
      CDG-guided parameterized quantum circuit
                    │
                    ▼
      <Zi>, <ZiZj> quantum expectation values
                    │
                    ▼
           Type-specific decoder
       ├─ continuous value heads
       ├─ categorical heads
       ├─ binary treatment heads
       └─ missing-mask heads
                    │
                    ▼
           Synthetic ICU record
                    │
                    ▼
        Conditional WGAN-GP critic
```

생성기는 다음과 같이 정의한다.

\[
\tilde{x}
=
G_{\Theta}(z,y)
=
G_{dec}
\left(
Q_{\theta}(E_{\omega}(z,y)),y
\right)
\]

- \(z\): 잠재 난수
- \(y\): 임상 조건
- \(E_{\omega}\): 고전 angle encoder
- \(Q_{\theta}\): CDG-guided PQC 및 측정
- \(G_{dec}\): mixed-type decoder

잠재 난수 \(z\)가 decoder로 직접 우회하는 skip connection은 기본 모델에서 허용하지 않는다. 그렇지 않으면 decoder가 양자 출력을 무시하고 고전 경로만으로 데이터를 생성할 수 있다.

---

# 9. Clinical Dependency Graph 구성

## 9.1 정의

CDG를 다음의 가중 무방향 그래프로 정의한다.

\[
\mathcal G_F=(V_F,E_F,A)
\]

- \(V_F\): 의료 특징 노드
- \(E_F\): 안정적인 의존 관계를 나타내는 간선
- \(A_{jk}\): 특징 \(j,k\) 사이의 연관도

CDG는 인과 그래프가 아니며, 간선 방향이나 intervention effect를 의미하지 않는다.

## 9.2 자료형별 연관도

자료형 조합에 따라 연관도 추정 방법을 다르게 사용한다.

\[
A^{data}_{jk}=
\begin{cases}
|\rho_s(x_j,x_k)|,
& \text{continuous--continuous}\\
V(x_j,x_k),
& \text{categorical--categorical}\\
\eta^2(x_j,x_k),
& \text{continuous--categorical}
\end{cases}
\]

- \(\rho_s\): Spearman correlation
- \(V\): Cramér's V
- \(\eta^2\): correlation ratio

비선형 관계를 보완하기 위해 normalized mutual information 또는 distance correlation을 secondary association으로 사용할 수 있다.

## 9.3 결측을 고려한 연관도

각 쌍의 연관도는 두 변수가 동시에 관측된 샘플에서 계산하되, 표본 수가 작은 쌍은 제외하거나 shrinkage를 적용한다. 결측 마스크 자체도 별도 노드 또는 별도 association matrix로 분석한다.

\[
A^{mask}_{jk}=\operatorname{Assoc}(m_j,m_k)
\]

값 관계와 결측 관계가 혼동되지 않도록 두 행렬을 분리하여 기록한다.

## 9.4 임상 사전 관계

공개된 생리학적 정의나 명백한 변수군 관계를 prior edge로 추가한다.

예시:

- \(SBP\leftrightarrow DBP\leftrightarrow MAP\)
- \(Hemoglobin\leftrightarrow Hematocrit\)
- \(Creatinine\leftrightarrow BUN\)
- \(Respiratory\ Rate\leftrightarrow SpO_2\)
- \(Lactate\leftrightarrow Bicarbonate\)

사전 간선은 인과관계가 아니라 구조적 관련성을 나타낸다. 단일 저자 연구에서는 임상의 검토가 없으므로, 문헌 또는 임상 정의가 명확한 관계만 사용한다.

## 9.5 데이터 기반 관계와 prior 결합

\[
A_{jk}
=
\lambda_A A^{data}_{jk}
+
(1-\lambda_A)A^{prior}_{jk}
\]

여기서 \(A^{prior}_{jk}\in\{0,1\}\) 또는 신뢰도 가중치이며, \(\lambda_A\)는 validation을 통해 결정한다.

## 9.6 Bootstrap 안정성 선택

우연한 표본 상관에 의해 회로 구조가 결정되는 것을 막기 위해 training set에서 bootstrap을 수행한다.

각 bootstrap \(b\)에서 상위 연관 간선 집합을 \(E^{(b)}\)라 할 때,

\[
\pi_{jk}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbb{I}[(j,k)\in E^{(b)}]
\]

안정성 \(\pi_{jk}\geq\tau_{stable}\)인 간선을 우선 선택한다. 기본 후보는 \(B=100\), \(\tau_{stable}=0.7\)이다.

## 9.7 특징 커뮤니티 구성

특징 수 \(d\)는 큐비트 수 \(n_q\)보다 크므로, 특징을 \(n_q\)개의 커뮤니티로 묶는다.

\[
\mathcal C=\{C_1,\ldots,C_{n_q}\}
\]

가능한 방법:

- spectral clustering
- Louvain community detection
- normalized-cut clustering

클러스터링 목적은 높은 내부 연관도와 낮은 외부 연관도를 갖도록 하는 것이다.

\[
\max_{\mathcal C}
\sum_{u=1}^{n_q}
\sum_{j,k\in C_u}A_{jk}
-
\lambda_C
\sum_{u\neq v}
\sum_{j\in C_u,k\in C_v}A_{jk}
\]

커뮤니티 결과는 임상적으로 해석 가능한지 검토하고, seed에 따른 안정성을 보고한다.

## 9.8 큐비트 수준 그래프

커뮤니티 \(C_u,C_v\) 사이의 가중치는 다음과 같이 정의한다.

\[
W_{uv}
=
\frac{1}{|C_u||C_v|}
\sum_{j\in C_u}
\sum_{k\in C_v}
A_{jk}
\]

연결된 그래프를 보장하기 위해 maximum spanning tree를 먼저 구성하고, 추가로 상위 가중치 간선을 선택한다.

\[
E_{CDG}
=
E_{MST}
\cup
E_{top-r}
\]

ring 및 random sparse ansatz와의 공정한 비교를 위해 two-qubit edge 수를 동일하게 맞춘다.

---

# 10. 생성기 구조

## 10.1 잠재변수와 조건 임베딩

\[
z\sim\mathcal U(-1,1)^{d_z}
\]

조건 \(y\)는 embedding으로 변환한다.

\[
h_0=[z;e_y(y)]
\]

기본 잠재 차원 후보는 \(d_z\in\{8,16,32\}\)이다.

## 10.2 Classical angle encoder

PQC에 입력할 각도를 고전 신경망이 생성한다.

\[
\phi^{(a)}_{\ell,q}
=
\pi\tanh
\left(
W^{(a)}_{\ell,q}h_0+b^{(a)}_{\ell,q}
\right)
\]

- \(\ell\): 회로 층
- \(q\): 큐비트
- \(a\in\{Y,Z\}\): 회전축

`tanh`를 이용해 각도를 제한하면 초기 상태가 지나치게 무작위화되는 현상을 완화할 수 있다.

## 10.3 초기 상태

기본 상태는

\[
|\psi_0\rangle=|0\rangle^{\otimes n_q}
\]

로 설정한다. 보조 실험으로 각 큐비트에 Hadamard gate를 적용한 균등 중첩 초기화를 비교할 수 있다.

## 10.4 Data re-uploading PQC

전체 회로는 \(L\)개의 반복 층으로 구성한다.

\[
U_{\theta}(z,y)
=
\prod_{\ell=1}^{L}
U_{ent}^{(\ell)}
U_{var}^{(\ell)}
U_{enc}^{(\ell)}(z,y)
\]

### 입력 인코딩

\[
U_{enc}^{(\ell)}
=
\prod_{q=1}^{n_q}
R_Y(\phi^{Y}_{\ell,q})
R_Z(\phi^{Z}_{\ell,q})
\]

### 학습 가능한 단일 큐비트 회전

\[
U_{var}^{(\ell)}
=
\prod_{q=1}^{n_q}
R_Y(\theta^{(1)}_{\ell,q})
R_Z(\theta^{(2)}_{\ell,q})
R_Y(\theta^{(3)}_{\ell,q})
\]

### CDG 기반 얽힘

\[
U_{ent}^{(\ell)}
=
\prod_{(u,v)\in E_{CDG}}
R_{ZZ}^{(u,v)}(\gamma_{\ell,uv})
\]

또는 hardware-efficient 비교를 위해 CNOT–rotation 구조를 사용할 수 있다.

\[
R_{ZZ}(\gamma)
=
\operatorname{CNOT}
\left(I\otimes R_Z(\gamma)\right)
\operatorname{CNOT}
\]

## 10.5 그래프 가중치 활용 방식

다음 두 방식을 비교한다.

### Topology-only

CDG는 연결 구조만 결정하고 모든 \(\gamma_{\ell,uv}\)를 독립 학습한다.

### Weight-informed initialization

\[
\gamma_{\ell,uv}^{(0)}
=\alpha_{init}W_{uv}+\epsilon_{uv}
\]

CDG edge weight를 초기화에만 사용하고 이후 자유롭게 학습한다. 이 방식은 데이터 의존적 inductive bias를 제공하지만 회로 표현력을 지나치게 제한하지 않는다.

## 10.6 큐비트 수와 깊이

\[
n_q\in\{4,6,8,10\}
\]

\[
L\in\{1,2,3,4\}
\]

주 모델 후보는 8큐비트, 2~3층으로 설정한다. 최종 설정은 validation 성능, two-qubit gate 수 및 gradient 안정성을 함께 고려하여 선택한다.

---

# 11. 양자 측정과 잠재표현

## 11.1 단일 큐비트 측정

\[
q_i=\langle Z_i\rangle
\]

## 11.2 CDG 간선 측정

\[
q_{uv}=\langle Z_uZ_v\rangle,
\qquad (u,v)\in E_{CDG}
\]

## 11.3 최종 quantum latent vector

\[
q(z,y)=
\left[
\langle Z_1\rangle,\ldots,\langle Z_{n_q}\rangle,
\{\langle Z_uZ_v\rangle\}_{(u,v)\in E_{CDG}}
\right]
\]

출력 차원은

\[
d_q=n_q+|E_{CDG}|
\]

이다. 예를 들어 8큐비트와 10개 CDG 간선을 사용하면 18차원 quantum latent를 얻는다.

## 11.4 측정 ablation

- 단일 \(Z\) expectation만 사용
- \(Z+ZZ\) expectation 사용
- 모든 pairwise \(ZZ\) 사용
- CDG 간선에 해당하는 \(ZZ\)만 사용

이를 통해 pairwise quantum measurement가 실제 구조 보존에 기여하는지 검증한다.

---

# 12. Type-Specific Decoder

## 12.1 공유 표현

\[
h_1=\operatorname{LeakyReLU}(W_1[q;e_y(y)]+b_1)
\]

\[
h_2=\operatorname{LeakyReLU}(W_2h_1+b_2)
\]

기본 구조 후보는 \(d_q\rightarrow128\rightarrow256\) 또는 \(d_q\rightarrow128\rightarrow128\)이다.

## 12.2 연속형 출력 head

Robust scaling을 사용할 경우

\[
\tilde{x}^{cont}_j
=
\tanh(W^{cont}_jh_2+b^{cont}_j)
\]

를 사용한다.

Mode-specific normalization을 사용할 경우 각 변수는 normalized scalar와 mixture component logits를 함께 출력한다.

## 12.3 범주형 출력 head

\[
\tilde{x}^{cat}_j
=
\operatorname{GumbelSoftmax}
(W^{cat}_jh_2+b^{cat}_j;\tau)
\]

## 12.4 이진형 출력 head

\[
\tilde{x}^{bin}_j
=
\sigma(W^{bin}_jh_2+b^{bin}_j)
\]

학습 중에는 Binary Concrete 또는 straight-through estimator를 사용한다.

## 12.5 결측 mask head

\[
\tilde{m}_j
=
\sigma(W^{mask}_jh_2+b^{mask}_j)
\]

최종 생성값은

\[
\tilde{x}^{final}_j
=
\begin{cases}
\tilde{x}_j,&\tilde m_j=1\\
\text{missing},&\tilde m_j=0
\end{cases}
\]

로 복원한다.

## 12.6 Quantum bottleneck 검증

Decoder가 quantum latent를 실제로 사용하는지 확인하기 위해 다음 분석을 수행한다.

- quantum latent permutation test
- latent zeroing test
- decoder input gradient norm
- latent dimension별 mutual information 추정
- PQC 출력을 고정했을 때 생성 다양성 변화

PQC 출력 순서를 무작위로 섞거나 0으로 치환했을 때 성능이 변하지 않는다면, decoder가 양자 표현을 유의미하게 활용하지 않는 것으로 해석한다.

---

# 13. Conditional Critic

## 13.1 기본 구조

```text
Input record x + missing mask m
             │
             ▼
        Linear(256)
      + LeakyReLU
             │
             ▼
        Linear(128)
      + LeakyReLU
             │
             ▼
         Linear(64)
      + LeakyReLU
             │
             ▼
       scalar critic score
```

Critic에는 Batch Normalization을 사용하지 않는다. Gradient penalty가 개별 입력 gradient를 사용하는 상황에서 batch-dependent normalization이 간섭할 수 있기 때문이다.

## 13.2 Projection conditioning

조건 \(y\)는 projection critic 방식으로 반영할 수 있다.

\[
D_{\psi}(x,y)
=f_{\psi}(x)+h_{\psi}(x)^Te_D(y)
\]

단순 concatenate 방식과 비교하여 안정적인 방식을 선택한다.

---

# 14. 목적함수

## 14.1 WGAN-GP Critic Loss

\[
\mathcal L_D
=
\mathbb E_{\tilde{x}\sim P_G}[D_{\psi}(\tilde{x},y)]
-
\mathbb E_{x\sim P_r}[D_{\psi}(x,y)]
+
\lambda_{gp}\mathcal L_{gp}
\]

\[
\mathcal L_{gp}
=
\mathbb E_{\hat{x}}
\left[
\left(
\|\nabla_{\hat{x}}D_{\psi}(\hat{x},y)\|_2-1
\right)^2
\right]
\]

\[
\hat{x}=\alpha x+(1-\alpha)\tilde{x},
\qquad
\alpha\sim\mathcal U(0,1)
\]

## 14.2 Generator Adversarial Loss

\[
\mathcal L_{adv}
=-\mathbb E_{\tilde{x}\sim P_G}[D_{\psi}(\tilde{x},y)]
\]

## 14.3 Clinical Constraint Loss

### 혈압 순서 관계

\[
DBP\leq MAP\leq SBP
\]

\[
\mathcal L_{order}
=
\mathbb E
\left[
\operatorname{ReLU}(DBP-MAP)
+
\operatorname{ReLU}(MAP-SBP)
\right]
\]

### 평균동맥압 근사 관계

\[
MAP\approx\frac{SBP+2DBP}{3}
\]

\[
\mathcal L_{MAP}
=
\mathbb E
\left[
\operatorname{Huber}
\left(
MAP-\frac{SBP+2DBP}{3}
\right)
\right]
\]

이 관계는 근사식이므로 hard equality가 아니라 soft penalty로 사용한다.

### 생리학적 범위

변수 \(j\)의 허용 범위를 \([l_j,u_j]\)라고 할 때,

\[
\mathcal L_{range}
=
\sum_j
\mathbb E
\left[
\operatorname{ReLU}(l_j-\tilde{x}_j)
+
\operatorname{ReLU}(\tilde{x}_j-u_j)
\right]
\]

### 임상 손실 결합

\[
\mathcal L_{clin}
=
\mathcal L_{order}
+
\alpha_{MAP}\mathcal L_{MAP}
+
\alpha_{range}\mathcal L_{range}
\]

진단 기준이나 치료 지침을 단일 검사 수치로 강제하는 hard rule은 사용하지 않는다. 실제 임상 데이터에는 치료 효과, 측정 오류, 진단 지연 및 예외가 존재하기 때문이다.

## 14.4 Dependency Structure Loss

연속형 변수의 batch correlation matrix를 각각 \(R^{real}\), \(R^{syn}\)이라 하면

\[
\mathcal L_{corr}
=
\|R^{real}-R^{syn}\|_F^2
\]

CDG 주요 간선에 가중치를 둔 구조 손실은

\[
\mathcal L_{struct}
=
\sum_{(j,k)\in E_F}
\omega_{jk}
\left(
\hat a^{real}_{jk}
-
\hat a^{syn}_{jk}
\right)^2
\]

로 정의한다.

범주형 변수는 soft category probability로 differentiable contingency table을 구성한다. Mixed-type association을 완전히 미분 가능하게 구현하기 어려운 경우, 연속형 correlation loss를 학습에 사용하고 mixed-type association은 평가 지표로만 사용한다.

## 14.5 Missingness Loss

변수별 결측률 보존:

\[
\mathcal L_{miss,marg}
=
\sum_j
\left|
\mathbb E[m_j]
-
\mathbb E[\tilde m_j]
\right|
\]

결측 마스크 간 상관 보존:

\[
\mathcal L_{miss,corr}
=
\|R_m^{real}-R_m^{syn}\|_F^2
\]

\[
\mathcal L_{miss}
=
\mathcal L_{miss,marg}
+
\alpha_m\mathcal L_{miss,corr}
\]

## 14.6 Condition Consistency Loss

Training set으로 auxiliary classifier \(C_{\xi}\)를 사전 학습한 뒤 고정한다.

\[
\mathcal L_{cond}
=
\operatorname{BCE}(C_{\xi}(\tilde{x}),y)
\]

최종 utility 평가는 auxiliary classifier와 다른 알고리즘을 포함하여 순환 평가를 피한다.

## 14.7 선택적 Diversity Regularization

Mode collapse가 관찰되는 경우에만 다음 regularizer를 ablation으로 사용한다.

\[
\mathcal L_{div}
=-
\frac{
\|G(z_1,y)-G(z_2,y)\|_1
}{
\|z_1-z_2\|_1+\epsilon
}
\]

모든 모델에 자동으로 적용하지 않고, collapse 대응 실험으로 분리한다.

## 14.8 전체 생성기 손실

\[
\boxed{
\mathcal L_G
=
\mathcal L_{adv}
+
\lambda_{clin}\mathcal L_{clin}
+
\lambda_{struct}\mathcal L_{struct}
+
\lambda_{miss}\mathcal L_{miss}
+
\lambda_{cond}\mathcal L_{cond}
+
\lambda_{div}\mathcal L_{div}
}
\]

## 14.9 손실 가중치 ramp-up

학습 초기부터 구조 손실을 크게 적용하면 평균적이고 제한된 샘플만 생성할 수 있다. 따라서 adversarial warm-up 이후 가중치를 점진적으로 증가시킨다.

\[
\lambda_k(t)
=
\lambda_{k,max}
\min
\left(
1,
\frac{t-t_0}{T_{ramp}}
\right)
\]

이는 연구 일정이 아니라 단일 학습 실행 내부의 최적화 절차이다.

---

# 15. 학습 알고리즘

## Algorithm 1. CDG-QGAN 학습

```text
Input:
    Training data Dtrain
    Clinical condition y
    CDG graph GCDG
    Generator parameters Θ = {ω, θ, decoder parameters}
    Critic parameters ψ

1. Fit preprocessing transforms using Dtrain only.
2. Construct stable feature-level CDG using bootstrap associations.
3. Cluster features into nq communities and build qubit-level CDG.
4. Initialize angle encoder, PQC, decoder, and conditional critic.
5. Repeat until stopping criterion:
    a. Repeat ncritic times:
        i.   Sample real minibatch (x, m, y).
        ii.  Sample z and generate (x~, m~) = GΘ(z, y).
        iii. Interpolate real and synthetic samples.
        iv.  Compute WGAN-GP critic loss.
        v.   Update ψ.
    b. Sample z and condition y.
    c. Compute quantum latent expectations <Zi> and <ZiZj>.
    d. Decode mixed-type synthetic records.
    e. Compute adversarial, clinical, structure,
       missingness, and condition losses.
    f. Update Θ using hybrid automatic differentiation.
    g. Record quantum/classical gradient norms and training stability.
6. Select checkpoint using validation composite score.
7. Freeze model and generate evaluation datasets.
8. Evaluate fidelity, clinical plausibility, utility,
   diversity, privacy, and NISQ robustness on held-out data.
```

## 15.1 Checkpoint 선택

단일 지표만 최적화하지 않고 validation composite score를 사용한다.

\[
S_{val}
=w_fS_{fidelity}
+w_uS_{utility}
+w_cS_{clinical}
-w_pR_{privacy-proxy}
\]

Test set은 checkpoint 선택에 사용하지 않는다. Privacy proxy는 validation 단계의 간단한 nearest-neighbor gap으로 제한하고, 최종 공격 평가는 test 단계에서 수행한다.

---

# 16. Differential Privacy 확장

## 16.1 명칭 구분

- **CDG-QGAN**: 공격 기반으로 프라이버시를 감사하는 기본 모델
- **DP-CDG-QGAN**: 명시적인 \((\varepsilon,\delta)\)-DP를 만족하도록 학습한 확장 모델

DP를 적용하지 않은 모델을 privacy-preserving이라고 부르지 않는다.

## 16.2 DP-SGD critic

각 환자 샘플에 대한 critic gradient \(g_i\)를 norm \(C\)로 clipping한다.

\[
\bar g_i
=g_i\cdot
\min\left(1,\frac{C}{\|g_i\|_2}\right)
\]

Gaussian noise를 추가한다.

\[
\tilde g
=
\frac{1}{B}
\left(
\sum_{i=1}^{B}\bar g_i
+
\mathcal N(0,\sigma^2C^2I)
\right)
\]

RDP 또는 PRV accountant를 이용하여 전체 privacy budget을 계산한다 [17,18].

\[
\delta<\frac{1}{N}
\]

를 기본 원칙으로 설정한다.

## 16.3 DP와 데이터 의존형 손실의 충돌

Critic만 DP-SGD로 학습하더라도 생성기 손실이 실제 training statistic을 비공개 방식으로 직접 사용하면 전체 DP 보장이 깨질 수 있다.

주의 대상:

- 실제 correlation matrix를 반복적으로 직접 참조
- 비DP auxiliary classifier 사용
- 비DP CDG edge weight 사용
- 실제 데이터 기반 임상 임계값의 비공개 추정

따라서 DP 확장에서는 다음 변형을 구분한다.

### DP-CDG-QGAN-Public

- 공개된 생리학적 관계로 CDG 구성
- 공개 임상 범위만 사용
- \(\mathcal L_{struct}\)에서 실제 batch statistic 제거
- DP critic만을 통해 데이터 정보 전달

### DP-CDG-QGAN-PrivateGraph

- CDG association에 DP noise 적용
- privacy accountant에 그래프 추정 비용 포함
- DP auxiliary statistics 사용

첫 논문에서 formal DP가 과도하게 복잡해지면 `DP-CDG-QGAN-Public`을 확장 실험으로 수행하고, 기본 기여는 privacy-audited CDG-QGAN에 둔다.

---

# 17. 구현 환경

## 17.1 소프트웨어

- Python
- PyTorch
- PennyLane
- Qiskit
- Qiskit Aer
- scikit-learn
- XGBoost
- SDMetrics 또는 SynthCity 기반 보조 평가
- Opacus 또는 TensorFlow Privacy 계열 DP 도구
- Anonymeter

## 17.2 양자 미분

### 이상적 simulator

- analytic expectation
- adjoint differentiation 또는 backpropagation 지원 simulator

### finite-shot 및 hardware-compatible 실험

Parameter-shift rule:

\[
\frac{\partial f(\theta)}{\partial\theta}
=
\frac{1}{2}
\left[
f\left(\theta+\frac{\pi}{2}\right)
-
f\left(\theta-\frac{\pi}{2}\right)
\right]
\]

실제 하드웨어에서 전체 GAN을 end-to-end 학습하는 것은 회로 호출량이 매우 커질 수 있으므로, 기본 설계는 simulator 학습 후 finite-shot/noisy inference와 선택적 소규모 fine-tuning으로 구성한다.

## 17.3 권장 초기 하이퍼파라미터

| 항목 | 초기값 또는 탐색 범위 |
|---|---:|
| Batch size | 64, 128 |
| Critic updates | generator 1회당 5회 |
| Classical optimizer | Adam |
| PQC optimizer | Adam 또는 AdamW |
| Critic learning rate | \(1\times10^{-4}\) |
| Decoder learning rate | \(1\times10^{-4}\) |
| PQC learning rate | \(1\times10^{-4}\), \(5\times10^{-5}\) |
| Adam \(\beta_1,\beta_2\) | \((0.0,0.9)\) |
| Gradient penalty | 10 |
| Qubits | 4, 6, 8, 10 |
| PQC layers | 1, 2, 3, 4 |
| Latent dimension | 8, 16, 32 |
| Shots | analytic, 256, 1024, 4096 |
| Random seeds | 최소 5, 핵심 비교 10 권장 |
| Max training epochs | 최대 1,000, early stopping 사용 |

초기 초안의 learning rate 0.001은 WGAN-GP와 PQC의 결합에서 불안정할 수 있으므로 기본값으로 사용하지 않는다.

## 17.4 초기화

- PQC: near-identity 또는 작은 균등분포 초기화
- Decoder/Critic: Xavier 또는 Kaiming initialization
- CDG edge parameter: 작은 난수 또는 weight-informed initialization
- 각 seed에서 동일한 전처리와 split 유지

## 17.5 Gradient 모니터링

다음을 층별로 기록한다.

\[
\|\nabla_{\theta_{\ell}}\mathcal L_G\|_2
\]

\[
\operatorname{Var}
\left[
\frac{\partial\mathcal L_G}{\partial\theta_{\ell,q}}
\right]
\]

깊이에 따라 gradient variance가 급격히 작아지는지 확인하여 barren plateau 또는 최적화 불안정을 분석한다 [12].

---

# 18. 비교 모델

## 18.1 고전 기준선

| 모델 | 목적 |
|---|---|
| Gaussian Copula | 단순 통계적 기준선 |
| CTGAN | 대표 mixed-type conditional GAN [5] |
| TVAE | VAE 계열 표형 기준선 |
| CTAB-GAN+ | 혼합형 변수, WGAN-GP 및 DP 비교 [7] |
| TabDDPM | diffusion 기반 강력한 기준선 [8] |
| Classical WGAN-GP | 동일 adversarial objective 비교 |
| Parameter-Matched MLP-GAN | 양자 코어의 효과를 분리하는 핵심 기준선 |
| Output-Matched MLP-GAN | 동일 latent output dimension 비교 |

## 18.2 양자 기준선

| 모델 | 비고 |
|---|---|
| Ring-QGAN | 고정 ring entanglement |
| RandomSparse-QGAN | 동일 edge 수의 무작위 sparse topology |
| AllToAll-QGAN | 고자원 참고 모델 |
| TabularQGAN | 의료 표형 QGAN 선행연구 [13] |
| QTabGAN | hybrid tabular QGAN 선행연구, 2026 preprint [14] |
| CDG-QGAN | 제안 모델 |

TabularQGAN 및 QTabGAN은 데이터 표현과 출력 구조가 다를 수 있으므로, 전체 특징 실험과 reduced-feature benchmark를 구분한다.

## 18.3 Parameter-Matched MLP-GAN

PQC를 고전 MLP \(M_{\varphi}\)로 교체한다.

\[
q_{classical}=M_{\varphi}(z,y)
\]

다음 조건을 고정한다.

- 동일 decoder
- 동일 critic
- 동일 손실함수
- 동일 batch sampling
- 동일 training steps
- 동일 latent output dimension

파라미터 조건:

\[
|\varphi|
\approx
|\theta_{PQC}|+|\omega_{angle}|
\]

양자 모델의 전체 파라미터에는 angle encoder를 반드시 포함한다. PQC 파라미터만 세고 고전 전처리 네트워크를 제외하면 공정하지 않다.

## 18.4 계산 비용 보고

- trainable parameters
- forward circuit evaluations
- gradient circuit evaluations
- wall-clock training time
- CPU/GPU utilization
- shot 수
- simulator type
- transpiled gate count

파라미터 효율성과 계산 효율성을 별도 개념으로 보고한다.

---

# 19. Ablation Study

| ID | 변경 사항 | 검증 목적 |
|---|---|---|
| A1 | CDG → ring | 데이터 의미 기반 topology의 효과 |
| A2 | CDG → random sparse | CDG가 임의 sparse graph보다 유효한지 |
| A3 | CDG → all-to-all | 고자원 회로와의 참고 비교 |
| A4 | \(\mathcal L_{clin}\) 제거 | 임상 제약 손실 효과 |
| A5 | \(\mathcal L_{struct}\) 제거 | 의존 구조 손실 효과 |
| A6 | \(\mathcal L_{miss}\) 제거 | 결측 구조 모델링 효과 |
| A7 | \(ZZ\) 측정 제거 | pairwise quantum measurement 필요성 |
| A8 | PQC → parameter-matched MLP | 양자 latent core 효과 |
| A9 | 조건 변수 제거 | conditional generation 효과 |
| A10 | data-driven CDG만 사용 | clinical prior 효과 |
| A11 | clinical prior만 사용 | 데이터 기반 graph 효과 |
| A12 | stable edge selection 제거 | bootstrap 안정성 선택 효과 |
| A13 | weight-informed initialization 제거 | edge weight 초기화 효과 |
| A14 | DP-SGD 적용 | privacy–utility trade-off |
| A15 | quantum latent zeroing/permutation | decoder의 quantum latent 의존성 |

All-to-all은 two-qubit gate 수가 증가하므로 동일 자원 비교가 아니라 고자원 참고 실험으로 명시한다.

---

# 20. NISQ 실험 설계

## 20.1 회로 규모 분석

\[
n_q\in\{4,6,8,10\},
\qquad
L\in\{1,2,3,4\}
\]

각 설정에서 다음을 기록한다.

- trainable quantum parameters
- total trainable parameters
- logical circuit depth
- transpiled depth
- single-qubit gate 수
- two-qubit gate 수
- measurement observables 수
- gradient variance
- 생성 품질
- 학습 시간

## 20.2 Shot 분석

\[
S\in\{256,1024,4096\}
\]

Analytic expectation을 기준으로 finite-shot 성능 저하를 계산한다.

\[
\Delta_{shot}(M)
=
\frac{M_{analytic}-M_{shot}}
{|M_{analytic}|+\epsilon}
\]

지표 방향이 반대인 경우 부호를 조정한다.

## 20.3 Noise 분석

비교 조건:

1. ideal analytic simulator
2. finite-shot noiseless simulator
3. backend-calibrated noise model
4. readout mitigation 적용
5. 가능한 경우 실제 양자 장치 inference

실제 장치 결과가 inference-only라면 논문에 명확히 표시한다.

## 20.4 Hardware topology와 routing

CDG topology가 실제 장치 coupling map과 불일치할 수 있으므로 다음을 보고한다.

- initial layout
- SWAP gate 수
- transpiled two-qubit depth
- mapping 전략
- CDG edge 보존율

가능하면 CDG edge weight와 hardware 연결 비용을 함께 고려한 layout heuristic을 보조 기여로 검토한다.

## 20.5 Noise mitigation

- readout error mitigation
- measurement calibration
- 가능한 경우 zero-noise extrapolation 보조 실험

Mitigation 설정은 validation 단계에서 사전 고정하고, 적용 전후 결과를 모두 보고한다.

---

# 21. 평가 체계

## 21.1 평가 원칙

합성 데이터의 품질을 단일 점수로 결론내리지 않는다.

\[
\text{Evaluation}
=
\text{Fidelity}
+
\text{Clinical plausibility}
+
\text{Utility}
+
\text{Diversity}
+
\text{Privacy}
+
\text{Resource analysis}
\]

## 21.2 통계적 충실도

### 연속형 단변량 분포

- Wasserstein-1 distance
- Kolmogorov–Smirnov statistic
- mean/standard deviation error
- quantile error

\[
E_{quantile,j}
=
\frac{1}{K}
\sum_{k=1}^{K}
|Q_{jk}^{real}-Q_{jk}^{syn}|
\]

### 범주형 분포

- Total Variation Distance
- Jensen–Shannon divergence
- category support coverage

\[
TVD(P,Q)=\frac{1}{2}\sum_i|P_i-Q_i|
\]

### 다변량 분포

- correlation matrix Frobenius error
- CDG edge-weighted association error
- mutual information error
- Maximum Mean Discrepancy
- energy distance
- real-vs-synthetic discriminator AUROC

### CDG edge-weighted error

\[
E_{CDG}
=
\frac{
\sum_{(j,k)\in E_F}
\omega_{jk}
|a_{jk}^{real}-a_{jk}^{syn}|
}{
\sum_{(j,k)\in E_F}\omega_{jk}
}
\]

이 값을 fidelity의 primary endpoint로 사용한다.

## 21.3 결측 패턴 평가

- 변수별 missing rate MAE
- mask correlation error
- 사망 조건별 missing rate
- 주요 검사 시행 패턴
- 관측값과 결측 여부의 조건부 관계

## 21.4 임상적 개연성

### 제약 위반율

\[
ViolationRate_r
=
\frac{
\#\{\tilde x:r(\tilde x)\text{ violated}\}
}{N_{syn}}
\]

### 위반 심각도

\[
Severity_r
=
\frac{1}{N_{syn}}
\sum_i
\operatorname{dist}(\tilde x_i,\mathcal C_r)
\]

### 관계별 평가

- SBP–DBP–MAP
- Hemoglobin–Hematocrit
- Creatinine–BUN
- Respiratory Rate–SpO₂
- Lactate–Bicarbonate

Real test set 자체의 위반율도 함께 제시한다. 합성 데이터의 위반율을 무조건 0으로 만드는 것이 목표가 아니라 실제 holdout의 구조를 과도하게 왜곡하지 않는 것이 중요하다.

## 21.5 머신러닝 활용성

### TRTR

\[
\text{Train Real, Test Real}
\]

### TSTR

\[
\text{Train Synthetic, Test Real}
\]

### TSRTR 또는 혼합 증강

\[
\text{Train Real + Synthetic, Test Real}
\]

### Downstream 모델

- Logistic Regression
- Random Forest
- XGBoost
- MLP

### 지표

- AUROC
- AUPRC
- Macro-F1
- sensitivity
- specificity
- Brier score
- Expected Calibration Error

사망률처럼 불균형한 outcome에서는 accuracy를 주 지표로 사용하지 않는다.

### Utility ratio

\[
U_{ratio}
=
\frac{M_{TSTR}}{M_{TRTR}}
\]

주 utility endpoint는 사망 예측 TSTR AUPRC로 설정한다.

## 21.6 저데이터 증강

실제 training data 비율

\[
p\in\{10\%,25\%,50\%\}
\]

에서 다음을 비교한다.

1. 실제 데이터 \(p\%\)만 사용
2. 실제 데이터 \(p\%\)+CTGAN 합성 데이터
3. 실제 데이터 \(p\%\)+TabDDPM 합성 데이터
4. 실제 데이터 \(p\%\)+CDG-QGAN 합성 데이터

이를 통해 합성 데이터가 실제 데이터를 완전히 대체하는 상황뿐 아니라 데이터 부족 환경의 증강 효과를 검증한다.

## 21.7 다양성

연속형 표형 데이터에 `numpy.unique`만 적용한 Mode Score는 부적절하다. 다음 지표를 사용한다.

- exact duplicate rate
- generated sample uniqueness
- category support coverage
- cluster coverage
- precision and recall for distributions
- density and coverage
- pairwise distance distribution
- minority-condition coverage

\[
DuplicateRate
=
\frac{\#\{\text{duplicated synthetic rows}\}}{N_{syn}}
\]

높은 uniqueness는 노이즈 데이터에서도 얻을 수 있으므로 fidelity와 함께 해석한다.

---

# 22. 프라이버시 평가

## 22.1 Membership Inference

DOMIAS는 생성모델의 국소적 과적합을 이용하는 밀도 기반 membership inference attack이며, 희귀 또는 과소대표 샘플에 특히 강한 공격 가능성을 보여준다 [19].

평가 지표:

- ROC-AUC
- attack accuracy
- TPR at fixed FPR
- attack advantage

\[
Adv_{MIA}=TPR-FPR
\]

균형 이진 공격에서 무작위 accuracy는 약 50%이므로 “공격 성공률 5% 이하”를 목표로 삼지 않는다. 경험적 안전 신호는 다음과 같다.

\[
AUC_{MIA}\approx0.5,
\qquad
Adv_{MIA}\approx0
\]

그러나 공격 실패는 formal privacy guarantee가 아니다.

## 22.2 Singling-out, Linkability, Inference

Anonymeter 계열 공격 기반 프레임워크를 이용해 다음을 평가한다 [20].

- **Singling-out**: 희귀한 속성 조합으로 개인을 고립시킬 위험
- **Linkability**: 서로 다른 속성 집합의 레코드를 동일인으로 연결할 위험
- **Inference**: 알려진 속성으로 민감 속성을 추론할 위험

각 공격은 control dataset과 비교하여 risk estimate 및 confidence interval을 보고한다.

## 22.3 Nearest-neighbor memorization

- Distance to Closest Record(DCR)
- Nearest Neighbor Distance Ratio(NNDR)
- exact training record match
- training–synthetic 거리와 holdout–synthetic 거리 비교

DCR은 보조 지표이며 단독으로 프라이버시를 보장하지 않는다.

## 22.4 Attribute Inference

공격자에게 일부 quasi-identifier를 제공하고 다음 민감 속성을 예측한다.

- 병원 내 사망
- 기계환기 여부
- 혈압상승제 사용
- 주요 검사 이상 여부

공격 모델은 Logistic Regression, Random Forest 및 XGBoost를 사용한다.

## 22.5 Subgroup privacy

전체 평균뿐 아니라 다음 subgroup을 별도로 평가한다.

- 사망군과 생존군
- 성별
- 연령 구간
- 인종 범주
- 희귀 ICU 유형
- 높은 결측률 환자
- 극단적 검사 수치 환자

희귀 subgroup에서 공격 위험이 높아질 수 있으므로 표본 수와 confidence interval을 함께 보고한다.

## 22.6 Formal DP 결과

DP-CDG-QGAN에서는 다음을 반드시 보고한다.

- \(\varepsilon\)
- \(\delta\)
- clipping norm
- noise multiplier
- sampling rate
- training steps
- accountant 종류
- 공격 기반 privacy score
- utility 감소

Formal DP와 empirical attack resistance를 서로 대체하지 않고 함께 제시한다.

---

# 23. 통계 분석

## 23.1 반복 실험

- 모든 주요 모델: 최소 5개 random seed
- 핵심 비교 모델: 가능하면 10개 seed
- 동일 split 및 동일 preprocessing 사용

\[
\bar M=\frac{1}{S}\sum_{s=1}^{S}M_s
\]

## 23.2 Confidence interval

- seed bootstrap
- test sample bootstrap
- 필요시 nested bootstrap

95% confidence interval을 보고한다.

## 23.3 가설 검정

- paired permutation test
- Wilcoxon signed-rank test
- bootstrap difference interval
- effect size

여러 metric을 동시에 검정하는 경우 Holm correction을 적용한다.

## 23.4 비열등성 분석

양자 모델이 절대적으로 우수하지 않더라도 더 적은 학습 파라미터로 유사 성능을 보이는지 평가하기 위해 비열등성 margin을 사전 정의한다.

예:

\[
\Delta_{AUPRC}
=AUPRC_{CDG-QGAN}-AUPRC_{MLP}
\]

비열등성 margin \(-\delta_U\)에 대해 confidence interval의 하한이 \(-\delta_U\)보다 크면 비열등으로 판단한다.

## 23.5 Primary endpoints

| 평가 축 | Primary endpoint |
|---|---|
| Fidelity | CDG edge-weighted association error |
| Utility | Mortality TSTR AUPRC |
| Clinical | 전체 clinical violation rate |
| Privacy | DOMIAS attack advantage |
| NISQ | noisy-to-ideal utility degradation |
| Resource | transpiled two-qubit gate 수 및 total trainable parameters |

나머지 지표는 secondary endpoint로 구분한다.

---

# 24. 사전 성공 기준

아래 수치는 결과 예측이 아니라 실험 전에 설정하는 운영 기준이다.

## 24.1 Utility

\[
\frac{AUPRC_{TSTR}}{AUPRC_{TRTR}}\geq0.85
\]

또는 parameter-matched MLP 대비 사전 정의된 비열등성 margin 충족.

## 24.2 구조 보존

- 동일 edge 수의 ring/random ansatz보다 CDG edge-weighted error 감소
- 다수 seed에서 일관된 방향의 개선
- effect size와 confidence interval 제시

## 24.3 임상 제약

- no-constraint ablation 대비 violation rate 30% 이상 감소를 목표
- TSTR AUPRC의 절대 감소가 0.03 이내

## 24.4 프라이버시

- 비DP 모델: 고전 baseline보다 DOMIAS attack advantage가 악화되지 않을 것
- DP 모델: \(\varepsilon\) 감소에 따라 공격 위험이 감소하는 경향 확인
- subgroup risk가 전체 평균에 가려지지 않도록 별도 보고

## 24.5 NISQ

- sparse CDG ansatz가 고밀도 ansatz보다 two-qubit gate 및 noise degradation 측면에서 유리
- finite-shot 조건에서 모델 순위가 완전히 역전되는지 확인

---

# 25. 예상 결과의 해석 원칙

## 25.1 CDG-QGAN이 고전 모델보다 우수한 경우

다음 조건이 충족되어야 유의미한 결과로 해석한다.

- 동일 decoder와 critic
- parameter-matched 또는 output-matched 비교
- 여러 seed에서 재현
- 강한 고전 baseline 포함
- wall-clock 및 회로 호출 비용 공개
- NISQ noise에서 성능 유지

이 경우에도 “quantum advantage”보다 “quantum-assisted inductive bias” 또는 “parameter-efficient performance”라는 표현을 사용한다.

## 25.2 비슷한 성능을 보이는 경우

더 적은 파라미터 또는 더 해석 가능한 회로 구조로 비열등한 결과를 얻었다면 다음과 같이 기여를 구성할 수 있다.

- resource-aware non-inferiority
- clinical structure preservation
- NISQ limitation characterization
- quantum latent bottleneck 분석

## 25.3 고전 모델보다 낮은 경우

부정적 결과도 다음 분석이 충분하면 연구 가치가 있다.

- 어떤 특징군에서 실패하는가
- 회로 깊이와 gradient vanishing의 관계
- decoder가 quantum latent를 무시하는가
- shot/noise가 어느 지표를 가장 크게 손상시키는가
- CDG topology가 hardware routing에서 불리한가
- 파라미터 효율성과 절대 성능의 trade-off

결과를 과장하지 않고 적용 가능한 규모와 한계를 정량화하는 것이 중요하다.

---

# 26. 재현성과 연구 관리

## 26.1 공개 산출물

- 데이터 추출 SQL
- feature dictionary
- preprocessing pipeline
- CDG construction code
- quantum circuit definition
- baseline configuration
- evaluation pipeline
- privacy attack configuration
- seed별 raw metrics
- environment lock file
- hardware/noise configuration

## 26.2 실험 추적

각 실행에서 다음을 저장한다.

- git commit hash
- configuration file
- random seed
- dataset split hash
- model parameter count
- circuit depth 및 gate count
- training loss
- validation metrics
- checkpoint selection reason

## 26.3 데이터 공개 제한

MIMIC-IV 원본 및 환자 단위 파생 데이터를 저장소에 포함하지 않는다. 합성 데이터 공개도 자동으로 안전하다고 가정하지 않고, privacy audit와 DUA 조건을 확인한 뒤 결정한다.

---

# 27. 윤리적 고려

- 합성 데이터는 실제 진료 또는 환자 의사결정을 위한 자료가 아니다.
- 통계적 유사성은 임상적 안전성을 의미하지 않는다.
- 낮은 공격 score는 절대적 익명성을 의미하지 않는다.
- 소수 집단의 utility 및 privacy를 별도로 평가한다.
- 임상 규칙은 공개된 생리학적 정의로 제한한다.
- 임상의 검토가 없다면 “clinical validity”보다 “clinical plausibility”를 사용한다.
- 인종 변수는 생물학적 인과 요인이 아니라 사회적·환경적 요인을 포함하는 데이터 속성으로 제한적으로 해석한다.
- 환자 결과를 조건으로 생성할 때 stigmatization 또는 subgroup distortion 가능성을 논의한다.

---

# 28. 단일 저자 연구로서의 권장 범위

## 핵심 본문에 포함

- MIMIC-IV 첫 ICU 입실 후 첫 24시간 정적 테이블
- 30~45개 mixed-type 특징
- CDG 구축과 qubit community mapping
- CDG-guided PQC
- parameter-matched MLP generator
- CTGAN, CTAB-GAN+, TabDDPM, WGAN-GP
- clinical constraint 및 dependency loss
- 종합 privacy audit
- ideal, finite-shot, noisy simulation
- gate count 및 gradient 분석

## 여력이 있을 때 확장

- DP-CDG-QGAN
- 실제 양자 장치 inference
- eICU 외부 검증
- 저데이터 증강 실험의 범위 확대

## 첫 논문에서 제외 권장

- 전체 longitudinal trajectory 생성
- 임상 노트와 영상의 multimodal 생성
- quantum diffusion 동시 개발
- 연합학습과 결합
- 여러 질환에 대한 다중 진단 조건
- 실제 임상 배치 또는 안전성 주장

---

# 29. 저널 방향

## 29.1 현실적 1차 목표

**Quantum Machine Intelligence**

적합한 이유:

- hybrid quantum–classical machine learning
- quantum generative model
- application-driven quantum AI
- NISQ 성능 및 자원 분석

필수 조건:

- CDG ansatz의 명확한 신규성
- parameter-matched comparison
- depth/noise/qubit ablation
- 강한 classical baseline
- 과장 없는 quantum claim

## 29.2 상향 목표

**Quantum Science and Technology**

추가로 요구될 가능성이 높은 요소:

- 실제 하드웨어 결과
- 여러 데이터셋 또는 비의료 tabular dataset 일반화
- hardware-aware mapping
- 회로 구조 자체의 일반적 방법론 기여

## 29.3 도전적 목표

**PRX Quantum**

단일 MIMIC-IV simulator 실험만으로는 현실적으로 매우 어렵다. 일반화 가능한 이론, 명확한 quantum resource insight, 강한 하드웨어 검증이 필요하다.

## 29.4 의료 AI 저널 전환

다음이 추가되면 JAMIA, IEEE Journal of Biomedical and Health Informatics, npj Digital Medicine 계열을 고려할 수 있다.

- eICU 외부 검증
- 임상의 검토
- subgroup fairness 및 calibration 확대
- 실제 연구 활용 시나리오
- 의료적 의미 중심의 결과 해석

의료 저널에서는 양자 회로 자체보다 외부 검증과 임상적 유용성이 더 중요하게 평가될 가능성이 높다.

---

# 30. 제안 논문 구성

1. Introduction
2. Related Work
   - Synthetic EHR generation
   - Tabular generative models
   - Quantum generative models
   - Privacy evaluation of synthetic data
3. Problem Formulation
4. Clinical Dependency Graph Construction
5. CDG-QGAN Architecture
6. Clinical and Structural Objectives
7. Experimental Setup
8. Fidelity and Utility Results
9. Privacy Audit
10. NISQ and Resource Analysis
11. Ablation and Failure Analysis
12. Discussion
13. Limitations and Ethical Considerations
14. Conclusion

---

# 31. 핵심 novelty 문장

### 국문

> 본 연구는 의료 변수 사이의 안정적인 통계적·임상적 의존 관계를 특징 그래프로 구성하고, 해당 그래프를 매개변수화 양자 회로의 얽힘 토폴로지와 pairwise measurement 구조로 변환하는 CDG-QGAN을 제안한다. 동일한 decoder, critic, 손실함수 및 유사한 파라미터 규모를 갖는 고전 생성기와 비교함으로써, 의료 표형 데이터 생성에서 데이터 의존적 양자 회로가 제공하는 귀납 편향과 NISQ 한계를 정량화한다.

### 영문

> We propose CDG-QGAN, a hybrid quantum–classical generative model that transforms stable statistical and clinically grounded dependencies among ICU variables into the entanglement topology and pairwise measurement structure of a parameterized quantum circuit. Under matched decoder, critic, objective, latent dimensionality, and parameter budgets, we quantify whether the resulting data-dependent quantum circuit provides a useful inductive bias for mixed-type clinical tabular synthesis and characterize its limitations under NISQ constraints.

---

# 32. 최종 연구 방향 요약

CDG-QGAN의 논문 가치를 결정하는 핵심은 “양자 GAN을 의료 데이터에 사용했다”가 아니다. 다음 세 가지가 중심이어야 한다.

1. **Clinical Dependency Graph를 양자 entanglement topology로 변환하는 명확한 방법론**
2. **동일 자원 조건의 고전 generator와의 공정한 통제 비교**
3. **fidelity, clinical plausibility, utility, privacy 및 NISQ 자원을 통합한 평가**

이 구조를 유지하면 본 연구는 단순 의료 응용 QGAN이 아니라, 의료 변수 구조를 양자 회로 설계에 반영하는 **data-dependent quantum generative modeling** 연구로 포지셔닝할 수 있다.

---

# 참고문헌

1. Goodfellow, I. et al. “Generative Adversarial Nets.” *NeurIPS*, 2014.  
   https://papers.nips.cc/paper/5423-generative-adversarial-nets

2. Arjovsky, M., Chintala, S., Bottou, L. “Wasserstein Generative Adversarial Networks.” *ICML/PMLR*, 2017.  
   https://proceedings.mlr.press/v70/arjovsky17a.html

3. Gulrajani, I. et al. “Improved Training of Wasserstein GANs.” *NeurIPS*, 2017.  
   https://proceedings.neurips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html

4. Choi, E. et al. “Generating Multi-label Discrete Patient Records using Generative Adversarial Networks.” *MLHC/PMLR*, 2017.  
   https://proceedings.mlr.press/v68/choi17a.html

5. Xu, L. et al. “Modeling Tabular Data using Conditional GAN.” *NeurIPS*, 2019.  
   https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html

6. Li, J., Cairns, B. J., Li, J., Zhu, T. “Generating synthetic mixed-type longitudinal electronic health records for artificial intelligent applications.” *npj Digital Medicine*, 2023.  
   https://doi.org/10.1038/s41746-023-00834-7

7. Zhao, Z. et al. “CTAB-GAN+: Enhancing Tabular Data Synthesis.” 2022.  
   https://arxiv.org/abs/2204.00401

8. Kotelnikov, A. et al. “TabDDPM: Modelling Tabular Data with Diffusion Models.” *ICML/PMLR*, 2023.  
   https://proceedings.mlr.press/v202/kotelnikov23a.html

9. Lloyd, S., Weedbrook, C. “Quantum Generative Adversarial Learning.” *Physical Review Letters*, 2018.  
   https://doi.org/10.1103/PhysRevLett.121.040502

10. Dallaire-Demers, P.-L., Killoran, N. “Quantum Generative Adversarial Networks.” *Physical Review A*, 2018.  
    https://doi.org/10.1103/PhysRevA.98.012324

11. Huang, H.-L. et al. “Experimental Quantum Generative Adversarial Networks for Image Generation.” *Physical Review Applied*, 2021.  
    https://doi.org/10.1103/PhysRevApplied.16.024051

12. McClean, J. R. et al. “Barren plateaus in quantum neural network training landscapes.” *Nature Communications*, 2018.  
    https://doi.org/10.1038/s41467-018-07090-4

13. Bhardwaj, P. et al. “TabularQGAN: A Quantum Generative Model for Tabular Data.” 2025 preprint.  
    https://arxiv.org/abs/2505.22533

14. Kumari, S., Achutha, R., Sivaraman, V. “QTabGAN: A Hybrid Quantum-Classical GAN for Tabular Data Synthesis.” 2026 preprint.  
    https://arxiv.org/abs/2602.12704

15. Johnson, A. E. W. et al. “MIMIC-IV, a freely accessible electronic health record dataset.” *Scientific Data*, 2023.  
    https://doi.org/10.1038/s41597-022-01899-x

16. PhysioNet. “MIMIC-IV v3.1.”  
    https://physionet.org/content/mimiciv/

17. Abadi, M. et al. “Deep Learning with Differential Privacy.” *ACM CCS*, 2016.  
    https://doi.org/10.1145/2976749.2978318

18. Mironov, I. “Rényi Differential Privacy.” *IEEE CSF*, 2017.  
    https://doi.org/10.1109/CSF.2017.11

19. van Breugel, B. et al. “Membership Inference Attacks against Synthetic Data through Overfitting Detection.” *AISTATS/PMLR*, 2023.  
    https://proceedings.mlr.press/v206/breugel23a.html

20. Giomi, M. et al. “A Unified Framework for Quantifying Privacy Risk in Synthetic Data.” *Proceedings on Privacy Enhancing Technologies*, 2023.  
    https://petsymposium.org/popets/2023/popets-2023-0055.php

21. Pérez-Salinas, A. et al. “Data re-uploading for a universal quantum classifier.” *Quantum*, 2020.  
    https://doi.org/10.22331/q-2020-02-06-226

22. Schuld, M. et al. “Evaluating analytic gradients on quantum hardware.” *Physical Review A*, 2019.  
    https://doi.org/10.1103/PhysRevA.99.032331

23. Qian, Z. et al. “Synthcity: facilitating innovative use cases of synthetic data in different data modalities.” *NeurIPS Datasets and Benchmarks*, 2023.  
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html

24. Sajjadi, M. S. M. et al. “Assessing Generative Models via Precision and Recall.” *NeurIPS*, 2018.  
    https://proceedings.neurips.cc/paper/2018/hash/f7696a9b362ac5a51c3dc8f098b73923-Abstract.html

25. Kynkäänniemi, T. et al. “Improved Precision and Recall Metric for Assessing Generative Models.” *NeurIPS*, 2019.  
    https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html

---

## 문서 사용 시 주의

- 본 문서는 연구 **계획서**이며 실험 결과를 미리 주장하지 않는다.
- `CDG-QGAN`의 CDG는 인과 그래프가 아니다.
- 비DP 기본 모델은 `privacy-audited`로 표현한다.
- SCI/SCIE 투고 전에는 사용한 데이터 버전, 라이브러리 버전, 양자 backend 및 최신 선행연구를 다시 확인한다.

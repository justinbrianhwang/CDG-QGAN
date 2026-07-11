---
title: "CDG-QGAN: 임상 조건부 의존 그래프와 얕은 양자 생성기의 수용영역 정렬을 이용한 합성 중환자 데이터 생성"
subtitle: "구조 재설계 연구계획서 — 이론·모델·실험 통합본"
lang: ko-KR
version: "2.0"
date: "2026-07-11"
status: "연구 설계안"
---

# CDG-QGAN 연구계획서 v2.0

## 정식 연구명

### 국문

**CDG-QGAN: 임상 조건부 의존 그래프와 얕은 양자 생성기의 수용영역 정렬을 이용한 구조 보존형 합성 중환자 데이터 생성**

### 영문

**CDG-QGAN: Aligning Clinical Dependency Graphs with the Depth-Limited Receptive Fields of Shallow Quantum Generators for Structure-Preserving Synthetic ICU Data**

### 약어

**CDG-QGAN** = **C**linical **D**ependency **G**raph-aligned **Q**uantum **G**enerative **A**dversarial **N**etwork

> **용어 경계**  
> 이 연구에서 CDG는 인과 그래프가 아니라, 훈련 데이터에서 추정한 **조건부 통계 의존 그래프**이다. 따라서 `causal graph`, `causal discovery`, `causal effect`라는 표현을 사용하지 않는다. 조건부 독립 그래프에 가까운 해석을 위해 단순 상관계수가 아니라 희소 정밀도 행렬과 부분상관을 사용한다.

---

# 연구 요약

의료 데이터 합성 연구는 데이터 접근성과 프라이버시 문제를 완화할 가능성이 있지만, 생성된 환자 레코드가 실제 변수의 주변분포만 닮고 변수 간 생리학적 관계를 잃는 문제가 있다. 양자 생성모델은 제한된 수의 파라미터로 비선형 분포를 표현할 가능성이 있으나, 기존 하이브리드 QGAN 설계에서는 전역 angle encoder와 dense decoder가 양자 회로 전후에서 모든 특징을 자유롭게 혼합한다. 이 경우 임상 그래프를 양자 회로의 얽힘 토폴로지로 사용하더라도, 특정 임상 특징과 특정 큐비트의 대응은 모델 내부에서 식별되지 않으며, 최종 출력의 상관구조가 양자 회로의 그래프 구조에서 발생했는지 고전 신경망에서 발생했는지 구분하기 어렵다.

본 연구는 이 문제를 해결하기 위해 **16개 임상 특징을 16개 큐비트에 일대일로 대응**시키고, 각 큐비트에 독립적인 local latent를 입력하며, 각 출력 특징이 대응 큐비트의 local observable만 읽도록 제한하는 **graph-local hybrid quantum generator**를 제안한다. 양자 회로는 임상 조건부 의존 그래프의 간선에만 `RZZ` 얽힘 게이트를 배치하고, 얽힘 뒤에 비가환 local mixing rotation을 두어 얽힘 위상 정보가 최종 `Z` 측정값에 실제로 반영되도록 한다.

이 구조에서는 깊이 \(L\)의 그래프 로컬 회로에서 출력 특징 \(u\)가 그래프 반경 \(L\) 이내의 local latent에만 의존한다. 조건 변수 \(y\)가 주어졌을 때 local latent가 서로 독립이고 두 출력의 backward light cone이 겹치지 않으면, 그래프 거리 \(d_G(u,v)>2L\)인 두 출력은 조건부로 독립하게 된다. 따라서 얕은 회로에서 어떤 의존 관계를 표현할 수 있는지는 얽힘 그래프의 거리 구조에 의해 제한된다. CDG-QGAN의 핵심 가설은 강한 임상 의존 관계를 짧은 그래프 거리 안에 배치하면, 동일한 깊이와 자원에서 임상 의미가 무작위로 섞인 그래프보다 해당 관계를 더 잘 복원할 수 있다는 것이다.

주 실험은 MIMIC-IV v3.1의 성인 첫 ICU 입실을 대상으로 한다. 관측 창 내 사망과 조기 퇴실이 결측 패턴을 통해 사망 라벨을 누설하는 문제를 피하기 위해, ICU 입실 후 24시간까지 생존하고 ICU에 재원한 환자만 포함하는 **24시간 landmark cohort**를 사용한다. 첫 24시간의 16개 연속형 생리·검사 특징을 생성하고, 결과 라벨은 24시간 이후 병원 내 사망으로 정의한다. 모델은 라벨 \(y\)에 조건부로 \(x\mid y\)를 생성하되, 이론과 의존성 평가는 모두 \(y\)에 조건부로 수행한다.

확증적 비교는 CDG-QGAN과 **isomorphic permuted-CDG-QGAN** 사이에서 수행한다. Permuted-CDG는 그래프의 간선 수, 차수열, 지름, 추상적 그래프 구조와 양자 자원을 완전히 유지하면서 임상 특징과 그래프 노드의 대응만 무작위로 섞는다. 따라서 두 모델의 차이는 단순한 그래프 복잡도나 파라미터 수가 아니라 **임상 의미와 회로 수용영역의 정렬**에서 발생한다. 확증적 primary endpoint는 모델 구성과 손실에 사용되지 않은 held-out 임상 의존 쌍에 대한 조건부 부분상관 복원 오차 하나로 제한한다.

주 모델의 생성 손실에는 임상 상관 손실, CDG edge loss, auxiliary condition loss를 사용하지 않는다. 생성기는 동일한 conditional WGAN 목적함수만으로 학습한다. 이 선택은 평가 지표를 직접 최적화하는 순환 논증을 제거하고, 관찰된 구조 복원 차이를 회로 토폴로지의 inductive bias로 해석할 수 있게 한다. Graph-local classical message-passing generator, no-entanglement quantum model, random Fourier feature control, global MLP-WGAN 및 TabDDPM을 비교하여 그래프 효과, locality 효과, 비선형 특징맵 효과와 양자 회로 효과를 분리한다.

본 연구는 보편적인 quantum advantage를 주장하지 않는다. 핵심 결과는 다음 질문에 대한 반증 가능한 답이다.

> **얕은 그래프 로컬 양자 생성기의 깊이 제한적 의존성 수용영역을 임상 조건부 의존 그래프에 맞추면, 동일한 토폴로지·깊이·파라미터 조건에서 임상 의미가 무작위로 섞인 회로보다 보지 못한 임상 의존 관계를 더 정확히 복원하는가?**

---

# 1. 연구 배경

## 1.1 의료 합성 데이터의 필요성

전자건강기록(Electronic Health Record, EHR)은 의료 인공지능 연구에 중요한 자원이지만, 환자 프라이버시, 기관별 데이터 사용 계약, 연구윤리 심의, 희귀 환자군의 작은 표본 수로 인해 접근과 공유가 제한된다. 합성 데이터는 원본 레코드를 직접 공개하지 않고 새로운 환자 표본을 생성할 수 있다는 장점이 있으나, 다음 문제가 남는다.

- 개별 변수의 평균과 분산은 유사하지만 변수 간 관계가 깨질 수 있다.
- 희귀 환자군과 극단값이 과도하게 평활화될 수 있다.
- 조건부 생성기가 결과 라벨을 지나치게 분리하여 비현실적인 \(P(y\mid x)\)를 만들 수 있다.
- 훈련 레코드의 암기와 근접 복제가 발생할 수 있다.
- 높은 통계적 유사성이 임상적으로 타당한 생리 구조를 의미하지는 않는다.

따라서 합성 의료 데이터 연구는 주변분포, 다변량 관계, downstream utility, calibration, 임상적 개연성 및 프라이버시 위험을 구분하여 평가해야 한다.

## 1.2 표형 생성모델의 발전과 한계

medGAN은 고차원 이산형 환자 기록 생성을 제안했고 [8], CTGAN은 혼합형 표형 데이터에서 조건부 학습과 mode-specific normalization을 사용했다 [9]. TabDDPM은 연속형·범주형 변수를 위한 diffusion 기반 강력한 기준선을 제공한다 [10]. 의료 데이터 생성에서는 단순한 평균 비교보다 TSTR(Train on Synthetic, Test on Real), 상관구조, 임상적 일관성 및 공격 기반 프라이버시 평가가 중요하다는 방향이 확립되고 있다.

그러나 이러한 고전 모델은 높은 용량의 전역 신경망을 사용하므로 특정 구조적 관계가 어떤 메커니즘으로 복원되었는지 해석하기 어렵다. 본 연구는 절대적인 생성 품질 경쟁만이 아니라, **제한된 깊이의 생성기가 어떤 의존성을 표현할 수 있는가**를 중심 질문으로 삼는다.

## 1.3 양자 표형 데이터 생성 연구

QGAN은 양자 회로를 생성기 또는 판별자로 사용하는 생성모델 계열이다 [11,12]. 최근 TabularQGAN은 MIMIC-III와 Adult Census에서 양자 표형 데이터 생성을 평가했고 [13], QTabGAN은 PQC 출력과 고전 매퍼를 결합한 하이브리드 표형 생성 구조를 제안했다 [14]. 두 연구는 양자 표형 생성의 가능성을 보여주지만, 2026년 7월 기준 확인 가능한 공개본은 arXiv preprint이므로 peer-reviewed 결과와 구분하여 인용해야 한다.

기존 설계에서 흔한 문제는 다음과 같다.

1. 모든 latent가 모든 큐비트에 입력되는 dense angle encoder
2. 모든 양자 측정값이 모든 출력 특징에 연결되는 dense decoder
3. 회로 토폴로지와 임상 특징 정체성의 비식별성
4. 양자 코어보다 고전 encoder·decoder의 파라미터가 훨씬 큰 구조
5. 임상 상관 손실과 동일한 지표로 모델을 평가하는 순환성
6. 이상적 simulator에서만 성능을 보고하고 측정·회로 자원을 분리하지 않는 분석

본 연구는 이 여섯 문제를 모델 구조와 실험 설계 단계에서 제거한다.

---

# 2. 연구의 중심 주장과 주장 경계

## 2.1 중심 주장

본 연구의 중심 주장은 다음과 같다.

> 깊이 \(L\)의 graph-local quantum generator는 그래프 거리로 제한된 dependency receptive field를 갖는다. 임상 조건부 의존 그래프를 회로 토폴로지로 사용하여 강한 임상 관계를 이 receptive field 안에 배치하면, 동일한 추상 그래프 구조와 자원을 갖지만 임상 노드 배정이 무작위인 회로보다 held-out 임상 의존 관계를 더 정확히 복원할 수 있다.

## 2.2 주장하지 않는 내용

다음은 본 연구 결과만으로 주장하지 않는다.

- 계산복잡도 관점의 보편적 quantum advantage
- 고전적으로 시뮬레이션할 수 없는 양자 우위
- 적은 파라미터 수만을 근거로 한 전체 계산 효율성 우위
- CDG 간선의 인과관계
- 합성 데이터의 완전한 익명성
- 실제 임상 배치 가능성 또는 임상 안전성
- simulator 결과만으로 입증한 실제 하드웨어 우위

## 2.3 논문의 권장 핵심 문장

> **We do not claim unconditional quantum advantage. We test whether aligning a shallow graph-local quantum generator with a clinically estimated conditional-dependency graph improves the recovery of unseen clinical relationships under matched topology, depth, parameter, and training conditions.**

---

# 3. 연구 질문과 가설

## 3.1 확증적 연구 질문

### RQ1

임상 특징과 그래프 노드의 의미적 정렬은 얕은 양자 생성기의 held-out 임상 의존 관계 복원에 기여하는가?

### 확증적 가설 H1

동일한 그래프 isomorphism class, 간선 수, 회로 깊이, 양자 파라미터 수, local encoder, local decoder, critic 및 학습 절차를 사용할 때,

\[
\mathrm{HDE}_{\mathrm{CDG}}
<
\mathrm{HDE}_{\mathrm{Permuted\text{-}CDG}}
\]

가 성립한다. 여기서 HDE는 held-out dependency error이다.

## 3.2 탐색적 연구 질문

### RQ2

회로 깊이와 그래프 거리는 의존성 복원 오차에 어떤 상호작용을 보이는가?

### H2

깊이 \(L\)이 증가하면 그래프 거리 \(d_G(u,v)\le 2L\)인 특징 쌍의 의존성 복원 오차가 우선 감소한다. \(d_G(u,v)>2L\)인 쌍은 얕은 회로에서 복원하기 어렵다.

### RQ3

CDG-QGAN의 이점은 양자 회로 자체에서 발생하는가, 아니면 graph-local inductive bias만으로 설명되는가?

### H3

CDG-QGAN이 동일 receptive field와 유사 파라미터 수를 갖는 graph-local classical generator보다 held-out dependency error가 낮을 때만 양자 코어의 추가적 이점을 주장한다. 두 모델이 동등하면 결과는 “CDG 정렬의 이점”으로 제한하여 해석한다.

### RQ4

회로의 얽힘이 필요한가?

### H4

No-entanglement 모델은 CDG-QGAN보다 그래프 거리 1–2의 관계 복원에서 열등하다. 차이가 없으면 양자 얽힘이 생성 성능에 실질적으로 기여하지 않았다고 판단한다.

### RQ5

NISQ 측정 및 장치 노이즈가 구조 복원에 미치는 영향은 무엇인가?

### H5

Finite-shot 및 calibrated noise는 구조 복원 오차를 증가시키며, shot-noise-aware fine-tuning은 해당 성능 저하를 일부 완화한다.

---

# 4. 핵심 이론: 그래프 로컬 양자 생성기의 깊이 제한적 수용영역

## 4.1 표기

임상 특징과 큐비트의 집합을

\[
V=\{1,\ldots,n\},\qquad n=16
\]

으로 둔다. 임상 조건부 의존 그래프는

\[
G=(V,E)
\]

이다. 특징 \(x_u\)와 큐비트 \(q_u\)는 일대일 대응한다.

결과 조건은

\[
y\in\{0,1\}
\]

이며, \(y=1\)은 24시간 landmark 이후 병원 내 사망을 의미한다. 각 큐비트에는 조건 \(y\) 아래 서로 독립인 local latent를 입력한다.

\[
z_u\overset{iid}{\sim}\mathcal U(-1,1),
\qquad
z_u\perp z_v\mid y
\]

## 4.2 회로 블록

한 블록은 Schrödinger 관점에서 다음 순서로 작동한다.

```text
Local data encoding
→ CDG-edge RZZ entanglement
→ Non-commuting local mixing rotation
```

수식으로는 오른쪽 연산자가 먼저 작동하도록

\[
U^{(\ell)}
=
R^{(\ell)}_{\mathrm{mix}}
E^{(\ell)}_G
S^{(\ell)}_{\mathrm{enc}}
\]

로 둔다.

전체 깊이 \(L\)의 회로는

\[
U_{G,\Theta}(z,y)
=
U^{(L)}\cdots U^{(2)}U^{(1)}
\]

이다.

## 4.3 Local encoding

각 큐비트 \(u\)의 encoding angle은 해당 local latent \(z_u\)와 조건 \(y\)만 사용한다.

\[
\alpha_{\ell,u}
=
a_{\ell,u}z_u+b_{\ell,u}y+c_{\ell,u}
\]

\[
S^{(\ell)}_{\mathrm{enc}}
=
\prod_{u\in V}
R_Y(\alpha_{\ell,u})
R_Z(\beta_{\ell,u})
\]

다른 노드의 latent \(z_v\)를 입력하는 dense encoder는 사용하지 않는다. \(a,b,c\)는 고정값 또는 소수의 feature-wise trainable scale·shift로 제한한다.

## 4.4 CDG 기반 얽힘

\[
E^{(\ell)}_G
=
\prod_{(u,v)\in E}
R_{ZZ}^{(u,v)}(\gamma_{\ell,uv})
\]

\[
R_{ZZ}(\gamma)
=
\exp\left(-i\frac{\gamma}{2}Z\otimes Z\right)
\]

`RZZ` 게이트는 동일 층 안에서 서로 가환하므로 논리적 그래프 얽힘 층으로 다룰 수 있다. 실제 장치에서의 two-qubit depth는 edge coloring과 transpilation 결과를 별도로 보고한다.

## 4.5 Local mixing

마지막 얽힘 층을 `Z` 측정 바로 앞에 두면 `RZZ`가 `Z` 및 `ZZ` 관측량과 가환하기 때문에 마지막 얽힘 파라미터가 측정값에 영향을 주지 않는다. 이를 피하기 위해 얽힘 뒤에 비가환 local rotation을 둔다.

\[
R^{(\ell)}_{\mathrm{mix}}
=
\prod_{u\in V}
R_X(\theta^{x}_{\ell,u})
R_Y(\theta^{y}_{\ell,u})
\]

이 순서에서는 entanglement가 만든 위상 정보가 local mixing을 거쳐 `Z`-basis population에 반영된다.

## 4.6 Local observable과 출력

각 특징 \(u\)는 대응 큐비트의 단일 관측량만 읽는다.

\[
q_u(z,y)
=
\langle 0|
U_{G,\Theta}^{\dagger}(z,y)
Z_u
U_{G,\Theta}(z,y)
|0\rangle
\]

최종 특징은 feature-specific local head로 생성한다.

\[
\tilde x_u
=
h_u(q_u,y)
\]

`h_u`는 폭 8 이하의 작은 MLP 또는 1차원 spline으로 구성한다. 다른 큐비트의 측정값은 입력받지 않는다.

## 4.7 Light-cone 명제

### 명제 1: Local observable support

위 회로에서 \(L\)개의 `RZZ → local mixing` 블록을 사용하면, Heisenberg 관점에서 \(Z_u\)의 backward support는 그래프 반경 \(L\) 이내에 포함된다.

\[
\operatorname{supp}
\left(
U_{G,\Theta}^{\dagger}Z_uU_{G,\Theta}
\right)
\subseteq
N_L(u)
\]

여기서

\[
N_L(u)
=
\{w\in V:d_G(u,w)\le L\}
\]

이다.

### 직관

- 최종 local mixing은 \(Z_u\)를 \(X_u,Y_u,Z_u\)의 선형결합으로 바꾼다.
- 한 `RZZ` 층은 \(X_u\) 또는 \(Y_u\) 성분에 인접 노드의 \(Z_v\)를 붙일 수 있으므로 support를 최대 한 graph hop 확장한다.
- 다음 local mixing이 인접 노드의 \(Z_v\)를 다시 \(X_v,Y_v\) 성분으로 바꾸고, 다음 `RZZ` 층에서 한 hop 더 확장된다.
- 따라서 블록 하나당 backward light cone은 최대 한 graph hop씩 증가한다.

정식 증명은 Pauli string의 켤레변환을 이용한 귀납법으로 부록에 제시한다. 유한 깊이 로컬 회로에서 상관관계와 정보 전파가 거리와 깊이로 제한된다는 일반적 배경은 Lieb–Robinson 및 finite-depth circuit locality 결과와 연결된다 [1,2].

## 4.8 조건부 독립성 따름정리

### 따름정리 1

다음 조건을 가정한다.

1. 초기 상태가 product state이다.
2. 입력 encoding이 local하다.
3. \(z_u\perp z_v\mid y\)이다.
4. 출력 \(\tilde x_u\)가 local observable \(q_u\)만 사용한다.
5. 그래프 거리 \(d_G(u,v)>2L\)이다.

그러면 두 backward light cone이 겹치지 않으므로

\[
\tilde x_u
\perp
\tilde x_v
\mid y
\]

가 성립한다.

따라서 조건부 연결상관은

\[
\operatorname{Cov}
(\tilde x_u,\tilde x_v\mid y)
=0
\]

이다.

### 해석

이 결과는 모든 실제 의료 관계가 정확히 그래프 로컬하다는 뜻이 아니다. 오히려 다음의 **구조적 제약**을 명시한다.

> 얕은 generator가 특정 특징 쌍의 조건부 의존성을 표현하려면, 해당 특징 쌍의 backward light cone이 겹치도록 그래프상 충분히 가깝게 배치되어야 한다.

따라서 CDG 설계의 목적은 강한 임상 관계의 그래프 거리를 줄이는 것이다.

## 4.9 전역 condition의 처리

조건 \(y\)는 모든 local encoder와 head에 입력되므로, 서로 먼 특징도 \(y\)를 공통 원인으로 하여 **무조건부** 상관을 가질 수 있다. 따라서 이론과 주요 평가는 다음 중 하나로 수행한다.

- 각 \(y=c\) 집단 안에서 별도 의존성 계산
- \(y\)를 회귀로 제거한 residual의 부분상관 계산

논문에서 light-cone 주장은 반드시 “conditional on \(y\)”로 표현한다.

---

# 5. 순열 대칭과 임상 노드 정체성 고정

## 5.1 기존 구조의 문제

Dense angle encoder와 dense decoder를 사용하면 큐비트 라벨을 임의로 순열하고 encoder 행, 회로 파라미터, 그래프 간선 및 decoder 입력 순서를 함께 순열해도 동일한 함수족을 얻을 수 있다. 이 경우 “신장 특징이 특정 큐비트에 배정되었다”는 의미가 모델 출력에 고정되지 않는다.

## 5.2 v2 구조의 해결

본 연구에서는 다음 대응을 고정한다.

\[
\text{clinical feature }x_u
\longleftrightarrow
\text{local latent }z_u
\longleftrightarrow
\text{qubit }q_u
\longleftrightarrow
\text{local head }h_u
\]

따라서 그래프 노드 라벨을 바꾸면서 출력 head를 같이 바꾸는 것은 단순한 재표기이지만, **특징–큐비트 대응을 유지한 채 그래프 간선만 임상적으로 잘못된 노드 쌍으로 옮기는 것**은 다른 모델이 된다.

## 5.3 Isomorphic permuted-CDG 대조군

임상 특징과 출력 head의 정체성은 고정하고, 그래프 adjacency만 순열한다.

\[
A_{\pi}
=
PAP^{\top}
\]

단, 큐비트 \(u\)는 계속 특징 \(x_u\)를 생성한다. 이 대조군은 다음을 완전히 보존한다.

- 노드 수
- 간선 수
- 차수열
- 그래프 지름
- 추상적 graph isomorphism class
- 양자 게이트 수
- trainable quantum parameter 수
- local encoder와 local head 크기

달라지는 것은 어떤 임상 특징 쌍이 가까운 그래프 거리를 갖는지뿐이다. 따라서 CDG와 permuted-CDG의 차이는 **임상 의미 정렬의 직접 검정**이다.

---

# 6. 데이터 설계

## 6.1 데이터셋

주 데이터셋은 **MIMIC-IV v3.1**이다 [15,16]. 연구 시점의 버전을 v3.1로 고정하고, 논문과 코드에 데이터 버전 및 SQL commit hash를 명시한다.

MIMIC-IV는 비식별화 데이터이지만 무제한 공개 데이터가 아니다. PhysioNet credentialing, 요구 교육 및 데이터 사용 동의를 준수하며, 원본 또는 환자 단위 파생 데이터를 공개하지 않는다.

## 6.2 Primary cohort: 24시간 landmark cohort

### 포함 기준

- 만 18세 이상 성인
- 환자별 최초 ICU 입실
- ICU 입실 후 24시간까지 생존
- ICU length of stay가 24시간 이상
- 첫 24시간에 사전 정의한 핵심 활력징후 중 일정 수 이상 관측

### 관측 창

\[
[t_{ICU},t_{ICU}+24h)
\]

### 결과 창

\[
[t_{ICU}+24h,t_{hospital\ discharge}]
\]

### 결과 라벨

\[
y
=
\mathbb I
\left[
\text{24시간 이후, 병원 퇴원 전 사망}
\right]
\]

이 설계는 입실 24시간 이전 사망이나 조기 퇴실로 인해 결측 마스크 자체가 결과 라벨을 직접 누설하는 문제를 줄인다. 대신 연구 모집단은 “ICU 입실 후 24시간까지 생존하고 ICU에 재원한 환자”로 제한되므로, 해당 selection을 논문의 한계로 명시한다. MIMIC 임상 예측 benchmark에서도 고정 관측 창보다 짧은 stay를 제외하는 방식이 사용되어 왔다 [17].

## 6.3 16개 핵심 특징

첫 논문에서는 1특징–1큐비트 대응을 유지하기 위해 16개 연속형 특징으로 제한한다.

| 번호 | 임상 영역 | 특징 | 24시간 요약 |
|---:|---|---|---|
| 1 | 순환 | Heart rate | 시간창 평균 |
| 2 | 순환 | Systolic blood pressure | 시간창 평균 |
| 3 | 순환 | Diastolic blood pressure | 시간창 평균 |
| 4 | 순환 | Mean arterial pressure | 시간창 평균 |
| 5 | 호흡 | Respiratory rate | 시간창 평균 |
| 6 | 호흡 | SpO₂ | 시간창 평균 |
| 7 | 체온 | Temperature | 시간창 평균 |
| 8 | 대사 | Glucose | 시간창 중앙값 |
| 9 | 전해질 | Sodium | 시간창 중앙값 |
| 10 | 전해질 | Potassium | 시간창 중앙값 |
| 11 | 전해질 | Chloride | 시간창 중앙값 |
| 12 | 산염기 | Bicarbonate | 시간창 중앙값 |
| 13 | 신장 | Creatinine | 시간창 중앙값 |
| 14 | 신장 | BUN | 시간창 중앙값 |
| 15 | 혈액 | Hemoglobin | 시간창 중앙값 |
| 16 | 혈액 | Platelet count | 시간창 중앙값 |

### 사전 정의 대체 변수

특정 변수가 training cohort에서 사전 기준보다 낮은 관측률을 보일 경우, test set을 확인하지 않고 다음 순서로 대체한다.

1. White blood cell count
2. Hematocrit
3. Calcium

최종 변수 목록은 본 실험 전에 고정하고 이후 변경하지 않는다.

## 6.4 혈압 요약의 정합성

다음 근사 관계는 동기화된 평균 또는 동일 시점 측정에서만 진단한다.

\[
MAP
\approx
\frac{SBP+2DBP}{3}
\]

최댓값이나 최솟값의 조합에는 적용하지 않는다.

\[
\max(MAP)
\neq
\frac{\max(SBP)+2\max(DBP)}{3}
\]

본 연구의 핵심 16개 특징은 혈압 변수에 평균값을 사용하므로, 해당 관계는 **평가 지표**로만 사용할 수 있다. 주 학습 손실에는 넣지 않는다.

## 6.5 데이터 분할

동일 환자가 여러 분할에 들어가지 않도록 `subject_id` 기준으로 분할한다.

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
70\%:15\%:15\%
\]

사망 여부를 기준으로 층화하되, 다음은 training set에서만 결정한다.

- 단위 변환 규칙
- 이상치 처리 경계
- scaling 통계량
- imputation 모델
- CDG 추정
- CDG edge split
- 하이퍼파라미터 및 checkpoint 선택

Test set은 최종 평가 전까지 모델 및 그래프 선택에 사용하지 않는다.

## 6.6 전처리

### 단위와 오류 처리

- 동일 임상 변수의 단위를 통일한다.
- 물리적으로 불가능한 값은 결측으로 처리한다.
- 임상적으로 가능한 극단값은 자동 제거하지 않는다.
- 오류 경계는 MIMIC Code Repository의 공개 개념 정의와 임상 문헌을 참고하여 사전에 고정한다.

### Scaling

각 특징은 training set의 median과 IQR을 사용해 robust scaling한다.

\[
\tilde x_j
=
\frac{x_j-\operatorname{median}_{train}(x_j)}
{\operatorname{IQR}_{train}(x_j)+\epsilon}
\]

최종 generator 출력에는 `tanh` 또는 bounded spline을 사용하고, 역변환 후 원 단위에서 평가한다.

## 6.7 결측 처리

본 연구의 primary mechanism experiment는 연속값 의존 구조에 집중하므로, 결측 마스크를 양자 generator의 주 출력에 포함하지 않는다.

### Primary analysis

1. 각 특징의 환자 수준 관측률을 계산한다.
2. 사전 정의된 관측률 기준을 충족하는 16개 특징을 확정한다.
3. Training set에서 iterative imputation 모델을 적합한다.
4. 동일 모델을 validation 및 test에 적용한다.
5. Imputation 이전의 mask는 별도로 보존하되 generator 입력이나 primary endpoint에는 사용하지 않는다.

### Sensitivity analyses

- Complete-case subset
- Training median imputation
- Multiple imputation 기반 CDG 안정성
- Value-only TSTR
- Value + original missingness indicator TSTR

Value+mask 성능이 value-only보다 현저히 높으면, utility가 생리값보다 측정 관행과 informative missingness에 의존할 가능성을 보고한다.

---

# 7. Clinical Dependency Graph 구축

## 7.1 CDG의 목적

CDG는 “임상적으로 관련 있어 보이는 변수”를 임의로 연결하는 지식 그래프가 아니다. 본 연구의 CDG는 training data에서 추정한 **조건부 통계 의존 구조**이며, 회로의 제한된 receptive field 안에 중요한 관계를 배치하는 데 사용한다.

## 7.2 조건 변수와 공변량 제거

각 특징을 다음 공변량에 대해 training set에서 회귀한다.

- 결과 조건 \(y\)
- 연령
- 성별
- ICU 유형

연속형 또는 범주형 공변량을 사용한 적절한 회귀모형의 residual을 얻는다.

\[
r_j
=
x_j-\widehat{\mathbb E}[x_j\mid y,\text{age},\text{sex},\text{ICU type}]
\]

CDG는 residual \(r_j\)의 조건부 관계를 추정한다. 이를 통해 단순히 사망군과 생존군의 평균 차이 때문에 모든 특징이 연결되는 문제를 줄인다.

## 7.3 Nonparanormal transformation

각 residual 특징에 rank-based inverse normal transformation을 적용한다 [4].

\[
\hat r_j
=
\Phi^{-1}
\left(
\frac{\operatorname{rank}(r_j)-0.5}{N}
\right)
\]

이는 비정규 주변분포를 가진 연속형 임상 변수에서 Gaussian copula graphical model을 근사하기 위한 절차이다.

## 7.4 희소 정밀도 행렬

정밀도 행렬 \(\Omega=\Sigma^{-1}\)을 graphical lasso로 추정한다 [3].

\[
\hat\Omega
=
\arg\min_{\Omega\succ0}
\left
\{
\operatorname{tr}(S\Omega)
-
\log\det\Omega
+
\lambda\|\Omega\|_{1,off}
\right
\}
\]

부분상관은 다음과 같다.

\[
\rho_{uv\mid V\setminus\{u,v\}}
=
-
\frac{\Omega_{uv}}
{\sqrt{\Omega_{uu}\Omega_{vv}}}
\]

간선 가중치는

\[
w_{uv}
=
|\rho_{uv\mid V\setminus\{u,v\}}|
\]

로 정의한다.

## 7.5 정규화 계수 선택과 안정성

Graphical lasso의 \(\lambda\)는 다음 중 하나를 사전에 선택한다.

- StARS 기반 안정성 선택 [5]
- Extended BIC

Bootstrap 또는 subsampling을 반복하여 각 간선의 선택 빈도를 계산한다.

\[
s_{uv}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbb I[(u,v)\in E^{(b)}]
\]

기본 안정성 임계값은

\[
s_{uv}\ge0.70
\]

으로 설정하고, 0.60과 0.80을 sensitivity analysis로 사용한다.

## 7.6 그래프 자원 제약

고차수 노드는 실제 장치의 two-qubit depth와 noise를 크게 증가시킬 수 있으므로 다음 제약을 적용한다.

- 노드 수: 16
- 최대 차수: \(\Delta\le3\) 권장
- 연결 그래프 유지
- 간선 수: 사전에 고정한 \(m\)개
- 임상적으로 강하고 안정적인 간선 우선

그래프가 분리될 경우 held-out 간선을 사용하지 않는 범위에서 가장 높은 가중치의 연결 간선을 추가한다. 모든 추가 간선과 그 이유를 공개한다.

## 7.7 Fit edge와 held-out dependency pair 분리

순환 평가를 피하기 위해 안정적인 후보 관계를 다음처럼 분리한다.

\[
E_{candidate}
=
E_{fit}\cup E_{holdout},
\qquad
E_{fit}\cap E_{holdout}=\varnothing
\]

- \(E_{fit}\): CDG topology 구성에만 사용
- \(E_{holdout}\): topology, 손실, hyperparameter 선택, checkpoint 선택에 사용하지 않음

기본 비율은

\[
70\%:30\%
\]

이다. 분할은 관계 강도와 임상 영역을 기준으로 층화하고 고정 seed를 사용한다.

`E_holdout`의 실제 목표값은 test cohort에서 다시 계산한다. 즉 training set에서 강한 관계로 식별되었더라도, 최종 평가는 독립된 test 환자에서 해당 관계가 유지되는지를 기준으로 한다.

## 7.8 비교 그래프

### Isomorphic permuted-CDG

같은 adjacency의 노드 라벨만 여러 번 무작위로 순열한다. 임상 의미 정렬의 확증적 대조군이다.

### Degree-preserving rewired graph

Double-edge swap으로 차수열과 간선 수를 유지하되 경로 구조를 바꾼다. 추상 그래프 구조 자체의 효과를 평가한다.

### Ring / ring-with-chords

회로 문헌에서 흔한 균일 토폴로지 기준선이다. 자원 비교 시 간선 수와 two-qubit gate 수를 맞추기 위해 필요한 경우 고정된 chord를 추가한다.

### No-entanglement graph

\[
E=\varnothing
\]

로 설정하여 local marginals만 생성하는 양자 모델을 구성한다.

---

# 8. CDG-QGAN 모델

## 8.1 전체 구조

```text
Condition y ──────────────────────────────────────────────┐
                                                          │
Independent local latents z1,...,z16                       │
      │                                                   │
      ▼                                                   │
Local angle encoding on corresponding qubits              │
      │                                                   │
      ▼                                                   │
CDG-edge RZZ entanglement                                  │
      │                                                   │
      ▼                                                   │
Local RX/RY mixing                                         │
      │                                                   │
      ├──────── repeat L blocks ────────┐                  │
      │                                 │                  │
      ▼                                 │                  │
Local Z expectations q1,...,q16                            │
      │                                                   │
      ▼                                                   │
Feature-specific local heads h1,...,h16 ◀──────────────────┘
      │
      ▼
Synthetic clinical vector x~ | y
      │
      ▼
Global conditional WGAN critic D(x~, y)
```

## 8.2 Local latent

\[
z
=(z_1,\ldots,z_{16}),
\qquad
z_u\sim\mathcal U(-1,1)
\]

각 \(z_u\)는 대응 특징의 local stochastic source이다. 전역 latent가 모든 큐비트에 연결되는 구조를 사용하지 않는다.

## 8.3 Data re-uploading

각 층에서 동일한 local latent를 다시 입력할 수 있다.

\[
S^{(\ell)}_u
=
R_Y(a_{\ell,u}z_u+b_{\ell,u}y+c_{\ell,u})
R_Z(d_{\ell,u}z_u+e_{\ell,u}y+f_{\ell,u})
\]

하지만 encoder가 양자 코어보다 큰 신경망이 되지 않도록 다음 중 하나를 사용한다.

### 기본안

- feature-wise affine scale·shift만 학습
- 층당 큐비트당 4–6개 이하 파라미터

### 엄격 대조안

- 고정 scale 사용
- \(z_u\)를 직접 \([-\pi,\pi]\)로 매핑
- trainable encoder parameter 없음

두 설정을 비교하여 학습 가능한 angle encoder가 성능을 대신 설명하는지 확인한다.

## 8.4 Entangling gate

각 CDG edge에 trainable `RZZ`를 적용한다.

\[
E_G^{(\ell)}
=
\prod_{(u,v)\in E_{fit}}
\exp
\left(
-i\frac{\gamma_{\ell,uv}}{2}Z_uZ_v
\right)
\]

초기값은 0 근방의 작은 범위에서 설정한다.

\[
\gamma_{\ell,uv}
\sim
\mathcal U(-\epsilon,\epsilon)
\]

CDG 가중치를 초기값 크기에 반영하는 실험은 exploratory로 제한한다. Primary comparison에서는 CDG와 permuted-CDG의 초기화 분포를 동일하게 유지한다.

## 8.5 Local mixing rotation

\[
R_{mix}^{(\ell)}
=
\prod_{u=1}^{16}
R_X(\theta^x_{\ell,u})
R_Y(\theta^y_{\ell,u})
\]

얽힘 이후 local mixing을 두어 `RZZ`가 최종 `Z` expectation에 영향을 주도록 한다.

## 8.6 측정

Generator에 전달하는 측정값은 각 큐비트의 `Z` expectation 하나로 제한한다.

\[
q_u=\langle Z_u\rangle
\]

Pairwise \(\langle Z_uZ_v\rangle\)는 진단과 noise 분석에는 사용할 수 있지만 decoder 입력으로 사용하지 않는다. 이를 decoder에 넣으면 출력의 receptive field가 넓어지고 이론적 경계가 달라지기 때문이다.

## 8.7 Feature-specific local head

\[
\tilde x_u
=h_u([q_u,y])
\]

권장 기본 구조는 다음과 같다.

```text
Input(q_u, y)
→ Linear(2, 8)
→ SiLU or Tanh
→ Linear(8, 1)
```

각 head는 다른 특징의 \(q_v\)를 볼 수 없다. Head를 공유할 수는 있지만 입력 혼합은 허용하지 않는다.

## 8.8 조건부 생성

훈련 시 실제 minibatch의 \(y\)를 사용하고, 생성 시 다음 중 하나로 \(y\)를 샘플링한다.

- 실제 training prevalence
- 균형 생성 후 목표 prevalence로 재표본화

Primary fidelity 평가에서는 실제 test prevalence에 맞춘 합성표본을 사용한다. TSTR에서는 class-balanced synthetic training과 prevalence-matched synthetic training을 모두 보고한다.

## 8.9 Critic

Critic은 전체 임상 벡터와 조건을 입력받는 전역 네트워크이다.

\[
D_\psi(x,y)
\]

권장 구조:

```text
Input(16 features + y)
→ Linear(128) + LeakyReLU
→ Linear(128) + LeakyReLU
→ Linear(64)  + LeakyReLU
→ Scalar
```

Critic이 전역 구조를 평가하는 것은 허용한다. Critic은 학습 신호를 제공하지만 생성 시 샘플 내 특징을 직접 혼합하지 않으므로 generator의 구조적 receptive field를 변경하지 않는다.

## 8.10 학습 손실

Primary model은 conditional WGAN-GP만 사용한다 [18].

### Critic loss

\[
\mathcal L_D
=
\mathbb E_{\tilde x\sim P_G}
[D_\psi(\tilde x,y)]
-
\mathbb E_{x\sim P_r}
[D_\psi(x,y)]
+
\lambda_{gp}\mathcal L_{gp}
\]

\[
\mathcal L_{gp}
=
\mathbb E_{\hat x}
\left[
\left(
\|\nabla_{\hat x}D_\psi(\hat x,y)\|_2-1
\right)^2
\right]
\]

### Generator loss

\[
\boxed{
\mathcal L_G
=
-
\mathbb E_{z,y}
[D_\psi(G_\Theta(z,y),y)]
}
\]

### Primary model에서 제외하는 손실

- CDG edge correlation loss
- Dependency matrix loss
- MAP constraint loss
- Physiological range loss
- Auxiliary mortality classifier loss
- Missing-mask correlation loss

이 항목들은 평가 지표를 직접 최적화하거나 조건 분리를 과장할 수 있으므로 주 모델에서 제거한다. 임상 제약 손실은 후속 exploratory ablation에서만 사용한다.

## 8.11 기본 하이퍼파라미터

| 항목 | 기본값 또는 탐색 범위 |
|---|---|
| 큐비트 수 | 16 고정 |
| 논리적 CDG block 깊이 | \(L\in\{1,2\}\), \(L=3\) exploratory |
| Batch size | 64 |
| Critic update | Generator 1회당 5회 |
| Optimizer | Adam |
| Critic learning rate | \(1\times10^{-4}\) |
| Generator learning rate | \(5\times10^{-5}\) 또는 \(1\times10^{-4}\) |
| Adam \((\beta_1,\beta_2)\) | \((0.0,0.9)\) |
| Gradient penalty | \(\lambda_{gp}=10\) |
| Early stopping | Validation two-sample score와 loss 안정성 기준 |
| Primary seed 수 | 최종 설정에서 10개 |
| Exploratory seed 수 | 5개 |

Epoch 수를 고정 성과 기준으로 사용하지 않고, 모든 모델에 동일한 최대 critic update budget을 부여한다.

## 8.12 파라미터 분해 보고

모든 모델에 대해 다음을 따로 보고한다.

| 구성 | 파라미터 수 |
|---|---:|
| Local angle encoding | 별도 보고 |
| Quantum single-qubit rotations | 별도 보고 |
| Quantum entangling angles | 별도 보고 |
| Local output heads | 별도 보고 |
| Generator total | 별도 보고 |
| Critic | 별도 보고 |

양자 파라미터 비중은 다음과 같이 공개한다.

\[
r_Q
=
\frac{N_{quantum}}
{N_{generator,total}}
\]

파라미터 수와 계산비용은 분리한다. 회로 시뮬레이션 비용, 회로 평가 횟수, 메모리, wall-clock time도 함께 보고한다.

---

# 9. 비교 모델

## 9.1 확증적 대조군

### 1. CDG-QGAN

임상 특징과 CDG 노드가 올바르게 정렬된 제안 모델.

### 2. Isomorphic permuted-CDG-QGAN

동일한 그래프 구조와 자원을 유지하면서 임상 특징–노드 대응만 섞은 모델. **가장 중요한 대조군**이다.

## 9.2 메커니즘 대조군

### 3. Degree-preserving rewired-QGAN

차수열과 간선 수는 같지만 경로 구조가 다른 회로.

### 4. No-entanglement QGAN

Local encoding, local mixing, local head는 동일하지만 `RZZ`를 제거한다.

### 5. Ring-QGAN 또는 resource-matched ring-with-chords

일반적인 균일 토폴로지 기준선. 간선 수와 gate budget을 맞춘 결과를 우선 보고한다.

## 9.3 고전 locality 대조군

### 6. CDG-local classical message-passing generator

동일한 CDG와 동일한 local latent를 사용한다.

\[
h_u^{(0)}
=\phi(z_u,y)
\]

\[
h_u^{(\ell+1)}
=
\sigma
\left(
W_s h_u^{(\ell)}
+
\sum_{v\in N(u)}W_m h_v^{(\ell)}
\right)
\]

\[
\tilde x_u
=h_u^{out}(h_u^{(L)},y)
\]

Message-passing round 수를 양자 회로의 논리적 깊이 \(L\)과 맞추고, generator 파라미터 수를 가능한 범위에서 일치시킨다. 이 모델이 CDG-QGAN과 동등하면 관찰된 이점은 양자성보다 graph-local inductive bias로 해석한다.

### 7. Global MLP-WGAN

동일한 critic과 WGAN-GP loss를 사용하되 전역 MLP generator를 사용한다. 절대적인 생성 품질을 비교하지만, 양자 기여의 핵심 증거로 사용하지 않는다.

## 9.4 비선형 특징맵 대조군

### 8. Random Fourier Feature generator

양자 회로 대신 고정된 random Fourier feature map을 사용한다 [19]. 출력 차원과 local head를 맞추어, “학습 가능한 양자 회로”가 아니라 단순한 비선형 특징맵만으로 성능이 설명되는지 확인한다.

## 9.5 고성능 맥락 기준선

### 9. TabDDPM

현대적인 고전 tabular diffusion 기준선이다. 전체 fidelity와 TSTR utility를 비교하지만, graph-local 메커니즘 검정에는 포함하지 않는다.

### 선택적 추가 모델

CTGAN, TVAE, Gaussian Copula는 계산 여유가 있을 때 맥락 수치로만 포함한다. 이 모델과의 우위를 양자 회로의 증거로 해석하지 않는다.

## 9.6 Clifford·matchgate 대조군에 대한 처리

Clifford 또는 matchgate 회로는 고전적 시뮬레이션 가능성에 대한 흥미로운 대조군이지만, arbitrary-angle continuous data encoding을 유지하면 Clifford simulability가 깨지고, matchgate는 게이트와 연결 구조에 별도 제약을 갖는다. 따라서 완전한 공정 비교가 가능한 구성이 확보된 경우에만 appendix 실험으로 수행한다. 핵심 비교군은 graph-local classical generator와 RFF control로 둔다.

---

# 10. 통제된 합성 그래프 실험

실제 임상 데이터에서는 참 그래프가 알려져 있지 않으므로, 이론과 구현을 검증하기 위한 controlled benchmark를 먼저 수행한다.

## 10.1 Teacher distribution

16차원 Gaussian copula 또는 nonparanormal distribution을 생성한다.

1. 알려진 sparse precision matrix \(\Omega^*\)를 정의한다.
2. \(\Sigma^*=(\Omega^*)^{-1}\)에서 Gaussian latent를 샘플링한다.
3. 각 차원에 서로 다른 비선형 단조 변환을 적용하여 비정규 주변분포를 만든다.

\[
g
\sim
\mathcal N(0,\Sigma^*)
\]

\[
x_u
=T_u(g_u)
\]

## 10.2 Teacher graph 종류

- Chain graph
- Modular graph
- Sparse degree-balanced graph

각 graph에서 aligned, permuted, rewired, ring, no-entanglement 및 classical local generator를 비교한다.

## 10.3 목적

- Light-cone 구현이 이론과 일치하는지 확인
- 깊이 \(L\)에 따라 복원 가능한 graph distance가 증가하는지 확인
- Dense decoder 없이도 의존성이 생성되는지 확인
- 회로 순서가 잘못되었을 때 마지막 `RZZ` gradient가 0이 되는지 unit test
- CDG 정렬 효과가 ground-truth graph가 알려진 조건에서 나타나는지 확인

## 10.4 핵심 그림

```text
x-axis: teacher graph distance d(u,v)
y-axis: conditional dependency restoration error
curves: L=1, L=2, L=3
models: aligned / permuted / rewired / classical-local
```

이 그림은 깊이가 증가할수록 dependency receptive field가 넓어지는 메커니즘을 직접 보여준다.

---

# 11. MIMIC-IV 실험 설계

## 11.1 Primary experiment

- 데이터: 24시간 landmark cohort
- 특징: 16개 연속형 변수
- 조건: 24시간 이후 병원 내 사망
- 회로 깊이: \(L=1,2\)
- 비교: CDG-QGAN vs isomorphic permuted-CDG-QGAN
- 손실: 동일 conditional WGAN-GP
- Generator 구조: local encoder, local observable, local head로 동일
- 확증적 endpoint: held-out conditional dependency error 하나

## 11.2 Permutation 반복

단일 permutation의 우연한 결과를 피하기 위해 사전에 고정한 여러 permutation을 사용한다.

- 기본: 3개 isomorphic permutations
- 각 permutation: 최종 설정에서 10개 seed 또는 계산 여건에 따라 5개 seed
- CDG-QGAN: 동일 seed 집합

Primary contrast는 CDG-QGAN의 HDE와 permutation 평균 HDE의 차이로 정의한다.

## 11.3 깊이 실험

### 주 깊이

\[
L\in\{1,2\}
\]

### 탐색 깊이

\[
L=3
\]

깊이가 커지면 대부분 특징 쌍의 light cone이 겹쳐 토폴로지 차이가 약해질 수 있으므로, 얕은 깊이를 주 조건으로 둔다.

## 11.4 Graph-distance stratification

각 pair를 CDG 거리로 나눈다.

- \(d=1\)
- \(d=2\)
- \(d=3\)
- \(d\ge4\)

각 깊이에서 의존성 복원 오차를 거리별로 보고한다. Permuted graph에 대해서도 동일한 특징 쌍의 permuted graph distance를 계산한다.

## 11.5 동일 자원 조건

CDG와 permuted-CDG는 완전히 동일한 자원을 사용한다. Rewired 및 ring 비교에서는 다음을 각각 보고한다.

- 동일 edge count
- 동일 trainable quantum parameter count
- 동일 logical block depth
- transpiled two-qubit gate count
- transpiled two-qubit depth

하나의 “공정성” 정의만 사용하지 않고, 논리적 구조와 실제 장치 자원을 분리하여 보고한다.

---

# 12. 평가 지표

## 12.1 유일한 확증적 primary endpoint

### Held-out Dependency Error, HDE

각 결과 조건 \(y=c\)에서 test real data와 synthetic data의 부분상관을 계산한다.

\[
\rho_{uv,real}^{(c)}
,
\qquad
\rho_{uv,syn}^{(c)}
\]

Fisher \(z\)-transform을 적용한다.

\[
z(\rho)
=
\operatorname{atanh}(\rho)
\]

조건별 held-out error는

\[
\mathrm{HDE}^{(c)}
=
\frac{1}{|E_{holdout}|}
\sum_{(u,v)\in E_{holdout}}
\left|
z(\rho_{uv,syn}^{(c)})
-
z(\rho_{uv,real}^{(c)})
\right|
\]

이다.

전체 HDE는 test prevalence로 가중한다.

\[
\boxed{
\mathrm{HDE}
=
\sum_{c\in\{0,1\}}
\hat p_{test}(y=c)
\mathrm{HDE}^{(c)}
}
\]

### Primary contrast

\[
\Delta_{HDE}
=
\mathrm{HDE}_{CDG}
-
\frac{1}{K}
\sum_{k=1}^{K}
\mathrm{HDE}_{Permuted,k}
\]

연구 가설을 지지하려면

\[
\Delta_{HDE}<0
\]

이고 95% confidence interval이 0을 포함하지 않아야 한다.

## 12.2 거리별 의존성 복원 오차

\[
E_d
=
\frac{1}{|P_d|}
\sum_{(u,v)\in P_d}
\left|
z(\rho_{uv,syn})-z(\rho_{uv,real})
\right|
\]

\[
P_d
=
\{(u,v):d_G(u,v)=d\}
\]

깊이와 거리의 interaction을 분석한다.

## 12.3 전체 의존 구조

Secondary metrics:

- 전체 partial-correlation matrix Frobenius error
- Spearman correlation matrix error
- Precision matrix edge recovery
- Top-\(k\) dependency overlap
- Conditional mutual information error

\[
E_{F}
=
\|R_{real}-R_{syn}\|_F
\]

## 12.4 주변분포 fidelity

각 특징에 대해 다음을 보고한다.

- Wasserstein-1 distance
- Kolmogorov–Smirnov statistic
- Median error
- IQR error
- Quantile error

\[
E_{quantile,j}
=
\frac{1}{K}
\sum_{k=1}^{K}
|Q_{real,j}(\tau_k)-Q_{syn,j}(\tau_k)|
\]

## 12.5 다변량 fidelity

- Maximum Mean Discrepancy
- Energy distance
- Real-vs-Synthetic classifier AUROC
- Precision/Recall for distributions
- Density and coverage

Real-vs-Synthetic AUROC가 0.5에 가까운 것은 보조 지표일 뿐, 임상적 타당성이나 프라이버시 보장을 의미하지 않는다.

## 12.6 머신러닝 활용성

### TRTR

\[
\text{Train Real, Test Real}
\]

### TSTR

\[
\text{Train Synthetic, Test Real}
\]

### 평가 모델

- Logistic Regression
- Random Forest
- XGBoost
- 작은 MLP

### 평가 지표

- AUROC
- AUPRC
- Brier score
- Expected Calibration Error
- Sensitivity at fixed specificity

TSTR이 TRTR보다 높다고 자동으로 성공으로 판정하지 않는다. 다음을 함께 분석한다.

- Synthetic와 real의 classifier score distribution
- Calibration curve
- Class-conditional feature overlap
- \(P(y\mid x)\)의 과도한 분리 여부

## 12.7 임상적 개연성

임상 제약은 학습 손실이 아니라 진단 지표로 사용한다.

### 혈압 순서

\[
DBP\le MAP\le SBP
\]

### 평균 MAP 근사

\[
MAP\approx\frac{SBP+2DBP}{3}
\]

합성 데이터의 위반율을 0으로 만드는 것이 목표가 아니다. 실제 test data와의 차이를 평가한다.

\[
E_{violation}
=
\left|
VR_{syn}-VR_{real}
\right|
\]

위반 심각도 분포도 비교한다.

\[
D_{severity}
=
D
\left(
P_{severity}^{syn},
P_{severity}^{real}
\right)
\]

## 12.8 다양성과 암기

- Synthetic exact duplicate rate
- Synthetic-to-train nearest-neighbor distance
- Synthetic-to-test nearest-neighbor distance
- Pairwise distance distribution
- Coverage

높은 uniqueness만으로 좋은 모델이라고 판단하지 않는다.

## 12.9 탐색적 프라이버시 감사

프라이버시는 첫 논문의 핵심 novelty와 제목에서 제외한다. 다만 합성 의료 데이터로서 최소한의 empirical audit를 수행할 수 있다.

### 권장 분석

- DOMIAS 또는 density-based membership inference [20]
- Nearest-neighbor membership score
- Attribute inference

### Fidelity-matched 비교

못 학습한 모델이 자동으로 안전해 보이는 교락을 피하기 위해 여러 checkpoint에서 다음 쌍을 계산한다.

\[
(
\text{fidelity},
\text{attack advantage}
)
\]

모델 간 privacy를 단일 MIA AUC로 비교하지 않고 privacy–fidelity Pareto frontier로 제시한다.

Formal differential privacy는 첫 논문에서 제외하고 후속 연구로 분리한다.

---

# 13. NISQ 및 측정 실험

## 13.1 범위 축소 원칙

모든 모델·깊이·shot·noise 조합을 전부 훈련하지 않는다. 다음 순서로 제한한다.

1. Ideal analytic simulation에서 모델 선택
2. CDG와 핵심 대조군의 최종 checkpoint 선택
3. 선택된 모델에만 finite-shot 평가
4. 선택된 일부 모델에 shot-noise-aware fine-tuning
5. 하나 또는 소수의 calibrated noise model 평가
6. 실제 장치 사용은 가능한 경우 소규모 inference validation으로 제한

## 13.2 Finite-shot 조건

\[
S\in\{256,1024,4096\}
\]

각 shot 조건에서 HDE, 전체 dependency error 및 marginal fidelity의 변화를 측정한다.

## 13.3 동일 측정 기저의 장점

모든 \(Z_i\)와 진단용 \(Z_iZ_j\)는 계산 기저에서 대각이다. 따라서 하나의 Z-basis bitstring batch에서 다음을 동시에 추정할 수 있다.

\[
\langle Z_i\rangle
\]

\[
\langle Z_iZ_j\rangle
\]

측정 observable 수가 늘어도 별도의 basis-setting 수가 간선 수만큼 증가하지 않는다. 다만 유한 shot에서 estimator covariance와 다중 관측량의 정확도 요구는 별도로 분석한다.

## 13.4 Shot-noise-aware training

Pauli observable \(O_a\)의 shot estimator는 근사적으로

\[
\operatorname{Var}(\hat O_a)
=
\frac{1-\mu_a^2}{S}
\]

을 갖는다. 동일 bitstring에서 추정한 관측량은 상관된 noise를 갖는다.

\[
\operatorname{Cov}(
\hat O_a,
\hat O_b
)
=
\frac{
\langle O_aO_b\rangle-\mu_a\mu_b
}{S}
\]

따라서 단순 독립 \(1/\sqrt S\) Gaussian noise 대신 다음 correlated surrogate를 사용한다.

\[
\epsilon
\sim
\mathcal N
\left(
0,
\frac{\Sigma_O}{S}
\right)
\]

\[
\tilde q
=q+\epsilon
\]

이 noise를 local expectation vector에 주입하여 짧은 fine-tuning을 수행한다. Exact finite-shot parameter-shift training은 회로 실행량이 지나치게 크므로 핵심 실험에서 제외한다.

## 13.5 Noise model

- Ideal statevector
- Finite-shot noiseless sampling
- Backend-calibrated depolarizing/readout noise
- Readout mitigation 전후

Noise model의 calibration date와 backend 정보를 고정해 보고한다. 여러 backend 중 가장 좋은 결과만 선택하지 않는다.

## 13.6 Trainability 분석

4–16큐비트에서 관찰된 gradient 감소를 `barren plateau`라고 단정하지 않는다. 다음 용어와 지표를 사용한다.

- Gradient norm
- Gradient variance
- Gradient signal-to-noise ratio
- Hessian 또는 empirical Fisher conditioning의 근사
- Convergence rate
- Seed sensitivity
- Plateau duration

Barren plateau라는 표현은 큐비트 수에 따른 지수적 scaling을 충분히 검증한 경우에만 사용한다 [21].

---

# 14. 구현 방법

## 14.1 주 시뮬레이터

Primary training은 PyTorch 기반 batched statevector simulator로 구현한다.

16큐비트 상태벡터는 샘플당

\[
2^{16}=65,536
\]

개의 complex amplitude를 가진다. Batch 64, `complex64` 원시 상태 텐서는 약 32 MiB이지만, autograd intermediate와 optimizer state를 포함하면 실제 메모리는 훨씬 커질 수 있으므로 gradient checkpointing과 custom gate kernel을 사용한다.

## 14.2 게이트 구현

- `RY`, `RZ`, `RX`: target axis reshape 후 \(2\times2\) matrix multiplication
- `RZZ`: basis index의 parity를 이용한 diagonal phase multiplication
- Expectation \(\langle Z_u\rangle\): basis probability의 부호 가중합
- Batched latent: `(batch, 2**n)` complex tensor

`RZZ`는 diagonal gate이므로 dense \(4\times4\) 행렬곱보다 phase multiplication으로 구현한다.

## 14.3 미분

Ideal simulation에서는 PyTorch reverse-mode autograd를 사용한다. Parameter-shift는 실제 장치 또는 framework 교차검증에만 사용한다.

## 14.4 정확성 검증

Custom simulator 결과를 PennyLane 또는 Qiskit의 소규모 회로와 비교한다.

### Unit tests

- 2–6큐비트 random circuit state fidelity
- \(Z\) expectation 일치
- PyTorch gradient와 parameter-shift gradient 일치
- `RZZ`가 측정 직전에 있을 때 \(\partial\langle Z\rangle/\partial\gamma=0\)인지 확인
- `RZZ → RX/RY → Z` 순서에서는 gradient가 비영인지 확인
- Permutation equivariance test
- Locality/light-cone support test

## 14.5 PennyLane·Qiskit의 역할

- Custom simulator correctness verification
- Circuit transpilation
- Hardware topology mapping
- Calibrated noise simulation
- 실제 backend inference

PennyLane 또는 Qiskit을 주 훈련 경로로 사용할지는 동일 회로에서 wall-clock benchmark 후 결정한다. 특정 framework가 항상 느리거나 빠르다고 사전 단정하지 않는다.

## 14.6 재현 가능한 실험 구성

모든 실험은 config 파일로 관리한다.

```yaml
model: cdg_qgan
n_qubits: 16
logical_depth: 1
graph_seed: 20260711
permutation_id: 0
training_seed: 1
batch_size: 64
critic_steps: 5
loss: wgan_gp
lambda_gp: 10
feature_set: core16_v1
cohort: landmark24h_v1
```

---

# 15. 학습 알고리즘

## 15.1 CDG-QGAN 학습 의사코드

```text
Input:
    training data {(x_i, y_i)}
    fitted graph G = (V, E_fit)
    local quantum generator G_Θ
    conditional critic D_ψ

Initialize Θ, ψ

repeat until stopping criterion:

    for t = 1,...,n_critic:
        sample real minibatch {(x_i, y_i)}
        sample independent local latents z_i,u ~ Uniform(-1, 1)

        q_i ← quantum_expectations(z_i, y_i; G, Θ)
        x_fake_i,u ← local_head_u(q_i,u, y_i)

        x_hat_i ← α_i x_i + (1-α_i) x_fake_i
        L_D ← mean[D_ψ(x_fake, y)]
              - mean[D_ψ(x_real, y)]
              + λ_gp GP(x_hat, y)

        update ψ using ∇_ψ L_D

    sample labels y and independent local latents z
    q ← quantum_expectations(z, y; G, Θ)
    x_fake_u ← local_head_u(q_u, y)

    L_G ← -mean[D_ψ(x_fake, y)]
    update Θ and local-head parameters using ∇ L_G

    evaluate validation diagnostics
    save checkpoint if pre-registered validation rule improves
```

## 15.2 Checkpoint 선택

Primary endpoint인 held-out dependency error를 validation checkpoint 선택에 사용하지 않는다. 권장 validation rule은 다음 조합이다.

- validation real-vs-synthetic classifier AUROC
- 평균 univariate Wasserstein distance
- critic/generator loss 안정성
- mode collapse indicator

Held-out dependency pair는 최종 test에서만 사용한다.

---

# 16. Ablation study

## 16.1 필수 ablation

| ID | 실험 | 목적 |
|---|---|---|
| A1 | Isomorphic permuted-CDG | 임상 특징–노드 정렬 효과 |
| A2 | Degree-preserving rewiring | 추상 topology와 경로 구조 효과 |
| A3 | No entanglement | 얽힘의 필요성 |
| A4 | Ring/resource-matched ring | 일반 회로 토폴로지 비교 |
| A5 | Graph-local classical generator | 양자 효과와 locality 효과 분리 |
| A6 | Fixed angle encoding | 학습 가능한 angle encoder 기여 분리 |
| A7 | Random Fourier feature control | 일반 비선형 특징맵과 비교 |
| A8 | \(L=1\) vs \(L=2\) | light-cone 확대 효과 |

## 16.2 선택적 ablation

| ID | 실험 | 목적 |
|---|---|---|
| B1 | Local head width 1/4/8/16 | 고전 head 용량 영향 |
| B2 | \(L=3\) | 토폴로지 차이 소실 여부 |
| B3 | Different graph stability thresholds | CDG 추정 민감도 |
| B4 | Median vs iterative imputation | 결측 처리 영향 |
| B5 | Shot-noise-aware fine-tuning | finite-shot 강건성 |
| B6 | Exploratory clinical constraint loss | 구조 prior와 손실 prior 분리 |

## 16.3 제외하는 과도한 grid

다음 전체 Cartesian grid는 수행하지 않는다.

\[
\text{qubit count}
\times
\text{depth}
\times
\text{shot}
\times
\text{noise}
\times
\text{ablation}
\times
\text{baseline}
\times
\text{seed}
\]

큐비트 수는 16으로 고정하고, 깊이는 1–2에 집중하며, shot/noise는 최종 모델에만 적용한다.

---

# 17. 통계 분석

## 17.1 Primary hypothesis 하나

확증적 검정은 다음 하나로 제한한다.

\[
H_0:
\Delta_{HDE}\ge0
\]

\[
H_1:
\Delta_{HDE}<0
\]

## 17.2 불확실성 추정

다음 변동성을 함께 반영한다.

- Training seed
- Permutation instance
- Synthetic sample draw
- Test patient resampling

권장 방법은 hierarchical bootstrap이다.

1. Model seed를 복원추출
2. Permutation을 복원추출
3. Test 환자를 복원추출
4. 동일 크기의 synthetic cohort를 다시 샘플링
5. \(\Delta_{HDE}\) 계산

95% confidence interval과 effect size를 보고한다.

## 17.3 Secondary analyses

거리·깊이·모델의 관계는 mixed-effects model로 분석할 수 있다.

\[
e_{uvs}
=
\beta_0
+
\beta_1\operatorname{Model}
+
\beta_2 d_G(u,v)
+
\beta_3 L
+
\beta_4(\operatorname{Model}\times d_G)
+
\beta_5(d_G\times L)
+
u_s+
\epsilon_{uvs}
\]

여기서 \(s\)는 training seed이다.

Secondary metric의 다중비교에는 Benjamini–Hochberg FDR 또는 Holm correction을 적용하되, 결과를 탐색적으로 명시한다.

## 17.4 보고 원칙

- 평균과 표준편차만이 아니라 seed별 raw result 공개
- 95% confidence interval
- 절대 차이와 상대 차이
- Effect size
- 실패한 seed와 mode collapse 포함
- 가장 좋은 seed만 선택해 제시하지 않음

---

# 18. 성공, 실패 및 반증 기준

## 18.1 중심 가설 지지

다음이 모두 충족될 때 임상 의미 정렬 효과를 지지한다.

1. \(\Delta_{HDE}<0\)
2. 95% CI가 0을 포함하지 않음
3. Controlled synthetic benchmark에서도 동일 방향
4. 특정 permutation 하나에만 의존하지 않음
5. Marginal fidelity가 현저히 악화된 결과가 아님

## 18.2 양자 코어의 추가적 기여

다음이 충족될 때만 “quantum core의 추가 이점”을 제한적으로 주장한다.

- CDG-QGAN이 CDG-local classical generator보다 낮은 HDE
- 전체 generator parameter budget이 유사
- Receptive field가 동일
- 동일 critic, loss, data split, update budget 사용
- 결과가 여러 seed에서 유지

동등하면 결론은 다음으로 제한한다.

> 데이터 의존적 graph-local inductive bias는 유효했지만, 본 설정에서 양자 회로가 matched classical message passing보다 추가 이점을 제공한다는 증거는 얻지 못했다.

## 18.3 중심 가설 기각

CDG와 permuted-CDG의 차이가 없으면 다음을 인정한다.

- 임상 노드 정렬이 성능에 기여하지 않았음
- 선택한 깊이에서 topology가 충분한 병목이 아니었을 수 있음
- CDG 추정이 불안정했을 수 있음
- Critic과 local heads가 회로 차이를 학습에서 상쇄했을 수 있음
- 실제 조건부 의존 구조가 선택한 graph-local 가정과 맞지 않을 수 있음

이 경우 CTGAN이나 TabDDPM보다 특정 지표가 좋다는 결과로 중심 주장을 대체하지 않는다.

## 18.4 No-entanglement와 동등한 경우

얽힘이 없는 모델과 동등하면 양자 얽힘의 기여를 주장하지 않는다. 해당 결과는 local nonlinear feature map만으로 성능이 설명된다는 근거로 보고한다.

---

# 19. 예상 결과 형태

이 연구는 특정 수치 개선을 사전에 단정하지 않는다. 기대하는 결과의 **형태**는 다음과 같다.

## 19.1 이론 검증 그림

- \(L=1\): 가까운 graph-distance pair에서만 의존성 복원
- \(L=2\): 복원 가능한 거리 범위 확대
- 깊이가 커질수록 aligned와 permuted의 차이가 감소할 가능성

## 19.2 임상 결과 그림

- Held-out dependency pair별 real vs synthetic partial correlation scatter
- CDG와 permuted-CDG의 HDE 분포
- Graph distance별 error curve
- Depth별 dependency restoration heatmap

## 19.3 양자·고전 비교

- CDG-QGAN vs graph-local classical generator의 Pareto plot
- x축: generator parameter 또는 runtime
- y축: HDE 또는 full dependency error

## 19.4 NISQ 결과

- Shot 수에 따른 HDE 변화
- Noise-aware fine-tuning 전후
- Transpiled two-qubit depth와 성능 저하 관계

---

# 20. 실현 가능성

## 20.1 단일 저자 핵심 범위

### 반드시 수행

- 16-feature MIMIC-IV landmark cohort
- CDG 추정과 held-out edge split
- Custom batched statevector simulator
- CDG-QGAN
- Isomorphic permuted-CDG
- No-entanglement model
- Graph-local classical generator
- Controlled synthetic graph benchmark
- HDE primary endpoint
- Distance-depth analysis
- 최소한의 marginal fidelity와 TSTR utility

### 계산 여유에 따라 수행

- Degree-preserving rewiring
- RFF control
- Ring-QGAN
- TabDDPM
- Finite-shot 및 하나의 calibrated noise model
- Privacy–fidelity Pareto audit

### 첫 논문에서 제외

- 30–45개 특징 커뮤니티 압축을 주 모델로 사용
- 전체 longitudinal EHR 생성
- Formal differential privacy
- 여러 실제 양자 backend의 대규모 학습
- Quantum diffusion 동시 개발
- 의료 영상·임상 노트 결합
- 다기관 외부 검증을 핵심 요구사항으로 설정

## 20.2 확장성 실험

30–45개 특징을 커뮤니티로 압축하는 구조는 첫 논문의 주 모델이 아니라 후속 확장성 실험으로 둔다. 확장 시에도 각 community-specific output head가 대응 qubit와 인접 pair observable만 읽도록 구조화해야 한다.

---

# 21. 연구 위험과 대응

## 21.1 CDG 추정 불안정

### 위험

MIMIC 변수의 missingness, 비정규성, 공통 중증도 요인으로 인해 partial-correlation graph가 불안정할 수 있다.

### 대응

- Nonparanormal transform
- Residualization
- Stability selection
- 여러 imputation에서 edge consistency 확인
- Graph threshold sensitivity analysis
- 간선별 bootstrap stability 공개

## 21.2 Local generator의 절대 생성 품질 저하

### 위험

강한 locality 제약 때문에 global MLP나 TabDDPM보다 전체 fidelity가 낮을 수 있다.

### 대응

이 연구의 핵심은 절대 SOTA가 아니라 구조적 메커니즘 검정이다. 다만 HDE 개선이 심각한 marginal fidelity 손실과 함께 발생하면 성공으로 해석하지 않는다. Fidelity–structure trade-off를 함께 제시한다.

## 21.3 Condition leakage와 과분리

### 위험

조건부 생성이 사망군과 생존군을 실제보다 과도하게 분리할 수 있다.

### 대응

- Auxiliary condition loss 사용 금지
- 실제 prevalence와 balanced generation 모두 평가
- Brier score, ECE, score distribution 확인
- Class-conditional two-sample test

## 21.4 결측 및 landmark selection

### 위험

Landmark cohort는 24시간 이전 사망자와 조기 퇴실자를 제외하여 일반화 범위를 제한한다.

### 대응

- 연구 모집단을 명확히 제한하여 기술
- 전체 ICU cohort에 대한 secondary descriptive analysis
- Value-only와 value+mask utility 분리
- 후속 연구에서 variable observation window 모델 검토

## 21.5 Custom simulator 오류

### 위험

직접 구현한 quantum simulator의 미세한 오류가 전체 결과를 왜곡할 수 있다.

### 대응

- PennyLane/Qiskit 교차검증
- State fidelity 및 gradient unit test
- 작은 회로 exhaustive test
- 회로 순서와 qubit indexing test
- 공개 코드와 continuous integration

## 21.6 계산비용

### 위험

16큐비트 batch statevector의 autograd memory가 커질 수 있다.

### 대응

- `complex64`
- Gradient checkpointing
- RZZ diagonal kernel
- Smaller batch + gradient accumulation
- Local head와 critic의 mixed precision
- 최종 설정 전 microbenchmark

---

# 22. 윤리·프라이버시·재현성

## 22.1 데이터 사용

- PhysioNet DUA 준수
- 원본 및 환자 단위 파생 데이터 비공개
- 승인되지 않은 외부 저장소 업로드 금지
- 데이터 접근 자격이 있는 환경에서만 실행

## 22.2 합성 데이터 공개

합성 데이터는 자동으로 안전하다고 간주하지 않는다. 공개 전 다음을 검토한다.

- Exact/near duplicate
- Membership inference
- Rare subgroup disclosure risk
- PhysioNet/MIMIC 이용 조건

안전성이 불확실하면 합성 데이터 파일 대신 생성 코드와 평가 결과만 공개한다.

## 22.3 임상 해석

단일 저자이며 임상의 검토가 없을 경우 다음 표현을 사용한다.

- `statistical clinical consistency`
- `physiological plausibility`
- `dependency preservation`

다음 표현은 피한다.

- `clinically validated`
- `safe for clinical use`
- `clinically equivalent to real patients`

## 22.4 공개 산출물

- Cohort SQL
- Feature dictionary
- Unit conversion rules
- Preprocessing code
- CDG estimation code
- Graph split seed
- Quantum simulator
- Model configs
- Evaluation code
- Seed별 결과
- Transpilation/noise settings
- Environment lock file

---

# 23. 논문 구성안

## 23.1 Introduction

1. 의료 합성 데이터의 구조 보존 문제
2. 기존 양자 표형 생성의 dense encoder/decoder 한계
3. 얕은 graph-local circuit의 depth-limited dependency receptive field
4. 임상 의존 그래프와 회로 토폴로지 정렬이라는 문제 정의
5. 기여 요약

## 23.2 Theory

- Graph-local generator 정의
- `RZZ → local mixing` 회로 구조
- Light-cone support 명제와 증명
- 조건부 독립성 따름정리
- Permuted-CDG 검정의 정당성

## 23.3 Methods

- MIMIC-IV landmark cohort
- 16-feature selection
- CDG 추정
- Held-out dependency pair
- CDG-QGAN
- Classical locality baseline
- Training and implementation

## 23.4 Experiments

- Controlled synthetic graph benchmark
- MIMIC-IV primary experiment
- Depth-distance analysis
- Baselines and ablations
- NISQ robustness

## 23.5 Results

- Primary HDE contrast
- Distance-depth curves
- Fidelity and utility
- Quantum vs classical local generator
- Resource accounting

## 23.6 Discussion

- 무엇이 CDG 정렬의 효과인지
- 양자 코어의 추가적 기여가 있었는지
- 결과가 없는 경우의 해석
- 임상 데이터와 NISQ의 한계
- 확장성 방향

---

# 24. 예상 기여

## 24.1 이론적 기여

그래프 로컬 `RZZ` 양자 생성기에서 깊이에 따라 local observable의 backward light cone이 어떻게 확장되는지 형식화하고, 조건부 독립 local latent 아래에서 출력 의존성의 거리 제한을 제시한다.

## 24.2 방법론적 기여

임상 조건부 의존 그래프를 얕은 양자 생성기의 dependency receptive field에 정렬하는 CDG-QGAN을 제안한다.

## 24.3 실험 설계 기여

추상 그래프 구조를 완전히 유지하면서 임상 노드 의미만 파괴하는 isomorphic permuted-CDG를 통해, 의료 그래프의 의미적 정렬이 실제로 기여하는지 직접 검정한다.

## 24.4 비교 방법 기여

- Graph-local classical generator
- No-entanglement quantum generator
- Fixed nonlinear feature map
- Global high-capacity baseline

을 함께 사용하여 graph prior, locality, nonlinear map 및 quantum core의 효과를 분리한다.

## 24.5 실용적 기여

Batched statevector implementation, one-basis measurement, shot-noise-aware fine-tuning 및 transpiled resource report를 통해 양자 표형 생성 연구의 재현 가능한 NISQ 평가 절차를 제시한다.

---

# 25. 저널 방향

## 25.1 현실적 1차 목표

**Quantum Machine Intelligence**

적합한 조건:

- CDG-QGAN의 방법론적 신규성
- Light-cone 이론과 실험의 일치
- Permuted-CDG 확증 비교
- Matched local classical baseline
- 재현 가능한 NISQ resource 분석

## 25.2 상향 목표

**Quantum Science and Technology**

추가로 필요한 요소:

- 이론 명제의 일반화
- 의료 외 controlled tabular benchmark
- 실제 하드웨어 또는 설득력 있는 hardware-aware 결과
- 양자·고전 local generator의 자원–성능 trade-off

## 25.3 의료 정보학 저널로의 전환

JAMIA 또는 IEEE Journal of Biomedical and Health Informatics를 목표로 할 경우 다음이 더 중요하다.

- eICU 외부 검증
- 더 넓은 임상 변수 집합
- 임상의 검토
- 데이터 활용 시나리오
- subgroup utility 및 fairness

현재 v2의 중심은 의료 응용 그 자체보다 **양자 생성기의 구조적 표현 메커니즘**이므로 양자 AI 저널이 더 적합하다.

---

# 26. 최종 연구 방향

본 연구는 다음 세 축으로 고정한다.

## 첫째, CDG가 실제로 출력까지 작동하게 한다

- 1특징 = 1큐비트
- Local latent
- Local angle encoding
- Local observable
- Feature-specific local head
- Dense decoder 제거

## 둘째, 성능 우위가 아니라 구조적 제약을 검증한다

- Light-cone 명제
- Depth-limited dependency receptive field
- Graph distance별 복원 오차
- Controlled graph benchmark

## 셋째, 하나의 확증적 질문에 집중한다

\[
\boxed{
\text{CDG alignment가 held-out clinical dependency 복원을 개선하는가?}
}
\]

이 질문에 대한 가장 강한 대조군은 CTGAN이 아니라 **isomorphic permuted-CDG**이다. Quantum core의 추가적 의미는 graph-local classical generator와의 비교 후에만 판단한다.

---

# 참고문헌

[1] Lieb, E. H., & Robinson, D. W. (1972). *The finite group velocity of quantum spin systems*. Communications in Mathematical Physics, 28, 251–257. https://doi.org/10.1007/BF01645779

[2] Bravyi, S., Hastings, M. B., & Verstraete, F. (2006). *Lieb-Robinson bounds and the generation of correlations and topological quantum order*. Physical Review Letters, 97, 050401. https://doi.org/10.1103/PhysRevLett.97.050401

[3] Friedman, J., Hastie, T., & Tibshirani, R. (2008). *Sparse inverse covariance estimation with the graphical lasso*. Biostatistics, 9(3), 432–441. https://doi.org/10.1093/biostatistics/kxm045

[4] Liu, H., Lafferty, J., & Wasserman, L. (2009). *The nonparanormal: Semiparametric estimation of high dimensional undirected graphs*. Journal of Machine Learning Research, 10, 2295–2328.

[5] Liu, H., Roeder, K., & Wasserman, L. (2010). *Stability approach to regularization selection (StARS) for high dimensional graphical models*. Advances in Neural Information Processing Systems 23.

[6] Meinshausen, N., & Bühlmann, P. (2010). *Stability selection*. Journal of the Royal Statistical Society: Series B, 72(4), 417–473. https://doi.org/10.1111/j.1467-9868.2010.00740.x

[7] Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). *MIMIC-IV, a freely accessible electronic health record dataset*. Scientific Data, 10, 1. https://doi.org/10.1038/s41597-022-01899-x

[8] Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017). *Generating multi-label discrete patient records using generative adversarial networks*. Proceedings of Machine Learning for Healthcare, PMLR 68, 286–305.

[9] Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). *Modeling tabular data using conditional GAN*. Advances in Neural Information Processing Systems 32.

[10] Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023). *TabDDPM: Modelling tabular data with diffusion models*. Proceedings of ICML 2023, PMLR 202.

[11] Lloyd, S., & Weedbrook, C. (2018). *Quantum generative adversarial learning*. Physical Review Letters, 121, 040502. https://doi.org/10.1103/PhysRevLett.121.040502

[12] Zoufal, C., Lucchi, A., & Woerner, S. (2019). *Quantum generative adversarial networks for learning and loading random distributions*. npj Quantum Information, 5, 103. https://doi.org/10.1038/s41534-019-0223-2

[13] Bhardwaj, P., Jones, C., Dierich, L., & Vučković, A. (2025). *TabularQGAN: A Quantum Generative Model for Tabular Data*. arXiv:2505.22533. **Preprint.** https://arxiv.org/abs/2505.22533

[14] Kumari, S., Achutha, R., & Sivaraman, V. (2026). *QTabGAN: A Hybrid Quantum-Classical GAN for Tabular Data Synthesis*. arXiv:2602.12704. **Preprint.** https://arxiv.org/abs/2602.12704

[15] Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). *MIMIC-IV* (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58

[16] Goldberger, A. L., Amaral, L. A. N., Glass, L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation, 101(23), e215–e220.

[17] Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., & Galstyan, A. (2019). *Multitask learning and benchmarking with clinical time series data*. Scientific Data, 6, 96. https://doi.org/10.1038/s41597-019-0103-9

[18] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). *Improved training of Wasserstein GANs*. Advances in Neural Information Processing Systems 30.

[19] Rahimi, A., & Recht, B. (2007). *Random features for large-scale kernel machines*. Advances in Neural Information Processing Systems 20.

[20] van Breugel, B., Sun, H., Qian, Z., & van der Schaar, M. (2023). *Membership inference attacks against synthetic data through overfitting detection*. Proceedings of AISTATS 2023, PMLR 206.

[21] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). *Barren plateaus in quantum neural network training landscapes*. Nature Communications, 9, 4812. https://doi.org/10.1038/s41467-018-07090-4

---

# 부록 A. 사전 등록 체크리스트

- [ ] MIMIC-IV 버전 v3.1 고정
- [ ] Cohort SQL 및 commit hash 고정
- [ ] 24시간 landmark 정의 고정
- [ ] 16개 특징과 대체 순서 고정
- [ ] Unit conversion 및 오류 경계 고정
- [ ] Train/validation/test subject split 고정
- [ ] Imputation 방식 고정
- [ ] CDG residualization 공변량 고정
- [ ] Graphical lasso 선택 규칙 고정
- [ ] Edge stability threshold 고정
- [ ] \(E_{fit}\)/\(E_{holdout}\) split seed 고정
- [ ] Permutation 목록 고정
- [ ] Primary endpoint HDE 수식 고정
- [ ] Primary comparison CDG vs permuted-CDG 고정
- [ ] Validation checkpoint rule 고정
- [ ] Primary seed 목록 고정
- [ ] Secondary metrics를 exploratory로 명시

# 부록 B. 필수 구현 검증

- [ ] RZZ 바로 뒤 Z 측정에서 entangling gradient가 0인지 검증
- [ ] RZZ 뒤 RX/RY를 둘 때 gradient가 비영인지 검증
- [ ] Custom simulator와 PennyLane statevector 일치
- [ ] Autograd와 parameter-shift gradient 일치
- [ ] Graph permutation test
- [ ] Local latent 독립성 test
- [ ] Local head가 다른 큐비트 입력을 읽지 않는지 architecture test
- [ ] Depth \(L\)별 numerical light-cone test
- [ ] 동일 Z-basis sample에서 Z와 ZZ estimator 검증


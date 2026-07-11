---
title: "CDG-QGAN: Structure-Preserving Synthetic ICU Data Generation via Receptive-Field Alignment between a Clinical Conditional-Dependency Graph and a Shallow Quantum Generator"
subtitle: "Architectural Redesign Research Plan — Integrated Theory, Model, and Experiments"
lang: en-US
version: "2.0"
date: "2026-07-11"
status: "Research design proposal"
---

# CDG-QGAN Research Plan v2.0

## Official Study Title

### Korean

**CDG-QGAN: 임상 조건부 의존 그래프와 얕은 양자 생성기의 수용영역 정렬을 이용한 구조 보존형 합성 중환자 데이터 생성**

### English

**CDG-QGAN: Aligning Clinical Dependency Graphs with the Depth-Limited Receptive Fields of Shallow Quantum Generators for Structure-Preserving Synthetic ICU Data**

### Acronym

**CDG-QGAN** = **C**linical **D**ependency **G**raph-aligned **Q**uantum **G**enerative **A**dversarial **N**etwork

> **Terminological boundary**  
> In this study, the CDG is not a causal graph but a **conditional statistical dependency graph** estimated from the training data. Accordingly, we do not use the expressions `causal graph`, `causal discovery`, or `causal effect`. To support an interpretation closer to a conditional-independence graph, we use a sparse precision matrix and partial correlations rather than simple correlation coefficients.

---

# Research Summary

Research on medical data synthesis has the potential to alleviate problems of data accessibility and privacy, but generated patient records often resemble only the marginal distributions of the real variables while losing the physiological relationships among them. Quantum generative models may be able to express nonlinear distributions with a limited number of parameters; however, in existing hybrid QGAN designs, a global angle encoder and a dense decoder freely mix all features before and after the quantum circuit. In that case, even if a clinical graph is used as the entanglement topology of the quantum circuit, the correspondence between a particular clinical feature and a particular qubit is not identifiable inside the model, and it is difficult to distinguish whether the correlation structure of the final output arose from the graph structure of the quantum circuit or from the classical neural network.

To resolve this problem, this study proposes a **graph-local hybrid quantum generator** that maps **16 clinical features one-to-one onto 16 qubits**, feeds an independent local latent into each qubit, and restricts each output feature to read only the local observable of its corresponding qubit. The quantum circuit places `RZZ` entangling gates only on the edges of the clinical conditional-dependency graph, and places a non-commuting local mixing rotation after the entanglement so that the entangling phase information is actually reflected in the final `Z` measurement values.

In this architecture, in a graph-local circuit of depth \(L\), an output feature \(u\) depends only on the local latents within graph radius \(L\). Given the condition variable \(y\), if the local latents are mutually independent and the backward light cones of two outputs do not overlap, then two outputs with graph distance \(d_G(u,v)>2L\) become conditionally independent. Therefore, which dependencies a shallow circuit can express is limited by the distance structure of the entanglement graph. The core hypothesis of CDG-QGAN is that placing strong clinical dependencies within short graph distances allows those relationships to be recovered better, at the same depth and with the same resources, than in a graph whose clinical meaning has been randomly shuffled.

The main experiments target the first ICU admissions of adults in MIMIC-IV v3.1. To avoid the problem in which death and early discharge within the observation window leak the mortality label through the missingness pattern, we use a **24-hour landmark cohort** that includes only patients who survive to 24 hours after ICU admission and remain in the ICU. We generate 16 continuous physiological and laboratory features from the first 24 hours, and the outcome label is defined as in-hospital death after 24 hours. The model generates \(x\mid y\) conditional on the label \(y\), while both the theory and the dependency evaluation are carried out conditional on \(y\).

The confirmatory comparison is performed between CDG-QGAN and an **isomorphic permuted-CDG-QGAN**. Permuted-CDG fully preserves the number of edges, the degree sequence, the diameter, the abstract graph structure, and the quantum resources of the graph, while randomly shuffling only the correspondence between clinical features and graph nodes. The difference between the two models therefore arises not from graph complexity or parameter count, but from the **alignment between clinical meaning and the circuit's receptive field**. The confirmatory primary endpoint is restricted to a single quantity: the conditional partial-correlation recovery error on held-out clinical dependency pairs that were not used in model construction or in the loss.

The generator loss of the main model does not use a clinical correlation loss, a CDG edge loss, or an auxiliary condition loss. The generator is trained with the identical conditional WGAN objective alone. This choice eliminates the circular argument of directly optimizing the evaluation metric and makes it possible to interpret the observed differences in structure recovery as the inductive bias of the circuit topology. By comparing a graph-local classical message-passing generator, a no-entanglement quantum model, a random Fourier feature control, a global MLP-WGAN, and TabDDPM, we separate the graph effect, the locality effect, the nonlinear feature-map effect, and the quantum circuit effect.

This study does not claim a universal quantum advantage. The core result is a falsifiable answer to the following question.

> **If the depth-limited dependency receptive field of a shallow graph-local quantum generator is aligned with a clinical conditional-dependency graph, does it recover unseen clinical dependencies more accurately than a circuit whose clinical meaning has been randomly shuffled, under matched topology, depth, and parameter conditions?**

---

# 1. Research Background

## 1.1 The Need for Synthetic Medical Data

Electronic Health Records (EHRs) are an important resource for medical artificial intelligence research, but their access and sharing are restricted by patient privacy, institution-specific data use agreements, research ethics review, and the small sample sizes of rare patient groups. Synthetic data offer the advantage of generating new patient samples without directly releasing the original records, but the following problems remain.

- The means and variances of individual variables may be similar while the relationships among variables are broken.
- Rare patient groups and extreme values may be excessively smoothed.
- A conditional generator may separate the outcome label too sharply, producing an unrealistic \(P(y\mid x)\).
- Memorization and near-duplication of training records may occur.
- High statistical similarity does not imply clinically valid physiological structure.

Research on synthetic medical data must therefore evaluate marginal distributions, multivariate relationships, downstream utility, calibration, clinical plausibility, and privacy risk separately.

## 1.2 Progress and Limitations of Tabular Generative Models

medGAN proposed the generation of high-dimensional discrete patient records [8], and CTGAN used conditional training and mode-specific normalization for mixed-type tabular data [9]. TabDDPM provides a strong diffusion-based baseline for continuous and categorical variables [10]. In medical data generation, a direction has been established in which TSTR (Train on Synthetic, Test on Real), correlation structure, clinical consistency, and attack-based privacy evaluation matter more than simple comparisons of means.

However, because these classical models use high-capacity global neural networks, it is difficult to interpret by which mechanism a particular structural relationship was recovered. This study takes as its central question not merely a competition over absolute generative quality, but **which dependencies a depth-limited generator can express**.

## 1.3 Research on Quantum Tabular Data Generation

QGANs are a family of generative models that use a quantum circuit as the generator or the discriminator [11,12]. Recently, TabularQGAN evaluated quantum tabular data generation on MIMIC-III and Adult Census [13], and QTabGAN proposed a hybrid tabular generative architecture combining PQC outputs with a classical mapper [14]. Both studies demonstrate the feasibility of quantum tabular generation, but as of July 2026 the publicly verifiable versions are arXiv preprints, so they must be cited as distinct from peer-reviewed results.

Common problems in existing designs are as follows.

1. A dense angle encoder in which every latent is fed into every qubit
2. A dense decoder in which every quantum measurement is connected to every output feature
3. Non-identifiability between the circuit topology and the identity of the clinical features
4. An architecture in which the classical encoder and decoder have far more parameters than the quantum core
5. Circularity in evaluating the model with the same metric as the clinical correlation loss
6. Analyses that report performance only on an ideal simulator and do not separate measurement and circuit resources

This study removes these six problems at the level of both the model architecture and the experimental design.

---

# 2. Central Claim and Claim Boundaries

## 2.1 Central Claim

The central claim of this study is as follows.

> A graph-local quantum generator of depth \(L\) has a dependency receptive field limited by graph distance. If a clinical conditional-dependency graph is used as the circuit topology so that strong clinical relationships are placed within this receptive field, the model can recover held-out clinical dependencies more accurately than a circuit that has the same abstract graph structure and resources but a random assignment of clinical nodes.

## 2.2 What Is Not Claimed

The following are not claimed on the basis of this study's results alone.

- A universal quantum advantage from the perspective of computational complexity
- A quantum advantage that cannot be classically simulated
- Overall computational efficiency superiority based solely on a small parameter count
- Causality of the CDG edges
- Complete anonymity of the synthetic data
- Real clinical deployability or clinical safety
- Real-hardware superiority demonstrated by simulator results alone

## 2.3 Recommended Core Sentence for the Paper

> **We do not claim unconditional quantum advantage. We test whether aligning a shallow graph-local quantum generator with a clinically estimated conditional-dependency graph improves the recovery of unseen clinical relationships under matched topology, depth, parameter, and training conditions.**

---

# 3. Research Questions and Hypotheses

## 3.1 Confirmatory Research Question

### RQ1

Does the semantic alignment between clinical features and graph nodes contribute to the recovery of held-out clinical dependencies by a shallow quantum generator?

### Confirmatory Hypothesis H1

When the same graph isomorphism class, edge count, circuit depth, number of quantum parameters, local encoder, local decoder, critic, and training procedure are used,

\[
\mathrm{HDE}_{\mathrm{CDG}}
<
\mathrm{HDE}_{\mathrm{Permuted\text{-}CDG}}
\]

holds, where HDE is the held-out dependency error.

## 3.2 Exploratory Research Questions

### RQ2

What interaction do circuit depth and graph distance exhibit with respect to the dependency recovery error?

### H2

As the depth \(L\) increases, the dependency recovery error decreases first for feature pairs with graph distance \(d_G(u,v)\le 2L\). Pairs with \(d_G(u,v)>2L\) are difficult to recover in a shallow circuit.

### RQ3

Does the benefit of CDG-QGAN arise from the quantum circuit itself, or is it explained by the graph-local inductive bias alone?

### H3

We claim an additional benefit of the quantum core only if CDG-QGAN attains a lower held-out dependency error than a graph-local classical generator with the same receptive field and a comparable parameter count. If the two models are equivalent, the result is interpreted restrictively as a "benefit of CDG alignment."

### RQ4

Is entanglement in the circuit necessary?

### H4

The no-entanglement model is inferior to CDG-QGAN in recovering relationships at graph distance 1–2. If there is no difference, we judge that quantum entanglement did not substantively contribute to generative performance.

### RQ5

What is the effect of NISQ measurement and device noise on structure recovery?

### H5

Finite-shot sampling and calibrated noise increase the structure recovery error, and shot-noise-aware fine-tuning partially mitigates that degradation.

---

# 4. Core Theory: The Depth-Limited Receptive Field of a Graph-Local Quantum Generator

## 4.1 Notation

Let the set of clinical features and qubits be

\[
V=\{1,\ldots,n\},\qquad n=16
\]

The clinical conditional-dependency graph is

\[
G=(V,E)
\]

Feature \(x_u\) and qubit \(q_u\) are in one-to-one correspondence.

The outcome condition is

\[
y\in\{0,1\}
\]

where \(y=1\) denotes in-hospital death after the 24-hour landmark. Into each qubit we feed a local latent that is mutually independent given the condition \(y\).

\[
z_u\overset{iid}{\sim}\mathcal U(-1,1),
\qquad
z_u\perp z_v\mid y
\]

## 4.2 Circuit Block

In the Schrödinger picture, a single block operates in the following order.

```text
Local data encoding
→ CDG-edge RZZ entanglement
→ Non-commuting local mixing rotation
```

In equations, with the rightmost operator acting first, we set

\[
U^{(\ell)}
=
R^{(\ell)}_{\mathrm{mix}}
E^{(\ell)}_G
S^{(\ell)}_{\mathrm{enc}}
\]

The full circuit of depth \(L\) is

\[
U_{G,\Theta}(z,y)
=
U^{(L)}\cdots U^{(2)}U^{(1)}
\]

## 4.3 Local Encoding

The encoding angle of each qubit \(u\) uses only its own local latent \(z_u\) and the condition \(y\).

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

We do not use a dense encoder that feeds in the latents \(z_v\) of other nodes. The quantities \(a,b,c\) are restricted to fixed values or to a small number of feature-wise trainable scales and shifts.

## 4.4 CDG-Based Entanglement

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

Because `RZZ` gates commute with one another within the same layer, they can be treated as a logical graph entangling layer. The two-qubit depth on an actual device is reported separately, together with the edge-coloring and transpilation results.

## 4.5 Local Mixing

If the final entangling layer is placed immediately before the `Z` measurement, `RZZ` commutes with the `Z` and `ZZ` observables, so the final entangling parameters have no effect on the measured values. To avoid this, we place a non-commuting local rotation after the entanglement.

\[
R^{(\ell)}_{\mathrm{mix}}
=
\prod_{u\in V}
R_X(\theta^{x}_{\ell,u})
R_Y(\theta^{y}_{\ell,u})
\]

In this ordering, the phase information created by the entanglement passes through the local mixing and is reflected in the `Z`-basis populations.

## 4.6 Local Observables and Outputs

Each feature \(u\) reads only a single observable of its corresponding qubit.

\[
q_u(z,y)
=
\langle 0|
U_{G,\Theta}^{\dagger}(z,y)
Z_u
U_{G,\Theta}(z,y)
|0\rangle
\]

The final feature is produced by a feature-specific local head.

\[
\tilde x_u
=
h_u(q_u,y)
\]

`h_u` is implemented as a small MLP of width at most 8, or as a one-dimensional spline. It does not take the measured values of other qubits as input.

## 4.7 The Light-Cone Proposition

### Proposition 1: Local observable support

In the circuit above, if \(L\) `RZZ → local mixing` blocks are used, then in the Heisenberg picture the backward support of \(Z_u\) is contained within graph radius \(L\).

\[
\operatorname{supp}
\left(
U_{G,\Theta}^{\dagger}Z_uU_{G,\Theta}
\right)
\subseteq
N_L(u)
\]

where

\[
N_L(u)
=
\{w\in V:d_G(u,w)\le L\}
\]

### Intuition

- The final local mixing turns \(Z_u\) into a linear combination of \(X_u,Y_u,Z_u\).
- One `RZZ` layer can attach the \(Z_v\) of a neighboring node to the \(X_u\) or \(Y_u\) component, so it expands the support by at most one graph hop.
- The next local mixing turns the neighboring node's \(Z_v\) back into \(X_v,Y_v\) components, and the next `RZZ` layer expands the support by one more hop.
- Therefore the backward light cone grows by at most one graph hop per block.

A formal proof by induction using the conjugation of Pauli strings is given in the appendix. The general background — that in a finite-depth local circuit, correlations and information propagation are limited by distance and depth — connects to the Lieb–Robinson bound and to finite-depth circuit locality results [1,2].

## 4.8 Conditional Independence Corollary

### Corollary 1

Assume the following conditions.

1. The initial state is a product state.
2. The input encoding is local.
3. \(z_u\perp z_v\mid y\).
4. The output \(\tilde x_u\) uses only the local observable \(q_u\).
5. The graph distance satisfies \(d_G(u,v)>2L\).

Then the two backward light cones do not overlap, so

\[
\tilde x_u
\perp
\tilde x_v
\mid y
\]

holds.

Consequently, the conditional connected correlation is

\[
\operatorname{Cov}
(\tilde x_u,\tilde x_v\mid y)
=0
\]

### Interpretation

This result does not mean that every real medical relationship is exactly graph-local. Rather, it makes explicit the following **structural constraint**.

> For a shallow generator to express the conditional dependency of a given feature pair, that feature pair must be placed close enough on the graph that their backward light cones overlap.

The purpose of the CDG design is therefore to reduce the graph distance of strong clinical relationships.

## 4.9 Handling of the Global Condition

Because the condition \(y\) is fed into every local encoder and every head, even distant features can have **unconditional** correlations with \(y\) as a common cause. The theory and the main evaluation are therefore carried out in one of the following ways.

- Computing dependencies separately within each \(y=c\) group
- Computing partial correlations of the residuals after regressing out \(y\)

In the paper, the light-cone claim is always stated as "conditional on \(y\)."

---

# 5. Permutation Symmetry and Fixing Clinical Node Identity

## 5.1 The Problem with the Existing Architecture

If a dense angle encoder and a dense decoder are used, one can arbitrarily permute the qubit labels and jointly permute the encoder rows, the circuit parameters, the graph edges, and the decoder input order, and still obtain the same function family. In that case, the statement "the renal feature was assigned to a particular qubit" is not fixed in the model output.

## 5.2 The Resolution in the v2 Architecture

In this study we fix the following correspondence.

\[
\text{clinical feature }x_u
\longleftrightarrow
\text{local latent }z_u
\longleftrightarrow
\text{qubit }q_u
\longleftrightarrow
\text{local head }h_u
\]

Therefore, relabeling the graph nodes while relabeling the output heads accordingly is a mere renaming, whereas **keeping the feature–qubit correspondence fixed and moving only the graph edges onto clinically incorrect node pairs** yields a different model.

## 5.3 The Isomorphic Permuted-CDG Control

We fix the identities of the clinical features and the output heads, and permute only the graph adjacency.

\[
A_{\pi}
=
PAP^{\top}
\]

Here qubit \(u\) still generates feature \(x_u\). This control preserves the following completely.

- Number of nodes
- Number of edges
- Degree sequence
- Graph diameter
- Abstract graph isomorphism class
- Number of quantum gates
- Number of trainable quantum parameters
- Sizes of the local encoder and the local heads

The only thing that changes is which clinical feature pairs have short graph distances. The difference between CDG and permuted-CDG is therefore a **direct test of clinical semantic alignment**.

---

# 6. Data Design

## 6.1 Dataset

The primary dataset is **MIMIC-IV v3.1** [15,16]. We fix the version at the time of the study to v3.1, and state the data version and the SQL commit hash in both the paper and the code.

MIMIC-IV is de-identified data, but it is not unrestricted public data. We comply with PhysioNet credentialing, the required training, and the data use agreement, and we do not release the raw data or any patient-level derived data.

## 6.2 Primary Cohort: the 24-Hour Landmark Cohort

### Inclusion criteria

- Adults aged 18 years or older
- The patient's first ICU admission
- Survival to 24 hours after ICU admission
- ICU length of stay of at least 24 hours
- At least a prespecified number of observations among the core vital signs in the first 24 hours

### Observation window

\[
[t_{ICU},t_{ICU}+24h)
\]

### Outcome window

\[
[t_{ICU}+24h,t_{hospital\ discharge}]
\]

### Outcome label

\[
y
=
\mathbb I
\left[
\text{death after 24 hours and before hospital discharge}
\right]
\]

This design reduces the problem in which the missingness mask itself directly leaks the outcome label because of death or early discharge before 24 hours of admission. In exchange, the study population is restricted to "patients who survive to 24 hours after ICU admission and remain in the ICU," so this selection is stated explicitly as a limitation of the paper. In MIMIC clinical prediction benchmarks as well, the practice of excluding stays shorter than a fixed observation window has been used [17].

## 6.3 The 16 Core Features

In this first paper, we restrict ourselves to 16 continuous features in order to maintain the one-feature–one-qubit correspondence.

| No. | Clinical domain | Feature | 24-hour summary |
|---:|---|---|---|
| 1 | Circulation | Heart rate | Window mean |
| 2 | Circulation | Systolic blood pressure | Window mean |
| 3 | Circulation | Diastolic blood pressure | Window mean |
| 4 | Circulation | Mean arterial pressure | Window mean |
| 5 | Respiration | Respiratory rate | Window mean |
| 6 | Respiration | SpO₂ | Window mean |
| 7 | Temperature | Temperature | Window mean |
| 8 | Metabolism | Glucose | Window median |
| 9 | Electrolytes | Sodium | Window median |
| 10 | Electrolytes | Potassium | Window median |
| 11 | Electrolytes | Chloride | Window median |
| 12 | Acid–base | Bicarbonate | Window median |
| 13 | Renal | Creatinine | Window median |
| 14 | Renal | BUN | Window median |
| 15 | Hematology | Hemoglobin | Window median |
| 16 | Hematology | Platelet count | Window median |

### Prespecified replacement variables

If a particular variable shows an observation rate below the prespecified criterion in the training cohort, it is replaced, without inspecting the test set, in the following order.

1. White blood cell count
2. Hematocrit
3. Calcium

The final variable list is fixed before the main experiments and is not changed thereafter.

## 6.4 Consistency of the Blood-Pressure Summaries

The following approximate relationship is diagnosed only for synchronized means or for measurements taken at the same time point.

\[
MAP
\approx
\frac{SBP+2DBP}{3}
\]

It is not applied to combinations of maxima or minima.

\[
\max(MAP)
\neq
\frac{\max(SBP)+2\max(DBP)}{3}
\]

Because the 16 core features of this study use mean values for the blood-pressure variables, this relationship may be used **only as an evaluation metric**. It is not included in the main training loss.

## 6.5 Data Splits

The data are split by `subject_id` so that the same patient does not appear in more than one split.

\[
\mathcal D
=
\mathcal D_{train}
\cup
\mathcal D_{validation}
\cup
\mathcal D_{test}
\]

The recommended ratio is as follows.

\[
70\%:15\%:15\%
\]

The split is stratified by mortality status, and the following are determined on the training set only.

- Unit conversion rules
- Outlier handling boundaries
- Scaling statistics
- Imputation model
- CDG estimation
- CDG edge split
- Hyperparameter and checkpoint selection

The test set is not used for model or graph selection until the final evaluation.

## 6.6 Preprocessing

### Units and error handling

- Units of the same clinical variable are harmonized.
- Physically impossible values are treated as missing.
- Clinically possible extreme values are not automatically removed.
- Error boundaries are fixed in advance with reference to the public concept definitions of the MIMIC Code Repository and to the clinical literature.

### Scaling

Each feature is robust-scaled using the median and IQR of the training set.

\[
\tilde x_j
=
\frac{x_j-\operatorname{median}_{train}(x_j)}
{\operatorname{IQR}_{train}(x_j)+\epsilon}
\]

The final generator output uses `tanh` or a bounded spline, and evaluation is performed in the original units after inverse transformation.

## 6.7 Handling of Missingness

Because the primary mechanism experiment of this study focuses on the dependency structure of continuous values, the missingness mask is not included in the main output of the quantum generator.

### Primary analysis

1. Compute the patient-level observation rate of each feature.
2. Fix the 16 features that satisfy the prespecified observation-rate criterion.
3. Fit an iterative imputation model on the training set.
4. Apply the same model to the validation and test sets.
5. Retain the pre-imputation mask separately, but do not use it as a generator input or in the primary endpoint.

### Sensitivity analyses

- Complete-case subset
- Training median imputation
- CDG stability across multiple imputations
- Value-only TSTR
- Value + original missingness indicator TSTR

If value+mask performance is markedly higher than value-only performance, we report the possibility that utility depends on measurement practice and informative missingness rather than on physiological values.

---

# 7. Construction of the Clinical Dependency Graph

## 7.1 Purpose of the CDG

The CDG is not a knowledge graph that arbitrarily connects "variables that appear clinically related." The CDG of this study is a **conditional statistical dependency structure** estimated from the training data, and it is used to place important relationships inside the circuit's limited receptive field.

## 7.2 Conditioning Variables and Covariate Adjustment

Each feature is regressed on the following covariates in the training set.

- The outcome condition \(y\)
- Age
- Sex
- ICU type

We obtain the residuals of an appropriate regression model using continuous or categorical covariates.

\[
r_j
=
x_j-\widehat{\mathbb E}[x_j\mid y,\text{age},\text{sex},\text{ICU type}]
\]

The CDG estimates the conditional relationships among the residuals \(r_j\). This reduces the problem in which all features become connected simply because of mean differences between the deceased and surviving groups.

## 7.3 Nonparanormal Transformation

A rank-based inverse normal transformation is applied to each residual feature [4].

\[
\hat r_j
=
\Phi^{-1}
\left(
\frac{\operatorname{rank}(r_j)-0.5}{N}
\right)
\]

This is a procedure for approximating a Gaussian copula graphical model for continuous clinical variables with non-normal marginal distributions.

## 7.4 Sparse Precision Matrix

The precision matrix \(\Omega=\Sigma^{-1}\) is estimated with the graphical lasso [3].

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

The partial correlation is

\[
\rho_{uv\mid V\setminus\{u,v\}}
=
-
\frac{\Omega_{uv}}
{\sqrt{\Omega_{uu}\Omega_{vv}}}
\]

The edge weight is defined as

\[
w_{uv}
=
|\rho_{uv\mid V\setminus\{u,v\}}|
\]

## 7.5 Selection of the Regularization Coefficient and Stability

The graphical lasso \(\lambda\) is chosen in advance by one of the following.

- Stability selection based on StARS [5]
- Extended BIC

Bootstrap or subsampling is repeated to compute the selection frequency of each edge.

\[
s_{uv}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbb I[(u,v)\in E^{(b)}]
\]

The default stability threshold is set to

\[
s_{uv}\ge0.70
\]

and 0.60 and 0.80 are used in sensitivity analyses.

## 7.6 Graph Resource Constraints

Because high-degree nodes can substantially increase the two-qubit depth and the noise on a real device, we apply the following constraints.

- Number of nodes: 16
- Maximum degree: \(\Delta\le3\) recommended
- Maintain a connected graph
- Number of edges: a prespecified \(m\)
- Prioritize clinically strong and stable edges

If the graph becomes disconnected, we add the highest-weight connecting edges, within the constraint of not using any held-out edge. All added edges and the reasons for adding them are disclosed.

## 7.7 Separation of Fit Edges and Held-Out Dependency Pairs

To avoid circular evaluation, the stable candidate relationships are separated as follows.

\[
E_{candidate}
=
E_{fit}\cup E_{holdout},
\qquad
E_{fit}\cap E_{holdout}=\varnothing
\]

- \(E_{fit}\): used only to construct the CDG topology
- \(E_{holdout}\): not used for topology, loss, hyperparameter selection, or checkpoint selection

The default ratio is

\[
70\%:30\%
\]

The split is stratified by relationship strength and clinical domain, and uses a fixed seed.

The actual target values for `E_holdout` are recomputed on the test cohort. That is, even if a relationship was identified as strong in the training set, the final evaluation is based on whether that relationship holds in independent test patients.

## 7.8 Comparison Graphs

### Isomorphic permuted-CDG

Only the node labels of the same adjacency are randomly permuted, several times. This is the confirmatory control for clinical semantic alignment.

### Degree-preserving rewired graph

Double-edge swaps preserve the degree sequence and edge count while changing the path structure. This evaluates the effect of the abstract graph structure itself.

### Ring / ring-with-chords

This is the uniform-topology baseline that is common in the circuit literature. When comparing resources, fixed chords are added if necessary to match the edge count and the two-qubit gate count.

### No-entanglement graph

Setting

\[
E=\varnothing
\]

yields a quantum model that generates only local marginals.

---

# 8. The CDG-QGAN Model

## 8.1 Overall Architecture

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

## 8.2 Local Latents

\[
z
=(z_1,\ldots,z_{16}),
\qquad
z_u\sim\mathcal U(-1,1)
\]

Each \(z_u\) is the local stochastic source of the corresponding feature. We do not use an architecture in which a global latent is connected to every qubit.

## 8.3 Data Re-uploading

The same local latent may be re-entered at each layer.

\[
S^{(\ell)}_u
=
R_Y(a_{\ell,u}z_u+b_{\ell,u}y+c_{\ell,u})
R_Z(d_{\ell,u}z_u+e_{\ell,u}y+f_{\ell,u})
\]

However, to prevent the encoder from becoming a larger neural network than the quantum core, we use one of the following.

### Default option

- Learn only feature-wise affine scales and shifts
- At most 4–6 parameters per qubit per layer

### Strict control option

- Use fixed scales
- Map \(z_u\) directly onto \([-\pi,\pi]\)
- No trainable encoder parameters

By comparing the two settings, we check whether a trainable angle encoder is what actually accounts for the performance.

## 8.4 Entangling Gate

A trainable `RZZ` is applied to each CDG edge.

\[
E_G^{(\ell)}
=
\prod_{(u,v)\in E_{fit}}
\exp
\left(
-i\frac{\gamma_{\ell,uv}}{2}Z_uZ_v
\right)
\]

Initial values are set within a small range around 0.

\[
\gamma_{\ell,uv}
\sim
\mathcal U(-\epsilon,\epsilon)
\]

Experiments that encode the CDG weights into the magnitude of the initial values are restricted to the exploratory setting. In the primary comparison, the initialization distributions of CDG and permuted-CDG are kept identical.

## 8.5 Local Mixing Rotation

\[
R_{mix}^{(\ell)}
=
\prod_{u=1}^{16}
R_X(\theta^x_{\ell,u})
R_Y(\theta^y_{\ell,u})
\]

Local mixing is placed after the entanglement so that `RZZ` affects the final `Z` expectation values.

## 8.6 Measurement

The measurements passed to the generator are limited to a single `Z` expectation per qubit.

\[
q_u=\langle Z_u\rangle
\]

The pairwise \(\langle Z_uZ_v\rangle\) may be used for diagnostics and noise analysis, but it is not used as a decoder input. Feeding it into the decoder would widen the receptive field of the output and change the theoretical bound.

## 8.7 Feature-Specific Local Head

\[
\tilde x_u
=h_u([q_u,y])
\]

The recommended default architecture is as follows.

```text
Input(q_u, y)
→ Linear(2, 8)
→ SiLU or Tanh
→ Linear(8, 1)
```

No head can see the \(q_v\) of another feature. Heads may be shared, but mixing of inputs is not permitted.

## 8.8 Conditional Generation

During training we use the \(y\) of the actual minibatch, and at generation time we sample \(y\) in one of the following ways.

- The actual training prevalence
- Balanced generation followed by resampling to the target prevalence

For the primary fidelity evaluation we use a synthetic sample matched to the actual test prevalence. For TSTR we report both class-balanced synthetic training and prevalence-matched synthetic training.

## 8.9 Critic

The critic is a global network that takes the full clinical vector and the condition as input.

\[
D_\psi(x,y)
\]

Recommended architecture:

```text
Input(16 features + y)
→ Linear(128) + LeakyReLU
→ Linear(128) + LeakyReLU
→ Linear(64)  + LeakyReLU
→ Scalar
```

It is permissible for the critic to evaluate global structure. The critic provides a training signal, but because it does not directly mix features within a sample at generation time, it does not change the structural receptive field of the generator.

## 8.10 Training Loss

The primary model uses only a conditional WGAN-GP [18].

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

### Losses excluded from the primary model

- CDG edge correlation loss
- Dependency matrix loss
- MAP constraint loss
- Physiological range loss
- Auxiliary mortality classifier loss
- Missing-mask correlation loss

These terms are removed from the main model because they either directly optimize the evaluation metric or can exaggerate the separation between conditions. Clinical constraint losses are used only in a subsequent exploratory ablation.

## 8.11 Default Hyperparameters

| Item | Default value or search range |
|---|---|
| Number of qubits | Fixed at 16 |
| Logical CDG block depth | \(L\in\{1,2\}\), \(L=3\) exploratory |
| Batch size | 64 |
| Critic updates | 5 per generator update |
| Optimizer | Adam |
| Critic learning rate | \(1\times10^{-4}\) |
| Generator learning rate | \(5\times10^{-5}\) or \(1\times10^{-4}\) |
| Adam \((\beta_1,\beta_2)\) | \((0.0,0.9)\) |
| Gradient penalty | \(\lambda_{gp}=10\) |
| Early stopping | Based on the validation two-sample score and loss stability |
| Number of primary seeds | 10 in the final setting |
| Number of exploratory seeds | 5 |

We do not use the number of epochs as a fixed performance criterion; instead, every model is given the same maximum critic-update budget.

## 8.12 Parameter Breakdown Reporting

For every model, the following are reported separately.

| Component | Parameter count |
|---|---:|
| Local angle encoding | Reported separately |
| Quantum single-qubit rotations | Reported separately |
| Quantum entangling angles | Reported separately |
| Local output heads | Reported separately |
| Generator total | Reported separately |
| Critic | Reported separately |

The share of quantum parameters is disclosed as

\[
r_Q
=
\frac{N_{quantum}}
{N_{generator,total}}
\]

Parameter count and computational cost are reported separately. Circuit simulation cost, the number of circuit evaluations, memory, and wall-clock time are also reported.

---

# 9. Comparison Models

## 9.1 Confirmatory Controls

### 1. CDG-QGAN

The proposed model, in which the clinical features and the CDG nodes are correctly aligned.

### 2. Isomorphic permuted-CDG-QGAN

A model that keeps the same graph structure and resources while shuffling only the clinical feature–node correspondence. This is the **most important control**.

## 9.2 Mechanism Controls

### 3. Degree-preserving rewired-QGAN

A circuit with the same degree sequence and edge count but a different path structure.

### 4. No-entanglement QGAN

The local encoding, local mixing, and local heads are identical, but `RZZ` is removed.

### 5. Ring-QGAN or resource-matched ring-with-chords

A generic uniform-topology baseline. We primarily report the results with the edge count and gate budget matched.

## 9.3 Classical Locality Control

### 6. CDG-local classical message-passing generator

Uses the same CDG and the same local latents.

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

The number of message-passing rounds is matched to the logical depth \(L\) of the quantum circuit, and the generator parameter count is matched as closely as possible. If this model is equivalent to CDG-QGAN, the observed benefit is interpreted as a graph-local inductive bias rather than as quantumness.

### 7. Global MLP-WGAN

Uses the same critic and WGAN-GP loss but a global MLP generator. It is used to compare absolute generative quality, but not as key evidence of a quantum contribution.

## 9.4 Nonlinear Feature-Map Control

### 8. Random Fourier Feature generator

Uses a fixed random Fourier feature map instead of a quantum circuit [19]. By matching the output dimension and the local heads, we check whether the performance is explained by a simple nonlinear feature map rather than by a "trainable quantum circuit."

## 9.5 High-Performance Context Baselines

### 9. TabDDPM

A modern classical tabular diffusion baseline. We compare overall fidelity and TSTR utility, but do not include it in the test of the graph-local mechanism.

### Optional additional models

CTGAN, TVAE, and Gaussian Copula are included only as context numbers when computational resources permit. Superiority over these models is not interpreted as evidence for the quantum circuit.

## 9.6 Treatment of Clifford and Matchgate Controls

Clifford and matchgate circuits are interesting controls with respect to classical simulability, but if arbitrary-angle continuous data encoding is retained, Clifford simulability breaks down, and matchgates carry separate constraints on the gates and the connectivity structure. We therefore run them as appendix experiments only if a configuration that permits a fully fair comparison can be secured. The core comparison groups remain the graph-local classical generator and the RFF control.

---

# 10. Controlled Synthetic Graph Experiments

Because the true graph is unknown in real clinical data, we first run a controlled benchmark to validate the theory and the implementation.

## 10.1 Teacher Distribution

We generate a 16-dimensional Gaussian copula or nonparanormal distribution.

1. Define a known sparse precision matrix \(\Omega^*\).
2. Sample Gaussian latents from \(\Sigma^*=(\Omega^*)^{-1}\).
3. Apply a different nonlinear monotone transformation to each dimension to produce non-normal marginal distributions.

\[
g
\sim
\mathcal N(0,\Sigma^*)
\]

\[
x_u
=T_u(g_u)
\]

## 10.2 Types of Teacher Graph

- Chain graph
- Modular graph
- Sparse degree-balanced graph

For each graph we compare the aligned, permuted, rewired, ring, and no-entanglement models as well as the classical local generator.

## 10.3 Objectives

- Confirm that the light-cone implementation agrees with the theory
- Confirm that the recoverable graph distance increases with depth \(L\)
- Confirm that dependencies are generated even without a dense decoder
- Unit-test that the gradient of the final `RZZ` becomes 0 when the circuit ordering is wrong
- Confirm that the CDG alignment effect appears under conditions where the ground-truth graph is known

## 10.4 Key Figure

```text
x-axis: teacher graph distance d(u,v)
y-axis: conditional dependency restoration error
curves: L=1, L=2, L=3
models: aligned / permuted / rewired / classical-local
```

This figure directly exhibits the mechanism by which the dependency receptive field widens as the depth increases.

---

# 11. MIMIC-IV Experimental Design

## 11.1 Primary Experiment

- Data: the 24-hour landmark cohort
- Features: 16 continuous variables
- Condition: in-hospital death after 24 hours
- Circuit depth: \(L=1,2\)
- Comparison: CDG-QGAN vs isomorphic permuted-CDG-QGAN
- Loss: identical conditional WGAN-GP
- Generator architecture: identical local encoder, local observable, and local head
- Confirmatory endpoint: a single held-out conditional dependency error

## 11.2 Permutation Repetitions

To avoid the chance outcome of a single permutation, we use several prespecified permutations.

- Default: 3 isomorphic permutations
- Per permutation: 10 seeds in the final setting, or 5 seeds depending on computational constraints
- CDG-QGAN: the same set of seeds

The primary contrast is defined as the difference between the HDE of CDG-QGAN and the mean HDE over permutations.

## 11.3 Depth Experiments

### Main depths

\[
L\in\{1,2\}
\]

### Exploratory depth

\[
L=3
\]

Because at larger depths the light cones of most feature pairs overlap and the topology difference can weaken, we take the shallow depths as the main condition.

## 11.4 Graph-Distance Stratification

Each pair is stratified by CDG distance.

- \(d=1\)
- \(d=2\)
- \(d=3\)
- \(d\ge4\)

At each depth we report the dependency recovery error by distance. For the permuted graph we likewise compute the permuted graph distance of the same feature pairs.

## 11.5 Matched-Resource Conditions

CDG and permuted-CDG use exactly the same resources. For the rewired and ring comparisons we report each of the following.

- Same edge count
- Same trainable quantum parameter count
- Same logical block depth
- Transpiled two-qubit gate count
- Transpiled two-qubit depth

We do not use a single definition of "fairness"; instead we report the logical structure and the actual device resources separately.

---

# 12. Evaluation Metrics

## 12.1 The Single Confirmatory Primary Endpoint

### Held-out Dependency Error, HDE

For each outcome condition \(y=c\), we compute the partial correlations of the real test data and of the synthetic data.

\[
\rho_{uv,real}^{(c)}
,
\qquad
\rho_{uv,syn}^{(c)}
\]

We apply the Fisher \(z\)-transform.

\[
z(\rho)
=
\operatorname{atanh}(\rho)
\]

The held-out error for each condition is

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

The overall HDE is weighted by the test prevalence.

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

To support the research hypothesis, we require

\[
\Delta_{HDE}<0
\]

and the 95% confidence interval must not include 0.

## 12.2 Dependency Recovery Error by Distance

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

We analyze the interaction between depth and distance.

## 12.3 Overall Dependency Structure

Secondary metrics:

- Frobenius error of the full partial-correlation matrix
- Spearman correlation matrix error
- Precision matrix edge recovery
- Top-\(k\) dependency overlap
- Conditional mutual information error

\[
E_{F}
=
\|R_{real}-R_{syn}\|_F
\]

## 12.4 Marginal Fidelity

For each feature we report the following.

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

## 12.5 Multivariate Fidelity

- Maximum Mean Discrepancy
- Energy distance
- Real-vs-Synthetic classifier AUROC
- Precision/Recall for distributions
- Density and coverage

A Real-vs-Synthetic AUROC close to 0.5 is only an auxiliary indicator; it does not imply clinical validity or a privacy guarantee.

## 12.6 Machine Learning Utility

### TRTR

\[
\text{Train Real, Test Real}
\]

### TSTR

\[
\text{Train Synthetic, Test Real}
\]

### Evaluation models

- Logistic Regression
- Random Forest
- XGBoost
- A small MLP

### Evaluation metrics

- AUROC
- AUPRC
- Brier score
- Expected Calibration Error
- Sensitivity at fixed specificity

We do not automatically declare success when TSTR exceeds TRTR. The following are analyzed together.

- The classifier score distributions of synthetic and real data
- The calibration curve
- Class-conditional feature overlap
- Whether \(P(y\mid x)\) is excessively separated

## 12.7 Clinical Plausibility

Clinical constraints are used as diagnostic metrics, not as training losses.

### Blood-pressure ordering

\[
DBP\le MAP\le SBP
\]

### Mean MAP approximation

\[
MAP\approx\frac{SBP+2DBP}{3}
\]

The goal is not to drive the violation rate of the synthetic data to zero. We evaluate the difference from the real test data.

\[
E_{violation}
=
\left|
VR_{syn}-VR_{real}
\right|
\]

We also compare the distribution of violation severity.

\[
D_{severity}
=
D
\left(
P_{severity}^{syn},
P_{severity}^{real}
\right)
\]

## 12.8 Diversity and Memorization

- Synthetic exact duplicate rate
- Synthetic-to-train nearest-neighbor distance
- Synthetic-to-test nearest-neighbor distance
- Pairwise distance distribution
- Coverage

High uniqueness alone is not taken as evidence of a good model.

## 12.9 Exploratory Privacy Audit

Privacy is excluded from the core novelty and the title of the first paper. Nevertheless, a minimal empirical audit can be performed, given that the data are synthetic medical data.

### Recommended analyses

- DOMIAS or density-based membership inference [20]
- Nearest-neighbor membership score
- Attribute inference

### Fidelity-matched comparison

To avoid the confounding in which a poorly trained model automatically appears safe, we compute the following pair at several checkpoints.

\[
(
\text{fidelity},
\text{attack advantage}
)
\]

Privacy across models is presented as a privacy–fidelity Pareto frontier rather than compared by a single MIA AUC.

Formal differential privacy is excluded from the first paper and separated into follow-up work.

---

# 13. NISQ and Measurement Experiments

## 13.1 Principle of Scope Reduction

We do not train every combination of model, depth, shot count, and noise. We restrict ourselves in the following order.

1. Model selection under ideal analytic simulation
2. Selection of the final checkpoints of CDG and the core controls
3. Finite-shot evaluation on the selected models only
4. Shot-noise-aware fine-tuning on a subset of the selected models
5. Evaluation with one or a few calibrated noise models
6. Real-device usage restricted, where possible, to small-scale inference validation

## 13.2 Finite-Shot Conditions

\[
S\in\{256,1024,4096\}
\]

Under each shot condition we measure the change in HDE, in the overall dependency error, and in marginal fidelity.

## 13.3 Advantage of a Common Measurement Basis

All \(Z_i\) and the diagnostic \(Z_iZ_j\) are diagonal in the computational basis. Therefore, from a single batch of Z-basis bitstrings, we can simultaneously estimate

\[
\langle Z_i\rangle
\]

\[
\langle Z_iZ_j\rangle
\]

Even as the number of measured observables grows, the number of separate basis settings does not grow with the number of edges. However, the estimator covariance and the accuracy requirements for multiple observables under finite shots are analyzed separately.

## 13.4 Shot-Noise-Aware Training

The shot estimator of a Pauli observable \(O_a\) approximately has variance

\[
\operatorname{Var}(\hat O_a)
=
\frac{1-\mu_a^2}{S}
\]

Observables estimated from the same bitstrings have correlated noise.

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

We therefore use the following correlated surrogate instead of simple independent \(1/\sqrt S\) Gaussian noise.

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

This noise is injected into the local expectation vector and a short fine-tuning is performed. Exact finite-shot parameter-shift training is excluded from the core experiments because the number of circuit executions is prohibitively large.

## 13.5 Noise Models

- Ideal statevector
- Finite-shot noiseless sampling
- Backend-calibrated depolarizing/readout noise
- Before and after readout mitigation

The calibration date and backend information of the noise model are fixed and reported. We do not select only the best result among several backends.

## 13.6 Trainability Analysis

We do not assert that a gradient decay observed at 4–16 qubits is a `barren plateau`. We use the following terms and metrics.

- Gradient norm
- Gradient variance
- Gradient signal-to-noise ratio
- An approximation of the Hessian or empirical Fisher conditioning
- Convergence rate
- Seed sensitivity
- Plateau duration

The term barren plateau is used only when exponential scaling with the number of qubits has been sufficiently verified [21].

---

# 14. Implementation

## 14.1 Primary Simulator

Primary training is implemented with a PyTorch-based batched statevector simulator.

A 16-qubit statevector has

\[
2^{16}=65,536
\]

complex amplitudes per sample. With a batch of 64, the raw `complex64` state tensor is about 32 MiB, but once autograd intermediates and optimizer state are included the actual memory can be far larger, so we use gradient checkpointing and custom gate kernels.

## 14.2 Gate Implementation

- `RY`, `RZ`, `RX`: reshape along the target axis, then a \(2\times2\) matrix multiplication
- `RZZ`: diagonal phase multiplication using the parity of the basis index
- Expectation \(\langle Z_u\rangle\): a sign-weighted sum of the basis probabilities
- Batched latents: a `(batch, 2**n)` complex tensor

Because `RZZ` is a diagonal gate, it is implemented as a phase multiplication rather than as a dense \(4\times4\) matrix product.

## 14.3 Differentiation

In ideal simulation we use PyTorch reverse-mode autograd. Parameter-shift is used only for real devices or for cross-validation across frameworks.

## 14.4 Correctness Verification

We compare the results of the custom simulator against small circuits in PennyLane or Qiskit.

### Unit tests

- State fidelity of random 2–6 qubit circuits
- Agreement of \(Z\) expectation values
- Agreement between PyTorch gradients and parameter-shift gradients
- Verification that \(\partial\langle Z\rangle/\partial\gamma=0\) when `RZZ` sits immediately before the measurement
- Verification that the gradient is nonzero in the `RZZ → RX/RY → Z` ordering
- Permutation equivariance test
- Locality/light-cone support test

## 14.5 The Role of PennyLane and Qiskit

- Custom simulator correctness verification
- Circuit transpilation
- Hardware topology mapping
- Calibrated noise simulation
- Real backend inference

Whether PennyLane or Qiskit is used as the primary training path is decided after a wall-clock benchmark on the same circuit. We do not assume in advance that any particular framework is always slower or faster.

## 14.6 Reproducible Experiment Configuration

All experiments are managed through configuration files.

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

# 15. Training Algorithm

## 15.1 CDG-QGAN Training Pseudocode

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

## 15.2 Checkpoint Selection

The primary endpoint, the held-out dependency error, is not used for validation checkpoint selection. The recommended validation rule is the following combination.

- Validation real-vs-synthetic classifier AUROC
- Mean univariate Wasserstein distance
- Stability of the critic/generator losses
- Mode collapse indicator

The held-out dependency pairs are used only in the final test.

---

# 16. Ablation Study

## 16.1 Required Ablations

| ID | Experiment | Purpose |
|---|---|---|
| A1 | Isomorphic permuted-CDG | Effect of clinical feature–node alignment |
| A2 | Degree-preserving rewiring | Effect of abstract topology and path structure |
| A3 | No entanglement | Necessity of entanglement |
| A4 | Ring/resource-matched ring | Comparison with a generic circuit topology |
| A5 | Graph-local classical generator | Separating the quantum effect from the locality effect |
| A6 | Fixed angle encoding | Isolating the contribution of the trainable angle encoder |
| A7 | Random Fourier feature control | Comparison with a generic nonlinear feature map |
| A8 | \(L=1\) vs \(L=2\) | Effect of widening the light cone |

## 16.2 Optional Ablations

| ID | Experiment | Purpose |
|---|---|---|
| B1 | Local head width 1/4/8/16 | Influence of classical head capacity |
| B2 | \(L=3\) | Whether the topology difference disappears |
| B3 | Different graph stability thresholds | Sensitivity of the CDG estimate |
| B4 | Median vs iterative imputation | Influence of missingness handling |
| B5 | Shot-noise-aware fine-tuning | Robustness under finite shots |
| B6 | Exploratory clinical constraint loss | Separating the structural prior from the loss prior |

## 16.3 Excessive Grids That Are Excluded

We do not perform the full Cartesian grid below.

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

The qubit count is fixed at 16, the depth is concentrated on 1–2, and shots/noise are applied only to the final models.

---

# 17. Statistical Analysis

## 17.1 A Single Primary Hypothesis

The confirmatory test is restricted to the following single test.

\[
H_0:
\Delta_{HDE}\ge0
\]

\[
H_1:
\Delta_{HDE}<0
\]

## 17.2 Uncertainty Estimation

The following sources of variability are accounted for jointly.

- Training seed
- Permutation instance
- Synthetic sample draw
- Test patient resampling

The recommended method is a hierarchical bootstrap.

1. Resample model seeds with replacement
2. Resample permutations with replacement
3. Resample test patients with replacement
4. Re-sample a synthetic cohort of the same size
5. Compute \(\Delta_{HDE}\)

We report the 95% confidence interval and the effect size.

## 17.3 Secondary Analyses

The relationship among distance, depth, and model can be analyzed with a mixed-effects model.

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

where \(s\) is the training seed.

For multiple comparisons among the secondary metrics we apply the Benjamini–Hochberg FDR or a Holm correction, while explicitly labeling the results as exploratory.

## 17.4 Reporting Principles

- Release the raw per-seed results, not only means and standard deviations
- 95% confidence intervals
- Absolute and relative differences
- Effect size
- Include failed seeds and mode collapse
- Do not present only the best-performing seed

---

# 18. Success, Failure, and Falsification Criteria

## 18.1 Support for the Central Hypothesis

We support the clinical semantic alignment effect when all of the following are satisfied.

1. \(\Delta_{HDE}<0\)
2. The 95% CI does not include 0
3. The same direction holds in the controlled synthetic benchmark
4. The result does not depend on a single specific permutation
5. The result is not obtained together with a markedly degraded marginal fidelity

## 18.2 Additional Contribution of the Quantum Core

We claim an "additional benefit of the quantum core," in a limited sense, only when the following are satisfied.

- CDG-QGAN attains a lower HDE than the CDG-local classical generator
- The total generator parameter budgets are comparable
- The receptive fields are identical
- The same critic, loss, data split, and update budget are used
- The result holds across multiple seeds

If they are equivalent, the conclusion is restricted to the following.

> The data-dependent graph-local inductive bias was effective, but we obtained no evidence that, in this setting, the quantum circuit provides an additional benefit over matched classical message passing.

## 18.3 Rejection of the Central Hypothesis

If there is no difference between CDG and permuted-CDG, we acknowledge the following.

- Clinical node alignment did not contribute to performance
- At the chosen depth, the topology may not have been a sufficient bottleneck
- The CDG estimate may have been unstable
- The critic and the local heads may have compensated for the circuit difference during training
- The true conditional dependency structure may not match the chosen graph-local assumption

In that case we do not substitute the central claim with a result showing that some metric is better than CTGAN or TabDDPM.

## 18.4 The Case of Equivalence with No-Entanglement

If the model is equivalent to the entanglement-free model, we do not claim a contribution from quantum entanglement. We report that result as evidence that the performance is explained by a local nonlinear feature map alone.

---

# 19. Expected Form of the Results

This study does not prespecify particular numerical improvements. The **form** of the expected results is as follows.

## 19.1 Theory Validation Figure

- \(L=1\): dependency recovery only for pairs at short graph distance
- \(L=2\): a wider range of recoverable distances
- The possibility that the gap between aligned and permuted shrinks as depth increases

## 19.2 Clinical Result Figures

- Real vs synthetic partial-correlation scatter for each held-out dependency pair
- The HDE distributions of CDG and permuted-CDG
- Error curves by graph distance
- Dependency restoration heatmaps by depth

## 19.3 Quantum–Classical Comparison

- Pareto plot of CDG-QGAN vs the graph-local classical generator
- x-axis: generator parameters or runtime
- y-axis: HDE or full dependency error

## 19.4 NISQ Results

- Change in HDE with the number of shots
- Before and after noise-aware fine-tuning
- Relationship between transpiled two-qubit depth and performance degradation

---

# 20. Feasibility

## 20.1 Core Scope for a Single Author

### Must be carried out

- The 16-feature MIMIC-IV landmark cohort
- CDG estimation and the held-out edge split
- The custom batched statevector simulator
- CDG-QGAN
- Isomorphic permuted-CDG
- The no-entanglement model
- The graph-local classical generator
- The controlled synthetic graph benchmark
- The HDE primary endpoint
- The distance–depth analysis
- A minimal set of marginal fidelity and TSTR utility results

### To be carried out if computational resources permit

- Degree-preserving rewiring
- The RFF control
- Ring-QGAN
- TabDDPM
- Finite-shot evaluation and one calibrated noise model
- The privacy–fidelity Pareto audit

### Excluded from the first paper

- Using community compression of 30–45 features as the main model
- Full longitudinal EHR generation
- Formal differential privacy
- Large-scale training on multiple real quantum backends
- Concurrent development of quantum diffusion
- Combination with medical imaging or clinical notes
- Setting multi-center external validation as a core requirement

## 20.2 Scalability Experiments

An architecture that compresses 30–45 features into communities is treated not as the main model of the first paper but as a follow-up scalability experiment. Even when scaling up, the architecture must be arranged so that each community-specific output head reads only its corresponding qubit and adjacent pair observables.

---

# 21. Research Risks and Mitigations

## 21.1 Instability of the CDG Estimate

### Risk

Because of missingness, non-normality, and a common severity factor among MIMIC variables, the partial-correlation graph may be unstable.

### Mitigation

- Nonparanormal transform
- Residualization
- Stability selection
- Checking edge consistency across multiple imputations
- Graph threshold sensitivity analysis
- Public release of per-edge bootstrap stability

## 21.2 Degraded Absolute Generative Quality of the Local Generator

### Risk

Because of the strong locality constraint, overall fidelity may be lower than that of a global MLP or TabDDPM.

### Mitigation

The core of this study is a test of a structural mechanism, not absolute SOTA. However, if an HDE improvement comes together with a serious loss of marginal fidelity, we do not interpret it as a success. We present the fidelity–structure trade-off alongside the results.

## 21.3 Condition Leakage and Over-Separation

### Risk

Conditional generation may separate the deceased and surviving groups more sharply than in reality.

### Mitigation

- Prohibit the use of an auxiliary condition loss
- Evaluate both the real prevalence and balanced generation
- Check the Brier score, ECE, and score distributions
- Class-conditional two-sample tests

## 21.4 Missingness and Landmark Selection

### Risk

The landmark cohort excludes patients who die before 24 hours and those discharged early, which limits the scope of generalization.

### Mitigation

- Describe the study population with clearly stated restrictions
- Secondary descriptive analysis of the full ICU cohort
- Separate value-only and value+mask utility
- Consider variable-observation-window models in follow-up work

## 21.5 Custom Simulator Errors

### Risk

Subtle errors in a hand-implemented quantum simulator could distort all of the results.

### Mitigation

- Cross-validation against PennyLane/Qiskit
- State fidelity and gradient unit tests
- Exhaustive tests on small circuits
- Tests of circuit ordering and qubit indexing
- Public code and continuous integration

## 21.6 Computational Cost

### Risk

The autograd memory of a 16-qubit batched statevector can become large.

### Mitigation

- `complex64`
- Gradient checkpointing
- RZZ diagonal kernel
- Smaller batch + gradient accumulation
- Mixed precision for the local heads and the critic
- Microbenchmarks before fixing the final settings

---

# 22. Ethics, Privacy, and Reproducibility

## 22.1 Data Use

- Compliance with the PhysioNet DUA
- Non-disclosure of the raw data and of patient-level derived data
- Prohibition of uploads to unapproved external repositories
- Execution only in environments with data access credentials

## 22.2 Release of Synthetic Data

Synthetic data are not automatically assumed to be safe. Before release, the following are reviewed.

- Exact/near duplicates
- Membership inference
- Rare subgroup disclosure risk
- The PhysioNet/MIMIC terms of use

If safety is uncertain, we release only the generation code and the evaluation results instead of the synthetic data files.

## 22.3 Clinical Interpretation

Since this is single-author work without clinician review, we use the following expressions.

- `statistical clinical consistency`
- `physiological plausibility`
- `dependency preservation`

We avoid the following expressions.

- `clinically validated`
- `safe for clinical use`
- `clinically equivalent to real patients`

## 22.4 Public Artifacts

- Cohort SQL
- Feature dictionary
- Unit conversion rules
- Preprocessing code
- CDG estimation code
- Graph split seed
- Quantum simulator
- Model configs
- Evaluation code
- Per-seed results
- Transpilation/noise settings
- Environment lock file

---

# 23. Paper Outline

## 23.1 Introduction

1. The structure-preservation problem in synthetic medical data
2. Limitations of the dense encoder/decoder in existing quantum tabular generation
3. The depth-limited dependency receptive field of a shallow graph-local circuit
4. Problem definition: aligning a clinical dependency graph with the circuit topology
5. Summary of contributions

## 23.2 Theory

- Definition of a graph-local generator
- The `RZZ → local mixing` circuit structure
- The light-cone support proposition and its proof
- The conditional independence corollary
- Justification of the permuted-CDG test

## 23.3 Methods

- The MIMIC-IV landmark cohort
- 16-feature selection
- CDG estimation
- Held-out dependency pairs
- CDG-QGAN
- The classical locality baseline
- Training and implementation

## 23.4 Experiments

- The controlled synthetic graph benchmark
- The MIMIC-IV primary experiment
- The depth–distance analysis
- Baselines and ablations
- NISQ robustness

## 23.5 Results

- The primary HDE contrast
- Distance–depth curves
- Fidelity and utility
- Quantum vs classical local generator
- Resource accounting

## 23.6 Discussion

- What exactly the effect of CDG alignment is
- Whether there was an additional contribution from the quantum core
- Interpretation in the case of a null result
- Limitations of clinical data and of NISQ
- Directions for scaling up

---

# 24. Expected Contributions

## 24.1 Theoretical Contribution

We formalize how, in a graph-local `RZZ` quantum generator, the backward light cone of a local observable expands with depth, and we establish a distance limit on output dependencies under conditionally independent local latents.

## 24.2 Methodological Contribution

We propose CDG-QGAN, which aligns a clinical conditional-dependency graph with the dependency receptive field of a shallow quantum generator.

## 24.3 Experimental Design Contribution

Through the isomorphic permuted-CDG, which fully preserves the abstract graph structure while destroying only the clinical meaning of the nodes, we directly test whether the semantic alignment of a medical graph actually contributes.

## 24.4 Contribution in Comparison Methodology

By jointly using

- a graph-local classical generator,
- a no-entanglement quantum generator,
- a fixed nonlinear feature map, and
- a global high-capacity baseline,

we separate the effects of the graph prior, locality, the nonlinear map, and the quantum core.

## 24.5 Practical Contribution

Through a batched statevector implementation, one-basis measurement, shot-noise-aware fine-tuning, and a transpiled resource report, we present a reproducible NISQ evaluation procedure for research on quantum tabular generation.

---

# 25. Journal Strategy

## 25.1 Realistic Primary Target

**Quantum Machine Intelligence**

Conditions of fit:

- The methodological novelty of CDG-QGAN
- Agreement between the light-cone theory and the experiments
- The confirmatory permuted-CDG comparison
- A matched local classical baseline
- A reproducible NISQ resource analysis

## 25.2 Stretch Target

**Quantum Science and Technology**

Additional elements required:

- Generalization of the theoretical propositions
- A controlled tabular benchmark outside the medical domain
- Real hardware, or convincing hardware-aware results
- The resource–performance trade-off between quantum and classical local generators

## 25.3 Redirection to a Medical Informatics Journal

If JAMIA or the IEEE Journal of Biomedical and Health Informatics is targeted, the following become more important.

- External validation on eICU
- A broader set of clinical variables
- Clinician review
- Data utilization scenarios
- Subgroup utility and fairness

Because the focus of the current v2 is the **structural expressive mechanism of a quantum generator** rather than the medical application itself, a quantum AI journal is the better fit.

---

# 26. Final Research Direction

This study is fixed along the following three axes.

## First, make the CDG actually operate all the way to the output

- One feature = one qubit
- Local latents
- Local angle encoding
- Local observables
- Feature-specific local heads
- Removal of the dense decoder

## Second, validate a structural constraint rather than a performance advantage

- The light-cone proposition
- The depth-limited dependency receptive field
- Recovery error by graph distance
- The controlled graph benchmark

## Third, focus on a single confirmatory question

\[
\boxed{
\text{Does CDG alignment improve the recovery of held-out clinical dependencies?}
}
\]

The strongest control for this question is not CTGAN but the **isomorphic permuted-CDG**. The additional significance of the quantum core is judged only after comparison with the graph-local classical generator.

---

# References

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

# Appendix A. Pre-registration Checklist

- [ ] MIMIC-IV version fixed at v3.1
- [ ] Cohort SQL and commit hash fixed
- [ ] The 24-hour landmark definition fixed
- [ ] The 16 features and the replacement order fixed
- [ ] Unit conversions and error boundaries fixed
- [ ] Train/validation/test subject split fixed
- [ ] Imputation method fixed
- [ ] CDG residualization covariates fixed
- [ ] Graphical lasso selection rule fixed
- [ ] Edge stability threshold fixed
- [ ] \(E_{fit}\)/\(E_{holdout}\) split seed fixed
- [ ] Permutation list fixed
- [ ] The primary endpoint HDE formula fixed
- [ ] The primary comparison, CDG vs permuted-CDG, fixed
- [ ] Validation checkpoint rule fixed
- [ ] Primary seed list fixed
- [ ] Secondary metrics explicitly labeled as exploratory

# Appendix B. Required Implementation Verification

- [ ] Verify that the entangling gradient is 0 when Z is measured immediately after RZZ
- [ ] Verify that the gradient is nonzero when RX/RY are placed after RZZ
- [ ] Agreement between the custom simulator and the PennyLane statevector
- [ ] Agreement between autograd and parameter-shift gradients
- [ ] Graph permutation test
- [ ] Local latent independence test
- [ ] Architecture test that a local head does not read the inputs of other qubits
- [ ] Numerical light-cone test for each depth \(L\)
- [ ] Verification of the Z and ZZ estimators from the same Z-basis samples


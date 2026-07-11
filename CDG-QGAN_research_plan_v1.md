---
title: "CDG-QGAN: Synthetic Intensive Care Data Generation with a Clinical Dependency Graph-Guided Hybrid Quantum GAN"
subtitle: "Single-Author Research Proposal"
lang: en-US
version: "1.0"
date: "2026-07-11"
---

# CDG-QGAN Research Plan

> **Superseded.** This is v1, kept for history. The current plan is [CDG-QGAN_research_plan_v2.md](CDG-QGAN_research_plan_v2.md), and where v2 and [REVISIONS.md](REVISIONS.md) conflict, REVISIONS.md wins.

## Official Research Title

### Korean

**CDG-QGAN: 임상 의존성 그래프 기반 하이브리드 양자–고전 GAN을 이용한 프라이버시 감사형 합성 중환자 데이터 생성**

### English

**CDG-QGAN: A Clinical Dependency Graph-Guided Hybrid Quantum–Classical Generative Adversarial Network for Privacy-Audited Synthetic ICU Data**

### Acronym

**CDG-QGAN** = **C**linical **D**ependency **G**raph-guided **Q**uantum **G**enerative **A**dversarial **N**etwork

> **Rationale for the name**  
> The name `CDG-QGAN` conveys, on its own, the central novelty of this work: that the dependency graph among clinical variables is reflected in the entanglement structure of the quantum circuit. It expresses the methodological contribution more sharply than `ClinQGAN` and is well suited to emphasizing the link between the medical application and the circuit design. However, since CDG is a **statistical and clinical dependency graph** rather than a causal graph, the terms causal graph and causal discovery are not used anywhere in the paper.

---

# Abstract-Style Research Summary

Electronic Health Records (EHRs) are indispensable for medical artificial intelligence research, but access and sharing are restricted by patient privacy concerns and data-use constraints. Synthetic medical data offer an alternative that can improve data accessibility while preserving the distributions and clinical relationships of real patient data. Existing generative models, however, struggle to simultaneously handle mixed-type variables, missingness structure, rare patient subgroups, clinical dependencies among variables, and privacy risk.

This study constructs the dependency relations among clinical variables as a **Clinical Dependency Graph (CDG)** and proposes **CDG-QGAN**, a hybrid quantum–classical generative model that converts this graph into the entanglement topology of a Parameterized Quantum Circuit (PQC). The proposed model generates a quantum latent representation with a small number of qubits and reconstructs it, through a classical type-specific decoder, into synthetic intensive care records comprising continuous, categorical, and binary variables as well as missingness masks. Generative training combines a conditional WGAN-GP objective, a clinical constraint loss, a variable dependency structure preservation loss, and a condition consistency loss.

The core object of validation in this study is not unconditional "quantum advantage." Rather, by comparing against a classical MLP generator with an identical decoder, critic, loss function, and output dimensionality, and a comparable number of trainable parameters, we analyze whether a CDG-based PQC provides a useful inductive bias or parameter efficiency for medical tabular data generation. We further evaluate how performance varies with the number of qubits, circuit depth, shot count, noise model, and transpiled two-qubit gate count.

Because statistical similarity alone does not guarantee privacy for synthetic data, the base model of this study is described as **privacy-audited** rather than **privacy-preserving**. We comprehensively evaluate membership inference, singling-out, linkability, attribute inference, and nearest-neighbor memorization, and in a separate extended model we apply DP-SGD to analyze the trade-off between \((\varepsilon,\delta)\)-differential privacy and utility.

---

# 1. Background and Motivation

## 1.1 Medical Data Accessibility and Privacy

EHRs contain demographic information, vital signs, laboratory results, diagnoses, procedures, and clinical outcomes, and are therefore highly valuable for developing medical artificial intelligence models. However, the research use of medical data is restricted for the following reasons.

- Risk of patient identification and re-identification
- Institution-specific data use agreements and ethics review
- Insufficient samples for rare diseases and minority patient groups
- Distributional differences across institutions and data standardization issues
- Complex structure combining missingness, imbalance, and mixed data types

Simple de-identification alone cannot eliminate the risk of re-identification through combinations of high-dimensional attributes. Synthetic data have the advantage of learning statistical patterns and generating new samples without directly releasing the original records, but if the generative model memorizes training samples, real patient information can be exposed. The quality and the privacy of synthetic data must therefore be evaluated together.

## 1.2 Developments in Synthetic EHR Generation

medGAN combined an autoencoder with a GAN to generate high-dimensional discrete patient records [4]. CTGAN proposed a conditional tabular GAN that handles multimodal continuous variables and imbalanced categorical variables [5]. EHR-M-GAN jointly generates mixed-type longitudinal EHRs containing both continuous and discrete variables, and evaluated fidelity, utility, and privacy on real ICU data [6]. CTAB-GAN+ combined mixed-type tabular data handling, WGAN-GP, and DP-SGD to address the privacy–utility trade-off [7]. TabDDPM applies diffusion models to mixed-type tabular data and provides a strong classical baseline that is competitive with GAN- and VAE-based approaches [8].

This body of work shows that synthetic EHRs must be evaluated not merely on per-variable mean reproduction but jointly on multivariate relationships, downstream utility, missingness structure, and privacy.

## 1.3 Quantum Generative Models and Tabular Data

QGANs are a family of quantum generative models that learn distributions through adversarial training between a quantum generator and a classical or quantum discriminator [9,10]. Although studies have demonstrated image distribution generation on actual superconducting quantum processors, on current NISQ devices the principal constraints are the limited number of qubits, noise, shot cost, gradient estimation cost, and barren plateaus [11,12].

Among recent studies specialized for tabular data, TabularQGAN compared quantum generative models against CTGAN and CopulaGAN on MIMIC-III and Adult Census [13], while QTabGAN proposed an architecture that maps the \(2^n\)-dimensional probability vector produced by a PQC onto tabular variables using a classical neural network [14]. Consequently, the mere fact that "a QGAN was applied to medical tabular data" is unlikely to constitute sufficient novelty.

## 1.4 Gaps in Prior Work

| Research gap | Limitation of existing approaches | Response in this study |
|---|---|---|
| Mismatch between medical variable structure and the quantum circuit | Use of semantics-agnostic entanglement such as ring, linear, or all-to-all | Construct a CDG and convert it into an entanglement topology |
| Ambiguity of the quantum effect | Fair comparison is difficult because decoder size or total parameter count differs | Construct parameter-matched and output-matched classical generators |
| Handling of mixed-type and missing EHR data | Focus on a small number of continuous features or simple binary data | Jointly generate continuous, categorical, and binary variables and missingness masks |
| Insufficient evaluation of clinical relationships | Focus on per-variable similarity | Reflect physiological constraints and clinical dependency structure preservation in both the loss and the evaluation |
| Overstated privacy | Assume automatic anonymity simply because the data are synthetic | Separate attack-based privacy audit from formal DP |
| Insufficient analysis of NISQ practicality | Report only ideal simulator results | Analyze shots, noise, depth, gate count, and gradient variance |

---

# 2. Research Objectives

## 2.1 Primary Objective

Design **CDG-QGAN**, a hybrid generative model that reflects the dependencies among clinical variables in the entanglement structure of a quantum circuit, and comprehensively evaluate it on real intensive care data with respect to statistical fidelity, clinical plausibility, machine learning utility, diversity, privacy risk, and NISQ resource efficiency.

## 2.2 Specific Objectives

1. Construct a mixed-type static patient table based on the first 24 hours following the first ICU admission in MIMIC-IV.
2. Construct a Clinical Dependency Graph by combining type-specific association measures with published physiological relationships.
3. Map the community structure of the CDG onto qubits and convert the graph edges into entangling gates of the PQC.
4. Combine an expectation-value-based quantum latent representation with a type-specific decoder.
5. Design a training objective that preserves clinical constraints and dependency structure.
6. Compare against strong classical generative models and a resource-controlled MLP generator.
7. Perform an attack-based privacy evaluation and an optional differential privacy extension.
8. Analyze robustness with respect to the number of qubits, circuit depth, shot count, noise, and hardware topology.

---

# 3. Research Questions and Hypotheses

| Research question | Hypothesis |
|---|---|
| **RQ1.** Is a CDG-based quantum latent generator more useful than a classical latent generator of the same scale? | **H1.** Under identical decoder, critic, and loss conditions, CDG-QGAN achieves a smaller CDG edge-weighted dependency error than a parameter-matched MLP generator, or is non-inferior in TSTR utility. |
| **RQ2.** Is an entanglement structure based on clinical dependencies more effective than generic circuit connectivity? | **H2.** Compared with ring and random sparse ansätze having the same number of two-qubit edges, the CDG ansatz preserves correlation and conditional dependency structure more accurately. |
| **RQ3.** Does the clinical constraint loss improve the plausibility of generated records? | **H3.** The Clinical Constraint Loss reduces the physiological violation rate, while the degradation in utility remains within a pre-specified tolerance. |
| **RQ4.** What is the relationship between improved generation quality and privacy risk? | **H4.** As fidelity increases, membership and attribute inference risk may increase for some rare samples; applying DP lowers the attacker's advantage at the cost of reduced utility. |
| **RQ5.** How robust is the proposed model under NISQ conditions? | **H5.** Under calibrated noise, the sparse CDG ansatz exhibits smaller performance degradation and more stable gradients than circuits with larger gate counts. |
| **RQ6.** Are the improvements from the quantum circuit explained merely by a difference in model size? | **H6.** Even after parameter-matched, output-matched, and compute-reported comparisons, some of the structure-preservation benefits of CDG-QGAN persist. |

---

# 4. Core Contributions

## 4.1 CDG-Based Quantum Circuit Design

We propose a reproducible procedure for estimating stable dependency relations among medical variables and converting them into qubit communities and an entanglement topology.

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

## 4.2 Comparison Controlled for Parameters and Output Dimensionality

The quantum circuit is replaced by a classical MLP with the same latent output dimension and a comparable number of trainable parameters. The decoder, critic, loss, and training protocol are held identical in order to isolate the effect of the quantum core.

## 4.3 Clinical Structure-Preserving Objective

WGAN-GP is combined with losses that reflect blood pressure relationships, physiological ranges, the variable dependency matrix, condition consistency, and missingness structure.

## 4.4 Comprehensive Privacy Audit

We evaluate not only membership inference but also singling-out, linkability, attribute inference, and memorization, and separately analyze the risk for rare patient subgroups.

## 4.5 NISQ Practicality Analysis

We report not only model performance but also qubits, depth, shots, noise, transpiled gate count, gradient variance, number of circuit evaluations, and wall-clock cost.

---

# 5. Scope and Boundaries of the Claims

## 5.1 What This Study Can Claim

- Whether CDG-based circuit design constitutes a useful inductive bias for generating a particular class of medical tabular data
- Fidelity, utility, clinical consistency, and parameter efficiency relative to classical models under matched resource conditions
- How performance degrades under NISQ conditions
- Attack-based privacy risk and the privacy–utility trade-off when DP is applied

## 5.2 What This Study Cannot Automatically Claim

- Universal quantum advantage from the standpoint of computational complexity
- Complete anonymity of the synthetic data
- Clinical safety or applicability to actual patient care
- Causality of the CDG edges
- Demonstration of superiority on real quantum hardware on the basis of simulator results alone
- Superiority in overall computational efficiency on the basis of a smaller parameter count alone

The central sentence of the paper is set as follows.

> **We do not claim unconditional quantum advantage. We investigate whether a clinical-dependency-guided parameterized quantum circuit provides a useful and resource-efficient inductive bias for mixed-type ICU data synthesis under matched classical baselines and NISQ constraints.**

---

# 6. Data Design

## 6.1 Primary Dataset

The primary dataset is **MIMIC-IV v3.1** [15,16]. MIMIC-IV contains de-identified emergency department and intensive care unit data from Beth Israel Deaconess Medical Center. Data access requires PhysioNet credentialing, completion of the required training, and agreement to the data use terms.

The raw data and patient-level derived data are not released; the following items are designated as publicly releasable reproducibility artifacts.

- Cohort definition
- SQL extraction code
- Variable dictionary
- Preprocessing code
- Model and evaluation code
- Hyperparameter settings
- Random seeds
- Circuit and transpilation settings

## 6.2 Unit of Analysis

The scope of the first paper is restricted not to full longitudinal time series but to a **per-patient static mixed-type table** of the following form.

- Each patient's first ICU admission
- The first 24 hours from the time of ICU admission
- Minimum, mean, and maximum values of vital signs, or clinically selected summary values
- First, minimum, maximum, or representative values of laboratory results
- Treatment status and admission information
- Per-variable missingness masks
- In-hospital mortality

This scope is more realistic for a single author to implement and validate than full longitudinal EHR generation, and it is well suited to clearly demonstrating the methodological contribution of the quantum generator.

## 6.3 Inclusion Criteria

- Adults aged 18 years or older
- Each patient's first ICU admission
- Patients for whom observable clinical records exist after ICU admission
- Patients for whom the primary outcome and key demographic variables can be identified

## 6.4 Exclusion and Bias Management

Uniformly excluding patients who died, were transferred, or were discharged before 24 hours can introduce a selection bias in which only patients with long survival times remain. We therefore use the values from the observable period and represent unobserved variables with a missingness mask.

Clear data errors and unit errors are treated as missing according to pre-specified rules. Extreme values are not removed unconditionally; clinically possible rare values are distinguished from physically impossible values.

## 6.5 Variable Composition

The target number of features is set at approximately 30 to 45.

| Variable group | Candidate variables |
|---|---|
| Demographics | Age, sex, race category |
| Admission information | Admission type, ICU type, emergency admission status |
| Vital signs | Heart rate, systolic blood pressure, diastolic blood pressure, mean arterial pressure, respiratory rate, oxygen saturation, body temperature |
| Renal/electrolytes | Creatinine, BUN, sodium, potassium, chloride |
| Metabolic/acid–base | Glucose, lactate, bicarbonate |
| Hematology | White blood cell count, hemoglobin, hematocrit, platelets |
| Treatment | Mechanical ventilation status, vasopressor use |
| Missingness information | Per-variable observation status or per-group masks |
| Condition/outcome | In-hospital mortality |

Variable selection is fixed in advance on the training set based on missingness rate, clinical importance, data extraction stability, and redundancy.

## 6.6 Condition Variable

The primary condition is in-hospital mortality.

\[
y \in \{0,1\}, \qquad y=1:\text{in-hospital mortality}
\]

Conditional generation makes it possible to evaluate not only simple replication of the overall distribution but also generation performance for the minority mortality group separately. ICU type may be considered as an additional condition, but the core condition in the first paper is restricted to a single variable.

## 6.7 Data Splits

Splits are performed at the `subject_id` level so that multiple admissions of the same patient do not fall into different splits.

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
70\% : 15\% : 15\%
\]

- Stratified by mortality status
- Preprocessing statistics estimated only on the training set
- The CDG constructed only on the training set
- The validation set used for hyperparameters and early stopping
- The test set used only for the final evaluation

---

# 7. Preprocessing Methods

## 7.1 Data Quality Cleaning

1. Consolidation of item identifiers for the same clinical item
2. Unit conversion and detection of unit inconsistencies
3. Fixed rules for handling duplicate measurements
4. Physically impossible values treated as missing
5. Clinically possible extreme values retained, with robust scaling applied
6. All rules recorded in the code and the variable dictionary

## 7.2 Continuous Variable Transformation

### Default option: robust scaling

Each continuous variable \(x_j\) is transformed using the median and IQR of the training set.

\[
\tilde{x}_j
=
\frac{x_j-\operatorname{median}_{train}(x_j)}
{\operatorname{IQR}_{train}(x_j)+\epsilon}
\]

Where necessary, values are clipped to the range \([-c,c]\) and then rescaled to \([-1,1]\). The inverse transform uses training set statistics only.

### Extended option: mode-specific normalization

For multimodal continuous variables, CTGAN-style mode-specific normalization can be applied [5]. A variational Gaussian mixture is fitted to the variable \(x_j\), and for the selected component \(k\) we represent

\[
\alpha_j
=
\frac{x_j-\mu_{jk}}{4\sigma_{jk}}
\]

together with the component one-hot vector \(\beta_j\). The generator outputs \(\alpha_j\) and \(\beta_j\), where \(\beta_j\) uses Gumbel-Softmax.

In the main analysis, whichever of robust scaling and mode-specific normalization yields more stable validation performance is selected, while the effect of the two preprocessing schemes on the model ranking is examined in a supplementary experiment.

## 7.3 Categorical Variables

- Low cardinality: one-hot representation
- High cardinality: rare categories merged into `Other` based on the training set
- Generator output: Gumbel-Softmax
- Final generation: argmax or categorical sampling

Gumbel-Softmax provides a differentiable approximation during training.

\[
\tilde c_{jk}
=
\frac{
\exp((\log \pi_{jk}+g_k)/\tau)
}{
\sum_r \exp((\log \pi_{jr}+g_r)/\tau)
}
\]

## 7.4 Missingness Structure

Because missingness in medical data is not merely random loss but can reflect test selection and patient status, missing values are not all removed via KNN imputation.

The observation status of each variable is defined as

\[
m_j=
\begin{cases}
1,&x_j\text{ observed}\\
0,&x_j\text{ missing}
\end{cases}
\]

The training input uses a temporary imputed value \(x_j^{imp}\) together with the mask \(m_j\).

\[
x_j^{model}=m_jx_j^{imp}+(1-m_j)c_j
\]

Here \(c_j\) is a neutral value after scaling. The generator jointly outputs values and masks, and in the final synthetic record the positions with \(\tilde m_j=0\) are restored as missing.

## 7.5 Class Imbalance

When the mortality group is small, conditional sampling is applied. The proportion of the condition \(y\) can be adjusted within each minibatch, but at final data generation time the following two settings are distinguished.

1. **Distribution-preserving generation**: reproduce the true training prevalence
2. **Balanced augmentation generation**: generate an equal number of samples per condition for downstream augmentation

The two sets of results are reported separately and are not mixed.

---

# 8. Proposed Method: CDG-QGAN

## 8.1 Overall Architecture

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

The generator is defined as follows.

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

- \(z\): latent random variable
- \(y\): clinical condition
- \(E_{\omega}\): classical angle encoder
- \(Q_{\theta}\): CDG-guided PQC and measurement
- \(G_{dec}\): mixed-type decoder

A skip connection that routes the latent random variable \(z\) directly to the decoder is not permitted in the base model. Otherwise, the decoder could ignore the quantum output and generate the data through the classical path alone.

---

# 9. Clinical Dependency Graph Construction

## 9.1 Definition

The CDG is defined as the following weighted undirected graph.

\[
\mathcal G_F=(V_F,E_F,A)
\]

- \(V_F\): medical feature nodes
- \(E_F\): edges representing stable dependency relations
- \(A_{jk}\): association between features \(j\) and \(k\)

The CDG is not a causal graph, and it does not imply edge directionality or intervention effects.

## 9.2 Type-Specific Association Measures

Different association estimation methods are used depending on the combination of data types.

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

To complement nonlinear relationships, normalized mutual information or distance correlation may be used as a secondary association measure.

## 9.3 Association Under Missingness

Each pairwise association is computed on the samples in which both variables are simultaneously observed; pairs with a small sample size are either excluded or subjected to shrinkage. The missingness masks themselves are also analyzed as separate nodes or as a separate association matrix.

\[
A^{mask}_{jk}=\operatorname{Assoc}(m_j,m_k)
\]

The two matrices are recorded separately so that value relations and missingness relations are not conflated.

## 9.4 Clinical Prior Relations

Published physiological definitions or obvious variable-group relations are added as prior edges.

Examples:

- \(SBP\leftrightarrow DBP\leftrightarrow MAP\)
- \(Hemoglobin\leftrightarrow Hematocrit\)
- \(Creatinine\leftrightarrow BUN\)
- \(Respiratory\ Rate\leftrightarrow SpO_2\)
- \(Lactate\leftrightarrow Bicarbonate\)

Prior edges denote structural relatedness rather than causality. Since a single-author study lacks clinician review, only relationships with clear literature support or clinical definitions are used.

## 9.5 Combining Data-Driven Relations with the Prior

\[
A_{jk}
=
\lambda_A A^{data}_{jk}
+
(1-\lambda_A)A^{prior}_{jk}
\]

Here \(A^{prior}_{jk}\in\{0,1\}\) or a confidence weight, and \(\lambda_A\) is determined via validation.

## 9.6 Bootstrap Stability Selection

To prevent the circuit structure from being determined by chance sample correlations, bootstrapping is performed on the training set.

Letting \(E^{(b)}\) denote the set of top-association edges in bootstrap replicate \(b\),

\[
\pi_{jk}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbb{I}[(j,k)\in E^{(b)}]
\]

Edges with stability \(\pi_{jk}\geq\tau_{stable}\) are selected preferentially. The default candidates are \(B=100\) and \(\tau_{stable}=0.7\).

## 9.7 Feature Community Construction

Because the number of features \(d\) exceeds the number of qubits \(n_q\), the features are grouped into \(n_q\) communities.

\[
\mathcal C=\{C_1,\ldots,C_{n_q}\}
\]

Possible methods:

- spectral clustering
- Louvain community detection
- normalized-cut clustering

The objective of the clustering is to achieve high within-community association and low between-community association.

\[
\max_{\mathcal C}
\sum_{u=1}^{n_q}
\sum_{j,k\in C_u}A_{jk}
-
\lambda_C
\sum_{u\neq v}
\sum_{j\in C_u,k\in C_v}A_{jk}
\]

The resulting communities are examined for clinical interpretability, and their stability across seeds is reported.

## 9.8 Qubit-Level Graph

The weight between communities \(C_u\) and \(C_v\) is defined as follows.

\[
W_{uv}
=
\frac{1}{|C_u||C_v|}
\sum_{j\in C_u}
\sum_{k\in C_v}
A_{jk}
\]

To guarantee a connected graph, a maximum spanning tree is constructed first, and additional top-weight edges are then selected.

\[
E_{CDG}
=
E_{MST}
\cup
E_{top-r}
\]

For a fair comparison with ring and random sparse ansätze, the number of two-qubit edges is matched.

---

# 10. Generator Architecture

## 10.1 Latent Variable and Condition Embedding

\[
z\sim\mathcal U(-1,1)^{d_z}
\]

The condition \(y\) is converted into an embedding.

\[
h_0=[z;e_y(y)]
\]

The default candidate latent dimensions are \(d_z\in\{8,16,32\}\).

## 10.2 Classical Angle Encoder

A classical neural network produces the angles fed into the PQC.

\[
\phi^{(a)}_{\ell,q}
=
\pi\tanh
\left(
W^{(a)}_{\ell,q}h_0+b^{(a)}_{\ell,q}
\right)
\]

- \(\ell\): circuit layer
- \(q\): qubit
- \(a\in\{Y,Z\}\): rotation axis

Bounding the angles with `tanh` mitigates excessive randomization of the initial state.

## 10.3 Initial State

The default state is set to

\[
|\psi_0\rangle=|0\rangle^{\otimes n_q}
\]

As a supplementary experiment, uniform-superposition initialization by applying a Hadamard gate to each qubit can be compared.

## 10.4 Data Re-uploading PQC

The full circuit consists of \(L\) repeated layers.

\[
U_{\theta}(z,y)
=
\prod_{\ell=1}^{L}
U_{ent}^{(\ell)}
U_{var}^{(\ell)}
U_{enc}^{(\ell)}(z,y)
\]

### Input encoding

\[
U_{enc}^{(\ell)}
=
\prod_{q=1}^{n_q}
R_Y(\phi^{Y}_{\ell,q})
R_Z(\phi^{Z}_{\ell,q})
\]

### Trainable single-qubit rotations

\[
U_{var}^{(\ell)}
=
\prod_{q=1}^{n_q}
R_Y(\theta^{(1)}_{\ell,q})
R_Z(\theta^{(2)}_{\ell,q})
R_Y(\theta^{(3)}_{\ell,q})
\]

### CDG-based entanglement

\[
U_{ent}^{(\ell)}
=
\prod_{(u,v)\in E_{CDG}}
R_{ZZ}^{(u,v)}(\gamma_{\ell,uv})
\]

Alternatively, a CNOT–rotation structure can be used for a hardware-efficient comparison.

\[
R_{ZZ}(\gamma)
=
\operatorname{CNOT}
\left(I\otimes R_Z(\gamma)\right)
\operatorname{CNOT}
\]

## 10.5 Ways of Using the Graph Weights

The following two schemes are compared.

### Topology-only

The CDG determines only the connectivity structure, and all \(\gamma_{\ell,uv}\) are learned independently.

### Weight-informed initialization

\[
\gamma_{\ell,uv}^{(0)}
=\alpha_{init}W_{uv}+\epsilon_{uv}
\]

The CDG edge weights are used only for initialization, after which the parameters are learned freely. This scheme provides a data-dependent inductive bias without excessively restricting the expressivity of the circuit.

## 10.6 Number of Qubits and Depth

\[
n_q\in\{4,6,8,10\}
\]

\[
L\in\{1,2,3,4\}
\]

The main model candidate is set to 8 qubits with 2–3 layers. The final setting is chosen by jointly considering validation performance, two-qubit gate count, and gradient stability.

---

# 11. Quantum Measurement and Latent Representation

## 11.1 Single-Qubit Measurement

\[
q_i=\langle Z_i\rangle
\]

## 11.2 CDG Edge Measurement

\[
q_{uv}=\langle Z_uZ_v\rangle,
\qquad (u,v)\in E_{CDG}
\]

## 11.3 Final Quantum Latent Vector

\[
q(z,y)=
\left[
\langle Z_1\rangle,\ldots,\langle Z_{n_q}\rangle,
\{\langle Z_uZ_v\rangle\}_{(u,v)\in E_{CDG}}
\right]
\]

The output dimension is

\[
d_q=n_q+|E_{CDG}|
\]

For example, using 8 qubits and 10 CDG edges yields an 18-dimensional quantum latent.

## 11.4 Measurement Ablation

- Use only single-qubit \(Z\) expectations
- Use \(Z+ZZ\) expectations
- Use all pairwise \(ZZ\) terms
- Use only the \(ZZ\) terms corresponding to CDG edges

This verifies whether pairwise quantum measurement actually contributes to structure preservation.

---

# 12. Type-Specific Decoder

## 12.1 Shared Representation

\[
h_1=\operatorname{LeakyReLU}(W_1[q;e_y(y)]+b_1)
\]

\[
h_2=\operatorname{LeakyReLU}(W_2h_1+b_2)
\]

The default architecture candidates are \(d_q\rightarrow128\rightarrow256\) or \(d_q\rightarrow128\rightarrow128\).

## 12.2 Continuous Output Head

When robust scaling is used, we employ

\[
\tilde{x}^{cont}_j
=
\tanh(W^{cont}_jh_2+b^{cont}_j)
\]

When mode-specific normalization is used, each variable outputs a normalized scalar together with mixture component logits.

## 12.3 Categorical Output Head

\[
\tilde{x}^{cat}_j
=
\operatorname{GumbelSoftmax}
(W^{cat}_jh_2+b^{cat}_j;\tau)
\]

## 12.4 Binary Output Head

\[
\tilde{x}^{bin}_j
=
\sigma(W^{bin}_jh_2+b^{bin}_j)
\]

During training, a Binary Concrete relaxation or a straight-through estimator is used.

## 12.5 Missingness Mask Head

\[
\tilde{m}_j
=
\sigma(W^{mask}_jh_2+b^{mask}_j)
\]

The final generated value is restored as

\[
\tilde{x}^{final}_j
=
\begin{cases}
\tilde{x}_j,&\tilde m_j=1\\
\text{missing},&\tilde m_j=0
\end{cases}
\]

## 12.6 Quantum Bottleneck Validation

To confirm that the decoder actually uses the quantum latent, the following analyses are performed.

- quantum latent permutation test
- latent zeroing test
- decoder input gradient norm
- estimation of mutual information per latent dimension
- change in generation diversity when the PQC output is held fixed

If performance does not change when the PQC output is randomly shuffled or replaced with zeros, this is interpreted as evidence that the decoder is not meaningfully exploiting the quantum representation.

---

# 13. Conditional Critic

## 13.1 Basic Architecture

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

Batch Normalization is not used in the critic, because batch-dependent normalization can interfere when the gradient penalty operates on per-sample input gradients.

## 13.2 Projection Conditioning

The condition \(y\) can be incorporated via a projection critic.

\[
D_{\psi}(x,y)
=f_{\psi}(x)+h_{\psi}(x)^Te_D(y)
\]

The more stable of this and simple concatenation is selected by comparison.

---

# 14. Objective Functions

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

### Blood pressure ordering relation

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

### Mean arterial pressure approximation

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

Because this relation is an approximation, it is used as a soft penalty rather than a hard equality.

### Physiological ranges

Letting \([l_j,u_j]\) denote the admissible range of variable \(j\),

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

### Combined clinical loss

\[
\mathcal L_{clin}
=
\mathcal L_{order}
+
\alpha_{MAP}\mathcal L_{MAP}
+
\alpha_{range}\mathcal L_{range}
\]

Hard rules that enforce diagnostic criteria or treatment guidelines on the basis of a single laboratory value are not used, because real clinical data contain treatment effects, measurement errors, diagnostic delays, and exceptions.

## 14.4 Dependency Structure Loss

Letting \(R^{real}\) and \(R^{syn}\) denote the batch correlation matrices of the continuous variables,

\[
\mathcal L_{corr}
=
\|R^{real}-R^{syn}\|_F^2
\]

The structure loss weighted toward the principal CDG edges is defined as

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

For categorical variables, a differentiable contingency table is constructed from soft category probabilities. Where a fully differentiable implementation of mixed-type association is difficult, the continuous correlation loss is used for training and mixed-type association is used only as an evaluation metric.

## 14.5 Missingness Loss

Preservation of per-variable missingness rates:

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

Preservation of correlations among missingness masks:

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

An auxiliary classifier \(C_{\xi}\) is pre-trained on the training set and then frozen.

\[
\mathcal L_{cond}
=
\operatorname{BCE}(C_{\xi}(\tilde{x}),y)
\]

The final utility evaluation includes algorithms different from the auxiliary classifier in order to avoid circular evaluation.

## 14.7 Optional Diversity Regularization

The following regularizer is used as an ablation only when mode collapse is observed.

\[
\mathcal L_{div}
=-
\frac{
\|G(z_1,y)-G(z_2,y)\|_1
}{
\|z_1-z_2\|_1+\epsilon
}
\]

It is not applied automatically to all models but is isolated as a collapse-mitigation experiment.

## 14.8 Total Generator Loss

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

## 14.9 Loss Weight Ramp-Up

Applying the structure losses with large weights from the very beginning of training can lead to the generation of only average, restricted samples. The weights are therefore increased gradually after an adversarial warm-up.

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

This is an optimization procedure internal to a single training run, not a research schedule.

---

# 15. Training Algorithm

## Algorithm 1. CDG-QGAN Training

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

## 15.1 Checkpoint Selection

Rather than optimizing a single metric, a validation composite score is used.

\[
S_{val}
=w_fS_{fidelity}
+w_uS_{utility}
+w_cS_{clinical}
-w_pR_{privacy-proxy}
\]

The test set is not used for checkpoint selection. The privacy proxy is limited to a simple nearest-neighbor gap at the validation stage, and the final attack evaluation is performed at the test stage.

---

# 16. Differential Privacy Extension

## 16.1 Naming Distinction

- **CDG-QGAN**: the base model, whose privacy is audited by attacks
- **DP-CDG-QGAN**: an extended model trained to satisfy explicit \((\varepsilon,\delta)\)-DP

A model without DP is not called privacy-preserving.

## 16.2 DP-SGD Critic

The critic gradient \(g_i\) for each patient sample is clipped to norm \(C\).

\[
\bar g_i
=g_i\cdot
\min\left(1,\frac{C}{\|g_i\|_2}\right)
\]

Gaussian noise is then added.

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

The overall privacy budget is computed using an RDP or PRV accountant [17,18].

\[
\delta<\frac{1}{N}
\]

is adopted as the basic principle.

## 16.3 Conflict Between DP and Data-Dependent Losses

Even if only the critic is trained with DP-SGD, the overall DP guarantee can be broken if the generator loss directly uses real training statistics in a non-private manner.

Points of concern:

- Repeatedly referring directly to the real correlation matrix
- Use of a non-DP auxiliary classifier
- Use of non-DP CDG edge weights
- Non-private estimation of clinical thresholds from real data

The DP extension therefore distinguishes the following variants.

### DP-CDG-QGAN-Public

- Construct the CDG from published physiological relationships
- Use only public clinical ranges
- Remove real batch statistics from \(\mathcal L_{struct}\)
- Transmit data information only through the DP critic

### DP-CDG-QGAN-PrivateGraph

- Apply DP noise to the CDG associations
- Include the cost of graph estimation in the privacy accountant
- Use DP auxiliary statistics

If formal DP becomes excessively complex for the first paper, `DP-CDG-QGAN-Public` is carried out as an extension experiment and the core contribution remains the privacy-audited CDG-QGAN.

---

# 17. Implementation Environment

## 17.1 Software

- Python
- PyTorch
- PennyLane
- Qiskit
- Qiskit Aer
- scikit-learn
- XGBoost
- Auxiliary evaluation based on SDMetrics or SynthCity
- DP tooling such as Opacus or TensorFlow Privacy
- Anonymeter

## 17.2 Quantum Differentiation

### Ideal simulator

- analytic expectation
- simulators supporting adjoint differentiation or backpropagation

### Finite-shot and hardware-compatible experiments

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

Because end-to-end training of the full GAN on real hardware can require an extremely large number of circuit evaluations, the default design consists of training on a simulator followed by finite-shot/noisy inference and optional small-scale fine-tuning.

## 17.3 Recommended Initial Hyperparameters

| Item | Initial value or search range |
|---|---:|
| Batch size | 64, 128 |
| Critic updates | 5 per generator update |
| Classical optimizer | Adam |
| PQC optimizer | Adam or AdamW |
| Critic learning rate | \(1\times10^{-4}\) |
| Decoder learning rate | \(1\times10^{-4}\) |
| PQC learning rate | \(1\times10^{-4}\), \(5\times10^{-5}\) |
| Adam \(\beta_1,\beta_2\) | \((0.0,0.9)\) |
| Gradient penalty | 10 |
| Qubits | 4, 6, 8, 10 |
| PQC layers | 1, 2, 3, 4 |
| Latent dimension | 8, 16, 32 |
| Shots | analytic, 256, 1024, 4096 |
| Random seeds | at least 5; 10 recommended for key comparisons |
| Max training epochs | up to 1,000, with early stopping |

The learning rate of 0.001 in the initial draft can be unstable in the combination of WGAN-GP and a PQC, and is therefore not used as the default.

## 17.4 Initialization

- PQC: near-identity or small uniform initialization
- Decoder/Critic: Xavier or Kaiming initialization
- CDG edge parameters: small random values or weight-informed initialization
- The same preprocessing and split are maintained across seeds

## 17.5 Gradient Monitoring

The following are recorded layer by layer.

\[
\|\nabla_{\theta_{\ell}}\mathcal L_G\|_2
\]

\[
\operatorname{Var}
\left[
\frac{\partial\mathcal L_G}{\partial\theta_{\ell,q}}
\right]
\]

We check whether the gradient variance shrinks rapidly with depth in order to analyze barren plateaus or optimization instability [12].

---

# 18. Comparison Models

## 18.1 Classical Baselines

| Model | Purpose |
|---|---|
| Gaussian Copula | Simple statistical baseline |
| CTGAN | Representative mixed-type conditional GAN [5] |
| TVAE | VAE-based tabular baseline |
| CTAB-GAN+ | Mixed-type variables, WGAN-GP, and DP comparison [7] |
| TabDDPM | Strong diffusion-based baseline [8] |
| Classical WGAN-GP | Comparison under the same adversarial objective |
| Parameter-Matched MLP-GAN | The key baseline for isolating the effect of the quantum core |
| Output-Matched MLP-GAN | Comparison at the same latent output dimension |

## 18.2 Quantum Baselines

| Model | Notes |
|---|---|
| Ring-QGAN | Fixed ring entanglement |
| RandomSparse-QGAN | Random sparse topology with the same number of edges |
| AllToAll-QGAN | High-resource reference model |
| TabularQGAN | Prior work on medical tabular QGANs [13] |
| QTabGAN | Prior work on hybrid tabular QGANs, 2026 preprint [14] |
| CDG-QGAN | Proposed model |

Because TabularQGAN and QTabGAN may differ in data representation and output structure, the full-feature experiments are kept separate from the reduced-feature benchmark.

## 18.3 Parameter-Matched MLP-GAN

The PQC is replaced by a classical MLP \(M_{\varphi}\).

\[
q_{classical}=M_{\varphi}(z,y)
\]

The following conditions are held fixed.

- Same decoder
- Same critic
- Same loss function
- Same batch sampling
- Same training steps
- Same latent output dimension

Parameter condition:

\[
|\varphi|
\approx
|\theta_{PQC}|+|\omega_{angle}|
\]

The total parameter count of the quantum model must include the angle encoder. Counting only the PQC parameters while excluding the classical preprocessing network would not be fair.

## 18.4 Computational Cost Reporting

- trainable parameters
- forward circuit evaluations
- gradient circuit evaluations
- wall-clock training time
- CPU/GPU utilization
- number of shots
- simulator type
- transpiled gate count

Parameter efficiency and computational efficiency are reported as separate concepts.

---

# 19. Ablation Study

| ID | Change | Purpose of validation |
|---|---|---|
| A1 | CDG → ring | Effect of a semantics-driven topology |
| A2 | CDG → random sparse | Whether the CDG is more effective than an arbitrary sparse graph |
| A3 | CDG → all-to-all | Reference comparison against a high-resource circuit |
| A4 | Remove \(\mathcal L_{clin}\) | Effect of the clinical constraint loss |
| A5 | Remove \(\mathcal L_{struct}\) | Effect of the dependency structure loss |
| A6 | Remove \(\mathcal L_{miss}\) | Effect of modeling the missingness structure |
| A7 | Remove \(ZZ\) measurements | Necessity of pairwise quantum measurement |
| A8 | PQC → parameter-matched MLP | Effect of the quantum latent core |
| A9 | Remove the condition variable | Effect of conditional generation |
| A10 | Use only the data-driven CDG | Effect of the clinical prior |
| A11 | Use only the clinical prior | Effect of the data-driven graph |
| A12 | Remove stable edge selection | Effect of bootstrap stability selection |
| A13 | Remove weight-informed initialization | Effect of edge weight initialization |
| A14 | Apply DP-SGD | Privacy–utility trade-off |
| A15 | Quantum latent zeroing/permutation | Dependence of the decoder on the quantum latent |

Because all-to-all increases the number of two-qubit gates, it is explicitly designated as a high-resource reference experiment rather than a matched-resource comparison.

---

# 20. NISQ Experimental Design

## 20.1 Circuit Scale Analysis

\[
n_q\in\{4,6,8,10\},
\qquad
L\in\{1,2,3,4\}
\]

The following are recorded for each setting.

- trainable quantum parameters
- total trainable parameters
- logical circuit depth
- transpiled depth
- number of single-qubit gates
- number of two-qubit gates
- number of measurement observables
- gradient variance
- generation quality
- training time

## 20.2 Shot Analysis

\[
S\in\{256,1024,4096\}
\]

Finite-shot performance degradation is computed relative to the analytic expectation.

\[
\Delta_{shot}(M)
=
\frac{M_{analytic}-M_{shot}}
{|M_{analytic}|+\epsilon}
\]

Where the direction of a metric is inverted, the sign is adjusted accordingly.

## 20.3 Noise Analysis

Comparison conditions:

1. ideal analytic simulator
2. finite-shot noiseless simulator
3. backend-calibrated noise model
4. with readout mitigation applied
5. where possible, inference on a real quantum device

If the real-device results are inference-only, this is clearly stated in the paper.

## 20.4 Hardware Topology and Routing

Because the CDG topology may not match the coupling map of the real device, the following are reported.

- initial layout
- number of SWAP gates
- transpiled two-qubit depth
- mapping strategy
- CDG edge preservation rate

Where possible, a layout heuristic that jointly considers CDG edge weights and hardware connection costs is examined as a secondary contribution.

## 20.5 Noise Mitigation

- readout error mitigation
- measurement calibration
- where possible, a supplementary zero-noise extrapolation experiment

The mitigation settings are fixed in advance at the validation stage, and results both before and after application are reported.

---

# 21. Evaluation Framework

## 21.1 Evaluation Principles

The quality of synthetic data is not concluded from a single score.

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

## 21.2 Statistical Fidelity

### Univariate continuous distributions

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

### Categorical distributions

- Total Variation Distance
- Jensen–Shannon divergence
- category support coverage

\[
TVD(P,Q)=\frac{1}{2}\sum_i|P_i-Q_i|
\]

### Multivariate distributions

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

This quantity is used as the primary endpoint for fidelity.

## 21.3 Missingness Pattern Evaluation

- per-variable missing rate MAE
- mask correlation error
- missing rate by mortality condition
- ordering patterns of key laboratory tests
- conditional relationship between observed values and missingness

## 21.4 Clinical Plausibility

### Constraint violation rate

\[
ViolationRate_r
=
\frac{
\#\{\tilde x:r(\tilde x)\text{ violated}\}
}{N_{syn}}
\]

### Violation severity

\[
Severity_r
=
\frac{1}{N_{syn}}
\sum_i
\operatorname{dist}(\tilde x_i,\mathcal C_r)
\]

### Evaluation by relation

- SBP–DBP–MAP
- Hemoglobin–Hematocrit
- Creatinine–BUN
- Respiratory Rate–SpO₂
- Lactate–Bicarbonate

The violation rate of the real test set itself is also presented. The goal is not to drive the violation rate of the synthetic data unconditionally to zero, but rather to avoid excessively distorting the structure of the real holdout.

## 21.5 Machine Learning Utility

### TRTR

\[
\text{Train Real, Test Real}
\]

### TSTR

\[
\text{Train Synthetic, Test Real}
\]

### TSRTR or mixed augmentation

\[
\text{Train Real + Synthetic, Test Real}
\]

### Downstream models

- Logistic Regression
- Random Forest
- XGBoost
- MLP

### Metrics

- AUROC
- AUPRC
- Macro-F1
- sensitivity
- specificity
- Brier score
- Expected Calibration Error

For imbalanced outcomes such as mortality, accuracy is not used as the primary metric.

### Utility ratio

\[
U_{ratio}
=
\frac{M_{TSTR}}{M_{TRTR}}
\]

The primary utility endpoint is set as the TSTR AUPRC for mortality prediction.

## 21.6 Low-Data Augmentation

At real training data fractions

\[
p\in\{10\%,25\%,50\%\}
\]

the following are compared.

1. Real data \(p\%\) only
2. Real data \(p\%\) + CTGAN synthetic data
3. Real data \(p\%\) + TabDDPM synthetic data
4. Real data \(p\%\) + CDG-QGAN synthetic data

This validates not only the scenario in which synthetic data fully replace real data but also the augmentation benefit in data-scarce settings.

## 21.7 Diversity

A Mode Score computed by simply applying `numpy.unique` to continuous tabular data is inappropriate. The following metrics are used instead.

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

Since high uniqueness can also be obtained from noisy data, it is interpreted together with fidelity.

---

# 22. Privacy Evaluation

## 22.1 Membership Inference

DOMIAS is a density-based membership inference attack that exploits local overfitting of the generative model, and it demonstrates particularly strong attack potential on rare or under-represented samples [19].

Evaluation metrics:

- ROC-AUC
- attack accuracy
- TPR at fixed FPR
- attack advantage

\[
Adv_{MIA}=TPR-FPR
\]

Because random accuracy in a balanced binary attack is roughly 50%, we do not set a target of "attack success rate below 5%." The empirical safety signals are as follows.

\[
AUC_{MIA}\approx0.5,
\qquad
Adv_{MIA}\approx0
\]

Attack failure, however, is not a formal privacy guarantee.

## 22.2 Singling-Out, Linkability, and Inference

Using an Anonymeter-style attack-based framework, the following are evaluated [20].

- **Singling-out**: the risk of isolating an individual through a rare combination of attributes
- **Linkability**: the risk of linking records with different attribute sets to the same person
- **Inference**: the risk of inferring a sensitive attribute from known attributes

For each attack, the risk estimate and confidence interval are reported in comparison with a control dataset.

## 22.3 Nearest-Neighbor Memorization

- Distance to Closest Record (DCR)
- Nearest Neighbor Distance Ratio (NNDR)
- exact training record match
- comparison of training–synthetic distances with holdout–synthetic distances

DCR is a secondary metric and does not by itself guarantee privacy.

## 22.4 Attribute Inference

The attacker is given some quasi-identifiers and predicts the following sensitive attributes.

- In-hospital mortality
- Mechanical ventilation status
- Vasopressor use
- Abnormality of key laboratory results

The attack models are Logistic Regression, Random Forest, and XGBoost.

## 22.5 Subgroup Privacy

Beyond the overall average, the following subgroups are evaluated separately.

- Mortality group and survival group
- Sex
- Age band
- Race category
- Rare ICU types
- Patients with high missingness rates
- Patients with extreme laboratory values

Because attack risk may be elevated in rare subgroups, the sample size and confidence interval are reported alongside the risk.

## 22.6 Formal DP Results

For DP-CDG-QGAN, the following must be reported.

- \(\varepsilon\)
- \(\delta\)
- clipping norm
- noise multiplier
- sampling rate
- training steps
- type of accountant
- attack-based privacy score
- reduction in utility

Formal DP and empirical attack resistance are presented together and are not treated as substitutes for each other.

---

# 23. Statistical Analysis

## 23.1 Repeated Experiments

- All main models: at least 5 random seeds
- Key comparison models: 10 seeds where possible
- The same splits and the same preprocessing are used

\[
\bar M=\frac{1}{S}\sum_{s=1}^{S}M_s
\]

## 23.2 Confidence Intervals

- seed bootstrap
- test sample bootstrap
- nested bootstrap where necessary

95% confidence intervals are reported.

## 23.3 Hypothesis Testing

- paired permutation test
- Wilcoxon signed-rank test
- bootstrap difference interval
- effect size

When multiple metrics are tested simultaneously, the Holm correction is applied.

## 23.4 Non-Inferiority Analysis

To assess whether the quantum model attains comparable performance with fewer trainable parameters even if it is not absolutely superior, a non-inferiority margin is pre-specified.

Example:

\[
\Delta_{AUPRC}
=AUPRC_{CDG-QGAN}-AUPRC_{MLP}
\]

For a non-inferiority margin \(-\delta_U\), non-inferiority is concluded if the lower bound of the confidence interval exceeds \(-\delta_U\).

## 23.5 Primary Endpoints

| Evaluation axis | Primary endpoint |
|---|---|
| Fidelity | CDG edge-weighted association error |
| Utility | Mortality TSTR AUPRC |
| Clinical | Overall clinical violation rate |
| Privacy | DOMIAS attack advantage |
| NISQ | Noisy-to-ideal utility degradation |
| Resource | Transpiled two-qubit gate count and total trainable parameters |

All remaining metrics are classified as secondary endpoints.

---

# 24. Pre-Specified Success Criteria

The figures below are operational criteria established before the experiments, not predictions of the results.

## 24.1 Utility

\[
\frac{AUPRC_{TSTR}}{AUPRC_{TRTR}}\geq0.85
\]

or satisfaction of the pre-specified non-inferiority margin relative to the parameter-matched MLP.

## 24.2 Structure Preservation

- A reduction in CDG edge-weighted error relative to ring/random ansätze with the same number of edges
- Improvement in a consistent direction across multiple seeds
- Presentation of effect sizes and confidence intervals

## 24.3 Clinical Constraints

- Target of at least a 30% reduction in violation rate relative to the no-constraint ablation
- Absolute decrease in TSTR AUPRC within 0.03

## 24.4 Privacy

- Non-DP model: DOMIAS attack advantage must not be worse than that of the classical baselines
- DP model: confirmation of a trend of decreasing attack risk as \(\varepsilon\) decreases
- Subgroup risk reported separately so that it is not masked by the overall average

## 24.5 NISQ

- The sparse CDG ansatz is favorable relative to dense ansätze in terms of two-qubit gates and noise degradation
- Verification of whether the model ranking is completely reversed under finite-shot conditions

---

# 25. Principles for Interpreting the Expected Results

## 25.1 If CDG-QGAN Outperforms the Classical Models

The result is interpreted as meaningful only if the following conditions are met.

- Identical decoder and critic
- Parameter-matched or output-matched comparison
- Reproduced across multiple seeds
- Strong classical baselines included
- Wall-clock and circuit evaluation costs disclosed
- Performance maintained under NISQ noise

Even in this case, the phrasing used is "quantum-assisted inductive bias" or "parameter-efficient performance" rather than "quantum advantage."

## 25.2 If Performance Is Comparable

If non-inferior results are obtained with fewer parameters or a more interpretable circuit structure, the contribution can be framed as follows.

- resource-aware non-inferiority
- clinical structure preservation
- NISQ limitation characterization
- quantum latent bottleneck analysis

## 25.3 If Performance Is Lower Than the Classical Models

Even a negative result has research value provided the following analyses are sufficient.

- In which feature groups does the model fail?
- What is the relationship between circuit depth and gradient vanishing?
- Does the decoder ignore the quantum latent?
- Which metrics are most severely damaged by shots/noise?
- Is the CDG topology disadvantageous under hardware routing?
- What is the trade-off between parameter efficiency and absolute performance?

What matters is quantifying the applicable scale and the limitations without overstating the results.

---

# 26. Reproducibility and Research Management

## 26.1 Public Artifacts

- Data extraction SQL
- feature dictionary
- preprocessing pipeline
- CDG construction code
- quantum circuit definition
- baseline configuration
- evaluation pipeline
- privacy attack configuration
- raw metrics per seed
- environment lock file
- hardware/noise configuration

## 26.2 Experiment Tracking

The following are stored for each run.

- git commit hash
- configuration file
- random seed
- dataset split hash
- model parameter count
- circuit depth and gate count
- training loss
- validation metrics
- checkpoint selection reason

## 26.3 Restrictions on Data Release

Neither the raw MIMIC-IV data nor patient-level derived data are included in the repository. Release of the synthetic data is likewise not assumed to be automatically safe; it is decided only after the privacy audit and the DUA conditions have been checked.

---

# 27. Ethical Considerations

- Synthetic data are not material for actual patient care or patient decision-making.
- Statistical similarity does not imply clinical safety.
- A low attack score does not imply absolute anonymity.
- Utility and privacy for minority groups are evaluated separately.
- Clinical rules are restricted to published physiological definitions.
- In the absence of clinician review, the term "clinical plausibility" is used rather than "clinical validity."
- The race variable is interpreted restrictively as a data attribute that encompasses social and environmental factors rather than a biological causal factor.
- When generating conditioned on patient outcomes, the potential for stigmatization or subgroup distortion is discussed.

---

# 28. Recommended Scope for a Single-Author Study

## Included in the main body

- Static table of the first 24 hours after the first ICU admission in MIMIC-IV
- 30–45 mixed-type features
- CDG construction and qubit community mapping
- CDG-guided PQC
- Parameter-matched MLP generator
- CTGAN, CTAB-GAN+, TabDDPM, WGAN-GP
- Clinical constraint and dependency losses
- Comprehensive privacy audit
- Ideal, finite-shot, and noisy simulation
- Gate count and gradient analysis

## Extensions if capacity allows

- DP-CDG-QGAN
- Inference on a real quantum device
- External validation on eICU
- Expanded scope of the low-data augmentation experiments

## Recommended exclusions from the first paper

- Generation of full longitudinal trajectories
- Multimodal generation of clinical notes and imaging
- Concurrent development of quantum diffusion
- Combination with federated learning
- Multiple diagnostic conditions across several diseases
- Claims of real clinical deployment or safety

---

# 29. Journal Strategy

## 29.1 Realistic Primary Target

**Quantum Machine Intelligence**

Reasons for fit:

- hybrid quantum–classical machine learning
- quantum generative model
- application-driven quantum AI
- NISQ performance and resource analysis

Essential requirements:

- Clear novelty of the CDG ansatz
- Parameter-matched comparison
- Depth/noise/qubit ablation
- Strong classical baselines
- Quantum claims free of exaggeration

## 29.2 Stretch Target

**Quantum Science and Technology**

Elements likely to be additionally required:

- Results on real hardware
- Generalization to multiple datasets or non-medical tabular datasets
- Hardware-aware mapping
- A general methodological contribution regarding circuit structure itself

## 29.3 Ambitious Target

**PRX Quantum**

Realistically very difficult on the basis of a single MIMIC-IV simulator experiment alone. It would require generalizable theory, clear quantum resource insight, and strong hardware validation.

## 29.4 Pivoting to a Medical AI Journal

With the following additions, journals such as JAMIA, IEEE Journal of Biomedical and Health Informatics, and npj Digital Medicine could be considered.

- External validation on eICU
- Clinician review
- Expanded subgroup fairness and calibration
- Realistic research use scenarios
- Interpretation of results centered on medical significance

In medical journals, external validation and clinical usefulness are likely to be weighted more heavily than the quantum circuit itself.

---

# 30. Proposed Paper Structure

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

# 31. Core Novelty Statement

### Korean

> 본 연구는 의료 변수 사이의 안정적인 통계적·임상적 의존 관계를 특징 그래프로 구성하고, 해당 그래프를 매개변수화 양자 회로의 얽힘 토폴로지와 pairwise measurement 구조로 변환하는 CDG-QGAN을 제안한다. 동일한 decoder, critic, 손실함수 및 유사한 파라미터 규모를 갖는 고전 생성기와 비교함으로써, 의료 표형 데이터 생성에서 데이터 의존적 양자 회로가 제공하는 귀납 편향과 NISQ 한계를 정량화한다.

### English

> We propose CDG-QGAN, a hybrid quantum–classical generative model that transforms stable statistical and clinically grounded dependencies among ICU variables into the entanglement topology and pairwise measurement structure of a parameterized quantum circuit. Under matched decoder, critic, objective, latent dimensionality, and parameter budgets, we quantify whether the resulting data-dependent quantum circuit provides a useful inductive bias for mixed-type clinical tabular synthesis and characterize its limitations under NISQ constraints.

---

# 32. Summary of the Final Research Direction

What determines the scholarly value of CDG-QGAN is not the claim that "a quantum GAN was used on medical data." The following three points must be central.

1. **A clear methodology for converting a Clinical Dependency Graph into a quantum entanglement topology**
2. **A fair, controlled comparison against a classical generator under matched resource conditions**
3. **An integrated evaluation spanning fidelity, clinical plausibility, utility, privacy, and NISQ resources**

By maintaining this structure, the work can be positioned not as a simple medical-application QGAN but as a study in **data-dependent quantum generative modeling** that reflects the structure of medical variables in quantum circuit design.

---

# References

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

## Cautions on Using This Document

- This document is a research **plan**; it does not assert experimental results in advance.
- The CDG in `CDG-QGAN` is not a causal graph.
- The non-DP base model is described as `privacy-audited`.
- Before submission to an SCI/SCIE journal, re-verify the data version, library versions, quantum backend, and the most recent prior work used.

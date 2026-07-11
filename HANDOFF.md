# CDG-QGAN Implementation Work Specification (for Codex)

Last updated: 2026-07-11 · Design, review, and interpretation by the PM; implementation by Codex

This document states only **what must be built and why** and **what the acceptance criteria are**.
How to implement it is at Codex's discretion.

---

## 0. Read these first

| Document | Contents |
|---|---|
| `CDG-QGAN_research_plan_v2_ko.md` | Research plan v2 (the reference document) |
| **`REVISIONS.md`** | **What the experiments overturned in v2. Where it conflicts with the plan, this takes precedence.** |
| `RESULTS_precheck.md` | Validation of the CDG alignment effect (real data) |
| `RESULTS_ceiling.md` | Circuit expressivity ceiling · light-cone validation |
| `DATA.md` | Data locations, environment pitfalls |
| `scripts/*` | Validated reference implementations. Extend them rather than rewriting. |

### Environment (must read)

conda env **`cdg-qgan`**. **numpy is pinned at 2.2.6. Never upgrade it.**
torch 2.11+cu128 is built against the numpy 2.2 ABI, so upgrading to 2.4 breaks the `shm.dll`
load (`OSError: [WinError 127]`). This is exactly how the base conda environment was destroyed.

```bash
python -m pip install -c .backup/constraints.txt <package>   # always apply the constraints file
```

Data paths are managed in one place, by `scripts/paths.py`.
**Both the raw data and everything derived from it live in the shared archive
(`D:\pythondata\Medical Data`). Never place patient data inside the code project** (DUA).

---

## 1. Already done (no rework needed)

| Item | Result |
|---|---|
| MIMIC-IV v3.1 cohort extraction | **51,668 patients / 5,350 deaths (10.4%)**. 48,561 with all 16 features fully observed |
| CDG estimation | 16 nodes · 23 edges · Δ≤3 · diameter 5 |
| **Alignment-effect precheck** | **`L=1`: z=+3.18, p=0.0004 PASS** (only 2 of 5,000 permutations beat the CDG) |
| Circuit expressivity ceiling | `L=1` adjacent pairs **0.991** → strong clinical relations are representable. `d≥3` is ~0.01 |
| light-cone corollary | **Numerically demonstrated.** The cliff falls exactly at `d=2L` |
| Head non-mixing | The Jacobian is exactly diagonal → classical parameters cannot create dependencies |
| light-cone simulator | **4096x saving** at n=16, L=1; agrees with the full simulator to 1e-6 |
| MAP/Hct identity removal | `REVISIONS.md` C-3 |

**No open items. The design is fully settled by the data.**

---

## 2. Where this conflicts with research plan v2, follow this document

Five things the experiments overturned. Evidence is in `REVISIONS.md`.

1. **`L=1` only.** `L=2` FAILS on the real data (z=+0.66, p=0.287), and `L=3` makes every pair
   reachable, so it is **null by definition**. → **No confirmatory experiments at `L=2` or `L=3`.**
2. **The primary endpoint is the dependency error over all 120 pairs, not over held-out edges.**
   Measured on held-out edges alone, the CDG comes out worse than a random permutation.
3. **The condition vector is `c = (mortality, age, sex, ICU type)`** — not `y` alone.
4. **The dependency metric is computed in the nonparanormal space.** Pearson correlation in raw
   units is forbidden.
5. **MAP is not a generated variable** (it is an arithmetic identity). Evaluation only.

---

## 3. Work packages

> **Order: WP-6 → WP-2 → WP-3 → WP-4 → WP-5**
> WP-1 (the data pipeline) is complete.

### WP-6. Simulator performance ← **first**

`qsim_lightcone.py` is **correct but slow.** The bottleneck is not arithmetic but **kernel-launch
overhead**. At `L=1, Δ≤3` a cone spans 4 qubits (16 dimensions), so the tensors are tiny, yet we
loop over 16 qubits × ~10 gates in Python, producing hundreds of micro-kernels per step.
(Measured: 83% of GPU time is spent waiting on launches, not computing.)

**Measured evidence**: in `benchmark_synthetic.py`, a single variant (3 seeds) takes **1.5+ hours**.
Completing 5 variants would need 7+ hours → aborted. This is untenable for the 90 training runs
of the confirmatory experiment.

**Optimization directions** (the numbers must not change)
1. **Batch the 16 cones onto a single axis.** All are ≤4 qubits, so they can be processed
   simultaneously as one `(B, 16, 16)` tensor. Pad to the maximum size where cone sizes differ.
2. Vectorize the single-qubit gates over the cone axis.
3. Eliminate launch overhead with `torch.compile` or CUDA graphs.

**Acceptance criteria**
- `qsim_lightcone.verify_against_full()` passes (agrees with the existing implementation to `<1e-5`)
- A 3000-step `L=1` training run finishes in **under 1 minute per seed**
- After optimization, **rerun `benchmark_synthetic.py`** → check whether `aligned < permuted`
  holds when the true graph is known (validation of the training stage of the confirmatory design)

---

### WP-2. Confirmatory experiment

**Depth fixed at `L=1`.** 10 seeds.

| Model | Purpose |
|---|---|
| CDG-QGAN | The proposed model |
| **isomorphic permuted-CDG × 3** | **Confirmatory control.** Destroys clinical alignment only |
| **distance-matched permuted × 3** | Also matches the held-out distance distribution. Tests whether the effect goes beyond mere distance placement |
| degree-preserving rewired | Abstract topology effect |
| ring (resource-matched) | Uniform-topology baseline |
| no-entanglement | Necessity of entanglement |

All identical in resources: edge count, depth, parameters, critic, loss, step budget.

**Primary endpoint**: conditional dependency error over all 120 pairs (nonparanormal space, Fisher-z).
It must catch both **false negatives** (dependencies not produced) and **false positives**
(dependencies manufactured where none exist).
The held-out edge error is secondary.

Run `L=2` **as a negative control only**, to confirm that the alignment effect vanishes
(expected: `z: +3.18 → +0.66 → 0.00`). **This decay curve is a figure in the paper.**

**Acceptance criteria**
- `Δ = mean(CDG) − mean(permuted) < 0`, with a hierarchical bootstrap 95% CI that excludes 0
- The result must not depend on any single particular permutation
- The result must not have been obtained at the cost of markedly degraded marginal fidelity

---

### WP-3. Classical controls (isolating the quantum contribution)

| Model | Purpose |
|---|---|
| CDG-local classical message-passing (GNN) | **The most important one.** Same receptive field, comparable parameter count |
| Random Fourier Feature generator | Tests whether a simple nonlinear feature map explains it |
| Global MLP-WGAN | Absolute-quality context (do not use as evidence of a quantum contribution) |
| TabDDPM | Modern classical baseline (context figure) |

**Interpretation rule (v2 §18.2)**: only if CDG-QGAN achieves a lower error than the GNN may we
make a limited claim of "an additional benefit from the quantum core". **If they are equivalent,
the conclusion is restricted to "the graph-local inductive bias was effective, but there is no
evidence of an additional benefit from the quantum circuit."** That result is also a paper.

> **PM judgment**: the GNN will probably win or tie. In that case the axis of the paper becomes
> the **parameter–performance Pareto frontier** (the quantum core attains the same receptive
> field with only ~23 entangling angles).
> Prepare that figure as a **main Results figure**, not as a side note in the Discussion.

---

### WP-4. Evaluation suite

- **Dependency**: error over all 120 pairs (primary), stratification by distance, held-out
  (secondary), precision matrix edge recovery, **false positive rate** (spurious dependencies
  manufactured on true non-edges)
- **Marginals**: Wasserstein-1, KS, quantile error
- **Multivariate**: MMD, energy distance, real-vs-synthetic AUROC
- **Clinical plausibility**:
  - The ordering relation `DBP ≤ MAP ≤ SBP`
  - **Compute `MAP~ = (SBP~ + 2·DBP~)/3`** and compare it against the real MAP distribution
    (MAP is not generated. Computing it and matching it is the stricter check.)
  - The goal is not to drive the violation rate to zero. **Minimize `|VR_syn − VR_real|`.**
- **Utility**: TRTR / TSTR (LR, RF, XGBoost, MLP) — AUROC, AUPRC, **Brier, ECE**
  - **Treat `TSTR > TRTR` as a red flag, not a success** (it means the conditional generation
    over-separated the classes). Always inspect calibration alongside it.
- **Privacy**: do not compare on a single MIA AUC. Present a **privacy–fidelity Pareto frontier**
  instead. (This avoids the confound in which a model that failed to learn automatically looks safe.)

---

### WP-5. NISQ

- finite-shot `S ∈ {256, 1024, 4096}` — applied **to the final model only**
- **Because every Z and ZZ is diagonal in the computational basis**, all of them can be estimated
  simultaneously from **a single set of Z-basis bitstrings**. The number of measurement bases does
  not grow with the number of edges. State this explicitly in the paper.
- shot-noise-aware fine-tuning: use a **correlated** surrogate `ε ~ N(0, Σ_O/S)`, not independent
  Gaussians
  > **PM observation**: `Cov(Ẑ_u, Ẑ_v) = (⟨Z_uZ_v⟩ − μ_uμ_v)/S` has **the same sign as the true
  > correlation.** That is, shot noise **systematically inflates** the dependencies in the output.
  > This can produce an artifact in which the metric looks better at low S. Derive the bias term
  > analytically and report it.
  > It is a small but genuine contribution.
- Report the transpiled two-qubit gate count and depth, the SWAP count, and the CDG edge
  preservation rate
- Use the term `barren plateau` only when exponential scaling in the number of qubits has actually
  been verified

---

## 4. Do not do

- **Confirmatory experiments at `L=2` / `L=3`** — at `L=3` all 120 pairs fall inside the light cone,
  making the CDG and the permuted graphs literally identical. `L=2` also FAILS on the real data.
  (`L=2` is run **as a negative control** only.)
- **Judging the primary endpoint on held-out edges alone** — the CDG comes out worse than a random
  permutation
- **Measuring dependency with Pearson correlation in raw units** — marginal errors contaminate the
  metric
- **Putting MAP back among the generated variables** — it is an arithmetic identity (R²=0.860) and
  it induces a spurious negative correlation between SBP and DBP (the sign of ρ flips, −0.508 ↔ +0.499)
- **Using hematocrit as a substitute variable** — `Hct ≈ 3×Hb` (r=0.962). The same identity
- **Using `r_Q` (the quantum parameter fraction) as a defense** — at 3.5% it backfires.
  Use instead: **"100% of the parameters that can create dependencies are quantum"**
- **Interpreting an advantage over CTGAN/TabDDPM as evidence of a quantum contribution** — the loss
  formulations differ, so these are context figures only
- **Training with the full statevector** — the light-cone subcircuit gives exactly the same values
  4096x more cheaply

---

## 5. Report to the PM immediately

1. **`verify_against_full()` fails after the WP-6 optimization** → it means the numbers changed.
   Stop immediately.
2. **`aligned` comes out worse than `permuted` in `benchmark_synthetic.py`** → it means there is a
   problem with the confirmatory design. Do not burn more GPU; report it.
3. **`TSTR > TRTR` is observed** → the conditional generation may have over-separated the classes.
4. **CDG-QGAN loses to the graph-local GNN** → switch the axis of the paper to the
   parameter–performance Pareto frontier.
5. **A new arithmetic identity is found** (R² > 0.8 between variables) → the variable set must be
   revisited.

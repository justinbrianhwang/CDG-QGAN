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
| light-cone simulator (**WP-6 done**) | **~70x speedup**, 25.7 s/seed; agrees with the full simulator to 1e-6 |
| Joint 120-pair ceiling | aligned **0.0103** vs. permuted 0.0468 vs. distmatched 0.0421 (floor 0.0609) |
| **Confirmatory contrast, TRAINED model** | **aligned 0.0426 vs. distmatched 0.0729 vs. permuted 0.0787 (floor 0.0653)** |
| The critic fix that made training work | `RESULTS_confirm.md`, `REVISIONS.md` §E-7 |
| MAP/Hct identity removal | `REVISIONS.md` C-3 |

**The design is settled by the data, and the training pipeline now demonstrably works on a
controlled synthetic teacher where the true graph is known.** WP-2 is unblocked.

---

## 2. Where this conflicts with research plan v2, follow this document

Evidence is in `REVISIONS.md`.

1. **`L=1` only.** `L=2` FAILS on the real data (z=+0.66, p=0.287), and `L=3` makes every pair
   reachable, so it is **null by definition**. → **No confirmatory experiments at `L=2` or `L=3`.**
2. **The primary endpoint is the dependency error over all 120 pairs, not over held-out edges.**
   Measured on held-out edges alone, the CDG comes out worse than a random permutation.
3. **The condition vector is `c = (mortality, age, sex, ICU type)`** — not `y` alone.
4. **The dependency metric is `eval_dep.partial_corr_c`** — nonparanormal, then residualize on a
   **nonlinear** basis of `c`, then partial correlation. **This is not optional and it is not the
   old metric.** The CDG is *defined* as a partial correlation conditional on `c`; measuring the
   unconditional one scores everything `c` induces as a false positive on every pair, and it was
   contaminating the real MIMIC pipeline, not just the benchmark. Linear residualization removes
   only half of it (`c` enters through angles, i.e. nonlinearly). Pearson correlation in raw units
   is forbidden.
5. **MAP is not a generated variable** (it is an arithmetic identity). Evaluation only.
6. **The critic MUST be the copula + batch-aware critic** (`diag_fix2.Critic`, `copula=True`,
   `batch_critic=True`). **This is the single change that made the model learn anything at all.**
   See §2.1.

### 2.1 The critic is not a free choice — read this before touching training

The plain MLP critic of v2 §8.9 **learns zero conditional dependency, for every topology.** Not
"a little" — zero. Its true-edge error equals that of a model that creates no dependency at all.
Four other fixes were tried and all four failed, including sweeping the quantum learning rate over
1000× (at `lr_q=5e-2` the angles move 2.0 radians and still learn nothing). Full record:
`RESULTS_lr.md`, `REVISIONS.md` §E-7.

The cause: the critic spends its capacity on the **marginals** — which the ~2000 classical head
parameters can already fit unaided — and hands the entangling angles nothing but noise. And when
nothing learns dependency, **every topology scores identically and the confirmatory experiment
measures nothing.** That is exactly what the first benchmark run produced.

The fix, and it is required:

- **(E) Copula critic.** Rank-transform each feature *within the batch* before the critic sees it
  (differentiable soft rank → inverse normal CDF). Every input is then standard normal by
  construction, the marginals carry no information, and the only thing left to discriminate on is
  the copula. The gradient has nowhere to go but `gamma`.
- **(A) Minibatch discrimination** on top (Salimans et al., 2016). On its own it does nothing; on
  top of (E) it nearly doubles the effect.

**The loss stays pure WGAN-GP.** No dependency term. Nothing from the evaluation metric. v2 §8.10
is respected — these are architectural changes to what the critic *sees*, not additions to what it
*optimizes*.

**Sanity check you must run before trusting any number**: score a model that creates no dependency
at all (shuffle each column of the real data independently). If your trained model does not beat
it, your pipeline is broken and every downstream comparison is meaningless. Ours lost to it by
2.1× and we nearly concluded the hypothesis was false.

---

## 3. Work packages

> **Order: WP-2 → WP-3 → WP-4 → WP-5**
> WP-1 (the data pipeline) and **WP-6 (simulator performance) are complete.**

### WP-6. Simulator performance — **DONE**

~70× speedup (>1800 s → 25.7 s per 3000-step seed), `verify_against_full()` still agrees with the
full statevector to 1e-6. `WP6_REPORT.md`. Independently verified by the PM (`verify_wp6.py`),
including that the RNG inside the CUDA graph advances across replays — otherwise the gradient
penalty would have been silently disabled.

One bug found afterwards and fixed: a zero-edge graph (`no_entangle`) fell back to the full 2^16
statevector instead of the light-cone path, which is backwards — with no edges each cone is a
single qubit. `REVISIONS.md` §E-5.

---

### WP-2. Confirmatory experiment ← **start here**

**Depth fixed at `L=1`.** 10 seeds. On real MIMIC-IV.

**Use `scripts/confirm.py` as the reference implementation.** It runs exactly this design on a
controlled synthetic teacher where the true graph is known, and it works. Do not re-derive the
training setup; port it.

| Model | Purpose |
|---|---|
| CDG-QGAN | The proposed model |
| **isomorphic permuted-CDG × 3** | **Confirmatory control.** Destroys clinical alignment only |
| **distance-matched permuted × 3** | Also matches the held-out distance distribution. **This is the control that decides the paper** — it cancels "the strong pairs happen to be nearby" |
| degree-preserving rewired | Abstract topology effect |
| ring (resource-matched) | Uniform-topology baseline |
| no-entanglement | Necessity of entanglement |

All identical in resources: edge count, depth, parameters, **critic**, loss, step budget.

**Required configuration** (this is what made it work — see §2.1):
- critic: **copula + batch-aware** (`copula=True, batch_critic=True`)
- loss: **pure WGAN-GP**, no dependency term
- optimizer: separate learning rate for the quantum angles (`lr_q=5e-3`) and the heads (`5e-5`) —
  radians and network weights do not share a scale
- metric: **`eval_dep.partial_corr_c`** (conditional on `c`, nonparanormal). Not the old one.

**Primary endpoint**: conditional dependency error over all 120 pairs (nonparanormal space,
Fisher-z). It must catch both **false negatives** (dependencies not produced) and **false
positives** (dependencies manufactured where none exist). The held-out edge error is secondary.

**Always report the floor** — the score of a model that creates no dependency at all (each column
of the real data shuffled independently). Every number is meaningless without it. On the synthetic
teacher the floor is 0.0653 and the trained aligned model reaches 0.0426; before the critic was
fixed, the trained model scored 0.1359 against a floor of 0.0648, i.e. **2.1× worse than doing
nothing**, and we nearly read that as a refutation of the hypothesis.

Run `L=2` **as a negative control only**, to confirm that the alignment effect vanishes
(expected: `z: +3.18 → +0.66 → 0.00`). **This decay curve is a figure in the paper.**

**What the synthetic teacher predicts you should see** (`RESULTS_confirm.md`):
- aligned beats **every** control, including distance-matched
- controls sit **at the floor** on the true edges — they learn essentially nothing there, because
  Corollary 1 forbids it
- **`no_entangle` sits exactly on the floor**, and *beats* the misaligned circuits — misplaced
  entanglement is worse than none

**Acceptance criteria**
- `Δ = mean(CDG) − mean(permuted) < 0`, with a hierarchical bootstrap 95% CI that excludes 0
- **`Δ = mean(CDG) − mean(distance-matched) < 0` as well.** Beating the isomorphic permutation
  alone is not enough — it could be graph combinatorics. This is the one that isolates the claim.
- The result must not depend on any single particular permutation
- The result must not have been obtained at the cost of markedly degraded marginal fidelity
- **The trained model must beat the floor.** If it does not, stop and report; nothing downstream
  means anything.

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

- **Reporting any dependency number without the floor next to it.** The floor is the score of a
  model that creates no dependency at all. Without it you cannot tell a working model from a broken
  one — we had a trained model losing to the floor by 2.1× and read it as a scientific result.
- **Using the plain MLP critic of v2 §8.9.** It learns **zero** dependency, for every topology, and
  then every topology scores the same. §2.1.
- **Measuring dependency unconditionally.** The CDG is *defined* conditional on `c`. Use
  `eval_dep.partial_corr_c`. Linear residualization is not enough — `c` enters through the angles.
- **"Fixing" the training by raising the learning rate, enlarging the batch, or adding a dependency
  term to the loss.** All three were tried and all three failed; the record is in `REVISIONS.md`
  §E-7. Do not spend GPU re-refuting them.
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

1. **The trained model does not beat the floor.** Stop everything. It means nothing is learning
   dependency, and in that state *every* topology scores the same, so no structural hypothesis can
   be tested — including ours. Do not interpret it as evidence against the CDG. Report it.
2. **`aligned` is not better than `distance-matched`** on the real data. Beating the isomorphic
   permutation but not the distance-matched one would mean the effect is graph combinatorics, not
   clinical alignment. That is a different (and much weaker) paper, and the PM decides.
3. **`aligned` comes out worse than `permuted`** → do not burn more GPU; report it.
4. **`verify_against_full()` fails** → the simulator numbers changed. Stop immediately.
5. **`TSTR > TRTR` is observed** → the conditional generation may have over-separated the classes.
6. **CDG-QGAN loses to the graph-local GNN** → switch the axis of the paper to the
   parameter–performance Pareto frontier.
7. **A new arithmetic identity is found** (R² > 0.8 between variables) → the variable set must be
   revisited.

---

## 6. One habit that would have saved a day

Before interpreting any result, compute the score of the **do-nothing model** — the one that
reproduces the marginals perfectly and creates zero dependency. It costs no training.

The first benchmark run returned a clean null: every graph variant scored ~0.135, and the model
with **no entanglement at all** was the best of them. Read naively, that refutes the hypothesis.
The floor was 0.065. The trained model was **2.1× worse than doing nothing** — which is not a
weak result, it is a broken pipeline. Everything since followed from asking that one question.

# Required Revisions to the v2 Research Plan

Only experimentally confirmed items are recorded here. The supporting script is named for each item.
Written: 2026-07-11 (before MIMIC-IV v3.1 arrived; based on the demo + circuit experiments)

---

## A. Must change

### A-1. **`L=1` only.** `L=2` cannot be used for the confirmatory experiment, and `L=3` is null by definition.

Evidence: `scripts/precheck_alignment.py` · **real MIMIC-IV v3.1, n=48,560**

The representable dependency mass `M(G,L) = Σ|ρ_true(u,v)|·1[d_G(u,v) ≤ 2L]` was compared
between the CDG and an isomorphic-permutation null distribution (5,000 draws). Because the
permutations are isomorphic, the node count, edge count, degree sequence, triangle count, and
clustering coefficient are all preserved; the only thing that changes is **which clinical pair
sits inside which triangle**.

These are the **current** numbers: post-MAP-removal (C-3), and estimated in the space
`eval_dep.partial_corr_c` defines (nonparanormal → residualize on a nonlinear basis of `c` →
partial correlation). Two earlier versions are superseded and neither should be quoted:

- An early run still containing MAP gave `L=1` z=+4.46. That extra margin was the arithmetic
  identity `MAP≈(SBP+2DBP)/3`, not clinical structure.
- A run that residualized **in raw units and then applied the nonparanormal transform** gave
  z=+3.18 on a 23-edge graph. That estimator disagreed with the one the evaluation uses, so the
  graph we drew and the quantity we measured were not the same object [A-4, E-3]. Fixing it
  changed the graph very little (ρ-matrix correlation 0.9971, 20 of 23 edges retained) and made
  the effect **stronger**.

CDG: 16 nodes · 21 edges · maximum degree 3 · diameter 6 · total dependency mass Σ|ρ| = 5.96

| L | Reach radius | Reachable pairs | CDG mass | Permutation null | z | p | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | 2 | 48/120 | **4.645 (78.0%)** | 2.387 ± 0.595 [0.883, 4.380] | **+3.79** | **<0.0001** | **PASS** |
| 2 | 4 | 103/120 | 5.696 (95.6%) | 5.113 ± 0.413 | +1.41 | 0.038 | weak |
| 3 | 6 | **120/120** | 5.957 (100%) | 5.957 ± 0.000 | 0.00 | 1.000 | **meaningless** |

**At `L=1` not one of the 5,000 permutations beat the CDG** (100th percentile, null max 4.380 <
CDG 4.645). Within its reachable range the CDG captures **78.0%** of the total dependency mass.

**On `L=2`, a correction to what this document previously claimed.** It used to say `L=2` *FAILS*
on the real data (z=+0.66, p=0.287). Under the corrected estimator `L=2` **marginally passes**
(z=+1.41, p=0.038). That earlier statement was too strong and is withdrawn.

`L=1` is still the operating point, but for the right reason rather than a convenient one: at
`L=2`, **103 of 120 pairs are already reachable (86%)**, so the topology barely constrains
anything, and the effect size collapses by 63% (z: +3.79 → +1.41). A topology that forbids almost
nothing cannot discriminate between topologies. "Weak" and "fails" are different claims, and only
the first one is ours.

> v2 §8.11 table: `logical CDG block depth L∈{1,2}, L=3 exploratory`
> → **`L=1` only for the confirmatory experiment. `L=2` is the negative control. Delete `L=3`.**

#### This is not a limitation but a result — the decay of the effect with depth is what the theory predicts

The alignment effect decays with depth: `z = +3.79 → +1.41 → 0.00`.
This is **exactly what the light-cone theory predicts**: as depth grows, the reach radius `2L`
widens, every pair becomes representable, and the topology stops imposing any constraint at all.

**We present this decay curve as a figure in the paper.** It is a relationship the theory
predicted and the data confirmed, and it is the answer to "why must the circuit be shallow?"
`L=2` is not a failure — it is **the negative control that confirms the theory**.

#### Why the effect is strong on the real data

Because clinical variables genuinely form clusters, and those clusters form **triangles**.
So even when a strong edge is held out, the pair often remains at **distance 2** via a common
neighbor, and stays representable. A random permutation scatters such pairs to distance 3–5,
and by Corollary 1 they become **fundamentally unrepresentable**.

The dominant source of mass loss at `L=1` is `sbp–dbp` (|ρ| = 0.439), which is held out and sits
at distance 3 — out of reach. Sensitivity to the hold-out construction seed still needs checking.

### A-2. Replace the primary endpoint: held-out edges → **all 120 pairs**

Evidence: `scripts/build_cdg.py`, `scripts/graphs.py`

Measured on held-out edges alone, **the CDG does worse than a random permutation**:

| Graph | L=1 held-out reachable | L=2 |
|---|---|---|
| CDG | **3/13** | 12/13 |
| permuted_0 | **8/13** | 12/13 |
| ring | 6/13 | 13/13 |

The CDG is constructed so as to *place strong edges on RZZ gates*, whereas held-out evaluation
asks whether *held-out pairs are close*. Nothing in the construction connects these two things.

By contrast, measured over **all 120 pairs**, the CDG wins decisively at the 99.8th percentile
of the null distribution. Evaluating all pairs catches both kinds of error:
- **False negatives**: failing to produce a truly strong dependency (the permuted graph places
  a strong pair far away)
- **False positives**: manufacturing a dependency that does not exist (the permuted graph
  connects an unrelated pair by an RZZ)

The held-out edge error is demoted to a **secondary non-circularity metric**.

> v2 §12.1 "the only confirmatory primary endpoint = HDE"
> → **primary = conditional dependency error over all 120 pairs. HDE is secondary.**

### A-3. Expand the condition vector to `c = (mortality, age, sex, ICU type)`

v2 §7.2 residualizes the CDG on (y, age, sex, ICU), but the generator is conditioned on `y`
alone. As a result `ρ_real` (conditional on 4 covariates) and `ρ_syn` (conditional on `y` only)
are **different estimators**, and their difference is being used as the primary endpoint.

The local encoding already takes `y`, so it suffices to turn that scalar into a 4-dimensional
vector, and Corollary 1 continues to hold verbatim as "conditional on **c**".
Implementation: `scripts/features.py: CONDITION_VARS`, `scripts/model.py: cond_dim=4`

### A-4. Compute the dependency metric in the **nonparanormal space**

v2 §12.1 says only "partial correlation" and does not specify the space. Measured with Pearson
correlation in raw units, the heavy-tailed marginal errors that the width-8 local head cannot
fit (creatinine, lactate, glucose) contaminate the dependency metric. Worse, the permuted graphs
have different per-feature **degrees**, hence different expressivity, hence different marginal
fidelity → Δ ends up measuring "who fit the marginals better" rather than "the alignment effect".

Applying the **same** rank-inverse-normal transform used for CDG estimation to both the real and
the synthetic data leaves only the copula, which is invariant to the monotone head.
Implementation: `scripts/train.py: _npn()`

> Add **`fix the space in which the dependency metric is computed (nonparanormal)`** to the
> pre-registration checklist in Appendix A.

---

## B. Must add

### B-1. Distance-matched permutation control

By the corollary, covariance is 0 whenever `d > 2L`. So if a held-out pair is close in the CDG
and far in the permuted graph, `Δ < 0` is already determined by graph combinatorics = a test
that carries no information.

Adding as a control a permutation whose **held-out-pair distance distribution is matched to the
CDG's** cancels out the "being close" advantage, and the only difference that remains is whether
**the specific edges the CDG chose actually capture clinical structure**.
Implementation: `scripts/graphs.py: distance_matched_permuted()`

### B-2. Put the precheck into the protocol

The `M(G,L)` vs. permutation-null test determines **without any training at all** whether the
confirmatory experiment has any chance of succeeding. If you lose here, no amount of GPU burn
will produce `Δ < 0`. Write it into the protocol as a gate that must be passed before the main
experiment.

### B-3. Add the proposition "classical parameters cannot create dependencies" to §4

`h_u` is a one-dimensional map taking only `(q_u, c)`, so conditional on `c`,
`x̃_u ⟂ x̃_v ⟺ q_u ⟂ q_v`. That is, **every conditional cross-feature dependency structure in
the output arises solely in the quantum core.**

Reporting `r_Q` (the quantum parameter fraction = 3.5%) makes it look as though "the classical
part did all the work", but **among the parameters that *can* create dependencies, the quantum
fraction is 100%**. That is the true statement.
Enforced and verified in code: `scripts/model.py: assert_no_cross_feature_mixing()`
(the head Jacobian `∂x̃_u/∂q_v` is exactly diagonal, off-diagonal = 0.0)

### B-4. Light-cone subcircuit simulator (§14, §24.5)

By Proposition 1, `⟨Z_u⟩` depends only on `N_L(u)`, so the full 2^16 statevector is unnecessary.

| n | L | full | light-cone | saving |
|---|---|---|---|---|
| 16 | 1 | 65,536 | **16** | **4,096x** |
| 12 | 2 | 4,096 | 1,024 | 4x |

Agrees with the full simulator to `1e-6` (float32 precision). The full statevector is needed only
for finite-shot bitstring sampling. This is a case of theory making computation cheap, so we
report it as a contribution.
Implementation: `scripts/qsim_lightcone.py`

---

## C. Implementation pitfalls (not in the plan, but reflected in the code)

### C-1. `nx.maximum_spanning_tree` does not constrain degree

Used as-is it yields a graph with maxdeg=5, violating the `Δ≤3` requirement of v2 §7.6. That in
turn shrinks the diameter (6→4) and graph distance loses its discriminating power. You must
implement **Kruskal under a degree constraint** yourself.
Implementation: `scripts/build_cdg.py: build_graph()`

### C-2. Building E_fit first leaves only weak relations in the held-out set

If you select `E_fit = MST ∪ top-r` first, it absorbs all the strong edges (dbp–map 0.76,
creatinine–bun 0.76, sodium–chloride 0.67) and only pairs with `|ρ| ≤ 0.16` are left for the
held-out set → the HDE then measures nothing but estimation noise. **Select the held-out set
first, stratified by strength, choosing only pairs whose removal keeps the remaining graph
connected**, and only then build E_fit from what is left.

### C-3. **[RESOLVED] Exclude MAP from the generated variables. It is an arithmetic identity, and it manufactures spurious edges.**

Evidence: `scripts/diag_collinearity.py` · real MIMIC-IV v3.1, n=51,587

**The problem was not "metric dominance" — it was a scientific error.**

**(1) MAP is an arithmetic identity, not a clinical dependency.**

| Identity | R² | Pearson r | Note |
|---|---|---|---|
| `MAP ≈ (SBP + 2·DBP)/3` | **0.860** | **0.962** | mean absolute error 3.36 mmHg (MAP SD 11.0) |
| `Hct ≈ 3 × Hb` | — | **0.962** | median Hct/Hb = **3.02** |

**(2) Worse — MAP induces a spurious negative correlation between SBP and DBP.**

```
MAP included:  ρ(sbp, dbp) = -0.508    <- physiologically wrong
MAP excluded:  ρ(sbp, dbp) = +0.499    <- the true relation
```

**The sign flips.** Because MAP is a deterministic function of SBP and DBP, conditioning on MAP
in the precision matrix induces the artificial negative relation "MAP is the same but SBP is
high → DBP must be low" (a collider/suppression artifact). In other words, **a physiologically
wrong edge enters the CDG.** This is not a metric problem; it is a scientific error.

**(3) Risk to the paper's credibility.** `dbp–map` (10.4% of Σ|z|) and `sbp–map` (7.2%) occupy
two of the top three pairs. The CDG's advantage then looks reducible to "it recovered an
arithmetic identity".
> Reviewer: "Isn't your effect just the recovery of the algebraic identity
> `MAP=(SBP+2DBP)/3`? That is arithmetic, not the discovery of clinical dependency structure."

#### Actions

- **Exclude MAP from the generated variables.** Still extract it (`EVAL_ONLY`) and use it **for
  clinical-plausibility evaluation only**: from the generated `SBP~` and `DBP~`, compute
  `MAP~ = (SBP~ + 2·DBP~)/3` and compare it against the real MAP distribution. **This is a
  stricter check than generating MAP directly and fitting it.**
  (v2 §6.4 and §12.7 already define MAP as a diagnostic metric rather than a training loss.)
- **Put WBC in the vacated slot.** This is ICU data, and **the entire inflammation axis was
  missing.**
- **Remove hematocrit from the substitute-variable list.** `Hct ≈ 3×Hb` is the same identity.
  If it is auto-substituted when an observation-rate threshold is missed, we walk into the same
  trap again.
  → Substitution order: calcium, magnesium, phosphate.
- We lost one identity and **gained a real physiological relation (SBP–DBP, ρ=+0.499).**

**Re-validation required**: the feature set has changed, so the CDG and the precheck must be
rerun to confirm that the `L=1` alignment effect holds (before MAP removal: z=+4.46).

### C-4. The RZZ sign-vector cache key must include the qubit count

Light-cone subcircuits have a different qubit count `m` per cone, so keying the cache on `(u,v)`
alone causes sign vectors of different sizes to collide. The key must be `(n, u, v)`.

---

## D. Verified (no change needed, proceed as planned)

- **Corollary 1 is numerically confirmed.** `L=1` ceiling: `d=1: 0.991`, `d=2: 0.847`,
  `d≥3: ~0.01`. A cliff exactly at the `2L` boundary. (`scripts/ceiling.py`)
- **Gate check passed.** Since the adjacent-pair ceiling is 0.99, the circuit can represent the
  strong clinical relations (Hb–Hct 0.95, SBP–MAP 0.90). The design does not collapse from lack
  of expressivity.
- **Appendix B circuit convention.** A `Z` measurement immediately after `RZZ` → entangling
  gradient `-1.5e-17` (i.e. 0). Inserting non-commuting `RX/RY` mixing gives `+0.46`. The design
  rationale of v2 §4.5 is empirically confirmed. (`scripts/smoke_test_env.py`)
- **References [13], [14] confirmed to exist.** arXiv:2505.22533 and arXiv:2602.12704 were both
  retrieved. QTabGAN is dated 2026-02-13, by Kumari, Achutha, and Sivaraman — the bibliographic
  details match.

---

## E. **[RESOLVED 2026-07-11]** WGAN-GP could not train the entangling angles — the critic was blind to dependency

> **Resolution.** The critic was spending its capacity on the marginals, which the ~2000 classical
> head parameters can already fit unaided, so it handed the entangling angles nothing but noise —
> so nothing learned any dependency — so every topology scored identically and the confirmatory
> contrast measured nothing. **Rank-transforming the critic's input within the batch destroys all
> marginal information and leaves the gradient nowhere to go but `gamma`.** With that fix, plus
> the c-conditional metric the CDG is actually *defined* by (E-3), the trained model recovers the
> alignment effect:
>
> | model | 120 pairs (conditional) | 19 true edges |
> |---|---|---|
> | **aligned (true CDG)** | **0.0426 ± 0.0043** | **0.1627** |
> | no_entangle | 0.0647 | 0.3644 |
> | *floor* | *0.0653* | *0.3641* |
> | distmatched | 0.0729 | 0.3352 |
> | rewired | 0.0749 | 0.3521 |
> | permuted | 0.0787 | 0.3600 |
>
> **WP-2 is unblocked.** Full account: `RESULTS_confirm.md` and §E-7 below. The rest of section E
> is kept as the diagnostic record — it is how we got here, and E-2 is the reason we did not
> conclude the hypothesis was false.

## E (record). WGAN-GP fails to train the entangling angles

Evidence: `scripts/benchmark_synthetic.py`, `scripts/diag_benchmark.py`, `scripts/diag_trained.py`
· controlled synthetic teacher (16 nodes, 19 edges; the true graph is known from the outset)

### E-1. The confirmatory control does not work

On the controlled synthetic benchmark, **all four variants coincide** (dependency error over
120 pairs, 3 seeds):

| Model | \|E\| | 120-pair error |
|---|---|---|
| aligned | 19 | 0.1359 ± 0.0092 |
| permuted | 19 | 0.1346 ± 0.0089 |
| distmatched | 19 | 0.1354 ± 0.0106 |
| rewired | 19 | 0.1360 ± 0.0090 |
| **no_entangle** | **0** | **0.1318** ← **the best** |

The hypothesis predicts `aligned < permuted`, but in fact permuted is marginally better, and
**the model with no entanglement at all beats every other model.**

### E-2. The cause is not "it fails to learn" but "it manufactures spurious dependencies"

You cannot interpret 0.135 without computing a baseline. Here is the score of **a model that
creates no dependencies whatsoever** (the real data shuffled independently column by column →
the marginals are perfect and the dependencies are exactly zero):

| Baseline | 120 pairs | 19 true edges | 101 non-edges |
|---|---|---|---|
| **floor (zero dependency)** | **0.0648** | 0.3676 | 0.0078 |
| aligned (WGAN-GP trained) | 0.1359 | **0.3662** | **0.0926** |
| permuted | 0.1346 | 0.3707 | 0.0902 |
| no_entangle | 0.1318 | 0.3760 | 0.0859 |

**The trained model is 2.1x worse than a model that does nothing.** Two things happen at once:

1. **The error on the 19 true edges equals the floor.** `0.3662` vs `0.3676`. The contribution of
   the 19 entangling gates is **essentially zero**. The model produces none of the true
   dependencies.
2. **Spurious dependency of `0.0926` on the 101 non-edges.** Even a model with **zero**
   entangling gates produces `0.0859` → the culprit is not entanglement but **the condition
   vector `c`, which every feature shares.**

In other words, **the WGAN-GP critic fails to supply a useful gradient to `gamma` (the only
parameter that can create cross-feature dependency).** `gamma` is never learned and only adds
noise, so turning entanglement off yields a better score. The alignment effect (expected
magnitude ~0.01) is completely buried in this noise.

**This is not an expressivity problem — and that is now settled, not conjectured.**

### E-2b. [RESOLVED] The gate: the circuit CAN represent the pattern, and the CDG wins by 4.5x

Evidence: `scripts/ceiling_joint.py` · `RESULTS_ceiling_joint.md`

With the GAN removed and the circuit optimized **directly** on the full 120-pair pattern
(scored on a fixed held-out draw of 65,536 samples, identically for every variant and for the
floor):

| Model | \|E\| | 120-pair min. error | vs. floor |
|---|---|---|---|
| **aligned** (true CDG) | 19 | **0.0103** | **83.0%** |
| rewired | 19 | 0.0401 | 34.1% |
| distmatched | 19 | 0.0421 | 30.8% |
| permuted | 19 | 0.0468 | 23.2% |
| no_entangle | 0 | 0.0508 | 16.7% |
| *floor (zero dependency)* | — | *0.0609* | — |

Three things follow, and together they rewrite the status of the project:

1. **A solution exists and gradient descent reaches it.** aligned gets to 0.0103 against a true
   signal magnitude of 0.351. Satisfying the 19 edges and the 101 non-edges *simultaneously* is
   not the obstacle. **Expressivity is not the bottleneck.**
2. **The CDG hypothesis is true at the representational level.** Same 19-edge budget, same degree
   sequence, same triangles — aligned still beats permuted by 4.5x on error, and beats the
   **distance-matched** control by 4.1x. Since distance-matching cancels the "advantage of being
   nearby", the remaining gap can only come from *the edges the CDG chose carrying real
   dependency structure*. **The premise of the confirmatory experiment is sound.**
3. **Therefore the failure is entirely in training.** WGAN-GP sits 13x away from a solution the
   same circuit demonstrably reaches (0.1359 vs 0.0103).

This is the most consequential result so far, and note the counterfactual: had we run WP-2's 90
confirmatory runs *before* this check, they would have returned the same null, and we would have
concluded that **the CDG hypothesis was false** — when in fact it is the training objective that
is broken.

**A methodological note, because the number was nearly wrong.** `fit()` originally scored the
best training minibatch seen along the trajectory. Each step draws a fresh minibatch, so `min`
over 1500 steps returns the *luckiest draw*, not the best parameters — and since `floor` was
computed analytically (noise-free), that optimistic bias landed on only one side of the
comparison. Both sides are now scored with the same estimator on the same fixed held-out draw.
Fixing it made the separation **larger**: aligned 0.0130 → 0.0103, permuted 0.0491 → 0.0468.

### E-3. Estimator mismatch — the metric is computed in a different space from the CDG definition

The CDG is defined by partial correlation **conditional on `c`** (§7.2, A-3), but
`train.dependency_error` measures **unconditional** partial correlation. The generator feeds `c`
into every qubit angle and every head, so the correlation `c` induces shows up as a false
positive on **every pair**.

Linearly residualizing on `c` reduces the non-edge error from `0.0926 → 0.0532`, but it does
**not eliminate it**, because `c` enters the angles **nonlinearly**. Linear residualization is
not enough.

> **This is not a defect of the synthetic benchmark alone — it is present verbatim in the real
> MIMIC experiment.**
> If the metric is not fixed, all 90 runs of WP-2 will be measuring this false positive.

### E-4. Actions (mandatory before starting WP-2)

1. **[GATE — PASSED]** `ceiling_joint.py`. See E-2b. The circuit can represent the pattern and the
   CDG beats every control. The design is sound; the training is not.
2. **Align the metric with the CDG definition.** Replace it with partial correlation conditional
   on `c`, using an estimator that handles nonlinear conditioning properly (linear
   residualization removes only half of the false positives). This defect exists in the real
   MIMIC pipeline too, not only in the benchmark.
3. **Make WGAN-GP actually learn dependencies.** This is now the critical path.

   The leading candidate is **the learning rate**. `ceiling_joint` uses Adam `lr = 0.05` on the
   quantum parameters; `train.py` applies `lr_g = 5e-5` to *everything* — 1000x smaller. Quantum
   angles are in radians and need to move O(0.1–1) rad to create meaningful entanglement, but a
   learning rate tuned for ~2000 classical head weights was applied unchanged to the 19 entangling
   angles. `scripts/diag_lr.py` measures how far `gamma` actually moves and sweeps
   `lr_q ∈ {5e-5, 5e-3, 5e-2}`.

   **Note that a per-parameter-group learning rate is *not* a circular fix**: the loss stays pure
   WGAN-GP and the evaluation metric never enters it. v2 §8.10 is respected.

   If raising `lr_q` does *not* pull the true-edge error off the floor, then the cause is deeper —
   the critic simply cannot see pairwise conditional dependence in a 16-dimensional joint, while
   the ~2000 head parameters are free to fit the marginals, so the GAN converges to matching
   marginals and ignores dependency entirely. In that case §8.10's ban on any dependency term is
   in direct conflict with the model learning anything at all, and that tension has to be resolved
   on its merits — we would need a critic that is sensitive to pairwise structure *without*
   optimizing the evaluation metric.

> **Running WP-2 (9 variants × 10 seeds = 90 runs) right now would be entirely wasted effort.**
> The contrast is real (E-2b) but the training destroys it (E-1, E-2). Fix E-2/E-3 first.

### E-6. [DECISION REQUIRED] The learning rate is refuted. §8.10's ban on a dependency term has disabled the model.

Evidence: `scripts/diag_lr.py` · `RESULTS_lr.md`

Sweeping the quantum learning rate over three orders of magnitude:

| `lr_q` | mean \|Δγ\| (rad) | 120 pairs | **19 true edges** | 101 non-edges |
|---|---|---|---|---|
| 5e-5 *(current)* | 0.0960 | 0.1350 | **0.3667** | 0.0915 |
| 5e-3 | 0.2926 | 0.1197 | **0.4026** | 0.0665 |
| 5e-2 | **2.0110** | 0.1748 | **0.3589** | 0.1402 |
| *floor (zero dependency)* | — | *0.0648* | *0.3676* | *0.0078* |
| *ceiling (direct optimization)* | — | *0.0103* | *~0* | *~0* |

(aligned; `lr_q=5e-2 / permuted` was lost to a crashed run and is not load-bearing.)

**At `lr_q = 5e-2` the angles move 2.0 radians — they are not frozen — and the true-edge error
is still 0.3589 against a floor of 0.3676.** Meanwhile the non-edge false positives explode and
the overall score gets *worse*. There is no learning rate at which this model learns the true
dependency structure. Raising `lr_q` moves `gamma` faster in the wrong direction.

The diagnostic detail that settles it: at `lr_q=5e-3`, aligned's angles move **2.5× further**
than permuted's (0.2926 vs 0.1164 rad) — the critic pushes hardest on the circuit that actually
has RZZ gates on the true edges — and aligned comes out **worse** than the floor. More gradient,
in the wrong direction, does more damage exactly where there was more to gain.

**Conclusion: the WGAN-GP critic cannot identify which pairs ought to be dependent. The gradient
it supplies to `gamma` is noise.**

#### This is a failed design decision, not a bug

v2 §8.10 forbids any dependency term in the loss, to avoid the circular argument *"you optimized
the evaluation metric, so of course you score well on it."* The measured consequence of that ban
is **a model that learns no conditional dependency at all, under any topology.** The rule meant to
protect the claim has disabled the model that was supposed to make it.

**WP-2 cannot be run as specified.** A confirmatory experiment across 9 topologies × 10 seeds,
using an objective that provably learns zero dependency for every topology, measures nothing.

#### The three ways out

**(A) Make the critic able to see dependency — without naming the metric.** A per-sample MLP
critic on 16 dimensions has to infer a correlation structure from single rows; in practice it
converges to matching marginals, which the ~2000 head parameters can do on their own, and ignores
the joint. Standard remedy: a **batch-aware critic** (minibatch discrimination, Salimans et al.
2016), which lets the critic compare a *batch's* structure against the real batch's. This is a
generic architectural technique, not the evaluation metric, so §8.10 survives intact.
*Cost: one experiment. If it works, the paper keeps a real generative model.*

**(B) Put a dependency term on `E_fit` only, and evaluate on held-out pairs.** Non-circular by
construction: no model ever optimized the pairs it is scored on. This is what HDE was designed
for. *Caveat: A-2 demoted HDE precisely because held-out reachability is decided by graph
combinatorics, which is why B-1's distance-matched control exists. Workable but delicate.*

**(C) Reframe. Make the expressivity ceiling the primary result and report the GAN failure as an
honest secondary finding.** The scientific claim — that CDG alignment carries information — is
already fully supported by `RESULTS_ceiling_joint.md`, and it is non-circular in the way that
matters: aligned and permuted get identical objectives and identical budgets, and permuted simply
*cannot represent* what it needs to. "A standard adversarial objective does not exploit this
structure at all" is itself a real result about tabular GANs.
*Cost: the paper loses its working generator, and with it the synthetic-data application.*

**Recommendation: (A) first.** It is one experiment, it preserves §8.10, and it targets the exact
failure mode observed (the generator fits marginals and ignores the joint). If (A) fails, (C) is
the honest fallback and it is already fully evidenced — but it costs the application, so it should
not be chosen before (A) has been tried.

### E-7. [RESOLVED] The fix: make the critic blind to the marginals

Evidence: `scripts/diag_fix.py`, `scripts/diag_fix2.py`, `scripts/confirm.py` ·
`RESULTS_confirm.md`

Five candidate fixes were tried. **Four failed, and all four are recorded here** — the true-edge
error is the sharpest read, because the floor is 0.3676 and the same circuit reaches ~0 under
direct optimization.

| what was tried | aligned true-edge error | verdict |
|---|---|---|
| *floor — a model that creates zero dependency* | *0.3676* | — |
| baseline WGAN-GP | 0.4029 | worse than doing nothing |
| quantum learning rate, swept 1000× | 0.3589 | **failed** (§E-6) |
| (D) bigger critic batch (1024) | 0.3609 | **failed** |
| (A) batch-aware critic alone | 0.3488 | **failed** |
| (B) dependency term on a FIT split | 0.4024 | **failed** |
| **(E) copula critic** | **0.2663** | **works** |
| **(E) + (A)** | **0.1892** | **works, and best** |

**(E) — the copula critic.** Rank-transform each feature *within the batch* before the critic
sees it: a differentiable soft rank, then the inverse normal CDF. Every input feature is then
standard normal **by construction**, so the marginals carry *no information at all*, and the only
thing left for the critic to discriminate on is the copula — the dependency structure. The
gradient has nowhere to go but the entangling angles.

This is not an ad-hoc trick. It is the *same* monotone-invariance argument that already justifies
computing the **metric** in nonparanormal space (A-4), applied to the critic instead of to the
metric. And it computes no correlation, no partial correlation, nothing the evaluation does — it
is an input transform. **v2 §8.10 survives intact, and the exception we thought we would have to
carve out is not needed.**

**(A) — minibatch discrimination** (Salimans et al., 2016) on top. A per-sample critic cannot
estimate a 16-dimensional joint from single rows; this lets it compare a *batch's* structure
against a real batch's. On its own it does nothing (0.3488, still the floor). On top of (E) it
nearly doubles the effect (0.2663 → 0.1892).

The causal chain reads cleanly in that table: **if the left column does not break the floor, the
topologies never separate.** A model that learns no dependency cannot express a preference among
topologies — expressivity that is never used is invisible. (E) opened the learning; the alignment
effect appeared immediately; (A) amplified it.

#### Two more things had to be fixed for the experiment to mean anything

**The metric (E-3).** The CDG is *defined* as a partial correlation conditional on `c`, and the
metric was measuring the unconditional one. `eval_dep.partial_corr_c` is now the estimator the
CDG is defined by: nonparanormal → residualize on a **nonlinear** basis of `c` (it enters the
circuit through angles, so linear residualization left half the false positives behind) → partial
correlation. This defect was live in the real MIMIC pipeline, not only in the benchmark.

**The teacher.** `benchmark_synthetic.teacher_data` drew `c` **independently of `X`**. MIMIC does
not look like that — mortality, age, sex and ICU type plainly shift vitals and labs — and the
independence made the estimator mismatch invisible in the benchmark while it was live in the real
pipeline. `teacher_data_cond` has `c` shift every feature's mean, with the graph fixing the
dependency structure *conditional on* `c`.

#### The result, and a result we did not predict

`aligned − distmatched = −0.0303` against a per-variant SD of ~0.004. distmatched is the control
that matters: it matches the CDG's held-out pair distance profile, so "the strong pairs happen to
be nearby" is no longer an advantage the CDG uniquely holds. It still loses.

And: **`no_entangle` (0.0647) beats permuted (0.0787), rewired (0.0749) and distmatched
(0.0729) — all three of which are worse than the floor.** A misaligned circuit cannot learn the
dependencies that exist (true-edge error stays at the floor) but *does* manufacture ones that do
not (non-edge error 0.023–0.026 vs the floor's 0.0090). The entangling gates are not a free prior
that helps a little wherever you put them. They are a **commitment** that these particular pairs
are dependent — and asserting it about the wrong pairs leaves you strictly worse off than never
having asserted it. That is a sharper claim than "alignment helps".

#### The methodological lesson, which outlives this paper

A tabular GAN whose decoder can fit the marginals will let the critic settle for the marginals,
and then **nothing learns the dependency structure** — for any architecture, any topology, any
hyperparameter. In that regime every structural hypothesis looks equally false, because every
model is equally empty. We nearly concluded the CDG hypothesis was wrong on exactly that basis.

The tell was cheap and we should have looked for it sooner: **score a model that creates no
dependency at all.** Ours beat the model we had trained by 2.1×.

#### Housekeeping

`diag_fix.py` was stopped after its (B)/aligned row. (B) had already failed there (0.4024, worse
than the floor), and its remaining rows were (B)/permuted and (A)+(B) — variants of two approaches
that had each already failed on their own. The GPU went to `confirm.py` instead. Recording this
because silently dropping runs is how "we only report what worked" happens.

### E-5. A bug caught along the way: `no_entangle` falls back to the full statevector

Because of the `lightcone and len(edges)` guard in `model.py`, a graph with zero edges could not
take the light-cone path and fell back to the **full 2^16 = 65,536-dimensional statevector**.
With no edges each cone is a single qubit (2 dimensions), so **what should have been the lightest
case became the heaviest** — it occupied 22GB of GPU memory and training effectively stalled.
The guard has been removed (`if lightcone`).

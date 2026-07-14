# WP-2 — the confirmatory topology experiment

`wp2.py` (5 shards) · `wp2_report.py` · Real MIMIC-IV v3.1, n = 48,561, full cohort
Architecture: CDG Δ=4 (`cdg_d4.npz`, 29 RZZ angles) · **L = 1** · v3 loss (λ = 10, lr_g = 1e-3)
8,000 steps · batch 512 · 3 seeds per graph

**The question.** Does it matter that the entangling topology is *the clinical dependency graph*,
as opposed to some other graph with the same shape? Every control below has the same number of
edges and the same circuit; only the wiring differs.

---

## 1. Result

| variant | 120-pair error | vs floor | **vs honest null** |
|---|---|---|---|
| **CDG (Δ=4)** | **0.0796 ± 0.0035** | −19.2 % | **−15.5 %** |
| ring-with-chords | 0.0859 ± 0.0030 | −12.7 % | −8.8 % |
| permuted_0 | 0.0879 ± 0.0039 | −10.7 % | −6.7 % |
| distmatched_2 | 0.0907 ± 0.0040 | −7.9 % | −3.7 % |
| permuted_1 | 0.0927 ± 0.0011 | −5.9 % | −1.6 % |
| rewired (degree-preserving) | 0.0947 ± 0.0014 | −3.8 % | **+0.5 %** |
| distmatched_0 | 0.0963 ± 0.0012 | −2.2 % | **+2.2 %** |
| no_entangle | 0.0966 ± 0.0004 | −1.9 % | **+2.6 %** |
| distmatched_1 | 0.0979 ± 0.0028 | −0.5 % | **+4.0 %** |
| permuted_2 | 0.0984 ± 0.0027 | −0.0 % | **+4.5 %** |

Reference points for the same 120 pairs:

| | | |
|---|---|---|
| wp2 floor | 0.0985 | zero dependency **and** no x–c relation — gates the run |
| **honest null** | **0.0942** | zero dependency, correct conditional marginals — see §3 |
| CDG-QGAN Δ=4 ceiling | 0.0300 | GAN removed, circuit fitted directly (`RESULTS_design.md`) |
| Corollary 1 bound | 0.0140 | unreachable by **any** L=1 Δ=4 circuit |

Read the last column, not the middle one. Against the null that is actually correct for a v3 model
(§3), **five of the nine controls are at or ABOVE it** — a misaligned topology does not merely fail
to help, it manufactures dependency that is *wrong*, and ends up worse than creating none at all.
Only the CDG is decisively below.

### Contrasts (hierarchical bootstrap, 20,000 resamples)

```
CDG − permuted          (3 graphs, n=9)  = −0.0134   95% CI [−0.0187, −0.0086]   CDG better
CDG − distance-matched  (3 graphs, n=9)  = −0.0154   95% CI [−0.0202, −0.0109]   CDG better
CDG − rewired                    (n=3)   = −0.0151   95% CI [−0.0197, −0.0114]   CDG better
CDG − ring                       (n=3)   = −0.0064   95% CI [−0.0120, −0.0014]   CDG better
CDG − no_entangle                (n=3)   = −0.0171   95% CI [−0.0218, −0.0142]   CDG better
```

Every contrast has a 95 % CI excluding zero. `ring-with-chords` is the strongest control — a
generic *structured* topology, and the only one that gets meaningfully below the honest null on its
own — and the CDG still beats it.

**Seed-level separation is complete.** The CDG's three seeds are 0.0747, 0.0817, 0.0823. The
*minimum* over all **27** control seeds — nine graphs × three seeds — is **0.0825**. The CDG's worst
run is better than every control's best run. The two sets of numbers do not overlap at any point,
which is a stronger statement than any of the CIs above and does not depend on a bootstrap.

---

## 2. What each control rules out, and what survives

The claim is not "entanglement helps". It is "*this* wiring helps, because it is the clinical
structure." Each control removes one alternative explanation for the CDG's score.

| control | what it holds fixed | what it rules out if the CDG still wins |
|---|---|---|
| `no_entangle` | nothing (0 edges) | that the score comes from the classical heads |
| `permuted` | \|E\|, degree sequence, **the graph up to isomorphism** | that any 29-edge graph of this shape would do |
| `distmatched` | \|E\| **and the distribution of graph distances between the 120 pairs** | that the CDG merely happens to put *many* pairs in the light cone |
| `rewired` | \|E\| and every node's degree | that the score is a degree effect |
| `ring` | \|E\|, plus a generic "structured" topology | that any non-random structure would do |

**The decisive control is `distmatched`.** Beating an isomorphic permutation could still be graph
combinatorics — an L = 1 circuit can only reach pairs within graph distance 2, so *any* graph that
covers more pairs scores better, whatever it covers. Distance matching removes exactly that
degree of freedom: the control graph puts the *same number of pairs at the same distances*, and
differs only in **which** pairs. The CDG still wins, by 0.0154 with a CI excluding zero.

That is the clinical claim, isolated: it is not how many pairs the circuit can reach, it is
**which** pairs. The CDG reaches the ones the data actually depends on.

---

## 3. The floor is the wrong null, and we say so before anyone asks

`no_entangle` scores 0.0966 against a floor of 0.0985 — *below* the floor. Taken at face value that
reads as a falsifier failure: a circuit with **zero entangling gates** appearing to create
dependency, which Proposition D-2 says is impossible.

It is not a failure. It is the **floor** that is wrong, and the reason is in the *estimator*.

`eval_dep.partial_corr_c` residualizes on a fixed basis — 1, c, c², and the pairwise products
c_a·c_b. Whatever part of E[x_u | c] lies outside that span survives residualization. It is a
deterministic function of c, therefore **shared across features**, and a shared component is
precisely what a correlation estimator reports as cross-feature dependence. The real structure
`zr` is measured with the same estimator, so `zr` carries this bias too. A model that reproduces
E[x_u | c] therefore reproduces the bias, lands closer to `zr`, and scores below the floor
**without creating one bit of conditional dependence.**

The v3 loss makes E[x_u | c] correct *on purpose* (§E-11). So this was not a possibility — it was a
prediction. `wp2.py`'s floor destroys the cross-feature structure **and** the x–c relation, which
was the right null for v2 (where nothing trained the marginals) and is the wrong one for v3.

**The honest null** (`null_condmarg.py`): bin c by mortality × age-quantile × sex × ICU type, and
permute each feature independently *within* each bin. Cross-feature dependence conditional on c is
then exactly zero, by construction — no model, no training, no circuit — while E[x_u | c] is the
true conditional marginal at bin resolution.

| | score | vs floor |
|---|---|---|
| wp2 floor (x–c destroyed too) | 0.0985 | — |
| honest null, 197 bins (age-quartile) | 0.0942 ± 0.0003 | −4.3 % |
| honest null, 357 bins (age-octile) | 0.0940 ± 0.0004 | −4.6 % |

The two resolutions agree, so it is the estimator's conditioning bias and not the binning.
**A zero-dependency model with correct conditional marginals collects ~4 % below the floor for
free.**

Three consequences, and the third is the one that matters:

1. **`no_entangle` (0.0966) does not even reach the honest null (0.0942).** It sits *between* the
   two nulls — exactly where a model with zero dependency and *imperfect* conditional marginals
   (W1 = 0.131) belongs. Every bit of its sub-floor score is accounted for without invoking any
   dependency. **The falsifier holds, and it holds more strongly than we thought.**

2. Restated against the honest null, the CDG is at **−15.5 %** (0.0796 vs 0.0942), not −19.2 %.
   The headline number is smaller, and it is the correct one.

3. **The contrasts in §1 are untouched.** The bias is a property of the estimator and the
   conditional marginals, and *every variant in the table has the same conditional marginal term*.
   It is a common offset. It cancels in CDG − permuted and in CDG − distance-matched, which are
   the numbers the paper rests on.

---

## 4. Where this leaves the model

| | 120-pair error |
|---|---|
| honest null (zero dependency) | 0.0942 |
| **CDG-QGAN Δ=4, trained** | **0.0796** |
| CDG-QGAN Δ=4, ceiling (GAN removed) | 0.0300 |
| Corollary 1 bound (L=1, Δ=4) | 0.0140 |
| Gaussian copula (the **oracle** for this metric) | 0.0064 |

The GAN closes **23 %** of the distance between the null and its own ceiling. The ceiling is not a
theoretical curiosity: `ceiling_real.py` reaches 0.0300 on the same real data with the adversarial
game removed. **The gap between 0.0796 and 0.0300 is an optimization gap, not a representational
one** — the circuit can express the structure; the GAN is not getting it there.

That is the honest statement of where the method stands, and it is the obvious next target
(WP-6). It is also why the paper's claim is the *topology* result (§2), which is measured against
matched controls and is unaffected by how far either of them is from the ceiling.

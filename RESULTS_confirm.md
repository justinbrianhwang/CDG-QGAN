# Result: the confirmatory contrast works — a trained model recovers the alignment effect

Script: `scripts/confirm.py` · Run: 2026-07-11 · RTX 5090

## What this experiment is, and what it is not

`RESULTS_ceiling_joint.md` asked whether the circuit **can represent** the target dependency
pattern. It removed the GAN and optimized the circuit directly. Answer: yes, and the CDG beats
every control by ~4x.

This experiment asks the harder question: **does an actual generative model, trained
adversarially, reach it?** Until now the answer was a flat no — WGAN-GP learned *zero*
dependency for every topology (`RESULTS_lr.md`), so all variants collapsed onto the same score
and the confirmatory contrast measured nothing.

Three things had to be fixed before this experiment could mean anything.

## The three fixes

**1. The critic could not see dependency at all.**

The critic was spending its capacity on the marginals — which the ~2000 classical head
parameters can already fit unaided — and handing the entangling angles nothing but noise.
Sweeping the quantum learning rate over 1000× did not help; at `lr_q=5e-2` the angles move 2.0
radians and still learn nothing (`RESULTS_lr.md`).

Two changes, **neither of which touches the evaluation metric**:

- **(E) Copula critic.** Rank-transform each feature *within the batch* before the critic sees
  it — a differentiable soft rank, then the inverse normal CDF. Every input feature is then
  standard normal by construction, so the marginals carry **no information at all** and the only
  thing left to discriminate on is the copula. The gradient has nowhere to go but `gamma`.
  This is the same monotone-invariance argument that already justifies measuring the *metric* in
  nonparanormal space (review A-4), applied to the critic instead.
- **(A) Batch-aware critic.** Minibatch discrimination (Salimans et al., 2016). A per-sample
  critic cannot estimate a 16-dimensional joint from single rows; this lets it compare a
  *batch's* structure against a real batch's.

**2. The benchmark's teacher was unrealistic.** It drew the condition vector `c` independently
of `X`. MIMIC does not look like that — mortality, age, sex and ICU type plainly shift vitals and
labs — and the independence made an estimator mismatch that is *live in the real pipeline*
invisible in the benchmark. `teacher_data_cond` now has `c` shift the mean of every feature,
with the graph fixing the dependency structure **conditional on `c`**.

**3. The metric was the wrong estimator.** The CDG is *defined* as a partial correlation
conditional on `c`, but the metric measured the unconditional one, so everything `c` induced was
scored as a false positive on **every pair** (review E-3). `eval_dep.partial_corr_c` is now the
estimator the CDG is defined by: nonparanormal, residualize on a nonlinear basis of `c`, then
partial correlation.

**The loss is still pure WGAN-GP.** No dependency term. Nothing from the evaluation metric.
v2 §8.10 stands, and it stands *without* the exception we thought we would have to carve out.

## Setup

16-node, 19-edge modular teacher (the true graph is known); `c` drives every feature's mean.
`L=1` · 3 seeds · 3000 steps · identical resources for every variant: same edge budget, same
critic, same optimizer, same objective. The **only** difference is which clinical pair sits under
which RZZ gate.

Primary endpoint: mean absolute Fisher-z error over all 120 pairs, **conditional on `c`**.
Reported alongside: the 19 true edges (false negatives) and the 101 non-edges (false positives).

## Table

![confirmatory experiment](figures/fig_confirm.png)

| Model | \|E\| | **120 pairs** | 19 true edges | 101 non-edges |
|---|---|---|---|---|
| **aligned (true CDG)** | 19 | **0.0426 ± 0.0043** | **0.1627** | 0.0200 |
| no_entangle (RZZ removed) | 0 | 0.0647 ± 0.0003 | 0.3644 | 0.0084 |
| *floor — a model that creates zero dependency* | — | *0.0653* | *0.3641* | *0.0090* |
| distmatched (distance-matched permutation) | 19 | 0.0729 ± 0.0020 | 0.3352 | 0.0235 |
| rewired (degree-preserving rewire) | 19 | 0.0749 ± 0.0051 | 0.3521 | 0.0228 |
| permuted (isomorphic permutation) | 19 | 0.0787 ± 0.0040 | 0.3600 | 0.0258 |

Contrasts, against a per-variant standard deviation of ~0.004:

| | Δ | |
|---|---|---|
| aligned − permuted | **−0.0361** | ≈ 9σ |
| aligned − rewired | **−0.0324** | |
| aligned − distmatched | **−0.0303** | ≈ 7σ · the strongest control |
| aligned − no_entangle | **−0.0222** | |

## Conclusions

**1. The trained model finally beats doing nothing.** aligned scores 0.0426 against a floor of
0.0653 — 35% better. Before the fixes, the trained model was **2.1× worse than the floor**
(0.1359 vs 0.0648). The direction has reversed.

**2. Alignment shows up in a trained model, for the first time.**
`aligned − permuted = −0.0361`, against a per-variant standard deviation of ~0.004. That is a
9σ separation. Against the *strongest* control it is `aligned − distmatched = −0.0303`, still 7σ.

**3. The mechanism is exactly the one the theory names.** Look at the true-edge column. Every
variant gets 19 entangling gates and the same critic. aligned pulls the true-edge error from
0.3641 (floor) down to **0.1627**. permuted stays at **0.3600** — it learns essentially *nothing*
on those edges. Not because it was trained less, but because Corollary 1 **forbids** it: its
strong pairs were scattered beyond `2L`, where the conditional covariance is exactly zero at
every parameter setting. Fixing the training did not help permuted, because permuted's problem
was never training. **Repairing the optimizer is what made the expressivity wall visible.**

**4. Distance matching does not rescue it.** `distmatched` matches the CDG's held-out pair
distance profile, so "the strong pairs happen to be nearby" is no longer an advantage the CDG
uniquely holds. It still loses, and its true-edge error (0.3352) sits just under the floor.
**Being close is not enough — the right pair has to be there.** This is the central claim of the
paper, and it now holds for a model that was actually trained, not merely for an upper bound.

**5. `no_entangle` lands exactly on the floor — so the effect really is the entanglement.**
0.0647 against a floor of 0.0653, with a standard deviation of 0.0003. Strip the RZZ gates and
the model creates no dependency whatsoever, as Corollary 1 says it must. Everything aligned
achieves, it achieves through the entangling angles. Nothing else in the model can do it.

**6. Misplaced entanglement is worse than no entanglement at all.** This one we did not
predict. `no_entangle` (0.0647) **beats** distmatched (0.0729), rewired (0.0749) and permuted
(0.0787). All three of them are *worse than the floor*.

The reason is visible in the split. A misaligned circuit still cannot learn the dependencies
that exist — its true-edge error sits at 0.335–0.360, essentially the floor's 0.3641 — but it
**does** manufacture dependencies that do not (non-edge error 0.023–0.026, against the floor's
0.0090). It pays the full price of entanglement and collects none of the benefit.

So the entangling gates are not a free prior that helps a little wherever you put them. They
are a **commitment**: they assert that these particular pairs are dependent. Assert it about the
wrong pairs and you are strictly worse off than never having asserted it. That is a sharper
statement than "alignment helps", and it is the strongest form of the claim that the *clinical*
content of the graph is load-bearing.

## The methodological finding, which outlives this paper

A tabular GAN with a decoder powerful enough to fit the marginals will let the critic settle for
the marginals, and then **nothing learns the dependency structure** — for any architecture, any
topology, any hyperparameter. In that regime every structural hypothesis you might want to test
looks equally false, because every model is equally empty.

We nearly concluded that the CDG hypothesis was wrong on exactly this basis. The tell was
cheap and we should have looked for it sooner: **compute the score of a model that creates no
dependency at all.** Ours was 2.1× better than the model we had trained.

## What this unblocks

WP-2 (the full confirmatory experiment: 9 topologies × 10 seeds on real MIMIC-IV) can now be
run. `REVISIONS.md` §E is resolved.

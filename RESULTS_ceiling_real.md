# Results: Joint Expressivity Ceiling on Real MIMIC-IV

Script: `scripts/ceiling_real.py` · Run: 2026-07-11
Data: MIMIC-IV v3.1, 24-hour landmark cohort, **n = 48,561** · CDG: 16 nodes, 21 edges, L = 1

> This is `ceiling_joint.py` (which ran on the synthetic teacher, `RESULTS_ceiling_joint.md`)
> moved onto the real cohort. It exists because the WP-2 smoke run failed, and there were two
> incompatible explanations for that failure. This experiment tells them apart.

---

## The question

The first WP-2 run put **every** trained variant, the CDG included, **above the floor** — worse
than a model that creates no dependency at all. Two readings, demanding opposite responses:

| | meaning | response |
|---|---|---|
| **(a)** the circuit cannot represent the real dependency pattern | the design is wrong | WP-2 is moot; redesign |
| **(b)** it can, but WGAN-GP does not find it | the training is wrong | fix the training |

So: **remove the GAN.** Optimize the circuit parameters directly, by gradient descent, on the
c-conditional dependency error, scored on a fixed held-out draw. This measures *representability*
with learnability taken out of the picture.

---

## The bar is not zero

72 of the 120 pairs sit **outside** the L=1 light cone (`d_G(u,v) > 2`). By Corollary 1 the model
produces **exactly zero** conditional covariance there, at every parameter setting. It cannot err
on them and it cannot help itself on them. So the best score any L=1 CDG circuit can possibly
achieve is not 0 — it is the error it must pay on those unreachable pairs:

```
irreducible = (1/120) · Σ_{d_G(u,v) > 2} |z_true(u,v)|  =  0.0331
```

**Read the ceiling against 0.0331, not against 0.** This number is a property of the graph, not
of the optimizer.

---

## Results

| model | \|E\| | 120-pair error | vs. floor | above irreducible |
|---|---|---|---|---|
| **aligned (CDG)** | 21 | **0.0437** | **−55.7%** | **+0.0106** |
| permuted (isomorphic) | 21 | 0.0632 | −35.9% | +0.0301 |
| distmatched | 21 | 0.0699 | −29.1% | +0.0369 |
| rewired (degree-preserving) | 21 | 0.0730 | −26.0% | +0.0399 |
| no_entangle | 0 | 0.0969 | −1.8% | +0.0638 |
| *floor (zero dependency)* | — | *0.0986* | — | *+0.0655* |

```
  aligned − permuted     = −0.0195
  aligned − distmatched  = −0.0262      <- the contrast that carries the clinical claim
  aligned − rewired      = −0.0293
  aligned − no_entangle  = −0.0531
```

**Verdict: (b).** The circuit represents the real pattern, alignment is what lets it, and the
WP-2 smoke failure is a **training** failure.

---

## What the numbers say

**1. The CDG circuit nearly saturates its own light cone.** It lands 0.0106 above a bound it
cannot cross. Of the dependency that is reachable at all, it captures almost all of it. There is
very little left on the table for a better optimizer — the headroom is in the *graph*, not in the
fitting.

**2. `no_entangle` sits on the floor.** 0.0969 against 0.0986 — indistinguishable. With the RZZ
gates removed the model creates **no conditional dependency whatsoever**, exactly as Proposition
D-2 requires: the ~2,000 head parameters can shape every marginal and still cannot manufacture a
single cross-feature dependence. This is the cleanest empirical confirmation of D-2 we have, and
it reproduces the synthetic result (`RESULTS_confirm.md`) on real data.

**3. Misplaced entanglement is worse than no entanglement — but only up to a point.** On the
synthetic teacher, permuted/rewired/distmatched all scored *worse* than `no_entangle`. Here they
score *better* than it (0.063–0.073 vs 0.097). The difference is that the real graph's clusters
are physiological: even a permuted CDG accidentally connects *some* genuinely dependent pairs,
because the strong dependencies in MIMIC (electrolytes, renal) are dense and a random relabelling
of a 21-edge graph will land on a few of them by chance. The teacher's edges were scattered, so a
permutation hit nothing.

That is worth saying plainly: **the permutation null on real data is not a zero-signal control.**
It is a weak-signal control. Which makes the aligned−permuted gap the *conservative* estimate of
the effect, not an inflated one.

**4. The distance-matched control is the one that matters, and the CDG beats it by 0.0262** —
larger than the gap to the plain permutation. Distance matching preserves the graph *and* the
distances of the held-out strong pairs, so beating it cannot be explained by combinatorics. What
is left is the specific clinical pairs the CDG chose.

---

## What this does not show

This is a **ceiling**, obtained by optimizing the dependency error directly. It is not a GAN
result and it is not a synthetic-data-generation result. It says the target is reachable; it says
nothing about whether adversarial training reaches it — and as of this run, **it does not**
(the WP-2 smoke run scored 0.1307, worse than the floor).

The ceiling is therefore not a result to report as if it were the model working. It is a
**diagnostic that tells us where to spend the next effort**: on the objective, not on the circuit.

Compare `RESULTS_ceiling_joint.md`, where the identical experiment on the synthetic teacher gave
the identical verdict, and the fix that followed (the copula critic) closed the gap. The same
program applies here.

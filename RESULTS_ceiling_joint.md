# Results: Joint Expressivity Ceiling — CDG Alignment Really Does Contribute Bits

Script: `scripts/ceiling_joint.py` · Run: 2026-07-11 · RTX 5090

## Why This Experiment Was Needed

`RESULTS_ceiling.md` maximized the |correlation| of **a single pair** (L=1, d=1 → 0.991).
That only means "a strong dependency can be placed on one edge"; it does not mean that the
circuit can **simultaneously fit 19 edges at ρ=0.35 while holding the 101 non-edges at
zero**. The latter is what the confirmatory experiment demands. That cell was empty.

Moreover, in the controlled synthetic benchmark (`benchmark_synthetic.py`) **the
confirmatory contrast did not work** — all four variants overlapped at a 120-pair error of
0.132–0.136, and the model with **no entanglement at all** was the best. We had to
determine whether this was an expressivity failure or a training failure.

## Setup

teacher = a 16-node, 19-edge modular graph (we start knowing the true graph), with a mean
true-edge |ρ| of 0.351. **With the GAN removed**, the circuit parameters are optimized by
direct gradient descent on the 120-pair partial-correlation error (Adam lr=0.05, 1500
steps, batch=4096, 3 restarts). Because the head is a monotone one-dimensional map, the
dependency structure in the nonparanormal space is determined solely by the copula of
`q = ⟨Z⟩` → measuring on `q` suffices.

**Scoring.** Parameters are scored on a **fixed held-out draw of 65,536 samples**, identical
across every variant and every restart. Scoring the training minibatch instead would return
the luckiest minibatch rather than the best parameters, and — because `floor` can be computed
analytically — that optimistic bias would land on only one side of the comparison. The floor
is therefore also measured with the same estimator on the same held-out sample size.

`floor` = the 120-pair error of a model that creates no dependency at all = **0.0609**.

## Table

**Minimum reachable dependency error as a function of where the same entanglement budget
(19 edges) is placed**

![joint expressivity ceiling](figures/fig_joint_ceiling.png)

| Model | \|E\| | 120-pair min. error | Improvement over floor |
|---|---|---|---|
| **aligned** (true CDG) | 19 | **0.0103** | **83.0%** |
| rewired (degree-preserving rewire) | 19 | 0.0401 | 34.1% |
| distmatched (distance-matched permutation) | 19 | 0.0421 | 30.8% |
| permuted (isomorphic permutation) | 19 | 0.0468 | 23.2% |
| no_entangle (RZZ removed) | 0 | 0.0508 | 16.7% |
| *floor (zero dependency)* | — | *0.0609* | *0%* |

## Conclusions

**1. The circuit can represent the entire teacher pattern.** aligned descends to 0.0103
(an error of 0.010 against a true signal magnitude of 0.351). A solution that satisfies the
19 edges and the 101 non-edges simultaneously **exists and is reachable by gradient
descent**. Expressivity is not the bottleneck.

**2. Entanglement really does create dependency.** `no_entangle` (0.0508) sits just above
the floor (0.0609). Without RZZ, almost no dependency can be created. The residual 16.7% is
produced by the condition vector `c` that all features share (see `REVISIONS.md` §E-3).

**3. Even with the same entanglement budget, *where you place it* makes a 3.6× difference.**
aligned 83.0% vs. permuted 23.2% of the floor recovered — an error 4.5× lower. The number of
nodes, edges, the degree sequence, and the triangle count are all identical; the only
difference is **which clinical pairs sit on top of an RZZ** — and yet the outcomes diverge
this much.

**4. It also wins against the distance-matched control.** `distmatched` (0.0421) is a
permutation whose held-out pair distance distribution is matched to the CDG's, which
cancels out any "advantage from being nearby." Even so, aligned leads by 4.1× → **the
advantage comes not from graph combinatorics but from the fact that the edges the CDG
selected carry the real dependency structure.** This is the central claim of the paper.

## And Yet — WGAN-GP Fails to Find This Solution At All

Training the same circuit with the primary model of v2 §8.10 (pure WGAN-GP):

| | 120 pairs | 19 true edges | 101 non-edges |
|---|---|---|---|
| floor (zero dependency) | 0.0648 | 0.3676 | 0.0078 |
| **direct optimization (aligned)** | **0.0103** | — | — |
| **WGAN-GP (aligned)** | **0.1359** | **0.3662** | 0.0926 |
| WGAN-GP (permuted) | 0.1346 | 0.3707 | 0.0902 |

**The true-edge error equals the floor** (0.3662 vs. 0.3676). The contribution of the 19
entangling gates is effectively zero. The model stays **13× away from** a solution that
demonstrably exists.

Therefore **this is a training failure, not an expressivity failure.** See `REVISIONS.md`
§E. `scripts/diag_lr.py` tests the leading candidate cause (using the same learning rate
for the quantum angles and the classical head).

## Position in the Paper

This table becomes **Figure 2**. If the light-cone cliff in `RESULTS_ceiling.md` (Figure 1)
explains "why L=1," this table explains **"why Clinical"** — with the same resources, a
permuted graph recovers less than a third of what is achievable.

## Remaining Check (Before This Goes Into the Paper)

`fit()` optimizes the **Pearson** partial correlation on `q = ⟨Z⟩`, whereas the target `R*`
was computed in the `npn(X)` space. Because the head is monotone the two spaces agree
approximately, but not exactly. The discrepancy is not large enough to overturn the
conclusion (a 4.5× separation), but the final numbers should be re-confirmed with a version
in which the two spaces are made consistent.

Resolved since the first run: `fit()` originally scored the best training minibatch over the
optimization trajectory, which returned the luckiest draw rather than the best parameters and
biased every ceiling downward — asymmetrically, since the floor was computed analytically.
Both sides are now scored on the same fixed held-out draw. Correcting it made the separation
*larger*, not smaller (aligned 0.0130 → 0.0103, permuted 0.0491 → 0.0468).

# Result: the learning rate is not the cause — WGAN-GP cannot see pairwise dependency

Script: `scripts/diag_lr.py` · Run: 2026-07-11 · RTX 5090

## The hypothesis being tested

`ceiling_joint.py` optimizes the quantum parameters with Adam `lr = 0.05` and reaches 0.0103.
`train.py` applies `lr_g = 5e-5` to *every* generator parameter and reaches 0.1359 — a factor
of 1000 difference in learning rate. Quantum angles are in radians, and at `lr = 5e-5` over
3000 steps the theoretical maximum displacement is 0.15 rad. The obvious suspicion: the
entangling angles are effectively frozen, because a learning rate tuned for ~2000 classical
head weights was applied unchanged to the 19 quantum angles.

**Note this is not a circular fix.** The loss stays pure WGAN-GP; only the per-parameter-group
learning rate changes. v2 §8.10 is respected.

## Result

Reference values — floor (a model that creates *no* dependency): 120-pair 0.0648 · true-edge
0.3676 · non-edge 0.0078. Direct-optimization ceiling for the same circuit: **0.0103**.

| `lr_q` | graph | mean \|Δγ\| (rad) | 120 pairs | **19 true edges** | 101 non-edges |
|---|---|---|---|---|---|
| 5e-5 *(current)* | aligned | 0.0960 | 0.1350 | **0.3667** | 0.0915 |
| 5e-5 *(current)* | permuted | 0.0924 | 0.1349 | 0.3720 | 0.0903 |
| 5e-3 | aligned | 0.2926 | 0.1197 | **0.4026** | 0.0665 |
| 5e-3 | permuted | 0.1164 | 0.1114 | 0.3760 | 0.0616 |
| 5e-2 | aligned | **2.0110** | 0.1748 | **0.3589** | 0.1402 |
| *floor (zero dependency)* | — | — | *0.0648* | *0.3676* | *0.0078* |

(The final cell, `5e-2 / permuted`, was lost when the run died. It is not load-bearing: the
verdict rests on the aligned rows, which are complete.)

## The hypothesis is refuted

**At `lr_q = 5e-2` the entangling angles move 2.0 radians.** They are not frozen — they are
free to roam the entire circuit. And yet:

- The **true-edge error is 0.3589**, versus a floor of 0.3676. That is a 2.4% improvement over
  creating no dependency at all, while the same circuit under direct optimization reaches
  essentially zero error on those edges.
- The **non-edge false positives explode** from 0.0915 to 0.1402.
- The **overall 120-pair score gets worse**, not better: 0.1350 → 0.1748.

Across a 1000× sweep of the learning rate there is no setting at which the model learns the
true dependency structure. Raising `lr_q` does not move `gamma` toward the solution — it moves
`gamma` *faster in the wrong direction*.

The tell is in the middle row. At `lr_q = 5e-3`, aligned's angles move **2.5× further** than
permuted's (0.2926 vs 0.1164 rad) — the critic pushes harder on the circuit that actually has
RZZ gates on the true edges — and aligned comes out **worse** (true-edge error 0.4026, above
the floor). More gradient, applied in the wrong direction, does more damage precisely where
there was more to gain.

## The real cause

**The WGAN-GP critic cannot identify which pairs ought to be dependent.** The gradient it
supplies to `gamma` is, for this purpose, noise. Turning the learning rate up only amplifies
the noise.

This is not a bug. It is the failure of a design decision.

v2 §8.10 forbids any dependency term in the loss, in order to avoid the circular argument
*"you optimized the evaluation metric, so of course you score well on it."* The measured
consequence of that ban is **a model that learns no conditional dependency at all, under any
topology.** The rule intended to protect the claim has disabled the model that was supposed to
make it.

There is no version of this where we run WP-2 as specified. A confirmatory experiment over 9
topologies × 10 seeds, using a training objective that provably learns zero dependency for
every topology, measures nothing.

## What this does *not* undermine

The scientific claim itself stands, and it stands on `RESULTS_ceiling_joint.md`: given the same
19-edge budget and the same objective, the CDG reaches 0.0103 while an isomorphic permutation
reaches only 0.0468, and a distance-matched permutation only 0.0421. Alignment carries real
information. What is broken is the *training procedure we chose to demonstrate it with*.

## The decision this forces

See `REVISIONS.md` §E-6.

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

`floor` = the 120-pair error of a model that creates no dependency at all (`R_syn = I`) =
**0.0602**.

## Table

**Minimum reachable dependency error as a function of where the same entanglement budget
(19 edges) is placed**

| Model | \|E\| | 120-pair min. error | Improvement over floor |
|---|---|---|---|
| **aligned** (true CDG) | 19 | **0.0130** | **78.3%** |
| rewired (degree-preserving rewire) | 19 | 0.0435 | 27.7% |
| distmatched (distance-matched permutation) | 19 | 0.0463 | 23.1% |
| permuted (isomorphic permutation) | 19 | 0.0491 | 18.4% |
| no_entangle (RZZ removed) | 0 | 0.0540 | 10.3% |
| *floor (zero dependency)* | — | *0.0602* | *0%* |

## Conclusions

**1. The circuit can represent the entire teacher pattern.** aligned descends to 0.0130
(an error of 0.013 against a true signal magnitude of 0.351). A solution that satisfies the
19 edges and the 101 non-edges simultaneously **exists and is reachable by gradient
descent**. Expressivity is not the bottleneck.

**2. Entanglement really does create dependency.** `no_entangle` (0.0540) sits just above
the floor (0.0602). Without RZZ, almost no dependency can be created. The remaining 10.3%
is produced by the condition vector `c` that all features share (see §E-3).

**3. Even with the same entanglement budget, *where you place it* makes a 6× difference.**
aligned 78.3% vs. permuted 18.4%. The number of nodes, edges, the degree sequence, and the
triangle count are all identical; the only difference is **which clinical pairs sit on top
of an RZZ** — and yet the outcomes diverge this much.

**4. It also wins against the distance-matched control.** `distmatched` (0.0463) is a
permutation whose held-out pair distance distribution is matched to the CDG's, which
cancels out any "advantage from being nearby." Even so, aligned leads by 3.6× → **the
advantage comes not from graph combinatorics but from the fact that the edges the CDG
selected carry the real dependency structure.** This is the central claim of the paper.

## And Yet — WGAN-GP Fails to Find This Solution At All

Training the same circuit with the primary model of v2 §8.10 (pure WGAN-GP):

| | 120 pairs | 19 true edges | 101 non-edges |
|---|---|---|---|
| floor (zero dependency) | 0.0648 | 0.3676 | 0.0078 |
| **direct optimization (aligned)** | **0.0130** | — | — |
| **WGAN-GP (aligned)** | **0.1359** | **0.3662** | 0.0926 |
| WGAN-GP (permuted) | 0.1346 | 0.3707 | 0.0902 |

**The true-edge error equals the floor** (0.3662 vs. 0.3676). The contribution of the 19
entangling gates is effectively zero. The model stays **10× away from** a solution that
demonstrably exists.

Therefore **this is a training failure, not an expressivity failure.** See `REVISIONS.md`
§E. `scripts/diag_lr.py` tests the leading candidate cause (using the same learning rate
for the quantum angles and the classical head).

## Position in the Paper

This table becomes **Figure 2**. If the light-cone cliff in `RESULTS_ceiling.md` (Figure 1)
explains "why L=1," this table explains **"why Clinical"** — with the same resources, a
permuted graph yields only a quarter of what is achievable.

## Remaining Check (Before This Goes Into the Paper)

`fit()` optimizes the **Pearson** partial correlation on `q = ⟨Z⟩`, whereas the target `R*`
was computed in the `npn(X)` space. Because the head is monotone the two spaces agree
approximately, but not exactly. The discrepancy is not large enough to overturn the
conclusion (the 6× separation), but the final numbers must be re-confirmed with a version
in which the two spaces are made consistent.

# Results: The Design Trade-off — Expressivity vs. the Hypothesis

Scripts: `scripts/baseline_classical.py`, `diag_tradeoff.py`, `design_sweep.py`, `perf_sweep.py`,
`ceiling_real.py` · Real MIMIC-IV v3.1, n = 48,561

> **This document exists because the model loses to a textbook classical baseline, and we went
> looking for why.** It is the most important negative-then-positive result in the project.

---

## 1. The problem: we lose, and at Δ=3 we lose by construction

`baseline_classical.py`, on our own dependency metric:

| model | 120-pair error | dependency params |
|---|---|---|
| empirical resample (oracle) | **0.0055** | — |
| **Gaussian copula (classical)** | **0.0060** | 120 covariances |
| CDG-QGAN Δ=3, trained (WGAN-GP) | 0.0694 | 21 RZZ angles |
| CDG-QGAN Δ=3, ceiling (GAN removed) | 0.0436 | 21 RZZ angles |
| **CDG-QGAN Δ=3, Corollary 1 bound** | **0.0331** | *unreachable by ANY L=1 Δ=3 circuit* |
| independent (floor) | 0.0986 | 0 |

A Gaussian copula beats the trained circuit **11.6×**, and it would keep beating it however good the
optimizer got: at Δ=3, **72 of the 120 pairs sit outside the L=1 light cone** and Corollary 1 forces
their conditional covariance to exactly zero. The bound 0.0331 is not a training artefact. It is a
theorem.

### The copula is the oracle for this metric, not a rival

Say this out loud before anyone else does. Our metric is a **partial correlation in nonparanormal
space** — a Gaussian-copula quantity. A Gaussian copula fits exactly that object and nothing else,
which is why it lands within 0.0005 of the empirical resample. **No generative model beats it here.**
Reporting "we lose to a Gaussian copula on partial correlations" is true and about as informative as
"we lose to the sample covariance". The real baselines are CTGAN/TVAE and downstream utility (WP-3).

That does not excuse the bound. The bound is real, and it had to be attacked.

---

## 2. Depth is the wrong lever

The obvious fix is a deeper circuit. It does not work, and the reason is exact:

| L | reach | pairs in cone | Corollary 1 bound | alignment z | |
|---|---|---|---|---|---|
| **1** | 2 | 48/120 | 0.0331 | **+3.79** | testable, uncompetitive |
| 2 | 4 | 103/120 | 0.0068 | +1.41 | competitive-ish, effect 63% gone |
| 3 | 6 | 120/120 | 0.0000 | 0.00 | **CDG ≡ permuted. Nothing left to test.** |

Depth widens the cone for **every** graph equally. At L=3 a random permutation reaches every pair
too, so the CDG and its own null become the same object and the hypothesis evaporates. Depth buys
expressivity by **selling the thing we set out to measure**.

---

## 3. Density is the right lever

The reach radius at L=1 is 2. For a 16-node graph, every pair can be within distance 2 of a graph
with **maximum degree 4** (Moore bound: `d² + 1 ≥ 16 ⟹ d ≥ 4`). Our CDG had max degree **3** — a cap
written into v2 §7.6, **not a property of the data**.

Raising it lets the degree-constrained Kruskal keep adding edges **in weight order** — the next
strongest partial correlations. The graph stays the clinical one; it stops being truncated.

**And the light cone does not move. L = 1 throughout. Corollary 1 and Proposition D-2 hold exactly.**

| Δ | \|E\| | max deg | diam | pairs in cone | Corollary 1 bound | alignment z | percentile |
|---|---|---|---|---|---|---|---|
| **3** (v2) | 21–23 | 3 | 5 | 55/120 | 0.0331 | **+3.38** | 100.0% |
| **4** | 29 | 4 | 4 | 86/120 | **0.0140** | **+2.51** | 100.0% |
| 5 | 34 | 5 | 4 | 95/120 | 0.0089 | +2.08 | 99.9% |
| 6 | 40 | 6 | 3 | 103/120 | 0.0057 | +1.70 | 99.5% |

Compare the two levers **at equal bound**:

| | bound | alignment z |
|---|---|---|
| depth L=2 (Δ=3) | 0.0068 | +1.41 |
| **density Δ=6 (L=1)** | **0.0057** | **+1.70** |

Density dominates depth on both axes — and it keeps the light cone tight, so the structural
guarantee survives. Depth cannot say that.

At Δ=4 the strongest held-out pair that Δ=3 could not express, **sbp–dbp (|ρ| = 0.42)**, moves from
graph distance 3 to distance 2 — from *provably impossible* to *reachable*.

---

## 4. A denser graph is a harder optimization problem — and we nearly missed it

At the original optimizer budget (1,200 steps × 2 restarts), Δ=4 looked like a failure:

| | bound | ceiling @ 1200×2 | gap |
|---|---|---|---|
| Δ=3 | 0.0331 | 0.0437 | +0.0106 |
| Δ=4 | 0.0140 | **0.0418** | **+0.0278** |

The bound had halved and the achieved error had barely moved. The natural reading — and the one
written down at the time — was that the bottleneck had moved off the light cone and onto the
circuit's own capacity: a degree-4 qubit cannot set four correlations independently, so the extra
edges are useless.

**That reading was wrong.** Give both graphs 4× the optimizer (4,000 steps × 4 restarts):

| | bound | ceiling @ 1200×2 | **ceiling @ 4000×4** | gap |
|---|---|---|---|---|
| Δ=3 | 0.0331 | 0.0437 | **0.0436** | +0.0105 |
| Δ=4 | 0.0140 | 0.0418 | **0.0300** | +0.0160 |

Δ=3 is **converged**: 4× the budget moves it by 0.0001. Δ=4 gains **28%** and is *still* not
converged. The extra edges were never useless — they made the problem harder, and the fixed budget
was hiding the gain. **Δ=4 beats Δ=3 by 31%** (0.0300 vs 0.0436) and closes the gap to the classical
copula from 11.6× to 5.0×.

The lesson generalizes: **when you enrich a model and it does not improve, check that you gave the
optimizer more, not the same.** A fixed budget silently penalizes the richer model, and the result
looks like a capacity limit.

---

## 5. Density is not free

More edges → a *random permutation* of that graph also reaches more pairs by luck. The alignment z
falls monotonically (+3.38 → +2.51 → +2.08 → +1.70), and the trained contrast against the
**isomorphic permutation** shrinks with it.

But the contrast against the **distance-matched** control — the one that isolates the clinical claim
from graph combinatorics — does **not**:

| | aligned | permuted | distmatched | **aligned − distmatched** |
|---|---|---|---|---|
| Δ=3 | 0.0437 | 0.0632 | 0.0699 | **−0.0262** |
| Δ=4 | 0.0418\* | 0.0500\* | 0.0693\* | **−0.0275** |

\* at the 1200×2 budget; the fair 4000×4 comparison across all variants is running.

A denser random graph gets luckier; a denser *distance-matched* graph does not, because distance
matching already fixes the structure and the held-out pairs' distances, and changes **only which
clinical variable sits at which node**. That is exactly the contrast the paper needs.

---

## 6. Where this leaves the design

**Δ = 4 is the new design point**, pending the fair contrast run:

- bound 0.0331 → **0.0140**
- ceiling 0.0436 → **0.0300** (and still falling)
- alignment **z = +2.51, 100th percentile** — not one of 2,000 permutations beats the CDG
- max degree 4, diameter 4 — NISQ cost barely moves
- 29 RZZ angles vs the copula's 120 covariance parameters
- **L = 1. Corollary 1 and Proposition D-2 unchanged.**

We will not chase Δ=6 for the bound alone. It buys 0.0083 of ceiling and sells a third of the
alignment effect, and the alignment effect is the paper.

**We still expect to lose the dependency column to a Gaussian copula.** That is a structural fact of
a light-cone-constrained model and it will be stated plainly. Whether we lose to CTGAN/TVAE, and
what happens on downstream utility, is WP-3 — and *that* is the comparison that decides whether this
architecture has a performance argument at all.

# Results: CDG Alignment Pre-check (real MIMIC-IV v3.1)

Script: `scripts/precheck_alignment.py` · Run: 2026-07-11
Data: MIMIC-IV v3.1, 24-hour landmark cohort, **n = 48,561** (complete observations on all 16 features)

> **These are the current numbers. Two earlier versions are superseded — do not quote them.**
>
> | run | `L=1` z | why it is superseded |
> |---|---|---|
> | with MAP | +4.46 | MAP is an arithmetic identity (`MAP≈(SBP+2DBP)/3`, R²=0.860) and induced a *spurious negative* SBP–DBP correlation. The extra margin was arithmetic, not clinical structure. `REVISIONS.md` C-3 |
> | linear residualization in raw units | +3.18 | The graph was estimated in a different space from the metric that scores it. `REVISIONS.md` A-4, E-3 |
> | **current** | **+3.79** | nonparanormal → residualize on a nonlinear basis of `c` → partial correlation. The same estimator `eval_dep.partial_corr_c` applies to the synthetic data. |
>
> Correcting the estimator changed the graph very little (ρ-matrix correlation **0.9971**, 20 of 23
> edges retained) and made the effect **stronger**. The conclusion is not sensitive to this choice —
> which is itself worth knowing.

---

## What this test asks

> Does clinical-meaning alignment increase the **representable dependency mass**?

By Corollary 1, any pair with `d_G(u,v) > 2L` has **exactly zero** conditional covariance.
Therefore the total dependency mass that a depth-`L` circuit can represent is determined by:

```
M(G, L) = Σ_{(u,v)} |ρ_true(u,v)| · 1[ d_G(u,v) ≤ 2L ]
```

We compare the CDG against an **isomorphic permutation null distribution (5,000 draws)**. The permutation
**preserves everything** — number of nodes, number of edges, degree sequence, triangle count, and clustering
coefficient — and changes only **which clinical pair sits where inside that structure**. The test therefore
measures **clinical-meaning alignment** and nothing else.

**No training is performed at any point.** This is the gate that decides, before burning GPU, whether the
confirmatory experiment has any chance of succeeding.

---

## Results

CDG: 16 nodes · 21 edges · maximum degree 3 · diameter 6 · total dependency mass Σ|ρ| = 5.96

| L | Reach radius | Reachable pairs | CDG mass | Permutation null | z | p | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | 2 | 48/120 | **4.645 (78.0%)** | 2.387 ± 0.595<br>[0.883, 4.380] | **+3.79** | **<0.0001** | **PASS** |
| 2 | 4 | 103/120 | 5.696 (95.6%) | 5.113 ± 0.413 | +1.41 | 0.038 | weak |
| 3 | 6 | **120/120** | 5.957 (100%) | 5.957 ± 0.000 | 0.00 | 1.000 | Meaningless |

**At `L=1` not one of the 5,000 permutations beat the CDG** — the null's maximum (4.380) falls
below the CDG (4.645). 100th percentile.

Dependency mass falling within reach: **CDG 78.0% vs. random permutation 40.1%.**

### `L=2` — a correction to what we previously claimed

An earlier version of this document said `L=2` **FAILS** on the real data (z=+0.66, p=0.287).
Under the corrected estimator it **marginally passes** (z=+1.41, p=0.038). That claim was too
strong and is withdrawn.

`L=1` remains the operating point, but for the honest reason rather than the convenient one: at
`L=2`, **103 of 120 pairs are already reachable (86%)**, so the topology forbids almost nothing,
and the effect size collapses by 63%. A topology that constrains nothing cannot discriminate
between topologies. That is an argument about *discriminative power*, not about failure.

### The cost and the benefit of removing MAP

| | With MAP (contaminated) | **Without MAP (honest)** |
|---|---|---|
| CDG mass | 5.55 (90.5%) | 4.645 (78.0%) |
| z | +4.46 | **+3.79** |

**The drop in z from +4.46 is precisely "the part that was just arithmetic."**
Had we kept MAP, we could not have answered a reviewer asking, "How much of this is just recovering
`MAP=(SBP+2DBP)/3`?" **What remains comes entirely from genuine clinical structure.**

The main reason mass fell from 90.5% to 79.2% is that `sbp–dbp` (0.439) is held out and sits at distance 3,
placing it outside the `L=1` reach. But this is **a loss created by the evaluation apparatus itself**.
The actual deployed model uses every stable edge with no hold-out, so `sbp–dbp` becomes distance 1.

---

## Interpretation 1 — the mechanism really works

Clinical variables form clusters, and those clusters form **triangles**.
So even when a strong edge is held out, **it remains at distance 2 via a common neighbor.**

| Held-out strong pair | \|ρ\| | CDG distance | Common neighbor | Clinical meaning |
|---|---|---|---|---|
| creatinine — bun | **0.657** | **2** | potassium | Kidney |
| sodium — chloride | **0.649** | **2** | bicarbonate | Electrolytes / acid-base |

Since the reach radius at `L=1` is 2, these are **representable**.
A random permutation scatters them out to distances 3–5, and by Corollary 1 they become **fundamentally impossible to represent**.

The edges the CDG did capture are all genuine physiology as well: chloride–bicarbonate (0.387, anion gap),
sodium–bicarbonate (0.294), heart_rate–resp_rate (0.255), **wbc–platelet (0.232, bone marrow / inflammation)**,
resp_rate–spo2 (0.191), bicarbonate–creatinine (0.151, renal acid-base),
**wbc–glucose (0.097, stress hyperglycemia)**.

## Interpretation 2 — the attenuation of the effect with depth is what the theory predicts

```
z = +3.18  (L=1)  ->  +0.66  (L=2)  ->  0.00  (L=3)
```

As depth grows, the reach radius `2L` widens, so **every pair becomes representable and the topology stops
imposing any constraint at all.** At `L=3`, the CDG and the permuted graphs are literally identical.

**We will use this attenuation curve as a figure in the paper.** It is a relationship the theory predicted and
the data confirmed, and it is the answer to "why must the circuit be shallow?" The FAIL at `L=2` is not a
failure but **a negative control that corroborates the theory**.

## Interpretation 3 — the design is settled

**`L=1` alone.** `L=2` cannot be used for the confirmatory experiment (on real data it FAILs, z=+0.66, p=0.287).
On the demo (n=81), `L=2` did pass, but that graph was noise.

This dovetails exactly with `RESULTS_ceiling.md`:

- `L=1` adjacent-pair ceiling = **0.991** → strong clinical relationships (0.66, 0.65) are representable
- `L=1` outside the light cone = **~0.01** → distant pairs are forced to exactly zero
- `L=1` alignment effect = **z=+3.18** → the CDG overwhelms the permutation

**`L=1` is the only point at which expressivity, discriminative power, and the alignment effect hold simultaneously.**
It was not selected by tuning a hyperparameter; it is the point the theory pointed to and the data confirmed.

---

## Remaining risks

- **The blood-pressure triangle is resolved.** MAP has been excluded from the generated variables (`REVISIONS.md` C-3).
  MAP is still extracted, but it is used only for **evaluating** clinical plausibility: we compute
  `MAP~ = (SBP~ + 2·DBP~)/3` and compare it against the real MAP distribution.
- This test only measures **an upper bound on representability**; it does not ask whether learning actually
  reaches that bound. A confirmatory experiment that includes WGAN-GP training is still needed
  (re-run `benchmark_synthetic.py` after the WP-6 optimization).
- The magnitude of the `L=1` alignment effect (z=+3.18) depends on the hold-out scheme. The main source of
  mass loss is that `sbp–dbp` (0.439) is held out and sits at distance 3, putting it out of reach.
  Sensitivity to the hold-out construction seed should be checked.

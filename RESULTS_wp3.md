# Results: WP-3 — against classical baselines

`scripts/wp3_baselines.py` (classical) · `scripts/wp3_qgan.py` (ours, v3 loss) · Real MIMIC-IV v3.1
Train n = 36,420 · **held-out real test n = 12,141** · 3 seeds · same split, same metrics, same code

> Every earlier comparison in this project was **internal** — the CDG against permuted,
> distance-matched, rewired, no-entangle versions of *itself*. Those show that *where* the
> entanglement goes matters (WP-2, and it does). They never showed the model was any good.

---

## The table

| model | dep. error | TSTR AUROC | TSTR AUPRC | marginal W₁ | dependency params |
|---|---|---|---|---|---|
| *real data (ceiling)* | — | **0.8569** | 0.4514 | — | — |
| gaussian copula (≈**oracle**, §1) | **0.0064** | **0.8073** | 0.3388 | 0.0070 | 120 |
| **CDG-QGAN Δ=4** + calibrated | 0.0765 | 0.7803 | 0.2914 | 0.0004 | **29** |
| **CDG-QGAN Δ=3** + calibrated | 0.0762 | 0.7807 | 0.2880 | 0.0004 | **21** |
| **no_entangle** + calibrated | 0.0975 | **0.7848** | 0.2913 | 0.0004 | **0** |
| **TVAE** | 0.0708 | 0.7736 | 0.3105 | 0.1938 | ~10⁶ |
| **CTGAN** | 0.0935 | 0.6999 | 0.2135 | 0.1724 | ~10⁶ |
| *independent (floor)* | *0.0990* | *0.6823* | *0.1805* | *0.0065* | *0* |
| CDG-QGAN Δ=4, ceiling (no GAN) | *0.0300* | — | | | 29 |
| CDG-QGAN Δ=4, Corollary 1 bound | *0.0140* | — | | | *unreachable* |

Raw (uncalibrated) rows, for the invariant check in §2: Δ=3 `0.0762 / 0.7740 / 0.1189`,
Δ=4 `0.0765 / 0.7777 / 0.1100`, no_entangle `0.0975 / 0.7729 / 0.1330`.

---

## 1. The Gaussian copula is the oracle, not a rival

Our dependency metric is a **partial correlation in nonparanormal space** — a Gaussian-copula
quantity. A Gaussian copula fits precisely that object and lands **0.0009 from the empirical
resample** (0.0064 vs 0.0055). Saying "we lose to a Gaussian copula on partial correlations" is
about as informative as "we lose to the sample covariance."

It is in the table for **calibration**, and must be labelled as such in the paper. Note that it does
not win everything: its TSTR (0.8073) is below the real-data ceiling (0.8569). It is unbeatable on
the one axis that is its own definition — and it is still the best generator in the table on TSTR,
which is a fact about this dataset that we report rather than hide.

## 2. The invariant check passed

Each model is scored twice — raw, and after a per-feature monotone quantile map onto the training
marginals. A monotone per-feature map **cannot change the copula**, so the dependency column *must*
be identical between the two rows. It is, to four decimals, for all three models:

```
Δ=3          0.0762  /  0.0762
Δ=4          0.0765  /  0.0765
no_entangle  0.0975  /  0.0975
```

That identity is the evidence that the dependency metric is measuring the copula and nothing else,
and that the calibration is not smuggling in structure. Only TSTR and W₁ move.

## 3. CTGAN barely beats a model that creates no dependency at all

| | dep. error | TSTR AUROC |
|---|---|---|
| **CTGAN** | **0.0935** | **0.6999** |
| **independent (floor)** | **0.0990** | **0.6823** |

CTGAN, with ~10⁶ parameters, lands **5 % below the floor on dependency**. It has learned the
marginals and almost nothing else.

**This is exactly the failure we diagnosed in ourselves and fixed** (§E-7): a tabular GAN whose
decoder can fit the marginals will let the critic settle for the marginals, and then nothing learns
the joint — for any architecture, any topology, any hyperparameter. We caught it only because we
scored a do-nothing model first; our own WGAN-GP was, at the time, **2.1× worse than doing nothing**.

**This is a reusable contribution and it is independent of the quantum claim:** *always score the
zero-dependency model; if your generator does not beat it, your dependency numbers mean nothing.*
That check would have caught CTGAN on this dataset too.

## 4. The pre-registered TSTR prediction was REFUTED

`wp3_qgan.py` carried this, written before the run:

> *"On TSTR we expect to do better than [the dependency ratio] suggests, because the CDG
> deliberately places the STRONG dependencies inside the light cone … If that holds it is a finding;
> if it does not, the honest conclusion is that at L=1 the model has no performance argument at all
> and the paper rests on the structural result."*

It does not hold.

| | dep. error | TSTR AUROC |
|---|---|---|
| CDG-QGAN Δ=4 (calibrated) | **0.0765** | 0.7803 |
| **no_entangle** (calibrated) | **0.0975** | **0.7848** |

**A circuit with zero entangling gates beats the CDG on TSTR.** The CDG is 22 % better on the
dependency metric and *loses* on the downstream task. The entire cross-feature dependency
structure — the subject of this paper — is worth **nothing** on in-ICU mortality prediction.

An earlier single-seed run had reported the opposite (CDG 0.7856 vs no_entangle 0.7710, "+0.015").
**That was seed noise.** At three seeds the ordering reverses. It is recorded here because it is
exactly the kind of number a paper gets built on by accident.

We beat TVAE (0.7803 vs 0.7736) — but so does `no_entangle` (0.7848). **The win is not coming from
the topology.** It is coming from the conditional marginals plus the quantile calibration, and any
model with those would get it. Claiming a topology win on TSTR would be false.

## 5. Δ=4 did not beat Δ=3 under training

| | Corollary 1 bound | ceiling (no GAN) | **trained** |
|---|---|---|---|
| Δ=3 | 0.0329 | 0.0435 | **0.0762** |
| Δ=4 | 0.0138 | 0.0300 | **0.0765** |

The redesign (§E-9) halved the bound and improved the achievable ceiling by 31 %. Under GAN training
it bought **nothing** — the two are within noise.

That is not a failure of the redesign; it is a measurement of where the real constraint is. The
trained model sits at 0.0765 against a ceiling of 0.0300, so it is realising **less than half** of
what its own circuit can express. **The binding constraint is the optimizer, not the light cone**, and
until that changes, widening the light cone cannot help. Same conclusion WP-2 reaches from the other
direction.

## 6. What has to be said plainly in the paper

- On the dependency metric we **lose to a Gaussian copula, by construction and forever at L=1**.
  Corollary 1 forces 34/120 pairs to exactly zero at Δ=4. The bound is a theorem. We state it.
- We **beat CTGAN**, and we **lose to TVAE on dependency** (0.0765 vs 0.0708) with 29 angles against
  ~10⁶ parameters. That ratio is worth reporting; the win is not.
- **The topology result is real and the performance result is not.** The CDG produces measurably and
  significantly better conditional dependency structure than every matched control (WP-2, decisive
  contrast CI [−0.0202, −0.0109], complete seed separation). It produces **no** measurable downstream
  benefit on this task.
- **This is a structural paper, not a performance paper**, and the abstract must say so. A paper that
  reports the first result and omits the second is selling something.

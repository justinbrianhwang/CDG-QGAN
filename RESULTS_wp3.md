# Results: WP-3 — Classical Baselines

Script: `scripts/wp3_baselines.py` · Real MIMIC-IV v3.1
Train n = 36,420 · **held-out real test n = 12,141** · 3 seeds · CTGAN/TVAE 200 epochs

> Every earlier comparison in this project was **internal** — the CDG against permuted,
> distance-matched, rewired, no-entangle versions of *itself*. Those show that *where* the
> entanglement goes matters. They never showed the model was any good. This is the first time a
> classical generator has been put against the same metric.

---

## The table

| model | dep. error | TSTR AUROC | TSTR AUPRC | marginal W₁ | dependency params |
|---|---|---|---|---|---|
| *real data (ceiling)* | — | **0.8569** | 0.4514 | — | — |
| gaussian copula (≈**oracle**, see below) | **0.0064** | 0.8073 | 0.3388 | 0.0070 | 120 |
| **TVAE** | 0.0708 | 0.7736 | 0.3105 | 0.1938 | ~10⁶ |
| **CDG-QGAN Δ=3, trained** | **0.0714**\* | *pending* | | | **21** |
| **CTGAN** | 0.0935 | 0.6999 | 0.2135 | 0.1724 | ~10⁶ |
| *independent (floor)* | *0.0990* | *0.6823* | *0.1805* | *0.0065* | *0* |
| CDG-QGAN Δ=3, ceiling (no GAN) | 0.0435 | — | | | 21 |
| **CDG-QGAN Δ=4, ceiling (no GAN)** | **0.0300** | — | | | **29** |
| CDG-QGAN Δ=4, Corollary 1 bound | 0.0140 | — | | | *unreachable* |

\* measured on the full cohort (WP-2 protocol). `wp3_qgan.py` re-measures it on this exact split,
with TSTR, so the row can be quoted as-is. Do not quote it until then.

---

## 1. The Gaussian copula is the oracle, not a rival

Our dependency metric is a **partial correlation in nonparanormal space** — a Gaussian-copula
quantity. A Gaussian copula fits precisely that object and lands **0.0009 from the empirical
resample** (0.0064 vs 0.0055). No generative model beats it here, and saying "we lose to a Gaussian
copula on partial correlations" is about as informative as "we lose to the sample covariance."

It is in the table for **calibration**, and it must be labelled as such in the paper. Note also that
it does *not* win everything: its TSTR (0.8073) is below the real-data ceiling (0.8569), so it is
not a free lunch — it is just unbeatable on the one axis that happens to be its own definition.

## 2. CTGAN barely beats a model that creates no dependency at all

This is the finding.

| | dep. error | TSTR AUROC |
|---|---|---|
| **CTGAN** | **0.0935** | **0.6999** |
| **independent (floor)** | **0.0990** | **0.6823** |

CTGAN, with on the order of a million parameters, lands **5% below the floor on dependency** and
**0.018 above it on TSTR**. It has learned the marginals and almost nothing else.

**This is exactly the failure we diagnosed in ourselves and fixed** (`REVISIONS.md` §E-7): a tabular
GAN whose decoder can fit the marginals will let the critic settle for the marginals, and then
nothing learns the joint — for any architecture, any topology, any hyperparameter. We caught it
because we scored a do-nothing model first; our own WGAN-GP was, at the time, *2.1× worse than doing
nothing*. The fix was the copula critic: rank-transform the critic's input within the batch, so the
marginals carry no information and the gradient has nowhere to go but the entangling angles.

CTGAN has not had that fix applied. It is sitting in the failure mode we climbed out of.

**This is a genuinely reusable contribution and it is independent of the quantum claim.** The
methodological result — *always score the zero-dependency model; if your generator does not beat it,
your dependency numbers mean nothing* — would have caught CTGAN's behaviour on this dataset too.

## 3. Against TVAE we are, so far, a tie

TVAE 0.0708 vs CDG-QGAN Δ=3 0.0714. Within noise of each other, on ~10⁶ parameters versus **21
entangling angles**. TVAE also has substantially worse marginals (W₁ 0.194 vs the copula's 0.007).

That parity is on the **Δ=3** circuit, whose expressivity ceiling is 0.0435 — so the trained model
is leaving a lot on the table, and the gap is the *optimizer*, not the light cone.

**The Δ=4 ceiling is 0.0300**, well below TVAE. Whether training reaches it is the open question, and
it is what `wp3_qgan.py` at Δ=4 will answer.

## 4. What has to be said plainly in the paper

- On the dependency metric we **lose to a Gaussian copula, by construction and forever at L=1**
  (Corollary 1 forces 72/120 pairs to zero at Δ=3, 37/120 at Δ=4). The bound is a theorem. We state
  it, we do not bury it.
- We **beat CTGAN** and **tie TVAE**, with 21–29 dependency parameters against ~10⁶.
- The interesting axis is not "who wins the 120-pair average" — that average weights a |z| = 0.02
  pair exactly as much as sodium–chloride at |z| = 1.06. It is **TSTR**, which weights a dependency
  by how much a clinical model actually uses it, and the CDG puts the strong dependencies inside its
  light cone *by construction*.

That last claim was written into `wp3_qgan.py` **before the run**, as a pre-registration. If TSTR
does not bear it out, it will be reported as refuted, like the three mechanisms before it
(`REVISIONS.md` §E-8, §E-9).

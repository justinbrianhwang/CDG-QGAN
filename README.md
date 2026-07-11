# CDG-QGAN

A hybrid quantum-classical GAN that takes the Clinical Dependency Graph as the
**entanglement topology** of a shallow parameterized quantum circuit, to generate synthetic
MIMIC-IV ICU data.

The core claim: if the conditional dependency graph among clinical variables (the CDG) is
planted directly into the circuit's RZZ layout, it recovers the dependency structure better
than an **isomorphic permutation** of that same graph using identical resources. In other
words, "Clinical" is not a mere modifier — it demonstrably contributes bits.

## Skeleton of the design

- **1 feature = 1 qubit.** Each feature gets its own local latent variable `z_u`, encoded as
  an angle.
- **RZZ only on CDG edges.** The entangling angle `gamma` is the only parameter that can
  create cross-feature dependency.
- **A per-feature one-dimensional local head** `h_u(q_u, c)`. It cannot see `q_v` from any
  other qubit. The classical parameters therefore **structurally cannot create** conditional
  cross-feature dependency (`model.assert_no_cross_feature_mixing` verifies that the head
  Jacobian is exactly diagonal).
- **Proposition 1 (light cone).** With a product initial state and local angle encoding,
  `<Z_u>` depends only on qubits within graph distance `L` of `u`. Corollary: if
  `d_G(u,v) > 2L` then `Cov(x_u, x_v | c) = 0` — exactly, not approximately. We also exploit
  this identity computationally: instead of a 2^16 statevector we simulate only a
  2^|N_L(u)| subcircuit per qubit.

### The light cone is exact, and it is why the circuit must be shallow

![the light-cone cliff](figures/fig_lightcone_cliff.png)

Optimizing the circuit to maximize the correlation of a single pair at graph distance `d`,
the achievable `|ρ|` falls off a cliff at exactly `d = 2L` — independently reproduced at
`L=1` and `L=2`. Outside the cone the maximum is 0.012, which is sampling noise, not
expressivity.

This is what forces `L=1`. At `L=3` all 120 pairs fall inside the light cone, the topology
constrains nothing, and the CDG becomes **identical to a permuted graph by definition**. The
alignment effect decays exactly as the theory predicts:

![alignment effect decays with depth](figures/fig_alignment_decay.png)

`L=1` is not a compromise. It is the only operating point at which expressivity and
discriminative power hold simultaneously. See `REVISIONS.md` A-1 for the full argument.

## Results

### Alignment carries real information — the CDG beats every control

With the GAN removed and the circuit optimized directly on the full 120-pair dependency
pattern, the placement of the entangling gates dominates the outcome. Every variant gets the
**same 19-edge budget**; only *which clinical pair sits on top of an RZZ* differs.

![joint expressivity ceiling](figures/fig_joint_ceiling.png)

`distmatched` is the control that matters most: it matches the CDG's held-out pair distance
distribution, so the "advantage of being nearby" is cancelled out. The CDG still leads it by
4.1×. **The advantage comes from the edges the CDG chose carrying real dependency structure,
not from graph combinatorics.** Details: `RESULTS_ceiling_joint.md`.

### Confirmed so far

| | Result | Document |
|---|---|---|
| The light-cone cliff appears exactly at `d = 2L` | outside it, max \|ρ\| = 0.012 | `RESULTS_ceiling.md` |
| Adjacent-pair expressivity ceiling (`L=1`) | \|ρ\| = 0.991 | `RESULTS_ceiling.md` |
| Alignment precheck `M(G,L)` on the real CDG | `L=1`: z = +3.18 (p = 0.0004) · `L=2`: z = +0.66 (rejected) | `RESULTS_precheck.md` |
| Joint 120-pair ceiling, CDG vs. controls | aligned 0.0103 vs. permuted 0.0468 (floor 0.0609) | `RESULTS_ceiling_joint.md` |
| Light-cone simulator optimization | ~70× speedup (>1800 s → 25.7 s / seed) | `WP6_REPORT.md` |

## Unresolved — current top priority

**WGAN-GP does not train the entangling angles at all.** The circuit demonstrably *can*
represent the target pattern (above), but adversarial training finds none of it.

| | 120 pairs | 19 true edges | 101 non-edges |
|---|---|---|---|
| a model that creates **no dependency at all** | 0.0648 | 0.3676 | 0.0078 |
| **direct optimization** (aligned) | **0.0103** | — | — |
| **WGAN-GP** (aligned) | 0.1359 | **0.3662** | 0.0926 |
| WGAN-GP (permuted) | 0.1346 | 0.3707 | 0.0902 |

Three things are wrong at once, and they compound:

1. **The trained model is 2× worse than doing nothing.** A model that generates zero
   dependency scores 0.0648; WGAN-GP scores 0.1359.
2. **The entangling gates contribute nothing.** The true-edge error (0.3662) equals the floor
   (0.3676). The model stays 13× away from a solution it can demonstrably reach.
3. **It manufactures dependency that does not exist.** 0.0926 of spurious partial correlation
   on the 101 non-edges — and a model with *zero* entangling gates manufactures 0.0859 of it
   too. So the culprit is not the entanglement but **the condition vector `c` that every
   feature shares**, which the metric fails to control for. The CDG is *defined* conditionally
   on `c` while the metric measures an *unconditional* partial correlation — an estimator
   mismatch that is present in the real MIMIC pipeline too, not just the benchmark.

Consequence: the four graph variants all collapse onto ~0.135 and the confirmatory contrast
measures nothing. **Running the 90-run confirmatory experiment in this state would be
wasted.** See `REVISIONS.md` §E for the full diagnosis and the fix list.

## Data

MIMIC-IV / eICU are subject to PhysioNet **credentialed access + a Data Use Agreement**.
Neither the raw data nor any **patient-level** data derived from it may ever enter this
repository. All of it stays in the local archive only. Paths and acquisition instructions are
in `DATA.md`.

## Environment

```
conda env create -f environment.yml   # env name: cdg-qgan
```

**numpy is pinned to 2.2.6. Do not upgrade it** — torch 2.11+cu128 is built against the
numpy 2.2 ABI, and upgrading breaks the loading of `shm.dll`. Any additional installation
must go through the constraints file:

```
python -m pip install -c .backup/constraints.txt <pkg>
```

`tools/mimic-code` is cloned separately:

```
git clone https://github.com/MIT-LCP/mimic-code tools/mimic-code
```

## Documents

- `CDG-QGAN_research_plan_v2.md` — the research plan (the reference document)
- `REVISIONS.md` — **where this conflicts with the plan, this takes precedence.** Confirmed
  corrections, and §E, the open problem above
- `HANDOFF.md` — work package specifications (WP-2 through WP-6)
- `DATA.md` — data locations and environment pitfalls
- `RESULTS_ceiling.md` · `RESULTS_ceiling_joint.md` · `RESULTS_precheck.md` — results
- `FIGURES.md` — figure specifications
- `CDG-QGAN_research_plan_v1.md` — superseded, kept for history

Figures under `figures/` are produced by `scripts/make_figures.py` from the measured values.
Nothing in this repository is an artist's impression of a result.

# CDG-QGAN

A hybrid quantum-classical GAN that takes the Clinical Dependency Graph as the
**entanglement topology** of a shallow parameterized quantum circuit. The goal is the
synthetic generation of MIMIC-IV ICU data.

The core claim: if the conditional dependency graph among clinical variables (the CDG) is
planted directly into the circuit's RZZ layout, it recovers the dependency structure better
than an **isomorphic permutation** of that graph using the same resources. In other words,
"Clinical" is not a mere modifier — it demonstrably contributes bits.

## Skeleton of the Design

- **1 feature = 1 qubit.** Each feature gets its own local latent variable `z_u`, encoded as
  an angle.
- **RZZ only on CDG edges.** The entangling angle `gamma` is the only parameter that can
  create cross-feature dependency.
- **A per-feature one-dimensional local head** `h_u(q_u, c)`. It cannot see `q_v` from any
  other qubit. The classical parameters therefore **structurally cannot create** conditional
  cross-feature dependency (`model.assert_no_cross_feature_mixing` verifies that the
  Jacobian is diagonal).
- **Proposition 1 (light cone).** With a product initial state and local angle encoding,
  `<Z_u>` depends only on qubits within graph distance `L`. Corollary: if `d_G(u,v) > 2L`
  then `Cov(x_u, x_v | c) = 0` (exactly). We also exploit this identity computationally:
  instead of a 2^16 statevector, we simulate only a 2^|N_L(u)| subcircuit per qubit.

`L=1` is the only viable operating point. At `L=3` all 120 pairs fall inside the light cone,
so the CDG and a permuted graph become identical by definition. See `REVISIONS.md` for the
full argument.

## What Has Been Confirmed So Far

| | Result | Document |
|---|---|---|
| The light-cone cliff appears exactly at `d = 2L` | outside it, max \|rho\|=0.012 | `RESULTS_ceiling.md` |
| Adjacent-pair expressivity ceiling (L=1) | \|rho\| = 0.991 | `RESULTS_ceiling.md` |
| Alignment precheck `M(G,L)` on the real CDG | L=1: z=+3.18 (p=0.0004) / L=2: z=+0.66 (rejected) | `RESULTS_precheck.md` |
| Light-cone simulator optimization | ~70× speedup (>1800s → 25.7s / seed) | `WP6_REPORT.md` |

## Unresolved — Current Top Priority

In the controlled synthetic graph benchmark, **the confirmatory contrast does not work.**
All four variants — `aligned`, `permuted`, `rewired`, `no_entangle` — overlap at a 120-pair
dependency error of 0.132–0.136, and `no_entangle`, which has **no entanglement at all**, is
the best of them.

Diagnostic findings (`scripts/diag_benchmark.py`, `scripts/diag_trained.py`):

- The error of a model that creates no dependency whatsoever = **0.065**. The trained model
  (0.136) is **2× worse than that**.
- Error on the 19 true edges: floor 0.368 → aligned 0.366. **The contribution of the
  entangling gates is effectively zero.**
- It creates a spurious dependency of 0.093 on the 101 non-edges. A model with zero
  entangling gates creates 0.086 as well → the culprit is **the condition vector `c` that
  all features share**, and the evaluation metric does not control for it (the CDG is
  defined conditionally on `c`, whereas the metric measures the unconditional partial
  correlation = an estimator mismatch).

In short, **the WGAN-GP critic is failing to supply a useful gradient to `gamma`.**
Whether this is an expressivity problem or a training problem is settled by
`scripts/ceiling_joint.py` (which removes the GAN and performs gradient descent directly on
the 120-pair pattern).

## Data

MIMIC-IV / eICU are subject to PhysioNet **credentialed access + a DUA**. Neither the raw
data nor any **patient-level** data derived from it may ever enter this repository. All of it
stays in the local archive only. Paths and acquisition instructions are in `DATA.md`.

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

- `CDG-QGAN_research_plan_v2_ko.md` — the research plan (the reference document)
- `REVISIONS.md` — **where this conflicts with v2, this takes precedence.** Review responses
- `HANDOFF.md` — work package specifications (WP-2 through WP-6)
- `DATA.md` — data locations and environment pitfalls

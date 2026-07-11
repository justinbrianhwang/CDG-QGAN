# WP-6 optimization report

## Changes

- Batched all output-qubit light cones into one padded `(batch, cone, amplitude)` tensor.
- Vectorized `RY`, `RZ`, and `RX` gates over the cone axis, using precomputed global-qubit gather maps and identity angles for padding qubits.
- Precomputed each cone's edge-parameter indices, eliminating all per-gate `list.index` scans.
- Fused the commuting `RZZ` gates in each layer into one vectorized diagonal phase multiply using precomputed `ZZ` sign tables.
- Precomputed observable sign tables and registered all maps/tables as non-persistent module buffers so device transfers remain correct.
- Captured the fixed-shape CUDA critic and generator training steps as CUDA Graphs. This removes the remaining launch overhead without requiring Triton (which is unavailable in the pinned environment). Warm-up parameter and optimizer mutations are reset before training begins.
- Added `tests/test_qsim_lightcone.py`, which retains the pre-WP6 algorithm as an independent oracle and checks both `L=1` and `L=2` outputs at `<1e-5`.

## Wall-clock performance

Hardware/software: NVIDIA GeForce RTX 5090, PyTorch 2.11.0+cu128, `cdg-qgan` environment.

Acceptance workload: `train()` with `n=16`, `depth=1`, `batch=256`, `critic_steps=5`, `steps=3000`, 20,000 synthetic rows, four condition columns, and a 3-regular graph.

| Version | Wall-clock per seed | Speedup |
|---|---:|---:|
| Before WP-6 | >1,800 s (lower bound from the recorded >1.5 h / 3-seed single-variant run) | — |
| After WP-6 | 30.43 s | >59.2x |

The after time includes model creation, graph warm-up/capture, all 3,000 training steps, and final CUDA synchronization. It passes the `<60 s` criterion.

## Exactness verification

`python scripts/qsim_lightcone.py`:

```text
n=8  L=1: max error 8.94e-07  PASS
n=10 L=1: max error 8.34e-07  PASS
n=12 L=1: max error 1.01e-06  PASS
n=16 L=1: max error 1.61e-06  PASS
n=12 L=2: max error 7.75e-07  PASS
```

The legacy regression test also passes for `n=10, L=1` and `n=12, L=2`, with the required tolerance of `1e-5`.

## Remaining bottleneck

For `L=1`, CUDA Graph replay and data-buffer copies now dominate rather than individual quantum gate launches. Depth 2 remains intrinsically much heavier because padding to the largest cone evolves `(batch, 12, 1024)` complex tensors; depth 2 is verified for exactness but is not the confirmatory training configuration.

---

## Independent PM verification (2026-07-11) — **APPROVED**

`scripts/verify_wp6.py`

We independently reproduced the accuracy and speed that Codex reported, and additionally checked
**items Codex did not verify**. Codex introduced CUDA Graphs into the training loop but **did not verify
that the graph path produces the same results as the ordinary path.** CUDA Graphs are a textbook example
of an optimization that silently produces wrong results, so this must be confirmed.

| Verification | Result |
|---|---|
| Simulator accuracy (independently reproduced) | PASS — agrees with the full statevector to `1.6e-6` |
| Regression test (independently reproduced) | PASS |
| **[new] RNG advancement inside the CUDA Graph** | **PASS** — different random numbers on every replay |
| **[new] Graph path vs. ordinary path equivalence** | **PASS** — dependency error 0.2058 vs. 0.2061 |
| Speed (3000 steps/seed) | **PASS — 25.7 s** (target 60 s) |

### Why the RNG check was necessary

The interpolation coefficient in `gradient_penalty`, `alpha = torch.rand(...)`, is generated **inside** the
CUDA Graph. If the RNG does not advance between graph replays, **every critic step would use the same
interpolation point and the gradient penalty would be neutralized.** Training would look like it was "working,"
but it would silently stop being WGAN-GP. PyTorch handles graph-safe RNG correctly, so there was no problem —
but this was not an item we could let pass unchecked.

### Actual speedup

| Configuration | 3000 steps / seed |
|---|---|
| Original (pre-WP-6 simulator) | **>1,800 s** |
| New simulator (without CUDA Graphs) | ~490 s |
| **New simulator + CUDA Graphs** | **25.7 s** |

**Roughly 70x.** The 90 training runs of the confirmatory experiment go from 45 hours to **about 40 minutes**.

### What was added

- The `Cfg.use_cuda_graph` flag — it must be possible to turn the graph path off and verify against it.

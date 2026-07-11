# WP-6: Optimize the light-cone quantum simulator

(English, on purpose: PowerShell `Get-Content` mangles UTF-8 Korean on this machine.)

## Goal

`LightconeGenerator` in `scripts/qsim_lightcone.py` is **numerically correct but far too slow.**
The confirmatory experiment needs 9 model variants x 10 seeds = 90 training runs.
At the current speed that is >40 hours. We need at least a 20-50x speedup.

## Diagnosed bottleneck

**It is kernel-launch overhead, not arithmetic.**

- With `L=1` and max degree `Δ<=3`, each light cone has at most 4 qubits, so its
  statevector is only **16 complex amplitudes**. The tensors are tiny.
- But `forward()` loops in **Python** over 16 qubits x ~10 gates, issuing hundreds of
  micro CUDA kernels per generator forward pass.
- Measured: GPU shows 83% "utilization" that is actually launch latency, not compute.
- Measured: `benchmark_synthetic.py` took **>1.5 hours** for ONE graph variant (3 seeds).

## What to do

1. **Batch the 16 cones into a single tensor axis.**
   Every cone has `<= 2^4 = 16` amplitudes, so all 16 cones fit in one `(B, 16, 16)`
   tensor and can be evolved simultaneously. Pad smaller cones to the max cone size and
   apply identity on padding qubits.
2. **Vectorize the single-qubit gates (`RY`/`RZ`/`RX`) over the cone axis.**
   Gather the per-(layer, qubit) parameters instead of indexing them one at a time.
3. Keep `RZZ` as a diagonal phase multiply, but vectorize it over the cone axis too.
4. Apply `torch.compile` (or CUDA graphs) to remove the remaining launch overhead.
5. In the current `forward()`, `self.edges.index(...)` is called **per gate**, which is an
   O(E) Python list scan every time. Precompute those indices in `__init__`.

## Acceptance criteria (ALL must pass)

1. **`python scripts/qsim_lightcone.py` must pass.**
   `verify_against_full()` must agree with the full-statevector implementation to `<1e-4`
   for `n in {8,10,12,16}, L=1` and for `n=12, L=2`.
   **If the numbers change, the task has failed.**
2. **Regression test**: same inputs -> same outputs as the current implementation, `<1e-5`.
   Add this test to the repo.
3. **Speed**: `L=1`, 3000 training steps must finish in **under 60 seconds per seed**.
   Measure by calling `train()` from `scripts/train.py` with `n=16`, `depth=1`,
   `batch=256`, `critic_steps=5`, `steps=3000`, and timing wall-clock.
4. Report the **before/after wall-clock and the speedup factor**.

## Hard rules

- **DO NOT upgrade numpy above 2.2.6.** torch 2.11+cu128 is built against the numpy 2.2 ABI;
  upgrading breaks `shm.dll` loading (`OSError: [WinError 127]`). If you must install
  anything: `python -m pip install -c .backup/constraints.txt <pkg>`
- **Numerical exactness beats speed.** Do not approximate. The light-cone subcircuit must
  produce *exactly* the same `<Z_u>` as the full statevector (up to float error).
  The whole point is that this is an exact identity, not an approximation.
- Environment: conda env `cdg-qgan`.
  Python: `C:\Users\sunju\miniconda3\envs\cdg-qgan\python.exe`
  (`conda run` cannot handle multi-line `python -c`; call that path directly.)
- The repo's `.md` files are UTF-8 Korean. If you read them via PowerShell, use
  `Get-Content -Encoding utf8`, or read them with your file tool instead.

## Why the subcircuit trick is exact (do not break this)

Proposition 1 (proved in the plan, verified numerically): with a product initial state and
purely local angle encoding, the Heisenberg backward support of `Z_u` after depth `L` is
contained in the graph neighborhood `N_L(u)`. Therefore `<Z_u>` depends only on the qubits
within graph distance `L` of `u`, and can be computed **exactly** on that subcircuit.
See the header comment of `qsim_lightcone.py` and `RESULTS_ceiling.md`.

## Deliverable

Write `WP6_REPORT.md` (English is fine) containing:
- what you changed
- before/after wall-clock and speedup factor
- `verify_against_full()` output
- any remaining bottleneck

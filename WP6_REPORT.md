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

## PM 독립 검증 (2026-07-11) — **승인**

`scripts/verify_wp6.py`

Codex가 보고한 정확도·속도를 독립적으로 재현했고, **Codex가 검증하지 않은 항목**을
추가로 확인했다. Codex는 CUDA Graph를 학습 루프에 도입했지만 **그래프 경로가 일반
경로와 같은 결과를 내는지는 검증하지 않았다.** CUDA Graph는 조용히 틀린 결과를 내는
대표적 최적화이므로 반드시 확인해야 한다.

| 검증 | 결과 |
|---|---|
| 시뮬레이터 정확도 (독립 재현) | PASS — 전체 상태벡터와 `1.6e-6` 일치 |
| 회귀 테스트 (독립 재현) | PASS |
| **[신규] CUDA Graph 내 RNG 전진** | **PASS** — replay마다 다른 난수 |
| **[신규] 그래프 경로 vs 일반 경로 동등성** | **PASS** — 의존성오차 0.2058 vs 0.2061 |
| 속도 (3000 step/seed) | **PASS — 25.7초** (목표 60초) |

### 왜 RNG 검증이 필요했는가

`gradient_penalty`의 보간 계수 `alpha = torch.rand(...)` 가 CUDA Graph **안**에서
생성된다. 그래프 replay 사이에 RNG가 전진하지 않으면 **모든 critic 스텝이 같은
보간점을 쓰게 되어 gradient penalty가 무력화된다.** 학습은 "돌아가는 것처럼" 보이지만
WGAN-GP가 아니게 되는 조용한 실패다. PyTorch가 graph-safe RNG를 올바로 처리하고
있어 문제없었으나, 확인 없이 넘어갈 수는 없는 항목이었다.

### 실제 가속

| 구성 | 3000 step / seed |
|---|---|
| 원래 (WP-6 이전 시뮬레이터) | **>1,800초** |
| 새 시뮬레이터 (CUDA Graph 없이) | ~490초 |
| **새 시뮬레이터 + CUDA Graph** | **25.7초** |

**약 70배.** 확증 실험 90회 학습이 45시간 → **약 40분**이 된다.

### 추가된 것

- `Cfg.use_cuda_graph` 플래그 — 그래프 경로를 끄고 검증할 수 있어야 한다.

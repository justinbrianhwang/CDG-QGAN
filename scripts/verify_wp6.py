"""WP-6 독립 검증 (PM).

Codex가 보고한 것:
  - 시뮬레이터 정확도 1.6e-6, 회귀 테스트 통과
  - 30.43 s/seed, 59배 가속

Codex가 검증하지 **않은** 것 — 여기서 잡는다:
  CUDA Graph를 학습 루프에 도입했는데, 그래프 경로가 일반 경로와 같은 결과를 내는지
  확인하지 않았다. CUDA Graph는 조용히 틀린 결과를 내는 대표적인 최적화다.

검증 항목
  [1] RNG가 그래프 replay 사이에 전진하는가
      gradient_penalty의 alpha가 그래프 **안**에서 생성된다. replay마다 같은 값이 나오면
      매 critic 스텝이 같은 보간점을 쓰게 되고 -> gradient penalty가 무력화된다.
      학습은 "돌아가는 것처럼" 보이지만 WGAN-GP가 아니게 된다. 조용한 실패.
  [2] 그래프 경로 vs 일반 경로가 동일 seed에서 같은 학습 결과를 내는가
  [3] 속도가 실제로 목표(60s/seed)를 치는가
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from train import DEVICE, Cfg, dependency_error, generate, train  # noqa: E402

N, P = 20000, 16


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    import networkx as nx
    G = nx.random_regular_graph(3, P, seed=seed)
    edges = [tuple(sorted(e)) for e in G.edges()]

    Om = np.eye(P)
    for u, v in G.edges():
        Om[u, v] = Om[v, u] = rng.uniform(0.3, 0.5) * rng.choice([-1, 1])
    Om += np.eye(P) * (abs(np.linalg.eigvalsh(Om).min()) + 0.15)
    S = np.linalg.inv(Om)
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    X = rng.multivariate_normal(np.zeros(P), S, size=N)
    X = (X - X.mean(0)) / X.std(0)
    y = (rng.random(N) < 0.1).astype(float)
    C = np.column_stack([y, rng.normal(0, 1, N), rng.random(N) < 0.5,
                         rng.integers(0, 3, N)]).astype(float)
    return X, C, edges


# ---------------------------------------------------------------------------
def test_rng_in_graph() -> bool:
    """[1] CUDA Graph replay 사이에 torch.rand가 다른 값을 내는가."""
    print("=" * 74)
    print("[1] CUDA Graph 안에서 RNG가 전진하는가")
    print("=" * 74)
    if DEVICE != "cuda":
        print("  CUDA 없음 — 건너뜀")
        return True

    out = torch.empty(8, device="cuda")

    def body():
        out.copy_(torch.rand(8, device="cuda"))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()

    draws = []
    for _ in range(4):
        g.replay()
        torch.cuda.synchronize()
        draws.append(out.clone().cpu().numpy())

    same = [np.allclose(draws[0], d) for d in draws[1:]]
    for i, d in enumerate(draws):
        print(f"  replay {i}: {d[:4].round(4)}")
    ok = not any(same)
    print(f"\n  replay마다 다른 난수: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  !! FAIL — gradient_penalty의 alpha가 매번 같아진다.")
        print("     모든 critic 스텝이 같은 보간점을 쓰게 되어 WGAN-GP가 무력화된다.")
    return ok


# ---------------------------------------------------------------------------
def test_graph_vs_eager(steps=300) -> bool:
    """[2] 그래프 경로와 일반 경로가 같은 학습 결과를 내는가."""
    print()
    print("=" * 74)
    print(f"[2] CUDA Graph 경로 vs 일반 경로  (동일 seed, {steps} step)")
    print("=" * 74)
    X, C, edges = make_data()

    res = {}
    for use_graph in (True, False):
        t0 = time.time()
        G, _ = train(X, C, edges, Cfg(depth=1, steps=steps, seed=0, use_cuda_graph=use_graph))
        Xs = generate(G, C, 20000, seed=0)
        res[use_graph] = {
            "err": dependency_error(X, Xs),
            "mean": Xs.mean(0),
            "std": Xs.std(0),
            "t": time.time() - t0,
        }
        tag = "CUDA Graph" if use_graph else "일반(eager)"
        print(f"  {tag:<14} 의존성오차={res[use_graph]['err']:.4f}  ({res[use_graph]['t']:.0f}s)")

    a, b = res[True], res[False]
    d_err = abs(a["err"] - b["err"])
    d_mean = np.abs(a["mean"] - b["mean"]).max()
    d_std = np.abs(a["std"] - b["std"]).max()
    print()
    print(f"  |의존성오차 차이|   = {d_err:.4f}")
    print(f"  |특징 평균 차이|max = {d_mean:.4f}")
    print(f"  |특징 표준편차 차이|max = {d_std:.4f}")
    print()
    print("  주: 두 경로의 RNG 소비 순서가 달라 bit-exact 일치는 기대하지 않는다.")
    print("      학습이 통계적으로 동등한 곳에 수렴하는지를 본다.")
    ok = d_err < 0.05 and d_mean < 0.15 and d_std < 0.15
    print(f"  판정: {'PASS — 두 경로가 통계적으로 동등' if ok else 'FAIL — 경로에 따라 결과가 다르다'}")
    return ok


# ---------------------------------------------------------------------------
def test_speed(steps=3000) -> bool:
    """[3] 합격 기준: seed당 60초 이내."""
    print()
    print("=" * 74)
    print(f"[3] 속도  ({steps} step, n=16, L=1, batch=256, critic=5)")
    print("=" * 74)
    X, C, edges = make_data()
    t0 = time.time()
    train(X, C, edges, Cfg(depth=1, steps=steps, seed=0))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    print(f"  wall-clock: {el:.1f}s  (목표 <60s)")
    ok = el < 60
    print(f"  판정: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = test_rng_in_graph()
    r2 = test_graph_vs_eager()
    r3 = test_speed()
    print()
    print("=" * 74)
    print(f"  [1] 그래프 내 RNG 전진   : {'PASS' if r1 else 'FAIL'}")
    print(f"  [2] 그래프 vs 일반 동등  : {'PASS' if r2 else 'FAIL'}")
    print(f"  [3] 속도 <60s            : {'PASS' if r3 else 'FAIL'}")
    print(f"\n  WP-6 최종: {'승인' if (r1 and r2 and r3) else '반려'}")
    print("=" * 74)

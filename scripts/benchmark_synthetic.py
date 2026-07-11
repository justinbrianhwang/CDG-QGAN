"""통제된 합성 그래프 벤치마크 (CDG-QGAN v2 §10).

실제 임상 데이터에서는 참 그래프를 모른다. 여기서는 참 그래프를 알고 시작한다.
-> 이론과 구현이 맞는지, 확증 설계가 실제로 작동하는지 MIMIC 없이 검증할 수 있다.

Teacher:
  1. 알려진 sparse precision matrix Omega* (모듈러 그래프 — 임상 클러스터를 모사)
  2. Gaussian latent ~ N(0, Omega*^-1)
  3. 차원별로 다른 비선형 단조 변환 -> 비정규 주변분포 (nonparanormal)

비교 (전부 동일 자원: 간선 수, 깊이, 파라미터, critic, loss, step 수):
  aligned      : 참 그래프를 회로 토폴로지로 사용
  permuted     : isomorphic 순열 — 임상 정렬만 파괴
  distmatched  : [리뷰 B] 거리 분포까지 맞춘 순열
  rewired      : 차수 보존 재배선
  no_entangle  : RZZ 제거

평가 [리뷰 B-2]:
  primary = 120쌍 전체 조건부 의존성 오차 (위음성 + 위양성)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from graphs import degree_preserving_rewire, distance_matched_permuted, isomorphic_permuted  # noqa: E402
from train import Cfg, dependency_error, generate, train  # noqa: E402

N_FEAT = 16
N_TRAIN = 20000
N_SYN = 20000
SEEDS = [0, 1, 2]
DEPTH = 1  # [findings] L=1이 유일한 작동점. L=3은 모든 쌍이 도달 가능해져 무의미.


def teacher_graph(rng) -> nx.Graph:
    """모듈러 그래프: 4개 클러스터(4노드씩) + 클러스터 간 다리.

    실제 임상 구조를 모사한다 — 순환/전해질/신장/혈액 같은 국소 클러스터.
    """
    G = nx.Graph()
    G.add_nodes_from(range(N_FEAT))
    for b in range(4):  # 각 블록 안에서 삼각형을 만든다 (거리 2 경로 확보)
        o = 4 * b
        G.add_edges_from([(o, o + 1), (o + 1, o + 2), (o, o + 2), (o + 2, o + 3)])
    G.add_edges_from([(3, 4), (7, 8), (11, 12)])  # 블록 간 다리
    return G


def teacher_data(G: nx.Graph, n: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """참 그래프에서 nonparanormal 데이터 생성. 반환: (X, C, Omega)."""
    p = N_FEAT
    Om = np.eye(p) * 1.0
    for u, v in G.edges():
        w = rng.uniform(0.35, 0.55) * rng.choice([-1, 1])
        Om[u, v] = Om[v, u] = w
    Om += np.eye(p) * (abs(np.linalg.eigvalsh(Om).min()) + 0.15)  # 양정치 보장

    Sigma = np.linalg.inv(Om)
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)

    g = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    # 차원별 비선형 단조 변환 -> 비정규 주변분포
    X = np.column_stack([
        np.sign(g[:, j]) * np.abs(g[:, j]) ** (1.0 + 0.5 * (j % 3)) if j % 3 else np.exp(0.5 * g[:, j])
        for j in range(p)
    ])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    y = (rng.random(n) < 0.1).astype(float)  # 조건 (사망 라벨 모사)
    C = np.column_stack([y, rng.normal(0, 1, n), rng.random(n) < 0.5, rng.integers(0, 3, n)]).astype(float)
    return X, C, Om


def main() -> None:
    rng = np.random.default_rng(20260711)
    G_true = teacher_graph(rng)
    X, C, _ = teacher_data(G_true, N_TRAIN, rng)

    print("=" * 78)
    print("통제된 합성 그래프 벤치마크 (v2 §10)")
    print("=" * 78)
    print(f"  teacher: {N_FEAT}노드 {G_true.number_of_edges()}간선  "
          f"지름={nx.diameter(G_true)}  군집계수={nx.average_clustering(G_true):.3f}")
    print(f"  깊이 L={DEPTH} (도달 반경 {2*DEPTH})   n_train={N_TRAIN:,}  seeds={SEEDS}")

    n_reach = sum(1 for u in range(N_FEAT) for v in range(u + 1, N_FEAT)
                  if nx.shortest_path_length(G_true, u, v) <= 2 * DEPTH)
    print(f"  aligned에서 도달 가능한 쌍: {n_reach}/120")
    print()

    gr = np.random.default_rng(7)
    E_true = list(G_true.edges())
    holdout = [(u, v) for u in range(N_FEAT) for v in range(u + 1, N_FEAT)
               if not G_true.has_edge(u, v)
               and nx.shortest_path_length(G_true, u, v) == 2][:10]

    variants = {
        "aligned": G_true,
        "permuted": isomorphic_permuted(G_true, gr),
        "distmatched": distance_matched_permuted(G_true, holdout, gr)[0],
        "rewired": degree_preserving_rewire(G_true, gr),
        "no_entangle": nx.empty_graph(N_FEAT),
    }

    print(f"  {'모델':<14} {'|E|':>4} {'전체120쌍 오차':>16}  (평균 ± 표준편차, seed 3개)")
    print("  " + "-" * 60)

    results = {}
    for name, Gv in variants.items():
        errs = []
        t0 = time.time()
        for s in SEEDS:
            cfg = Cfg(depth=DEPTH, seed=s)
            Gm, _ = train(X, C, list(Gv.edges()), cfg)
            Xs = generate(Gm, C, N_SYN, seed=s)
            errs.append(dependency_error(X, Xs))  # 120쌍 전체 [리뷰 B-2]
        results[name] = np.array(errs)
        print(f"  {name:<14} {Gv.number_of_edges():>4} "
              f"{errs and np.mean(errs):>10.4f} ± {np.std(errs):.4f}   ({time.time()-t0:.0f}s)",
              flush=True)

    print()
    print("=" * 78)
    print("확증 대조")
    print("=" * 78)
    a = results["aligned"]
    for ref in ("permuted", "distmatched", "rewired", "no_entangle"):
        b = results[ref]
        delta = a.mean() - b.mean()
        print(f"  aligned - {ref:<12} = {delta:+.4f}   "
              f"{'aligned 우세' if delta < 0 else 'aligned 열세'}")
    print()
    print("  기대: aligned < permuted (임상 정렬 효과)")
    print("        aligned <= distmatched 이면, 효과가 단순 거리 배치를 넘어선다는 뜻 [리뷰 B]")
    print("        no_entangle이 가장 나쁘면, 얽힘이 실제로 의존성을 만든다는 뜻")

    np.savez("results_benchmark.npz", **results)


if __name__ == "__main__":
    main()

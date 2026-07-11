"""[결정적 사전 검사] CDG 정렬이 애초에 효과를 낼 수 있는 구조인가?

훈련을 한 번도 하지 않고, 그래프와 참 부분상관만으로 판정한다.
GPU 시간을 쓰기 전에 반드시 통과해야 하는 관문이다.

--------------------------------------------------------------------------
핵심 통계량: 표현 가능한 의존성 질량 (expressible dependency mass)

    M(G, L) = sum_{(u,v)} |rho_true(u,v)| * 1[ d_G(u,v) <= 2L ]

따름정리 1에 의해 d_G(u,v) > 2L 인 쌍은 조건부 공분산이 정확히 0이다.
따라서 깊이 L의 회로가 표현할 수 있는 의존성의 총량은 위와 같이 정해진다.
CDG의 주장이 옳다면, M(CDG, L)이 무작위 permutation의 분포보다 위에 있어야 한다.

permutation은 isomorphic이므로 노드 수, 간선 수, 차수열, 삼각형 수,
군집계수가 전부 보존된다. 달라지는 것은 "어떤 임상 쌍이 그 삼각형 안에 앉는가"뿐이다.
따라서 이 검정은 정확히 "임상 의미 정렬"만을 잰다.

--------------------------------------------------------------------------
판정

  CDG의 M이 permutation 귀무분포의 상위 꼬리에 있으면  -> 실험 진행
  귀무분포 한가운데 있으면                              -> 효과 없음. 설계를 바꿔야 함.

이 검정에서 지면, 아무리 GPU를 태워도 Delta_HDE < 0 은 나오지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import paths  # noqa: E402

N_PERM = 5000


def dist_matrix(G: nx.Graph, p: int) -> np.ndarray:
    D = np.full((p, p), 99)
    for u, lengths in nx.all_pairs_shortest_path_length(G):
        for v, d in lengths.items():
            D[u, v] = d
    return D


def mass(D: np.ndarray, R: np.ndarray, L: int) -> float:
    """깊이 L에서 표현 가능한 의존성 질량."""
    reach = (D <= 2 * L) & (D > 0)
    return float((np.abs(R) * reach).sum() / 2.0)  # 대칭이므로 /2


def main() -> None:
    d = np.load(paths.PROCESSED / "cdg.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    rho = d["rho"]
    E_fit = [tuple(e) for e in d["E_fit"]]
    p = len(names)

    G = nx.Graph()
    G.add_nodes_from(range(p))
    G.add_edges_from(E_fit)

    R = np.abs(rho.copy())
    np.fill_diagonal(R, 0.0)
    total = R.sum() / 2.0

    print("=" * 78)
    print("사전 검사: CDG 정렬이 표현 가능한 의존성 질량을 늘리는가")
    print("=" * 78)
    print(f"  노드 {p}  간선 {G.number_of_edges()}  최대차수 {max(dict(G.degree()).values())}  "
          f"지름 {nx.diameter(G)}  군집계수 {nx.average_clustering(G):.3f}")
    print(f"  전체 의존성 질량 sum|rho| = {total:.2f}")
    print()

    rng = np.random.default_rng(20260711)
    D_cdg = dist_matrix(G, p)

    for L in (1, 2, 3):
        m_cdg = mass(D_cdg, R, L)

        null = np.empty(N_PERM)
        for k in range(N_PERM):
            perm = rng.permutation(p)
            H = nx.Graph()
            H.add_nodes_from(range(p))
            H.add_edges_from((int(perm[u]), int(perm[v])) for u, v in G.edges())
            null[k] = mass(dist_matrix(H, p), R, L)

        pct = float((null < m_cdg).mean() * 100)
        pval = float((null >= m_cdg).mean())
        z = (m_cdg - null.mean()) / (null.std() + 1e-12)

        n_reach = int(((D_cdg <= 2 * L) & (D_cdg > 0)).sum() / 2)
        n_pairs = p * (p - 1) // 2

        print(f"  L={L}  (도달 반경 {2*L})")
        print(f"    도달 가능 쌍       : {n_reach}/{n_pairs}")
        print(f"    CDG 질량           : {m_cdg:.3f}  ({m_cdg/total*100:.1f}% of total)")
        print(f"    permutation 귀무    : {null.mean():.3f} +- {null.std():.3f}   "
              f"[{null.min():.3f}, {null.max():.3f}]")
        print(f"    CDG의 백분위        : {pct:.1f}%   z={z:+.2f}   p={pval:.4f}")

        if n_reach == n_pairs:
            verdict = "무의미 — 모든 쌍이 도달 가능. 토폴로지가 아무 제약도 안 건다."
        elif pval < 0.05:
            verdict = "PASS — CDG가 귀무분포 상위 꼬리. 정렬 효과가 존재할 수 있다."
        else:
            verdict = "FAIL — CDG가 무작위 순열과 구별되지 않는다."
        print(f"    판정: {verdict}")
        print()

    print("=" * 78)
    print("  해석")
    print("=" * 78)
    print("  이 검정은 회로를 한 번도 돌리지 않는다. 순수하게 그래프 구조와")
    print("  참 부분상관만으로, 확증 실험이 성공할 가능성이 있는지를 판정한다.")
    print()
    print("  FAIL이면 GPU를 아무리 태워도 Delta_HDE < 0 은 나오지 않는다.")
    print("  그 경우 다음을 검토해야 한다:")
    print("    - 회로 깊이 L을 낮춘다 (L=1). 도달 반경이 좁을수록 토폴로지가 변별력을 갖는다.")
    print("    - 간선 수를 줄여 지름을 키운다 (지금은 near-3-regular라 지름이 작다).")
    print("    - held-out 방식을 버리고 120쌍 전체 (위양성 포함) 평가로 간다.")
    print("    - CDG가 실제로 임상 클러스터(삼각형)를 담고 있는지 확인한다.")


if __name__ == "__main__":
    main()

"""대조 그래프 생성 (CDG-QGAN v2 §7.8, §9) + 리뷰 반영 추가 대조군.

확증 대조군:
  - isomorphic permuted-CDG : adjacency만 순열, 특징-큐비트 대응은 고정 (v2 §5.3)
  - distance-matched permuted-CDG : [리뷰 B] 아래 설명

메커니즘 대조군:
  - degree-preserving rewired
  - ring / ring-with-chords
  - no-entanglement

[리뷰 B] 왜 distance-matched permutation이 필요한가
--------------------------------------------------
따름정리 1에 의해 d_G(u,v) > 2L 이면 조건부 공분산이 정확히 0이다.
그런데 held-out 임상 쌍은 CDG에서 거리 2 근처(공통 이웃을 공유)에 놓이고,
무작위 permutation에서는 거리 3~5로 흩어진다. 따라서 L=1에서
Delta_HDE < 0 은 "실험 결과"가 아니라 그래프 조합론으로 이미 결정된 사실이다.
= 실패할 수 없는 검정.

distance-matched permutation은 held-out 쌍의 거리 분포를 CDG와 같게 맞춘다.
그러면 "가까이 있다"는 이점이 상쇄되고, 남는 차이는 오직
"CDG가 고른 바로 그 간선들이 실제 임상 구조를 담고 있는가"뿐이다.
이것이 결과를 미리 알 수 없는, 정보량 있는 검정이다.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def _dist_profile(G: nx.Graph, pairs: list[tuple[int, int]]) -> tuple[int, ...]:
    """주어진 쌍들의 그래프 거리 분포 (정렬된 튜플)."""
    out = []
    for u, v in pairs:
        out.append(nx.shortest_path_length(G, u, v) if nx.has_path(G, u, v) else 99)
    return tuple(sorted(out))


def isomorphic_permuted(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """adjacency만 순열 (v2 §5.3).

    노드 수, 간선 수, 차수열, 지름, isomorphism class, 게이트 수, 파라미터 수를
    전부 보존한다. 달라지는 것은 "어떤 임상 특징 쌍이 가까운가"뿐.
    큐비트 u는 계속 특징 x_u를 생성한다 (특징-큐비트 대응은 고정).
    """
    n = G.number_of_nodes()
    perm = rng.permutation(n)
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from((int(perm[u]), int(perm[v])) for u, v in G.edges())
    return H


def distance_matched_permuted(
    G: nx.Graph,
    holdout: list[tuple[int, int]],
    rng: np.random.Generator,
    n_tries: int = 20000,
) -> tuple[nx.Graph, bool]:
    """[리뷰 B] held-out 쌍의 거리 분포가 CDG와 같은 permutation을 찾는다.

    반환: (그래프, 정확히 매칭됐는지). 정확 매칭이 없으면 가장 가까운 것.
    """
    target = _dist_profile(G, holdout)
    best, best_cost = None, np.inf

    for _ in range(n_tries):
        H = isomorphic_permuted(G, rng)
        prof = _dist_profile(H, holdout)
        if prof == target:
            return H, True
        cost = sum(abs(a - b) for a, b in zip(prof, target))
        if cost < best_cost:
            best, best_cost = H, cost

    return best, False


def degree_preserving_rewire(G: nx.Graph, rng: np.random.Generator, n_swaps: int = 200) -> nx.Graph:
    """double-edge swap. 차수열과 간선 수는 유지, 경로 구조는 바꿈 (v2 §7.8)."""
    H = G.copy()
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 50, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass  # 스왑 가능한 조합이 부족하면 가능한 만큼만
    return H


def ring_with_chords(n: int, n_edges: int, rng: np.random.Generator) -> nx.Graph:
    """자원(간선 수)을 맞춘 균일 토폴로지 기준선 (v2 §9.2)."""
    H = nx.cycle_graph(n)
    extra = n_edges - H.number_of_edges()
    if extra > 0:
        cand = [(i, j) for i in range(n) for j in range(i + 2, n) if not H.has_edge(i, j)]
        rng.shuffle(cand)
        H.add_edges_from(cand[:extra])
    return H


def no_entanglement(n: int) -> nx.Graph:
    """E = 공집합. local marginal만 생성하는 양자 모델 (v2 §7.8)."""
    H = nx.Graph()
    H.add_nodes_from(range(n))
    return H


def summarize(name: str, G: nx.Graph, holdout: list[tuple[int, int]], L_list=(1, 2)) -> str:
    """대조군이 자원을 정말 보존하는지 + held-out 쌍이 도달 가능한지 확인."""
    deg = dict(G.degree())
    prof = _dist_profile(G, holdout) if G.number_of_edges() else tuple([99] * len(holdout))
    arr = np.array(prof)
    reach = "  ".join(f"L={L}:{int((arr <= 2*L).sum())}/{len(arr)}" for L in L_list)
    conn = nx.is_connected(G) if G.number_of_edges() else False
    return (f"  {name:<26} |E|={G.number_of_edges():>2}  maxdeg={max(deg.values()) if deg else 0}  "
            f"connected={str(conn):<5}  held-out 도달 {reach}")


def build_all(G_cdg: nx.Graph, holdout: list[tuple[int, int]], seed: int = 20260711,
              n_perms: int = 3) -> dict[str, nx.Graph]:
    """확증/메커니즘 대조군 일괄 생성."""
    rng = np.random.default_rng(seed)
    n, m = G_cdg.number_of_nodes(), G_cdg.number_of_edges()

    out: dict[str, nx.Graph] = {"cdg": G_cdg}
    for k in range(n_perms):
        out[f"permuted_{k}"] = isomorphic_permuted(G_cdg, rng)
    for k in range(n_perms):
        H, exact = distance_matched_permuted(G_cdg, holdout, rng)
        out[f"distmatched_{k}"] = H
        if not exact:
            print(f"  [경고] distance-matched permutation {k}: 정확 매칭 실패, 최근접 사용")
    out["rewired"] = degree_preserving_rewire(G_cdg, rng)
    out["ring"] = ring_with_chords(n, m, rng)
    out["no_entangle"] = no_entanglement(n)
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    d = np.load(Path("data/processed/cdg.npz"), allow_pickle=True)
    names = list(d["names"])
    E_fit = [tuple(e) for e in d["E_fit"]]
    holdout = [tuple(e) for e in d["E_holdout"]]

    G = nx.Graph()
    G.add_nodes_from(range(len(names)))
    G.add_edges_from(E_fit)

    print("=" * 78)
    print("대조 그래프  (자원은 동일, 임상 정렬만 파괴)")
    print("=" * 78)
    graphs = build_all(G, holdout)
    print()
    for k, H in graphs.items():
        print(summarize(k, H, holdout))
    print()
    print("  핵심: permuted는 held-out 도달 수가 CDG보다 작아야 하고,")
    print("        distmatched는 CDG와 같아야 한다. 같은데도 CDG가 이기면")
    print("        그건 거리 배치가 아니라 임상 구조 때문이다. [리뷰 B]")

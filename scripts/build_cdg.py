"""Clinical Dependency Graph 추정 (CDG-QGAN v2 §7) + 리뷰 반영 수정.

절차:
  1. 조건 벡터 c = (y, age, sex, icu_type)에 대해 residualize
  2. nonparanormal (rank -> inverse normal) 변환
  3. graphical lasso로 희소 정밀도 행렬 -> 부분상관
  4. bootstrap 안정성 선택
  5. E_fit = MST(w) ∪ top-r   <-- 연결성 자동 보장
     E_holdout = 안정적 강한 쌍 중 E_fit에 없는 것

리뷰 반영 (v2 계획서 대비 변경):
  [A-1] 조건 벡터를 y 하나에서 c=(y,age,sex,icu_type)로 확장.
        v2는 CDG를 (y,age,sex,ICU)에 residualize하면서 generator는 y만 조건으로
        받으므로, rho_real과 rho_syn이 서로 다른 추정량이 되는 버그가 있었다.
  [C-3] E_fit을 "E_candidate의 70%"로 뽑으면 16노드 연결에 필요한 15간선보다
        적어져 그래프가 분리되고 filler 간선이 토폴로지를 희석한다.
        MST ∪ top-r로 바꾸면 연결성이 자동 보장되고 filler가 불필요하다.
  [B]   held-out 쌍의 그래프 거리 분포를 보고한다. Delta_HDE가 단순히 거리
        배치로 설명되는지 사전에 알 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import graphical_lasso
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from features import CONDITION_VARS, CORE16  # noqa: E402

MAX_DEGREE = 3       # v2 §7.6
STABILITY_TAU = 0.70  # v2 §7.5
N_BOOTSTRAP = 100


def nonparanormal(X: np.ndarray) -> np.ndarray:
    """rank -> inverse normal (v2 §7.3). HDE도 반드시 이 공간에서 계산해야 한다 [A-2]."""
    N = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / N) for j in range(X.shape[1])])


def residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """조건 벡터 c에 대해 회귀 후 residual (v2 §7.2). c는 real/synthetic 양쪽에 동일 적용."""
    return X - LinearRegression().fit(C, X).predict(C)


def partial_corr(X: np.ndarray, alpha: float) -> np.ndarray:
    """graphical lasso -> 부분상관 행렬 (v2 §7.4)."""
    S = np.corrcoef(X, rowvar=False)
    _, prec = graphical_lasso(S, alpha=alpha, max_iter=200)
    d = np.sqrt(np.diag(prec))
    rho = -prec / np.outer(d, d)
    np.fill_diagonal(rho, 1.0)
    return rho


def stability(X: np.ndarray, alpha: float, B: int, rng: np.random.Generator) -> np.ndarray:
    """subsample 반복에서 각 간선이 선택된 빈도 (v2 §7.5)."""
    n, p = X.shape
    m = int(0.8 * n)
    cnt = np.zeros((p, p))
    for _ in range(B):
        idx = rng.choice(n, m, replace=False)
        try:
            rho = partial_corr(X[idx], alpha)
        except Exception:
            continue
        cnt += (np.abs(rho) > 1e-4).astype(float)
    np.fill_diagonal(cnt, 0)
    return cnt / B


def build_graph(rho, stab, names, rng, n_holdout_strong=3):
    """E_holdout을 강도 층화로 먼저 뽑고, 나머지로 E_fit = MST ∪ top-r 를 만든다.

    순서가 중요하다. E_fit을 먼저 MST∪top-r로 만들면 강한 간선을 전부 가져가버려
    E_holdout에 |rho|~0.05짜리 약한 관계만 남는다 -> HDE가 추정 노이즈만 재게 된다.
    (demo 실행에서 실제로 확인됨: held-out이 전부 |rho| <= 0.16)

    따라서:
      1) 강한 간선 중 n_holdout_strong개를, "빼도 나머지가 연결을 유지하는" 것만 골라 held-out
      2) 약한 간선 일부도 held-out (강도 스펙트럼을 덮기 위해)
      3) 남은 간선으로 E_fit = MST ∪ top-r  -> 연결성 보장, filler 불필요
    """
    p = len(names)
    w = np.abs(rho) * (stab >= STABILITY_TAU)
    np.fill_diagonal(w, 0)
    cand = sorted(((i, j, w[i, j]) for i in range(p) for j in range(i + 1, p) if w[i, j] > 0),
                  key=lambda e: -e[2])

    def connected_without(excluded: set) -> bool:
        H = nx.Graph()
        H.add_nodes_from(range(p))
        H.add_edges_from((i, j) for i, j, _ in cand if tuple(sorted((i, j))) not in excluded)
        return nx.is_connected(H)

    # 1) 강한 간선을 held-out으로 (연결성 유지 조건 하에)
    holdout: set = set()
    for i, j, _ in cand:
        if len(holdout) >= n_holdout_strong:
            break
        e = tuple(sorted((i, j)))
        if connected_without(holdout | {e}):
            holdout.add(e)

    # 2) 약한 간선도 일부 held-out (강도 스펙트럼 커버)
    weak = [tuple(sorted((i, j))) for i, j, _ in cand[len(cand) // 2:]
            if tuple(sorted((i, j))) not in holdout]
    rng.shuffle(weak)
    for e in weak[: max(3, len(weak) // 3)]:
        if connected_without(holdout | {e}):
            holdout.add(e)

    # 3) 남은 간선으로 E_fit 구성 — 차수 제약 Delta<=3 을 처음부터 강제한다.
    #
    #    nx.maximum_spanning_tree는 차수를 제약하지 않아 maxdeg=5짜리 조밀한 그래프가
    #    나온다. 그러면 지름이 작아져 그래프 거리가 변별력을 잃고, light-cone 기반
    #    확증 검정이 무력해진다 (demo에서 L=2일 때 CDG와 permuted가 12/13으로 동일).
    #    -> Kruskal을 차수 제약 하에서 직접 돌린다 (degree-constrained spanning forest).
    rest = [(i, j, wt) for i, j, wt in cand if tuple(sorted((i, j))) not in holdout]

    G = nx.Graph()
    G.add_nodes_from(range(p))
    uf = {i: i for i in range(p)}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    # 3a) 가중치 순 Kruskal + 차수 제약 -> 신장 숲
    for i, j, _ in rest:
        if G.degree(i) < MAX_DEGREE and G.degree(j) < MAX_DEGREE and find(i) != find(j):
            G.add_edge(i, j)
            uf[find(i)] = find(j)

    # 3b) 아직 분리된 성분이 있으면, 차수 여유가 있는 노드끼리 최고 가중치로 잇는다
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        best = None
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                for u in comps[a]:
                    for v in comps[b]:
                        if G.degree(u) < MAX_DEGREE and G.degree(v) < MAX_DEGREE:
                            if best is None or w[u, v] > best[2]:
                                best = (u, v, w[u, v])
        if best is None:  # 차수 제약을 지키며 연결할 방법이 없음
            raise RuntimeError(f"Delta<={MAX_DEGREE} 하에서 연결 불가. MAX_DEGREE를 올리십시오.")
        G.add_edge(best[0], best[1])

    # 3c) 차수 여유 안에서 상위 가중치 간선 추가 (top-r)
    for i, j, _ in rest:
        if not G.has_edge(i, j) and G.degree(i) < MAX_DEGREE and G.degree(j) < MAX_DEGREE:
            G.add_edge(i, j)

    E_fit = [tuple(sorted(e)) for e in G.edges()]
    return G, E_fit, sorted(holdout)


def report_holdout_distances(G: nx.Graph, E_holdout: list, names: list[str], rho: np.ndarray) -> None:
    """[B] held-out 쌍의 그래프 거리 분포. Delta_HDE가 거리로 설명되는지 사전 확인."""
    print("\n  held-out 쌍의 CDG 그래프 거리 (light cone 도달 여부를 결정)")
    dists = []
    for (i, j) in E_holdout:
        d = nx.shortest_path_length(G, i, j) if nx.has_path(G, i, j) else 99
        dists.append(d)
        print(f"    d={d}  |rho|={abs(rho[i,j]):.3f}   {names[i]} -- {names[j]}")
    if dists:
        arr = np.array(dists)
        print(f"\n    거리 분포: " + ", ".join(f"d={d}:{int((arr==d).sum())}" for d in sorted(set(dists))))
        for L in (1, 2):
            n_in = int((arr <= 2 * L).sum())
            print(f"    L={L} (도달 반경 {2*L}): {n_in}/{len(arr)} 쌍이 표현 가능")


def main(path: Path, alpha: float) -> None:
    df = pd.read_parquet(path)
    names = [f.name for f in CORE16]
    df = df.dropna(subset=names + CONDITION_VARS[1:])
    print("=" * 70)
    print(f"CDG 추정: {path}   (n={len(df):,}, p={len(names)})")
    print("=" * 70)

    if len(df) < 200:
        print("\n  ** 경고 ** 표본이 너무 작습니다 (demo). graphical lasso가 불안정합니다.")
        print("     이 실행은 파이프라인 검증용이며, 나온 그래프를 실험에 쓰면 안 됩니다.")

    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)  # 조건 벡터 c [A-1]

    X = nonparanormal(residualize(X, C))
    rho = partial_corr(X, alpha)

    rng = np.random.default_rng(20260711)
    stab = stability(X, alpha, N_BOOTSTRAP, rng)

    G, E_fit, E_hold = build_graph(rho, stab, names, rng)
    print(f"\n  E_fit     : {len(E_fit)} 간선  (연결={nx.is_connected(G)}, 최대차수={max(dict(G.degree()).values())})")
    print(f"  E_holdout : {len(E_hold)} 간선  (topology/loss/checkpoint에 사용 금지)")

    print("\n  E_fit 간선 (회로의 RZZ 위치)")
    for (i, j) in sorted(E_fit, key=lambda e: -abs(rho[e[0], e[1]])):
        print(f"    |rho|={abs(rho[i,j]):.3f}  stab={stab[i,j]:.2f}   {names[i]} -- {names[j]}")

    report_holdout_distances(G, E_hold, names, rho)

    import paths

    out = paths.PROCESSED / "cdg.npz"
    np.savez(out, rho=rho, stab=stab, E_fit=np.array(E_fit), E_holdout=np.array(E_hold),
             names=np.array(names))
    print(f"\n  저장: {out}")


if __name__ == "__main__":
    import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    data = a.data or paths.PROCESSED / ("cohort_demo.parquet" if a.demo else "cohort_v31.parquet")
    main(data, a.alpha)

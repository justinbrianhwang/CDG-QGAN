"""학습된 모델이 실제로 무엇을 만들고 있는가 (벤치마크 null 진단 2단계).

diag_benchmark.py 결과:
    의존성 0 모델      = 0.0648
    학습된 모델        = 0.1359   <- 2.1배 나쁘다

=> 모델은 참 의존성을 못 배우는 게 아니라, **없는 의존성을 지어내고 있다**.

가설: 조건 벡터 c 때문이다.
    생성기는 x~_u = h_u(q_u, c) 이고 c가 모든 특징에 공유된다.
    -> c를 통제하지 않고 재는 무조건부 부분상관에는 c가 만든 상관이 전부 섞인다.
    -> 참값(합성 teacher)에는 그 성분이 없다 (C를 X와 독립으로 뽑았으므로).
    CDG는 "c 조건부"로 정의되는데 평가는 무조건부로 하고 있다 = 추정량 불일치.

검증:
    [A] 무조건부 부분상관  (현재 train.dependency_error)
    [B] c 잔차화 후 부분상관 (CDG 정의와 일치)
    가설이 맞으면 [B]에서 오차가 floor 아래로 내려가고 aligned/permuted가 갈라진다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_SYN, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from train import Cfg, _npn, generate, partial_corr_matrix, train  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731
SEEDS = [0, 1, 2]


def partial_corr_resid(X: np.ndarray, C: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """c를 통제한 부분상관 — CDG 정의와 같은 공간.

    nonparanormal 변환 후 c(및 상수항)에 선형회귀하고 잔차의 부분상관을 잰다.
    build_cdg.py가 CDG를 만들 때 쓰는 것과 동일한 절차.
    """
    Xn = _npn(X)
    D = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(D, Xn, rcond=None)
    R = Xn - D @ beta
    S = np.corrcoef(R, rowvar=False)
    P = np.linalg.inv(S + ridge * np.eye(S.shape[0]))
    d = np.sqrt(np.diag(P))
    M = -P / np.outer(d, d)
    np.fill_diagonal(M, 1.0)
    return M


def split(zr, zs, G):
    e = [abs(zs[i, j] - zr[i, j]) for i, j in G.edges()]
    nn = [abs(zs[i, j] - zr[i, j]) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)
          if not G.has_edge(i, j)]
    return float(np.mean(e + nn)), float(np.mean(e)), float(np.mean(nn))


def main() -> None:
    rng = np.random.default_rng(20260711)
    G = teacher_graph(rng)
    X, C, _ = teacher_data(G, N_TRAIN, rng)
    Gp = isomorphic_permuted(G, np.random.default_rng(7))

    zr_u = Z(partial_corr_matrix(X))          # 무조건부 참값
    zr_c = Z(partial_corr_resid(X, C))        # c 잔차화 참값

    # floor: 의존성 0 모델
    fl_u, fl_c = [], []
    for s in range(3):
        r = np.random.default_rng(100 + s)
        Xi = np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])
        Ci = C[r.permutation(len(C))]
        fl_u.append(split(zr_u, Z(partial_corr_matrix(Xi)), G))
        fl_c.append(split(zr_c, Z(partial_corr_resid(Xi, Ci)), G))

    print("=" * 88)
    print("학습된 모델은 무엇을 만들고 있는가")
    print("=" * 88)
    print(f"  {'모델':<22} {'[A] 무조건부 (현재 지표)':>30}   {'[B] c 잔차화 (CDG 정의)':>30}")
    print(f"  {'':<22} {'120쌍':>9}{'간선19':>9}{'비간선101':>11}   "
          f"{'120쌍':>9}{'간선19':>9}{'비간선101':>11}")
    print("  " + "-" * 84)

    def row(name, a, b):
        print(f"  {name:<22} {a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>11.4f}   "
              f"{b[0]:>9.4f}{b[1]:>9.4f}{b[2]:>11.4f}", flush=True)

    row("floor (의존성 0)", np.mean(fl_u, 0), np.mean(fl_c, 0))

    for name, Gv in [("aligned", G), ("permuted", Gp), ("no_entangle", nx.empty_graph(N_FEAT))]:
        au, ac = [], []
        t0 = time.time()
        for s in SEEDS:
            Gm, _ = train(X, C, list(Gv.edges()), Cfg(depth=1, seed=s))
            Xs = generate(Gm, C, N_SYN, seed=s)
            Cs = C[np.random.default_rng(s).integers(0, len(C), N_SYN)]  # generate와 동일 seed 규칙
            au.append(split(zr_u, Z(partial_corr_matrix(Xs)), G))
            ac.append(split(zr_c, Z(partial_corr_resid(Xs, Cs)), G))
        row(f"{name} ({time.time()-t0:.0f}s)", np.mean(au, 0), np.mean(ac, 0))

    print()
    print("  읽는 법:")
    print("    [A]에서 '비간선101' 열이 크면 -> 가짜 의존성을 뿜고 있다 (위양성).")
    print("    [B]에서 그게 줄면        -> 원인은 공유 조건 c다. 지표를 CDG 정의에 맞추면 된다.")
    print("    [B]에서도 안 줄면        -> 원인은 c가 아니라 학습/회로다. 더 파야 한다.")
    print("    aligned vs permuted 가 [B]에서 갈라지면 확증 설계가 살아난다.")


if __name__ == "__main__":
    main()

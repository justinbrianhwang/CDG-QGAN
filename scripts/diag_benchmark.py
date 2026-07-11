"""벤치마크 null 결과 진단 (학습 불필요).

benchmark_synthetic.py가 네 변형 모두 0.135 +- 0.009를 내놓았다.
그런데 우리는 **0.135가 좋은 값인지 나쁜 값인지 모른다.** 기준선이 없다.

여기서 학습 없이 두 가지 기준선을 계산한다:

  [floor]  의존성을 전혀 만들지 않는 모델 (열별 독립 셔플).
           참 주변분포는 완벽히 맞추고 의존성만 0인 모델의 120쌍 오차.
           이것이 "아무것도 학습 안 함"의 점수다.

  [ceil]   참 데이터를 다시 뽑은 것 (같은 Omega, 다른 seed).
           유한표본 잡음만 있는, 도달 가능한 최선의 점수.

그리고 오차를 쪼갠다:
  - 참 간선 19쌍   (위음성: 있어야 할 의존성을 못 만듦)
  - 비간선 101쌍   (위양성: 없어야 할 의존성을 지어냄)

만약 학습된 모델(0.135)이 floor보다 **나쁘면**, 모델은 의존성을 배우는 게 아니라
가짜 의존성을 뿜고 있는 것이고, 120쌍 평균 지표는 그 잡음에 지배당한다.
-> 확증 실험(90 run)을 돌리기 전에 반드시 해결해야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from train import partial_corr_matrix  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731


def split_error(zr: np.ndarray, zs: np.ndarray, G: nx.Graph):
    edge, non = [], []
    for i in range(N_FEAT):
        for j in range(i + 1, N_FEAT):
            (edge if G.has_edge(i, j) else non).append(abs(zs[i, j] - zr[i, j]))
    allp = edge + non
    return np.mean(allp), np.mean(edge), np.mean(non), len(edge), len(non)


def main() -> None:
    rng = np.random.default_rng(20260711)
    G = teacher_graph(rng)
    X, C, Om = teacher_data(G, N_TRAIN, rng)

    Rr = partial_corr_matrix(X)
    zr = Z(Rr)

    print("=" * 78)
    print("벤치마크 진단 — 0.135는 좋은 값인가 나쁜 값인가")
    print("=" * 78)

    # 참 부분상관의 크기 — 애초에 잡을 신호가 얼마나 되는가
    e_true = [abs(Rr[i, j]) for i, j in G.edges()]
    n_true = [abs(Rr[i, j]) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)
              if not G.has_edge(i, j)]
    print(f"\n  참 데이터의 부분상관 |rho|")
    print(f"    간선 19쌍   : 평균 {np.mean(e_true):.3f}  (범위 {min(e_true):.3f}~{max(e_true):.3f})")
    print(f"    비간선 101쌍: 평균 {np.mean(n_true):.3f}  (범위 {min(n_true):.3f}~{max(n_true):.3f})")
    print(f"    -> Fisher-z로 간선의 평균 크기 = {np.mean(np.abs(Z(np.array(e_true)))):.3f}")

    print(f"\n  {'기준선':<34} {'120쌍':>8} {'간선19':>8} {'비간선101':>10}")
    print("  " + "-" * 64)

    # [floor] 의존성 0 모델: 각 열을 독립 셔플 -> 주변분포 완벽, 의존성 0
    errs = []
    for s in range(5):
        r = np.random.default_rng(100 + s)
        Xi = np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])
        errs.append(split_error(zr, Z(partial_corr_matrix(Xi)), G))
    f = np.mean([e[0] for e in errs]), np.mean([e[1] for e in errs]), np.mean([e[2] for e in errs])
    print(f"  {'[floor] 의존성 0 (독립 셔플)':<34} {f[0]:>8.4f} {f[1]:>8.4f} {f[2]:>10.4f}")

    # [ceil] 참 분포 재추출 — 유한표본 잡음만
    errs = []
    for s in range(5):
        Xn, _, _ = teacher_data(G, N_TRAIN, np.random.default_rng(200 + s))
        errs.append(split_error(zr, Z(partial_corr_matrix(Xn)), G))
    c = np.mean([e[0] for e in errs]), np.mean([e[1] for e in errs]), np.mean([e[2] for e in errs])
    print(f"  {'[ceil]  참 분포 재추출 (잡음만)':<34} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>10.4f}")

    print(f"  {'[obs]   학습된 모델 (벤치마크)':<34} {0.1359:>8.4f} {'?':>8} {'?':>10}")

    print()
    print("=" * 78)
    print("해석")
    print("=" * 78)
    print(f"  의존성을 전혀 안 만드는 모델의 점수 = {f[0]:.4f}")
    print(f"  학습된 모델의 점수                  = 0.1359")
    if 0.1359 > f[0]:
        print()
        print("  >> 학습된 모델이 '아무것도 안 하는 모델'보다 **나쁘다**.")
        print("     모델은 참 의존성을 배우는 게 아니라 **가짜 의존성을 뿜고 있다**.")
        print("     120쌍 평균은 그 잡음(위양성)에 지배당하고, 정렬 효과는 묻힌다.")
        print("     -> 확증 실험을 돌려도 아무것도 안 나온다. 원인 제거가 먼저다.")
    else:
        print()
        print("  >> 학습된 모델이 floor보다는 낫다. 그러면 지표가 아니라 신호 희석 문제다.")
    print()
    print(f"  도달 가능한 최선(잡음 바닥) = {c[0]:.4f}")
    print(f"  floor - ceil = {f[0]-c[0]:.4f}  <- 이 구간이 '학습으로 벌 수 있는 전부'다.")
    print("     이 구간이 좁으면 어떤 대조도 통계적으로 안 잡힌다.")


if __name__ == "__main__":
    main()

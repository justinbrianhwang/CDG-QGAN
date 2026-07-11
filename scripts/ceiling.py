"""게이트 실험: 얕은 graph-local 양자 생성기의 상관 표현력 천장.

질문:
    깊이 L의 회로에서, 그래프 거리 d인 두 특징 사이에 만들어낼 수 있는
    조건부 의존성의 최대 크기는 얼마인가?

왜 중요한가:
    x~_u = h_u(q_u, c) 이고 h_u는 단조 1차원 맵이므로, HDE가 계산되는
    nonparanormal 공간의 상관은 head와 무관하게 q의 copula로만 결정된다.
    따라서 천장은 순전히 회로의 성질이다.

    실제 임상 부분상관에는 |rho| > 0.9 인 쌍이 있다 (Hb-Hct, SBP-MAP-DBP).
    천장이 그보다 낮으면 CDG와 permuted-CDG 양쪽이 천장에 붙어버려
    확증 대조가 뭉개진다. -> 설계를 바꿔야 한다.

부수 효과:
    d > 2L 에서 상관이 0으로 나오면 v2 §4.8의 따름정리가 수치로 검증된다.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from qsim import GraphLocalQuantumGenerator, normal_score_corr, pearson  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 400
RESTARTS = 3
EVAL_BATCH = 16384  # no_grad -> 중간텐서가 없으므로 크게 잡아도 된다
MEM_BUDGET = 6e9    # autograd 중간텐서 예산 (bytes)


def train_batch(n: int, depth: int) -> int:
    """autograd 중간텐서가 메모리 예산 안에 들도록 훈련 배치를 정한다.

    중간텐서 개수 ~ 5*L*n, 각각 B * 2^n * 8 bytes (complex64).
    n=13, L=3에서 배치 4096이면 ~50GB로 32GB VRAM을 넘긴다.
    """
    per = 5 * depth * n * (2**n) * 8
    return int(np.clip(MEM_BUDGET / per, 256, 8192))


def max_corr(depth: int, dist: int, seed_base: int = 0) -> tuple[float, float, int]:
    """경로 그래프에서 거리 dist인 쌍의 최대 달성 가능 상관.

    반환: (nonparanormal 공간 |상관|, Pearson |상관|, 훈련 배치)
    """
    # u 양옆으로 light cone 반경 L 만큼 패딩. 경로 그래프이므로 거리 = 인덱스 차.
    n = 2 * depth + dist + 1
    u, v = depth, depth + dist
    edges = [(i, i + 1) for i in range(n - 1)]
    B = train_batch(n, depth)

    best_ns, best_p = 0.0, 0.0
    for r in range(RESTARTS):
        torch.manual_seed(seed_base + 1000 * r)
        gen = GraphLocalQuantumGenerator(n, edges, depth, seed=seed_base + 1000 * r).to(DEVICE)
        opt = torch.optim.Adam(gen.parameters(), lr=0.05)

        z = 2 * torch.rand(B, n, device=DEVICE) - 1   # local latent, iid
        y = torch.zeros(B, device=DEVICE)             # 조건 고정 (conditional on c)

        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            q = gen(z, y)
            loss = -pearson(q[:, u], q[:, v]).abs()   # 미분 가능한 대리 목적함수
            loss.backward()
            opt.step()

        # 평가는 큰 배치에서 no_grad로 -> 추정 오차를 줄인다
        with torch.no_grad():
            ze = 2 * torch.rand(EVAL_BATCH, n, device=DEVICE) - 1
            ye = torch.zeros(EVAL_BATCH, device=DEVICE)
            qe = gen(ze, ye).cpu().numpy()
        ns = abs(normal_score_corr(qe[:, u], qe[:, v]))  # HDE가 사는 공간에서 보고
        pe = abs(float(np.corrcoef(qe[:, u], qe[:, v])[0, 1]))
        best_ns, best_p = max(best_ns, ns), max(best_p, pe)
        del gen, z, y
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    return best_ns, best_p, B


def main() -> None:
    print("=" * 74)
    print("회로 상관 표현력 천장  (graph-local RZZ 생성기, 경로 그래프)")
    print(f"device={DEVICE}  steps={STEPS}  restarts={RESTARTS}  eval_batch={EVAL_BATCH}")
    print("=" * 74)
    print()
    print("  값 = 달성 가능한 최대 |상관| (nonparanormal 공간 = HDE가 계산되는 공간)")
    print("  회색 배경(--) = 이론상 0이어야 하는 영역 (d > 2L)")
    print()

    depths = [1, 2, 3]
    dists = [1, 2, 3, 4, 5, 6]

    header = "  L \\ d  " + "".join(f"{d:>10}" for d in dists)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: dict[tuple[int, int], tuple[float, float]] = {}
    rows = []
    for L in depths:
        row = f"  L={L}    "
        for d in dists:
            t0 = time.time()
            ns, pe, B = max_corr(L, d, seed_base=17)
            results[(L, d)] = (ns, pe)
            mark = "" if d <= 2 * L else "*"
            row += f"{ns:>9.3f}{mark:<1}"
            print(f"    [L={L} d={d}] n={2*L+d+1:2d} batch={B:5d} "
                  f"ns={ns:.4f} pearson={pe:.4f} ({time.time()-t0:.0f}s)", flush=True)
        rows.append(row)
        print(row, flush=True)
    print()
    print("  요약")
    print(header)
    for r in rows:
        print(r)
    print()
    print("  * = d > 2L, 따름정리 1에 의해 0이어야 하는 칸")
    print()

    # ---- 따름정리 검증 ----
    print("=" * 74)
    print("검증 1: light-cone 따름정리 (d > 2L  =>  조건부 상관 = 0)")
    print("=" * 74)
    inside = [(k, v[0]) for k, v in results.items() if k[1] <= 2 * k[0]]
    outside = [(k, v[0]) for k, v in results.items() if k[1] > 2 * k[0]]
    max_out = max((v for _, v in outside), default=0.0)
    min_in = min((v for _, v in inside), default=0.0)
    print(f"  light cone 안 (d <= 2L): 최소 |corr| = {min_in:.4f}   ({len(inside)}칸)")
    print(f"  light cone 밖 (d >  2L): 최대 |corr| = {max_out:.4f}   ({len(outside)}칸)")
    ok = max_out < 0.02
    print(f"  -> 따름정리 1 {'PASS' if ok else 'FAIL'}: 밖에서는 상관을 만들 수 없다")
    print()

    # ---- 게이트 판정 ----
    print("=" * 74)
    print("검증 2: 게이트 — 실제 임상 부분상관이 천장 안에 들어오는가")
    print("=" * 74)
    targets = {
        "Hemoglobin-Hematocrit": 0.95,
        "SBP-MAP":               0.90,
        "Creatinine-BUN":        0.60,
        "Na-Cl":                 0.55,
        "약한 임상 관계 다수":     0.20,
    }
    print()
    for L in depths:
        c1 = results[(L, 1)][0]  # 인접 쌍 = 최선의 경우
        print(f"  L={L}, 인접 쌍(d=1) 천장 = {c1:.3f}")
        for name, tgt in targets.items():
            ok_t = c1 >= tgt
            print(f"      {'OK  ' if ok_t else 'FAIL'}  {name:<24} |rho|={tgt:.2f}")
        print()

    np.save("results_ceiling.npy", results, allow_pickle=True)
    print("  -> results_ceiling.npy 저장")


if __name__ == "__main__":
    main()

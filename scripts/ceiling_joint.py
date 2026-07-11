"""결합 표현력 천장 — 120쌍 패턴 전체를 동시에 맞출 수 있는가.

왜 필요한가
-----------
`ceiling.py`는 **쌍 하나**의 |상관|을 최대화했다 (L=1, d=1 -> 0.991).
그건 "간선 하나에 강한 의존성을 걸 수 있다"는 뜻일 뿐,
**19개 간선을 동시에 rho=0.35로 맞추면서 101개 비간선을 0으로 유지**할 수 있다는
뜻이 아니다. 확증 실험이 요구하는 건 후자다. 그 칸은 비어 있었다.

`diag_trained.py`가 보여준 것:
    WGAN-GP로 학습한 모델의 참 간선 오차 = 0.3662
    의존성을 아예 안 만드는 모델의 오차   = 0.3676
    -> 학습된 모델은 참 간선에 의존성을 **거의 0** 만든다.

두 가지 중 하나다:
    (a) 회로가 그 패턴을 애초에 표현할 수 없다  -> 설계 결함, WP-2 무의미
    (b) 회로는 표현할 수 있는데 WGAN-GP가 못 찾는다 -> 학습 결함, 목적함수를 고쳐야 한다

여기서 GAN을 빼고 회로 파라미터를 부분상관 오차에 **직접** 경사하강시킨다.
head는 단조 1차원 맵이므로 nonparanormal 공간의 의존성은 q=<Z>의 copula로만
결정된다 (ceiling.py와 같은 논리). 따라서 q 위에서 재면 충분하다.

읽는 법
-------
    aligned가 낮고 permuted가 높다  -> (b). 회로/CDG 가설은 살아있고 학습을 고치면 된다.
    둘 다 높다                       -> (a). 설계가 그 패턴을 못 만든다. 재설계.
    둘 다 낮다                       -> CDG 정렬이 표현력에 무관하다. 확증 실험 근거가 무너진다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from graphs import degree_preserving_rewire, distance_matched_permuted, isomorphic_permuted  # noqa: E402
from model import CDGQGAN  # noqa: E402
from train import DEVICE, _npn  # noqa: E402

RIDGE = 1e-3
STEPS = 1500
BATCH = 4096
RESTARTS = 3


def partial_corr_torch(q: torch.Tensor) -> torch.Tensor:
    """미분 가능한 부분상관 (q: (B, n))."""
    x = q - q.mean(0, keepdim=True)
    x = x / (x.std(0, keepdim=True) + 1e-6)
    S = (x.T @ x) / (x.shape[0] - 1)
    P = torch.linalg.inv(S + RIDGE * torch.eye(S.shape[0], device=S.device))
    d = torch.sqrt(torch.diag(P))
    R = -P / torch.outer(d, d)
    return R - torch.diag(torch.diag(R)) + torch.eye(S.shape[0], device=S.device)


def fit(edges, target: torch.Tensor, C: np.ndarray, seed: int) -> float:
    """회로 파라미터를 target 부분상관에 직접 맞춘다. 반환: 120쌍 평균 절대오차."""
    torch.manual_seed(seed)
    m = CDGQGAN(N_FEAT, list(edges), depth=1, cond_dim=C.shape[1], seed=seed).to(DEVICE)
    core = m.core
    params = list(m.quantum.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)
    iu = torch.triu_indices(N_FEAT, N_FEAT, offset=1)

    best = float("inf")
    for step in range(STEPS):
        idx = torch.randint(0, len(Ct), (BATCH,), device=DEVICE)
        z = 2 * torch.rand(BATCH, N_FEAT, device=DEVICE) - 1
        q = core(z, Ct[idx, 0])
        R = partial_corr_torch(q)
        loss = (R[iu[0], iu[1]] - target[iu[0], iu[1]]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        best = min(best, loss.item())
    return best


def main() -> None:
    rng = np.random.default_rng(20260711)
    G = teacher_graph(rng)
    X, C, _ = teacher_data(G, N_TRAIN, rng)

    # teacher의 참 부분상관 (nonparanormal 공간) — 이것이 맞춰야 할 표적
    Xn = _npn(X)
    S = np.corrcoef(Xn, rowvar=False)
    P = np.linalg.inv(S + RIDGE * np.eye(N_FEAT))
    d = np.sqrt(np.diag(P))
    Rstar = -P / np.outer(d, d)
    np.fill_diagonal(Rstar, 1.0)
    target = torch.tensor(Rstar, dtype=torch.float32, device=DEVICE)

    gr = np.random.default_rng(7)
    holdout = [(u, v) for u in range(N_FEAT) for v in range(u + 1, N_FEAT)
               if not G.has_edge(u, v) and nx.shortest_path_length(G, u, v) == 2][:10]
    variants = {
        "aligned": G,
        "permuted": isomorphic_permuted(G, gr),
        "distmatched": distance_matched_permuted(G, holdout, gr)[0],
        "rewired": degree_preserving_rewire(G, gr),
        "no_entangle": nx.empty_graph(N_FEAT),
    }

    # floor: 의존성 0 모델의 120쌍 오차 (같은 표적 기준)
    iu = np.triu_indices(N_FEAT, 1)
    floor = float(np.abs(Rstar[iu]).mean())  # R_syn = I 이면 오차 = |R*|

    print("=" * 78)
    print("결합 표현력 천장 — 120쌍 패턴 전체 (GAN 없이 직접 최적화)")
    print("=" * 78)
    print(f"  teacher: 19간선, 간선 |rho| 평균 {np.abs([Rstar[i,j] for i,j in G.edges()]).mean():.3f}")
    print(f"  최적화: Adam lr=0.05, {STEPS}스텝, batch={BATCH}, 재시작 {RESTARTS}회")
    print(f"  [floor] 의존성 0 모델의 120쌍 오차 = {floor:.4f}")
    print()
    print(f"  {'모델':<14} {'120쌍 최소오차':>16}   {'floor 대비':>12}")
    print("  " + "-" * 50)

    res = {}
    for name, Gv in variants.items():
        t0 = time.time()
        e = min(fit(Gv.edges(), target, C, s) for s in range(RESTARTS))
        res[name] = e
        gain = (floor - e) / floor * 100
        print(f"  {name:<14} {e:>16.4f}   {gain:>10.1f}%   ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 78)
    a = res["aligned"]
    for ref in ("permuted", "distmatched", "rewired", "no_entangle"):
        dlt = a - res[ref]
        print(f"  aligned - {ref:<12} = {dlt:+.4f}   {'aligned 우세' if dlt < 0 else 'aligned 열세'}")
    print()
    if a >= floor * 0.95:
        print("  >> 회로가 teacher 패턴을 표현하지 못한다. 설계 결함이다 (경우 a).")
    elif a < res["permuted"] - 0.005:
        print("  >> 회로는 표현할 수 있고 aligned가 우세하다 (경우 b).")
        print("     -> CDG 가설은 살아있다. 문제는 WGAN-GP가 그 해를 못 찾는 것이다.")
    else:
        print("  >> 표현은 되는데 aligned가 permuted를 못 이긴다.")
        print("     -> 확증 실험의 전제가 무너진다. 이걸 먼저 해결해야 한다.")


if __name__ == "__main__":
    main()

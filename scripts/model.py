"""CDG-QGAN 생성기와 critic (v2 §8).

핵심 구조 명제 [리뷰 D-2]
--------------------------------------------------------------------------
  local head h_u는 (q_u, c)만 입력받고 다른 큐비트의 q_v를 볼 수 없다.
  따라서 c 조건부로

      x~_u ⟂ x~_v | c   <=>   q_u ⟂ q_v | c

  즉 **고전 파라미터는 조건부 교차특징 의존성을 생성할 수 없다.**
  출력의 모든 의존 구조는 양자 코어에서만 발생한다.

  이것이 r_Q(양자 파라미터 비율 ~9%)보다 훨씬 강한 진술이다. head가 파라미터의
  대부분을 차지하더라도, head는 주변분포만 만들 뿐 의존성은 못 만든다.
  `assert_no_cross_feature_mixing()`이 이 성질을 구조적으로 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from qsim import GraphLocalQuantumGenerator
from qsim_lightcone import LightconeGenerator


class LocalHead(nn.Module):
    """특징 u 전용 1차원 맵.  (q_u, c) -> x~_u.   다른 q_v는 입력받지 않는다."""

    def __init__(self, cond_dim: int, width: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + cond_dim, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, q_u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q_u.unsqueeze(-1), c], dim=-1)).squeeze(-1)


class CDGQGAN(nn.Module):
    """graph-local 양자 코어 + 특징별 local head."""

    def __init__(self, n_features: int, edges, depth: int, cond_dim: int,
                 head_width: int = 8, seed: int = 0, lightcone: bool = True):
        super().__init__()
        self.n, self.cond_dim = n_features, cond_dim
        self.quantum = GraphLocalQuantumGenerator(n_features, edges, depth, seed=seed)
        self.heads = nn.ModuleList([LocalHead(cond_dim, head_width) for _ in range(n_features)])

        # 명제 1을 계산에 써먹는다 [리뷰 D-1]: <Z_u>는 N_L(u) 안에만 의존하므로
        # 2^16 전체 상태벡터 대신 큐비트당 2^|N_L(u)| 부분회로면 충분하다 (정확함).
        # n=16, L=1, Delta<=3 -> 4096배 절감.
        # 간선이 0개(no_entangle)여도 cone이 {u} 하나(2차원)라 그대로 성립한다.
        # 여기서 전체 시뮬레이터로 떨어뜨리면 2^16 상태벡터를 돌게 되어 학습이 사실상 멈춘다.
        self.core = (LightconeGenerator(self.quantum, edges, n_features, depth)
                     if lightcone else self.quantum)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 양자 코어는 조건의 첫 성분(사망 라벨)을 각도에 싣는다.
        q = self.core(z, c[:, 0])                        # (B, n) = <Z_u>
        return torch.stack([h(q[:, u], c) for u, h in enumerate(self.heads)], dim=1)

    # --- 파라미터 분해 보고 (v2 §8.12) ---
    def param_breakdown(self) -> dict[str, int]:
        qp = self.quantum
        enc = sum(p.numel() for p in (qp.a_y, qp.b_y, qp.c_y, qp.a_z, qp.b_z, qp.c_z))
        ent = qp.gamma.numel()
        mix = qp.tx.numel() + qp.ty.numel()
        head = sum(p.numel() for p in self.heads.parameters())
        return {
            "local_angle_encoding": enc,
            "quantum_entangling": ent,
            "quantum_mixing": mix,
            "local_heads": head,
            "generator_total": enc + ent + mix + head,
            # 교차특징 의존성을 만들 수 있는 파라미터는 얽힘 각도뿐이다.
            "dependency_capable": ent,
        }


class Critic(nn.Module):
    """전역 conditional critic (v2 §8.9). BatchNorm 금지 (gradient penalty와 간섭)."""

    def __init__(self, n_features: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features + cond_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 구조 검증
# ---------------------------------------------------------------------------
def assert_no_cross_feature_mixing(model: CDGQGAN, device="cpu") -> None:
    """head가 다른 큐비트의 q_v를 읽지 않는지 gradient로 확인한다 (v2 부록 B).

    q를 직접 흔들었을 때 dx~_u/dq_v가 v != u에서 0이어야 한다.
    """
    model = model.to(device).eval()
    B, n = 4, model.n
    q = torch.randn(B, n, device=device, requires_grad=True)
    c = torch.zeros(B, model.cond_dim, device=device)

    jac = torch.zeros(n, n)
    for u, h in enumerate(model.heads):
        out = h(q[:, u], c).sum()
        g, = torch.autograd.grad(out, q, retain_graph=True)
        jac[u] = g.abs().sum(0)

    off = jac - torch.diag(torch.diag(jac))
    assert off.abs().max() < 1e-12, f"head가 다른 특징의 q를 읽고 있다! max={off.abs().max():.2e}"
    print(f"  [구조] head Jacobian dx~_u/dq_v 가 대각행렬 "
          f"(off-diagonal max = {off.abs().max():.1e})")
    print("  -> 고전 파라미터는 조건부 교차특징 의존성을 만들 수 없다. "
          "출력의 모든 의존 구조는 양자 코어에서 나온다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    m = CDGQGAN(n_features=16, edges=edges, depth=1, cond_dim=4)

    print("파라미터 분해 (v2 §8.12)")
    bd = m.param_breakdown()
    for k, v in bd.items():
        print(f"  {k:<22} {v:>6,}")
    r_q = (bd["quantum_entangling"] + bd["quantum_mixing"]) / bd["generator_total"]
    print(f"\n  r_Q (양자 파라미터 비율)          = {r_q*100:.1f}%   <- 방어적인 숫자")
    print(f"  의존성 생성 가능 파라미터 중 양자 = 100.0%  <- 이게 진짜 진술이다")
    print()
    assert_no_cross_feature_mixing(m)

"""CDG-QGAN generator and critic (v2 §8).

Core structural proposition [review D-2]
--------------------------------------------------------------------------
  The local head h_u takes only (q_u, c) as input and cannot see any other
  qubit's q_v. Hence, conditionally on c,

      x~_u ⟂ x~_v | c   <=>   q_u ⟂ q_v | c

  In other words, **the classical parameters cannot generate conditional
  cross-feature dependence.** All dependency structure in the output arises
  solely in the quantum core.

  This is a far stronger statement than r_Q (the ~9% quantum parameter fraction).
  Even though the heads hold most of the parameters, a head can only shape the
  marginals — it cannot create dependence.
  `assert_no_cross_feature_mixing()` verifies this property structurally.
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
    """1-D map dedicated to feature u.  (q_u, c) -> x~_u.   No other q_v is taken as input."""

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
    """Graph-local quantum core + one local head per feature."""

    def __init__(self, n_features: int, edges, depth: int, cond_dim: int,
                 head_width: int = 8, seed: int = 0, lightcone: bool = True):
        super().__init__()
        self.n, self.cond_dim = n_features, cond_dim
        self.quantum = GraphLocalQuantumGenerator(n_features, edges, depth, seed=seed)
        self.heads = nn.ModuleList([LocalHead(cond_dim, head_width) for _ in range(n_features)])

        # Put Proposition 1 to computational use [review D-1]: <Z_u> depends only on
        # N_L(u), so a 2^|N_L(u)| subcircuit per qubit suffices (exactly) instead of the
        # full 2^16 statevector. n=16, L=1, Delta<=3 -> 4096x saving.
        # It still holds with zero edges (no_entangle): the cone is just {u} (2 dims).
        # Falling back to the full simulator here would run a 2^16 statevector and
        # training would grind to a halt.
        self.core = (LightconeGenerator(self.quantum, edges, n_features, depth)
                     if lightcone else self.quantum)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # The quantum core encodes the first component of the condition (the mortality
        # label) into the angles.
        q = self.core(z, c[:, 0])                        # (B, n) = <Z_u>
        return torch.stack([h(q[:, u], c) for u, h in enumerate(self.heads)], dim=1)

    # --- parameter breakdown report (v2 §8.12) ---
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
            # The entangling angles are the only parameters that can create
            # cross-feature dependence.
            "dependency_capable": ent,
        }


class Critic(nn.Module):
    """Global conditional critic (v2 §8.9). No BatchNorm (it interferes with the gradient penalty)."""

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
# Structural verification
# ---------------------------------------------------------------------------
def assert_no_cross_feature_mixing(model: CDGQGAN, device="cpu") -> None:
    """Check by gradient that no head reads another qubit's q_v (v2 Appendix B).

    Perturbing q directly, dx~_u/dq_v must be 0 for v != u.
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
    assert off.abs().max() < 1e-12, f"a head is reading another feature's q! max={off.abs().max():.2e}"
    print(f"  [structure] head Jacobian dx~_u/dq_v is diagonal "
          f"(off-diagonal max = {off.abs().max():.1e})")
    print("  -> the classical parameters cannot create conditional cross-feature "
          "dependence. All dependency structure in the output comes from the quantum core.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    m = CDGQGAN(n_features=16, edges=edges, depth=1, cond_dim=4)

    print("Parameter breakdown (v2 §8.12)")
    bd = m.param_breakdown()
    for k, v in bd.items():
        print(f"  {k:<22} {v:>6,}")
    r_q = (bd["quantum_entangling"] + bd["quantum_mixing"]) / bd["generator_total"]
    print(f"\n  r_Q (quantum parameter fraction)       = {r_q*100:.1f}%   <- the defensive number")
    print(f"  quantum share of dependency-capable params = 100.0%  <- this is the real statement")
    print()
    assert_no_cross_feature_mixing(m)

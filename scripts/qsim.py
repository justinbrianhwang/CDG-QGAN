"""Batched statevector simulator for the graph-local quantum generator (CDG-QGAN v2 §4.2).

Circuit block (rightmost acts first):
    U^(l) = R_mix^(l) . E_G^(l) . S_enc^(l)

  S_enc : per-qubit local encoding.  The angles use only that qubit's local latent
          z_u and the condition y.
  E_G   : RZZ on the CDG edges only (diagonal gate).
  R_mix : per-qubit non-commuting RX/RY.  Without it, RZZ commutes with Z and the
          gradient of the entangling parameters is exactly 0 (confirmed numerically
          in Appendix B-1/B-2).

Qubit 0 = most significant bit.
"""

from __future__ import annotations

import numpy as np
import torch

C64 = torch.complex64


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def _apply_1q(psi: torch.Tensor, n: int, q: int, a00, a01, a10, a11) -> torch.Tensor:
    """Apply the 2x2 gate [[a00,a01],[a10,a11]] to qubit q.

    psi: (B, 2**n) complex. The coefficients a** must broadcast as (B,1) or as
    scalars (so that data encoding with a per-sample angle is supported).
    """
    B = psi.shape[0]
    left, right = 2**q, 2 ** (n - q - 1)
    v = psi.reshape(B, left, 2, right)
    v0, v1 = v[:, :, 0, :], v[:, :, 1, :]

    def bc(a):
        return a.reshape(-1, 1, 1) if torch.is_tensor(a) and a.ndim >= 1 else a

    a00, a01, a10, a11 = bc(a00), bc(a01), bc(a10), bc(a11)
    out = torch.stack([a00 * v0 + a01 * v1, a10 * v0 + a11 * v1], dim=2)
    return out.reshape(B, 2**n)


def ry(psi, n, q, t):
    c, s = torch.cos(t / 2).to(C64), torch.sin(t / 2).to(C64)
    return _apply_1q(psi, n, q, c, -s, s, c)


def rx(psi, n, q, t):
    c, s = torch.cos(t / 2).to(C64), torch.sin(t / 2).to(C64)
    return _apply_1q(psi, n, q, c, -1j * s, -1j * s, c)


def rz(psi, n, q, t):
    e = torch.exp(-0.5j * t.to(C64))
    return _apply_1q(psi, n, q, e, 0.0, 0.0, e.conj())


def _zsign(n: int, q: int, device) -> torch.Tensor:
    """Z eigenvalue (+1 / -1) of qubit q in the computational basis."""
    idx = torch.arange(2**n, device=device)
    bit = (idx >> (n - q - 1)) & 1
    return 1.0 - 2.0 * bit.to(torch.float32)


def rzz(psi, n, u, v, gamma, cache):
    """RZZ(gamma) = exp(-i gamma/2 Z_u Z_v). Diagonal, so implemented as a phase multiply.

    The cache key must include n. The light-cone subcircuits have a qubit count m
    that differs from cone to cone, so keying on (u,v) alone would make sign vectors
    of different lengths collide.
    """
    key = (n, u, v)
    if key not in cache:
        cache[key] = _zsign(n, u, psi.device) * _zsign(n, v, psi.device)
    zz = cache[key]
    phase = torch.exp(-0.5j * (gamma.to(C64) * zz.to(C64)))
    return psi * phase


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
class GraphLocalQuantumGenerator(torch.nn.Module):
    """The graph-local quantum core of v2 §8. Outputs one <Z_u> per qubit."""

    def __init__(self, n_qubits: int, edges: list[tuple[int, int]], depth: int, seed: int = 0):
        super().__init__()
        self.n, self.edges, self.L = n_qubits, list(edges), depth
        g = torch.Generator().manual_seed(seed)
        n, L, E = n_qubits, depth, len(self.edges)

        def p(*shape, scale=1.0):
            return torch.nn.Parameter(scale * (2 * torch.rand(*shape, generator=g) - 1))

        # local encoding: angle = a*z + b*y + c   (separately per axis)
        self.a_y, self.b_y, self.c_y = p(L, n, scale=np.pi), p(L, n), p(L, n)
        self.a_z, self.b_z, self.c_z = p(L, n, scale=np.pi), p(L, n), p(L, n)
        # entangling: small initial values near 0 (v2 §8.4)
        self.gamma = p(L, max(E, 1), scale=0.1)
        # local mixing
        self.tx, self.ty = p(L, n), p(L, n)

        self._zcache: dict = {}
        self._zsign_cache: dict = {}

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """z: (B, n) local latents.  y: (B,) condition.  Returns: (B, n) = <Z_u>."""
        B, n = z.shape
        psi = torch.zeros(B, 2**n, dtype=C64, device=z.device)
        psi[:, 0] = 1.0
        yv = y.reshape(-1)

        for l in range(self.L):
            for u in range(n):  # S_enc : local
                psi = ry(psi, n, u, self.a_y[l, u] * z[:, u] + self.b_y[l, u] * yv + self.c_y[l, u])
                psi = rz(psi, n, u, self.a_z[l, u] * z[:, u] + self.b_z[l, u] * yv + self.c_z[l, u])
            for k, (u, v) in enumerate(self.edges):  # E_G : CDG edges
                psi = rzz(psi, n, u, v, self.gamma[l, k], self._zcache)
            for u in range(n):  # R_mix : non-commuting. Without it gamma never learns.
                psi = rx(psi, n, u, self.tx[l, u])
                psi = ry(psi, n, u, self.ty[l, u])

        probs = (psi.conj() * psi).real  # (B, 2**n)
        for u in range(n):
            if u not in self._zsign_cache:
                self._zsign_cache[u] = _zsign(n, u, z.device)
        signs = torch.stack([self._zsign_cache[u] for u in range(n)], dim=0)  # (n, 2**n)
        return probs @ signs.T  # (B, n)


# --------------------------------------------------------------------------
# Correlation metrics
# --------------------------------------------------------------------------
def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False) + 1e-12)


def normal_score_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation in the nonparanormal (rank -> inverse normal) space.

    This is the space HDE is actually computed in, and it is invariant to a
    monotone head h_u.
    """
    from scipy.stats import norm, rankdata

    N = len(a)
    za = norm.ppf((rankdata(a) - 0.5) / N)
    zb = norm.ppf((rankdata(b) - 0.5) / N)
    return float(np.corrcoef(za, zb)[0, 1])

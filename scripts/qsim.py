"""Graph-local 양자 생성기의 batched statevector 시뮬레이터 (CDG-QGAN v2 §4.2).

회로 블록 (오른쪽이 먼저 작동):
    U^(l) = R_mix^(l) . E_G^(l) . S_enc^(l)

  S_enc : 큐비트별 local encoding.  각도는 해당 큐비트의 local latent z_u와 조건 y만 사용.
  E_G   : CDG 간선에만 RZZ (대각 게이트).
  R_mix : 큐비트별 비가환 RX/RY.  이게 없으면 RZZ가 Z와 가환이라 얽힘 파라미터의
          gradient가 0이 된다 (부록 B-1/B-2에서 수치 확인됨).

큐비트 0 = 최상위 비트.
"""

from __future__ import annotations

import numpy as np
import torch

C64 = torch.complex64


# --------------------------------------------------------------------------
# 게이트
# --------------------------------------------------------------------------
def _apply_1q(psi: torch.Tensor, n: int, q: int, a00, a01, a10, a11) -> torch.Tensor:
    """큐비트 q에 2x2 게이트 [[a00,a01],[a10,a11]] 적용.

    psi: (B, 2**n) complex. 계수 a**는 (B,1) 또는 스칼라로 브로드캐스트 가능해야 함
    (샘플마다 각도가 다른 data encoding을 지원하기 위함).
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
    """계산 기저에서 큐비트 q의 Z 고유값 (+1 / -1)."""
    idx = torch.arange(2**n, device=device)
    bit = (idx >> (n - q - 1)) & 1
    return 1.0 - 2.0 * bit.to(torch.float32)


def rzz(psi, n, u, v, gamma, cache):
    """RZZ(gamma) = exp(-i gamma/2 Z_u Z_v). 대각이므로 위상 곱으로 구현.

    캐시 키에 n을 반드시 포함해야 한다. light-cone 부분회로는 큐비트 수 m이
    cone마다 다르므로, (u,v)만으로 키를 잡으면 크기가 다른 부호 벡터가 충돌한다.
    """
    key = (n, u, v)
    if key not in cache:
        cache[key] = _zsign(n, u, psi.device) * _zsign(n, v, psi.device)
    zz = cache[key]
    phase = torch.exp(-0.5j * (gamma.to(C64) * zz.to(C64)))
    return psi * phase


# --------------------------------------------------------------------------
# 생성기
# --------------------------------------------------------------------------
class GraphLocalQuantumGenerator(torch.nn.Module):
    """v2 §8의 graph-local 양자 코어. 출력은 큐비트별 <Z_u> 하나씩."""

    def __init__(self, n_qubits: int, edges: list[tuple[int, int]], depth: int, seed: int = 0):
        super().__init__()
        self.n, self.edges, self.L = n_qubits, list(edges), depth
        g = torch.Generator().manual_seed(seed)
        n, L, E = n_qubits, depth, len(self.edges)

        def p(*shape, scale=1.0):
            return torch.nn.Parameter(scale * (2 * torch.rand(*shape, generator=g) - 1))

        # local encoding: angle = a*z + b*y + c   (축별로 따로)
        self.a_y, self.b_y, self.c_y = p(L, n, scale=np.pi), p(L, n), p(L, n)
        self.a_z, self.b_z, self.c_z = p(L, n, scale=np.pi), p(L, n), p(L, n)
        # 얽힘: 0 근방 작은 초기값 (v2 §8.4)
        self.gamma = p(L, max(E, 1), scale=0.1)
        # local mixing
        self.tx, self.ty = p(L, n), p(L, n)

        self._zcache: dict = {}
        self._zsign_cache: dict = {}

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """z: (B, n) local latents.  y: (B,) 조건.  반환: (B, n) = <Z_u>."""
        B, n = z.shape
        psi = torch.zeros(B, 2**n, dtype=C64, device=z.device)
        psi[:, 0] = 1.0
        yv = y.reshape(-1)

        for l in range(self.L):
            for u in range(n):  # S_enc : local
                psi = ry(psi, n, u, self.a_y[l, u] * z[:, u] + self.b_y[l, u] * yv + self.c_y[l, u])
                psi = rz(psi, n, u, self.a_z[l, u] * z[:, u] + self.b_z[l, u] * yv + self.c_z[l, u])
            for k, (u, v) in enumerate(self.edges):  # E_G : CDG 간선
                psi = rzz(psi, n, u, v, self.gamma[l, k], self._zcache)
            for u in range(n):  # R_mix : 비가환. 없으면 gamma가 안 배운다.
                psi = rx(psi, n, u, self.tx[l, u])
                psi = ry(psi, n, u, self.ty[l, u])

        probs = (psi.conj() * psi).real  # (B, 2**n)
        for u in range(n):
            if u not in self._zsign_cache:
                self._zsign_cache[u] = _zsign(n, u, z.device)
        signs = torch.stack([self._zsign_cache[u] for u in range(n)], dim=0)  # (n, 2**n)
        return probs @ signs.T  # (B, n)


# --------------------------------------------------------------------------
# 상관 지표
# --------------------------------------------------------------------------
def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False) + 1e-12)


def normal_score_corr(a: np.ndarray, b: np.ndarray) -> float:
    """nonparanormal(rank -> inverse normal) 공간의 상관.

    HDE가 실제로 계산되는 공간이며, 단조 head h_u에 대해 불변이다.
    """
    from scipy.stats import norm, rankdata

    N = len(a)
    za = norm.ppf((rankdata(a) - 0.5) / N)
    zb = norm.ppf((rankdata(b) - 0.5) / N)
    return float(np.corrcoef(za, zb)[0, 1])

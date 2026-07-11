"""Light-cone subcircuit simulator — putting our own theorem to computational use [review D-1].

Proposition 1 (v2 §4.7):
    With a product initial state and local encoding, <Z_u> depends only on the
    qubits within graph radius L of u.

So computing <Z_u> does not require the full 2^16 = 65,536-dimensional statevector.
It suffices to simulate the subcircuit induced by N_L(u) — exactly, not approximately.

    L=1, Delta<=3  ->  |N_1(u)| <= 4   ->  16 dimensions
    L=2, Delta<=3  ->  |N_2(u)| <= 10  ->  1,024 dimensions

Memory drops by a factor of thousands and training gets far faster. The full
statevector is only needed for finite-shot bitstring sampling (inference).

`verify_against_full()` guarantees exactness by checking against the full simulator.
"""

from __future__ import annotations

import networkx as nx
import torch

from qsim import C64, _zsign


def neighborhood(edges, n: int, u: int, L: int) -> list[int]:
    """N_L(u) — the nodes within graph distance L of u (sorted)."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return sorted(nx.single_source_shortest_path_length(G, u, cutoff=L).keys())


class LightconeGenerator(torch.nn.Module):
    """Simulates only the per-qubit light-cone subcircuits instead of the full statevector.

    The output is numerically identical to the full simulator (not an approximation).
    Parameters are shared with the full model, so it is a drop-in replacement.
    """

    def __init__(self, full_gen, edges, n: int, depth: int):
        super().__init__()
        self.g = full_gen  # parameter owner (GraphLocalQuantumGenerator)
        self.n, self.L = n, depth
        self.edges = list(edges)
        edge_idx = {tuple(sorted(e)): k for k, e in enumerate(self.edges)}

        # For each output qubit u: precompute the nodes and edges of its subcircuit
        self.cones, self.local_edges, self.local_idx = [], [], []
        cone_edge_idx = []
        for u in range(n):
            S = neighborhood(self.edges, n, u, depth)
            idx = {g: l for l, g in enumerate(S)}  # global -> local
            self.cones.append(S)
            self.local_edges.append([(idx[a], idx[b]) for a, b in self.edges
                                     if a in idx and b in idx])
            cone_edge_idx.append([edge_idx[tuple(sorted((a, b)))] for a, b in self.edges
                                  if a in idx and b in idx])
            self.local_idx.append(idx[u])

        # All cones are evolved in one (sample, cone, amplitude) tensor.  Shorter
        # cones use trailing padding qubits whose gates are identities.
        self.max_qubits = max(map(len, self.cones))
        self.max_edges = max(map(len, self.local_edges), default=0)
        nodes = torch.zeros(n, self.max_qubits, dtype=torch.long)
        node_valid = torch.zeros(n, self.max_qubits, dtype=torch.bool)
        edge_param = torch.zeros(n, self.max_edges, dtype=torch.long)
        edge_valid = torch.zeros(n, self.max_edges, dtype=torch.bool)
        zz = torch.zeros(n, self.max_edges, 2**self.max_qubits)
        signs = torch.empty(n, 2**self.max_qubits)
        for u, (S, le, ei) in enumerate(zip(self.cones, self.local_edges, cone_edge_idx)):
            nodes[u, :len(S)] = torch.tensor(S)
            node_valid[u, :len(S)] = True
            if ei:
                edge_param[u, :len(ei)] = torch.tensor(ei)
                edge_valid[u, :len(ei)] = True
            for j, (a, b) in enumerate(le):
                zz[u, j] = _zsign(self.max_qubits, a, "cpu") * _zsign(
                    self.max_qubits, b, "cpu")
            signs[u] = _zsign(self.max_qubits, self.local_idx[u], "cpu")
        self.register_buffer("node_indices", nodes, persistent=False)
        self.register_buffer("node_valid", node_valid, persistent=False)
        self.register_buffer("edge_param_indices", edge_param, persistent=False)
        self.register_buffer("edge_valid", edge_valid, persistent=False)
        self.register_buffer("zz_signs", zz, persistent=False)
        self.register_buffer("observable_signs", signs, persistent=False)

    @staticmethod
    def _apply_1q(psi: torch.Tensor, q: int, a00, a01, a10, a11) -> torch.Tensor:
        """Apply one gate position to every cone; coefficients are (B, cones)."""
        B, C, A = psi.shape
        left, right = 2**q, A // (2 ** (q + 1))
        v = psi.reshape(B, C, left, 2, right)
        v0, v1 = v[:, :, :, 0, :], v[:, :, :, 1, :]

        def bc(a):
            return a[:, :, None, None] if torch.is_tensor(a) and a.ndim == 2 else a

        out = torch.stack([bc(a00) * v0 + bc(a01) * v1,
                           bc(a10) * v0 + bc(a11) * v1], dim=3)
        return out.reshape(B, C, A)

    def _ry(self, psi, q, theta):
        c, s = torch.cos(theta / 2).to(C64), torch.sin(theta / 2).to(C64)
        return self._apply_1q(psi, q, c, -s, s, c)

    def _rx(self, psi, q, theta):
        c, s = torch.cos(theta / 2).to(C64), torch.sin(theta / 2).to(C64)
        return self._apply_1q(psi, q, c, -1j * s, -1j * s, c)

    def _rz(self, psi, q, theta):
        e = torch.exp(-0.5j * theta.to(C64))
        return self._apply_1q(psi, q, e, 0.0, 0.0, e.conj())

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        psi = torch.zeros(B, self.n, 2**self.max_qubits, dtype=C64, device=z.device)
        psi[:, :, 0] = 1.0
        y = y.reshape(B, 1)
        safe_nodes = self.node_indices

        for l in range(self.L):
            for q in range(self.max_qubits):
                idx = safe_nodes[:, q]
                valid = self.node_valid[:, q].to(z.dtype)[None, :]
                zq = z[:, idx]
                ty = (self.g.a_y[l, idx][None, :] * zq
                      + self.g.b_y[l, idx][None, :] * y + self.g.c_y[l, idx][None, :]) * valid
                tz = (self.g.a_z[l, idx][None, :] * zq
                      + self.g.b_z[l, idx][None, :] * y + self.g.c_z[l, idx][None, :]) * valid
                psi = self._ry(psi, q, ty)
                psi = self._rz(psi, q, tz)

            if self.max_edges:
                gamma = self.g.gamma[l, self.edge_param_indices] * self.edge_valid
                total = (gamma[:, :, None] * self.zz_signs).sum(dim=1)
                psi = psi * torch.exp(-0.5j * total.to(C64))[None, :, :]

            for q in range(self.max_qubits):
                idx = safe_nodes[:, q]
                valid = self.node_valid[:, q].to(z.dtype)[None, :]
                psi = self._rx(psi, q, self.g.tx[l, idx][None, :] * valid)
                psi = self._ry(psi, q, self.g.ty[l, idx][None, :] * valid)

        probs = (psi.conj() * psi).real
        return (probs * self.observable_signs[None, :, :]).sum(dim=2)


def verify_against_full(n=8, depth=1, seed=0, tol=1e-4) -> None:
    """Verify that the subcircuit result agrees with the full statevector."""
    import numpy as np

    from qsim import GraphLocalQuantumGenerator

    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(3, n, seed=seed)
    edges = [tuple(sorted(e)) for e in G.edges()]

    full = GraphLocalQuantumGenerator(n, edges, depth, seed=seed)
    cone = LightconeGenerator(full, edges, n, depth)

    z = torch.tensor(2 * rng.random((64, n)) - 1, dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 2, 64), dtype=torch.float32)

    with torch.no_grad():
        qf, qc = full(z, y), cone(z, y)
    err = (qf - qc).abs().max().item()

    sizes = [len(c) for c in cone.cones]
    print(f"  n={n} L={depth} Delta=3")
    print(f"    full statevector : 2^{n} = {2**n:,} dimensions")
    print(f"    light-cone part  : at most 2^{max(sizes)} = {2**max(sizes):,} dimensions  "
          f"(cone sizes {min(sizes)}~{max(sizes)})")
    print(f"    speedup factor   : {2**n / 2**max(sizes):.0f}x")
    print(f"    max <Z_u> error  : {err:.2e}   {'PASS' if err < tol else 'FAIL'}")
    assert err < tol, f"subcircuit disagrees with the full simulator! err={err}"


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 66)
    print("Light-cone subcircuit simulator verification [review D-1]")
    print("=" * 66)
    for n, L in [(8, 1), (10, 1), (12, 1), (16, 1), (12, 2)]:
        verify_against_full(n=n, depth=L)
        print()
    print("  -> Exploiting Proposition 1 computationally shrinks the statevector")
    print("     dimension by a factor of thousands while keeping <Z_u> exact.")

"""Regression tests for the vectorized light-cone simulator."""

from pathlib import Path
import sys
import unittest

import networkx as nx
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from qsim import C64, GraphLocalQuantumGenerator, _zsign, rx, ry, rz, rzz  # noqa: E402
from qsim_lightcone import LightconeGenerator, neighborhood  # noqa: E402


def legacy_forward(g, edges, n, depth, z, y):
    """The pre-WP6 implementation, retained here as an independent oracle."""
    edges = list(edges)
    cache = {}
    out = []
    for u in range(n):
        cone = neighborhood(edges, n, u, depth)
        index = {global_q: local_q for local_q, global_q in enumerate(cone)}
        local_edges = [(index[a], index[b]) for a, b in edges if a in index and b in index]
        psi = torch.zeros(z.shape[0], 2**len(cone), dtype=C64, device=z.device)
        psi[:, 0] = 1.0
        for layer in range(depth):
            for local_q, global_q in enumerate(cone):
                psi = ry(psi, len(cone), local_q,
                         g.a_y[layer, global_q] * z[:, global_q]
                         + g.b_y[layer, global_q] * y + g.c_y[layer, global_q])
                psi = rz(psi, len(cone), local_q,
                         g.a_z[layer, global_q] * z[:, global_q]
                         + g.b_z[layer, global_q] * y + g.c_z[layer, global_q])
            for a, b in local_edges:
                edge = (cone[a], cone[b])
                k = edges.index(edge) if edge in edges else edges.index(edge[::-1])
                psi = rzz(psi, len(cone), a, b, g.gamma[layer, k], cache)
            for local_q, global_q in enumerate(cone):
                psi = rx(psi, len(cone), local_q, g.tx[layer, global_q])
                psi = ry(psi, len(cone), local_q, g.ty[layer, global_q])
        probs = (psi.conj() * psi).real
        out.append(probs @ _zsign(len(cone), index[u], z.device))
    return torch.stack(out, dim=1)


class LightconeRegressionTest(unittest.TestCase):
    def test_vectorized_matches_legacy(self):
        for n, depth in ((10, 1), (12, 2)):
            graph = nx.random_regular_graph(3, n, seed=17)
            edges = [tuple(sorted(edge)) for edge in graph.edges()]
            generator = GraphLocalQuantumGenerator(n, edges, depth, seed=23)
            vectorized = LightconeGenerator(generator, edges, n, depth)
            torch.manual_seed(31)
            z = 2 * torch.rand(7, n) - 1
            y = torch.randint(0, 2, (7,), dtype=torch.float32)
            expected = legacy_forward(generator, edges, n, depth, z, y)
            actual = vectorized(z, y)
            error = (actual - expected).abs().max().item()
            self.assertLess(error, 1e-5, f"n={n}, depth={depth}, max error={error}")


if __name__ == "__main__":
    unittest.main()

"""Independent verification of WP-6 (PM).

What Codex reported:
  - simulator accuracy 1.6e-6, regression tests pass
  - 30.43 s/seed, 59x speedup

What Codex did **not** verify — caught here:
  A CUDA Graph was introduced into the training loop, but it was never checked that the
  graph path produces the same results as the ordinary path. The CUDA Graph is a textbook
  example of an optimization that silently produces wrong results.

Checks
  [1] Does the RNG advance between graph replays?
      The alpha of the gradient penalty is generated **inside** the graph. If every replay
      yields the same value, every critic step uses the same interpolation point ->
      the gradient penalty is neutralized.
      Training would "look like it runs" but would no longer be WGAN-GP. A silent failure.
  [2] Do the graph path and the ordinary path yield the same training result at the same seed?
  [3] Does the speed actually hit the target (60s/seed)?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from train import DEVICE, Cfg, dependency_error, generate, train  # noqa: E402

N, P = 20000, 16


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    import networkx as nx
    G = nx.random_regular_graph(3, P, seed=seed)
    edges = [tuple(sorted(e)) for e in G.edges()]

    Om = np.eye(P)
    for u, v in G.edges():
        Om[u, v] = Om[v, u] = rng.uniform(0.3, 0.5) * rng.choice([-1, 1])
    Om += np.eye(P) * (abs(np.linalg.eigvalsh(Om).min()) + 0.15)
    S = np.linalg.inv(Om)
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    X = rng.multivariate_normal(np.zeros(P), S, size=N)
    X = (X - X.mean(0)) / X.std(0)
    y = (rng.random(N) < 0.1).astype(float)
    C = np.column_stack([y, rng.normal(0, 1, N), rng.random(N) < 0.5,
                         rng.integers(0, 3, N)]).astype(float)
    return X, C, edges


# ---------------------------------------------------------------------------
def test_rng_in_graph() -> bool:
    """[1] Does torch.rand yield different values between CUDA Graph replays?"""
    print("=" * 74)
    print("[1] Does the RNG advance inside the CUDA Graph?")
    print("=" * 74)
    if DEVICE != "cuda":
        print("  No CUDA — skipped")
        return True

    out = torch.empty(8, device="cuda")

    def body():
        out.copy_(torch.rand(8, device="cuda"))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()

    draws = []
    for _ in range(4):
        g.replay()
        torch.cuda.synchronize()
        draws.append(out.clone().cpu().numpy())

    same = [np.allclose(draws[0], d) for d in draws[1:]]
    for i, d in enumerate(draws):
        print(f"  replay {i}: {d[:4].round(4)}")
    ok = not any(same)
    print(f"\n  different random numbers on each replay: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  !! FAIL — the alpha of gradient_penalty is the same every time.")
        print("     Every critic step ends up using the same interpolation point, neutralizing WGAN-GP.")
    return ok


# ---------------------------------------------------------------------------
def test_graph_vs_eager(steps=300) -> bool:
    """[2] Do the graph path and the ordinary path yield the same training result?"""
    print()
    print("=" * 74)
    print(f"[2] CUDA Graph path vs ordinary path  (same seed, {steps} steps)")
    print("=" * 74)
    X, C, edges = make_data()

    res = {}
    for use_graph in (True, False):
        t0 = time.time()
        G, _ = train(X, C, edges, Cfg(depth=1, steps=steps, seed=0, use_cuda_graph=use_graph))
        Xs = generate(G, C, 20000, seed=0)
        res[use_graph] = {
            "err": dependency_error(X, Xs),
            "mean": Xs.mean(0),
            "std": Xs.std(0),
            "t": time.time() - t0,
        }
        tag = "CUDA Graph" if use_graph else "ordinary(eager)"
        print(f"  {tag:<14} dep_error={res[use_graph]['err']:.4f}  ({res[use_graph]['t']:.0f}s)")

    a, b = res[True], res[False]
    d_err = abs(a["err"] - b["err"])
    d_mean = np.abs(a["mean"] - b["mean"]).max()
    d_std = np.abs(a["std"] - b["std"]).max()
    print()
    print(f"  |dep_error difference|      = {d_err:.4f}")
    print(f"  |feature mean difference|max = {d_mean:.4f}")
    print(f"  |feature std difference|max  = {d_std:.4f}")
    print()
    print("  Note: the two paths consume RNG in a different order, so bit-exact agreement is not expected.")
    print("        What we look at is whether training converges to a statistically equivalent place.")
    ok = d_err < 0.05 and d_mean < 0.15 and d_std < 0.15
    print(f"  Verdict: {'PASS — the two paths are statistically equivalent' if ok else 'FAIL — the result depends on the path'}")
    return ok


# ---------------------------------------------------------------------------
def test_speed(steps=3000) -> bool:
    """[3] Acceptance criterion: within 60 seconds per seed."""
    print()
    print("=" * 74)
    print(f"[3] Speed  ({steps} steps, n=16, L=1, batch=256, critic=5)")
    print("=" * 74)
    X, C, edges = make_data()
    t0 = time.time()
    train(X, C, edges, Cfg(depth=1, steps=steps, seed=0))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    print(f"  wall-clock: {el:.1f}s  (target <60s)")
    ok = el < 60
    print(f"  Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = test_rng_in_graph()
    r2 = test_graph_vs_eager()
    r3 = test_speed()
    print()
    print("=" * 74)
    print(f"  [1] RNG advances inside graph    : {'PASS' if r1 else 'FAIL'}")
    print(f"  [2] graph vs ordinary equivalent : {'PASS' if r2 else 'FAIL'}")
    print(f"  [3] speed <60s                   : {'PASS' if r3 else 'FAIL'}")
    print(f"\n  WP-6 final: {'APPROVED' if (r1 and r2 and r3) else 'REJECTED'}")
    print("=" * 74)

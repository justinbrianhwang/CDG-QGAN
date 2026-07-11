"""Gate experiment: the correlation-expressivity ceiling of a shallow graph-local quantum
generator.

The question:
    In a depth-L circuit, what is the largest conditional dependency that can be created
    between two features at graph distance d?

Why it matters:
    Since x~_u = h_u(q_u, c) and h_u is a monotone one-dimensional map, the correlation in
    the nonparanormal space — the space in which the HDE is computed — is determined solely
    by the copula of q, independently of the head. The ceiling is therefore purely a property
    of the circuit.

    Real clinical partial correlations include pairs with |rho| > 0.9 (Hb-Hct, SBP-MAP-DBP).
    If the ceiling is below that, both the CDG and the permuted CDG saturate against it and
    the confirmatory contrast is washed out. -> the design must change.

Side benefit:
    If the correlation comes out as 0 for d > 2L, the corollary of v2 §4.8 is verified
    numerically.
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
EVAL_BATCH = 16384  # no_grad -> no intermediate tensors, so this can be large
MEM_BUDGET = 6e9    # budget for the autograd intermediate tensors (bytes)


def train_batch(n: int, depth: int) -> int:
    """Choose the training batch so the autograd intermediate tensors fit in the memory budget.

    Number of intermediate tensors ~ 5*L*n, each B * 2^n * 8 bytes (complex64).
    At n=13, L=3, a batch of 4096 comes to ~50GB, which blows past 32GB of VRAM.
    """
    per = 5 * depth * n * (2**n) * 8
    return int(np.clip(MEM_BUDGET / per, 256, 8192))


def max_corr(depth: int, dist: int, seed_base: int = 0) -> tuple[float, float, int]:
    """The maximum achievable correlation for a pair at distance dist on a path graph.

    Returns: (|correlation| in the nonparanormal space, Pearson |correlation|, training batch)
    """
    # pad on both sides of u by the light cone radius L. On a path graph, distance = index
    # difference.
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
        y = torch.zeros(B, device=DEVICE)             # condition held fixed (conditional on c)

        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            q = gen(z, y)
            loss = -pearson(q[:, u], q[:, v]).abs()   # differentiable surrogate objective
            loss.backward()
            opt.step()

        # evaluate under no_grad on a large batch -> reduces the estimation error
        with torch.no_grad():
            ze = 2 * torch.rand(EVAL_BATCH, n, device=DEVICE) - 1
            ye = torch.zeros(EVAL_BATCH, device=DEVICE)
            qe = gen(ze, ye).cpu().numpy()
        ns = abs(normal_score_corr(qe[:, u], qe[:, v]))  # report in the space where the HDE lives
        pe = abs(float(np.corrcoef(qe[:, u], qe[:, v])[0, 1]))
        best_ns, best_p = max(best_ns, ns), max(best_p, pe)
        del gen, z, y
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    return best_ns, best_p, B


def main() -> None:
    print("=" * 74)
    print("Circuit correlation-expressivity ceiling  (graph-local RZZ generator, path graph)")
    print(f"device={DEVICE}  steps={STEPS}  restarts={RESTARTS}  eval_batch={EVAL_BATCH}")
    print("=" * 74)
    print()
    print("  value = maximum achievable |correlation| (nonparanormal space = the space in which")
    print("          the HDE is computed)")
    print("  shaded (--) = the region that should be 0 in theory (d > 2L)")
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
    print("  Summary")
    print(header)
    for r in rows:
        print(r)
    print()
    print("  * = d > 2L, the cells that must be 0 by Corollary 1")
    print()

    # ---- Verify the corollary ----
    print("=" * 74)
    print("Check 1: the light-cone corollary (d > 2L  =>  conditional correlation = 0)")
    print("=" * 74)
    inside = [(k, v[0]) for k, v in results.items() if k[1] <= 2 * k[0]]
    outside = [(k, v[0]) for k, v in results.items() if k[1] > 2 * k[0]]
    max_out = max((v for _, v in outside), default=0.0)
    min_in = min((v for _, v in inside), default=0.0)
    print(f"  inside the light cone  (d <= 2L): min |corr| = {min_in:.4f}   ({len(inside)} cells)")
    print(f"  outside the light cone (d >  2L): max |corr| = {max_out:.4f}   ({len(outside)} cells)")
    ok = max_out < 0.02
    print(f"  -> Corollary 1 {'PASS' if ok else 'FAIL'}: no correlation can be created outside the cone")
    print()

    # ---- Gate verdict ----
    print("=" * 74)
    print("Check 2: the gate — do the real clinical partial correlations fit under the ceiling?")
    print("=" * 74)
    targets = {
        "Hemoglobin-Hematocrit": 0.95,
        "SBP-MAP":               0.90,
        "Creatinine-BUN":        0.60,
        "Na-Cl":                 0.55,
        "many weak clinical relations": 0.20,
    }
    print()
    for L in depths:
        c1 = results[(L, 1)][0]  # adjacent pair = the best case
        print(f"  L={L}, ceiling for an adjacent pair (d=1) = {c1:.3f}")
        for name, tgt in targets.items():
            ok_t = c1 >= tgt
            print(f"      {'OK  ' if ok_t else 'FAIL'}  {name:<24} |rho|={tgt:.2f}")
        print()

    np.save("results_ceiling.npy", results, allow_pickle=True)
    print("  -> saved results_ceiling.npy")


if __name__ == "__main__":
    main()

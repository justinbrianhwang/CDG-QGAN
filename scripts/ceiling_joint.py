"""Joint expressivity ceiling — can the full 120-pair pattern be matched simultaneously?

Why this is needed
------------------
`ceiling.py` maximized the |correlation| of a **single pair** (L=1, d=1 -> 0.991).
That only means "a strong dependency can be placed on one edge"; it does **not** mean
that **19 edges can be brought to rho=0.35 simultaneously while holding the 101
non-edges at 0**. The latter is what the confirmatory experiment requires. That box was
left empty.

What `diag_trained.py` showed:
    true-edge error of the WGAN-GP-trained model = 0.3662
    error of a model that creates no dependency at all = 0.3676
    -> the trained model creates **almost zero** dependency on the true edges.

It is one of two things:
    (a) the circuit cannot represent that pattern in the first place -> design flaw,
        WP-2 is pointless
    (b) the circuit can represent it but WGAN-GP fails to find it -> training flaw, the
        objective must be fixed

Here we remove the GAN and gradient-descend the circuit parameters **directly** on the
partial-correlation error. The head is a monotone 1-D map, so dependency in the
nonparanormal space is determined solely by the copula of q=<Z> (the same logic as in
ceiling.py). Measuring on q is therefore sufficient.

How to read this
----------------
    aligned low and permuted high -> (b). The circuit/CDG hypothesis survives; fix the
                                     training.
    both high                     -> (a). The design cannot produce that pattern.
                                     Redesign.
    both low                      -> CDG alignment is irrelevant to expressivity. The
                                     rationale for the confirmatory experiment collapses.
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
N_EVAL = 65536   # held-out draw used to score parameters (see fit())
EVAL_EVERY = 50


def partial_corr_torch(q: torch.Tensor) -> torch.Tensor:
    """Differentiable partial correlation (q: (B, n))."""
    x = q - q.mean(0, keepdim=True)
    x = x / (x.std(0, keepdim=True) + 1e-6)
    S = (x.T @ x) / (x.shape[0] - 1)
    P = torch.linalg.inv(S + RIDGE * torch.eye(S.shape[0], device=S.device))
    d = torch.sqrt(torch.diag(P))
    R = -P / torch.outer(d, d)
    return R - torch.diag(torch.diag(R)) + torch.eye(S.shape[0], device=S.device)


def fit(edges, target: torch.Tensor, C: np.ndarray, seed: int) -> float:
    """Fit the circuit parameters directly to the target partial correlation.

    Returns: mean absolute error over the 120 pairs, scored on a **held-out draw**.

    Scoring the training minibatch would be wrong here. Each step draws a fresh z and a
    fresh row sample, so `min` over the trajectory would return the luckiest minibatch,
    not the loss of the best parameters. With BATCH=4096 the per-entry sampling noise of
    a partial correlation is ~1/sqrt(4096) = 0.016, and taking a min over 1500 steps and
    then again over RESTARTS compounds the optimistic bias. Since `floor` is computed
    analytically (noise-free), that bias would not cancel between the two sides of the
    comparison — it would manufacture a ceiling that is better than the circuit can
    actually reach.

    So: score periodically on a fixed held-out draw (same z, same rows for every variant
    and every restart), under no_grad, and return the best held-out score.
    """
    torch.manual_seed(seed)
    m = CDGQGAN(N_FEAT, list(edges), depth=1, cond_dim=C.shape[1], seed=seed).to(DEVICE)
    core = m.core
    params = list(m.quantum.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)
    iu = torch.triu_indices(N_FEAT, N_FEAT, offset=1)

    # Fixed held-out draw — identical across variants and restarts, so the comparison is
    # not decided by which model happened to see a friendlier evaluation sample.
    g = torch.Generator(device=DEVICE).manual_seed(12345)
    z_eval = 2 * torch.rand(N_EVAL, N_FEAT, device=DEVICE, generator=g) - 1
    y_eval = Ct[torch.randint(0, len(Ct), (N_EVAL,), device=DEVICE, generator=g), 0]

    @torch.no_grad()
    def held_out_error() -> float:
        R = partial_corr_torch(core(z_eval, y_eval))
        return (R[iu[0], iu[1]] - target[iu[0], iu[1]]).abs().mean().item()

    best = held_out_error()
    for step in range(STEPS):
        idx = torch.randint(0, len(Ct), (BATCH,), device=DEVICE)
        z = 2 * torch.rand(BATCH, N_FEAT, device=DEVICE) - 1
        q = core(z, Ct[idx, 0])
        R = partial_corr_torch(q)
        loss = (R[iu[0], iu[1]] - target[iu[0], iu[1]]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % EVAL_EVERY == 0 or step == STEPS - 1:
            best = min(best, held_out_error())
    return best


def main() -> None:
    rng = np.random.default_rng(20260711)
    G = teacher_graph(rng)
    X, C, _ = teacher_data(G, N_TRAIN, rng)

    # The teacher's true partial correlation (nonparanormal space) — this is the target to match
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

    # floor: the 120-pair error of a zero-dependency model (against the same target).
    # Scored with the SAME estimator on the SAME held-out sample size as the variants, so
    # that the finite-sample noise of the partial-correlation estimate does not sit on only
    # one side of the comparison. (Scoring it analytically as mean|R*| would give a
    # noise-free floor and an optimistically-scored ceiling.)
    iu_t = torch.triu_indices(N_FEAT, N_FEAT, offset=1)
    gf = torch.Generator(device=DEVICE).manual_seed(999)
    fl = []
    for _ in range(5):
        q_indep = torch.randn(N_EVAL, N_FEAT, device=DEVICE, generator=gf)
        R = partial_corr_torch(q_indep)
        fl.append((R[iu_t[0], iu_t[1]] - target[iu_t[0], iu_t[1]]).abs().mean().item())
    floor = float(np.mean(fl))

    print("=" * 78)
    print("Joint expressivity ceiling — the full 120-pair pattern (direct optimization, no GAN)")
    print("=" * 78)
    print(f"  teacher: 19 edges, mean edge |rho| {np.abs([Rstar[i,j] for i,j in G.edges()]).mean():.3f}")
    print(f"  optimization: Adam lr=0.05, {STEPS} steps, batch={BATCH}, {RESTARTS} restarts")
    print(f"  [floor] 120-pair error of a zero-dependency model = {floor:.4f}")
    print()
    print(f"  {'model':<14} {'min 120-pair err':>16}   {'vs. floor':>12}")
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
        print(f"  aligned - {ref:<12} = {dlt:+.4f}   {'aligned better' if dlt < 0 else 'aligned worse'}")
    print()
    if a >= floor * 0.95:
        print("  >> The circuit cannot represent the teacher pattern. This is a design flaw (case a).")
    elif a < res["permuted"] - 0.005:
        print("  >> The circuit can represent it and aligned is better (case b).")
        print("     -> The CDG hypothesis survives. The problem is that WGAN-GP fails to find that solution.")
    else:
        print("  >> It is representable, but aligned cannot beat permuted.")
        print("     -> The premise of the confirmatory experiment collapses. This must be resolved first.")


if __name__ == "__main__":
    main()

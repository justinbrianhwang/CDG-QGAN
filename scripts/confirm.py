"""The confirmatory experiment, with everything fixed.

What changed since the null result
----------------------------------
1. **The critic.**  WGAN-GP learned no dependency at all, for any topology (RESULTS_lr.md).
   The critic was spending its capacity on marginals — which the ~2000 head parameters can fit
   unaided — and handing `gamma` nothing but noise. Two changes, neither of which touches the
   evaluation metric:
     (E) COPULA CRITIC. Rank-transform each feature within the batch before the critic sees it
         (differentiable soft rank -> inverse normal CDF). Every input feature is then standard
         normal by construction, so the marginals carry no information and the only thing left
         to discriminate on is the dependency structure. The gradient has nowhere to go but the
         entangling angles.
     (A) BATCH-AWARE CRITIC. Minibatch discrimination (Salimans et al. 2016). A per-sample
         critic cannot estimate a 16-dimensional joint from single rows; this lets it compare a
         *batch's* structure against a real batch's.
   Screening: true-edge error 0.4029 (baseline) -> 0.2663 (E) -> 0.1892 (E)+(A), against a
   floor of 0.3676. The model finally learns dependency.

2. **The teacher.**  The old teacher drew c independently of x, which is not what MIMIC looks
   like and which made the estimator mismatch invisible. Now c shifts the mean of every
   feature, and the graph determines the dependency structure *conditional on c* — exactly the
   situation the CDG is defined for.

3. **The metric.**  `eval_dep.partial_corr_c`: nonparanormal, then residualize on a nonlinear
   basis of c, then partial correlation. This is the estimator the CDG is *defined* by. The old
   metric was unconditional, so everything c induced was scored as a false positive on every
   pair (review E-3).

The loss is still pure WGAN-GP. No dependency term, nothing from the evaluation metric.
v2 §8.10 stands.

Primary endpoint: mean absolute Fisher-z error over all 120 pairs, conditional on c.
Reported alongside: the 19 true edges (false negatives) and the 101 non-edges (false positives).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_SYN, N_TRAIN, teacher_data_cond, teacher_graph  # noqa: E402
from diag_fix2 import Cfg, train_fix2  # noqa: E402
from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from graphs import degree_preserving_rewire, distance_matched_permuted, isomorphic_permuted  # noqa: E402
from train import DEVICE  # noqa: E402

import torch  # noqa: E402

SEEDS = [0, 1, 2]
STEPS = 3000


@torch.no_grad()
def gen(G, C, n_samples, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n_samples)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n_samples, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy(), C[idx]


def main() -> None:
    rng = np.random.default_rng(20260711)
    Gt = teacher_graph(rng)
    X, C, _ = teacher_data_cond(Gt, N_TRAIN, rng)

    ALL = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
    EDGES = [tuple(sorted(e)) for e in Gt.edges()]
    NON = [p for p in ALL if p not in EDGES]

    zr = fisher_z(partial_corr_c(X, C))

    def score(Xs, Cs):
        zs = fisher_z(partial_corr_c(Xs, Cs))
        return (dep_error(zr, zs, ALL), dep_error(zr, zs, EDGES), dep_error(zr, zs, NON))

    # floor: a model that reproduces the marginals exactly and creates zero dependency
    fl = []
    for s in range(3):
        r = np.random.default_rng(100 + s)
        Xi = np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)])
        fl.append(score(Xi, C[r.permutation(len(C))]))
    fl = np.mean(fl, 0)

    gr = np.random.default_rng(7)
    holdout = [(u, v) for u, v in NON if nx.shortest_path_length(Gt, u, v) == 2][:10]
    variants = {
        "aligned (true CDG)": Gt,
        "permuted (isomorphic)": isomorphic_permuted(Gt, gr),
        "distmatched": distance_matched_permuted(Gt, holdout, gr)[0],
        "rewired": degree_preserving_rewire(Gt, gr),
        "no_entangle": nx.empty_graph(N_FEAT),
    }

    print("=" * 100)
    print("Confirmatory experiment — copula + batch-aware critic, realistic teacher, "
          "c-conditional metric")
    print("=" * 100)
    print(f"  teacher: {N_FEAT} nodes · {Gt.number_of_edges()} edges · c drives the mean of "
          f"every feature")
    print(f"  metric : partial correlation CONDITIONAL ON c, nonparanormal space (eval_dep)")
    print(f"  loss   : pure WGAN-GP. No dependency term. v2 §8.10 respected.")
    print(f"  L=1 · {len(SEEDS)} seeds · {STEPS} steps")
    print()
    print(f"  {'model':<24} {'|E|':>4} {'120 pairs':>11} {'19 edges':>10} {'101 non-edges':>14}")
    print("  " + "-" * 70)
    print(f"  {'floor (zero dep.)':<24} {'—':>4} {fl[0]:>11.4f} {fl[1]:>10.4f} {fl[2]:>14.4f}")
    print()

    res = {}
    for name, Gv in variants.items():
        rows, t0 = [], time.time()
        for s in SEEDS:
            cfg = Cfg(steps=STEPS, seed=s, copula=True, batch_critic=True, lr_q=5e-3)
            Gm = train_fix2(X, C, list(Gv.edges()), cfg)
            Xs, Cs = gen(Gm, C, N_SYN, s)
            rows.append(score(Xs, Cs))
        m, sd = np.mean(rows, 0), np.std(rows, 0)
        res[name] = (m, sd)
        print(f"  {name:<24} {Gv.number_of_edges():>4} {m[0]:>11.4f} {m[1]:>10.4f} "
              f"{m[2]:>14.4f}   ±{sd[0]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 100)
    a = res["aligned (true CDG)"][0]
    for ref in list(variants)[1:]:
        d = a[0] - res[ref][0][0]
        print(f"  aligned − {ref:<24} = {d:+.4f}   "
              f"{'aligned better' if d < 0 else 'ALIGNED WORSE'}")
    print()
    print("  Hypothesis: aligned < every control on the 120-pair conditional dependency error.")
    print("  aligned <= distmatched means the effect survives distance matching, i.e. it is not")
    print("  graph combinatorics but the specific clinical pairs the CDG chose.")

    np.savez("results_confirm.npz", **{k: v[0] for k, v in res.items()}, floor=fl)


if __name__ == "__main__":
    main()

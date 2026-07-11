"""WP-2: the confirmatory experiment on real MIMIC-IV.

The design is `confirm.py`, ported to real data. Read `RESULTS_confirm.md` first — it is the
same experiment run against a synthetic teacher whose true graph is known, and it works there.

What is different on real data
------------------------------
On the synthetic teacher we knew the true graph, so we could report the error on the *true*
edges and watch only the aligned circuit learn them. Here **the CDG is our estimate**, not
ground truth. There is no privileged "true edge" column. The primary endpoint is therefore the
conditional dependency error over **all 120 pairs**, which catches both failure modes at once:
false negatives (a real dependency not produced) and false positives (a dependency manufactured
where none exists).

Required configuration (HANDOFF §2.1 — do not deviate)
------------------------------------------------------
  critic   : copula + batch-aware. A plain MLP critic learns ZERO dependency for every
             topology, and then every topology scores the same and this experiment measures
             nothing. That is not a hypothesis; it is measured (`RESULTS_lr.md`).
  loss     : pure WGAN-GP. No dependency term. v2 §8.10 respected.
  metric   : eval_dep.partial_corr_c — conditional on c, nonparanormal. The space the CDG is
             *defined* in (build_cdg uses the same transform).
  floor    : always reported. It is the score of a model that creates no dependency at all.
             Without it you cannot tell a working model from a broken one.

Variants — all identical in resources (edge count, depth, parameters, critic, loss, steps).
The only difference is which clinical pair sits under which RZZ gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from diag_fix2 import Cfg, train_fix2  # noqa: E402
from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import degree_preserving_rewire, distance_matched_permuted, isomorphic_permuted, ring_with_chords  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]


@torch.no_grad()
def generate(G, C: np.ndarray, n: int, seed: int):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy(), C[idx]


def build_variants(G_cdg: nx.Graph, E_hold, rng: np.random.Generator) -> dict:
    """Every variant gets the same |E| and the same max degree. Only the labelling changes."""
    v = {"cdg": G_cdg}
    for k in range(3):
        v[f"permuted_{k}"] = isomorphic_permuted(G_cdg, rng)
    for k in range(3):
        v[f"distmatched_{k}"] = distance_matched_permuted(G_cdg, E_hold, rng)[0]
    v["rewired"] = degree_preserving_rewire(G_cdg, rng)
    v["ring"] = ring_with_chords(N_FEAT, G_cdg.number_of_edges(), rng)
    v["no_entangle"] = nx.empty_graph(N_FEAT)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=1, help="L. 1 = confirmatory; 2 = negative control")
    ap.add_argument("--n-syn", type=int, default=20000)
    args = ap.parse_args()

    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    z = np.load(PROCESSED / "cdg.npz", allow_pickle=True)
    G_cdg = nx.Graph()
    G_cdg.add_nodes_from(range(N_FEAT))
    G_cdg.add_edges_from([tuple(e) for e in z["E_fit"]])
    E_hold = [tuple(e) for e in z["E_holdout"]]

    zr = fisher_z(partial_corr_c(X, C))   # the real conditional dependency structure

    def score(Xs, Cs) -> float:
        return dep_error(zr, fisher_z(partial_corr_c(Xs, Cs)), ALL_PAIRS)

    # floor — a model with perfect marginals and zero dependency. Report it or report nothing.
    fl = []
    for s in range(5):
        r = np.random.default_rng(1000 + s)
        fl.append(score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)]),
                        C[r.permutation(len(C))]))
    floor = float(np.mean(fl))

    variants = build_variants(G_cdg, E_hold, np.random.default_rng(20260711))

    print("=" * 92)
    print(f"WP-2 confirmatory experiment — real MIMIC-IV · L={args.depth} · "
          f"{args.seeds} seeds · {args.steps} steps")
    print("=" * 92)
    print(f"  cohort : n={len(df):,}  ·  CDG {G_cdg.number_of_edges()} edges  "
          f"max degree {max(dict(G_cdg.degree()).values())}  diameter {nx.diameter(G_cdg)}")
    print(f"  critic : copula + batch-aware   ·   loss: pure WGAN-GP, no dependency term")
    print(f"  metric : conditional partial correlation over all 120 pairs (eval_dep)")
    print(f"  floor  : {floor:.4f}   <- a model that creates no dependency at all")
    if args.depth != 1:
        print(f"\n  ** L={args.depth} is a NEGATIVE CONTROL. The effect is expected to vanish. **")
    print()
    print(f"  {'variant':<18} {'|E|':>4} {'120-pair error':>16} {'vs floor':>10}")
    print("  " + "-" * 56)

    res = {}
    for name, Gv in variants.items():
        rows, t0 = [], time.time()
        for s in range(args.seeds):
            cfg = Cfg(steps=args.steps, seed=s, copula=True, batch_critic=True, lr_q=5e-3)
            Gm = train_fix2(X, C, list(Gv.edges()), cfg)
            Xs, Cs = generate(Gm, C, args.n_syn, s)
            rows.append(score(Xs, Cs))
        res[name] = rows
        m, sd = float(np.mean(rows)), float(np.std(rows))
        print(f"  {name:<18} {Gv.number_of_edges():>4} {m:>10.4f} ± {sd:.4f} "
              f"{(m - floor) / floor * 100:>+9.1f}%   ({time.time()-t0:.0f}s)", flush=True)

    # ---- gate: did anything learn at all? --------------------------------
    #
    # If the model does not beat the floor, NOTHING below is interpretable. Every topology
    # scores the same when nothing learns dependency, and a bootstrap CI over near-identical
    # numbers will happily report a "significant" difference of 0.0007. We nearly published
    # that mistake once (REVISIONS §E-2); the code refuses to let it happen quietly.
    print()
    print("=" * 92)
    cdg = np.array(res["cdg"])
    if cdg.mean() >= floor:
        print("  ** STOP — the trained CDG model LOSES to a model that creates no dependency **")
        print(f"     CDG {cdg.mean():.4f}  vs  floor {floor:.4f}")
        print()
        print("     Nothing has learned any dependency, so every topology scores alike and the")
        print("     contrasts below are noise. Do NOT read them as evidence for or against the")
        print("     hypothesis — a bootstrap CI over near-identical numbers will report")
        print("     'significance' that means nothing.")
        print()
        print("     Likely causes, in order:")
        print("       - too few steps (this is what a short smoke run looks like)")
        print("       - the critic is not the copula + batch-aware one (HANDOFF §2.1)")
        print("       - the metric is not eval_dep.partial_corr_c")
        print("     See RESULTS_lr.md and REVISIONS §E-7.")
        print("=" * 92)


    def group(prefix):
        return np.concatenate([res[k] for k in res if k.startswith(prefix)])

    rng = np.random.default_rng(0)
    for label, ctrl in (("permuted (3 graphs)", group("permuted")),
                        ("distance-matched (3 graphs)", group("distmatched")),
                        ("rewired", np.array(res["rewired"])),
                        ("ring", np.array(res["ring"])),
                        ("no_entangle", np.array(res["no_entangle"]))):
        d = cdg.mean() - ctrl.mean()
        # hierarchical bootstrap over seeds
        boot = [rng.choice(cdg, len(cdg)).mean() - rng.choice(ctrl, len(ctrl)).mean()
                for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ok = hi < 0
        print(f"  CDG − {label:<28} = {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"{'CDG better' if ok else 'not significant' if lo < 0 < hi else 'CDG WORSE'}")

    print()
    print("  Primary: CDG − permuted < 0, CI excludes 0.")
    print("  Decisive: CDG − distance-matched < 0. Beating the isomorphic permutation alone")
    print("            could be graph combinatorics; distance matching is what isolates the")
    print("            clinical claim.")
    print(f"  Sanity  : every variant must be compared against the floor ({floor:.4f}). A model")
    print("            that loses to it has learned nothing, and nothing downstream is meaningful.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"wp2_L{args.depth}.json"
    out.write_text(json.dumps({"floor": floor, "results": res,
                               "seeds": args.seeds, "steps": args.steps}, indent=2))
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()

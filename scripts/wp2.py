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
  steps    : >= 8000.  NOT 3000.  See below — this is not a tuning preference.
  batch    : 512.      NOT 256.   See below — this one decides the effect size.

Two settings that are load-bearing, and why (REVISIONS §E-8)
-----------------------------------------------------------
The first attempt at this experiment ran 3000 steps at batch 256 and put every variant, the CDG
included, ABOVE the floor. Neither number was innocent:

  steps.  On real MIMIC the CDG error curve crosses the floor at step ~4000 and keeps falling to
          0.0679 by 12000 (`scripts/diag_wp2_probe.py`). At 3000 it reads 0.0988 — sitting exactly
          ON the floor of 0.0985. Real data converges ~4x slower than the synthetic teacher
          because its dependencies are weaker (mean edge |z| 0.21 vs 0.36). A 10x10 run at 3000
          steps yields ten numbers clustered on the floor, and a bootstrap that reports
          "significance" on the noise between them.

  batch.  The copula transform is a soft rank computed WITHIN the batch, so the batch size sets
          how sharply the critic sees the very object it is meant to judge — a 16-dimensional
          joint. At batch 256 the measured CDG-permuted gap is 0.0065. At batch 512 it is 0.0207,
          which is what the training-free ceiling says the true gap is (0.0195, RESULTS_ceiling_real).
          The small batch did not just add noise; it COMPRESSED THE EFFECT to a third of its size.

So: an undersized batch and an early stop would each, on their own, have turned a real effect into
a null. Both were present. This is why the floor is computed first and the run refuses to
interpret itself if the CDG does not beat it.

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

from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from train_v3 import CfgV3, train_v3  # noqa: E402
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
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512,
                    help="512, not 256. The copula rank is computed within the batch; "
                         "at 256 the CDG-permuted effect shrinks to a third. REVISIONS E-8.")
    ap.add_argument("--depth", type=int, default=1, help="L. 1 = confirmatory; 2 = negative control")
    ap.add_argument("--n-syn", type=int, default=20000)
    # v3 loss (REVISIONS §E-10). The copula critic supplies the dependency; a strictly 1-D
    # conditional marginal term supplies the shape. Without it the marginals are wrong by two
    # orders of magnitude (W1 0.68) and TSTR falls BELOW the zero-dependency floor.
    ap.add_argument("--lambda-marg", type=float, default=10.0)
    ap.add_argument("--lr-g", type=float, default=1e-3,
                    help="head lr. 5e-5 was set when nothing trained the heads on the marginals; "
                         "it is far too small now that they must.")
    ap.add_argument("--cdg", type=str, default="cdg_d4.npz",
                    help="cdg.npz = Δ3 (v2 §7.6's degree cap); cdg_d4.npz = Δ4, the redesign")
    # Sharding. One run is ~30 min at 8000x512, and the full design is 10 variants x 10 seeds =
    # 100 runs = ~50 h serial. These runs leave the GPU mostly idle (light-cone subcircuits, small
    # batches), so 3 concurrent shards cost ~1.7x each in wall-clock and win ~1.8x in throughput.
    # Each shard writes its own JSON; `wp2_report.py` aggregates and does the floor gate + bootstrap.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(
        subset=names + ["age", "sex", "icu_type"])
    X = df[names].to_numpy(float)
    C = df[["y", "age", "sex", "icu_type"]].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    z = np.load(PROCESSED / args.cdg, allow_pickle=True)
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

    # The variant set is built from a FIXED seed, so every shard constructs the identical graphs
    # and can then take its own slice by name. Sharding must not change the experiment.
    variants = build_variants(G_cdg, E_hold, np.random.default_rng(20260711))
    mine = {k: v for i, (k, v) in enumerate(variants.items())
            if i % args.nshards == args.shard}

    print("=" * 92)
    print(f"WP-2 confirmatory experiment — real MIMIC-IV · L={args.depth} · {args.cdg} · "
          f"{args.seeds} seeds · {args.steps} steps · batch {args.batch}")
    if args.nshards > 1:
        print(f"  shard {args.shard + 1}/{args.nshards}: {', '.join(mine)}")
    print("=" * 92)
    print(f"  cohort : n={len(df):,}  ·  CDG {G_cdg.number_of_edges()} edges  "
          f"max degree {max(dict(G_cdg.degree()).values())}  diameter {nx.diameter(G_cdg)}")
    print(f"  loss   : copula+batch-aware critic (dependency)  +  λ={args.lambda_marg} × "
          f"1-D conditional marginal term (shape)")
    print(f"           the marginal term's Jacobian is diagonal (model_v3), so it cannot create")
    print(f"           cross-feature dependence. v2 §8.10 holds. lr_g={args.lr_g}")
    print(f"  metric : conditional partial correlation over all 120 pairs (eval_dep)")
    print(f"  floor  : {floor:.4f}   <- a model that creates no dependency at all")
    if args.depth != 1:
        print(f"\n  ** L={args.depth} is a NEGATIVE CONTROL. The effect is expected to vanish. **")
    print()
    print(f"  {'variant':<18} {'|E|':>4} {'120-pair error':>16} {'vs floor':>10}")
    print("  " + "-" * 56)

    # Written after EVERY variant, not at the end. One variant is ~7 GPU-hours here; a shard that
    # crashed on its last one used to take every earlier result down with it, unsaved. If the file
    # already holds variants (a resumed shard), they are kept and skipped.
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"wp2_L{args.depth}.json" if args.nshards == 1
                     else f"wp2_L{args.depth}_shard{args.shard}.json")

    # The config this run is producing. A resumed file is only usable if it was produced by the
    # SAME one — otherwise we would silently mix Δ=3 rows into a Δ=4 table, or v2-loss rows into a
    # v3 table, and the mixture would be invisible in the output. That nearly happened: the first
    # v3 smoke run cheerfully "resumed" three variants left over from the Δ=3 / v2 / 4-shard run.
    cfgkey = {"cdg": args.cdg, "depth": args.depth, "steps": args.steps, "batch": args.batch,
              "seeds": args.seeds, "lambda_marg": args.lambda_marg, "lr_g": args.lr_g,
              "nshards": args.nshards, "shard": args.shard, "loss": "v3"}
    res = {}
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("config") == cfgkey:
            res = prev["results"]
            if res:
                print(f"  resuming: {', '.join(res)} already done\n")
        else:
            stale = out.with_suffix(".json.stale")
            out.replace(stale)
            print(f"  ** the existing {out.name} was produced by a DIFFERENT configuration.")
            print(f"     moved to {stale.name}; starting clean. **\n")

    def save():
        out.write_text(json.dumps({"config": cfgkey, "floor": floor, "results": res,
                                   "seeds": args.seeds, "steps": args.steps,
                                   "batch": args.batch}, indent=2))

    for name, Gv in mine.items():
        if name in res:
            continue
        rows, t0 = [], time.time()
        for s in range(args.seeds):
            cfg = CfgV3(steps=args.steps, batch=args.batch, seed=s,
                        lambda_marg=args.lambda_marg, lr_g=args.lr_g)
            Gm = train_v3(X, C, list(Gv.edges()), cfg)
            Xs, Cs = generate(Gm, C, args.n_syn, s)
            rows.append(score(Xs, Cs))
        res[name] = rows
        save()
        m, sd = float(np.mean(rows)), float(np.std(rows))
        print(f"  {name:<18} {Gv.number_of_edges():>4} {m:>10.4f} ± {sd:.4f} "
              f"{(m - floor) / floor * 100:>+9.1f}%   ({time.time()-t0:.0f}s)", flush=True)

    # No analysis here. A shard holds only part of the design, and a shard that tried to
    # interpret itself would be reading contrasts against controls it never ran. Aggregation,
    # the floor gate and the bootstrap all live in `wp2_report.py`, which refuses to run until
    # every variant is present.
    print(f"\n  saved: {out}")
    if args.nshards > 1:
        print(f"  shard {args.shard + 1}/{args.nshards} done. "
              f"Run `python scripts/wp2_report.py` once all shards finish.")


if __name__ == "__main__":
    main()

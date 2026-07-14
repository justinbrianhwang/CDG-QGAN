"""CDG-QGAN under the WP-3 protocol — the same split, the same metrics, as the classical baselines.

`wp3_baselines.py` scores CTGAN / TVAE / a Gaussian copula on a 75/25 split, measuring dependency
error, TSTR (train on synthetic, test on real) and marginal W1. WP-2 scores our circuit on the full
cohort and measures dependency error only. Those two tables cannot be put next to each other.

This script closes the gap: it trains CDG-QGAN on the SAME training slice, samples, and runs the
SAME three metrics, so the numbers land in one table. Anything else would be a comparison between
two protocols, not two models.

Two graphs are run:
    Δ=3  (cdg.npz)     the original design. Corollary 1 bound 0.0331 — 72 of 120 pairs are outside
                       the L=1 light cone and are forced to exactly zero.
    Δ=4  (cdg_d4.npz)  the redesign (`design_sweep.py`). Bound 0.0140, 37 pairs outside. Same L=1,
                       so Corollary 1 and Proposition D-2 are untouched; the degree cap of v2 §7.6
                       is what moved, and it was never a claim about the data.

and, as always, `no_entangle` — the control that proves the dependency has to come from the
entangling angles and cannot come from the ~2,000 classical head parameters.

What we expect, written down BEFORE the run
-------------------------------------------
On dependency error we expect to LOSE to the Gaussian copula at both Δ, and possibly to CTGAN/TVAE
as well. The 120-pair average weights a |z|=0.02 pair exactly as much as the |z|=1.06 sodium–chloride
pair, and 37–72 of those pairs are unreachable by construction.

On TSTR we expect to do better than that ratio suggests, because the CDG deliberately places the
STRONG dependencies inside the light cone (sodium–chloride 1.06, creatinine–bun 0.78, chloride–
bicarbonate 0.66 are all at distance <= 2) and those are the ones a mortality model uses. If that
holds it is a finding; if it does not, the honest conclusion is that at L=1 the model has no
performance argument at all and the paper rests on the structural result.

This paragraph is the pre-registration. It is being written before the numbers exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from train_v3 import CfgV3, train_v3  # noqa: E402
from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402
from wp3_baselines import COND, marginal_w1, tstr  # noqa: E402  (same functions, not a copy)

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_SYN = 20000


def load_graph(fname: str) -> nx.Graph:
    z = np.load(PROCESSED / fname, allow_pickle=True)
    G = nx.Graph()
    G.add_nodes_from(range(N_FEAT))
    G.add_edges_from([tuple(e) for e in z["E_fit"]])
    return G


def bound_of(G: nx.Graph, az: np.ndarray) -> tuple[float, int]:
    """Corollary 1: what the circuit cannot reach, it pays for in full."""
    out = [p for p in ALL_PAIRS
           if not nx.has_path(G, *p) or nx.shortest_path_length(G, *p) > 2]
    return float(np.sum([az[i, j] for i, j in out]) / len(ALL_PAIRS)), len(out)


@torch.no_grad()
def sample(G, C, n, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy(), C[idx]


def calibrate_marginals(Xs: np.ndarray, Xtr: np.ndarray) -> np.ndarray:
    """Per-feature monotone quantile mapping:  x_u -> F_train,u^{-1}( F_syn,u(x_u) ).

    Why this is necessary
    ---------------------
    The copula critic (REVISIONS §E-7) is DELIBERATELY blind to the marginals — that is what
    forces the gradient into the entangling angles instead of letting it settle for the marginals,
    and it is what made the model learn any dependency at all. But it has a consequence nobody
    priced in: **there is then no term in the loss that trains the heads to match the marginals.**
    So they don't. Measured: marginal W1 = 0.676, against a Gaussian copula's 0.007.

    The dependency metric never saw this, because it is a nonparanormal (rank) quantity and is
    invariant to any monotone per-feature map. TSTR is not, and TSTR fell BELOW the floor.

    Why this fix is free, and not a cheat
    -------------------------------------
    The map is monotone and applied to one feature at a time. A monotone per-feature map leaves
    the COPULA exactly unchanged — so the conditional dependency structure, which is the entire
    scientific claim, is bit-for-bit identical before and after. It cannot manufacture a
    dependency, and `no_entangle` still lands on the floor after calibration.

    It also does not touch Proposition D-2: it is another 1-D map of x_u, reading no other
    feature. It is the same object the local head already is — we are simply enforcing the
    monotonicity the theory assumed all along (and which §E-8 measured the free head violating on
    58% of the q-grid).

    Fitted on the TRAIN split only. Using the training marginals is what every generative model
    does; the held-out patients are never touched.

    Interpolate, do not index
    -------------------------
    A first version mapped the rank to an integer index into the sorted training column. That is
    monotone but NOT STRICTLY monotone — many synthetic values collapse onto the same training
    value — and ties change ranks, so the nonparanormal transform moved and the dependency error
    shifted by ~0.003 on every variant (including `no_entangle`, uniformly, which is why it was a
    tie artefact and not injected structure). Linear interpolation between the order statistics is
    strictly increasing, so the ranks — and therefore the copula — are preserved exactly.
    """
    n_ref = Xtr.shape[0]
    grid = (np.arange(n_ref) + 0.5) / n_ref          # plotting positions of the order statistics
    out = np.empty_like(Xs)
    for j in range(Xs.shape[1]):
        r = rankdata(Xs[:, j], method="average") / (len(Xs) + 1)     # F_syn,u(x) in (0,1)
        out[:, j] = np.interp(r, grid, np.sort(Xtr[:, j]))           # F_train,u^{-1}(.)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lambda-marg", type=float, default=10.0)
    ap.add_argument("--lr-g", type=float, default=1e-3)
    # One variant is 3 trainings ~= 2.6 GPU-hours. Run the four variants as four processes and
    # merge with --merge; the GPU is saturated either way, so this buys wall-clock, not throughput.
    ap.add_argument("--only", type=str, default=None,
                    help="run one variant only: d3 | d4 | permuted | no_entangle")
    ap.add_argument("--merge", action="store_true",
                    help="merge the per-variant JSONs into wp3_qgan.json and print the table")
    args = ap.parse_args()

    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(subset=names + COND[1:])
    X = df[names].to_numpy(float)
    C = df[COND].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # the SAME split as wp3_baselines.py — same seed, same stratification
    Xtr, Xte, Ctr, Cte = train_test_split(X, C, test_size=0.25, random_state=0,
                                          stratify=C[:, 0])

    zr = fisher_z(partial_corr_c(Xtr, Ctr))
    az = np.abs(zr)

    def dep(Xs, Cs):
        return dep_error(zr, fisher_z(partial_corr_c(Xs, Cs)), ALL_PAIRS)

    trtr = tstr(Xtr, Ctr, Xte, Cte)

    variants = {}
    for tag, fname in (("CDG-QGAN Δ=3", "cdg.npz"), ("CDG-QGAN Δ=4", "cdg_d4.npz")):
        if (PROCESSED / fname).exists():
            variants[tag] = load_graph(fname)
    G3 = variants.get("CDG-QGAN Δ=3")
    if G3 is not None:
        variants["  permuted (Δ=3)"] = isomorphic_permuted(G3, np.random.default_rng(20260711))
    variants["  no_entangle"] = nx.empty_graph(N_FEAT)

    KEY = {"d3": "CDG-QGAN Δ=3", "d4": "CDG-QGAN Δ=4",
           "permuted": "  permuted (Δ=3)", "no_entangle": "  no_entangle"}
    if args.merge:
        merged = {"tstr_ceiling": list(trtr)}
        for part in sorted(RESULTS.glob("wp3_qgan_part_*.json")):
            merged.update({k: v for k, v in json.loads(part.read_text()).items()
                           if k != "tstr_ceiling"})
        (RESULTS / "wp3_qgan.json").write_text(json.dumps(merged, indent=2, default=float))
        print(f"  TSTR ceiling (train on REAL): AUROC {trtr[0]:.4f}  AUPRC {trtr[1]:.4f}\n")
        print(f"  {'model':<22} {'|E|':>4} {'bound':>7} {'dep':>8} {'TSTR AUROC':>11} "
              f"{'TSTR AUPRC':>11} {'W1':>8}")
        for k, v in merged.items():
            if k == "tstr_ceiling":
                continue
            print(f"  {k:<22} {v['edges']:>4} {v['bound']:>7.4f} {v['dep']:>8.4f} "
                  f"{v['auroc']:>11.4f} {v['auprc']:>11.4f} {v['w1']:>8.4f}")
        print(f"\n  saved: {RESULTS / 'wp3_qgan.json'}")
        return

    if args.only:
        want = KEY[args.only]
        variants = {k: v for k, v in variants.items() if k == want}
        if not variants:
            sys.exit(f"--only {args.only}: that graph file is not present")

    print("=" * 108)
    print("CDG-QGAN under the WP-3 protocol — same split, same metrics as the classical baselines")
    print("=" * 108)
    print(f"  train n={len(Xtr):,}  ·  held-out real test n={len(Xte):,}  ·  "
          f"{args.seeds} seeds · {args.steps} steps · batch {args.batch}")
    print(f"  TSTR ceiling (train on REAL): AUROC {trtr[0]:.4f}  AUPRC {trtr[1]:.4f}")
    print()
    print("  Each model is scored twice: raw, and after a per-feature monotone quantile map onto")
    print("  the training marginals (`calibrate_marginals`). The map cannot change the copula, so")
    print("  the dependency column MUST be identical between the two — that identity is the proof")
    print("  the calibration is not smuggling in structure. Only the marginals and TSTR move.")
    print()
    print(f"  {'model':<20} {'|E|':>4} {'bound':>7} {'dep. error':>11} {'TSTR AUROC':>11} "
          f"{'TSTR AUPRC':>11} {'marg. W1':>9}")
    print("  " + "-" * 88)

    out = {"tstr_ceiling": list(trtr)}
    for tag, Gv in variants.items():
        b, n_out = bound_of(Gv, az) if Gv.number_of_edges() else (float(np.mean(az[np.triu_indices(N_FEAT, 1)])), 120)
        raw = {"de": [], "au": [], "apr": [], "w1": []}
        cal = {"de": [], "au": [], "apr": [], "w1": []}
        t0 = time.time()
        for s in range(args.seeds):
            cfg = CfgV3(steps=args.steps, batch=args.batch, seed=s, lr_q=5e-3,
                        lr_g=args.lr_g, lambda_marg=args.lambda_marg)
            Gm = train_v3(Xtr, Ctr, list(Gv.edges()), cfg)
            Xs, Cs = sample(Gm, Ctr, N_SYN, s)
            Xc = calibrate_marginals(Xs, Xtr)
            for d, A in ((raw, Xs), (cal, Xc)):
                d["de"].append(dep(A, Cs))
                a, p = tstr(A, Cs, Xte, Cte, s)
                d["au"].append(a); d["apr"].append(p)
                d["w1"].append(marginal_w1(A, Xtr))

        for label, d in (("", raw), ("  + calibrated", cal)):
            key = tag.strip() + label.strip()
            out[key] = {"edges": Gv.number_of_edges(), "bound": b,
                        "dep": float(np.mean(d["de"])), "auroc": float(np.nanmean(d["au"])),
                        "auprc": float(np.nanmean(d["apr"])), "w1": float(np.mean(d["w1"]))}
            name = tag if not label else label
            print(f"  {name:<20} {Gv.number_of_edges() if not label else '':>4} "
                  f"{b if not label else float('nan'):>7.4f} {np.mean(d['de']):>11.4f} "
                  f"{np.nanmean(d['au']):>11.4f} {np.nanmean(d['apr']):>11.4f} "
                  f"{np.mean(d['w1']):>9.4f}", flush=True)
        print(f"       ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 108)
    print("  'bound' is the Corollary 1 limit: the error this circuit MUST pay on the pairs outside")
    print("  its L=1 light cone, at every parameter setting. It is a property of the graph, not of")
    print("  the optimizer, and no amount of training can go below it.")
    print()
    print("  The copula critic is blind to the marginals BY DESIGN — that is what forced the")
    print("  gradient into the entangling angles (REVISIONS §E-7). The price, which we did not see")
    print("  until TSTR was measured, is that nothing in the loss trained the heads to match the")
    print("  marginals, and they didn't (W1 = 0.676, TSTR BELOW the floor). The v3 loss fixes that")
    print("  IN THE OBJECTIVE with a strictly 1-D conditional marginal term (§E-11), not with a")
    print("  post-hoc map — post-hoc calibration fixed W1 and made TSTR WORSE (0.641 -> 0.573).")
    print()
    print("  The calibration rows are kept only as an INVARIANT CHECK: a monotone per-feature map")
    print("  cannot change the copula, so the dep. column must be identical between the two rows.")
    print("  If it moves, something in the pipeline is not rank-invariant and the metric is lying.")
    print("=" * 108)

    RESULTS.mkdir(exist_ok=True)
    # With --only, each variant is its own process and MUST write its own file. Three processes
    # writing wp3_qgan.json would clobber one another and the last one to finish would silently
    # win, leaving a one-row table that looks complete. `--merge` assembles the parts.
    dst = RESULTS / (f"wp3_qgan_part_{args.only}.json" if args.only else "wp3_qgan.json")
    dst.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  saved: {dst}")


if __name__ == "__main__":
    main()

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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from diag_fix2 import Cfg, train_fix2  # noqa: E402
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
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

    print("=" * 108)
    print("CDG-QGAN under the WP-3 protocol — same split, same metrics as the classical baselines")
    print("=" * 108)
    print(f"  train n={len(Xtr):,}  ·  held-out real test n={len(Xte):,}  ·  "
          f"{args.seeds} seeds · {args.steps} steps · batch {args.batch}")
    print(f"  TSTR ceiling (train on REAL): AUROC {trtr[0]:.4f}  AUPRC {trtr[1]:.4f}")
    print()
    print(f"  {'model':<20} {'|E|':>4} {'bound':>7} {'dep. error':>11} {'TSTR AUROC':>11} "
          f"{'TSTR AUPRC':>11} {'marg. W1':>9}")
    print("  " + "-" * 88)

    out = {"tstr_ceiling": list(trtr)}
    for tag, Gv in variants.items():
        b, n_out = bound_of(Gv, az) if Gv.number_of_edges() else (float(np.mean(az[np.triu_indices(N_FEAT, 1)])), 120)
        de, au, apr, w1 = [], [], [], []
        t0 = time.time()
        for s in range(args.seeds):
            cfg = Cfg(steps=args.steps, batch=args.batch, seed=s,
                      copula=True, batch_critic=True, lr_q=5e-3)
            Gm = train_fix2(Xtr, Ctr, list(Gv.edges()), cfg)
            Xs, Cs = sample(Gm, Ctr, N_SYN, s)
            de.append(dep(Xs, Cs))
            a, p = tstr(Xs, Cs, Xte, Cte, s)
            au.append(a); apr.append(p)
            w1.append(marginal_w1(Xs, Xtr))
        out[tag.strip()] = {"edges": Gv.number_of_edges(), "bound": b,
                            "dep": float(np.mean(de)), "auroc": float(np.nanmean(au)),
                            "auprc": float(np.nanmean(apr)), "w1": float(np.mean(w1))}
        print(f"  {tag:<20} {Gv.number_of_edges():>4} {b:>7.4f} {np.mean(de):>11.4f} "
              f"{np.nanmean(au):>11.4f} {np.nanmean(apr):>11.4f} {np.mean(w1):>9.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 108)
    print("  'bound' is the Corollary 1 limit: the error this circuit MUST pay on the pairs outside")
    print("  its L=1 light cone, at every parameter setting. It is a property of the graph, not of")
    print("  the optimizer, and no amount of training can go below it.")
    print()
    print("  Put this table next to wp3_baselines.py. If we lose the dependency column but hold")
    print("  TSTR, say exactly that: the 120-pair average is unweighted and the CDG spends its")
    print("  light cone on the dependencies that carry clinical signal.")
    print("=" * 108)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "wp3_qgan.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  saved: {RESULTS / 'wp3_qgan.json'}")


if __name__ == "__main__":
    main()

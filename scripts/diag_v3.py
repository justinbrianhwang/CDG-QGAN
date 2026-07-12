"""Does the v3 loss fix the marginals WITHOUT smuggling dependency in?

Two questions, and the second one can kill the paper.

  1. Does the 1-D conditional marginal term fix what it was added to fix?
       marginal W1   0.676 -> ?      (a Gaussian copula gets 0.007)
       TSTR AUROC    0.641 -> ?      (a ZERO-dependency model gets 0.682; real data 0.857)

  2. Does it leave the dependency claim intact?
       dependency error of the CDG      -> must not get worse
       dependency error of `no_entangle` -> **MUST STAY ON THE FLOOR**

The second is the falsifier. `no_entangle` has no RZZ gates, so by Proposition D-2 it can create no
conditional cross-feature dependence at all, and it has landed exactly on the dependency floor every
time we have measured it (Δ=3: 0.0951/0.0986; Δ=4: 0.0951/0.0986; 12,000-step training curve:
converges to 0.0986 and stays). If switching on the marginal term lets a zero-entanglement model
start beating that floor, then the term is a source of dependency, v2 §8.10 is violated, and
`train_v3.py` must be reverted — no matter how good the TSTR looks.

`model_v3.assert_marginal_loss_is_1d` already proves the term's Jacobian is diagonal (off-diagonal
max = 0.0e+00), so this should not happen. But "should not happen" is exactly the kind of claim this
project has falsified four times, and the control costs one extra row.
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

from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402
from train_v3 import CfgV3, train_v3  # noqa: E402
from wp3_baselines import COND, marginal_w1, tstr  # noqa: E402
from wp3_qgan import calibrate_marginals  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
N_SYN = 20000

# reference points, all measured (RESULTS_wp3.md, REVISIONS §E-10)
REF = {"copula_dep": 0.0064, "copula_w1": 0.0070, "copula_tstr": 0.8073,
       "tvae_dep": 0.0708, "tvae_tstr": 0.7736,
       "floor_dep": 0.0990, "floor_tstr": 0.6823, "real_tstr": 0.8569,
       "v2_dep": 0.0748, "v2_tstr": 0.6412, "v2_w1": 0.6759}


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
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lambda-marg", type=float, default=1.0)
    # The head learning rate was 5e-5 — chosen when NOTHING trained the heads on the marginals, so
    # it never mattered. Now it does: the heads are the only thing that can fix a marginal, and at
    # 5e-5 they cannot move far enough (v2 sat at W1 = 0.68 after 8,000 steps). It is a free
    # parameter again and has to be swept.
    ap.add_argument("--lr-g", type=float, default=5e-5)
    ap.add_argument("--cdg", type=str, default="cdg_d4.npz")
    ap.add_argument("--only-cdg", action="store_true",
                    help="skip the controls — for hyperparameter sweeps, where only W1/TSTR matter")
    args = ap.parse_args()

    names = [f.name for f in CORE16]
    df = pd.read_parquet(PROCESSED / "cohort_v31.parquet").dropna(subset=names + COND[1:])
    X = df[names].to_numpy(float)
    C = df[COND].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    Xtr, Xte, Ctr, Cte = train_test_split(X, C, test_size=0.25, random_state=0,
                                          stratify=C[:, 0])
    zr = fisher_z(partial_corr_c(Xtr, Ctr))

    def dep(A, Cs):
        return dep_error(zr, fisher_z(partial_corr_c(A, Cs)), ALL_PAIRS)

    # the dependency floor, on THIS split — the number no_entangle must not beat
    fl = []
    for s in range(5):
        r = np.random.default_rng(1000 + s)
        fl.append(dep(np.column_stack([r.permutation(Xtr[:, j]) for j in range(N_FEAT)]),
                      Ctr[r.permutation(len(Ctr))]))
    floor = float(np.mean(fl))

    z = np.load(PROCESSED / args.cdg, allow_pickle=True)
    G_cdg = nx.Graph()
    G_cdg.add_nodes_from(range(N_FEAT))
    G_cdg.add_edges_from([tuple(e) for e in z["E_fit"]])

    variants = {f"CDG ({args.cdg[:-4]})": G_cdg}
    if not args.only_cdg:
        variants["permuted"] = isomorphic_permuted(G_cdg, np.random.default_rng(20260711))
        variants["no_entangle  <= FALSIFIER"] = nx.empty_graph(N_FEAT)

    print("=" * 104)
    print(f"v3 loss — copula critic + 1-D conditional marginal term (lambda = {args.lambda_marg})")
    print("=" * 104)
    print(f"  {args.cdg}: {G_cdg.number_of_edges()} edges · L=1 · {args.seeds} seeds · "
          f"{args.steps} steps · batch {args.batch} · lambda {args.lambda_marg} · lr_g {args.lr_g}")
    print(f"  dependency floor on this split: {floor:.4f}")
    print()
    print("  reference (measured):")
    print(f"    real data          TSTR {REF['real_tstr']:.4f}")
    print(f"    gaussian copula    dep {REF['copula_dep']:.4f}  TSTR {REF['copula_tstr']:.4f}  "
          f"W1 {REF['copula_w1']:.4f}")
    print(f"    TVAE               dep {REF['tvae_dep']:.4f}  TSTR {REF['tvae_tstr']:.4f}")
    print(f"    independent floor  dep {REF['floor_dep']:.4f}  TSTR {REF['floor_tstr']:.4f}")
    print(f"    ** v2 (no marginal term)  dep {REF['v2_dep']:.4f}  TSTR {REF['v2_tstr']:.4f}  "
          f"W1 {REF['v2_w1']:.4f}  <- what we are fixing")
    print()
    print(f"  {'model':<26} {'dep. error':>11} {'TSTR AUROC':>11} {'TSTR AUPRC':>11} "
          f"{'marg. W1':>9}")
    print("  " + "-" * 76)

    out = {"floor_dep": floor, "lambda_marg": args.lambda_marg, "cdg": args.cdg}
    for tag, Gv in variants.items():
        acc = {"de": [], "au": [], "ap": [], "w1": []}
        t0 = time.time()
        for s in range(args.seeds):
            cfg = CfgV3(steps=args.steps, batch=args.batch, seed=s,
                        lambda_marg=args.lambda_marg, lr_g=args.lr_g)
            Gm = train_v3(Xtr, Ctr, list(Gv.edges()), cfg)
            Xs, Cs = sample(Gm, Ctr, N_SYN, s)
            acc["de"].append(dep(Xs, Cs))
            a, p = tstr(Xs, Cs, Xte, Cte, s)
            acc["au"].append(a); acc["ap"].append(p)
            acc["w1"].append(marginal_w1(Xs, Xtr))
        out[tag] = {k: float(np.nanmean(v)) for k, v in acc.items()}
        print(f"  {tag:<26} {np.mean(acc['de']):>11.4f} {np.nanmean(acc['au']):>11.4f} "
              f"{np.nanmean(acc['ap']):>11.4f} {np.mean(acc['w1']):>9.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    print()
    print("=" * 104)
    if args.only_cdg:
        print("  (sweep mode: the controls were skipped, so the falsifier was NOT checked.")
        print("   No configuration is adopted from a run that did not test `no_entangle`.)")
        print("=" * 104)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"diag_v3_lam{args.lambda_marg}_lrg{args.lr_g}.json").write_text(
            json.dumps(out, indent=2))
        return

    ne = out["no_entangle  <= FALSIFIER"]["de"]
    if ne < floor - 0.005:
        print(f"  ** REVERT train_v3.py **")
        print(f"     no_entangle scored {ne:.4f}, BELOW the dependency floor {floor:.4f}.")
        print("     A model with zero RZZ gates cannot create dependency (Prop. D-2). If it now")
        print("     beats the floor, the marginal term is supplying dependency and v2 §8.10 is")
        print("     violated. The TSTR numbers above are worthless — do not report them.")
    else:
        print(f"  falsifier held: no_entangle {ne:.4f} vs floor {floor:.4f} — still on the floor.")
        print("  The marginal term shapes 1-D conditionals and supplies NO dependency. All")
        print("  cross-feature structure still comes from the entangling angles alone.")
    print("=" * 104)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"diag_v3_lam{args.lambda_marg}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

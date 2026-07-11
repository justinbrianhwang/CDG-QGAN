"""Does the real-data model EVER beat the floor? Score the training trajectory, not the endpoint.

Why this exists
---------------
The WP-2 smoke run (200 steps) put every variant — CDG included — 33% ABOVE the floor. That is
the regime where nothing has learned any dependency and every topology scores alike, so the
contrasts mean nothing (REVISIONS §E-2). 200 steps is far too few to conclude anything, but the
full run is 10 variants x 10 seeds x 3000 steps and I am not willing to find out after twelve
hours of GPU that 3000 steps was also too few.

So: train on the REAL cohort and score every EVERY steps. Three graphs only — the CDG, one
isomorphic permutation, and no_entangle. The questions, in order of importance:

  1. Does the CDG curve ever cross below the floor?  If it never does, the full run is dead on
     arrival regardless of how the variants rank, and the fix has to come first.
  2. Where does it cross?  That sets the step count for the real run.
  3. Do CDG and permuted separate, and does that separation grow or shrink with steps?

no_entangle is the control that matters most here: on the synthetic teacher it landed exactly on
the floor (it can create no dependency at all, by construction). If on real data it drifts BELOW
the floor, the metric or the sampler is broken, not the model.
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

from diag_fix2 import Cfg, Critic, gradient_penalty  # noqa: E402
from eval_dep import dep_error, fisher_z, partial_corr_c  # noqa: E402
from features import CORE16  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from model import CDGQGAN  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402
from train import DEVICE  # noqa: E402

import pandas as pd  # noqa: E402

N_FEAT = 16
ALL_PAIRS = [(i, j) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)]
TOTAL_STEPS = 12000
EVERY = 1000
N_SYN = 20000


@torch.no_grad()
def generate(G, C, n, seed):
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).integers(0, len(C), n)
    c = torch.tensor(C[idx], dtype=torch.float32, device=DEVICE)
    z = 2 * torch.rand(n, G.n, device=DEVICE) - 1
    return G(z, c).cpu().numpy(), C[idx]


def train_scored(X, C, edges, cfg: Cfg, score_fn, total: int, every: int):
    """train_fix2, but scored on a held-out draw every `every` steps instead of only at the end."""
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = CDGQGAN(n, edges, 1, C.shape[1], cfg.head_width, seed=cfg.seed).to(DEVICE)
    D = Critic(n, C.shape[1], cfg.copula, cfg.batch_critic, cfg.tau).to(DEVICE)
    og = torch.optim.Adam([{"params": list(G.quantum.parameters()), "lr": cfg.lr_q},
                           {"params": list(G.heads.parameters()), "lr": cfg.lr_g}],
                          betas=(0.0, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9))

    def real(b):
        i = torch.randint(0, N, (b,), device=DEVICE)
        return Xt[i], Ct[i]

    def zz(b):
        return 2 * torch.rand(b, n, device=DEVICE) - 1

    curve = []
    for step in range(1, total + 1):
        for _ in range(cfg.critic_steps):
            xr, c = real(cfg.batch)
            with torch.no_grad():
                xf = G(zz(cfg.batch), c)
            ld = (D(xf, c).mean() - D(xr, c).mean()
                  + cfg.lambda_gp * gradient_penalty(D, xr, xf, c))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()

        _, c = real(cfg.batch)
        lg = -D(G(zz(cfg.batch), c), c).mean()
        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()

        if step % every == 0:
            Xs, Cs = generate(G, C, N_SYN, cfg.seed)
            curve.append((step, score_fn(Xs, Cs)))
    return curve


def main() -> None:
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

    zr = fisher_z(partial_corr_c(X, C))

    def score(Xs, Cs):
        return dep_error(zr, fisher_z(partial_corr_c(Xs, Cs)), ALL_PAIRS)

    fl = []
    for s in range(5):
        r = np.random.default_rng(1000 + s)
        fl.append(score(np.column_stack([r.permutation(X[:, j]) for j in range(N_FEAT)]),
                        C[r.permutation(len(C))]))
    floor = float(np.mean(fl))

    variants = {
        "cdg": G_cdg,
        "permuted": isomorphic_permuted(G_cdg, np.random.default_rng(20260711)),
        "no_entangle": nx.empty_graph(N_FEAT),
    }

    print("=" * 88)
    print(f"WP-2 probe — does the real-data model ever beat the floor?  ({TOTAL_STEPS} steps)")
    print("=" * 88)
    print(f"  cohort n={len(df):,}   ·   floor = {floor:.4f}   <- zero-dependency model")
    print("  A curve that never crosses below the floor means nothing has learned dependency,")
    print("  and the full WP-2 run cannot be interpreted no matter how the variants rank.")
    print()
    hdr = "  ".join(f"{s:>6}" for s in range(EVERY, TOTAL_STEPS + 1, EVERY))
    print(f"  {'variant':<12} {hdr}")
    print("  " + "-" * (12 + len(hdr) + 2))

    out = {"floor": floor}
    for name, Gv in variants.items():
        t0 = time.time()
        cfg = Cfg(seed=0, copula=True, batch_critic=True, lr_q=5e-3)
        curve = train_scored(X, C, list(Gv.edges()), cfg, score, TOTAL_STEPS, EVERY)
        out[name] = [(s, float(v)) for s, v in curve]
        cells = "  ".join(f"{v:>6.4f}" for _, v in curve)
        print(f"  {name:<12} {cells}   ({time.time()-t0:.0f}s)", flush=True)

    print()
    cdg = np.array([v for _, v in out["cdg"]])
    steps = np.array([s for s, _ in out["cdg"]])
    below = steps[cdg < floor]
    print("=" * 88)
    if len(below):
        print(f"  CDG crosses below the floor at step {below[0]}  "
              f"(best {cdg.min():.4f} at step {steps[int(cdg.argmin())]}, floor {floor:.4f})")
        print("  -> the full WP-2 run is interpretable; use at least this many steps.")
    else:
        print(f"  CDG NEVER beats the floor in {TOTAL_STEPS} steps (best {cdg.min():.4f} "
              f"vs floor {floor:.4f}).")
        print("  -> the full WP-2 run as configured is DEAD. Steps are not the problem;")
        print("     something about the real data breaks what worked on the teacher.")
    print("=" * 88)

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "wp2_probe.json"
    import json
    p.write_text(json.dumps(out, indent=2))
    print(f"  saved: {p}")


if __name__ == "__main__":
    main()

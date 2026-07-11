"""Why does WGAN-GP fail to learn the entanglement — the learning-rate hypothesis.

Background
----------
ceiling_joint.py (direct optimization, no GAN):
    aligned  0.0130   permuted 0.0491   <- the circuit CAN represent the pattern,
                                           and the CDG beats a permutation by ~4x
train.py (WGAN-GP):
    aligned  0.1359   permuted 0.1346   <- the gap vanishes, and the true-edge error
                                           equals the floor

The decisive difference between the two experiments:
    ceiling_joint : Adam lr = 0.05    on the quantum parameters
    train.py      : Adam lr_g = 5e-5  on every parameter      <- 1000x smaller

Quantum angles are in radians. For gamma to create meaningful entanglement it has to move
O(0.1..1) rad, but at lr=5e-5 over 3000 steps the theoretical maximum displacement is
0.15 rad. In practice it is frozen near its initialization. The learning rate was tuned
for the ~2000 classical head weights and then applied unchanged to the 19 quantum angles.

What this script measures
-------------------------
  [1] How far gamma actually moves during WGAN-GP training (displacement from its init)
  [2] With a larger lr on the quantum parameters only: (a) does gamma move,
      (b) does the model learn dependency, (c) do aligned and permuted separate

Note: this is NOT putting the evaluation metric into the loss. The loss stays pure
      WGAN-GP. Only the per-parameter-group learning rate changes -> no circular reasoning.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_synthetic import N_FEAT, N_SYN, N_TRAIN, teacher_data, teacher_graph  # noqa: E402
from graphs import isomorphic_permuted  # noqa: E402
from model import CDGQGAN, Critic  # noqa: E402
from train import DEVICE, Cfg, generate, gradient_penalty, partial_corr_matrix  # noqa: E402

Z = lambda R: np.arctanh(np.clip(R, -0.999, 0.999))  # noqa: E731
SEEDS = [0, 1, 2]


def train_split_lr(X, C, edges, cfg: Cfg, lr_q: float):
    """Identical WGAN-GP to train.train(), except the quantum parameters get their own lr.

    Returns: (G, mean |delta gamma|, mean |gamma_init|)
    """
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = CDGQGAN(n, edges, cfg.depth, C.shape[1], cfg.head_width, seed=cfg.seed).to(DEVICE)
    D = Critic(n, C.shape[1]).to(DEVICE)

    q_params = list(G.quantum.parameters())
    h_params = list(G.heads.parameters())
    og = torch.optim.Adam([{"params": q_params, "lr": lr_q},
                           {"params": h_params, "lr": cfg.lr_g}], betas=(0.0, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9))

    gamma0 = G.quantum.gamma.detach().clone()

    def sample_real(b):
        i = torch.randint(0, N, (b,), device=DEVICE)
        return Xt[i], Ct[i]

    def sample_z(b):
        return 2 * torch.rand(b, n, device=DEVICE) - 1

    for _ in range(cfg.steps):
        for _ in range(cfg.critic_steps):
            xr, c = sample_real(cfg.batch)
            with torch.no_grad():
                xf = G(sample_z(cfg.batch), c)
            ld = (D(xf, c).mean() - D(xr, c).mean()
                  + cfg.lambda_gp * gradient_penalty(D, xr, xf, c))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()

        _, c = sample_real(cfg.batch)
        lg = -D(G(sample_z(cfg.batch), c), c).mean()
        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()

    move = (G.quantum.gamma.detach() - gamma0).abs().mean().item()
    return G, move, gamma0.abs().mean().item()


def main() -> None:
    rng = np.random.default_rng(20260711)
    G_true = teacher_graph(rng)
    X, C, _ = teacher_data(G_true, N_TRAIN, rng)
    G_perm = isomorphic_permuted(G_true, np.random.default_rng(7))
    zr = Z(partial_corr_matrix(X))

    def err(Xs):
        zs = Z(partial_corr_matrix(Xs))
        e = [abs(zs[i, j] - zr[i, j]) for i, j in G_true.edges()]
        nn = [abs(zs[i, j] - zr[i, j]) for i in range(N_FEAT) for j in range(i + 1, N_FEAT)
              if not G_true.has_edge(i, j)]
        return float(np.mean(e + nn)), float(np.mean(e)), float(np.mean(nn))

    print("=" * 92)
    print("Learning-rate hypothesis — is WGAN-GP moving gamma at all?")
    print("=" * 92)
    print("  floor (zero dependency) = 0.0648 / edges 0.3676 / non-edges 0.0078")
    print("  direct-optimization ceiling: aligned 0.0130,  permuted 0.0491")
    print()
    print(f"  {'lr_q':>8} {'graph':<10} {'mean |dgamma|':>14} {'120pair':>9} {'edge19':>9} {'nonedge101':>11}")
    print("  " + "-" * 74)

    for lr_q in (5e-5, 5e-3, 5e-2):
        for name, Gv in (("aligned", G_true), ("permuted", G_perm)):
            res, mv = [], []
            t0 = time.time()
            for s in SEEDS:
                cfg = Cfg(depth=1, seed=s)
                Gm, move, _ = train_split_lr(X, C, list(Gv.edges()), cfg, lr_q)
                Xs = generate(Gm, C, N_SYN, seed=s)
                res.append(err(Xs))
                mv.append(move)
            r = np.mean(res, 0)
            tag = "  (current setting)" if lr_q == 5e-5 else ""
            print(f"  {lr_q:>8.0e} {name:<10} {np.mean(mv):>14.4f} "
                  f"{r[0]:>9.4f} {r[1]:>9.4f} {r[2]:>11.4f}   ({time.time()-t0:.0f}s){tag}",
                  flush=True)
        print()

    print("=" * 92)
    print("  How to read this:")
    print("    |dgamma| near zero        -> the entangling angles are frozen. The learning rate is the cause.")
    print("    'edge19' falls below 0.368 (the floor) as lr_q rises -> the model has started learning entanglement.")
    print("    aligned then drops below permuted -> the confirmatory experiment works.")


if __name__ == "__main__":
    main()

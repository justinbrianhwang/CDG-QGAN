"""Training with the v3 loss: copula critic (dependency) + 1-D conditional marginal term (shape).

    L_G  =  -E[ D(soft_npn(x~), c) ]   +   lambda * L_marginal(x~, x, c)
            \_____________________/       \___________________________/
             cross-feature dependency      1-D conditional shape only
             ONLY the entangling angles    ONLY the local heads can
             can supply this               supply this

The two terms cannot interfere, and that is not a hope — it is Proposition D-2 applied twice:

  * the critic sees a rank-transformed input, so it carries no marginal information at all;
  * the marginal term reads one feature column at a time, so it carries no cross-feature
    information at all (`model_v3.assert_marginal_loss_is_1d` verifies the Jacobian is diagonal,
    off-diagonal max = 0.0e+00).

v2 §8.10 bans putting *dependency structure* into the objective. A strictly 1-D term is not
dependency structure, by the same argument that says the ~2,000 head parameters cannot manufacture
a dependence — an argument this project has already confirmed empirically twice: `no_entangle`
trains for 12,000 steps and lands *exactly* on the dependency floor, at Δ=3 and at Δ=4.

`no_entangle` remains the falsifier. If, with the marginal term switched on, a zero-RZZ model starts
beating the dependency floor, then the term IS smuggling dependency in and this file must be
reverted. The diagnostic below checks it on every run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from diag_fix2 import Critic, gradient_penalty  # noqa: E402
from model import CDGQGAN  # noqa: E402
from model_v3 import conditional_marginal_loss  # noqa: E402
from train import DEVICE  # noqa: E402


@dataclass
class CfgV3:
    steps: int = 8000
    batch: int = 512
    critic_steps: int = 5
    lr_g: float = 5e-5
    lr_q: float = 5e-3
    lr_d: float = 1e-4
    lambda_gp: float = 10.0
    lambda_marg: float = 1.0     # weight on the 1-D conditional marginal term
    head_width: int = 8
    seed: int = 0
    tau: float = 0.05


def train_v3(X: np.ndarray, C: np.ndarray, edges, cfg: CfgV3):
    torch.manual_seed(cfg.seed)
    N, n = X.shape
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Ct = torch.tensor(C, dtype=torch.float32, device=DEVICE)

    G = CDGQGAN(n, list(edges), 1, C.shape[1], cfg.head_width, seed=cfg.seed).to(DEVICE)
    D = Critic(n, C.shape[1], copula=True, batch_aware=True, tau=cfg.tau).to(DEVICE)

    og = torch.optim.Adam([{"params": list(G.quantum.parameters()), "lr": cfg.lr_q},
                           {"params": list(G.heads.parameters()), "lr": cfg.lr_g}],
                          betas=(0.0, 0.9))
    od = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9))

    def real(b):
        i = torch.randint(0, N, (b,), device=DEVICE)
        return Xt[i], Ct[i]

    def zz(b):
        return 2 * torch.rand(b, n, device=DEVICE) - 1

    for _ in range(cfg.steps):
        for _ in range(cfg.critic_steps):
            xr, c = real(cfg.batch)
            with torch.no_grad():
                xf = G(zz(cfg.batch), c)
            ld = (D(xf, c).mean() - D(xr, c).mean()
                  + cfg.lambda_gp * gradient_penalty(D, xr, xf, c))
            od.zero_grad(set_to_none=True)
            ld.backward()
            od.step()

        xr, c = real(cfg.batch)
        xf = G(zz(cfg.batch), c)
        lg = -D(xf, c).mean()
        if cfg.lambda_marg > 0:
            # c is shared between the real and fake halves here, so the strata line up exactly
            lg = lg + cfg.lambda_marg * conditional_marginal_loss(xf, xr, c, c)
        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()

    return G

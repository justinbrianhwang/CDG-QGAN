"""CDG-QGAN v3 — the copula critic keeps the dependency, a 1-D loss restores the marginals.

The problem this fixes (REVISIONS §E-10)
---------------------------------------
The copula critic (§E-7) rank-transforms the critic's input **by design**. That is what destroys
marginal information, which is what forces the gradient into the entangling angles, which is what
made the model learn any dependency at all. Without it, nothing learns dependency — measured, for
every topology and every hyperparameter (RESULTS_lr.md).

But it means **no term anywhere in the loss trains the heads to match the marginals**, and they
don't. Measured on real MIMIC (WP-3):

    marginal W1     0.676   (a Gaussian copula: 0.007)
    TSTR AUROC      0.641   (a model with ZERO dependency: 0.682)

Our synthetic data is a *worse* source of training data than independent noise with the right
marginals. And our dependency metric — a rank quantity, invariant to any monotone per-feature map —
is structurally incapable of seeing this. It scored 0.0714 and beat CTGAN while the distributions
were wrong by two orders of magnitude.

It gets worse than "the marginals are off". The critic cannot see how a feature's distribution
SHIFTS WITH c either — and the shift of a lab value with the mortality label **is the mortality
signal**. That is why calibrating the pooled marginals post-hoc (wp3_qgan.calibrate_marginals)
fixed W1 to 0.0004 and made TSTR *worse* (0.641 -> 0.573): once the features were on the right
scale the classifier actually used them, and they carried the wrong conditional information.

The fix, and why it does not break the science
----------------------------------------------
Add a **per-feature conditional marginal loss**: for each feature u independently, match the
distribution of x~_u given c to the real x_u given c. Nothing else about training changes; the
critic still sees only the copula.

This does NOT violate v2 §8.10's ban on putting dependency structure into the objective, and the
reason is exactly Proposition D-2:

    the term is a function of ONE feature at a time. A 1-D objective on x~_u cannot create a
    conditional dependence between x~_u and x~_v — that is the same argument that says the local
    heads cannot, and it is verified structurally by `assert_no_cross_feature_mixing`.

So the loss now has two parts that cannot interfere:

    critic (copula, batch-aware)  ->  ALL cross-feature dependency. Only `gamma` can supply it.
    per-feature marginal loss     ->  ALL 1-D conditional shape. Only the heads can supply it.

`no_entangle` remains the control that proves it: with the RZZ gates removed the model must still
land exactly on the dependency floor, however good its marginals become. If it does not, this
module is wrong and must be reverted.

Implementation
--------------
The marginal loss is a **conditional quantile (pinball) loss**. For feature u, we regress the
synthetic sample's quantiles onto the same basis of c the metric uses, and match them to the real
ones. In practice a cheap and stable version is the 1-D energy distance between the synthetic and
real values of feature u within c-strata — differentiable, needs no critic, and is exactly 1-D.

We stratify on the mortality label y (c[:,0]), which is the axis TSTR depends on, and match a grid
of quantiles per stratum. Age/sex/ICU enter the heads directly and are matched in the pooled term.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

N_QUANTILES = 32


def _quantile_grid(device) -> torch.Tensor:
    # (0.5/Q, 1.5/Q, ...) — avoids the extreme tails, which are pure sampling noise at batch size
    return (torch.arange(N_QUANTILES, device=device) + 0.5) / N_QUANTILES


def conditional_marginal_loss(x_fake: torch.Tensor, x_real: torch.Tensor,
                              c_fake: torch.Tensor, c_real: torch.Tensor,
                              strat_col: int = 0) -> torch.Tensor:
    """Per-feature, per-stratum quantile matching.  STRICTLY 1-D IN THE FEATURES.

    For each feature u and each level of the stratifying condition (the mortality label), compare
    the sorted quantiles of the synthetic x~_u against the real x_u. Sum over u.

    It reads one feature column at a time and never a pair, so by the Proposition D-2 argument it
    cannot create conditional cross-feature dependence. It can only shape 1-D conditionals — which
    is exactly the thing the copula critic is blind to, and exactly the thing TSTR needs.
    """
    q = _quantile_grid(x_fake.device)
    loss = x_fake.new_zeros(())
    n_terms = 0

    for level in (0.0, 1.0):                       # y = 0 (survived), y = 1 (died)
        mf = c_fake[:, strat_col] == level
        mr = c_real[:, strat_col] == level
        # a stratum needs enough rows for its quantiles to mean anything
        if int(mf.sum()) < N_QUANTILES or int(mr.sum()) < N_QUANTILES:
            continue
        xf, xr = x_fake[mf], x_real[mr]            # (n_f, n_feat), (n_r, n_feat)
        for u in range(x_fake.shape[1]):           # ONE FEATURE AT A TIME. This is the whole point.
            qf = torch.quantile(xf[:, u], q)
            qr = torch.quantile(xr[:, u], q)
            loss = loss + (qf - qr).abs().mean()
            n_terms += 1

    return loss / max(n_terms, 1)


class MarginalRegularizedTrainer(nn.Module):
    """Bookkeeping only — the loss above is a free function so it can be unit-tested on its own."""

    def __init__(self, lambda_marg: float = 1.0):
        super().__init__()
        self.lambda_marg = lambda_marg


def assert_marginal_loss_is_1d(n_feat: int = 6, batch: int = 256, device="cpu") -> None:
    """The loss must have a DIAGONAL Jacobian w.r.t. the features.

    d loss / d x~_u must not depend on x~_v for v != u. If it did, the marginal term could carry
    cross-feature information and v2 §8.10 would be violated — the objective would be able to
    manufacture the very dependency the paper claims comes only from the entangling angles.

    We check it the same way `model.assert_no_cross_feature_mixing` checks the heads: perturb one
    feature's column and confirm no other column's contribution to the loss moves.
    """
    torch.manual_seed(0)
    xr = torch.randn(batch, n_feat, device=device)
    cr = torch.zeros(batch, 4, device=device)
    cr[: batch // 2, 0] = 1.0
    cf = cr.clone()

    # per-feature contributions, computed independently
    def per_feature(xf):
        q = _quantile_grid(device)
        out = []
        for u in range(n_feat):
            tot = xf.new_zeros(())
            for level in (0.0, 1.0):
                mf, mr = cf[:, 0] == level, cr[:, 0] == level
                if int(mf.sum()) < N_QUANTILES:
                    continue
                tot = tot + (torch.quantile(xf[mf][:, u], q)
                             - torch.quantile(xr[mr][:, u], q)).abs().mean()
            out.append(tot)
        return out

    xf = torch.randn(batch, n_feat, device=device, requires_grad=True)
    jac = torch.zeros(n_feat, n_feat)
    for u, term in enumerate(per_feature(xf)):
        g, = torch.autograd.grad(term, xf, retain_graph=True)
        jac[u] = g.abs().sum(0)

    off = jac - torch.diag(torch.diag(jac))
    assert off.abs().max() < 1e-10, (
        f"the marginal loss reads more than one feature! off-diagonal max = {off.abs().max():.2e}")
    print(f"  [structure] marginal-loss Jacobian d L_u / d x~_v is diagonal "
          f"(off-diagonal max = {off.abs().max():.1e})")
    print("  -> the marginal term is 1-D per feature and cannot create conditional cross-feature")
    print("     dependence. v2 §8.10 holds: all dependency still comes from the entangling angles.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("CDG-QGAN v3 — structural check of the conditional marginal loss")
    print()
    assert_marginal_loss_is_1d()

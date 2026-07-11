"""The dependency metric, computed in the space the CDG is actually defined in.

The estimator mismatch [review E-3]
-----------------------------------
The CDG is defined as a partial correlation **conditional on c** (v2 §7.2, review A-3):
`build_cdg.py` residualizes on c before it estimates anything. But `train.dependency_error`
measures the **unconditional** partial correlation. The generator feeds c into every qubit
angle and every head, so whatever correlation c induces shows up as a false positive on
**every pair** — and it did: a model with *zero* entangling gates still produced 0.0859 of
spurious non-edge dependency.

Two estimators, one target. That is not a benchmark artifact; it is present verbatim in the
real MIMIC pipeline, and it would have contaminated all 90 runs of WP-2.

What this module does
---------------------
`partial_corr_c(X, C)` is the estimator the CDG is defined by:

    1. nonparanormal transform (rank -> inverse normal)   [review A-4]
       — makes the metric invariant to the monotone local heads, so it measures the copula
         and not "who fit the heavy tails better"
    2. residualize on a basis expansion of c
       — c enters the circuit through the angles, i.e. **nonlinearly**. Linear residualization
         removed only half the false positives (0.0926 -> 0.0532). The basis adds squares and
         the y x age interaction, which is what the angle encoding actually generates.
    3. partial correlation of the residuals

Use this for every dependency number from here on. `train.partial_corr_matrix` (unconditional)
is kept only to reproduce the earlier results it produced.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

RIDGE = 1e-3


def npn(X: np.ndarray) -> np.ndarray:
    """rank -> inverse normal. Invariant to any monotone per-feature map."""
    n = X.shape[0]
    return np.column_stack([norm.ppf((rankdata(X[:, j]) - 0.5) / n) for j in range(X.shape[1])])


def cond_basis(C: np.ndarray) -> np.ndarray:
    """Design matrix for residualizing on c.

    c = (mortality, age, sex, icu_type). The generator puts c into RY/RZ angles, so its effect
    on the output is a smooth nonlinear function of c, not a linear one. A linear design
    matrix leaves half of it behind (measured). Squares and the mortality x age interaction
    cover the leading terms.
    """
    n, k = C.shape
    cols = [np.ones(n)]
    cols += [C[:, j] for j in range(k)]
    cols += [C[:, j] ** 2 for j in range(k)]
    for a in range(k):
        for b in range(a + 1, k):
            cols.append(C[:, a] * C[:, b])
    return np.column_stack(cols)


def partial_corr_c(X: np.ndarray, C: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Partial correlation conditional on c, in nonparanormal space. The CDG's own estimator."""
    Xn = npn(X)
    D = cond_basis(C)
    beta, *_ = np.linalg.lstsq(D, Xn, rcond=None)
    R = Xn - D @ beta
    S = np.corrcoef(R, rowvar=False)
    P = np.linalg.inv(S + ridge * np.eye(S.shape[0]))
    d = np.sqrt(np.diag(P))
    M = -P / np.outer(d, d)
    np.fill_diagonal(M, 1.0)
    return M


def fisher_z(R: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(R, -0.999, 0.999))


def dep_error(z_real: np.ndarray, z_syn: np.ndarray, pairs) -> float:
    """Mean absolute Fisher-z difference over the given pairs."""
    return float(np.mean([abs(z_syn[i, j] - z_real[i, j]) for i, j in pairs]))

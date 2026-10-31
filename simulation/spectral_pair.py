"""Stationary weighting of one estimated transition matrix.

Use this function separately on each original, bootstrap, or null matrix.
It requires an irreducible empirical matrix and never adds pseudocounts.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve, svd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def stationary_spectral_pair(P, gamma=0.9):
    """Return mu, weighted P, and the smallest nonstationary singular mode.

    P must already be row-normalized.  mu is solved from this P, not from
    empirical state frequencies or a separate sample.  The function returns
    f=M^{-1/2}v and h=(I-gamma*P)f/sigma, both in state coordinates.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or len(P) < 2:
        raise ValueError("P must be a square matrix with at least two states.")
    if not (0 < gamma < 1):
        raise ValueError("gamma must lie strictly between zero and one.")
    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("P must contain finite nonnegative probabilities.")
    if not np.allclose(P.sum(axis=1), 1, rtol=0, atol=1e-11):
        raise ValueError("Normalize nonempty count rows before calling this function.")
    if connected_components(csr_matrix(P > 0), directed=True,
                            connection="strong", return_labels=False) != 1:
        raise ValueError("P is not irreducible. Inspect communicating classes; "
                         "do not silently add pseudocounts or invent stationary weights.")
    n = len(P)
    lhs = np.eye(n) - P.T
    lhs[-1, :] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    mu = solve(lhs, rhs)
    residual = float(np.max(np.abs(mu @ P - mu)))
    if np.min(mu) <= 0 or residual > 1e-10:
        raise ArithmeticError("Positive stationary distribution was not resolved numerically.")
    s = np.sqrt(mu)
    weighted_P = s[:, None] * P / s[None, :]
    A = np.eye(n) - gamma * weighted_P
    _, singular_values, vt = svd(A)
    v = vt[-2].copy()
    # Fix only the arbitrary sign; do not alter the singular direction.
    if v[np.argmax(np.abs(v))] < 0:
        v *= -1
    sigma = float(singular_values[-2])
    f = v / s
    h = (f - gamma * (P @ f)) / sigma
    if abs(singular_values[-1] - (1 - gamma)) > 1e-9:
        raise ArithmeticError("Discounted stationary floor check failed.")
    return dict(mu=mu, weighted_P=weighted_P, singular_values=singular_values,
                sigma=sigma, v=v, f=f, h=h,
                epsilon=float(np.linalg.norm(v - weighted_P @ v)),
                stationarity_residual_inf=residual)

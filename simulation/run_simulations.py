## python run_simulations.py --outdir results

#!/usr/bin/env python3
"""Run the two compact experiments accompanying the manuscript review."""
from __future__ import annotations
import os

# Set before importing NumPy/SciPy. Avoid oversubscription on small matrices.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import matplotlib
from scipy.special import logsumexp
from spectral_pair import stationary_spectral_pair
from rmt_experiments import run_rmt
from irl_experiments import run_irl


def soft_optimum(Pa, reward, gamma, tau):
    """Finite discounted MCE optimum; Pa has shape (actions, states, states)."""
    V = np.zeros(len(reward))
    for _ in range(20000):
        Q = reward[None, :] + gamma * np.einsum("aij,j->ai", Pa, V)
        nxt = tau * logsumexp(Q / tau, axis=0)
        if np.max(np.abs(nxt - V)) < 2e-13:
            V = nxt
            break
        V = nxt
    else:
        raise ArithmeticError("Soft Bellman iteration failed to converge.")
    Q = reward[None, :] + gamma * np.einsum("aij,j->ai", Pa, V)
    logpi = (Q / tau - logsumexp(Q / tau, axis=0)[None, :]).T
    return V, logpi


def check_general_response(seed):
    """One nonsymmetric multi-action sanity check, not another experiment.

    Compare the analytic reward-to-policy score with a central difference
    through an independent soft Bellman solve; check normalization and the
    two Fisher bounds on the same stationary pair.
    """
    rng = np.random.default_rng(seed)
    n, actions, gamma, tau = 9, 3, 0.9, 1.0
    Pa = rng.exponential(size=(actions, n, n))
    Pa /= Pa.sum(axis=2, keepdims=True)
    reward = rng.normal(scale=0.3, size=n)
    _, logpi = soft_optimum(Pa, reward, gamma, tau)
    pi = np.exp(logpi)
    P = np.einsum("ia,aij->ij", pi, Pa)
    pair = stationary_spectral_pair(P, gamma)
    mu, h, v = pair["mu"], pair["h"], pair["v"]
    sigma, eps, Pt = pair["sigma"], pair["epsilon"], pair["weighted_P"]
    dV = np.linalg.solve(np.eye(n) - gamma * P, h)
    response = np.einsum("aij,j->ai", Pa, dV).T
    score = gamma / tau * (response - (P @ dV)[:, None])
    step = 1e-4
    _, lp_plus = soft_optimum(Pa, reward + step * h, gamma, tau)
    _, lp_minus = soft_optimum(Pa, reward - step * h, gamma, tau)
    numeric_score = (lp_plus - lp_minus) / (2 * step)
    fisher = float(np.sum(mu[:, None] * pi * score ** 2))
    loss = 1 - np.dot(Pt @ v, Pt @ v)
    bound1 = float(gamma ** 2 * loss / (tau ** 2 * sigma ** 2))
    bound2 = float(gamma / (tau ** 2 * (1 - gamma)) *
                   (1 - (1 - gamma) ** 2 / sigma ** 2))
    identity_error = abs(sigma ** 2 - ((1 - gamma) ** 2 + gamma * eps ** 2 +
                                     gamma * (1 - gamma) * loss))
    score_error = float(np.max(np.abs(numeric_score - score)))
    if score_error > 2e-7 or identity_error > 1e-11:
        raise ArithmeticError("General local-response or spectral identity check failed.")
    if abs(np.dot(mu, h * h) - 1) > 1e-11 or abs(np.dot(mu, h)) > 1e-10:
        raise ArithmeticError("Paired reward is not centered and normalized.")
    if fisher > bound1 + 1e-11 or bound1 > bound2 + 1e-11:
        raise ArithmeticError("Fisher information bound check failed.")
    return dict(states=n, actions=actions, score_central_difference_step=step,
                max_score_difference=score_error,
                floor_identity_error=float(identity_error),
                stationary_residual=pair["stationarity_residual_inf"],
                paired_reward_norm=float(np.sqrt(np.dot(mu, h * h))),
                paired_reward_mean=float(np.dot(mu, h)),
                sigma=sigma, epsilon=eps, fisher=fisher,
                stationary_fisher_bound=bound1, spectral_fisher_bound=bound2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small smoke run; not paper settings.")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Checking a general multi-action local response...", flush=True)
    checks = check_general_response(args.seed + 2)
    print("Running RMT ensembles...", flush=True)
    rmt = run_rmt(args.outdir, seed=args.seed, quick=args.quick)
    print("Running finite-demonstration preference recovery and transfer...", flush=True)
    irl = run_irl(args.outdir, seed=args.seed + 1, quick=args.quick)
    summary = dict(seed=args.seed, quick=args.quick,
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                   general_response_checks=checks, rmt=rmt, irl=irl,
                   runtime_seconds=time.perf_counter() - started)
    (args.outdir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed. Outputs: {args.outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

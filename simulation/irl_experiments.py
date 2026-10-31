"""Two-room finite-candidate maximum-likelihood IRL mechanism experiment.

The true reward is r=(beta,-beta).  In the paper's raw-contrast family
r_theta=(beta+theta,-beta-theta), the candidates are theta=0 and theta=-2*beta.
Policy evaluation takes the absolute reward amplitude beta+theta.  This is
not an implementation or performance evaluation of AIRL.  The observed data
consist of N actual K-step state/action/next-state trajectories.  Closed-form
binomial error probabilities provide an independent Monte Carlo check.

The Fisher calculation uses the paired normalized direction at the true
reward, held fixed during differentiation.  It is distinct from the raw
contrast h=(1,-1) used to index the two reward candidates.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import expit, xlogy
from scipy.stats import binom


GAMMA = 0.9
TAU = 1.0
BETA = 1.0
THETA_TRUE = 0.0
THETA_OPPOSITE = -2.0 * BETA
P_TARGET = 0.4
N_TRAJECTORIES = 20


def _policy(p: float, reward_amplitude: float) -> np.ndarray:
    """Policy for (amplitude,-amplitude); rows C,H, columns stay,cross.

    In the paper's raw-contrast family, reward_amplitude = BETA + theta.
    """
    d = 1.0 - GAMMA + GAMMA * p
    q = float(expit(2.0 * GAMMA * p * reward_amplitude / (TAU * d)))
    return np.array([[q, 1.0 - q], [1.0 - q, q]])


def _paired_reward_direction(q: float) -> np.ndarray:
    """Centered unit direction under mu=(q,1-q); freeze at the true reward."""
    return np.array([1.0 - q, -q]) / np.sqrt(q * (1.0 - q))


def _policy_value(p: float, policy: np.ndarray) -> np.ndarray:
    """Evaluate true reward + tau times causal action entropy."""
    crossing = p * policy[:, 1]
    transition = np.array(
        [[1.0 - crossing[0], crossing[0]],
         [crossing[1], 1.0 - crossing[1]]]
    )
    entropy = -np.sum(xlogy(policy, policy), axis=1)
    reward = np.array([BETA, -BETA])
    return np.linalg.solve(np.eye(2) - GAMMA * transition, reward + TAU * entropy)


def _wrong_policy_regret(p: float) -> float:
    """Uniform-initial-state regret of replanning with the wrong candidate."""
    oracle = _policy_value(p, _policy(p, BETA))
    wrong = _policy_value(p, _policy(p, -BETA))
    regret = float(np.mean(oracle - wrong))
    if regret < -1e-10:
        raise AssertionError("The oracle must dominate the wrong-candidate policy.")
    return max(regret, 0.0)


def _wrong_probability(p, decisions: int):
    p = np.asarray(p, dtype=float)
    d = 1.0 - GAMMA + GAMMA * p
    q = expit(2.0 * GAMMA * p * BETA / (TAU * d))
    # All budgets are even; ties are broken with a fair random coin.
    return binom.cdf(decisions // 2 - 1, decisions, q) + 0.5 * binom.pmf(
        decisions // 2, decisions, q
    )


def _simulate_trajectories(p: float, steps: int, replicates: int, rng):
    """Generate actual transitions; retain sufficient statistics per dataset.

    A preferred action means stay in C or attempt-cross in H.  Its probability
    q is independent of the current state, but the states still evolve using
    the sampled action and a separate Bernoulli crossing outcome.
    """
    q = _policy(p, BETA)[0, 0]
    shape = (replicates, N_TRAJECTORIES)
    states = rng.integers(0, 2, size=shape, dtype=np.int8)
    initial_c = np.sum(states == 0, axis=1)
    preferred_count = np.zeros(replicates, dtype=np.int64)
    crossing_count = np.zeros(replicates, dtype=np.int64)
    attempt_count = np.zeros(replicates, dtype=np.int64)
    c_occupation = np.zeros(replicates, dtype=np.int64)
    for _ in range(steps):
        preferred = rng.random(shape) < q
        # C=0 and H=1, so the preferred action has the same numeric state label.
        actions = np.where(preferred, states, 1 - states)
        crosses = (actions == 1) & (rng.random(shape) < p)
        next_states = np.where(crosses, 1 - states, states).astype(np.int8)
        preferred_count += preferred.sum(axis=1)
        crossing_count += crosses.sum(axis=1)
        attempt_count += (actions == 1).sum(axis=1)
        c_occupation += (states == 0).sum(axis=1)
        states = next_states
    decisions = N_TRAJECTORIES * steps
    tie = preferred_count == decisions // 2
    wrong = preferred_count < decisions // 2
    wrong = wrong | (tie & (rng.random(replicates) < 0.5))
    return {
        "preferred_count": preferred_count,
        "tie": tie,
        "wrong": wrong,
        "crossing_count": crossing_count,
        "attempt_count": attempt_count,
        "c_occupation": c_occupation,
        "initial_c": initial_c,
    }


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sanity_checks() -> dict:
    """Small direct Bellman and trajectory-KL checks, not a test suite."""
    from scipy.special import logsumexp
    from spectral_pair import stationary_spectral_pair

    max_bellman_residual = 0.0
    max_policy_difference = 0.0
    max_kl_difference = 0.0
    max_trajectory_kl_difference = 0.0
    max_reference_difference = 0.0
    max_paired_score_difference = 0.0
    max_paired_fisher_difference = 0.0
    max_parameter_mapping_difference = 0.0
    for p in (1e-4, 0.01, 0.3, P_TARGET):
        pi = _policy(p, BETA)
        value = _policy_value(p, pi)
        transition_cross = np.array([[1.0 - p, p], [p, 1.0 - p]])
        action_value = np.array([BETA, -BETA])[:, None] + GAMMA * np.column_stack(
            [value, transition_cross @ value]
        )
        soft_value = TAU * logsumexp(action_value / TAU, axis=1)
        soft_policy = np.exp((action_value - soft_value[:, None]) / TAU)
        max_bellman_residual = max(max_bellman_residual, float(np.max(abs(value - soft_value))))
        max_policy_difference = max(max_policy_difference, float(np.max(abs(pi - soft_policy))))
        opposite = _policy(p, -BETA)
        state_kl = np.sum(pi * np.log(pi / opposite), axis=1)
        x = GAMMA * p * BETA / (TAU * (1.0 - GAMMA + GAMMA * p))
        expected_kl = 2.0 * x * np.tanh(x)
        max_kl_difference = max(max_kl_difference, float(np.max(abs(state_kl - expected_kl))))

        # Resolve the stationary pair from the actual true-expert transition.
        transitions = (np.eye(2), transition_cross)
        transition = np.einsum("sa,asj->sj", pi, np.asarray(transitions))
        pair = stationary_spectral_pair(transition, GAMMA)
        q = float(pi[0, 0])
        mu = np.array([q, 1.0 - q])
        h_nt = _paired_reward_direction(q)
        resolved_h = pair["h"]
        if np.dot(resolved_h, h_nt) < 0:
            resolved_h = -resolved_h
        d = 1.0 - GAMMA + GAMMA * p
        max_reference_difference = max(
            max_reference_difference,
            float(np.max(abs(pair["mu"] - mu))),
            float(np.max(abs(resolved_h - h_nt))),
            abs(pair["sigma"] - d),
            abs(float(np.dot(mu, h_nt))),
            abs(float(np.dot(mu, h_nt**2)) - 1.0),
        )
        response = np.linalg.solve(np.eye(2) - GAMMA * transition, h_nt)
        score = GAMMA / TAU * np.column_stack(
            [(kernel - transition) @ response for kernel in transitions]
        )
        information_scale = GAMMA * p / (TAU * d)
        preferred = np.eye(2)
        expected_score = information_scale / np.sqrt(q * (1.0 - q)) * (preferred - q)
        max_paired_score_difference = max(
            max_paired_score_difference, float(np.max(abs(score - expected_score)))
        )
        fisher = float(np.sum(mu[:, None] * pi * score**2))
        max_paired_fisher_difference = max(
            max_paired_fisher_difference, abs(fisher - information_scale**2)
        )
        max_parameter_mapping_difference = max(
            max_parameter_mapping_difference,
            float(np.max(abs(_policy(p, BETA + THETA_TRUE) - pi))),
            float(np.max(abs(_policy(p, BETA + THETA_OPPOSITE) - opposite))),
        )

        # Enumerate every single-trajectory path for K=2, including next states.
        trajectory_kl = 0.0
        for s0 in range(2):
            for a0 in range(2):
                for s1 in range(2):
                    for a1 in range(2):
                        for s2 in range(2):
                            prob = 0.5 * pi[s0, a0] * transitions[a0][s0, s1] * pi[s1, a1] * transitions[a1][s1, s2]
                            if prob > 0:
                                trajectory_kl += prob * (
                                    np.log(pi[s0, a0] / opposite[s0, a0])
                                    + np.log(pi[s1, a1] / opposite[s1, a1])
                                )
        max_trajectory_kl_difference = max(max_trajectory_kl_difference, abs(trajectory_kl - 2.0 * expected_kl))

    checks = {
        "max_soft_bellman_residual": max_bellman_residual,
        "max_policy_formula_difference": max_policy_difference,
        "max_single_action_KL_difference": max_kl_difference,
        "max_K2_enumerated_trajectory_KL_difference": max_trajectory_kl_difference,
        "max_true_reward_stationary_pair_difference": max_reference_difference,
        "max_paired_normalized_score_difference": max_paired_score_difference,
        "max_paired_normalized_fisher_difference": max_paired_fisher_difference,
        "max_local_parameter_policy_mapping_difference": max_parameter_mapping_difference,
    }
    if max(checks.values()) > 1e-10:
        raise AssertionError(f"Two-room numerical consistency check failed: {checks}")
    return checks


def _plot_irl(rows, p_grid, budgets, outdir: Path, replicates: int) -> dict[str, str]:
    """Draw the combined three-panel figure and each panel separately."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dense_p = np.geomspace(p_grid[0], p_grid[-1], 400)
    dense_d = 1.0 - GAMMA + GAMMA * dense_p
    colors = ["#2468A2", "#D57A20", "#21856E"]

    def draw_a(ax):
        # Paired normalized direction at r=(BETA,-BETA), not the raw contrast.
        ax.loglog(GAMMA * dense_p,
                  GAMMA**2 * dense_p**2 / (TAU**2 * dense_d**2),
                  color=colors[0], lw=2.2, label="Local Fisher information")
        ax.loglog(GAMMA * dense_p,
                  GAMMA**2 * (2.0 * dense_p - dense_p**2) / (TAU**2 * dense_d**2),
                  color=colors[2], ls="--", lw=1.8, label="Upper bound in (14)")
        ax.loglog(GAMMA * dense_p,
                  GAMMA / (TAU**2 * (1.0 - GAMMA)) *
                  (1.0 - (1.0 - GAMMA)**2 / dense_d**2),
                  color=colors[1], ls=":", lw=2, label="Upper bound in (15)")
        ax.set_title("(a) Local Fisher information\nbound")
        ax.set_xlabel(r"$\sigma_{\mathrm{nt}}-(1-\gamma)$")
        ax.set_ylabel(r"Value")
        ax.legend(loc="lower right", fontsize=18)

    def draw_b(ax):
        for color, steps in zip(colors, budgets):
            decisions = N_TRAJECTORIES * steps
            subset = [row for row in rows if row["K"] == steps]
            ax.semilogx(dense_p, _wrong_probability(dense_p, decisions),
                        color=color, lw=1.9, label=f"NK = {decisions:,}")
            ax.errorbar([row["p_source"] for row in subset],
                        [row["preference_error_mc"] for row in subset],
                        yerr=[1.96 * row["preference_error_mc_se"] for row in subset],
                        fmt="o", ms=3.5, color=color, elinewidth=0.9, capsize=2)
        ax.axhline(0.5, color="#89929B", lw=0.8, ls=":")
        ax.set_ylim(-0.015, 0.78)
        ax.set_title("(b) Finite-candidate\npreference recovery")
        ax.set_xlabel("Source crossing probability p")
        ax.set_ylabel(r"Probability of recovering $-r$")
        ax.legend(loc="upper right", fontsize=18)

    def draw_c(ax):
        midbudget = N_TRAJECTORIES * 50
        x = GAMMA * dense_p * BETA / (TAU * dense_d)
        # Initial-state and transition factors cancel in the likelihood ratio.
        # Both rooms have action KL 2*x*tanh(x); tanh(x) <= x gives the bound.
        exact_kl = 2.0 * midbudget * x * np.tanh(x)
        upper_bound = 2.0 * midbudget * x**2
        ax.loglog(dense_p, exact_kl, color=colors[0], lw=2.2,
                  label="Exact KL")
        ax.loglog(dense_p, upper_bound, color=colors[1], ls="--", lw=1.8,
                  label="Upper bound")
        ax.set_xlim(float(p_grid[0]), float(p_grid[-1]))
        ax.set_title("(c) Trajectory-distribution KL\nand upper bound")
        ax.set_xlabel("Source crossing probability p")
        ax.set_ylabel("Trajectory-data KL divergence")
        ax.legend(loc="upper left", fontsize=18)
        ax.text(0.97, 0.08, "NK = 1,000", transform=ax.transAxes,
                ha="right", fontsize=18)

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 18,
        "axes.titlesize": 20, "axes.labelsize": 22,
        "xtick.labelsize": 19, "ytick.labelsize": 19,
        "legend.fontsize": 18, "axes.titlepad": 14, "axes.labelpad": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(17.2, 8.4))
        draw_a(axes[0])
        draw_b(axes[1])
        draw_c(axes[2])
        for ax in axes:
            ax.grid(which="major", color="#E4E8ED", lw=0.65)
            ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.07, right=0.95, bottom=0.20, top=0.86, wspace=0.46)
        fig.suptitle("Two-room mechanism: finite-candidate maximum-likelihood IRL",
                     fontsize=23, y=0.985)
        fig.text(0.5, 0.070,
                 f"beta = {BETA:g}; candidates: r = (beta, -beta), -r; gamma = 0.9; "
                 f"tau = 1; N = 20; {replicates:,} Monte Carlo datasets per setting.",
                 ha="center", fontsize=15.5)
        fig.text(0.5, 0.022,
                 "(a): paired normalized direction at r. (b): Monte Carlo +/-1.96 SE. "
                 "(c): exact KL (solid), upper bound (dashed).",
                 ha="center", fontsize=14.5, color="#4B5563")
        pdf = outdir / "fig_irl_mechanism.pdf"
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

        panel_files: dict[str, str] = {}
        for letter, draw in (("a", draw_a), ("b", draw_b), ("c", draw_c)):
            panel_fig = plt.figure(figsize=(7.0, 6.1))
            panel_ax = panel_fig.add_axes([0.22, 0.18, 0.75, 0.68])
            draw(panel_ax)
            panel_ax.grid(which="major", color="#E4E8ED", lw=0.65)
            panel_ax.set_axisbelow(True)
            panel_pdf = outdir / f"fig_irl_mechanism_panel_{letter}.pdf"
            panel_fig.savefig(panel_pdf, facecolor="white")
            plt.close(panel_fig)
            panel_files[f"panel_{letter}_pdf"] = str(panel_pdf)
    return {"figure_pdf": str(pdf), **panel_files}


def run_irl(outdir, seed: int = 20261015, quick: bool = False) -> dict:
    """Write CSV data and PDF figures; return JSON-safe metadata."""
    import platform

    import matplotlib
    import scipy

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    replicates = 150 if quick else 1000
    p_grid = np.geomspace(1e-4, 0.3, 7 if quick else 13)
    budgets = (10, 50, 250)
    target_regret = _wrong_policy_regret(P_TARGET)
    target_tv = float(abs(_policy(P_TARGET, BETA)[0, 0] - _policy(P_TARGET, -BETA)[0, 0]))
    rows, raw_rows = [], []
    checks = _sanity_checks()

    for p_index, p in enumerate(p_grid):
        p = float(p)
        d = 1.0 - GAMMA + GAMMA * p
        x = GAMMA * p * BETA / (TAU * d)
        q = float(expit(2.0 * x))
        h_nt = _paired_reward_direction(q)
        source_regret = _wrong_policy_regret(p)
        for steps in budgets:
            decisions = N_TRAJECTORIES * steps
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), p_index, steps, 731]))
            result = _simulate_trajectories(p, steps, replicates, rng)
            errors = int(result["wrong"].sum())
            mc_error = errors / replicates
            exact_error = float(_wrong_probability(p, decisions))
            mc_se = float(np.sqrt(mc_error * (1.0 - mc_error) / replicates))
            exact_mc_se = float(np.sqrt(exact_error * (1.0 - exact_error) / replicates))
            row = {
                "p_source": p,
                "p_target": P_TARGET,
                "gamma": GAMMA,
                "tau": TAU,
                "beta": BETA,
                "theta_true": THETA_TRUE,
                "theta_opposite": THETA_OPPOSITE,
                "N": N_TRAJECTORIES,
                "K": steps,
                "M": decisions,
                "replicates": replicates,
                "sigma_nontrivial": d,
                "discounted_floor": 1.0 - GAMMA,
                "sigma_minus_floor": GAMMA * p,
                "persistence_time_steps": 1.0 / p,
                "preferred_action_probability": q,
                "stationary_probability_C": q,
                "stationary_probability_H": 1.0 - q,
                "paired_direction_C": float(h_nt[0]),
                "paired_direction_H": float(h_nt[1]),
                "fisher_paired_at_true_per_action": GAMMA**2 * p**2 / (TAU**2 * d**2),
                "fisher_paired_at_true_dataset": decisions * GAMMA**2 * p**2 / (TAU**2 * d**2),
                "fisher_raw_contrast_at_true_per_action": 4.0 * q * (1.0 - q) * GAMMA**2 * p**2 / (TAU**2 * d**2),
                "stationary_fisher_upper_bound": GAMMA**2 * (2.0 * p - p**2) / (TAU**2 * d**2),
                "floor_fisher_upper_bound": GAMMA / (TAU**2 * (1.0 - GAMMA)) * (1.0 - (1.0 - GAMMA)**2 / d**2),
                "KL_plus_to_minus_exact": 2.0 * decisions * x * np.tanh(x),
                "KL_plus_to_minus_upper_bound": 2.0 * decisions * x**2,
                "preference_error_exact": exact_error,
                "preference_error_mc": mc_error,
                "preference_error_mc_se": mc_se,
                "preference_error_exact_mc_se": exact_mc_se,
                "error_count": errors,
                "tie_count": int(result["tie"].sum()),
                "tie_probability_exact": float(binom.pmf(decisions // 2, decisions, q)),
                "preferred_count_mean_mc": float(result["preferred_count"].mean()),
                "preferred_count_mean_exact": decisions * q,
                "preferred_count_mean_mc_se_exact": float(np.sqrt(decisions * q * (1.0 - q) / replicates)),
                "crossings_mean_mc": float(result["crossing_count"].mean()),
                "source_policy_max_state_TV": float(np.tanh(x)),
                "target_policy_max_state_TV": target_tv,
                "source_wrong_policy_regret": source_regret,
                "target_wrong_policy_regret": target_regret,
                "source_expected_regret_exact": exact_error * source_regret,
                "source_expected_regret_mc": mc_error * source_regret,
                "source_expected_regret_mc_se": mc_se * source_regret,
                "target_expected_regret_exact": exact_error * target_regret,
                "target_expected_regret_mc": mc_error * target_regret,
                "target_expected_regret_mc_se": mc_se * target_regret,
            }
            rows.append(row)
            if steps == 50:
                for rep in range(replicates):
                    raw_rows.append({
                        "p_source": p,
                        "N": N_TRAJECTORIES,
                        "K": steps,
                        "M": decisions,
                        "replicate": rep,
                        "preferred_count_B": int(result["preferred_count"][rep]),
                        "tie": int(result["tie"][rep]),
                        "wrong": int(result["wrong"][rep]),
                        "beta": BETA,
                        "theta_hat": THETA_OPPOSITE if result["wrong"][rep] else THETA_TRUE,
                        "reward_amplitude_hat": -BETA if result["wrong"][rep] else BETA,
                        "initial_C_count": int(result["initial_c"][rep]),
                        "C_state_action_count": int(result["c_occupation"][rep]),
                        "attempt_cross_count": int(result["attempt_count"][rep]),
                        "successful_cross_count": int(result["crossing_count"][rep]),
                    })

    # Under the true reward, preferred-action indicators are iid Bernoulli(q),
    # even though the states are dependent.  This score uses h_nt, held fixed
    # at r, rather than the raw contrast that defines the candidate indices.
    fisher_replicates = 1000 if quick else 5000
    fisher_rows = []
    fisher_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1909]))
    for p in p_grid:
        p = float(p)
        d = 1.0 - GAMMA + GAMMA * p
        information_scale = GAMMA * p / (TAU * d)
        q = float(_policy(p, BETA)[0, 0])
        h_nt = _paired_reward_direction(q)
        score_per_count = information_scale / np.sqrt(q * (1.0 - q))
        decisions = 1000
        reference_counts = fisher_rng.binomial(decisions, q, size=fisher_replicates)
        dataset_scores = score_per_count * (reference_counts - decisions * q)
        predicted_information = decisions * information_scale**2
        sample_variance = float(np.var(dataset_scores, ddof=1))
        fisher_rows.append({
            "p_source": p,
            "beta": BETA,
            "reference_reward_C": BETA,
            "reference_reward_H": -BETA,
            "perturbation_reference": 0.0,
            "fisher_direction": "paired_normalized_room_contrast",
            "preferred_action_probability": q,
            "paired_direction_C": float(h_nt[0]),
            "paired_direction_H": float(h_nt[1]),
            "M": decisions,
            "replicates": fisher_replicates,
            "score_mean_mc": float(dataset_scores.mean()),
            "score_mean_exact": 0.0,
            "score_mean_mc_se_exact": float(np.sqrt(predicted_information / fisher_replicates)),
            "score_sample_variance": sample_variance,
            "M_times_I_mu": predicted_information,
            "variance_ratio": sample_variance / predicted_information,
            "simulation_method": "Binomial(M,q) paired-normalized trajectory score at the true reward",
        })

    summary_csv = outdir / "irl_summary.csv"
    raw_csv = outdir / "irl_mle_raw.csv"
    fisher_csv = outdir / "irl_finite_fisher_check.csv"
    _write_csv(summary_csv, rows)
    _write_csv(raw_csv, raw_rows)
    _write_csv(fisher_csv, fisher_rows)

    figure_files = _plot_irl(rows, p_grid, budgets, outdir, replicates)
    middle_rows = [row for row in rows if row["K"] == 50]

    standardized = [abs(row["preference_error_mc"] - row["preference_error_exact"]) / row["preference_error_exact_mc_se"]
                    for row in rows if min(replicates * row["preference_error_exact"],
                                           replicates * (1.0 - row["preference_error_exact"])) >= 5]
    # This diagnostic is reported, not used as a brittle stochastic pass/fail gate.
    checks["max_MC_error_deviation_in_exact_SE_regular_regime"] = float(max(standardized, default=0.0))
    checks["MC_error_comparison_settings_with_at_least_5_expected_errors_and_successes"] = len(standardized)
    first_middle = middle_rows[0]
    return {
        "experiment": "Two-room finite-candidate maximum-likelihood IRL (not AIRL)",
        "seed": int(seed), "quick": bool(quick), "gamma": GAMMA, "tau": TAU,
        "beta": BETA, "theta_true": THETA_TRUE,
        "theta_opposite": THETA_OPPOSITE,
        "candidate_thetas": [THETA_TRUE, THETA_OPPOSITE],
        "true_reward": [BETA, -BETA], "opposite_reward": [-BETA, BETA],
        "candidate_reward_amplitudes": [BETA, -BETA],
        "raw_contrast_parameterization": "r_theta=(beta+theta,-beta-theta); h=(1,-1)",
        "policy_argument": "absolute reward amplitude = beta + theta",
        "versions": dict(python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        "p_source": [float(p) for p in p_grid], "p_target": P_TARGET,
        "N": N_TRAJECTORIES, "K": list(budgets),
        "M": [N_TRAJECTORIES * steps for steps in budgets], "mc_replicates": replicates,
        "initial_distribution": [0.5, 0.5], "tie_rule": "independent fair coin",
        "regret_definition": "Uniform-initial-state discounted true reward plus tau action entropy: oracle value minus learned-reward policy value",
        "target_wrong_policy_regret": target_regret,
        "target_policy_max_state_TV": target_tv,
        "narrowest_source_midbudget": {
            key: first_middle[key] for key in (
                "p_source", "M", "source_policy_max_state_TV", "preference_error_exact",
                "preference_error_mc", "source_expected_regret_exact", "target_expected_regret_exact"
            )
        },
        "sanity_checks": checks,
        "files": {"summary_csv": str(summary_csv), "midbudget_raw_csv": str(raw_csv),
                  "finite_fisher_check_csv": str(fisher_csv),
                  **figure_files},
        "finite_fisher_check": {
            "beta": BETA, "reference_reward": [BETA, -BETA],
            "perturbation_reference": 0.0,
            "direction": "paired_normalized_room_contrast",
            "direction_formula": "h_nt=(1-q,-q)/sqrt(q*(1-q))",
            "perturbation_family": "r + local_step*h_nt, with h_nt fixed at r",
            "M": 1000, "replicates": fisher_replicates,
            "variance_ratio_min": min(row["variance_ratio"] for row in fisher_rows),
            "variance_ratio_max": max(row["variance_ratio"] for row in fisher_rows),
            "simulation_method": "Binomial(M,q) paired-normalized trajectory score at the true reward",
        },
        "raw_data_scope": "Per-dataset sufficient statistics for M=1000; all other budgets retain MC aggregates and standard errors.",
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outputs/irl")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_irl(args.outdir, args.seed, args.quick), indent=2))

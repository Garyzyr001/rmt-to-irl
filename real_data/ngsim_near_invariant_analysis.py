## python ngsim_near_invariant_analysis.py "RECONSTRUCTED+trajectories-400-0415_NO+MOTORCYCLES.csv" --outdir results_ngsim --bootstrap 200 --null-reps 500

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NGSIM I-80 near-invariant / persistent-mode analysis

Reproduces the pilot analysis discussed in chat:

1) Build state features from reconstructed NGSIM trajectories
   x_t = (ego speed, leader gap proxy, relative speed)

2) Coarse-grain continuous states by quantile binning.

3) Estimate the expert-policy transition matrix by row-wise multinomial MLE:
       P_hat(i,j) = N_ij / N_i

4) Move to stationary L2(mu) geometry:
       P_tilde = M^{1/2} P_hat M^{-1/2}

5) Analyze
       A_tilde = I - gamma P_tilde
   and its smallest nontrivial singular mode.  Report the full spectrum in
   descending order: sigma_1 >= ... >= sigma_K, including the stationary
   sigma_K = 1-gamma.  The persistent nontrivial mode is sigma_{K-1}.

6) Compute the near-invariance defect
       epsilon = ||(I - P_tilde) v||_2

7) Vehicle-level block bootstrap.

8) No-temporal-dependence null calibration.

9) Robustness to temporal lag and state resolution.

10) Map the persistent mode back to physical traffic variables and
    report empirical cross-regime transition flow.

The default discount is 0.9 per second.  At transition lag h seconds,
gamma_h = 0.9 ** h, using the actual frame-rounded lag.  Changing gamma
requires recomputing the singular vectors and all mode-dependent statistics,
including epsilon, interpretation, cuts, bootstrap and null calibration.

Dependencies:
    pandas
    numpy
    scipy
    matplotlib

Example:
    python ngsim_near_invariant_analysis.py \
        "RECONSTRUCTED+trajectories-400-0415_NO+MOTORCYCLES.csv" \
        --outdir results_ngsim \
        --gamma-per-second 0.9 \
        --bootstrap 200 \
        --null-reps 500
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.linalg import svd


DEFAULT_GAMMA_PER_SECOND = 0.9
FRAME_RATE_HZ = 10.0


def validate_gamma(gamma: float) -> None:
    if not np.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be finite and strictly between 0 and 1.")


def resolve_discount(lag_seconds: float, gamma_per_second: float):
    """Use the same physical time interval for transitions and discounting."""
    validate_gamma(gamma_per_second)
    if not np.isfinite(lag_seconds) or lag_seconds <= 0:
        raise ValueError("lag must be a finite positive number of seconds.")
    lag_frames = int(round(lag_seconds * FRAME_RATE_HZ))
    if lag_frames < 1:
        raise ValueError("lag must round to at least one frame at 10 Hz.")
    effective_lag = lag_frames / FRAME_RATE_HZ
    gamma = gamma_per_second ** effective_lag
    validate_gamma(gamma)
    return {
        "requested_lag_seconds": lag_seconds,
        "lag_seconds": effective_lag,
        "lag_frames": lag_frames,
        "gamma_per_second": gamma_per_second,
        "gamma": gamma,
    }


def discount_diagnostics(sigma: float, gamma: float):
    """Report the scale relative to its lag-specific discount floor."""
    floor = 1.0 - gamma
    return {
        "discount_floor": floor,
        "sigma_over_discount_floor": sigma / floor,
        "sigma_excess_above_floor": sigma - floor,
    }


class TeeStream:
    """Keep the console and the saved run log in sync."""

    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


# ---------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------

def load_and_construct_features(csv_path: str) -> pd.DataFrame:
    """
    Construct:
        Ego_Speed
        Gap_Proxy = leader Local_Y - ego Local_Y
        Rel_Speed = leader speed - ego speed

    The uploaded reconstructed file does not contain a direct headway column,
    so Gap_Proxy is a longitudinal leader-distance proxy in the dataset's
    native distance units.
    """
    df = pd.read_csv(csv_path)

    required = {
        "Vehicle_ID", "Frame_ID", "Lane_ID", "Local_Y",
        "Mean_Speed", "Mean_Accel", "Vehicle_Length",
        "Vehicle_Class_ID", "Follower_ID", "Leader_ID"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    leader = df[
        ["Vehicle_ID", "Frame_ID", "Local_Y", "Mean_Speed", "Vehicle_Length"]
    ].rename(
        columns={
            "Vehicle_ID": "Leader_ID",
            "Local_Y": "Leader_Y",
            "Mean_Speed": "Leader_Speed",
            "Vehicle_Length": "Leader_Length",
        }
    )

    x = df.merge(
        leader,
        on=["Leader_ID", "Frame_ID"],
        how="left",
        validate="many_to_one",
    )

    # Keep observations whose stated leader is actually observed in the same frame.
    x = x.loc[x["Leader_Y"].notna()].copy()

    x["Ego_Speed"] = x["Mean_Speed"].astype(float)
    x["Gap_Proxy"] = x["Leader_Y"] - x["Local_Y"]
    x["Rel_Speed"] = x["Leader_Speed"] - x["Mean_Speed"]

    good = (
        np.isfinite(x["Ego_Speed"])
        & np.isfinite(x["Gap_Proxy"])
        & np.isfinite(x["Rel_Speed"])
        & (x["Gap_Proxy"] > 0)
    )
    x = x.loc[good].copy()

    return x


# ---------------------------------------------------------------------
# 2. Finite-state abstraction
# ---------------------------------------------------------------------

def add_quantile_states(x: pd.DataFrame, bins_per_axis: int = 4) -> pd.DataFrame:
    """
    Quantile discretization of:
        speed × gap × relative speed

    For bins_per_axis = 4, this gives 4^3 = 64 states.
    """
    y = x.copy()

    features = ["Ego_Speed", "Gap_Proxy", "Rel_Speed"]
    labels = []

    for col in features:
        b = pd.qcut(
            y[col],
            q=bins_per_axis,
            labels=False,
            duplicates="drop",
        )
        if b.nunique() != bins_per_axis:
            raise ValueError(
                f"{col}: qcut produced {b.nunique()} bins, "
                f"expected {bins_per_axis}."
            )
        labels.append(b.astype(int).to_numpy())

    b0, b1, b2 = labels
    K = bins_per_axis ** 3
    state = (
        b0 * bins_per_axis * bins_per_axis
        + b1 * bins_per_axis
        + b2
    )

    y["state"] = state.astype(int)

    if y["state"].min() < 0 or y["state"].max() >= K:
        raise RuntimeError("State encoding error.")

    return y


# ---------------------------------------------------------------------
# 3. Transition MLE
# ---------------------------------------------------------------------

def build_transitions(
    x_state: pd.DataFrame,
    lag_frames: int,
) -> pd.DataFrame:
    """
    Match each vehicle at frame t to the same vehicle at t + lag_frames.
    """
    cur = x_state[["Vehicle_ID", "Frame_ID", "state"]].copy()
    cur["Frame_ID_next"] = cur["Frame_ID"] + int(lag_frames)

    nxt = x_state[["Vehicle_ID", "Frame_ID", "state"]].rename(
        columns={
            "Frame_ID": "Frame_ID_next",
            "state": "state_next",
        }
    )

    tr = cur.merge(
        nxt,
        on=["Vehicle_ID", "Frame_ID_next"],
        how="inner",
        validate="one_to_one",
    )

    return tr[["Vehicle_ID", "state", "state_next"]]


def transition_count_matrix(
    transitions: pd.DataFrame,
    K: int,
    weights=None,
) -> np.ndarray:
    s0 = transitions["state"].to_numpy(dtype=int)
    s1 = transitions["state_next"].to_numpy(dtype=int)

    flat = s0 * K + s1

    if weights is None:
        counts = np.bincount(flat, minlength=K * K)
    else:
        counts = np.bincount(flat, weights=weights, minlength=K * K)

    return counts.reshape(K, K).astype(float)


def rowwise_mle(counts: np.ndarray) -> np.ndarray:
    row_sum = counts.sum(axis=1)

    if np.any(row_sum <= 0):
        bad = np.where(row_sum <= 0)[0]
        raise ValueError(
            f"At least one coarse state has no outgoing transitions: {bad.tolist()}. "
            "Reduce state resolution or change lag."
        )

    return counts / row_sum[:, None]


# ---------------------------------------------------------------------
# 4. Stationary weighted geometry
# ---------------------------------------------------------------------

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Left stationary vector mu^T P = mu^T.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))

    mu = np.real(eigvecs[:, idx])

    # Global eigenvector sign is arbitrary.
    if mu.sum() < 0:
        mu = -mu

    # Numerical cleanup.
    mu[np.abs(mu) < 1e-14] = 0.0

    if np.any(mu <= 0):
        # Power iteration is more robust if eig numerical signs become awkward.
        mu = np.ones(P.shape[0], dtype=float) / P.shape[0]
        for _ in range(100000):
            mu_new = mu @ P
            if np.max(np.abs(mu_new - mu)) < 1e-14:
                break
            mu = mu_new

    mu = mu / mu.sum()

    if np.any(mu <= 0):
        raise ValueError(
            "Stationary distribution has zero/negative mass. "
            "The empirical state graph may not be irreducible."
        )

    return mu


def weighted_operator(P: np.ndarray, mu: np.ndarray) -> np.ndarray:
    sqrt_mu = np.sqrt(mu)
    return (sqrt_mu[:, None] * P) / sqrt_mu[None, :]


# ---------------------------------------------------------------------
# 5. Persistent-mode analysis
# ---------------------------------------------------------------------

def analyze_transition_matrix(P: np.ndarray, gamma: float):
    """
    Returns the smallest NONTRIVIAL singular mode of
        A_tilde = I - gamma P_tilde.

    The trivial stationary mode has singular value exactly 1-gamma
    in stationary L2(mu) geometry.

    The minimizing vector v depends on gamma for a general nonreversible
    chain.  Epsilon and the Rayleigh quotient must therefore be recomputed
    at each gamma, even though their formulas have no explicit gamma.
    """
    validate_gamma(gamma)
    K = P.shape[0]
    mu = stationary_distribution(P)
    P_tilde = weighted_operator(P, mu)

    A_tilde = np.eye(K) - gamma * P_tilde

    U, s_desc, Vt = svd(A_tilde)
    order = np.argsort(s_desc)
    s = s_desc[order]

    # order[0] is the trivial stationary direction, sigma = 1-gamma.
    idx_nontrivial = order[1]
    v = Vt.T[:, idx_nontrivial].copy()

    sigma = float(s[1])
    epsilon = float(np.linalg.norm((np.eye(K) - P_tilde) @ v))
    rayleigh = float(v @ P_tilde @ v)

    # Map weighted-coordinate singular vector back to the state function.
    f = v / np.sqrt(mu)

    # Normalize f in L2(mu) for numerical clarity.
    f = f / np.sqrt(np.sum(mu * f * f))

    return {
        "gamma": gamma,
        **discount_diagnostics(sigma, gamma),
        "mu": mu,
        "P_tilde": P_tilde,
        "A_tilde": A_tilde,
        "singular_values_descending": s_desc,
        # Retained for existing numerical consumers; reporting is descending.
        "singular_values_ascending": s,
        "sigma_nontrivial": sigma,
        "epsilon_nontrivial": epsilon,
        "rayleigh_nontrivial": rayleigh,
        "v_weighted": v,
        "f_state": f,
    }


# ---------------------------------------------------------------------
# 6. Mode interpretation
# ---------------------------------------------------------------------

def state_summary(x_state: pd.DataFrame, K: int) -> pd.DataFrame:
    out = (
        x_state.groupby("state")
        .agg(
            mean_speed=("Ego_Speed", "mean"),
            mean_gap=("Gap_Proxy", "mean"),
            mean_rel_speed=("Rel_Speed", "mean"),
            observations=("state", "size"),
        )
        .reindex(range(K))
    )
    return out


def orient_mode_for_interpretation(f: np.ndarray, summary: pd.DataFrame):
    """
    Singular-vector sign is arbitrary.
    Choose sign so that the correlation with mean speed is positive.
    """
    corr = np.corrcoef(f, summary["mean_speed"].to_numpy())[0, 1]
    if corr < 0:
        f = -f
    return f


def empirical_cut_statistics(P: np.ndarray, mu: np.ndarray, f: np.ndarray):
    """
    Split states by sign of the persistent mode.

    This is a descriptive cut statistic. We do NOT require reversibility here.
    """
    neg = f < 0
    pos = ~neg

    mu_neg = float(mu[neg].sum())
    mu_pos = float(mu[pos].sum())

    Q_neg_pos = float(
        np.sum(mu[neg, None] * P[np.ix_(neg, pos)])
    )
    Q_pos_neg = float(
        np.sum(mu[pos, None] * P[np.ix_(pos, neg)])
    )

    p_pos_given_neg = Q_neg_pos / mu_neg
    p_neg_given_pos = Q_pos_neg / mu_pos

    conductance_like = Q_neg_pos / min(mu_neg, mu_pos)

    return {
        "mu_negative_side": mu_neg,
        "mu_positive_side": mu_pos,
        "stationary_flow_neg_to_pos": Q_neg_pos,
        "stationary_flow_pos_to_neg": Q_pos_neg,
        "P(pos | neg)": p_pos_given_neg,
        "P(neg | pos)": p_neg_given_pos,
        "conductance_like_cut": conductance_like,
    }


# ---------------------------------------------------------------------
# 7. Vehicle-level bootstrap
# ---------------------------------------------------------------------

def vehicle_bootstrap(
    transitions: pd.DataFrame,
    K: int,
    gamma: float,
    full_v: np.ndarray,
    B: int = 200,
    seed: int = 20260920,
) -> pd.DataFrame:
    """
    Block bootstrap by whole vehicle IDs.

    Each bootstrap sample resamples vehicles with replacement.
    All transitions contributed by a selected vehicle receive the corresponding
    multiplicity.
    """
    rng = np.random.default_rng(seed)

    vehicle_ids, vehicle_code = np.unique(
        transitions["Vehicle_ID"].to_numpy(),
        return_inverse=True,
    )
    n_vehicle = len(vehicle_ids)

    s0 = transitions["state"].to_numpy(dtype=int)
    s1 = transitions["state_next"].to_numpy(dtype=int)
    flat = s0 * K + s1

    rows = []

    for b in range(B):
        sampled = rng.integers(0, n_vehicle, size=n_vehicle)
        multiplicity = np.bincount(sampled, minlength=n_vehicle)

        transition_weights = multiplicity[vehicle_code]

        counts = np.bincount(
            flat,
            weights=transition_weights,
            minlength=K * K,
        ).reshape(K, K)

        # A bootstrap draw can very rarely empty a row at a finer discretization.
        if np.any(counts.sum(axis=1) == 0):
            continue

        P_b = rowwise_mle(counts)
        fit_b = analyze_transition_matrix(P_b, gamma)

        v_b = fit_b["v_weighted"]
        alignment = abs(float(np.dot(v_b, full_v)))

        rows.append(
            {
                "bootstrap": b,
                "gamma": gamma,
                "sigma": fit_b["sigma_nontrivial"],
                **discount_diagnostics(fit_b["sigma_nontrivial"], gamma),
                "epsilon": fit_b["epsilon_nontrivial"],
                "alignment_abs": alignment,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 8. No-temporal-dependence null
# ---------------------------------------------------------------------

def temporal_independence_null(
    counts_observed: np.ndarray,
    gamma: float,
    reps: int = 500,
    seed: int = 20260921,
) -> pd.DataFrame:
    """
    Null model:
        keep each source-state row sample size fixed,
        but draw all next states from one common pooled next-state distribution.

    This destroys state-to-state temporal dependence while preserving:
        - number of outgoing observations per row
        - pooled destination frequencies approximately
    """
    rng = np.random.default_rng(seed)

    K = counts_observed.shape[0]
    row_n = counts_observed.sum(axis=1).astype(int)
    pooled = counts_observed.sum(axis=0)
    pooled = pooled / pooled.sum()

    out = []

    for b in range(reps):
        C0 = np.vstack(
            [rng.multinomial(int(n), pooled) for n in row_n]
        ).astype(float)

        P0 = rowwise_mle(C0)
        fit0 = analyze_transition_matrix(P0, gamma)

        out.append(
            {
                "replicate": b,
                "gamma": gamma,
                "sigma_nontrivial": fit0["sigma_nontrivial"],
                **discount_diagnostics(fit0["sigma_nontrivial"], gamma),
            }
        )

    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# 9. One complete run
# ---------------------------------------------------------------------

def run_single(
    x_features: pd.DataFrame,
    bins_per_axis: int,
    lag_seconds: float,
    gamma_per_second: float,
):
    discount = resolve_discount(lag_seconds, gamma_per_second)

    x_state = add_quantile_states(
        x_features,
        bins_per_axis=bins_per_axis,
    )

    tr = build_transitions(x_state, lag_frames=discount["lag_frames"])

    K = bins_per_axis ** 3
    counts = transition_count_matrix(tr, K=K)
    P = rowwise_mle(counts)

    fit = analyze_transition_matrix(P, discount["gamma"])

    return {
        "x_state": x_state,
        "transitions": tr,
        "counts": counts,
        "P": P,
        **discount,
        "K": K,
        **fit,
    }


# ---------------------------------------------------------------------
# 10. Plotting
# ---------------------------------------------------------------------

def spectrum_table(fit):
    """Export sigma_1 >= ... >= sigma_K, including the stationary last mode."""
    scales = fit["singular_values_descending"]
    K = len(scales)
    return pd.DataFrame({
        "mode_index": np.arange(1, K + 1),
        "is_stationary": np.arange(K) == K - 1,
        "gamma_per_second": fit["gamma_per_second"],
        "gamma": fit["gamma"],
        "lag_seconds": fit["lag_seconds"],
        "sigma": scales,
        **discount_diagnostics(scales, fit["gamma"]),
    })


def plot_spectrum(fit, null_df, outpath: Path, n_show=None):
    """Show the full descending spectrum, including sigma_K = 1-gamma.

    If n_show is given, show that many smallest nontrivial values and the
    stationary endpoint, keeping their indices in the full descending order.
    The null band calibrates only the minimum nontrivial singular value.
    """
    s = fit["singular_values_descending"]
    K = len(s)
    if n_show is not None and n_show < 1:
        raise ValueError("n_show must be positive or None for the full spectrum.")
    first_index = 1 if n_show is None else max(1, K - n_show)
    observed = s[first_index - 1:]

    null_lo = null_df["sigma_nontrivial"].quantile(0.025)
    null_hi = null_df["sigma_nontrivial"].quantile(0.975)

    floor = 1.0 - fit["gamma"]

    plt.figure(figsize=(7.5, 4.5))
    x = np.arange(first_index, K + 1)

    plt.plot(
        x,
        observed,
        marker="o",
        markersize=3,
        label="Observed singular values (descending)",
    )
    if K >= 3 and first_index <= K - 2:
        plt.scatter(
            [K - 2], [s[-3]], marker="^", s=52, color="tab:green", zorder=4,
        )
    plt.scatter(
        [K - 1], [s[-2]], marker="s", s=48, color="tab:blue", zorder=4,
    )
    # plt.scatter(
    #     [K], [s[-1]], marker="D", s=52, color="tab:orange", zorder=4,
    #     label=rf"Stationary mode $\sigma_{{{K}}}$",
    # )
    plt.axhline(
        floor,
        linestyle="--",
        label=rf"Discount floor $1-\gamma={floor:.3g}$",
    )
    plt.fill_between(
        [first_index, K],
        [null_lo, null_lo],
        [null_hi, null_hi],
        alpha=0.2,
        label="Null minimum nontrivial scale: 95% band",
    )

    # A zero baseline prevents the floor from appearing below its true value.
    # At gamma=0.9 this locator also explicitly labels the 0.1 tick.
    ax = plt.gca()
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 5, 10]))
    ax.set_xlim(first_index - 0.5, K + 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= first_index) & (ticks <= K)]
    ax.set_xticks(np.unique(np.r_[first_index, ticks, K]))
    if K >= 3 and first_index <= K - 2:
        ax.annotate(
            rf"$\sigma_{{{K-2}}}={s[-3]:.4f}$",
            xy=(K - 2, s[-3]), xytext=(-90, 36), textcoords="offset points",
            ha="right", fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "tab:green"},
        )
    ax.annotate(
        rf"$\sigma_{{{K-1}}}={s[-2]:.4f}$",
        xy=(K - 1, s[-2]), xytext=(-90, 20), textcoords="offset points",
        ha="right", fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
    )
    ax.annotate(
        rf"$\sigma_{{{K}}}={s[-1]:.3g}$",
        xy=(K, s[-1]), xytext=(-90, -4), textcoords="offset points",
        ha="right", va="top", fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "tab:orange"},
    )

    plt.xlabel(r"Singular value index $j$", fontsize=14)
    plt.ylabel(r"Singular value of $I-\gamma \widetilde P$", fontsize=14)
    plt.title(
        f"NGSIM I-80: singular spectrum (lag={fit['lag_seconds']:g} s, "
        f"gamma={fit['gamma']:.4f})", fontsize=14,
    )
    plt.legend(loc="lower left", bbox_to_anchor=(0.0, 0.16))
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_mode_interpretation(summary, f, outpath: Path):
    z = summary.copy()
    z["mode_value"] = f

    neg = z["mode_value"] < 0
    pos = ~neg

    plt.figure(figsize=(7.5, 5.0))

    plt.scatter(
        z.loc[neg, "mean_speed"],
        z.loc[neg, "mean_gap"],
        s=70 * np.abs(z.loc[neg, "mode_value"]),
        marker="x",
        label="Negative side of the state contrast",
    )

    plt.scatter(
        z.loc[pos, "mean_speed"],
        z.loc[pos, "mean_gap"],
        s=70 * np.abs(z.loc[pos, "mode_value"]),
        marker="o",
        label="Positive side of the state contrast",
    )

    plt.xlabel("Mean speed (dataset units)", fontsize=14)
    plt.ylabel("Mean leader-distance proxy (dataset units)", fontsize=14)
    plt.title(
        "State contrast: low-speed/short-gap vs high-speed/long-gap states", fontsize=14
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("csv", type=str)
    parser.add_argument("--outdir", type=str, default="ngsim_results")

    parser.add_argument(
        "--gamma-per-second",
        type=float,
        default=DEFAULT_GAMMA_PER_SECOND,
        help=("One-second discount (default: %(default)s). At lag h seconds, "
              "the transition discount is gamma_per_second ** h."),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Quantile bins per feature axis; 4 => 64 states.",
    )

    parser.add_argument(
        "--lag",
        type=float,
        default=1.0,
        help="Main transition lag in seconds.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--null-reps",
        type=int,
        default=500,
    )

    args = parser.parse_args(argv)
    try:
        resolve_discount(args.lag, args.gamma_per_second)
    except ValueError as exc:
        parser.error(str(exc))
    if args.bins < 2:
        parser.error("--bins must be at least 2.")
    if args.bootstrap < 1 or args.null_reps < 1:
        parser.error("--bootstrap and --null-reps must be positive.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "run_log.txt").open("w", encoding="utf-8") as log:
        with redirect_stdout(TeeStream(sys.stdout, log)):
            run_analysis(args, outdir)


def run_analysis(args, outdir: Path):
    """Regenerate all reported statistics and figures for one configuration."""

    print("Loading data...")
    x = load_and_construct_features(args.csv)

    print(f"Rows in raw file:       {pd.read_csv(args.csv, usecols=['Vehicle_ID']).shape[0]:,}")
    print(f"Unique vehicles:        {x['Vehicle_ID'].nunique():,}")
    print(f"Leader-matched rows:    {len(x):,}")

    # ---------------- Main fit ----------------
    print("\nRunning main analysis...")
    fit = run_single(
        x,
        bins_per_axis=args.bins,
        lag_seconds=args.lag,
        gamma_per_second=args.gamma_per_second,
    )

    K = fit["K"]
    summary = state_summary(fit["x_state"], K)

    f = orient_mode_for_interpretation(
        fit["f_state"].copy(),
        summary,
    )

    # Keep weighted v sign consistent with physical f sign for display only.
    if np.dot(f, fit["f_state"]) < 0:
        fit["v_weighted"] = -fit["v_weighted"]

    corr_speed = float(
        np.corrcoef(f, summary["mean_speed"].to_numpy())[0, 1]
    )
    corr_gap = float(
        np.corrcoef(f, summary["mean_gap"].to_numpy())[0, 1]
    )
    corr_rel = float(
        np.corrcoef(f, summary["mean_rel_speed"].to_numpy())[0, 1]
    )

    cut = empirical_cut_statistics(
        fit["P"],
        fit["mu"],
        f,
    )

    print(f"Usable transitions:     {len(fit['transitions']):,}")
    print(f"Number of states K:     {K}")
    print(f"gamma per second:       {fit['gamma_per_second']:.8f}")
    print(f"requested lag (s):      {fit['requested_lag_seconds']:g}")
    print(f"actual lag (s/frames):  {fit['lag_seconds']:g} / {fit['lag_frames']}")
    print(f"gamma at main lag:      {fit['gamma']:.8f}")
    print(f"discount floor:         {1-fit['gamma']:.8f}")
    print("Spectrum order:         sigma_1 >= ... >= sigma_K")
    print(f"sigma_1 (maximum):      {fit['singular_values_descending'][0]:.10f}")
    print(f"sigma_{K-1}^perp:         {fit['sigma_nontrivial']:.10f}")
    print(f"sigma_{K} (stationary):  {fit['singular_values_descending'][-1]:.10f}")
    print(f"sigma / discount floor: {fit['sigma_over_discount_floor']:.8f}")
    print(f"sigma - discount floor: {fit['sigma_excess_above_floor']:.10f}")
    print(f"epsilon (mode {K-1}):      {fit['epsilon_nontrivial']:.10f}")
    print(f"<v, P_tilde v>:         {fit['rayleigh_nontrivial']:.10f}")

    print("\nMode interpretation")
    print(f"corr(mode, speed):      {corr_speed:.4f}")
    print(f"corr(mode, gap):        {corr_gap:.4f}")
    print(f"corr(mode, rel speed):  {corr_rel:.4f}")

    print("\nEmpirical sign-cut statistics")
    for k, v0 in cut.items():
        print(f"{k:28s} {v0:.8f}")

    # Save state interpretation table.
    state_out = summary.copy()
    state_out["mode_value"] = f
    state_out["side"] = np.where(f < 0, "negative", "positive")
    state_out["stationary_mass"] = fit["mu"]
    state_out["gamma_per_second"] = fit["gamma_per_second"]
    state_out["gamma"] = fit["gamma"]
    state_out["lag_seconds"] = fit["lag_seconds"]
    state_out.to_csv(
        outdir / "state_mode_interpretation.csv",
        index=True,
    )

    spectrum_table(fit).to_csv(outdir / "singular_spectrum.csv", index=False)

    # ---------------- Bootstrap ----------------
    print("\nRunning vehicle-level bootstrap...")
    boot = vehicle_bootstrap(
        fit["transitions"],
        K=K,
        gamma=fit["gamma"],
        full_v=fit["v_weighted"],
        B=args.bootstrap,
    )
    if boot.empty:
        raise RuntimeError("No bootstrap draw had all state rows populated.")
    boot["gamma_per_second"] = fit["gamma_per_second"]
    boot["lag_seconds"] = fit["lag_seconds"]
    boot.to_csv(outdir / "vehicle_bootstrap.csv", index=False)

    def q95(series):
        return (
            float(series.quantile(0.025)),
            float(series.quantile(0.975)),
        )

    sigma_ci = q95(boot["sigma"])
    eps_ci = q95(boot["epsilon"])
    align_ci = q95(boot["alignment_abs"])
    ratio_ci = q95(boot["sigma_over_discount_floor"])

    print(f"sigma bootstrap 95% CI: {sigma_ci}")
    print(f"ratio bootstrap 95% CI: {ratio_ci}")
    print(f"eps bootstrap 95% CI:   {eps_ci}")
    print(f"align bootstrap 95% CI: {align_ci}")

    # ---------------- Null ----------------
    print("\nRunning no-temporal-dependence null...")
    null = temporal_independence_null(
        fit["counts"],
        gamma=fit["gamma"],
        reps=args.null_reps,
    )
    null["gamma_per_second"] = fit["gamma_per_second"]
    null["lag_seconds"] = fit["lag_seconds"]
    null.to_csv(outdir / "temporal_independence_null.csv", index=False)

    null_q = null["sigma_nontrivial"].quantile([0.025, 0.5, 0.975])

    print(
        "null sigma 2.5/50/97.5%: "
        f"{null_q.loc[0.025]:.6f}, "
        f"{null_q.loc[0.5]:.6f}, "
        f"{null_q.loc[0.975]:.6f}"
    )

    # ---------------- Lag robustness ----------------
    print("\nLag robustness...")
    lag_rows = []

    for lag_sec in [0.5, 1.0, 2.0]:
        z = run_single(
            x,
            bins_per_axis=args.bins,
            lag_seconds=lag_sec,
            gamma_per_second=args.gamma_per_second,
        )

        lag_rows.append(
            {
                "requested_lag_seconds": z["requested_lag_seconds"],
                "lag_seconds": z["lag_seconds"],
                "lag_frames": z["lag_frames"],
                "gamma_per_second": z["gamma_per_second"],
                "gamma": z["gamma"],
                "transitions": len(z["transitions"]),
                "sigma_nontrivial": z["sigma_nontrivial"],
                **discount_diagnostics(z["sigma_nontrivial"], z["gamma"]),
                "epsilon_nontrivial": z["epsilon_nontrivial"],
                "rayleigh": z["rayleigh_nontrivial"],
            }
        )

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(outdir / "lag_robustness.csv", index=False)
    print(lag_df.to_string(index=False))

    # ---------------- Resolution robustness ----------------
    print("\nState-resolution robustness...")
    res_rows = []

    for nb in [3, 4, 5]:
        z = run_single(
            x,
            bins_per_axis=nb,
            lag_seconds=args.lag,
            gamma_per_second=args.gamma_per_second,
        )

        res_rows.append(
            {
                "bins_per_axis": nb,
                "K": nb ** 3,
                "lag_seconds": z["lag_seconds"],
                "gamma_per_second": z["gamma_per_second"],
                "gamma": z["gamma"],
                "sigma_nontrivial": z["sigma_nontrivial"],
                **discount_diagnostics(z["sigma_nontrivial"], z["gamma"]),
                "epsilon_nontrivial": z["epsilon_nontrivial"],
                "rayleigh": z["rayleigh_nontrivial"],
            }
        )

    res_df = pd.DataFrame(res_rows)
    res_df.to_csv(
        outdir / "state_resolution_robustness.csv",
        index=False,
    )
    print(res_df.to_string(index=False))

    # ---------------- Plots ----------------
    for extension in ("pdf", "png"):
        plot_spectrum(fit, null, outdir / f"spectrum_vs_null.{extension}")
        plot_mode_interpretation(summary, f, outdir / f"mode_interpretation.{extension}")

    # ---------------- Overall summary ----------------
    overall = pd.DataFrame(
        [
            ["raw_rows", pd.read_csv(args.csv, usecols=["Vehicle_ID"]).shape[0]],
            ["unique_vehicles", x["Vehicle_ID"].nunique()],
            ["leader_matched_rows", len(x)],
            ["usable_main_lag_transitions", len(fit["transitions"])],
            ["K", K],
            ["persistent_nontrivial_mode_index", K - 1],
            ["stationary_mode_index", K],
            ["frame_rate_hz", FRAME_RATE_HZ],
            ["requested_lag_seconds", fit["requested_lag_seconds"]],
            ["lag_seconds", fit["lag_seconds"]],
            ["lag_frames", fit["lag_frames"]],
            ["gamma_per_second", fit["gamma_per_second"]],
            ["gamma", fit["gamma"]],
            ["discount_floor", 1 - fit["gamma"]],
            ["sigma_nontrivial", fit["sigma_nontrivial"]],
            ["sigma_over_discount_floor", fit["sigma_over_discount_floor"]],
            ["sigma_excess_above_floor", fit["sigma_excess_above_floor"]],
            ["epsilon_nontrivial", fit["epsilon_nontrivial"]],
            ["rayleigh", fit["rayleigh_nontrivial"]],
            ["corr_mode_speed", corr_speed],
            ["corr_mode_gap", corr_gap],
            ["corr_mode_rel_speed", corr_rel],
            ["bootstrap_sigma_ci_low", sigma_ci[0]],
            ["bootstrap_sigma_ci_high", sigma_ci[1]],
            ["bootstrap_sigma_over_floor_ci_low", ratio_ci[0]],
            ["bootstrap_sigma_over_floor_ci_high", ratio_ci[1]],
            ["bootstrap_epsilon_ci_low", eps_ci[0]],
            ["bootstrap_epsilon_ci_high", eps_ci[1]],
            ["bootstrap_alignment_ci_low", align_ci[0]],
            ["bootstrap_alignment_ci_high", align_ci[1]],
            ["null_sigma_q025", null_q.loc[0.025]],
            ["null_sigma_median", null_q.loc[0.5]],
            ["null_sigma_q975", null_q.loc[0.975]],
            ["null_sigma_over_floor_median", null_q.loc[0.5] / fit["discount_floor"]],
            ["cut_P_pos_given_neg", cut["P(pos | neg)"]],
            ["cut_P_neg_given_pos", cut["P(neg | pos)"]],
            ["cut_conductance_like", cut["conductance_like_cut"]],
        ],
        columns=["quantity", "value"],
    )

    overall.to_csv(outdir / "overall_summary.csv", index=False)

    metadata = {
        "csv_path": str(Path(args.csv).resolve()),
        "discount_convention": "gamma(lag) = gamma_per_second ** actual_lag_seconds",
        "gamma_per_second": fit["gamma_per_second"],
        "gamma": fit["gamma"],
        "discount_floor": fit["discount_floor"],
        "frame_rate_hz": FRAME_RATE_HZ,
        "requested_lag_seconds": fit["requested_lag_seconds"],
        "lag_seconds": fit["lag_seconds"],
        "lag_frames": fit["lag_frames"],
        "bins_per_axis": args.bins,
        "spectrum_order": "descending",
        "persistent_nontrivial_mode_index": K - 1,
        "stationary_mode_index": K,
        "bootstrap_requested": args.bootstrap,
        "bootstrap_completed": len(boot),
        "bootstrap_seed": 20260920,
        "null_reps": len(null),
        "null_seed": 20260921,
    }
    (outdir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )

    print("\nDone.")
    print(f"Results written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

"""Reproducible nonsymmetric random-transition experiments.

All stationary weights are computed from the realized transition matrix.
Transition weights are independent Gamma(shape=0.155, scale=1/0.155) draws,
normalized rowwise (giving Dirichlet(0.155, ..., 0.155) dense rows).
The persistent-group construction shares the realized dense stationary law,
as required by the theorem; no uniform-weight substitution is made.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals


GAMMA = 0.9
FLOOR = 1.0 - GAMMA
NEAR_FLOOR_THRESHOLD = FLOOR + GAMMA / 2.0


def _stationary_distribution(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve stationary equations by GTH elimination without regularization.

    Positive off-diagonal arithmetic avoids cancellation when Gamma weights
    produce nearly closed classes or very small stationary probabilities.
    See https://quanteconpy.readthedocs.io/en/latest/markov/gth_solve.html.
    """
    n = p.shape[0]
    reduced = p.copy()
    for last in range(n - 1, 0, -1):
        exit_mass = float(np.sum(reduced[last, :last]))
        if exit_mass <= 0 or not np.isfinite(exit_mass):
            raise FloatingPointError("Stationary elimination requires positive finite exit mass.")
        reduced[:last, last] /= exit_mass
        reduced[:last, :last] += np.outer(reduced[:last, last], reduced[last, :last])
    mu = np.ones(n)
    for state in range(1, n):
        mu[state] = np.dot(mu[:state], reduced[:state, state])
    mu /= mu.sum()
    residual = _check_stationarity(p, mu)
    return mu, residual


def _check_stationarity(p: np.ndarray, mu: np.ndarray) -> float:
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(mu)):
        raise FloatingPointError("Nonfinite transition or stationary values.")
    if np.min(p) < 0 or np.min(mu) <= 0:
        raise FloatingPointError("Negative transitions or nonpositive stationary law.")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=5e-13):
        raise FloatingPointError("Transition rows are not normalized.")
    if abs(mu.sum() - 1.0) > 5e-12:
        raise FloatingPointError("Stationary law is not normalized.")
    residual = float(np.max(np.abs(mu @ p - mu)))
    if residual > 1e-10:
        raise FloatingPointError(f"Stationarity residual too large: {residual:g}")
    return residual


def _spectral_statistics(p: np.ndarray, mu: np.ndarray) -> dict[str, float]:
    """Use singular values of the full nonsymmetric weighted operators."""
    residual = _check_stationarity(p, mu)
    root_mu = np.sqrt(mu)
    weighted = root_mu[:, None] * p / root_mu[None, :]
    singular = svdvals(np.eye(p.shape[0]) - GAMMA * weighted,
                       check_finite=False)
    centered_norm = float(svdvals(weighted - np.outer(root_mu, root_mu),
                                  check_finite=False)[0])
    sigma = float(singular[-2])
    stationary_sigma = float(singular[-1])
    if abs(stationary_sigma - FLOOR) > 5e-10 or sigma < FLOOR - 5e-10:
        raise FloatingPointError("Discounted spectral floor check failed.")
    return {
        "sigma_nonstationary": sigma,
        "sigma_stationary": stationary_sigma,
        "weighted_centered_norm": centered_norm,
        "stationarity_residual_inf": residual,
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "max_relative_stationary_deviation": float(np.max(np.abs(len(mu) * mu - 1))),
    }


def _row_record(experiment: str, n: int, replicate: int, seed: int,
                stats: dict, alpha: float = 0.0, restricted_rows: int = 0,
                forbidden_per_restricted_row: int = 0,
                dense_centered_norm: float | None = None) -> dict:
    location = 1.0 - GAMMA * alpha
    reference_gap = GAMMA * (1.0 - alpha)
    relative_error = abs(stats["sigma_nonstationary"] - location) / reference_gap
    row = {
        "experiment": experiment,
        "n": n,
        "replicate": replicate,
        "seed": seed,
        "gamma": GAMMA,
        "alpha": alpha,
        "restricted_rows": restricted_rows,
        "forbidden_per_restricted_row": forbidden_per_restricted_row,
        **stats,
        "reference_sigma": location,
        "reference_gap_above_floor": reference_gap,
        "absolute_reference_error": abs(stats["sigma_nonstationary"] - location),
        "relative_reference_error": relative_error,
        "dense_background_centered_norm": "" if dense_centered_norm is None else dense_centered_norm,
        "near_floor_threshold": NEAR_FLOOR_THRESHOLD,
        "near_floor_event": int(stats["sigma_nonstationary"] <= NEAR_FLOOR_THRESHOLD),
    }
    if experiment.startswith("persistent") and dense_centered_norm is not None:
        if relative_error > dense_centered_norm + 5e-10:
            raise FloatingPointError("Persistent-group perturbation bound violated.")
    return row


def _summarize(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _group_summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["experiment"], row["n"], row["alpha"])
        groups.setdefault(key, []).append(row)
    result = []
    for (experiment, n, alpha), group in groups.items():
        result.append({
            "experiment": experiment, "n": n, "alpha": alpha,
            "replicates": len(group),
            "sigma_nonstationary": _summarize([r["sigma_nonstationary"] for r in group]),
            "weighted_centered_norm": _summarize([r["weighted_centered_norm"] for r in group]),
            "relative_reference_error": _summarize([r["relative_reference_error"] for r in group]),
            "near_floor_count": sum(r["near_floor_event"] for r in group),
            "near_floor_fraction": float(np.mean([r["near_floor_event"] for r in group])),
        })
    return result


def _plot(rows: list[dict], ns: list[int], outdir: Path) -> tuple[Path, Path, dict[str, str]]:
    colors = {"dense": "#2463A6", "localized": "#D67C22", "persistent": "#397A57"}

    def draw_a(ax):
        log_n = np.log2(np.asarray(ns, dtype=float))
        for experiment, label in [("dense", "Dense"), ("localized", "Localized obstacles")]:
            groups = [[r["sigma_nonstationary"] for r in rows
                       if r["experiment"] == experiment and r["n"] == n] for n in ns]
            means = [np.mean(g) for g in groups]
            lo = [np.quantile(g, .025) for g in groups]
            hi = [np.quantile(g, .975) for g in groups]
            ax.fill_between(log_n, lo, hi, color=colors[experiment], alpha=.16, linewidth=0)
            ax.plot(log_n, means, "o-", color=colors[experiment], label=label, markersize=4)
        # ax.axhline(1, color="#6D7580", linestyle=":", linewidth=1, label="Reference: 1")
        ax.axhline(FLOOR, color="#B3443A", linestyle="--", linewidth=1.2, label="Floor: 0.1")
        ax.set(xlabel=r"$\log_2 n$", ylabel=r"$\sigma_{n-1}(\widetilde A_n)$",
               title="A   Dense and localized\ntransitions", ylim=(0, 1.06))
        ax.set_xticks(log_n, labels=[f"{value:g}" for value in log_n])
        ax.legend(frameon=False, fontsize=18, loc="lower right",
                  bbox_to_anchor=(1.0, 0.10), borderaxespad=0.25)

    def draw_b(ax):
        alphas = sorted({r["alpha"] for r in rows if r["experiment"] == "persistent_fixed"})
        groups = [[r["sigma_nonstationary"] for r in rows
                   if r["experiment"] == "persistent_fixed" and r["alpha"] == a] for a in alphas]
        means = np.asarray([np.mean(g) for g in groups])
        lo = np.asarray([np.quantile(g, .025) for g in groups])
        hi = np.asarray([np.quantile(g, .975) for g in groups])
        # Draw intervals directly: empirical means need not lie between tail quantiles.
        ax.vlines(alphas, lo, hi, color=colors["persistent"], linewidth=2.2)
        ax.plot(alphas, means, "o", color=colors["persistent"], markersize=5,
                label="Mean / 95% range")
        aa = np.linspace(0, 1, 201)
        ax.plot(aa, 1 - GAMMA * aa, color="#3F4853", linestyle="--", linewidth=1.2,
                label=r"Reference: $1-\gamma\alpha_n$")
        ax.axhline(FLOOR, color="#B3443A", linestyle=":", linewidth=1)
        ax.set(xlabel=r"Persistence strength $\alpha$", ylabel=r"$\sigma_{n-1}(\widetilde A_n)$",
               title=f"B   Persistent groups\n($n={max(ns)}$)", xlim=(-.03, 1.03), ylim=(0, 1.04))
        ax.set_xticks(alphas, labels=[f"{alpha:g}" for alpha in alphas])
        ax.tick_params(axis="x", labelsize=16)
        ax.legend(frameon=False, fontsize=15, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), borderaxespad=0.25,
                  handlelength=1.6, handletextpad=0.5, labelspacing=0.25)

    def draw_c(ax):
        for field, label, color, marker in [
            ("relative_reference_error", "Relative spectral deviation", "#7756A2", "o"),
            ("dense_background_centered_norm", "Perturbation upper bound", "#70853A", "s"),
        ]:
            groups = [[r[field] for r in rows
                       if r["experiment"] == "persistent_sequence" and r["n"] == n] for n in ns]
            means = [np.mean(g) for g in groups]
            lo = [np.quantile(g, .025) for g in groups]
            hi = [np.quantile(g, .975) for g in groups]
            ax.fill_between(ns, lo, hi, color=color, alpha=.14, linewidth=0)
            ax.plot(ns, means, marker + "-", color=color, markersize=4, label=label)
        ax.plot(ns, np.asarray(ns, dtype=float) ** -.5, color="#7A7F86", linestyle="--",
                linewidth=1, label=r"$n^{-1/2}$ guide")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns, labels=[str(n) for n in ns])
        ax.set(xlabel="Number of states $n$", ylabel="Dimensionless deviation / bound",
               title="C   Approach to the floor:\n" + r"$\alpha_n=1-n^{-1/2}$")
        ax.legend(frameon=False, fontsize=18, loc="lower left")

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 18,
        "axes.labelsize": 22, "axes.titlesize": 20,
        "xtick.labelsize": 19, "ytick.labelsize": 19,
        "legend.fontsize": 18, "axes.titlepad": 14, "axes.labelpad": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8))
        draw_a(axes[0])
        draw_b(axes[1])
        draw_c(axes[2])
        for axis in axes:
            axis.grid(axis="y", alpha=.15, linewidth=.6)
        fig.subplots_adjust(left=0.060, right=0.970, bottom=0.18,
                            top=0.83, wspace=0.45)
        pdf = outdir / "rmt_spectral_experiments.pdf"
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

        panel_files: dict[str, str] = {}
        for letter, draw in (("A", draw_a), ("B", draw_b), ("C", draw_c)):
            panel_fig = plt.figure(figsize=(6.2, 5.3))
            panel_ax = panel_fig.add_axes([0.24, 0.18, 0.70, 0.65])
            draw(panel_ax)
            panel_ax.grid(axis="y", alpha=.15, linewidth=.6)
            panel_pdf = outdir / f"rmt_spectral_experiments_panel_{letter}.pdf"
            panel_fig.savefig(panel_pdf, facecolor="white")
            plt.close(panel_fig)
            panel_files[f"panel_{letter.lower()}_pdf"] = str(panel_pdf)
    return pdf, panel_files


def run_rmt(outdir, seed=20261015, quick=False):
    """Write replicate-level data, a three-panel figure, and JSON summary.

    Full: n=(8,16,32,64,128,256,512,1024), 100 replicates per n.
    Quick: n=(8,64,256), 12 replicates per n (smoke run only).
    Panels A and C use all n. Fixed-alpha comparisons in panel B use the
    largest n, with all alphas paired on the same dense background.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ns = [8, 64, 256] if quick else [8, 16, 32, 64, 128, 256, 512, 1024]
    repeats = 12 if quick else 100
    alphas = [0.0, 0.3, 0.6, 0.9]
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        k = int(np.floor(np.sqrt(n)))
        forbidden = n // 2
        allowed = np.ones((n, n), dtype=bool)
        # One deterministic mask per dimension, unchanged over replicates.
        # The first k rows forbid the final floor(n/2) destinations.
        allowed[:k, n - forbidden:] = False
        split = n // 2
        alpha_sequence = 1.0 - 1.0 / np.sqrt(n)
        for replicate in range(repeats):
            weights = rng.gamma(shape=0.155, scale=1.0 / 0.155, size=(n, n))
            # Gamma draws are strictly positive almost surely. Do not
            # modify a zero draw with pseudocounts if a generator ever emits one.
            if np.any(weights <= 0):
                raise FloatingPointError("Gamma generator returned a nonpositive draw.")
            dense = weights / weights.sum(axis=1, keepdims=True)
            mu, _ = _stationary_distribution(dense)
            dense_stats = _spectral_statistics(dense, mu)
            dense_norm = dense_stats["weighted_centered_norm"]
            rows.append(_row_record("dense", n, replicate, seed, dense_stats,
                                    dense_centered_norm=dense_norm))

            local_weights = weights * allowed
            localized = local_weights / local_weights.sum(axis=1, keepdims=True)
            local_mu, _ = _stationary_distribution(localized)
            rows.append(_row_record("localized", n, replicate, seed,
                                    _spectral_statistics(localized, local_mu),
                                    restricted_rows=k,
                                    forbidden_per_restricted_row=forbidden,
                                    dense_centered_norm=dense_norm))

            q = np.zeros_like(dense)
            q[:split, :split] = mu[:split] / mu[:split].sum()
            q[split:, split:] = mu[split:] / mu[split:].sum()
            persistent = alpha_sequence * q + (1.0 - alpha_sequence) * dense
            rows.append(_row_record("persistent_sequence", n, replicate, seed,
                                    _spectral_statistics(persistent, mu),
                                    alpha=float(alpha_sequence), dense_centered_norm=dense_norm))
            if n == max(ns):
                for alpha in alphas:
                    persistent = alpha * q + (1.0 - alpha) * dense
                    stats = dense_stats if alpha == 0 else _spectral_statistics(persistent, mu)
                    rows.append(_row_record("persistent_fixed", n, replicate, seed, stats,
                                            alpha=alpha, dense_centered_norm=dense_norm))

    raw_path = outdir / "rmt_replicates.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pdf, panel_files = _plot(rows, ns, outdir)
    summary = {
        "seed": int(seed), "quick": bool(quick), "gamma": GAMMA,
        "discounted_floor": FLOOR, "n_values": ns, "replicates_per_n": repeats,
        "fixed_alphas_at_largest_n": alphas,
        "sequence_alpha": "1 - n**(-1/2)",
        "weights": "Independent Gamma(shape=0.155, scale=1/0.155, mean=1), paired across constructions within each replicate",
        "localized_mask": "First floor(sqrt(n)) rows forbid the final floor(n/2) destinations; fixed per n",
        "stationary_weighting": "Solved separately for dense and localized matrices; persistent groups preserve the dense stationary law",
        "stationary_solver": "GTH elimination using positive off-diagonal arithmetic; no regularization",
        "singular_value_method": "Full nonsymmetric SVD of I - gamma * diag(sqrt(mu)) * P * diag(1/sqrt(mu))",
        "near_floor_event": f"sigma_nonstationary <= {NEAR_FLOOR_THRESHOLD:g}",
        "band_definition": "Pointwise empirical 2.5% and 97.5% replicate quantiles; not confidence intervals for a mean",
        "relative_deviation_definition": "abs(sigma_nonstationary - (1-gamma*alpha)) / (gamma*(1-alpha))",
        "max_stationarity_residual_inf": max(r["stationarity_residual_inf"] for r in rows),
        "max_stationary_floor_error": max(abs(r["sigma_stationary"] - FLOOR) for r in rows),
        "raw_rows": len(rows),
        "groups": _group_summaries(rows),
        "files": {"raw_csv": str(raw_path), "figure_pdf": str(pdf), **panel_files},
    }
    summary_path = outdir / "rmt_summary.json"
    summary["files"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/rmt")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_rmt(args.outdir, args.seed, args.quick)
    print(json.dumps({"files": result["files"], "raw_rows": result["raw_rows"],
                      "max_stationarity_residual_inf": result["max_stationarity_residual_inf"]}, indent=2))

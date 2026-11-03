"""Prepare and draw the v1.3 Figure 2 PNG review figure from frozen E1/E2 results."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STYLE_ROOT = REPOSITORY_ROOT / ".codex" / "skills" / "dynags-figure" / "scripts"
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))

from plot_style import paper_style, style_axis


P = 10_000
NTRS = (100, 200, 400, 800, 1600)
H2S = (0.1, 0.4, 0.8)
REFERENCE_CONDITION = "P10000_NTR400_H0.8"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 73
FIGURE_SIZE = (7.1, 5.9)
LAYOUT_LEFT = 0.085
LAYOUT_RIGHT = 0.925
LAYOUT_BOTTOM = 0.075
LAYOUT_TOP = 0.905
PANEL_LABEL_X = 0.025
COLORS = {"Oracle": "#0072B2", "Screened": "#0072B2", "Truth": "#000000", "Observed": "#B3B3B3"}
MARKERS = {"Oracle": "o", "Screened": "s"}
LINESTYLES = {"Oracle": "-", "Screened": "--"}

DEFAULT_TABLE_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/tables"
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "dataset/experiments"
DEFAULT_SIMULATION_ROOT = REPOSITORY_ROOT / "outputs/simulation"
DEFAULT_DATA = DEFAULT_TABLE_ROOT / "figures/figure_02_data.csv"
DEFAULT_METADATA = REPOSITORY_ROOT / "outputs/paper_experiments/manifests/figures/figure_02_metadata.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs/paper_experiments/figures/fig2_oracle_screened.png"


def bootstrap_summary(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Every Figure 2 condition requires 10 finite independent repeats")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    boot_means = values[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def performance_axis_limits(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Return shared limits that retain every repeat and confidence interval."""
    raw = frame[
        frame["record_type"].eq("performance_replicate")
        & frame["metric"].eq(metric)
    ]["value"].to_numpy(float)
    summary = frame[
        frame["record_type"].eq("performance_summary")
        & frame["metric"].eq(metric)
    ]
    low = min(float(raw.min()), float(summary["ci_low"].min()))
    high = max(float(raw.max()), float(summary["ci_high"].max()))
    pad = max(0.04 * (high - low), 0.01 if metric == "pcc" else 0.001)
    return low - pad, high + pad


def inflection(values: np.ndarray, time: np.ndarray, scale: float) -> float | None:
    """Shared PCHIP-1000 main-inflection protocol from the frozen experiment plan."""
    if float(np.ptp(values)) < 0.01 * scale:
        return None
    dense = np.linspace(float(time[0]), float(time[-1]), 1000)
    derivative = np.asarray(PchipInterpolator(time, values).derivative()(dense), dtype=float)
    margin = 0.05 * float(time[-1] - time[0])
    internal = np.flatnonzero((dense >= time[0] + margin) & (dense <= time[-1] - margin))
    local = int(np.argmax(derivative[internal]))
    if local in (0, len(internal) - 1):
        return None
    value = float(derivative[internal[local]])
    return None if not np.isfinite(value) or value <= 0 else float(dense[internal[local]])


def reference_paths(simulation_root: Path, repetition: int) -> tuple[Path, Path]:
    suffix = Path(REFERENCE_CONDITION) / f"rep_{repetition:02d}" / "test_predictions.npz"
    return simulation_root / "experiment1_oracle" / suffix, simulation_root / "experiment2_screened" / suffix


def choose_reference_repeat(metrics: pd.DataFrame) -> int:
    subset = metrics.loc[
        metrics["condition"].eq(REFERENCE_CONDITION)
        & metrics["experiment"].eq("E2")
        & metrics["mode"].eq("Screened")
        & metrics["target"].eq("observed")
        & metrics["metric"].eq("pcc"),
        ["repetition", "value"],
    ].sort_values("repetition")
    if len(subset) != 10:
        raise ValueError("Reference-repeat selection requires 10 Screened PCC values")
    median = float(subset["value"].median())
    return int(subset.iloc[np.argmin(np.abs(subset["value"].to_numpy() - median))]["repetition"])


def select_samples(prediction: np.lib.npyio.NpzFile) -> list[dict[str, object]]:
    observed_means = prediction["targets"][:, 0, :].mean(axis=1)
    ordered = np.argsort(observed_means, kind="stable")
    output = []
    for level, positions in zip(("Low", "Middle", "High"), np.array_split(ordered, 3)):
        position = int(positions[np.argmin(np.abs(observed_means[positions] - np.median(observed_means[positions])))])
        output.append({"level": level, "position": position, "sample_id": str(prediction["sample_id"][position]), "sample_idx": int(prediction["sample_idx"][position])})
    return output


def prepare_data(table_root: Path, dataset_root: Path, simulation_root: Path) -> tuple[pd.DataFrame, list[Path], dict[str, object]]:
    overall_path = table_root / "metrics_overall_long.csv"
    screening_path = table_root / "screening_metrics.csv"
    overall = pd.read_csv(overall_path)
    screen = pd.read_csv(screening_path)

    nrmse_audit = overall.loc[
        overall["p"].eq(P)
        & overall["experiment"].isin(["E1", "E2"])
        & overall["target"].eq("observed")
        & overall["metric"].isin(["rmse", "nrmse"]),
        [
            "experiment", "condition", "ntr", "heritability", "repetition",
            "mode", "metric", "value", "train_p95_p5",
        ],
    ].copy()
    audit_keys = [
        "experiment", "condition", "ntr", "heritability", "repetition", "mode"
    ]
    audit_wide = nrmse_audit.pivot(
        index=audit_keys, columns="metric", values="value"
    )
    denominators = nrmse_audit[nrmse_audit["metric"].eq("nrmse")].set_index(
        audit_keys
    )["train_p95_p5"]
    expected_nrmse = audit_wide["rmse"] / denominators
    formula_error = np.abs(audit_wide["nrmse"] - expected_nrmse)
    if (
        len(audit_wide) != 300
        or not np.isfinite(denominators).all()
        or not denominators.gt(0).all()
        or float(formula_error.max()) > 1e-12
    ):
        raise ValueError("NRMSE audit failed the training-set P95-P5 protocol")
    paired_denominators = denominators.reset_index().pivot(
        index=["condition", "ntr", "heritability", "repetition"],
        columns="mode",
        values="train_p95_p5",
    )
    paired_denominator_difference = np.abs(
        paired_denominators["Oracle"] - paired_denominators["Screened"]
    )
    if float(paired_denominator_difference.max()) > 1e-12:
        raise ValueError("Paired Oracle and Screened NRMSE denominators differ")
    perf = overall.loc[
        overall["p"].eq(P)
        & overall["experiment"].isin(["E1", "E2"])
        & overall["target"].eq("observed")
        & overall["metric"].isin(["pcc", "nrmse"]),
        ["experiment", "condition", "ntr", "heritability", "repetition", "mode", "metric", "value"],
    ].copy()
    if len(perf) != 600 or set(perf["mode"]) != {"Oracle", "Screened"}:
        raise ValueError("Figure 2 requires 15 conditions × 10 repeats × 2 modes × 2 metrics")
    rows: list[dict[str, object]] = []
    for item in perf.itertuples(index=False):
        rows.append({"record_type": "performance_replicate", "panel": "B", "mode": item.mode, "metric": item.metric, "condition": item.condition, "ntr": item.ntr, "heritability": item.heritability, "repetition": item.repetition, "sample_level": "", "sample_id": "", "sample_idx": np.nan, "time": np.nan, "series": "", "value": item.value, "ci_low": np.nan, "ci_high": np.nan})
    for keys, group in perf.groupby(["condition", "ntr", "heritability", "mode", "metric"], sort=True):
        condition, ntr, h2, mode, metric = keys
        mean, low, high = bootstrap_summary(group.sort_values("repetition")["value"].to_numpy())
        rows.append({"record_type": "performance_summary", "panel": "B", "mode": mode, "metric": metric, "condition": condition, "ntr": ntr, "heritability": h2, "repetition": np.nan, "sample_level": "", "sample_id": "", "sample_idx": np.nan, "time": np.nan, "series": "", "value": mean, "ci_low": low, "ci_high": high})

    recovery = screen.loc[
        screen["p"].eq(P) & screen["k"].eq(20),
        [
            "run_id", "experiment", "condition", "ntr", "heritability",
            "repetition", "model", "mode", "k", "exact_recall",
            "ld_aware_recall",
        ],
    ].copy()
    recovery_counts = recovery.groupby(
        ["condition", "ntr", "heritability"], sort=True
    )["repetition"].nunique()
    if (
        len(recovery) != 150
        or recovery["run_id"].nunique() != 150
        or len(recovery_counts) != 15
        or not recovery_counts.eq(10).all()
        or set(recovery["experiment"]) != {"E2"}
        or set(recovery["model"]) != {"DynaGS"}
        or set(recovery["mode"]) != {"Screened"}
    ):
        raise ValueError("Figure 2 requires 15 conditions × 10 Recall@20 repetitions")
    ld_aware_identical = bool(
        np.allclose(recovery["exact_recall"], recovery["ld_aware_recall"], equal_nan=True)
    )
    if not ld_aware_identical:
        raise ValueError(
            "Exact and LD-aware Recall@20 differ; the conditional D panel must be restored"
        )
    for keys, group in recovery.groupby(
        ["condition", "ntr", "heritability"], sort=True
    ):
        condition, ntr, h2 = keys
        mean, _, _ = bootstrap_summary(
            group.sort_values("repetition")["exact_recall"].to_numpy()
        )
        rows.append({"record_type": "recovery_summary", "panel": "C", "mode": "Screened", "metric": "exact_recall", "condition": condition, "ntr": ntr, "heritability": h2, "repetition": np.nan, "sample_level": "", "sample_id": "", "sample_idx": np.nan, "time": np.nan, "series": "", "value": mean, "ci_low": np.nan, "ci_high": np.nan})

    reference_repeat = choose_reference_repeat(overall)
    oracle_path, screened_path = reference_paths(simulation_root, reference_repeat)
    oracle = np.load(oracle_path, allow_pickle=False)
    screened = np.load(screened_path, allow_pickle=False)
    if not (np.array_equal(oracle["sample_idx"], screened["sample_idx"]) and np.array_equal(oracle["sample_id"], screened["sample_id"]) and np.allclose(oracle["targets"], screened["targets"])):
        raise ValueError("Reference Oracle and Screened predictions are not paired on identical test samples")
    samples = select_samples(screened)
    dataset_dir = dataset_root / REFERENCE_CONDITION / f"rep_{reference_repeat:02d}"
    truth_path = dataset_dir / "truth/true_curve.csv"
    truth = pd.read_csv(truth_path).set_index("SampleID")
    time = oracle["time_coordinates"].astype(float)
    split = np.load(dataset_dir / "splits/split_seed73.npz", allow_pickle=False)
    scale = np.percentile(pd.read_csv(dataset_dir / "sim_pheno.csv").iloc[split["train_idx"], 1:].to_numpy(float), 95) - np.percentile(pd.read_csv(dataset_dir / "sim_pheno.csv").iloc[split["train_idx"], 1:].to_numpy(float), 5)
    for sample in samples:
        pos = int(sample["position"])
        arrays = {"Truth": truth.loc[sample["sample_id"], [f"T{i}" for i in range(1, 21)]].to_numpy(float), "Observed": oracle["targets"][pos, 0], "Oracle": oracle["predictions"][pos, 0], "Screened": screened["predictions"][pos, 0]}
        inflections = {series: inflection(values, time, scale) for series, values in arrays.items() if series != "Observed"}
        for series, values in arrays.items():
            for coordinate, value in zip(time, values):
                rows.append({"record_type": "trajectory", "panel": "A", "mode": series if series in {"Oracle", "Screened"} else "", "metric": "", "condition": REFERENCE_CONDITION, "ntr": 400, "heritability": 0.8, "repetition": reference_repeat, "sample_level": sample["level"], "sample_id": sample["sample_id"], "sample_idx": sample["sample_idx"], "time": coordinate, "series": series, "value": value, "ci_low": np.nan, "ci_high": np.nan})
        truth_time = inflections["Truth"]
        for series, point in inflections.items():
            rows.append({"record_type": "inflection", "panel": "A", "mode": series if series in {"Oracle", "Screened"} else "", "metric": "", "condition": REFERENCE_CONDITION, "ntr": 400, "heritability": 0.8, "repetition": reference_repeat, "sample_level": sample["level"], "sample_id": sample["sample_id"], "sample_idx": sample["sample_idx"], "time": point if point is not None else np.nan, "series": series, "value": np.nan if point is None or truth_time is None else point - truth_time, "ci_low": np.nan, "ci_high": np.nan})
    frame = pd.DataFrame(rows)
    pcc_limits = performance_axis_limits(frame, "pcc")
    nrmse_limits = performance_axis_limits(frame, "nrmse")
    sources = [overall_path, screening_path, oracle_path, screened_path, truth_path, dataset_dir / "sim_pheno.csv", dataset_dir / "splits/split_seed73.npz"]
    scientific = {
        "main_figure_scope": "P=10000 only; P=5000 is supplementary",
        "reference_condition": REFERENCE_CONDITION,
        "reference_repeat": reference_repeat,
        "selected_samples": [
            {key: value for key, value in sample.items() if key != "position"}
            for sample in samples
        ],
        "selection_rule": "Screened observed PCC nearest the 10-repeat median; then observed-mean test tertiles with the tertile-median sample.",
        "replicate_unit": "independent_simulation_repetition",
        "performance_panel": {
            "evaluation_target": "noisy observed test phenotype",
            "large_markers_and_lines": "mean of 10 independent simulation repetitions",
            "error_bars": "repetition-level percentile bootstrap 95% CI",
            "pale_points": "individual independent simulation repetitions",
            "pcc_display_limits": [float(value) for value in pcc_limits],
            "pcc_full_minus_one_to_one": False,
            "nrmse_display_limits": [float(value) for value in nrmse_limits],
        },
        "pcc_protocol": "mean of valid timepoint-wise Pearson correlations across test individuals",
        "nrmse_denominator": "training-individual observed phenotype P95 minus P5",
        "nrmse_audit": {
            "mode_metric_rows_checked": int(len(audit_wide)),
            "formula": "nrmse = rmse / train_p95_p5",
            "maximum_absolute_formula_error": float(formula_error.max()),
            "all_denominators_finite_and_positive": True,
            "oracle_screened_denominators_match_within_repetition": True,
            "maximum_paired_denominator_difference": float(
                paired_denominator_difference.max()
            ),
        },
        "inflection_protocol": "PCHIP_1000_grid; interior_5_percent; maximum_first_derivative",
        "inflection_grid_spacing": float((20 - 1) / (1000 - 1)),
        "inflection_interpretation": "A displayed delta near 0.02 can equal one interpolation-grid interval and is not biological timing precision.",
        "supplementary_inflection_reporting": [
            "all-test-individual inflection MAE",
            "valid-inflection fraction",
            "valid-individual count",
        ],
        "recovery_formal_result_audit": {
            "source": str(screening_path.resolve()),
            "rows_checked": int(len(recovery)),
            "unique_run_ids": int(recovery["run_id"].nunique()),
            "conditions_checked": int(len(recovery_counts)),
            "repetitions_per_condition": 10,
            "experiment": "E2",
            "model": "DynaGS",
            "mode": "Screened",
            "k": 20,
        },
        "ld_aware_protocol": "training-individual dosage r_squared >= 0.8",
        "ld_aware_audit": {
            "rows_checked": int(len(recovery)),
            "exact_equals_ld_aware_in_every_row": ld_aware_identical,
            "main_figure_decision": "omit duplicate LD-aware panel; report in supplementary material",
        },
        "uncertainty": "10,000-resample repetition-level percentile bootstrap 95% CI; seed 73",
    }
    return frame, sources, scientific


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    """Place every major-panel label at one shared figure x coordinate."""
    bounds = ax.get_position()
    fig.text(
        PANEL_LABEL_X, bounds.y1 + 0.008, label,
        fontsize=9, fontweight="bold", ha="left", va="bottom",
    )


def plot_trajectories(axes: list[plt.Axes], frame: pd.DataFrame) -> None:
    data = frame[frame["record_type"].eq("trajectory")]
    bounds = data["value"].agg(["min", "max"])
    pad = 0.05 * (bounds["max"] - bounds["min"])
    inflections = frame[frame["record_type"].eq("inflection")]
    for ax, level in zip(axes, ("Low", "Middle", "High")):
        subset = data[data["sample_level"].eq(level)]
        sample_id = subset["sample_id"].iloc[0]
        for series in ("Truth", "Observed", "Oracle", "Screened"):
            values = subset[subset["series"].eq(series)].sort_values("time")
            if series == "Observed":
                ax.scatter(values["time"], values["value"], color=COLORS[series], s=8, zorder=4)
            else:
                ax.plot(
                    values["time"], values["value"], color=COLORS[series],
                    linestyle=LINESTYLES.get(series, "-"), marker=MARKERS.get(series),
                    markerfacecolor="white" if series == "Oracle" else COLORS[series],
                    markeredgecolor=COLORS[series], markeredgewidth=0.75,
                    markevery=3, markersize=3.2, linewidth=1.15, zorder=3,
                )
        inset = inflections[inflections["sample_level"].eq(level)].set_index("series")
        for series in ("Truth", "Oracle", "Screened"):
            point = inset.loc[series, "time"]
            if np.isfinite(point):
                ax.axvline(point, color=COLORS.get(series, "#000000"), linestyle=LINESTYLES.get(series, ":"), linewidth=0.65, alpha=0.75, zorder=1)
        delta_o = inset.loc["Oracle", "value"]
        delta_s = inset.loc["Screened", "value"]
        value_o = r"\mathrm{N/A}" if not np.isfinite(delta_o) else f"{delta_o:+.2f}"
        value_s = r"\mathrm{N/A}" if not np.isfinite(delta_s) else f"{delta_s:+.2f}"
        annotation = (
            rf"$\Delta t_{{\mathrm{{O}}}}={value_o},\quad "
            rf"\Delta t_{{\mathrm{{S}}}}={value_s}$"
        )
        ax.text(
            0.98, 0.08, annotation, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.35,
        )
        ax.set_title(f"{level} · {sample_id}", loc="left", pad=3)
        ax.set_xlim(1, 20)
        ax.set_ylim(bounds["min"] - pad, bounds["max"] + pad)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        style_axis(ax, grid="y")
    axes[0].set_ylabel("Simulated phenotype")
    axes[1].set_xlabel("Time", labelpad=1.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def plot_performance(axes: np.ndarray, frame: pd.DataFrame) -> None:
    raw = frame[frame["record_type"].eq("performance_replicate")]
    summary = frame[frame["record_type"].eq("performance_summary")]
    row_limits = {
        metric: performance_axis_limits(frame, metric) for metric in ("pcc", "nrmse")
    }
    for row, metric in enumerate(("pcc", "nrmse")):
        for col, h2 in enumerate(H2S):
            ax = axes[row, col]
            x = np.arange(len(NTRS), dtype=float)
            for mode, offset in (("Oracle", -0.045), ("Screened", 0.045)):
                mode_raw = raw[
                    raw["metric"].eq(metric)
                    & raw["heritability"].eq(h2)
                    & raw["mode"].eq(mode)
                ]
                mode_stats = summary[
                    summary["metric"].eq(metric)
                    & summary["heritability"].eq(h2)
                    & summary["mode"].eq(mode)
                ].set_index("ntr").loc[list(NTRS)]
                for ntr_index, ntr in enumerate(NTRS):
                    individual = mode_raw[mode_raw["ntr"].eq(ntr)].sort_values("repetition")
                    deterministic_jitter = np.linspace(-0.055, 0.055, len(individual))
                    ax.scatter(
                        np.full(len(individual), x[ntr_index] + offset) + deterministic_jitter,
                        individual["value"], color=COLORS[mode], marker=MARKERS[mode],
                        s=7, alpha=0.15, linewidths=0, zorder=1,
                    )
                mean = mode_stats["value"].to_numpy(float)
                lower = mean - mode_stats["ci_low"].to_numpy(float)
                upper = mode_stats["ci_high"].to_numpy(float) - mean
                ax.plot(
                    x + offset, mean, color=COLORS[mode], linestyle=LINESTYLES[mode],
                    marker=MARKERS[mode], markerfacecolor="white", markeredgewidth=0.8,
                    markersize=3.8, linewidth=1.05, zorder=3,
                )
                ax.errorbar(
                    x + offset, mean, yerr=np.vstack([lower, upper]), fmt="none",
                    color=COLORS[mode], capsize=2.0, linewidth=0.75, zorder=2,
                )
            if row == 0:
                ax.set_title(f"h² = {h2:g}", pad=3)
            if col == 0:
                ax.set_ylabel("PCC ↑" if metric == "pcc" else "NRMSE ↓")
            else:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_xlabel("NTR", labelpad=1.5)
                ax.set_xticks(range(5), ["100", "200", "400", "800", "1,600"])
                ax.tick_params(axis="x", labelsize=6.5)
            else:
                ax.set_xticks(range(5), [])
            ax.set_ylim(*row_limits[metric])
            ax.yaxis.set_major_locator(MaxNLocator(4))
            style_axis(ax, grid="y")


def plot_heatmap(ax: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    summary = frame[(frame["record_type"].eq("recovery_summary")) & (frame["metric"].eq(metric))]
    matrix = summary.pivot(index="heritability", columns="ntr", values="value").reindex(index=H2S, columns=NTRS)
    image = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for y, h2 in enumerate(H2S):
        for x, ntr in enumerate(NTRS):
            value = matrix.loc[h2, ntr]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6.8, color="white" if value < 0.55 else "black")
    ax.set_xticks(range(5), ["100", "200", "400", "800", "1,600"])
    ax.set_yticks(range(3), [f"{value:g}" for value in H2S])
    ax.set_xlabel("NTR", labelpad=1.5)
    ax.set_ylabel("h²")
    ax.set_title(title, pad=2)
    return image


def draw_figure(data_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(data_path)
    required = {"trajectory", "inflection", "performance_replicate", "performance_summary", "recovery_summary"}
    if set(frame["record_type"]) != required:
        raise ValueError("Figure 2 prepared data are incomplete")
    with paper_style():
        fig = plt.figure(figsize=FIGURE_SIZE)
        outer = fig.add_gridspec(
            3, 1, height_ratios=(1.0, 1.64, 1.0), hspace=0.40,
            left=LAYOUT_LEFT, right=LAYOUT_RIGHT,
            bottom=LAYOUT_BOTTOM, top=LAYOUT_TOP,
        )
        trajectory_axes = [
            fig.add_subplot(spec)
            for spec in outer[0].subgridspec(1, 3, wspace=0.20)
        ]
        performance_axes = outer[1].subgridspec(
            2, 3, hspace=0.22, wspace=0.20
        ).subplots()
        exact_axis = fig.add_subplot(outer[2])
        plot_trajectories(trajectory_axes, frame)
        plot_performance(performance_axes, frame)
        exact_image = plot_heatmap(exact_axis, frame, "exact_recall", "Exact Recall@20")
        exact_bounds = exact_axis.get_position()
        colorbar_axis = fig.add_axes(
            [0.945, exact_bounds.y0, 0.012, exact_bounds.height]
        )
        fig.colorbar(exact_image, cax=colorbar_axis, label="Mean recall")
        panel_label(fig, trajectory_axes[0], "A")
        panel_label(fig, performance_axes[0, 0], "B")
        panel_label(fig, exact_axis, "C")
        fig.legend(
            handles=[
                Line2D([0], [0], color="#000000", label="Latent truth"),
                Line2D([0], [0], color="#B3B3B3", marker="o", linestyle="None", label="Observed"),
                Line2D([0], [0], color="#0072B2", marker="o", markerfacecolor="white", linestyle="-", label="Oracle"),
                Line2D([0], [0], color="#0072B2", marker="s", markerfacecolor="#0072B2", linestyle="--", label="Screened"),
            ],
            loc="upper center", bbox_to_anchor=(0.505, 0.988), ncol=4,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")
        plt.close(fig)


def write_metadata(path: Path, data_path: Path, output_path: Path, sources: list[Path], scientific: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "figure_id": "Figure 2", "specification_version": "v1.4", "status": "png_review_only", "created_at_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()), "generation_command": "uv run python -B experiments/plotting/plot_figure_02_oracle_screened.py", "figure_data": {"path": str(data_path.resolve())}, "source_files": [{"path": str(item.resolve())} for item in sources], "outputs": {"png": {"path": str(output_path.resolve())}}, "deferred_final_formats": ["svg", "pdf", "tiff"], "plotting_options": {"figure_size_inches": list(FIGURE_SIZE), "shared_main_bounds": {"left": LAYOUT_LEFT, "right": LAYOUT_RIGHT}, "panel_label_figure_x": PANEL_LABEL_X, "outer_height_ratios": [1.0, 1.64, 1.0], "outer_hspace": 0.40, "column_wspace": 0.20, "performance_row_hspace": 0.22, "colorbar_uses_independent_axis": True}, "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}, **scientific}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--simulation-root", type=Path, default=DEFAULT_SIMULATION_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame, sources, scientific = prepare_data(args.table_root, args.dataset_root, args.simulation_root)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.data, index=False)
    draw_figure(args.data, args.output)
    write_metadata(args.metadata, args.data, args.output, sources, scientific)


if __name__ == "__main__":
    main()

"""Prepare and plot the first four DynaGS supplementary figures.

Each figure is rendered only from its exported ``figure_SXX_data.csv``. Source
summary tables are used solely by the preparation step in the same command.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STYLE_ROOT = REPOSITORY_ROOT / ".codex/skills/dynags-figure/scripts"
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))

from plot_style import MODEL_STYLES, figure_size, paper_style, style_axis


DEFAULT_TABLE_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/tables"
DEFAULT_FIGURE_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/figures/supplementary"
DEFAULT_FIGURE_DATA_ROOT = DEFAULT_TABLE_ROOT / "figures"
DEFAULT_METADATA_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/manifests/figures"
REFERENCE_CONDITION = "P10000_NTR400_H0.8"
MODEL_ORDER = tuple(MODEL_STYLES)
MODE_STYLES = {
    "Oracle": {"color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    "Screened": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
}


def _source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve())}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _write_figure_data(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_png(fig: plt.Figure, output_stem: Path) -> dict[str, Path]:
    """Save a review figure as PNG only; final vector exports are deferred."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_stem.with_suffix(".png")
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return {"png": output_path}


def _export_metadata(
    figure_id: str,
    data_path: Path,
    source_paths: list[Path],
    figure_paths: dict[str, Path],
    metadata_root: Path,
    scientific: dict[str, Any],
) -> Path:
    metadata_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "figure_id": figure_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "generation_command": (
            "uv run python -B experiments/plotting/plot_supplementary.py "
            f"--figure {figure_id.lower()}"
        ),
        "figure_data": _source_record(data_path),
        "source_tables": [_source_record(path) for path in source_paths],
        "outputs": {
            extension: {"path": str(path.resolve())}
            for extension, path in figure_paths.items()
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "visual_style": {
            "profile": "double",
            "font_family": ["Arial", "DejaVu Sans"],
            "axis_label_pt": 8,
            "tick_and_legend_pt": 7,
            "panel_label_pt": 9,
            "png_dpi": 300,
            "final_vector_and_tiff_exports": "deferred_until_layout_approval",
        },
        **scientific,
    }
    path = metadata_root / f"figure_{figure_id}_metadata.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _prepare_s01(table_root: Path) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    raw_path = table_root / "metrics_by_time.csv"
    summary_path = table_root / "metrics_by_time_summary.csv"
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    filters = (
        raw["condition"].eq(REFERENCE_CONDITION)
        & raw["experiment"].isin(["E1", "E2"])
        & raw["target"].eq("observed")
        & raw["metric"].isin(["pcc", "nrmse"])
    )
    raw = raw.loc[
        filters,
        [
            "experiment", "condition", "p", "ntr", "heritability", "repetition",
            "model", "mode", "target", "time_index", "time", "metric", "value",
            "valid_n", "total_n",
        ],
    ].copy()
    raw["record_type"] = "replicate"
    for column in ("mean", "ci_low", "ci_high", "summary_valid_n", "summary_total_n"):
        raw[column] = np.nan

    summary_filter = (
        summary["condition"].eq(REFERENCE_CONDITION)
        & summary["experiment"].isin(["E1", "E2"])
        & summary["target"].eq("observed")
        & summary["metric"].isin(["pcc", "nrmse"])
    )
    selected_summary = summary.loc[summary_filter].copy()
    selected_summary["record_type"] = "summary"
    selected_summary["value"] = np.nan
    selected_summary["repetition"] = np.nan
    selected_summary["valid_n"] = np.nan
    selected_summary["total_n"] = np.nan
    selected_summary = selected_summary.rename(
        columns={"valid_n": "summary_valid_n", "total_n": "summary_total_n"}
    )
    # The rename above also affects the newly added placeholders only if pandas
    # encounters duplicate names, so assign from the original summary columns.
    selected_summary["summary_valid_n"] = summary.loc[summary_filter, "valid_n"].to_numpy()
    selected_summary["summary_total_n"] = summary.loc[summary_filter, "total_n"].to_numpy()
    columns = [
        "record_type", "experiment", "condition", "p", "ntr", "heritability",
        "repetition", "model", "mode", "target", "time_index", "time", "metric",
        "value", "mean", "ci_low", "ci_high", "valid_n", "total_n",
        "summary_valid_n", "summary_total_n",
    ]
    frame = pd.concat(
        [raw.reindex(columns=columns), selected_summary.reindex(columns=columns)],
        ignore_index=True,
    )
    if len(raw) != 800 or len(selected_summary) != 80:
        raise ValueError("Figure S01 expects 800 raw and 80 summary rows")
    return frame, [raw_path, summary_path], {
        "condition": REFERENCE_CONDITION,
        "target": "observed",
        "metrics": ["PCC", "NRMSE"],
        "replicate_unit": "independent_simulation_repetition",
        "uncertainty": "10,000-resample repetition-level percentile bootstrap 95% CI",
        "inclusion": "10/10 Oracle and 10/10 Screened runs",
        "exclusion": "none",
    }


def _plot_s01(data_path: Path, output_stem: Path) -> dict[str, Path]:
    frame = pd.read_csv(data_path)
    raw = frame[frame["record_type"].eq("replicate")]
    summary = frame[frame["record_type"].eq("summary")]
    with paper_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figure_size("double", 3.0),
            layout="constrained",
        )
        for ax, metric, label, ylabel in zip(
            axes,
            ("pcc", "nrmse"),
            ("A", "B"),
            ("PCC", "NRMSE"),
        ):
            for mode in ("Oracle", "Screened"):
                style = MODE_STYLES[mode]
                points = raw[(raw["metric"] == metric) & (raw["mode"] == mode)]
                stats = summary[(summary["metric"] == metric) & (summary["mode"] == mode)].sort_values("time")
                offset = -0.07 if mode == "Oracle" else 0.07
                ax.scatter(
                    points["time"] + offset,
                    points["value"],
                    s=6,
                    color=style["color"],
                    marker=style["marker"],
                    alpha=0.20,
                    linewidths=0,
                    zorder=1,
                )
                ax.fill_between(
                    stats["time"].to_numpy(),
                    stats["ci_low"].to_numpy(),
                    stats["ci_high"].to_numpy(),
                    color=style["color"],
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(
                    stats["time"],
                    stats["mean"],
                    label=mode,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markevery=2,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    zorder=3,
                )
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.5, 20.5)
            ax.set_xticks([1, 5, 10, 15, 20])
            if metric == "pcc":
                ax.set_ylim(-1.0, 1.0)
            else:
                ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            style_axis(ax, grid="y")
            _panel_label(ax, label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2)
        return _save_png(fig, output_stem)


def _prepare_s02(table_root: Path) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    raw_path = table_root / "curve_metrics_run_summary.csv"
    summary_path = table_root / "curve_metrics_condition_summary.csv"
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    metrics = [
        "ccc",
        "curve_nrmse",
        "inflection_abs_error",
        "within_one_sampling_interval",
    ]
    raw_filter = (
        raw["condition"].eq(REFERENCE_CONDITION)
        & raw["experiment"].isin(["E1", "E2"])
        & raw["target"].eq("latent")
        & raw["metric"].isin(metrics)
    )
    raw = raw.loc[
        raw_filter,
        [
            "experiment", "condition", "p", "ntr", "heritability", "repetition",
            "model", "mode", "target", "metric", "run_value",
            "aggregation_across_individuals", "valid_n", "total_n",
        ],
    ].copy()
    raw["record_type"] = "replicate"
    for column in ("mean", "ci_low", "ci_high", "summary_valid_n", "summary_total_n"):
        raw[column] = np.nan

    summary_filter = (
        summary["condition"].eq(REFERENCE_CONDITION)
        & summary["experiment"].isin(["E1", "E2"])
        & summary["target"].eq("latent")
        & summary["metric"].isin(metrics)
    )
    selected_summary = summary.loc[summary_filter].copy()
    selected_summary["record_type"] = "summary"
    selected_summary["run_value"] = np.nan
    selected_summary["repetition"] = np.nan
    selected_summary["valid_n_individual"] = np.nan
    selected_summary["total_n_individual"] = np.nan
    selected_summary = selected_summary.rename(
        columns={"valid_n": "summary_valid_n", "total_n": "summary_total_n"}
    )
    raw = raw.rename(
        columns={"valid_n": "valid_n_individual", "total_n": "total_n_individual"}
    )
    columns = [
        "record_type", "experiment", "condition", "p", "ntr", "heritability",
        "repetition", "model", "mode", "target", "metric", "run_value",
        "aggregation_across_individuals", "valid_n_individual", "total_n_individual",
        "mean", "ci_low", "ci_high", "summary_valid_n", "summary_total_n",
    ]
    frame = pd.concat(
        [raw.reindex(columns=columns), selected_summary.reindex(columns=columns)],
        ignore_index=True,
    )
    if len(raw) != 80 or len(selected_summary) != 8:
        raise ValueError("Figure S02 expects 80 raw and 8 summary rows")
    return frame, [raw_path, summary_path], {
        "condition": REFERENCE_CONDITION,
        "target": "latent truth",
        "replicate_unit": "independent_simulation_repetition",
        "curve_aggregation": {
            "CCC": "median across test individuals within each repetition",
            "curve NRMSE": "mean across test individuals within each repetition",
            "inflection-time MAE": "mean absolute error across valid test individuals",
            "within one interval": "proportion across valid test individuals",
        },
        "uncertainty": "10,000-resample repetition-level percentile bootstrap 95% CI",
        "inclusion": "10 paired Oracle/Screened repetitions",
        "exclusion": "none",
    }


def _plot_s02(data_path: Path, output_stem: Path) -> dict[str, Path]:
    frame = pd.read_csv(data_path)
    raw = frame[frame["record_type"].eq("replicate")]
    summary = frame[frame["record_type"].eq("summary")]
    panels = (
        ("ccc", "CCC", "A"),
        ("curve_nrmse", "Curve NRMSE", "B"),
        ("inflection_abs_error", "Inflection-time MAE", "C"),
        ("within_one_sampling_interval", "Within one interval", "D"),
    )
    with paper_style():
        fig, axes = plt.subplots(
            2,
            2,
            figsize=figure_size("double", 5.2),
            layout="constrained",
        )
        for ax, (metric, ylabel, label) in zip(axes.flat, panels):
            subset = raw[raw["metric"].eq(metric)]
            pivot = subset.pivot(index="repetition", columns="mode", values="run_value")
            for repetition, row in pivot.iterrows():
                ax.plot(
                    [0, 1],
                    [row["Oracle"], row["Screened"]],
                    color="#B3B3B3",
                    linewidth=0.65,
                    alpha=0.75,
                    zorder=1,
                )
            for x, mode in enumerate(("Oracle", "Screened")):
                style = MODE_STYLES[mode]
                values = subset[subset["mode"].eq(mode)]
                jitter = np.linspace(-0.045, 0.045, len(values))
                ax.scatter(
                    np.full(len(values), x) + jitter,
                    values["run_value"],
                    s=18,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=2,
                )
                stats = summary[(summary["metric"].eq(metric)) & (summary["mode"].eq(mode))].iloc[0]
                ax.errorbar(
                    x,
                    stats["mean"],
                    yerr=np.asarray([[stats["mean"] - stats["ci_low"]], [stats["ci_high"] - stats["mean"]]]),
                    fmt="D",
                    color="#000000",
                    markerfacecolor="white",
                    markersize=4,
                    linewidth=1.0,
                    capsize=2.2,
                    zorder=3,
                )
            ax.set_xticks([0, 1], ["Oracle", "Screened"])
            ax.set_ylabel(ylabel)
            ax.set_xlim(-0.35, 1.35)
            if metric == "ccc":
                values = subset["run_value"].to_numpy()
                padding = max(0.01, 0.12 * float(values.max() - values.min()))
                ax.set_ylim(max(-1.0, float(values.min()) - padding), min(1.0, float(values.max()) + padding))
            elif metric == "within_one_sampling_interval":
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            style_axis(ax, grid="y")
            _panel_label(ax, label)
        legend = [
            Line2D([0], [0], marker=MODE_STYLES[mode]["marker"], color="none", markerfacecolor=MODE_STYLES[mode]["color"], markeredgecolor="white", label=mode)
            for mode in ("Oracle", "Screened")
        ]
        fig.legend(legend, [item.get_label() for item in legend], loc="outside upper center", ncol=2)
        return _save_png(fig, output_stem)


def _prepare_s03(table_root: Path) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    source_path = table_root / "metrics_by_time_summary.csv"
    summary = pd.read_csv(source_path)
    filters = (
        summary["experiment"].eq("E3")
        & summary["condition"].eq(REFERENCE_CONDITION)
        & summary["target"].eq("observed")
        & summary["metric"].isin(["pcc", "nrmse"])
        & summary["model"].isin(MODEL_ORDER)
    )
    frame = summary.loc[
        filters,
        [
            "experiment", "condition", "p", "ntr", "heritability", "model", "mode",
            "target", "time_index", "time", "metric", "mean", "sd", "ci_low",
            "ci_high", "valid_n", "total_n", "bootstrap_replicates", "bootstrap_seed",
            "bootstrap_unit",
        ],
    ].copy()
    frame["model_order"] = frame["model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    frame = frame.sort_values(["metric", "model_order", "time_index"])
    if len(frame) != 480:
        raise ValueError("Figure S03 expects 12 models x 20 times x 2 metrics")
    return frame, [source_path], {
        "condition": REFERENCE_CONDITION,
        "target": "observed",
        "replicate_unit": "independent_simulation_repetition",
        "heatmap_value": "mean across 10 repetitions",
        "pcc_missingness": "undefined constant-prediction cells remain NA and are shown gray",
        "inclusion": "120/120 successful E3 runs",
        "exclusion": "no model or repetition excluded; E3 timing is not used",
        "model_display_order": list(MODEL_ORDER),
    }


def _plot_s03(data_path: Path, output_stem: Path) -> dict[str, Path]:
    frame = pd.read_csv(data_path)
    with paper_style():
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figure_size("double", 6.4),
            layout="constrained",
        )
        for ax, metric, title, label in (
            (axes[0], "pcc", "PCC (higher is better)", "A"),
            (axes[1], "nrmse", "NRMSE (lower is better)", "B"),
        ):
            subset = frame[frame["metric"].eq(metric)]
            matrix = (
                subset.pivot(index="model", columns="time_index", values="mean")
                .reindex(index=MODEL_ORDER, columns=range(20))
                .to_numpy(dtype=np.float64)
            )
            cmap = matplotlib.colormaps["viridis"].copy()
            cmap.set_bad("#D9D9D9")
            if metric == "pcc":
                image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
            else:
                image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=float(np.nanmax(matrix)))
            ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
            ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
            ax.set_xlabel("Time")
            ax.set_title(title, pad=4)
            ax.tick_params(length=2.5)
            for tick in ax.get_yticklabels():
                if tick.get_text() == "DynaGS":
                    tick.set_color(MODEL_STYLES["DynaGS"]["color"])
                    tick.set_fontweight("bold")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
            colorbar.ax.tick_params(labelsize=7, direction="out", width=0.8)
            _panel_label(ax, label)
        return _save_png(fig, output_stem)


def _prepare_s04(table_root: Path) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    metric_path = table_root / "metrics_overall_long.csv"
    history_path = table_root / "training_history_e1_e2.csv"
    metrics = pd.read_csv(metric_path)
    histories = pd.read_csv(history_path)
    candidate = metrics[
        metrics["condition"].eq(REFERENCE_CONDITION)
        & metrics["experiment"].isin(["E1", "E2"])
        & metrics["target"].eq("observed")
        & metrics["metric"].eq("pcc")
    ].copy()
    selections: dict[str, dict[str, Any]] = {}
    blocks = []
    for experiment in ("E1", "E2"):
        values = candidate[candidate["experiment"].eq(experiment)].copy()
        median = float(values["value"].median())
        values["distance_to_median"] = (values["value"] - median).abs()
        selected = values.sort_values(["distance_to_median", "repetition"]).iloc[0]
        repetition = int(selected["repetition"])
        selections[experiment] = {
            "mode": selected["mode"],
            "repetition": repetition,
            "observed_pcc": float(selected["value"]),
            "ten_repetition_median_pcc": median,
            "absolute_distance_to_median": float(selected["distance_to_median"]),
        }
        block = histories[
            histories["condition"].eq(REFERENCE_CONDITION)
            & histories["experiment"].eq(experiment)
            & histories["repetition"].eq(repetition)
        ].copy()
        block["selection_metric"] = "observed_test_pcc_closest_to_10_repetition_median"
        block["selection_metric_value"] = float(selected["value"])
        block["selection_metric_median"] = median
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    if frame.empty or set(frame["experiment"]) != {"E1", "E2"}:
        raise ValueError("Figure S04 representative histories are incomplete")
    return frame, [metric_path, history_path], {
        "condition": REFERENCE_CONDITION,
        "representative_repeat_rule": "observed PCC closest to the 10-repetition median, ties resolved by lower repetition number",
        "selected_repetitions": selections,
        "best_epoch_source": "saved training summary, carried into the unified history table",
        "inclusion": "all epochs from the selected Oracle and Screened runs",
        "exclusion": "no epoch excluded and no smoothing applied",
    }


def _plot_s04(data_path: Path, output_stem: Path) -> dict[str, Path]:
    frame = pd.read_csv(data_path)
    with paper_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figure_size("double", 3.1),
            layout="constrained",
        )
        for ax, experiment, label in zip(axes, ("E1", "E2"), ("A", "B")):
            subset = frame[frame["experiment"].eq(experiment)].sort_values("epoch")
            mode = str(subset["mode"].iloc[0])
            repetition = int(subset["repetition"].iloc[0])
            best_epoch = int(subset["best_epoch"].iloc[0])
            ax.plot(subset["epoch"], subset["train_loss"], color="#4D4D4D", linestyle="--", label="Training")
            ax.plot(subset["epoch"], subset["val_loss"], color="#0072B2", linestyle="-", label="Validation")
            ax.axvline(best_epoch, color="#E69F00", linestyle=":", linewidth=1.0, label="Selected epoch")
            selected_row = subset.iloc[int(np.argmin(np.abs(subset["epoch"].to_numpy() - best_epoch)))]
            ax.scatter(best_epoch, selected_row["val_loss"], marker="*", s=38, color="#E69F00", edgecolor="white", linewidth=0.5, zorder=3)
            ax.set_title(f"{mode} (rep {repetition:02d})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_xlim(left=1)
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            style_axis(ax, grid="y")
            _panel_label(ax, label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3)
        return _save_png(fig, output_stem)


FIGURES: dict[str, tuple[Callable, Callable]] = {
    "s01": (_prepare_s01, _plot_s01),
    "s02": (_prepare_s02, _plot_s02),
    "s03": (_prepare_s03, _plot_s03),
    "s04": (_prepare_s04, _plot_s04),
}


def generate(figure_key: str, args: argparse.Namespace) -> dict[str, Path]:
    prepare, plot = FIGURES[figure_key]
    figure_id = figure_key.upper()
    frame, source_paths, scientific = prepare(args.table_root)
    data_path = args.figure_data_root / f"figure_{figure_id}_data.csv"
    output_stem = args.figure_root / f"figure_{figure_id}"
    if args.metadata_only:
        png_path = output_stem.with_suffix(".png")
        if not png_path.is_file():
            raise FileNotFoundError(
                "Cannot refresh metadata because the PNG figure is missing: "
                + str(png_path)
            )
        figure_paths = {"png": png_path}
    else:
        _write_figure_data(frame, data_path)
        figure_paths = plot(data_path, output_stem)
    metadata_path = _export_metadata(
        figure_id,
        data_path,
        source_paths,
        figure_paths,
        args.metadata_root,
        scientific,
    )
    return {"data": data_path, "metadata": metadata_path, **figure_paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", *FIGURES), default="all")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Refresh metadata for existing figures without writing figure data or image files.",
    )
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--figure-data-root", type=Path, default=DEFAULT_FIGURE_DATA_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    args = parser.parse_args()
    for name in ("table_root", "figure_root", "figure_data_root", "metadata_root"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    args = parse_args()
    selected = FIGURES if args.figure == "all" else {args.figure: FIGURES[args.figure]}
    for figure_key in selected:
        outputs = generate(figure_key, args)
        print(f"figure_{figure_key.upper()}")
        for name, path in outputs.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

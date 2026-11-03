"""Plot PNG-only full-factor E1/E2 drafts and E3 metric bars."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
DEFAULT_FIGURE_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/figures/drafts"
DEFAULT_DATA_ROOT = DEFAULT_TABLE_ROOT / "figures"
DEFAULT_METADATA_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/manifests/figures"

NTR_LEVELS = (100, 200, 400, 800, 1600)
H2_LEVELS = (0.1, 0.4, 0.8)
P_LEVELS = (5000, 10000)
NTR_STYLES = {
    100: {"color": "#4D4D4D", "marker": "o"},
    200: {"color": "#E69F00", "marker": "s"},
    400: {"color": "#009E73", "marker": "D"},
    800: {"color": "#CC79A7", "marker": "^"},
    1600: {"color": "#0072B2", "marker": "v"},
}
MODEL_ORDER = tuple(MODEL_STYLES)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _write_metadata(
    figure_id: str,
    data_path: Path,
    source_paths: list[Path],
    png_path: Path,
    metadata_root: Path,
    scientific: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "figure_id": figure_id,
        "draft_status": "png_review_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "generation_command": (
            "uv run python -B experiments/plotting/plot_experiment_drafts.py "
            f"--figure {figure_id.lower()}"
        ),
        "figure_data": {
            "path": str(data_path.resolve()),
        },
        "source_tables": [
            {"path": str(path.resolve())}
            for path in source_paths
        ],
        "outputs": {
            "png": {"path": str(png_path.resolve())}
        },
        "deferred_final_formats": ["svg", "pdf", "tiff"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "replicate_unit": "independent_simulation_repetition",
        **scientific,
    }
    metadata_root.mkdir(parents=True, exist_ok=True)
    path = metadata_root / f"figure_{figure_id}_metadata.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _prepare_factor_data(experiment: str, table_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    source = table_root / "metrics_by_time_summary.csv"
    frame = pd.read_csv(source)
    selected = frame[
        frame["experiment"].eq(experiment)
        & frame["target"].eq("observed")
        & frame["metric"].isin(["pcc", "nrmse"])
        & frame["p"].isin(P_LEVELS)
        & frame["ntr"].isin(NTR_LEVELS)
        & frame["heritability"].isin(H2_LEVELS)
    ].copy()
    selected = selected.sort_values(
        ["metric", "p", "heritability", "ntr", "time_index"]
    )
    expected = 2 * 2 * 3 * 5 * 20
    if len(selected) != expected:
        raise ValueError(f"{experiment} full-factor figure expects {expected} rows")
    if not (selected["valid_n"] == 10).all():
        raise ValueError(f"{experiment} has an incomplete repetition summary")
    return selected, [source]


def _plot_factor(frame: pd.DataFrame, experiment: str, output: Path) -> None:
    row_specs = (
        ("pcc", 5000, "PCC", "P = 5,000"),
        ("pcc", 10000, "PCC", "P = 10,000"),
        ("nrmse", 5000, "NRMSE", "P = 5,000"),
        ("nrmse", 10000, "NRMSE", "P = 10,000"),
    )
    panel_labels = iter("ABCDEFGHIJKL")
    with paper_style():
        fig, axes = plt.subplots(
            4,
            3,
            figsize=figure_size("double", 7.1),
            sharex=True,
            layout="constrained",
        )
        nrmse_max = float(frame[frame["metric"].eq("nrmse")]["ci_high"].max()) * 1.06
        for row_index, (metric, p, metric_label, p_label) in enumerate(row_specs):
            for column_index, h2 in enumerate(H2_LEVELS):
                ax = axes[row_index, column_index]
                subset = frame[
                    frame["metric"].eq(metric)
                    & frame["p"].eq(p)
                    & np.isclose(frame["heritability"], h2)
                ]
                for ntr in NTR_LEVELS:
                    values = subset[subset["ntr"].eq(ntr)].sort_values("time")
                    style = NTR_STYLES[ntr]
                    ax.fill_between(
                        values["time"].to_numpy(),
                        values["ci_low"].to_numpy(),
                        values["ci_high"].to_numpy(),
                        color=style["color"],
                        alpha=0.075,
                        linewidth=0,
                    )
                    ax.plot(
                        values["time"],
                        values["mean"],
                        color=style["color"],
                        marker=style["marker"],
                        markerfacecolor="white",
                        markeredgewidth=0.65,
                        markevery=3,
                        linewidth=1.05,
                        label=str(ntr),
                    )
                if row_index == 0:
                    ax.set_title(f"h² = {h2:g}", pad=4)
                if column_index == 0:
                    ax.set_ylabel(f"{metric_label}\n{p_label}")
                if row_index == 3:
                    ax.set_xlabel("Time")
                ax.set_xlim(0.5, 20.5)
                ax.set_xticks([1, 5, 10, 15, 20])
                if metric == "pcc":
                    ax.set_ylim(-1.0, 1.0)
                else:
                    ax.set_ylim(0.0, nrmse_max)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                style_axis(ax, grid="y")
                _panel_label(ax, next(panel_labels))
        handles = [
            Line2D(
                [0],
                [0],
                color=NTR_STYLES[ntr]["color"],
                marker=NTR_STYLES[ntr]["marker"],
                markerfacecolor="white",
                markeredgewidth=0.65,
                label=f"NTR = {ntr}",
            )
            for ntr in NTR_LEVELS
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _prepare_e3(table_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    raw_path = table_root / "metrics_overall_long.csv"
    summary_path = table_root / "condition_summary.csv"
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    metrics = ["pcc", "nrmse", "time_adjusted_r2"]
    raw = raw[
        raw["experiment"].eq("E3")
        & raw["target"].eq("observed")
        & raw["metric"].isin(metrics)
        & raw["model"].isin(MODEL_ORDER)
    ][
        [
            "experiment", "condition", "repetition", "model", "target", "metric",
            "value", "valid_n", "total_n",
        ]
    ].copy()
    raw["record_type"] = "replicate"
    for column in ("mean", "ci_low", "ci_high", "summary_valid_n", "summary_total_n"):
        raw[column] = np.nan
    selected_summary = summary[
        summary["experiment"].eq("E3")
        & summary["target"].eq("observed")
        & summary["metric"].isin(metrics)
        & summary["model"].isin(MODEL_ORDER)
    ].copy()
    selected_summary["record_type"] = "summary"
    selected_summary["value"] = np.nan
    selected_summary["repetition"] = np.nan
    selected_summary = selected_summary.rename(
        columns={"valid_n": "summary_valid_n", "total_n": "summary_total_n"}
    )
    selected_summary["valid_n"] = np.nan
    selected_summary["total_n"] = np.nan
    columns = [
        "record_type", "experiment", "condition", "repetition", "model", "target",
        "metric", "value", "mean", "ci_low", "ci_high", "valid_n", "total_n",
        "summary_valid_n", "summary_total_n",
    ]
    frame = pd.concat(
        [raw.reindex(columns=columns), selected_summary.reindex(columns=columns)],
        ignore_index=True,
    )
    if len(raw) != 360 or len(selected_summary) != 36:
        raise ValueError("E3 bar figure expects 360 raw and 36 summary rows")
    return frame, [raw_path, summary_path]


def _plot_e3_bars(frame: pd.DataFrame, output: Path) -> None:
    raw = frame[frame["record_type"].eq("replicate")]
    summary = frame[frame["record_type"].eq("summary")]
    panels = (
        ("pcc", "PCC", "A"),
        ("nrmse", "NRMSE", "B"),
        ("time_adjusted_r2", "Time-adjusted R²", "C"),
    )
    positions = np.arange(len(MODEL_ORDER))
    rng = np.random.default_rng(73)
    jitter_template = np.linspace(-0.18, 0.18, 10)
    with paper_style():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figure_size("double", 5.1),
            sharey=True,
            layout="constrained",
        )
        for ax, (metric, xlabel, panel) in zip(axes, panels):
            stats = summary[summary["metric"].eq(metric)].set_index("model").reindex(MODEL_ORDER)
            for y, model in enumerate(MODEL_ORDER):
                style = MODEL_STYLES[model]
                mean = stats.loc[model, "mean"]
                if pd.isna(mean):
                    continue
                ax.barh(
                    y,
                    mean,
                    height=0.58,
                    color=style["color"],
                    alpha=0.68,
                    edgecolor="#333333",
                    linewidth=0.45,
                    zorder=1,
                )
                ax.errorbar(
                    mean,
                    y,
                    xerr=np.asarray(
                        [[mean - stats.loc[model, "ci_low"]], [stats.loc[model, "ci_high"] - mean]]
                    ),
                    fmt="none",
                    color="#1A1A1A",
                    linewidth=0.8,
                    capsize=1.8,
                    zorder=3,
                )
                points = raw[(raw["metric"].eq(metric)) & (raw["model"].eq(model))]
                finite = points["value"].dropna().to_numpy()
                ax.scatter(
                    finite,
                    y + jitter_template[: len(finite)],
                    s=8,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.9,
                    zorder=4,
                )
            ax.set_xlabel(xlabel)
            ax.set_yticks(positions, MODEL_ORDER)
            ax.invert_yaxis()
            if metric == "pcc":
                ax.set_xlim(-1.0, 1.0)
                mean_position = MODEL_ORDER.index("Mean")
                ax.text(-0.94, mean_position, "NA", va="center", ha="left", fontsize=7, color="#4D4D4D")
            elif metric == "nrmse":
                ax.set_xlim(left=0.0)
            else:
                lower = min(-0.2, float(raw[raw["metric"].eq(metric)]["value"].min()) * 1.1)
                upper = max(0.65, float(raw[raw["metric"].eq(metric)]["value"].max()) * 1.1)
                ax.set_xlim(lower, upper)
                ax.axvline(0.0, color="#7A7A7A", linestyle="--", linewidth=0.75, zorder=0)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            style_axis(ax, grid="x")
            _panel_label(ax, panel)
        for tick in axes[0].get_yticklabels():
            if tick.get_text() == "DynaGS":
                tick.set_color(MODEL_STYLES["DynaGS"]["color"])
                tick.set_fontweight("bold")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def generate(figure: str, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.figure_root.mkdir(parents=True, exist_ok=True)
    if figure in {"e1", "e2"}:
        experiment = figure.upper()
        frame, sources = _prepare_factor_data(experiment, args.table_root)
        figure_id = f"{experiment}_full_factor_draft"
        data_path = args.data_root / f"figure_{figure_id}_data.csv"
        png_path = args.figure_root / f"figure_{figure_id}.png"
        frame.to_csv(data_path, index=False)
        plotted = pd.read_csv(data_path)
        _plot_factor(plotted, experiment, png_path)
        metadata = _write_metadata(
            figure_id,
            data_path,
            sources,
            png_path,
            args.metadata_root,
            {
                "experiment": experiment,
                "target": "observed",
                "metrics": ["PCC", "NRMSE"],
                "factor_encoding": {
                    "time": "x-axis using actual coordinates 1..20",
                    "ntr": "five colored and marked mean trajectories",
                    "heritability": "three columns",
                    "p": "separate rows within each metric",
                },
                "uncertainty": "10,000-resample repetition-level percentile bootstrap 95% CI",
                "inclusion": "30 conditions x 10 repetitions",
                "exclusion": "none",
            },
        )
    else:
        frame, sources = _prepare_e3(args.table_root)
        figure_id = "E3_metric_bars_draft"
        data_path = args.data_root / f"figure_{figure_id}_data.csv"
        png_path = args.figure_root / f"figure_{figure_id}.png"
        frame.to_csv(data_path, index=False)
        plotted = pd.read_csv(data_path)
        _plot_e3_bars(plotted, png_path)
        metadata = _write_metadata(
            figure_id,
            data_path,
            sources,
            png_path,
            args.metadata_root,
            {
                "experiment": "E3",
                "condition": "P10000_NTR400_H0.8",
                "target": "observed",
                "metrics": ["PCC", "NRMSE", "Time-adjusted R²"],
                "bar_statistic": "mean across 10 independent repetitions",
                "uncertainty": "10,000-resample repetition-level percentile bootstrap 95% CI",
                "raw_points": "all available repetition-level values overlaid",
                "missing_values": "Mean PCC remains NA and is labeled NA",
                "inclusion": "120/120 E3 runs",
                "exclusion": "E3 timing and resources are not used",
            },
        )
    return data_path, metadata, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "e1", "e2", "e3"), default="all")
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    args = parser.parse_args()
    for name in ("table_root", "figure_root", "data_root", "metadata_root"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> None:
    args = parse_args()
    selected = ("e1", "e2", "e3") if args.figure == "all" else (args.figure,)
    for figure in selected:
        data, metadata, png = generate(figure, args)
        print(f"{figure}: data={data}")
        print(f"{figure}: metadata={metadata}")
        print(f"{figure}: png={png}")


if __name__ == "__main__":
    main()

"""Prepare and draw the E3 Figure 3 PNG directly from metrics.json files."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex/skills/dynags-figure/scripts"))
from plot_style import MODEL_STYLES, paper_style, style_axis


TABLES = ROOT / "outputs/paper_experiments/tables"
RUNS = ROOT / "outputs/paper_experiments/runs/E3/P10000_NTR400_H0.8"
PAIR_DATA = TABLES / "figures/fig3_pairwise_effects.csv"
TIME_DATA = TABLES / "figures/fig3_timewise_metrics.csv"
REPORT = TABLES / "figures/fig3_results.md"
METADATA = ROOT / "outputs/paper_experiments/manifests/figures/figure_03_metadata.json"
PNG = ROOT / "outputs/paper_experiments/figures/fig3_model_comparison.png"
MODELS = ("DynaGS", "XGBoost", "Random Forest", "Cropformer")
BASELINES = (
    "XGBoost", "Random Forest", "Ridge-A+D", "Ridge-All", "GBLUP-All",
    "RR-GBLUP", "DNNGP-SNP", "DLGWAS", "Cropformer", "WheatGP", "Mean",
)
MARKERS = {"DynaGS": "o", "XGBoost": "h", "Random Forest": "P", "Cropformer": "p"}
SLUG_TO_MODEL = {
    "dynags": "DynaGS", "xgboost": "XGBoost", "random_forest": "Random Forest",
    "cropformer": "Cropformer", "ridge_a_plus_d": "Ridge-A+D",
    "ridge_all": "Ridge-All", "gblup_all": "GBLUP-All", "rr_gblup": "RR-GBLUP",
    "dnngp_snp": "DNNGP-SNP", "dlgwas": "DLGWAS",
    "wheatgp": "WheatGP", "mean": "Mean",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 73


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].str.strip()
    return frame


def bootstrap_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Summarize ten independent repetitions without changing source metrics."""
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Every Figure 3 summary requires 10 finite repetitions")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, 10, size=(BOOTSTRAP_REPLICATES, 10))
    low, high = np.quantile(values[indexes].mean(axis=1), (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def prepare_data() -> list[Path]:
    """Read only E3 metrics.json inputs; preserve their reported metric values."""
    json_paths = sorted(RUNS.glob("rep_*/*/metrics.json"))
    if len(json_paths) != 120:
        raise ValueError("E3 requires 10 repetitions × 12 formal metrics.json files")
    overall_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    for path in json_paths:
        repetition = int(path.parent.parent.name.removeprefix("rep_"))
        model = SLUG_TO_MODEL[path.parent.name]
        metrics = json.loads(path.read_text(encoding="utf-8"))
        overall = metrics["overall"]
        if len(metrics["by_time"]) != 20:
            raise ValueError(f"Expected 20 recorded times: {path}")
        for metric in ("pcc", "nrmse"):
            value = overall[metric]
            if value is not None:
                overall_rows.append({"repetition": repetition, "model": model,
                                     "metric": metric, "value": float(value)})
        if model in MODELS:
            for point in metrics["by_time"]:
                for metric in ("pcc", "nrmse"):
                    value = point[metric]
                    if value is not None:
                        time_rows.append({"repetition": repetition, "model": model,
                                          "metric": metric, "time": float(point["time"]),
                                          "value": float(value)})
    overall = pd.DataFrame(overall_rows)
    time_raw = pd.DataFrame(time_rows)
    if overall.groupby(["model", "metric"])["repetition"].nunique().min() != 10:
        raise ValueError("Overall metrics are missing paired repetitions")
    pair_rows: list[dict[str, object]] = []
    for metric in ("pcc", "nrmse"):
        wide = overall[overall["metric"].eq(metric)].pivot(
            index="repetition", columns="model", values="value"
        )
        for model in BASELINES:
            if metric == "pcc" and model == "Mean":
                continue
            if model not in wide or wide[["DynaGS", model]].isna().any().any():
                raise ValueError(f"Incomplete {metric} pairing for {model}")
            values = (wide["DynaGS"] - wide[model] if metric == "pcc"
                      else wide[model] - wide["DynaGS"])
            for repetition, value in values.items():
                pair_rows.append({"record_type": "repeat", "metric": metric,
                                  "comparison_model": model, "repetition": repetition,
                                  "value": float(value)})
            mean, low, high = bootstrap_summary(values.to_numpy(float))
            pair_rows.append({"record_type": "summary", "metric": metric,
                              "comparison_model": model, "repetition": None,
                              "value": mean, "ci_low": low, "ci_high": high})
        pair_rows.append({"record_type": "reference", "metric": metric,
                          "comparison_model": "DynaGS", "repetition": None,
                          "value": 0.0})
    pair_frame = pd.DataFrame(pair_rows)[
        ["record_type", "metric", "comparison_model", "repetition", "value", "ci_low", "ci_high"]
    ]
    time_summaries: list[dict[str, object]] = []
    for (model, metric, coordinate), group in time_raw.groupby(["model", "metric", "time"]):
        ordered = group.sort_values("repetition")
        if ordered["repetition"].nunique() != 10:
            raise ValueError(f"Incomplete timewise metrics: {model}, {metric}, {coordinate}")
        mean, low, high = bootstrap_summary(ordered["value"].to_numpy(float))
        time_summaries.append({"model": model, "metric": metric, "time": coordinate,
                               "mean": mean, "ci_low": low, "ci_high": high})
    time_frame = pd.DataFrame(time_summaries)
    if len(pair_frame) != 233 or len(time_frame) != 160:
        raise ValueError("Figure 3 expected 210 paired repeats, 21 summaries, 2 references, and 160 time summaries")
    for path, frame in ((PAIR_DATA, pair_frame), (TIME_DATA, time_frame)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return json_paths


def forest(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    raw = data[data["record_type"].eq("repeat") & data["metric"].eq(metric)]
    summary = data[data["record_type"].eq("summary") & data["metric"].eq(metric)]
    reference = data[data["record_type"].eq("reference") & data["metric"].eq(metric)]
    if len(reference) != 1 or reference.iloc[0]["comparison_model"] != "DynaGS" or reference.iloc[0]["value"] != 0:
        raise ValueError("DynaGS reference row must be an explicit zero without a confidence interval")
    baselines = [model for model in BASELINES if model in set(summary["comparison_model"])]
    summary = summary.set_index("comparison_model").loc[baselines]
    y = np.arange(1, len(baselines) + 1)
    all_values = np.concatenate((raw["value"].to_numpy(float), summary["ci_low"].to_numpy(float), summary["ci_high"].to_numpy(float)))
    extent = max(abs(float(all_values.min())), abs(float(all_values.max()))) * 1.10
    for position, model in enumerate(baselines, start=1):
        values = raw.loc[raw["comparison_model"].eq(model)].sort_values("repetition")["value"].to_numpy(float)
        ax.scatter(values, np.full(10, position) + np.linspace(-0.14, 0.14, 10),
                   s=6, color="#0072B2", alpha=0.22, linewidths=0, zorder=2)
    means = summary["value"].to_numpy(float)
    ax.errorbar(means, y, xerr=np.vstack((means - summary["ci_low"].to_numpy(float),
                                         summary["ci_high"].to_numpy(float) - means)),
                fmt="o", color="#0072B2", markerfacecolor="white", markersize=3.5,
                linewidth=0.85, capsize=1.8, zorder=3)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=0.7, zorder=1)
    ax.scatter([0], [0], marker="D", s=26, color="#0072B2", zorder=4)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(len(baselines) + 0.45, -0.55)
    ax.set_yticks(np.arange(len(baselines) + 1), ["DynaGS (reference)", *baselines])
    ax.tick_params(axis="y", labelsize=6.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel("ΔPCC" if metric == "pcc" else "ΔNRMSE", labelpad=2)
    ax.set_title("PCC advantage" if metric == "pcc" else "NRMSE advantage", pad=4)
    style_axis(ax, grid="x")


def time_plot(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    for model in MODELS:
        subset = data[data["model"].eq(model) & data["metric"].eq(metric)].sort_values("time")
        color = MODEL_STYLES[model]["color"]
        x = subset["time"].to_numpy(float)
        ax.fill_between(x, subset["ci_low"].to_numpy(float), subset["ci_high"].to_numpy(float),
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(x, subset["mean"].to_numpy(float), color=color,
                linestyle=MODEL_STYLES[model]["linestyle"], marker=MARKERS[model],
                markevery=4, markersize=3.2, linewidth=1.1)
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Time")
    ax.set_ylabel("PCC ↑" if metric == "pcc" else "NRMSE ↓")
    ax.set_title("PCC over time" if metric == "pcc" else "NRMSE over time", pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    style_axis(ax, grid="y")


def draw_figure() -> None:
    pair_data = read_table(PAIR_DATA)
    time_data = read_table(TIME_DATA)
    with paper_style():
        fig = plt.figure(figsize=(7.1, 6.35))
        grid = fig.add_gridspec(2, 2, left=0.19, right=0.96, bottom=0.085,
                               top=0.90, wspace=0.64, hspace=0.26,
                               height_ratios=(1.53, 1.0))
        axes = [[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)]
        forest(axes[0][0], pair_data, "pcc")
        forest(axes[0][1], pair_data, "nrmse")
        time_plot(axes[1][0], time_data, "pcc")
        time_plot(axes[1][1], time_data, "nrmse")
        for label, ax in zip("ABCD", [*axes[0], *axes[1]]):
            bounds = ax.get_position()
            fig.text(bounds.x0 - 0.16, bounds.y1 + 0.008, label,
                     fontsize=9, fontweight="bold", ha="left", va="bottom")
        handles = [Line2D([0], [0], label=model, color=MODEL_STYLES[model]["color"],
                          marker=MARKERS[model], linestyle=MODEL_STYLES[model]["linestyle"],
                          markersize=4) for model in MODELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.985),
                   ncol=4, frameon=False)
        fig.text(0.575, 0.027, "Positive values favour DynaGS", ha="center", fontsize=7)
        PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG, dpi=300, facecolor="white")
        plt.close(fig)


def main() -> None:
    source_paths = prepare_data()
    draw_figure()
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "figure_id": "Figure 3", "status": "png_review_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source_type": "metrics.json only",
        "source_files": [str(path.resolve()) for path in source_paths],
        "source_fields": {"forest": ["overall.pcc", "overall.nrmse"],
                          "time_curves": ["by_time.time", "by_time.pcc", "by_time.nrmse"]},
        "source_values_recomputed": False,
        "figure_data": [str(PAIR_DATA.resolve()), str(TIME_DATA.resolve())],
        "result_report": str(REPORT.resolve()),
        "output_png": str(PNG.resolve()),
        "scope": "E3; P10000_NTR400_H0.8; observed phenotype; 10 paired independent repeats",
        "pairwise_effects": "PCC: DynaGS minus baseline; NRMSE: baseline minus DynaGS",
        "forest_reference": "DynaGS appears explicitly at zero as the reference diamond; no CI is defined for a model compared with itself",
        "plotting_options": {"figure_size_inches": [7.1, 6.35], "legend_center_x": 0.50,
                             "panel_hspace": 0.26, "panel_wspace": 0.64,
                             "shared_plotting_bounds": {"left": 0.19, "right": 0.96}},
        "uncertainty": "repetition-level percentile bootstrap 95% CI; 10000 resamples; seed 73",
        "source_count": len(source_paths),
        "specific_source_repair": {
            "path": str((RUNS / "rep_01/wheatgp/metrics.json").resolve()),
            "fields": {"overall.pcc": -0.06089955036066654,
                       "overall.time_adjusted_r2": -0.015306731509561411},
            "basis": "PCC agrees with by_time mean; R2 agrees with same-run test_predictions.npz under existing metric protocol",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    METADATA.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

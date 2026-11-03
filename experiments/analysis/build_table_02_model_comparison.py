"""Build manuscript Table 2 directly from formal E3 metrics.json values."""

from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "outputs/paper_experiments/runs/E3/P10000_NTR400_H0.8"
OUTPUT_ROOT = ROOT / "outputs/paper_experiments/tables"
DATA_PATH = OUTPUT_ROOT / "table_02_model_comparison.csv"
MARKDOWN_PATH = OUTPUT_ROOT / "table_02_model_comparison.md"
LATEX_PATH = OUTPUT_ROOT / "table_02_model_comparison.tex"
METADATA_PATH = ROOT / "outputs/paper_experiments/manifests/table_02_metadata.json"
MODEL_SLUGS = (
    ("Mean", "mean"), ("Ridge-All", "ridge_all"),
    ("Ridge-A+D", "ridge_a_plus_d"), ("GBLUP-All", "gblup_all"),
    ("RR-GBLUP", "rr_gblup"), ("Random Forest", "random_forest"),
    ("XGBoost", "xgboost"), ("DNNGP-SNP", "dnngp_snp"),
    ("DLGWAS", "dlgwas"), ("WheatGP", "wheatgp"),
    ("Cropformer", "cropformer"), ("DynaGS", "dynags"),
)
METRICS = (
    ("pcc", "PCC ↑", 3, "max"),
    ("nrmse", "NRMSE ↓", 4, "min"),
    ("time_adjusted_r2", "Time-adjusted R² ↑", 3, "max"),
)


def summarize() -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, object]] = []
    sources: list[Path] = []
    for model, slug in MODEL_SLUGS:
        by_metric: dict[str, list[float]] = {metric: [] for metric, *_ in METRICS}
        for repetition in range(1, 11):
            path = RUN_ROOT / f"rep_{repetition:02d}" / slug / "metrics.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("metrics_protocol") != "time_adjusted_r2_nan_pcc_v2":
                raise ValueError(f"Metric protocol differs from frozen E3 protocol: {path}")
            overall = document["overall"]
            sources.append(path)
            for metric in by_metric:
                value = overall[metric]
                if value is None:
                    if metric == "pcc" and model == "Mean":
                        continue
                    raise ValueError(f"Unexpected missing {metric}: {path}")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {metric}: {path}")
                by_metric[metric].append(value)
        row: dict[str, object] = {"model": model}
        for metric, _, _, _ in METRICS:
            values = np.asarray(by_metric[metric], dtype=float)
            expected = 0 if metric == "pcc" and model == "Mean" else 10
            if len(values) != expected:
                raise ValueError(f"Expected {expected} {metric} values for {model}")
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) else np.nan
        rows.append(row)
    if len(sources) != 120:
        raise ValueError("Table 2 requires 12 models × 10 E3 metrics.json files")
    return pd.DataFrame(rows), sources


def write_outputs(frame: pd.DataFrame, sources: list[Path]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    best = {}
    for metric, _, _, direction in METRICS:
        values = frame[f"{metric}_mean"].dropna()
        best[metric] = float(values.max() if direction == "max" else values.min())

    def cell(row: pd.Series, metric: str, digits: int, latex: bool) -> str:
        mean = row[f"{metric}_mean"]
        if pd.isna(mean):
            return "N/A"
        if latex:
            value = f"${mean:.{digits}f} \\pm {row[f'{metric}_sd']:.{digits}f}$"
        else:
            value = f"{mean:.{digits}f} ± {row[f'{metric}_sd']:.{digits}f}"
        if np.isclose(mean, best[metric], rtol=0, atol=1e-12):
            return f"\\textbf{{{value}}}" if latex else f"**{value}**"
        return value

    headers = ["Model", *[label for _, label, _, _ in METRICS]]
    markdown = [
        "# Table 2. Overall predictive performance on simulated E3 data",
        "",
        "| " + " | ".join(headers) + " |",
        "|:--|--:|--:|--:|",
    ]
    for _, row in frame.iterrows():
        markdown.append("| " + " | ".join(
            [str(row["model"]), *[cell(row, metric, digits, False) for metric, _, digits, _ in METRICS]]
        ) + " |")
    markdown.extend([
        "",
        "Values are mean ± sample SD across 10 independent E3 simulation repetitions",
        "(P = 10,000; NTR = 400; h² = 0.8; QTL = 20; T = 20).",
        "PCC is undefined for the training-timepoint Mean baseline.",
        "Bold marks the best mean in each metric column; negative R² values are retained.",
        "All metric values come directly from each run's `metrics.json` `overall` block.",
        "CI and paired comparisons belong in Table S1, not this table.",
        "",
    ])
    MARKDOWN_PATH.write_text("\n".join(markdown), encoding="utf-8")

    latex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & PCC $\uparrow$ & NRMSE $\downarrow$ & Time-adjusted $R^2$ $\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        latex_lines.append(" & ".join(
            [str(row["model"]), *[cell(row, metric, digits, True) for metric, _, digits, _ in METRICS]]
        ) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    LATEX_PATH.write_text("\n".join(latex_lines), encoding="utf-8")

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "table_id": "Table 2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "generation_command": "uv run python -B experiments/analysis/build_table_02_model_comparison.py",
        "source_type": "metrics.json overall fields only",
        "source_files": [str(path.resolve()) for path in sources],
        "outputs": {"csv": str(DATA_PATH.resolve()), "markdown": str(MARKDOWN_PATH.resolve()),
                    "latex": str(LATEX_PATH.resolve())},
        "condition": "P10000_NTR400_H0.8", "repetitions_per_model": 10,
        "metric_protocol": "time_adjusted_r2_nan_pcc_v2",
        "sd_ddof": 1, "display_precision": {"pcc": 3, "nrmse": 4, "time_adjusted_r2": 3},
        "model_order": [model for model, _ in MODEL_SLUGS],
        "source_values_recomputed": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    frame, sources = summarize()
    write_outputs(frame, sources)


if __name__ == "__main__":
    main()

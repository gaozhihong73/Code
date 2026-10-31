"""从 E3 run manifests 生成统一长表；不读取训练日志猜测字段。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.e3.schemas.e3 import E3RunManifest


def summarize(run_root: Path, output_dir: Path) -> dict[str, Path]:
    manifests = sorted(run_root.rglob("run_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"未找到 run_manifest.json: {run_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows, overall_rows, time_rows, resource_rows, failures = [], [], [], [], []
    seen = set()
    for path in manifests:
        run = E3RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if run.run_id in seen:
            raise ValueError(f"重复 run_id: {run.run_id}")
        seen.add(run.run_id)
        row = {
            "run_id": run.run_id,
            "condition": run.condition,
            "repetition": run.repetition,
            "model": run.model,
            "seed": run.seed,
            "status": run.status,
            "manifest_path": str(path.resolve()),
        }
        if run.data is not None:
            row.update(
                split_sha256=run.data.split.sha256,
                raw_snp_count=run.data.raw_snp_count,
                model_input_snp_count=run.data.model_input_snp_count,
            )
        run_rows.append(row)
        if run.status != "succeeded":
            if run.status == "failed":
                failures.append(
                    {
                        **row,
                        "exception_type": run.failure.exception_type,
                        "message": run.failure.message,
                    }
                )
            continue
        metrics = json.loads(Path(run.prediction.metrics_path).read_text(encoding="utf-8"))
        for metric, value in metrics["overall"].items():
            overall_rows.append(
                {
                    "run_id": run.run_id,
                    "condition": run.condition,
                    "repetition": run.repetition,
                    "model": run.model,
                    "target": "observed",
                    "metric": metric,
                    "value": value,
                }
            )
        for item in metrics["by_time"]:
            for metric in ("rmse", "nrmse", "mae", "pcc", "spearman"):
                time_rows.append(
                    {
                        "run_id": run.run_id,
                        "condition": run.condition,
                        "repetition": run.repetition,
                        "model": run.model,
                        "time_index": item["time_index"],
                        "time": item["time"],
                        "metric": metric,
                        "value": item[metric],
                        "valid_n": item["valid_n"],
                    }
                )
        resource_rows.append(
            {
                "run_id": run.run_id,
                "condition": run.condition,
                "repetition": run.repetition,
                "model": run.model,
                **run.resources.model_dump(),
                "best_epoch": run.training.best_epoch,
                "parameter_count": run.training.parameter_count,
            }
        )

    outputs = {
        "run_index": output_dir / "run_index.csv",
        "metrics_overall_long": output_dir / "metrics_overall_long.csv",
        "metrics_by_time": output_dir / "metrics_by_time.csv",
        "resource_metrics": output_dir / "resource_metrics.csv",
        "run_failures": output_dir / "run_failures.csv",
    }
    pd.DataFrame(run_rows).to_csv(outputs["run_index"], index=False)
    pd.DataFrame(overall_rows).to_csv(outputs["metrics_overall_long"], index=False)
    pd.DataFrame(time_rows).to_csv(outputs["metrics_by_time"], index=False)
    pd.DataFrame(resource_rows).to_csv(outputs["resource_metrics"], index=False)
    pd.DataFrame(failures).to_csv(outputs["run_failures"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/paper_experiments/runs/E3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_experiments/tables/E3"))
    args = parser.parse_args()
    for name, path in summarize(args.run_root.resolve(), args.output_dir.resolve()).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

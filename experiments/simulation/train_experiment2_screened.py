"""实验二：批量执行未知QTL筛选后TFPP训练。

独立入口，不依赖仓库根目录的train_sim脚本。默认遍历正式清单中的全部
300套数据；每项由独立子进程执行，输出与实验一完全隔离。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import build_tfpp_config
from trainer.TFPP_Trainer import TFPPTrainer

DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "dataset/experiments"
DEFAULT_CONFIG = REPOSITORY_ROOT / "config/profiles/simulation.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/simulation/experiment2_screened"
REDUNDANT_HISTORY = tuple(
    f"phenotype_{metric}_history.csv" for metric in ("mse", "rmse", "mae", "pcc", "r2")
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _data_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "data_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据清单: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _snp_file(data_dir: Path, manifest: dict[str, Any]) -> Path:
    candidates = [
        data_dir / name
        for name in manifest.get("model_inputs", [])
        if Path(name).name.startswith("sim_snp")
    ]
    candidates.extend((data_dir / "sim_snp.npy", data_dir / "sim_snp.csv"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到sim_snp.npy或sim_snp.csv: {data_dir}")


def build_training_config(
    data_dir: Path, config_file: Path, output_root: Path, device: str
) -> Any:
    """构建未知QTL条件筛选配置，筛选参数来自正式profile。"""
    data_dir = data_dir.resolve()
    manifest = _data_manifest(data_dir)
    if manifest.get("split", {}).get("seed") != 73:
        raise ValueError(f"正式协议要求split seed=73: {data_dir}")
    split_file = data_dir / "splits/split_seed73.npz"
    sample_ids_file = data_dir / "sample_ids.csv"
    for path in (split_file, sample_ids_file, data_dir / "sim_pheno.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"缺少训练输入: {path}")
    return build_tfpp_config(
        config_file,
        exp_name=f"{data_dir.parent.name}/{data_dir.name}",
        output_root=str(output_root.resolve()),
        device=device,
        snp_file=str(_snp_file(data_dir, manifest)),
        pheno_dir=str(data_dir),
        phenotype_names=["sim_pheno"],
        time_coordinates=manifest["time_coordinates"],
        split_file=str(split_file),
        sample_ids_file=str(sample_ids_file),
        preselected_snp_file=None,
    )


def _compact(run_dir: Path) -> list[str]:
    """删除重复表示，并按保留文件重新封存checkpoint完整性清单。"""
    removed = []
    paths = [run_dir / "training_history" / name for name in REDUNDANT_HISTORY]
    paths.extend((run_dir / "test_predictions.csv", *run_dir.glob("*.log")))
    for path in paths:
        if path.is_file():
            path.unlink()
            removed.append(str(path.relative_to(run_dir)))
    from utils.ArtifactIntegrity import seal_checkpoint

    seal_checkpoint(
        run_dir,
        extra_files=(
            "test_evaluation.json",
            "test_predictions.npz",
            "preprocessing/screened_indices.npy",
        ),
    )
    return removed


def _is_complete(run_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            run_dir / "run_complete.json",
            run_dir / "models/best_model.pt",
            run_dir / "models/checkpoint_integrity.json",
            run_dir / "training_history/training_summary.json",
            run_dir / "training_history/training_history_metrics.csv",
            run_dir / "test_predictions.npz",
            run_dir / "test_evaluation.json",
            run_dir / "preprocessing/preprocessing.json",
            run_dir / "preprocessing/screened_indices.npy",
        )
    )


def _archive_incomplete(run_dir: Path, output_root: Path) -> Path | None:
    """把不可原地续训的半成品移入可恢复归档，然后允许整项重跑。"""
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    relative = run_dir.relative_to(output_root)
    destination = output_root / "_batch/incomplete_archives" / relative / stamp
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))
    return destination


def train_one(
    data_dir: Path,
    config_file: Path,
    output_root: Path,
    device: str,
    artifact_policy: str,
) -> None:
    config = build_training_config(data_dir, config_file, output_root, device)
    run_dir = Path(config.output_dir)
    if _is_complete(run_dir):
        raise FileExistsError(f"训练已经完成: {run_dir}")
    result = TFPPTrainer(config=config).train()
    removed = _compact(run_dir) if artifact_policy == "compact" else []
    _atomic_json(
        run_dir / "run_complete.json",
        {
            "schema_version": 1,
            "experiment": "experiment2_screened",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(data_dir.resolve()),
            "device": device,
            "artifact_policy": artifact_policy,
            "removed_redundant_artifacts": removed,
            "best_epoch": int(result["best_epoch"]),
            "screen_k": int(config.screen_k),
            "gwas_k": int(config.gwas_k),
        },
    )


def _discover(data_root: Path, conditions: set[str], reps: set[int]) -> list[dict[str, Any]]:
    payload = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    jobs = []
    for record in payload["records"]:
        if record.get("status") != "generated_valid":
            raise ValueError(f"数据未通过生成校验: {record}")
        if conditions and record["condition"] not in conditions:
            continue
        if reps and int(record["repetition"]) not in reps:
            continue
        jobs.append(record)
    return sorted(
        jobs,
        key=lambda x: (x["n_snps"], x["n_train"], x["heritability"], x["repetition"]),
    )


def _launch(
    job: dict[str, Any], config_file: Path, output_root: Path, device: str, policy: str
) -> tuple[int, str]:
    failure_dir = output_root / "_batch/failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    label = f"{job['condition']}__rep_{int(job['repetition']):02d}"
    temporary_log = failure_dir / f".{label}.running.log"
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--run-one",
        str(Path(job["data_dir"]).resolve()),
        "--config",
        str(config_file.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--device",
        device,
        "--artifact-policy",
        policy,
    ]
    with temporary_log.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if process.returncode == 0:
        temporary_log.unlink(missing_ok=True)
        return 0, ""
    failure_log = failure_dir / f"{label}.log"
    temporary_log.replace(failure_log)
    return process.returncode, str(failure_log)


def _record(job: dict[str, Any], output_root: Path, status: str, **extra: Any) -> dict[str, Any]:
    run_dir = output_root / job["condition"] / f"rep_{int(job['repetition']):02d}"
    record = {
        "condition": job["condition"],
        "repetition": int(job["repetition"]),
        "simulation_seed": int(job["simulation_seed"]),
        "data_dir": str(Path(job["data_dir"]).resolve()),
        "run_dir": str(run_dir.resolve()),
        "status": status,
    }
    summary_path = run_dir / "training_history/training_summary.json"
    if status in {"completed", "skipped_complete"} and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["test_results"]["denormalized"]["overall"]
        record.update(
            best_epoch=int(summary["best_epoch"]),
            training_time_seconds=float(summary["training_time_seconds"]),
            **{
                key: value if math.isfinite(value := float(metrics[key])) else None
                for key in ("mse", "rmse", "mae", "pcc", "r2")
            },
        )
    record.update(extra)
    return record


def _write_batch(output_root: Path, records: list[dict[str, Any]], planned: int) -> None:
    ordered = sorted(records, key=lambda x: (x["condition"], x["repetition"]))
    path = output_root / "_batch/batch_manifest.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "experiment": "experiment2_screened",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "planned_jobs": planned,
            "status_counts": {
                status: sum(row["status"] == status for row in ordered)
                for status in sorted({row["status"] for row in ordered})
            },
            "records": ordered,
        },
    )
    if not ordered:
        return
    columns = sorted({key for row in ordered for key in row})
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ordered)
        temporary = Path(handle.name)
    temporary.replace(path.with_suffix(".csv"))


def run_batch(args: argparse.Namespace) -> int:
    jobs = _discover(args.data_root.resolve(), set(args.condition), set(args.rep))
    jobs = jobs if args.limit is None else jobs[: args.limit]
    print(f"[experiment2_screened] planned_jobs={len(jobs)}")
    if args.plan_only:
        for job in jobs:
            print(f"{job['condition']}/rep_{int(job['repetition']):02d}")
        return 0
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices不能为空")
    records, pending = [], []
    for job in jobs:
        run_dir = args.output_root / job["condition"] / f"rep_{int(job['repetition']):02d}"
        if _is_complete(run_dir):
            if not args.resume:
                raise FileExistsError(f"输出已完成且未启用resume: {run_dir}")
            records.append(_record(job, args.output_root, "skipped_complete"))
        else:
            archived = _archive_incomplete(run_dir, args.output_root)
            if archived is not None:
                print(f"[archive incomplete] {run_dir} -> {archived}")
            pending.append(job)
    _write_batch(args.output_root, records, len(jobs))
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _launch,
                job,
                args.config,
                args.output_root,
                devices[index % len(devices)],
                args.artifact_policy,
            ): job
            for index, job in enumerate(pending)
        }
        for future in as_completed(futures):
            job = futures[future]
            returncode, failure_log = future.result()
            if returncode == 0:
                row = _record(job, args.output_root, "completed")
            else:
                failures += 1
                row = _record(
                    job,
                    args.output_root,
                    "failed",
                    returncode=returncode,
                    failure_log=failure_log,
                )
            records.append(row)
            _write_batch(args.output_root, records, len(jobs))
            print(f"[{len(records)}/{len(jobs)}] {job['condition']}/rep_{int(job['repetition']):02d} {row['status']}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--devices", default="cuda:0", help="逗号分隔，如cuda:0,cuda:1")
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--rep", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="仅列出任务，不训练")
    parser.add_argument("--artifact-policy", choices=("compact", "full"), default="compact")
    parser.add_argument("--run-one", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers和limit必须大于0")
    if any(rep < 1 or rep > 10 for rep in args.rep):
        parser.error("--rep必须在1到10之间")
    return args


def main() -> None:
    args = parse_args()
    if args.run_one is not None:
        if args.device is None:
            raise ValueError("内部单任务模式必须指定--device")
        train_one(
            args.run_one,
            args.config.resolve(),
            args.output_root.resolve(),
            args.device,
            args.artifact_policy,
        )
        return
    raise SystemExit(run_batch(args))


if __name__ == "__main__":
    main()

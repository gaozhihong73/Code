"""E3 分阶段批量调度器：pilot 为3重复，formal 为10重复。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e3.runners.run_e3 import (
    DEFAULT_CONFIG,
    _load_config,
    _run_id,
    _slug,
    execute_run,
)
from experiments.e3.schemas.e3 import E3ExperimentConfig, E3RunManifest


PILOT_REPETITIONS = (1, 2, 3)
PILOT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/pilots/E3"
FORMAL_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/paper_experiments/runs/E3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _stage_config(
    base: E3ExperimentConfig, *, stage: str, output_root: Path
) -> E3ExperimentConfig:
    """派生不可混淆的阶段配置，不修改冻结的基准 JSON。"""
    note = (
        f"{base.notes}\n"
        f"batch_stage={stage}; derived_at_utc={_now()}; "
        f"derived_output_root={output_root.resolve()}"
    )
    return base.model_copy(update={"output_root": str(output_root.resolve()), "notes": note})


def _existing_success(run_dir: Path) -> bool:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return False
    try:
        return E3RunManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        ).status == "succeeded"
    except (OSError, ValueError):
        return False


def _write_batch_manifest(
    output_root: Path,
    *,
    stage: str,
    config: E3ExperimentConfig,
    records: list[dict[str, Any]],
    planned_jobs: int,
) -> None:
    directory = output_root / "_batch"
    path = directory / "batch_manifest.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "experiment": "E3",
            "stage": stage,
            "updated_at_utc": _now(),
            "planned_jobs": planned_jobs,
            "completed_records": len(records),
            "status_counts": {
                status: sum(row["status"] == status for row in records)
                for status in sorted({row["status"] for row in records})
            },
            "derived_config": config.model_dump(mode="json"),
            "records": records,
        },
    )
    if not records:
        return
    columns = sorted({key for record in records for key in record})
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=directory, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
        temporary = Path(handle.name)
    temporary.replace(path.with_suffix(".csv"))


def _jobs(
    config: E3ExperimentConfig, *, models: list[str], repetitions: list[int]
) -> list[tuple[int, str]]:
    unknown_models = sorted(set(models).difference(config.models))
    if unknown_models:
        raise ValueError(f"未知 E3 模型: {unknown_models}")
    invalid_reps = sorted(set(repetitions).difference(config.repetitions))
    if invalid_reps:
        raise ValueError(f"重复编号不在 E3 配置中: {invalid_reps}")
    return [
        (repetition, model)
        for repetition in repetitions
        for model in models
    ]


def run_batch(args: argparse.Namespace) -> int:
    base = _load_config(args.config.resolve())
    default_repetitions = (
        list(PILOT_REPETITIONS) if args.stage == "pilot" else list(base.repetitions)
    )
    models = args.model or list(base.models)
    repetitions = args.rep or default_repetitions
    jobs = _jobs(base, models=models, repetitions=repetitions)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (PILOT_OUTPUT_ROOT if args.stage == "pilot" else FORMAL_OUTPUT_ROOT)
    )
    config = _stage_config(base, stage=args.stage, output_root=output_root)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    for repetition, model in jobs:
        print(f"{args.stage}: {config.condition}/rep_{repetition:02d}/{model}")
    if not args.execute:
        print("未传 --execute：仅列出批量计划，不写入任何产物。")
        return 0
    if args.stage == "formal" and not args.confirm_formal:
        raise ValueError("正式 E3 必须显式传入 --confirm-formal")
    if args.stage == "formal" and base.protocol_status != "formal_frozen":
        raise ValueError("正式 E3 必须使用 protocol_status=formal_frozen 的配置")

    records: list[dict[str, Any]] = []
    failures = 0
    for position, (repetition, model) in enumerate(jobs, 1):
        run_dir = output_root / config.condition / f"rep_{repetition:02d}" / _slug(model)
        record = {
            "run_id": _run_id(config, repetition, model),
            "stage": args.stage,
            "condition": config.condition,
            "repetition": repetition,
            "model": model,
            "run_dir": str(run_dir),
        }
        if args.resume and _existing_success(run_dir):
            record["status"] = "skipped_complete"
        elif run_dir.exists():
            failures += 1
            record.update(
                status="existing_incomplete_or_failed",
                message="拒绝覆盖已有非成功 run；请使用新的 --output-root 保留重试轨迹",
            )
        else:
            data_dir = (
                REPOSITORY_ROOT
                / config.data_root
                / config.condition
                / f"rep_{repetition:02d}"
            )
            try:
                execute_run(
                    config,
                    repetition=repetition,
                    model=model,
                    data_dir=data_dir,
                    run_dir=run_dir,
                    formal=True,
                )
                record["status"] = "succeeded"
            except Exception as error:
                failures += 1
                record.update(
                    status="failed",
                    exception_type=type(error).__name__,
                    message=str(error),
                )
        records.append(record)
        _write_batch_manifest(
            output_root,
            stage=args.stage,
            config=config,
            records=records,
            planned_jobs=len(jobs),
        )
        print(f"[{position}/{len(jobs)}] {record['model']} {record['status']}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--rep", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="实际串行执行计划")
    parser.add_argument(
        "--confirm-formal", action="store_true", help="确认启动10重复正式阶段"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过已有成功 run；默认开启",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正数")
    return args


def main() -> None:
    raise SystemExit(run_batch(parse_args()))


if __name__ == "__main__":
    main()

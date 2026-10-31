"""E3 单模型单重复调度器；默认只列计划，不启动正式训练。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e3.baselines.data import E3DataBundle, file_sha256
from experiments.e3.baselines.registry import IMPLEMENTATION_REGISTRY, build_adapter
from experiments.e3.metrics import compute_prediction_metrics
from experiments.e3.schemas.e3 import E3ExperimentConfig, E3RunManifest

DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments/e3/configs/e3.json"


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


def _load_config(path: Path) -> E3ExperimentConfig:
    return E3ExperimentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    return name.lower().replace("+", "_plus_").replace("-", "_").replace(" ", "_")


def _run_id(config: E3ExperimentConfig, repetition: int, model: str) -> str:
    return f"E3_{config.condition}_rep{repetition:02d}_{_slug(model)}_seed{config.seed}"


def _git_commit() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def _software() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import xgboost

        result["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return result


def _initial_manifest(
    config: E3ExperimentConfig,
    repetition: int,
    model: str,
    run_dir: Path,
    started: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "run_id": _run_id(config, repetition, model),
        "experiment": "E3",
        "condition": config.condition,
        "repetition": repetition,
        "model": model,
        "seed": config.seed,
        "status": "running",
        "started_at_utc": started,
        "completed_at_utc": None,
        "parsed_config_path": str(run_dir / "parsed_config.json"),
        "git_commit": _git_commit(),
    }


def _save_predictions(
    run_dir: Path, data: E3DataBundle, prediction: np.ndarray, target: np.ndarray
) -> Path:
    path = run_dir / "test_predictions.npz"
    np.savez_compressed(
        path,
        sample_idx=data.split.test_idx.astype(np.int64),
        sample_id=data.sample_ids[data.split.test_idx],
        time_coordinates=data.time_coordinates,
        prediction=np.asarray(prediction, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
    )
    return path


def _dynags_reference(
    config: E3ExperimentConfig,
    repetition: int,
    data: E3DataBundle,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    source = (
        (REPOSITORY_ROOT / config.dynags_source_root).resolve()
        / config.condition
        / f"rep_{repetition:02d}"
    )
    required = {
        "prediction": source / "test_predictions.npz",
        "evaluation": source / "test_evaluation.json",
        "preprocessing": source / "preprocessing/preprocessing.json",
        "summary": source / "training_history/training_summary.json",
        "model_config": source / "models/model_config.json",
        "complete": source / "run_complete.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DynaGS 既有 E2 run 不完整:\n" + "\n".join(missing))
    evaluation = json.loads(required["evaluation"].read_text(encoding="utf-8"))
    preprocessing = json.loads(required["preprocessing"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    model_config = json.loads(required["model_config"].read_text(encoding="utf-8"))
    complete = json.loads(required["complete"].read_text(encoding="utf-8"))
    if evaluation.get("test_evaluation_count") != 1:
        raise ValueError("DynaGS 来源 run 的 test_evaluation_count 必须为1")
    if preprocessing.get("split_sha256") != file_sha256(data.split_path):
        raise ValueError("DynaGS 来源 run 与 E3 数据 split 哈希不一致")
    if preprocessing.get("original_snp_dim") != data.genotype.shape[1]:
        raise ValueError("DynaGS 来源 run 的原始 SNP 数不一致")
    with np.load(required["prediction"], allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_idx"], data.split.test_idx):
            raise ValueError("DynaGS 来源 run 的测试索引不一致")
        prediction = archive["predictions"][:, 0, :].copy()
        target = archive["targets"][:, 0, :].copy()
    expected_target = data.phenotype[data.split.test_idx]
    if not np.allclose(target, expected_target, rtol=0, atol=1e-6):
        raise ValueError("DynaGS 来源 run 的测试目标与当前数据不一致")
    training = {
        "fit_scope": "existing_e2_screened_run_reference",
        "selection_metric": "validation_loss_from_existing_e2_run",
        "best_parameters": {
            "source_run": str(source),
            "source_prediction_sha256": file_sha256(required["prediction"]),
            "ablation_mode": model_config.get("ablation_mode"),
            "screen_k": preprocessing.get("screen_k"),
            "gwas_k": preprocessing.get("gwas_k"),
            "screening_algorithm": preprocessing.get("screening_algorithm"),
        },
        "best_epoch": int(summary["best_epoch"]),
        "validation_history_path": str(source / "training_history/training_history_metrics.csv"),
        "effective_feature_count": int(preprocessing["snp_dim"]),
        "parameter_count": None,
        "best_epochs_by_time": None,
        "preprocessing_artifacts": {
            "source_preprocessing_path": str(required["preprocessing"]),
            "source_preprocessing_sha256": file_sha256(required["preprocessing"]),
        },
        "model_artifacts": {
            "source_checkpoint_path": str(source / "models/best_model.pt"),
            "source_checkpoint_sha256": file_sha256(source / "models/best_model.pt"),
        },
        "implementation": {
            **IMPLEMENTATION_REGISTRY["DynaGS"],
            "source_run_complete": complete,
            "adaptation": "read-only materialization of frozen E2 screened predictions",
        },
    }
    resources = {
        "train_seconds": float(summary["training_time_seconds"]),
        "predict_seconds": 0.0,
    }
    return prediction, target, training, resources


def execute_run(
    config: E3ExperimentConfig,
    *,
    repetition: int,
    model: str,
    data_dir: Path,
    run_dir: Path,
    formal: bool,
) -> Path:
    """执行一个 run；调用方负责 formal 授权门禁。"""
    if model not in config.models or repetition not in config.repetitions:
        raise ValueError("模型或重复不在冻结配置中")
    if run_dir.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = _now()
    started_clock = time.perf_counter()
    base = _initial_manifest(config, repetition, model, run_dir, started_at)
    _atomic_json(run_dir / "parsed_config.json", config.model_dump(mode="json"))
    _atomic_json(run_dir / "run_manifest.json", base)
    failure_context: dict[str, Any] = {}
    try:
        data = E3DataBundle.load(
            data_dir,
            formal=formal,
            expected_condition=config.condition,
            expected_repetition=repetition,
        )
        train_target, val_target, test_target = data.phenotype_splits()
        failure_context["data"] = data.schema_record(
            model_input_snp_count=0 if model == "Mean" else int(data.genotype.shape[1])
        )
        train_seconds = 0.0
        predict_seconds = 0.0
        adapter_resources: dict[str, float | None] = {}
        if model == "DynaGS":
            prediction, test_target, training, source_resource = _dynags_reference(
                config, repetition, data, run_dir
            )
            train_seconds = source_resource["train_seconds"]
            predict_seconds = source_resource["predict_seconds"]
            model_input_snp_count = int(training["effective_feature_count"])
        else:
            adapter = build_adapter(
                model, seed=config.seed, options=config.model_options.get(model, {})
            )
            if adapter.uses_genotype:
                x_train, x_val, x_test, imputation = data.dosage_splits()
                model_input_snp_count = int(x_train.shape[1])
                imputation_path = run_dir / "training_imputation_values.npy"
                np.save(
                    imputation_path,
                    np.asarray(imputation["fill_values"], dtype=np.int8),
                    allow_pickle=False,
                )
                preprocessing_artifacts = {
                    "genotype_imputation": {
                        "strategy": imputation["strategy"],
                        "training_missing_count": imputation["missing_count_train"],
                        "path": str(imputation_path),
                        "sha256": file_sha256(imputation_path),
                    }
                }
            else:
                x_train = x_val = x_test = None
                model_input_snp_count = 0
                adapter.test_size = len(test_target)
                preprocessing_artifacts = {"genotype": "not_used"}
            fit_started = time.perf_counter()
            fit = adapter.fit(x_train, train_target, x_val, val_target)
            train_seconds = time.perf_counter() - fit_started
            history_path = run_dir / "validation_history.csv"
            pd.DataFrame(fit.validation_history).to_csv(history_path, index=False)
            arrays = adapter.preprocessing_arrays()
            if arrays:
                feature_path = run_dir / "feature_preprocessing.npz"
                np.savez_compressed(feature_path, **arrays)
                preprocessing_artifacts["feature_preprocessing"] = {
                    "path": str(feature_path),
                    "sha256": file_sha256(feature_path),
                    "arrays": sorted(arrays),
                    "fit_scope": "train_only",
                }
            model_artifacts = adapter.save_artifacts(run_dir)
            for key, value in list(model_artifacts.items()):
                if key.endswith("_path"):
                    model_artifacts[key.removesuffix("_path") + "_sha256"] = file_sha256(
                        value
                    )
            predict_started = time.perf_counter()
            prediction = adapter.predict(x_test)
            predict_seconds = time.perf_counter() - predict_started
            adapter_resources = adapter.resource_metrics()
            training = {
                "fit_scope": "train_only_validation_selection_test_once",
                "selection_metric": config.selection_metric,
                "best_parameters": fit.best_parameters,
                "best_epoch": fit.best_epoch,
                "best_epochs_by_time": fit.best_epochs_by_time,
                "validation_history_path": str(history_path),
                "effective_feature_count": fit.effective_feature_count,
                "parameter_count": fit.parameter_count,
                "preprocessing_artifacts": preprocessing_artifacts,
                "model_artifacts": model_artifacts,
                "implementation": IMPLEMENTATION_REGISTRY[model],
            }

        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != test_target.shape or not np.isfinite(prediction).all():
            raise ValueError("测试预测必须与目标同形且全部有限")
        metrics = compute_prediction_metrics(
            prediction, test_target, train_target, data.time_coordinates
        )
        metrics_path = run_dir / "metrics.json"
        _atomic_json(metrics_path, metrics)
        prediction_path = _save_predictions(run_dir, data, prediction, test_target)
        np.savez_compressed(
            run_dir / "split_indices.npz",
            train_idx=data.split.train_idx,
            validation_idx=data.split.val_idx,
            test_idx=data.split.test_idx,
        )
        completed_at = _now()
        payload = {
            **base,
            "status": "succeeded",
            "completed_at_utc": completed_at,
            "data": data.schema_record(model_input_snp_count=model_input_snp_count),
            "training": training,
            "prediction": {
                "path": str(prediction_path),
                "sha256": file_sha256(prediction_path),
                "test_evaluation_count": 1,
                "metrics_path": str(metrics_path),
                "metrics_sha256": file_sha256(metrics_path),
                "prediction_source": (
                    "frozen_existing_prediction" if model == "DynaGS" else "new_model_prediction"
                ),
            },
            "resources": {
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "end_to_end_seconds": time.perf_counter() - started_clock,
                "peak_cpu_rss_mb": _peak_rss_mb(),
                "peak_gpu_memory_mb": adapter_resources.get("peak_gpu_memory_mb"),
                "hardware": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "cpu_count": os.cpu_count(),
                },
                "software": _software(),
                "measurement_notes": {
                    "peak_cpu_rss_mb": "process ru_maxrss upper bound",
                    "peak_gpu_memory_mb": "not yet instrumented; null is explicit",
                },
            },
        }
        validated = E3RunManifest.model_validate(payload)
        _atomic_json(run_dir / "run_manifest.json", validated.model_dump(mode="json"))
        return run_dir
    except Exception as error:
        traceback_path = run_dir / "failure_traceback.txt"
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        failed = {
            **base,
            **failure_context,
            "status": "failed",
            "completed_at_utc": _now(),
            "failure": {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback_path": str(traceback_path),
            },
        }
        validated = E3RunManifest.model_validate(failed)
        _atomic_json(run_dir / "run_manifest.json", validated.model_dump(mode="json"))
        raise


def _planned_jobs(config: E3ExperimentConfig, models: list[str], reps: list[int]):
    selected_models = models or list(config.models)
    selected_reps = reps or list(config.repetitions)
    for repetition in selected_reps:
        for model in selected_models:
            yield repetition, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--rep", action="append", type=int, default=[])
    parser.add_argument("--execute", action="store_true", help="实际执行；默认仅列出计划")
    parser.add_argument("--smoke", action="store_true", help="允许在非正式数据上执行")
    parser.add_argument(
        "--allow-formal",
        action="store_true",
        help="显式授权正式参考条件执行；开发阶段不要使用",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="仅 smoke 单重复覆盖")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config.resolve())
    unknown = sorted(set(args.model).difference(config.models))
    if unknown:
        raise ValueError(f"未知模型: {unknown}")
    jobs = list(_planned_jobs(config, args.model, args.rep))
    for repetition, model in jobs:
        status = IMPLEMENTATION_REGISTRY[model]["status"]
        print(f"{config.condition}/rep_{repetition:02d}/{model} [{status}]")
    if not args.execute:
        print("未传 --execute：仅列出计划，没有训练或写入 run 产物。")
        return
    if args.smoke == args.allow_formal:
        raise ValueError("执行时必须且只能选择 --smoke 或 --allow-formal")
    if args.smoke and (args.data_dir is None or len(jobs) != 1):
        raise ValueError("smoke 执行必须用 --data-dir 且只选择一个 model/rep")
    if args.allow_formal and config.protocol_status != "formal_frozen":
        raise ValueError("配置尚未标记 formal_frozen，拒绝正式 E3 训练")
    output_root = (args.output_root or Path(config.output_root)).resolve()
    for repetition, model in jobs:
        data_dir = (
            args.data_dir.resolve()
            if args.smoke
            else (REPOSITORY_ROOT / config.data_root / config.condition / f"rep_{repetition:02d}").resolve()
        )
        run_dir = output_root / config.condition / f"rep_{repetition:02d}" / _slug(model)
        execute_run(
            config,
            repetition=repetition,
            model=model,
            data_dir=data_dir,
            run_dir=run_dir,
            formal=args.allow_formal,
        )


if __name__ == "__main__":
    main()

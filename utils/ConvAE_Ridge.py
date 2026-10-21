"""ConvAE 表征独立线性探针：固定 train-only Ridge 与单次 held-out 评估。"""

import hashlib
import json
import math
import pickle
import platform
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .Calculate_Metrics import calculate_metrics_by_average, compute_metrics_vectorized
from .ConvAE_Export import LATENT_LAYOUT, ROLES, _git_provenance, load_exported_features
from .ConvAE_Preprocessing import convae_split_fingerprint
from .Inference import file_sha256, transform_phenotypes
from .Normalization import denormalize_phenotype
from .Split import load_split

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
METRICS = ("pcc", "r2", "mse", "rmse", "mae")


def baseline_config(seed: int, phenotype_sha256: str) -> dict:
    """固定模型/归一化/选择口径；不提供扩大搜索范围的参数。"""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    return {"schema_version": 1, "metrics_protocol": "time_adjusted_r2_nan_pcc_v2", "std_ddof": 0, "description": "ConvAE 表征的独立线性探针/下游基线",
            "seed": seed, "model": "multioutput_Ridge", "fit_intercept": True,
            "solver": "lsqr", "tol": 1e-6, "max_iter": 10000, "alphas": list(ALPHAS),
            "selection": "normalized_overall_val_mse", "tie_break": "larger_alpha",
            "tie_rtol": 1e-12, "tie_atol": 1e-12, "refit_on_train_val": False,
            "feature_scaler": "train_only_StandardScaler_per_feature",
            "phenotype_scaler": "train_only_StandardScaler_global_per_phenotype",
            "target_layout": "phenotype_major_time_minor_C_order", "threads": 1,
            "phenotype_file_sha256": phenotype_sha256,
            "pcc_policy": "undefined_NaN_finite_mean_with_valid_counts",
            "r2_policy": "TFPP_time_adjusted_global_no_clipping",
            "metrics_aggregation": "time_mean_per_phenotype_then_equal_phenotype_mean"}


def _digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_probe_export(export_dir: str | Path, split_file: str | Path | None = None):
    """验证全部特征哈希，再把每个 role 的 ID 顺序与原始 manifest 逐项比较。"""
    root = Path(export_dir)
    arrays, meta = load_exported_features(root)
    required = {"split_file", "split_sha256", "split_fingerprint", "fold_metadata", "phenotype_names",
                "time_points", "time_labels", "time_coordinates", "checkpoint_sha256", "model_config",
                "preprocessing_schema_version", "encoder_only", "reconstruction_mask_ratio", "seed",
                "num_chunks", "d_ext", "latent_dim", "sample_ids_sha256", "sample_id_source",
                "preprocessing_sha256", "genotype_encoding_version", "snp_file_sha256"}
    if meta.get("paper_eligible") is not True:
        raise ValueError("旧/未封存 ConvAE export 不具备正式结果资格；请使用当前严格 checkpoint 重新导出")
    if not required.issubset(meta) or meta["preprocessing_schema_version"] != 3:
        raise ValueError("导出 metadata 缺少 schema 3 来源字段")
    if (meta["latent_layout"] != LATENT_LAYOUT or meta["encoder_only"] is not True
            or meta["reconstruction_mask_ratio"] != 0
            or meta["latent_dim"] != meta["num_chunks"] * meta["d_ext"]):
        raise ValueError("导出表示/未遮蔽编码契约不匹配")
    for key in ("checkpoint_sha256", "preprocessing_sha256", "split_sha256", "snp_file_sha256"):
        digest = meta[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"导出来源哈希不合法: {key}")
    if (meta["model_config"].get("num_chunks") != meta["num_chunks"]
            or meta["model_config"].get("d_ext") != meta["d_ext"]
            or set(meta["fold_metadata"]) != {"fold_index", "n_folds", "kfold_val_ratio", "validation_seed"}):
        raise ValueError("导出模型/chunk/fold 配置不一致")
    names, times = meta["phenotype_names"], meta["time_labels"]
    if (not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names)
            or len(times) != meta["time_points"] or not times or len(set(times)) != len(times)
            or meta["model_config"].get("n_phenotypes") != len(names)
            or meta["model_config"].get("n_time_points") != len(times)):
        raise ValueError("导出表型或时间点配置不一致")
    manifest = Path(split_file) if split_file is not None else Path(meta["split_file"])
    if file_sha256(manifest) != meta["split_sha256"]:
        raise ValueError("导出 split 文件哈希不匹配")
    split = load_split(manifest)
    if convae_split_fingerprint(split) != meta["split_fingerprint"]:
        raise ValueError("导出 split/fold 指纹不匹配")
    if (meta["sample_ids_sha256"] != split.metadata["sample_ids_sha256"]
            or meta["sample_id_source"] != split.metadata["sample_id_source"]):
        raise ValueError("导出 sample_id 身份来源/顺序哈希不一致")
    for key, value in meta["fold_metadata"].items():
        if split.metadata.get(key) != value:
            raise ValueError("导出 fold 元数据不一致")
    all_ids = np.asarray(split.metadata["sample_ids"])
    for role in ROLES:
        _, ids = arrays[role]
        expected = all_ids[getattr(split, role + "_idx")]
        if not len(ids) or not np.array_equal(ids, expected):
            raise ValueError(f"{role} 特征 sample_id 缺失/额外/错序，与 manifest 不一致")
    return arrays, meta


def save_probe_targets(path: str | Path, targets: dict, *, phenotype_names: list[str], time_labels: list[str]):
    """显式标签交换格式；targets[role]=(原始 [N,P,T] 表型, manifest 顺序的 sample_ids)。"""
    payload = {"phenotype_names": np.asarray(phenotype_names), "time_labels": np.asarray(time_labels)}
    for role in ROLES:
        values, ids = targets[role]
        payload[role + "_phenotype"] = np.asarray(values, dtype=np.float64)
        payload[role + "_sample_ids"] = np.asarray(ids)
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)


def load_role_targets(path: str | Path, role: str, ids: np.ndarray, meta: dict) -> np.ndarray:
    """惰性读取单集合标签；test 仅在模型选择结束后调用。拒绝静默重排标签。"""
    with np.load(path, allow_pickle=False) as saved:
        if (saved["phenotype_names"].tolist() != meta["phenotype_names"]
                or saved["time_labels"].tolist() != meta["time_labels"]):
            raise ValueError("标签 phenotype/time 顺序与导出 metadata 不一致")
        actual = saved[role + "_sample_ids"]
        if actual.dtype != ids.dtype or not np.array_equal(actual, ids):
            raise ValueError(f"{role} phenotype sample_id 重复/缺失/额外/错序")
        values = saved[role + "_phenotype"].astype(np.float64)
    if values.shape != (len(ids), len(meta["phenotype_names"]), meta["time_points"]) or not np.isfinite(values).all():
        raise ValueError(f"{role} 表型 shape 不匹配或含缺失值；策略为 error")
    return values


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
              phenotype_names: list[str], *, seed: int = 73, timings: dict | None = None) -> tuple[dict, list[dict]]:
    """此 API 不接收 test。九个候选均只 fit train，保留选中对象，不合并 val 重拟合。"""
    preprocessing_start = time.perf_counter()
    fit_seconds, validation_seconds = 0.0, 0.0
    train_x, val_x = np.asarray(train_x, dtype=np.float64), np.asarray(val_x, dtype=np.float64)
    feature_scaler = StandardScaler().fit(train_x)
    scalers = {"norm_method": "global", "phenotype_names": phenotype_names,
               "pheno_meta": {"scalers": {name: StandardScaler().fit(train_y[:, p].reshape(-1, 1))
                   for p, name in enumerate(phenotype_names)}}}
    train = feature_scaler.transform(train_x)
    val = feature_scaler.transform(val_x)
    y = transform_phenotypes(train_y, scalers).numpy().reshape(len(train_y), -1).astype(np.float64)
    val_target = transform_phenotypes(val_y, scalers).numpy().reshape(len(val_y), -1).astype(np.float64)
    fit_seconds += time.perf_counter() - preprocessing_start
    candidates, records = [], []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-6,
                      max_iter=10000, random_state=seed)
        fit_start = time.perf_counter()
        model.fit(train, y)
        fit_seconds += time.perf_counter() - fit_start
        validation_start = time.perf_counter()
        prediction = model.predict(val)
        score = float(np.mean((prediction - val_target) ** 2))
        validation_seconds += time.perf_counter() - validation_start
        if not math.isfinite(score):
            raise ValueError("val MSE 非有限，禁止选择模型")
        records.append({"alpha": alpha, "normalized_val_mse": score, "fit_rows": len(train_y)})
        candidates.append(model)
    minimum = min(record["normalized_val_mse"] for record in records)
    tied = [i for i, record in enumerate(records) if np.isclose(record["normalized_val_mse"], minimum,
                                                            rtol=1e-12, atol=1e-12)]
    best = max(tied, key=lambda i: records[i]["alpha"])
    for i, record in enumerate(records):
        record["selected"] = i == best
    if timings is not None:
        timings.update(training=fit_seconds, validation=validation_seconds)
    return {"model": candidates[best], "feature_scaler": feature_scaler, "phenotype_scaler": scalers,
            "target_shape": train_y.shape[1:], "alpha": records[best]["alpha"],
            "test_evaluation_count": 0}, records


def predict_probe(package: dict, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """仅预测，绝不读取标签或 fit；返回归一化和原始量纲的 [N,P,T]。"""
    transformed = package["feature_scaler"].transform(np.asarray(features, dtype=np.float64))
    norm = package["model"].predict(transformed).reshape(len(features), *package["target_shape"])
    raw = denormalize_phenotype(norm, package["phenotype_scaler"]).numpy()
    return norm, raw


def _metric_block(pred: np.ndarray, target: np.ndarray) -> dict:
    return compute_metrics_vectorized(pred, target)


def _average_blocks(blocks: dict) -> dict:
    return calculate_metrics_by_average(blocks)


def metric_hierarchy(pred: np.ndarray, target: np.ndarray, names: list[str], time_labels: list[str]) -> dict:
    """沿用 TFPP 时间校正 R² 及先时间后表型均值；只更正未定义 PCC 的表示。"""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (len(names), len(time_labels)):
        raise ValueError("指标必须为匹配的 [N,P,T]")
    phenotypes = {name: _metric_block(pred[:, p], target[:, p]) for p, name in enumerate(names)}
    cells = {name: {label: _metric_block(pred[:, p, t:t + 1], target[:, p, t:t + 1])
                    for t, label in enumerate(time_labels)} for p, name in enumerate(names)}
    timepoints = {label: _average_blocks({name: cells[name][label] for name in names}) for label in time_labels}
    return {"overall": _average_blocks(phenotypes), "phenotypes": phenotypes,
            "timepoints": timepoints, "phenotype_timepoints": cells}


def evaluate_test_once(package: dict, features: np.ndarray, labels: np.ndarray, meta: dict):
    """选择完成后的唯一测试入口；重复调用明确失败。"""
    if package["test_evaluation_count"] != 0:
        raise ValueError("test 已评估，禁止重复模型选择/测试评估")
    package["test_evaluation_count"] = 1
    prediction_start = time.perf_counter()
    norm_pred, raw_pred = predict_probe(package, features)
    package["test_inference_seconds"] = time.perf_counter() - prediction_start
    norm_target = transform_phenotypes(labels, package["phenotype_scaler"]).numpy()
    args = (meta["phenotype_names"], meta["time_labels"])
    return norm_pred, raw_pred, {
        "normalized": metric_hierarchy(norm_pred, norm_target, *args),
        "denormalized": metric_hierarchy(raw_pred, labels, *args)}


def load_ridge_probe(directory: str | Path) -> dict:
    """加载本项目产生的可信本地 pickle，校验哈希；不重拟合任何对象。"""
    root = Path(directory)
    manifest = json.loads((root / "artifacts.json").read_text())
    required = {"config.json", "alpha_search.json", "feature_scaler.pkl", "phenotype_scaler.pkl",
                "ridge_model.pkl", "metrics.json", "test_predictions.npz", "test_predictions.csv",
                "summary.json", "environment.json", "run.log"}
    if set(manifest) != required:
        raise ValueError("Ridge 产物哈希清单不完整")
    for name, digest in manifest.items():
        if Path(name).name != name or file_sha256(root / name) != digest:
            raise ValueError(f"Ridge 产物哈希不匹配: {name}")
    with (root / "ridge_model.pkl").open("rb") as handle:
        package = pickle.load(handle)
    for name, key in (("feature_scaler.pkl", "feature_scaler"), ("phenotype_scaler.pkl", "phenotype_scaler")):
        with (root / name).open("rb") as handle:
            package[key] = pickle.load(handle)
    config = json.loads((root / "config.json").read_text())
    model = package["model"]
    if (config != baseline_config(config["seed"], config["phenotype_file_sha256"])
            or model.solver != "lsqr" or model.tol != 1e-6 or model.max_iter != 10000
            or model.fit_intercept is not True or model.alpha != package["alpha"]
            or package["alpha"] not in ALPHAS or package["test_evaluation_count"] != 1):
        raise ValueError("保存的 Ridge 配置/测试评估状态不一致")
    return package


def run_ridge_baseline(export_dir: str | Path, phenotype_file: str | Path, *,
                       output_root: str | Path = "outputs/baselines/convae_ridge",
                       split_file: str | Path | None = None, seed: int = 73,
                       before_test: Callable[[], None] | None = None) -> Path:
    """独立线性探针；test 标签只在九个 alpha 的 train/val 选择完全结束后读取。"""
    started = time.perf_counter()
    export_dir, phenotype_file = Path(export_dir).resolve(), Path(phenotype_file).resolve()
    export_hash = file_sha256(export_dir / "metadata.json")
    arrays, meta = load_probe_export(export_dir, split_file)
    label_hash = file_sha256(phenotype_file)  # 字节指纹仅溯源，不计算标签统计。
    config = baseline_config(seed, label_hash)
    config_hash = _digest(config)
    output = Path(output_root) / export_hash / config_hash
    output.mkdir(parents=True, exist_ok=False)
    try:
        log = []
        def record(message):
            log.append(message)
        record("ConvAE 表征的独立线性探针/下游基线；固定 Ridge，禁止据此声称优于监督筛选")
        train_y = load_role_targets(phenotype_file, "train", arrays["train"][1], meta)
        val_y = load_role_targets(phenotype_file, "val", arrays["val"][1], meta)
        with threadpool_limits(limits=1):
            fit_start = time.perf_counter()
            timings = {}
            package, search = fit_probe(arrays["train"][0].astype(np.float64), train_y,
                arrays["val"][0].astype(np.float64), val_y, meta["phenotype_names"], seed=seed, timings=timings)
            training_seconds = time.perf_counter() - fit_start
            record(f"fit/selection finished: alpha={package['alpha']}; train-only, no refit")
            # 唯一一次读取 test 标签，严格位于模型选择之后。
            if before_test is not None:
                before_test()
            test_y = load_role_targets(phenotype_file, "test", arrays["test"][1], meta)
            infer_start = time.perf_counter()
            norm_pred, raw_pred, metrics = evaluate_test_once(package, arrays["test"][0], test_y, meta)
            test_evaluation_seconds = time.perf_counter() - infer_start
        for candidate in search:
            record(f"alpha={candidate['alpha']}: normalized_val_mse={candidate['normalized_val_mse']:.12g}; selected={candidate['selected']}")
        record("test_evaluation_count=1")
        for name, key in (("feature_scaler.pkl", "feature_scaler"), ("phenotype_scaler.pkl", "phenotype_scaler")):
            with (output / name).open("wb") as handle:
                pickle.dump(package[key], handle)
        model_package = {key: value for key, value in package.items() if key not in ("feature_scaler", "phenotype_scaler", "test_inference_seconds")}
        with (output / "ridge_model.pkl").open("wb") as handle:
            pickle.dump(model_package, handle)
        np.savez_compressed(output / "test_predictions.npz", y_true=test_y, y_pred=raw_pred,
            y_pred_normalized=norm_pred, sample_ids=arrays["test"][1], split="test", phenotype_names=meta["phenotype_names"],
            time_labels=meta["time_labels"], split_fingerprint=meta["split_fingerprint"],
            fold=-1 if meta["fold_metadata"]["fold_index"] is None else meta["fold_metadata"]["fold_index"])
        rows = []
        for i, sample_id in enumerate(arrays["test"][1]):
            for p, phenotype in enumerate(meta["phenotype_names"]):
                for t, label in enumerate(meta["time_labels"]):
                    rows.append({"sample_id": int(sample_id), "split": "test", "split_fingerprint": meta["split_fingerprint"],
                        "fold": meta["fold_metadata"]["fold_index"], "phenotype": phenotype, "time_point": label,
                        "time_coordinate": None if meta["time_coordinates"] is None else meta["time_coordinates"][t],
                        "y_true": test_y[i, p, t], "y_pred": raw_pred[i, p, t]})
        pd.DataFrame(rows).to_csv(output / "test_predictions.csv", index=False)
        summary = {**_git_provenance(), "schema_version": 1, "alpha": package["alpha"],
            "export_metadata_sha256": export_hash, "export_directory": str(export_dir),
            "config_sha256": config_hash, "seed": seed, "phenotype_file": str(phenotype_file), "phenotype_file_sha256": label_hash,
            "split_sha256": meta["split_sha256"], "split_fingerprint": meta["split_fingerprint"],
            "fold_metadata": meta["fold_metadata"], "convae_checkpoint_sha256": meta["checkpoint_sha256"],
            "convae_use_pheno_help": meta["use_pheno_help"],
            "feature_shapes": {role: list(arrays[role][0].shape) for role in ROLES},
            "fit_scope": {"features": "train", "phenotype": "train", "Ridge": "train", "alpha": "val"},
            "test_evaluation_count": package["test_evaluation_count"], "training_seconds": training_seconds,
            "fit_seconds": timings["training"], "validation_seconds": timings["validation"],
            "test_prediction_and_metrics_seconds": test_evaluation_seconds,
            "inference_seconds": package["test_inference_seconds"], "prediction_replay_check": True,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_memory_scope": "Linux process lifetime peak RSS, includes imports and earlier work",
            "elapsed_seconds": time.perf_counter() - started}
        environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                       "sklearn": sklearn.__version__, "scipy": scipy.__version__, "threads": 1}
        for filename, data in (("config.json", config), ("alpha_search.json", search), ("metrics.json", metrics),
                               ("summary.json", summary), ("environment.json", environment)):
            (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
        if file_sha256(phenotype_file) != label_hash or file_sha256(export_dir / "metadata.json") != export_hash:
            raise ValueError("运行期间输入文件变化")
        files = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}
        (output / "artifacts.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
        restored = load_ridge_probe(output)
        with threadpool_limits(limits=1):
            _, restored_pred = predict_probe(restored, arrays["test"][0])
        np.testing.assert_allclose(raw_pred, restored_pred, rtol=1e-7, atol=1e-7)
        return output
    except Exception:
        shutil.rmtree(output)
        raise

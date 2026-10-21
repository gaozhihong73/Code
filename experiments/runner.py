"""Experiment runner v1：复用现有训练器，严格状态机和一次性 test reservation。"""

import json
import math
import platform
import resource
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from config import TFPPConfig
from trainer import TFPPTrainer
from utils.ConvAE_Ridge import (load_ridge_probe, metric_hierarchy, run_ridge_baseline, save_probe_targets)
from utils.Genotype import TrainingDosageImputer, dosage_to_model_codes, prepare_screening_dosage
from utils.Screening import (SCREENING_ALGORITHM_VERSION, array_fingerprint, dc_sis_screen,
                             gwas_screen, _multivariate_pheno_matrix)
from utils.Split import IndividualSplit, load_split, save_split
from .spec import UnitSpec, code_identity, digest, file_digest, random_indices
from .results import ResultManifest, RunStatus


def write_json(path, value):
    """同目录原子替换状态/清单，避免中断留下半个 JSON。"""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class RunnerTFPPConfig(TFPPConfig):
    run_directory: str

    @property
    def output_dir(self):
        return self.run_directory


class RecordedTFPPTrainer(TFPPTrainer):
    """保持现有训练目标/优化器/早停；仅记录时间并预约单次 test。"""

    def __init__(self, config, reserve_test):
        self.timings = {"training": 0.0, "validation": 0.0, "inference": 0.0}
        self.reserve_test = reserve_test
        try:
            super().__init__(config)
        except Exception:
            if hasattr(self, "logger"):
                for handler in list(self.logger.logger.handlers):
                    handler.close()
                    self.logger.logger.removeHandler(handler)
            raise

    def _train_epoch(self, *args, **kwargs):
        start = time.perf_counter()
        result = super()._train_epoch(*args, **kwargs)
        self.timings["training"] += time.perf_counter() - start
        return result

    def _validate_epoch(self, *args, **kwargs):
        start = time.perf_counter()
        result = super()._validate_epoch(*args, **kwargs)
        self.timings["validation"] += time.perf_counter() - start
        return result

    def _evaluate_test_set(self, *args, **kwargs):
        self.reserve_test()  # 在任何 test 前向之前持久化；崩溃后也不自动重测。
        start = time.perf_counter()
        result = super()._evaluate_test_set(*args, **kwargs)
        self.timings["inference"] += time.perf_counter() - start
        return result


def select_snps(unit: dict, train_dosage: np.ndarray, train_y: np.ndarray | None, traits: list[str]):
    """显式串联，不沿用 screen_snps 对小矩阵的并行并集启发式。"""
    route, width = unit["route"], train_dosage.shape[1]
    if route == "fixed_indices":
        indices = np.load(unit["fixed_indices"]["path"], allow_pickle=False)
        return indices, {"stages": [], "label_free": True, "source_sha256": unit["fixed_indices"]["sha256"]}
    if route == "random_k":
        return random_indices(width, unit["k"], unit["seed"]), {"stages": [], "label_free": True,
            "random_policy": "numpy_default_rng_choice_sorted_v1"}
    stages = []
    base = np.arange(width, dtype=np.int64)
    for algorithm in (["GWAS", "DC-SIS"] if route == "GWAS->DC-SIS" else [route]):
        k = unit["gwas_k"] if algorithm == "GWAS" else unit["k"]
        x = train_dosage[:, base]
        per_trait = {}
        if unit["screening_mode"] == "joint":
            if algorithm == "GWAS":
                chosen, _ = gwas_screen(x, train_y, k, multivariate=True, verbose=False)
            else:
                chosen, _ = dc_sis_screen(x, _multivariate_pheno_matrix(train_y), k, verbose=False)
            groups = [base[chosen]]
        else:
            groups = []
            for p, trait in enumerate(traits):
                if algorithm == "GWAS":
                    chosen, _ = gwas_screen(x, train_y[:, p], k, verbose=False)
                else:
                    chosen, _ = dc_sis_screen(x, train_y[:, p], k, verbose=False)
                groups.append(base[chosen])
                per_trait[trait] = {"count": len(chosen), "original_indices": base[chosen].tolist()}
            for trait, group in zip(traits, groups):
                other = set().union(*(set(g) for g in groups if g is not group))
                per_trait[trait]["exclusive_count"] = len(set(group) - other)
        combined = np.concatenate(groups)
        selected = np.unique(combined)
        stages.append({"algorithm": algorithm, "mode": unit["screening_mode"],
            "input_dim": len(base), "requested_k": k, "pre_dedup_count": len(combined),
            "post_dedup_count": len(selected), "per_trait": per_trait,
            "joint_contribution_policy": "joint association; no independent per-trait attribution" if not per_trait else None,
            "ordered_original_indices": selected.tolist(), "ordered_indices_sha256": digest(selected.tolist())})
        base = selected
    return base, {"stages": stages, "label_free": False, "algorithm_version": SCREENING_ALGORITHM_VERSION,
                  "GWAS_parameters": {"n_pcs": 5, "use_rint": True}}


def _verify_inputs(resolved):
    refs = [resolved["data"]["snp"], *resolved["data"]["phenotypes"].values(), resolved["outer_split"]]
    unit = resolved["unit"]
    if unit["fixed_indices"]:
        refs.append(unit["fixed_indices"])
    refs.extend(unit["exports"])
    for ref in refs:
        if file_digest(ref["path"]) != ref["sha256"]:
            raise ValueError(f"输入内容指纹不匹配: {ref['path']}")


def _load_data(resolved):
    data = resolved["data"]
    x = np.load(data["snp"]["path"], allow_pickle=False)
    if x.shape != (data["n_samples"], data["n_snps"]):
        raise ValueError("SNP 实际维度与 spec 不同")
    # 校验契约，但不使用全体基因型拟合任何统计。
    dosage_to_model_codes(x)
    blocks = []
    for name in resolved["unit"]["traits"]:
        frame = pd.read_csv(data["phenotypes"][name]["path"])
        if list(frame.columns) != data["time_labels"] or frame.shape != (data["n_samples"], len(data["time_labels"])):
            raise ValueError("phenotype 的个体/时间列与 spec 不一致")
        blocks.append(frame.to_numpy(dtype=np.float32))
    y = np.stack(blocks, axis=1)
    if not np.isfinite(y).all():
        raise ValueError("表型缺失策略为 error")
    return x, y


def _axes(resolved):
    unit = resolved["unit"]
    return {"split_mode": resolved["split_mode"], "split_family_sha256": resolved["split_family_sha256"],
            "seed": unit["seed"], "train_size": unit["train_size"], "trait_set": unit["traits"],
            "screening_route": unit["route"], "method_id": resolved["method_id"], "ablation": unit["ablation"],
            "fold": resolved["fold"], "expected_folds": resolved["expected_folds"],
            "statistical_unit_sha256": resolved["statistical_unit_sha256"], "development_smoke": resolved["development_smoke"]}


def run_unit(item: dict, *, resume: bool = False) -> Path:
    """resume=同一配置的成功结果校验/no-op；仅 test 前失败可新 attempt 重启。"""
    resolved, config_hash = item["resolved"], item["config_hash"]
    if digest(resolved) != config_hash or code_identity() != resolved["code"]:
        raise ValueError("配置哈希或代码 commit/dirty tree 已变化")
    if resolved["code"]["dirty"] and not resolved["development_smoke"]:
        raise ValueError("正式运行默认拒绝 dirty 工作区；开发 smoke 必须显式放宽并记录")
    UnitSpec.model_validate(resolved["unit"])
    root = Path(item["output_dir"])
    if root.exists():
        if not resume:
            raise FileExistsError(f"实验已存在，禁止覆盖: {root}")
        _verify_inputs(resolved)
        if json.loads((root / "resolved_config.json").read_text()) != resolved:
            raise ValueError("resume config/split/data/code 不完全相同")
        status = json.loads((root / "status.json").read_text())
        if status["config_hash"] != config_hash:
            raise ValueError("resume 状态配置哈希不一致")
        if status["state"] == "success":
            validate_result(root)
            return root
        if status["test_evaluation_count"]:
            raise ValueError("test 已预约/评估，禁止失败重试中读取或再次评估 test；请人工审查")
        attempt = status["attempt"] + 1
    else:
        if resume:
            raise ValueError("resume 找不到完全相同 canonical hash 的已有实验")
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "resolved_config.json", resolved)
        attempt = 1
    state = {"schema_version": 1, "state": "running", "config_hash": config_hash,
             "attempt": attempt, "test_evaluation_count": 0}
    write_json(root / "status.json", state)
    work = root / f"attempt_{attempt:03d}"
    work.mkdir()
    trainer = None
    start = time.perf_counter()
    def reserve():
        if state["test_evaluation_count"]:
            raise ValueError("test 已预约，禁止重复评估")
        state["test_evaluation_count"] = 1
        write_json(root / "status.json", state)
    try:
        _verify_inputs(resolved)
        unit = resolved["unit"]
        subset = resolved["subset"]
        split = IndividualSplit(*(np.asarray(subset[name], dtype=np.int64) for name in ("train_idx", "val_idx", "test_idx")),
                                metadata=subset["metadata"])
        split_path = work / "split.npz"
        if not split.metadata["excluded_train_idx"]:
            shutil.copyfile(resolved["outer_split"]["path"], split_path)
        else:
            save_split(split, split_path)
        x, y = _load_data(resolved)
        traits = unit["traits"]
        torch.manual_seed(unit["seed"])
        np.random.seed(unit["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(unit["seed"])
            torch.cuda.reset_peak_memory_stats()
        with threadpool_limits(limits=1):
            if unit["route"] == "ConvAE->Ridge":
                backend = _run_ridge(resolved, work, split, y, reserve)
            else:
                screen_start = time.perf_counter()
                train = x[split.train_idx]
                imputer = None
                label_free = unit["route"] in ("fixed_indices", "random_k")
                if not label_free:
                    strategy = resolved["tfpp_resolved"]["screen_missing_strategy"]
                    if strategy == "mode":
                        imputer = TrainingDosageImputer("mode").fit(train, train_idx=np.arange(len(train)))
                    train = prepare_screening_dosage(train, missing_strategy=strategy, imputer=imputer)
                indices, provenance = select_snps(unit, train, None if label_free else y[split.train_idx], traits)
                np.save(work / "screened_indices.npy", indices)
                # 读实际保存的索引维度，不把请求 k 当成实际维度。
                actual = np.load(work / "screened_indices.npy", allow_pickle=False)
                selection = {**provenance, "actual_input_dim": len(actual), "requested_k": unit["k"],
                    "ordered_indices_sha256": digest(actual.tolist()), "method_id": resolved["method_id"],
                    "training_indices_sha256": digest(split.train_idx.tolist()),
                    "training_genotype_sha256": array_fingerprint(x[split.train_idx]),
                    "training_phenotype_sha256": None if label_free else array_fingerprint(y[split.train_idx]),
                    "imputer": None if imputer is None else {**imputer.metadata_, "original_train_idx": split.train_idx.tolist()},
                    "cache_policy": "runner_no_cache_explicit_saved_indices"}
                write_json(work / "selection.json", selection)
                screening_seconds = time.perf_counter() - screen_start
                # 文件名按 trait 引用生成，不改动输入目录，也不改变 CSV 内容。
                pheno_dir = work / "phenotypes"
                pheno_dir.mkdir()
                for name in traits:
                    shutil.copyfile(resolved["data"]["phenotypes"][name]["path"], pheno_dir / f"{name}.csv")
                options = dict(resolved["tfpp_resolved"])
                options.update(exp_name=config_hash, run_directory=str(work / "backend"),
                    snp_file=resolved["data"]["snp"]["path"], pheno_dir=str(pheno_dir), split_file=str(split_path),
                    seed=unit["seed"], phenotype_names=traits, time_coordinates=resolved["data"]["time_coordinates"],
                    screen_k=0, gwas_k=0, use_kfold=False, fold_index=resolved["fold"],
                    n_folds=split.metadata.get("n_folds", 2), is_save_training_info=True,
                    kfold_val_ratio=split.metadata.get("kfold_val_ratio", 0.1),
                    preselected_snp_file=str(work / "screened_indices.npy"),
                    ablation_mode={"full": "all", "wide_only": "wide", "deep_only": "deep", "no_gate_add": "add"}[unit["ablation"]])
                init_start = time.perf_counter()
                actual_config = RunnerTFPPConfig(**options)
                write_json(work / "backend_config.json", actual_config.model_dump(mode="json"))
                trainer = RecordedTFPPTrainer(actual_config, reserve)
                initialization = time.perf_counter() - init_start
                trainer.logger.set_console_level("ERROR")
                trainer.train()
                preds = trainer.test_predictions
                norm, raw, true = preds["predictions_normalized"], preds["predictions"], y[split.test_idx]
                norm_true = preds["targets_normalized"]
                metrics = {"normalized": metric_hierarchy(norm, norm_true, traits, resolved["data"]["time_labels"]),
                           "denormalized": metric_hierarchy(raw, true, traits, resolved["data"]["time_labels"])}
                backend = {"raw": raw, "norm": norm, "true": true, "metrics": metrics,
                    "times": {**trainer.timings, "screening": screening_seconds, "preprocessing_model_init": initialization},
                    "parameters": sum(p.numel() for p in trainer.model.parameters()), "actual_input_dim": len(actual),
                    "history": str(trainer.training_history_save_dir.relative_to(root)),
                    "model_config": str((trainer.best_model_save_dir / "model_config.json").relative_to(root)),
                    "model_artifact": str((trainer.best_model_save_dir / "best_model.pt").relative_to(root)),
                    "feature_artifact": str((work / "screened_indices.npy").relative_to(root))}
        ids = np.asarray(split.metadata["sample_ids"])[split.test_idx]
        np.savez_compressed(work / "predictions.npz", sample_ids=ids, predictions=backend["raw"],
                            targets=backend["true"], predictions_normalized=backend["norm"])
        rows = [{"sample_id": int(ids[i]), "fold": resolved["fold"], "split": "test",
                 "split_sha256": resolved["outer_split"]["sha256"], "phenotype": trait,
                 "time_point": label, "true": float(backend["true"][i, p, t]), "prediction": float(backend["raw"][i, p, t])}
                for i in range(len(ids)) for p, trait in enumerate(traits)
                for t, label in enumerate(resolved["data"]["time_labels"])]
        pd.DataFrame(rows).to_csv(work / "predictions.csv", index=False)
        write_json(work / "metrics.json", backend["metrics"])
        write_json(work / "environment.json", {"platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": np.__version__, "torch_threads": torch.get_num_threads(), "blas_thread_limit": 1})
        manifest = {"schema_version": 1, "config_hash": config_hash, "code": resolved["code"], "axes": _axes(resolved),
            "subset_sha256": resolved["subset_sha256"], "sample_ids": {role: np.asarray(split.metadata["sample_ids"])[getattr(split, role + "_idx")].tolist() for role in ("train", "val", "test")},
            "actual_input_dim": backend["actual_input_dim"], "model_parameter_count": backend["parameters"],
            "times_seconds": {**backend["times"], "total": time.perf_counter() - start},
            "peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cpu_memory_scope": "Linux process lifetime high water RSS",
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "test_evaluation_count": state["test_evaluation_count"], "history": backend["history"],
            "model_config_artifact": backend["model_config"], "model_artifact": backend["model_artifact"],
            "feature_artifact": backend["feature_artifact"], "split_artifact": str(split_path.relative_to(root)),
            "predictions": str((work / "predictions.npz").relative_to(root)),
            "predictions_csv": str((work / "predictions.csv").relative_to(root)),
            "metrics": str((work / "metrics.json").relative_to(root))}
        write_json(root / "manifest.json", manifest)
        files = {str(p.relative_to(root)): file_digest(p) for p in work.rglob("*") if p.is_file()}
        files.update({"resolved_config.json": file_digest(root / "resolved_config.json"), "manifest.json": file_digest(root / "manifest.json")})
        write_json(root / "artifacts.json", files)
        state["state"] = "success"
        write_json(root / "status.json", state)
        validate_result(root)
        return root
    except Exception as error:
        state.update(state="failed", error_type=type(error).__name__, error=str(error))
        (work / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(root / "status.json", state)
        raise
    finally:
        if trainer is not None:
            for handler in list(trainer.logger.logger.handlers):
                handler.close()
                trainer.logger.logger.removeHandler(handler)


def _run_ridge(resolved, work, split, y, reserve):
    from utils.ConvAE_Ridge import load_probe_export
    from utils.ConvAE_Preprocessing import convae_split_fingerprint
    export_ref = next(ref for ref in resolved["unit"]["exports"]
        if json.loads(Path(ref["path"]).read_text())["split_fingerprint"] == convae_split_fingerprint(split))
    directory = Path(export_ref["path"]).parent
    arrays, meta = load_probe_export(directory)
    labels = work / "ridge_targets.npz"
    save_probe_targets(labels, {role: (y[getattr(split, role + "_idx")], arrays[role][1]) for role in ("train", "val", "test")},
        phenotype_names=resolved["unit"]["traits"], time_labels=resolved["data"]["time_labels"])
    # Ridge 的 test 预约必须精确发生在模型选择之后；通过显式回调接口传递。
    result = run_ridge_baseline(directory, labels, output_root=work / "ridge", seed=resolved["unit"]["seed"],
                                before_test=reserve)
    summary = json.loads((result / "summary.json").read_text())
    package = load_ridge_probe(result)
    with np.load(result / "test_predictions.npz", allow_pickle=False) as saved:
        raw, norm, true = saved["y_pred"], saved["y_pred_normalized"], saved["y_true"]
    shutil.copyfile(export_ref["path"], work / "export_metadata.json")
    write_json(work / "feature_hashes.json", meta["arrays"])
    return {"raw": raw, "norm": norm, "true": true, "metrics": json.loads((result / "metrics.json").read_text()),
        "times": {"training": summary["fit_seconds"], "validation": summary["validation_seconds"],
                  "inference": summary["inference_seconds"], "screening": 0.0},
        "parameters": int(package["model"].coef_.size + package["model"].intercept_.size),
        "actual_input_dim": meta["latent_dim"], "history": str((result / "alpha_search.json").relative_to(work.parent)),
        "model_config": str((result / "config.json").relative_to(work.parent)),
        "model_artifact": str((result / "ridge_model.pkl").relative_to(work.parent)),
        "feature_artifact": str((work / "feature_hashes.json").relative_to(work.parent))}


def validate_result(root: str | Path) -> tuple[dict, dict]:
    """仅接受 success、完整哈希和逐样本 ID 均校验通过的实验。"""
    root = Path(root)
    status = json.loads((root / "status.json").read_text())
    RunStatus.model_validate(status)
    if status.get("schema_version") != 1 or status.get("state") != "success" or status.get("test_evaluation_count") != 1:
        raise ValueError("仅可汇总 schema 1、success 且 test 评估一次的实验")
    resolved = json.loads((root / "resolved_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    ResultManifest.model_validate(manifest)
    UnitSpec.model_validate(resolved["unit"])
    hashes = json.loads((root / "artifacts.json").read_text())
    if (digest(resolved) != status["config_hash"] or manifest.get("config_hash") != status["config_hash"]
            or manifest.get("code") != resolved.get("code") or not resolved["code"].get("commit")
            or root.name != status["config_hash"] or manifest.get("schema_version") != 1):
        raise ValueError("结果 config hash/commit/schema 不一致")
    if (manifest["axes"] != _axes(resolved) or manifest["subset_sha256"] != resolved["subset_sha256"]
            or manifest["test_evaluation_count"] != 1):
        raise ValueError("结果实验轴/训练子集/评估状态不一致")
    required = {"manifest.json", "resolved_config.json", manifest["split_artifact"], manifest["predictions"],
                manifest["predictions_csv"], manifest["metrics"], manifest["feature_artifact"],
                manifest["model_config_artifact"], manifest["model_artifact"]}
    if not required.issubset(hashes):
        raise ValueError("缺少 split/预测/指标/索引或特征哈希")
    for name, expected in hashes.items():
        path = root / name
        if root.resolve() not in path.resolve().parents or file_digest(path) != expected:
            raise ValueError(f"结果产物哈希不匹配: {name}")
    if not any(name == manifest["history"] or name.startswith(manifest["history"] + "/") for name in hashes):
        raise ValueError("缺少训练 history 哈希")
    split = load_split(root / manifest["split_artifact"])
    for role in ("train", "val", "test"):
        if (getattr(split, role + "_idx").tolist() != resolved["subset"][role + "_idx"]
                or manifest["sample_ids"][role] != np.asarray(split.metadata["sample_ids"])[getattr(split, role + "_idx")].tolist()):
            raise ValueError("split/manifest/resolved sample_id 或行序不一致")
    feature_path = root / manifest["feature_artifact"]
    if resolved["unit"]["route"] != "ConvAE->Ridge":
        indices = np.load(feature_path, allow_pickle=False)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(np.unique(indices)) != len(indices)
                or len(indices) != manifest["actual_input_dim"] or (indices < 0).any()
                or (indices >= resolved["data"]["n_snps"]).any()):
            raise ValueError("实际 SNP 索引维度/范围不一致")
        selection = json.loads(feature_path.with_name("selection.json").read_text())
        if selection["ordered_indices_sha256"] != digest(indices.tolist()):
            raise ValueError("最终有序 SNP 索引哈希不一致")
        required.add(str(feature_path.with_name("selection.json").relative_to(root)))
    else:
        source = feature_path.with_name("export_metadata.json")
        export = json.loads(source.read_text())
        if (file_digest(source) not in [ref["sha256"] for ref in resolved["unit"]["exports"]]
                or export["arrays"] != json.loads(feature_path.read_text())
                or export["latent_dim"] != manifest["actual_input_dim"]):
            raise ValueError("特征来源/维度/数组哈希不一致")
        required.add(str(source.relative_to(root)))
    if not required.issubset(hashes):
        raise ValueError("缺少来源哈希")
    model_config = json.loads((root / manifest["model_config_artifact"]).read_text())
    if resolved["unit"]["route"] != "ConvAE->Ridge":
        if (model_config.get("snp_dim") != manifest["actual_input_dim"]
                or model_config.get("ablation_mode") != resolved["tfpp_resolved"]["ablation_mode"]
                or model_config.get("phenotype_names") != resolved["unit"]["traits"]
                or model_config.get("time_coordinates") != resolved["data"]["time_coordinates"]):
            raise ValueError("实际模型输入/消融/trait/time 配置不匹配")
    else:
        for key, value in resolved["ridge_protocol"].items():
            if key != "target_data_refs_sha256" and model_config.get(key) != value:
                raise ValueError("Ridge 固定协议不匹配")
    expected = np.asarray(split.metadata["sample_ids"])[split.test_idx]
    with np.load(root / manifest["predictions"], allow_pickle=False) as saved:
        if (not np.array_equal(saved["sample_ids"], expected)
                or saved["predictions"].shape != (len(expected), len(resolved["unit"]["traits"]), len(resolved["data"]["time_labels"]))
                or saved["targets"].shape != saved["predictions"].shape
                or saved["predictions_normalized"].shape != saved["predictions"].shape
                or not np.isfinite(saved["predictions_normalized"]).all()
                or not np.isfinite(saved["predictions"]).all() or not np.isfinite(saved["targets"]).all()):
            raise ValueError("预测 sample_id/shape/有限值不一致")
    metrics = json.loads((root / manifest["metrics"]).read_text())
    names, labels = resolved["unit"]["traits"], resolved["data"]["time_labels"]
    for scale in ("normalized", "denormalized"):
        hierarchy = metrics[scale]
        if (not {"overall", "phenotypes", "timepoints", "phenotype_timepoints"}.issubset(hierarchy)
                or set(hierarchy["phenotypes"]) != set(names)
                or set(hierarchy["timepoints"]) != set(labels)
                or set(hierarchy["phenotype_timepoints"]) != set(names)
                or any(set(hierarchy["phenotype_timepoints"][n]) != set(labels) for n in names)):
            raise ValueError("指标层级/表型/时间点不完整")
        blocks = [hierarchy["overall"], *hierarchy["phenotypes"].values(), *hierarchy["timepoints"].values(),
                  *[cell for n in names for cell in hierarchy["phenotype_timepoints"][n].values()]]
        for block in blocks:
            for key in ("pcc", "r2", "mse", "rmse", "mae"):
                value = block[key]
                if not isinstance(value, (int, float)) or (not math.isfinite(value) and not (key == "pcc" and math.isnan(value))):
                    raise ValueError("指标类型或非有限值不合法")
            count, total = block["pcc_valid_count"], block["pcc_total_count"]
            if (not isinstance(count, int) or not isinstance(total, int) or not 0 <= count <= total
                    or (count == 0) != math.isnan(block["pcc"])):
                raise ValueError("PCC NaN 与有效计数不一致")
    frame = pd.read_csv(root / manifest["predictions_csv"], dtype={"time_point": str})
    columns = {"sample_id", "fold", "split", "split_sha256", "phenotype", "time_point", "true", "prediction"}
    if (not columns.issubset(frame) or len(frame) != len(expected) * len(names) * len(labels)
            or frame["sample_id"].tolist() != np.repeat(expected, len(names) * len(labels)).tolist()
            or frame["phenotype"].tolist() != list(np.tile(np.repeat(names, len(labels)), len(expected)))
            or frame["time_point"].tolist() != labels * (len(expected) * len(names))
            or not (frame["split"] == "test").all()
            or not (frame["split_sha256"] == resolved["outer_split"]["sha256"]).all()):
        raise ValueError("逐样本预测 CSV 身份/trait/time 行序不完整")
    return manifest, metrics


def summarize(roots: list[str | Path]) -> dict:
    """fold/seed/train size/traits/route/ablation 均保留，拒绝不同实验单位混合。"""
    groups, rejected = {}, []
    for root in roots:
        try:
            manifest, metrics = validate_result(root)
            axes = dict(manifest["axes"])
            fold = axes.pop("fold")
            key = digest(axes)
            group = groups.setdefault(key, {"axes": axes, "folds": [], "details": []})
            if fold in group["folds"]:
                raise ValueError("重复 fold/普通实验单位，不允许重复计权")
            group["folds"].append(fold)
            group["details"].append({"fold": fold, "path": str(root), "metrics": metrics})
        except (ValueError, KeyError, OSError, TypeError) as error:
            rejected.append({"path": str(root), "error": str(error)})
    def statistics(values):
        finite = np.asarray(values, dtype=float)[np.isfinite(values)]
        return {"mean": float(finite.mean()) if len(finite) else float("nan"),
                "std": float(finite.std(ddof=0)) if len(finite) else float("nan"),
                "valid_count": int(len(finite)), "values": values}
    def reduce_metrics(values):
        if isinstance(values[0], dict):
            return {key: reduce_metrics([value[key] for value in values]) for key in values[0]}
        return statistics(values)
    for group in groups.values():
        axes = group["axes"]
        expected = list(range(1, axes["expected_folds"] + 1)) if axes["split_mode"] == "kfold" else [None]
        group["complete"] = set(group["folds"]) == set(expected)
        group["statistics"] = None
        group["metric_statistics"] = None
        if group["complete"]:
            group["statistics"] = {key: statistics([row["metrics"]["denormalized"]["overall"][key]
                for row in group["details"]]) for key in ("pcc", "r2", "mse", "rmse", "mae")}
            group["metric_statistics"] = reduce_metrics([row["metrics"] for row in group["details"]])
    return {"schema_version": 1, "groups": list(groups.values()), "rejected": rejected}

"""真实数据工程 pre-test：只训练/验证，永不创建或读取 test 标签视图。"""

import argparse
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from config import TFPPConfig
from dataloader import TFPPDataset
from experiments.spec import code_identity, digest, file_digest
from trainer import TFPPTrainer
from utils.RealDataManifest import verify_manifest
from utils.Screening import gwas_dc_sis_screen_mmap
from utils.Split import generate_split, save_split
from utils.model_loader import load_tfpp_model


PURPOSE = "engineering_pretest_smoke"
FROZEN_MANIFEST_SHA256 = "e82ebb8840ad16fff9b5f059ec9fcb86ec788d38f9ee617c9f5704a1195afc61"
TIME_COORDINATES = [10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101]
ROUTES = {
    "random_k": {"k": 1000, "gwas_k": 0},
    "gwas_dc_sis": {"k": 1000, "gwas_k": 5000},
}
GWAS_PCA_ALGORITHM_VERSION = "sklearn_randomized_centered_even_panel_v1"


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _plan(manifest_path, output_root, route, block_columns=16384,
          gwas_n_pcs=5, gwas_pca_panel_size=20000):
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.parent.name != FROZEN_MANIFEST_SHA256:
        raise ValueError("必须使用第12步冻结 manifest 目录，禁止替换数据版本")
    manifest = verify_manifest(manifest_path)
    if manifest["manifest_sha256"] != FROZEN_MANIFEST_SHA256:
        raise ValueError("真实数据 manifest SHA-256 与冻结值不一致")
    if manifest["identity"]["alignment_status"] != "confirmed_by_user_source_order":
        raise ValueError("当前 pretest 只接受已确认的稳定源行顺序")
    snp_path = Path(manifest["genotype"]["path"])
    rgbvi = next((p for p in manifest["phenotypes"] if p["trait"] == "RGBVI"), None)
    if rgbvi is None or rgbvi["time"]["coordinates"] != [float(x) for x in TIME_COORDINATES]:
        raise ValueError("RGBVI 或14个DAS时间坐标与冻结 manifest 不一致")
    spec = {"schema_version": 1, "purpose": PURPOSE, "paper_eligible": False,
            "manifest_sha256": manifest["manifest_sha256"], "snp_sha256": manifest["genotype"]["sha256"],
            "phenotype_sha256": rgbvi["sha256"], "phenotype": "RGBVI",
            "time_coordinates": TIME_COORDINATES, "seed": 73, "route": route,
            "k": ROUTES[route]["k"], "gwas_k": ROUTES[route]["gwas_k"],
            "missing_strategy": "mode", "epochs": 1, "batch_size": TFPPConfig.model_fields["batch_size"].default,
            "ablation_mode": "all", "screening_block_columns": block_columns,
            "gwas_population_structure": "pca", "gwas_n_pcs": gwas_n_pcs,
            "gwas_pca_panel_size": gwas_pca_panel_size,
            "gwas_pca_panel_seed": None,
            "gwas_pca_algorithm_version": GWAS_PCA_ALGORITHM_VERSION,
            "gwas_pca_svd_seed": 42, "code": code_identity(),
            "replay_rtol": 1e-5, "replay_atol": 1e-6}
    config_hash = digest(spec)
    output = Path(output_root).resolve() / config_hash
    split = generate_split(manifest["genotype"]["n_samples"], ratios=(.8, .1, .1), seed=73)
    free = shutil.disk_usage(Path(output_root).resolve().parent if not Path(output_root).exists() else output_root).free
    return manifest, snp_path, rgbvi, spec, config_hash, output, split, free


def dry_run(manifest_path, output_root, routes, block_columns=16384,
            gwas_n_pcs=5, gwas_pca_panel_size=20000):
    plans = []
    for route in routes:
        manifest, _, rgbvi, spec, config_hash, output, split, free = _plan(
            manifest_path, output_root, route, block_columns,
            gwas_n_pcs, gwas_pca_panel_size)
        plans.append({"route": route, "manifest_sha256": manifest["manifest_sha256"],
                      "split": {"train_n": len(split.train_idx), "val_n": len(split.val_idx),
                                "test_n": len(split.test_idx),
                                "sample_ids_sha256": split.metadata["sample_ids_sha256"]},
                      "config": spec, "config_hash": config_hash, "output_dir": str(output),
                      "disk_free_bytes": free, "minimum_required_bytes": 2 * 1024 ** 3,
                      "will_read_test_phenotype": False, "will_evaluate_test": False})
        if free < 2 * 1024 ** 3:
            raise OSError("可用磁盘不足2 GiB，停止真实 pretest；不删除任何用户文件")
    return {"purpose": PURPOSE, "paper_eligible": False, "plans": plans, "writes_performed": False}


def _predictions(model, loader, device):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x.to(device)).cpu() for x, _ in loader]).numpy()


def run_route(manifest_path, output_root, route, block_columns=16384,
              gwas_n_pcs=5, gwas_pca_panel_size=20000):
    manifest, snp_path, rgbvi, spec, config_hash, output, split, free = _plan(
        manifest_path, output_root, route, block_columns,
        gwas_n_pcs, gwas_pca_panel_size)
    if output.exists():
        raise FileExistsError(f"相同 pretest 配置禁止覆盖: {output}")
    output.mkdir(parents=True)
    status = {"state": "running", "purpose": PURPOSE, "paper_eligible": False,
              "test_accessed": False, "config_hash": config_hash}
    _write_json(output / "status.json", status)
    try:
        split_path = output / "input_split.npz"
        save_split(split, split_path)
        selection_path = output / "screened_indices.npy"
        selection_started = time.perf_counter()
        if route == "random_k":
            rng = np.random.default_rng(73)
            selected = rng.choice(manifest["genotype"]["n_snps"], size=spec["k"], replace=False).astype(np.int64)
            screening = {"algorithm": "train_label_free_numpy_choice_v1", "selected_count": len(selected),
                         "phenotype_accessed": False, "mmap": True, "full_matrix_copy_created": False}
        else:
            probe = TFPPDataset(str(snp_path), str(Path(rgbvi["path"]).parent), ["RGBVI"],
                                snp_indices=np.array([0]),
                                phenotype_row_indices=np.concatenate([split.train_idx, split.val_idx]))
            selected, screening = gwas_dc_sis_screen_mmap(
                snp_path, split.train_idx, probe.phenotype[split.train_idx],
                gwas_k=spec["gwas_k"], screen_k=spec["k"], block_columns=block_columns,
                n_pcs=spec["gwas_n_pcs"],
                pca_marker_count=spec["gwas_pca_panel_size"], multivariate=True)
            panel_indices = np.asarray(screening.pop("pca_panel_indices"), dtype=np.int64)
            panel_path = output / "gwas_pca_panel_indices.npy"
            np.save(panel_path, panel_indices)
            screening["pca_panel_indices_file"] = panel_path.name
            screening["pca_panel_indices_file_sha256"] = file_digest(panel_path)
        selection_seconds = time.perf_counter() - selection_started
        np.save(selection_path, selected)

        panel_file = screening.get("pca_panel_indices_file")
        panel_file_hash = screening.get("pca_panel_indices_file_sha256")
        config = TFPPConfig(
            exp_name=config_hash, output_root=str(Path(output_root).resolve()),
            run_stage="engineering_pretest", split_file=str(split_path),
            preselected_snp_file=str(selection_path), snp_file=str(snp_path),
            pheno_dir=str(Path(rgbvi["path"]).parent), phenotype_names=["RGBVI"],
            time_coordinates=TIME_COORDINATES, seed=73, epochs=1,
            screen_k=0, gwas_k=0, screen_missing_strategy="mode",
            wide_missing_strategy="mode", ablation_mode="all", device="cuda",
            gwas_population_structure=spec["gwas_population_structure"],
            gwas_n_pcs=spec["gwas_n_pcs"],
            gwas_pca_panel_size=spec["gwas_pca_panel_size"],
            gwas_pca_panel_seed=spec["gwas_pca_panel_seed"],
            gwas_pca_algorithm_version=spec["gwas_pca_algorithm_version"],
            gwas_pca_panel_indices_file=panel_file,
            gwas_pca_panel_indices_sha256=panel_file_hash,
        )
        if Path(config.output_dir).resolve() != output:
            raise ValueError("config_hash 输出路径不一致")
        torch.cuda.reset_peak_memory_stats()
        init_started = time.perf_counter()
        trainer = TFPPTrainer(config)
        initialization_seconds = time.perf_counter() - init_started
        test_rows = set(split.test_idx.tolist())
        parsed = set(np.concatenate(trainer.dataset.parsed_phenotype_rows).tolist())
        if parsed & test_rows or parsed != set(np.concatenate([split.train_idx, split.val_idx]).tolist()):
            raise AssertionError("pretest phenotype 行访问越界")
        training = trainer.train()
        original_predictions = _predictions(trainer.model, trainer.val_loader, trainer.device)
        reload_started = time.perf_counter()
        loaded, info = load_tfpp_model(output, device="cuda", verbose=False)
        replay = _predictions(loaded, trainer.val_loader, "cuda")
        replay_seconds = time.perf_counter() - reload_started
        replay_ok = bool(np.allclose(original_predictions, replay, rtol=spec["replay_rtol"],
                                     atol=spec["replay_atol"]))
        if not replay_ok or info["paper_eligible"]:
            raise AssertionError("pretest 严格重载复现失败或错误标记为论文资格")
        if trainer.test_loader is not None or (output / "test_evaluation.json").exists():
            raise AssertionError("pretest 禁止创建 test loader/预约")
        if any("test" in path.name.lower() and path.name != "status.json" for path in output.rglob("*")):
            raise AssertionError("pretest 输出不得包含 test 预测/指标")

        artifact_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        record = {**spec, "config_hash": config_hash, "split_sha256": file_digest(split_path),
                  "train_indices_sha256": trainer.train_indices_sha256,
                  "val_indices_sha256": hashlib.sha256(split.val_idx.tobytes()).hexdigest(),
                  "test_indices_sha256": hashlib.sha256(split.test_idx.tobytes()).hexdigest(),
                  "train_n": len(split.train_idx), "val_n": len(split.val_idx), "test_n": len(split.test_idx),
                  "test_access": {"loader_created": False, "phenotype_rows_parsed": sorted(parsed),
                                  "test_rows_parsed": [], "metrics_computed": False,
                                  "predictions_generated": False, "reservation_created": False},
                  "screening": {**screening, "selection_seconds": selection_seconds,
                                "selected_indices_sha256": file_digest(selection_path),
                                "screening_cache_bytes": 0},
                  "best_val_loss": float(training["best_val_loss"]),
                  "val_loss_finite": bool(np.isfinite(training["best_val_loss"])),
                  "strict_reload_replay": {"passed": replay_ok, "rtol": spec["replay_rtol"],
                                           "atol": spec["replay_atol"], "seconds": replay_seconds},
                  "timings_seconds": {"selection": selection_seconds,
                                      "preprocessing_model_init": initialization_seconds,
                                      "training_validation": training["training_time"]},
                  "resources": {"peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "checkpoint_bytes": (output / "models/best_model.pt").stat().st_size,
                                "output_bytes_before_manifest": artifact_bytes,
                                "disk_free_bytes_after": shutil.disk_usage(output).free},
                  "source_manifest": str(Path(manifest_path).resolve()),
                  "original_snp_indices": True}
        _write_json(output / "engineering_manifest.json", record)
        status.update(state="success", test_accessed=False, manifest="engineering_manifest.json")
        _write_json(output / "status.json", status)
        return record
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", test_accessed=False)
        _write_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "run"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default="outputs/engineering_pretest_smoke")
    parser.add_argument("--route", choices=tuple(ROUTES), action="append")
    parser.add_argument("--block_columns", type=int, default=16384)
    parser.add_argument("--gwas_n_pcs", type=int, default=5)
    parser.add_argument("--gwas_pca_panel_size", type=int, default=20000)
    args = parser.parse_args()
    routes = args.route or list(ROUTES)
    if args.command == "dry-run":
        print(json.dumps(dry_run(
            args.manifest, args.output_root, routes, args.block_columns,
            args.gwas_n_pcs, args.gwas_pca_panel_size),
                         ensure_ascii=False, indent=2))
    else:
        for route in routes:
            result = run_route(
                args.manifest, args.output_root, route, args.block_columns,
                args.gwas_n_pcs, args.gwas_pca_panel_size)
            print(json.dumps({"route": route, "config_hash": result["config_hash"],
                              "status": "success"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

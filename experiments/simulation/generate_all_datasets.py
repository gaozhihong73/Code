"""生成并验证正式模拟实验的完整参数网格。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.simulation.generate_dataset import (
    TRAIN_TO_TOTAL,
    _heritability_label,
    generate_dataset,
)
from validate_simulation import validate_tfpp_loading


N_SNPS_GRID = (5000, 10000)
N_TRAIN_GRID = (100, 200, 400, 800, 1600)
HERITABILITY_GRID = (0.1, 0.4, 0.8)
REPETITIONS = tuple(range(1, 11))
SPLIT_SEED = 73
MAX_REALIZED_H2_ERROR = 0.01
MAX_MEAN_REALIZED_H2_ERROR = 0.002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition(n_snps: int, n_train: int, heritability: float) -> str:
    return f"P{n_snps}_NTR{n_train}_H{_heritability_label(heritability)}"


def _validate_dataset(
    data_dir: Path,
    *,
    n_snps: int,
    n_train: int,
    heritability: float,
    repetition: int,
    simulation_seed: int,
) -> dict:
    request_path = data_dir / "generation_request.json"
    manifest_path = data_dir / "data_manifest.json"
    config_path = data_dir / "simulation_config.json"
    for path in (request_path, manifest_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"数据集不完整，缺少: {path}")

    request = json.loads(request_path.read_text(encoding="utf-8"))
    expected_request = {
        "condition": _condition(n_snps, n_train, heritability),
        "repetition": repetition,
        "n_train_requested": n_train,
        "n_total_generated": TRAIN_TO_TOTAL[n_train],
        "n_snps": n_snps,
        "n_qtl": 20,
        "qtl_pool_size": min(5000, n_snps),
        "heritability": heritability,
        "n_epistasis": 0,
        "seed": simulation_seed,
        "snp_format": "npy",
    }
    for key, expected in expected_request.items():
        if request.get(key) != expected:
            raise ValueError(
                f"{data_dir} 的 generation_request.{key}={request.get(key)!r}，"
                f"预期为 {expected!r}"
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_shape = [TRAIN_TO_TOTAL[n_train], n_snps]
    if manifest.get("genotype_shape") != expected_shape:
        raise ValueError(f"{data_dir} 基因型形状错误")
    split = manifest.get("split", {})
    expected_holdout = TRAIN_TO_TOTAL[n_train] - n_train
    expected_val = expected_holdout // 2
    if (
        split.get("seed") != SPLIT_SEED
        or split.get("train_n") != n_train
        or split.get("val_n") != expected_val
        or split.get("test_n") != expected_holdout - expected_val
    ):
        raise ValueError(f"{data_dir} 的固定划分不符合正式协议")
    if manifest.get("truth_is_model_input") is not False:
        raise ValueError(f"{data_dir} 错误地把模拟真值列为模型输入")

    for relative, expected_hash in manifest.get("output_files", {}).items():
        path = data_dir / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"数据文件缺失或哈希不一致: {path}")

    effects = pd.read_csv(data_dir / "truth" / "causal_effects.csv")
    if len(effects) != 20 or effects["snp_index"].nunique() != 20:
        raise ValueError(f"{data_dir} 必须包含20个互异真实QTL")
    if effects["snp_index"].min() < 1 or effects["snp_index"].max() > 5000:
        raise ValueError(f"{data_dir} 的真实QTL没有限制在前5000个位点")

    loading = validate_tfpp_loading(data_dir)
    required_checks = (
        "finite",
        "split_disjoint",
        "split_complete",
        "scaler_fit_train_only",
    )
    if not all(loading[name] for name in required_checks):
        raise ValueError(f"{data_dir} 未通过TFPP加载验证")

    simulation = json.loads(config_path.read_text(encoding="utf-8"))
    realized = simulation.get("realized_heritability", [])
    realized_errors = [abs(value - heritability) for value in realized]
    if (
        len(realized) != 20
        or max(realized_errors) > MAX_REALIZED_H2_ERROR
        or sum(realized_errors) / len(realized_errors)
        > MAX_MEAN_REALIZED_H2_ERROR
    ):
        raise ValueError(f"{data_dir} 的实现遗传力与目标不一致")
    provenance = manifest.get("provenance_bundle", {})
    bundle_manifest = Path(provenance.get("path", "")) / "bundle_manifest.json"
    if (
        not bundle_manifest.is_file()
        or _sha256(bundle_manifest) != provenance.get("manifest_sha256")
    ):
        raise ValueError(f"{data_dir} 的生成器溯源包无效")
    return {
        "data_manifest_sha256": _sha256(manifest_path),
        "genotype_sha256": manifest["output_files"]["sim_snp.npy"],
        "phenotype_sha256": manifest["output_files"]["sim_pheno.csv"],
        "split_sha256": manifest["output_files"]["splits/split_seed73.npz"],
        "provenance_bundle_id": provenance["bundle_id"],
        "realized_h2_mean": sum(realized) / len(realized),
        "realized_h2_min": min(realized),
        "realized_h2_max": max(realized),
        "realized_h2_max_abs_error": max(realized_errors),
    }


def _write_records(output_root: Path, records: list[dict]) -> None:
    json_path = output_root / "dataset_manifest.json"
    csv_path = output_root / "dataset_manifest.csv"
    payload = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "n_snps": list(N_SNPS_GRID),
            "n_train": list(N_TRAIN_GRID),
            "heritability": list(HERITABILITY_GRID),
            "repetitions": list(REPETITIONS),
            "simulation_seeds": {str(rep): 72 + rep for rep in REPETITIONS},
            "split_seed": SPLIT_SEED,
            "n_qtl": 20,
            "n_epistasis": 0,
            "time_coordinates": list(range(1, 21)),
            "realized_h2_tolerance": {
                "max_timepoint_abs_error": MAX_REALIZED_H2_ERROR,
                "max_mean_abs_error": MAX_MEAN_REALIZED_H2_ERROR,
            },
            "batch_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "records": records,
    }
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)

    fieldnames = [
        "condition",
        "repetition",
        "simulation_seed",
        "data_dir",
        "n_snps",
        "n_train",
        "n_total",
        "heritability",
        "n_qtl",
        "status",
        "data_manifest_sha256",
        "genotype_sha256",
        "phenotype_sha256",
        "split_sha256",
        "provenance_bundle_id",
        "realized_h2_mean",
        "realized_h2_min",
        "realized_h2_max",
        "realized_h2_max_abs_error",
    ]
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary_csv.replace(csv_path)


def generate_all(output_root: str | Path, *, resume: bool = True) -> list[dict]:
    """顺序生成300份正式数据；已有数据仅在完整复验后跳过。"""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total = (
        len(N_SNPS_GRID)
        * len(N_TRAIN_GRID)
        * len(HERITABILITY_GRID)
        * len(REPETITIONS)
    )
    task_index = 0
    for n_snps in N_SNPS_GRID:
        for n_train in N_TRAIN_GRID:
            for heritability in HERITABILITY_GRID:
                condition = _condition(n_snps, n_train, heritability)
                for repetition in REPETITIONS:
                    task_index += 1
                    simulation_seed = 72 + repetition
                    data_dir = output_root / condition / f"rep_{repetition:02d}"
                    reused = data_dir.exists()
                    if reused and not resume:
                        raise FileExistsError(f"正式数据已存在: {data_dir}")
                    print(
                        f"[{task_index:03d}/{total}] {condition}/rep_{repetition:02d} "
                        f"seed={simulation_seed} {'validate' if reused else 'generate'}",
                        flush=True,
                    )
                    if not reused:
                        generate_dataset(
                            output_root=output_root,
                            n_train=n_train,
                            n_snps=n_snps,
                            heritability=heritability,
                            repetition=repetition,
                            seed=simulation_seed,
                            snp_format="npy",
                            n_epistasis=0,
                        )
                    validation = _validate_dataset(
                        data_dir,
                        n_snps=n_snps,
                        n_train=n_train,
                        heritability=heritability,
                        repetition=repetition,
                        simulation_seed=simulation_seed,
                    )
                    records.append(
                        {
                            "condition": condition,
                            "repetition": repetition,
                            "simulation_seed": simulation_seed,
                            "data_dir": str(data_dir),
                            "n_snps": n_snps,
                            "n_train": n_train,
                            "n_total": TRAIN_TO_TOTAL[n_train],
                            "heritability": heritability,
                            "n_qtl": 20,
                            "status": "reused_valid" if reused else "generated_valid",
                            **validation,
                        }
                    )
                    _write_records(output_root, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="dataset/experiments")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="若任何目标数据目录已存在则报错；默认复验后跳过",
    )
    args = parser.parse_args()
    records = generate_all(args.output_root, resume=not args.no_resume)
    print(
        json.dumps(
            {
                "output_root": str(Path(args.output_root).resolve()),
                "dataset_count": len(records),
                "status": "complete",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

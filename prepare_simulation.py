"""将 R 生成的完整模拟数据转换为原 TFPP 输入格式。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.Genotype import validate_dosage
from utils.Split import load_or_create_split, validate_individual_split


FORMAT_VERSION = "dynags_tfpp_simulation_v1"
REQUIRED_INPUTS = ("genotype.csv", "phenotype.csv", "sample_ids.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["SampleID"]:
        raise ValueError(f"{path.name} 必须且只能包含 SampleID 列")
    ids = frame["SampleID"].tolist()
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("SampleID 必须非空且唯一")
    return ids


def _read_matrix(path: Path, expected_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    if not len(frame.columns) or frame.columns[0] != "SampleID":
        raise ValueError(f"{path.name} 第一列必须为 SampleID")
    ids = frame["SampleID"].astype(str).tolist()
    if len(ids) != len(expected_ids):
        raise ValueError(f"{path.name} 与 sample_ids.csv 行数不一致")
    if ids != expected_ids:
        raise ValueError(f"{path.name} 的 SampleID 顺序与 sample_ids.csv 不一致")
    values = frame.drop(columns="SampleID")
    if values.shape[1] == 0:
        raise ValueError(f"{path.name} 不含数据列")
    try:
        array = values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path.name} 包含非数值数据") from error
    return values, array


def prepare_simulation(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 73,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """校验完整模拟数据并以原 TFPP 格式原子发布，不覆盖已有目录。"""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    missing = [name for name in REQUIRED_INPUTS if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"模拟输入缺少文件: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output_dir}")

    sample_ids = _read_ids(input_dir / "sample_ids.csv")
    genotype_frame, genotype_float = _read_matrix(
        input_dir / "genotype.csv", sample_ids
    )
    phenotype_frame, phenotype = _read_matrix(
        input_dir / "phenotype.csv", sample_ids
    )
    if genotype_float.shape[0] != phenotype.shape[0]:
        raise ValueError("基因型、表型和 SampleID 行数不一致")
    if not np.isfinite(genotype_float).all():
        raise ValueError("基因型必须为有限的 0/1/2 剂量")
    validate_dosage(genotype_float)
    genotype = genotype_float.astype(np.int8)
    if not np.isfinite(phenotype).all():
        raise ValueError("表型必须全部为有限值")
    phenotype = phenotype.astype(np.float32)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-sim-", dir=output_dir.parent))
    try:
        pheno_dir = temporary / "cleaned_pheno"
        split_dir = temporary / "splits"
        truth_dir = temporary / "truth"
        pheno_dir.mkdir()
        split_dir.mkdir()
        truth_dir.mkdir()

        genotype_path = temporary / "genotype.npy"
        phenotype_path = pheno_dir / "sim_pheno.csv"
        sample_ids_path = temporary / "sample_ids.csv"
        split_path = split_dir / f"split_seed{seed}.npz"
        np.save(genotype_path, genotype, allow_pickle=False)
        pd.DataFrame(phenotype, columns=phenotype_frame.columns).to_csv(
            phenotype_path, index=False
        )
        pd.DataFrame({"SampleID": sample_ids}).to_csv(sample_ids_path, index=False)
        split = load_or_create_split(
            split_path,
            sample_ids=sample_ids,
            ratios=ratios,
            seed=seed,
        )
        validate_individual_split(split, sample_ids=sample_ids)

        truth_files = []
        for name in ("true_gv.csv", "causal_effects.csv", "epistatic_effects.csv"):
            source = input_dir / name
            if source.is_file():
                destination = truth_dir / name
                shutil.copyfile(source, destination)
                truth_files.append(destination)

        config_path = input_dir / "simulation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        time_coordinates = config.get("time_coordinates")
        if time_coordinates is None:
            time_coordinates = phenotype_frame.columns.tolist()
        if len(time_coordinates) != phenotype.shape[1]:
            raise ValueError("simulation_config 时间点数量与 phenotype.csv 不一致")

        inputs = [input_dir / name for name in REQUIRED_INPUTS]
        if config_path.is_file():
            inputs.append(config_path)
        manifest = {
            "data_format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "input_files": {str(path): _sha256(path) for path in inputs},
            "output_files": {
                "genotype.npy": _sha256(genotype_path),
                "cleaned_pheno/sim_pheno.csv": _sha256(phenotype_path),
                "sample_ids.csv": _sha256(sample_ids_path),
                f"splits/{split_path.name}": _sha256(split_path),
                **{
                    f"truth/{path.name}": _sha256(path) for path in truth_files
                },
            },
            "genotype_shape": list(genotype.shape),
            "genotype_dtype": str(genotype.dtype),
            "phenotype_shape": list(phenotype.shape),
            "phenotype_dtype": str(phenotype.dtype),
            "phenotype_columns": phenotype_frame.columns.tolist(),
            "time_coordinates": time_coordinates,
            "sample_ids_sha256": split.metadata["sample_ids_sha256"],
            "split": {
                "path": str(output_dir / "splits" / split_path.name),
                "seed": split.metadata["seed"],
                "ratios": split.metadata["ratios"],
                "train_n": len(split.train_idx),
                "val_n": len(split.val_idx),
                "test_n": len(split.test_idx),
                "generator": split.metadata["generator"],
            },
            "simulation_config_path": str(config_path) if config_path.is_file() else None,
            "model_inputs": ["genotype.npy", "cleaned_pheno/sim_pheno.csv"],
            "truth_is_model_input": False,
        }
        manifest_path = temporary / "data_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖: {output_dir}")
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    manifest = prepare_simulation(
        args.input_dir,
        args.output_dir,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

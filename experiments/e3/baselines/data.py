"""E3 共享数据契约；只加载既有 split，绝不重新划分。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.Genotype import TrainingDosageImputer, load_dosage_file, validate_dosage
from utils.Split import IndividualSplit, load_split


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


@dataclass
class E3DataBundle:
    data_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    genotype_path: Path
    phenotype_path: Path
    sample_ids_path: Path
    split_path: Path
    simulation_config_path: Path | None
    simulation_config: dict[str, Any] | None
    genotype: np.ndarray
    phenotype: np.ndarray
    sample_ids: np.ndarray
    time_coordinates: np.ndarray
    split: IndividualSplit

    @classmethod
    def load(
        cls,
        data_dir: str | Path,
        *,
        formal: bool,
        expected_condition: str | None = None,
        expected_repetition: int | None = None,
    ) -> "E3DataBundle":
        data_dir = Path(data_dir).resolve()
        manifest_path = data_dir / "data_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺少数据清单: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = [
            data_dir / name
            for name in manifest.get("model_inputs", [])
            if Path(name).name.startswith("sim_snp")
        ]
        candidates.extend((data_dir / "sim_snp.npy", data_dir / "sim_snp.csv"))
        genotype_path = next((path for path in candidates if path.is_file()), None)
        if genotype_path is None:
            raise FileNotFoundError(f"缺少 sim_snp.npy/csv: {data_dir}")
        phenotype_path = data_dir / "sim_pheno.csv"
        sample_ids_path = data_dir / "sample_ids.csv"
        split_path = data_dir / "splits/split_seed73.npz"
        simulation_config_path = data_dir / "simulation_config.json"
        for path in (phenotype_path, sample_ids_path, split_path):
            if not path.is_file():
                raise FileNotFoundError(f"缺少 E3 输入: {path}")
        declared_hashes = manifest.get("output_files", {})
        hash_paths = [genotype_path, phenotype_path, sample_ids_path, split_path]
        if simulation_config_path.is_file():
            hash_paths.append(simulation_config_path)
        for path in hash_paths:
            relative = path.relative_to(data_dir).as_posix()
            expected_hash = declared_hashes.get(relative)
            if expected_hash is not None and file_sha256(path) != expected_hash:
                raise ValueError(f"数据文件与 data_manifest 哈希不一致: {relative}")

        sample_frame = pd.read_csv(sample_ids_path)
        if list(sample_frame.columns) != ["SampleID"]:
            raise ValueError("sample_ids.csv 必须且只能包含 SampleID 列")
        sample_ids = sample_frame["SampleID"].astype(str).to_numpy()
        if len(sample_ids) != len(set(sample_ids.tolist())):
            raise ValueError("SampleID 必须唯一")
        genotype, genotype_ids, _ = load_dosage_file(genotype_path)
        validate_dosage(genotype, allow_missing=True)
        if genotype_ids is not None and not np.array_equal(genotype_ids, sample_ids):
            raise ValueError("SNP CSV 与 sample_ids.csv 的身份或顺序不一致")

        phenotype_frame = pd.read_csv(phenotype_path)
        if not len(phenotype_frame.columns) or phenotype_frame.columns[0] != "SampleID":
            raise ValueError("sim_pheno.csv 第一列必须为 SampleID")
        phenotype_ids = phenotype_frame.pop("SampleID").astype(str).to_numpy()
        if not np.array_equal(phenotype_ids, sample_ids):
            raise ValueError("表型与 sample_ids.csv 的身份或顺序不一致")
        phenotype = phenotype_frame.to_numpy(dtype=np.float32)
        if not np.isfinite(phenotype).all():
            raise ValueError("表型必须全部为有限值")
        if genotype.shape[0] != len(sample_ids) or phenotype.shape[0] != len(sample_ids):
            raise ValueError("SNP、表型与 SampleID 行数不一致")

        split = load_split(split_path, sample_ids=sample_ids.tolist())
        time_coordinates = np.asarray(manifest.get("time_coordinates", []), dtype=np.float64)
        if time_coordinates.shape != (phenotype.shape[1],):
            raise ValueError("time_coordinates 与表型时间列数不一致")
        if split.metadata["seed"] != 73:
            raise ValueError("E3 必须复用 seed=73 的原始 split")
        if formal:
            expected = (500, 10000, 20, 400, 50, 50)
            observed = (
                genotype.shape[0],
                genotype.shape[1],
                phenotype.shape[1],
                len(split.train_idx),
                len(split.val_idx),
                len(split.test_idx),
            )
            if observed != expected:
                raise ValueError(f"正式 E3 数据维度不匹配: {observed}")
            if manifest.get("truth_is_model_input") is not False:
                raise ValueError("正式 E3 数据清单必须声明 truth_is_model_input=false")
            if not simulation_config_path.is_file():
                raise FileNotFoundError(f"正式 E3 缺少 simulation_config.json: {data_dir}")
        simulation_config = (
            json.loads(simulation_config_path.read_text(encoding="utf-8"))
            if simulation_config_path.is_file()
            else None
        )
        if formal:
            expected = (expected_condition, expected_repetition, 500, 10000, 20, 0.8)
            observed = (
                simulation_config.get("scenario"),
                simulation_config.get("repetition"),
                simulation_config.get("n_samples"),
                simulation_config.get("n_snps"),
                simulation_config.get("qtl_count"),
                simulation_config.get("target_heritability"),
            )
            if observed != expected:
                raise ValueError(f"正式 E3 simulation_config 与任务不匹配: {observed}")
        return cls(
            data_dir=data_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            genotype_path=genotype_path,
            phenotype_path=phenotype_path,
            sample_ids_path=sample_ids_path,
            split_path=split_path,
            simulation_config_path=(
                simulation_config_path if simulation_config_path.is_file() else None
            ),
            simulation_config=simulation_config,
            genotype=genotype,
            phenotype=phenotype,
            sample_ids=sample_ids,
            time_coordinates=time_coordinates,
            split=split,
        )

    def dosage_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """众数填补只在 train 上拟合，然后应用到三组。"""
        imputer = TrainingDosageImputer("mode").fit(
            self.genotype, train_idx=self.split.train_idx
        )
        blocks = tuple(
            imputer.transform(np.asarray(self.genotype[index]))
            for index in (self.split.train_idx, self.split.val_idx, self.split.test_idx)
        )
        train_dosage = np.asarray(self.genotype[self.split.train_idx])
        metadata = {
            "strategy": "training_mode_imputation",
            "missing_count_train": int(
                (train_dosage == -1).sum() + np.isnan(train_dosage).sum()
            ),
            "fill_values": imputer.fill_values_.tolist(),
        }
        return *blocks, metadata

    def phenotype_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.asarray(self.phenotype[index], dtype=np.float32)
            for index in (self.split.train_idx, self.split.val_idx, self.split.test_idx)
        )

    def schema_record(self, *, model_input_snp_count: int) -> dict[str, Any]:
        train_idx = self.split.train_idx.astype(np.int64)
        return {
            "data_dir": str(self.data_dir),
            "data_manifest_path": str(self.manifest_path),
            "data_manifest_sha256": file_sha256(self.manifest_path),
            "genotype_path": str(self.genotype_path),
            "genotype_sha256": file_sha256(self.genotype_path),
            "phenotype_path": str(self.phenotype_path),
            "phenotype_sha256": file_sha256(self.phenotype_path),
            "simulation_config_path": (
                str(self.simulation_config_path) if self.simulation_config_path else None
            ),
            "simulation_config_sha256": (
                file_sha256(self.simulation_config_path) if self.simulation_config_path else None
            ),
            "simulation_seed": (
                self.simulation_config.get("seeds", {}).get("simulation")
                if self.simulation_config
                else None
            ),
            "raw_snp_count": int(self.genotype.shape[1]),
            "model_input_snp_count": model_input_snp_count,
            "n_samples": int(self.genotype.shape[0]),
            "n_time_points": int(self.phenotype.shape[1]),
            "split": {
                "path": str(self.split_path),
                "sha256": file_sha256(self.split_path),
                "sample_ids_sha256": self.split.metadata["sample_ids_sha256"],
                "train_indices_sha256": array_sha256(train_idx),
                "train_indices": train_idx.tolist(),
                "validation_indices": self.split.val_idx.astype(np.int64).tolist(),
                "test_indices": self.split.test_idx.astype(np.int64).tolist(),
            },
        }

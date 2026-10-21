"""Experiment spec v1：显式实验单位，不隐式生成全轴笛卡尔积。"""

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import TFPPConfig
from utils.Split import IndividualSplit, load_split, validate_individual_split


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileRef(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataSpec(StrictModel):
    snp: FileRef
    phenotypes: dict[str, FileRef]
    n_samples: int = Field(ge=3)
    n_snps: int = Field(ge=1)
    sample_id_source: Literal["row_index"] = "row_index"
    sample_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    time_labels: list[str] = Field(min_length=2)
    time_coordinates: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def dimensions(self):
        if (len(self.time_labels) != len(self.time_coordinates)
                or len(set(self.time_labels)) != len(self.time_labels)
                or not np.isfinite(self.time_coordinates).all() or not self.phenotypes):
            raise ValueError("表型、时间列/坐标必须非空、唯一且一致")
        return self


class SplitSpec(StrictModel):
    mode: Literal["regular", "kfold"]
    manifests: list[FileRef] = Field(min_length=1)


class TrainSize(StrictModel):
    count: int | None = Field(default=None, ge=2)
    fraction: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def exclusive(self):
        if (self.count is None) == (self.fraction is None):
            raise ValueError("train_size 必须且只能提供 count 或 fraction")
        return self


class UnitSpec(StrictModel):
    route: Literal["fixed_indices", "random_k", "GWAS", "DC-SIS", "GWAS->DC-SIS", "ConvAE->Ridge"]
    seed: int = Field(default=73, ge=0)
    traits: list[str] = Field(min_length=1)
    train_size: TrainSize = Field(default_factory=lambda: TrainSize(fraction=1))
    ablation: Literal["full", "wide_only", "deep_only", "no_gate_add"] = "full"
    k: int = Field(default=10000, ge=1)
    gwas_k: int = Field(default=10000, ge=1)
    screening_mode: Literal["joint", "per_trait_union"] = "joint"
    fixed_indices: FileRef | None = None
    exports: list[FileRef] = Field(default_factory=list, description="各 manifest 对应 export metadata.json")
    tfpp: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_continuous(cls, value):
        if isinstance(value, dict) and "continuous" in str(value.get("route", "")).lower():
            raise ValueError("continuous_feature TFPP 第9步尚未实现，当前 schema 明确拒绝；不能回退到离散 Embedding")
        return value

    @model_validator(mode="after")
    def supported(self):
        if len(set(self.traits)) != len(self.traits):
            raise ValueError("traits 不得重复")
        if (self.route == "fixed_indices") != (self.fixed_indices is not None):
            raise ValueError("仅 fixed_indices 路线必须提供索引文件引用")
        if self.route == "ConvAE->Ridge":
            if not self.exports or self.ablation != "full" or self.tfpp:
                raise ValueError("ConvAE->Ridge 只接受匹配的 exports 与固定 Ridge，不接受 TFPP 消融/配置")
        elif self.exports:
            raise ValueError("TFPP 离散路线不能提供 ConvAE 特征")
        protected = {"exp_name", "phenotype_names", "time_coordinates", "snp_file", "pheno_dir", "split_file",
                     "screen_k", "gwas_k", "use_kfold", "fold_index", "ablation_mode", "seed",
                     "preselected_snp_file", "screen_cache_dir", "is_save_training_info", "screen_multivariate"}
        if protected & self.tfpp.keys() or set(self.tfpp) - TFPPConfig.model_fields.keys():
            raise ValueError("tfpp 包含未知字段或 runner 管理的字段")
        TFPPConfig(exp_name="runner", **self.tfpp)
        return self


class ExperimentSpec(StrictModel):
    schema_version: Literal[1] = 1
    data: DataSpec
    splits: SplitSpec
    experiments: list[UnitSpec] = Field(min_length=1)
    output_root: str = "outputs/experiments"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    status = git("status", "--porcelain")
    tree = hashlib.sha256(git("diff", "HEAD", "--binary") + status)
    for name in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if name:
            path = root / name.decode()
            tree.update(name + file_digest(path).encode())
    return {"commit": git("rev-parse", "HEAD").decode().strip(), "dirty": bool(status),
            "working_tree_sha256": tree.hexdigest()}


def read_spec(path: str | Path) -> ExperimentSpec:
    path = Path(path).resolve()
    spec = ExperimentSpec.model_validate_json(path.read_text(encoding="utf-8"))
    for ref in [spec.data.snp, *spec.data.phenotypes.values(), *spec.splits.manifests,
                *[ref for unit in spec.experiments for ref in ([unit.fixed_indices] if unit.fixed_indices else []) + unit.exports]]:
        ref.path = str((path.parent / ref.path).resolve())
    spec.output_root = str((path.parent / spec.output_root).resolve())
    return spec


def nested_subset(split: IndividualSplit, size: TrainSize, seed: int) -> IndividualSplit:
    """按 stable sample_id+seed 哈希排序取前缀；不依赖标签，保留 val/test 原始顺序。"""
    if split.metadata.get("excluded_train_idx"):
        raise ValueError("计划输入必须为完整外层 split，不能将已有子集当成外层 train")
    parent = split.train_idx
    count = size.count if size.count is not None else math.floor(len(parent) * size.fraction)
    if not 2 <= count <= len(parent):
        raise ValueError("train_size 超出原训练集或少于2个体")
    ids = split.metadata["sample_ids"]
    order = sorted(parent.tolist(), key=lambda i: (digest([seed, ids[i]]), json.dumps(ids[i])))
    selected = set(order[:count])
    chosen = np.asarray([i for i in parent if i in selected], dtype=np.int64)
    excluded = np.asarray([i for i in parent if i not in selected], dtype=np.int64)
    meta = {**split.metadata, "parent_train_idx": parent.tolist(), "excluded_train_idx": excluded.tolist(),
            "subset_policy": "sample_id_seed_sha256_prefix_v1", "subset_seed": seed}
    result = IndividualSplit(chosen, split.val_idx.copy(), split.test_idx.copy(), meta)
    validate_individual_split(result)
    return result


def random_indices(n_snps: int, k: int, seed: int) -> np.ndarray:
    """唯一输入为训练基因型列数、k 和预先固定 seed；函数不接收任何 phenotype。"""
    return np.sort(np.random.default_rng(seed).choice(n_snps, size=min(n_snps, k), replace=False)).astype(np.int64)


def plan(spec: ExperimentSpec, *, development_smoke: bool = False) -> list[dict]:
    """只读取小型 split/JSON/固定索引；不加载 SNP、表型或潜在特征大数组。"""
    code = code_identity()
    refs = [spec.data.snp, *spec.data.phenotypes.values(), *spec.splits.manifests]
    for ref in refs:
        if not Path(ref.path).is_file():
            raise FileNotFoundError(ref.path)
    splits = []
    for ref in spec.splits.manifests:
        if file_digest(ref.path) != ref.sha256:
            raise ValueError("split 哈希不匹配")
        split = load_split(ref.path, n_samples=spec.data.n_samples)
        if split.metadata["sample_ids_sha256"] != spec.data.sample_ids_sha256:
            raise ValueError("sample_id 顺序哈希不匹配")
        if not len(split.val_idx) or not len(split.test_idx):
            raise ValueError("实验要求非空 val/test；不得将空测试结果作为成功")
        splits.append(split)
    if spec.splits.mode == "regular":
        if len(splits) != 1 or splits[0].metadata.get("fold_index") is not None:
            raise ValueError("regular 必须恰好一个普通 split，不能混入 fold")
    else:
        folds = [s.metadata.get("fold_index") for s in splits]
        if sorted(folds, key=lambda x: -1 if x is None else x) != list(range(1, len(splits) + 1)) or len(splits) < 2:
            raise ValueError("kfold 必须提供完整且唯一的 fold 1..K")
        for split in splits:
            ratio, validation_seed = split.metadata.get("kfold_val_ratio"), split.metadata.get("validation_seed")
            if not isinstance(ratio, (int, float)) or not 0 < ratio < 1 or not isinstance(validation_seed, int):
                raise ValueError("fold 验证比例或 seed 不合法")
            if split.metadata.get("n_folds") != len(splits):
                raise ValueError("fold 声明的折数不一致")
        tests = np.concatenate([s.test_idx for s in splits])
        if not np.array_equal(np.sort(tests), np.arange(spec.data.n_samples)):
            raise ValueError("每个样本必须恰好在一个外层 fold 中作为 test")
    family_hash = digest({"mode": spec.splits.mode, "splits": [r.model_dump() for r in spec.splits.manifests]})
    result = []
    for unit in spec.experiments:
        if not set(unit.traits).issubset(spec.data.phenotypes):
            raise ValueError("trait set 不在 data.phenotypes 中")
        if unit.route == "ConvAE->Ridge" and len(unit.exports) != len(splits):
            raise ValueError("ConvAE->Ridge 必须按 manifest 顺序显式给出每折 export metadata 引用")
        fixed = None
        if unit.fixed_indices:
            if file_digest(unit.fixed_indices.path) != unit.fixed_indices.sha256:
                raise ValueError("fixed_indices 哈希不匹配")
            fixed = np.load(unit.fixed_indices.path, allow_pickle=False)
            if (fixed.ndim != 1 or fixed.dtype.kind not in "iu" or not fixed.size
                    or len(np.unique(fixed)) != len(fixed) or (fixed < 0).any() or (fixed >= spec.data.n_snps).any()):
                raise ValueError("fixed_indices 必须为非空、唯一、有序的合法原始列索引；保持文件顺序")
        for index, (split, ref) in enumerate(zip(splits, spec.splits.manifests)):
            subset = nested_subset(split, unit.train_size, unit.seed)
            export = None
            if unit.route == "ConvAE->Ridge":
                export_ref = unit.exports[index]
                if file_digest(export_ref.path) != export_ref.sha256:
                    raise ValueError("export metadata 哈希不匹配")
                export = json.loads(Path(export_ref.path).read_text())
                if (export.get("paper_eligible") is not True or export["seed"] != unit.seed or export["snp_file_sha256"] != spec.data.snp.sha256
                        or export["phenotype_names"] != unit.traits or export["time_coordinates"] != spec.data.time_coordinates
                        or export["time_labels"] != spec.data.time_labels):
                    raise ValueError("ConvAE export trait set/time 与任务不一致")
                from utils.ConvAE_Preprocessing import convae_split_fingerprint
                if export["split_fingerprint"] != convae_split_fingerprint(subset):
                    raise ValueError("ConvAE export 必须来自当前 train 子集/fold，不能复用全 train 编码器")
            tfpp = TFPPConfig(exp_name="runner", phenotype_names=unit.traits,
                time_coordinates=spec.data.time_coordinates, seed=unit.seed, snp_file=spec.data.snp.path,
                pheno_dir="materialized_by_runner", screen_k=0, gwas_k=0, screen_multivariate=unit.screening_mode == "joint",
                ablation_mode={"full": "all", "wide_only": "wide", "deep_only": "deep", "no_gate_add": "add"}[unit.ablation],
                **unit.tfpp).model_dump(mode="json")
            from utils.ConvAE_Ridge import baseline_config
            ridge_protocol = baseline_config(unit.seed, digest({"data": spec.data.model_dump(), "traits": unit.traits})) if export else None
            if ridge_protocol is not None:
                ridge_protocol["target_data_refs_sha256"] = ridge_protocol.pop("phenotype_file_sha256")
            method_id = unit.route + (":" + unit.screening_mode if unit.route in ("GWAS", "DC-SIS", "GWAS->DC-SIS") else "")
            resolved = {"schema_version": 1, "metrics_protocol": "time_adjusted_r2_nan_pcc_v2", "std_ddof": 0,
                "code": code, "development_smoke": development_smoke,
                "data": spec.data.model_dump(), "split_mode": spec.splits.mode, "split_family_sha256": family_hash,
                "outer_split": ref.model_dump(), "fold": split.metadata.get("fold_index"),
                "subset": {"train_idx": subset.train_idx.tolist(), "val_idx": subset.val_idx.tolist(),
                           "test_idx": subset.test_idx.tolist(), "metadata": subset.metadata},
                "subset_sha256": digest({"train": subset.train_idx.tolist(), "parent": ref.sha256, "seed": unit.seed}),
                "unit": unit.model_dump(), "method_id": method_id,
                "tfpp_resolved": None if export else tfpp, "ridge_protocol": ridge_protocol,
                "early_stopping": "min_val_loss" if export is None else "min_normalized_val_mse_fixed_Ridge_grid",
                "metrics_policy": "TFPP_time_adjusted_R2; mean_time_then_trait; undefined_PCC_NaN_valid_counts"}
            resolved["statistical_unit_sha256"] = digest({"unit": unit.model_dump(), "data": spec.data.model_dump(),
                "code": code, "family": family_hash, "tfpp": resolved["tfpp_resolved"], "smoke": development_smoke})
            resolved["expected_folds"] = len(splits)
            config_hash = digest(resolved)
            result.append({"config_hash": config_hash, "output_dir": str(Path(spec.output_root) / config_hash), "resolved": resolved})
    if len({p["config_hash"] for p in result}) != len(result):
        raise ValueError("计划中有重复实验单位")
    return result

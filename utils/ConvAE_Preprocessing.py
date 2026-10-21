"""ConvAE 数据管线 schema 3：保存并严格校验共享 split、scaler 和分块语义。"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .Genotype import GENOTYPE_ENCODING_VERSION
from .Inference import file_sha256
from .Normalization import load_scalers, save_scalers
from .Screening import array_fingerprint, training_indices_hash
from .Split import IndividualSplit, load_split, save_split, validate_individual_split


def _fold_metadata(split: IndividualSplit) -> dict:
    values = {key: split.metadata.get(key) for key in
              ("fold_index", "n_folds", "kfold_val_ratio", "validation_seed")}
    fold = values["fold_index"]
    if fold is None:
        if any(value is not None for value in values.values()):
            raise ValueError("ConvAE 普通 split 含有不完整的 fold 标识")
    else:
        count, ratio, seed = values["n_folds"], values["kfold_val_ratio"], values["validation_seed"]
        if (not isinstance(fold, int) or not isinstance(count, int) or not 1 <= fold <= count
                or count < 2 or not isinstance(ratio, (int, float)) or not 0 < ratio < 1
                or not isinstance(seed, int)):
            raise ValueError("ConvAE fold 编号、折数、验证比例或 seed 不合法")
    return values


def convae_split_fingerprint(split) -> str:
    """对全部三组行序、个体身份及 fold 标识哈希，不依赖文件名/生成时间。"""
    validate_individual_split(split)
    _fold_metadata(split)
    payload = {name: getattr(split, name).tolist() for name in ("train_idx", "val_idx", "test_idx")}
    payload.update({key: split.metadata.get(key) for key in
                    ("sample_ids_sha256", "sample_id_source", "n_samples", "seed", "generator",
                     "fold_index", "n_folds", "kfold_val_ratio", "validation_seed")})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _contract(dataset, config) -> dict:
    times = config.time_coordinates
    if times is not None and (len(times) != dataset.time_points or not np.isfinite(times).all()):
        raise ValueError("time_coordinates 必须与原始表型时间列一一对应且为有限值")
    if config.chunk_size != dataset.chunk_size or config.overlap != dataset.overlap:
        raise ValueError("配置 chunk_size/overlap 与 Dataset 不一致")
    return {
        "n_samples": len(dataset), "sample_id_source": dataset.sample_id_source,
        "phenotype_names": list(dataset.phenotype_names), "time_points": dataset.time_points,
        "time_labels": dataset.time_labels, "time_coordinates": times,
        "time_coordinate_source": "csv_columns" if times is None else "config",
        "genotype_encoding_version": GENOTYPE_ENCODING_VERSION,
        "input_mode": "categorical0123_missing0", "original_length": dataset.original_length,
        "target_length": dataset.target_length, "chunk_size": dataset.chunk_size,
        "overlap": dataset.overlap, "step_size": dataset.step_size, "num_chunks": dataset.num_chunks,
        "padding_mask_semantics": "true_is_structural_padding_not_genotype_missing",
        "phenotype_missing_strategy": config.phenotype_missing_strategy,
        "norm_method": "global", "scaler_type": "zscore",
    }


@dataclass
class ConvAEPreprocessingArtifacts:
    metadata: dict
    split: IndividualSplit
    scalers: dict
    padding_mask: np.ndarray
    chunk_padding_mask: np.ndarray


def save_convae_preprocessing(dataset, data, config, *, split_source=None) -> dict:
    """保存预处理而非潜在特征；已有实验不允许被不同数据契约覆盖。"""
    root = Path(config.output_dir)
    prep = root / "preprocessing"
    contract = _contract(dataset, config)
    train_fingerprint = array_fingerprint(dataset.phenotype[data.split.train_idx])
    if (prep / "preprocessing.json").exists():
        artifacts = load_convae_preprocessing(root, dataset=dataset, config=config)
        if split_source is not None and file_sha256(split_source) != artifacts.metadata["split_sha256"]:
            raise ValueError("来源 split/fold 文件的哈希已变化，请使用新的 exp_name")
        if artifacts.metadata["train_phenotype_sha256"] != train_fingerprint:
            raise ValueError("已有实验的 train phenotype 已变化，请使用新的 exp_name")
        if artifacts.metadata["split_fingerprint"] != convae_split_fingerprint(data.split):
            raise ValueError("已有实验 split 不同，禁止覆盖")
        return artifacts.metadata
    prep.mkdir(parents=True, exist_ok=True)
    copied = prep / "split.npz"
    if split_source is not None:
        source = Path(split_source)
        if source.resolve() != copied.resolve():
            if copied.exists() and copied.read_bytes() != source.read_bytes():
                raise ValueError("已有 split 副本与来源文件不同，禁止覆盖")
            shutil.copyfile(source, copied)
    elif not copied.exists():
        save_split(data.split, copied)
    saved = load_split(copied)
    if convae_split_fingerprint(saved) != convae_split_fingerprint(data.split):
        raise ValueError("保存的 split 与使用的内存索引不一致")
    scaler_path = root / "scalers" / "scaler.pkl"
    save_scalers(data.scalers, scaler_path)
    np.save(prep / "padding_mask.npy", dataset.padding_mask.numpy())
    np.save(prep / "chunk_padding_mask.npy", dataset.chunk_padding_mask.numpy())
    (root / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    meta = {
        **contract, "schema_version": 3, "pipeline": "convae_train_only_v1",
        "split_source": "in_memory" if split_source is None else str(Path(split_source).resolve()),
        "split_sha256": file_sha256(copied), "split_fingerprint": convae_split_fingerprint(saved),
        "sample_ids_sha256": saved.metadata["sample_ids_sha256"],
        "train_indices_sha256": training_indices_hash(saved.train_idx),
        "train_phenotype_sha256": train_fingerprint,
        "fold_metadata": _fold_metadata(saved),
        "scaler_sha256": file_sha256(scaler_path),
        "padding_mask_sha256": file_sha256(prep / "padding_mask.npy"),
        "chunk_padding_mask_sha256": file_sha256(prep / "chunk_padding_mask.npy"),
        "config_sha256": file_sha256(root / "config.json"),
    }
    (prep / "preprocessing.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    load_convae_preprocessing(root, dataset=dataset, config=config)
    return meta


def load_convae_preprocessing(root: str | Path, *, dataset=None, config=None) -> ConvAEPreprocessingArtifacts:
    """校验所有必需产物及数据契约，只加载 scaler，绝不重新拟合。"""
    root = Path(root)
    prep = root / "preprocessing"
    meta_path = prep / "preprocessing.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 ConvAE 预处理记录: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "pipeline", "n_samples", "sample_id_source", "sample_ids_sha256",
        "phenotype_names", "time_points", "time_labels", "time_coordinates", "time_coordinate_source",
        "genotype_encoding_version", "input_mode", "original_length", "target_length",
        "chunk_size", "overlap", "step_size", "num_chunks", "padding_mask_semantics",
        "phenotype_missing_strategy", "norm_method", "scaler_type", "fold_metadata", "split_source",
        "split_fingerprint", "train_indices_sha256", "train_phenotype_sha256",
    }
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("ConvAE schema 3 缺少必需预处理字段")
    if meta["schema_version"] != 3 or meta["pipeline"] != "convae_train_only_v1":
        raise ValueError("ConvAE 预处理版本不兼容")
    paths = {"split": prep / "split.npz", "scaler": root / "scalers" / "scaler.pkl",
             "padding_mask": prep / "padding_mask.npy", "chunk_padding_mask": prep / "chunk_padding_mask.npy",
             "config": root / "config.json"}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少 ConvAE 训练产物: {path}")
        if file_sha256(path) != meta.get(key + "_sha256"):
            raise ValueError(f"ConvAE 产物哈希缺失或不匹配: {key}")
    split = load_split(paths["split"])
    if (not isinstance(meta["fold_metadata"], dict)
            or set(meta["fold_metadata"]) != {"fold_index", "n_folds", "kfold_val_ratio", "validation_seed"}):
        raise ValueError("ConvAE fold 元数据不完整")
    if (split.metadata["n_samples"] != meta["n_samples"]
            or split.metadata["sample_id_source"] != meta["sample_id_source"]
            or split.metadata["sample_ids_sha256"] != meta["sample_ids_sha256"]
            or training_indices_hash(split.train_idx) != meta["train_indices_sha256"]
            or convae_split_fingerprint(split) != meta["split_fingerprint"]
            or _fold_metadata(split) != meta["fold_metadata"]):
        raise ValueError("ConvAE split/fold 或个体身份顺序与训练记录不一致")
    length, chunk, overlap = meta["original_length"], meta["chunk_size"], meta["overlap"]
    if (not all(isinstance(value, int) for value in (length, chunk, overlap))
            or length < 1 or chunk < 1 or not 0 <= overlap < chunk):
        raise ValueError("ConvAE 原始有效长度或 chunk 参数不合法")
    step = chunk - overlap
    count = 1 + (max(0, length - chunk) + step - 1) // step
    target = chunk + (count - 1) * step
    if (step, count, target) != (meta["step_size"], meta["num_chunks"], meta["target_length"]):
        raise ValueError("ConvAE chunk/padding 长度记录不一致")
    padding = np.load(paths["padding_mask"], allow_pickle=False)
    chunks = np.load(paths["chunk_padding_mask"], allow_pickle=False)
    expected = np.arange(target) >= length
    expected_chunks = np.stack([expected[start:start + chunk] for start in range(0, target - chunk + 1, step)])
    if (padding.dtype != np.bool_ or chunks.dtype != np.bool_
            or not np.array_equal(padding, expected) or not np.array_equal(chunks, expected_chunks)):
        raise ValueError("ConvAE 结构性 padding mask 与原始有效长度不一致")
    if (meta["genotype_encoding_version"] != GENOTYPE_ENCODING_VERSION
            or meta["input_mode"] != "categorical0123_missing0"
            or meta["padding_mask_semantics"] != "true_is_structural_padding_not_genotype_missing"
            or meta["phenotype_missing_strategy"] != "error"
            or meta["norm_method"] != "global" or meta["scaler_type"] != "zscore"):
        raise ValueError("ConvAE 编码、padding 或归一化约定不兼容")
    saved_config = json.loads(paths["config"].read_text(encoding="utf-8"))
    for key in ("chunk_size", "overlap", "phenotype_names", "time_coordinates", "phenotype_missing_strategy"):
        if saved_config.get(key) != meta[key]:
            raise ValueError(f"ConvAE 保存配置与预处理元数据不一致: {key}")
    times = meta["time_coordinates"]
    if (len(meta["time_labels"]) != meta["time_points"] or meta["time_points"] < 1
            or (times is not None and (len(times) != meta["time_points"] or not np.isfinite(times).all()))):
        raise ValueError("ConvAE 表型时间点记录不一致")
    if meta["time_coordinate_source"] != ("csv_columns" if times is None else "config"):
        raise ValueError("ConvAE 时间坐标来源与保存的配置不一致")
    scalers = load_scalers(paths["scaler"])
    if scalers.get("phenotype_names") != meta["phenotype_names"] or scalers.get("norm_method") != "global":
        raise ValueError("ConvAE scaler 表型或归一化方式不一致")
    for name in meta["phenotype_names"]:
        scaler = scalers["pheno_meta"]["scalers"][name]
        if scaler.n_features_in_ != 1 or scaler.n_samples_seen_ != len(split.train_idx) * meta["time_points"]:
            raise ValueError("ConvAE scaler 训练样本/时间点数量与 split 不一致")
    if dataset is not None:
        if (dataset.sample_ids.tolist() != split.metadata["sample_ids"]
                or dataset.sample_id_source != meta["sample_id_source"]
                or len(dataset) != meta["n_samples"]
                or dataset.phenotype_names != meta["phenotype_names"]
                or dataset.time_points != meta["time_points"] or dataset.time_labels != meta["time_labels"]
                or dataset.original_length != length or dataset.chunk_size != chunk or dataset.overlap != overlap):
            raise ValueError("当前 ConvAE 数据的个体顺序、表型/时间点或有效长度/chunk 参数不一致")
        if array_fingerprint(dataset.phenotype[split.train_idx]) != meta["train_phenotype_sha256"]:
            raise ValueError("ConvAE train phenotype 指纹已变化，请使用新的 exp_name")
    if config is not None:
        for key in ("phenotype_names", "time_coordinates", "chunk_size", "overlap", "phenotype_missing_strategy"):
            if getattr(config, key) != meta[key]:
                raise ValueError(f"当前 ConvAE 配置不匹配训练数据契约: {key}")
        if config.model_dump(mode="json") != saved_config:
            raise ValueError("已有 ConvAE 实验配置不同，请使用新的 exp_name，禁止覆盖旧产物")
    return ConvAEPreprocessingArtifacts(meta, split, scalers, padding, chunks)

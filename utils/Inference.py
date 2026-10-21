"""只读取训练产物的 TFPP 推理预处理；此模块没有 fit 或筛选入口。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .Genotype import GENOTYPE_ENCODING_VERSION, dosage_to_model_codes, model_code_tensor
from .Normalization import load_scalers
from .Screening import array_fingerprint, training_indices_hash
from .Split import IndividualSplit, load_split


def file_sha256(path: str | Path) -> str:
    """训练产物的字节哈希。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class InferenceArtifacts:
    metadata: dict
    config: dict
    split: IndividualSplit
    indices: np.ndarray
    scalers: dict
    wide_fill_values: np.ndarray
    hashes: dict


def load_inference_artifacts(exp_dir: str | Path, config: dict) -> InferenceArtifacts:
    """严格读取第2步 schema 2 或带显式哈希的 schema 3；绝不补建缺失产物。"""
    root = Path(exp_dir)
    prep = root / "preprocessing"
    paths = {
        "metadata": prep / "preprocessing.json",
        "split": prep / "split.npz",
        "screened_indices": prep / "screened_indices.npy",
        "scaler": root / "scalers" / "scaler.pkl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少训练产物: {path}；请恢复原实验产物，禁止重新拟合")
    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not isinstance(config, dict):
        raise ValueError("预处理元数据和模型配置必须为 JSON 对象")
    required = {
        "schema_version", "training_mode", "split_sha256", "sample_ids_sha256",
        "train_indices_sha256", "original_snp_dim", "snp_dim", "phenotype_names",
        "screening_dosage_encoding_version", "screen_missing_strategy",
        "wide_input_mode", "wide_imputer",
        "screen_k", "gwas_k", "screen_multivariate", "screening_algorithm_version",
    }
    missing = required - meta.keys()
    if missing:
        raise ValueError(f"预处理元数据缺少字段: {sorted(missing)}")
    if meta["schema_version"] not in (2, 3) or meta["training_mode"] not in ("regular", "kfold"):
        raise ValueError("只支持完整的普通/折内预处理产物 schema 2/3")
    if meta["training_mode"] == "kfold":
        fold = meta.get("fold_index")
        if (meta["schema_version"] != 3 or not isinstance(fold, int) or fold < 1
                or config.get("fold_index") != fold):
            raise ValueError("折内预处理编号或版本与模型配置不一致")
    if meta["screening_dosage_encoding_version"] != GENOTYPE_ENCODING_VERSION:
        raise ValueError("基因型预处理版本不兼容，不能猜测编码")
    if meta["wide_input_mode"] != "dosage_v1":
        raise ValueError("预处理输入模式必须为 dosage_v1")
    if meta["screen_missing_strategy"] not in ("error", "mode"):
        raise ValueError("未知的筛选缺失策略")
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes["split"] != meta["split_sha256"]:
        raise ValueError("split 文件哈希与训练记录不一致")
    split = load_split(paths["split"])
    if meta["training_mode"] == "kfold" and split.metadata.get("fold_index") != meta["fold_index"]:
        raise ValueError("split 中的 fold_index 与训练记录不一致")
    if meta["training_mode"] == "kfold":
        if (config.get("n_folds") != split.metadata.get("n_folds")
                or config.get("kfold_val_ratio") != split.metadata.get("kfold_val_ratio")
                or meta.get("model_initialization_seed") != config.get("seed")):
            raise ValueError("折数、折内验证比例或模型初始化 seed 与保存的记录不一致")
    if (split.metadata["sample_ids_sha256"] != meta["sample_ids_sha256"]
            or training_indices_hash(split.train_idx) != meta["train_indices_sha256"]):
        raise ValueError("split 个体顺序或训练索引哈希不一致")
    indices = np.load(paths["screened_indices"], allow_pickle=False)
    original_dim, dim = meta["original_snp_dim"], meta["snp_dim"]
    if (not isinstance(original_dim, int) or not isinstance(dim, int)
            or not 0 < dim <= original_dim or indices.shape != (dim,)
            or indices.dtype.kind not in "iu" or np.unique(indices).size != dim
            or (indices < 0).any() or (indices >= original_dim).any()):
        raise ValueError("screened_indices 必须是互异、合法的原始 SNP 列索引")
    for key in ("snp_dim", "phenotype_names", "wide_input_mode", "screen_missing_strategy",
                "screen_k", "gwas_k", "screen_multivariate", "screening_algorithm_version"):
        if key not in config or config[key] != meta[key]:
            raise ValueError(f"model_config 与预处理元数据不一致: {key}")
    if config.get("genotype_encoding_version") != GENOTYPE_ENCODING_VERSION:
        raise ValueError("model_config 缺少或不兼容的基因型编码版本")
    times = config.get("time_coordinates")
    if (not isinstance(times, list) or not times or not np.isfinite(times).all()
            or config.get("time_points") != len(times)):
        raise ValueError("model_config 时间点或时间坐标不合法")
    names = config["phenotype_names"]
    if (not isinstance(names, list) or not names or len(names) != len(set(names))
            or not all(isinstance(name, str) and name for name in names)):
        raise ValueError("表型名称必须非空且互异")
    wide = meta["wide_imputer"]
    if not isinstance(wide, dict):
        raise ValueError("缺少训练期 Wide 预处理参数")
    strategy = wide.get("strategy")
    if strategy != config.get("wide_missing_strategy") or strategy not in ("mode", "error"):
        raise ValueError("Wide 缺失策略不一致")
    if strategy == "mode":
        if "fill_values" not in wide:
            raise ValueError("缺少训练期 Wide 填充值")
        fill = np.asarray(wide["fill_values"], dtype=np.float32)
        if (fill.shape != (dim,) or not np.isin(fill, [0, 1, 2]).all()
                or wide.get("train_idx") != split.train_idx.tolist()
                or wide.get("encoding_version") != GENOTYPE_ENCODING_VERSION):
            raise ValueError("Wide 填充值、训练行号或编码版本不一致")
    else:
        fill = np.array([], dtype=np.float32)  # error 策略的持久化空 buffer，拒绝缺失。
    hashes["wide_fill_values"] = array_fingerprint(fill)
    if meta["schema_version"] == 3:
        for key in ("screened_indices", "scaler", "wide_fill_values"):
            if meta.get(key + "_sha256") != hashes[key]:
                raise ValueError(f"训练产物缺少哈希或哈希不一致: {key}")
        if meta.get("time_coordinates") != times:
            raise ValueError("训练时间坐标与模型配置不一致")
    scalers = load_scalers(paths["scaler"])
    if not isinstance(scalers, dict) or scalers.get("phenotype_names") != names:
        raise ValueError("scaler 的表型名称/顺序与模型不一致")
    if ("norm_method" not in scalers or "pheno_meta" not in scalers
            or set(scalers["pheno_meta"].get("scalers", {})) != set(names)):
        raise ValueError("scaler 缺少归一化方法或表型缩放器")
    # 空合成块只做 transform，提前校验 scaler 的时间维及残差均值曲线。
    transform_phenotypes(np.zeros((1, len(names), len(times))), scalers)
    return InferenceArtifacts(meta, config, split, indices, scalers, fill, hashes)


def prepare_inference_input(dosage: np.ndarray, artifacts: InferenceArtifacts,
                            *, input_stage: str) -> torch.Tensor:
    """输入阶段必须显式指定；raw 切片一次，screened 不再切片。"""
    if input_stage not in ("raw", "screened"):
        raise ValueError("必须显式指定 input_stage=raw 或 screened，不能按维度猜测")
    array = np.asarray(dosage)
    expected = artifacts.metadata["original_snp_dim" if input_stage == "raw" else "snp_dim"]
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(f"{input_stage} SNP 应为二维且具有 {expected} 列；请检查输入阶段及原始列顺序")
    # 先校验全矩阵，防止 raw 的未选中列携带错误的类别编码。
    codes = dosage_to_model_codes(array)
    if input_stage == "raw":
        codes = codes[:, artifacts.indices]
    if artifacts.metadata["wide_imputer"]["strategy"] == "error" and (codes == 0).any():
        raise ValueError("训练期 Wide 使用 error 策略，输入不能有缺失；禁止推理期重新填补")
    return model_code_tensor(codes)


def transform_phenotypes(values: np.ndarray, scalers: dict) -> torch.Tensor:
    """仅调用保存的 scaler.transform，包含训练期残差曲线的减法。"""
    data = np.asarray(values, dtype=np.float32).copy()
    names = scalers["phenotype_names"]
    if data.ndim != 3 or data.shape[1] != len(names) or not np.isfinite(data).all():
        raise ValueError("测试表型必须为有限值 [个体, 表型, 时间点]")
    method = scalers["norm_method"]
    if method not in ("global", "timepoint", "residual_global", "residual_timepoint"):
        raise ValueError(f"不支持保存的归一化方法: {method}")
    meta = scalers["pheno_meta"]
    if "residual" in method:
        curve = np.asarray(meta["mean_curve"])
        if curve.shape != (1, data.shape[1], data.shape[2]):
            raise ValueError("保存的残差均值曲线时间维不匹配")
        data -= curve
    for i, name in enumerate(names):
        scaler = meta["scalers"][name]
        block = data[:, i]
        transformed = scaler.transform(block if "timepoint" in method else block.reshape(-1, 1))
        data[:, i] = transformed.reshape(block.shape)
    return torch.tensor(data, dtype=torch.float32)

"""集中定义基因型语义：筛选剂量 0/1/2，模型类别 0=缺失、1/2/3=剂量0/1/2。"""

from typing import Literal

import numpy as np
import torch

GENOTYPE_ENCODING_VERSION = "dosage012_missing-1_categorical0123_v1"
MissingStrategy = Literal["error", "mode"]


def validate_dosage(data: np.ndarray, *, allow_missing: bool = False) -> np.ndarray:
    """只接受二维整数剂量；原始缺失可显式标为 -1 或 NaN，不能猜测类别码。"""
    array = np.asarray(data)
    if array.ndim != 2 or array.dtype.kind not in "iuf":
        raise ValueError("screening dosage 必须是二维数值矩阵")
    observed = (array == 0) | (array == 1) | (array == 2)
    if allow_missing:
        observed |= (array == -1) | np.isnan(array)
    if not observed.all():
        raise ValueError(
            "screening dosage 只允许 0/1/2；模型类别码须先显式解码，缺失须按训练策略处理"
        )
    return array


def dosage_to_model_codes(dosage: np.ndarray) -> np.ndarray:
    """原始 -1/NaN → 模型缺失类别0；剂量0/1/2 → 类别1/2/3。"""
    dosage = validate_dosage(dosage, allow_missing=True)
    clean = np.where(np.isnan(dosage), -1, dosage).astype(np.int8)
    return clean + 1


def model_codes_to_dosage(codes: np.ndarray) -> np.ndarray:
    """显式解码；模型缺失类别0恢复为 -1，绝不恢复成真实剂量0。"""
    codes = np.asarray(codes)
    if (
        codes.ndim != 2
        or codes.dtype.kind not in "iuf"
        or not np.isin(codes, [0, 1, 2, 3]).all()
    ):
        raise ValueError("model categorical code 必须是二维 0/1/2/3 整数矩阵")
    return codes.astype(np.int8) - 1


def model_code_tensor(codes: torch.Tensor | np.ndarray) -> torch.Tensor:
    """只将已明确声明的模型类别转为 Embedding 索引；拒绝截断连续特征。"""
    tensor = torch.as_tensor(codes)
    if tensor.ndim < 2 or tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError("模型类别必须为至少二维的 0/1/2/3 整数张量")
    if not torch.isfinite(tensor).all() or not ((tensor >= 0) & (tensor <= 3)).all():
        raise ValueError("模型类别只能为 0/1/2/3，0 表示缺失或 ConvAE 边界填充")
    if tensor.is_floating_point() and not torch.equal(tensor, tensor.trunc()):
        raise ValueError("连续特征不能截断为模型类别码")
    return tensor.to(dtype=torch.long)


class TrainingDosageImputer:
    """仅按显式 train_idx 拟合逐位点众数；并列取较小剂量，全缺失位点报错。

    普通训练必须在完成划分后再 fit，不能全量拟合。
    metadata 可作为 JSON 随实验保存，包含训练行号、每列填充值和缺失计数。
    """

    def __init__(self, strategy: MissingStrategy = "mode"):
        if strategy not in ("error", "mode"):
            raise ValueError("未知的筛选缺失策略")
        self.strategy = strategy
        self.fill_values_: np.ndarray | None = None
        self.metadata_: dict | None = None

    def fit(
        self, dosage: np.ndarray, *, train_idx: np.ndarray
    ) -> "TrainingDosageImputer":
        self.fill_values_, self.metadata_ = None, None
        dosage = validate_dosage(dosage, allow_missing=True)
        indices = np.asarray(train_idx)
        if (
            indices.ndim != 1
            or indices.size == 0
            or indices.dtype.kind not in "iu"
            or np.unique(indices).size != indices.size
            or (indices < 0).any()
            or (indices >= len(dosage)).any()
        ):
            raise ValueError("fit 需要非空、互异且范围合法的训练个体索引 train_idx")
        train = dosage[indices]
        missing = (train == -1) | np.isnan(train)
        if self.strategy == "error" and missing.any():
            raise ValueError("训练个体存在缺失基因型，error 策略拒绝拟合")
        counts = np.stack([(train == value).sum(axis=0) for value in (0, 1, 2)])
        empty = np.flatnonzero(counts.sum(axis=0) == 0)
        if empty.size:
            raise ValueError(f"训练个体中全缺失的 SNP 列无法拟合: {empty.tolist()}")
        self.fill_values_ = counts.argmax(axis=0).astype(np.int8)
        self.metadata_ = {
            "encoding_version": GENOTYPE_ENCODING_VERSION,
            "strategy": self.strategy,
            "tie_break": "lowest_observed_dosage",
            "train_idx": indices.astype(np.int64).tolist(),
            "train_missing_counts": missing.sum(axis=0).tolist(),
            "fill_values": self.fill_values_.tolist(),
        }
        return self

    def transform(self, dosage: np.ndarray) -> np.ndarray:
        if self.fill_values_ is None:
            raise ValueError("必须先仅在 train_idx 上 fit，不能自动全量填补")
        dosage = validate_dosage(dosage, allow_missing=True)
        if dosage.shape[1] != len(self.fill_values_):
            raise ValueError("填补器的 SNP 列数与输入不一致")
        missing = (dosage == -1) | np.isnan(dosage)
        if self.strategy == "error" and missing.any():
            raise ValueError("error 策略拒绝缺失基因型")
        result = np.where(missing, self.fill_values_[None, :], dosage).astype(np.int8)
        return validate_dosage(result)


def prepare_screening_dosage(
    dosage: np.ndarray,
    *,
    missing_strategy: MissingStrategy = "error",
    imputer: TrainingDosageImputer | None = None,
) -> np.ndarray:
    """校验筛选剂量；mode 遇到缺失必须提供已在训练个体上拟合的填补器。"""
    if missing_strategy not in ("error", "mode"):
        raise ValueError("未知的筛选缺失策略")
    dosage = validate_dosage(dosage, allow_missing=True)
    if imputer is not None:
        if imputer.strategy != missing_strategy:
            raise ValueError("填补器策略与筛选配置不一致")
        return imputer.transform(dosage)
    if ((dosage == -1) | np.isnan(dosage)).any():
        raise ValueError(
            f"筛选发现缺失基因型 (策略={missing_strategy})；禁止当作剂量0。"
            "mode 需要仅在 train_idx 上 fit 的填补器，禁止自动全量 fit"
        )
    return validate_dosage(dosage)


def tfpp_input_views(
    codes: torch.Tensor,
    wide_fill_values: torch.Tensor | None = None,
    *,
    legacy_categorical: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分离 Deep 类别与 Wide 剂量；旧路径仅供未迁移的 K折/模拟与旧权重。"""
    deep = model_code_tensor(codes)
    if legacy_categorical:
        return deep, deep.float() / 2.0
    dosage = deep.float() - 1.0
    missing = deep == 0
    if wide_fill_values is not None and wide_fill_values.numel():
        fill = wide_fill_values.to(device=deep.device, dtype=dosage.dtype)
        if fill.shape != (deep.shape[-1],) or not torch.isfinite(fill).all():
            raise ValueError("Wide 填补参数必须与 SNP 列一一对应")
        if not ((fill >= 0) & (fill <= 2)).all():
            raise ValueError("Wide 填补剂量必须在 [0,2] 内")
        dosage = torch.where(missing, fill, dosage)
    elif missing.any():
        raise ValueError("Wide 缺失值必须使用训练期保存的填补参数")
    return deep, dosage / 2.0

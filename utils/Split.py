"""按个体生成、保存和校验可复用的 train/val/test 行索引。"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass
class IndividualSplit:
    """三组索引均指向原始个体行；不会沿表型或时间维度拆分。"""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: dict


def _sample_identity(n_samples=None, sample_ids=None):
    if sample_ids is not None:
        sample_ids = list(sample_ids)
        if n_samples is None:
            n_samples = len(sample_ids)
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples 必须为正整数")
    n_samples = int(n_samples)
    source = "row_index" if sample_ids is None else "provided"
    if sample_ids is None:
        sample_ids = list(range(n_samples))
    if len(sample_ids) != n_samples:
        raise ValueError("sample_ids 长度与 n_samples 不匹配")
    ids = []
    for value in sample_ids:
        if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            ids.append(int(value))
        elif isinstance(value, str) and value:
            ids.append(value)
        else:
            raise ValueError("sample_id 必须为整数或非空字符串")
    tokens = [json.dumps(value, ensure_ascii=False) for value in ids]
    if len(set(tokens)) != n_samples:
        raise ValueError("sample_ids 不得重复；每行必须对应一个独立个体")
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return n_samples, ids, source, digest


def _ratios(ratios):
    values = np.asarray(ratios, dtype=np.float64)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("ratios 必须为非负且总和为 1 的 (train, val, test) 三元组")
    return values.tolist()


def validate_split_indices(train_idx, val_idx, test_idx, n_samples: int) -> None:
    """拒绝非整数、重复、越界、交叉或未完整覆盖个体的索引。"""
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples 必须为正整数")
    arrays = []
    for name, indices in zip(("train", "val", "test"), (train_idx, val_idx, test_idx)):
        array = np.asarray(indices)
        if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
            raise ValueError(f"{name} 索引必须为一维整数数组")
        if (array < 0).any() or (array >= n_samples).any():
            raise ValueError(f"{name} 索引越界")
        arrays.append(array.astype(np.int64))
    combined = np.concatenate(arrays)
    if np.unique(combined).size != combined.size:
        raise ValueError("集合内索引重复或集合之间存在交集")
    if combined.size != n_samples:
        raise ValueError("三组索引未完整覆盖全部个体")


def generate_split(
    n_samples: int | None = None,
    *,
    sample_ids: Sequence[str | int] | None = None,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 73,
) -> IndividualSplit:
    """使用局部随机生成器划分个体，保持现有 torch.random_split 的顺序。"""
    n_samples, ids, source, digest = _sample_identity(n_samples, sample_ids)
    ratios = _ratios(ratios)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed 必须为整数")
    seed = int(seed)
    # 与 random_split 的比例取整一致：向下取整后按 train/val/test 补余数。
    sizes = [int(np.floor(n_samples * ratio)) for ratio in ratios]
    for i in range(n_samples - sum(sizes)):
        sizes[i % 3] += 1
    order = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed)).numpy()
    train_idx, val_idx, test_idx = np.split(order, np.cumsum(sizes)[:2])
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "ratios": ratios,
        "n_samples": n_samples,
        "sample_id_source": source,
        "sample_ids": ids,
        "sample_ids_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "torch_randperm_random_split_v1",
        "torch_version": str(torch.__version__),
    }
    result = IndividualSplit(train_idx, val_idx, test_idx, metadata)
    _validate_split(result)
    return result


def split_from_indices(
    train_idx, val_idx, test_idx, n_samples: int, *, seed: int = 73,
    sample_ids: Sequence[str | int] | None = None,
) -> IndividualSplit:
    """将外部确定的个体索引封装成同一 manifest 格式，不重新随机划分。"""
    n_samples, ids, source, digest = _sample_identity(n_samples, sample_ids)
    validate_split_indices(train_idx, val_idx, test_idx, n_samples)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed 必须为整数")
    arrays = [np.asarray(values, dtype=np.int64).copy()
              for values in (train_idx, val_idx, test_idx)]
    result = IndividualSplit(*arrays, metadata={
        "schema_version": 1, "seed": int(seed),
        "ratios": [len(values) / n_samples for values in arrays],
        "n_samples": n_samples, "sample_id_source": source,
        "sample_ids": ids, "sample_ids_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "explicit_individual_indices_v1",
    })
    _validate_split(result)
    return result


def _validate_split(split: IndividualSplit) -> None:
    meta = split.metadata
    required = {
        "schema_version", "seed", "ratios", "n_samples", "sample_id_source",
        "sample_ids", "sample_ids_sha256", "created_at_utc", "generator",
    }
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("划分文件缺少必要元数据")
    if meta["schema_version"] != 1:
        raise ValueError("不支持的划分文件版本")
    _, ids, _, digest = _sample_identity(meta["n_samples"], meta["sample_ids"])
    if digest != meta["sample_ids_sha256"]:
        raise ValueError("划分文件中的 sample_ids 哈希不匹配")
    if meta["sample_id_source"] not in ("row_index", "provided"):
        raise ValueError("未知的 sample_id_source")
    if meta["sample_id_source"] == "row_index" and ids != list(range(meta["n_samples"])):
        raise ValueError("行索引 sample_id 的顺序不合法")
    _ratios(meta["ratios"])
    validate_split_indices(
        split.train_idx, split.val_idx, split.test_idx, meta["n_samples"]
    )


def save_split(split: IndividualSplit, path: str | Path) -> Path:
    """保存纯数值数组及 JSON 元数据，不覆盖已有文件。"""
    _validate_split(split)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("划分文件必须使用 .npz 后缀")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            test_idx=split.test_idx,
            metadata=np.asarray(json.dumps(split.metadata, ensure_ascii=False)),
        )
    return path


def validate_individual_split(
    split: IndividualSplit,
    *,
    n_samples: int | None = None,
    sample_ids: Sequence[str | int] | None = None,
) -> None:
    """同时验证内存 manifest 的元数据、互斥覆盖和当前个体身份顺序。"""
    _validate_split(split)
    if n_samples is not None or sample_ids is not None:
        count, _, source, digest = _sample_identity(n_samples, sample_ids)
        if count != split.metadata["n_samples"]:
            raise ValueError("当前样本数与划分文件不匹配")
        if (
            source != split.metadata["sample_id_source"]
            or digest != split.metadata["sample_ids_sha256"]
        ):
            raise ValueError("当前 sample_ids 类型、顺序或哈希与划分文件不匹配")


def load_split(
    path: str | Path, *, n_samples: int | None = None,
    sample_ids: Sequence[str | int] | None = None,
) -> IndividualSplit:
    """加载并校验；提供当前样本数/ID 时，严格检查身份与行顺序。"""
    with np.load(path, allow_pickle=False) as archive:
        if not {"train_idx", "val_idx", "test_idx", "metadata"}.issubset(archive.files):
            raise ValueError("划分文件缺少索引或元数据")
        result = IndividualSplit(
            archive["train_idx"].copy(), archive["val_idx"].copy(),
            archive["test_idx"].copy(), json.loads(str(archive["metadata"].item())),
        )
    validate_individual_split(result, n_samples=n_samples, sample_ids=sample_ids)
    return result


def default_split_path(
    n_samples: int | None = None,
    *,
    sample_ids: Sequence[str | int] | None = None,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 73,
    directory: str | Path = "splits",
) -> Path:
    """文件名包含 seed、比例和完整个体顺序哈希，供多个实验复用。"""
    count, _, source, digest = _sample_identity(n_samples, sample_ids)
    ratio_tag = "_".join(str(value) for value in _ratios(ratios))
    return Path(directory) / f"split_n{count}_seed{seed}_r{ratio_tag}_{source}_{digest}.npz"


def load_or_create_split(
    path: str | Path,
    n_samples: int | None = None,
    *,
    sample_ids: Sequence[str | int] | None = None,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 73,
) -> IndividualSplit:
    """已有文件为划分依据；不存在时生成。'auto' 使用可复用默认命名。"""
    # 先固定 ID 顺序，兼容一次性迭代器。
    if sample_ids is not None:
        sample_ids = list(sample_ids)
    count, _, _, _ = _sample_identity(n_samples, sample_ids)
    if str(path) == "auto":
        path = default_split_path(count, sample_ids=sample_ids, ratios=ratios, seed=seed)
    path = Path(path)
    if path.exists():
        return load_split(path, n_samples=count, sample_ids=sample_ids)
    result = generate_split(count, sample_ids=sample_ids, ratios=ratios, seed=seed)
    try:
        save_split(result, path)
    except FileExistsError:
        return load_split(path, n_samples=count, sample_ids=sample_ids)
    return result

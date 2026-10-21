"""ConvAE（带表型辅助任务的 MAE）原始数据与训练个体专用归一化。"""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from utils.Genotype import dosage_to_model_codes, model_code_tensor
from utils.Inference import transform_phenotypes
from utils.Split import IndividualSplit, generate_split, validate_individual_split


class ConvAEDataset(Dataset):
    """只加载原始表型、类别 SNP 和个体顺序，建立分块及独立 padding mask。

    原始 dosage -1/NaN 映射为缺失类别0，剂量0/1/2映射为类别1/2/3。
    结构性 padding 虽也编码0，但 padding_mask=True；真实缺失位置始终有效。
    __getitem__ 保持 (chunks, raw_phenotype) 两项接口，不执行统计拟合。
    """

    def __init__(self, snp_file: str, pheno_dir: str, phenotype_names: list[str],
                 chunk_size: int = 12000, overlap: int = 2000):
        if (isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1
                or isinstance(overlap, bool) or not isinstance(overlap, Integral)
                or not 0 <= overlap < chunk_size):
            raise ValueError("chunk_size 必须为正整数，overlap 必须为 [0, chunk_size) 内的整数")
        codes = dosage_to_model_codes(np.load(snp_file, allow_pickle=False))
        n_samples, self.original_length = codes.shape
        if not n_samples or not self.original_length:
            raise ValueError("SNP 个体数和原始有效长度必须非零")
        self.sample_ids = np.arange(n_samples, dtype=np.int64)
        self.sample_id_source = "row_index"  # 输入没有已验证的生物学个体 ID。
        self.chunk_size = self.chunks_size = int(chunk_size)  # 保留旧 chunks_size 属性。
        self.overlap = int(overlap)
        self.step_size = self.chunk_size - self.overlap
        extra = max(0, self.original_length - self.chunk_size)
        self.num_chunks = 1 + (extra + self.step_size - 1) // self.step_size
        self.target_length = self.chunk_size + (self.num_chunks - 1) * self.step_size
        self.padding_mask = torch.arange(self.target_length) >= self.original_length
        self.valid_position_mask = ~self.padding_mask
        self.chunk_padding_mask = self.padding_mask.unfold(0, self.chunk_size, self.step_size).contiguous()
        self.chunk_valid_mask = ~self.chunk_padding_mask
        self.chunk_starts = torch.arange(self.num_chunks) * self.step_size
        self.chunk_valid_lengths = self.chunk_valid_mask.sum(dim=1)
        codes = np.pad(codes, ((0, 0), (0, self.target_length - self.original_length)))
        self.snp = model_code_tensor(codes)
        self.pheno_dir = pheno_dir
        self.phenotype_names = phenotype_names
        self.phenotype = self.load_phenotypes()
        self.time_points = self.phenotype.shape[2]
        if len(self.phenotype) != n_samples:
            raise ValueError("SNP 与表型的个体行数不一致，不能复用个体 split")

    def load_phenotypes(self) -> np.ndarray:
        """保留 CSV 列顺序及原始数值；缺失表型明确报错，不静默填补。"""
        names = self.phenotype_names
        if not names or len(set(names)) != len(names):
            raise ValueError("phenotype_names 必须非空且互异")
        blocks = []
        self.time_labels = None
        for name in names:
            frame = pd.read_csv(Path(self.pheno_dir) / f"{name}.csv")
            labels = [str(column) for column in frame.columns]
            if self.time_labels is not None and labels != self.time_labels:
                raise ValueError("各表型 CSV 的时间列和顺序必须一致")
            self.time_labels = labels
            block = frame.to_numpy(dtype=np.float32)
            if not block.size or not np.isfinite(block).all():
                raise ValueError(f"表型 {name} 存在缺失/非有限值；phenotype_missing_strategy=error，禁止用全体或 test 统计填补")
            if blocks and block.shape != blocks[0].shape:
                raise ValueError("各表型的个体数和时间点数必须一致")
            blocks.append(block)
        return np.stack(blocks, axis=1)

    def __len__(self) -> int:
        return len(self.snp)

    def apply_global_zscore_norm(self):
        raise ValueError("禁止在原始 Dataset 上拟合表型统计；请向 get_mae_dataloaders 传入 IndividualSplit")

    def __getitem__(self, idx):
        chunks = self.snp[idx].unfold(0, self.chunk_size, self.step_size).contiguous()
        return chunks, self.phenotype[idx]

    def sample_with_metadata(self, idx: int) -> dict:
        """可复用的数据接口；第6步再接入模型注意力、池化和重建损失。"""
        chunks, phenotype = self[idx]
        return {"snp": chunks, "phenotype": phenotype, "sample_id": self.sample_ids[idx],
                "padding_mask": self.chunk_padding_mask,
                "valid_position_mask": self.chunk_valid_mask,
                "missing_genotype_mask": (chunks == 0) & self.chunk_valid_mask,
                "original_length": self.original_length}


class NormalizedPhenotypeSubset(Subset):
    """保留原始行号和 sample_id 顺序的独立归一化视图，不修改原始 Dataset。"""

    def __init__(self, dataset: Dataset, indices: np.ndarray, phenotype: torch.Tensor):
        super().__init__(dataset, indices.tolist())
        self.normalized_phenotype = phenotype
        self.sample_ids = np.asarray(getattr(dataset, "sample_ids", np.arange(len(dataset))))[indices]

    @property
    def chunk_padding_mask(self):
        return self.dataset.chunk_padding_mask

    def __getitem__(self, idx):
        chunks, _ = self.dataset[self.indices[idx]]
        return chunks, self.normalized_phenotype[idx]

    def __getitems__(self, indices):
        # Subset 自带的批量接口会跳过本类 __getitem__，必须显式分派到归一化视图。
        return [self[idx] for idx in indices]


@dataclass
class MAEDataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scalers: dict
    split: IndividualSplit


def get_mae_dataloaders(dataset: Dataset, indices: IndividualSplit, batch_size: int = 32,
                        num_workers: int = 8, seed: int = 73) -> MAEDataLoaders:
    """只接受已确定的三集合划分，只在 train phenotype 上拟合全局 Z-score。"""
    ids = np.asarray(getattr(dataset, "sample_ids", np.arange(len(dataset))))
    source = getattr(dataset, "sample_id_source", "row_index")
    validate_individual_split(indices, n_samples=len(dataset),
                              sample_ids=ids.tolist() if source == "provided" else None)
    if ids.tolist() != indices.metadata["sample_ids"]:
        raise ValueError("Dataset sample_id 顺序与 split 不一致")
    if not len(indices.train_idx) or not len(indices.val_idx):
        raise ValueError("ConvAE train 和 val 必须非空；test 可保留为空以兼容旧两集合实验")
    raw = np.asarray(dataset.phenotype, dtype=np.float32)
    names = dataset.phenotype_names
    if (raw.ndim != 3 or raw.shape[0] != len(dataset) or raw.shape[1] != len(names)
            or not raw.shape[2] or not np.isfinite(raw).all()):
        raise ValueError("原始表型必须为有限值 [个体, 表型, 时间]；缺失策略为 error")
    train = raw[indices.train_idx]
    scalers = {"norm_method": "global", "phenotype_names": list(names),
               "pheno_meta": {"scalers": {
                   name: StandardScaler().fit(train[:, p].reshape(-1, 1))
                   for p, name in enumerate(names)
               }}}
    loaders = []
    generator = torch.Generator().manual_seed(seed)
    for role, rows in zip(("train", "val", "test"),
                          (indices.train_idx, indices.val_idx, indices.test_idx)):
        norm = (transform_phenotypes(raw[rows], scalers) if len(rows)
                else torch.empty((0, len(names), raw.shape[2]), dtype=torch.float32))
        view = NormalizedPhenotypeSubset(dataset, rows, norm)
        loaders.append(DataLoader(view, batch_size=batch_size, shuffle=role == "train",
                                  num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                  generator=generator if role == "train" else None))
    return MAEDataLoaders(*loaders, scalers=scalers, split=indices)


def get_mae_dataloader(dataset: Dataset, batch_size: int = 32, train_ratio: float = 0.8,
                       val_ratio: float = 0.2, seed: int = 73, num_workers: int = 8,
                       indices: IndividualSplit | None = None) -> tuple[DataLoader, DataLoader]:
    """兼容旧的两项返回值；归一化改为只 fit train，test 留给新三集合接口。"""
    if indices is None:
        indices = generate_split(len(dataset), ratios=(train_ratio, val_ratio, max(0.0, 1 - train_ratio - val_ratio)), seed=seed)
    data = get_mae_dataloaders(dataset, indices, batch_size=batch_size,
                              num_workers=num_workers, seed=seed)
    return data.train_loader, data.val_loader

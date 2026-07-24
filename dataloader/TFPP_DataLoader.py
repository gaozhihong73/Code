from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import NORM_FUNC_REGISTRY, SCALER_REGISTRY


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def _apply_normalization(
    train_phenotype,
    val_phenotype,
    test_phenotype,
    phenotype_names: list[str],
    norm_method: str,
    scaler_type: str,
):
    """
    公共辅助函数：动态获取并执行归一化策略
    """
    if scaler_type not in SCALER_REGISTRY:
        raise ValueError(
            f"不支持的 scaler_type: '{scaler_type}'。支持的类型有: {list(SCALER_REGISTRY.keys())}"
        )

    if norm_method not in NORM_FUNC_REGISTRY:
        raise ValueError(
            f"不支持的 norm_method: '{norm_method}'。支持的方法有: {list(NORM_FUNC_REGISTRY.keys())}"
        )

    scaler_cls = SCALER_REGISTRY[scaler_type]
    apply_norm_func = NORM_FUNC_REGISTRY[norm_method]

    return apply_norm_func(
        train_phenotype, val_phenotype, test_phenotype, phenotype_names, scaler_cls
    )


def _build_dataloaders_and_scalers(
    train_snp,
    train_phe_norm,
    val_snp,
    val_phe_norm,
    test_snp,
    test_phe_norm,
    phenotype_names: list[str],
    norm_method: str,
    pheno_meta: dict,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    公共辅助函数：组装 TensorDataset、构建 DataLoader 并打包 scalers 信息
    """
    # 1. 封装合并数据集
    train_dataset = TensorDataset(to_tensor(train_snp), to_tensor(train_phe_norm))
    val_dataset = TensorDataset(to_tensor(val_snp), to_tensor(val_phe_norm))
    test_dataset = TensorDataset(to_tensor(test_snp), to_tensor(test_phe_norm))

    # 2. 构建 DataLoader
    pin_memory = device != "cpu"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # 3. 构建 Scalers 信息包
    scalers = {
        "phenotype_names": phenotype_names,
        "norm_method": norm_method,
        "pheno_meta": pheno_meta,
    }

    return train_loader, val_loader, test_loader, scalers


class TFPPDataset:
    """SNP和表型数据集类"""

    def __init__(self, snp_file: str, pheno_dir: str, phenotype_names: list[str]):
        """
        初始化数据集
        Args:
            snp_file: SNP数据文件路径(.npy格式)
            phenotype_names: 选择的表型名称列表,例如['RGBVI', 'ExB', 'WI', 'CIVE']
        """
        if not Path(snp_file).exists():
            raise FileNotFoundError(f"SNP文件不存在: {snp_file}")

        # 加载SNP数据
        self.snp = np.load(snp_file) + 1  # [n_sample, n_feature]

        self.pheno_dir = pheno_dir
        self.phenotype_names = phenotype_names

        # 加载表型数据(未归一化)
        self.phenotype = self.load_phenotypes()  # [n_sample, n_pheno, n_time_points]

    def load_phenotypes(self):
        """
        加载并合并所有表型数据(不进行归一化)
        phenotype_names 表型名称

        Returns:
            phenotype_data: numpy数组,形状为[n_sample, n_pheno, n_time_points]
        """
        phenotypes = []

        # 遍历每个表型文件
        for phenotype_name in self.phenotype_names:
            phenotype = pd.read_csv(Path(f"{self.pheno_dir}/{phenotype_name}.csv"))
            phenotype = phenotype.values[:, np.newaxis, :].astype(np.float32)
            phenotypes.append(phenotype)

        # 合并所有表型数据: [n_sample, n_pheno, n_time_points]
        pheno_data = np.concatenate(phenotypes, axis=1)
        return pheno_data

    def __len__(self):
        """返回数据集中的样本数量"""
        return self.snp.shape[0]

    def __getitem__(self, idx):
        """获取单个样本"""
        return self.snp[idx], self.phenotype[idx]


def get_dataloader(
    dataset,
    batch_size: int,
    norm_method: str,
    scaler_type: str,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 73,
    device: str = "cpu",
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Args:
        dataset: 数据集
        batch_size: 批次数量
        train_ratio: 训练集所占比例
        test_ratio: 测试集所占比例
        val_ratio: 验证集所占比例
        seed: 随机数种子
        device: 训练设备
        num_workers: 加载数据的线程数量
        norm_method: 归一化策略选择
            - 'global': 整个表型统一归一化
            - 'timepoint': 每个时间点独立归一化
            - 'residual_global': 减均值 -> 整个表型归一化
            - 'residual_timepoint': 减均值 -> 每个时间点独立归一化
        scaler_type: 缩放器类型 ('zscore', 'minmax', 'robust'). 默认 'zscore'.
    """
    # 1. 划分数据集
    snp_data = dataset.snp  # [n_sample, n_feature]
    phenotype_data = dataset.phenotype  # [n_sample, n_pheno, n_time_points]

    generator = torch.Generator().manual_seed(seed)
    # 此处的 random_split 只是为了获取纯净的互斥索引
    train_subset, val_subset, test_subset = random_split(
        range(
            len(dataset)
        ),  # 直接对长度进行 split，避免生成复杂的 Subset 套娃 # type: ignore
        lengths=[train_ratio, val_ratio, test_ratio],
        generator=generator,
    )

    # 将索引提取出来进行纯 NumPy 数组切片，保证数据统一干净
    train_snp, train_phenotype = (
        snp_data[train_subset.indices],
        phenotype_data[train_subset.indices],
    )
    val_snp, val_phenotype = (
        snp_data[val_subset.indices],
        phenotype_data[val_subset.indices],
    )
    test_snp, test_phenotype = (
        snp_data[test_subset.indices],
        phenotype_data[test_subset.indices],
    )

    # 2. 归一化表型
    train_phe_norm, val_phe_norm, test_phe_norm, pheno_meta = _apply_normalization(
        train_phenotype,
        val_phenotype,
        test_phenotype,
        dataset.phenotype_names,
        norm_method,
        scaler_type,
    )

    # 3. 调用公共 DataLoader 构建逻辑
    return _build_dataloaders_and_scalers(
        train_snp,
        train_phe_norm,
        val_snp,
        val_phe_norm,
        test_snp,
        test_phe_norm,
        dataset.phenotype_names,
        norm_method,
        pheno_meta,
        batch_size,
        num_workers,
        device,
    )


def get_dataloader_with_kfold(
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    dataset: TFPPDataset,
    batch_size: int,
    device: str,
    norm_method: str,
    scaler_type: str,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Args:
        train_indices: 验证集索引集
        val_indices：验证集索引集
        test_indices：索引集
        dataset: 数据集
        batch_size: 批次数量
        device: 训练设备
        num_workers: 加载数据的线程数量
        norm_method: 归一化策略选择
            - 'global': 整个表型统一归一化
            - 'timepoint': 每个时间点独立归一化
            - 'residual_global': 减均值 -> 整个表型归一化
            - 'residual_timepoint': 减均值 -> 每个时间点独立归一化
        scaler_type: 缩放器类型 ('zscore', 'minmax', 'robust'). 默认 'zscore'.
    """
    # ===================== 1. 划分数据集 =====================
    snp_data = dataset.snp
    phenotype_data = dataset.phenotype

    # 1. 提取各数据集的SNP和表型数据
    train_snp, train_phenotype = snp_data[train_indices], phenotype_data[train_indices]
    val_snp, val_phenotype = snp_data[val_indices], phenotype_data[val_indices]
    test_snp, test_phenotype = snp_data[test_indices], phenotype_data[test_indices]

    # 核心归一化策略函数库
    # ===================== 2. 归一化表型 =====================
    # 2. 调用公共归一化逻辑
    train_phe_norm, val_phe_norm, test_phe_norm, pheno_meta = _apply_normalization(
        train_phenotype,
        val_phenotype,
        test_phenotype,
        dataset.phenotype_names,
        norm_method,
        scaler_type,
    )

    # 3. 调用公共 DataLoader 构建逻辑
    return _build_dataloaders_and_scalers(
        train_snp,
        train_phe_norm,
        val_snp,
        val_phe_norm,
        test_snp,
        test_phe_norm,
        dataset.phenotype_names,
        norm_method,
        pheno_meta,
        batch_size,
        num_workers,
        device,
    )

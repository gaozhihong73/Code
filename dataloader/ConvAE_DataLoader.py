"""ConvAE 卷积自编码器的数据加载与预处理模块。

本模块负责 ConvAE 训练所需的数据管线：从 .npy 加载 SNP 基因型（+1 偏移处理缺失值）、
滑动窗口分块（chunk_size + overlap）、动态零填充对齐、表型全局 Z-score 归一化，
最终构建 PyTorch DataLoader。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from utils.Split import IndividualSplit, validate_split_indices
from utils.Genotype import dosage_to_model_codes, model_code_tensor


class ConvAEDataset(Dataset):
    """ConvAE 模型专用数据集，集成滑动窗口分块、归一化与动态填充处理。

    从 ``.npy`` 文件加载 SNP 数据（+1 偏移将缺失值 -1 映射为 0），
    通过滑动窗口将长序列切分为多个等长子块（``chunk_size`` 窗口，
    ``overlap`` 重叠），不足部分自动零填充。同时从 CSV 文件加载表型数据
    并执行全局 Z-score 归一化。

    Parameters
    ----------
    snp_file : str
        SNP 数据文件路径（``.npy`` 格式），加载后形状为 ``(n_sample, n_feature)``。
    pheno_dir : str
        表型 CSV 文件存放目录，每个表型对应一个 ``{name}.csv`` 文件。
    phenotype_names : list[str]
        需要加载的表型名称列表。
    chunk_size : int, optional
        每个滑动窗口的 SNP 位点数（窗口大小），默认 ``12000``。
    overlap : int, optional
        相邻窗口之间的重叠位点数，默认 ``2000``。
    """

    def __init__(
        self,
        snp_file: str,
        pheno_dir: str,
        phenotype_names: list[str],
        chunk_size: int = 12000,
        overlap: int = 2000,
    ):
        """初始化数据集，加载 SNP 数据并执行滑动窗口分块与动态填充。

        Parameters
        ----------
        snp_file : str
            SNP 数据文件路径（``.npy`` 格式）。
        pheno_dir : str
            表型 CSV 文件存放目录。
        phenotype_names : list[str]
            需要加载的表型名称列表。
        chunk_size : int, optional
            每个滑动窗口的 SNP 位点数（窗口大小），默认 ``12000``。
        overlap : int, optional
            相邻窗口之间的重叠位点数，默认 ``2000``。

        Raises
        ------
        FileNotFoundError
            当 ``snp_file`` 指向的文件不存在时抛出。
        """
        # 1. 检查文件是否存在
        if not Path(snp_file).exists():
            raise FileNotFoundError(f"SNP文件不存在: {snp_file}")

        # 2. 集中转换：剂量0/1/2 → 模型类别1/2/3；-1/NaN → 缺失类别0。
        self.snp = dosage_to_model_codes(np.load(snp_file))

        # 计算滑动步长
        self.chunks_size = chunk_size
        self.overlap = overlap
        self.step_size = chunk_size - overlap  # 例如: 12000 - 2000 = 10000

        # 1. 获取当前数据的实际特征长度
        num_samples, current_length = self.snp.shape

        # 2. 动态计算：为了让最后一个窗口也完整，序列总长度至少需要达到多少
        # 如果刚好能整除或少一部分，我们需要计算出“超越当前长度的下一个完美对齐长度”
        if current_length <= chunk_size:
            # 如果总长度还不如一个窗口大，直接补满一个窗口
            self.target_length = chunk_size
            self.num_chunks = 1
        else:
            # 减去第一个窗口后，计算剩余部分需要多少个步长(step_size)能覆盖，向上取整
            remaining_length = current_length - chunk_size
            # 向上取整计算：(A + B - 1) // B 是整数除法向上取整的快捷写方法
            num_extra_steps = (remaining_length + self.step_size - 1) // self.step_size

            # 最终的完美对齐总长度
            self.target_length = chunk_size + num_extra_steps * self.step_size
            # 动态生成的总块数
            self.num_chunks = 1 + num_extra_steps

        # 3. 动态边界填充 (Padding)
        # 此时 target_length 一定是 >= current_length 的
        if current_length < self.target_length:
            pad_width = self.target_length - current_length
            # 创建填充矩阵 (全 0 矩阵代表全为缺失值)
            padding = np.zeros((num_samples, pad_width), dtype=self.snp.dtype)
            self.snp = np.concatenate([self.snp, padding], axis=1, dtype=self.snp.dtype)
        self.snp = model_code_tensor(self.snp)  # 方便使用 unfold 函数

        print(
            f"数据加载完毕。SNP 原始长度: {current_length}，填充后形状: {self.snp.shape}，动态切分为 {self.num_chunks} 块"
        )

        # 4. 加载并归一化表型
        self.pheno_dir = pheno_dir
        self.phenotype_names = phenotype_names
        self.phenotype = self.load_phenotypes()  # 加载表型
        self.phenotype_norm = self.apply_global_zscore_norm()  # 归一化表型

    def load_phenotypes(self):
        """加载并合并所有表型数据（不进行归一化）。

        遍历 ``phenotype_names`` 中的每个表型名称，读取对应 CSV 文件，
        沿表型维度拼接为统一的三维数组。

        Returns
        -------
        pheno_data : np.ndarray
            合并后的表型数据，形状为 ``(n_sample, n_pheno, n_time_points)``。
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

    def apply_global_zscore_norm(self):
        """对表型数据执行逐表型的全局 Z-score 归一化。

        对每个表型独立计算其在所有样本和时间点上的全局均值与标准差，
        然后执行 ``(x - mean) / std`` 标准化。

        Returns
        -------
        pheno_norm : np.ndarray
            归一化后的表型数据，形状与 ``self.phenotype`` 相同，
            即 ``(n_sample, n_pheno, n_time_points)``。
        """
        # 深拷贝数据，防止直接修改传入的原始数组引用的数据
        pheno_norm = np.copy(self.phenotype)

        # 获取表型数量 (即第 1 维度的大小)
        num_phenotypes = self.phenotype.shape[1]

        # 遍历每一个表型进行独立的全局归一化
        for i in range(num_phenotypes):
            # 提取当前第 i 个表型的所有样本和所有时间点数据
            # current_pheno_data 的形状为 [N, T]
            current_pheno_data = self.phenotype[:, i, :]

            # 计算该表型在所有样本和时间点上的全局均值和标准差
            mean = np.mean(current_pheno_data)
            std = np.std(current_pheno_data)

            # 加上一个极小的值 (epsilon)，防止当所有数据相同时标准差为 0 导致除以 0 的错误 (NaN)
            if std == 0:
                std = 1e-8

            # 进行 Z-score 计算: (x - mean) / std
            # 利用 numpy 的广播机制，直接对 [N, T] 形状的数组进行批量计算，并赋值回原位置
            pheno_norm[:, i, :] = (current_pheno_data - mean) / std

        return pheno_norm

    def __len__(self):
        """返回数据集中的样本数量。"""
        return self.snp.shape[0]

    def __getitem__(self, idx):
        """获取单个样本，使用 ``unfold`` 将 SNP 序列切分为多个滑动窗口子块。"""
        sample_snps = self.snp[idx]
        # 用 unfold 动态切块
        x_chunks = sample_snps.unfold(
            dimension=0, size=self.chunks_size, step=self.step_size
        )
        return x_chunks.contiguous(), self.phenotype_norm[idx]


# ================================
# 划分训练/验证/测试集的工厂函数
# ================================
def get_mae_dataloader(
    dataset: ConvAEDataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 73,
    num_workers: int = 8,
    indices: IndividualSplit | None = None,
) -> tuple[DataLoader, DataLoader]:
    """按指定比例划分训练/验证集并构建 DataLoader。

    使用 ``random_split`` 对 ``ConvAEDataset`` 进行互斥划分，
    创建训练集（shuffle=True）和验证集（shuffle=False）的 DataLoader。

    Parameters
    ----------
    dataset : ConvAEDataset
        ConvAE 数据集实例，包含已分块和归一化的 SNP 与表型数据。
    batch_size : int, optional
        每个 DataLoader 的批次大小，默认 ``32``。
    train_ratio : float, optional
        训练集所占比例，默认 ``0.8``。
    val_ratio : float, optional
        验证集所占比例，默认 ``0.2``。
    seed : int, optional
        随机数种子，默认 ``73``。
    num_workers : int, optional
        DataLoader 的数据加载线程数，默认 ``8``。
    indices : IndividualSplit | None, optional
        共享三集合划分；仅返回其中 train/val loader，test 个体保留。

    Returns
    -------
    train_loader : DataLoader
        训练集 DataLoader（shuffle=True）。
    val_loader : DataLoader
        验证集 DataLoader（shuffle=False）。
    """

    generator = torch.Generator().manual_seed(seed)

    if indices is None:
        # 未指定共享划分时，保持原有两集合行为。
        train_subset, val_subset = random_split(
            dataset, lengths=[train_ratio, val_ratio], generator=generator
        )
    else:
        validate_split_indices(
            indices.train_idx, indices.val_idx, indices.test_idx, len(dataset)
        )
        train_subset = Subset(dataset, indices.train_idx.tolist())
        val_subset = Subset(dataset, indices.val_idx.tolist())
        # test_idx 保留给下游评估，不合并到 ConvAE 的训练集或验证集。

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),  # 4090 加速
    }

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader

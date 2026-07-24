from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ConvAEDataset(Dataset):
    """
    智能分块数据集类 (集成归一化与填充处理)
    """

    def __init__(
        self,
        snp_file: str,
        pheno_dir: str,
        phenotype_names: list[str],
        chunk_size: int = 12000,
        overlap: int = 2000,
    ):
        """
        参数:
        ------------
        snp_file: str
            SNP数据文件路径 (.npy格式)
        pheno_dir:
            表型数据存放目录
        phenotype_names: np.ndarray
            要使用的表型名称
        chunk_size: int
            每个分块的 SNP 位点数（窗口大小）
        overlap: int
            相邻分块之间的重叠位点数
        """
        # 1. 检查文件是否存在
        if not Path(snp_file).exists():
            raise FileNotFoundError(f"SNP文件不存在: {snp_file}")

        # 2. 从 .npy 文件直接加载，并处理缺失值 -1
        # np.load(snp_file) + 1 会将 -1 映射为 0，0->1, 1->2, 2->3[cite: 1]
        # 从而直接转化为标准的类别索引，供后续 Embedding 层使用
        self.snp = np.load(snp_file).astype(np.int8) + 1

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
        self.snp = torch.tensor(
            self.snp, dtype=torch.long
        )  # 转化成 Tensor ，方便使用 unfold 函数

        print(
            f"数据加载完毕。SNP 原始长度: {current_length}，填充后形状: {self.snp.shape}，动态切分为 {self.num_chunks} 块"
        )

        # 4. 加载并归一化表型
        self.pheno_dir = pheno_dir
        self.phenotype_names = phenotype_names
        self.phenotype = self.load_phenotypes()  # 加载表型
        self.phenotype_norm = self.apply_global_zscore_norm()  # 归一化表型

    def load_phenotypes(self):
        """
        加载并合并所有表型数据(进行归一化)
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

    def apply_global_zscore_norm(self):
        """
        对表型数据进行全局的 Z-score 归一化 (Global Z-score per Phenotype)。

        参数:
            phenotypes (np.ndarray): 形状为 [样本数量(N), 表型数量(P), 时间点数(T)] 的表型数据。

        返回:
            pheno_norm (np.ndarray): 归一化后具有相同形状 [N, P, T] 的表型数据。
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
        return self.snp.shape[0]

    def __getitem__(self, idx):
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
) -> tuple[DataLoader, DataLoader]:
    """
    参考 TFPP_DataLoader.py 逻辑，利用随机索引安全划分 DataLoader
    """

    generator = torch.Generator().manual_seed(seed)

    # 划分纯净的互斥子集
    train_subset, val_subset = random_split(
        dataset, lengths=[train_ratio, val_ratio], generator=generator
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),  # 4090 加速
    }

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader

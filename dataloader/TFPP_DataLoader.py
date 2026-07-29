"""TFPP 模型的数据加载与预处理模块。

本模块负责 TFPP 训练/推理所需的数据管线：从 .npy 加载 SNP 基因型、从 CSV 加载多维时序表型、
执行表型归一化（仅用训练集拟合 scaler，防止数据泄露）、按比例或 KFold 索引划分数据集，
最终构建 PyTorch DataLoader。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils import NORM_FUNC_REGISTRY, SCALER_REGISTRY


def to_tensor(x):
    """将输入数组转换为 float32 类型的 PyTorch 张量。"""
    return torch.tensor(x, dtype=torch.float32)


def _apply_normalization(
    train_phenotype,
    val_phenotype,
    test_phenotype,
    phenotype_names: list[str],
    norm_method: str,
    scaler_type: str,
):
    """动态获取并执行归一化策略。

    根据指定的 ``scaler_type`` 和 ``norm_method``，从注册表中获取对应的
    缩放器类与归一化函数，对训练/验证/测试三组表型数据执行拟合并变换。

    Parameters
    ----------
    train_phenotype : np.ndarray
        训练集表型数据，形状为 ``(n_sample_train, n_pheno, n_time)``。
    val_phenotype : np.ndarray
        验证集表型数据，形状为 ``(n_sample_val, n_pheno, n_time)``。
    test_phenotype : np.ndarray
        测试集表型数据，形状为 ``(n_sample_test, n_pheno, n_time)``。
    phenotype_names : list[str]
        表型名称列表，用于在 scaler 元信息中标识各表型。
    norm_method : str
        归一化策略名称（如 ``'global'``、``'timepoint'``、``'residual_global'`` 等），
        必须在 ``NORM_FUNC_REGISTRY`` 中注册。
    scaler_type : str
        缩放器类型（如 ``'zscore'``、``'minmax'``、``'robust'``），
        必须在 ``SCALER_REGISTRY`` 中注册。

    Returns
    -------
    train_phe_norm : np.ndarray
        归一化后的训练集表型。
    val_phe_norm : np.ndarray
        归一化后的验证集表型。
    test_phe_norm : np.ndarray
        归一化后的测试集表型。
    pheno_meta : dict
        包含 scaler 实例、表型名称等元信息的字典，用于推理阶段逆变换。

    Raises
    ------
    ValueError
        当 ``scaler_type`` 或 ``norm_method`` 不在对应注册表中时抛出。
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
    """组装 TensorDataset、构建 DataLoader 并打包 scaler 元信息。

    将训练/验证/测试三组的 SNP 与归一化表型数据封装为 ``TensorDataset``，
    创建对应的 ``DataLoader``，并将 scaler 相关信息打包返回，供推理阶段使用。

    Parameters
    ----------
    train_snp : np.ndarray
        训练集 SNP 数据，形状为 ``(n_sample_train, n_feature)``。
    train_phe_norm : np.ndarray
        归一化后的训练集表型数据，形状为 ``(n_sample_train, n_pheno, n_time)``。
    val_snp : np.ndarray
        验证集 SNP 数据，形状为 ``(n_sample_val, n_feature)``。
    val_phe_norm : np.ndarray
        归一化后的验证集表型数据，形状为 ``(n_sample_val, n_pheno, n_time)``。
    test_snp : np.ndarray
        测试集 SNP 数据，形状为 ``(n_sample_test, n_feature)``。
    test_phe_norm : np.ndarray
        归一化后的测试集表型数据，形状为 ``(n_sample_test, n_pheno, n_time)``。
    phenotype_names : list[str]
        表型名称列表。
    norm_method : str
        归一化策略名称。
    pheno_meta : dict
        表型元信息（如 scaler 实例等），由 ``_apply_normalization`` 返回。
    batch_size : int
        每个 DataLoader 的批次大小。
    num_workers : int
        DataLoader 的数据加载线程数。
    device : str
        训练设备标识（如 ``'cuda'`` 或 ``'cpu'``），用于决定是否启用 ``pin_memory``。

    Returns
    -------
    train_loader : DataLoader
        训练集 DataLoader（shuffle=True）。
    val_loader : DataLoader
        验证集 DataLoader（shuffle=False）。
    test_loader : DataLoader
        测试集 DataLoader（shuffle=False）。
    scalers : dict
        包含 ``phenotype_names``、``norm_method``、``pheno_meta`` 的字典，
        用于推理阶段的逆归一化。
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
        "snp_dim": train_snp.shape[1],
    }

    return train_loader, val_loader, test_loader, scalers


class TFPPDataset:
    """TFPP 模型专用数据集，管理 SNP 基因型与多维时序表型的加载与索引。

    从 ``.npy`` 文件加载 SNP 数据（+1 偏移将缺失值 -1 映射为 0），
    从 ``pheno_dir`` 目录中按表型名称逐个读取 CSV 文件并沿表型维度拼接，
    最终提供 ``(snp, phenotype)`` 样本对供 DataLoader 使用。

    Parameters
    ----------
    snp_file : str
        SNP 数据文件路径（``.npy`` 格式），加载后形状为 ``(n_sample, n_feature)``。
    pheno_dir : str
        表型 CSV 文件存放目录，每个表型对应一个 ``{name}.csv`` 文件，
        文件形状为 ``(n_sample, n_time_points)``。
    phenotype_names : list[str]
        需要加载的表型名称列表，如 ``['RGBVI', 'ExB', 'WI', 'CIVE']``。
    """

    def __init__(self, snp_file: str, pheno_dir: str, phenotype_names: list[str]):
        """初始化数据集，加载 SNP 与表型数据。

        Parameters
        ----------
        snp_file : str
            SNP 数据文件路径（``.npy`` 格式）。
        pheno_dir : str
            表型 CSV 文件存放目录。
        phenotype_names : list[str]
            需要加载的表型名称列表，如 ``['RGBVI', 'ExB', 'WI', 'CIVE']``。

        Raises
        ------
        FileNotFoundError
            当 ``snp_file`` 指向的文件不存在时抛出。
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
    """根据比例划分数据集并构建训练/验证/测试 DataLoader。

    使用 ``random_split`` 对数据集索引进行互斥划分，对表型数据
    执行归一化（仅用训练集拟合 scaler），最终构建三组 DataLoader。

    注意: SNP 特征筛选由上层 Trainer 在数据划分前统一执行，本函数不再负责筛选。

    Parameters
    ----------
    dataset : TFPPDataset
        TFPP 数据集实例，包含 SNP 和表型数据。
    batch_size : int
        每个 DataLoader 的批次大小。
    norm_method : str
        归一化策略选择：

        - ``'global'``: 整个表型统一归一化
        - ``'timepoint'``: 每个时间点独立归一化
        - ``'residual_global'``: 减均值后整个表型归一化

    scaler_type : str
        缩放器类型（``'zscore'``、``'minmax'``、``'robust'``）。
    train_ratio : float, optional
        训练集所占比例，默认 ``0.8``。
    test_ratio : float, optional
        测试集所占比例，默认 ``0.1``。
    val_ratio : float, optional
        验证集所占比例，默认 ``0.1``。
    seed : int, optional
        随机数种子，默认 ``73``。
    device : str, optional
        训练设备标识，默认 ``'cpu'``。
    num_workers : int, optional
        DataLoader 的数据加载线程数，默认 ``8``。

    Returns
    -------
    train_loader : DataLoader
        训练集 DataLoader（shuffle=True）。
    val_loader : DataLoader
        验证集 DataLoader（shuffle=False）。
    test_loader : DataLoader
        测试集 DataLoader（shuffle=False）。
    scalers : dict
        包含 scaler 元信息的字典，用于推理阶段逆归一化。
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
    """基于 KFold 预划分的索引构建训练/验证/测试 DataLoader。

    接收外部 KFold 划分好的索引列表，提取对应的 SNP 和表型数据，
    执行归一化（仅用训练集拟合 scaler）后构建三组 DataLoader。

    注意: SNP 特征筛选由上层 Trainer 在数据划分前统一执行，本函数不再负责筛选。

    Parameters
    ----------
    train_indices : list[int]
        训练集样本索引列表。
    val_indices : list[int]
        验证集样本索引列表。
    test_indices : list[int]
        测试集样本索引列表。
    dataset : TFPPDataset
        TFPP 数据集实例，包含 SNP 和表型数据。
    batch_size : int
        每个 DataLoader 的批次大小。
    device : str
        训练设备标识（如 ``'cuda'`` 或 ``'cpu'``）。
    norm_method : str
        归一化策略选择：

        - ``'global'``: 整个表型统一归一化
        - ``'timepoint'``: 每个时间点独立归一化
        - ``'residual_global'``: 减均值后整个表型归一化

    scaler_type : str
        缩放器类型（``'zscore'``、``'minmax'``、``'robust'``）。
    num_workers : int, optional
        DataLoader 的数据加载线程数，默认 ``4``。

    Returns
    -------
    train_loader : DataLoader
        训练集 DataLoader（shuffle=True）。
    val_loader : DataLoader
        验证集 DataLoader（shuffle=False）。
    test_loader : DataLoader
        测试集 DataLoader（shuffle=False）。
    scalers : dict
        包含 scaler 元信息的字典，用于推理阶段逆归一化。
    """
    # 1. 提取各数据集的SNP和表型数据
    snp_data = dataset.snp
    phenotype_data = dataset.phenotype

    train_snp, train_phenotype = snp_data[train_indices], phenotype_data[train_indices]
    val_snp, val_phenotype = snp_data[val_indices], phenotype_data[val_indices]
    test_snp, test_phenotype = snp_data[test_indices], phenotype_data[test_indices]

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

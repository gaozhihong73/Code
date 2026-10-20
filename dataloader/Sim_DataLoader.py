"""模拟数据专用 DataLoader

与 TFPP_DataLoader 的关键差异:
- 数据已经是 Train/Val/Test 预划分,无需 random_split / KFold
- 数据格式为 .csv (DynaGS_S*_P*_H*_N*_Rep*_{Train|Val|Test}_{Genotype|Phenotype|TrueGV}.csv)
- V1..V20 视为同一表型 sim_pheno 的 20 个时间点
- 复用 TFPP_DataLoader 中的 _apply_normalization 与 _build_dataloaders_and_scalers 公共逻辑
- 使用 utils.Screening 中的共享筛选逻辑 (DC-SIS + GWAS), 支持磁盘缓存
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils import compute_cache_key, screen_snps
from config.TFPP_Config import TFPPConfig
from utils.Genotype import dosage_to_model_codes, prepare_screening_dosage, validate_dosage

from .TFPP_DataLoader import _apply_normalization, _build_dataloaders_and_scalers


def _load_snp_csv(csv_path: str) -> np.ndarray:
    """读取 Genotype csv, 丢弃首列样本ID"""
    df = pd.read_csv(csv_path)
    snp = df.drop(columns=[df.columns[0]]).values
    return validate_dosage(snp, allow_missing=True).astype(np.float32)


def _load_pheno_csv(csv_path: str) -> np.ndarray:
    """读取 Phenotype csv, 丢弃首列样本ID, reshape 为 [N, 1, T] (单表型多时间点)"""
    df = pd.read_csv(csv_path)
    pheno = df.drop(columns=[df.columns[0]]).values.astype(np.float32)
    return pheno[:, np.newaxis, :]  # [N, 1, T]


class SimTFPPDataset:
    """模拟数据专用数据集（预划分 Train/Val/Test）。

    与 :class:`TFPPDataset` 不同，本类直接加载已划分好的三组 SNP 与表型 CSV 文件，
    无需 ``random_split`` 或 KFold 划分。V1..V20 视为同一表型 ``sim_pheno``
    的 20 个时间点。

    支持双重特征筛选（仅在训练集计算，列索引取并集应用到 val/test，避免数据泄露）：

    - **DC-SIS 筛选** (``screen_k``): 基于距离相关系数筛 top-k 个 SNP。
    - **GWAS 筛选** (``gwas_k``): 基于 OLS + PCA 协变量校正的 min-P 筛 top-k 个 SNP。
    - 最终取两者并集作为模型输入特征。

    Parameters
    ----------
    train_snp_csv : str
        训练集基因型 CSV 文件路径。
    train_pheno_csv : str
        训练集表型 CSV 文件路径。
    val_snp_csv : str
        验证集基因型 CSV 文件路径。
    val_pheno_csv : str
        验证集表型 CSV 文件路径。
    test_snp_csv : str
        测试集基因型 CSV 文件路径。
    test_pheno_csv : str
        测试集表型 CSV 文件路径。
    phenotype_names : list[str]
        表型名称列表，默认 ``['sim_pheno']``。
    screen_k : int, optional
        DC-SIS 筛选保留的 SNP 数目，设为 ``0`` 则跳过，默认 ``100``。
    gwas_k : int, optional
        GWAS 筛选保留的 SNP 数目，设为 ``0`` 则跳过，默认 ``100``。
    """

    def __init__(
        self,
        train_snp_csv: str,
        train_pheno_csv: str,
        val_snp_csv: str,
        val_pheno_csv: str,
        test_snp_csv: str,
        test_pheno_csv: str,
        phenotype_names: list[str],
        screen_k: int = 0,
        gwas_k: int = 0,
        screen_multivariate: bool = TFPPConfig.model_fields["screen_multivariate"].default,
        screen_missing_strategy: str = TFPPConfig.model_fields["screen_missing_strategy"].default,
    ):
        """初始化模拟数据集，加载预划分的 SNP/表型 CSV 并执行双重特征筛选。

        Parameters
        ----------
        train_snp_csv : str
            训练集基因型 CSV 文件路径。
        train_pheno_csv : str
            训练集表型 CSV 文件路径。
        val_snp_csv : str
            验证集基因型 CSV 文件路径。
        val_pheno_csv : str
            验证集表型 CSV 文件路径。
        test_snp_csv : str
            测试集基因型 CSV 文件路径。
        test_pheno_csv : str
            测试集表型 CSV 文件路径。
        phenotype_names : list[str]
            表型名称列表，若为 ``None`` 则默认 ``['sim_pheno']``。
        screen_k : int, optional
            DC-SIS 筛选保留的 SNP 数目，默认 ``0``，表示不筛选。
        gwas_k : int, optional
            GWAS 筛选保留的 SNP 数目，默认 ``0``，表示不筛选。

        Raises
        ------
        FileNotFoundError
            当任一 CSV 文件路径不存在时抛出。
        ValueError
            当各 split 的 SNP 维度或时间点数不一致时抛出。
        """
        if phenotype_names is None:
            phenotype_names = ["sim_pheno"]

        for p in [
            train_snp_csv,
            train_pheno_csv,
            val_snp_csv,
            val_pheno_csv,
            test_snp_csv,
            test_pheno_csv,
        ]:
            if not Path(p).exists():
                raise FileNotFoundError(f"模拟数据文件不存在: {p}")

        self.phenotype_names = phenotype_names

        # 读取各 split 的 snp 与 pheno
        self.snp_train = _load_snp_csv(train_snp_csv)
        self.snp_val = _load_snp_csv(val_snp_csv)
        self.snp_test = _load_snp_csv(test_snp_csv)

        self.pheno_train = _load_pheno_csv(train_pheno_csv)
        self.pheno_val = _load_pheno_csv(val_pheno_csv)
        self.pheno_test = _load_pheno_csv(test_pheno_csv)

        # 校验 snp_dim 与表型维度一致 (筛选前)
        if not (
            self.snp_train.shape[1] == self.snp_val.shape[1] == self.snp_test.shape[1]
        ):
            raise ValueError(
                f"各 split snp_dim 不一致: "
                f"{self.snp_train.shape[1]} / {self.snp_val.shape[1]} / {self.snp_test.shape[1]}"
            )

        if not (
            self.pheno_train.shape[2]
            == self.pheno_val.shape[2]
            == self.pheno_test.shape[2]
        ):
            raise ValueError(
                f"各 split 时间点数不一致: "
                f"{self.pheno_train.shape[2]} / {self.pheno_val.shape[2]} / {self.pheno_test.shape[2]}"
            )

        # ---- 特征筛选 (仅用训练集计算, 应用到 val/test, 避免数据泄露) ----
        cache_key = compute_cache_key(
            snp_file=train_snp_csv,
            pheno_source=train_pheno_csv,
            phenotype_names=phenotype_names,
            screen_k=screen_k,
            gwas_k=gwas_k,
            seed=0,
            split_info="pre_split",
            multivariate=screen_multivariate,
            missing_strategy=screen_missing_strategy,
        )
        print(
            f"筛选配置: screen_multivariate={screen_multivariate}, "
            f"screen_missing_strategy={screen_missing_strategy}"
        )
        screened_idx = None
        if screen_k > 0 or gwas_k > 0:
            screened_idx = screen_snps(
                prepare_screening_dosage(
                    self.snp_train, missing_strategy=screen_missing_strategy
                ),
                self.pheno_train,
                screen_k=screen_k,
                gwas_k=gwas_k,
                multivariate=screen_multivariate,
                cache_key=cache_key,
            )

        if screened_idx is not None:
            self.snp_train = self.snp_train[:, screened_idx].copy()
            self.snp_val = self.snp_val[:, screened_idx].copy()
            self.snp_test = self.snp_test[:, screened_idx].copy()
            self._screened_indices = screened_idx
            print(
                f"  最终特征维度: train={self.snp_train.shape}  "
                f"val={self.snp_val.shape}  test={self.snp_test.shape}"
            )
        else:
            self._screened_indices = None

    def __len__(self):
        """返回训练集中的样本数量。"""
        return self.snp_train.shape[0]


def get_sim_dataloader(
    dataset: SimTFPPDataset,
    batch_size: int,
    norm_method: str,
    scaler_type: str,
    num_workers: int = 4,
    device: str = "cpu",
):
    """基于预划分的 train/val/test 数据构建 DataLoader。

    直接使用 ``SimTFPPDataset`` 中已划分好的三组 SNP 与表型数组，
    对表型执行归一化（仅用训练集拟合 scaler）后构建三组 DataLoader。

    Parameters
    ----------
    dataset : SimTFPPDataset
        模拟数据集实例，包含预划分的 SNP 与表型数据。
    batch_size : int
        每个 DataLoader 的批次大小。
    norm_method : str
        归一化策略（``'global'`` / ``'timepoint'`` / ``'residual_global'``）。
    scaler_type : str
        缩放器类型（``'zscore'`` / ``'minmax'`` / ``'robust'``）。
    num_workers : int, optional
        DataLoader 的工作进程数，默认 ``4``。
    device : str, optional
        训练设备标识，默认 ``'cpu'``。

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
    # 1. 直接使用 dataset 中已划分好的三组数组
    # 与真实管线共用编码边界：筛选保持剂量，送入模型前显式转为四类别码。
    train_snp, train_phenotype = dosage_to_model_codes(dataset.snp_train), dataset.pheno_train
    val_snp, val_phenotype = dosage_to_model_codes(dataset.snp_val), dataset.pheno_val
    test_snp, test_phenotype = dosage_to_model_codes(dataset.snp_test), dataset.pheno_test

    # 2. 归一化 (仅用 train 拟合, 应用到 val/test)
    train_phe_norm, val_phe_norm, test_phe_norm, pheno_meta = _apply_normalization(
        train_phenotype,
        val_phenotype,
        test_phenotype,
        dataset.phenotype_names,
        norm_method,
        scaler_type,
    )

    # 3. 复用公共构建逻辑
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

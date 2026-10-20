"""模拟数据实验配置 (SimTFPPConfig)。

本模块定义 SimTFPPConfig (Pydantic BaseModel)，管理模拟数据 TFPP 训练的全部超参数。

与 TFPPConfig 的关键差异:
- 数据由 6 个 csv 路径指定 (Train/Val/Test × Genotype/Phenotype),而非 snp_file + pheno_dir
- 数据已预划分, 不使用 use_kfold / train_ratio / val_ratio / test_ratio / snp_file / pheno_dir
- time_coordinates 默认 [1..20], phenotype_names 默认 ['sim_pheno']
- 新增 exp_group / rep_name 字段用于多组实验汇总
- 输出目录改为 outputs/sim_tfpp/{exp_name}
"""

from datetime import datetime

import torch
from pydantic import BaseModel, Field

from .TFPP_Config import TFPPConfig


class SimTFPPConfig(BaseModel):
    """模拟数据 TFPP 表型预测模型的训练配置，支持预划分的基因型/表型 CSV 输入。

    与 TFPPConfig 相比新增损失函数类型选择 (loss_type)、L1 稀疏正则 (l1_wide_weight)、
    特征筛选 (screen_k / gwas_k) 等字段。degree 默认为 0 (纯加性遗传架构,
    仅保留勒让德多项式常数项)。seed 默认 73, 训练与推理必须保持一致。
    """

    # ============ 实验标识 ============
    exp_name: str = Field(
        default_factory=lambda: (
            f"sim_exp_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
        ),
        description="实验名称, 决定输出子目录 outputs/sim_tfpp/{exp_name}",
    )
    exp_group: str = Field(
        default="", description="实验组标识 (如 S1/P10000_N1600_H0.8), 用于多 Rep 汇总"
    )
    rep_name: str = Field(
        default="", description="重复名称 (如 Rep01), 用于多 Rep 汇总"
    )

    # ============ 数据 csv 路径 (6 个, 模拟数据已预划分) ============
    train_snp_csv: str = ""
    train_pheno_csv: str = ""
    val_snp_csv: str = ""
    val_pheno_csv: str = ""
    test_snp_csv: str = ""
    test_pheno_csv: str = ""

    # ============ 训练超参数 ============
    epochs: int = Field(default=200, ge=1, le=500)
    lr: float = Field(default=0.0001, gt=0)
    weight_decay: float = 0.0001
    min_lr: float = 1e-6
    batch_size: int = Field(default=32, ge=1, multiple_of=2)
    patience: int = 20
    seed: int = 73
    dropout: float = 0.0
    grad_clip_norm: float = 1.0
    is_save_training_info: bool = True
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ============ 学习率调度器参数 ============
    use_warmup: bool = True
    warmup_epochs: int = 5  # 短 warmup, 让 lr 快速到位
    warmup_start_factor: float = 0.1
    warmup_end_factor: float = 1.0

    # ============ 模型参数 ============
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 1
    d_ff: int = 128
    degree: int = 5  # Legendre 多项式阶数
    ablation_mode: str = Field(default="all", pattern="^(all|wide|deep|add)$")
    time_coordinates: list[float] = Field(
        default_factory=lambda: list(range(1, 21)),  # [1..20]
        min_length=1,
        description="模拟数据时间点, 默认 1..20",
    )
    phenotype_names: list[str] = Field(
        default_factory=lambda: ["sim_pheno"], min_length=1
    )
    num_embeddings: int = 4  # 模型类别0=缺失，1/2/3=剂量0/1/2

    # ============ 损失函数 ============
    loss_type: str = Field(
        default="mse",
        pattern="^(mse|mse_pcc|mse_offset)$",
        description="mse=纯MSE, mse_pcc=MSE+PCC, mse_offset=MSE+个体偏移MSE",
    )
    offset_weight: float = 0.5
    pcc_weight: float = 0.0

    # ============ L1 稀疏正则化 (宽通路) ============
    l1_wide_weight: float = 1e-5  # L1 惩罚系数, 0=关闭

    # ============ 特征筛选 ============
    # DC-SIS 筛选: 距离相关系数选 top-k SNP, 0=关闭
    screen_k: int = 100  # DC-SIS 筛选 SNP 数 (默认 100, 与 gwas_k 各 100)
    # GWAS 筛选: OLS+PCA 协变量校正 + Mean/AUC/Slope 三性状 min-P, 0=关闭
    gwas_k: int = 100  # GWAS 筛选 SNP 数 (默认 100, 与 screen_k 并集)
    screen_multivariate: bool = TFPPConfig.model_fields["screen_multivariate"].default
    screen_missing_strategy: str = Field(
        default=TFPPConfig.model_fields["screen_missing_strategy"].default,
        pattern="^(error|mode)$",
    )

    # ============ DataLoader / 其他 ============
    num_workers: int = 8

    # ============ 归一化 ============
    norm_method: str = Field(
        default="global", pattern="^(global|timepoint|residual_global)$"
    )
    scaler_type: str = Field(default="zscore", pattern="^(zscore|minmax|robust)$")

    # ============ 输出目录属性 ============
    @property
    def output_dir(self) -> str:
        """训练输出文件的保存总目录"""
        return f"outputs/sim_tfpp/{self.exp_name}"

    @property
    def best_model_save_dir(self) -> str:
        """训练好模型的保存目录"""
        return f"{self.output_dir}/models"

    @property
    def training_history_save_dir(self) -> str:
        """训练历史文件的保存目录"""
        return f"{self.output_dir}/training_history"

    @property
    def scaler_save_dir(self) -> str:
        """表型的归一化器保存目录"""
        return f"{self.output_dir}/scalers"

    @property
    def log_file(self) -> str:
        """日志文件路径"""
        log_name = self.exp_name.split("/")[-1]
        return f"{self.output_dir}/{log_name}.log"

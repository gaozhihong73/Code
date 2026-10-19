"""TFPP 主模型的训练与推理配置定义。

本模块定义 TFPPConfig (Pydantic BaseModel)，集中管理 TFPP 模型训练/推理所需的
全部超参数、数据路径、模型结构参数和输出目录。
"""

from datetime import datetime

import torch
from pydantic import BaseModel, Field, field_validator


def _default_exp_name() -> str:
    return f"exp_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"


class TFPPConfig(BaseModel):
    """TFPP 表型预测模型的训练与推理配置。

    所有字段均通过 Pydantic Field 进行类型校验与约束检查。
    注意: batch_size 必须为 2 的倍数 (K-fold 分组需要);
    ablation_mode 决定模型是否实例化 Wide/Deep 分支 (联动 model/TFPP.py);
    seed 默认 73, 训练与推理必须保持一致以复现数据划分。
    """

    exp_name: str = Field(
        default_factory=_default_exp_name,
        description="实验名称，若不指定则自动按时间戳生成 (格式: exp_YYYYMMDD_HHMMSS)",
    )

    @field_validator("exp_name", mode="before")
    @classmethod
    def _resolve_exp_name(cls, v):
        return _default_exp_name() if v is None else v

    # 文件目录
    pheno_dir: str = "dataset/maize/cleaned_pheno"
    snp_file: str = "dataset/maize/snp/snp_50000.npy"

    # 训练参数
    epochs: int = Field(default=200, ge=1, le=500)
    lr: float = Field(default=0.0001, gt=0)
    weight_decay: float = 0.0001
    min_lr: float = 1e-7
    batch_size: int = Field(default=32, ge=1, multiple_of=2)
    patience: int = 20
    seed: int = 73
    dropout: float = 0.25
    grad_clip_norm: float = 1.0
    is_save_training_info: bool = True  # 是否保存训练好的模型到本地
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )  # 训练设备

    # 学习率调度器参数
    use_warmup: bool = True  # 使用warmup
    warmup_epochs: int = 5  # warmup轮数
    warmup_start_factor: float = 0.1  # 从 base_lr * 0.1 开始
    warmup_end_factor: float = 1.0  # 持续增长到 base_lr * 1.0

    # 模型参数
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 1  # 编码器层数
    num_embeddings: int = 4  # 嵌入层维度
    d_ff: int = 128  # 前馈网络隐藏维度
    degree: int = Field(
        default=5,
        ge=0,
        description="Legendre 多项式阶数, 0=退化为纯加性(仅常数项)",
    )
    ablation_mode: str = Field(
        default="all",
        pattern="^(all|wide|deep|add)$",
        description="模型消融模式: all=完整模型, wide=仅宽通路, deep=仅深通路, add=去掉门控直接相加",
    )
    time_coordinates: list[float] = Field(
        default_factory=lambda: [
            10,
            12,
            14,
            15,
            17,
            19,
            24,
            26,
            37,
            44,
            51,
            65,
            80,
            101,
        ],
        min_length=1,
        description="选择的表型时间点",
    )
    phenotype_names: list[str] = Field(
        default_factory=lambda: ["RGBVI"],
        min_length=1,
        description="选择的表型特征列表",
    )

    # K折交叉验证参数
    use_kfold: bool = False  # 是否使用K折交叉验证
    n_folds: int = 10  # K折数量

    # 数据集参数
    train_ratio: float = 0.8
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    num_workers: int = 8

    # SNP 特征筛选参数 (0=不筛选)
    screen_k: int = Field(
        default=10000,
        ge=0,
        description="DC-SIS 筛选保留的 SNP 数目, 0=跳过",
    )
    gwas_k: int = Field(
        default=10000,
        ge=0,
        description="GWAS 筛选保留的 SNP 数目, 0=跳过",
    )
    screen_multivariate: bool = Field(
        default=True,
        description="DC-SIS 多元模式: True=多表型拼接联合筛选, False=逐表型筛选取并集",
    )

    # 损失函数
    loss_type: str = Field(
        default="mse",
        pattern="^(mse|mse_pcc|mse_offset)$",
        description="mse=纯MSE, mse_pcc=MSE+PCC, mse_offset=MSE+个体偏移MSE",
    )
    offset_weight: float = 0.5
    pcc_weight: float = 0.2

    # L1 稀疏正则化 (宽通路)
    l1_wide_weight: float = 1e-5  # L1 惩罚系数, 0=关闭

    # 规范字符串输入
    norm_method: str = Field(
        default="residual_global",
        pattern="^(global|timepoint|residual_global)$",
        description="仅支持 global, timepoint, residual_global",
    )
    scaler_type: str = Field(
        default="zscore",
        pattern="^(zscore|minmax|robust)$",
        description="仅支持 zscore, minmax, robust",
    )

    @property
    def output_dir(self) -> str:
        """训练输出文件的保存总目录"""
        return f"outputs/tfpp/{self.exp_name}"

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
        """日志文件名"""
        log_name = self.exp_name.split("/")[-1]
        if self.use_kfold:
            return f"{self.output_dir}/{log_name}_kfold.log"
        else:
            return f"{self.output_dir}/{log_name}.log"

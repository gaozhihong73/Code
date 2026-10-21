"""ConvAE（带表型辅助任务的 masked autoencoder）的训练配置定义。

本模块定义 ConvAEConfig (Pydantic BaseModel)，管理现有 masked autoencoder 训练所需的
超参数、数据路径和输出目录。ConvAE 用于将高维 SNP 数据压缩为低维潜在表示，
当前压缩结果仅支持独立 Ridge 下游基线；continuous_feature TFPP 尚未实现。
"""

from datetime import datetime
import math
from typing import Literal

import torch
from pydantic import BaseModel, Field, computed_field, model_validator


class ConvAEConfig(BaseModel):
    """ConvAE 数据与训练配置；类名沿用，辅助头维度与掩码策略显式记录。

    损失函数由 recon_weight (重构损失) 和 pheno_weight (表型辅助损失) 加权组合，
    两者之和必须为 1.0；关闭辅助时显式使用 recon_weight=1、pheno_weight=0。
    注意: batch_size 必须为 2 的倍数; seed 默认 73。
    """

    use_pheno_help: bool = True  # 是否使用表型辅助压缩
    exp_name: str = Field(default_factory=lambda: f"exp_{datetime.now():%Y%m%d_%H%M%S}")
    output_root: str = "outputs/convae"
    split_tag: str | None = None  # trainer 根据完整个体划分内容确定，不依赖文件名。

    # 文件目录
    pheno_dir: str = "dataset/maize/cleaned_pheno"
    snp_file: str = "dataset/maize/snp/snp_2860770.npy"

    # 训练参数
    epochs: int = Field(default=100, ge=1, le=500)
    lr: float = Field(default=0.0001, gt=0)
    weight_decay: float = 0.0001
    min_lr: float = 1e-6
    batch_size: int = Field(default=2, ge=1, multiple_of=2)
    patience: int = 10
    seed: int = 73
    dropout: float = Field(default=0.3, ge=0, lt=1, allow_inf_nan=False)
    grad_clip_norm: float = 1.0
    is_save_training_info: bool = True  # 是否保存训练好的模型到本地
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )  # 训练设备
    num_workers: int = 8

    # 损失权重
    recon_weight: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    pheno_weight: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    train_mask_ratio: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    val_mask_ratio: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)

    # 保留原模型默认规模，可为 CPU 合成冒烟显式缩小。
    patch_size: int = Field(default=20, ge=1)
    d_model: int = Field(default=128, ge=1)
    d_ext: int = Field(default=166, ge=1)
    num_heads: int = Field(default=4, ge=1)
    encoder_layers: int = Field(default=4, ge=1)
    decoder_layers: int = Field(default=2, ge=1)
    n_phenotypes: int | None = Field(default=None, ge=1)
    n_time_points: int | None = Field(default=None, ge=1)

    # 学习率调度器参数
    use_warmup: bool = True  # 使用warmup
    warmup_epochs: int = 5  # warmup轮数
    warmup_start_factor: float = 0.1  # 从 base_lr * 0.1 开始
    warmup_end_factor: float = 1.0  # 持续增长到 base_lr * 1.0

    phenotype_names: list[str] = Field(
        default_factory=lambda: ["RGBVI"],
        min_length=1,  # 自动验证：列表至少需要包含一个元素
        description="选择的表型特征列表",
    )

    # 数据集参数
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    chunk_size: int = Field(default=12000, ge=1)
    overlap: int = Field(default=2000, ge=0)
    time_coordinates: list[float] | None = None
    phenotype_missing_strategy: Literal["error"] = "error"
    split_file: str | None = Field(
        default=None,
        description="已有 TFPP 普通/fold manifest 路径；显式路径必须存在，None/auto 在共享 splits/ 中生成或复用",
    )

    @model_validator(mode="after")
    def _validate_chunks(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if not math.isclose(self.recon_weight + self.pheno_weight, 1, rel_tol=0, abs_tol=1e-8):
            raise ValueError("recon_weight 与 pheno_weight 之和必须为1")
        if not self.use_pheno_help and (self.pheno_weight != 0 or self.recon_weight <= 0):
            raise ValueError("关闭表型辅助时请显式设置 pheno_weight=0、recon_weight=1")
        if self.n_phenotypes is not None and self.n_phenotypes != len(self.phenotype_names):
            raise ValueError("n_phenotypes 必须与 phenotype_names 一致")
        self.n_phenotypes = len(self.phenotype_names)
        if self.time_coordinates is not None:
            if not self.time_coordinates or not all(math.isfinite(value) for value in self.time_coordinates):
                raise ValueError("time_coordinates 必须非空且为有限值")
            if self.n_time_points is not None and self.n_time_points != len(self.time_coordinates):
                raise ValueError("n_time_points 必须与 time_coordinates 一致")
            self.n_time_points = len(self.time_coordinates)
        return self

    @computed_field
    @property
    def selection_metric(self) -> str:
        return "val_total_loss" if self.use_pheno_help else "val_recon_loss"

    @computed_field
    @property
    def selection_definition(self) -> str:
        return ("recon_weight * val_recon_loss + pheno_weight * val_pheno_loss"
                if self.use_pheno_help else "val_recon_loss")

    @computed_field
    @property
    def reconstruction_loss_definition(self) -> str:
        return "mean_ce_selected_observed_nonpadding_snps_zero_if_empty"

    @computed_field
    @property
    def phenotype_loss_definition(self) -> str:
        return "mean_squared_error_over_batch_phenotypes_times" if self.use_pheno_help else "disabled_zero"

    @property
    def output_dir(self) -> str:
        """训练输出文件的保存总目录"""
        base = f"{self.output_root}/{self.exp_name}"
        return base if self.split_tag is None else f"{base}/{self.split_tag}"

    @property
    def best_model_save_dir(self) -> str:
        """训练好模型的保存目录"""
        return f"{self.output_dir}/models"

    @property
    def training_history_save_dir(self) -> str:
        """训练历史文件保存目录"""
        return f"{self.output_dir}/training_history"

    @property
    def compressed_snp_save_dir(self) -> str:
        """压缩后的SNP文件目录"""
        return "dataset/maize/snp_zip"

    @property
    def log_file(self) -> str:
        """日志文件路径"""
        return f"{self.output_dir}/convae.log"

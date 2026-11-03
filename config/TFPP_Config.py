"""TFPP 主模型的训练与推理配置定义。

本模块定义 TFPPConfig (Pydantic BaseModel)，集中管理 TFPP 模型训练/推理所需的
全部超参数、数据路径、模型结构参数和输出目录。
"""

import math
from datetime import datetime
from typing import Literal

import torch
from pydantic import BaseModel, Field, field_validator, model_validator


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

    # 数据集标识：这些字段没有跨数据集的合理默认值，必须由 profile 或 runner 提供。
    pheno_dir: str = Field(
        description="表型 CSV 目录；每个 phenotype_names 元素对应其中的 <名称>.csv。"
    )
    snp_file: str = Field(
        description="原始 SNP 剂量矩阵 .npy 或 .csv；行顺序必须与表型和 sample_ids_file 一致。"
    )
    time_coordinates: list[float] = Field(
        min_length=1,
        description="表型列对应的时间坐标；顺序和长度必须与每个表型 CSV 的时间列一致。",
    )

    @field_validator("time_coordinates")
    @classmethod
    def _validate_time_coordinates(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("time_coordinates 必须全部为有限值")
        if len(values) > 1 and any(
            current <= previous for previous, current in zip(values, values[1:])
        ):
            raise ValueError("time_coordinates 必须按采样先后严格递增")
        return values
    phenotype_names: list[str] = Field(
        min_length=1,
        description="待预测表型名称；名称同时用于定位 pheno_dir 中的 <名称>.csv 文件。",
    )

    # 训练控制。
    epochs: int = Field(default=200, ge=1, le=500, description="最大训练 epoch 数。")
    lr: float = Field(default=1e-4, gt=0, description="优化器初始学习率。")
    weight_decay: float = Field(default=1e-4, ge=0, description="优化器 L2 权重衰减系数。")
    min_lr: float = Field(default=1e-6, gt=0, description="余弦退火学习率的下限。")
    batch_size: int = Field(
        default=32,
        ge=1,
        multiple_of=2,
        description="每个 DataLoader 批次的样本数；必须为偶数以满足当前分组训练约束。",
    )
    patience: int = Field(default=20, ge=1, description="验证损失未改善时的早停耐心轮数。")
    seed: int = Field(default=73, ge=0, description="随机划分、初始化及可复现实验使用的全局随机种子。")
    dropout: float = Field(default=0.0, ge=0, lt=1, description="模型各 dropout 层的失活概率。")
    grad_clip_norm: float = Field(default=1.0, gt=0, description="反向传播时的梯度 L2 范数裁剪阈值。")
    is_save_training_info: bool = Field(
        default=True, description="是否保存 checkpoint、scaler、训练历史和测试评估产物。"
    )
    run_stage: Literal["full", "engineering_pretest"] = Field(
        default="full",
        description="运行协议：full 为正式训练；engineering_pretest 为不接触测试标签的工程预检。",
    )
    output_root: str = Field(default="outputs", description="实验输出根目录；最终目录为 output_root/exp_name。")
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="PyTorch 设备标识；未指定时按当前环境自动选择 cuda 或 cpu。",
    )

    # 学习率 warmup；关闭时 warmup_* 参数不参与调度。
    use_warmup: bool = Field(default=True, description="是否在训练开始阶段启用线性 warmup。")
    warmup_epochs: int = Field(default=5, ge=0, description="warmup 持续的 epoch 数；仅 use_warmup=true 时生效。")
    warmup_start_factor: float = Field(default=0.1, gt=0, le=1, description="warmup 起点相对于 lr 的倍率。")
    warmup_end_factor: float = Field(default=1.0, gt=0, le=1, description="warmup 终点相对于 lr 的倍率。")

    # 模型结构。
    d_model: int = Field(default=32, ge=1, description="Transformer 隐藏表示维度。")
    num_heads: int = Field(default=4, ge=1, description="多头注意力头数；应能整除 d_model。")
    num_layers: int = Field(default=1, ge=1, description="Transformer 编码器层数。")
    num_embeddings: int = Field(default=4, ge=1, description="SNP 离散编码的 embedding 类别数；当前剂量/缺失编码使用 4 类。")
    d_ff: int = Field(default=128, ge=1, description="Transformer 前馈网络的中间隐藏维度。")
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
    # K 折交叉验证；use_kfold=true 时不能同时提供单次 split_file。
    use_kfold: bool = Field(default=False, description="是否执行外层 K 折交叉验证，而非单次 train/val/test 划分。")
    n_folds: int = Field(default=10, ge=2, description="use_kfold=true 时的外层折数。")
    kfold_val_ratio: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="外层 train 内再划分给早停验证集的比例，不使用 test 选模型",
    )
    fold_index: int | None = Field(
        default=None, ge=1, description="内部折实验编号，从1开始"
    )

    # 数据划分和加载；比例仅在未提供 split_file 的普通训练中用于生成划分。
    train_ratio: float = Field(default=0.8, gt=0, lt=1, description="普通训练生成划分时的训练集比例。")
    test_ratio: float = Field(default=0.1, gt=0, lt=1, description="普通训练生成划分时的最终测试集比例。")
    val_ratio: float = Field(default=0.1, gt=0, lt=1, description="普通训练生成划分时的早停验证集比例。")
    num_workers: int = Field(default=8, ge=0, description="DataLoader 子进程数；0 表示在主进程加载数据。")
    split_file: str | None = Field(
        default=None,
        description="个体划分 .npz 路径；已有文件优先，不存在则生成，auto 使用 splits/；仅普通训练",
    )
    sample_ids_file: str | None = Field(
        default=None,
        description="可选 SampleID 单列 CSV；用于校验 split 中的个体身份与顺序",
    )

    @model_validator(mode="after")
    def _validate_split_mode(self):
        if self.use_kfold and self.split_file is not None:
            raise ValueError(
                "split_file 为单次 train/val/test 划分，不能与 use_kfold 同时使用"
            )
        if self.use_kfold and self.fold_index is not None:
            raise ValueError("fold_index 仅用于已确定划分的内部折实验")
        if self.run_stage == "engineering_pretest" and (
            self.use_kfold
            or self.split_file in (None, "auto")
            or self.preselected_snp_file is None
        ):
            raise ValueError(
                "engineering_pretest 必须使用显式 split_file 和预选原始 SNP 索引，且不支持 K 折"
            )
        if self.run_stage == "full" and (
            self.gwas_pca_panel_size is not None
            or self.gwas_pca_algorithm_version
            != "sklearn_centered_all_train_markers_v1"
        ):
            raise ValueError(
                "普通/K折GWAS固定使用当前训练矩阵全部列PCA；panel参数仅供engineering_pretest"
            )
        if self.run_stage == "engineering_pretest" and (
            self.gwas_pca_panel_size is None
            or self.gwas_pca_panel_seed is not None
            or self.gwas_pca_algorithm_version
            != "sklearn_randomized_centered_even_panel_v1"
        ):
            raise ValueError(
                "engineering_pretest 必须显式记录evenly-spaced PCA panel；该算法不使用panel seed"
            )
        return self

    preselected_snp_file: str | None = Field(
        default=None,
        description="可选的原始 SNP 有序列索引 .npy；提供后跳过内部筛选并直接使用该特征集合。",
    )

    # SNP 特征筛选参数 (gwas_k=0 时不筛选)
    gwas_k: int = Field(
        default=0,
        ge=0,
        description="质量控制后加性+显性联合 GWAS 的候选池大小；0=关闭内部筛选。",
    )
    gwas_population_structure: Literal["pca"] = Field(default="pca", description="GWAS 群体结构校正方法；当前只支持 PCA。")
    gwas_n_pcs: int = Field(default=5, ge=1, description="GWAS 中作为协变量的主成分数量。")
    screen_final_k: int = Field(
        default=20,
        ge=1,
        description="候选内 DC-SIS 最终保留的 SNP 数；按实验显式设置。",
    )

    @model_validator(mode="after")
    def _validate_screening_protocol(self):
        if self.gwas_k > 0 and len(self.time_coordinates) < 2:
            raise ValueError("GWAS 动态性状筛选至少需要两个实际时间坐标")
        if self.gwas_k > 0 and self.screen_final_k > self.gwas_k:
            raise ValueError("screen_final_k 不能超过 gwas_k 候选池")
        if (
            self.gwas_k > 0
            and not self.screen_multivariate
            and len(self.phenotype_names) != 1
        ):
            raise ValueError(
                "screen_multivariate=false 仅支持单表型；多表型正式筛选必须使用联合模式"
            )
        return self

    gwas_pca_panel_size: int | None = Field(
        default=None,
        ge=1,
        description="None=普通/K折GWAS使用当前训练矩阵全部列；工程pretest显式使用固定panel",
    )
    gwas_pca_panel_seed: int | None = Field(
        default=None,
        ge=0,
        description="随机panel时使用；当前evenly-spaced算法不使用随机数，必须为None",
    )
    gwas_pca_algorithm_version: str = Field(
        default="sklearn_centered_all_train_markers_v1",
        description="GWAS PCA 的可复现算法版本标识；必须与 run_stage 对应的协议一致。",
    )
    gwas_pca_panel_indices_file: str | None = Field(
        default=None,
        description="engineering_pretest 使用的 PCA marker panel 原始列索引文件；普通训练必须为空。",
    )
    gwas_pca_panel_indices_sha256: str | None = Field(
        default=None,
        pattern="^[0-9a-f]{64}$",
        description="PCA marker panel 索引文件的 SHA-256；engineering_pretest 用于重放时校验文件未变化。",
    )
    screen_multivariate: bool = Field(
        default=True,
        description="True=多表型拼接联合筛选；False=单表型模式，不支持多表型逐项并集。",
    )
    screen_missing_strategy: str = Field(
        default="mode",
        pattern="^(error|mode)$",
        description="筛选缺失策略：error=拒绝，mode=仅在当前普通/折内 train_idx 上拟合众数",
    )
    wide_missing_strategy: str = Field(
        default="mode",
        pattern="^(error|mode)$",
        description="普通/折内 Wide 剂量缺失策略，独立于筛选；mode 仅在训练个体拟合",
    )
    screen_cache_dir: str = Field(
        default="dataset/screen_cache",
        description="DC-SIS/GWAS 筛选结果缓存目录；缓存键包含训练个体和筛选参数。",
    )

    @property
    def preprocessing_save_dir(self) -> str:
        return f"{self.output_dir}/preprocessing"

    # 损失函数
    loss_type: str = Field(
        default="mse",
        pattern="^(mse|mse_pcc|mse_offset)$",
        description="mse=纯MSE, mse_pcc=MSE+PCC, mse_offset=MSE+个体偏移MSE",
    )
    offset_weight: float = Field(default=0.5, ge=0, description="mse_offset 中个体偏移 MSE 项的权重。")
    pcc_weight: float = Field(default=0.1, ge=0, description="mse_pcc 中 PCC 辅助损失项的权重。")

    # L1 稀疏正则化 (宽通路)
    l1_wide_weight: float = Field(default=1e-5, ge=0, description="Wide 分支权重的 L1 稀疏正则系数；0 表示关闭。")

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
        return f"{self.output_root}/{self.exp_name}"

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

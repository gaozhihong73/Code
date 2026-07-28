from datetime import datetime

import torch
from pydantic import BaseModel, Field


class TFPPConfig(BaseModel):
    """TFPP 表型预测模型的训练与推理配置。"""

    exp_name: str = Field(
        default_factory=lambda: (
            f"exp_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
        ),
        description="实验名称，如果不指定则按时间自动生成",
    )

    # 文件目录
    pheno_dir: str = "dataset/maize/cleaned_pheno"
    snp_file: str = "dataset/maize/snp/snp_50000.npy"

    # 训练参数
    epochs: int = Field(default=100, ge=1, le=500)
    lr: float = Field(default=0.0001, gt=0)
    weight_decay: float = 0.0001
    min_lr: float = 1e-7
    batch_size: int = Field(default=32, ge=1, multiple_of=2)
    patience: int = 10
    seed: int = 73
    dropout: float = 0.3
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
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2  # 编码器层数
    num_embeddings: int = 4  # 嵌入层维度
    d_ff: int = 256  # 前馈网络隐藏维度
    degree: int = Field(
        default=3,
        ge=0,
        description="Legendre 多项式阶数, 0=退化为纯加性(仅常数项)",
    )
    ablation_mode: str = Field(
        default="none",
        pattern="^(none|wide|deep|add)$",
        description="模型消融模式: none=完整模型, wide=仅宽通路, deep=仅深通路, add=去掉门控直接相加",
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
    use_kfold: bool = True  # 是否使用K折交叉验证
    n_folds: int = 10  # K折数量

    # 数据集参数
    train_ratio: float = 0.8
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    num_workers: int = 4
    # 规范字符串输入
    norm_method: str = Field(
        default="global",
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
        """训练好模型的保存目录"""
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

import torch
from pydantic import BaseModel, Field


class ConvAEConfig(BaseModel):
    """实验配置类"""

    use_pheno_help: bool = True  # 是否使用表型辅助压缩

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
    dropout: float = 0.3
    grad_clip_norm: float = 1.0
    is_save_training_info: bool = True  # 是否保存训练好的模型到本地
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )  # 训练设备
    num_workers: int = 8

    # 损失权重
    recon_weight: float = 0.7
    pheno_weight: float = 0.3

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
    val_ratio: float = 0.2
    test_ratio: float = 0.0

    @property
    def output_dir(self) -> str:
        """训练输出文件的保存总目录"""
        return "outputs/convae"

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
        return f"{self.output_dir}/convae.log"

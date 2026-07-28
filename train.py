import argparse

from config import TFPPConfig
from trainer import TFPPTrainer


def _str_to_bool(s: str) -> bool:
    """将 'true'/'false' 字符串正确解析为布尔值，避免 argparse type=bool 的陷阱。"""
    return s.strip().lower() == "true"


def parse_args():
    """解析命令行参数并返回 argparse Namespace 对象。"""
    parser = argparse.ArgumentParser(description="DynaGS: TFPP 模型训练入口 (真实数据)")

    # ================= 实验标识与数据路径 =================
    parser.add_argument(
        "--exp_name", type=str, default=None, help="实验名称，默认为时间戳格式"
    )
    parser.add_argument(
        "--snp_file",
        type=str,
        default="dataset/maize/snp/snp_50000.npy",
        help="SNP 数据文件路径 (.npy)",
    )
    parser.add_argument(
        "--pheno_dir",
        type=str,
        default="dataset/maize/cleaned_pheno",
        help="表型数据所在目录",
    )

    # ================= 数据集与特征配置 =================
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="测试集比例 (需与推理时一致)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="验证集比例 (需与推理时一致)"
    )
    parser.add_argument(
        "--pheno_names",
        type=str,
        nargs="+",
        default=["GreenCoverage"],
        help="需要预测的表型名称列表，需与表型文件名对应，例如: --pheno_names Yield PH GW",
    )
    parser.add_argument(
        "--time_coordinates",
        type=float,
        nargs="+",
        default=[10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101],
        help="采样时间点坐标列表，例如: --time_coordinates 10 14 26",
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="global",
        choices=["global", "timepoint", "residual_global"],
        help="表型归一化策略",
    )
    parser.add_argument(
        "--scaler_type",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "robust"],
        help="归一化缩放器类型",
    )

    # ================= 训练超参数 =================
    parser.add_argument("--epochs", type=int, default=100, help="训练总轮数")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="批次大小 (必须为 2 的倍数)"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="初始学习率")
    parser.add_argument(
        "--min_lr", type=float, default=1e-7, help="学习率下限 (CosineAnnealing 最终值)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="权重衰减 (L2 正则化)"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比例")
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="梯度裁剪最大范数, 0=不裁剪",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="早停耐心值 (连续多少轮无提升则停止)"
    )
    parser.add_argument("--seed", type=int, default=73, help="全局随机种子")
    parser.add_argument(
        "--is_save",
        type=_str_to_bool,
        default=True,
        help="是否保存训练好的模型与历史到本地 (true/false)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader 工作线程数"
    )

    # ================= SNP 特征筛选 =================
    parser.add_argument(
        "--screen_k",
        type=int,
        default=0,
        help="DC-SIS 筛选保留的 SNP 数目, 0=跳过 (两阶段: GWAS 粗筛后在子集上精筛)",
    )
    parser.add_argument(
        "--gwas_k",
        type=int,
        default=0,
        help="GWAS 筛选保留的 SNP 数目, 0=跳过 (向量化, 适合高维 SNP 粗筛)",
    )

    # ================= 学习率调度器 (Warmup) =================
    parser.add_argument(
        "--use_warmup",
        type=_str_to_bool,
        default=True,
        help="是否使用 Warmup 学习率预热 (true/false)",
    )
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup 轮数")
    parser.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Warmup 起始学习率系数 (base_lr × start_factor)",
    )
    parser.add_argument(
        "--warmup_end_factor",
        type=float,
        default=1.0,
        help="Warmup 结束学习率系数 (base_lr × end_factor)",
    )

    # ================= 模型结构参数 =================
    parser.add_argument("--d_model", type=int, default=64, help="Transformer 隐藏维度")
    parser.add_argument("--num_heads", type=int, default=4, help="多头注意力机制的头数")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="编码器层数 (网络深度)"
    )
    parser.add_argument("--d_ff", type=int, default=256, help="前馈网络隐藏层维度")
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=4,
        help="SNP 类别嵌入维度 (通常等于等位基因类别数)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Legendre 多项式阶数 (宽通路), 0=退化为纯加性 (仅常数项)",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "wide", "deep", "add"],
        help="模型消融模式: none=完整, wide=仅宽通路, deep=仅深通路, add=去掉门控直接相加",
    )

    # ================= K 折交叉验证 =================
    parser.add_argument(
        "--use_kfold",
        type=_str_to_bool,
        default=False,
        help="是否启用 K 折交叉验证 (true/false)",
    )
    parser.add_argument("--n_folds", type=int, default=10, help="K 折交叉验证的折数")

    return parser.parse_args()


def main():
    """TFPP 模型训练入口：解析参数、构建配置、初始化训练器并执行训练。"""
    args = parse_args()

    config = TFPPConfig(
        exp_name=args.exp_name,
        snp_file=args.snp_file,
        pheno_dir=args.pheno_dir,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        phenotype_names=args.pheno_names,
        time_coordinates=args.time_coordinates,
        norm_method=args.norm_method,
        scaler_type=args.scaler_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        seed=args.seed,
        is_save_training_info=args.is_save,
        num_workers=args.num_workers,
        screen_k=args.screen_k,
        gwas_k=args.gwas_k,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        warmup_end_factor=args.warmup_end_factor,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_embeddings=args.num_embeddings,
        degree=args.degree,
        ablation_mode=args.ablation_mode,
        use_kfold=args.use_kfold,
        n_folds=args.n_folds,
    )

    print(f"[*] 正在初始化 TFPPTrainer，实验名称: {config.exp_name}")
    trainer = TFPPTrainer(config=config)

    print("[*] 开始执行训练流程...")
    trainer.train()
    print("[*] 流程执行完毕。")


if __name__ == "__main__":
    main()

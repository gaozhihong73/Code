"""DynaGS: TFPP 模型训练入口 (真实数据)。

所有命令行参数的默认值均与 config/TFPP_Config.py 保持一致。
"""

import argparse

from config import TFPPConfig
from trainer import TFPPTrainer


def _str_to_bool(s: str) -> bool:
    """将 'true'/'false' 字符串正确解析为布尔值，避免 argparse type=bool 的陷阱。"""
    if s.strip().lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("请显式使用 true 或 false")
    return s.strip().lower() == "true"


def parse_args():
    """解析命令行参数并返回 argparse Namespace 对象。"""
    parser = argparse.ArgumentParser(
        description="DynaGS: TFPP 模型训练入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ================= 实验标识与数据路径 =================
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="实验名称，用于标识本次训练。若不指定则自动按时间戳生成 (格式: exp_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--snp_file",
        type=str,
        default="dataset/maize/snp/snp_50000.npy",
        help="SNP 基因型数据文件路径 (.npy 格式, 形状为 [样本数, SNP数], 编码值 0/1/2)",
    )
    parser.add_argument(
        "--pheno_dir",
        type=str,
        default="dataset/maize/cleaned_pheno",
        help="表型数据所在目录，目录下每个 CSV 文件对应一个表型 (文件名即表型名)",
    )

    # ================= 数据集划分与预处理 =================
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="个体划分 .npz 路径，存在则复用其 seed/比例，不存在则生成；auto 按 seed/比例/行索引哈希命名，仅普通训练",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集占全部样本的比例",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集占全部样本的比例 (用于早停和模型选择)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集占全部样本的比例 (用于最终评估, 需与推理时一致)",
    )
    parser.add_argument(
        "--pheno_names",
        type=str,
        nargs="+",
        default=["RGBVI"],
        help="需要预测的表型名称列表，需与 pheno_dir 中的 CSV 文件名对应。例如: --pheno_names RGBVI GreenCoverage",
    )
    parser.add_argument(
        "--time_coordinates",
        type=float,
        nargs="+",
        default=[10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101],
        help="采样时间点坐标列表 (天)，对应表型 CSV 中的列名。例如: --time_coordinates 10 14 26",
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="residual_global",
        choices=["global", "timepoint", "residual_global"],
        help=(
            "表型归一化策略: "
            "global=所有时间点共享同一均值/方差; "
            "timepoint=每个时间点独立归一化; "
            "residual_global=先去除时间点均值再全局归一化 (推荐)"
        ),
    )
    parser.add_argument(
        "--scaler_type",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "robust"],
        help=(
            "归一化缩放器类型: "
            "zscore=标准化 (均值0方差1); "
            "minmax=缩放到[0,1]; "
            "robust=基于中位数和四分位距, 对离群值鲁棒"
        ),
    )

    # ================= 训练超参数 =================
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="训练总轮数上限 (配合早停使用, 实际训练轮数可能少于此值)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="每批次样本数 (必须为 2 的倍数, K-fold 分组需要)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Deep 通路基础学习率; Wide 通路学习率自动缩放为 lr×0.1",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="学习率下限 (CosineAnnealing 衰减的最终值)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减系数 (AdamW 的 L2 正则化)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout 比例",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="梯度裁剪最大范数, 防止梯度爆炸。0=不裁剪",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="早停耐心值: 验证损失连续多少轮无改善则终止训练",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="全局随机种子, 控制数据划分和模型初始化的可复现性 (训练与推理必须一致)",
    )
    parser.add_argument(
        "--is_save",
        type=_str_to_bool,
        default=True,
        help="是否保存训练好的模型权重、训练历史和归一化器到本地 (true/false)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader 数据加载工作线程数, 0=主进程加载",
    )

    # ================= SNP 特征筛选 =================
    parser.add_argument(
        "--screen_k",
        type=int,
        default=10000,
        help=(
            "DC-SIS (距离相关确定独立性筛选) 保留的 SNP 数目。"
            "基于非线性距离相关系数衡量每个 SNP 与表型的关联强度。"
            "0=跳过此步骤"
        ),
    )
    parser.add_argument(
        "--gwas_k",
        type=int,
        default=10000,
        help=(
            "GWAS (全基因组关联分析) 筛选保留的 SNP 数目。"
            "使用向量化 F 检验快速粗筛高维 SNP。"
            "普通入口按生产筛选策略：P>50000串联，否则两者并集；严格串联请用统一 runner。"
            "0=跳过此步骤"
        ),
    )
    parser.add_argument(
        "--screen_multivariate",
        type=_str_to_bool,
        default=TFPPConfig.model_fields["screen_multivariate"].default,
        help=(
            "DC-SIS 多元模式: true=将多表型 z-score 标准化后拼接为多维 Y,"
            " SNP 与 Y=(Y1,Y2,...) 联合关联一次筛选;"
            " false=逐表型筛选后取并集"
        ),
    )
    parser.add_argument(
        "--screen_missing_strategy",
        choices=["error", "mode"],
        default=TFPPConfig.model_fields["screen_missing_strategy"].default,
        help="筛选缺失策略；普通/折内 mode 仅在当前 train_idx 上拟合众数",
    )
    parser.add_argument(
        "--wide_missing_strategy", choices=["error", "mode"],
        default=TFPPConfig.model_fields["wide_missing_strategy"].default,
        help="普通/折内 Wide 剂量缺失策略；不改变 Deep 的缺失类别0",
    )
    parser.add_argument(
        "--screen_cache_dir", default=TFPPConfig.model_fields["screen_cache_dir"].default,
        help="筛选索引缓存目录",
    )

    # ================= 损失函数与正则化 =================
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "mse_pcc", "mse_offset"],
        help=(
            "损失函数类型: "
            "mse=纯均方误差; "
            "mse_pcc=MSE + PCC损失 (鼓励预测值与真实值的线性相关性); "
            "mse_offset=MSE + 个体偏移MSE (惩罚每个样本在所有时间点上的系统性偏差)"
        ),
    )
    parser.add_argument(
        "--pcc_weight",
        type=float,
        default=0.2,
        help="PCC 损失权重 (仅 mse_pcc/mse_offset 模式生效): loss = MSE + pcc_weight × (1-PCC)",
    )
    parser.add_argument(
        "--offset_weight",
        type=float,
        default=0.5,
        help="个体偏移损失权重 (仅 mse_offset 模式生效): 惩罚单样本跨时间点的均值偏移",
    )
    parser.add_argument(
        "--l1_wide_weight",
        type=float,
        default=1e-5,
        help="宽通路 L1 稀疏正则系数, 对 Wide 层参数施加 L1 惩罚促进稀疏性。0=关闭",
    )

    # ================= 学习率调度器 (Warmup + Cosine) =================
    parser.add_argument(
        "--use_warmup",
        type=_str_to_bool,
        default=True,
        help="是否使用学习率预热: 前 warmup_epochs 轮从低学习率线性增长到基础学习率 (true/false)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup 预热轮数, 结束后切换为 CosineAnnealing 衰减",
    )
    parser.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Warmup 起始学习率系数: 初始 lr = base_lr × warmup_start_factor",
    )
    parser.add_argument(
        "--warmup_end_factor",
        type=float,
        default=1.0,
        help="Warmup 结束学习率系数: 预热结束时 lr = base_lr × warmup_end_factor",
    )

    # ================= 模型结构参数 =================
    parser.add_argument(
        "--d_model",
        type=int,
        default=32,
        help="Transformer 隐藏维度 (Token 嵌入维度, 需能被 num_heads 整除)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="多头自注意力的头数 (d_model 必须能被 num_heads 整除)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Transformer 编码器堆叠层数 (网络深度)",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=128,
        help="编码器内前馈网络 (FFN) 的隐藏层维度",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=4,
        help="SNP 类别嵌入的类别数 (对应基因型编码 0/1/2 + padding, 通常为 4)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=5,
        help=(
            "Legendre 多项式阶数 (宽通路): "
            "控制时间基函数的复杂度, 阶数越高拟合时间动态越灵活。"
            "0=退化为纯加性模型 (仅常数项)"
        ),
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="all",
        choices=["all", "wide", "deep", "add"],
        help=(
            "模型消融模式: "
            "all=完整模型 (Wide+Deep+门控融合); "
            "wide=仅宽通路 (Legendre 多项式线性模型); "
            "deep=仅深通路 (Transformer 编码器); "
            "add=去掉门控网络, Wide 和 Deep 输出直接相加"
        ),
    )

    # ================= K 折交叉验证 =================
    parser.add_argument(
        "--use_kfold",
        type=_str_to_bool,
        default=False,
        help="是否启用 K 折交叉验证模式 (true/false)。启用后忽略 train/val/test_ratio, 每折独立训练并评估",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="K 折交叉验证的折数 (仅 use_kfold=true 时生效)",
    )
    parser.add_argument(
        "--kfold_val_ratio", type=float,
        default=TFPPConfig.model_fields["kfold_val_ratio"].default,
        help="每折外层 train 中划给早停 val 的比例；test 仅最终评分",
    )

    # 配置是默认值唯一来源，避免 CLI 副本随配置漂移。
    aliases = {"pheno_names": "phenotype_names", "is_save": "is_save_training_info"}
    defaults = {}
    for action in parser._actions:
        field = TFPPConfig.model_fields.get(aliases.get(action.dest, action.dest))
        if field is not None and action.dest != "exp_name":
            defaults[action.dest] = field.get_default(call_default_factory=True)
    parser.set_defaults(**defaults)
    return parser.parse_args()


def main():
    """TFPP 模型训练入口：解析参数、构建配置、初始化训练器并执行训练。"""
    args = parse_args()

    config = TFPPConfig(
        exp_name=args.exp_name,
        snp_file=args.snp_file,
        pheno_dir=args.pheno_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_file=args.split_file,
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
        screen_multivariate=args.screen_multivariate,
        screen_missing_strategy=args.screen_missing_strategy,
        wide_missing_strategy=args.wide_missing_strategy,
        screen_cache_dir=args.screen_cache_dir,
        loss_type=args.loss_type,
        pcc_weight=args.pcc_weight,
        offset_weight=args.offset_weight,
        l1_wide_weight=args.l1_wide_weight,
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
        kfold_val_ratio=args.kfold_val_ratio,
    )

    TFPPTrainer(config=config).train()


if __name__ == "__main__":
    main()

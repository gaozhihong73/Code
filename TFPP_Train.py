"""
表型预测Transformer模型训练脚本
支持单次训练和批量训练所有实验组合
"""

from pathlib import Path

import pandas as pd

from config.TFPP_Config import TFPPConfig
from trainer.TFPP_Trainer import TFPPTrainer


def _build_config(phenotypes: list[str]) -> TFPPConfig:
    """构建 TFPPConfig，抽取共用的超参数，仅表型相关字段按需传入。

    Parameters
    ----------
    phenotypes : list[str]
        需要一起训练的表型列表，同时拼接作为实验名称。
    """
    return TFPPConfig(
        exp_name="_".join(phenotypes),  # 表型名拼接作为实验名称，输出目录互不覆盖
        snp_file="dataset/maize/snp/snp_50000.npy",
        pheno_dir="dataset/maize/pheno",
        epochs=200,
        lr=0.0001,
        min_lr=1e-7,  # 学习率下限
        weight_decay=0.0001,
        batch_size=32,
        patience=20,
        seed=73,  # 随机种子，训练与推理保持一致以复现数据划分
        dropout=0.3,
        use_kfold=False,  # 使用K折交叉验证
        use_warmup=True,  # 使用warmup学习率调度
        warmup_epochs=10,  # warmup轮数
        warmup_start_factor=0.1,  # 从 base_lr * 0.1 开始
        warmup_end_factor=1.0,  # 持续增长到 base_lr * 1.0
        loss_type="mse",  # mse=纯MSE, mse_pcc=MSE+PCC, mse_offset=MSE+个体偏移MSE
        offset_weight=0.5,  # mse_offset 的偏移损失权重
        pcc_weight=0.2,  # mse_pcc 的 PCC 损失权重
        phenotype_names=phenotypes,  # 一次训练多个表型
        time_coordinates=[10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101],
        norm_method="residual_global",  # 归一化策略：global, timepoint, residual_global
        scaler_type="zscore",  # 归一化函数：zscore，minmax，robust
        d_model=32,  # Transformer 嵌入维度
        num_heads=4,  # 注意力头数
        num_layers=1,  # 编码器层数
        num_embeddings=4,  # 嵌入层维度
        d_ff=128,  # 前馈网络隐藏维度
        degree=5,  # Legendre 多项式阶数, 0=退化为纯加性(仅常数项)
        ablation_mode="all",  # 模型消融模式: all|wide|deep|add
        train_ratio=0.8,  # 训练集比例
        test_ratio=0.1,  # 测试集比例
        val_ratio=0.1,  # 验证集比例
        num_workers=8,  # DataLoader 工作进程数
        gwas_k=100000,
        screen_k=10000,
        screen_multivariate=True,
    )


# 玉米数据
def train_maize():
    """玉米数据集实验（多表型一起训练，单模型多输出头）"""

    # ============ 创建训练器 ============
    trainer = TFPPTrainer(config=_build_config(["RGBVI"]))

    # ============ 开始训练 ============
    results = trainer.train()

    return results


def _train_single_phenotype(phenotype: str) -> dict:
    """训练单个表型并返回汇总指标"""
    print(f"\n========== 开始训练表型: {phenotype} ==========")
    trainer = TFPPTrainer(config=_build_config([phenotype]))
    results = trainer.train()

    # 单表型训练，反归一化后的整体指标即为该表型的指标
    denorm_overall = results["test_results"]["denormalized"]["overall"]
    return {
        "phenotype": phenotype,
        "best_epoch": results["best_epoch"],
        "best_val_loss": round(results["best_val_loss"], 6),
        "training_time_s": round(results["training_time"], 2),
        **{f"test_{metric}": val for metric, val in denorm_overall.items()},
    }


def train_multi_phenotype(phenotype_names: list[str] | None = None) -> pd.DataFrame:
    """逐个训练多个单表型，并将测试集指标结果汇总到一个 CSV 文件。

    Parameters
    ----------
    phenotype_names : list[str] | None
        需要依次训练的表型列表，默认为 ["GLI2", "NDI", "RGBVI"]。

    Returns
    -------
    pd.DataFrame
        每个表型一行的指标汇总表。
    """
    if phenotype_names is None:
        phenotype_names = ["GLI2", "NDI", "RGBVI"]

    summary_dir = Path("outputs/tfpp")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "GLI2_NDI_RGBVI_summary.csv"

    rows = []
    for phenotype in phenotype_names:
        try:
            row = _train_single_phenotype(phenotype)
            rows.append(row)
            print(f"[完成] {phenotype}: {row}")
        except Exception as e:
            print(f"[失败] {phenotype}: {e}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n多表型指标汇总已保存至: {summary_csv}")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    # train_multi_phenotype()
    train_maize()

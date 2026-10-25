"""20 个真实 QTL 的 oracle 筛选诊断实验。

该脚本仅用于动态非线性遗传场景下的机制验证：它从模拟真值中读取
20 个因果 SNP，再将这些已知列作为 TFPP 的输入。真实数据和常规模拟
比较中均不可使用这种预知信息，因此结果必须标记为 oracle 诊断结果，
不得与未知 QTL 的筛选实验作公平性能比较。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.TFPP_Config import TFPPConfig
from trainer.TFPP_Trainer import TFPPTrainer


DEFAULT_DATA_DIR = Path(
    "/root/DynaGS/dataset/sim_data/tfpp_ready/"
    "scenario_tfpp_dynamic_sparse_qtl_n1600_p10000_q20_no_epistasis_h2_04/rep_01"
)
DEFAULT_OUTPUT_ROOT = Path("/root/DynaGS/outputs/tfpp")
DEFAULT_EXP_NAME = "scenario_tfpp_dynamic_sparse_qtl_n1600_p10000_q20_no_epistasis_h2_04/rep_01"
EXPECTED_QTL_COUNT = 20


def extract_true_qtl_indices(data_dir: Path, output_path: Path) -> np.ndarray:
    """从模拟真值提取并保存 TFPP 使用的 0-based QTL 列索引。"""
    truth_path = data_dir / "truth" / "causal_effects.csv"
    genotype_path = data_dir / "genotype.npy"
    if not truth_path.is_file():
        raise FileNotFoundError(f"缺少模拟真值文件: {truth_path}")
    if not genotype_path.is_file():
        raise FileNotFoundError(f"缺少基因型文件: {genotype_path}")

    effects = pd.read_csv(truth_path)
    required_columns = {"snp_index", "snp_name"}
    if not required_columns.issubset(effects.columns):
        raise ValueError(f"causal_effects.csv 缺少列: {required_columns - set(effects.columns)}")
    if len(effects) != EXPECTED_QTL_COUNT:
        raise ValueError(
            f"此实验要求 {EXPECTED_QTL_COUNT} 个真实 QTL，实际为 {len(effects)}"
        )

    one_based = effects["snp_index"].to_numpy(dtype=np.int64)
    if len(np.unique(one_based)) != len(one_based) or (one_based < 1).any():
        raise ValueError("模拟真值中的 snp_index 必须是互异的正整数")
    expected_names = np.array([f"SNP{index}" for index in one_based])
    if not np.array_equal(effects["snp_name"].to_numpy(dtype=str), expected_names):
        raise ValueError("snp_name 与 snp_index 不一致，拒绝使用不明确的真值索引")

    snp_dim = np.load(genotype_path, mmap_mode="r", allow_pickle=False).shape[1]
    indices = np.sort(one_based - 1)
    if (indices >= snp_dim).any():
        raise ValueError(f"真实 QTL 索引越界：基因型只有 {snp_dim} 列")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = np.load(output_path, allow_pickle=False)
        if not np.array_equal(existing, indices):
            raise FileExistsError(f"已有不同的 oracle QTL 索引文件: {output_path}")
    else:
        np.save(output_path, indices)
    return indices


def build_config(
    data_dir: Path,
    output_root: Path,
    exp_name: str,
    epochs: int,
) -> TFPPConfig:
    """读取数据 manifest，并构建使用真实 QTL 预选索引的 TFPP 配置。"""
    manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    split_seed = manifest["split"]["seed"]
    selected_path = output_root / exp_name / "oracle_selection" / "true_qtl_indices_zero_based.npy"
    selected = extract_true_qtl_indices(data_dir, selected_path)
    print(
        "[Oracle diagnostic] 使用模拟真值预选 "
        f"{len(selected)} 个 QTL；原始 SNP 维度为 {manifest['genotype_shape'][1]}。"
    )

    return TFPPConfig(
        exp_name=exp_name,
        output_root=str(output_root),
        snp_file=str(data_dir / "genotype.npy"),
        pheno_dir=str(data_dir / "cleaned_pheno"),
        phenotype_names=["sim_pheno"],
        time_coordinates=manifest["time_coordinates"],
        split_file=str(data_dir / "splits" / f"split_seed{split_seed}.npz"),
        sample_ids_file=str(data_dir / "sample_ids.csv"),
        preselected_snp_file=str(selected_path),
        # 禁用内部监督筛选；仅使用上方的真实 QTL oracle 预选。
        screen_k=0,
        gwas_k=0,
        screen_multivariate=True,
        # 与 train_sim.py 保持一致的训练与模型设置。
        epochs=epochs,
        lr=0.0005,
        min_lr=1e-6,
        weight_decay=0.0001,
        batch_size=64,
        patience=20,
        seed=73,
        dropout=0.0,
        device="cuda",
        num_workers=8,
        use_warmup=True,
        warmup_epochs=10,
        warmup_start_factor=0.1,
        warmup_end_factor=1.0,
        d_model=32,
        num_heads=4,
        num_layers=1,
        num_embeddings=4,
        d_ff=128,
        degree=5,
        ablation_mode="add",
        loss_type="mse",
        pcc_weight=0.1,
        norm_method="residual_global",
        scaler_type="zscore",
    )


def train_oracle_qtl_experiment(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    exp_name: str = DEFAULT_EXP_NAME,
    epochs: int = 200,
) -> dict:
    """运行真实 QTL oracle 预选后的标准 TFPP 训练与一次测试评估。"""
    config = build_config(data_dir, output_root, exp_name, epochs)
    trainer = TFPPTrainer(config=config)
    return trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exp_name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    train_oracle_qtl_experiment(
        data_dir=args.data_dir,
        output_root=args.output_root,
        exp_name=args.exp_name,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

"""在 Oracle 20-QTL 模拟输入上运行传统模型、RF 与 XGBoost。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trainer.Baseline_Trainer import (
    SUPPORTED_MODELS,
    BaselineConfig,
    BaselineTrainer,
    ORACLE_SCENARIO_LABEL,
)

DEFAULT_DATA_DIR = Path(
    "dataset/experiments/P5000_NTR400_H0.4/rep_01"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/baseline_comparison/oracle_qtl20/P5000_NTR400_H0.4/rep_01"
)
EXPECTED_QTL_COUNT = 20


def _load_true_qtl_indices(data_dir: Path) -> tuple[int, ...]:
    """读取真值文件中的 20 个 1-based QTL 索引并转为模型列索引。"""
    truth_path = data_dir / "truth" / "causal_effects.csv"
    if not truth_path.is_file():
        raise FileNotFoundError(f"缺少 Oracle 真值文件: {truth_path}")
    effects = pd.read_csv(truth_path)
    if "snp_index" not in effects or len(effects) != EXPECTED_QTL_COUNT:
        raise ValueError("Oracle baseline 要求 causal_effects.csv 恰好记录20个 QTL")
    indices = effects["snp_index"].to_numpy(dtype=int) - 1
    if (indices < 0).any() or len(set(indices.tolist())) != EXPECTED_QTL_COUNT:
        raise ValueError("causal_effects.csv 的 snp_index 必须是互异的正整数")
    return tuple(sorted(indices.tolist()))


def run_all_baselines(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = 73,
    models: tuple[str, ...] = SUPPORTED_MODELS,
) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    genotype_file = data_dir / "sim_snp.npy"
    phenotype_file = data_dir / "sim_pheno.csv"
    sample_ids_file = data_dir / "sample_ids.csv"
    split_file = data_dir / "splits" / f"split_seed{seed}.npz"
    config = BaselineConfig(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=seed,
        models=models,
        scenario_label=ORACLE_SCENARIO_LABEL,
        genotype_file=str(genotype_file),
        phenotype_file=str(phenotype_file),
        sample_ids_file=str(sample_ids_file),
        split_file=str(split_file),
        preselected_snp_indices=_load_true_qtl_indices(data_dir),
    )
    return BaselineTrainer(config).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
    )
    args = parser.parse_args()
    summary = run_all_baselines(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        models=tuple(args.models),
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

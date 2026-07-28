"""导出测试集预测结果为 CSV, 格式与原始 Test_Phenotype.csv 一致。

用法:
    python export_predictions.py [--model_dir <path>] [--output <path>]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.load_model_from_checkpoint import (
    denormalize_predictions,
    load_model_from_checkpoint,
    load_scaler_and_mean_curve,
    load_test_csvs,
    run_inference,
)

_MODEL_DIRNAME_RE = re.compile(
    r"^(?P<scenario>S\d+)_P(?P<p>\d+)_N(?P<n>\d+)_H(?P<h>[0-9.]+)_Rep(?P<rep>\d+)"
)

_DATA_DIR = Path("dataset/simulated_data")


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="导出测试集预测结果为 CSV")
    p.add_argument(
        "--model_dir",
        type=str,
        default="../outputs/sim_tfpp/S1_P10000_N1600_H0.8_Rep01_abl-wide",
        help="模型输出目录",
    )
    p.add_argument("--device", type=str, default="cuda", help="推理设备 (cuda/cpu)")
    return p.parse_args()


def _parse_meta(dirname: str) -> dict:
    """从模型目录名中解析实验参数 (scenario, p, n, h, rep)。"""
    m = _MODEL_DIRNAME_RE.match(dirname)
    if not m:
        raise ValueError(f"无法从目录名 '{dirname}' 解析实验参数")
    return {
        "scenario": m.group("scenario"),
        "p": int(m.group("p")),
        "n": int(m.group("n")),
        "h": float(m.group("h")),
        "rep": int(m.group("rep")),
    }


def _load_pheno_df(meta: dict) -> pd.DataFrame:
    """加载测试集 phenotype CSV 以获取样本 ID 与列名, 用于输出格式化。"""
    group_dir = _DATA_DIR / meta["scenario"] / f"P{meta['p']}_N{meta['n']}_H{meta['h']}"
    prefix = (
        f"DynaGS_{meta['scenario']}_P{meta['p']}_H{meta['h']}"
        f"_N{meta['n']}_Rep{meta['rep']:02d}"
    )
    pheno_path = group_dir / f"{prefix}_Test_Phenotype.csv"
    if not pheno_path.exists():
        raise FileNotFoundError(f"缺失文件: {pheno_path}")
    return pd.read_csv(pheno_path, index_col=0)


def main():
    """导出测试集预测结果为 CSV 的主入口。"""
    args = parse_args()
    device = args.device
    if device == "cuda" and not __import__("torch").cuda.is_available():
        device = "cpu"

    model_dir = Path(args.model_dir)

    # 1. 从目录名解析实验参数
    meta = _parse_meta(model_dir.name)
    print(
        f"实验: {meta['scenario']} P={meta['p']} N={meta['n']} "
        f"H={meta['h']} Rep={meta['rep']:02d}"
    )

    # 2. 加载模型 & scaler (复用 plot_utils)
    model, _ = load_model_from_checkpoint(model_dir, device)
    scaler, mean_curve = load_scaler_and_mean_curve(model_dir)

    # 3. 加载测试数据 (复用 plot_utils), pheno_df 仅用于输出格式化
    X_test, _ = load_test_csvs(
        meta["scenario"], meta["p"], meta["n"], meta["h"], meta["rep"]
    )
    pheno_df = _load_pheno_df(meta)
    sample_ids = pheno_df.index.tolist()
    cols = pheno_df.columns

    print(
        f"测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} SNP, {len(cols)} 个时间点"
    )

    # 4. 推理 + 反归一化 (复用 plot_utils)
    pred_norm = run_inference(model, X_test, device)
    pred = denormalize_predictions(pred_norm, scaler, mean_curve)  # [N, 1, T]

    # 5. 构造 CSV (格式与 Test_Phenotype.csv 一致)
    df_out = pd.DataFrame(pred[:, 0, :], index=sample_ids, columns=cols)
    df_out.index.name = None

    out_path = model_dir / f"{model_dir.name}_predictions.csv"
    df_out.to_csv(out_path)
    print(f"已导出: {out_path}  ({df_out.shape[0]} 行 × {df_out.shape[1]} 列)")


if __name__ == "__main__":
    main()

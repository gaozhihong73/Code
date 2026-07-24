"""模拟数据 TFPP 训练入口

支持三种输入粒度:
1) 单个 Rep 的某个 Train_Genotype.csv 文件 -> 只跑这一个 Rep
2) 一个实验组目录 (P*_N*_H*) -> 跑该组下所有 Rep (10个), 组内汇总
3) S0 / S1 / simulated_data 根目录 -> 递归扫描所有组, 组内+跨组汇总

输出:
- 每个独立 Rep 的训练结果在 outputs/sim_tfpp/{exp_name}/
- 多 Rep 汇总在 outputs/sim_tfpp/summary/
    - {exp_group}_per_rep.csv   每行: rep, rmse, mae, pcc, r2 (反归一化)
    - {exp_group}_summary.csv   单行: rmse_mean±std, mae_.., pcc_.., r2_.. (跨10个Rep)
    - all_experiments_summary.csv  跨所有组汇总

参考: Baseline_Train.py 的多轮结果汇总写法。
"""

import argparse
import re
import traceback
from pathlib import Path

import pandas as pd

from config import SimTFPPConfig
from trainer import SimTFPPTrainer

# 文件命名规律: DynaGS_{S}_P{P}_H{H}_N{N}_Rep{rep}_{split}_{type}.csv
_FILENAME_RE = re.compile(
    r"^DynaGS_(?P<scenario>S\d+)_P(?P<p>\d+)_H(?P<h>[0-9.]+)_N(?P<n>\d+)_Rep(?P<rep>\d+)_(?P<split>Train|Val|Test)_(?P<type>Genotype|Phenotype|TrueGV)\.csv$"
)
# 实验组目录名: P{p}_N{n}_H{h}
_GROUPDIR_RE = re.compile(r"^P(?P<p>\d+)_N(?P<n>\d+)_H(?P<h>[0-9.]+)$")


def parse_args():
    parser = argparse.ArgumentParser(description="DynaGS 模拟数据 TFPP 训练入口")
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="/root/DynaGS/dataset/simulated_data/S1/P10000_N1600_H0.8/DynaGS_S1_P10000_H0.8_N1600_Rep01_Train_Genotype.csv",
        # default="/root/DynaGS/dataset/simulated_data/S0/P10000_N1600_H0.0/DynaGS_S0_P10000_H0.0_N1600_Rep01_Train_Genotype.csv",
        help="输入路径: 一个 *_*_RepXX_Train_Genotype.csv 文件, "
        "或一个 P*_N*_H* 实验组目录, "
        "或 S0/S1/simulated_data 根目录",
    )
    # 训练超参 (透传)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32, help="必须为 2 的倍数")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_embeddings", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument(
        "--degree", type=int, default=5, help="Legendre 多项式阶数, 0=纯加性"
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="residual_global",
        choices=["global", "timepoint", "residual_global"],
    )
    parser.add_argument(
        "--scaler_type",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "robust"],
    )
    parser.add_argument(
        "--use_warmup", type=lambda s: s.lower() == "true", default=True
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "wide", "deep", "add"],
        help="模型消融模式: none=完整, wide=只宽通路, deep=只深通路, add=去掉门控直接相加",
    )
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--is_save", type=lambda s: s.lower() == "true", default=True)
    # 损失函数
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "mse_pcc", "mse_offset"],
    )
    parser.add_argument("--offset_weight", type=float, default=0.5)
    parser.add_argument("--pcc_weight", type=float, default=0.3)
    parser.add_argument(
        "--l1_wide_weight",
        type=float,
        default=1e-5,
        help="宽通路 L1 正则系数, 抑制 p >> n 时的 SNP 噪声过拟合, 0=关闭",
    )
    parser.add_argument(
        "--screen_k",
        type=int,
        default=100,
        help="DC-SIS 筛选 SNP 数 (仅用训练集计算距离相关系数, 0=关闭)",
    )
    parser.add_argument(
        "--gwas_k",
        type=int,
        default=100,
        help="GWAS 筛选 SNP 数 (PCA 协变量 + Mean/AUC/Slope 三性状 min-P, 0=关闭)",
    )
    return parser.parse_args()


def _build_config_kwargs(args) -> dict:
    """从 CLI 提取训练超参, 复用于每个 Rep"""
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "degree": args.degree,
        "norm_method": args.norm_method,
        "scaler_type": args.scaler_type,
        "use_warmup": args.use_warmup,
        "warmup_epochs": args.warmup_epochs,
        "num_workers": args.num_workers,
        "is_save_training_info": args.is_save,
        "ablation_mode": args.ablation_mode,
        "loss_type": args.loss_type,
        "offset_weight": args.offset_weight,
        "pcc_weight": args.pcc_weight,
        "l1_wide_weight": args.l1_wide_weight,
        "screen_k": args.screen_k,
        "gwas_k": args.gwas_k,
    }


def _group_id(scenario: str, p: int, n: int, h: str) -> str:
    """构造实验组标识, 与目录路径一致"""
    return f"{scenario}/P{p}_N{n}_H{h}"


def _exp_name(rep_meta: dict, ablation_mode: str = "none") -> str:
    """构造实验名 (= 输出子目录); 非默认消融时加后缀以避免覆盖"""
    base = f"{rep_meta['scenario']}_P{rep_meta['p']}_N{rep_meta['n']}_H{rep_meta['h']}_Rep{rep_meta['rep']}"
    if ablation_mode and ablation_mode != "none":
        return f"{base}_abl-{ablation_mode}"
    return base


def _find_rep_files(group_dir: Path, rep_meta: dict) -> dict:
    """根据命名规律在 group_dir 中定位该 Rep 的 6 个 csv"""
    scenario = rep_meta["scenario"]
    p = rep_meta["p"]
    h = rep_meta["h"]
    n = rep_meta["n"]
    rep = rep_meta["rep"]
    prefix = f"DynaGS_{scenario}_P{p}_H{h}_N{n}_Rep{rep}"

    files = {
        "train_snp_csv": group_dir / f"{prefix}_Train_Genotype.csv",
        "train_pheno_csv": group_dir / f"{prefix}_Train_Phenotype.csv",
        "val_snp_csv": group_dir / f"{prefix}_Val_Genotype.csv",
        "val_pheno_csv": group_dir / f"{prefix}_Val_Phenotype.csv",
        "test_snp_csv": group_dir / f"{prefix}_Test_Genotype.csv",
        "test_pheno_csv": group_dir / f"{prefix}_Test_Phenotype.csv",
    }
    for k, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"缺失文件 {k}: {path}")
    return {k: str(v) for k, v in files.items()}


def _discover_reps(data_path: Path) -> list[tuple[dict, Path]]:
    """返回 [(rep_meta, group_dir), ...], rep_meta 含 scenario/p/h/n/rep

    支持三种粒度:
    - data_path 是 *_*_RepXX_Train_Genotype.csv -> 1 个 Rep
    - data_path 是 P*_N*_H* 目录 -> 该目录下所有 Rep
    - data_path 是 S0/S1/simulated_data 目录 -> 递归所有组 × 所有 Rep
    """
    results = []

    if data_path.is_file():
        # 粒度1: 单文件
        m = _FILENAME_RE.match(data_path.name)
        if not m or m.group("split") != "Train" or m.group("type") != "Genotype":
            raise ValueError(
                f"输入文件命名不符合预期 (需要 *_RepXX_Train_Genotype.csv): {data_path.name}"
            )
        rep_meta = {
            "scenario": m.group("scenario"),
            "p": int(m.group("p")),
            "h": m.group("h"),
            "n": int(m.group("n")),
            "rep": m.group("rep"),
        }
        results.append((rep_meta, data_path.parent))

    elif data_path.is_dir() and _GROUPDIR_RE.match(data_path.name):
        # 粒度2: 实验组目录
        # scenario = data_path.parent.name  # 上一级目录名是 S0/S1
        group_reps = []
        for f in sorted(data_path.glob("DynaGS_*_Rep*_Train_Genotype.csv")):
            m = _FILENAME_RE.match(f.name)
            if not m:
                continue
            rep_meta = {
                "scenario": m.group("scenario"),
                "p": int(m.group("p")),
                "h": m.group("h"),
                "n": int(m.group("n")),
                "rep": m.group("rep"),
            }
            group_reps.append((rep_meta, data_path))
        if not group_reps:
            raise FileNotFoundError(f"实验组目录下未找到任何 Rep: {data_path}")
        results.extend(group_reps)

    elif data_path.is_dir():
        # 粒度3: 多组根目录 (S0 / S1 / simulated_data)
        for group_dir in sorted(data_path.rglob("P*_N*_H*")):
            if not group_dir.is_dir():
                continue
            if not _GROUPDIR_RE.match(group_dir.name):
                continue
            for f in sorted(group_dir.glob("DynaGS_*_Rep*_Train_Genotype.csv")):
                m = _FILENAME_RE.match(f.name)
                if not m:
                    continue
                rep_meta = {
                    "scenario": m.group("scenario"),
                    "p": int(m.group("p")),
                    "h": m.group("h"),
                    "n": int(m.group("n")),
                    "rep": m.group("rep"),
                }
                results.append((rep_meta, group_dir))
        if not results:
            raise FileNotFoundError(f"根目录下未找到任何实验组: {data_path}")

    else:
        raise FileNotFoundError(f"路径不存在: {data_path}")

    return results


def _run_single_rep(
    rep_meta: dict, group_dir: Path, cli_kwargs: dict, summary_root: Path
) -> dict:
    """运行单个 Rep, 返回其测试集反归一化指标 (扁平字典)"""
    files = _find_rep_files(group_dir, rep_meta)

    ablation_mode = cli_kwargs.get("ablation_mode", "none")
    exp_name = _exp_name(rep_meta, ablation_mode=ablation_mode)
    group_id = _group_id(
        rep_meta["scenario"], rep_meta["p"], rep_meta["n"], rep_meta["h"]
    )

    config = SimTFPPConfig(
        exp_name=exp_name,
        exp_group=group_id,
        rep_name=f"Rep{rep_meta['rep']}",
        **files,
        **cli_kwargs,
    )

    print(f"\n{'=' * 80}")
    print(f"开始训练: {exp_name}")
    print(f"{'=' * 80}")

    trainer = SimTFPPTrainer(config=config)
    results = trainer.train()

    # 提取反归一化整体测试指标
    denorm_overall = results["test_results"]["denormalized"]["overall"]
    denorm_pheno = results["test_results"]["denormalized"]["phenotypes"]

    row = {
        "exp_group": group_id,
        "rep": f"Rep{rep_meta['rep']}",
        "scenario": rep_meta["scenario"],
        "P": rep_meta["p"],
        "N": rep_meta["n"],
        "H": rep_meta["h"],
        "best_epoch": results["best_epoch"],
        "training_time_min": results["training_time"] / 60,
    }
    # 整体指标 (反归一化)
    for metric in ["mse", "rmse", "mae", "pcc", "r2"]:
        if metric in denorm_overall:
            row[metric] = float(denorm_overall[metric])

    # 各表型各时间点指标 (反归一化) - 模拟数据只有1个表型, 展平记录
    for p_name, p_metrics in denorm_pheno.items():
        for metric in ["mse", "rmse", "mae", "pcc", "r2"]:
            if metric in p_metrics:
                row[f"{p_name}_{metric}"] = float(p_metrics[metric])

    return row


def _summarize_group(
    per_rep_rows: list[dict], group_id: str, summary_root: Path
) -> dict:
    """对单个实验组内的多个 Rep 汇总 mean±std, 参照 Baseline_Train.py"""
    df = pd.DataFrame(per_rep_rows)
    per_rep_path = summary_root / f"{group_id.replace('/', '_')}_per_rep.csv"
    df.to_csv(per_rep_path, index=False)
    print(f"\n📄 该实验组每 Rep 明细已保存至: {per_rep_path}")

    # 跨 Rep 汇总 (整体指标)
    summary_row = {"exp_group": group_id, "n_reps": len(df)}
    metric_cols = ["mse", "rmse", "mae", "pcc", "r2"]
    for m in metric_cols:
        if m in df.columns:
            vals = df[m].dropna().astype(float)
            summary_row[f"{m}_mean"] = float(vals.mean())
            summary_row[f"{m}_std"] = float(vals.std(ddof=0))

    summary_df = pd.DataFrame([summary_row])
    summary_path = summary_root / f"{group_id.replace('/', '_')}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 该实验组跨 Rep 汇总已保存至: {summary_path}")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    return summary_row


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    cli_kwargs = _build_config_kwargs(args)

    # 1. 发现要跑的所有 Rep
    rep_list = _discover_reps(data_path)
    print(f"\n🔍 共发现 {len(rep_list)} 个 Rep 需要训练")

    # 2. 按实验组聚合
    groups: dict[str, list[tuple[dict, Path]]] = {}
    for rep_meta, group_dir in rep_list:
        gid = _group_id(
            rep_meta["scenario"], rep_meta["p"], rep_meta["n"], rep_meta["h"]
        )
        groups.setdefault(gid, []).append((rep_meta, group_dir))

    print(f"📦 共 {len(groups)} 个实验组\n")

    summary_root = Path("outputs/sim_tfpp/summary")
    summary_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    all_per_rep_rows = []

    # 3. 逐组逐 Rep 训练
    for gid, reps_in_group in groups.items():
        print(f"\n{'#' * 80}")
        print(f"# 实验组: {gid}  ({len(reps_in_group)} 个 Rep)")
        print(f"{'#' * 80}")

        group_per_rep_rows = []
        for rep_meta, group_dir in reps_in_group:
            try:
                row = _run_single_rep(rep_meta, group_dir, cli_kwargs, summary_root)
                group_per_rep_rows.append(row)
                all_per_rep_rows.append(row)
            except Exception:
                print(f"❌ Rep {rep_meta['rep']} (group {gid}) 训练失败:")
                traceback.print_exc()

        # 4. 组内汇总 (只有多个 Rep 时才需要)
        if len(rep_list) > 1 and group_per_rep_rows:
            summary_row = _summarize_group(group_per_rep_rows, gid, summary_root)
            all_summary_rows.append(summary_row)

    # 5. 跨组全局汇总 (只有多组实验时才需要)
    if len(groups) > 1 and all_summary_rows:
        all_summary_df = pd.DataFrame(all_summary_rows)
        all_summary_path = summary_root / "all_experiments_summary.csv"
        all_summary_df.to_csv(all_summary_path, index=False)
        print(f"\n🌐 跨实验组全局汇总已保存至: {all_summary_path}")
        print("-" * 80)
        print(all_summary_df.to_string(index=False))

    # 单个 Rep 模式下, 仅展示该 Rep 的结果, 不做汇总
    elif len(rep_list) == 1:
        print("\n✅ 单 Rep 训练完成, 详细结果见对应日志与训练历史目录")
        if all_per_rep_rows:
            df = pd.DataFrame(all_per_rep_rows)
            one_path = (
                summary_root
                / f"{all_per_rep_rows[0]['exp_group'].replace('/', '_')}_{all_per_rep_rows[0]['rep']}.csv"
            )
            df.to_csv(one_path, index=False)
            print("   主要测试指标 (反归一化):")
            print(df.to_string(index=False))

    print("\n✅ 所有训练完成!")


if __name__ == "__main__":
    main()

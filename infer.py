"""DynaGS: TFPP 模型推理脚本。

支持两种预测模式:
1. 外部文件模式: 传入 --input_snp 指定待预测的 SNP 数据 (.npy)
2. 测试集模式: 不传 --input_snp, 自动复现训练时的测试集进行预测

模型结构参数通过 utils/model_loader.py 自动从 checkpoint 推测, 无需手动指定。
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataloader.TFPP_DataLoader import TFPPDataset, get_dataloader
from config import TFPPConfig
from utils.Genotype import (
    dosage_to_model_codes,
    model_codes_to_dosage,
    model_code_tensor,
    prepare_screening_dosage,
)
from utils import (
    compute_cache_key,
    denormalize_phenotype,
    load_scalers,
    load_tfpp_model,
    screen_snps,
)


def _str_to_bool(s: str) -> bool:
    """将 'true'/'false' 字符串正确解析为布尔值，避免 argparse type=bool 的陷阱。"""
    return s.strip().lower() == "true"


def parse_args():
    """解析推理脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="DynaGS: TFPP 模型推理脚本 (支持外部文件或默认测试集)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ================= 核心必填路径 =================
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "训练好的模型路径。支持两种形式: "
            "1) 实验目录 (如 outputs/tfpp/exp_name), 自动定位 models/best_model.pt; "
            "2) 直接指定 .pt 文件路径"
        ),
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=None,
        help=(
            "归一化器文件路径 (.pkl)。若不指定, 自动从 --model_path 推导: "
            "{exp_dir}/scalers/scaler.pkl"
        ),
    )

    # ================= 预测模式控制 =================
    parser.add_argument(
        "--input_snp",
        type=str,
        default=None,
        help=(
            "外部待预测的 SNP 数据路径 (.npy, 形状 [样本数, SNP数])。"
            "若不传入此项, 则自动复现训练时的测试集进行预测"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="预测结果 CSV 保存路径。默认自动生成: outputs/infer/inference_YYYYMMDD_HHMMSS.csv",
    )

    # ================= 测试集复现参数 (不传 --input_snp 时使用) =================
    parser.add_argument(
        "--snp_file",
        type=str,
        default="dataset/maize/snp/snp_50000.npy",
        help="训练时使用的 SNP 数据文件路径 (需与训练时一致以复现相同的数据划分)",
    )
    parser.add_argument(
        "--pheno_dir",
        type=str,
        default="dataset/maize/cleaned_pheno",
        help="训练时使用的表型数据目录 (需与训练时一致)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (需与训练时一致, 用于复现相同的数据划分)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="验证集比例 (需与训练时一致)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例 (需与训练时一致)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="随机种子 (必须与训练时一致, 否则数据划分不同导致测试集不匹配)",
    )
    parser.add_argument(
        "--scaler_type",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "robust"],
        help="归一化缩放器类型 (需与训练时一致, 影响测试集的归一化方式)",
    )

    # ================= SNP 特征筛选 (需与训练时一致) =================
    parser.add_argument(
        "--screen_k",
        type=int,
        default=10000,
        help="DC-SIS 筛选保留的 SNP 数目 (需与训练时一致, 0=训练时未筛选)",
    )
    parser.add_argument(
        "--gwas_k",
        type=int,
        default=10000,
        help="GWAS 筛选保留的 SNP 数目 (需与训练时一致, 0=训练时未筛选)",
    )
    parser.add_argument(
        "--screen_multivariate",
        type=_str_to_bool,
        default=TFPPConfig.model_fields["screen_multivariate"].default,
        help=(
            "DC-SIS 多元模式: true=多表型拼接联合筛选, false=逐表型并集"
            " (需与训练时一致)"
        ),
    )
    parser.add_argument(
        "--screen_missing_strategy",
        choices=["error", "mode"],
        default=TFPPConfig.model_fields["screen_missing_strategy"].default,
        help="筛选缺失策略，需与训练一致；本步骤不在推理中拟合填补器",
    )

    # ================= 推理运行时参数 =================
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="推理批次大小 (不影响结果, 仅影响显存占用和速度)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader 工作线程数",
    )

    return parser.parse_args()


def predict():
    """加载模型与归一化器，对 SNP 数据执行推理并导出预测结果 CSV。"""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("outputs/infer").mkdir(parents=True, exist_ok=True)

    output_path = args.output or (
        f"outputs/infer/inference_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # 自动推导 scaler_path: {exp_dir}/scalers/scaler.pkl
    if args.scaler_path is None:
        model_p = Path(args.model_path)
        exp_dir = model_p if model_p.is_dir() else model_p.parent.parent
        args.scaler_path = str(exp_dir / "scalers" / "scaler.pkl")

    # 1. 加载模型 (自动推测结构参数)
    print(f"[*] 正在加载模型: {args.model_path}")
    model, model_info = load_tfpp_model(args.model_path, device=str(device))
    if model_info.get("wide_input_mode") == "dosage_v1" and args.input_snp is None:
        raise ValueError(
            "新普通训练模型必须复用 preprocessing/split.npz 和 screened_indices.npy。"
            "测试集推理入口尚未迁移，禁止重新使用全体表型筛选；"
            "可先按保存索引准备 --input_snp，模型会加载训练期 Wide 填补参数。"
        )
    phenotype_names = model_info["phenotype_names"]
    print(
        f"[*] 模型结构: d_model={model_info['d_model']}, "
        f"num_layers={model_info['num_layers']}, "
        f"degree={model_info['degree']}, "
        f"ablation_mode={model_info['ablation_mode']}"
    )
    print(f"[*] 表型列表: {phenotype_names}")
    print(
        f"[*] 筛选配置: screen_multivariate={args.screen_multivariate}, "
        f"screen_missing_strategy={args.screen_missing_strategy}"
    )

    # 2. 加载归一化器
    print(f"[*] 正在加载归一化器: {args.scaler_path}")
    scalers_pkg = load_scalers(args.scaler_path)
    norm_method = scalers_pkg["norm_method"]
    print(f"[*] 归一化策略: {norm_method}")

    # 3. 准备数据
    if args.input_snp:
        # 模式 A: 外部文件预测
        print(f"[*] 外部输入模式: {args.input_snp}")
        if not Path(args.input_snp).exists():
            raise FileNotFoundError(f"SNP 文件不存在: {args.input_snp}")
        external_snp_data = np.load(args.input_snp)
        snp_dim = external_snp_data.shape[1]
        expected_dim = model_info["snp_dim"]
        if snp_dim != expected_dim:
            raise ValueError(
                f"外部 SNP 维度 ({snp_dim}) 与模型期望维度 ({expected_dim}) 不匹配。"
                f"请确认数据经过相同的特征筛选 (screen_k={args.screen_k}, gwas_k={args.gwas_k})"
            )
        test_loader = None
    else:
        # 模式 B: 复现训练时的测试集
        print(f"[*] 测试集模式: 复现训练时的数据划分 (seed={args.seed})...")
        dataset = TFPPDataset(args.snp_file, args.pheno_dir, phenotype_names)

        # 与训练时相同的 SNP 筛选流程
        if args.screen_k > 0 or args.gwas_k > 0:
            cache_key = compute_cache_key(
                snp_file=args.snp_file,
                pheno_source=args.pheno_dir,
                phenotype_names=phenotype_names,
                screen_k=args.screen_k,
                gwas_k=args.gwas_k,
                seed=args.seed,
                multivariate=args.screen_multivariate,
                missing_strategy=args.screen_missing_strategy,
            )
            screened_idx = screen_snps(
                prepare_screening_dosage(
                    model_codes_to_dosage(dataset.snp),
                    missing_strategy=args.screen_missing_strategy,
                ),
                dataset.phenotype,
                screen_k=args.screen_k,
                gwas_k=args.gwas_k,
                multivariate=args.screen_multivariate,
                cache_key=cache_key,
            )
            if screened_idx is not None:
                dataset.snp = dataset.snp[:, screened_idx]

        snp_dim = dataset.snp.shape[1]
        expected_dim = model_info["snp_dim"]
        if snp_dim != expected_dim:
            raise ValueError(
                f"筛选后 SNP 维度 ({snp_dim}) 与模型期望维度 ({expected_dim}) 不匹配。"
                f"请检查 --screen_k 和 --gwas_k 是否与训练时一致"
            )

        _, _, test_loader, _ = get_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            norm_method=norm_method,
            scaler_type=args.scaler_type,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=str(device),
            num_workers=args.num_workers,
        )
        if test_loader is None:
            raise ValueError("测试集加载失败")

    # 4. 执行推理并反归一化
    results = []

    if args.input_snp:
        # 外部文件明确是原始剂量；缺失映射到模型类别0，禁止猜测类别码输入。
        snp_tensor = model_code_tensor(dosage_to_model_codes(external_snp_data)).to(device)
        with torch.no_grad():
            preds_norm = model(snp_tensor)

        preds_denorm = denormalize_phenotype(preds_norm.cpu(), scalers_pkg)
        num_samples, num_phenos, num_times = preds_denorm.shape

        for i in range(num_samples):
            for p_idx, p_name in enumerate(phenotype_names):
                row = {"sample_idx": i, "phenotype": p_name}
                for t_idx in range(num_times):
                    row[f"T{t_idx + 1}"] = preds_denorm[i, p_idx, t_idx].item()
                results.append(row)
    else:
        # 测试集: 同时输出预测值和真实值
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in test_loader:  # type: ignore
                X = X.to(device)
                preds_norm = model(X)
                all_preds.append(preds_norm.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        preds_denorm = denormalize_phenotype(all_preds, scalers_pkg)
        targets_denorm = denormalize_phenotype(all_targets, scalers_pkg)

        num_samples, num_phenos, num_times = preds_denorm.shape
        for i in range(num_samples):
            for p_idx, p_name in enumerate(phenotype_names):
                row = {"sample_idx": i, "phenotype": p_name}
                for t_idx in range(num_times):
                    row[f"T{t_idx + 1}_Pred"] = preds_denorm[i, p_idx, t_idx].item()
                    row[f"T{t_idx + 1}_True"] = targets_denorm[i, p_idx, t_idx].item()
                results.append(row)

    # 5. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[*] 预测完成! 结果已保存至: {output_path}")
    print(
        f"[*] 共 {len(df)} 条记录 ({len(df) // len(phenotype_names)} 样本 × {len(phenotype_names)} 表型)"
    )


if __name__ == "__main__":
    predict()

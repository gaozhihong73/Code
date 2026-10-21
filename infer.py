"""TFPP 推理：只读取训练产物，绝不筛选或拟合测试数据。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import TFPPConfig
from utils.Genotype import dosage_to_model_codes, model_code_tensor
from utils.Inference import file_sha256, prepare_inference_input, transform_phenotypes
from utils.Normalization import denormalize_phenotype, load_scalers
from utils.model_loader import load_tfpp_model


def _str_to_bool(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("请使用 true 或 false")
    return value.lower() == "true"


def parse_args(argv=None):
    """输入阶段必须由用户声明；预处理配置从实验目录读取。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="实验目录或 models/*.pt")
    parser.add_argument("--input_snp", help="外部原始剂量 .npy，缺失为 -1/NaN")
    parser.add_argument("--input_stage", choices=["raw", "screened"],
                        help="外部输入必填：raw 为原始全维，screened 为训练索引顺序的已筛选列")
    parser.add_argument("--snp_file", default="dataset/maize/snp/snp_50000.npy",
                        help="测试集复现使用的原始全维剂量文件，行序必须与训练一致")
    parser.add_argument("--pheno_dir", default=None,
                        help="可选测试表型目录；仅用保存的 scaler.transform，不拟合")
    parser.add_argument("--output", help="输出 CSV；同时保存 .metadata.json 推理记录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cpu 或 cuda；默认自动选择")
    parser.add_argument("--legacy", action="store_true",
                        help="显式允许旧 checkpoint，仅限外部 screened 输入；禁止用于论文正式汇总")
    parser.add_argument("--scaler_path", help="仅 legacy 模式允许指定旧 scaler")
    # 兼容旧 CLI 的配置开关；实际值只从训练产物读取，不再执行筛选。
    parser.add_argument("--screen_multivariate", type=_str_to_bool,
                        default=TFPPConfig.model_fields["screen_multivariate"].default,
                        help="兼容旧 CLI；实际筛选配置从训练产物读取，本参数不重新筛选")
    args = parser.parse_args(argv)
    if args.input_snp and args.input_stage is None:
        parser.error("外部 SNP 必须显式提供 --input_stage raw 或 screened")
    if not args.input_snp and args.input_stage is not None:
        parser.error("--input_stage 仅用于 --input_snp 外部输入")
    if args.batch_size < 1:
        parser.error("--batch_size 必须为正数")
    return args


def prepare_test_data(snp_file, artifacts, pheno_dir=None):
    """按保存的 test_idx 取行；标签读取和 transform 不影响预测输入。"""
    raw = np.load(snp_file, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[0] != artifacts.split.metadata["n_samples"]:
        raise ValueError("原始 SNP 个体数与保存的 split 不一致；需保持训练时行顺序")
    test_idx = artifacts.split.test_idx
    if not len(test_idx):
        raise ValueError("保存的 split 没有测试个体")
    codes = prepare_inference_input(raw[test_idx], artifacts, input_stage="raw")
    targets = None
    if pheno_dir is not None:
        names = artifacts.config["phenotype_names"]
        times = artifacts.config["time_points"]
        blocks = []
        for name in names:
            frame = pd.read_csv(Path(pheno_dir) / f"{name}.csv")
            if frame.shape != (len(raw), times):
                raise ValueError(f"{name}.csv 应有 {len(raw)} 个体、{times} 时间点，且行序与训练一致")
            blocks.append(frame.iloc[test_idx].to_numpy(dtype=np.float32))
        targets = transform_phenotypes(np.stack(blocks, axis=1), artifacts.scalers)
    return codes, targets, test_idx


def predict(argv=None):
    """按保存的实验配置批量推理，用同一训练 scaler 反归一化。"""
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, info = load_tfpp_model(args.model_path, device=device, legacy=args.legacy)
    artifacts = info["artifacts"]
    targets = None
    if info["legacy"]:
        if not args.input_snp or args.input_stage != "screened":
            raise ValueError("legacy 仅支持显式外部 --input_stage screened；不能猜测旧测试划分或重筛选")
        if info["num_embeddings"] != 4:
            raise ValueError("旧三类别 checkpoint 不支持此剂量转四类别入口；请使用对应旧版本诊断流程，不能混用编码")
        path = Path(info["ckpt_path"]).parent.parent / "scalers" / "scaler.pkl"
        scalers = load_scalers(args.scaler_path or path)
        if scalers.get("phenotype_names") != info["phenotype_names"]:
            raise ValueError("legacy scaler 表型顺序与模型不一致")
        raw = np.load(args.input_snp, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != info["snp_dim"]:
            raise ValueError("legacy screened 输入维度与模型不一致")
        codes = model_code_tensor(dosage_to_model_codes(raw))
        sample_idx = np.arange(len(codes))
    else:
        if args.scaler_path is not None:
            raise ValueError("新格式实验必须使用实验目录保存的 scaler，不能覆盖 --scaler_path")
        scalers = artifacts.scalers
        print(f"[*] 复用训练筛选配置: screen_multivariate={artifacts.config['screen_multivariate']}")
        if args.input_snp:
            codes = prepare_inference_input(
                np.load(args.input_snp, allow_pickle=False), artifacts, input_stage=args.input_stage
            )
            sample_idx = np.arange(len(codes))
        else:
            codes, targets, sample_idx = prepare_test_data(args.snp_file, artifacts, args.pheno_dir)
    if not len(codes):
        raise ValueError("推理输入没有个体")
    predictions = []
    with torch.inference_mode():
        for batch in codes.split(args.batch_size):
            predictions.append(model(batch.to(device)).cpu())
    predicted = denormalize_phenotype(torch.cat(predictions), scalers)
    truth = None if targets is None else denormalize_phenotype(targets, scalers)
    rows = []
    for i, sample in enumerate(sample_idx):
        for p, name in enumerate(info["phenotype_names"]):
            row = {"sample_idx": int(sample), "phenotype": name,
                   "legacy": info["legacy"], "paper_eligible": info["paper_eligible"]}
            for t in range(info["time_points"]):
                row[f"T{t+1}_Pred"] = predicted[i, p, t].item()
                if truth is not None:
                    row[f"T{t+1}_True"] = truth[i, p, t].item()
            rows.append(row)
    output = Path(args.output or f"outputs/infer/inference_{datetime.now():%Y%m%d_%H%M%S}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    record = {
        "checkpoint": info["ckpt_path"], "legacy": info["legacy"],
        "checkpoint_sha256": file_sha256(info["ckpt_path"]),
        "paper_eligible": info["paper_eligible"],
        "input_stage": args.input_stage or "saved_test_split",
        "phenotype_names": info["phenotype_names"], "time_coordinates": info["time_coordinates"],
        "artifact_hashes": {} if artifacts is None else artifacts.hashes,
        "wide_input_mode": info["wide_input_mode"],
        "preprocessing_version": None if artifacts is None else artifacts.metadata["schema_version"],
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[*] 预测已保存至 {output}")
    return predicted


if __name__ == "__main__":
    predict()

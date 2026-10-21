"""从严格最佳 ConvAE checkpoint 确定性导出 train/val/test 潜在特征，不读取表型。"""

import argparse
import json
from pathlib import Path

from utils.ConvAE_Export import export_convae_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="实验目录或 models/best_model.pt")
    parser.add_argument("--split_file", required=True, help="必须与 checkpoint 一致的普通/fold manifest")
    parser.add_argument("--snp_file", default=None, help="原始 dosage .npy，默认保存配置中的路径；缺失 -1/NaN")
    parser.add_argument("--output_root", default=None, help="默认实验目录/exports；已有导出禁止覆盖")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint_sha256", default=None, help="可选：任务预期的 checkpoint SHA256")
    parser.add_argument("--model_config", default=None, help="可选：任务预期的 model_config.json")
    args = parser.parse_args()
    config = None if args.model_config is None else json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    output = export_convae_features(args.model_path, split_file=args.split_file, snp_file=args.snp_file,
        output_root=args.output_root, batch_size=args.batch_size, device=args.device,
        expected_checkpoint_sha256=args.checkpoint_sha256, expected_model_config=config)
    print(f"ConvAE 潜在特征已导出并校验: {output}")


if __name__ == "__main__":
    main()

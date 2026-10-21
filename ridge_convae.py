"""ConvAE 表征的独立线性探针/下游基线；固定 Ridge，不进行模型/solver 扩展。"""

import argparse

from utils.ConvAE_Ridge import run_ridge_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_dir", required=True, help="第7步包含 metadata.json/features 的导出目录")
    parser.add_argument("--phenotype_file", required=True, help="显式 sample_id 对齐的原始表型 NPZ，见 README")
    parser.add_argument("--split_file", default=None, help="可选迁移路径，内容哈希必须与导出 manifest 相同")
    parser.add_argument("--output_root", default="outputs/baselines/convae_ridge")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    output = run_ridge_baseline(args.export_dir, args.phenotype_file,
        output_root=args.output_root, split_file=args.split_file, seed=args.seed)
    print(f"ConvAE 独立 Ridge 基线已保存并通过预测复现校验: {output}")


if __name__ == "__main__":
    main()

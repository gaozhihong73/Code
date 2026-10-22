"""真实输入只读 preflight；不启动模型、筛选或 split 创建。"""

import argparse
import json
from pathlib import Path

from utils.RealDataManifest import (RealDataManifest, experiment_template, scan_data,
                                    verify_manifest, _write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snp_file")
    parser.add_argument("--pheno_dir")
    parser.add_argument("--output_root", default="outputs/preflight")
    parser.add_argument("--expected_traits", type=int, default=24)
    parser.add_argument("--block_columns", type=int, default=16384)
    parser.add_argument("--attestation_date", help="用户已明确确认当前源顺序时才提供 YYYY-MM-DD")
    parser.add_argument("--attestation_file", help="源哈希变化后，用户重新确认的完整哈希凭据JSON；不得自动生成确认")
    parser.add_argument("--snp_ids_file", help="已有 SNP 行生物学 ID 单列 CSV，禁止自动推断")
    parser.add_argument("--verify_manifest", help="只读重新核验冻结清单、当前源文件哈希和确认有效性")
    parser.add_argument("--write_schema", help="只写不包含真实数据的 JSON schema")
    args = parser.parse_args()
    if args.write_schema:
        with Path(args.write_schema).open("x", encoding="utf-8") as handle:
            json.dump(RealDataManifest.model_json_schema(), handle, ensure_ascii=False, indent=2)
        return 0
    if args.verify_manifest:
        payload = verify_manifest(args.verify_manifest)
        print(json.dumps({"status": "valid", "manifest_sha256": payload["manifest_sha256"]}))
        return 0
    if not args.snp_file or not args.pheno_dir:
        parser.error("扫描必须显式提供 --snp_file 和 --pheno_dir，不自动寻找或替代数据")
    dest = scan_data(args.snp_file, args.pheno_dir, args.output_root, expected_traits=args.expected_traits,
                     block_columns=args.block_columns, attestation_date=args.attestation_date, snp_ids_file=args.snp_ids_file,
                     attestation_file=args.attestation_file)
    payload = json.loads((dest / "real_data_manifest.json").read_text())
    if payload["status"] == "ready":
        verify_manifest(dest / "real_data_manifest.json")
        _write_json(dest / "experiment.template.json", experiment_template(payload))
        (dest / "next_steps.txt").write_text(
            "先 verify_manifest，再按研究目标填写主性状、时间坐标含义、seed、K折数、候选k、训练折内缺失策略、输出根目录。\n"
            "模板尚未完成，不是可直接运行的 spec。runner 当前不接收 real_manifest 字段；先 verify 再使用从清单提取的 FileRefs。\n"
            "普通split（只给出模板，本次未执行）：\n"
            ".venv/bin/python -B -c 'from utils.Split import generate_split,save_split; "
            f"save_split(generate_split({payload['genotype']['n_samples']},ratios=(0.8,0.1,0.1),seed=<SEED>),\"<SHARED_SPLIT>.npz\")'\n"
            "填写模板后：.venv/bin/python -B run_experiments.py dry-run <COMPLETED_SPEC.json>\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "manifest": str(dest / "real_data_manifest.json"),
                      "manifest_sha256": payload["manifest_sha256"], "errors": payload["errors"]}, ensure_ascii=False))
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

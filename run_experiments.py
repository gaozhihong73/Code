"""DynaGS experiment spec v1：dry-run、显式运行、严格汇总或小型合成 smoke。"""

import argparse
import json
from pathlib import Path

from experiments.runner import run_unit, summarize
from experiments.spec import digest, plan, read_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "run"):
        command = sub.add_parser(name)
        command.add_argument("spec")
        command.add_argument(
            "--development-smoke",
            action="store_true",
            help="仅开发 smoke 放宽 dirty 拒绝并记录",
        )
        if name == "run":
            command.add_argument(
                "--resume",
                action="store_true",
                help="同配置成功结果 no-op；test 前失败新 attempt 重启",
            )
    command = sub.add_parser("summarize")
    command.add_argument("directories", nargs="+")
    command = sub.add_parser("smoke")
    command.add_argument(
        "--directory", required=True, help="新的合成临时目录，不得复用已有目录"
    )
    args = parser.parse_args()
    if args.command == "summarize":
        print(json.dumps(summarize(args.directories), ensure_ascii=False, indent=2))
    elif args.command == "smoke":
        from experiments.smoke import smoke

        print(json.dumps(smoke(args.directory), ensure_ascii=False, indent=2))
    else:
        spec = read_spec(args.spec)
        items = plan(spec, development_smoke=args.development_smoke)
        if args.command == "dry-run":
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dry_run": True,
                        "note": "仅校验小型 manifest/JSON；大数据指纹与实际 shape 在运行时验证，不生成结果",
                        "experiments": items,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            paths = [str(run_unit(item, resume=args.resume)) for item in items]
            summary = summarize(paths)
            location = (
                Path(spec.output_root)
                / "summaries"
                / (digest([item["config_hash"] for item in items]) + ".json")
            )
            location.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(summary, ensure_ascii=False, indent=2)
            if location.exists():
                if location.read_text(encoding="utf-8") != text:
                    raise ValueError("汇总文件内容变化，禁止覆盖")
            else:
                with location.open("x", encoding="utf-8") as handle:
                    handle.write(text)
            print(
                json.dumps(
                    {
                        "runs": paths,
                        "summary_file": str(location),
                        "rejected": summary["rejected"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()

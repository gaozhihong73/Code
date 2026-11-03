"""maize_gdb2026 数据集的 TFPP 训练入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import build_tfpp_config, load_tfpp_profile
from trainer import TFPPTrainer


REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPOSITORY_ROOT / "dataset/maize_gdb2026"
SOURCE_GENOTYPE = DATA_ROOT / "genotype.npy"
MODEL_GENOTYPE = DATA_ROOT / "genotype_tfpp_missing.npy"
MODEL_GENOTYPE_METADATA = DATA_ROOT / "genotype_tfpp_missing.json"
PROFILE_BY_CONDITION = {
    "HN": REPOSITORY_ROOT / "config/profiles/maize_gdb2026_hn.json",
    "LN": REPOSITORY_ROOT / "config/profiles/maize_gdb2026_ln.json",
}
TIME_LABELS = ["V9", "VT", "R1"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 maize_gdb2026 的 HN/LN 三时期表型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        type=str.upper,
        choices=sorted(PROFILE_BY_CONDITION),
        default="HN",
        help="氮处理条件",
    )
    trait_group = parser.add_mutually_exclusive_group()
    trait_group.add_argument(
        "--traits",
        nargs="+",
        help="一个或多个性状名称，例如 PH PW；未指定时使用 profile 的 phenotype_names",
    )
    trait_group.add_argument(
        "--all-traits",
        action="store_true",
        help="联合训练 selected_traits.csv 中的全部 10 个性状",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="自定义 profile；未指定时按 condition 选择仓库内 profile",
    )
    parser.add_argument("--exp-name", default=None, help="自定义实验输出名称")
    parser.add_argument("--output-root", default=None, help="覆盖 profile 的输出根目录")
    parser.add_argument("--device", default=None, help="覆盖 profile 的设备，例如 cuda:0 或 cpu")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--plan-only",
        action="store_true",
        help="只校验数据并打印最终配置，不创建输出或启动训练",
    )
    action_group.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成 TFPP 可读取的显式缺失基因型，不启动训练",
    )
    return parser.parse_args()


def _read_single_column(path: Path, column: str) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [column]:
            raise ValueError(f"{path} 必须且只能包含 {column} 列")
        values = [row[column] for row in reader]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{path} 的 {column} 必须非空且唯一")
    return values


def _available_traits() -> list[str]:
    path = DATA_ROOT / "phenotype/selected_traits.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "trait" not in reader.fieldnames:
            raise ValueError(f"缺少 trait 列: {path}")
        traits = [row["trait"] for row in reader]
    if not traits or len(traits) != len(set(traits)):
        raise ValueError(f"性状清单必须非空且唯一: {path}")
    return traits


def _resolve_traits(
    args: argparse.Namespace,
    available: list[str],
    profile_traits: list[str],
) -> list[str]:
    if args.all_traits:
        requested = available
    elif args.traits:
        requested = args.traits
    else:
        requested = profile_traits
    if (
        not isinstance(requested, list)
        or not requested
        or not all(isinstance(trait, str) and trait for trait in requested)
    ):
        raise ValueError("profile 的 phenotype_names 必须是非空字符串列表")
    canonical = {trait.upper(): trait for trait in available}
    unknown = sorted({trait for trait in requested if trait.upper() not in canonical})
    if unknown:
        raise ValueError(
            f"未知性状: {', '.join(unknown)}；可选值: {', '.join(available)}"
        )
    traits = [canonical[trait.upper()] for trait in requested]
    if len(traits) != len(set(traits)):
        raise ValueError("--traits 不能包含重复性状")
    return traits


def _validate_inputs(condition: str, traits: list[str]) -> dict[str, object]:
    genotype = np.load(SOURCE_GENOTYPE, mmap_mode="r", allow_pickle=False)
    if genotype.ndim != 2:
        raise ValueError(f"基因型必须为二维矩阵，实际形状: {genotype.shape}")
    mean_imputed_count = 0
    for start in range(0, len(genotype), 16):
        block = np.asarray(genotype[start : start + 16])
        if not np.isfinite(block).all() or (block < 0).any() or (block > 2).any():
            raise ValueError("原始基因型包含非有限值或超出 [0, 2] 的剂量")
        observed = (block == 0) | (block == 1) | (block == 2)
        mean_imputed_count += int((~observed).sum())

    condition_dir = DATA_ROOT / "phenotype" / condition
    sample_ids_path = condition_dir / "sample_ids.csv"
    sample_ids = _read_single_column(sample_ids_path, "SampleID")
    if genotype.shape[0] != len(sample_ids):
        raise ValueError(
            f"基因型与样本清单行数不一致: {genotype.shape[0]} != {len(sample_ids)}"
        )

    with (DATA_ROOT / "sample_ids.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        root_rows = list(csv.DictReader(handle))
    root_ids = [row.get("sample_id", "") for row in root_rows]
    if root_ids != sample_ids:
        raise ValueError("条件目录 sample_ids.csv 与基因型根样本顺序不一致")

    for trait in traits:
        path = condition_dir / f"{trait}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != ["SampleID", *TIME_LABELS]:
                raise ValueError(
                    f"{path} 表头应为 SampleID,{','.join(TIME_LABELS)}，实际为 {header}"
                )
            rows = list(reader)
        phenotype_ids = [row[0] for row in rows if len(row) == 4]
        if len(phenotype_ids) != len(rows) or phenotype_ids != sample_ids:
            raise ValueError(f"{path} 的行数、列数或 SampleID 顺序不正确")
        values = np.asarray([row[1:] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} 含缺失值或非有限表型值")

    return {
        "genotype_shape": list(genotype.shape),
        "genotype_dtype": str(genotype.dtype),
        "detectable_mean_imputed_count": mean_imputed_count,
        "detectable_mean_imputed_fraction": mean_imputed_count / genotype.size,
        "model_genotype": str(MODEL_GENOTYPE),
        "sample_count": len(sample_ids),
        "time_labels": TIME_LABELS,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_model_genotype() -> dict[str, object]:
    """将可识别的全样本均值填充值恢复为 -1，供训练集内重新填补。"""
    source_sha256 = _sha256(SOURCE_GENOTYPE)
    if MODEL_GENOTYPE.is_file() and MODEL_GENOTYPE_METADATA.is_file():
        try:
            metadata = json.loads(MODEL_GENOTYPE_METADATA.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}
        if (
            metadata.get("source_sha256") == source_sha256
            and metadata.get("output_sha256") == _sha256(MODEL_GENOTYPE)
        ):
            return {**metadata, "reused": True}

    source = np.load(SOURCE_GENOTYPE, mmap_mode="r", allow_pickle=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".genotype_tfpp_missing_",
            suffix=".npy",
            dir=DATA_ROOT,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        output = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=source.shape,
        )
        restored_count = 0
        for start in range(0, len(source), 16):
            block = np.asarray(source[start : start + 16], dtype=np.float32).copy()
            if not np.isfinite(block).all() or (block < 0).any() or (block > 2).any():
                raise ValueError("原始基因型包含非有限值或超出 [0, 2] 的剂量")
            observed = (block == 0) | (block == 1) | (block == 2)
            restored_count += int((~observed).sum())
            block[~observed] = -1
            output[start : start + len(block)] = block
        output.flush()
        del output
        temporary_path.replace(MODEL_GENOTYPE)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    metadata = {
        "schema_version": 1,
        "source": str(SOURCE_GENOTYPE.resolve()),
        "source_sha256": source_sha256,
        "output": str(MODEL_GENOTYPE.resolve()),
        "output_sha256": _sha256(MODEL_GENOTYPE),
        "shape": list(source.shape),
        "dtype": "float32",
        "transformation": "non_integer_mean_imputed_dosage_to_missing_minus_one_v1",
        "detectable_mean_imputed_restored_count": restored_count,
        "detectable_mean_imputed_restored_fraction": restored_count / source.size,
        "missing_value": -1,
        "limitation": (
            "原始均值填充值若恰为整数 0/1/2，则无法与真实剂量区分，"
            "不能从当前 genotype.npy 无损恢复"
        ),
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix=".genotype_tfpp_missing_",
        suffix=".json",
        dir=DATA_ROOT,
        delete=False,
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        metadata_temporary = Path(handle.name)
    metadata_temporary.replace(MODEL_GENOTYPE_METADATA)
    return {**metadata, "reused": False}


def _default_exp_name(condition: str, traits: list[str], all_traits: list[str]) -> str:
    trait_slug = "all_traits" if traits == all_traits else "-".join(traits)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{condition}/{trait_slug}_{stamp}"


def main() -> None:
    args = _parse_args()
    os.chdir(REPOSITORY_ROOT)
    profile_path = (args.config or PROFILE_BY_CONDITION[args.condition]).resolve()
    profile = load_tfpp_profile(profile_path)
    available = _available_traits()
    traits = _resolve_traits(args, available, profile.get("phenotype_names", []))
    input_summary = _validate_inputs(args.condition, traits)
    preparation = None if args.plan_only else _prepare_model_genotype()
    condition_dir = DATA_ROOT / "phenotype" / args.condition
    config = build_tfpp_config(
        profile_path,
        exp_name=args.exp_name
        or _default_exp_name(args.condition, traits, available),
        output_root=args.output_root,
        device=args.device,
        snp_file=str(MODEL_GENOTYPE.resolve()),
        pheno_dir=str(condition_dir.resolve()),
        sample_ids_file=str((condition_dir / "sample_ids.csv").resolve()),
        phenotype_names=traits,
    )
    plan = {
        "profile": str(profile_path),
        "condition": args.condition,
        "traits": traits,
        "input": input_summary,
        "genotype_preparation": preparation,
        "output_dir": str(Path(config.output_dir).resolve()),
        "config": config.model_dump(mode="json"),
    }
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.prepare_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 CUDA；请传 --device cpu")

    print(
        f"启动 maize_gdb2026: condition={args.condition}, "
        f"traits={','.join(traits)}, output={config.output_dir}"
    )
    TFPPTrainer(config=config).train()


if __name__ == "__main__":
    main()

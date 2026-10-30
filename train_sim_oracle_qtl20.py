"""仅使用 20 个真实 QTL 的 TFPP Oracle 诊断实验。

Oracle 会读取模拟真值，因此只能作为模型可学习性与筛选损失的诊断上界，
不能与未知 QTL 的筛选实验作为公平方法比较。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import build_tfpp_config, load_tfpp_profile
from config.TFPP_Config import TFPPConfig
from trainer.TFPP_Trainer import TFPPTrainer


DEFAULT_DATA_DIR = Path(
    "/root/DynaGS/dataset/experiments/P5000_NTR1600_H0.8/rep_09"
)
DEFAULT_PROFILE = Path("config/profiles/simulation.json")
EXPECTED_QTL_COUNT = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_exp_name(data_dir: Path) -> str:
    """由条件目录和重复编号生成稳定、独立的 Oracle 输出名。"""
    if data_dir.parent == data_dir:
        raise ValueError(f"无法从数据目录推导实验名: {data_dir}")
    return f"sim_oracle_qtl20/{data_dir.parent.name}/{data_dir.name}"


def _resolve_snp_path(data_dir: Path, manifest: dict) -> Path:
    """按 manifest 解析模型实际使用的 NPY 或 CSV 基因型。"""
    candidates = [
        data_dir / name
        for name in manifest.get("model_inputs", [])
        if Path(name).name.startswith("sim_snp")
    ]
    candidates.extend([data_dir / "sim_snp.npy", data_dir / "sim_snp.csv"])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{data_dir} 中没有可用的 sim_snp.npy/sim_snp.csv")


def extract_true_qtl_indices(
    data_dir: Path,
    genotype_snp_dim: int,
    output_path: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """严格校验模拟真值，并保存 TFPP 使用的 0-based QTL 列索引。"""
    truth_path = data_dir / "truth" / "causal_effects.csv"
    if not truth_path.is_file():
        raise FileNotFoundError(f"缺少模拟真值文件: {truth_path}")

    effects = pd.read_csv(truth_path)
    required_columns = {"snp_index", "snp_name", "effect_type"}
    missing = required_columns - set(effects.columns)
    if missing:
        raise ValueError(f"causal_effects.csv 缺少列: {sorted(missing)}")
    if len(effects) != EXPECTED_QTL_COUNT:
        raise ValueError(
            f"Oracle 实验要求 {EXPECTED_QTL_COUNT} 个真实 QTL，实际为 {len(effects)}"
        )

    one_based = effects["snp_index"].to_numpy(dtype=np.int64)
    if len(np.unique(one_based)) != len(one_based) or (one_based < 1).any():
        raise ValueError("模拟真值中的 snp_index 必须是互异的正整数")
    expected_names = np.array([f"SNP{index}" for index in one_based])
    if not np.array_equal(effects["snp_name"].to_numpy(dtype=str), expected_names):
        raise ValueError("snp_name 与 snp_index 不一致，拒绝使用不明确的真值索引")

    indices = np.sort(one_based - 1)
    if (indices >= genotype_snp_dim).any():
        raise ValueError(f"真实 QTL 索引越界：基因型只有 {genotype_snp_dim} 列")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            existing = np.load(output_path, allow_pickle=False)
            if not np.array_equal(existing, indices):
                raise FileExistsError(f"已有不同的 Oracle QTL 索引文件: {output_path}")
        else:
            np.save(output_path, indices)
    return indices, effects


def build_config(
    data_dir: Path,
    config_file: Path = DEFAULT_PROFILE,
    output_root: Path | None = None,
    exp_name: str | None = None,
    publish_artifacts: bool = True,
) -> tuple[TFPPConfig, dict]:
    """构建与未知QTL实验共用划分、但禁用内部筛选的配置。"""
    data_dir = data_dir.resolve()
    profile = load_tfpp_profile(config_file)
    resolved_output_root = (
        output_root.resolve()
        if output_root is not None
        else Path(profile.get("output_root", "outputs/tfpp")).resolve()
    )
    manifest_path = data_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少数据 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split = manifest.get("split", {})
    if split.get("train_n") is None or split.get("seed") is None:
        raise ValueError("data_manifest.json 缺少 split.train_n 或 split.seed")
    if split["seed"] != 73:
        raise ValueError(f"正式协议要求 split seed=73，实际为 {split['seed']}")
    if manifest.get("genotype_shape", [None, None])[1] is None:
        raise ValueError("data_manifest.json 缺少 genotype_shape")

    snp_path = _resolve_snp_path(data_dir, manifest)
    pheno_path = data_dir / "sim_pheno.csv"
    sample_ids_path = data_dir / "sample_ids.csv"
    split_path = data_dir / "splits" / "split_seed73.npz"
    for required in (pheno_path, sample_ids_path, split_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少 Oracle 实验输入: {required}")

    # resolved_exp_name = exp_name or _default_exp_name(data_dir)
    resolved_exp_name = exp_name or _default_exp_name(data_dir)
    run_dir = resolved_output_root / resolved_exp_name
    selected_path = run_dir / "oracle_selection" / "true_qtl_indices_zero_based.npy"
    indices, effects = extract_true_qtl_indices(
        data_dir,
        int(manifest["genotype_shape"][1]),
        selected_path if publish_artifacts else None,
    )

    config = build_tfpp_config(
        config_file,
        exp_name=resolved_exp_name,
        output_root=str(resolved_output_root),
        snp_file=str(snp_path),
        pheno_dir=str(data_dir),
        phenotype_names=["sim_pheno"],
        time_coordinates=manifest["time_coordinates"],
        split_file=str(split_path),
        sample_ids_file=str(sample_ids_path),
        preselected_snp_file=str(selected_path),
        # Oracle 与内部监督筛选互斥；这不是可调超参数。
        screen_k=0,
        gwas_k=0,
    )

    oracle_manifest = {
        "protocol": "oracle_true_qtl20_v1",
        "purpose": "diagnostic_upper_reference_not_fair_method_comparison",
        "data_dir": str(data_dir),
        "data_manifest_sha256": _sha256(manifest_path),
        "causal_effects_file": str(data_dir / "truth" / "causal_effects.csv"),
        "causal_effects_sha256": _sha256(data_dir / "truth" / "causal_effects.csv"),
        "qtl_count": int(len(indices)),
        "qtl_indices_zero_based": indices.tolist(),
        "qtl_indices_one_based": (indices + 1).tolist(),
        "qtl_effect_type_counts": {
            str(key): int(value)
            for key, value in effects["effect_type"].value_counts().sort_index().items()
        },
        "train_n": int(split["train_n"]),
        "split_seed": int(split["seed"]),
        "snp_file": str(snp_path),
        "screen_k": 0,
        "gwas_k": 0,
        "ablation_mode": config.ablation_mode,
    }
    oracle_manifest_path = selected_path.parent / "oracle_manifest.json"
    if publish_artifacts:
        if oracle_manifest_path.exists():
            existing = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
            if existing != oracle_manifest:
                raise FileExistsError(f"已有不同的 Oracle 配置: {oracle_manifest_path}")
        else:
            oracle_manifest_path.write_text(
                json.dumps(oracle_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    print(
        "[Oracle diagnostic] "
        f"data={data_dir.name}, train_n={split['train_n']}, "
        f"QTL={len(indices)}/{manifest['genotype_shape'][1]}, split_seed=73"
    )
    return config, oracle_manifest


def train_oracle_qtl_experiment(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path = DEFAULT_PROFILE,
    output_root: Path | None = None,
    exp_name: str | None = None,
    dry_run: bool = False,
) -> dict:
    """准备 Oracle 真值索引，并按固定划分训练和评估一次。"""
    config, oracle_manifest = build_config(
        data_dir=data_dir,
        config_file=config_file,
        output_root=output_root,
        exp_name=exp_name,
        publish_artifacts=not dry_run,
    )
    if dry_run:
        print(f"[Dry run] 配置验证通过，输出目录: {config.output_dir}")
        return {"config": config.model_dump(), "oracle": oracle_manifest}
    run_dir = Path(config.output_dir)
    existing_training_outputs = [
        run_dir / "test_evaluation.json",
        run_dir / "models" / "best_model.pt",
        run_dir / "training_history" / "training_summary.json",
    ]
    occupied = [path for path in existing_training_outputs if path.exists()]
    if occupied:
        raise FileExistsError(
            "Oracle 输出目录已有训练结果，拒绝覆盖；请更换 --exp_name: "
            + ", ".join(str(path) for path in occupied)
        )
    trainer = TFPPTrainer(config=config)
    return trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅校验数据、QTL索引和配置，不启动训练",
    )
    args = parser.parse_args()
    train_oracle_qtl_experiment(
        data_dir=args.data_dir,
        config_file=args.config,
        output_root=args.output_root,
        exp_name=args.exp_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

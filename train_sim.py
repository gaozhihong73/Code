"""模拟数据的未知 QTL TFPP 训练入口。

固定使用条件残差 GWAS/DC-SIS 筛选。数据、划分、时间坐标和模型超参数
由此文件统一管理，批量实验无需重复长命令。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import build_tfpp_config
from config.TFPP_Config import TFPPConfig
from trainer.TFPP_Trainer import TFPPTrainer


DEFAULT_DATA_DIR = Path("dataset/experiments/P5000_NTR400_H0.4/rep_01")
DEFAULT_PROFILE = Path("config/profiles/simulation.json")


def _resolve_snp_file(data_dir: Path, manifest: dict) -> Path:
    candidates = [
        data_dir / name
        for name in manifest.get("model_inputs", [])
        if Path(name).name.startswith("sim_snp")
    ]
    candidates.extend([data_dir / "sim_snp.npy", data_dir / "sim_snp.csv"])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到 sim_snp.npy 或 sim_snp.csv: {data_dir}")


def _default_exp_name(data_dir: Path) -> str:
    return f"sim_conditional/{data_dir.parent.name}/{data_dir.name}"


def build_config(
    data_dir: Path,
    config_file: Path = DEFAULT_PROFILE,
    output_root: Path | None = None,
    exp_name: str | None = None,
) -> TFPPConfig:
    """读取模拟实验配置，并补充当前数据集必需的动态路径。"""
    data_dir = data_dir.resolve()
    manifest_path = data_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 data_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_seed = manifest["split"]["seed"]
    if split_seed != 73:
        raise ValueError(f"模拟正式协议要求 split seed=73，实际为 {split_seed}")
    snp_file = _resolve_snp_file(data_dir, manifest)
    for path in (
        data_dir / "sim_pheno.csv",
        data_dir / "sample_ids.csv",
        data_dir / "splits" / f"split_seed{split_seed}.npz",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"缺少模拟训练输入: {path}")

    resolved_exp_name = exp_name or _default_exp_name(data_dir)
    overrides = {
        "exp_name": resolved_exp_name,
        "snp_file": str(snp_file),
        "pheno_dir": str(data_dir),
        "phenotype_names": ["sim_pheno"],
        "time_coordinates": manifest["time_coordinates"],
        "split_file": str(data_dir / "splits" / f"split_seed{split_seed}.npz"),
        "sample_ids_file": str(data_dir / "sample_ids.csv"),
    }
    if output_root is not None:
        overrides["output_root"] = str(output_root.resolve())
    return build_tfpp_config(config_file, **overrides)


def train_simulation(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path = DEFAULT_PROFILE,
    output_root: Path | None = None,
    exp_name: str | None = None,
    dry_run: bool = False,
) -> dict:
    """运行一次模拟训练；dry_run 只显示确定的配置。"""
    config = build_config(
        data_dir=data_dir,
        config_file=config_file,
        output_root=output_root,
        exp_name=exp_name,
    )
    if dry_run:
        print(json.dumps(config.model_dump(), ensure_ascii=False, indent=2))
        return config.model_dump()

    existing = [
        Path(config.output_dir) / "test_evaluation.json",
        Path(config.best_model_save_dir) / "best_model.pt",
    ]
    occupied = [str(path) for path in existing if path.exists()]
    if occupied:
        raise FileExistsError(
            "输出目录已有训练结果；请改用 --exp_name: " + ", ".join(occupied)
        )
    return TFPPTrainer(config=config).train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    train_simulation(
        data_dir=args.data_dir,
        config_file=args.config,
        output_root=args.output_root,
        exp_name=args.exp_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

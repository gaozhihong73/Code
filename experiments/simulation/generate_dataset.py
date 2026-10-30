"""生成一个冻结参数的模拟数据集，并保存完整生成溯源。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prepare_simulation import prepare_simulation


TRAIN_TO_TOTAL = {
    100: 124,
    200: 250,
    400: 500,
    800: 1000,
    1600: 2000,
}


def _heritability_label(value: float) -> str:
    return format(value, ".12g")


def generate_dataset(
    *,
    output_root: str | Path,
    n_train: int,
    n_snps: int,
    heritability: float,
    repetition: int,
    seed: int,
    snp_format: str,
    n_epistasis: int = 0,
    epistasis_strength: float = 0.5,
) -> Path:
    if n_train not in TRAIN_TO_TOTAL:
        raise ValueError(f"n_train 必须是 {sorted(TRAIN_TO_TOTAL)} 之一")
    if n_snps not in (5000, 10000):
        raise ValueError("正式设计的 n_snps 仅支持 5000 或 10000")
    if heritability not in (0.1, 0.4, 0.8):
        raise ValueError("正式设计的 heritability 仅支持 0.1、0.4 或 0.8")
    if repetition < 1:
        raise ValueError("repetition 必须从1开始")
    if snp_format not in ("npy", "csv"):
        raise ValueError("snp_format 仅支持 npy 或 csv")
    if n_epistasis < 0 or n_epistasis > 190:
        raise ValueError("n_epistasis 必须在0到190之间")
    if epistasis_strength < 0:
        raise ValueError("epistasis_strength 必须为非负数")

    repository_root = REPOSITORY_ROOT
    output_root = Path(output_root).resolve()
    condition = (
        f"P{n_snps}_NTR{n_train}_H{_heritability_label(heritability)}"
    )
    if n_epistasis:
        condition += f"_EPI{n_epistasis}_S{_heritability_label(epistasis_strength)}"
    output_dir = output_root / condition / f"rep_{repetition:02d}"
    if output_dir.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {output_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=".generate-simulation-", dir=output_root)
    )
    raw_dir = staging_parent / "raw"
    n_total = TRAIN_TO_TOTAL[n_train]
    qtl_pool_size = min(5000, n_snps)
    command = [
        "Rscript",
        str(repository_root / "r/generate_simulation.R"),
        "--output_dir",
        str(raw_dir),
        "--scenario",
        condition,
        "--rep",
        str(repetition),
        "--n_samples",
        str(n_total),
        "--n_snps",
        str(n_snps),
        "--n_qtl",
        "20",
        "--qtl_pool_size",
        str(qtl_pool_size),
        "--n_timepoints",
        "20",
        "--heritability",
        str(heritability),
        "--ld",
        "0.3",
        "--dominance_ratio",
        "0.3",
        "--prop_add",
        "0.4",
        "--prop_dom",
        "0.3",
        "--prop_mixed",
        "0.3",
        "--n_epistasis",
        str(n_epistasis),
        "--epistasis_strength",
        str(epistasis_strength),
        "--seed",
        str(seed),
    ]
    request = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "repetition": repetition,
        "n_train_requested": n_train,
        "n_total_generated": n_total,
        "expected_split": {
            "train": n_train,
            "validation": (n_total - n_train) // 2,
            "test": n_total - n_train - (n_total - n_train) // 2,
        },
        "n_snps": n_snps,
        "n_qtl": 20,
        "qtl_pool_size": qtl_pool_size,
        "heritability": heritability,
        "time_coordinates": list(range(1, 21)),
        "n_epistasis": n_epistasis,
        "epistasis_strength": epistasis_strength,
        "seed": seed,
        "snp_format": snp_format,
        "generator_command": command,
    }

    try:
        subprocess.run(command, cwd=repository_root, check=True)
        (raw_dir / "generation_request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = prepare_simulation(
            raw_dir,
            output_dir,
            seed=73,
            ratios=(0.8, 0.1, 0.1),
            snp_format=snp_format,
            provenance_root=output_root / "provenance",
        )
        actual_train = manifest["split"]["train_n"]
        if actual_train != n_train:
            raise RuntimeError(
                f"划分后训练样本数错误: expected={n_train}, actual={actual_train}"
            )
        return output_dir
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root", default="dataset/experiments"
    )
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_snps", type=int, required=True)
    parser.add_argument("--heritability", type=float, required=True)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--snp_format", choices=("npy", "csv"), default="npy")
    parser.add_argument("--n_epistasis", type=int, default=0)
    parser.add_argument("--epistasis_strength", type=float, default=0.5)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else 72 + args.rep
    output = generate_dataset(
        output_root=args.output_root,
        n_train=args.n_train,
        n_snps=args.n_snps,
        heritability=args.heritability,
        repetition=args.rep,
        seed=seed,
        snp_format=args.snp_format,
        n_epistasis=args.n_epistasis,
        epistasis_strength=args.epistasis_strength,
    )
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

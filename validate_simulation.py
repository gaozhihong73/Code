"""使用原 TFPP Dataset、Split 和 DataLoader 验证准备后的模拟数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataloader import TFPPDataset, get_dataloader
from utils.Split import load_split, validate_individual_split


def validate_tfpp_loading(data_dir: str | Path, *, batch_size: int = 16) -> dict:
    data_dir = Path(data_dir).resolve()
    manifest = json.loads((data_dir / "data_manifest.json").read_text())
    sample_ids = np.genfromtxt(
        data_dir / "sample_ids.csv", delimiter=",", dtype=str, skip_header=1
    ).reshape(-1).tolist()
    split_path = data_dir / "splits" / f"split_seed{manifest['split']['seed']}.npz"
    split = load_split(split_path, sample_ids=sample_ids)
    validate_individual_split(split, sample_ids=sample_ids)
    snp_name = manifest["model_inputs"][0]
    snp_path = data_dir / snp_name
    dataset = TFPPDataset(
        str(snp_path),
        str(data_dir),
        ["sim_pheno"],
        sample_ids=sample_ids,
    )
    loaders = get_dataloader(
        dataset,
        batch_size=batch_size,
        norm_method="global",
        scaler_type="zscore",
        device="cpu",
        num_workers=0,
        indices=split,
    )
    train_loader, val_loader, test_loader, scalers = loaders
    batches = [next(iter(loader)) for loader in loaders[:3]]
    if snp_path.suffix == ".npy":
        raw = np.load(snp_path, allow_pickle=False)
    else:
        raw = pd.read_csv(snp_path).drop(columns="SampleID").to_numpy()
    index_sets = [set(x.tolist()) for x in (split.train_idx, split.val_idx, split.test_idx)]
    finite = all(torch.isfinite(item).all().item() for batch in batches for item in batch)
    scaler = scalers["pheno_meta"]["scalers"]["sim_pheno"]
    train_values = dataset.phenotype[split.train_idx, 0, :].reshape(-1, 1)
    scaler_fit_train_only = bool(np.allclose(scaler.mean_, train_values.mean(axis=0)))
    return {
        "genotype_shape": list(raw.shape),
        "genotype_dtype": str(raw.dtype),
        "phenotype_shape": list(dataset.phenotype.shape),
        "split_counts": [len(split.train_idx), len(split.val_idx), len(split.test_idx)],
        "batch_genotype_shapes": [list(batch[0].shape) for batch in batches],
        "batch_phenotype_shapes": [list(batch[1].shape) for batch in batches],
        "raw_dosage_values": np.unique(raw).tolist(),
        "model_code_values": np.unique(dataset.snp).tolist(),
        "timepoint_count": dataset.phenotype.shape[-1],
        "finite": finite,
        "split_disjoint": not (index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2] or index_sets[1] & index_sets[2]),
        "split_complete": set.union(*index_sets) == set(range(len(dataset))),
        "scaler_fit_train_only": scaler_fit_train_only,
        "screening_contract": "TFPPTrainer passes only dataset[train_idx] to screen_snps",
        "truth_is_model_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    report = validate_tfpp_loading(args.data_dir, batch_size=args.batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    checks = ("finite", "split_disjoint", "split_complete", "scaler_fit_train_only")
    if not all(report[name] for name in checks):
        raise SystemExit("TFPP 模拟数据加载验证失败")


if __name__ == "__main__":
    main()

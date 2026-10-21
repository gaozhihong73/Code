"""仅40个体/13 SNP 的 CPU smoke，实际执行所有六条当前路线。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ConvAEConfig
from trainer import ConvAETrainer
from utils.ConvAE_Export import export_convae_features
from utils.Split import generate_split, save_split
from .runner import run_unit, summarize
from .spec import ExperimentSpec, file_digest, plan


def create_smoke_spec(directory: str | Path, *, include_ridge: bool = True) -> ExperimentSpec:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(73)
    x = rng.integers(0, 3, size=(40, 13), dtype=np.int8)
    x[0, 0] = -1
    y = rng.normal(size=(40, 2, 3)) + x[:, :2, None]
    np.save(root / "snp.npy", x)
    for p, name in enumerate(("A", "B")):
        pd.DataFrame(y[:, p], columns=["0", "1", "2"]).to_csv(root / f"{name}.csv", index=False)
    np.save(root / "fixed.npy", np.array([5, 1, 7], dtype=np.int64))
    split = generate_split(40, ratios=(0.6, 0.2, 0.2), seed=73)
    save_split(split, root / "split.npz")
    def ref(name):
        path = root / name
        return {"path": str(path), "sha256": file_digest(path)}
    options = {"epochs": 1, "batch_size": 4, "num_workers": 0, "device": "cpu", "d_model": 8,
               "num_heads": 2, "num_layers": 1, "d_ff": 16, "degree": 1, "dropout": 0,
               "use_warmup": False, "screen_missing_strategy": "mode"}
    units = [{"route": route, "seed": 73, "traits": ["A", "B"], "k": 3, "gwas_k": 5, "tfpp": options,
              **({"fixed_indices": ref("fixed.npy")} if route == "fixed_indices" else {})}
             for route in ("fixed_indices", "random_k", "GWAS", "DC-SIS", "GWAS->DC-SIS")]
    if include_ridge:
        torch.manual_seed(73)
        trainer = ConvAETrainer(ConvAEConfig(exp_name="smoke", output_root=str(root / "convae"),
            snp_file=str(root / "snp.npy"), pheno_dir=str(root), phenotype_names=["A", "B"], time_coordinates=[1, 2, 3],
            split_file=str(root / "split.npz"), device="cpu", num_workers=0, epochs=1, batch_size=4,
            chunk_size=8, overlap=2, patch_size=3, d_model=8, d_ext=4, num_heads=2,
            encoder_layers=1, decoder_layers=1, dropout=0, use_warmup=False))
        trainer.logger.set_console_level("ERROR")
        try:
            trainer.train()
            export = export_convae_features(trainer.config.output_dir, split_file=root / "split.npz", output_root=root / "exports")
        finally:
            for handler in list(trainer.logger.logger.handlers):
                handler.close()
                trainer.logger.logger.removeHandler(handler)
        units.append({"route": "ConvAE->Ridge", "traits": ["A", "B"],
                      "exports": [{"path": str(export / "metadata.json"), "sha256": file_digest(export / "metadata.json")}]})
    spec = ExperimentSpec.model_validate({"data": {"snp": ref("snp.npy"), "phenotypes": {n: ref(f"{n}.csv") for n in ("A", "B")},
        "n_samples": 40, "n_snps": 13, "sample_ids_sha256": split.metadata["sample_ids_sha256"],
        "time_labels": ["0", "1", "2"], "time_coordinates": [1, 2, 3]},
        "splits": {"mode": "regular", "manifests": [ref("split.npz")]}, "experiments": units,
        "output_root": str(root / "runs")})
    (root / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return spec


def smoke(directory: str | Path) -> dict:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec = create_smoke_spec(directory)
        items = plan(spec, development_smoke=True)
        roots = [str(run_unit(item)) for item in items]
        summary = summarize(roots)
        summary_path = Path(directory) / "smoke_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"schema_version": 1, "synthetic_only": True, "runs": roots, "summary_file": str(summary_path),
                "successful_groups": len(summary["groups"]), "rejected": summary["rejected"]}
    finally:
        torch.set_num_threads(old_threads)

"""E3 首批适配器的小型端到端 smoke。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.e3.analysis.summarize_e3 import summarize
from experiments.e3.baselines.deep_models import (
    CropformerModel,
    DLGWASModel,
    DNNGPSNPModel,
    WheatGPModel,
)
from experiments.e3.baselines.registry import build_adapter
from experiments.e3.runners.run_e3 import DEFAULT_CONFIG, _load_config, execute_run
from experiments.e3.runners.run_e3_batch import run_batch
from experiments.e3.schemas.e3 import E3RunManifest
from utils.Split import generate_split, save_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_dataset(root: Path) -> Path:
    rng = np.random.default_rng(73)
    data_dir = root / "toy_data"
    (data_dir / "splits").mkdir(parents=True)
    n_samples, n_snps, n_time = 40, 24, 4
    sample_ids = [f"toy_{index:03d}" for index in range(n_samples)]
    genotype = rng.integers(0, 3, size=(n_samples, n_snps), dtype=np.int8)
    weights = rng.normal(size=(n_snps, 3))
    basis = np.stack(
        (np.ones(n_time), np.linspace(-1, 1, n_time), np.linspace(-1, 1, n_time) ** 2),
        axis=0,
    )
    phenotype = genotype @ weights @ basis + rng.normal(0, 0.2, size=(n_samples, n_time))
    np.save(data_dir / "sim_snp.npy", genotype, allow_pickle=False)
    pd.DataFrame({"SampleID": sample_ids}).to_csv(data_dir / "sample_ids.csv", index=False)
    frame = pd.DataFrame(phenotype, columns=[f"T{i}" for i in range(1, n_time + 1)])
    frame.insert(0, "SampleID", sample_ids)
    frame.to_csv(data_dir / "sim_pheno.csv", index=False)
    split = generate_split(sample_ids=sample_ids, seed=73)
    save_split(split, data_dir / "splits/split_seed73.npz")
    manifest = {
        "data_format_version": "e3_smoke_v1",
        "model_inputs": ["sim_snp.npy", "sim_pheno.csv"],
        "genotype_shape": [n_samples, n_snps],
        "phenotype_shape": [n_samples, n_time],
        "time_coordinates": list(range(1, n_time + 1)),
        "truth_is_model_input": False,
        "output_files": {
            "sim_snp.npy": _sha256(data_dir / "sim_snp.npy"),
            "sim_pheno.csv": _sha256(data_dir / "sim_pheno.csv"),
            "sample_ids.csv": _sha256(data_dir / "sample_ids.csv"),
            "splits/split_seed73.npz": _sha256(data_dir / "splits/split_seed73.npz"),
        },
    }
    (data_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return data_dir


class E3SmokeTests(unittest.TestCase):
    def test_batch_runner_plan_only(self) -> None:
        args = Namespace(
            stage="pilot",
            config=DEFAULT_CONFIG,
            model=["Mean"],
            rep=[1],
            limit=None,
            execute=False,
            confirm_formal=False,
            output_root=None,
            resume=True,
        )
        self.assertEqual(run_batch(args), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_deep_adapters_cuda_smoke(self) -> None:
        rng = np.random.default_rng(73)
        x_train = rng.integers(0, 3, size=(16, 24), dtype=np.int8)
        x_val = rng.integers(0, 3, size=(4, 24), dtype=np.int8)
        x_test = rng.integers(0, 3, size=(4, 24), dtype=np.int8)
        y_train = rng.normal(size=(16, 2)).astype(np.float32)
        y_val = rng.normal(size=(4, 2)).astype(np.float32)
        common = {
            "device": "cuda",
            "batch_size": 8,
            "max_epochs": 1,
            "patience": 1,
            "pooled_length": 4,
            "dense_units": 8,
            "segment_tokens": 2,
            "hidden_size": 8,
            "embedding_dim": 8,
            "num_heads": 2,
            "max_tokens": 8,
        }
        for name in ("DNNGP-SNP", "DLGWAS", "WheatGP", "Cropformer"):
            options = dict(common)
            options["channels"] = [2, 4, 4] if name == "DNNGP-SNP" else 2
            adapter = build_adapter(name, seed=73, options=options)
            result = adapter.fit(x_train, y_train, x_val, y_val)
            prediction = adapter.predict(x_test)
            self.assertEqual(prediction.shape, (4, 2))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertEqual(len(result.best_epochs_by_time), 2)
            self.assertGreater(adapter.resource_metrics()["peak_gpu_memory_mb"], 0)
            del adapter
            torch.cuda.empty_cache()

    def test_deep_models_accept_full_marker_dimension(self) -> None:
        dosage = torch.zeros((2, 1, 10_000), dtype=torch.float32)
        one_hot = torch.zeros((2, 4, 10_000), dtype=torch.float32)
        models_and_inputs = (
            (DNNGPSNPModel(), dosage),
            (DLGWASModel(), one_hot),
            (WheatGPModel(), dosage),
            (CropformerModel(input_snps=10_000), dosage),
        )
        with torch.no_grad():
            for model, values in models_and_inputs:
                model.eval()
                self.assertEqual(model(values).shape, (2,))

    def test_first_batch_adapters_end_to_end(self) -> None:
        config = _load_config(DEFAULT_CONFIG)
        config = config.model_copy(
            update={
                "model_options": {
                    "Ridge-All": {"alphas": [0.1, 1.0]},
                    "Ridge-A+D": {"alphas": [0.1, 1.0]},
                    "GBLUP-All": {"lambdas": [0.1, 1.0]},
                    "RR-GBLUP": {"degrees": [2], "lambdas": [0.1, 1.0]},
                    "Random Forest": {
                        "n_jobs": 1,
                        "candidates": [
                            {"n_estimators": 5, "max_features": 0.5, "min_samples_leaf": 1}
                        ],
                    },
                    "XGBoost": {
                        "device": "cpu",
                        "n_jobs": 1,
                        "early_stopping_rounds": 3,
                        "candidates": [
                            {
                                "n_estimators": 12,
                                "learning_rate": 0.1,
                                "max_depth": 2,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                            }
                        ],
                    },
                    "DNNGP-SNP": {
                        "device": "cpu",
                        "batch_size": 8,
                        "max_epochs": 2,
                        "patience": 1,
                        "channels": [2, 4, 4],
                        "pooled_length": 4,
                        "dense_units": 8,
                        "dropout_input": 0.1,
                        "dropout_dense": 0.1,
                    },
                    "DLGWAS": {
                        "device": "cpu",
                        "batch_size": 8,
                        "max_epochs": 2,
                        "patience": 1,
                        "channels": 2,
                        "pooled_length": 4,
                        "dropout": 0.1,
                    },
                    "WheatGP": {
                        "device": "cpu",
                        "batch_size": 8,
                        "max_epochs": 2,
                        "patience": 1,
                        "segment_tokens": 2,
                        "hidden_size": 8,
                        "cnn_dropout": 0.1,
                        "lstm_dropout": 0.1,
                    },
                    "Cropformer": {
                        "device": "cpu",
                        "batch_size": 8,
                        "max_epochs": 2,
                        "patience": 1,
                        "embedding_dim": 8,
                        "num_heads": 2,
                        "max_tokens": 8,
                        "hidden_dropout": 0.1,
                        "attention_dropout": 0.1,
                    },
                }
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = _toy_dataset(root)
            for model in (
                "Mean",
                "Ridge-All",
                "Ridge-A+D",
                "GBLUP-All",
                "RR-GBLUP",
                "Random Forest",
                "XGBoost",
                "DNNGP-SNP",
                "DLGWAS",
                "WheatGP",
                "Cropformer",
            ):
                run_dir = root / "runs" / model.replace(" ", "_")
                execute_run(
                    config,
                    repetition=1,
                    model=model,
                    data_dir=data_dir,
                    run_dir=run_dir,
                    formal=False,
                )
                manifest = E3RunManifest.model_validate_json(
                    (run_dir / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest.status, "succeeded")
                self.assertEqual(manifest.prediction.test_evaluation_count, 1)
                expected_snps = 0 if model == "Mean" else 24
                self.assertEqual(manifest.data.model_input_snp_count, expected_snps)
                if model in ("DNNGP-SNP", "DLGWAS", "WheatGP", "Cropformer"):
                    self.assertEqual(len(manifest.training.best_epochs_by_time), 4)
                    checkpoint = torch.load(
                        manifest.training.model_artifacts["checkpoint_path"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assertEqual(len(checkpoint["state_dicts"]), 4)
                with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as archive:
                    self.assertEqual(archive["prediction"].shape, archive["target"].shape)
                    self.assertTrue(
                        np.array_equal(archive["sample_idx"], manifest.data.split.test_indices)
                    )
            tables = summarize(root / "runs", root / "tables")
            run_index = pd.read_csv(tables["run_index"])
            self.assertEqual(len(run_index), 11)
            self.assertEqual(set(run_index["status"]), {"succeeded"})


if __name__ == "__main__":
    unittest.main()

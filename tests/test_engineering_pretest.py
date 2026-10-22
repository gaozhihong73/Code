"""工程 pretest 只使用合成数据，验证 train/val-only 生产训练闭环。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import TFPPConfig
from experiments.runner import summarize
from experiments.spec import file_digest
from trainer import TFPPTrainer
from utils.Screening import gwas_dc_sis_screen_mmap
from utils.Split import split_from_indices, save_split
from utils.model_loader import load_tfpp_model


class EngineeringPretestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(73)
        self.snp = rng.integers(0, 3, size=(16, 31), dtype=np.int8)
        self.snp[0, 0] = -1
        self.snp_path = self.root / "snp.npy"
        np.save(self.snp_path, self.snp)
        self.pheno = rng.normal(size=(16, 3)).astype(np.float32)
        pd.DataFrame(self.pheno, columns=["X10", "X12", "X14"]).to_csv(
            self.root / "RGBVI.csv", index=False)
        self.split = split_from_indices(np.arange(10), np.arange(10, 13), np.arange(13, 16), 16)
        self.split_path = self.root / "split.npz"
        save_split(self.split, self.split_path)
        self.indices_path = self.root / "selected.npy"
        np.save(self.indices_path, np.array([7, 0, 12, 2, 18], dtype=np.int64))

    def config(self):
        return TFPPConfig(
            exp_name="pretest", output_root=str(self.root / "out"),
            run_stage="engineering_pretest", split_file=str(self.split_path),
            preselected_snp_file=str(self.indices_path), snp_file=str(self.snp_path),
            pheno_dir=str(self.root), phenotype_names=["RGBVI"],
            time_coordinates=[10, 12, 14], epochs=1, batch_size=2,
            num_workers=0, screen_k=0, gwas_k=0, screen_missing_strategy="mode",
            wide_missing_strategy="mode", device="cpu", d_model=8, num_heads=2,
            d_ff=16, degree=1, dropout=0, use_warmup=False,
            gwas_pca_panel_size=31,
            gwas_pca_algorithm_version="sklearn_randomized_centered_even_panel_v1")

    def test_train_val_only_reuses_production_train_and_strict_checkpoint(self):
        config = self.config()
        root = Path(config.output_dir)
        root.mkdir(parents=True)
        panel_path = root / "gwas_pca_panel_indices.npy"
        np.save(panel_path, np.arange(31, dtype=np.int64))
        config = config.model_copy(update={
            "gwas_pca_panel_indices_file": panel_path.name,
            "gwas_pca_panel_indices_sha256": file_digest(panel_path),
        })
        trainer = TFPPTrainer(config)
        parsed = set(np.concatenate(trainer.dataset.parsed_phenotype_rows).tolist())
        self.assertEqual(parsed, set(range(13)))
        self.assertTrue(parsed.isdisjoint(self.split.test_idx.tolist()))
        self.assertIsNone(trainer.test_loader)
        with patch.object(trainer, "_train_epoch", wraps=trainer._train_epoch) as production_train, \
             patch.object(trainer, "_evaluate_test_set", side_effect=AssertionError("test forbidden")) as test_eval:
            result = trainer.train()
        self.assertEqual(production_train.call_count, 1)
        test_eval.assert_not_called()
        self.assertIsNone(result["test_results"])
        self.assertFalse(result["paper_eligible"])
        self.assertFalse((root / "test_evaluation.json").exists())
        self.assertFalse(any("prediction" in p.name.lower() for p in root.rglob("*")))
        model, info = load_tfpp_model(root, device="cpu", verbose=False)
        self.assertFalse(info["paper_eligible"])
        self.assertEqual(info["artifacts"].metadata["purpose"], "engineering_pretest_smoke")
        self.assertEqual(info["artifacts"].metadata["phenotype_rows_parsed"], [list(range(13))])
        self.assertEqual(info["snp_dim"], 5)
        model_config = json.loads((root / "models/model_config.json").read_text())
        preprocessing = json.loads((root / "preprocessing/preprocessing.json").read_text())
        for record in (model_config, preprocessing):
            self.assertEqual(record["gwas_population_structure"], "pca")
            self.assertEqual(record["gwas_n_pcs"], 5)
            self.assertEqual(record["gwas_pca_panel_size"], 31)
            self.assertIsNone(record["gwas_pca_panel_seed"])
            self.assertEqual(record["gwas_pca_algorithm_version"],
                             "sklearn_randomized_centered_even_panel_v1")
            self.assertEqual(record["gwas_pca_panel_indices_file"], panel_path.name)
            self.assertEqual(record["gwas_pca_panel_indices_sha256"], file_digest(panel_path))
        summary = summarize([root])
        self.assertEqual(summary["groups"], [])
        self.assertEqual(len(summary["rejected"]), 1)
        with self.assertRaises(FileExistsError):
            TFPPTrainer(self.config())

    def test_test_labels_never_change_scaler_or_checkpoint(self):
        first = TFPPTrainer(self.config())
        scaler_before = first.scalers["pheno_meta"]["mean_curve"].copy()
        frame = pd.read_csv(self.root / "RGBVI.csv")
        frame.iloc[self.split.test_idx] += 1e6
        frame.to_csv(self.root / "RGBVI.csv", index=False)
        second_config = self.config().model_copy(update={"exp_name": "pretest_changed_test"})
        second = TFPPTrainer(second_config)
        np.testing.assert_array_equal(scaler_before, second.scalers["pheno_meta"]["mean_curve"])
        np.testing.assert_array_equal(first.dataset.phenotype[self.split.train_idx],
                                      second.dataset.phenotype[self.split.train_idx])
        self.assertTrue(np.isnan(second.dataset.phenotype[self.split.test_idx]).all())

    def test_streaming_screen_uses_train_rows_and_is_heldout_invariant(self):
        phenotype = self.pheno[:, None, :]
        selected, meta = gwas_dc_sis_screen_mmap(
            self.snp_path, self.split.train_idx, phenotype[self.split.train_idx],
            gwas_k=12, screen_k=5, block_columns=7, pca_marker_count=31)
        self.assertEqual(len(selected), 5)
        self.assertTrue(meta["mmap"])
        self.assertFalse(meta["full_matrix_copy_created"])
        self.assertEqual(meta["gwas_population_structure"], "pca")
        self.assertEqual(meta["gwas_n_pcs"], 5)
        self.assertIsNone(meta["gwas_pca_panel_seed"])
        self.assertEqual(meta["gwas_pca_algorithm_version"],
                         "sklearn_randomized_centered_even_panel_v1")
        self.assertEqual(meta["pca_scaling"], "center_only_no_standard_deviation")
        self.assertEqual(len(meta["pca_panel_indices"]), 31)
        changed = self.snp.copy()
        changed[self.split.val_idx] = 2 - changed[self.split.val_idx]
        changed[self.split.test_idx] = 1
        changed_phenotype = phenotype.copy()
        changed_phenotype[self.split.val_idx] += 1e6
        changed_phenotype[self.split.test_idx] -= 1e6
        other = self.root / "heldout_changed.npy"
        np.save(other, changed)
        selected_again, heldout_meta = gwas_dc_sis_screen_mmap(
            other, self.split.train_idx, changed_phenotype[self.split.train_idx],
            gwas_k=12, screen_k=5, block_columns=5, pca_marker_count=31)
        np.testing.assert_array_equal(selected, selected_again)
        for key in ("pca_panel_indices_sha256", "pca_scores_sha256",
                    "gwas_candidate_scores_sha256", "training_indices_sha256"):
            self.assertEqual(meta[key], heldout_meta[key])

        repeated, repeated_meta = gwas_dc_sis_screen_mmap(
            self.snp_path, self.split.train_idx, phenotype[self.split.train_idx],
            gwas_k=12, screen_k=5, block_columns=7, pca_marker_count=31)
        np.testing.assert_array_equal(selected, repeated)
        for key in ("pca_panel_indices_sha256", "pca_scores_sha256",
                    "gwas_candidate_scores_sha256"):
            self.assertEqual(meta[key], repeated_meta[key])

        train_changed = self.snp.copy()
        train_changed[self.split.train_idx[0], 0] = (
            int(train_changed[self.split.train_idx[0], 0]) + 1) % 3
        train_path = self.root / "train_changed.npy"
        np.save(train_path, train_changed)
        _, train_meta = gwas_dc_sis_screen_mmap(
            train_path, self.split.train_idx, phenotype[self.split.train_idx],
            gwas_k=12, screen_k=5, block_columns=7, pca_marker_count=31)
        self.assertNotEqual(meta["pca_scores_sha256"], train_meta["pca_scores_sha256"])

        subset = self.split.train_idx[:-1]
        _, subset_meta = gwas_dc_sis_screen_mmap(
            self.snp_path, subset, phenotype[subset], gwas_k=12, screen_k=5,
            block_columns=7, pca_marker_count=31)
        self.assertNotEqual(meta["training_indices_sha256"],
                            subset_meta["training_indices_sha256"])

    def test_zero_variance_pca_panel_is_finite_and_cache_binds_configuration(self):
        phenotype = self.pheno[:, None, :]
        constant = self.snp.copy()
        constant[self.split.train_idx, :4] = 1
        path = self.root / "constant_panel.npy"
        np.save(path, constant)
        selected, meta = gwas_dc_sis_screen_mmap(
            path, self.split.train_idx, phenotype[self.split.train_idx],
            gwas_k=12, screen_k=5, block_columns=6, pca_marker_count=31)
        self.assertEqual(len(selected), 5)
        self.assertGreaterEqual(meta["pca_zero_variance_marker_count"], 4)
        self.assertRegex(meta["pca_scores_sha256"], r"^[0-9a-f]{64}$")

        from utils.Screening import compute_cache_key
        kwargs = dict(snp_file=str(path), pheno_source=str(self.root),
                      phenotype_names=["RGBVI"], screen_k=5, gwas_k=12,
                      seed=73, train_indices=self.split.train_idx,
                      train_genotype=constant[self.split.train_idx],
                      train_phenotype=phenotype[self.split.train_idx])
        base = compute_cache_key(**kwargs)
        self.assertNotEqual(base, compute_cache_key(**kwargs, gwas_n_pcs=3))
        self.assertNotEqual(base, compute_cache_key(
            **kwargs, gwas_pca_panel_size=20,
            gwas_pca_algorithm_version="panel_v1"))

    def test_config_rejects_implicit_or_kfold_pretest(self):
        base = self.config().model_dump()
        for update in ({"split_file": None}, {"preselected_snp_file": None}, {"use_kfold": True}):
            with self.assertRaises(ValueError):
                TFPPConfig.model_validate({**base, **update})


if __name__ == "__main__":
    unittest.main()

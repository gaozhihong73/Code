"""普通 TFPP 的训练个体隔离、双视图与 CPU 训练回归。"""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from config import TFPPConfig
from model import TFPP
from trainer import TFPPTrainer
from utils.Genotype import dosage_to_model_codes, tfpp_input_views, validate_dosage
from utils.model_loader import load_tfpp_model
from utils.Screening import screen_snps, training_indices_hash
from utils.Split import generate_split, load_split, save_split


class LocalConfig(TFPPConfig):
    artifact_root: str

    @property
    def output_dir(self):
        return str(Path(self.artifact_root) / self.exp_name)


class RegularTFPPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split = generate_split(40, ratios=(0.6, 0.2, 0.2), seed=73)
        self.split_path = self.root / "split.npz"
        save_split(self.split, self.split_path)
        rng = np.random.default_rng(73)
        self.snp = rng.integers(0, 3, (40, 12), dtype=np.int8)
        self.pheno = np.stack(
            [
                self.snp[:, 0, None]
                + np.array([0, 1, 2])
                + rng.normal(0, 0.2, (40, 3)),
                self.snp[:, 3, None] * np.array([1, 2, 3])
                + rng.normal(0, 0.2, (40, 3)),
            ],
            axis=1,
        ).astype(np.float32)
        self.snp[self.split.train_idx[0], 0] = -1
        self.snp[self.split.train_idx[1], 1] = -1
        self.counter = 0
        self.write_inputs(self.snp, self.pheno)

    def write_inputs(self, snp, pheno):
        np.save(self.root / "snp.npy", snp)
        for i, name in enumerate(("A", "B")):
            pd.DataFrame(pheno[:, i]).to_csv(self.root / f"{name}.csv", index=False)

    def config(self, **kwargs):
        self.counter += 1
        options = dict(
            artifact_root=str(self.root),
            exp_name=f"run_{self.counter}",
            snp_file=str(self.root / "snp.npy"),
            pheno_dir=str(self.root),
            phenotype_names=["A", "B"],
            time_coordinates=[1, 2, 3],
            split_file=str(self.split_path),
            device="cpu",
            num_workers=0,
            screen_cache_dir=str(self.root / "cache"),
            screen_k=4,
            gwas_k=4,
            screen_multivariate=True,
            screen_missing_strategy="mode",
            wide_missing_strategy="mode",
            d_model=8,
            num_heads=2,
            num_layers=1,
            d_ff=16,
            degree=1,
            epochs=1,
            batch_size=8,
            dropout=0,
            use_warmup=False,
            l1_wide_weight=0,
        )
        options.update(kwargs)
        return LocalConfig(**options)

    def trainer(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            result = TFPPTrainer(self.config(**kwargs))
        self.addCleanup(self.close_logger, result)
        return result

    @staticmethod
    def close_logger(trainer):
        for handler in list(trainer.logger.logger.handlers):
            handler.close()
            trainer.logger.logger.removeHandler(handler)

    def assert_same_training_artifacts(self, first, second):
        np.testing.assert_array_equal(first.screened_indices, second.screened_indices)
        np.testing.assert_array_equal(first.wide_fill_values, second.wide_fill_values)
        np.testing.assert_array_equal(
            first.train_loader.dataset.tensors[1],
            second.train_loader.dataset.tensors[1],
        )
        self.assertEqual(first.screening_cache_key, second.screening_cache_key)
        for key in ("screening_imputer", "wide_imputer", "train_indices_sha256"):
            self.assertEqual(
                first.preprocessing_metadata[key], second.preprocessing_metadata[key]
            )
        for name in ("A", "B"):
            a = first.scalers["pheno_meta"]["scalers"][name]
            b = second.scalers["pheno_meta"]["scalers"][name]
            for attr in ("mean_", "var_", "scale_"):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
        np.testing.assert_array_equal(
            first.scalers["pheno_meta"]["mean_curve"],
            second.scalers["pheno_meta"]["mean_curve"],
        )

    def test_held_out_labels_and_genotypes_cannot_change_training_artifacts(self):
        base = self.trainer()
        held = np.concatenate([self.split.val_idx, self.split.test_idx])
        for kind in ("phenotype", "genotype", "both"):
            with self.subTest(kind=kind):
                snp, pheno = self.snp.copy(), self.pheno.copy()
                if kind in ("phenotype", "both"):
                    pheno[held] = np.random.default_rng(9).normal(
                        10000, 1000, pheno[held].shape
                    )
                if kind in ("genotype", "both"):
                    snp[held] = -1  # 改变 held-out 缺失分布也不能改变填补参数。
                self.write_inputs(snp, pheno)
                current = self.trainer()
                self.assert_same_training_artifacts(base, current)
                # 缓存命中和计算路径都必须不受 held-out 影响。
                with (
                    redirect_stdout(io.StringIO()),
                    patch("utils.Screening._load_cache", return_value=None),
                ):
                    recomputed = self.trainer()
                self.assert_same_training_artifacts(base, recomputed)

    def test_training_changes_invalidate_cache(self):
        base = self.trainer()
        changed = self.pheno.copy()
        changed[self.split.train_idx[0], 0, 0] += 123
        self.write_inputs(self.snp, changed)
        self.assertNotEqual(
            base.screening_cache_key, self.trainer().screening_cache_key
        )
        snp = self.snp.copy()
        snp[self.split.train_idx[2], 2] = (int(snp[self.split.train_idx[2], 2]) + 1) % 3
        self.write_inputs(snp, self.pheno)
        self.assertNotEqual(
            base.screening_cache_key, self.trainer().screening_cache_key
        )
        other_split = self.root / "other.npz"
        save_split(generate_split(40, ratios=(0.6, 0.2, 0.2), seed=74), other_split)
        self.assertNotEqual(
            base.screening_cache_key,
            self.trainer(split_file=str(other_split)).screening_cache_key,
        )

    def test_selector_receives_only_training_dosage_and_labels(self):
        with patch("utils.screen_snps", wraps=screen_snps) as selector:
            trainer = self.trainer()
        selector.assert_called_once()
        dosage, labels = selector.call_args.args[:2]
        self.assertEqual(dosage.shape, (len(self.split.train_idx), 12))
        validate_dosage(dosage)
        np.testing.assert_array_equal(labels, self.pheno[self.split.train_idx])
        self.assertTrue(selector.call_args.kwargs["multivariate"])
        np.testing.assert_array_equal(
            trainer.dataset.snp,
            dosage_to_model_codes(self.snp)[:, trainer.screened_indices],
        )
        self.assertIn(0, trainer.dataset.snp)  # Deep 缺失类别未被筛选填补覆盖。
        np.testing.assert_array_equal(trainer.dataset.sample_ids, np.arange(40))

    def test_preprocessing_artifacts_are_complete_and_verifiable(self):
        trainer = self.trainer()
        directory = Path(trainer.config.preprocessing_save_dir)
        meta = json.loads((directory / "preprocessing.json").read_text())
        copied = load_split(directory / "split.npz", n_samples=40)
        np.testing.assert_array_equal(copied.train_idx, self.split.train_idx)
        self.assertEqual(
            meta["split_sha256"],
            hashlib.sha256((directory / "split.npz").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            meta["train_indices_sha256"], training_indices_hash(self.split.train_idx)
        )
        self.assertEqual(meta["original_snp_dim"], 12)
        self.assertEqual(meta["snp_dim"], trainer.snp_dim)
        self.assertEqual(meta["screen_multivariate"], True)
        self.assertEqual(meta["screen_missing_strategy"], "mode")
        self.assertEqual(
            meta["screening_imputer"]["original_train_idx"],
            self.split.train_idx.tolist(),
        )
        self.assertEqual(
            meta["wide_imputer"]["train_idx"], self.split.train_idx.tolist()
        )
        self.assertTrue(Path(trainer.scaler_file).exists())
        indices = np.load(directory / "screened_indices.npy", allow_pickle=False)
        np.testing.assert_array_equal(indices, trainer.screened_indices)
        self.assertTrue((indices < 12).all())
        saved = json.loads(
            (trainer.best_model_save_dir / "model_config.json").read_text()
        )
        self.assertEqual(saved["snp_dim"], trainer.snp_dim)
        self.assertEqual(saved["wide_input_mode"], "dosage_v1")

    def test_correct_deep_and_wide_views_and_missing_fill(self):
        codes = torch.tensor([[1, 2, 3], [0, 0, 0]])
        deep, wide = tfpp_input_views(codes, torch.tensor([2.0, 1.0, 0.0]))
        torch.testing.assert_close(deep, codes)
        torch.testing.assert_close(
            wide, torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
        )
        with self.assertRaisesRegex(ValueError, "训练期保存"):
            tfpp_input_views(codes)
        trainer = self.trainer()
        sample = torch.tensor(trainer.dataset.snp)
        deep, wide = tfpp_input_views(sample, trainer.model.wide_fill_values)
        missing = deep == 0
        fills = trainer.model.wide_fill_values.expand_as(wide)
        torch.testing.assert_close(wide[missing], fills[missing] / 2)
        self.assertTrue(((wide >= 0) & (wide <= 1)).all())

    def test_all_wide_and_deep_modes_forward_backward(self):
        for mode in ("all", "wide", "deep"):
            with self.subTest(mode=mode):
                trainer = self.trainer(ablation_mode=mode)
                x, y = next(iter(trainer.train_loader))
                prediction = trainer.model(x)
                self.assertEqual(prediction.shape, y.shape)
                loss = (prediction - y).square().mean()
                loss.backward()
                grads = [
                    p.grad for p in trainer.model.parameters() if p.grad is not None
                ]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_cpu_one_epoch_and_checkpoint_round_trip(self):
        trainer = self.trainer()
        with redirect_stdout(io.StringIO()):
            result = trainer.train()
        self.assertTrue(np.isfinite(result["best_val_loss"]))
        self.assertEqual(result["best_epoch"], 1)
        loaded, info = load_tfpp_model(trainer.config.output_dir, verbose=False)
        self.assertEqual(info["wide_input_mode"], "dosage_v1")
        torch.testing.assert_close(
            loaded.wide_fill_values, trainer.model.wide_fill_values
        )
        trainer.model.eval()
        x = trainer.test_loader.dataset.tensors[0]
        with torch.inference_mode():
            torch.testing.assert_close(loaded(x), trainer.model(x), rtol=0, atol=0)

    def test_legacy_checkpoint_preserves_explicit_legacy_input_mode(self):
        model = TFPP(
            3,
            8,
            2,
            1,
            16,
            4,
            [1, 2, 3],
            0.0,
            ["A"],
            wide_input_mode="legacy_categorical",
        ).eval()
        path = self.root / "legacy.pt"
        torch.save(model.state_dict(), path)
        with self.assertWarnsRegex(RuntimeWarning, "LEGACY"):
            restored, info = load_tfpp_model(path, num_heads=2, verbose=False, legacy=True)
        self.assertEqual(info["wide_input_mode"], "legacy_categorical")
        with torch.inference_mode():
            x = torch.tensor([[0.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_no_screening_still_persists_identity_column_indices(self):
        trainer = self.trainer(screen_k=0, gwas_k=0)
        np.testing.assert_array_equal(trainer.screened_indices, np.arange(12))
        self.assertEqual(trainer.snp_dim, 12)

    def test_explicit_multivariate_override_is_logged_saved_and_cached(self):
        joint = self.trainer()
        with patch("utils.screen_snps", wraps=screen_snps) as selector:
            separate = self.trainer(screen_multivariate=False)
        self.assertIs(selector.call_args.kwargs["multivariate"], False)
        self.assertNotEqual(joint.screening_cache_key, separate.screening_cache_key)
        self.assertIs(separate.preprocessing_metadata["screen_multivariate"], False)
        saved = json.loads(
            (separate.best_model_save_dir / "model_config.json").read_text()
        )
        self.assertIs(saved["screen_multivariate"], False)
        separate.logger.flush()
        self.assertIn(
            "screen_multivariate=False", Path(separate.config.log_file).read_text()
        )

    def test_error_screening_rejects_training_missing_and_kfold_stays_legacy(self):
        with self.assertRaisesRegex(ValueError, "禁止当作剂量0"):
            self.trainer(screen_missing_strategy="error")
        self.write_inputs(np.maximum(self.snp, 0), self.pheno)
        with (
            patch("utils.screen_snps", wraps=screen_snps) as selector,
            patch("utils.compute_cache_key", return_value=None),
        ):
            trainer = self.trainer(
                use_kfold=True, split_file=None, screen_missing_strategy="error"
            )
        self.assertEqual(selector.call_args.args[0].shape[0], 40)
        self.assertEqual(trainer.model.wide_input_mode, "legacy_categorical")

    def test_new_model_cannot_silently_use_legacy_inference_rescreening(self):
        import infer

        trainer = self.trainer()
        torch.save(trainer.model.state_dict(), trainer.best_model_save_dir / "best_model.pt")
        (Path(trainer.config.preprocessing_save_dir) / "split.npz").unlink()
        with patch("utils.Screening.screen_snps") as selector:
            with self.assertRaisesRegex(FileNotFoundError, "split.npz"):
                infer.predict(["--model_path", trainer.config.output_dir, "--legacy"])
        selector.assert_not_called()


if __name__ == "__main__":
    unittest.main()

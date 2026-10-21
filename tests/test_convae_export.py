"""严格最佳模型的 SNP-only、确定性、manifest 行序和可追溯导出回归。"""

import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import test_convae_model as model_tests
from utils.ConvAE_Export import (
    LATENT_LAYOUT, encode_feature_batches, export_convae_features, load_exported_features,
)
from utils.Genotype import dosage_to_model_codes
from utils.Inference import file_sha256
from utils.Split import generate_split, save_split
from utils.convae_model_loader import load_convae_model


class ConvAEExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.helper = model_tests.ConvAEModelTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.fixture = self.helper.fixture
        self.root = self.fixture.root
        self.trainer = self.helper.trainer()
        self.trainer.train()
        self.experiment = Path(self.trainer.config.output_dir)
        self.counter = 0

    def export(self, **options):
        self.counter += 1
        defaults = dict(split_file=self.fixture.split_path, snp_file=self.root / "snp.npy",
                        output_root=self.root / f"export_{self.counter}", batch_size=4)
        defaults.update(options)
        return export_convae_features(self.experiment, **defaults)

    def test_encoder_only_has_no_random_masks_heads_decoder_or_phenotype_inputs(self):
        model, _ = load_convae_model(self.experiment)
        x, padding = self.helper.inputs()
        model.train()
        rng = torch.get_rng_state().clone()
        with (patch.object(model.phenotype_head, "forward", side_effect=AssertionError("辅助头被调用")),
              patch.object(model.decoder, "forward", side_effect=AssertionError("decoder 被调用")),
              patch.object(importlib.import_module("model.ConvAE"), "sample_patch_mask",
                           side_effect=AssertionError("随机掩码被调用"))):
            features = model.encode_features(x, padding_mask=padding)
            torch.testing.assert_close(features, model.encode_features(x, padding_mask=padding), rtol=0, atol=0)
        self.assertFalse(model.training)
        self.assertFalse(features.requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        with torch.inference_mode():
            native = model(x, mask_ratio=0, padding_mask=padding)[0]
        torch.testing.assert_close(features, native.flatten(1), rtol=0, atol=0)
        self.assertEqual(features.shape, (2, 8))
        with self.assertRaisesRegex(ValueError, "padding_mask"):
            model.encode_features(x, padding_mask=None)
        with self.assertRaises(TypeError):
            model.encode_features(x, padding_mask=padding, phenotype=torch.zeros(2))

    def test_three_role_shapes_ids_layout_hashes_and_raw_snp_equivalence(self):
        output = self.export()
        arrays, meta = load_exported_features(output)
        self.assertEqual(meta["latent_layout"], LATENT_LAYOUT)
        self.assertEqual(meta["latent_dim"], 8)
        self.assertEqual(meta["original_snp_dim"], 13)
        self.assertEqual(meta["structural_padding_count"], 1)
        self.assertTrue(meta["encoder_only"])
        self.assertEqual(meta["reconstruction_mask_ratio"], 0)
        self.assertEqual(len(meta["git_commit"]), 40)
        self.assertEqual(meta["checkpoint_sha256"], file_sha256(self.experiment / "models/best_model.pt"))
        model, info = load_convae_model(self.experiment)
        for role, (features, ids) in arrays.items():
            indices = getattr(self.fixture.split, role + "_idx")
            np.testing.assert_array_equal(ids, indices)
            self.assertEqual(features.shape, (len(indices), 8))
            self.assertEqual(features.dtype, np.float32)
            codes = dosage_to_model_codes(self.fixture.snp[indices])
            padded = torch.tensor(np.pad(codes, ((0, 0), (0, 1)))).long()
            chunks = padded.unfold(1, 8, 6)
            direct = model.encode_features(chunks, padding_mask=torch.as_tensor(info["artifacts"].chunk_padding_mask))
            np.testing.assert_allclose(features, direct.numpy(), rtol=1e-5, atol=1e-6)
            self.assertEqual(meta["arrays"][role]["sample_ids"], indices.tolist())

    def test_all_roles_phenotype_changes_or_absent_csv_never_change_features(self):
        original, _ = load_exported_features(self.export())
        for role in ("train", "val", "test"):
            changed = self.fixture.pheno.copy()
            changed[getattr(self.fixture.split, role + "_idx")] = np.nan
            self.fixture.write_inputs(self.fixture.snp, changed)
            current, _ = load_exported_features(self.export())
            for key in original:
                np.testing.assert_array_equal(original[key][0], current[key][0])
        for name in ("A", "B"):
            (self.root / f"{name}.csv").unlink()
        with patch("pandas.read_csv", side_effect=AssertionError("导出不得读取表型")):
            current, _ = load_exported_features(self.export())
        for role in original:
            np.testing.assert_array_equal(original[role][0], current[role][0])

    def test_repeated_batch_sizes_and_shuffled_loader_restore_exact_manifest_order(self):
        original, _ = load_exported_features(self.export(batch_size=1))
        repeated, _ = load_exported_features(self.export(batch_size=1))
        shuffled, _ = load_exported_features(self.export(batch_size=7, loader_factory=lambda ds, bs:
            DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(99))))
        for role in original:
            np.testing.assert_array_equal(original[role][0], repeated[role][0])
            np.testing.assert_array_equal(original[role][1], shuffled[role][1])
            np.testing.assert_allclose(original[role][0], shuffled[role][0], rtol=1e-5, atol=1e-6)

    def test_padding_values_including_nan_do_not_change_encoder_features(self):
        model, _ = load_convae_model(self.experiment)
        x, padding = self.helper.inputs()
        first = model.encode_features(x, padding_mask=padding)
        changed = x.float()
        changed[:, padding] = float("nan")
        torch.testing.assert_close(first, model.encode_features(changed, padding_mask=padding), rtol=0, atol=0)
        # 不完整 chunk 的最后真实 SNP 仍属于编码路径。
        changed = x.clone()
        changed[:, 1, 3] = (changed[:, 1, 3] % 3) + 1
        self.assertFalse(torch.equal(first, model.encode_features(changed, padding_mask=padding)))

    def test_duplicate_missing_extra_and_phenotype_batches_fail(self):
        model, _ = load_convae_model(self.experiment)
        x, padding = self.helper.inputs()
        for batches, message in (([(x, [1, 1])], "重复"), ([(x[:1], [1])], "缺失"),
                                 ([(x, [1, 3])], "额外"), ([(x, [1.0, 2.0])], "sample_id"),
                                 ([(x, torch.ones(2, 2, 3), [1, 2])], "phenotype")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                encode_feature_batches(model, batches, padding_mask=padding, expected_sample_ids=[1, 2])
        with self.assertRaisesRegex(ValueError, "重复"):
            encode_feature_batches(model, [], padding_mask=padding, expected_sample_ids=[1, 1])

    def test_expected_checkpoint_split_and_config_mismatch_fail_before_write(self):
        model, info = load_convae_model(self.experiment)
        with self.assertRaisesRegex(ValueError, "checkpoint 哈希"):
            self.export(expected_checkpoint_sha256="0" * 64)
        changed = {**info["model_config"], "d_ext": 99}
        with self.assertRaisesRegex(ValueError, "模型配置"):
            self.export(expected_model_config=changed)
        other = self.root / "other.npz"
        save_split(generate_split(40, seed=74), other)
        with self.assertRaisesRegex(ValueError, "split/fold 文件哈希"):
            self.export(split_file=other)
        self.export(expected_model_config=info["model_config"],
                    expected_checkpoint_sha256=file_sha256(self.experiment / "models/best_model.pt"))

    def test_missing_or_tampered_artifacts_and_invalid_snp_fail(self):
        source = self.root / "bad.npy"
        for values in (np.zeros((39, 13)), np.zeros((40, 12)), np.full((40, 13), 3)):
            np.save(source, values)
            with self.assertRaises(ValueError):
                self.export(snp_file=source)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            self.export(batch_size=0)
        prep = self.experiment / "preprocessing/preprocessing.json"
        saved = prep.read_bytes()
        prep.unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        prep.write_bytes(saved)
        for filename, field, value in (("config.json", "d_ext", 99),
                                       ("preprocessing/preprocessing.json", "schema_version", 2)):
            path = self.experiment / filename
            content = path.read_bytes()
            changed = json.loads(content)
            changed[field] = value
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.export()
            path.write_bytes(content)
        split = self.experiment / "preprocessing/split.npz"
        split.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "哈希"):
            self.export()

    def test_post_write_reload_rejects_content_or_row_order_corruption(self):
        output = self.export()
        features = output / "features/test.npy"
        original = features.read_bytes()
        features.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "哈希"):
            load_exported_features(output)
        features.write_bytes(original)
        meta_path = output / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["arrays"]["test"]["sample_ids"].reverse()
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "顺序"):
            load_exported_features(output)

    def test_different_checkpoints_splits_and_folds_never_overwrite(self):
        base = self.root / "exports"
        first = self.export(output_root=base)
        original = (first / "metadata.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.export(output_root=base)
        # 相同 split 的不同最佳模型。
        second_trainer = self.helper.trainer()
        second_trainer.train()
        second = export_convae_features(second_trainer.config.output_dir, split_file=self.fixture.split_path,
            snp_file=self.root / "snp.npy", output_root=base)
        self.assertNotEqual(first, second)
        # 新 split，以及一个明确的 TFPP-compatible fold manifest。
        for fold in (False, True):
            new_split = generate_split(40, ratios=(0.6, 0.2, 0.2), seed=81 if fold else 80)
            if fold:
                new_split.metadata.update(fold_index=1, n_folds=2, kfold_val_ratio=0.2, validation_seed=81)
            split_path = self.root / f"split_{fold}.npz"
            save_split(new_split, split_path)
            trainer = self.helper.trainer(split_file=str(split_path))
            trainer.train()
            result = export_convae_features(trainer.config.output_dir, split_file=split_path,
                snp_file=self.root / "snp.npy", output_root=base)
            self.assertNotEqual(first.parts[-3], result.parts[-3])
            _, meta = load_exported_features(result)
            self.assertEqual(meta["fold_metadata"]["fold_index"], 1 if fold else None)
        self.assertEqual((first / "metadata.json").read_bytes(), original)

    def test_one_epoch_cpu_checkpoint_cli_exports_without_phenotype_files(self):
        for name in ("A", "B"):
            (self.root / f"{name}.csv").unlink()
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        process = subprocess.run([sys.executable, "-B", "export_convae.py", "--model_path", str(self.experiment),
            "--split_file", str(self.fixture.split_path), "--snp_file", str(self.root / "snp.npy"),
            "--output_root", str(self.root / "cli_export"), "--batch_size", "3", "--device", "cpu"],
            check=True, capture_output=True, text=True, env=env)
        self.assertIn("已导出并校验", process.stdout)
        outputs = list((self.root / "cli_export").glob("*/*/*/metadata.json"))
        self.assertEqual(len(outputs), 1)
        arrays, _ = load_exported_features(outputs[0].parent)
        self.assertEqual([len(arrays[role][0]) for role in ("train", "val", "test")], [24, 8, 8])

    def test_no_auxiliary_model_exports_same_fixed_layout(self):
        trainer = self.helper.trainer(use_pheno_help=False, recon_weight=1, pheno_weight=0)
        trainer.train()
        output = export_convae_features(trainer.config.output_dir, split_file=self.fixture.split_path,
            output_root=self.root / "no_aux", batch_size=3)
        arrays, meta = load_exported_features(output)
        self.assertFalse(meta["use_pheno_help"])
        self.assertEqual(meta["latent_dim"], 8)
        self.assertEqual(arrays["train"][0].shape, (24, 8))


if __name__ == "__main__":
    unittest.main()

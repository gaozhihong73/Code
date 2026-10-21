"""ConvAE 训练个体隔离、共享 split、padding 和 schema 3 CPU 数据管线测试。"""

import copy
import importlib
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import test_tfpp_regular as regular_tests
from config import ConvAEConfig, TFPPConfig
from dataloader.ConvAE_DataLoader import ConvAEDataset, get_mae_dataloader, get_mae_dataloaders
from trainer import ConvAETrainer
from utils.ConvAE_Preprocessing import load_convae_preprocessing
from utils.Inference import file_sha256, transform_phenotypes
from utils.Normalization import denormalize_phenotype
from utils.Split import generate_split, save_split


class ConvAEDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = regular_tests.RegularTFPPTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.counter = 0

    def dataset(self, **options):
        return ConvAEDataset(str(self.fixture.root / "snp.npy"), str(self.fixture.root),
                             ["A", "B"], **options)

    def config(self, **options):
        self.counter += 1
        defaults = dict(exp_name=f"data_{self.counter}", output_root=str(self.fixture.root / "mae"),
                        split_file=str(self.fixture.split_path), snp_file=str(self.fixture.root / "snp.npy"),
                        pheno_dir=str(self.fixture.root), phenotype_names=["A", "B"], chunk_size=8,
                        overlap=2, time_coordinates=[1, 2, 3], num_workers=0, device="cpu")
        defaults.update(options)
        return ConvAEConfig(**defaults)

    def trainer(self, *, indices=None, **options):
        trainer = ConvAETrainer(self.config(**options), indices=indices, initialize_model=False)
        self.addCleanup(self.fixture.close_logger, trainer)
        return trainer

    def test_raw_dataset_never_fits_or_normalizes_phenotypes(self):
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("Dataset 不得拟合")):
            dataset = self.dataset(chunk_size=8, overlap=2)
        np.testing.assert_array_equal(dataset.phenotype, self.fixture.pheno)
        np.testing.assert_array_equal(dataset[0][1], self.fixture.pheno[0])
        self.assertFalse(hasattr(dataset, "phenotype_norm"))
        with self.assertRaisesRegex(ValueError, "IndividualSplit"):
            dataset.apply_global_zscore_norm()

    def test_fit_receives_only_training_phenotypes_and_batches_use_transformed_views(self):
        dataset = self.dataset(chunk_size=8, overlap=2)
        seen = []
        fit = StandardScaler.fit

        def record(scaler, values, *args, **kwargs):
            seen.append(values.copy())
            return fit(scaler, values, *args, **kwargs)

        with patch.object(StandardScaler, "fit", record):
            data = get_mae_dataloaders(dataset, self.fixture.split, batch_size=3, num_workers=0)
        self.assertEqual(len(seen), 2)
        for p, name in enumerate(dataset.phenotype_names):
            train = self.fixture.pheno[self.fixture.split.train_idx, p].reshape(-1, 1)
            np.testing.assert_array_equal(seen[p], train)
            scaler = data.scalers["pheno_meta"]["scalers"][name]
            np.testing.assert_allclose(scaler.mean_, train.astype(float).mean(axis=0))
            np.testing.assert_allclose(scaler.var_, train.astype(float).var(axis=0))
            self.assertEqual(scaler.n_samples_seen_, train.size)
        for loader, rows in ((data.val_loader, self.fixture.split.val_idx),
                             (data.test_loader, self.fixture.split.test_idx)):
            batched = torch.cat([phenotype for _, phenotype in loader])
            expected = transform_phenotypes(self.fixture.pheno[rows], data.scalers)
            torch.testing.assert_close(batched, expected)
            np.testing.assert_allclose(denormalize_phenotype(batched, data.scalers),
                                       self.fixture.pheno[rows], atol=1e-5)
        np.testing.assert_array_equal(dataset.phenotype, self.fixture.pheno)

    def test_shared_roles_are_disjoint_and_all_sample_orders_match_tfpp(self):
        tfpp = self.fixture.trainer(screen_k=0, gwas_k=0)
        mae = self.trainer()
        for key, loader in (("train_idx", mae.train_loader), ("val_idx", mae.val_loader),
                            ("test_idx", mae.test_loader)):
            expected = getattr(tfpp.split, key)
            np.testing.assert_array_equal(getattr(mae.split, key), expected)
            np.testing.assert_array_equal(loader.dataset.sample_ids, tfpp.dataset.sample_ids[expected])
            self.assertEqual(loader.dataset.indices, expected.tolist())
        groups = [set(getattr(mae.split, key)) for key in ("train_idx", "val_idx", "test_idx")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(set.union(*groups), set(range(40)))
        self.assertEqual(mae.preprocessing_metadata["sample_id_source"], "row_index")

    def test_heldout_labels_cannot_change_scaler_or_train_view(self):
        first = self.trainer()
        changed = self.fixture.pheno.copy()
        held = np.concatenate([self.fixture.split.val_idx, self.fixture.split.test_idx])
        changed[held] += 10000
        self.fixture.write_inputs(self.fixture.snp, changed)
        second = self.trainer()
        for name in ("A", "B"):
            a = first.scalers["pheno_meta"]["scalers"][name]
            b = second.scalers["pheno_meta"]["scalers"][name]
            np.testing.assert_array_equal(a.mean_, b.mean_)
            np.testing.assert_array_equal(a.scale_, b.scale_)
        for key in ("scaler_sha256", "train_phenotype_sha256", "train_indices_sha256", "split_sha256"):
            self.assertEqual(first.preprocessing_metadata[key], second.preprocessing_metadata[key])
        torch.testing.assert_close(first.train_loader.dataset.normalized_phenotype,
                                   second.train_loader.dataset.normalized_phenotype, rtol=0, atol=0)
        self.assertFalse(torch.equal(first.test_loader.dataset.normalized_phenotype,
                                     second.test_loader.dataset.normalized_phenotype))

    def test_training_label_changes_change_saved_scaler(self):
        first = self.trainer()
        changed = self.fixture.pheno.copy()
        changed[self.fixture.split.train_idx, 0] += 10
        self.fixture.write_inputs(self.fixture.snp, changed)
        second = self.trainer()
        self.assertNotEqual(first.preprocessing_metadata["scaler_sha256"], second.preprocessing_metadata["scaler_sha256"])
        self.assertNotEqual(first.preprocessing_metadata["train_phenotype_sha256"], second.preprocessing_metadata["train_phenotype_sha256"])

    def test_tfpp_fold_manifest_is_reused_without_random_split(self):
        tfpp = self.fixture.trainer(use_kfold=True, split_file=None, n_folds=2, kfold_val_ratio=0.2)
        with redirect_stdout(io.StringIO()):
            fold = tfpp._prepare_fold(1, tfpp._kfold_splits()[0])
        self.addCleanup(self.fixture.close_logger, fold)
        path = Path(fold.config.preprocessing_save_dir) / "split.npz"
        module = importlib.import_module("trainer.ConvAE_Trainer")
        with (patch.object(module, "load_or_create_split", side_effect=AssertionError("不能重划分")),
              patch("dataloader.ConvAE_DataLoader.generate_split", side_effect=AssertionError("不能重划分"))):
            mae = self.trainer(split_file=str(path), seed=999, train_ratio=0.1, val_ratio=0.2, test_ratio=0.7)
        for key in ("train_idx", "val_idx", "test_idx"):
            np.testing.assert_array_equal(getattr(mae.split, key), getattr(fold.split, key))
        self.assertTrue(mae.config.split_tag.startswith("fold_01_"))
        self.assertEqual(mae.preprocessing_metadata["split_sha256"], file_sha256(path))
        self.assertEqual(mae.preprocessing_metadata["fold_metadata"]["fold_index"], 1)
        np.testing.assert_array_equal(mae.test_loader.dataset.sample_ids,
                                      fold.dataset.sample_ids[fold.split.test_idx])

    def test_in_memory_manifest_and_invalid_identity_or_indices(self):
        mae = self.trainer(indices=self.fixture.split, split_file=None)
        self.assertEqual(mae.preprocessing_metadata["split_source"], "in_memory")
        data = self.dataset(chunk_size=8, overlap=2)
        for kind in ("overlap", "hash", "order"):
            split = copy.deepcopy(self.fixture.split)
            if kind == "overlap":
                split.train_idx[0] = split.test_idx[0]
            elif kind == "hash":
                split.metadata["sample_ids_sha256"] = "bad"
            else:
                data.sample_ids = data.sample_ids[::-1]
            with patch.object(StandardScaler, "fit") as fit:
                with self.assertRaises(ValueError):
                    get_mae_dataloaders(data, split, num_workers=0)
                fit.assert_not_called()

        invalid_fold = copy.deepcopy(self.fixture.split)
        invalid_fold.metadata.update(fold_index=99, n_folds=2, kfold_val_ratio=0.2, validation_seed=73)
        with patch.object(StandardScaler, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "fold"):
                self.trainer(indices=invalid_fold, split_file=None)
            fit.assert_not_called()

    def test_missing_shared_manifest_is_not_silently_created(self):
        path = self.fixture.root / "missing_fold.npz"
        with self.assertRaisesRegex(FileNotFoundError, "禁止猜测 fold"):
            self.trainer(split_file=str(path))
        self.assertFalse(path.exists())

    def test_default_shared_path_and_ratios_match_tfpp(self):
        module = importlib.import_module("trainer.ConvAE_Trainer")
        path = self.fixture.root / "auto.npz"
        with patch.object(module, "default_split_path", return_value=path) as naming:
            mae = self.trainer(split_file="auto")
        expected = generate_split(40, seed=73)
        for key in ("train_idx", "val_idx", "test_idx"):
            np.testing.assert_array_equal(getattr(mae.split, key), getattr(expected, key))
        ratios = tuple(TFPPConfig.model_fields[key].default for key in ("train_ratio", "val_ratio", "test_ratio"))
        self.assertEqual(naming.call_args.kwargs["ratios"], ratios)

    def test_real_missing_and_structural_padding_stay_distinct_across_chunk_layouts(self):
        rng = np.random.default_rng(5)
        for length in (1, 12, 17):
            raw = rng.integers(0, 3, (40, length)).astype(float)
            raw[0, 0] = -1
            if length > 1:
                raw[0, 1] = np.nan
            np.save(self.fixture.root / "snp.npy", raw)
            for chunk, overlap in ((4, 0), (6, 2), (20, 0), (7, 6), (12, 0)):
                with self.subTest(length=length, chunk=chunk, overlap=overlap):
                    dataset = self.dataset(chunk_size=chunk, overlap=overlap)
                    sample = dataset.sample_with_metadata(0)
                    self.assertEqual(dataset.original_length, length)
                    np.testing.assert_array_equal(dataset.padding_mask, np.arange(dataset.target_length) >= length)
                    self.assertEqual(sample["snp"].shape, (dataset.num_chunks, chunk))
                    covered = []
                    for i, start in enumerate(dataset.chunk_starts.tolist()):
                        positions = np.arange(start, start + chunk)
                        valid = positions < length
                        np.testing.assert_array_equal(sample["valid_position_mask"][i], valid)
                        self.assertEqual(dataset.chunk_valid_lengths[i].item(), int(valid.sum()))
                        covered.extend(positions[valid])
                        observed = raw[0, positions[valid]]
                        expected_missing = (observed == -1) | np.isnan(observed)
                        np.testing.assert_array_equal(sample["missing_genotype_mask"][i, valid], expected_missing)
                    np.testing.assert_array_equal(np.unique(covered), np.arange(length))
                    self.assertFalse((sample["missing_genotype_mask"] & sample["padding_mask"]).any())
                    self.assertTrue((sample["snp"][sample["padding_mask"]] == 0).all())
                    self.assertFalse(dataset.padding_mask[0])
                    self.assertEqual(dataset.snp[0, 0], 0)

    def test_missing_phenotypes_are_rejected_in_every_role(self):
        for rows in (self.fixture.split.train_idx, self.fixture.split.val_idx, self.fixture.split.test_idx):
            changed = self.fixture.pheno.copy()
            changed[rows[0], 0, 0] = np.nan
            self.fixture.write_inputs(self.fixture.snp, changed)
            with patch.object(StandardScaler, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "phenotype_missing_strategy=error"):
                    self.trainer()
                fit.assert_not_called()

    def test_bad_chunk_parameters_and_time_columns_fail(self):
        for chunk, overlap in ((0, 0), (4, 4), (4, -1), (4, 5), (4.5, 1)):
            with self.assertRaises(ValueError):
                self.dataset(chunk_size=chunk, overlap=overlap)
        with self.assertRaisesRegex(ValueError, "time_coordinates"):
            self.trainer(time_coordinates=[1, 2])
        pd.DataFrame(self.fixture.pheno[:, 1], columns=["x", "y", "z"]).to_csv(self.fixture.root / "B.csv", index=False)
        with self.assertRaisesRegex(ValueError, "时间列"):
            self.dataset(chunk_size=8, overlap=2)

    def test_schema3_roundtrip_without_any_refit(self):
        trainer = self.trainer()
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("不能重新fit")):
            loaded = load_convae_preprocessing(trainer.config.output_dir, dataset=trainer.dataset, config=trainer.config)
        self.assertEqual(loaded.metadata["schema_version"], 3)
        self.assertEqual(loaded.metadata["original_length"], 12)
        np.testing.assert_array_equal(loaded.padding_mask, trainer.dataset.padding_mask)
        np.testing.assert_array_equal(loaded.chunk_padding_mask, trainer.dataset.chunk_padding_mask)
        self.assertEqual(loaded.metadata["split_sha256"], file_sha256(self.fixture.split_path))

    def test_missing_artifacts_and_tampered_contract_are_rejected(self):
        trainer = self.trainer()
        root = Path(trainer.config.output_dir)
        paths = ["preprocessing/split.npz", "preprocessing/preprocessing.json", "scalers/scaler.pkl",
                 "preprocessing/padding_mask.npy", "preprocessing/chunk_padding_mask.npy", "config.json"]
        for name in paths:
            path = root / name
            content = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                    load_convae_preprocessing(root)
            finally:
                path.write_bytes(content)
        path = root / "preprocessing/preprocessing.json"
        content = path.read_bytes()
        for key in ("genotype_encoding_version", "original_length", "time_points", "fold_metadata", "scaler_sha256"):
            meta = json.loads(content)
            meta.pop(key)
            path.write_text(json.dumps(meta))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_convae_preprocessing(root)
        for key, value in (("original_length", 13), ("chunk_size", 7), ("phenotype_names", ["B", "A"]),
                           ("genotype_encoding_version", "bad"), ("sample_ids_sha256", "bad"),
                           ("time_points", 4), ("fold_metadata", {"fold_index": 8})):
            meta = json.loads(content)
            meta[key] = value
            path.write_text(json.dumps(meta))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_convae_preprocessing(root)
        path.write_bytes(content)
        mask = root / "preprocessing/padding_mask.npy"
        values = np.load(mask)
        values[0] = True  # 即使更新哈希，真实缺失位点也不能被标为结构性 padding。
        np.save(mask, values)
        meta = json.loads(content)
        meta["padding_mask_sha256"] = file_sha256(mask)
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "padding mask"):
            load_convae_preprocessing(root)

    def test_split_tag_separates_experiments_and_incompatible_layout_cannot_overwrite(self):
        first = self.trainer(exp_name="same")
        second = self.trainer(exp_name="same", indices=generate_split(40, seed=74), split_file=None)
        self.assertNotEqual(first.config.output_dir, second.config.output_dir)
        before = Path(first.config.output_dir, "preprocessing/preprocessing.json").read_bytes()
        with self.assertRaises(ValueError):
            self.trainer(exp_name="same", chunk_size=10, overlap=2)
        self.assertEqual(before, Path(first.config.output_dir, "preprocessing/preprocessing.json").read_bytes())

    def test_legacy_two_loader_interface_still_fits_only_train_and_keeps_raw_dataset(self):
        dataset = self.dataset(chunk_size=8, overlap=2)
        train, val = get_mae_dataloader(dataset, num_workers=0)
        expected = generate_split(40, ratios=(0.8, 0.2, 0), seed=73)
        self.assertEqual(train.dataset.indices, expected.train_idx.tolist())
        self.assertEqual(val.dataset.indices, expected.val_idx.tolist())
        np.testing.assert_array_equal(dataset.phenotype, self.fixture.pheno)

    def test_prepare_only_does_not_construct_model_train_or_export_features(self):
        with (patch.object(ConvAETrainer, "_create_model") as model,
              patch.object(ConvAETrainer, "_compress_data") as export):
            trainer = self.trainer()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "仅准备数据管线"):
                trainer.train()
        model.assert_not_called()
        export.assert_not_called()
        self.assertFalse(list(Path(trainer.config.output_dir).rglob("*compressed*")))
        import train_convae
        with (patch.object(sys, "argv", ["train_convae.py", "--prepare_only", "--split_file", str(self.fixture.split_path)]),
              patch.object(train_convae, "ConvAETrainer") as factory, redirect_stdout(io.StringIO())):
            train_convae.train_one()
        self.assertFalse(factory.call_args.kwargs["initialize_model"])
        factory.return_value.train.assert_not_called()


if __name__ == "__main__":
    unittest.main()

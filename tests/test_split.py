"""个体划分及两种 DataLoader 的 CPU 合成数据回归测试。"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from config import ConvAEConfig, TFPPConfig
from dataloader.ConvAE_DataLoader import get_mae_dataloader
from dataloader.TFPP_DataLoader import get_dataloader, get_dataloader_with_kfold
from utils.Split import (
    default_split_path,
    generate_split,
    load_or_create_split,
    load_split,
    save_split,
    validate_split_indices,
)
from utils.Normalization import denormalize_phenotype


class ToyDataset(Dataset):
    phenotype_names = ["toy"]

    def __init__(self, n_samples=40):
        self.snp = np.arange(n_samples, dtype=np.float32).reshape(-1, 1)
        # 同一个体的 3 个时间点编码为 row_id * 10 + time_id。
        self.phenotype = self.snp[:, None, :] * 10 + np.arange(3, dtype=np.float32)

    def __len__(self):
        return len(self.snp)

    def __getitem__(self, idx):
        return torch.from_numpy(self.snp[idx]), torch.from_numpy(self.phenotype[idx])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "shared.npz"

    def assert_same_indices(self, first, second):
        for name in ("train_idx", "val_idx", "test_idx"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_reproducible_and_matches_existing_random_split(self):
        for count in (40, 43):
            with self.subTest(count=count):
                first = generate_split(count, seed=73)
                self.assert_same_indices(first, generate_split(count, seed=73))
                original = random_split(
                    range(count), [0.8, 0.1, 0.1],
                    generator=torch.Generator().manual_seed(73),
                )
                for actual, expected in zip(
                    (first.train_idx, first.val_idx, first.test_idx), original
                ):
                    np.testing.assert_array_equal(actual, expected.indices)
                self.assertFalse(np.array_equal(first.train_idx, generate_split(count, seed=74).train_idx))

    def test_disjoint_and_complete_coverage(self):
        split = generate_split(43)
        sets = [set(split.train_idx), set(split.val_idx), set(split.test_idx)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(43)))
        self.assertEqual(sum(map(len, sets)), 43)

    def test_round_trip_including_metadata(self):
        split = generate_split(sample_ids=[f"maize-{i}" for i in range(40)])
        save_split(split, self.path)
        restored = load_split(self.path, sample_ids=split.metadata["sample_ids"])
        self.assert_same_indices(split, restored)
        self.assertEqual(split.metadata, restored.metadata)
        self.assertEqual(restored.metadata["sample_id_source"], "provided")

    def test_row_identity_is_explicit(self):
        split = generate_split(40)
        self.assertEqual(split.metadata["sample_id_source"], "row_index")
        self.assertEqual(split.metadata["sample_ids"], list(range(40)))
        self.assertEqual(len(split.metadata["sample_ids_sha256"]), 64)

    def test_reject_sample_count_mismatch_without_overwrite(self):
        load_or_create_split(self.path, 40)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "样本数"):
            load_or_create_split(self.path, 41)
        self.assertEqual(before, self.path.read_bytes())

    def test_reject_changed_or_reordered_ids(self):
        ids = [f"id-{i}" for i in range(40)]
        save_split(generate_split(sample_ids=ids), self.path)
        for current in (ids[::-1], ids[:-1] + ["other"]):
            with self.subTest(current=current[-1]):
                with self.assertRaisesRegex(ValueError, "哈希"):
                    load_or_create_split(self.path, sample_ids=current)
        # 只有行数不能证明显式个体 ID 的身份和顺序。
        with self.assertRaises(ValueError):
            load_split(self.path, n_samples=40)

    def test_invalid_indices_are_rejected(self):
        invalid = [
            ([0, 0], [1], [2]),  # 集合内重复
            ([0], [0], [1, 2]),  # 集合间交叉
            ([0], [1], []),  # 缺失个体
            ([-1], [1], [2]),  # 负索引
            ([0], [1], [3]),  # 越界
            ([0.0], [1], [2]),  # 浮点索引
            ([[0]], [1], [2]),  # 非一维
        ]
        for arrays in invalid:
            with self.subTest(arrays=arrays):
                with self.assertRaises(ValueError):
                    validate_split_indices(*arrays, n_samples=3)

    def test_invalid_generation_inputs_are_rejected(self):
        for kwargs in (
            {"n_samples": 0}, {"n_samples": 2.5},
            {"sample_ids": ["a", "a"]},
            {"n_samples": 3, "sample_ids": ["a", "b"]},
            {"n_samples": 40, "ratios": (0.8, 0.2, 0.1)},
            {"n_samples": 40, "ratios": (1.0, -0.1, 0.1)},
            {"n_samples": 40, "ratios": (np.nan, 0.1, 0.1)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    generate_split(**kwargs)

    def test_existing_split_is_authoritative_and_not_overwritten(self):
        first = load_or_create_split(self.path, 40, ratios=(0.8, 0.1, 0.1), seed=73)
        before = self.path.read_bytes()
        # ConvAE 的默认比例不同，但显式指定已有文件时必须复用该文件。
        second = load_or_create_split(self.path, 40, ratios=(0.8, 0.2, 0.0), seed=9)
        self.assert_same_indices(first, second)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(before, self.path.read_bytes())
        with self.assertRaises(FileExistsError):
            save_split(first, self.path)

    def test_default_paths_and_auto_creation(self):
        base = default_split_path(40)
        self.assertNotEqual(base, default_split_path(40, seed=74))
        self.assertNotEqual(base, default_split_path(40, ratios=(0.8, 0.2, 0)))
        ids = [f"id-{i}" for i in range(40)]
        self.assertNotEqual(default_split_path(sample_ids=ids), default_split_path(sample_ids=ids[::-1]))
        with patch("utils.Split.default_split_path", return_value=self.path):
            created = load_or_create_split("auto", 40)
            self.assert_same_indices(created, load_or_create_split("auto", 40))

    def test_corrupt_metadata_and_indices_are_rejected_on_load(self):
        split = generate_split(40)
        for field in ("hash", "indices"):
            with self.subTest(field=field):
                metadata = dict(split.metadata)
                train_idx = split.train_idx.copy()
                if field == "hash":
                    metadata["sample_ids_sha256"] = "invalid"
                else:
                    train_idx[0] = split.val_idx[0]
                np.savez(self.path, train_idx=train_idx, val_idx=split.val_idx,
                         test_idx=split.test_idx, metadata=np.asarray(json.dumps(metadata)))
                with self.assertRaises(ValueError):
                    load_split(self.path)

    def test_shared_loaders_keep_all_timepoints_and_hold_out_test(self):
        dataset = ToyDataset()
        split = generate_split(len(dataset))
        tfpp = get_dataloader(
            dataset, 4, "global", "zscore", num_workers=0, indices=split,
            train_ratio=0.2, val_ratio=0.4, test_ratio=0.4, seed=9,
        )
        mae = get_mae_dataloader(
            dataset, batch_size=4, num_workers=0, indices=split,
            train_ratio=0.5, val_ratio=0.5, seed=9,
        )
        for loader, expected in zip(tfpp[:3], (split.train_idx, split.val_idx, split.test_idx)):
            np.testing.assert_array_equal(loader.dataset.tensors[0][:, 0], expected)
            self.assertEqual(loader.dataset.tensors[1].shape, (len(expected), 1, 3))
            restored = denormalize_phenotype(loader.dataset.tensors[1], tfpp[3]).numpy()
            np.testing.assert_allclose(restored, dataset.phenotype[expected], atol=2e-5)
        self.assertEqual(mae[0].dataset.indices, split.train_idx.tolist())
        self.assertEqual(mae[1].dataset.indices, split.val_idx.tolist())
        self.assertFalse(set(mae[0].dataset.indices + mae[1].dataset.indices) & set(split.test_idx))

    def test_loaders_without_explicit_split_keep_legacy_behavior(self):
        dataset = ToyDataset()
        split = generate_split(40)
        loaders = get_dataloader(dataset, 4, "global", "zscore", num_workers=0)
        for loader, expected in zip(loaders[:3], (split.train_idx, split.val_idx, split.test_idx)):
            np.testing.assert_array_equal(loader.dataset.tensors[0][:, 0], expected)
        mae = get_mae_dataloader(dataset, num_workers=0)
        original = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(73))
        self.assertEqual(mae[0].dataset.indices, original[0].indices)
        self.assertEqual(mae[1].dataset.indices, original[1].indices)

    def test_kfold_loader_validates_explicit_indices(self):
        with self.assertRaises(ValueError):
            get_dataloader_with_kfold([0, 1], [1], list(range(2, 40)), ToyDataset(),
                                     4, "cpu", "global", "zscore", num_workers=0)
        with self.assertRaises(ValueError):
            TFPPConfig(use_kfold=True, split_file=str(self.path))

    def test_training_cli_forwards_split_file(self):
        import train

        with patch.object(sys, "argv", ["train.py", "--split_file", str(self.path)]):
            with patch.object(train, "TFPPTrainer") as trainer:
                train.main()
        self.assertEqual(trainer.call_args.kwargs["config"].split_file, str(self.path))
        trainer.return_value.train.assert_called_once()

    def test_both_trainers_read_the_shared_file(self):
        from trainer import ConvAETrainer, TFPPTrainer

        shared = load_or_create_split(self.path, 40)
        dataset = ToyDataset()
        for kind, trainer_class, loader_name in (
            ("TFPP", TFPPTrainer, "get_dataloader"),
            ("ConvAE", ConvAETrainer, "get_mae_dataloader"),
        ):
            with self.subTest(kind=kind):
                module = importlib.import_module(f"trainer.{kind}_Trainer")
                obj = trainer_class.__new__(trainer_class)
                obj.config = (TFPPConfig if kind == "TFPP" else ConvAEConfig)(split_file=str(self.path), device="cpu")
                obj.dataset, obj.num_sample = dataset, len(dataset)
                obj.train_ratio, obj.val_ratio, obj.test_ratio = 0.8, 0.2, 0.0
                obj.seed, obj.batch_size, obj.num_workers, obj.device = 9, 4, 0, "cpu"
                obj.norm_method, obj.scaler_type = "global", "zscore"
                obj.scaler_file, obj.logger = "unused", Mock()
                obj._log_configuration = Mock()
                obj._create_optimizer_and_scheduler = Mock(side_effect=RuntimeError("stop-before-training"))
                loader = getattr(module, loader_name)
                with patch.object(module, loader_name, wraps=loader) as wrapped:
                    with patch.object(module, "save_scalers", create=True):
                        with self.assertRaisesRegex(RuntimeError, "stop-before-training"):
                            (obj._train_regular if kind == "TFPP" else obj.train)()
                self.assert_same_indices(shared, wrapped.call_args.kwargs["indices"])


if __name__ == "__main__":
    unittest.main()

"""固定 Ridge 独立线性探针的泄漏边界、指标与 CPU CLI 冒烟。"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import test_convae_model as model_tests
from utils.Calculate_Metrics import compute_metrics_vectorized
from utils.ConvAE_Export import export_convae_features, load_exported_features
from utils.ConvAE_Ridge import (
    ALPHAS, evaluate_test_once, fit_probe, load_probe_export, load_ridge_probe, load_role_targets,
    metric_hierarchy, predict_probe, run_ridge_baseline, save_probe_targets,
)
from utils.Inference import file_sha256
from utils.Split import generate_split, save_split


class ConvAERidgeTests(unittest.TestCase):
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
        self.export = export_convae_features(self.trainer.config.output_dir,
            split_file=self.fixture.split_path, output_root=self.root / "exports", batch_size=4)
        self.arrays, self.meta = load_exported_features(self.export)
        self.targets = {role: (self.fixture.pheno[getattr(self.fixture.split, role + "_idx")].copy(),
                               self.arrays[role][1].copy()) for role in ("train", "val", "test")}
        self.target_file = self.root / "targets.npz"
        self.write_targets(self.target_file)
        self.counter = 0

    def write_targets(self, path, targets=None):
        save_probe_targets(path, self.targets if targets is None else targets,
                           phenotype_names=["A", "B"], time_labels=["0", "1", "2"])

    def run_probe(self, **kwargs):
        self.counter += 1
        defaults = dict(output_root=self.root / f"ridge_{self.counter}")
        defaults.update(kwargs)
        return run_ridge_baseline(self.export, self.target_file, **defaults)

    def test_fixed_candidates_train_only_fits_lazy_test_and_no_refit(self):
        events, ridge_rows, scaler_values = [], [], []
        original_fit = Ridge.fit
        original_scaler = StandardScaler.fit

        def fit(model, x, y, *args, **kwargs):
            events.append("fit")
            ridge_rows.append((x.shape, y.shape, model.get_params()))
            return original_fit(model, x, y, *args, **kwargs)

        def scaler_fit(scaler, x, *args, **kwargs):
            scaler_values.append(np.asarray(x).copy())
            return original_scaler(scaler, x, *args, **kwargs)

        def read(path, role, ids, meta):
            if role == "test":
                self.assertEqual(events.count("fit"), 9)
            events.append("labels_" + role)
            return load_role_targets(path, role, ids, meta)

        with (patch.object(Ridge, "fit", new=fit), patch.object(StandardScaler, "fit", new=scaler_fit),
              patch("utils.ConvAE_Ridge.load_role_targets", side_effect=read)):
            output = self.run_probe()
        self.assertEqual([v[0][0] for v in ridge_rows], [24] * 9)
        self.assertEqual([v[1] for v in ridge_rows], [(24, 6)] * 9)
        self.assertEqual([v[2]["alpha"] for v in ridge_rows], list(ALPHAS))
        for _, _, config in ridge_rows:
            self.assertEqual((config["solver"], config["tol"], config["max_iter"], config["fit_intercept"]),
                             ("lsqr", 1e-6, 10000, True))
        self.assertEqual(len(scaler_values), 3)
        np.testing.assert_array_equal(scaler_values[0], self.arrays["train"][0])
        for p in range(2):
            np.testing.assert_array_equal(scaler_values[p + 1], self.targets["train"][0][:, p].reshape(-1, 1))
        self.assertEqual(events[-1], "labels_test")
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["test_evaluation_count"], 1)

    def test_changed_test_labels_only_change_metrics_not_alpha_scalers_or_predictions(self):
        first = self.run_probe()
        changed = {role: (values.copy(), ids.copy()) for role, (values, ids) in self.targets.items()}
        changed["test"][0][:] += 500
        alternate = self.root / "changed_test.npz"
        self.write_targets(alternate, changed)
        second = run_ridge_baseline(self.export, alternate, output_root=self.root / "changed")
        old, new = load_ridge_probe(first), load_ridge_probe(second)
        for name in ("ridge_model.pkl", "feature_scaler.pkl", "phenotype_scaler.pkl", "alpha_search.json"):
            self.assertEqual(file_sha256(first / name), file_sha256(second / name))
        self.assertEqual(old["alpha"], new["alpha"])
        np.testing.assert_array_equal(old["model"].coef_, new["model"].coef_)
        np.testing.assert_array_equal(old["model"].intercept_, new["model"].intercept_)
        for key in ("mean_", "scale_"):
            np.testing.assert_array_equal(getattr(old["feature_scaler"], key), getattr(new["feature_scaler"], key))
            for name in ("A", "B"):
                np.testing.assert_array_equal(getattr(old["phenotype_scaler"]["pheno_meta"]["scalers"][name], key),
                                              getattr(new["phenotype_scaler"]["pheno_meta"]["scalers"][name], key))
        np.testing.assert_array_equal(predict_probe(old, self.arrays["test"][0])[1],
                                      predict_probe(new, self.arrays["test"][0])[1])
        self.assertNotEqual(json.loads((first / "metrics.json").read_text())["denormalized"]["overall"]["mse"],
                            json.loads((second / "metrics.json").read_text())["denormalized"]["overall"]["mse"])

    def test_val_can_change_alpha_but_not_train_scalers_and_ties_choose_larger(self):
        x = np.arange(20, dtype=np.float64).reshape(-1, 1)
        y = (2 * x + 1).reshape(20, 1, 1)
        val = np.array([[30.], [35.], [40.]])
        good, _ = fit_probe(x, y, val, (2 * val + 1).reshape(3, 1, 1), ["P"])
        bad, _ = fit_probe(x, y, val, np.full((3, 1, 1), y.mean()), ["P"])
        self.assertNotEqual(good["alpha"], bad["alpha"])
        np.testing.assert_array_equal(good["feature_scaler"].mean_, bad["feature_scaler"].mean_)
        np.testing.assert_array_equal(good["phenotype_scaler"]["pheno_meta"]["scalers"]["P"].mean_,
                                      bad["phenotype_scaler"]["pheno_meta"]["scalers"]["P"].mean_)
        tie, records = fit_probe(np.zeros((20, 2)), y, np.zeros((3, 2)), np.ones((3, 1, 1)), ["P"])
        self.assertEqual(tie["alpha"], 10000)
        self.assertEqual(sum(v["selected"] for v in records), 1)

    def test_alignment_duplicate_missing_extra_wrong_order_fail(self):
        for kind in ("duplicate", "missing", "extra", "order"):
            changed = {role: (values.copy(), ids.copy()) for role, (values, ids) in self.targets.items()}
            values, ids = changed["train"]
            if kind == "duplicate":
                ids[1] = ids[0]
            elif kind == "missing":
                changed["train"] = (values[:-1], ids[:-1])
            elif kind == "extra":
                changed["train"] = (np.concatenate([values, values[:1]]), np.append(ids, 999))
            else:
                changed["train"] = (values[::-1], ids[::-1])
            path = self.root / f"bad_{kind}.npz"
            self.write_targets(path, changed)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "sample_id"):
                run_ridge_baseline(self.export, path, output_root=self.root / kind)

    def test_export_array_metadata_and_split_hashes_are_strict(self):
        file = self.export / "features/train.npy"
        content = file.read_bytes()
        file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "哈希"):
            self.run_probe()
        file.write_bytes(content)
        meta_path = self.export / "metadata.json"
        content = meta_path.read_bytes()
        meta = json.loads(content)
        # 即使 ID 文件和 metadata 同时变动，也必须匹配独立 manifest。
        ids_path = self.export / "features/train_sample_ids.npy"
        ids_bytes = ids_path.read_bytes()
        ids = np.load(ids_path)[::-1]
        np.save(ids_path, ids)
        meta["arrays"]["train"]["sample_ids"] = ids.tolist()
        meta["arrays"]["train"]["files"][ids_path.name] = file_sha256(ids_path)
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.run_probe()
        ids_path.write_bytes(ids_bytes)
        meta_path.write_bytes(content)
        alternate = self.root / "other_split.npz"
        save_split(generate_split(40, seed=74), alternate)
        with self.assertRaisesRegex(ValueError, "split 文件哈希"):
            self.run_probe(split_file=alternate)

    def test_repeat_saved_model_replay_and_test_guard(self):
        first, second = self.run_probe(), self.run_probe()
        a, b = load_ridge_probe(first), load_ridge_probe(second)
        self.assertEqual(a["alpha"], b["alpha"])
        np.testing.assert_array_equal(a["model"].coef_, b["model"].coef_)
        norm, raw = predict_probe(a, self.arrays["test"][0])
        with np.load(first / "test_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["y_pred"], raw)
            np.testing.assert_array_equal(saved["y_pred_normalized"], norm)
            np.testing.assert_array_equal(saved["sample_ids"], self.arrays["test"][1])
        with self.assertRaisesRegex(ValueError, "test 已评估"):
            evaluate_test_once(a, self.arrays["test"][0], self.targets["test"][0], self.meta)
        self.assertGreater(json.loads((first / "summary.json").read_text())["peak_memory_bytes"], 0)
        (first / "ridge_model.pkl").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "哈希"):
            load_ridge_probe(first)

    def test_metrics_match_tfpp_hierarchy_negative_r2_nan_pcc(self):
        target = np.arange(1, 19, dtype=np.float64).reshape(3, 2, 3)
        prediction = target + 100
        metrics = metric_hierarchy(prediction, target, ["A", "B"], ["0", "1", "2"])
        self.assertEqual(len(metrics["phenotype_timepoints"]), 2)
        self.assertEqual(len(metrics["timepoints"]), 3)
        for p, name in enumerate(("A", "B")):
            existing = compute_metrics_vectorized(prediction[:, p], target[:, p])
            for key in ("pcc", "r2", "mse", "rmse", "mae"):
                self.assertAlmostEqual(metrics["phenotypes"][name][key], existing[key])
        self.assertLess(metrics["overall"]["r2"], 0)
        constant = metric_hierarchy(np.ones_like(target), target, ["A", "B"], ["0", "1", "2"])
        self.assertTrue(np.isnan(constant["overall"]["pcc"]))
        self.assertEqual(constant["overall"]["pcc_valid_count"], 0)
        self.assertEqual(constant["overall"]["pcc_total_count"], 6)
        self.assertTrue(np.isnan(constant["phenotype_timepoints"]["A"]["0"]["pcc"]))
        with self.assertRaises(ValueError):
            metric_hierarchy(prediction[..., :2], target, ["A", "B"], ["0", "1", "2"])

    def test_inverse_scaling_prediction_rows_and_search_records(self):
        output = self.run_probe()
        package = load_ridge_probe(output)
        norm, raw = predict_probe(package, self.arrays["test"][0])
        for p, name in enumerate(("A", "B")):
            scaler = package["phenotype_scaler"]["pheno_meta"]["scalers"][name]
            self.assertEqual(scaler.n_samples_seen_, 24 * 3)
            np.testing.assert_allclose(raw[:, p], norm[:, p] * scaler.scale_[0] + scaler.mean_[0], rtol=1e-6)
        frame = pd.read_csv(output / "test_predictions.csv")
        self.assertEqual(frame.shape[0], 8 * 2 * 3)
        np.testing.assert_array_equal(frame.sample_id.values.reshape(8, 6)[:, 0], self.arrays["test"][1])
        self.assertEqual(frame.phenotype[:6].tolist(), ["A"] * 3 + ["B"] * 3)
        self.assertTrue((frame["split"] == "test").all())
        search = json.loads((output / "alpha_search.json").read_text())
        self.assertEqual([v["alpha"] for v in search], list(ALPHAS))
        self.assertTrue(all("test" not in key for v in search for key in v))

    def test_export_and_config_hash_directories_and_no_overwrite(self):
        base = self.root / "results"
        first = self.run_probe(output_root=base)
        with self.assertRaises(FileExistsError):
            self.run_probe(output_root=base)
        second = self.run_probe(output_root=base, seed=74)
        self.assertNotEqual(first, second)
        split = generate_split(40, ratios=(0.6, 0.2, 0.2), seed=80)
        split.metadata.update(fold_index=1, n_folds=2, kfold_val_ratio=0.2, validation_seed=80)
        split_file = self.root / "fold.npz"
        save_split(split, split_file)
        trainer = self.helper.trainer(split_file=str(split_file))
        trainer.train()
        exported = export_convae_features(trainer.config.output_dir, split_file=split_file, output_root=self.root / "fold_export")
        arrays, _ = load_exported_features(exported)
        labels = self.root / "fold_labels.npz"
        self.write_targets(labels, {role: (self.fixture.pheno[getattr(split, role + "_idx")], arrays[role][1]) for role in self.targets})
        third = run_ridge_baseline(exported, labels, output_root=base)
        self.assertNotEqual(first.parent, third.parent)
        self.assertEqual(json.loads((third / "summary.json").read_text())["fold_metadata"]["fold_index"], 1)

    def test_cpu_cli_with_step7_synthetic_export(self):
        process = subprocess.run([sys.executable, "-B", "ridge_convae.py", "--export_dir", str(self.export),
            "--phenotype_file", str(self.target_file), "--output_root", str(self.root / "cli")],
            check=True, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertIn("预测复现校验", process.stdout)
        summary_files = list((self.root / "cli").glob("*/*/summary.json"))
        self.assertEqual(len(summary_files), 1)
        summary = json.loads(summary_files[0].read_text())
        self.assertIn(summary["alpha"], ALPHAS)
        self.assertEqual(summary["test_evaluation_count"], 1)


if __name__ == "__main__":
    unittest.main()

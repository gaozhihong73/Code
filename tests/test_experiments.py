"""实验计划、共享 split、子集、一次性 test、严格追溯和六路线 CPU smoke。"""

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from experiments.runner import RecordedTFPPTrainer, run_unit, select_snps, summarize, validate_result
from experiments.smoke import create_smoke_spec
from experiments.spec import ExperimentSpec, TrainSize, digest, file_digest, nested_subset, plan, random_indices
from utils.Normalization import load_scalers
from utils.Split import load_split, save_split, split_from_indices, validate_individual_split


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = create_smoke_spec(self.root / "fixture", include_ridge=False)
        self.split = load_split(self.spec.splits.manifests[0].path)

    def items(self, spec=None):
        return plan(self.spec if spec is None else spec, development_smoke=True)

    def execute(self, item, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run_unit(item, **kwargs)

    def test_dry_plan_shared_split_no_large_arrays_or_writes(self):
        original = np.load
        forbidden = self.spec.data.snp.path
        def small_only(path, *args, **kwargs):
            if str(path) == forbidden or "features" in str(path):
                raise AssertionError("dry-run 加载了大数组")
            return original(path, *args, **kwargs)
        before = set(self.root.rglob("*"))
        with patch("numpy.load", side_effect=small_only):
            items = self.items()
        self.assertEqual(len(items), 5)
        self.assertEqual(before, set(self.root.rglob("*")))
        self.assertEqual(len({v["resolved"]["outer_split"]["sha256"] for v in items}), 1)
        for role in ("train_idx", "val_idx", "test_idx"):
            self.assertEqual(len({tuple(v["resolved"]["subset"][role]) for v in items}), 1)

    def test_cpu_cli_dry_run_writes_no_results(self):
        spec_path = Path(self.spec.data.snp.path).parent / "spec.json"
        before = set(self.root.rglob("*"))
        process = subprocess.run([sys.executable, "-B", "run_experiments.py", "dry-run", str(spec_path),
                                  "--development-smoke"], check=True, text=True, capture_output=True)
        payload = json.loads(process.stdout)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(len(payload["experiments"]), 5)
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_hash_stability_axes_and_unsupported_routes_ablation_fail(self):
        self.assertEqual([p["config_hash"] for p in self.items()], [p["config_hash"] for p in self.items()])
        raw = self.spec.model_dump()
        raw["experiments"] = [raw["experiments"][1]]
        for field, invalid in (("route", "ConvAE->continuous_feature_TFPP"), ("ablation", "fake_gate")):
            bad = copy.deepcopy(raw)
            bad["experiments"][0][field] = invalid
            with self.assertRaises(ValueError):
                ExperimentSpec.model_validate(bad)
        variants = []
        for seed in (73, 74):
            for ablation in ("full", "wide_only", "deep_only", "no_gate_add"):
                unit = {**raw["experiments"][0], "seed": seed, "ablation": ablation}
                variants.append(unit)
        raw["experiments"] = variants
        items = self.items(ExperimentSpec.model_validate(raw))
        self.assertEqual(len({v["output_dir"] for v in items}), len(variants))

    def test_nested_subsets_and_exclusion_validation(self):
        small = nested_subset(self.split, TrainSize(count=8), 73)
        medium = nested_subset(self.split, TrainSize(fraction=0.5), 73)
        full = nested_subset(self.split, TrainSize(fraction=1), 73)
        self.assertTrue(set(small.train_idx) < set(medium.train_idx) < set(full.train_idx))
        for view in (small, medium, full):
            validate_individual_split(view)
            np.testing.assert_array_equal(view.val_idx, self.split.val_idx)
            np.testing.assert_array_equal(view.test_idx, self.split.test_idx)
            self.assertFalse(set(view.metadata["excluded_train_idx"]) & set(view.train_idx))
        broken = copy.deepcopy(small)
        broken.metadata["parent_train_idx"] = [0]
        with self.assertRaises(ValueError):
            validate_individual_split(broken)
        path = self.root / "subset.npz"
        save_split(small, path)
        np.testing.assert_array_equal(load_split(path).train_idx, small.train_idx)

    def test_random_k_is_label_free_and_fixed_order_hash_is_preserved(self):
        unit = self.spec.experiments[1].model_dump()
        x = np.zeros((8, 13))
        a, _ = select_snps(unit, x, None, [])
        b, _ = select_snps(unit, x + 2, np.full((8, 1, 3), 999), ["unused"])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, random_indices(13, 3, 73))
        item = self.items()[0]
        result = self.execute(item)
        manifest, _ = validate_result(result)
        indices = np.load(result / manifest["feature_artifact"])
        np.testing.assert_array_equal(indices, [5, 1, 7])
        self.assertEqual(manifest["actual_input_dim"], 3)
        self.assertEqual(manifest["test_evaluation_count"], 1)
        with self.assertRaises(FileExistsError):
            self.execute(item)
        self.assertEqual(self.execute(item, resume=True), result)

    def test_subset_preprocessing_refits_and_add_gate_is_tested(self):
        raw = self.spec.model_dump()
        template = raw["experiments"][3]
        raw["experiments"] = [{**template, "train_size": {"count": n}, "ablation": "no_gate_add"} for n in (12, 20)]
        results = [self.execute(item) for item in self.items(ExperimentSpec.model_validate(raw))]
        metadata, indices = [], []
        for root in results:
            manifest, _ = validate_result(root)
            work = root / "attempt_001"
            prep = json.loads((work / "backend/preprocessing/preprocessing.json").read_text())
            model_config = json.loads((work / "backend/models/model_config.json").read_text())
            self.assertEqual(model_config["ablation_mode"], "add")
            scalers = load_scalers(work / "backend/scalers/scaler.pkl")
            self.assertEqual(scalers["pheno_meta"]["scalers"]["A"].n_samples_seen_, len(manifest["sample_ids"]["train"]) * 3)
            metadata.append(prep)
            indices.append(set(manifest["sample_ids"]["train"]))
        self.assertTrue(indices[0] < indices[1])
        self.assertNotEqual(metadata[0]["train_indices_sha256"], metadata[1]["train_indices_sha256"])
        self.assertNotEqual(metadata[0]["wide_imputer"]["train_idx"], metadata[1]["wide_imputer"]["train_idx"])

    def test_joint_union_contributions_and_explicit_cascade(self):
        raw = self.spec.model_dump()
        template = raw["experiments"][4]
        raw["experiments"] = [{**template, "screening_mode": mode} for mode in ("joint", "per_trait_union")]
        items = self.items(ExperimentSpec.model_validate(raw))
        self.assertNotEqual(items[0]["resolved"]["method_id"], items[1]["resolved"]["method_id"])
        for item in items:
            root = self.execute(item)
            selected = json.loads((root / "attempt_001/selection.json").read_text())
            first, second = selected["stages"]
            self.assertEqual(first["algorithm"], "GWAS")
            self.assertEqual(second["algorithm"], "DC-SIS")
            self.assertEqual(second["input_dim"], first["post_dedup_count"])
            self.assertTrue(set(second["ordered_original_indices"]).issubset(first["ordered_original_indices"]))
            if item["resolved"]["unit"]["screening_mode"] == "per_trait_union":
                self.assertEqual(set(second["per_trait"]), {"A", "B"})
                self.assertEqual(second["pre_dedup_count"], sum(v["count"] for v in second["per_trait"].values()))

    def test_test_label_changes_do_not_change_training_artifacts(self):
        first = self.execute(self.items()[3])
        raw = self.spec.model_dump()
        for trait, ref in raw["data"]["phenotypes"].items():
            frame = pd.read_csv(ref["path"])
            frame.iloc[self.split.test_idx] += 1000
            frame.to_csv(ref["path"], index=False)
            ref["sha256"] = file_digest(ref["path"])
        changed_spec = ExperimentSpec.model_validate(raw)
        second = self.execute(self.items(changed_spec)[3])
        self.assertNotEqual(first.name, second.name)  # 完整数据 provenance 改变目录，但不影响训练期产物。
        for name in ("selection.json", "screened_indices.npy", "backend/scalers/scaler.pkl", "backend/models/best_model.pt"):
            self.assertEqual(file_digest(first / "attempt_001" / name), file_digest(second / "attempt_001" / name))
        m1 = json.loads((first / "attempt_001/backend/preprocessing/preprocessing.json").read_text())
        m2 = json.loads((second / "attempt_001/backend/preprocessing/preprocessing.json").read_text())
        for key in ("training_genotype_sha256", "training_phenotype_sha256", "train_indices_sha256", "screening_cache_key", "wide_imputer"):
            # 路径是缓存溯源的一部分；筛选由 runner 执行，不复用训练器禁用筛选的缓存。
            if key != "screening_cache_key":
                self.assertEqual(m1[key], m2[key])

    def test_failure_preserved_resume_only_before_test_and_mismatch_rejected(self):
        item = self.items()[1]
        with patch.object(RecordedTFPPTrainer, "_train_epoch", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.execute(item)
        root = Path(item["output_dir"])
        status = json.loads((root / "status.json").read_text())
        self.assertEqual((status["state"], status["test_evaluation_count"]), ("failed", 0))
        self.assertTrue((root / "attempt_001/error.log").is_file())
        wrong = copy.deepcopy(item)
        wrong["resolved"]["unit"]["seed"] = 999
        with self.assertRaises(ValueError):
            self.execute(wrong, resume=True)
        self.execute(item, resume=True)
        self.assertTrue((root / "attempt_001/error.log").is_file())
        self.assertEqual(json.loads((root / "status.json").read_text())["attempt"], 2)
        other = self.items()[2]
        def fail_after_reservation(trainer, *args, **kwargs):
            trainer.reserve_test()
            raise RuntimeError("after test reservation")
        with patch.object(RecordedTFPPTrainer, "_evaluate_test_set", new=fail_after_reservation):
            with self.assertRaises(RuntimeError):
                self.execute(other)
        with self.assertRaisesRegex(ValueError, "test 已"):
            self.execute(other, resume=True)

    def test_incomplete_failed_or_missing_hash_results_cannot_enter_summary(self):
        root = self.execute(self.items()[1])
        self.assertEqual(len(summarize([root])["groups"]), 1)
        hashes = root / "artifacts.json"
        content = hashes.read_text()
        changed = json.loads(content)
        manifest = json.loads((root / "manifest.json").read_text())
        changed.pop(manifest["feature_artifact"])
        hashes.write_text(json.dumps(changed))
        summary = summarize([root])
        self.assertFalse(summary["groups"])
        self.assertEqual(len(summary["rejected"]), 1)
        hashes.write_text(content)
        status_path = root / "status.json"
        status = json.loads(status_path.read_text())
        status["state"] = "failed"
        status_path.write_text(json.dumps(status))
        self.assertFalse(summarize([root])["groups"])

    def test_kfold_complete_statistics_and_regular_never_mix(self):
        refs = []
        for fold, (outer, test) in enumerate(KFold(2, shuffle=True, random_state=73).split(np.arange(40)), 1):
            split = split_from_indices(outer[4:], outer[:4], test, 40)
            split.metadata.update(fold_index=fold, n_folds=2, kfold_val_ratio=0.2, validation_seed=73)
            path = self.root / f"fold_{fold}.npz"
            save_split(split, path)
            refs.append({"path": str(path), "sha256": file_digest(path)})
        raw = self.spec.model_dump()
        raw["splits"] = {"mode": "kfold", "manifests": refs}
        raw["experiments"] = [raw["experiments"][1]]
        folded = self.items(ExperimentSpec.model_validate(raw))
        roots = [self.execute(item) for item in folded]
        partial = summarize(roots[:1])["groups"][0]
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["statistics"])
        regular = self.execute(self.items()[1])
        combined = summarize([*roots, regular])
        self.assertEqual(len(combined["groups"]), 2)
        cv = next(g for g in combined["groups"] if g["axes"]["split_mode"] == "kfold")
        self.assertTrue(cv["complete"])
        self.assertEqual(cv["statistics"]["mse"]["valid_count"], 2)
        self.assertIn("phenotype_timepoints", cv["metric_statistics"]["denormalized"])

    def test_bad_fixed_indices_data_and_resume_code_changes_rejected(self):
        raw = self.spec.model_dump()
        ref = raw["experiments"][0]["fixed_indices"]
        original = Path(ref["path"]).read_bytes()
        for values in (np.array([1, 1]), np.array([13]), np.array([1.5])):
            np.save(ref["path"], values)
            ref["sha256"] = file_digest(ref["path"])
            with self.assertRaises(ValueError):
                self.items(ExperimentSpec.model_validate(raw))
        Path(ref["path"]).write_bytes(original)
        item = self.items()[1]
        root = self.execute(item)
        changed = copy.deepcopy(item["resolved"]["code"])
        changed["commit"] = "f" * 40
        with patch("experiments.runner.code_identity", return_value=changed):
            with self.assertRaisesRegex(ValueError, "代码"):
                self.execute(item, resume=True)
        path = Path(self.spec.data.snp.path)
        data = path.read_bytes()
        path.write_bytes(data + b"changed")
        with self.assertRaisesRegex(ValueError, "指纹"):
            self.execute(item, resume=True)
        path.write_bytes(data)
        self.assertEqual(json.loads((root / "status.json").read_text())["state"], "success")

    def test_different_k_and_optimization_protocols_are_not_fold_pooled(self):
        raw = self.spec.model_dump()
        template = raw["experiments"][1]
        raw["experiments"] = [{**template, "k": 2}, {**template, "k": 4},
            {**template, "tfpp": {**template["tfpp"], "lr": 0.001}}]
        roots = [self.execute(item) for item in self.items(ExperimentSpec.model_validate(raw))]
        self.assertEqual(len(summarize(roots)["groups"]), 3)

    def test_formal_dirty_rejected_and_all_six_routes_smoke(self):
        item = plan(self.spec)[0]
        if item["resolved"]["code"]["dirty"]:
            with self.assertRaisesRegex(ValueError, "dirty"):
                self.execute(item)
        with redirect_stdout(io.StringIO()):
            spec = create_smoke_spec(self.root / "six", include_ridge=True)
            roots = [self.execute(item) for item in self.items(spec)]
        summary = summarize(roots)
        self.assertEqual(len(summary["groups"]), 6)
        self.assertFalse(summary["rejected"])
        self.assertTrue(all(group["complete"] for group in summary["groups"]))


if __name__ == "__main__":
    unittest.main()

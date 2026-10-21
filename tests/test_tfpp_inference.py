"""推理产物完整性、显式输入阶段和 CPU 预测复现。"""

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import infer
import test_tfpp_regular as regular_tests
from utils.Genotype import dosage_to_model_codes, tfpp_input_views
from utils.Inference import prepare_inference_input, transform_phenotypes
from utils.Normalization import denormalize_phenotype
from utils.model_loader import load_tfpp_model
from utils.load_model_from_checkpoint import load_model_from_checkpoint, run_inference


class InferenceTests(unittest.TestCase):
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
        self.trainer = self.fixture.trainer()
        self.root = Path(self.trainer.config.output_dir)
        self.ckpt = self.root / "models" / "best_model.pt"
        torch.save(self.trainer.model.state_dict(), self.ckpt)
        self.model, self.info = load_tfpp_model(self.root, verbose=False)
        self.artifacts = self.info["artifacts"]
        self.meta_path = self.root / "preprocessing" / "preprocessing.json"

    def rewrite_metadata(self, edit):
        meta = json.loads(self.meta_path.read_text())
        edit(meta)
        self.meta_path.write_text(json.dumps(meta))

    def predict(self, *extra):
        argv = ["--model_path", str(self.root), "--device", "cpu", "--batch_size", "3",
                "--snp_file", str(self.fixture.root / "snp.npy"),
                "--output", str(self.fixture.root / "pred.csv"), *extra]
        with redirect_stdout(io.StringIO()):
            return infer.predict(argv)

    def test_raw_and_screened_are_exact_training_test_columns(self):
        raw = self.fixture.snp[self.artifacts.split.test_idx]
        codes = prepare_inference_input(raw, self.artifacts, input_stage="raw")
        expected = self.trainer.test_loader.dataset.tensors[0]
        torch.testing.assert_close(codes, expected.long())
        selected = raw[:, self.artifacts.indices]
        screened = prepare_inference_input(selected, self.artifacts, input_stage="screened")
        torch.testing.assert_close(screened, expected.long())
        np.testing.assert_array_equal(codes, dosage_to_model_codes(selected))

    def test_input_stage_is_required_even_when_dimensions_match(self):
        with self.assertRaisesRegex(ValueError, "显式指定"):
            prepare_inference_input(self.fixture.snp, self.artifacts, input_stage=None)
        trainer = self.fixture.trainer(screen_k=0, gwas_k=0)
        torch.save(trainer.model.state_dict(), trainer.best_model_save_dir / "best_model.pt")
        _, info = load_tfpp_model(trainer.config.output_dir, verbose=False)
        for stage in ("raw", "screened"):
            result = prepare_inference_input(self.fixture.snp, info["artifacts"], input_stage=stage)
            np.testing.assert_array_equal(result, dosage_to_model_codes(self.fixture.snp))

    def test_bad_dimensions_and_encoding_are_actionable(self):
        for stage in ("raw", "screened"):
            with self.assertRaisesRegex(ValueError, "列"):
                prepare_inference_input(np.zeros((2, 99)), self.artifacts, input_stage=stage)
        raw = self.fixture.snp.copy().astype(float)
        for invalid in (3, 0.5, np.inf, -2):
            raw[0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "0/1/2"):
                prepare_inference_input(raw, self.artifacts, input_stage="raw")

    def test_missing_artifacts_fail_even_with_legacy_flag(self):
        paths = [self.root / "preprocessing" / name
                 for name in ("split.npz", "screened_indices.npy", "preprocessing.json")]
        paths += [self.root / "scalers" / "scaler.pkl", self.root / "models" / "model_config.json"]
        for path in paths:
            content = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(path=path.name), self.assertRaises(FileNotFoundError):
                    load_tfpp_model(self.root, legacy=True, verbose=False)
            finally:
                path.write_bytes(content)

    def test_invalid_duplicate_and_out_of_range_indices_fail(self):
        path = self.root / "preprocessing" / "screened_indices.npy"
        original = path.read_bytes()
        for values in (np.full(self.artifacts.indices.shape, 99),
                       np.zeros_like(self.artifacts.indices), np.array([0.5])):
            np.save(path, values)
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "原始 SNP 列索引"):
                load_tfpp_model(self.root, verbose=False)
        path.write_bytes(original)

    def test_missing_versions_fills_and_hashes_fail(self):
        original = self.meta_path.read_bytes()
        edits = [lambda m: m.pop("schema_version"),
                 lambda m: m.pop("screening_dosage_encoding_version"),
                 lambda m: m["wide_imputer"].pop("fill_values"),
                 lambda m: m.pop("wide_fill_values_sha256"),
                 lambda m: m.pop("scaler_sha256"),
                 lambda m: m.pop("screened_indices_sha256")]
        for edit in edits:
            self.meta_path.write_bytes(original)
            self.rewrite_metadata(edit)
            with self.assertRaises(ValueError):
                load_tfpp_model(self.root, legacy=True, verbose=False)

    def test_tampered_files_and_wide_checkpoint_buffer_fail(self):
        path = self.root / "preprocessing" / "screened_indices.npy"
        indices = self.artifacts.indices.copy()[::-1]
        np.save(path, indices)
        with self.assertRaisesRegex(ValueError, "哈希"):
            load_tfpp_model(self.root, verbose=False)
        np.save(path, self.artifacts.indices)
        state = torch.load(self.ckpt, weights_only=True)
        state["wide_fill_values"][0] = (state["wide_fill_values"][0] + 1) % 3
        torch.save(state, self.ckpt)
        with self.assertRaisesRegex(ValueError, "Wide 填充值"):
            load_tfpp_model(self.root, verbose=False)

    def test_config_dimensions_modes_names_times_and_overrides_fail(self):
        path = self.root / "models" / "model_config.json"
        original = path.read_bytes()
        changes = {"snp_dim": 99, "wide_input_mode": "legacy_categorical",
                   "phenotype_names": ["B", "A"], "time_points": 4,
                   "time_coordinates": [1, 2, 9], "num_embeddings": 3}
        for key, value in changes.items():
            config = json.loads(original)
            config[key] = value
            path.write_text(json.dumps(config))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_tfpp_model(self.root, verbose=False)
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "覆盖"):
            load_tfpp_model(self.root, phenotype_names=["B", "A"], verbose=False)

    def test_heldout_labels_never_affect_inputs_or_predictions_and_no_fit(self):
        # 屏蔽所有已知的重新筛选/拟合入口；CLI 仍应完整执行成功。
        with (patch("utils.Screening.screen_snps", side_effect=AssertionError("筛选")) as screen,
              patch("dataloader.TFPP_DataLoader.get_dataloader", side_effect=AssertionError("loader")),
              patch("sklearn.preprocessing.StandardScaler.fit", side_effect=AssertionError("fit"))):
            first = self.predict("--pheno_dir", str(self.fixture.root))
            x, y, ids = infer.prepare_test_data(self.fixture.root / "snp.npy", self.artifacts,
                                                self.fixture.root)
            changed = self.fixture.pheno.copy()
            changed[self.artifacts.split.test_idx] += 1000
            self.fixture.write_inputs(self.fixture.snp, changed)
            second = self.predict("--pheno_dir", str(self.fixture.root))
            x2, y2, ids2 = infer.prepare_test_data(self.fixture.root / "snp.npy", self.artifacts,
                                                  self.fixture.root)
        screen.assert_not_called()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(x, x2, rtol=0, atol=0)
        self.assertFalse(torch.equal(y, y2))
        np.testing.assert_array_equal(ids, ids2)
        torch.testing.assert_close(y, self.trainer.test_loader.dataset.tensors[1])

    def test_changed_missing_distribution_uses_saved_fill(self):
        before = self.model.wide_fill_values.clone()
        raw = self.fixture.snp.copy()
        raw[self.artifacts.split.test_idx] = -1
        self.fixture.write_inputs(raw, self.fixture.pheno)
        codes, _, _ = infer.prepare_test_data(self.fixture.root / "snp.npy", self.artifacts)
        deep, wide = tfpp_input_views(codes, self.model.wide_fill_values)
        self.assertTrue((deep == 0).all())
        torch.testing.assert_close(wide, (before / 2).expand_as(wide))
        self.predict()
        torch.testing.assert_close(before, self.model.wide_fill_values, rtol=0, atol=0)
        _, loaded = load_tfpp_model(self.root, verbose=False)
        np.testing.assert_array_equal(loaded["artifacts"].wide_fill_values, before.numpy())

    def test_cpu_cli_prediction_matches_one_epoch_training_checkpoint(self):
        with redirect_stdout(io.StringIO()):
            self.trainer.train()
        result = self.predict()
        x = self.trainer.test_loader.dataset.tensors[0]
        self.trainer.model.eval()
        with torch.inference_mode():
            expected = denormalize_phenotype(self.trainer.model(x), self.trainer.scalers)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        raw = self.fixture.snp[self.artifacts.split.test_idx]
        for stage, data in (("raw", raw), ("screened", raw[:, self.artifacts.indices])):
            path = self.fixture.root / f"{stage}.npy"
            np.save(path, data)
            external = self.predict("--input_snp", str(path), "--input_stage", stage)
            torch.testing.assert_close(external, expected, rtol=1e-5, atol=1e-5)
        record = json.loads((self.fixture.root / "pred.metadata.json").read_text())
        self.assertFalse(record["legacy"])
        self.assertIn("wide_fill_values", record["artifact_hashes"])

    def test_existing_schema2_step2_artifacts_load_without_refitting(self):
        self.rewrite_metadata(lambda meta: meta.update(schema_version=2))
        _, info = load_tfpp_model(self.root, verbose=False)
        self.assertFalse(info["legacy"])

    def test_all_wide_and_deep_checkpoint_predictions_match(self):
        for mode in ("all", "wide", "deep"):
            trainer = self.fixture.trainer(ablation_mode=mode)
            trainer.model.eval()
            torch.save(trainer.model.state_dict(), trainer.best_model_save_dir / "best_model.pt")
            restored, info = load_tfpp_model(trainer.config.output_dir, verbose=False)
            codes = prepare_inference_input(self.fixture.snp, info["artifacts"], input_stage="raw")
            with torch.inference_mode():
                torch.testing.assert_close(restored(codes), trainer.model(codes), rtol=0, atol=0)

    def test_both_model_loading_apis_are_strict(self):
        restored, info = load_model_from_checkpoint(self.root)
        self.assertEqual(info["snp_dim"], self.artifacts.metadata["snp_dim"])
        torch.testing.assert_close(restored.wide_fill_values, self.model.wide_fill_values)
        with self.assertRaisesRegex(ValueError, "artifacts"):
            run_inference(restored, self.fixture.snp, "cpu")
        predicted = run_inference(restored, self.fixture.snp, "cpu",
                                  artifacts=info["artifacts"], input_stage="raw")
        self.assertEqual(predicted.shape, (40, 2, 3))
        self.ckpt.unlink()
        with self.assertRaises(FileNotFoundError):
            load_model_from_checkpoint(self.root)

    def test_saved_scaler_transform_inverse_for_each_normalization(self):
        for method in ("global", "timepoint", "residual_global"):
            trainer = self.fixture.trainer(norm_method=method)
            y = self.fixture.pheno[trainer.split.test_idx]
            norm = transform_phenotypes(y, trainer.scalers)
            torch.testing.assert_close(norm, trainer.test_loader.dataset.tensors[1])
            np.testing.assert_allclose(denormalize_phenotype(norm, trainer.scalers), y, atol=1e-5)

    def test_error_strategy_keeps_empty_saved_buffer_and_rejects_missing(self):
        self.fixture.write_inputs(np.maximum(self.fixture.snp, 0), self.fixture.pheno)
        trainer = self.fixture.trainer(wide_missing_strategy="error", screen_missing_strategy="error")
        ckpt = trainer.best_model_save_dir / "best_model.pt"
        torch.save(trainer.model.state_dict(), ckpt)
        model, info = load_tfpp_model(trainer.config.output_dir, verbose=False)
        self.assertEqual(model.wide_fill_values.numel(), 0)
        with self.assertRaisesRegex(ValueError, "不能有缺失"):
            prepare_inference_input(np.full((2, 12), -1), info["artifacts"], input_stage="raw")
        state = torch.load(ckpt, weights_only=True)
        state.pop("wide_fill_values")
        torch.save(state, ckpt)
        with self.assertRaisesRegex(ValueError, "Wide 填充值"):
            load_tfpp_model(trainer.config.output_dir, legacy=True, verbose=False)

    def test_legacy_requires_opt_in_warns_and_marks_every_output(self):
        legacy = self.fixture.root / "legacy"
        (legacy / "models").mkdir(parents=True)
        (legacy / "scalers").mkdir()
        state = torch.load(self.ckpt, weights_only=True)
        state.pop("wide_fill_values")
        torch.save(state, legacy / "models" / "best_model.pt")
        config = self.artifacts.config.copy()
        config["wide_input_mode"] = "legacy_categorical"
        (legacy / "models" / "model_config.json").write_text(json.dumps(config))
        shutil.copyfile(self.root / "scalers" / "scaler.pkl", legacy / "scalers" / "scaler.pkl")
        with self.assertRaisesRegex(ValueError, "legacy"):
            load_tfpp_model(legacy, verbose=False)
        raw = self.fixture.snp[:, self.artifacts.indices]
        path = self.fixture.root / "legacy_input.npy"
        np.save(path, raw)
        output = self.fixture.root / "legacy.csv"
        with self.assertWarnsRegex(RuntimeWarning, "LEGACY"), redirect_stdout(io.StringIO()):
            infer.predict(["--model_path", str(legacy), "--legacy", "--input_snp", str(path),
                           "--input_stage", "screened", "--device", "cpu", "--output", str(output)])
        frame = pd.read_csv(output)
        self.assertTrue(frame["legacy"].all())
        self.assertFalse(frame["paper_eligible"].any())
        record = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertFalse(record["paper_eligible"])
        with self.assertWarnsRegex(RuntimeWarning, "LEGACY"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "仅支持"):
                infer.predict(["--model_path", str(legacy), "--legacy", "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()

"""折内筛选隔离、独立模型和 schema 3 推理复现；仅 CPU 合成数据。"""

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

import infer
import test_tfpp_regular as regular_tests
from config import TFPPConfig
from trainer import TFPPTrainer
from utils.Genotype import TrainingDosageImputer, model_codes_to_dosage, validate_dosage
from utils.Inference import prepare_inference_input
from utils.model_loader import load_tfpp_model
from utils.Screening import compute_cache_key, screen_snps
from utils.Split import split_from_indices


class KFoldTests(unittest.TestCase):
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

    def parent(self, **options):
        defaults = dict(use_kfold=True, n_folds=2, split_file=None, kfold_val_ratio=0.2)
        defaults.update(options)
        return self.fixture.trainer(**defaults)

    def prepare(self, parent, fold=1, split=None):
        if split is None:
            split = parent._kfold_splits()[fold - 1]
        with redirect_stdout(io.StringIO()):
            child = parent._prepare_fold(fold, split)
        self.addCleanup(self.fixture.close_logger, child)
        return child

    def assert_same_preprocessing(self, first, second):
        np.testing.assert_array_equal(first.screened_indices, second.screened_indices)
        np.testing.assert_array_equal(first.wide_fill_values, second.wide_fill_values)
        self.assertEqual(first.screening_cache_key, second.screening_cache_key)
        self.assertEqual(first.train_indices_sha256, second.train_indices_sha256)
        for key in ("training_genotype_sha256", "training_phenotype_sha256", "screening_imputer",
                    "wide_imputer", "wide_fill_values_sha256", "scaler_sha256"):
            self.assertEqual(first.preprocessing_metadata[key], second.preprocessing_metadata[key])
        for name in first.phenotype_names:
            a = first.scalers["pheno_meta"]["scalers"][name]
            b = second.scalers["pheno_meta"]["scalers"][name]
            np.testing.assert_array_equal(a.mean_, b.mean_)
            np.testing.assert_array_equal(a.scale_, b.scale_)

    def test_outer_coordinator_does_not_screen_fit_or_create_model(self):
        with (patch("utils.screen_snps") as screen,
              patch.object(TrainingDosageImputer, "fit") as fit,
              patch.object(TFPPTrainer, "_create_model") as create):
            parent = self.parent()
        screen.assert_not_called()
        fit.assert_not_called()
        create.assert_not_called()
        self.assertIsNone(parent.model)
        self.assertEqual(parent.dataset.snp.shape, (40, 12))

    def test_outer_test_once_disjoint_roles_and_reproducible_splits(self):
        for folds in (2, 3, 5):
            parent = self.parent(n_folds=folds)
            splits = parent._kfold_splits()
            other = parent._kfold_splits()
            test_rows = np.concatenate([split.test_idx for split in splits])
            np.testing.assert_array_equal(np.sort(test_rows), np.arange(40))
            for split, repeat in zip(splits, other):
                for name in ("train_idx", "val_idx", "test_idx"):
                    np.testing.assert_array_equal(getattr(split, name), getattr(repeat, name))
                combined = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
                self.assertEqual(len(np.unique(combined)), 40)
                self.assertGreater(len(split.val_idx), 0)

    def test_each_selector_receives_only_its_training_rows_and_legal_dosage(self):
        parent = self.parent()
        for fold, split in enumerate(parent._kfold_splits(), 1):
            with patch("utils.screen_snps", wraps=screen_snps) as screen:
                child = self.prepare(parent, fold, split)
            dosage, phenotype = screen.call_args.args
            validate_dosage(dosage)
            self.assertEqual(dosage.shape, (len(split.train_idx), 12))
            np.testing.assert_array_equal(phenotype, self.fixture.pheno[split.train_idx])
            observed = self.fixture.snp[split.train_idx]
            np.testing.assert_array_equal(dosage[observed >= 0], observed[observed >= 0])
            self.assertTrue(screen.call_args.kwargs["multivariate"])
            self.assertEqual(child.preprocessing_metadata["screening_imputer"]["original_train_idx"],
                             split.train_idx.tolist())
            self.assertEqual(child.model.wide_input_mode, "dosage_v1")

    def test_heldout_genotypes_and_labels_do_not_change_any_fitted_artifact(self):
        parent = self.parent(norm_method="residual_global")
        split = parent._kfold_splits()[0]
        first = self.prepare(parent, split=split)
        held = np.concatenate([split.val_idx, split.test_idx])
        snp, pheno = self.fixture.snp.copy(), self.fixture.pheno.copy()
        snp[held] = -1  # 全部缺失也不能改变训练填补规则。
        pheno[held] += 900
        self.fixture.write_inputs(snp, pheno)
        second = self.prepare(self.parent(norm_method="residual_global"), split=split)
        self.assert_same_preprocessing(first, second)
        np.testing.assert_array_equal(first.scalers["pheno_meta"]["mean_curve"],
                                      second.scalers["pheno_meta"]["mean_curve"])
        self.assertTrue((second.test_loader.dataset.tensors[0] == 0).all())

    def test_training_genotype_or_labels_change_cache_key(self):
        parent = self.parent()
        split = parent._kfold_splits()[0]
        first = self.prepare(parent, split=split)
        for kind in ("genotype", "phenotype"):
            snp, pheno = self.fixture.snp.copy(), self.fixture.pheno.copy()
            if kind == "genotype":
                snp[split.train_idx[0], 0] = (int(snp[split.train_idx[0], 0]) + 1) % 3
            else:
                pheno[split.train_idx[0], 0, 0] += 10
            self.fixture.write_inputs(snp, pheno)
            changed = self.prepare(self.parent(), split=split)
            self.assertNotEqual(first.screening_cache_key, changed.screening_cache_key)

    def test_fold_number_is_part_of_key_even_with_identical_training_rows(self):
        parent = self.parent()
        split = parent._kfold_splits()[0]
        first = self.prepare(parent, fold=1, split=split)
        other = split_from_indices(split.train_idx, split.val_idx, split.test_idx, 40)
        other.metadata.update(split.metadata, fold_index=2)
        second = self.prepare(parent, fold=2, split=other)
        self.assertNotEqual(first.screening_cache_key, second.screening_cache_key)
        # 排除 seed 差异，直接验证 fold 标识参与实际调用的缓存键。
        with patch("utils.compute_cache_key", wraps=compute_cache_key) as cache:
            self.prepare(self.parent(), fold=1, split=split)
        kwargs = cache.call_args.kwargs.copy()
        args = cache.call_args.args
        first_key = compute_cache_key(*args, **kwargs)
        kwargs["split_info"] = kwargs["split_info"].replace("fold_01", "fold_02")
        self.assertNotEqual(first_key, compute_cache_key(*args, **kwargs))

    def test_different_dimensions_rebuild_models_and_indices_remain_original(self):
        parent = self.parent()
        splits = parent._kfold_splits()
        for fold, indices in ((1, np.array([0, 3, 10])), (2, np.array([2, 9]))):
            with patch("utils.screen_snps", return_value=indices):
                child = self.prepare(parent, fold, splits[fold - 1])
            self.assertEqual(child.model.wide_layer.snp_importance.shape[1], len(indices))
            self.assertEqual(child.wide_fill_values.shape, (len(indices),))
            np.testing.assert_array_equal(child.screened_indices, indices)
            np.testing.assert_array_equal(model_codes_to_dosage(child.dataset.snp),
                                          self.fixture.snp[:, indices])
        self.assertEqual(parent.dataset.snp.shape[1], 12)

    def test_two_fold_cpu_training_independent_state_and_inference_replay(self):
        parent = self.parent()
        children, optimizers, schedulers, initial_embeddings = [], [], [], []
        test_calls = []
        prepare = parent._prepare_fold
        optimizer_factory = TFPPTrainer._create_optimizer_and_scheduler
        evaluate = TFPPTrainer._evaluate_test_set

        def prepare_record(fold, split):
            child = prepare(fold, split)
            children.append(child)
            initial_embeddings.append(child.model.snp_emb.weight.detach().clone())
            return child

        def optimizer_record(child):
            optimizer, scheduler = optimizer_factory(child)
            self.assertFalse(optimizer.state)
            model_params = {id(p) for p in child.model.parameters() if p.requires_grad}
            optim_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(model_params, optim_params)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            return optimizer, scheduler

        def evaluate_record(child, loader, scalers):
            self.assertIs(loader, child.test_loader)
            self.assertIsNot(loader, child.val_loader)
            test_calls.append(child.config.fold_index)
            return evaluate(child, loader, scalers)

        with (patch.object(parent, "_prepare_fold", side_effect=prepare_record),
              patch.object(TFPPTrainer, "_create_optimizer_and_scheduler", optimizer_record),
              patch.object(TFPPTrainer, "_evaluate_test_set", evaluate_record),
              redirect_stdout(io.StringIO())):
            result = parent.train()
        self.assertEqual(test_calls, [1, 2])
        for collection in (children, optimizers, schedulers):
            self.assertIsNot(collection[0], collection[1])
        self.assertIsNot(children[0].model, children[1].model)
        self.assertIsNot(children[0].scalers, children[1].scalers)
        self.assertFalse(torch.equal(initial_embeddings[0], initial_embeddings[1]))
        params = [{p.data_ptr() for p in child.model.parameters()} for child in children]
        self.assertFalse(params[0] & params[1])
        for child in children:
            root = Path(child.config.output_dir)
            required = ["models/best_model.pt", "models/model_config.json", "preprocessing/split.npz",
                        "preprocessing/screened_indices.npy", "preprocessing/preprocessing.json",
                        "scalers/scaler.pkl", "test_predictions.npz", "test_predictions.csv",
                        "metrics.json", "config.json"]
            self.assertTrue(all((root / name).is_file() for name in required))
            loaded, info = load_tfpp_model(root, verbose=False)
            self.assertFalse(info["legacy"])
            self.assertEqual(info["artifacts"].metadata["schema_version"], 3)
            self.assertEqual(info["artifacts"].metadata["training_mode"], "kfold")
            with np.load(root / "test_predictions.npz") as saved:
                codes = prepare_inference_input(self.fixture.snp[saved["sample_idx"]],
                                                 info["artifacts"], input_stage="raw")
                with torch.inference_mode():
                    np.testing.assert_allclose(loaded(codes), saved["predictions_normalized"],
                                               rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(saved["sample_id"], saved["sample_idx"])
                if child.config.fold_index == 1:
                    with redirect_stdout(io.StringIO()):
                        replay = infer.predict(["--model_path", str(root), "--device", "cpu",
                                                "--snp_file", str(self.fixture.root / "snp.npy"),
                                                "--output", str(root / "replay.csv")])
                    np.testing.assert_allclose(replay, saved["predictions"], rtol=1e-5, atol=1e-5)
        history = Path(parent.config.training_history_save_dir)
        predictions = pd.read_csv(history / "kfold_predictions.csv")
        for name in parent.phenotype_names:
            frame = predictions[predictions.phenotype == name]
            np.testing.assert_array_equal(frame.sample_id, np.arange(40))
            self.assertEqual(len(frame), 40)
        summary = json.loads((history / "kfold_summary.json").read_text())
        for key, values in result["overall_denorm_metrics"].items():
            self.assertAlmostEqual(summary["overall_statistics"][key]["mean"], np.mean(values))
            self.assertAlmostEqual(summary["overall_statistics"][key]["std"], np.std(values))

    def test_only_validation_selects_checkpoint_and_test_runs_after_selection(self):
        parent = self.parent(epochs=2)
        first_epoch = {}
        validate = TFPPTrainer._validate_epoch
        evaluate = TFPPTrainer._evaluate_test_set

        def validation(child, loader, criterion):
            self.assertIs(loader, child.val_loader)
            _, per_pheno, overall = validate(child, loader, criterion)
            fold = child.config.fold_index
            if fold not in first_epoch:
                first_epoch[fold] = {k: v.detach().cpu().clone() for k, v in child.model.state_dict().items()}
                loss = 1.0
            else:
                loss = 2.0
            return loss, per_pheno, overall

        def evaluation(child, loader, scalers):
            for key, value in child.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), first_epoch[child.config.fold_index][key], rtol=0, atol=0)
            return evaluate(child, loader, scalers)

        with (patch.object(TFPPTrainer, "_validate_epoch", validation),
              patch.object(TFPPTrainer, "_evaluate_test_set", evaluation), redirect_stdout(io.StringIO())):
            result = parent.train()
        self.assertEqual([fold["best_epoch"] for fold in result["fold_results"]], [1, 1])

    def test_invalid_fold_configs_and_stale_split_are_rejected(self):
        for options in ({"n_folds": 1}, {"kfold_val_ratio": 0}, {"kfold_val_ratio": 1}):
            with self.assertRaises(ValueError):
                TFPPConfig(**options)
        with self.assertRaisesRegex(ValueError, "不能超过"):
            self.parent(n_folds=41)
        parent = self.parent()
        splits = parent._kfold_splits()
        child = self.prepare(parent, split=splits[0])
        other = split_from_indices(splits[0].val_idx, splits[0].train_idx, splits[0].test_idx, 40)
        other.metadata.update(splits[0].metadata)
        with self.assertRaisesRegex(ValueError, "已有折划分"):
            self.prepare(parent, split=other)
        torch.save(child.model.state_dict(), child.best_model_save_dir / "best_model.pt")
        config_path = child.best_model_save_dir / "model_config.json"
        config = json.loads(config_path.read_text())
        config["fold_index"] = 99
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "编号"):
            load_tfpp_model(child.config.output_dir, verbose=False)

    def test_cli_forwards_inner_validation_ratio(self):
        import train

        with (patch.object(sys, "argv", ["train.py", "--use_kfold", "true", "--n_folds", "2",
                                         "--kfold_val_ratio", "0.25"]),
              patch.object(train, "TFPPTrainer") as trainer):
            train.main()
        config = trainer.call_args.kwargs["config"]
        self.assertEqual(config.kfold_val_ratio, 0.25)
        self.assertTrue(config.use_kfold)


if __name__ == "__main__":
    unittest.main()

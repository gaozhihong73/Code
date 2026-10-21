"""最终审计：真实核心函数证据、完整训练泄漏回归、封存与回放闭环。"""

import importlib
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import infer
import test_convae_model as mae_tests
import test_experiments as experiment_tests
import test_tfpp_kfold as fold_tests
import test_tfpp_regular as regular_tests
from config import TFPPConfig
from dataloader import TFPPDataset
from experiments.runner import run_unit, summarize, validate_result
from experiments.smoke import create_smoke_spec
from experiments.spec import plan, file_digest
from utils.Calculate_Metrics import compute_metrics_vectorized, calculate_metrics_by_average
from utils.ConvAE_Export import export_convae_features, load_exported_features
from utils.ConvAE_Ridge import metric_hierarchy, save_probe_targets
from utils.model_loader import load_tfpp_model


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, cls=regular_tests.RegularTFPPTests):
        obj = cls()
        obj.setUp()
        self.addCleanup(obj.doCleanups)
        return obj

    def train(self, trainer):
        with redirect_stdout(io.StringIO()):
            return trainer.train()

    def same_weights(self, a, b):
        for name, value in a.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), b.model.state_dict()[name].cpu(), rtol=0, atol=0)

    def test_training_rng_isolated_from_other_instances_and_loader_inspection(self):
        f = self.fixture()
        a, b = f.trainer(dropout=0.2), f.trainer(dropout=0.2)
        self.train(a)
        torch.rand(100)
        self.train(b)
        self.same_weights(a, b)
        helper = self.fixture(mae_tests.ConvAEModelTests)
        a, b = helper.trainer(dropout=0.2), helper.trainer(dropout=0.2)
        next(iter(b.train_loader))  # 官方训练前检查 batch 不应改变正式训练 RNG。
        self.train(a)
        torch.rand(100)
        self.train(b)
        self.same_weights(a, b)

    def test_regular_full_training_seed_and_test_label_genotype_isolation(self):
        f = self.fixture()
        first = f.trainer()
        original = self.train(first)
        pheno = f.pheno.copy()
        pheno[f.split.test_idx] += 500
        f.write_inputs(f.snp, pheno)
        second = f.trainer()
        changed_labels = self.train(second)
        self.same_weights(first, second)
        f.assert_same_training_artifacts(first, second)
        self.assertEqual(original['best_val_loss'], changed_labels['best_val_loss'])
        np.testing.assert_array_equal(first.test_predictions['predictions'], second.test_predictions['predictions'])
        self.assertNotEqual(original['test_results']['denormalized']['overall']['mse'],
                            changed_labels['test_results']['denormalized']['overall']['mse'])
        genotype = f.snp.copy()
        row, column = f.split.test_idx[0], first.screened_indices[0]
        genotype[row, column] = (max(0, int(genotype[row, column])) + 1) % 3
        f.write_inputs(genotype, f.pheno)
        third = f.trainer()
        self.train(third)
        self.same_weights(first, third)
        f.assert_same_training_artifacts(first, third)
        np.testing.assert_allclose(first.test_predictions['predictions'][1:], third.test_predictions['predictions'][1:], rtol=0, atol=0)
        self.assertFalse(np.array_equal(first.test_predictions['predictions'][0], third.test_predictions['predictions'][0]))

    def test_kfold_full_training_isolation_applies_within_each_fold(self):
        helper = self.fixture(fold_tests.KFoldTests)
        f = helper.fixture
        parent = helper.parent()
        split = parent._kfold_splits()[0]
        first = helper.prepare(parent, split=split)
        self.train(first)
        phenotype = f.pheno.copy()
        phenotype[split.test_idx] -= 900
        genotype = f.snp.copy()
        genotype[split.test_idx[0], first.screened_indices[0]] = -1
        f.write_inputs(genotype, phenotype)
        second = helper.prepare(helper.parent(), split=split)
        self.train(second)
        helper.assert_same_preprocessing(first, second)
        self.same_weights(first, second)
        np.testing.assert_allclose(first.test_predictions['predictions'][1:], second.test_predictions['predictions'][1:], rtol=0, atol=0)

    def test_convae_full_training_encoder_features_ignore_test_labels_and_distribution(self):
        helper = self.fixture(mae_tests.ConvAEModelTests)
        f = helper.fixture
        first = helper.trainer()
        result = self.train(first)
        exported = export_convae_features(first.config.output_dir, split_file=f.split_path, output_root=f.root / 'base_features')
        initial, _ = load_exported_features(exported)
        labels = f.pheno.copy()
        labels[f.split.test_idx] += 900
        f.write_inputs(f.snp, labels)
        second = helper.trainer()
        repeated = self.train(second)
        self.same_weights(first, second)
        self.assertEqual(result['best_selection_loss'], repeated['best_selection_loss'])
        after, _ = load_exported_features(export_convae_features(second.config.output_dir, split_file=f.split_path, output_root=f.root / 'label_features'))
        for role in initial:
            np.testing.assert_array_equal(initial[role][0], after[role][0])
        genotype = f.snp.copy()
        genotype[f.split.test_idx[0]] = -1
        f.write_inputs(genotype, f.pheno)
        third = helper.trainer()
        self.train(third)
        self.same_weights(first, third)
        altered, _ = load_exported_features(export_convae_features(third.config.output_dir, split_file=f.split_path, output_root=f.root / 'genotype_features'))
        for role in ('train', 'val'):
            np.testing.assert_array_equal(initial[role][0], altered[role][0])
        np.testing.assert_allclose(initial['test'][0][1:], altered['test'][0][1:], rtol=0, atol=0)
        self.assertFalse(np.array_equal(initial['test'][0][0], altered['test'][0][0]))
        # 换掉训练行的 SNP 则不能复用原 checkpoint 作同一任务导出。
        genotype[f.split.train_idx[0], 0] = 2
        f.write_inputs(genotype, f.pheno)
        with self.assertRaisesRegex(ValueError, 'train genotype'):
            export_convae_features(first.config.output_dir, split_file=f.split_path, output_root=f.root / 'wrong_train')

    def test_time_columns_missing_labels_and_cli_booleans_fail_early(self):
        f = self.fixture()
        frame = pd.read_csv(f.root / 'B.csv')
        frame.iloc[:, ::-1].to_csv(f.root / 'B.csv', index=False)
        with self.assertRaisesRegex(ValueError, '时间列'):
            TFPPDataset(str(f.root / 'snp.npy'), str(f.root), ['A', 'B'])
        f.write_inputs(f.snp, f.pheno)
        with self.assertRaisesRegex(ValueError, 'time_coordinates'):
            f.trainer(time_coordinates=[1, 2])
        f.pheno[0, 0, 0] = np.nan
        f.write_inputs(f.snp, f.pheno)
        with self.assertRaisesRegex(ValueError, '有限'):
            f.trainer()
        from train import _str_to_bool
        self.assertTrue(_str_to_bool('true'))
        self.assertFalse(_str_to_bool('false'))
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            _str_to_bool('typo')

    def test_finite_val_loss_and_one_time_test_guard_standalone(self):
        f = self.fixture()
        first = f.trainer()
        with patch.object(first, '_validate_epoch', return_value=(float('nan'), {}, {})):
            with self.assertRaisesRegex(ValueError, '验证损失'):
                self.train(first)
        self.assertFalse((Path(first.config.output_dir) / 'test_evaluation.json').exists())
        second = f.trainer()
        self.train(second)
        with self.assertRaisesRegex(ValueError, 'test 已'):
            self.train(second)
        from trainer import TFPPTrainer
        with self.assertRaises(FileExistsError):
            TFPPTrainer(second.config)

    def test_checkpoint_integrity_time_order_and_inference_are_strict(self):
        f = self.fixture()
        trainer = f.trainer()
        self.train(trainer)
        model, info = load_tfpp_model(trainer.config.output_dir, verbose=False)
        self.assertTrue(info['paper_eligible'])
        with (patch('utils.screen_snps', side_effect=AssertionError('推理重筛选')),
              patch('sklearn.preprocessing.StandardScaler.fit', side_effect=AssertionError('推理fit'))):
            codes, _, _ = infer.prepare_test_data(f.root / 'snp.npy', info['artifacts'])
            with torch.inference_mode():
                from utils.Normalization import denormalize_phenotype
                replay = denormalize_phenotype(model(codes), info['artifacts'].scalers).numpy()
        np.testing.assert_allclose(replay, trainer.test_predictions['predictions'], rtol=1e-5, atol=1e-6)
        for name in ('A', 'B'):
            frame = pd.read_csv(f.root / (name + '.csv'))
            frame.iloc[:, ::-1].to_csv(f.root / (name + '.csv'), index=False)
        with self.assertRaisesRegex(ValueError, '时间列'):
            infer.prepare_test_data(f.root / 'snp.npy', info['artifacts'], f.root)
        modified = f.snp.copy()
        modified[f.split.train_idx[0], 0] = 2
        np.save(f.root / 'wrong.npy', modified)
        with self.assertRaisesRegex(ValueError, '训练行'):
            infer.prepare_test_data(f.root / 'wrong.npy', info['artifacts'])
        checkpoint = Path(trainer.config.output_dir) / 'models/best_model.pt'
        weights = torch.load(checkpoint, weights_only=True)
        key = next(k for k, v in weights.items() if v.is_floating_point() and v.numel() > 1)
        weights[key].add_(1)
        torch.save(weights, checkpoint)
        with self.assertRaisesRegex(ValueError, '哈希'):
            load_tfpp_model(trainer.config.output_dir, verbose=False)

    def test_public_metrics_and_probe_share_nan_counts_negative_r2(self):
        target = np.arange(18).reshape(3, 2, 3).astype(float)
        pred = target + 100
        hierarchy = metric_hierarchy(pred, target, ['A', 'B'], ['0', '1', '2'])
        direct = {name: compute_metrics_vectorized(pred[:, p], target[:, p]) for p, name in enumerate(('A', 'B'))}
        self.assertEqual(hierarchy['overall'], calculate_metrics_by_average(direct))
        self.assertLess(hierarchy['overall']['r2'], 0)
        constant = compute_metrics_vectorized(np.ones((4, 3)), np.arange(12).reshape(4, 3))
        self.assertTrue(np.isnan(constant['pcc']))
        self.assertEqual(constant['pcc_valid_count'], 0)
        self.assertEqual(constant['pcc_total_count'], 3)

    def test_six_routes_reach_real_cores_and_production_replay(self):
        helper = self.fixture(experiment_tests.ExperimentTests)
        root = helper.root / 'production_evidence'
        runtime = importlib.import_module('experiments.runner')
        export_module = importlib.import_module('utils.ConvAE_Export')
        from utils.convae_model_loader import load_convae_model
        from utils.model_loader import load_tfpp_model as load_tfpp
        from trainer import TFPPTrainer
        from sklearn.linear_model import Ridge
        evidence = []
        real_tfpp_train = TFPPTrainer.train
        real_ridge_fit = Ridge.fit
        def train(obj):
            evidence.append(('TFPPTrainer.train', obj.config.ablation_mode))
            return real_tfpp_train(obj)
        def ridge_fit(model, x, y, *args, **kwargs):
            evidence.append(('Ridge.fit', len(x)))
            return real_ridge_fit(model, x, y, *args, **kwargs)
        with (patch.object(export_module, 'load_convae_model', wraps=load_convae_model) as mae_loader,
              patch.object(runtime, 'gwas_screen', wraps=runtime.gwas_screen) as gwas,
              patch.object(runtime, 'dc_sis_screen', wraps=runtime.dc_sis_screen) as dcsis,
              patch('utils.model_loader.load_tfpp_model', wraps=load_tfpp) as tfpp_loader,
              patch.object(TFPPTrainer, 'train', new=train), patch.object(Ridge, 'fit', new=ridge_fit),
              redirect_stdout(io.StringIO())):
            spec = create_smoke_spec(root)
            items = plan(spec, development_smoke=True)
            for item in items:
                run_unit(item)
        self.assertEqual(mae_loader.call_count, 1)
        self.assertEqual(tfpp_loader.call_count, 5)
        self.assertEqual(sum(name == 'TFPPTrainer.train' for name, _ in evidence), 5)
        self.assertEqual(sum(name == 'Ridge.fit' for name, _ in evidence), 9)
        self.assertEqual(gwas.call_count, 2)
        self.assertEqual(dcsis.call_count, 2)
        for call in [*gwas.call_args_list, *dcsis.call_args_list]:
            x, y = call.args[:2]
            self.assertEqual(len(x), 24)
            self.assertTrue(np.isin(x, [0, 1, 2]).all())
            self.assertEqual(len(y), 24)
        self.assertEqual(dcsis.call_args_list[-1].args[0].shape[1], 5)
        summaries = summarize([i['output_dir'] for i in items])
        self.assertFalse(summaries['rejected'])
        self.assertEqual(summaries['std_ddof'], 0)
        self.assertTrue(all(validate_result(i['output_dir'])[0]['replay_verified'] for i in items))

    def test_result_numeric_tampering_and_stale_test_state_are_rejected(self):
        helper = self.fixture(experiment_tests.ExperimentTests)
        item = helper.items()[1]
        root = helper.execute(item)
        manifest, _ = validate_result(root)
        csv = root / manifest['predictions_csv']
        frame = pd.read_csv(csv)
        frame.loc[0, 'prediction'] += 100
        frame.to_csv(csv, index=False)
        artifacts = root / 'artifacts.json'
        hashes = json.loads(artifacts.read_text())
        hashes[manifest['predictions_csv']] = file_digest(csv)
        artifacts.write_text(json.dumps(hashes))
        self.assertFalse(summarize([root])['groups'])
        status_path = root / 'status.json'
        status = json.loads(status_path.read_text())
        status.update(state='failed', test_evaluation_count=0)
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, 'test 已'):
            helper.execute(item, resume=True)

    def test_all_supported_cli_help_and_synthetic_train_export_ridge_replay(self):
        f = self.fixture()
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        def cli(script, *args):
            return subprocess.run([sys.executable, '-B', str(repo / script), *map(str, args)],
                cwd=f.root, env=env, check=True, text=True, capture_output=True)
        for script in ('train.py', 'infer.py', 'train_convae.py', 'export_convae.py', 'ridge_convae.py', 'run_experiments.py'):
            self.assertIn('usage:', cli(script, '--help').stdout)
        for sub in ('dry-run', 'run', 'smoke', 'summarize'):
            self.assertIn('usage:', cli('run_experiments.py', sub, '--help').stdout)
        common = ['--snp_file', f.root / 'snp.npy', '--pheno_dir', f.root, '--pheno_names', 'A', 'B',
            '--time_coordinates', 1, 2, 3, '--epochs', 1, '--batch_size', 4, '--num_workers', 0,
            '--d_model', 8, '--num_heads', 2, '--num_layers', 1, '--d_ff', 16, '--degree', 1,
            '--use_warmup', 'false', '--screen_k', 4, '--gwas_k', 0, '--screen_missing_strategy', 'mode',
            '--screen_cache_dir', f.root / 'cache']
        cli('train.py', '--exp_name', 'ordinary', '--split_file', f.split_path, *common)
        cli('train.py', '--exp_name', 'cv', '--use_kfold', 'true', '--n_folds', 2, *common)
        ordinary = f.root / 'outputs/tfpp/ordinary'
        cli('infer.py', '--model_path', ordinary, '--snp_file', f.root / 'snp.npy', '--device', 'cpu', '--output', f.root / 'replay.csv')
        cli('train_convae.py', '--exp_name', 'mae', '--output_root', f.root / 'mae', '--split_file', f.split_path,
            '--snp_file', f.root / 'snp.npy', '--pheno_dir', f.root, '--pheno_names', 'A', 'B', '--time_coordinates', 1, 2, 3,
            '--epochs', 1, '--batch_size', 4, '--num_workers', 0, '--device', 'cpu', '--chunk_size', 8, '--overlap', 2,
            '--patch_size', 3, '--d_model', 8, '--d_ext', 4, '--num_heads', 2, '--encoder_layers', 1, '--decoder_layers', 1,
            '--use_warmup', 'false')
        mae_root = next((f.root / 'mae/mae').glob('split_*'))
        cli('export_convae.py', '--model_path', mae_root, '--split_file', f.split_path,
            '--snp_file', f.root / 'snp.npy', '--output_root', f.root / 'features', '--device', 'cpu')
        exported = next((f.root / 'features').glob('*/*/*/metadata.json')).parent
        arrays, meta = load_exported_features(exported)
        targets = f.root / 'targets.npz'
        save_probe_targets(targets, {role: (f.pheno[getattr(f.split, role + '_idx')], arrays[role][1]) for role in arrays},
            phenotype_names=['A', 'B'], time_labels=['0', '1', '2'])
        cli('ridge_convae.py', '--export_dir', exported, '--phenotype_file', targets, '--output_root', f.root / 'ridge')
        self.assertTrue(list((f.root / 'ridge').glob('*/*/summary.json')))


if __name__ == '__main__':
    unittest.main()

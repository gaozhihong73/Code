"""ConvAE v2 输出、开关、有效目标、固定验证 mask 和单轮 CPU 训练回归。"""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader

import test_tfpp_regular as regular_tests
from config import ConvAEConfig
from model.ConvAE import ConvAE, sample_patch_mask
from trainer import ConvAETrainer
from utils.ConvAE_Objective import convae_loss_terms, fixed_reconstruction_masks
from utils.convae_model_loader import load_convae_model


class ConvAEModelTests(unittest.TestCase):
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
        self.fixture.snp = np.concatenate([self.fixture.snp, np.ones((40, 1), dtype=np.int8)], axis=1)
        self.fixture.snp[self.fixture.split.val_idx[0], 0] = -1
        self.fixture.write_inputs(self.fixture.snp, self.fixture.pheno)
        self.counter = 0

    def trainer(self, **options):
        self.counter += 1
        defaults = dict(exp_name=f"model_{self.counter}", output_root=str(self.fixture.root / "mae"),
                        split_file=str(self.fixture.split_path), snp_file=str(self.fixture.root / "snp.npy"),
                        pheno_dir=str(self.fixture.root), phenotype_names=["A", "B"], time_coordinates=[1, 2, 3],
                        chunk_size=8, overlap=2, patch_size=3, d_model=8, d_ext=4, num_heads=2,
                        encoder_layers=1, decoder_layers=1, dropout=0, num_workers=0, device="cpu",
                        epochs=1, use_warmup=False, batch_size=4)
        defaults.update(options)
        trainer = ConvAETrainer(ConvAEConfig(**defaults))
        self.addCleanup(self.fixture.close_logger, trainer)
        trainer.logger.set_console_level("ERROR")
        return trainer

    def model(self, **options):
        defaults = dict(chunk_size=8, patch_size=3, d_model=8, d_ext=4, num_chunks=2,
                        n_phenotypes=2, n_time_points=3, num_heads=2,
                        encoder_layers=1, decoder_layers=1, dropout=0)
        defaults.update(options)
        return ConvAE(**defaults)

    def inputs(self):
        x = torch.randint(1, 4, (2, 2, 8), generator=torch.Generator().manual_seed(73))
        padding = torch.zeros((2, 8), dtype=torch.bool)
        padding[1, 4:] = True
        x[:, padding] = 0
        x[0, 0, 0] = 0  # 真实缺失仍是有效位置。
        return x, padding

    def test_single_14_and_multiple_phenotypes_forward_backward(self):
        x, padding = self.inputs()
        for phenotypes, times in ((1, 14), (2, 3), (3, 5)):
            model = self.model(n_phenotypes=phenotypes, n_time_points=times)
            output = model(x, padding_mask=padding)
            self.assertEqual(output[1].shape, (2, phenotypes, times))
            terms = convae_loss_terms(output, x, torch.ones(2, phenotypes, times), padding, use_pheno_help=True)
            self.assertGreater(terms.recon_count, 0)
            terms.total(0.7, 0.3).backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_disabled_head_is_absent_not_called_and_not_in_optimizer(self):
        trainer = self.trainer(use_pheno_help=False, recon_weight=1, pheno_weight=0)
        self.assertIsNone(trainer.model.phenotype_head)
        self.assertFalse(any(name.startswith("phenotype_head") for name, _ in trainer.model.named_parameters()))
        optimizer, _ = trainer._create_optimizer_and_scheduler()
        parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(parameters, {id(p) for p in trainer.model.parameters()})
        x, y, _ = next(iter(trainer.train_loader))
        output = trainer.model(x, padding_mask=trainer.dataset.chunk_padding_mask)
        self.assertIsNone(output[1])
        terms = convae_loss_terms(output, x, y, trainer.dataset.chunk_padding_mask, use_pheno_help=False)
        self.assertEqual(terms.pheno_count, 0)
        self.assertEqual(terms.pheno_loss.item(), 0)
        terms.total(1, 0).backward()

    def test_padding_values_do_not_change_pooling_phenotypes_or_valid_logits(self):
        model = self.model().eval()
        x, padding = self.inputs()
        changed = x.float().clone()
        changed[:, padding] = float("nan")
        selected = torch.tensor([[[False, True, False], [True, False, False]]]).expand(2, -1, -1)
        with torch.inference_mode():
            for kwargs in ({"mask_ratio": 0}, {"reconstruction_mask": selected}):
                first = model(x, padding_mask=padding, **kwargs)
                second = model(changed, padding_mask=padding, **kwargs)
                for index in (0, 1):
                    torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)
                if first[2] is not None:
                    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_real_missing_uses_embedding_but_padding_has_zero_embedding_contribution(self):
        model = self.model().eval()
        x, padding = self.inputs()
        observed = x.clone()
        observed[observed == 0] = 1
        with torch.no_grad():
            before = model.feature_extractor(observed.reshape(4, 8), padding.expand(2, -1, -1).reshape(4, 8))
            model.feature_extractor.snp_embedding.weight[0].add_(7)
            after = model.feature_extractor(observed.reshape(4, 8), padding.expand(2, -1, -1).reshape(4, 8))
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            missing = observed.clone()
            missing[0, 0, 0] = 0
            actual = model.feature_extractor(missing.reshape(4, 8), padding.expand(2, -1, -1).reshape(4, 8))
        self.assertFalse(torch.equal(after[0, 0], actual[0, 0]))

    def test_loss_denominator_and_gradients_only_use_selected_observed_nonpadding_positions(self):
        codes = torch.tensor([[[1, 0, 3, 2, 0, 99, 99]]])
        padding = torch.tensor([[False] * 5 + [True] * 2])
        logits = torch.randn(1, 3, 3, 4, requires_grad=True)
        selected = torch.tensor([[True, False, False]])
        output = (None, None, logits, selected)
        terms = convae_loss_terms(output, codes, torch.zeros(1, 1, 1), padding, use_pheno_help=False)
        self.assertEqual(terms.recon_count, 2)
        expected = F.cross_entropy(logits[0, 0, [0, 2]], torch.tensor([1, 3]))
        torch.testing.assert_close(terms.recon_loss, expected)
        terms.recon_loss.backward()
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        allowed[0, 0, [0, 2]] = True
        self.assertTrue((logits.grad[~allowed] == 0).all())
        self.assertGreater(logits.grad[allowed].abs().sum().item(), 0)

    def test_last_partial_chunk_and_partial_patch_receive_reconstruction_supervision(self):
        trainer = self.trainer()
        codes, y, _ = next(iter(trainer.val_loader))
        codes = torch.ones_like(codes)
        padding = trainer.dataset.chunk_padding_mask
        selected = torch.zeros(codes.shape[0] * 2, 3, dtype=torch.bool)
        selected[1::2, -1] = True  # 最后一个 chunk 的最后 patch 仅有一个真实 SNP。
        logits = torch.zeros(codes.shape[0] * 2, 3, 3, 4, requires_grad=True)
        terms = convae_loss_terms((None, None, logits, selected), codes, y, padding, use_pheno_help=False)
        self.assertEqual(terms.recon_count, codes.shape[0])
        self.assertAlmostEqual(terms.recon_loss.item(), np.log(4), places=6)
        terms.recon_loss.backward()
        self.assertGreater(logits.grad[1::2, -1, 0].abs().sum(), 0)
        self.assertTrue((logits.grad[1::2, -1, 1:] == 0).all())

    def test_all_padding_skips_attention_and_produces_finite_zero_loss(self):
        model = self.model(use_pheno_help=False)
        x = torch.full((2, 2, 8), float("nan"))
        padding = torch.ones((2, 8), dtype=torch.bool)
        with (patch.object(model.encoder, "forward", side_effect=AssertionError("padding 不得进入 encoder")),
              patch.object(model.decoder, "forward", side_effect=AssertionError("padding 不得进入 decoder"))):
            output = model(x, padding_mask=padding)
        self.assertTrue((output[0] == 0).all())
        self.assertFalse(output[3].any())
        terms = convae_loss_terms(output, x, torch.ones(2, 2, 3), padding, use_pheno_help=False)
        self.assertEqual(terms.recon_count, 0)
        self.assertEqual(terms.recon_loss.item(), 0)
        terms.total(1, 0).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_single_visible_patch_all_masked_uses_safe_null_context(self):
        model = self.model(chunk_size=2, patch_size=3, num_chunks=1, use_pheno_help=False)
        x = torch.tensor([[[1, 2]]])
        output = model(x, reconstruction_mask=torch.ones(1, 1, 1, dtype=torch.bool))
        terms = convae_loss_terms(output, x, torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool), use_pheno_help=False)
        self.assertTrue(torch.isfinite(terms.recon_loss))
        self.assertGreater(terms.recon_loss.item(), 0)
        terms.recon_loss.backward()

    def test_no_observed_targets_returns_zero_and_skips_optimizer_weight_decay(self):
        self.fixture.write_inputs(np.full_like(self.fixture.snp, -1), self.fixture.pheno)
        trainer = self.trainer(use_pheno_help=False, recon_weight=1, pheno_weight=0, weight_decay=0.5)
        optimizer, _ = trainer._create_optimizer_and_scheduler()
        before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
        losses = trainer._train_epoch(trainer.train_loader, optimizer)
        self.assertEqual(losses, (0, 0, 0))
        self.assertFalse(optimizer.state)
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)
        self.assertEqual(trainer._validate_epoch(trainer.val_loader), (0, 0, 0))

    def test_validation_reconstruction_nonzero_fixed_and_independent_of_batch_order(self):
        trainer = self.trainer()
        first = trainer._validate_epoch(trainer.val_loader)
        self.assertGreater(first[1], 0)
        torch.manual_seed(999)
        repeated = trainer._validate_epoch(trainer.val_loader)
        self.assertEqual(first, repeated)
        reordered = DataLoader(trainer.val_loader.dataset, batch_size=3, shuffle=True,
                               generator=torch.Generator().manual_seed(8))
        np.testing.assert_allclose(trainer._validate_epoch(reordered), first, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(first[0], 0.7 * first[1] + 0.3 * first[2])
        with self.assertRaisesRegex(ValueError, "不属于保存的 val"):
            trainer._validate_epoch(trainer.test_loader)

    def test_fixed_masks_follow_sample_ids_and_never_select_padding(self):
        padding = torch.tensor([[False] * 3 + [True] * 5, [True] * 8])
        options = dict(patch_size=2, seed=73, split_fingerprint="split", mask_ratio=0.5)
        first = fixed_reconstruction_masks(padding, sample_ids=["A", "B"], **options)
        reverse = fixed_reconstruction_masks(padding, sample_ids=["B", "A"], **options)
        torch.testing.assert_close(first, reverse.flip(0), rtol=0, atol=0)
        self.assertFalse(first[:, 1].any())
        self.assertFalse(first[:, 0, 2:].any())
        masks = sample_patch_mask(torch.tensor([[True, True, False], [False, False, False]]), 0.5)
        self.assertFalse(masks[:, 2].any())

    def test_invalid_shapes_dimensions_flags_and_weights_fail_early(self):
        options = ({"recon_weight": -0.1}, {"pheno_weight": float("nan")},
                   {"recon_weight": 0, "pheno_weight": 0}, {"recon_weight": 0.2, "pheno_weight": 0.3},
                   {"use_pheno_help": False}, {"n_phenotypes": 2},
                   {"time_coordinates": [1, 2, 3], "n_time_points": 14}, {"val_mask_ratio": 0})
        for option in options:
            with self.subTest(option=option), self.assertRaises(ValueError):
                ConvAEConfig(**option)
        with self.assertRaisesRegex(ValueError, "n_time_points"):
            self.trainer(n_time_points=4, time_coordinates=None)
        model = self.model()
        x, padding = self.inputs()
        with self.assertRaisesRegex(ValueError, "输入"):
            model(x[:, :, :-1], padding_mask=padding)
        with self.assertRaisesRegex(ValueError, "结构性 padding"):
            model(x, padding_mask=padding, reconstruction_mask=torch.ones(2, 2, 3, dtype=torch.bool))

    def test_single_epoch_cpu_training_and_strict_checkpoint_roundtrip_in_both_modes(self):
        for enabled in (True, False):
            options = {} if enabled else dict(use_pheno_help=False, recon_weight=1, pheno_weight=0)
            trainer = self.trainer(**options)
            self.assertEqual(trainer.dataset.original_length, 13)
            self.assertTrue(trainer.dataset.chunk_padding_mask.any())
            self.assertTrue((trainer.dataset.snp[:, :13] == 0).any())
            with patch.object(trainer, "_compress_data", side_effect=AssertionError("禁止导出特征")):
                result = trainer.train()
            self.assertEqual(result["best_epoch"], 1)
            self.assertGreater(result["history"]["val_recon_loss"][0], 0)
            expected_metric = "val_total_loss" if enabled else "val_recon_loss"
            self.assertEqual(result["selection_metric"], expected_metric)
            if not enabled:
                self.assertEqual(result["history"]["train_pheno_loss"], [0])
                self.assertEqual(result["history"]["val_pheno_loss"], [0])
            loaded, info = load_convae_model(trainer.config.output_dir)
            self.assertEqual(info["metadata"]["selection_metric"], expected_metric)
            self.assertEqual(info["metadata"]["use_pheno_help"], enabled)
            x, _, ids = next(iter(trainer.val_loader))
            mask = trainer._validation_mask_for_ids(ids)
            with torch.inference_mode():
                old = trainer.model(x, padding_mask=trainer.dataset.chunk_padding_mask, reconstruction_mask=mask)
                new = loaded(x, padding_mask=trainer.dataset.chunk_padding_mask, reconstruction_mask=mask)
                for index in (0, 2):
                    torch.testing.assert_close(old[index], new[index], rtol=0, atol=0)
                if enabled:
                    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
            history = pd.read_csv(trainer.training_history_save_dir / "training_history.csv")
            self.assertIn("val_recon_loss", history)
            config = json.loads(Path(trainer.config.output_dir, "config.json").read_text())
            self.assertEqual(config["selection_metric"], expected_metric)

    def test_model_selection_uses_predeclared_metric_and_restores_best_weights(self):
        for enabled in (True, False):
            options = {} if enabled else dict(use_pheno_help=False, recon_weight=1, pheno_weight=0)
            trainer = self.trainer(epochs=2, **options)
            epoch = [0]

            def train(loader, optimizer):
                epoch[0] += 1
                with torch.no_grad():
                    trainer.model.feature_extractor.snp_embedding.weight.fill_(epoch[0])
                optimizer.step()
                trainer.last_train_stats = {"recon_count": 1}
                return (1, 1, 0)

            def validate(loader):
                self.assertIs(loader, trainer.val_loader)
                trainer.last_val_stats = {"recon_count": 1, "empty_batches": 0}
                recon, pheno = ((0.1, 4) if epoch[0] == 1 else (0.5, 0.1)) if enabled else (0.1 * epoch[0], 0)
                return trainer.recon_weight * recon + trainer.pheno_weight * pheno, recon, pheno

            with (patch.object(trainer, "_train_epoch", side_effect=train),
                  patch.object(trainer, "_validate_epoch", side_effect=validate)):
                result = trainer.train()
            best = 2 if enabled else 1
            self.assertEqual(result["best_epoch"], best)
            self.assertTrue((trainer.model.feature_extractor.snp_embedding.weight == best).all())

    def test_legacy_checkpoint_and_corrupted_new_weights_fail_explicitly(self):
        old = self.fixture.root / "legacy.pt"
        torch.save(self.model().state_dict(), old)
        with self.assertRaisesRegex(ValueError, "旧 ConvAE checkpoint"):
            load_convae_model(old)
        trainer = self.trainer()
        trainer.train()
        checkpoint = trainer.best_model_save_dir / "best_model.pt"
        masks = Path(trainer.config.output_dir) / "preprocessing/validation_masks.npz"
        content = masks.read_bytes()
        masks.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "验证掩码"):
            load_convae_model(trainer.config.output_dir)
        masks.write_bytes(content)
        package = torch.load(checkpoint, weights_only=True)
        key = "phenotype_head.3.weight"
        package["state_dict"][key] = torch.zeros(14, 128)
        torch.save(package, checkpoint)
        from utils.ArtifactIntegrity import seal_checkpoint
        seal_checkpoint(trainer.config.output_dir, extra_files=("config.json", "preprocessing/validation_masks.npz", "models/checkpoint_metadata.json"))
        with self.assertRaisesRegex(ValueError, "权重形状"):
            load_convae_model(trainer.config.output_dir)

    def test_cli_explicitly_disables_auxiliary_objective(self):
        import train_convae
        with (patch.object(sys, "argv", ["train_convae.py", "--use_pheno_help", "false", "--prepare_only"]),
              patch.object(train_convae, "ConvAETrainer") as factory, redirect_stdout(io.StringIO())):
            train_convae.train_one()
        config = factory.call_args.kwargs["config"]
        self.assertFalse(config.use_pheno_help)
        self.assertEqual((config.recon_weight, config.pheno_weight), (1, 0))
        self.assertEqual(config.selection_metric, "val_recon_loss")


if __name__ == "__main__":
    unittest.main()

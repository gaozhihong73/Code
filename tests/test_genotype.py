"""编码、筛选、缺失边界和配置的一致性测试；仅 CPU 合成数据。"""

import hashlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from config import SimTFPPConfig, TFPPConfig
from dataloader.ConvAE_DataLoader import ConvAEDataset
from dataloader.Sim_DataLoader import SimTFPPDataset, get_sim_dataloader
from dataloader.TFPP_DataLoader import TFPPDataset
from model import TFPP, ConvAE
from trainer import TFPPTrainer
from utils.Genotype import (
    GENOTYPE_ENCODING_VERSION,
    TrainingDosageImputer,
    dosage_to_model_codes,
    model_code_tensor,
    model_codes_to_dosage,
    prepare_screening_dosage,
    validate_dosage,
)
from utils.model_loader import save_model_config
from utils.Screening import (
    SCREENING_ALGORITHM_VERSION,
    compute_cache_key,
    dc_sis_screen,
    gwas_screen,
    screen_snps,
)
from utils.Split import generate_split


class GenotypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(73)
        self.dosage = rng.integers(0, 3, size=(40, 12), dtype=np.int8)
        self.pheno = (
            self.dosage[:, 0, None] * np.array([1, 2, 3]) + rng.normal(0, 0.2, (40, 3))
        ).astype(np.float32)[:, None, :]

    def write_inputs(self):
        snp_path = self.root / "snp.npy"
        np.save(snp_path, self.dosage)
        pd.DataFrame(self.pheno[:, 0]).to_csv(self.root / "toy.csv", index=False)
        paths = {}
        for group in ("train", "val", "test"):
            n = 40 if group == "train" else 5
            for kind, values in (
                ("snp", self.dosage[:n]),
                ("pheno", self.pheno[:n, 0]),
            ):
                path = self.root / f"{group}_{kind}.csv"
                df = pd.DataFrame(values)
                df.insert(0, "id", [f"{group}-{i}" for i in range(n)])
                df.to_csv(path, index=False)
                paths[f"{group}_{kind}_csv"] = str(path)
        return str(snp_path), paths

    def test_round_trip_and_identical_screening_scores_and_rankings(self):
        codes = dosage_to_model_codes(self.dosage)
        recovered = model_codes_to_dosage(codes)
        np.testing.assert_array_equal(recovered, self.dosage)
        for function in (gwas_screen, dc_sis_screen):
            with self.subTest(function=function.__name__):
                idx, scores = function(self.dosage, self.pheno[:, 0], 4, verbose=False)
                recovered_idx, recovered_scores = function(
                    recovered, self.pheno[:, 0], 4, verbose=False
                )
                np.testing.assert_array_equal(idx, recovered_idx)
                np.testing.assert_array_equal(scores, recovered_scores)

    def test_wrong_model_codes_are_rejected_by_every_screening_entry(self):
        codes = dosage_to_model_codes(self.dosage)
        for function in (gwas_screen, dc_sis_screen):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "screening dosage"):
                    function(codes, self.pheno[:, 0], 4, verbose=False)
        with self.assertRaises(ValueError):
            screen_snps(codes, self.pheno, screen_k=4, verbose=False)
        # 即使关闭筛选或命中旧缓存，也必须先拒绝非法编码。
        with patch("utils.Screening._load_cache", return_value=np.array([0])) as cache:
            with self.assertRaises(ValueError):
                screen_snps(
                    codes, self.pheno, screen_k=4, cache_key="invalid", verbose=False
                )
            cache.assert_not_called()
        with self.assertRaises(ValueError):
            screen_snps(codes, self.pheno)

    def test_missing_round_trip_never_becomes_real_zero(self):
        raw = np.array([[-1, np.nan, 0, 1, 2]], dtype=np.float32)
        codes = dosage_to_model_codes(raw)
        np.testing.assert_array_equal(codes, [[0, 0, 1, 2, 3]])
        recovered = model_codes_to_dosage(codes)
        np.testing.assert_array_equal(recovered, [[-1, -1, 0, 1, 2]])
        for strategy in ("error", "mode"):
            with self.assertRaisesRegex(ValueError, "禁止当作剂量0"):
                prepare_screening_dosage(recovered, missing_strategy=strategy)
        for function in (gwas_screen, dc_sis_screen):
            with self.assertRaises(ValueError):
                function(recovered, np.zeros((1, 3)), 1, verbose=False)
        with self.assertRaises(ValueError):
            screen_snps(recovered, np.zeros((1, 1, 3)), screen_k=1, verbose=False)

    def test_invalid_dosages_and_codes_fail_without_truncation(self):
        for value in (3, -2, 0.5, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    dosage_to_model_codes(np.array([[value]]))
        for values in ([[0.1, 1.9]], [[4, 0]], [[np.nan, 0]], [[True, False]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    model_code_tensor(torch.tensor(values))
        for values in ([[0.5]], [[4]], [[-1]]):
            with self.assertRaises(ValueError):
                model_codes_to_dosage(np.array(values))
        with self.assertRaises(ValueError):
            validate_dosage(np.array([0, 1, 2]))

    def test_mode_imputer_fits_only_explicit_training_individuals(self):
        raw = np.array([[2, 1], [2, 0], [-1, 1], [1, 0], [0, 2], [0, 2], [0, 2]])
        train_idx = np.array([0, 1, 2, 3])
        imputer = TrainingDosageImputer().fit(raw, train_idx=train_idx)
        np.testing.assert_array_equal(imputer.fill_values_, [2, 0])
        transformed = prepare_screening_dosage(
            raw, missing_strategy="mode", imputer=imputer
        )
        self.assertEqual(transformed[2, 0], 2)
        self.assertEqual(transformed[4, 0], 0)  # 观测到的剂量0仍是0。
        changed = raw.copy()
        changed[4:] = 1
        other = TrainingDosageImputer().fit(changed, train_idx=train_idx)
        self.assertEqual(imputer.metadata_, other.metadata_)
        self.assertEqual(imputer.metadata_["train_idx"], train_idx.tolist())
        self.assertEqual(imputer.metadata_["train_missing_counts"], [1, 0])
        self.assertEqual(json.loads(json.dumps(imputer.metadata_)), imputer.metadata_)

    def test_imputer_rejects_unfitted_all_missing_or_invalid_training_rows(self):
        with self.assertRaisesRegex(ValueError, "先.*train_idx.*fit"):
            TrainingDosageImputer().transform(self.dosage)
        raw = np.array([[-1, 1], [-1, 2], [0, 1]])
        with self.assertRaisesRegex(ValueError, "全缺失"):
            TrainingDosageImputer().fit(raw, train_idx=np.array([0, 1]))
        for indices in ([], [0, 0], [-1], [3], [0.5]):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    TrainingDosageImputer().fit(raw, train_idx=np.asarray(indices))
        with self.assertRaises(ValueError):
            TrainingDosageImputer("error").fit(raw, train_idx=np.array([0, 1]))
        fitted = TrainingDosageImputer().fit(self.dosage, train_idx=np.arange(20))
        with self.assertRaises(ValueError):
            prepare_screening_dosage(
                self.dosage, missing_strategy="error", imputer=fitted
            )

    def test_real_and_simulated_pipelines_screen_identical_dosage(self):
        snp_path, _ = self.write_inputs()
        dataset = TFPPDataset(snp_path, str(self.root), ["toy"])
        with redirect_stdout(io.StringIO()):
            real_indices = screen_snps(
                model_codes_to_dosage(dataset.snp), dataset.phenotype,
                screen_k=4, gwas_k=4, multivariate=True, verbose=False,
            )
        with self.assertWarns(DeprecationWarning):
            simulated = SimTFPPDataset(snp_path, str(self.root), ["toy"])
        np.testing.assert_array_equal(dataset.snp, simulated.snp)
        np.testing.assert_array_equal(
            model_codes_to_dosage(simulated.snp)[:, real_indices],
            self.dosage[:, real_indices],
        )
        split = generate_split(len(simulated))
        with self.assertWarns(DeprecationWarning):
            loaders = get_sim_dataloader(
                simulated, 4, "global", "zscore", num_workers=0, indices=split
            )
        for loader, indices in zip(loaders[:3], (
            split.train_idx, split.val_idx, split.test_idx
        )):
            np.testing.assert_array_equal(
                loader.dataset.tensors[0], simulated.snp[indices]
            )

    def test_real_and_convae_dataset_preserve_missing_category(self):
        snp_path, _ = self.write_inputs()
        raw = self.dosage.astype(np.float32)
        raw[0, :2] = [-1, np.nan]
        np.save(snp_path, raw)
        dataset = TFPPDataset(snp_path, str(self.root), ["toy"])
        np.testing.assert_array_equal(dataset.snp[0, :2], [0, 0])
        with redirect_stdout(io.StringIO()):
            mae = ConvAEDataset(
                snp_path, str(self.root), ["toy"], chunk_size=20, overlap=0
            )
        np.testing.assert_array_equal(mae.snp[:, :12], dataset.snp)
        np.testing.assert_array_equal(mae.snp[:, 12:], 0)

    def test_kfold_init_defers_fit_and_sim_uses_same_missing_category(self):
        snp_path, _ = self.write_inputs()
        raw = self.dosage.astype(np.float32)
        raw[0, 0] = -1
        np.save(snp_path, raw)
        real_module = importlib.import_module("trainer.TFPP_Trainer")
        with (
            patch.object(Path, "mkdir"),
            patch.object(real_module, "Logger", return_value=Mock()),
            patch("utils.screen_snps") as screen,
            patch.object(TrainingDosageImputer, "fit") as fit,
        ):
            trainer = TFPPTrainer(TFPPConfig(
                time_coordinates=[1, 2, 3],
                snp_file=snp_path, pheno_dir=str(self.root), phenotype_names=["toy"],
                screen_k=4, gwas_k=4, screen_missing_strategy="mode",
                use_kfold=True, device="cpu",
            ))
            self.assertIsNone(trainer.model)
            screen.assert_not_called()
            fit.assert_not_called()
        with self.assertWarns(DeprecationWarning):
            simulated = SimTFPPDataset(snp_path, str(self.root), ["toy"])
        self.assertEqual(simulated.snp[0, 0], 0)

    def test_cli_defaults_and_explicit_overrides_have_one_source(self):
        import infer
        import train

        expected = TFPPConfig.model_fields["screen_multivariate"].default
        self.assertIs(expected, True)
        self.assertEqual(SimTFPPConfig().screen_multivariate, expected)
        for module, argv in (
            (train, ["train.py"]),
            (infer, ["infer.py", "--model_path", "unused"]),
        ):
            with self.subTest(module=module.__name__):
                with patch.object(sys, "argv", argv):
                    self.assertEqual(module.parse_args().screen_multivariate, expected)
                for value in ("true", "false"):
                    with patch.object(
                        sys, "argv", argv + ["--screen_multivariate", value]
                    ):
                        self.assertEqual(
                            module.parse_args().screen_multivariate, value == "true"
                        )

    def test_saved_config_records_effective_flags_and_versions(self):
        for config_class in (TFPPConfig, SimTFPPConfig):
            for enabled in (True, False):
                with self.subTest(config=config_class.__name__, enabled=enabled):
                    config = config_class(
                        screen_multivariate=enabled,
                        screen_missing_strategy="mode",
                        device="cpu",
                    )
                    save_model_config(config, self.root)
                    saved = json.loads((self.root / "model_config.json").read_text())
                    self.assertEqual(saved["screen_multivariate"], enabled)
                    self.assertEqual(saved["screen_missing_strategy"], "mode")
                    self.assertEqual(
                        saved["genotype_encoding_version"], GENOTYPE_ENCODING_VERSION
                    )
                    self.assertEqual(
                        saved["screening_algorithm_version"],
                        SCREENING_ALGORITHM_VERSION,
                    )

    def test_cache_versions_and_missing_policy_invalidate_keys(self):
        kwargs = dict(
            snp_file="not-present.npy",
            pheno_source="not-present-pheno",
            phenotype_names=["toy"],
            screen_k=4,
            gwas_k=4,
            seed=73,
            multivariate=True,
        )
        legacy = hashlib.md5()
        legacy.update(b"not-present.npy")
        legacy.update(b"not-present-pheno")
        legacy.update(b"toy")
        legacy.update(b"4_4_73_")
        legacy.update(b"method=mv_gwas_sis_v2")
        new_key = compute_cache_key(**kwargs)
        self.assertNotEqual(new_key, legacy.hexdigest())
        self.assertNotEqual(
            new_key, compute_cache_key(**kwargs, missing_strategy="mode")
        )
        for field in ("GENOTYPE_ENCODING_VERSION", "SCREENING_ALGORITHM_VERSION"):
            with patch(f"utils.Screening.{field}", "different-version"):
                self.assertNotEqual(new_key, compute_cache_key(**kwargs))
        # 旧文件保留，但新缓存命名绝不加载它。
        np.save(self.root / f"{legacy.hexdigest()}.npy", [999])
        with redirect_stdout(io.StringIO()):
            indices = screen_snps(
                self.dosage,
                self.pheno,
                screen_k=4,
                gwas_k=4,
                multivariate=True,
                cache_key=new_key,
                cache_dir=str(self.root),
                verbose=False,
            )
        self.assertTrue((indices < self.dosage.shape[1]).all())
        self.assertTrue((self.root / f"{new_key}.npy").exists())

    def test_cpu_model_forward_uses_validated_categorical_codes(self):
        model = TFPP(
            snp_dim=12,
            d_model=8,
            num_heads=2,
            num_layers=1,
            d_ff=16,
            num_embeddings=4,
            time_coordinates=[1, 2, 3],
            dropout=0,
            phenotype_names=["toy"],
            degree=1,
            ablation_mode="all",
        ).eval()
        codes = model_code_tensor(dosage_to_model_codes(self.dosage[:2]))
        with torch.inference_mode():
            self.assertEqual(tuple(model(codes).shape), (2, 1, 3))
            with self.assertRaisesRegex(ValueError, "连续特征"):
                model(codes.float() / 2 + 0.1)
        mae = ConvAE(
            chunk_size=40, patch_size=20, d_model=8, d_ext=4, num_chunks=1
        ).eval()
        with torch.inference_mode():
            codes = model_code_tensor(
                dosage_to_model_codes(np.tile(self.dosage[:2], (1, 4))[:, :40])
            )
            self.assertEqual(
                tuple(mae(codes[:, None], mask_ratio=0)[0].shape), (2, 1, 4)
            )

    def test_inference_module_has_no_screening_or_fitting_loader(self):
        import infer

        self.assertFalse(hasattr(infer, "screen_snps"))
        self.assertFalse(hasattr(infer, "get_dataloader"))

    def test_external_inference_requires_explicit_input_stage(self):
        import infer

        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                infer.parse_args(["--model_path", "unused", "--input_snp", "raw.npy"])
        for stage in ("raw", "screened"):
            args = infer.parse_args(["--model_path", "unused", "--input_snp", "x.npy",
                                     "--input_stage", stage])
            self.assertEqual(args.input_stage, stage)


if __name__ == "__main__":
    unittest.main()

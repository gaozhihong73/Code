"""固定 split 的传统模型与机器学习比较最小测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trainer.Baseline_Trainer import (
    REPORT_COLUMNS,
    SCENARIO_LABEL,
    BaselineConfig,
    BaselineTrainer,
)
from utils.Split import generate_split, save_split


class BaselineComparisonTests(unittest.TestCase):
    def test_models_share_raw_snps_split_and_report_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "cleaned_pheno").mkdir(parents=True)
            (data / "splits").mkdir()
            (data / "truth").mkdir()

            rng = np.random.default_rng(73)
            genotype = rng.integers(0, 3, size=(60, 12), dtype=np.int8)
            effects = np.asarray(
                [[0.4, 0.2, 0.1], [-0.2, 0.3, 0.5], [0.1, -0.1, 0.2]]
            )
            phenotype = (
                genotype[:, :3] @ effects
                + rng.normal(0, 0.1, size=(len(genotype), effects.shape[1]))
            ).astype(np.float32)
            sample_ids = [f"ID{i + 1}" for i in range(len(genotype))]
            np.save(data / "genotype.npy", genotype)
            pd.DataFrame(phenotype, columns=["T1", "T2", "T3"]).to_csv(
                data / "cleaned_pheno/sim_pheno.csv", index=False
            )
            pd.DataFrame({"SampleID": sample_ids}).to_csv(
                data / "sample_ids.csv", index=False
            )
            pd.DataFrame(
                {
                    "SampleID": sample_ids,
                    **{f"T{i + 1}": phenotype[:, i] for i in range(3)},
                }
            ).to_csv(data / "truth/true_gv.csv", index=False)
            (data / "data_manifest.json").write_text("{}", encoding="utf-8")

            split_path = data / "splits/split_seed73.npz"
            split = generate_split(sample_ids=sample_ids, seed=73)
            save_split(split, split_path)

            output = root / "output"
            models = ("Ridge-All", "Ridge-A+D", "GBLUP-All", "RF", "XGBoost")
            config = BaselineConfig(
                data_dir=str(data),
                output_dir=str(output),
                alphas=(0.1, 1.0),
                models=models,
                rf_n_estimators=4,
                rf_n_jobs=1,
                xgb_n_estimators=4,
                xgb_early_stopping_rounds=2,
                xgb_n_jobs=1,
            )
            result = BaselineTrainer(config).run()

            self.assertEqual(tuple(result.columns), REPORT_COLUMNS)
            self.assertEqual(set(result["model"]), {"Train-Mean", *models})
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["scenario"], SCENARIO_LABEL)
            self.assertIn("不代表所有遗传结构", protocol["scope_statement"])
            for name in ("Train-Mean", *models):
                with np.load(output / name / "test_predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["sample_idx"], split.test_idx)
                    np.testing.assert_array_equal(
                        saved["sample_ids"], np.asarray(sample_ids)[split.test_idx]
                    )
                details = json.loads((output / name / "metrics.json").read_text())
                self.assertEqual(
                    details["split_sha256"], protocol["split_sha256"]
                )
                if name != "Train-Mean":
                    self.assertEqual(details["raw_snp_count"], genotype.shape[1])


if __name__ == "__main__":
    unittest.main()

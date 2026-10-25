"""完整模拟数据到生产 TFPP 数据管线的回归测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataloader import TFPPDataset, get_dataloader
from prepare_simulation import prepare_simulation
from utils.Split import load_split


class SimulationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.ids = [f"ID{i:03d}" for i in range(20)]
        rng = np.random.default_rng(73)
        self.genotype = rng.integers(0, 3, (20, 12), dtype=np.int8)
        self.phenotype = rng.normal(size=(20, 4)).astype(np.float32)
        self._write_inputs()

    def _write_inputs(self, *, ids=None, genotype=None, phenotype=None):
        ids = self.ids if ids is None else ids
        genotype = self.genotype if genotype is None else genotype
        phenotype = self.phenotype if phenotype is None else phenotype
        pd.DataFrame({"SampleID": ids}).to_csv(self.input / "sample_ids.csv", index=False)
        pd.DataFrame(genotype, columns=[f"SNP{i}" for i in range(genotype.shape[1])]).assign(
            SampleID=ids
        )[["SampleID", *[f"SNP{i}" for i in range(genotype.shape[1])]]].to_csv(
            self.input / "genotype.csv", index=False
        )
        pd.DataFrame(phenotype, columns=[f"T{i + 1}" for i in range(phenotype.shape[1])]).assign(
            SampleID=ids
        )[["SampleID", *[f"T{i + 1}" for i in range(phenotype.shape[1])]]].to_csv(
            self.input / "phenotype.csv", index=False
        )
        pd.DataFrame({"SampleID": ids, "T1": np.arange(len(ids))}).to_csv(
            self.input / "true_gv.csv", index=False
        )
        pd.DataFrame({"snp_index": [1], "effect": [0.5]}).to_csv(
            self.input / "causal_effects.csv", index=False
        )
        (self.input / "simulation_config.json").write_text(json.dumps({
            "time_coordinates": [1, 2, 3, 4]
        }))

    def _prepare(self, name="output"):
        output = self.root / name
        manifest = prepare_simulation(self.input, output, seed=73)
        return output, manifest

    def test_complete_data_converts_and_preserves_raw_dosage(self):
        output, manifest = self._prepare()
        saved = np.load(output / "genotype.npy", allow_pickle=False)
        np.testing.assert_array_equal(saved, self.genotype)
        self.assertEqual(saved.dtype, np.int8)
        self.assertEqual(set(np.unique(saved)), {0, 1, 2})
        self.assertEqual(manifest["genotype_shape"], [20, 12])

    def test_tfpp_dataset_encodes_categories_and_pheno_has_no_id(self):
        output, _ = self._prepare()
        dataset = TFPPDataset(
            str(output / "genotype.npy"), str(output / "cleaned_pheno"), ["sim_pheno"],
            sample_ids=self.ids,
        )
        self.assertEqual(set(np.unique(dataset.snp)), {1, 2, 3})
        frame = pd.read_csv(output / "cleaned_pheno" / "sim_pheno.csv")
        self.assertNotIn("SampleID", frame.columns)
        self.assertEqual(frame.shape, (20, 4))

    def test_row_mismatch_duplicate_ids_and_invalid_dosage_are_rejected(self):
        cases = []
        duplicate = self.ids[:-1] + [self.ids[0]]
        cases.append(("duplicate", duplicate, self.genotype, self.phenotype))
        invalid = self.genotype.copy()
        invalid[0, 0] = 4
        cases.append(("invalid dosage", self.ids, invalid, self.phenotype))
        for index, (label, ids, genotype, phenotype) in enumerate(cases):
            with self.subTest(label=label):
                self._write_inputs(ids=ids, genotype=genotype, phenotype=phenotype)
                with self.assertRaises(ValueError):
                    prepare_simulation(self.input, self.root / f"bad-{index}")

    def test_independent_file_row_mismatch_is_rejected(self):
        frame = pd.read_csv(self.input / "phenotype.csv").iloc[:-1]
        frame.to_csv(self.input / "phenotype.csv", index=False)
        with self.assertRaisesRegex(ValueError, "行数"):
            prepare_simulation(self.input, self.root / "bad")

    def test_split_is_disjoint_complete_and_reproducible(self):
        first, _ = self._prepare("first")
        second, _ = self._prepare("second")
        splits = [load_split(path / "splits/split_seed73.npz", sample_ids=self.ids)
                  for path in (first, second)]
        for field in ("train_idx", "val_idx", "test_idx"):
            np.testing.assert_array_equal(getattr(splits[0], field), getattr(splits[1], field))
        sets = [set(x.tolist()) for x in (
            splits[0].train_idx, splits[0].val_idx, splits[0].test_idx
        )]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(20)))

    def test_existing_output_is_not_overwritten(self):
        output, _ = self._prepare()
        before = (output / "data_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_simulation(self.input, output)
        self.assertEqual(before, (output / "data_manifest.json").read_bytes())

    def test_truth_is_separate_and_three_batches_load(self):
        output, manifest = self._prepare()
        dataset = TFPPDataset(
            str(output / "genotype.npy"), str(output / "cleaned_pheno"), ["sim_pheno"],
            sample_ids=self.ids,
        )
        split = load_split(output / "splits/split_seed73.npz", sample_ids=self.ids)
        loaders = get_dataloader(
            dataset, 4, "global", "zscore", indices=split, num_workers=0
        )
        for loader in loaders[:3]:
            snp, phenotype = next(iter(loader))
            self.assertEqual(snp.ndim, 2)
            self.assertEqual(phenotype.shape[1:], (1, 4))
            self.assertTrue(np.isfinite(phenotype.numpy()).all())
        self.assertTrue((output / "truth/true_gv.csv").is_file())
        self.assertTrue((output / "truth/causal_effects.csv").is_file())
        self.assertFalse(manifest["truth_is_model_input"])
        self.assertEqual(manifest["model_inputs"], [
            "genotype.npy", "cleaned_pheno/sim_pheno.csv"
        ])

    def test_r_supported_cli_does_not_export_split_files(self):
        script = (Path(__file__).parents[1] / "r/generate_simulation.R").read_text()
        for name in (
            "Train_Genotype.csv", "Val_Genotype.csv", "Test_Genotype.csv",
            "Train_Phenotype.csv", "Val_Phenotype.csv", "Test_Phenotype.csv",
        ):
            self.assertNotIn(name, script)


if __name__ == "__main__":
    unittest.main()

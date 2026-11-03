import unittest

import numpy as np

from experiments.analysis.build_supplementary_tables import (
    _ccc,
    _inflection,
    _metric_rows,
    _shape_error,
)


class SupplementaryAnalysisTests(unittest.TestCase):
    def test_identical_curves_have_perfect_agreement(self):
        values = np.linspace(0.0, 1.0, 20)
        self.assertAlmostEqual(_ccc(values, values), 1.0)
        self.assertAlmostEqual(_shape_error(values, values), 0.0)

    def test_constant_curve_has_undefined_shape_metrics(self):
        constant = np.ones(20)
        self.assertIsNone(_ccc(constant, constant))
        self.assertIsNone(_shape_error(constant, constant))

    def test_logistic_inflection_and_fixed_shift(self):
        times = np.arange(1.0, 21.0)
        values = 1.0 / (1.0 + np.exp(-(times - 10.2)))
        shifted = 1.0 / (1.0 + np.exp(-(times - 11.2)))
        inflection, rate, status = _inflection(values, times, scale=1.0)
        shifted_inflection, _, shifted_status = _inflection(
            shifted, times, scale=1.0
        )
        self.assertEqual(status, "valid")
        self.assertEqual(shifted_status, "valid")
        self.assertAlmostEqual(inflection, 10.2, delta=0.25)
        self.assertAlmostEqual(shifted_inflection - inflection, 1.0, delta=0.05)
        self.assertGreater(rate, 0.0)

    def test_constant_predictions_keep_pcc_as_na(self):
        prediction = np.ones((5, 2))
        target = np.arange(10, dtype=np.float64).reshape(5, 2)
        overall, by_time = _metric_rows(
            {"run_id": "test"},
            "observed",
            prediction,
            target,
            scale=9.0,
            times=np.asarray([1.0, 2.0]),
        )
        pcc = next(row for row in overall if row["metric"] == "pcc")
        self.assertIsNone(pcc["value"])
        self.assertEqual(pcc["valid_n"], 0)
        self.assertTrue(
            all(row["value"] is None for row in by_time if row["metric"] == "pcc")
        )


if __name__ == "__main__":
    unittest.main()

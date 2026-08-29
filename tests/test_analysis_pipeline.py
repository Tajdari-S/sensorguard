import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from evaluation import (  # noqa: E402
    assert_group_disjoint,
    first_alert_index,
    hours_required_for_zero_event_bound,
    poisson_zero_event_upper,
    run_alert,
    validate_labels,
    wilson_interval,
)
from features import SIGNALS, extract_run, feature_names, stage2_names  # noqa: E402


class AnalysisPipelineTest(unittest.TestCase):
    def test_feature_contract(self):
        names = feature_names()
        self.assertEqual(len(names), 166)
        reduced = stage2_names(names)
        self.assertEqual(len(reduced), 131)
        self.assertNotIn("power_w__mean", reduced)
        self.assertIn("power_w__std", reduced)

    def test_extract_run_is_causal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvml.csv"
            rows = []
            for sample in range(60):
                row = {
                    "status": "ok",
                    "gpu_index": 0,
                    "t_target_raw_s": sample,
                }
                row.update({signal: float(sample + 1) for signal in SIGNALS})
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            before = extract_run(path, 0, window_s=30, stride_s=15)
            changed = pd.DataFrame(rows)
            changed.loc[changed.index >= 30, SIGNALS] = 1e9
            changed.to_csv(path, index=False)
            after = extract_run(path, 0, window_s=30, stride_s=15)
            pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])

    def test_alert_rule_uses_consecutive_windows(self):
        self.assertTrue(run_alert(np.array([0.8, 0.8, 0.1, 0.8, 0.1])))
        self.assertFalse(run_alert(np.array([0.8, 0.1, 0.8, 0.1, 0.1])))
        self.assertFalse(run_alert(np.array([])))

    def test_first_alert_index(self):
        self.assertEqual(first_alert_index(np.array([0.8, 0.8, 0.1, 0.8, 0.1])), 3)
        self.assertEqual(first_alert_index(np.array([0.8, 0.8, 0.8])), 2)
        self.assertIsNone(first_alert_index(np.array([0.8, 0.1, 0.8, 0.1, 0.1])))
        self.assertIsNone(first_alert_index(np.array([])))

    def test_group_audit(self):
        groups = np.array(["a", "a", "b", "b"])
        assert_group_disjoint([0, 1], [2, 3], groups)
        with self.assertRaises(AssertionError):
            assert_group_disjoint([0, 2], [1, 3], groups)

    def test_missing_publication_group_fails_closed(self):
        labels = pd.DataFrame({
            "run_id": ["r1"], "trace_path": ["x.csv"], "gpu_index": [0],
            "label": ["training"], "family": ["train"],
            "gpu_uuid": [None], "collection_day": ["2026-08-25"],
        })
        with self.assertRaises(ValueError):
            validate_labels(labels, "gpu_uuid")
        validate_labels(labels, "run_id")

    def test_uncertainty_helpers(self):
        low, high = wilson_interval(22, 23)
        self.assertAlmostEqual(low, 0.7900884493)
        self.assertAlmostEqual(high, 0.9922833339)
        self.assertAlmostEqual(poisson_zero_event_upper(11.4533), 0.26156, places=4)
        self.assertAlmostEqual(hours_required_for_zero_event_bound(1 / 24), 71.8976, places=3)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from evasion_transfer import (  # noqa: E402
    normalize_labels,
    plan_comparisons,
    plan_declared_holdout,
)


def sample_labels():
    rows = []
    for run_id, label, evasion, split in [
        ("ordinary", "training", "", "fit"),
        ("infer-fit", "inference", "", "fit"),
        ("hpc-fit", "non_ml", "", "fit"),
        ("infer-test", "inference", "", "control_test"),
        ("hpc-test", "non_ml", "", "control_test"),
        ("shape-a", "training", "timing_shaping", "fit"),
        ("interleave-a", "training", "interleaving", "fit"),
        ("memory-a", "training", "memory_minimization", "fit"),
        ("sealed-a", "training", "fused_update_kernel", "fit"),
    ]:
        rows.append({
            "run_id": run_id,
            "trace_path": f"{run_id}.csv",
            "gpu_index": 0,
            "label": label,
            "family": evasion or label,
            "evasion_family": evasion,
            "split": split,
            "gpu_uuid": "gpu-a",
            "collection_day": "2026-08-29",
        })
    return pd.DataFrame(rows)


class EvasionTransferPlanTest(unittest.TestCase):
    def test_pairwise_and_leave_one_out_plans_are_disjoint(self):
        runs = normalize_labels(sample_labels())
        plans = plan_comparisons(runs, "fused_update_kernel")
        self.assertEqual(len(plans), 12)
        for plan in plans:
            self.assertFalse(set(plan["fit_run_ids"]) & set(plan["test_run_ids"]))
            self.assertNotIn("sealed-a", plan["fit_run_ids"])
            self.assertNotIn("sealed-a", plan["test_run_ids"])

    def test_target_family_is_never_in_fit(self):
        runs = normalize_labels(sample_labels())
        for plan in plan_comparisons(runs, "fused_update_kernel"):
            target_ids = set(runs.loc[
                runs["evasion_family"] == plan["target_family"], "run_id"
            ])
            self.assertFalse(target_ids & set(plan["fit_run_ids"]))
            self.assertTrue(target_ids <= set(plan["test_run_ids"]))

    def test_rejects_training_in_control_test(self):
        runs = sample_labels()
        runs.loc[runs["run_id"] == "ordinary", "split"] = "control_test"
        with self.assertRaises(ValueError):
            normalize_labels(runs)

    def test_declared_two_seen_one_unseen(self):
        runs = normalize_labels(sample_labels())
        plans = plan_declared_holdout(
            runs, ["timing_shaping", "interleaving"],
            "memory_minimization", "fused_update_kernel",
        )
        self.assertEqual(len(plans), 3)
        for plan in plans:
            self.assertEqual(plan["target_family"], "memory_minimization")
            self.assertIn("memory-a", plan["test_run_ids"])
            self.assertNotIn("memory-a", plan["fit_run_ids"])
            self.assertNotIn("sealed-a", plan["fit_run_ids"])

    def test_declared_plan_cannot_open_final_family(self):
        runs = normalize_labels(sample_labels())
        with self.assertRaises(ValueError):
            plan_declared_holdout(
                runs, ["timing_shaping", "interleaving"],
                "fused_update_kernel", "fused_update_kernel",
            )


if __name__ == "__main__":
    unittest.main()

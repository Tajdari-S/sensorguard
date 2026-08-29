import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.analysis.summarize_negative_exposure import audit_manifest, overlap_pairs


class NegativeExposureSummaryTest(unittest.TestCase):
    def write_manifest(self, root: Path, run_id: str, *, status="completed", duration=1800,
                       health="pass", start="2026-08-29T19:00:00+00:00",
                       end="2026-08-29T19:31:00+00:00") -> Path:
        path = root / run_id / "manifest.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump({
            "run_id": run_id,
            "start_utc": start,
            "end_utc": end,
            "status": status,
            "workload": {"duration_s": duration},
            "sensor_channels": [{"channel_id": "nvml.gpu0", "health": health}],
        }))
        return path

    def test_completed_healthy_duration_is_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_manifest(Path(tmp), "20260829_neg-gemm_node2-gpu0_s0_n000")
            row = audit_manifest(path)
            self.assertTrue(row["eligible"])
            self.assertEqual(row["host"], "node2")
            self.assertEqual(row["gpu"], 0)

    def test_failed_or_unhealthy_run_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_manifest(
                Path(tmp), "20260829_neg-gemm_node1-gpu0_s0_n000",
                status="flagged_channel_health", health="fail",
            )
            row = audit_manifest(path)
            self.assertFalse(row["eligible"])
            self.assertIn("flagged_channel_health", row["exclusion_reason"])
            self.assertIn("unhealthy_channels", row["exclusion_reason"])

    def test_same_gpu_overlap_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = audit_manifest(self.write_manifest(
                root, "20260829_neg-gemm_node2-gpu0_s0_n000",
                start="2026-08-29T19:00:00+00:00", end="2026-08-29T19:31:00+00:00"))
            second = audit_manifest(self.write_manifest(
                root, "20260829_neg-fft_node2-gpu0_s0_n001",
                start="2026-08-29T19:30:00+00:00", end="2026-08-29T20:01:00+00:00"))
            self.assertEqual(len(overlap_pairs([first, second])), 1)


if __name__ == "__main__":
    unittest.main()

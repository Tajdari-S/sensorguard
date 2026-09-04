import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts" / "experiments" / "run_synchronized_physical.py"
    spec = importlib.util.spec_from_file_location("run_synchronized_physical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StrongerPaperPlanTest(unittest.TestCase):
    def build(self, output: Path, repetitions: int = 2) -> tuple[dict, dict]:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "experiments" / "build_stronger_paper_plan.py"),
                "--output-dir", str(output),
                "--gpu-uuid", "GPU-test-uuid",
                "--development-repetitions", str(repetitions),
                "--sealed-repetitions", str(repetitions),
                "--start-epoch-s", "2000000000",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        return (
            json.loads((output / "development_plan.json").read_text()),
            json.loads((output / "sealed_lora_plan.json").read_text()),
        )

    def test_development_and_sealed_families_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            development, sealed = self.build(Path(directory))
        development_families = {run["family"] for run in development["runs"]}
        sealed_families = {run["family"] for run in sealed["runs"]}
        self.assertEqual(development_families & sealed_families, {"inference_control"})
        self.assertNotIn("lora_dilution", development_families)
        self.assertTrue(sealed["requires_frozen_detector_manifest"])
        self.assertFalse(development["requires_frozen_detector_manifest"])

    def test_repetition_counts_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            development, sealed = self.build(Path(directory), repetitions=3)
        self.assertEqual(len(development["runs"]), 15)
        self.assertEqual(len(sealed["runs"]), 6)
        for run in development["runs"] + sealed["runs"]:
            expected = 0 if run["mode"] == "inference_control" else 1
            self.assertEqual(run["target"], expected)

    def test_runner_forwards_redteam_parameters(self):
        runner = load_runner()
        command = runner.node_command(
            {
                "kind": "redteam",
                "mode": "lora_dilution",
                "duration_s": 300,
                "start_epoch_s": 2000000000,
                "seed": 7,
                "cuda_device": "cuda:0",
                "dilution": 20,
                "lora_rank": 8,
            },
            "/python",
            Path("out.json"),
        )
        self.assertIn("physical_redteam_workload.py", command[1])
        self.assertEqual(command[command.index("--dilution") + 1], "20")
        self.assertEqual(command[command.index("--lora-rank") + 1], "8")

    def test_sealed_runner_requires_complete_frozen_manifest(self):
        runner = load_runner()
        sealed_plan = {"requires_frozen_detector_manifest": True}
        with self.assertRaises(RuntimeError):
            runner.validate_frozen_manifest(sealed_plan, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"status": "frozen"}))
            with self.assertRaises(RuntimeError):
                runner.validate_frozen_manifest(sealed_plan, path)
            path.write_text(json.dumps({
                "status": "frozen",
                "feature_contract": "fixed-Hz-v1",
                "threshold": 0.85,
                "run_rule": "3-of-5",
                "fit_run_ids": ["dev-1"],
            }))
            digest = runner.validate_frozen_manifest(sealed_plan, path)
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()

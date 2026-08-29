import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "roofline"))

from benchmark_kernels import case_metadata
from compare_cross_gpu_roofline import build_table
from materialize_application_pairs import materialize
from parse_application_ncu import summarize
from parse_ncu import dominant_kernel_bytes, kernel_matches_case, parse_launches, scaled_number
from run_application_roofline import plan


class RooflineToolsTest(unittest.TestCase):
    def test_gemm_metadata(self):
        row = case_metadata("gemm_1024", 2)
        self.assertEqual(row["flops"], 2 * 1024 ** 3)
        self.assertEqual(row["minimum_bytes"], 2 * 3 * 1024 ** 2)
        self.assertGreater(row["arithmetic_intensity_min"], 300)

    def test_copy_has_zero_intensity(self):
        self.assertEqual(case_metadata("copy", 2)["arithmetic_intensity_min"], 0)

    def test_ncu_metric_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
                # init kernel launch: must not win over the benchmark kernel
                writer.writerow(["0", "fill_kernel", "dram__bytes_read.sum", "Mbyte", "1"])
                writer.writerow(["0", "fill_kernel", "dram__bytes_write.sum", "Mbyte", "1"])
                for launch in (1, 2, 3):
                    writer.writerow([str(launch), "sgemm_kernel", "dram__bytes_read.sum", "Mbyte", "100"])
                    writer.writerow([str(launch), "sgemm_kernel", "dram__bytes_write.sum", "Mbyte", "25"])
            launches = parse_launches(path)
            self.assertEqual(len(launches), 4)
            kernel, dram_bytes = dominant_kernel_bytes(launches)
            self.assertEqual(kernel, "sgemm_kernel")
            self.assertEqual(dram_bytes, 125e6)

    def test_unit_scaling(self):
        self.assertEqual(scaled_number("1.5", "Gbyte"), 1.5e9)

    def test_expected_kernel_validation(self):
        self.assertTrue(kernel_matches_case("gemm_1024", "ampere_sgemm_128x64_nn"))
        self.assertTrue(kernel_matches_case("gemv", "cublasGemvKernel"))
        self.assertFalse(kernel_matches_case("copy", "normal_and_transform_kernel"))
        self.assertFalse(kernel_matches_case("gemm_1024", "normal_and_transform_kernel"))

    def test_svg_plot_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            output = Path(directory) / "roofline.svg"
            summary.write_text(json.dumps({"points": [{
                "case": "gemm_1024", "arithmetic_intensity_measured": 100.0,
                "arithmetic_intensity_min": 120.0, "tflops": 20.0
            }]}))
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "roofline" / "plot_roofline.py"),
                str(summary), "--peak-tflops", "35.58", "--peak-gbps", "936.2",
                "--output", str(output)
            ], check=True, capture_output=True, text=True)
            self.assertIn("RTX 3090 measured microbenchmark roofline", output.read_text())

    def test_application_ncu_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "application.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
                writer.writerow(["0", "kernel_a", "dram__bytes_read.sum", "Mbyte", "2"])
                writer.writerow(["0", "kernel_a", "dram__bytes_write.sum", "Mbyte", "1"])
                writer.writerow(["1", "kernel_b", "dram__bytes_read.sum", "Mbyte", "1"])
                writer.writerow(["1", "kernel_b", "dram__bytes_write.sum", "Mbyte", "0"])
            timing = {
                "case_id": "bridge", "suite": "cross_gpu_bridge", "platform": "rtx",
                "repetition": 1, "mode": "gpt2_train", "dtype": "float16",
                "batch_size": 2, "seq_len": 256, "decode_tokens": 1, "gap_ms": 0,
                "iterations": 1, "total_flops": 8e6, "active_tflops": 2.0,
                "wall_tflops": 1.0, "gpu_name": "test",
            }
            point = summarize(timing, path, peak_tflops=10.0, peak_gbps=1000.0)
            self.assertEqual(point["measured_dram_bytes"], 4e6)
            self.assertEqual(point["arithmetic_intensity"], 2.0)
            self.assertAlmostEqual(point["normalized_arithmetic_intensity"], 0.2)
            self.assertAlmostEqual(point["normalized_wall_throughput"], 0.1)

    def test_application_plan_is_dry_run_safe(self):
        matrix = {"repetitions": 3, "cases": [{
            "case_id": "x", "suite": "cross_gpu_bridge", "mode": "gpt2_prefill",
            "dtype": "float16", "batch_size": 2, "seq_len": 256,
            "decode_tokens": 1, "gap_ms": 0, "iterations": 2,
        }]}
        args = SimpleNamespace(
            repetitions=1, suite="cross_gpu_bridge", output_root=Path("out"),
            python="python3", platform="rtx3090", gpu_index=0, warmup=2,
            seed=1, ncu="ncu", cuda_device="cuda:0", physical_gpu_uuid="",
        )
        items = plan(args, matrix)
        self.assertEqual(len(items), 1)
        self.assertIn("--profile-from-start", items[0]["ncu_command"])
        self.assertIn("--profile-range", items[0]["ncu_command"])
        self.assertIn("--skip-flop-profiler", items[0]["ncu_command"])
        self.assertNotIn("--profile-range", items[0]["timing_command"])
        self.assertNotIn("--skip-flop-profiler", items[0]["timing_command"])

    def test_application_plan_records_uuid_pinning(self):
        matrix = {"repetitions": 1, "cases": [{
            "case_id": "x", "suite": "cross_gpu_bridge", "mode": "gpt2_prefill",
            "dtype": "float16", "batch_size": 2, "seq_len": 256,
            "decode_tokens": 1, "gap_ms": 0, "iterations": 2,
        }]}
        args = SimpleNamespace(
            repetitions=1, suite="cross_gpu_bridge", output_root=Path("out"),
            python="python3", platform="node1-gpu1", gpu_index=1, warmup=2,
            seed=1, ncu="ncu", cuda_device="cuda:0",
            physical_gpu_uuid="GPU-immutable",
        )
        command = plan(args, matrix)[0]["timing_command"]
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(
            command[command.index("--physical-gpu-uuid") + 1], "GPU-immutable"
        )

    def test_cross_gpu_bridge_requires_two_platforms(self):
        base = {
            "case_id": "bridge", "suite": "cross_gpu_bridge", "repetition": 1,
            "mode": "gpt2_train", "gap_ms": 0, "arithmetic_intensity": 10.0,
            "wall_tflops": 2.0, "peak_tflops": 20.0, "peak_gbps": 1000.0,
            "normalized_arithmetic_intensity": 0.5,
            "normalized_wall_throughput": 0.1,
        }
        with self.assertRaises(ValueError):
            build_table([pd.DataFrame([{**base, "platform": "rtx"}])])
        rows = []
        for platform in ["rtx", "h200"]:
            for repetition in range(1, 4):
                rows.append({
                    **base, "platform": platform, "repetition": repetition,
                    "wall_tflops": 2.0 if platform == "rtx" else 5.0,
                })
        table = build_table([pd.DataFrame(rows)])
        self.assertEqual(set(table["platform"]), {"rtx", "h200"})

    def test_cross_gpu_bridge_requires_three_repetitions(self):
        rows = []
        for platform in ["rtx", "h200"]:
            for repetition in range(1, 3):
                rows.append({
                    "case_id": "bridge", "suite": "cross_gpu_bridge",
                    "repetition": repetition, "platform": platform,
                    "mode": "gpt2_train", "gap_ms": 0,
                    "arithmetic_intensity": 10.0, "wall_tflops": 2.0,
                    "peak_tflops": 20.0, "peak_gbps": 1000.0,
                    "normalized_arithmetic_intensity": 0.5,
                    "normalized_wall_throughput": 0.1,
                })
        with self.assertRaises(ValueError):
            build_table([pd.DataFrame(rows)])

    def test_application_pairs_are_declared_not_posthoc(self):
        rows = []
        for case_id, mode in [("train", "gpt2_train"), ("infer", "gpt2_prefill")]:
            for repetition in range(1, 4):
                rows.append({
                    "case_id": case_id, "platform": "rtx", "suite": "app",
                    "repetition": repetition, "mode": mode,
                    "arithmetic_intensity": repetition,
                    "wall_tflops": 2 * repetition,
                })
        pairs = pd.DataFrame([{
            "pair_id": "fixed-pair", "training_case": "train",
            "inference_case": "infer", "model_family": "gpt2",
            "freeze_note": "fixed before collection",
        }])
        result = materialize(pd.DataFrame(rows), pairs)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["role"]), {"training", "inference"})
        self.assertEqual(set(result["arithmetic_intensity"]), {2.0})

    def test_application_pairs_require_three_repetitions(self):
        points = pd.DataFrame([{
            "case_id": case_id, "platform": "rtx", "suite": "app",
            "repetition": 1, "mode": mode, "arithmetic_intensity": 1.0,
            "wall_tflops": 2.0,
        } for case_id, mode in [("train", "gpt2_train"), ("infer", "gpt2_prefill")]])
        pairs = pd.DataFrame([{
            "pair_id": "fixed-pair", "training_case": "train",
            "inference_case": "infer", "model_family": "gpt2",
            "freeze_note": "fixed before collection",
        }])
        with self.assertRaises(ValueError):
            materialize(points, pairs)


if __name__ == "__main__":
    unittest.main()

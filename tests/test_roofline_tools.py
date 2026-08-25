import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "roofline"))

from benchmark_kernels import case_metadata
from parse_ncu import dominant_kernel_bytes, parse_launches, scaled_number


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
            self.assertIn("RTX 3090 empirical roofline", output.read_text())


if __name__ == "__main__":
    unittest.main()

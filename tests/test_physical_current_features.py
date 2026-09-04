import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from physical_current_features import (  # noqa: E402
    calibrate_rescue_threshold,
    fixed_frequency_features,
    optimizer_event_features,
    resample_antialiased,
    rescue_alert,
)


class PhysicalCurrentFeaturesTest(unittest.TestCase):
    def test_same_rate_does_not_change_trace(self):
        values = np.linspace(-1, 1, 100)
        np.testing.assert_array_equal(resample_antialiased(values, 10_000, 10_000), values)

    def test_decimation_filters_tone_above_new_nyquist(self):
        time_s = np.arange(20_000) / 10_000
        high_tone = np.sin(2 * np.pi * 3_000 * time_s)
        filtered = resample_antialiased(high_tone, 10_000, 1_000)
        # A stride-only decimator aliases this tone at full amplitude.
        self.assertLess(float(np.sqrt(np.mean(filtered[50:-50] ** 2))), 0.05)

    def test_bands_are_in_physical_hertz(self):
        sample_hz = 10_000
        time_s = np.arange(sample_hz) / sample_hz
        features = fixed_frequency_features(np.sin(2 * np.pi * 120 * time_s), sample_hz)
        self.assertGreater(features["bandpower_100_200_hz"], 0.99)
        self.assertLess(features["bandpower_500_1000_hz"], 1e-6)

    def test_optimizer_dip_is_detected(self):
        values = np.ones(10_000)
        values[4_900:5_100] = 0.1
        features = optimizer_event_features(values, 10_000)
        self.assertGreater(features["optimizer_deep_dip"], 0.8)
        self.assertAlmostEqual(features["optimizer_dip_time_s"], 0.5, delta=0.03)

    def test_rescue_threshold_adds_no_calibration_alert(self):
        dips = np.array([0.1, 0.2, 0.3, 0.9])
        low_frequency = np.array([0.8, 0.7, 0.6, 0.1])
        threshold = calibrate_rescue_threshold(dips, low_frequency)
        for dip, fraction in zip(dips, low_frequency):
            self.assertFalse(rescue_alert(False, dip, fraction, threshold))
        self.assertTrue(rescue_alert(False, 0.4, 0.9, threshold))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Rate-aware features and optimizer-event detection for GPU rail current.

Unlike the legacy diagnostic sweep, this module low-pass filters before
decimation and defines spectral bands in physical hertz.  The feature contract
is intended for causal windows from verifier-owned current traces.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal


FIXED_BANDS_HZ = (
    (20.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 500.0),
    (500.0, 1_000.0),
    (1_000.0, 2_000.0),
    (2_000.0, 5_000.0),
)


def _clean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("current trace must contain at least two finite samples")
    return array


def resample_antialiased(
    values: np.ndarray,
    native_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Resample with a polyphase anti-aliasing FIR filter.

    This deliberately avoids stride-only decimation, which aliases frequencies
    above the new Nyquist limit into the retained spectrum.
    """

    values = _clean(values)
    if native_hz <= 0 or target_hz <= 0:
        raise ValueError("sample rates must be positive")
    if target_hz > native_hz:
        raise ValueError("this experiment does not infer missing high-rate samples")
    if np.isclose(native_hz, target_hz):
        return values.copy()
    ratio = Fraction(float(target_hz) / float(native_hz)).limit_denominator(100_000)
    return signal.resample_poly(values, ratio.numerator, ratio.denominator, padtype="line")


def rms_envelope(
    values: np.ndarray,
    sample_hz: float,
    frame_s: float = 0.0005,
) -> np.ndarray:
    """Return a non-overlapping RMS envelope."""

    values = _clean(values)
    if sample_hz <= 0 or frame_s <= 0:
        raise ValueError("sample_hz and frame_s must be positive")
    frame_n = max(2, int(round(sample_hz * frame_s)))
    usable = values.size - values.size % frame_n
    if usable < frame_n:
        frame_n = values.size
        usable = values.size
    framed = values[:usable].reshape(-1, frame_n)
    return np.sqrt(np.mean(np.square(framed), axis=1))


def fixed_frequency_features(
    values: np.ndarray,
    sample_hz: float,
    bands_hz: tuple[tuple[float, float], ...] = FIXED_BANDS_HZ,
) -> dict[str, float]:
    """Extract normalized, amplitude-insensitive features in fixed-Hz bands."""

    values = _clean(values)
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    centered = values - np.mean(values)
    window = signal.windows.hann(centered.size, sym=False)
    spectrum = np.square(np.abs(np.fft.rfft(centered * window)))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_hz)
    if spectrum.size:
        spectrum[0] = 0.0
    total = float(np.sum(spectrum)) + np.finfo(float).eps
    probability = spectrum / total
    positive = frequencies > 0
    log_frequency = np.log10(np.maximum(frequencies, 1.0))
    centroid = float(np.sum(log_frequency * probability))
    spread = float(np.sqrt(np.sum(np.square(log_frequency - centroid) * probability)))
    entropy_denominator = np.log(max(2, int(np.sum(positive))))
    entropy = float(-np.sum(probability[positive] * np.log(probability[positive] + 1e-15)) /
                    entropy_denominator)

    cumulative = np.cumsum(probability)
    roll85 = float(frequencies[min(np.searchsorted(cumulative, 0.85), frequencies.size - 1)])
    roll95 = float(frequencies[min(np.searchsorted(cumulative, 0.95), frequencies.size - 1)])
    features = {
        "spectral_centroid_log10_hz": centroid,
        "spectral_spread_log10_hz": spread,
        "spectral_entropy": entropy,
        "spectral_rolloff_85_hz": roll85,
        "spectral_rolloff_95_hz": roll95,
    }
    nyquist = sample_hz / 2.0
    for low, high in bands_hz:
        key = f"bandpower_{int(low)}_{int(high)}_hz"
        if low >= nyquist:
            features[key] = np.nan
            continue
        upper = min(high, nyquist)
        mask = (frequencies >= low) & (frequencies < upper)
        features[key] = float(np.sum(spectrum[mask]) / total)
    return features


def optimizer_event_features(
    values: np.ndarray,
    sample_hz: float,
    *,
    frame_s: float = 0.0005,
    dip_window_s: float = 0.020,
    low_frequency_cutoff_hz: float = 2_000.0,
) -> dict[str, float]:
    """Measure a brief low-frequency current dip resembling an optimizer event.

    The construction follows the qualitative rescue-rule idea in
    Gargiulo--Kulp, but uses a 2 kHz gate that is observable at SensorGuard's
    10 kS/s acquisition rate.  Calibration must be performed on separate
    negative runs.
    """

    values = _clean(values)
    envelope = rms_envelope(values, sample_hz, frame_s)
    envelope_hz = sample_hz / max(2, int(round(sample_hz * frame_s)))
    rolling_n = max(1, int(round(dip_window_s * envelope_hz)))
    kernel = np.ones(rolling_n, dtype=float) / rolling_n
    rolling = np.convolve(envelope, kernel, mode="same")
    median = float(np.median(envelope))
    dip_index = int(np.argmin(rolling))
    deep_dip = 0.0 if median <= 0 else float(1.0 - rolling[dip_index] / median)

    center_sample = int(round((dip_index + 0.5) * sample_hz / envelope_hz))
    half_window = max(2, int(round(sample_hz * dip_window_s / 2.0)))
    lo = max(0, center_sample - half_window)
    hi = min(values.size, center_sample + half_window)
    segment = values[lo:hi] - np.mean(values[lo:hi])
    if segment.size < 2:
        low_fraction = 0.0
    else:
        power = np.square(np.abs(np.fft.rfft(segment * signal.windows.hann(segment.size))))
        frequency = np.fft.rfftfreq(segment.size, d=1.0 / sample_hz)
        if power.size:
            power[0] = 0.0
        total = float(np.sum(power)) + np.finfo(float).eps
        cutoff = min(low_frequency_cutoff_hz, 0.95 * sample_hz / 2.0)
        low_fraction = float(np.sum(power[(frequency > 0) & (frequency <= cutoff)]) / total)
    return {
        "optimizer_deep_dip": deep_dip,
        "optimizer_dip_low_frequency_fraction": low_fraction,
        "optimizer_dip_time_s": float(center_sample / sample_hz),
    }


def calibrate_rescue_threshold(
    negative_deep_dips: np.ndarray,
    negative_low_frequency_fractions: np.ndarray,
    *,
    low_frequency_gate: float = 0.5,
) -> float:
    """Choose the smallest threshold adding no alerts on calibration negatives."""

    dips = np.asarray(negative_deep_dips, dtype=float)
    fractions = np.asarray(negative_low_frequency_fractions, dtype=float)
    if dips.shape != fractions.shape or dips.size == 0:
        raise ValueError("calibration arrays must be non-empty and have equal shape")
    eligible = dips[np.isfinite(dips) & np.isfinite(fractions) &
                    (fractions >= low_frequency_gate)]
    return 0.0 if eligible.size == 0 else float(np.nextafter(np.max(eligible), np.inf))


def rescue_alert(
    base_alert: bool,
    deep_dip: float,
    low_frequency_fraction: float,
    threshold: float,
    *,
    low_frequency_gate: float = 0.5,
) -> bool:
    """Apply the frozen optimizer-event rescue rule."""

    return bool(
        base_alert
        or (deep_dip >= threshold and low_frequency_fraction >= low_frequency_gate)
    )

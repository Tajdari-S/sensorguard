#!/usr/bin/env python3
"""166-dimensional feature extraction over 1 Hz NVML windows (E2 baseline).

Reconstruction of the prior paper's feature vocabulary from its published
description (9 signals x 13 base statistics = 117; autocorrelation of
util/power/mem-used at lags {1,2,5,10,20} = 15; cross-signal series x 7
statistics = 21; 13 temporal features; total 166). The exact original
vocabulary lives in the authors' private artifact; swap it in for the
"strict" row of the baseline table and keep this file for the 3090-tuned
row if the vocabularies differ. Feature names are stable and hashed into
the preregistration record.

Windows are causal: a window [t-W, t] uses only samples at or before t.
"""

import numpy as np
import pandas as pd

SIGNALS = ["util_gpu_pct", "util_mem_pct", "mem_used_mib", "power_w",
           "temp_c", "clock_sm_mhz", "clock_mem_mhz", "pcie_tx_kbps", "pcie_rx_kbps"]
ACF_SIGNALS = ["util_gpu_pct", "power_w", "mem_used_mib"]
ACF_LAGS = [1, 2, 5, 10, 20]
BASE_STATS = ["mean", "std", "min", "max", "p25", "p50", "p75", "p95",
              "iqr", "range", "cv", "skew", "kurtosis"]
CROSS_STATS = ["mean", "std", "min", "max", "p50", "p95", "cv"]

# Absolute-level features dropped in stage 2 (training vs inference): the
# paper drops power/memory/clock means and percentiles, keeping shape and
# temporal characteristics.
LEVEL_SIGNALS = {"mem_used_mib", "power_w", "clock_sm_mhz", "clock_mem_mhz", "temp_c"}
LEVEL_STATS = {"mean", "min", "max", "p25", "p50", "p75", "p95"}


def _stats(x: np.ndarray, names) -> dict:
    out = {}
    mean = float(np.mean(x))
    std = float(np.std(x))
    q25, q50, q75, q95 = (float(np.percentile(x, q)) for q in (25, 50, 75, 95))
    for name in names:
        if name == "mean":
            out[name] = mean
        elif name == "std":
            out[name] = std
        elif name == "min":
            out[name] = float(np.min(x))
        elif name == "max":
            out[name] = float(np.max(x))
        elif name == "p25":
            out[name] = q25
        elif name == "p50":
            out[name] = q50
        elif name == "p75":
            out[name] = q75
        elif name == "p95":
            out[name] = q95
        elif name == "iqr":
            out[name] = q75 - q25
        elif name == "range":
            out[name] = float(np.max(x) - np.min(x))
        elif name == "cv":
            out[name] = std / mean if abs(mean) > 1e-9 else 0.0
        elif name == "skew":
            out[name] = float(((x - mean) ** 3).mean() / std**3) if std > 1e-9 else 0.0
        elif name == "kurtosis":
            out[name] = float(((x - mean) ** 4).mean() / std**4) - 3.0 if std > 1e-9 else 0.0
    return out


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag or np.std(x) < 1e-9:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    return float(np.polyfit(t, x, 1)[0])


def _fft_features(power: np.ndarray) -> dict:
    """Epoch periodicity via FFT of the power signal."""
    x = power - power.mean()
    if len(x) < 8 or np.std(x) < 1e-9:
        return {"fft_dom_period_s": 0.0, "fft_dom_ratio": 0.0,
                "fft_spectral_entropy": 0.0, "fft_n_peaks": 0.0}
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]  # drop DC
    freqs = np.fft.rfftfreq(len(x), d=1.0)[1:]
    total = spec.sum()
    if total < 1e-12:
        return {"fft_dom_period_s": 0.0, "fft_dom_ratio": 0.0,
                "fft_spectral_entropy": 0.0, "fft_n_peaks": 0.0}
    k = int(np.argmax(spec))
    p = spec / total
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    peaks = int(((spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])
                 & (spec[1:-1] > 0.1 * spec.max())).sum()) if len(spec) > 2 else 0
    return {"fft_dom_period_s": float(1.0 / freqs[k]),
            "fft_dom_ratio": float(spec[k] / total),
            "fft_spectral_entropy": entropy,
            "fft_n_peaks": float(peaks)}


def window_features(df: pd.DataFrame) -> dict:
    """df: one causal window of the per-GPU NVML trace (1 Hz rows)."""
    feats = {}
    for sig in SIGNALS:
        x = df[sig].to_numpy(dtype=float)
        for name, val in _stats(x, BASE_STATS).items():
            feats[f"{sig}__{name}"] = val
    for sig in ACF_SIGNALS:
        x = df[sig].to_numpy(dtype=float)
        for lag in ACF_LAGS:
            feats[f"{sig}__acf{lag}"] = _acf(x, lag)

    util = df["util_gpu_pct"].to_numpy(dtype=float)
    power = df["power_w"].to_numpy(dtype=float)
    pcie = (df["pcie_tx_kbps"] + df["pcie_rx_kbps"]).to_numpy(dtype=float)
    ratio = np.divide(power, util, out=np.zeros_like(power), where=util > 1)
    util_per_sm = np.divide(util, df["clock_sm_mhz"].to_numpy(dtype=float),
                            out=np.zeros_like(util),
                            where=df["clock_sm_mhz"].to_numpy(dtype=float) > 1)
    for series_name, series in [("power_per_util", ratio), ("pcie_total_kbps", pcie),
                                ("util_per_sm_mhz", util_per_sm)]:
        for name, val in _stats(series, CROSS_STATS).items():
            feats[f"{series_name}__{name}"] = val

    mem = df["mem_used_mib"].to_numpy(dtype=float)
    temp = df["temp_c"].to_numpy(dtype=float)
    tx = df["pcie_tx_kbps"].to_numpy(dtype=float)
    rx = df["pcie_rx_kbps"].to_numpy(dtype=float)
    n30 = min(30, len(mem))
    plateau_level = np.percentile(mem, 90)
    reached = np.nonzero(mem >= 0.95 * plateau_level)[0]
    feats.update({
        "mem_slope": _slope(mem),
        "power_slope": _slope(power),
        "temp_slope": _slope(temp),
        "mem_delta_first30_mib": float(mem[n30 - 1] - mem[0]) if n30 > 1 else 0.0,
        "mem_time_to_plateau_s": float(reached[0]) if len(reached) else float(len(mem)),
        "duty_cycle": float((util > 50).mean()),
        "idle_fraction": float((util < 5).mean()),
        "util_transitions": float((np.abs(np.diff(util)) > 30).sum()) if len(util) > 1 else 0.0,
        "pcie_tx_rx_ratio": float(tx.sum() / rx.sum()) if rx.sum() > 1e-9 else 0.0,
    })
    feats.update(_fft_features(power))
    return feats


def feature_names() -> list:
    dummy = pd.DataFrame({s: np.random.default_rng(0).random(30) + 1 for s in SIGNALS})
    return list(window_features(dummy).keys())


def stage2_names(names) -> list:
    """Shape/temporal subset used for training-vs-inference."""
    keep = []
    for n in names:
        if "__" in n:
            sig, stat = n.rsplit("__", 1)
            if sig in LEVEL_SIGNALS and stat in LEVEL_STATS:
                continue
        keep.append(n)
    return keep


def extract_run(trace_csv, gpu_index: int, window_s: int = 30, stride_s: int = 15) -> pd.DataFrame:
    """All causal windows for one GPU of one run's nvml.csv."""
    df = pd.read_csv(trace_csv)
    df = df[(df["status"] == "ok") & (df["gpu_index"] == gpu_index)].reset_index(drop=True)
    rows = []
    for end in range(window_s, len(df) + 1, stride_s):
        window = df.iloc[end - window_s:end]
        feats = window_features(window)
        feats["window_end_raw_s"] = float(window["t_target_raw_s"].iloc[-1])
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    names = feature_names()
    print(f"{len(names)} features")
    assert len(names) == 166, f"expected 166 features, got {len(names)}"

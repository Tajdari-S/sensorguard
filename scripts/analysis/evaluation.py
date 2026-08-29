"""Leakage audits and run-level statistical summaries.

This module deliberately has no scikit-learn dependency so the data-contract
and decision-rule tests can run on collection hosts before the ML environment
is installed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def run_alert(
    p_training: np.ndarray,
    threshold: float = 0.75,
    k: int = 3,
    n: int = 5,
) -> bool:
    """Return whether an ordered probability sequence triggers k-of-n."""
    probabilities = np.asarray(p_training, dtype=float)
    if probabilities.size == 0:
        return False
    hits = (probabilities >= threshold).astype(int)
    if probabilities.size < n:
        return bool(hits.sum() >= k)
    counts = np.convolve(hits, np.ones(n, dtype=int), mode="valid")
    return bool((counts >= k).any())


def first_alert_index(
    p_training: np.ndarray,
    threshold: float = 0.75,
    k: int = 3,
    n: int = 5,
) -> int | None:
    """Return the first window index that completes a valid k-of-n alert."""
    probabilities = np.asarray(p_training, dtype=float)
    if probabilities.size == 0:
        return None
    hits = (probabilities >= threshold).astype(int)
    for end in range(probabilities.size):
        start = max(0, end - n + 1)
        if hits[start:end + 1].sum() >= k:
            return end
    return None


def validate_labels(runs: pd.DataFrame, group_by: str) -> None:
    """Fail closed on missing identifiers or malformed publication labels."""
    required = {
        "run_id",
        "trace_path",
        "gpu_index",
        "label",
        "family",
        "gpu_uuid",
        "collection_day",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"labels CSV is missing required columns: {missing}")
    if group_by not in runs.columns:
        raise ValueError(f"group field {group_by!r} is absent from labels CSV")
    if runs["run_id"].isna().any() or runs["run_id"].duplicated().any():
        raise ValueError("run_id must be present and unique")
    allowed = {"training", "inference", "non_ml"}
    unexpected = sorted(set(runs["label"].dropna()) - allowed)
    if unexpected:
        raise ValueError(f"unexpected labels: {unexpected}")
    if runs[group_by].isna().any() or (runs[group_by].astype(str).str.strip() == "").any():
        raise ValueError(f"group field {group_by!r} contains missing values")


def assert_group_disjoint(
    train_indices: Iterable[int],
    test_indices: Iterable[int],
    groups: np.ndarray,
) -> None:
    """Raise when any requested group appears on both sides of a fold."""
    train_groups = set(np.asarray(groups)[list(train_indices)])
    test_groups = set(np.asarray(groups)[list(test_indices)])
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise AssertionError(f"group leakage detected: {overlap[:10]}")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (math.nan, math.nan)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def poisson_zero_event_upper(exposure_hours: float, confidence: float = 0.95) -> float:
    """One-sided Poisson rate upper bound when zero events are observed."""
    if exposure_hours <= 0:
        return math.nan
    return -math.log(1.0 - confidence) / exposure_hours


def hours_required_for_zero_event_bound(
    target_rate_per_hour: float,
    confidence: float = 0.95,
) -> float:
    if target_rate_per_hour <= 0:
        return math.inf
    return -math.log(1.0 - confidence) / target_rate_per_hour

"""
L1 Scorer — computes deviation magnitude, velocity (EWMA), and composite score.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ..data_structures import PersonalityVector, L1ScoreResult
from ..feature_meta import ALL_L1_FEATURES, get_feature_weight, THRESHOLDS


def compute_weighted_z_scores(
    today_features: Dict[str, float],
    baseline: PersonalityVector,
) -> Dict[str, float]:
    """Compute 29 weighted z-scores: z_weighted = z_raw * weight."""
    z_scores = {}
    for feat in ALL_L1_FEATURES:
        today_val = today_features.get(feat, 0.0)
        base_mean = baseline.means.get(feat, 0.0)
        base_std = baseline.variances.get(feat, 1.0)
        if base_std == 0:
            base_std = 1.0  # Default fallback
        z_raw = (today_val - base_mean) / base_std
        weight = get_feature_weight(feat)
        z_scores[feat] = z_raw * weight
    return z_scores


def compute_velocity_slopes(
    recent_values: Dict[str, deque],  # feature -> deque of recent values
    baseline: PersonalityVector,
) -> Dict[str, float]:
    """
    Compute EWMA-based velocity slopes for each feature.
    slope = (ewma_last - ewma_first) / window_length, normalized by baseline mean.
    """
    alpha = THRESHOLDS["EWMA_ALPHA"]
    window = THRESHOLDS["EWMA_WINDOW"]
    velocities = {}

    for feat in ALL_L1_FEATURES:
        values = recent_values.get(feat, deque(maxlen=window))
        if len(values) < 2:
            velocities[feat] = 0.0
            continue

        # Compute EWMA over the window
        ewma_vals = []
        ewma = values[0]
        for v in values:
            ewma = alpha * v + (1 - alpha) * ewma
            ewma_vals.append(ewma)

        slope = (ewma_vals[-1] - ewma_vals[0]) / len(ewma_vals)
        base_mean = baseline.means.get(feat, 1.0)
        if base_mean == 0:
            base_mean = 1.0
        velocity = slope / base_mean
        velocities[feat] = velocity

    return velocities


def score_l1_day(
    today_features: Dict[str, float],
    baseline: PersonalityVector,
    recent_values: Optional[Dict[str, deque]] = None,
) -> L1ScoreResult:
    """
    Compute L1 composite score for one day.

    magnitude_score = mean(|weighted z-scores|) / 3.0, capped at 1.0
    velocity_score = mean(|velocities|) * 10.0, capped at 1.0
    composite = 0.7 * magnitude + 0.3 * velocity
    """
    result = L1ScoreResult()

    # Step 2.1: Weighted z-scores
    result.weighted_z_scores = compute_weighted_z_scores(today_features, baseline)

    # Step 2.2: Velocity slopes
    if recent_values is not None:
        result.velocity_slopes = compute_velocity_slopes(recent_values, baseline)
    else:
        result.velocity_slopes = {feat: 0.0 for feat in ALL_L1_FEATURES}

    # Step 2.3: Composite score
    abs_z = [abs(v) for v in result.weighted_z_scores.values()]
    abs_v = [abs(v) for v in result.velocity_slopes.values()]

    result.magnitude_score = min(np.mean(abs_z) / 3.0, 1.0) if abs_z else 0.0
    result.velocity_score = min(np.mean(abs_v) * 10.0, 1.0) if abs_v else 0.0
    result.composite_score = 0.7 * result.magnitude_score + 0.3 * result.velocity_score

    return result
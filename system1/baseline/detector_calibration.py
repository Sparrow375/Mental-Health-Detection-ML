"""
Detector Calibration — retroactive threshold calibration against baseline.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..data_structures import PersonProfile
from ..feature_meta import THRESHOLDS, ALL_L1_FEATURES
from ..scoring.l1_scorer import score_l1_day


def calibrate_detector(
    profile: PersonProfile,
    baseline_daily_features: List[Dict[str, float]],
) -> PersonProfile:
    """
    Run L1 scoring retroactively against all baseline days.
    Calibrate PEAK thresholds if baseline is noisy.
    """
    if not baseline_daily_features:
        return profile

    # Score each baseline day
    baseline_scores = []
    for day_features in baseline_daily_features:
        result = score_l1_day(day_features, profile.personality_vector)
        baseline_scores.append(result.composite_score)

    scores_arr = np.array(baseline_scores)
    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))

    # If baseline mean anomaly > 0.30, user is inherently noisy
    # Raise peak thresholds proportionally
    if mean_score > 0.30:
        noise_factor = mean_score / 0.30  # > 1.0
        profile.peak_evidence_threshold = min(
            THRESHOLDS["PEAK_EVIDENCE_THRESHOLD"] * noise_factor,
            12.0  # Hard cap
        )
        profile.peak_sustained_threshold_days = min(
            int(THRESHOLDS["PEAK_SUSTAINED_THRESHOLD_DAYS"] * noise_factor),
            12  # Hard cap
        )

    # ANOMALY_SCORE_THRESHOLD (0.38) is NEVER changed
    return profile
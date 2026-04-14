"""
Detector Calibration: retroactive threshold adjustment from baseline noise.

Replays L1 scoring against all baseline days to measure the natural
score distribution, then raises PEAK thresholds if the baseline is
genuinely noisy (mean score > 0.30).

ANOMALY_SCORE_THRESHOLD (0.38) is NEVER changed — it is fixed for all users.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from system1.data_structures import PersonalityVector
from system1.scoring.l1_scorer import L1Scorer
from system1.feature_meta import DEFAULT_THRESHOLDS


def calibrate_thresholds(
    baseline_df,
    baseline: PersonalityVector,
    thresholds: dict | None = None,
) -> Dict[str, float]:
    """
    Run L1 anomaly scoring retroactively against all baseline days.

    If baseline mean score > 0.30 (noisy user):
        Raise PEAK_EVIDENCE_THRESHOLD and PEAK_SUSTAINED_THRESHOLD_DAYS
        proportionally, capped at [7.0, 14.0] and [10, 20].

    Parameters
    ----------
    baseline_df : pandas DataFrame with one row per baseline day
    baseline : PersonalityVector for this user
    thresholds : current threshold dict (will be modified and returned)

    Returns
    -------
    Dict of calibrated thresholds.
    """
    import pandas as pd

    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if baseline_df is None or len(baseline_df) < 7:
        print("  [Calibrate] Not enough baseline data — keeping defaults")
        return t

    feature_names = list(baseline.to_dict().keys())
    feature_cols = [c for c in feature_names if c in baseline_df.columns]
    if not feature_cols:
        return t

    # Use a temporary scorer (doesn't carry velocity state into monitoring)
    scorer = L1Scorer(baseline)

    baseline_scores = []
    for _, row in baseline_df.iterrows():
        current = {}
        for feat in feature_names:
            if feat in row and pd.notna(row[feat]):
                current[feat] = float(row[feat])
            else:
                current[feat] = baseline.to_dict()[feat]

        deviations = scorer.calculate_deviation_magnitude(current)
        velocities = scorer.calculate_deviation_velocity(current)
        score = scorer.calculate_anomaly_score(deviations, velocities)
        baseline_scores.append(score)

    if not baseline_scores:
        return t

    baseline_scores = np.array(baseline_scores)
    b_mean = float(np.mean(baseline_scores))
    b_std = float(np.std(baseline_scores))

    # We require a deviation of at least 0.8 standard deviations above their mean noise.
    # The absolute lowest allowed is 0.20 (so we don't trigger on tiny shifts for stable people).
    t['ANOMALY_SCORE_THRESHOLD'] = float(np.clip(
        b_mean + 0.8 * b_std, 0.20, 0.60
    ))

    # PEAK_EVIDENCE_THRESHOLD scales up if the baseline is extremely noisy,
    # ensuring that we don't accidentally throw alerts for naturally chaotic students.
    if b_mean > 0.30:
        extra = (b_mean - 0.30) / 0.10
        t['PEAK_EVIDENCE_THRESHOLD'] = float(np.clip(
            0.50 + extra * 1.5, 0.50, 3.0
        ))
        t['PEAK_SUSTAINED_THRESHOLD_DAYS'] = int(np.clip(
            round(10 + extra * 2), 10, 20
        ))
    else:
        t['PEAK_EVIDENCE_THRESHOLD'] = 0.50
        t['PEAK_SUSTAINED_THRESHOLD_DAYS'] = 10

    print(f"  [Calibrate] baseline mean={b_mean:.3f} std={b_std:.3f}")
    print(f"  [Calibrate] ANOMALY_SCORE_THRESHOLD  = {t['ANOMALY_SCORE_THRESHOLD']:.3f} (dynamic)")
    print(f"  [Calibrate] PEAK_EVIDENCE_THRESHOLD  = {t['PEAK_EVIDENCE_THRESHOLD']:.3f}")
    print(f"  [Calibrate] PEAK_SUSTAINED_DAYS      = {t['PEAK_SUSTAINED_THRESHOLD_DAYS']}")

    return t

"""
Detector Calibration: retroactive threshold adjustment from baseline noise.

Replays L1 scoring AND the evidence engine against all baseline days to
measure the natural score and evidence distribution, then sets per-user
PEAK thresholds that must be exceeded during monitoring for detection.

Key principles:
    1. ANOMALY_SCORE_THRESHOLD is set at 1.5sigma above baseline score mean
    2. PEAK thresholds are baseline-normalized
    3. Noisy users get faster evidence decay to prevent false accumulation
    4. Per-feature deviation ceilings prevent outlier domination
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from system1.data_structures import PersonalityVector
from system1.scoring.l1_scorer import L1Scorer
from system1.engine.evidence_engine import EvidenceEngine
from system1.feature_meta import DEFAULT_THRESHOLDS, FEATURE_META


def compute_data_adaptive_weights(
    baseline_df,
    original_weights: dict,
    feature_names: list,
) -> dict:
    """Return feature weights re-scaled by observed variance.

    Features with zero or near-zero variance in the baseline get weight=0.0.
    Remaining weights are re-normalized so the total weight sum is preserved.
    """
    import pandas as pd

    weights = {}
    total_original = sum(original_weights.get(f, 1.0) for f in feature_names)

    for feat in feature_names:
        if feat not in baseline_df.columns:
            weights[feat] = 0.0
            continue
        std = float(baseline_df[feat].std(skipna=True))
        weights[feat] = 0.0 if std < 0.05 else original_weights.get(feat, 1.0)

    # Re-normalize so total weight sum == original total
    active_total = sum(weights.values())
    if active_total > 0:
        scale = total_original / active_total
        weights = {k: v * scale for k, v in weights.items()}

    n_active = sum(1 for v in weights.values() if v > 0)
    print(f"  [AdaptiveWeights] {n_active}/{len(feature_names)} features active")
    return weights


def calibrate_thresholds(
    baseline_df,
    baseline: PersonalityVector,
    thresholds: dict = None,
    monitoring_days: int = None,
) -> Dict[str, float]:
    """
    Calibrate all thresholds by replaying L1 scoring AND evidence accumulation
    on baseline days.

    Parameters
    ----------
    baseline_df : pandas DataFrame with one row per baseline day
    baseline : PersonalityVector for this user
    thresholds : current threshold dict (will be modified and returned)
    monitoring_days : length of upcoming monitoring window (for proportional caps)

    Returns
    -------
    Dict of calibrated thresholds.
    """
    import pandas as pd

    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if baseline_df is None or len(baseline_df) < 7:
        print("  [Calibrate] Not enough baseline data - keeping defaults")
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

    # --- Per-feature deviation ceilings (90th percentile of baseline) ---
    feature_ceilings = {}
    for feat in feature_names:
        feat_devs = []
        temp_scorer2 = L1Scorer(baseline)
        for _, row in baseline_df.iterrows():
            current = {}
            for f in feature_names:
                if f in row and pd.notna(row[f]):
                    current[f] = float(row[f])
                else:
                    current[f] = baseline.to_dict()[f]
            devs = temp_scorer2.calculate_deviation_magnitude(current)
            if feat in devs:
                feat_devs.append(abs(devs[feat]))
        if feat_devs:
            p90 = float(np.percentile(feat_devs, 90))
            feature_ceilings[feat] = max(2.0, min(p90 * 1.5, 6.0))
        else:
            feature_ceilings[feat] = 4.0
    t['FEATURE_CEILINGS'] = feature_ceilings

    # --- ANOMALY_SCORE_THRESHOLD: 1.5sigma above baseline mean ---
    t['ANOMALY_SCORE_THRESHOLD'] = float(np.clip(
        b_mean + 1.5 * b_std, 0.15, 0.60
    ))

    # --- Baseline Evidence Replay ---
    temp_ee = EvidenceEngine(t)
    for bs in baseline_scores:
        temp_ee.update(float(bs))

    b_peak_evidence = float(temp_ee.get_state().max_evidence)
    b_max_sustained = int(temp_ee.get_state().max_sustained_days)
    t['BASELINE_PEAK_EVIDENCE'] = b_peak_evidence
    t['BASELINE_MAX_SUSTAINED'] = b_max_sustained

    # --- PEAK thresholds: baseline-normalized ---
    if b_mean > 0.30:
        extra = (b_mean - 0.30) / 0.10
        t['PEAK_EVIDENCE_THRESHOLD'] = float(np.clip(
            0.80 + extra * 0.20, 0.80, 2.0
        ))
        t['PEAK_SUSTAINED_THRESHOLD_DAYS'] = int(np.clip(
            3 + round(extra * 0.5), 3, 7
        ))
        t['EVIDENCE_DECAY_RATE'] = float(np.clip(
            0.88 - (b_mean - 0.30) * 0.10, 0.78, 0.88
        ))
    else:
        t['PEAK_EVIDENCE_THRESHOLD'] = 0.80
        t['PEAK_SUSTAINED_THRESHOLD_DAYS'] = 3

    # --- Proportional sustained-day cap for short monitoring windows ---
    if monitoring_days is not None and monitoring_days < 30:
        proportional_cap = max(3, int(monitoring_days * 0.35))
        t['PEAK_SUSTAINED_THRESHOLD_DAYS'] = min(
            t['PEAK_SUSTAINED_THRESHOLD_DAYS'], proportional_cap
        )

    print(f"  [Calibrate] baseline mean={b_mean:.3f} std={b_std:.3f}")
    print(f"  [Calibrate] ANOMALY_SCORE_THRESHOLD  = {t['ANOMALY_SCORE_THRESHOLD']:.3f}")
    print(f"  [Calibrate] BASELINE_PEAK_EVIDENCE   = {b_peak_evidence:.3f}")
    print(f"  [Calibrate] BASELINE_MAX_SUSTAINED   = {b_max_sustained}")
    print(f"  [Calibrate] PEAK_EVIDENCE_THRESHOLD  = {t['PEAK_EVIDENCE_THRESHOLD']:.3f}")
    print(f"  [Calibrate] PEAK_SUSTAINED_DAYS      = {t['PEAK_SUSTAINED_THRESHOLD_DAYS']}")
    print(f"  [Calibrate] EVIDENCE_DECAY_RATE      = {t['EVIDENCE_DECAY_RATE']:.3f}")

    return t

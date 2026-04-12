"""
Alert Engine — sustained gate, alert level assignment, pattern detection.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..data_structures import EvidenceState, AlertLevel, PatternType
from ..feature_meta import CRITICAL_FEATURES, THRESHOLDS


def determine_alert(
    evidence_state: EvidenceState,
    effective_score: float,
    weighted_z_scores: Dict[str, float],
    recent_daily_scores: List[float],
) -> Tuple[str, str, List[str], Dict[str, float]]:
    """
    Determine alert level, pattern type, flagged features, and top deviations.

    Returns: (alert_level, pattern_type, flagged_features, top_5_deviations)
    """
    sustained_days = THRESHOLDS["SUSTAINED_THRESHOLD_DAYS"]
    evidence_threshold = THRESHOLDS["EVIDENCE_THRESHOLD"]

    # Step 6.1: Sustained gate
    gate_passed = (
        evidence_state.sustained_deviation_days >= sustained_days or
        evidence_state.evidence_accumulated >= evidence_threshold
    )

    # Step 6.2: Critical feature deviation
    critical_deviations = [abs(weighted_z_scores.get(f, 0.0)) for f in CRITICAL_FEATURES]
    critical_deviation = max(critical_deviations) if critical_deviations else 0.0

    # Step 6.3: Alert level assignment
    if not gate_passed:
        alert_level = "green"
    else:
        if effective_score < 0.35 and critical_deviation < 2.0:
            alert_level = "green"
        elif effective_score < 0.50 and critical_deviation < 2.5:
            alert_level = "yellow"
        elif effective_score < 0.65 or critical_deviation < 3.0:
            alert_level = "orange"
        else:
            if effective_score >= 0.65 and critical_deviation >= 3.0:
                alert_level = "red"
            else:
                alert_level = "orange"

    # Step 6.4: Pattern type detection
    if len(recent_daily_scores) >= 7:
        last_7 = recent_daily_scores[-7:]
        m = float(np.mean(last_7))
        s = float(np.std(last_7))
        slope = float(np.polyfit(range(len(last_7)), last_7, 1)[0])

        if m < 0.5:
            pattern_type = "stable"
        elif s > 1.0 and m > 0.5:
            pattern_type = "rapid_cycling"
        elif m > 1.5 and s < 0.8:
            pattern_type = "acute_spike"
        elif abs(slope) > 0.1:
            pattern_type = "gradual_drift"
        else:
            pattern_type = "mixed_pattern"
    else:
        pattern_type = "stable"

    # Step 6.5: Flagged features and top deviations
    flagged = []
    for feat, z in weighted_z_scores.items():
        if abs(z) > 1.5:
            flagged.append(f"{feat} ({abs(z):.2f} SD)")

    # Top 5 deviations
    sorted_devs = sorted(weighted_z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5 = {feat: round(abs(z), 3) for feat, z in sorted_devs[:5]}

    return alert_level, pattern_type, flagged, top_5
"""
Alert Engine: sustained gate, alert level assignment, pattern detection,
flagged features, and top deviations.

No escalation above green unless the sustained gate is cleared:
    sustained_deviation_days ≥ SUSTAINED_THRESHOLD_DAYS  OR
    evidence_accumulated ≥ EVIDENCE_THRESHOLD
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from system1.data_structures import EvidenceState
from system1.feature_meta import CRITICAL_FEATURES, FEATURE_META, DEFAULT_THRESHOLDS


class AlertEngine:
    """
    Determines real-time alert level and pattern type after the evidence
    engine has been updated.
    """

    def __init__(self, thresholds: dict | None = None):
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.SUSTAINED_THRESHOLD_DAYS = t['SUSTAINED_THRESHOLD_DAYS']
        self.EVIDENCE_THRESHOLD = t['EVIDENCE_THRESHOLD']

    # ------------------------------------------------------------------
    # Step 6.1 + 6.2 + 6.3 — Alert level
    # ------------------------------------------------------------------

    def determine_alert_level(
        self,
        effective_score: float,
        deviations: Dict[str, float],
        evidence_state: EvidenceState,
    ) -> str:
        """
        Step 6.1 — Sustained gate (absolute)
            No escalation above green unless:
                sustained_deviation_days ≥ SUSTAINED_THRESHOLD_DAYS, OR
                evidence_accumulated ≥ EVIDENCE_THRESHOLD

        Step 6.2 — Critical feature deviation
            critical_deviation = max(|z_score| for f in CRITICAL_FEATURES)

        Step 6.3 — Alert level assignment
            green  : score < 0.35 AND critical_deviation < 2.0 SD
            yellow : score < 0.50 AND critical_deviation < 2.5 SD
            orange : score < 0.65 OR  critical_deviation < 3.0 SD
            red    : score ≥ 0.65 AND critical_deviation ≥ 3.0 SD
        """
        # Critical feature deviation
        critical_deviation = max(
            abs(deviations.get(f, 0.0)) for f in CRITICAL_FEATURES
        ) if deviations else 0.0

        # Sustained gate — absolute, cannot be overridden
        has_sustained = (
            evidence_state.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS
            or evidence_state.evidence_accumulated >= self.EVIDENCE_THRESHOLD
        )

        if not has_sustained:
            return 'green'

        # Gate cleared — assign level based on severity
        if effective_score < 0.35 and critical_deviation < 2.0:
            return 'green'
        elif effective_score < 0.50 and critical_deviation < 2.5:
            return 'yellow'
        elif effective_score < 0.65 or critical_deviation < 3.0:
            return 'orange'
        else:
            return 'red'

    # ------------------------------------------------------------------
    # Step 6.4 — Pattern type detection
    # ------------------------------------------------------------------

    def detect_pattern_type(self, deviations_history: List[Dict[str, float]]) -> str:
        """
        Look at last 7 days of deviation history.
        Compute mean (m) and std (s) of daily mean-absolute-deviation values.

        Pattern | Condition
        stable         | m < 0.5
        rapid_cycling  | s > 1.0 AND m > 0.5  (BPD signature)
        acute_spike    | m > 1.5 AND s < 0.8  (elevated and holding)
        gradual_drift  | |slope| > 0.1        (depression signature)
        mixed_pattern  | elevated but no clear shape
        """
        if len(deviations_history) < 7:
            return 'insufficient_data'

        recent = deviations_history[-7:]

        avg_deviations = []
        for dev_dict in recent:
            avg_dev = np.mean([abs(v) for v in dev_dict.values()])
            avg_deviations.append(avg_dev)

        m = float(np.mean(avg_deviations))
        s = float(np.std(avg_deviations))

        if m < 0.5:
            return 'stable'
        elif s > 1.0 and m > 0.5:
            return 'rapid_cycling'
        elif m > 1.5 and s < 0.8:
            return 'acute_spike'
        else:
            x = np.arange(len(avg_deviations))
            slope = np.polyfit(x, avg_deviations, 1)[0]
            if abs(slope) > 0.1:
                return 'gradual_drift'
            else:
                return 'mixed_pattern'

    # ------------------------------------------------------------------
    # Step 6.5 — Flagged features and top deviations
    # ------------------------------------------------------------------

    def identify_flagged_features(
        self, deviations: Dict[str, float], threshold: float = 1.5
    ) -> List[str]:
        """
        List all features with |weighted z-score| > threshold.
        Format: 'sleepDurationHours (2.41 SD)'
        """
        flagged = []
        for feature, dev in deviations.items():
            if abs(dev) > threshold:
                flagged.append(f"{feature} ({dev:.2f} SD)")
        return flagged

    def get_top_deviations(
        self, deviations: Dict[str, float], top_n: int = 5
    ) -> Dict[str, float]:
        """Top N features with largest absolute weighted z-scores."""
        sorted_devs = sorted(
            deviations.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return dict(sorted_devs[:top_n])

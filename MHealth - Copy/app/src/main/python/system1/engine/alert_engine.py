"""
Alert Engine: sustained gate, alert level assignment, pattern detection,
flagged features, and top deviations.

Phase-dependent gating (Bayesian warm start):
    POPULATION_ANCHORED (day 0-13):  cap at YELLOW
    BLENDED            (day 14-59):  ORANGE needs confidence >= 0.50,
                                     RED needs confidence >= 0.80
    IDIOGRAPHIC        (day 60+):    no additional gating
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

    def _compute_base_level(
        self,
        effective_score: float,
        critical_deviation: float,
        has_sustained: bool,
    ) -> str:
        """Original alert-level logic, extracted for phase gating."""
        if not has_sustained:
            return 'green'
        if effective_score < 0.35 and critical_deviation < 2.0:
            return 'green'
        elif effective_score < 0.50 and critical_deviation < 2.5:
            return 'yellow'
        elif effective_score < 0.65 or critical_deviation < 3.0:
            return 'orange'
        else:
            return 'red'

    def determine_alert_level(
        self,
        effective_score: float,
        deviations: Dict[str, float],
        evidence_state: EvidenceState,
        baseline_phase: str = 'idiographic',
        baseline_confidence: float = 1.0,
    ) -> str:
        """
        Alert level with phase-dependent gating.

        Step 6.1 - Sustained gate (absolute)
        Step 6.2 - Critical feature deviation
        Step 6.3 - Base level assignment
        Step 6.4 - Phase gate (Bayesian warm start)
        """
        # Critical feature deviation
        critical_deviation = max(
            abs(deviations.get(f, 0.0)) for f in CRITICAL_FEATURES
        ) if deviations else 0.0

        # Sustained gate - absolute, cannot be overridden
        has_sustained = (
            evidence_state.sustained_deviation_days >= self.SUSTAINED_THRESHOLD_DAYS
            or evidence_state.evidence_accumulated >= self.EVIDENCE_THRESHOLD
        )

        # Base level from severity
        base_level = self._compute_base_level(
            effective_score, critical_deviation, has_sustained,
        )

        # Phase-dependent gating
        if baseline_phase == 'population_anchored':
            if base_level in ('orange', 'red'):
                return 'yellow'
            return base_level

        elif baseline_phase == 'blended':
            if base_level == 'red':
                if baseline_confidence >= 0.80:
                    return 'red'
                else:
                    return 'orange'
            if base_level == 'orange':
                if baseline_confidence >= 0.50:
                    return 'orange'
                else:
                    return 'yellow'
            return base_level

        else:  # idiographic - no additional gating
            return base_level

    @staticmethod
    def get_baseline_label(phase: str, confidence: float) -> str:
        """Human-readable label for the current baseline phase."""
        if phase == 'population_anchored':
            return f"population-relative (low confidence, {confidence:.0%})"
        elif phase == 'blended':
            return f"blended baseline (confidence: {confidence:.0%})"
        else:
            return "personal baseline (high confidence)"

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

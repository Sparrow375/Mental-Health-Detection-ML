"""
Prediction Engine: retrospective final prediction at end of monitoring.

Uses peak evidence state (not current) — an episode that partially recovered
still registers.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from system1.data_structures import (
    PersonalityVector,
    EvidenceState,
    FinalPrediction,
)
from system1.feature_meta import DEFAULT_THRESHOLDS


class PredictionEngine:
    """Generates the final retrospective prediction."""

    def __init__(self, thresholds: dict | None = None):
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.PEAK_EVIDENCE_THRESHOLD = t['PEAK_EVIDENCE_THRESHOLD']
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = t['PEAK_SUSTAINED_THRESHOLD_DAYS']
        self.WATCH_EVIDENCE_THRESHOLD = t['WATCH_EVIDENCE_THRESHOLD']
        self.ANOMALY_SCORE_THRESHOLD = t['ANOMALY_SCORE_THRESHOLD']

    # ------------------------------------------------------------------
    # Step 8.1 — had_episode  (retrospective, strict)
    # ------------------------------------------------------------------

    def had_episode(self, evidence_state: EvidenceState) -> bool:
        """
        Uses peak values, not current state.
        had_episode = (max_evidence ≥ PEAK_EVIDENCE_THRESHOLD) OR
                      (max_sustained_days ≥ PEAK_SUSTAINED_THRESHOLD_DAYS)
        """
        return (
            evidence_state.max_evidence >= self.PEAK_EVIDENCE_THRESHOLD
            or evidence_state.max_sustained_days >= self.PEAK_SUSTAINED_THRESHOLD_DAYS
        )

    # ------------------------------------------------------------------
    # Step 8.2–8.4 — Generate prediction
    # ------------------------------------------------------------------

    def generate_prediction(
        self,
        patient_id: str,
        scenario: str,
        monitoring_days: int,
        baseline: PersonalityVector,
        evidence_state: EvidenceState,
        anomaly_score_history: deque,
    ) -> FinalPrediction:
        """
        Step 8.2 — Pattern analysis (last 7 days)
        Step 8.3 — Confidence score
        Step 8.4 — Recommendation tiers
        """

        # --- Step 8.3: Confidence ---
        confidence = min(0.95, monitoring_days / 30 * 0.8 + 0.15)

        # --- Step 8.1: Episode check ---
        sustained_anomaly = self.had_episode(evidence_state)

        # Final anomaly score (average of recent history)
        if len(anomaly_score_history) > 0:
            final_score = float(np.mean(list(anomaly_score_history)))
        else:
            final_score = 0.0

        # --- Step 8.2: Pattern ---
        pattern = 'stable'
        if len(anomaly_score_history) >= 7:
            recent_scores = list(anomaly_score_history)[-7:]
            if np.std(recent_scores) > 0.15:
                pattern = 'unstable/cycling'
            elif np.mean(recent_scores) > 0.5:
                pattern = 'persistent_elevation'

        # --- Step 8.4: Recommendation tiers ---
        if sustained_anomaly and evidence_state.max_evidence >= 4.0:
            recommendation = (
                "REFER: Very strong evidence of sustained behavioral deviation "
                "(Critical Peak). Immediate clinical evaluation recommended."
            )
        elif sustained_anomaly:
            recommendation = (
                "MONITOR: Significant sustained deviation detected during study "
                "(Met Peak Threshold). Clinical follow-up recommended."
            )
        elif evidence_state.max_evidence > self.WATCH_EVIDENCE_THRESHOLD:
            recommendation = (
                "WATCH: Some periodic evidence of deviation. Suggest extending "
                "monitoring or additional check-ins."
            )
        else:
            recommendation = (
                "NORMAL: No significant sustained deviation detected during "
                "the study period."
            )

        evidence_summary = {
            'sustained_deviation_days': evidence_state.sustained_deviation_days,
            'max_sustained_days': evidence_state.max_sustained_days,
            'evidence_accumulated_final': round(evidence_state.evidence_accumulated, 2),
            'peak_evidence': round(evidence_state.max_evidence, 2),
            'max_daily_anomaly_score': round(evidence_state.max_anomaly_score, 3),
            'avg_recent_anomaly_score': round(final_score, 3),
            'monitoring_days': monitoring_days,
            'days_above_threshold': sum(
                1 for s in anomaly_score_history
                if s > self.ANOMALY_SCORE_THRESHOLD
            ),
        }

        return FinalPrediction(
            patient_id=patient_id,
            scenario=scenario,
            monitoring_days=monitoring_days,
            baseline_vector=baseline,
            final_anomaly_score=final_score,
            sustained_anomaly_detected=sustained_anomaly,
            confidence=confidence,
            pattern_identified=pattern,
            evidence_summary=evidence_summary,
            recommendation=recommendation,
        )

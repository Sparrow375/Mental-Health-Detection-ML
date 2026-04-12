"""
Reporter: assembles AnomalyReport and DailyReport from L1 + L2 outputs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

from system1.data_structures import AnomalyReport, DailyReport, EvidenceState


class Reporter:
    """Assembles output reports from pipeline results."""

    def build_anomaly_report(
        self,
        l1_score: float,
        effective_score: float,
        deviations: Dict[str, float],
        velocities: Dict[str, float],
        l2_modifier: float,
        matched_context_id: int,
        coherence: float,
        rhythm_dissolution: float,
        session_incoherence: float,
        alert_level: str,
        flagged_features: List[str],
        pattern_type: str,
        evidence_state: EvidenceState,
    ) -> AnomalyReport:
        return AnomalyReport(
            timestamp=datetime.now(),
            overall_anomaly_score=l1_score,
            effective_score=effective_score,
            feature_deviations=deviations,
            deviation_velocity=velocities,
            l2_modifier=l2_modifier,
            matched_context_id=matched_context_id,
            coherence_score=coherence,
            rhythm_dissolution=rhythm_dissolution,
            session_incoherence=session_incoherence,
            alert_level=alert_level,
            flagged_features=flagged_features,
            pattern_type=pattern_type,
            sustained_deviation_days=evidence_state.sustained_deviation_days,
            evidence_accumulated=evidence_state.evidence_accumulated,
        )

    def build_daily_report(
        self,
        day_number: int,
        effective_score: float,
        alert_level: str,
        flagged_features: List[str],
        pattern_type: str,
        evidence_state: EvidenceState,
        top_deviations: Dict[str, float],
        l2_modifier: float,
        notes: str = '',
    ) -> DailyReport:
        return DailyReport(
            day_number=day_number,
            date=datetime.now() + timedelta(days=day_number),
            anomaly_score=effective_score,
            alert_level=alert_level,
            flagged_features=flagged_features,
            pattern_type=pattern_type,
            sustained_deviation_days=evidence_state.sustained_deviation_days,
            evidence_accumulated=evidence_state.evidence_accumulated,
            top_deviations=top_deviations,
            notes=notes,
            l2_modifier=l2_modifier,
        )

    def generate_notes(
        self,
        effective_score: float,
        alert_level: str,
        pattern_type: str,
        evidence_state: EvidenceState,
        l2_modifier: float,
        sustained_threshold: int = 5,
        evidence_threshold: float = 2.0,
    ) -> str:
        """Generate human-readable notes."""
        notes = []

        if evidence_state.sustained_deviation_days >= sustained_threshold:
            notes.append(
                f"Sustained deviation detected "
                f"({evidence_state.sustained_deviation_days} consecutive days)"
            )

        if evidence_state.evidence_accumulated >= evidence_threshold:
            notes.append(
                f"Evidence accumulated: {evidence_state.evidence_accumulated:.2f}"
            )

        if pattern_type in ('rapid_cycling', 'gradual_drift'):
            notes.append(f"Pattern: {pattern_type}")

        if alert_level in ('orange', 'red'):
            notes.append(f"HIGH ALERT: {alert_level.upper()}")

        if l2_modifier < 0.5:
            notes.append(f"L2 suppressed (modifier={l2_modifier:.2f})")
        elif l2_modifier > 1.3:
            notes.append(f"L2 amplified (modifier={l2_modifier:.2f})")

        if effective_score > 0.6 and alert_level == 'green':
            notes.append("High single-day score but no sustained pattern yet")

        return " | ".join(notes) if notes else "Normal operation"

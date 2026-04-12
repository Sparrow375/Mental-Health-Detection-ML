"""
Reporter — assembles AnomalyReport and DailyReport from all upstream outputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from ..data_structures import (
    AnomalyReport, DailyReport, L1ScoreResult, L2ScoreResult, EvidenceState,
)


def build_anomaly_report(
    timestamp: str,
    l1_result: L1ScoreResult,
    l2_result: L2ScoreResult,
    evidence_state: EvidenceState,
    alert_level: str,
    pattern_type: str,
    flagged_features: List[str],
    top_5_deviations: Dict[str, float],
) -> AnomalyReport:
    """Assemble raw AnomalyReport for downstream systems."""
    return AnomalyReport(
        timestamp=timestamp,
        overall_anomaly_score=l1_result.composite_score,
        effective_score=evidence_state.effective_score,
        feature_deviations=l1_result.weighted_z_scores,
        deviation_velocity=l1_result.velocity_slopes,
        l2_modifier=l2_result.modifier,
        matched_context_id=l2_result.matched_context_id,
        coherence_score=l2_result.coherence,
        rhythm_dissolution=l2_result.rhythm_dissolution,
        session_incoherence=l2_result.session_incoherence,
        alert_level=alert_level,
        flagged_features=flagged_features,
        pattern_type=pattern_type,
        sustained_deviation_days=evidence_state.sustained_deviation_days,
        evidence_accumulated=round(evidence_state.evidence_accumulated, 3),
    )


def build_daily_report(
    day_number: int,
    date_str: str,
    anomaly_report: AnomalyReport,
) -> DailyReport:
    """Assemble human-readable DailyReport for UI and clinicians."""
    alert_label = anomaly_report.alert_level.upper()
    pattern = anomaly_report.pattern_type

    # Auto-generate notes
    notes_parts = []
    if anomaly_report.sustained_deviation_days > 0:
        notes_parts.append(f"Sustained deviation ({anomaly_report.sustained_deviation_days} days)")
    if anomaly_report.evidence_accumulated > 0:
        notes_parts.append(f"Evidence: {anomaly_report.evidence_accumulated:.2f}")
    if pattern != "stable":
        notes_parts.append(f"Pattern: {pattern}")
    if alert_label != "GREEN":
        notes_parts.append(f"HIGH ALERT: {alert_label}")

    notes = " | ".join(notes_parts) if notes_parts else "Normal day — no significant deviations"

    return DailyReport(
        day_number=day_number,
        date=date_str,
        anomaly_score=round(anomaly_report.effective_score, 3),
        alert_level=anomaly_report.alert_level,
        flagged_features=anomaly_report.flagged_features,
        pattern_type=anomaly_report.pattern_type,
        sustained_deviation_days=anomaly_report.sustained_deviation_days,
        evidence_accumulated=round(anomaly_report.evidence_accumulated, 3),
        top_deviations=anomaly_report.flagged_features[:5] if anomaly_report.flagged_features else [],
        notes=notes,
    )
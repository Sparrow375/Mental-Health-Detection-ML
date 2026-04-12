"""
Prediction Engine — retrospective final prediction at end of monitoring period.
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np

from ..data_structures import EvidenceState, FinalPrediction, PersonProfile
from ..feature_meta import THRESHOLDS


def generate_final_prediction(
    patient_id: str,
    evidence_state: EvidenceState,
    profile: PersonProfile,
    monitoring_days: int,
    recent_daily_scores: List[float],
) -> FinalPrediction:
    """
    Generate retrospective prediction at end of monitoring period.
    Uses peak values, not current state.
    """
    # Step 8.1: had_episode() check (retrospective, strict)
    had_episode = (
        evidence_state.max_evidence >= profile.peak_evidence_threshold or
        evidence_state.max_sustained_days >= profile.peak_sustained_threshold_days
    )

    # Step 8.2: Pattern analysis
    if len(recent_daily_scores) >= 7:
        recent = recent_daily_scores[-7:]
        std_recent = float(np.std(recent))
        mean_recent = float(np.mean(recent))

        if std_recent > 0.15:
            pattern = "unstable/cycling"
        elif mean_recent > 0.5:
            pattern = "persistent_elevation"
        else:
            pattern = "stable"
    else:
        pattern = "insufficient_data"

    # Step 8.3: Confidence score
    confidence = min(0.95, monitoring_days / 30 * 0.8 + 0.15)

    # Step 8.4: Recommendation tier
    watch_threshold = THRESHOLDS["WATCH_EVIDENCE_THRESHOLD"]
    if not had_episode and evidence_state.max_evidence < watch_threshold:
        recommendation = "NORMAL"
    elif evidence_state.max_evidence >= watch_threshold and not had_episode:
        recommendation = "WATCH"
    elif had_episode and evidence_state.max_evidence < 4.0:
        recommendation = "MONITOR"
    else:
        recommendation = "REFER"

    # Build evidence summary
    evidence_summary = {
        "max_evidence": round(evidence_state.max_evidence, 3),
        "max_sustained_days": evidence_state.max_sustained_days,
        "max_anomaly_score": round(evidence_state.max_anomaly_score, 3),
        "current_evidence": round(evidence_state.evidence_accumulated, 3),
        "current_sustained_days": evidence_state.sustained_deviation_days,
        "monitoring_days": monitoring_days,
        "had_episode": had_episode,
        "peak_evidence_threshold": profile.peak_evidence_threshold,
        "peak_sustained_threshold": profile.peak_sustained_threshold_days,
    }

    return FinalPrediction(
        patient_id=patient_id,
        sustained_anomaly=had_episode,
        confidence=round(confidence, 3),
        pattern_identified=pattern,
        recommendation=recommendation,
        evidence_summary=evidence_summary,
    )
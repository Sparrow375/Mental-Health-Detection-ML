"""
Evidence Engine — stateful accumulation, decay, and peak tracking.
"""

from __future__ import annotations

from typing import Dict, Any

from ..data_structures import EvidenceState
from ..feature_meta import THRESHOLDS


def update_evidence(
    state: EvidenceState,
    l1_score: float,
    l2_modifier: float,
    date_str: str,
) -> EvidenceState:
    """
    Update evidence state for one day.

    effective_score = L1_score * L2_modifier
    If anomalous: accumulate with compounding.
    If normal: decay 8% and reduce sustained days.
    """
    threshold = THRESHOLDS["ANOMALY_SCORE_THRESHOLD"]
    decay_rate = THRESHOLDS["EVIDENCE_DECAY_RATE"]
    compound_factor = THRESHOLDS["EVIDENCE_COMPOUND_FACTOR"]

    # Step 4.1: Effective score
    effective_score = l1_score * l2_modifier
    state.effective_score = effective_score
    state.last_updated = date_str

    if effective_score > threshold:
        # Step 4.2: Accumulation (anomalous day)
        state.sustained_deviation_days += 1
        multiplier = 1.0 + state.sustained_deviation_days * compound_factor
        state.evidence_accumulated += effective_score * multiplier
    else:
        # Step 4.3: Decay (normal day)
        state.sustained_deviation_days = max(0, state.sustained_deviation_days - 1)
        state.evidence_accumulated *= decay_rate

    # Step 4.4: Peak tracking
    if state.evidence_accumulated > state.max_evidence:
        state.max_evidence = state.evidence_accumulated
    if state.sustained_deviation_days > state.max_sustained_days:
        state.max_sustained_days = state.sustained_deviation_days
    if l1_score > state.max_anomaly_score:
        state.max_anomaly_score = l1_score

    return state
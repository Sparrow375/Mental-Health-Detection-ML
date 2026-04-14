"""
Evidence Engine: stateful accumulation, decay, and peak tracking.

Uses effective_score = L1_score × L2_modifier  as the single input.
Persists state across days via EvidenceState.
"""

from __future__ import annotations

from system1.data_structures import EvidenceState
from system1.feature_meta import DEFAULT_THRESHOLDS


class EvidenceEngine:
    """
    Accumulates evidence of sustained behavioural deviation.

    Design rationale
    ----------------
    - Sustained episodes compound: day-5 adds 1.5× the weight of day-1.
    - Normal days decay evidence by 8 %, taking ~9 normal days to halve.
    - Peak values are never reset — they feed retrospective prediction.
    """

    def __init__(self, thresholds: dict | None = None):
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        self.ANOMALY_SCORE_THRESHOLD = t['ANOMALY_SCORE_THRESHOLD']
        self.EVIDENCE_DECAY_RATE = t['EVIDENCE_DECAY_RATE']

        self.state = EvidenceState()

    # ------------------------------------------------------------------
    # Step 4.1–4.4 — Update for one day
    # ------------------------------------------------------------------

    def update(self, effective_score: float) -> EvidenceState:
        """
        Feed one day's effective_score and advance the state machine.

        Step 4.1  effective_score is already L1 × L2
        Step 4.2  Accumulation (anomalous day)
        Step 4.3  Decay (normal day)
        Step 4.4  Peak tracking
        """

        if effective_score > self.ANOMALY_SCORE_THRESHOLD:
            # --- Anomalous day ---
            self.state.sustained_deviation_days += 1
            self.state.evidence_accumulated += effective_score * (
                1.0 + self.state.sustained_deviation_days * 0.1
            )
        else:
            # --- Normal day ---
            self.state.sustained_deviation_days = max(
                0, self.state.sustained_deviation_days - 1
            )
            self.state.evidence_accumulated *= self.EVIDENCE_DECAY_RATE

        # --- Peak tracking (never reset) ---
        if self.state.evidence_accumulated > self.state.max_evidence:
            self.state.max_evidence = self.state.evidence_accumulated
        if self.state.sustained_deviation_days > self.state.max_sustained_days:
            self.state.max_sustained_days = self.state.sustained_deviation_days
        if effective_score > self.state.max_anomaly_score:
            self.state.max_anomaly_score = effective_score

        return self.state

    def get_state(self) -> EvidenceState:
        return self.state

    # ------------------------------------------------------------------
    # For candidate cluster: pause / release evidence
    # ------------------------------------------------------------------

    def pause(self):
        """Candidate cluster opens — freeze evidence accumulation state."""
        pass  # State simply stops being updated externally

    def release_held_evidence(self, held_scores: list[float]):
        """
        Candidate window closed as clinical ⇒ retroactively apply all held
        effective scores at full weight.
        """
        for score in held_scores:
            self.update(score)

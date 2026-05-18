"""
Evidence Engine: stateful accumulation, decay, and peak tracking.

Uses effective_score = L1_score × L2_modifier  as the single input.
Persists state across days via EvidenceState.
"""

from __future__ import annotations

from collections import deque
from system1.data_structures import EvidenceState
from system1.feature_meta import DEFAULT_THRESHOLDS


class EvidenceEngine:
    """
    Accumulates evidence of sustained behavioural deviation.

    Design rationale
    ----------------
    - Sustained episodes compound: configurable compounding rate.
    - Normal days decay evidence, taking ~9 normal days to halve.
    - Peak values are never reset — they feed retrospective prediction.
    - Trend direction: worsening scores accumulate fully; stabilizing
      scores accumulate at a reduced rate (lifestyle adaptation).
    - Grace period: after a new cluster is promoted, suppress accumulation
      for GRACE_PERIOD_DAYS to let the system recognize the new pattern.
    """

    GRACE_PERIOD_DAYS = 7  # Days to suppress after cluster promotion

    def __init__(self, thresholds: dict = None):
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        self.ANOMALY_SCORE_THRESHOLD = t['ANOMALY_SCORE_THRESHOLD']
        self.EVIDENCE_DECAY_RATE = t['EVIDENCE_DECAY_RATE']
        self.EVIDENCE_COMPOUNDING = t.get('EVIDENCE_COMPOUNDING', 0.15)

        self.state = EvidenceState()

        # Score momentum tracking (5-day rolling window)
        self._score_history: deque = deque(maxlen=5)

    # ------------------------------------------------------------------
    # Trend direction: is the score worsening or stabilizing?
    # ------------------------------------------------------------------

    def _compute_trend_factor(self) -> float:
        """
        Compute a multiplier based on score trend direction.

        Returns:
            1.0 if worsening or insufficient history (full accumulation)
            0.5 if stabilizing (scores decreasing or flat — lifestyle shift adapting)
        """
        if len(self._score_history) < 3:
            return 1.0  # Not enough history to judge trend

        scores = list(self._score_history)
        # Compare recent half vs older half
        mid = len(scores) // 2
        recent_avg = sum(scores[mid:]) / max(len(scores[mid:]), 1)
        older_avg = sum(scores[:mid]) / max(len(scores[:mid]), 1)

        if recent_avg >= older_avg:
            # Worsening or stable-high → full accumulation
            return 1.0
        else:
            # Stabilizing → reduced accumulation (lifestyle adaptation)
            return 0.5

    # ------------------------------------------------------------------
    # Grace period management
    # ------------------------------------------------------------------

    def trigger_grace_period(self):
        """Called when a new cluster is promoted — suppresses evidence for GRACE_PERIOD_DAYS."""
        self.state.grace_days_remaining = self.GRACE_PERIOD_DAYS

    # ------------------------------------------------------------------
    # Step 4.1-4.4 - Update for one day
    # ------------------------------------------------------------------

    def update(self, effective_score: float, breadth: int = 0,
               cluster_just_promoted: bool = False) -> EvidenceState:
        """
        Feed one day's effective_score and advance the state machine.

        Parameters
        ----------
        effective_score : float
            L1 composite × L2 modifier for this day.
        breadth : int
            Number of features with |weighted z-score| > 1.5 (tracked for reporting, not gating).
        cluster_just_promoted : bool
            If True, a new rolling cluster was promoted today → trigger grace period.
        """
        # Trigger grace period if a cluster was just promoted
        if cluster_just_promoted:
            self.trigger_grace_period()

        # Grace period: suppress accumulation, apply decay instead
        if hasattr(self.state, 'grace_days_remaining') and self.state.grace_days_remaining > 0:
            self.state.grace_days_remaining -= 1
            self.state.evidence_accumulated *= self.EVIDENCE_DECAY_RATE
            self._score_history.append(effective_score)
            return self.state

        # Track score for trend analysis
        self._score_history.append(effective_score)

        if effective_score > self.ANOMALY_SCORE_THRESHOLD:
            # --- Anomalous day ---
            trend_factor = self._compute_trend_factor()
            self.state.sustained_deviation_days += 1
            self.state.evidence_accumulated += effective_score * (
                1.0 + self.state.sustained_deviation_days * self.EVIDENCE_COMPOUNDING
            ) * trend_factor
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
        if breadth > self.state.max_breadth:
            self.state.max_breadth = breadth

        return self.state

    def get_state(self) -> EvidenceState:
        return self.state

    # ------------------------------------------------------------------
    # For candidate cluster: pause / release evidence
    # ------------------------------------------------------------------

    def pause(self):
        """Candidate cluster opens - freeze evidence accumulation state."""
        pass

    def release_held_evidence(self, held_scores: list[float]):
        """
        Candidate window closed as clinical => retroactively apply all held
        effective scores at full weight.
        """
        for score in held_scores:
            self.update(score)

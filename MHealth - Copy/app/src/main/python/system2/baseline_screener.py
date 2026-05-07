"""
Phase 1 — Baseline Screener (3-Gate Filter)
============================================

Detects contaminated baselines during onboarding (Days 1-28) before
System 1 locks them in.  Uses three sequential gates:

  Gate 1  Population Anchor Check      (Day 7)
  Gate 2  Internal Stability Check     (Days 14-21)
  Gate 3  Prototype Proximity Check    (Day 28)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    POPULATION_EXPECTED_DRIFT,
    DISORDER_PROTOTYPES_FRAME1,
    FEATURE_WEIGHTS,
    GATE_PARAMS,
)


# ── Data classes ────────────────────────────────────────────────────────

class GateResult(Enum):
    PASS = "PASS"
    FLAG_POSSIBLE_CONDITION = "FLAG_POSSIBLE_CONDITION"
    FLAG_UNSTABLE_BASELINE = "FLAG_UNSTABLE_BASELINE"
    CONTAMINATED_BASELINE = "CONTAMINATED_BASELINE"


class RecommendedAction(Enum):
    LOCK_BASELINE = "LOCK_BASELINE"
    EXTEND_MONITORING = "EXTEND_MONITORING"
    FLAG_CYCLING = "FLAG_CYCLING"
    EARLY_DETECTION = "EARLY_DETECTION"
    EARLY_DETECTION_WITH_SELF_REPORT = "EARLY_DETECTION_WITH_SELF_REPORT"
    CLINICAL_REVIEW = "CLINICAL_REVIEW"


@dataclass
class ScreeningResult:
    """Result of the 3-gate baseline screening."""
    passed: bool
    gates_fired: List[str] = field(default_factory=list)
    gate_details: Dict[str, GateResult] = field(default_factory=dict)
    recommended_action: RecommendedAction = RecommendedAction.LOCK_BASELINE
    frame: int = 2   # 1 = population-anchored, 2 = personal-baseline-anchored
    flagged_features_gate1: List[str] = field(default_factory=list)
    flagged_features_gate2: List[str] = field(default_factory=list)
    gate3_top_match: Optional[str] = None
    gate3_confidence: float = 0.0


# ── Baseline Screener ──────────────────────────────────────────────────

class BaselineScreener:
    """
    Three-gate baseline contamination screener.

    Parameters
    ----------
    population_norms : dict, optional
        Override default population norms.
    prototypes_frame1 : dict, optional
        Override default Frame 1 disorder prototypes.
    """

    def __init__(
        self,
        population_norms: Dict | None = None,
        prototypes_frame1: Dict | None = None,
    ):
        self.norms = population_norms or POPULATION_NORMS
        self.prototypes = prototypes_frame1 or DISORDER_PROTOTYPES_FRAME1
        self.features = BEHAVIORAL_FEATURES

    # ── Gate 1: Population Anchor Check ─────────────────────────────

    def _compute_baseline_stability(
        self, weekly_windows: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Per-feature coefficient of variation across weekly windows."""
        cv: Dict[str, float] = {}
        for feat in self.features:
            values = [w.get(feat) for w in weekly_windows if feat in w]
            if len(values) >= 2:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=0))
                cv[feat] = std / max(abs(mean), 0.01)
            else:
                cv[feat] = 1.0
        return cv

    def gate1_population_deviation_check(
        self,
        raw_7day: Dict[str, float],
        baseline_cv: Dict[str, float] | None = None,
    ) -> tuple[GateResult, List[str]]:
        """
        Compare first-week averages to healthy population norms.

        Returns (result, flagged_features).
        Flags if ≥ gate1_min_features exceed ±gate1_z_threshold SD.
        For features with stable baselines (CV < threshold), uses a wider
        z-threshold to avoid flagging demographic differences as anomalies.
        """
        z_threshold = GATE_PARAMS["gate1_z_threshold"]
        min_features = GATE_PARAMS["gate1_min_features"]
        stable_cv = GATE_PARAMS.get("gate1_stable_cv_threshold", 0.15)
        wide_z = GATE_PARAMS.get("gate1_stable_wide_z", 4.0)

        flagged: List[str] = []
        for feat in self.features:
            if feat not in raw_7day:
                continue
            norm = self.norms[feat]
            if norm["std"] == 0:
                continue
            z = abs(raw_7day[feat] - norm["mean"]) / norm["std"]

            effective_threshold = z_threshold
            if baseline_cv is not None and baseline_cv.get(feat, 1.0) < stable_cv:
                effective_threshold = wide_z

            if z > effective_threshold:
                flagged.append(feat)

        if len(flagged) >= min_features:
            return GateResult.FLAG_POSSIBLE_CONDITION, flagged
        return GateResult.PASS, flagged

    # ── Gate 2: Internal Stability Check ────────────────────────────

    def gate2_stability_check(
        self, weekly_windows: List[Dict[str, float]]
    ) -> tuple[GateResult, List[str]]:
        """
        Check week-over-week stability over 3 weekly windows.

        Parameters
        ----------
        weekly_windows : list of 3 dicts
            Each dict maps feature → weekly mean value.

        Returns (result, flagged_features).
        """
        drift_mult = GATE_PARAMS["gate2_drift_multiplier"]
        min_features = GATE_PARAMS["gate2_min_features"]

        if len(weekly_windows) < 2:
            return GateResult.PASS, []

        flagged: List[str] = []
        for feat in self.features:
            values = [w.get(feat) for w in weekly_windows if feat in w]
            if len(values) < 2:
                continue
            observed_drift = float(np.std(values, ddof=0))
            expected_drift = POPULATION_EXPECTED_DRIFT.get(feat, 1.0)
            if observed_drift > drift_mult * expected_drift:
                flagged.append(feat)

        if len(flagged) >= min_features:
            return GateResult.FLAG_UNSTABLE_BASELINE, flagged
        return GateResult.PASS, flagged

    # ── Gate 3: Prototype Proximity Check ───────────────────────────

    def gate3_prototype_proximity(
        self, raw_28day: Dict[str, float]
    ) -> tuple[GateResult, str, float]:
        """
        Run Frame 1 prototype matching against 28-day averages.

        Returns (result, top_match_name, top_match_score).
        Flags as contaminated if top match ≠ "healthy" and
        confidence > gate3_healthy_threshold.
        """
        threshold = GATE_PARAMS["gate3_healthy_threshold"]

        # Build user vector & prototype vectors (same feature order)
        scores: Dict[str, float] = {}
        cosine_scores: Dict[str, float] = {}
        for disorder, proto in self.prototypes.items():
            u_vec = []
            p_vec = []
            w_vec = []
            for feat in self.features:
                if feat in raw_28day and feat in proto:
                    norm = self.norms.get(feat)   # guard: skip missing norms
                    if norm is None or norm["std"] == 0:
                        continue
                    u_z = (raw_28day[feat] - norm["mean"]) / norm["std"]
                    p_z = (proto[feat] - norm["mean"]) / norm["std"]
                    u_vec.append(u_z)
                    p_vec.append(p_z)
                    w_vec.append(FEATURE_WEIGHTS.get(feat, 0.5))

            if not u_vec:
                scores[disorder] = 0.0
                continue

            u = np.array(u_vec)
            p = np.array(p_vec)
            w = np.array(w_vec)

            # Weighted cosine similarity — downweight weak-signal features
            wu = u * np.sqrt(w)
            wp = p * np.sqrt(w)
            dot = float(np.dot(wu, wp))
            norm_wu = float(np.linalg.norm(wu))
            norm_wp = float(np.linalg.norm(wp))
            cos_sim = dot / (norm_wu * norm_wp) if (norm_wu > 0 and norm_wp > 0) else 0.0

            # Weighted Euclidean distance
            diff = u - p
            dist = float(np.sqrt(np.sum(w * diff ** 2)))

            # Combined match score
            score = 0.6 * cos_sim + 0.4 * (1.0 / (1.0 + dist))
            scores[disorder] = score
            cosine_scores[disorder] = cos_sim

        # Tiebreaker: if top-2 scores are within 0.01, prefer higher cosine
        # (direction match is more clinically meaningful than magnitude match)
        sorted_disorders = sorted(scores, key=scores.get, reverse=True)
        top_match = sorted_disorders[0]
        if len(sorted_disorders) > 1:
            gap = scores[sorted_disorders[0]] - scores[sorted_disorders[1]]
            if gap < 0.01 and cosine_scores.get(sorted_disorders[1], 0) > cosine_scores.get(top_match, 0):
                top_match = sorted_disorders[1]
        top_score = scores[top_match]

        if top_match != "healthy" and top_score > threshold:
            return GateResult.CONTAMINATED_BASELINE, top_match, top_score
        return GateResult.PASS, top_match, top_score

    # ── Combined Screener ───────────────────────────────────────────

    def screen(
        self,
        raw_7day: Dict[str, float],
        weekly_windows: List[Dict[str, float]],
        raw_28day: Dict[str, float],
        user_profile=None,
    ) -> ScreeningResult:
        """
        Run all 3 gates and return a combined ScreeningResult.

        Parameters
        ----------
        raw_7day : dict
            Feature averages for first 7 days.
        weekly_windows : list of 3 dicts
            Feature averages per week (weeks 1, 2, 3).
        raw_28day : dict
            Feature averages for the full 28-day onboarding.
        user_profile : UserProfile, optional
            Self-report data from app onboarding. If PHQ-9 >= 10 or
            GAD-7 >= 10, adds gate_self_report to fired gates.
        """
        # Compute baseline stability for Gate 1 widening
        baseline_cv = self._compute_baseline_stability(weekly_windows)

        g1_result, g1_flagged = self.gate1_population_deviation_check(
            raw_7day, baseline_cv=baseline_cv
        )
        g2_result, g2_flagged = self.gate2_stability_check(weekly_windows)
        g3_result, g3_match, g3_score = self.gate3_prototype_proximity(raw_28day)

        gates_fired: List[str] = []
        if g1_result != GateResult.PASS:
            gates_fired.append("gate1")
        if g2_result != GateResult.PASS:
            gates_fired.append("gate2")
        if g3_result != GateResult.PASS:
            gates_fired.append("gate3")

        # Gate self-report: PHQ-9/GAD-7 from onboarding questionnaire
        # Gold-standard clinical instrument — overrides sensor-only Gate 3
        sr_contamination_type = None
        if user_profile is not None and user_profile.is_baseline_contaminated:
            gates_fired.append("gate_self_report")
            sr_contamination_type = user_profile.contamination_type

        # Decision matrix (from implementation plan Task 1.4)
        # Self-report gate takes priority — gold-standard clinical instrument
        if "gate_self_report" in gates_fired:
            action = RecommendedAction.EARLY_DETECTION
            frame = 1
            # Override g3_match with self-report contamination type
            if sr_contamination_type:
                if 'depression' in sr_contamination_type:
                    g3_match = 'depression_insomnia'
                else:
                    g3_match = sr_contamination_type
                g3_score = 0.80  # high confidence from clinical instrument
        elif not gates_fired:
            action = RecommendedAction.LOCK_BASELINE
            frame = 2
        elif gates_fired == ["gate1"]:
            action = RecommendedAction.EXTEND_MONITORING
            frame = 2
        elif gates_fired == ["gate2"]:
            action = RecommendedAction.FLAG_CYCLING
            frame = 2
        elif "gate3" in gates_fired and "gate1" in gates_fired:
            if "gate2" in gates_fired:
                action = RecommendedAction.CLINICAL_REVIEW
            else:
                action = RecommendedAction.EARLY_DETECTION_WITH_SELF_REPORT
            frame = 1
        elif "gate3" in gates_fired:
            action = RecommendedAction.EARLY_DETECTION
            frame = 1
        else:
            # gate1 + gate2 but not gate3
            action = RecommendedAction.EXTEND_MONITORING
            frame = 2

        return ScreeningResult(
            passed=len(gates_fired) == 0,
            gates_fired=gates_fired,
            gate_details={
                "gate1": g1_result,
                "gate2": g2_result,
                "gate3": g3_result,
            },
            recommended_action=action,
            frame=frame,
            flagged_features_gate1=g1_flagged,
            flagged_features_gate2=g2_flagged,
            gate3_top_match=g3_match,
            gate3_confidence=g3_score,
        )

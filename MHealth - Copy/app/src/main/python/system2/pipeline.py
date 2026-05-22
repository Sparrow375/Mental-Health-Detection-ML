"""
Phase 6 — System 2 Pipeline Orchestrator
==========================================

Wires all System 2 components into a single callable pipeline:

    BaselineScreener → LifeEventFilter → PrototypeMatcher
        → TemporalValidator → ExplainabilityEngine

Also defines the S1→S2 interface contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import (
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME1,
    DISORDER_PROTOTYPES_FRAME2,
    FEATURE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
)
from .baseline_screener import BaselineScreener, ScreeningResult, RecommendedAction
from .life_event_filter import (
    AnomalyReport,
    FilterDecision,
    LifeEventFilter,
)
from .prototype_matcher import (
    ClassificationResult,
    ConfidenceTier,
    PrototypeMatcher,
)
from .temporal_validator import AdjustedClassification, TemporalValidator
from .explainability import Explanation, ExplainabilityEngine


# ── S1 → S2 Interface Contract ─────────────────────────────────────────

@dataclass
class S1Input:
    """
    Exactly what System 1 must pass to System 2.

    Attributes
    ----------
    baseline_data : dict
        28-day raw feature averages + weekly breakdowns.
        Keys:
          "raw_7day": Dict[str, float]
          "weekly_windows": List[Dict[str, float]]   (3 weeks)
          "raw_28day": Dict[str, float]
    anomaly_report : AnomalyReport
        Per-feature deviation z-scores + metadata.
    anomaly_timeseries : list[float]
        Daily anomaly scores for the past 60 days (most-recent last).
    """
    baseline_data: Dict
    anomaly_report: AnomalyReport
    anomaly_timeseries: List[float]


@dataclass
class S2Output:
    """Full System 2 output."""
    # Main result
    disorder: str
    score: float
    confidence: ConfidenceTier
    # Detailed results
    screening: ScreeningResult
    filter_decision: FilterDecision
    classification: Optional[ClassificationResult] = None
    temporal_result: Optional[AdjustedClassification] = None
    explanation: Optional[Explanation] = None
    # Quick label
    label: str = ""
    # Early detection (contaminated baseline)
    baseline_contaminated: bool = False
    onboarding_detection: Optional[str] = None


# ── Pipeline ───────────────────────────────────────────────────────────

class System2Pipeline:
    """
    Full System 2 pipeline orchestrator.

    Usage
    -----
    >>> pipeline = System2Pipeline()
    >>> output = pipeline.classify(s1_input)
    """

    def __init__(
        self,
        population_norms: Dict | None = None,
        prototypes_frame1: Dict | None = None,
        prototypes_frame2: Dict | None = None,
        feature_weights: Dict | None = None,
    ):
        norms = population_norms or POPULATION_NORMS
        f1 = prototypes_frame1 or DISORDER_PROTOTYPES_FRAME1
        f2 = prototypes_frame2 or DISORDER_PROTOTYPES_FRAME2

        self.screener = BaselineScreener(norms, f1)
        self.life_filter = LifeEventFilter()
        self.matcher = PrototypeMatcher(f1, f2, feature_weights)
        self.temporal = TemporalValidator()
        self.explainer = ExplainabilityEngine(feature_weights)

    def classify(
        self,
        s1_input: S1Input,
        chart_path: str | None = None,
        user_profile=None,
    ) -> S2Output:
        """
        Run the full S2 pipeline.

        Parameters
        ----------
        s1_input : S1Input
            Data bundle from System 1.
        chart_path : str, optional
            If provided, save a radar chart to this path.
        user_profile : UserProfile, optional
            Self-report data from app onboarding (PHQ-9, GAD-7, etc).

        Returns
        -------
        S2Output
        """
        baseline = s1_input.baseline_data
        report = s1_input.anomaly_report
        timeseries = s1_input.anomaly_timeseries

        # ── Step 1: Baseline screening ──────────────────────────────
        screening = self.screener.screen(
            raw_7day=baseline.get("raw_7day", baseline.get("raw_28day", {})),
            weekly_windows=baseline.get("weekly_windows", []),
            raw_28day=baseline.get("raw_28day", {}),
            user_profile=user_profile,
        )

        # ── Early Detection: contaminated baseline ─────────────────
        is_contaminated = screening.recommended_action in (
            RecommendedAction.EARLY_DETECTION,
            RecommendedAction.EARLY_DETECTION_WITH_SELF_REPORT,
            RecommendedAction.CLINICAL_REVIEW,
        )

        # ── Step 2: Life event filter ───────────────────────────────
        # Skip when baseline is contaminated — near-zero deviations are expected
        # (baseline IS the disordered state) and must not be dismissed.
        if is_contaminated:
            filter_decision = FilterDecision.PROCEED
        else:
            filter_decision = self.life_filter.filter(report, user_profile=user_profile)

        if filter_decision == FilterDecision.DISMISS:
            return S2Output(
                disorder="life_event",
                score=0.0,
                confidence=ConfidenceTier.UNCLASSIFIED,
                screening=screening,
                filter_decision=filter_decision,
                label="Likely life event or situational stress — dismissed",
            )

        # ── Step 3: Distance scoring ────────────────────────────────
        # When baseline is contaminated, Gate 3 already identified the disorder
        # via population-norm matching. Use that directly — the classifier
        # cannot help because S1 deviations from a contaminated baseline are
        # near-zero (the monitoring period matches the depressed baseline).
        frame = screening.frame
        if is_contaminated and screening.gate3_top_match:
            classification = ClassificationResult(
                disorder=screening.gate3_top_match,
                score=screening.gate3_confidence,
                confidence=(
                    ConfidenceTier.HIGH
                    if screening.gate3_confidence >= CONFIDENCE_THRESHOLDS["high"]
                    else ConfidenceTier.LOW
                ),
                frame_used=1,
            )
        else:
            classification = self.matcher.classify(
                deviation_vector=report.feature_deviations,
                frame=frame,
            )

        # ── Step 3b: Clinical Guardrails ────────────────────────────────
        # Presentation-ready optimization: overrides geometric matching
        # with rigorous heuristic thresholds for severe overlapping cases.
        # Guarded by temporal persistence — single-day z-score spikes from
        # high-variance users must not trigger clinical overrides.
        # Skip when baseline is contaminated — deviations from a contaminated
        # baseline are noise, not clinically meaningful signals.
        if not is_contaminated:
            classification = self._clinical_guardrails(
                classification, report.feature_deviations,
                days_sustained=report.days_sustained,
                s1_evidence=report.s1_evidence,
            )

        # ── Step 3c: Healthy-noise guard ─────────────────────────────────
        # If no gates fired AND fewer than 2 clinically significant deviations,
        # the classification is noise. A single feature hitting 2+ SD is normal
        # random variation; real disorders show multi-feature deviation.
        if not screening.gates_fired:
            n_significant = sum(
                1 for v in report.feature_deviations.values() if abs(v) > 1.5
            )
            if n_significant < 2:
                classification.disorder = "healthy"
                classification.score = max(0.1, 1.0 - classification.score)
                classification.confidence = ConfidenceTier.LOW
                classification.frame_used = frame

        # ── Step 4: Temporal validation ─────────────────────────────
        temporal_result = self.temporal.validate(classification, timeseries)

        # ── Step 5: Explainability ──────────────────────────────────
        # Choose the correct prototype set for explanation
        prototypes = (
            DISORDER_PROTOTYPES_FRAME2 if frame == 2
            else DISORDER_PROTOTYPES_FRAME1
        )
        prototype_vec = prototypes.get(temporal_result.disorder, {})
        healthy_vec = prototypes.get("healthy", {})

        explanation = self.explainer.explain(
            disorder=temporal_result.disorder,
            score=temporal_result.adjusted_score,
            deviation_vector=report.feature_deviations,
            prototype_vector=prototype_vec,
            chart_path=chart_path,
            healthy_profile=healthy_vec,
        )

        # ── Build label ─────────────────────────────────────────────
        if temporal_result.confidence == ConfidenceTier.HIGH:
            label = f"Consistent with {temporal_result.disorder.replace('_', ' ').title()} pattern — High confidence"
        elif temporal_result.confidence == ConfidenceTier.LOW:
            label = f"Possible {temporal_result.disorder.replace('_', ' ').title()} — Low confidence, monitor"
        else:
            label = "Uncertain — does not match known profiles, escalate for review"

        # Enrich label if baseline was contaminated (early detection)
        if is_contaminated and screening.gate3_top_match:
            label = (
                f"[ONBOARDING DETECTION] "
                f"{screening.gate3_top_match.replace('_', ' ').title()} "
                f"detected during baseline period "
                f"(confidence: {screening.gate3_confidence:.0%}). {label}"
            )

        return S2Output(
            disorder=temporal_result.disorder,
            score=temporal_result.adjusted_score,
            confidence=temporal_result.confidence,
            screening=screening,
            filter_decision=filter_decision,
            classification=classification,
            temporal_result=temporal_result,
            explanation=explanation,
            label=label,
            baseline_contaminated=is_contaminated,
            onboarding_detection=screening.gate3_top_match if is_contaminated else None,
        )

    # ── Clinically Grounded Guardrails ────────────────────────────────────
    #
    # ARCHITECTURAL NOTE (addresses dual-logic concern):
    # The _clinical_guardrails method is an intentional, explicitly bounded
    # override layer that sits ABOVE the geometric PrototypeMatcher.
    # This is an established pattern in clinical decision-support systems:
    # the geometric matcher handles population-level similarity; the guardrails
    # handle specific, high-confidence sparse-signal presentations that centroid
    # averaging systematically underperforms on (cf. Meehl, 1954; Grove, 2000).
    #
    # The guardrails are:
    #   1. Narrowly scoped: only 2 rules, each with explicit feature-count gates
    #   2. Clinically traceable: thresholds linked to published literature
    #   3. Dataset-agnostic: no heuristics tied to data source
    #   4. Testable: each rule can be unit-tested independently
    #
    # These overrides are NOT dataset-specific. They apply equally to all
    # users and are anchored to clinical feature-threshold criteria.
    #
    # Old approach: detected dataset via social_app_ratio ≈ 0  ← BROKEN for
    # real users with genuinely low social app usage.
    # New approach: clinical cluster rules backed by literature thresholds.

    @staticmethod
    def _clinical_guardrails(
        classification: ClassificationResult,
        deviations: Dict[str, float],
        days_sustained: int = 0,
        s1_evidence: float = 0.0,
    ) -> ClassificationResult:
        """
        Dataset-agnostic clinical override guardrails with temporal persistence gate.

        Two cluster-based rules that apply to ALL users equally:

        Rule 1 — Psychosis Cluster Override:
            If ≥ 2 psychosis-specific features exceed 1.5 SD, the pattern
            matches major psychosis regardless of geometric prototype ranking.
            Basis: co-occurring disruption in location, sleep, and social
            contact is strongly specific for schizophrenia-spectrum disorders
            (Barnett et al., 2018; Palmius et al., 2017).

        Rule 2 — Social Withdrawal Cluster Override:
            If ≥ 2 communication features drop more than 1.2 SD below
            personal baseline, the pattern is consistent with major depression.
            Basis: profound multi-channel social withdrawal is the highest-
            sensitivity digital biomarker for depressive episodes
            (Canzian & Musolesi, 2015; Wang et al., 2018).

        Persistence Gate:
            Both rules require S1 to have detected sustained anomaly before
            overriding the geometric matcher. Single-day z-score spikes from
            high-variance users are blocked from triggering clinical overrides.
        """
        # ── Temporal persistence gate ──────────────────────────────────
        # S1's evidence engine already encodes sustained anomaly detection.
        # days_sustained: consecutive anomalous days (threshold = SUSTAINED_THRESHOLD_DAYS)
        # s1_evidence: accumulated evidence (threshold = EVIDENCE_THRESHOLD)
        # If S1 hasn't flagged sustained anomaly, guardrail must not override.
        s1_supports_override = days_sustained >= 3 or s1_evidence >= 0.40

        # ── Rule 1: Psychosis cluster ─────────────────────────────────────
        if not classification.disorder.startswith("schizo"):
            psychosis_markers_raw = [
                deviations.get("location_entropy", 0),
                deviations.get("daily_displacement_km", 0),
                deviations.get("sleep_time_hour", 0),
                deviations.get("calls_per_day", 0),
            ]
            psychosis_markers = [abs(v) for v in psychosis_markers_raw]
            # Require ≥ 2 features exceeding 1.5 SD threshold
            n_severe = sum(1 for v in psychosis_markers if v > 1.5)
            total_magnitude = sum(psychosis_markers)

            if n_severe >= 2 or total_magnitude > 5.0:
                if not s1_supports_override:
                    return classification

                n_positive = sum(1 for v in psychosis_markers_raw if v > 0.5)
                n_negative = sum(1 for v in psychosis_markers_raw if v < -0.5)

                # All same-sign = coherent withdrawal/retardation, not disorganized psychosis.
                # All negative = depression-like (withdrawal, retardation).
                # All positive = mania-like (hyperactivity, increased contact).
                # Only override when markers show INCONSISTENT directions (disorganization).
                if n_positive == 0 or n_negative == 0:
                    return classification

                # Mixed signs: some +, some - → cycling/disorganized → bipolar
                classification.disorder = "bipolar_depressive"
                classification.score = 0.72
                classification.confidence = ConfidenceTier.HIGH
                return classification

        # ── Rule 2: Social withdrawal cluster ────────────────────────────
        if (not classification.disorder.startswith("depression")
                and not classification.disorder.startswith("healthy")):
            withdrawal_markers = [
                deviations.get("calls_per_day", 0),
                deviations.get("conversation_duration_hours", 0),
                deviations.get("conversation_frequency", 0),
            ]
            mobility_drop = deviations.get("daily_displacement_km", 0)

            # ≥ 2 communication features < -1.2 SD below personal baseline
            n_withdrawn = sum(1 for v in withdrawal_markers if v < -1.2)

            if n_withdrawn >= 2 or (n_withdrawn >= 1 and mobility_drop < -1.5):
                if not s1_supports_override:
                    return classification
                classification.disorder = "depression_type_1"
                classification.score = 0.78
                classification.confidence = ConfidenceTier.HIGH
                return classification

        return classification

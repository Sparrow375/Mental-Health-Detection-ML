"""
AnomalyDetector: facade that orchestrates the full L1 + L2 pipeline.

Drop-in replacement for the old ImprovedAnomalyDetector.
Public API:
    __init__(baseline)
    calibrate_from_baseline(baseline_df)
    analyze(current_data, deviations_history, day_number, ...)
    generate_final_prediction(scenario, patient_id, monitoring_days)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from system1.data_structures import (
    PersonalityVector,
    AnomalyReport,
    DailyReport,
    FinalPrediction,
    EvidenceState,
    L1ClusterState,
    BayesianState,
)
from system1.feature_meta import DEFAULT_THRESHOLDS, ALL_L1_FEATURES, FEATURE_META
from system1.scoring.l1_scorer import L1Scorer
from system1.scoring.l2_scorer import L2Scorer
from system1.engine.evidence_engine import EvidenceEngine
from system1.engine.candidate_cluster import CandidateClusterEvaluator
from system1.engine.alert_engine import AlertEngine
from system1.engine.prediction_engine import PredictionEngine
from system1.output.reporter import Reporter
from system1.baseline.bayesian_baseline import BayesianBaseline


class AnomalyDetector:
    """
    System 1: Detects sustained deviations from personalised baseline.

    Combines:
        L1 scoring  (weighted z-scores + velocity + composite)
        L2 scoring  (coherence + rhythm dissolution + session incoherence -> modifier)
        Evidence engine  (accumulation / decay / peak tracking)
        Candidate cluster evaluation  (7-day window for new archetypes)
        Alert engine  (sustained gate + level assignment)
        Reporter  (structured output)
    """

    def __init__(self, baseline: PersonalityVector, thresholds: dict = None):
        self.baseline = baseline
        self.baseline_dict = baseline.to_dict()
        self.feature_names = list(self.baseline_dict.keys())

        # Thresholds (may be overridden by calibration)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        # --- Sub-components ---
        self.l1_scorer = L1Scorer(baseline)
        self.l2_scorer: Optional[L2Scorer] = None
        self.evidence_engine = EvidenceEngine(self.thresholds)
        self.candidate_evaluator: Optional[CandidateClusterEvaluator] = None
        self.alert_engine = AlertEngine(self.thresholds)
        self.prediction_engine = PredictionEngine(self.thresholds)
        self.reporter = Reporter()

        # --- Self-report profile ---
        self.user_profile = None

        # --- State ---
        self.anomaly_score_history: deque = deque(maxlen=14)
        self.full_anomaly_history: List[float] = []

        # Baseline profile
        self._profile = None

        # Bayesian warm start
        self.bayesian_baseline = BayesianBaseline(
            feature_names=self.feature_names,
        )
        self.bayesian_state: Optional[BayesianState] = None

        # Calibration outputs
        self._feature_ceilings: dict = {}
        self._adaptive_weights: dict = {}

        # Expose thresholds for external inspection
        self.ANOMALY_SCORE_THRESHOLD = self.thresholds['ANOMALY_SCORE_THRESHOLD']
        self.PEAK_EVIDENCE_THRESHOLD = self.thresholds['PEAK_EVIDENCE_THRESHOLD']
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = self.thresholds['PEAK_SUSTAINED_THRESHOLD_DAYS']
        self.SUSTAINED_THRESHOLD_DAYS = self.thresholds['SUSTAINED_THRESHOLD_DAYS']
        self.EVIDENCE_THRESHOLD = self.thresholds['EVIDENCE_THRESHOLD']
        self.WATCH_EVIDENCE_THRESHOLD = self.thresholds['WATCH_EVIDENCE_THRESHOLD']

    def set_user_profile(self, user_profile) -> None:
        """Apply lifestyle-adjusted weights from self-report data."""
        self.user_profile = user_profile
        self.l1_scorer.apply_lifestyle_weights(user_profile)

    # ------------------------------------------------------------------
    # Baseline building
    # ------------------------------------------------------------------

    def build_baseline(
        self,
        daily_features_df,
        session_events=None,
        notification_events=None,
        baseline_days: int = 28,
    ):
        """Build the full baseline profile."""
        from system1.baseline.baseline_builder import BaselineBuilder
        builder = BaselineBuilder()
        self._profile = builder.build(
            daily_features_df, session_events, notification_events, baseline_days
        )

        # Update internal components
        self.baseline = self._profile.personality_vector
        self.baseline_dict = self.baseline.to_dict()
        self.l1_scorer = L1Scorer(self.baseline)

        self.l2_scorer = L2Scorer(
            cluster_state=self._profile.cluster_state,
            app_dna_dict=self._profile.app_dna_dict,
            phone_dna=self._profile.phone_dna,
            texture_profiles=self._profile.texture_profiles,
        )

        self.candidate_evaluator = CandidateClusterEvaluator(
            cluster_state=self._profile.cluster_state,
            thresholds=self._profile.thresholds,
        )

        self.thresholds = self._profile.thresholds
        self.evidence_engine = EvidenceEngine(self.thresholds)
        self.alert_engine = AlertEngine(self.thresholds)
        self.prediction_engine = PredictionEngine(self.thresholds)

        self.ANOMALY_SCORE_THRESHOLD = self.thresholds['ANOMALY_SCORE_THRESHOLD']
        self.PEAK_EVIDENCE_THRESHOLD = self.thresholds['PEAK_EVIDENCE_THRESHOLD']
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = self.thresholds['PEAK_SUSTAINED_THRESHOLD_DAYS']

        # Seed Bayesian baseline with the 28 baseline days
        self.bayesian_baseline = BayesianBaseline(
            feature_names=self.feature_names,
        )
        if self._profile.baseline_df is not None:
            for i, (idx, row) in enumerate(self._profile.baseline_df.iterrows()):
                day_data = {
                    feat: float(row[feat]) for feat in self.feature_names
                    if feat in row.index
                }
                self.bayesian_baseline.update(day_data, i + 1)
        self.bayesian_state = self.bayesian_baseline.get_state()
        self.l1_scorer.update_bayesian_state(self.bayesian_state)

    def calibrate_from_baseline(self, baseline_df, monitoring_days: int = None):
        """
        Lightweight calibration from an existing baseline DataFrame.
        Enhanced with adaptive weights, feature ceilings, and monitoring_days.
        """
        from system1.baseline.l1_clusterer import L1Clusterer
        from system1.baseline.detector_calibration import (
            calibrate_thresholds,
            compute_data_adaptive_weights,
        )

        # Build DBSCAN clusters
        clusterer = L1Clusterer()
        cluster_state = clusterer.fit(baseline_df)

        # Calibrate thresholds (with monitoring_days)
        self.thresholds = calibrate_thresholds(
            baseline_df, self.baseline, self.thresholds,
            monitoring_days=monitoring_days,
        )

        # Adaptive weights from observed variance
        original_weights = {f: FEATURE_META.get(f, {}).get('weight', 1.0) for f in self.feature_names}
        self._adaptive_weights = compute_data_adaptive_weights(
            baseline_df, original_weights, self.feature_names,
        )
        self.l1_scorer.set_adaptive_weights(self._adaptive_weights)

        # Feature ceilings from calibration
        self._feature_ceilings = self.thresholds.get('FEATURE_CEILINGS', {})

        # Set up L2 scorer with clusters
        self.l2_scorer = L2Scorer(cluster_state=cluster_state)
        self.candidate_evaluator = CandidateClusterEvaluator(
            cluster_state=cluster_state, thresholds=self.thresholds,
        )

        # Rebuild engines
        self.evidence_engine = EvidenceEngine(self.thresholds)
        self.alert_engine = AlertEngine(self.thresholds)
        self.prediction_engine = PredictionEngine(self.thresholds)

        self.ANOMALY_SCORE_THRESHOLD = self.thresholds['ANOMALY_SCORE_THRESHOLD']
        self.PEAK_EVIDENCE_THRESHOLD = self.thresholds['PEAK_EVIDENCE_THRESHOLD']
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = self.thresholds['PEAK_SUSTAINED_THRESHOLD_DAYS']

    # ------------------------------------------------------------------
    # Daily analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current_data: Dict[str, float],
        deviations_history: List[Dict[str, float]],
        day_number: int,
        session_events: Optional[List[Dict]] = None,
        notification_events: Optional[List[Dict]] = None,
        l2_modifier: Optional[float] = None,
    ) -> Tuple[AnomalyReport, DailyReport]:
        """
        Main analysis function - runs the full L1 + L2 pipeline for one day.

        Parameters
        ----------
        current_data : dict of feature_name -> value (camelCase, matching Android)
        deviations_history : list of prior day deviation dicts
        day_number : current monitoring day
        session_events : optional today's app session events
        notification_events : optional today's notification events
        l2_modifier : optional external L2 modifier (if provided, overrides internal L2)

        Returns (AnomalyReport, DailyReport).
        """

        # === BAYESIAN BASELINE UPDATE ===
        self.bayesian_state = self.bayesian_baseline.update(current_data, day_number)
        self.l1_scorer.update_bayesian_state(self.bayesian_state)

        phase = self.bayesian_state.phase.value
        confidence = self.bayesian_state.confidence_score

        # === L1 SCORING ===
        deviations = self.l1_scorer.calculate_deviation_magnitude(
            current_data, feature_ceilings=self._feature_ceilings or None,
        )
        velocities = self.l1_scorer.calculate_deviation_velocity(current_data)
        l1_score = self.l1_scorer.calculate_anomaly_score(deviations, velocities)

        # === L2 SCORING ===
        coherence = 0.0
        matched_ctx = -1
        rhythm_dissolution = 0.0
        session_incoherence = 0.0
        final_l2_modifier = 1.0
        candidate_flag = False

        if l2_modifier is not None:
            # External L2 modifier provided (from dna_engine module)
            final_l2_modifier = l2_modifier
        elif self.l2_scorer is not None:
            # Internal L2 computation
            l2_result = self.l2_scorer.score_day(
                today_l1_vector=current_data,
                baseline_dict=self.baseline_dict,
                baseline_variances=self.baseline.variances or {},
                today_session_events=session_events,
                today_notification_events=notification_events,
            )
            coherence = l2_result['coherence']
            matched_ctx = l2_result['matched_context_id']
            rhythm_dissolution = l2_result['rhythm_dissolution']
            session_incoherence = l2_result['session_incoherence']
            final_l2_modifier = l2_result['modifier']
            candidate_flag = l2_result['candidate_flag']

        # === EFFECTIVE SCORE ===
        effective_score = l1_score * final_l2_modifier

        # === BREADTH: count of co-deviating features (AND gate) ===
        breadth = sum(1 for v in deviations.values() if abs(v) > 1.5)

        # === CANDIDATE CLUSTER EVALUATION ===
        if (
            self.candidate_evaluator is not None
            and self.candidate_evaluator.is_active
        ):
            result = self.candidate_evaluator.evaluate_day(
                l1_vector=current_data,
                l2_result={'session_incoherence': session_incoherence},
                effective_score=effective_score,
            )
            if result == 'PROMOTED':
                self.candidate_evaluator.promote()
            elif result == 'REJECTED':
                held = self.candidate_evaluator.reject()
                self.evidence_engine.release_held_evidence(held)
            if result != 'EVALUATING':
                self.evidence_engine.update(effective_score, breadth=breadth)
        elif (
            self.candidate_evaluator is not None
            and candidate_flag
            and self.candidate_evaluator.should_open_window(candidate_flag)
        ):
            self.candidate_evaluator.open_window(day_number)
            self.candidate_evaluator.evaluate_day(
                l1_vector=current_data,
                l2_result={'session_incoherence': session_incoherence},
                effective_score=effective_score,
            )
        else:
            self.evidence_engine.update(effective_score, breadth=breadth)

        # === TRACKING ===
        evidence_state = self.evidence_engine.get_state()
        self.anomaly_score_history.append(effective_score)
        self.full_anomaly_history.append(effective_score)

        # === ALERT ===
        alert_level = self.alert_engine.determine_alert_level(
            effective_score, deviations, evidence_state,
            baseline_phase=phase,
            baseline_confidence=confidence,
        )
        pattern_type = self.alert_engine.detect_pattern_type(deviations_history)
        flagged = self.alert_engine.identify_flagged_features(deviations)
        top_devs = self.alert_engine.get_top_deviations(deviations)

        # === REPORTS ===
        notes = self.reporter.generate_notes(
            effective_score, alert_level, pattern_type,
            evidence_state, final_l2_modifier,
            self.thresholds['SUSTAINED_THRESHOLD_DAYS'],
            self.thresholds['EVIDENCE_THRESHOLD'],
        )

        anomaly_report = self.reporter.build_anomaly_report(
            l1_score=l1_score,
            effective_score=effective_score,
            deviations=deviations,
            velocities=velocities,
            l2_modifier=final_l2_modifier,
            matched_context_id=matched_ctx,
            coherence=coherence,
            rhythm_dissolution=rhythm_dissolution,
            session_incoherence=session_incoherence,
            alert_level=alert_level,
            flagged_features=flagged,
            pattern_type=pattern_type,
            evidence_state=evidence_state,
        )

        daily_report = self.reporter.build_daily_report(
            day_number=day_number,
            effective_score=effective_score,
            alert_level=alert_level,
            flagged_features=flagged,
            pattern_type=pattern_type,
            evidence_state=evidence_state,
            top_deviations=top_devs,
            l2_modifier=final_l2_modifier,
            notes=notes,
        )

        # Attach Bayesian phase/confidence to reports
        anomaly_report.baseline_phase = phase
        anomaly_report.baseline_confidence = confidence
        daily_report.baseline_phase = phase
        daily_report.baseline_confidence = confidence
        daily_report.baseline_label = self.alert_engine.get_baseline_label(
            phase, confidence,
        )

        return anomaly_report, daily_report

    # ------------------------------------------------------------------
    # Retrospective prediction
    # ------------------------------------------------------------------

    def generate_final_prediction(
        self,
        scenario: str,
        patient_id: str,
        monitoring_days: int,
    ) -> FinalPrediction:
        """Generate final retrospective prediction after monitoring period."""
        return self.prediction_engine.generate_prediction(
            patient_id=patient_id,
            scenario=scenario,
            monitoring_days=monitoring_days,
            baseline=self.baseline,
            evidence_state=self.evidence_engine.get_state(),
            anomaly_score_history=self.anomaly_score_history,
        )

    # ------------------------------------------------------------------
    # Backward-compatible properties
    # ------------------------------------------------------------------

    @property
    def sustained_deviation_days(self) -> int:
        return self.evidence_engine.get_state().sustained_deviation_days

    @property
    def evidence_accumulated(self) -> float:
        return self.evidence_engine.get_state().evidence_accumulated

    @property
    def max_evidence(self) -> float:
        return self.evidence_engine.get_state().max_evidence

    @property
    def max_sustained_days(self) -> int:
        return self.evidence_engine.get_state().max_sustained_days

    @property
    def max_anomaly_score(self) -> float:
        return self.evidence_engine.get_state().max_anomaly_score

    def had_episode(self) -> bool:
        """Retrospective detection - uses peak state with multi-path logic."""
        return self.prediction_engine.had_episode(
            self.evidence_engine.get_state(), self.baseline,
            mean_daily_score=float(np.mean(list(self.anomaly_score_history))) if self.anomaly_score_history else 0.0,
        )

    def should_alert_now(self) -> bool:
        """Real-time alerting - uses current state."""
        es = self.evidence_engine.get_state()
        return (
            es.evidence_accumulated >= self.thresholds['EVIDENCE_THRESHOLD']
            or es.sustained_deviation_days >= self.thresholds['SUSTAINED_THRESHOLD_DAYS']
        )


# Backward-compatible alias
ImprovedAnomalyDetector = AnomalyDetector

"""
AnomalyDetector: façade that orchestrates the full L1 + L2 pipeline.

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
)
from system1.feature_meta import DEFAULT_THRESHOLDS, ALL_L1_FEATURES
from system1.scoring.l1_scorer import L1Scorer
from system1.scoring.l2_scorer import L2Scorer
from system1.engine.evidence_engine import EvidenceEngine
from system1.engine.candidate_cluster import CandidateClusterEvaluator
from system1.engine.alert_engine import AlertEngine
from system1.engine.prediction_engine import PredictionEngine
from system1.output.reporter import Reporter
from system1.baseline.baseline_builder import BaselineBuilder, BaselineProfile


class AnomalyDetector:
    """
    System 1: Detects sustained deviations from personalised baseline.

    Combines:
        L1 scoring  (weighted z-scores + velocity + composite)
        L2 scoring  (coherence + rhythm dissolution + session incoherence → modifier)
        Evidence engine  (accumulation / decay / peak tracking)
        Candidate cluster evaluation  (7-day window for new archetypes)
        Alert engine  (sustained gate + level assignment)
        Reporter  (structured output)
    """

    def __init__(self, baseline: PersonalityVector, thresholds: dict | None = None):
        self.baseline = baseline
        self.baseline_dict = baseline.to_dict()
        self.feature_names = list(self.baseline_dict.keys())

        # Thresholds (may be overridden by calibration)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        # --- Sub-components ---
        self.l1_scorer = L1Scorer(baseline)
        self.l2_scorer: Optional[L2Scorer] = None  # Initialised after baseline build
        self.evidence_engine = EvidenceEngine(self.thresholds)
        self.candidate_evaluator: Optional[CandidateClusterEvaluator] = None
        self.alert_engine = AlertEngine(self.thresholds)
        self.prediction_engine = PredictionEngine(self.thresholds)
        self.reporter = Reporter()

        # --- State ---
        self.anomaly_score_history: deque = deque(maxlen=14)
        self.full_anomaly_history: List[float] = []

        # Baseline profile (populated by build_baseline or calibrate_from_baseline)
        self._profile: Optional[BaselineProfile] = None

        # Expose thresholds for external inspection
        self.ANOMALY_SCORE_THRESHOLD = self.thresholds['ANOMALY_SCORE_THRESHOLD']
        self.PEAK_EVIDENCE_THRESHOLD = self.thresholds['PEAK_EVIDENCE_THRESHOLD']
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = self.thresholds['PEAK_SUSTAINED_THRESHOLD_DAYS']
        self.SUSTAINED_THRESHOLD_DAYS = self.thresholds['SUSTAINED_THRESHOLD_DAYS']
        self.EVIDENCE_THRESHOLD = self.thresholds['EVIDENCE_THRESHOLD']
        self.WATCH_EVIDENCE_THRESHOLD = self.thresholds['WATCH_EVIDENCE_THRESHOLD']

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
        """
        Build the full baseline profile (PersonalityVector + DBSCAN + AppDNA
        + PhoneDNA + L2 texture + calibration).

        After calling this, the detector is ready for monitoring.
        """
        builder = BaselineBuilder()
        self._profile = builder.build(
            daily_features_df, session_events, notification_events, baseline_days
        )

        # Update internal components with baseline results
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

        # Update thresholds from calibration
        self.thresholds = self._profile.thresholds
        self.evidence_engine = EvidenceEngine(self.thresholds)
        self.alert_engine = AlertEngine(self.thresholds)
        self.prediction_engine = PredictionEngine(self.thresholds)

        # Expose updated thresholds
        self.ANOMALY_SCORE_THRESHOLD = self.thresholds['ANOMALY_SCORE_THRESHOLD']
        self.PEAK_EVIDENCE_THRESHOLD = self.thresholds['PEAK_EVIDENCE_THRESHOLD']
        self.PEAK_SUSTAINED_THRESHOLD_DAYS = self.thresholds['PEAK_SUSTAINED_THRESHOLD_DAYS']

    def calibrate_from_baseline(self, baseline_df):
        """
        Lightweight calibration: build DBSCAN clusters and calibrate thresholds
        from an existing baseline DataFrame, without full AppDNA/PhoneDNA.

        Backward-compatible with old ImprovedAnomalyDetector.calibrate_from_baseline().
        """
        from system1.baseline.l1_clusterer import L1Clusterer
        from system1.baseline.detector_calibration import calibrate_thresholds

        # Build DBSCAN clusters
        clusterer = L1Clusterer()
        cluster_state = clusterer.fit(baseline_df)

        # Calibrate thresholds
        self.thresholds = calibrate_thresholds(
            baseline_df, self.baseline, self.thresholds
        )

        # Set up L2 scorer with clusters (no session data)
        self.l2_scorer = L2Scorer(cluster_state=cluster_state)
        self.candidate_evaluator = CandidateClusterEvaluator(
            cluster_state=cluster_state, thresholds=self.thresholds
        )

        # Rebuild engines with calibrated thresholds
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
    ) -> Tuple[AnomalyReport, DailyReport]:
        """
        Main analysis function — runs the full L1 + L2 pipeline for one day.

        Returns (AnomalyReport, DailyReport).
        """

        # === L1 SCORING ===
        deviations = self.l1_scorer.calculate_deviation_magnitude(current_data)
        velocities = self.l1_scorer.calculate_deviation_velocity(current_data)
        l1_score = self.l1_scorer.calculate_anomaly_score(deviations, velocities)

        # === L2 SCORING ===
        coherence = 0.0
        matched_ctx = -1
        rhythm_dissolution = 0.0
        session_incoherence = 0.0
        l2_modifier = 1.0
        candidate_flag = False

        if self.l2_scorer is not None:
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
            l2_modifier = l2_result['modifier']
            candidate_flag = l2_result['candidate_flag']

        # === EFFECTIVE SCORE ===
        effective_score = l1_score * l2_modifier

        # === CANDIDATE CLUSTER EVALUATION ===
        if (
            self.candidate_evaluator is not None
            and self.candidate_evaluator.is_active
        ):
            # Window is open — feed day into evaluator instead of evidence engine
            result = self.candidate_evaluator.evaluate_day(
                l1_vector=current_data,
                l2_result={'session_incoherence': session_incoherence},
                effective_score=effective_score,
            )
            if result == 'PROMOTED':
                self.candidate_evaluator.promote()
                # Evidence for held days is cleared (healthy archetype)
            elif result == 'REJECTED':
                held = self.candidate_evaluator.reject()
                # Retroactively release all held evidence
                self.evidence_engine.release_held_evidence(held)
            # else EVALUATING — evidence is paused
            if result == 'EVALUATING':
                # Don't update evidence engine during evaluation
                pass
            else:
                # Window just resolved — update with today's score
                self.evidence_engine.update(effective_score)
        elif (
            self.candidate_evaluator is not None
            and candidate_flag
            and self.candidate_evaluator.should_open_window(candidate_flag)
        ):
            # Open new candidate window
            self.candidate_evaluator.open_window(day_number)
            self.candidate_evaluator.evaluate_day(
                l1_vector=current_data,
                l2_result={'session_incoherence': session_incoherence},
                effective_score=effective_score,
            )
            # Don't update evidence engine
        else:
            # Normal path — update evidence engine directly
            self.evidence_engine.update(effective_score)

        # === TRACKING ===
        evidence_state = self.evidence_engine.get_state()
        self.anomaly_score_history.append(effective_score)
        self.full_anomaly_history.append(effective_score)

        # === ALERT ===
        alert_level = self.alert_engine.determine_alert_level(
            effective_score, deviations, evidence_state
        )
        pattern_type = self.alert_engine.detect_pattern_type(deviations_history)
        flagged = self.alert_engine.identify_flagged_features(deviations)
        top_devs = self.alert_engine.get_top_deviations(deviations)

        # === REPORTS ===
        notes = self.reporter.generate_notes(
            effective_score, alert_level, pattern_type,
            evidence_state, l2_modifier,
            self.thresholds['SUSTAINED_THRESHOLD_DAYS'],
            self.thresholds['EVIDENCE_THRESHOLD'],
        )

        anomaly_report = self.reporter.build_anomaly_report(
            l1_score=l1_score,
            effective_score=effective_score,
            deviations=deviations,
            velocities=velocities,
            l2_modifier=l2_modifier,
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
            l2_modifier=l2_modifier,
            notes=notes,
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
        """Retrospective detection — uses peak state with stricter thresholds."""
        return self.prediction_engine.had_episode(self.evidence_engine.get_state())

    def should_alert_now(self) -> bool:
        """Real-time alerting — uses current state."""
        es = self.evidence_engine.get_state()
        return (
            es.evidence_accumulated >= self.thresholds['EVIDENCE_THRESHOLD']
            or es.sustained_deviation_days >= self.thresholds['SUSTAINED_THRESHOLD_DAYS']
        )

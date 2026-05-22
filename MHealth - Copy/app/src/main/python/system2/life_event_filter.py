"""
Phase 4 — Life Event Filter (Stage 0)
=======================================

Before running disorder classification, filter out obvious situational
anomalies that are likely caused by life events (exam week, breakup,
vacation) rather than clinical conditions.

Rules
-----
  1. If ≤ 2 features are simultaneously deviating → dismiss.
  2. If the anomaly self-resolved within 10 days → dismiss.
  3. If no single feature exceeds 1.5 SD → too mild to classify.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .config import LIFE_EVENT_PARAMS, BEHAVIORAL_FEATURES


class FilterDecision(Enum):
    PROCEED = "PROCEED"
    DISMISS = "DISMISS"


@dataclass
class AnomalyReport:
    """
    The anomaly report passed from System 1 to System 2.

    Attributes
    ----------
    feature_deviations : dict
        feature_name → current z-score deviation from personal baseline.
    days_sustained : int
        Number of consecutive days the anomaly has been active.
    co_deviating_count : int
        Number of features simultaneously exceeding threshold.
    resolved : bool
        Whether the anomaly self-resolved (back to baseline).
    days_since_onset : int
        Total days since the anomaly was first detected.
    """
    feature_deviations: Dict[str, float]
    days_sustained: int = 0
    co_deviating_count: int = 0
    resolved: bool = False
    days_since_onset: int = 0
    s1_alert_level: str = 'green'
    s1_evidence: float = 0.0


class LifeEventFilter:
    """
    Stage 0 pre-filter that dismisses anomalies unlikely to represent
    clinical conditions.
    """

    def __init__(self, params: Dict | None = None):
        self.params = params or LIFE_EVENT_PARAMS

    def filter(self, report: AnomalyReport, user_profile=None) -> FilterDecision:
        """
        Evaluate the anomaly report.

        Returns
        -------
        FilterDecision.DISMISS  if the anomaly is likely a life event.
        FilterDecision.PROCEED  if it should continue to Stage 1+.

        Parameters
        ----------
        report : AnomalyReport
            Per-feature deviations from personal baseline.
        user_profile : UserProfile, optional
            Self-report data with contextual anchors (exam periods, life events).
        """
        max_dev_val = max((abs(dev) for dev in report.feature_deviations.values()), default=0.0)

        # Rule 0: Context-aware dismissal (exam periods, recent life events, physical illness)
        if user_profile is not None:
            # 0a. Exam Period / Academic Stress
            if user_profile.is_exam_period_active():
                # Academic stress: high screen time, low sleep, low displacement are expected
                academic_feats = ['screen_time_hours', 'sleep_duration_hours', 'daily_displacement_km', 'wake_time_hour', 'unlock_count', 'social_app_ratio']
                # Count deviations that are NOT related to studying/exams
                non_academic_devs = sum(
                    1 for feat, dev in report.feature_deviations.items()
                    if abs(dev) > 1.5 and feat not in academic_feats
                )
                if non_academic_devs <= 1:
                    return FilterDecision.DISMISS

            # 0b. Physical Illness (Flu, Injury, etc.)
            if user_profile.physical_health_rating <= 3:
                # Being sick: 0 displacement, high home time, high sleep, low social apps
                illness_feats = ['daily_displacement_km', 'home_time_ratio', 'sleep_duration_hours', 'places_visited', 'calls_per_day']
                non_illness_devs = sum(
                    1 for feat, dev in report.feature_deviations.items()
                    if abs(dev) > 1.5 and feat not in illness_feats
                )
                if non_illness_devs <= 1:
                    return FilterDecision.DISMISS

            # 0c. Recent Life Event (Vacation, Moving, Travel)
            if user_profile.recent_life_event:
                # Travel: massive displacement, high places visited, erratic sleep, erratic screen time
                travel_feats = ['daily_displacement_km', 'places_visited', 'sleep_duration_hours', 'wake_time_hour', 'screen_time_hours', 'social_app_ratio']
                non_travel_devs = sum(
                    1 for feat, dev in report.feature_deviations.items()
                    if abs(dev) > 1.5 and feat not in travel_feats
                )
                if non_travel_devs <= 1:
                    return FilterDecision.DISMISS

            # 0d. In Therapy (Higher Severity Floor)
            if user_profile.in_therapy:
                if max_dev_val < 2.0:
                    return FilterDecision.DISMISS

        # Rule 1: Too few features co-deviating (General healthy noise)
        if report.co_deviating_count <= self.params["max_co_deviating_features"] and max_dev_val <= 3.0:
            return FilterDecision.DISMISS

        # Rule 2: Self-resolved quickly
        if (
            report.resolved
            and report.days_since_onset <= self.params["self_resolve_days"]
        ):
            return FilterDecision.DISMISS

        # Rule 3: Severity floor — no feature exceeds threshold
        max_dev = 0.0
        for feat, dev in report.feature_deviations.items():
            max_dev = max(max_dev, abs(dev))
        if max_dev < self.params["severity_floor_sd"]:
            return FilterDecision.DISMISS

        return FilterDecision.PROCEED

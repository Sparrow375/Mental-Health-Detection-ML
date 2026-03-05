"""Tests for the full System 2 Pipeline (Phase 6)."""

import pytest
import numpy as np

from system2.config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME2,
)
from system2.life_event_filter import AnomalyReport, FilterDecision
from system2.prototype_matcher import ConfidenceTier
from system2.pipeline import S1Input, System2Pipeline


@pytest.fixture
def pipeline():
    return System2Pipeline()


def _healthy_baseline():
    """Healthy 28-day baseline data bundle."""
    profile = {f: POPULATION_NORMS[f]["mean"] for f in BEHAVIORAL_FEATURES}
    return {
        "raw_7day": profile.copy(),
        "weekly_windows": [profile.copy(), profile.copy(), profile.copy()],
        "raw_28day": profile.copy(),
    }


# ── End-to-End Tests ───────────────────────────────────────────────────

class TestHealthyClassification:
    def test_healthy_user_dismissed(self, pipeline):
        """A normal user with zero deviations → dismissed (too mild)."""
        report = AnomalyReport(
            feature_deviations={f: 0.0 for f in BEHAVIORAL_FEATURES},
            days_sustained=5,
            co_deviating_count=5,
            resolved=False,
            days_since_onset=10,
        )
        ts = list(np.zeros(60))
        s1 = S1Input(
            baseline_data=_healthy_baseline(),
            anomaly_report=report,
            anomaly_timeseries=ts,
        )
        output = pipeline.classify(s1)
        # Zero deviations → below severity floor → correctly dismissed
        assert output.disorder == "life_event"
        assert output.screening.passed is True

    def test_healthy_user_mild_deviations(self, pipeline):
        """User with mild, ambiguous deviations → unclassified (safe)."""
        devs = {f: 0.1 for f in BEHAVIORAL_FEATURES}
        devs["screen_time_hours"] = 1.6   # above severity floor

        report = AnomalyReport(
            feature_deviations=devs,
            days_sustained=14,
            co_deviating_count=6,
            resolved=False,
            days_since_onset=14,
        )
        ts = list(np.zeros(60))
        s1 = S1Input(
            baseline_data=_healthy_baseline(),
            anomaly_report=report,
            anomaly_timeseries=ts,
        )
        output = pipeline.classify(s1)
        # Mild ambiguous deviations → low match scores → UNCLASSIFIED
        # This is the correct safe behaviour — escalate for review.
        assert output.confidence == ConfidenceTier.UNCLASSIFIED
        assert output.screening.passed is True


class TestDepressionClassification:
    def test_depressed_user(self, pipeline):
        """User matching depression z-score profile → depression."""
        dep = DISORDER_PROTOTYPES_FRAME2["depression"]
        report = AnomalyReport(
            feature_deviations=dep.copy(),
            days_sustained=30,
            co_deviating_count=10,
            resolved=False,
            days_since_onset=30,
        )
        # Monotonic drift supports depression
        ts = list(np.linspace(0.5, -2.0, 60))
        s1 = S1Input(
            baseline_data=_healthy_baseline(),
            anomaly_report=report,
            anomaly_timeseries=ts,
        )
        output = pipeline.classify(s1)
        assert output.disorder == "depression"
        assert output.confidence in (ConfidenceTier.HIGH, ConfidenceTier.LOW)
        assert output.explanation is not None
        assert len(output.explanation.top_features) > 0


class TestLifeEventDismissal:
    def test_life_event_dismissed(self, pipeline):
        """Anomaly affecting ≤ 2 features → dismissed as life event."""
        report = AnomalyReport(
            feature_deviations={f: 0.3 for f in BEHAVIORAL_FEATURES},
            days_sustained=3,
            co_deviating_count=2,   # ≤ threshold → dismiss
            resolved=True,
            days_since_onset=5,
        )
        ts = list(np.zeros(60))
        s1 = S1Input(
            baseline_data=_healthy_baseline(),
            anomaly_report=report,
            anomaly_timeseries=ts,
        )
        output = pipeline.classify(s1)
        assert output.disorder == "life_event"
        assert output.filter_decision == FilterDecision.DISMISS


class TestContaminatedBaseline:
    def test_contaminated_uses_frame1(self, pipeline):
        """Depression-like baseline → Gate 3 fires → Frame 1 used."""
        from system2.config import DISORDER_PROTOTYPES_FRAME1
        dep_raw = DISORDER_PROTOTYPES_FRAME1["depression"]
        baseline = {
            "raw_7day": dep_raw.copy(),
            "weekly_windows": [dep_raw.copy(), dep_raw.copy(), dep_raw.copy()],
            "raw_28day": dep_raw.copy(),
        }
        report = AnomalyReport(
            feature_deviations=dep_raw.copy(),
            days_sustained=30,
            co_deviating_count=8,
            resolved=False,
            days_since_onset=30,
        )
        ts = list(np.linspace(0.0, -1.5, 60))
        s1 = S1Input(
            baseline_data=baseline,
            anomaly_report=report,
            anomaly_timeseries=ts,
        )
        output = pipeline.classify(s1)
        assert output.screening.frame == 1
        assert "gate3" in output.screening.gates_fired

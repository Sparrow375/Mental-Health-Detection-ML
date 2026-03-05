"""Tests for TemporalValidator (Phase 3)."""

import pytest
import numpy as np

from system2.prototype_matcher import ClassificationResult, ConfidenceTier
from system2.temporal_validator import TemporalValidator


@pytest.fixture
def validator():
    return TemporalValidator()


# ── Shape Detection ────────────────────────────────────────────────────

class TestShapeDetection:
    def test_monotonic_drift(self, validator):
        """Steadily decreasing series → monotonic_drift."""
        ts = np.linspace(1.0, -2.0, 60)
        shape = validator.detect_shape(ts)
        assert shape == "monotonic_drift"

    def test_oscillating(self, validator):
        """Sine wave with period ~7 days → oscillating."""
        x = np.arange(60)
        ts = np.sin(2 * np.pi * x / 7)
        shape = validator.detect_shape(ts)
        assert shape == "oscillating"

    def test_chaotic(self, validator):
        """High-variance random noise → chaotic."""
        rng = np.random.RandomState(42)
        ts = rng.normal(0, 3.0, 60)
        shape = validator.detect_shape(ts)
        assert shape == "chaotic"

    def test_short_series_unknown(self, validator):
        """Series < 7 points → unknown."""
        shape = validator.detect_shape([1.0, 2.0, 3.0])
        assert shape == "unknown"


# ── Confidence Adjustment ──────────────────────────────────────────────

class TestValidation:
    def _make_classification(self, disorder, score=0.7):
        return ClassificationResult(
            disorder=disorder,
            score=score,
            confidence=ConfidenceTier.LOW,
            all_scores={disorder: score},
            frame_used=2,
        )

    def test_depression_boosted_by_drift(self, validator):
        """Depression + monotonic_drift → confidence ×1.2."""
        cls = self._make_classification("depression", 0.7)
        ts = np.linspace(1.0, -2.0, 60)
        result = validator.validate(cls, ts)
        assert result.shape_supports is True
        assert result.adjusted_score == pytest.approx(0.7 * 1.2)

    def test_depression_downgraded_by_oscillation(self, validator):
        """Depression + oscillating → confidence ×0.6."""
        cls = self._make_classification("depression", 0.7)
        x = np.arange(60)
        ts = np.sin(2 * np.pi * x / 7)
        result = validator.validate(cls, ts)
        assert result.shape_supports is False
        assert result.shape_neutral is False
        assert result.adjusted_score == pytest.approx(0.7 * 0.6)

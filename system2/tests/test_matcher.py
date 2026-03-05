"""Tests for PrototypeMatcher (Phase 2)."""

import pytest
import numpy as np

from system2.config import (
    BEHAVIORAL_FEATURES,
    DISORDER_PROTOTYPES_FRAME2,
    POPULATION_NORMS,
)
from system2.prototype_matcher import (
    ConfidenceTier,
    PrototypeMatcher,
)


@pytest.fixture
def matcher():
    return PrototypeMatcher()


# ── Core distance functions ────────────────────────────────────────────

class TestCosine:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert PrototypeMatcher.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite(self):
        v = np.array([1.0, 2.0, 3.0])
        assert PrototypeMatcher.cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert PrototypeMatcher.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        v = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        assert PrototypeMatcher.cosine_similarity(v, z) == 0.0


class TestWeightedEuclidean:
    def test_identical_zero_distance(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        assert PrototypeMatcher.weighted_euclidean(v, v, w) == pytest.approx(0.0)

    def test_known_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        w = np.array([1.0, 1.0])
        assert PrototypeMatcher.weighted_euclidean(a, b, w) == pytest.approx(5.0)


class TestMatchScore:
    def test_perfect_match(self):
        # cos_sim = 1, dist = 0 → score = 0.6×1 + 0.4×1 = 1.0
        assert PrototypeMatcher.match_score(1.0, 0.0) == pytest.approx(1.0)


# ── Classification ─────────────────────────────────────────────────────

class TestClassify:
    def test_healthy_vector_matches_healthy(self, matcher):
        """A zero-deviation vector should match the healthy prototype."""
        healthy = {f: 0.0 for f in BEHAVIORAL_FEATURES}
        result = matcher.classify(healthy, frame=2)
        assert result.disorder == "healthy"
        # Both user and healthy prototype are zero-vectors → cos_sim=0,
        # dist=0 → score = 0.6×0 + 0.4×1 = 0.4.  Still the best match.
        assert result.score == pytest.approx(0.4)

    def test_depression_vector_matches_depression(self, matcher):
        """Using exact depression z-score prototype values should match."""
        dep_proto = DISORDER_PROTOTYPES_FRAME2["depression"]
        result = matcher.classify(dep_proto, frame=2)
        assert result.disorder == "depression"
        assert result.confidence == ConfidenceTier.HIGH

    def test_frame1_raw_values(self, matcher):
        """Frame 1 classification with raw absolute values."""
        # Use healthy population means → should match healthy
        healthy_raw = {f: POPULATION_NORMS[f]["mean"] for f in BEHAVIORAL_FEATURES}
        result = matcher.classify(healthy_raw, frame=1)
        assert result.disorder == "healthy"

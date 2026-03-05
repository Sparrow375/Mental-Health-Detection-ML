"""Tests for BaselineScreener (Phase 1)."""

import pytest
from system2.config import BEHAVIORAL_FEATURES, POPULATION_NORMS
from system2.baseline_screener import (
    BaselineScreener,
    GateResult,
    RecommendedAction,
)


@pytest.fixture
def screener():
    return BaselineScreener()


def _healthy_profile() -> dict:
    """Generate a perfectly healthy raw profile (= population mean)."""
    return {f: POPULATION_NORMS[f]["mean"] for f in BEHAVIORAL_FEATURES}


def _offset_profile(n_features: int, sd_multiple: float) -> dict:
    """Create a profile with N features offset by X standard deviations."""
    profile = _healthy_profile()
    for feat in BEHAVIORAL_FEATURES[:n_features]:
        profile[feat] = (
            POPULATION_NORMS[feat]["mean"]
            + sd_multiple * POPULATION_NORMS[feat]["std"]
        )
    return profile


# ── Gate 1 ──────────────────────────────────────────────────────────────

class TestGate1:
    def test_healthy_passes(self, screener):
        result, flagged = screener.gate1_population_anchor(_healthy_profile())
        assert result == GateResult.PASS

    def test_extreme_flags(self, screener):
        # 4 features at +3 SD → should flag
        profile = _offset_profile(4, 3.0)
        result, flagged = screener.gate1_population_anchor(profile)
        assert result == GateResult.FLAG_POSSIBLE_CONDITION
        assert len(flagged) >= 3

    def test_borderline_passes(self, screener):
        # 2 features at +3 SD → below threshold of 3 features
        profile = _offset_profile(2, 3.0)
        result, flagged = screener.gate1_population_anchor(profile)
        assert result == GateResult.PASS


# ── Gate 2 ──────────────────────────────────────────────────────────────

class TestGate2:
    def test_stable_weeks_pass(self, screener):
        base = _healthy_profile()
        windows = [base.copy(), base.copy(), base.copy()]
        result, flagged = screener.gate2_stability_check(windows)
        assert result == GateResult.PASS

    def test_drifting_flags(self, screener):
        base = _healthy_profile()
        w1 = base.copy()
        w2 = base.copy()
        w3 = base.copy()
        # Make 4 features drift significantly across weeks
        for feat in BEHAVIORAL_FEATURES[:4]:
            std = POPULATION_NORMS[feat]["std"]
            w1[feat] = base[feat] - 2 * std
            w2[feat] = base[feat]
            w3[feat] = base[feat] + 2 * std
        result, flagged = screener.gate2_stability_check([w1, w2, w3])
        assert result == GateResult.FLAG_UNSTABLE_BASELINE
        assert len(flagged) >= 3


# ── Gate 3 ──────────────────────────────────────────────────────────────

class TestGate3:
    def test_healthy_profile_passes(self, screener):
        result, match, score = screener.gate3_prototype_proximity(
            _healthy_profile()
        )
        assert result == GateResult.PASS
        assert match == "healthy"

    def test_depression_profile_flags(self, screener):
        # Use the depression prototype values as the user's raw data
        from system2.config import DISORDER_PROTOTYPES_FRAME1
        dep = DISORDER_PROTOTYPES_FRAME1["depression"]
        result, match, score = screener.gate3_prototype_proximity(dep)
        assert result == GateResult.CONTAMINATED_BASELINE
        assert match == "depression"


# ── Combined Screener ──────────────────────────────────────────────────

class TestScreenCombined:
    def test_all_pass_locks_baseline(self, screener):
        profile = _healthy_profile()
        result = screener.screen(profile, [profile, profile, profile], profile)
        assert result.passed is True
        assert result.recommended_action == RecommendedAction.LOCK_BASELINE
        assert result.frame == 2

    def test_gate3_replaces_baseline(self, screener):
        from system2.config import DISORDER_PROTOTYPES_FRAME1
        dep = DISORDER_PROTOTYPES_FRAME1["depression"]
        healthy = _healthy_profile()
        result = screener.screen(healthy, [healthy, healthy, healthy], dep)
        assert "gate3" in result.gates_fired
        assert result.frame == 1

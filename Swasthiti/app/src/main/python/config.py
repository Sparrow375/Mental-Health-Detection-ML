"""
System 2 Configuration
======================

Central source of truth for:
  - Behavioral feature definitions (16 features, voice + SMS excluded)
  - Population norms (healthy mean/std from StudentLife + literature)
  - Disorder prototype vectors — Frame 1 (absolute) & Frame 2 (z-scores)
  - Feature diagnostic weights
  - Confidence thresholds
  - Temporal shape detection parameters & compatibility matrix
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List

# ── Path helpers ─────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent / "data"

# ── 1. Behavioral Feature Set (16 features, voice + SMS excluded) ───────
BEHAVIORAL_FEATURES: List[str] = [
    "screen_time_hours",
    "unlock_count",
    "social_app_ratio",
    "calls_per_day",
    "unique_contacts",
    "daily_displacement_km",
    "location_entropy",
    "home_time_ratio",
    "places_visited",
    "wake_time_hour",
    "sleep_time_hour",
    "sleep_duration_hours",
    "dark_duration_hours",
    "charge_duration_hours",
    "conversation_duration_hours",
    "conversation_frequency",
]

# ── 2. Population Norms (healthy baseline) ──────────────────────────────
# CALIBRATED from StudentLife healthy cohort (PHQ-9 < 5, N=27).
# Each entry: {"mean": float, "std": float}  (healthy population reference)
# SMS features (texts_per_day, response_time_minutes) removed — not
# available on modern Android; their absence injected ghost z-scores.
POPULATION_NORMS: Dict[str, Dict[str, float]] = {
    "screen_time_hours":          {"mean": 11.3,  "std": 1.3},
    "unlock_count":               {"mean": 3.6,   "std": 1.5},
    "social_app_ratio":           {"mean": 0.08,  "std": 0.05},
    "calls_per_day":              {"mean": 31.8,  "std": 1.8},
    "unique_contacts":            {"mean": 7.2,   "std": 3.5},
    "daily_displacement_km":      {"mean": 11.0,  "std": 2.0},
    "location_entropy":           {"mean": 1.5,   "std": 0.7},
    "home_time_ratio":            {"mean": 0.52,  "std": 0.12},
    "places_visited":             {"mean": 8.1,   "std": 1.5},
    "wake_time_hour":             {"mean": 6.5,   "std": 1.0},
    "sleep_time_hour":            {"mean": 10.9,  "std": 1.0},
    "sleep_duration_hours":       {"mean": 5.8,   "std": 0.9},
    "dark_duration_hours":        {"mean": 9.6,   "std": 1.2},
    "charge_duration_hours":      {"mean": 6.2,   "std": 1.0},
    "conversation_duration_hours": {"mean": 4.9,  "std": 0.6},
    "conversation_frequency":     {"mean": 28.4,  "std": 2.5},
}

# Expected week-over-week drift for healthy individuals (used by Gate 2).
# Represents the normal SD of weekly means across a 3-week window.
POPULATION_EXPECTED_DRIFT: Dict[str, float] = {
    "screen_time_hours":          0.4,
    "unlock_count":               6.0,
    "social_app_ratio":           0.03,
    "calls_per_day":              0.6,
    "unique_contacts":            1.2,
    "daily_displacement_km":      0.8,
    "location_entropy":           0.2,
    "home_time_ratio":            0.04,
    "places_visited":             0.5,
    "wake_time_hour":             0.3,
    "sleep_time_hour":            0.3,
    "sleep_duration_hours":       0.3,
    "dark_duration_hours":        0.4,
    "charge_duration_hours":      0.3,
    "conversation_duration_hours": 0.2,
    "conversation_frequency":     0.8,
}

# ── 3. Disorder Prototypes — Frame 1 (Absolute Values) ─────────────────
# Population-anchored.  Used during baseline screening (Days 1-28).
# Sources: StudentLife high-PHQ-9, schizophrenia dataset, DSM-5 + literature.
DISORDER_PROTOTYPES_FRAME1: Dict[str, Dict[str, float]] = {
    # CALIBRATED from StudentLife healthy cohort (PHQ-9 < 5, N=27)
    "healthy": {
        "screen_time_hours": 11.33,
        "unlock_count": 3.58,
        "social_app_ratio": 0.08,
        "calls_per_day": 31.75,
        "unique_contacts": 7.15,
        "daily_displacement_km": 11.03,
        "location_entropy": 1.51,
        "home_time_ratio": 0.52,
        "places_visited": 8.10,
        "wake_time_hour": 6.47,
        "sleep_time_hour": 10.86,
        "sleep_duration_hours": 5.80,
        "dark_duration_hours": 9.60,
        "charge_duration_hours": 6.19,
        "conversation_duration_hours": 4.93,
        "conversation_frequency": 28.38,
    },
    "depression_insomnia": {
        "screen_time_hours": 12.06,
        "unlock_count": 3.91,
        "social_app_ratio": 0.07,
        "calls_per_day": 18.47,
        "unique_contacts": 1.81,
        "daily_displacement_km": 6.43,
        "location_entropy": 1.32,
        "home_time_ratio": 0.65,
        "places_visited": 7.96,
        "wake_time_hour": 5.22,
        "sleep_time_hour": 10.73,
        "sleep_duration_hours": 4.0,  # Insomnia pattern
        "dark_duration_hours": 10.63,
        "charge_duration_hours": 6.89,
        "conversation_duration_hours": 3.89,
        "conversation_frequency": 21.97,
    },
    "depression_hypersomnia": {
        "screen_time_hours": 12.06,
        "unlock_count": 3.91,
        "social_app_ratio": 0.07,
        "calls_per_day": 18.47,
        "unique_contacts": 1.81,
        "daily_displacement_km": 6.43,
        "location_entropy": 1.32,
        "home_time_ratio": 0.65,
        "places_visited": 7.96,
        "wake_time_hour": 9.22,   # Late wake time
        "sleep_time_hour": 10.73,
        "sleep_duration_hours": 10.0,  # Hypersomnia pattern
        "dark_duration_hours": 10.63,
        "charge_duration_hours": 6.89,
        "conversation_duration_hours": 3.89,
        "conversation_frequency": 21.97,
    },
    # CALIBRATED from CrossCheck dataset schz-positive group (N=30, EMA VOICES+SEEING > 0.5)
    "schizophrenia": {
        "screen_time_hours": 2.97,     # lower than non-schz (4.0h)
        "unlock_count": 57.2,
        "social_app_ratio": 0.48,      # similar to non-schz
        "calls_per_day": 7.2,
        "unique_contacts": 7.2,
        "daily_displacement_km": 3.5,  # literature-based (GPS data unreliable)
        "location_entropy": 1.0,       # more constrained
        "home_time_ratio": 0.75,       # more homebound
        "places_visited": 3.1,
        "wake_time_hour": 2.2,         # CrossCheck mean (early waking)
        "sleep_time_hour": 0.3,
        "sleep_duration_hours": 9.0,   # CrossCheck mean
        "dark_duration_hours": 19.8,
        "charge_duration_hours": 6.2,  # same as population norms
        "conversation_duration_hours": 4.1,
        "conversation_frequency": 20.0,
    },
    "bpd": {
        "screen_time_hours": 5.0,
        "unlock_count": 70.0,           # frequent checking
        "social_app_ratio": 0.30,       # variable — swings
        "calls_per_day": 5.0,           # bursts
        "unique_contacts": 6.0,
        "daily_displacement_km": 4.0,   # variable
        "location_entropy": 2.5,
        "home_time_ratio": 0.55,
        "places_visited": 4.0,
        "wake_time_hour": 8.0,
        "sleep_time_hour": 0.5,
        "sleep_duration_hours": 6.0,
        "dark_duration_hours": 6.5,
        "charge_duration_hours": 3.5,
        "conversation_duration_hours": 1.0,
        "conversation_frequency": 4.0,
    },
    "bipolar_depressive": {
        "screen_time_hours": 5.0,
        "unlock_count": 28.0,
        "social_app_ratio": 0.09,
        "calls_per_day": 1.2,
        "unique_contacts": 2.5,
        "daily_displacement_km": 1.5,
        "location_entropy": 1.0,
        "home_time_ratio": 0.85,
        "places_visited": 2.0,
        "wake_time_hour": 10.0,
        "sleep_time_hour": 1.5,
        "sleep_duration_hours": 9.0,
        "dark_duration_hours": 10.5,
        "charge_duration_hours": 5.0,
        "conversation_duration_hours": 0.2,
        "conversation_frequency": 1.0,
    },
    "bipolar_manic": {
        "screen_time_hours": 7.5,
        "unlock_count": 110.0,
        "social_app_ratio": 0.55,
        "calls_per_day": 8.0,
        "unique_contacts": 15.0,
        "daily_displacement_km": 7.0,
        "location_entropy": 3.8,
        "home_time_ratio": 0.25,
        "places_visited": 8.0,
        "wake_time_hour": 5.0,
        "sleep_time_hour": 2.0,
        "sleep_duration_hours": 3.5,
        "dark_duration_hours": 3.5,
        "charge_duration_hours": 1.5,
        "conversation_duration_hours": 3.0,
        "conversation_frequency": 12.0,
    },
    "anxiety": {
        "screen_time_hours": 5.2,
        "unlock_count": 75.0,           # checking behaviour
        "social_app_ratio": 0.18,
        "calls_per_day": 2.0,
        "unique_contacts": 5.0,
        "daily_displacement_km": 2.1,
        "location_entropy": 1.8,
        "home_time_ratio": 0.72,
        "places_visited": 3.0,
        "wake_time_hour": 6.5,
        "sleep_time_hour": 0.0,
        "sleep_duration_hours": 5.8,
        "dark_duration_hours": 6.0,
        "charge_duration_hours": 3.5,
        "conversation_duration_hours": 0.8,
        "conversation_frequency": 3.0,
    },
}

# ── 4. Disorder Prototypes — Frame 2 (Z-Scores from Clean Baseline) ────
# Personal-baseline-anchored.  Used during ongoing monitoring (Day 28+).
# Values = expected standard deviations from the user's verified-clean baseline.
DISORDER_PROTOTYPES_FRAME2: Dict[str, Dict[str, float]] = {
    "healthy": {f: 0.0 for f in BEHAVIORAL_FEATURES},
    # --- DEPRESSION SUBTYPES ---
    "depression_type_1": {
        "screen_time_hours": -0.60,
        "unlock_count": -0.42,
        "social_app_ratio": -0.03,
        "calls_per_day": -1.23,
        "texts_per_day": -1.24,
        "unique_contacts": -0.11,
        "response_time_minutes": -0.05,
        "daily_displacement_km": -0.43,
        "location_entropy": 0.18,
        "home_time_ratio": -0.18,
        "places_visited": 0.04,
        "wake_time_hour": -0.29,
        "sleep_time_hour": 0.51,
        "sleep_duration_hours": -0.87,
        "dark_duration_hours": 0.17,
        "charge_duration_hours": -0.03,
        "conversation_duration_hours": -0.55,
        "conversation_frequency": -0.81,
    },
    "depression_type_2": {
        "screen_time_hours": 5.00,
        "unlock_count": 5.00,
        "social_app_ratio": 0.69,
        "calls_per_day": 0.00,
        "texts_per_day": 5.00,
        "unique_contacts": 0.00,
        "response_time_minutes": 0.00,
        "daily_displacement_km": 0.00,
        "location_entropy": 3.41,
        "home_time_ratio": 0.00,
        "places_visited": 3.50,
        "wake_time_hour": -2.37,
        "sleep_time_hour": 2.50,
        "sleep_duration_hours": 0.22,
        "dark_duration_hours": 5.00,
        "charge_duration_hours": 0.00,
        "conversation_duration_hours": 2.50,
        "conversation_frequency": 2.50,
    },
    "depression_type_3": {
        "screen_time_hours": 5.00,
        "unlock_count": 5.00,
        "social_app_ratio": 2.20,
        "calls_per_day": 5.00,
        "texts_per_day": 1.06,
        "unique_contacts": 5.00,
        "response_time_minutes": -0.40,
        "daily_displacement_km": 5.00,
        "location_entropy": 1.06,
        "home_time_ratio": 0.00,
        "places_visited": 1.04,
        "wake_time_hour": -1.07,
        "sleep_time_hour": 0.48,
        "sleep_duration_hours": 0.80,
        "dark_duration_hours": -5.00,
        "charge_duration_hours": 0.00,
        "conversation_duration_hours": -0.03,
        "conversation_frequency": 1.84,
    },

    # --- SCHIZOPHRENIA SUBTYPES ---
    "schizophrenia_type_1": {
        "screen_time_hours": -0.08,
        "unlock_count": -0.14,
        "social_app_ratio": -0.16,
        "calls_per_day": -0.20,
        "texts_per_day": -0.20,
        "unique_contacts": -0.20,
        "response_time_minutes": -0.19,
        "daily_displacement_km": 0.04,
        "location_entropy": 1.15,
        "home_time_ratio": 0.00,
        "places_visited": 1.04,
        "wake_time_hour": -0.17,
        "sleep_time_hour": 1.39,
        "sleep_duration_hours": 0.35,
        "dark_duration_hours": 0.17,
        "charge_duration_hours": 0.00,
        "conversation_duration_hours": 0.47,
        "conversation_frequency": 0.18,
    },
    "schizophrenia_type_2": {
        "screen_time_hours": 4.01,
        "unlock_count": 4.83,
        "social_app_ratio": 3.24,
        "calls_per_day": 0.00,
        "texts_per_day": -0.13,
        "unique_contacts": 0.00,
        "response_time_minutes": 0.00,
        "daily_displacement_km": 1.67,
        "location_entropy": 2.77,
        "home_time_ratio": 0.00,
        "places_visited": 2.68,
        "wake_time_hour": -2.21,
        "sleep_time_hour": 0.00,
        "sleep_duration_hours": 3.11,
        "dark_duration_hours": 5.00,
        "charge_duration_hours": 0.00,
        "conversation_duration_hours": 3.25,
        "conversation_frequency": 3.25,
    },
    "schizophrenia_type_3": {
        "screen_time_hours": 5.00,
        "unlock_count": 2.49,
        "social_app_ratio": 1.14,
        "calls_per_day": 0.20,
        "texts_per_day": -0.09,
        "unique_contacts": 0.20,
        "response_time_minutes": -0.19,
        "daily_displacement_km": 0.99,
        "location_entropy": 1.28,
        "home_time_ratio": 0.00,
        "places_visited": 1.17,
        "wake_time_hour": -1.30,
        "sleep_time_hour": 2.75,
        "sleep_duration_hours": -2.92,
        "dark_duration_hours": -4.61,
        "charge_duration_hours": 0.00,
        "conversation_duration_hours": 1.67,
        "conversation_frequency": 1.46,
    },

    # --- HEALTHY SUBTYPES ---
    "healthy_type_1": {
        "screen_time_hours": 0.75,
        "unlock_count": 1.42,
        "social_app_ratio": 0.80,
        "calls_per_day": 0.49,
        "texts_per_day": 1.04,
        "unique_contacts": 0.53,
        "response_time_minutes": -0.11,
        "daily_displacement_km": 0.17,
        "location_entropy": 1.60,
        "home_time_ratio": -0.00,
        "places_visited": 1.50,
        "wake_time_hour": 0.13,
        "sleep_time_hour": 2.16,
        "sleep_duration_hours": 2.29,
        "dark_duration_hours": 1.39,
        "charge_duration_hours": 0.00,
        "conversation_duration_hours": -0.02,
        "conversation_frequency": 0.05,
    },
    "healthy_type_2": {
        "screen_time_hours": 2.57,
        "unlock_count": 3.49,
        "social_app_ratio": 2.98,
        "calls_per_day": 0.27,
        "texts_per_day": 1.05,
        "unique_contacts": 0.27,
        "response_time_minutes": -0.05,
        "daily_displacement_km": 0.87,
        "location_entropy": 1.73,
        "home_time_ratio": -0.00,
        "places_visited": 1.75,
        "wake_time_hour": -1.00,
        "sleep_time_hour": 1.82,
        "sleep_duration_hours": 0.63,
        "dark_duration_hours": 0.27,
        "charge_duration_hours": -0.00,
        "conversation_duration_hours": 4.49,
        "conversation_frequency": 4.69,
    },
    "healthy_type_3": {
        "screen_time_hours": -0.43,
        "unlock_count": -0.48,
        "social_app_ratio": 0.21,
        "calls_per_day": -1.15,
        "texts_per_day": -0.88,
        "unique_contacts": -0.12,
        "response_time_minutes": -0.12,
        "daily_displacement_km": -0.40,
        "location_entropy": -0.35,
        "home_time_ratio": -0.03,
        "places_visited": -0.33,
        "wake_time_hour": 0.04,
        "sleep_time_hour": 0.05,
        "sleep_duration_hours": 0.10,
        "dark_duration_hours": 0.02,
        "charge_duration_hours": -0.05,
        "conversation_duration_hours": -0.28,
        "conversation_frequency": -0.15,
    },

    # NOTE: Heuristic uncalibrated prototypes (bipolar, bpd, anxiety) have been 
    # temporarily disabled. Because they are not derived from empirical averages,
    # their extreme theoretical values act as geometric "sinkholes" that intercept
    # extreme true-positive Schizophrenia or extreme situational_stress anomalies.
    # To re-add them, they must be calibrated mathematically from real S1 deviations.
}

# ── 5. Feature Diagnostic Weights ──────────────────────────────────────
# CALIBRATED: weights reflect real discriminative power from StudentLife.
# Higher weight = larger real z-score difference (depressed vs healthy).
# Range [0, 1].  Used in Weighted Euclidean distance (Phase 2).
# SMS features removed — weights redistributed to call/contact features.
FEATURE_WEIGHTS: Dict[str, float] = {
    "screen_time_hours":          0.5,   # real z=+0.56, moderate; good coverage
    "unlock_count":               0.2,   # real z=+0.02, no signal
    "social_app_ratio":           0.1,   # real z=-0.14, no signal
    "calls_per_day":              0.6,   # bumped from 0.5; absorbs texts signal
    "unique_contacts":            0.5,   # bumped from 0.4; absorbs texts signal
    "daily_displacement_km":      0.9,   # real z=-2.30, STRONG signal; good coverage
    "location_entropy":           0.4,   # real z=-0.27, moderate
    "home_time_ratio":            0.7,   # real z=+1.07, good signal
    "places_visited":             0.2,   # real z=-0.09, no signal
    "wake_time_hour":             0.7,   # real z=-1.24, good signal
    "sleep_time_hour":            0.1,   # real z=-0.13, no signal
    "sleep_duration_hours":       0.1,   # real z=-0.13, no signal
    "dark_duration_hours":        0.6,   # real z=+0.86, moderate; good coverage
    "charge_duration_hours":      0.3,   # real z=+0.70, moderate
    "conversation_duration_hours": 0.9,  # real z=-1.73, STRONG signal
    "conversation_frequency":     0.9,   # real z=-2.56, STRONGEST signal
}

# ── 6. Confidence Thresholds ───────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "high":          0.75,   # ≥ 0.75 → high-confidence classification
    "low":           0.55,   # 0.55–0.75 → "Possible [X]"
    "unclassified":  0.55,   # < 0.55 → "Unclassified — escalate"
}

# ── 7. Gate Screening Parameters ──────────────────────────────────────
GATE_PARAMS = {
    "gate1_z_threshold":       2.5,   # population z-score threshold
    "gate1_min_features":      3,     # features that must exceed threshold
    "gate2_drift_multiplier":  1.5,   # drift must be > N× expected
    "gate2_min_features":      3,
    "gate3_healthy_threshold": 0.65,  # confidence above which flag
}

# ── 8. Temporal Shape Detection Parameters ─────────────────────────────
TEMPORAL_SHAPES = {
    "monotonic_drift": {
        "min_slope": -0.02,       # linear regression slope
        "min_r_squared": 0.6,
    },
    "oscillating": {
        "autocorr_lag_min": 3,    # days
        "autocorr_lag_max": 10,
        "autocorr_threshold": 0.4,
    },
    "chaotic": {
        "min_variance_ratio": 2.0,   # vs population expected
        "max_autocorr": 0.2,
    },
    "episodic_spike": {
        "spike_sd_threshold": 2.0,
        "recovery_window_days": 14,
    },
    "phase_flip": {
        "diff_sd_threshold": 3.0,    # between consecutive weeks
    },
}

# ── 9. Shape–Disorder Compatibility Matrix ─────────────────────────────
# Values: +1 = supports,  0 = neutral,  -1 = contradicts.
# Covers BOTH parent disorder names AND all Frame 2 subtype names so that
# temporal boost/downgrade fires correctly for every classification output.
_DEP = 1    # supports depression pattern
_SCHZ = 1   # supports schizophrenia pattern
SHAPE_DISORDER_MATRIX: Dict[str, Dict[str, int]] = {
    "monotonic_drift": {
        # parent names (Frame 1)
        "healthy": -1, "depression": 1, "schizophrenia": 0,
        "bpd": -1, "bipolar_depressive": 1, "bipolar_manic": -1, "anxiety": 0,
        # Frame 2 subtypes
        "healthy_type_1": -1, "healthy_type_2": -1, "healthy_type_3": -1,
        "depression_type_1": 1, "depression_type_2": 1, "depression_type_3": 1,
        "schizophrenia_type_1": 0, "schizophrenia_type_2": 0, "schizophrenia_type_3": 0,
    },
    "oscillating": {
        "healthy": -1, "depression": -1, "schizophrenia": 0,
        "bpd": 1, "bipolar_depressive": -1, "bipolar_manic": 0, "anxiety": 0,
        "healthy_type_1": -1, "healthy_type_2": -1, "healthy_type_3": -1,
        "depression_type_1": -1, "depression_type_2": -1, "depression_type_3": -1,
        "schizophrenia_type_1": 0, "schizophrenia_type_2": 0, "schizophrenia_type_3": 0,
    },
    "chaotic": {
        "healthy": -1, "depression": -1, "schizophrenia": 1,
        "bpd": 0, "bipolar_depressive": -1, "bipolar_manic": 0, "anxiety": 0,
        "healthy_type_1": -1, "healthy_type_2": -1, "healthy_type_3": -1,
        "depression_type_1": -1, "depression_type_2": -1, "depression_type_3": -1,
        "schizophrenia_type_1": 1, "schizophrenia_type_2": 1, "schizophrenia_type_3": 1,
    },
    "episodic_spike": {
        "healthy": 0, "depression": -1, "schizophrenia": -1,
        "bpd": 0, "bipolar_depressive": -1, "bipolar_manic": 0, "anxiety": 1,
        "healthy_type_1": 0, "healthy_type_2": 0, "healthy_type_3": 0,
        "depression_type_1": -1, "depression_type_2": -1, "depression_type_3": -1,
        "schizophrenia_type_1": -1, "schizophrenia_type_2": -1, "schizophrenia_type_3": -1,
    },
    "phase_flip": {
        "healthy": -1, "depression": -1, "schizophrenia": 0,
        "bpd": 0, "bipolar_depressive": 0, "bipolar_manic": 1, "anxiety": -1,
        "healthy_type_1": -1, "healthy_type_2": -1, "healthy_type_3": -1,
        "depression_type_1": -1, "depression_type_2": -1, "depression_type_3": -1,
        "schizophrenia_type_1": 0, "schizophrenia_type_2": 0, "schizophrenia_type_3": 0,
    },
}

# ── 10. Confidence Adjustment Factors ──────────────────────────────────
TEMPORAL_BOOST  = 1.2     # multiply when shape confirms
TEMPORAL_DOWNGRADE = 0.6  # multiply when shape contradicts

# ── 11. Life-Event Filter Parameters ──────────────────────────────────
LIFE_EVENT_PARAMS = {
    "max_co_deviating_features": 3,     # ≤ 3 features → likely situational
                                        # raised from 2: schizophrenia often presents
                                        # as 2-3 focused deviations, not broad spread
    "self_resolve_days":         10,     # resolved within N days → dismiss
    "severity_floor_sd":         1.0,   # no feature exceeds 1.0 SD → too mild to classify
}


# ── Helper: load population norms from JSON ────────────────────────────
def load_population_norms_json(path: pathlib.Path | None = None) -> Dict:
    """Load population norms from the JSON file in data/."""
    path = path or (_DATA_DIR / "population_norms.json")
    with open(path, "r") as f:
        return json.load(f)


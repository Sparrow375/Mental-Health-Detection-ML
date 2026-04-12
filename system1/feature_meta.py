"""
Feature metadata: weights, critical features, L1/L2 feature lists, and thresholds.
"""

from typing import Dict, List, Tuple

# ── All 29 L1 features with clinical weights ──────────────────────────────────
# Weight ranges from 0.4 (quasi-static) to 1.6 (highest clinical priority)

FEATURE_META: Dict[str, Dict[str, float]] = {
    # Group A — Screen & App Activity
    "screenTimeHours":        {"weight": 1.4, "group": "screen_app"},
    "unlockCount":            {"weight": 1.2, "group": "screen_app"},
    "appLaunchCount":         {"weight": 0.9, "group": "screen_app"},
    "notificationsToday":     {"weight": 0.8, "group": "screen_app"},
    "socialAppRatio":         {"weight": 1.3, "group": "screen_app"},

    # Group B — Communication
    "callsPerDay":            {"weight": 1.3, "group": "communication"},
    "callDurationMinutes":    {"weight": 1.2, "group": "communication"},
    "uniqueContacts":         {"weight": 1.1, "group": "communication"},
    "conversationFrequency":  {"weight": 0.9, "group": "communication"},

    # Group C — Location & Movement
    "dailyDisplacementKm":    {"weight": 1.5, "group": "location"},
    "locationEntropy":        {"weight": 1.3, "group": "location"},
    "homeTimeRatio":          {"weight": 1.2, "group": "location"},
    "placesVisited":          {"weight": 1.1, "group": "location"},

    # Group D — Sleep & Circadian
    "wakeTimeHour":           {"weight": 1.4, "group": "sleep"},
    "sleepTimeHour":          {"weight": 1.3, "group": "sleep"},
    "sleepDurationHours":     {"weight": 1.6, "group": "sleep"},
    "darkDurationHours":      {"weight": 1.0, "group": "sleep"},

    # Group E — System Usage
    "chargeDurationHours":    {"weight": 0.8, "group": "system"},
    "memoryUsagePercent":     {"weight": 0.5, "group": "system"},
    "networkWifiMB":          {"weight": 0.6, "group": "system"},
    "networkMobileMB":        {"weight": 0.6, "group": "system"},
    "storageUsedGB":          {"weight": 0.4, "group": "system"},

    # Group F — Behavioural Signals
    "totalAppsCount":         {"weight": 0.8, "group": "behavioral"},
    "upiTransactionsToday":   {"weight": 1.1, "group": "behavioral"},
    "appUninstallsToday":     {"weight": 0.9, "group": "behavioral"},
    "appInstallsToday":       {"weight": 0.8, "group": "behavioral"},

    # Group G — Calendar & Engagement
    "calendarEventsToday":    {"weight": 0.9, "group": "engagement"},
    "mediaCountToday":        {"weight": 0.7, "group": "engagement"},
    "downloadsToday":         {"weight": 0.6, "group": "engagement"},
    "backgroundAudioHours":   {"weight": 1.1, "group": "engagement"},
}

# Extra feature present in Android but not in the 29-feature anomaly model
EXTRA_FEATURES = ["dailySteps"]

# Canonical ordered list of all 29 L1 features
ALL_L1_FEATURES: List[str] = list(FEATURE_META.keys())

# ── Critical features for alert determination ──────────────────────────────────
CRITICAL_FEATURES: List[str] = [
    "sleepDurationHours",
    "dailyDisplacementKm",
    "screenTimeHours",
    "socialAppRatio",
    "wakeTimeHour",
    "callsPerDay",
    "upiTransactionsToday",
    "totalAppsCount",
]

# ── 12 features used for L1 DBSCAN clustering ─────────────────────────────────
L1_CLUSTER_FEATURES: List[str] = [
    "sleepDurationHours",
    "wakeTimeHour",
    "sleepTimeHour",
    "dailyDisplacementKm",
    "locationEntropy",
    "placesVisited",
    "callsPerDay",
    "callDurationMinutes",
    "screenTimeHours",
    "unlockCount",
    "socialAppRatio",
    "darkDurationHours",
]

# ── 22-feature L2 texture vector ──────────────────────────────────────────────
L2_TEXTURE_FEATURES: List[str] = [
    # Temporal anchoring (4)
    "time_in_primary_window_ratio",
    "temporal_anchor_deviation",
    "first_pickup_hour_deviation",
    "rhythm_dissolution_score",
    # Session quality (5)
    "weighted_abandon_rate",
    "deep_session_ratio",
    "micro_session_ratio",
    "session_duration_collapse",
    "interaction_density_ratio",
    # Agency & initiation (4)
    "self_open_ratio",
    "notification_open_rate",
    "notification_ignore_rate",
    "pickup_burst_rate",
    # Attention coherence (4)
    "app_switching_rate",
    "app_cooccurrence_consistency",
    "distinct_apps_ratio",
    "session_context_match",
    # Rhythm & structure (3)
    "daily_rhythm_regularity",
    "weekday_weekend_alignment",
    "dead_zone_count",
    # Notification relationship (2)
    "notification_response_latency_shift",
    "notification_to_session_ratio",
]

# ── Threshold constants ────────────────────────────────────────────────────────
THRESHOLDS = {
    "ANOMALY_SCORE_THRESHOLD": 0.38,       # Fixed, never calibrated
    "SUSTAINED_THRESHOLD_DAYS": 5,          # Min consecutive anomalous days for alert
    "EVIDENCE_THRESHOLD": 2.0,              # Min evidence for alert
    "PEAK_EVIDENCE_THRESHOLD": 7.0,         # Retrospective, calibrated per person
    "PEAK_SUSTAINED_THRESHOLD_DAYS": 10,    # Retrospective, calibrated per person
    "WATCH_EVIDENCE_THRESHOLD": 1.5,        # WATCH recommendation threshold
    "L2_CANDIDATE_TEXTURE_THRESHOLD": 0.35, # Session incoherence threshold for candidate
    "L2_COHERENCE_MATCH_RADIUS_FACTOR": 1.5,# Multiplier on cluster radius
    "MIN_APP_BASELINE_APPEARANCES": 3,      # Minimum sessions for AppDNA
    "MIN_ARCHETYPE_DAYS_FOR_KMEANS": 10,    # Minimum days for K-means texture
    "DBSCAN_MIN_SAMPLES": 3,
    "CANDIDATE_WINDOW_LENGTH": 7,           # Days for candidate evaluation
    "EVIDENCE_DECAY_RATE": 0.92,            # Daily decay on normal days
    "EWMA_ALPHA": 0.4,                      # EWMA smoothing factor
    "EWMA_WINDOW": 7,                       # Days for velocity computation
    "EVIDENCE_COMPOUND_FACTOR": 0.1,        # Sustained days multiplier
}

# ── Feature weight shortcut ───────────────────────────────────────────────────
def get_feature_weight(feature: str) -> float:
    """Return the clinical weight for a feature, defaulting to 1.0."""
    return FEATURE_META.get(feature, {}).get("weight", 1.0)

def get_feature_group(feature: str) -> str:
    """Return the group name for a feature."""
    return FEATURE_META.get(feature, {}).get("group", "unknown")
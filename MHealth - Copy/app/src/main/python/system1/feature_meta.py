"""
Feature metadata: clinical weights, feature groups, critical features, and feature lists.

Adapted for MHealth Android app — uses camelCase feature names matching
Android's PersonalityVector.toMap() keys exactly.

Groups:
  A  Screen & App Activity      (5 features)
  B  Communication              (4 features)
  C  Location & Movement        (3 features)
  D  Sleep & Circadian          (4 features)
  E  System Usage               (5 features)
  F  Behavioural Signals        (4 features)
  G  Calendar & Engagement      (4 features)

Total: 29 real-valued features per day.
"""

from typing import Dict, List


# ============================================================================
# FEATURE WEIGHTS  (clinical importance)
# ============================================================================
# Weight range: 0.4 (quasi-static, e.g. storage) → 1.6 (highest clinical, e.g. sleep)
#
# Directionality (asymmetric scoring):
#   shrink_matters — decrease from baseline is clinically significant; increase is noise
#   grow_matters   — increase from baseline is clinically significant; decrease is noise
#   u_shaped       — deviation in EITHER direction matters (e.g. too-much/too-little sleep)
#   both           — no asymmetry (default)

DIRECTIONALITY_DAMPENING = 0.15

FEATURE_META: Dict[str, Dict] = {
    # ── Group A: Screen & App Activity ────────────────────────────────────────
    "screenTimeHours":       {"group": "screen",        "weight": 1.4, "directionality": "grow_matters"},
    "unlockCount":           {"group": "screen",        "weight": 1.2, "directionality": "grow_matters"},
    "appLaunchCount":        {"group": "screen",        "weight": 0.9, "directionality": "both"},
    "notificationsToday":    {"group": "screen",        "weight": 0.8, "directionality": "grow_matters"},
    "socialAppRatio":        {"group": "screen",        "weight": 1.3, "directionality": "grow_matters"},

    # ── Group B: Communication ─────────────────────────────────────────────
    "callsPerDay":           {"group": "communication", "weight": 1.3, "directionality": "shrink_matters"},
    "callDurationMinutes":   {"group": "communication", "weight": 1.2, "directionality": "shrink_matters"},
    "uniqueContacts":        {"group": "communication", "weight": 1.1, "directionality": "shrink_matters"},
    "conversationFrequency": {"group": "communication", "weight": 0.9, "directionality": "shrink_matters"},

    # ── Group C: Location & Movement ──────────────────────────────────────
    "dailyDisplacementKm":   {"group": "movement",      "weight": 1.5, "directionality": "shrink_matters"},
    "locationEntropy":       {"group": "movement",      "weight": 1.3, "directionality": "shrink_matters"},
    "homeTimeRatio":         {"group": "movement",      "weight": 1.2, "directionality": "grow_matters"},

    # ── Group D: Sleep & Circadian ────────────────────────────────────────
    "wakeTimeHour":          {"group": "sleep",         "weight": 1.4, "directionality": "grow_matters"},
    "sleepTimeHour":         {"group": "sleep",         "weight": 1.3, "directionality": "grow_matters"},
    "sleepDurationHours":    {"group": "sleep",         "weight": 1.6, "directionality": "u_shaped"},
    "darkDurationHours":     {"group": "sleep",         "weight": 1.0, "directionality": "u_shaped"},

    # ── Group E: System Usage ─────────────────────────────────────────────
    "chargeDurationHours":   {"group": "system",        "weight": 0.8, "directionality": "both"},
    "memoryUsagePercent":    {"group": "system",        "weight": 0.5, "directionality": "both"},
    "networkWifiMB":         {"group": "system",        "weight": 0.6, "directionality": "both"},
    "networkMobileMB":       {"group": "system",        "weight": 0.6, "directionality": "both"},
    "storageUsedGB":         {"group": "system",        "weight": 0.4, "directionality": "both"},

    # ── Group F: Behavioural Signals ──────────────────────────────────────
    "totalAppsCount":        {"group": "behaviour",     "weight": 0.8, "directionality": "shrink_matters"},
    "upiTransactionsToday":  {"group": "behaviour",     "weight": 1.1, "directionality": "shrink_matters"},
    "appUninstallsToday":    {"group": "behaviour",     "weight": 0.9, "directionality": "shrink_matters"},
    "appInstallsToday":      {"group": "behaviour",     "weight": 0.8, "directionality": "shrink_matters"},

    # ── Group G: Calendar & Engagement ────────────────────────────────────
    "calendarEventsToday":   {"group": "engagement",    "weight": 0.9, "directionality": "shrink_matters"},
    "mediaCountToday":       {"group": "engagement",    "weight": 0.7, "directionality": "shrink_matters"},
    "downloadsToday":        {"group": "engagement",    "weight": 0.6, "directionality": "shrink_matters"},
    "backgroundAudioHours":  {"group": "engagement",    "weight": 0.9, "directionality": "both"},
}

# Ordered list of all feature names (camelCase, matching Android)
ALL_L1_FEATURES: List[str] = list(FEATURE_META.keys())

# Features the alert engine treats as high-priority for escalation decisions
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

# Feature subset used for L1 DBSCAN context clustering
L1_CLUSTERING_FEATURES: List[str] = [
    "sleepDurationHours",
    "wakeTimeHour",
    "sleepTimeHour",
    "dailyDisplacementKm",
    "locationEntropy",
    "callsPerDay",
    "conversationFrequency",
    "screenTimeHours",
    "unlockCount",
    "socialAppRatio",
    "darkDurationHours",
]

# ============================================================================
# L2 TEXTURE VECTOR  (22 features — derived from session & notification events)
# ============================================================================

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

# Feature group mapping for the 22 L2 features
L2_FEATURE_GROUPS: Dict[str, List[str]] = {
    "Temporal Anchoring": L2_TEXTURE_FEATURES[0:4],
    "Session Quality": L2_TEXTURE_FEATURES[4:9],
    "Agency & Initiation": L2_TEXTURE_FEATURES[9:13],
    "Attention Coherence": L2_TEXTURE_FEATURES[13:17],
    "Rhythm & Structure": L2_TEXTURE_FEATURES[17:20],
    "Notification Relationship": L2_TEXTURE_FEATURES[20:22],
}

# ============================================================================
# THRESHOLD DEFAULTS
# ============================================================================

DEFAULT_THRESHOLDS = {
    # Daily scoring gate — fixed for all users
    "ANOMALY_SCORE_THRESHOLD": 0.38,

    # Real-time alerting
    "SUSTAINED_THRESHOLD_DAYS": 5,
    "EVIDENCE_THRESHOLD": 2.0,

    # Retrospective (peak-based)
    "PEAK_EVIDENCE_THRESHOLD": 7.0,
    "PEAK_SUSTAINED_THRESHOLD_DAYS": 10,

    # Secondary watch flag
    "WATCH_EVIDENCE_THRESHOLD": 1.5,

    # Evidence accumulation
    "EVIDENCE_DECAY_RATE": 0.88,
    "EVIDENCE_COMPOUNDING": 0.15,

    # AND gate: min co-deviating features for episode confirmation
    "MIN_BREADTH_FOR_EPISODE": 3,

    # Evidence ratio: monitoring_peak / baseline_peak must exceed this
    "EVIDENCE_RATIO_THRESHOLD": 1.5,

    # Candidate cluster evaluation
    "CANDIDATE_WINDOW_DAYS": 7,
    "CANDIDATE_TEXTURE_THRESHOLD": 0.35,

    # L2 coherence
    "COHERENCE_MATCH_RADIUS_FACTOR": 1.5,

    # DBSCAN
    "DBSCAN_MIN_SAMPLES": 2,
    "MIN_APP_BASELINE_APPEARANCES": 3,
    "MIN_ARCHETYPE_DAYS_FOR_KMEANS": 10,

    # Per-feature deviation ceilings (populated by calibration per user)
    "FEATURE_CEILINGS": {},
}
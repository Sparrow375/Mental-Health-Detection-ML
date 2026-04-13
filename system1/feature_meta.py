"""
Feature metadata: clinical weights, feature groups, critical features, and feature lists.

L1 PersonalityVector — 29 aggregate daily features.
L2 Texture Vector   — 22 session/notification-derived features.
L1 DBSCAN clustering uses a 12-feature subset of the 29 L1 features.
"""

from typing import Dict, List


# ============================================================================
# L1 FEATURE WEIGHTS  (clinical importance, per the implementation plan)
# ============================================================================
# Weight range: 0.4 (quasi-static, e.g. storage) → 1.6 (highest clinical, e.g. sleep)

FEATURE_META: Dict[str, Dict] = {
    # Voice & Prosody
    'voice_pitch_mean':            {'weight': 0.8,  'group': 'Voice & Prosody'},
    'voice_pitch_std':             {'weight': 0.8,  'group': 'Voice & Prosody'},
    'voice_energy_mean':           {'weight': 1.2,  'group': 'Voice & Prosody'},
    'voice_speaking_rate':         {'weight': 1.0,  'group': 'Voice & Prosody'},

    # Activity & Digital
    'screen_time_hours':           {'weight': 1.3,  'group': 'Activity & Digital'},
    'unlock_count':                {'weight': 0.9,  'group': 'Activity & Digital'},
    'social_app_ratio':            {'weight': 1.2,  'group': 'Activity & Digital'},
    'app_launch_count':            {'weight': 0.8,  'group': 'Activity & Digital'},
    'notifications_today':         {'weight': 0.7,  'group': 'Activity & Digital'},
    'total_apps_count':            {'weight': 0.6,  'group': 'Activity & Digital'},

    # Communications
    'calls_per_day':               {'weight': 1.2,  'group': 'Communications'},
    'texts_per_day':               {'weight': 1.0,  'group': 'Communications'},
    'unique_contacts':             {'weight': 1.4,  'group': 'Communications'},
    'response_time_minutes':       {'weight': 1.2,  'group': 'Communications'},

    # Movement & Mobility
    'daily_displacement_km':       {'weight': 1.4,  'group': 'Movement & Mobility'},
    'location_entropy':            {'weight': 1.0,  'group': 'Movement & Mobility'},
    'home_time_ratio':             {'weight': 1.2,  'group': 'Movement & Mobility'},
    'places_visited':              {'weight': 1.0,  'group': 'Movement & Mobility'},

    # Circadian & Environment
    'wake_time_hour':              {'weight': 1.0,  'group': 'Circadian & Environment'},
    'sleep_time_hour':             {'weight': 0.7,  'group': 'Circadian & Environment'},
    'sleep_duration_hours':        {'weight': 1.6,  'group': 'Circadian & Environment'},
    'dark_duration_hours':         {'weight': 0.8,  'group': 'Circadian & Environment'},
    'charge_duration_hours':       {'weight': 0.7,  'group': 'Circadian & Environment'},

    # Social & Audio
    'conversation_duration_hours': {'weight': 1.2,  'group': 'Social & Audio'},
    'conversation_frequency':      {'weight': 1.2,  'group': 'Social & Audio'},

    # Calendar & Engagement / Financial / Device
    'calendar_events_today':       {'weight': 0.6,  'group': 'Calendar & Engagement'},
    'upi_transactions_today':      {'weight': 0.9,  'group': 'Calendar & Engagement'},
    'background_audio_hours':      {'weight': 0.9,  'group': 'Calendar & Engagement'},
    'storage_used_gb':             {'weight': 0.4,  'group': 'Calendar & Engagement'},
}


# Ordered list of all 29 L1 feature names
ALL_L1_FEATURES: List[str] = list(FEATURE_META.keys())


# Features the alert engine treats as high-priority for escalation decisions
CRITICAL_FEATURES: List[str] = [
    'sleep_duration_hours',
    'daily_displacement_km',
    'screen_time_hours',
    'social_app_ratio',
    'wake_time_hour',
    'calls_per_day',
    'upi_transactions_today',
    'total_apps_count',
]


# 12-feature subset used for L1 DBSCAN context clustering
L1_CLUSTERING_FEATURES: List[str] = [
    'sleep_duration_hours',
    'wake_time_hour',
    'sleep_time_hour',
    'daily_displacement_km',
    'location_entropy',
    'places_visited',
    'calls_per_day',
    'conversation_duration_hours',
    'screen_time_hours',
    'unlock_count',
    'social_app_ratio',
    'dark_duration_hours',
]


# ============================================================================
# L2 TEXTURE VECTOR  (22 features — derived from session & notification events)
# ============================================================================

L2_TEXTURE_FEATURES: List[str] = [
    # Temporal anchoring (4)
    'time_in_primary_window_ratio',
    'temporal_anchor_deviation',
    'first_pickup_hour_deviation',
    'rhythm_dissolution_score',

    # Session quality (5)
    'weighted_abandon_rate',
    'deep_session_ratio',
    'micro_session_ratio',
    'session_duration_collapse',
    'interaction_density_ratio',

    # Agency & initiation (4)
    'self_open_ratio',
    'notification_open_rate',
    'notification_ignore_rate',
    'pickup_burst_rate',

    # Attention coherence (4)
    'app_switching_rate',
    'app_cooccurrence_consistency',
    'distinct_apps_ratio',
    'session_context_match',

    # Rhythm & structure (3)
    'daily_rhythm_regularity',
    'weekday_weekend_alignment',
    'dead_zone_count',

    # Notification relationship (2)
    'notification_response_latency_shift',
    'notification_to_session_ratio',
]


# Feature group mapping for the 22 L2 features (for reporting)
L2_FEATURE_GROUPS: Dict[str, List[str]] = {
    'Temporal Anchoring': L2_TEXTURE_FEATURES[0:4],
    'Session Quality': L2_TEXTURE_FEATURES[4:9],
    'Agency & Initiation': L2_TEXTURE_FEATURES[9:13],
    'Attention Coherence': L2_TEXTURE_FEATURES[13:17],
    'Rhythm & Structure': L2_TEXTURE_FEATURES[17:20],
    'Notification Relationship': L2_TEXTURE_FEATURES[20:22],
}


# ============================================================================
# THRESHOLD DEFAULTS  (from implementation plan docx)
# ============================================================================

DEFAULT_THRESHOLDS = {
    # Daily scoring gate — fixed for all users
    'ANOMALY_SCORE_THRESHOLD': 0.32,

    # Real-time alerting
    'SUSTAINED_THRESHOLD_DAYS': 5,
    'EVIDENCE_THRESHOLD': 0.40,

    # Retrospective (peak-based) — calibrated per user if baseline is noisy
    'PEAK_EVIDENCE_THRESHOLD': 0.50,
    'PEAK_SUSTAINED_THRESHOLD_DAYS': 10,

    # Secondary watch flag
    'WATCH_EVIDENCE_THRESHOLD': 0.20,

    # Evidence accumulation
    'EVIDENCE_DECAY_RATE': 0.92,

    # Candidate cluster evaluation
    'CANDIDATE_WINDOW_DAYS': 7,
    'CANDIDATE_TEXTURE_THRESHOLD': 0.35,   # session_incoherence

    # L2 coherence
    'COHERENCE_MATCH_RADIUS_FACTOR': 1.5,  # × cluster radius

    # DBSCAN
    'DBSCAN_MIN_SAMPLES': 2,
    'MIN_APP_BASELINE_APPEARANCES': 3,
    'MIN_ARCHETYPE_DAYS_FOR_KMEANS': 10,
}

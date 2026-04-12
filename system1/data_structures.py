"""
Core data structures for the L1 + L2 Behavioral Anomaly Detection pipeline.

L1 PersonalityVector: 29 aggregate daily features (the baseline reference).
L2 Texture Vector: 22 session/notification-derived features (per-archetype texture).
AppDNA / PhoneDNA: per-app and device-level behavioural fingerprints from session events.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# L1 — PersonalityVector  (29 features)
# ---------------------------------------------------------------------------

@dataclass
class PersonalityVector:
    """
    Baseline personality profile — 29 aggregate daily behavioural features.
    Frozen after baseline establishment (days 1-28+).
    """

    # Voice & Prosody (4)
    voice_pitch_mean: float = 0.0
    voice_pitch_std: float = 0.0
    voice_energy_mean: float = 0.0
    voice_speaking_rate: float = 0.0

    # Activity & Digital (6)
    screen_time_hours: float = 0.0
    unlock_count: float = 0.0
    social_app_ratio: float = 0.0
    app_launch_count: float = 0.0
    notifications_today: float = 0.0
    total_apps_count: float = 0.0

    # Communications (4)
    calls_per_day: float = 0.0
    texts_per_day: float = 0.0
    unique_contacts: float = 0.0
    response_time_minutes: float = 0.0

    # Movement & Mobility (4)
    daily_displacement_km: float = 0.0
    location_entropy: float = 0.0
    home_time_ratio: float = 0.0
    places_visited: float = 0.0

    # Circadian & Environment (5)
    wake_time_hour: float = 0.0
    sleep_time_hour: float = 0.0
    sleep_duration_hours: float = 0.0
    dark_duration_hours: float = 0.0
    charge_duration_hours: float = 0.0

    # Social & Audio (2)
    conversation_duration_hours: float = 0.0
    conversation_frequency: float = 0.0

    # Calendar & Engagement (4)
    calendar_events_today: float = 0.0
    upi_transactions_today: float = 0.0
    background_audio_hours: float = 0.0
    storage_used_gb: float = 0.0

    # Per-feature standard deviations (populated during baseline)
    variances: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        """Return all 29 feature values as an ordered dict."""
        return {
            'voice_pitch_mean': self.voice_pitch_mean,
            'voice_pitch_std': self.voice_pitch_std,
            'voice_energy_mean': self.voice_energy_mean,
            'voice_speaking_rate': self.voice_speaking_rate,
            'screen_time_hours': self.screen_time_hours,
            'unlock_count': self.unlock_count,
            'social_app_ratio': self.social_app_ratio,
            'app_launch_count': self.app_launch_count,
            'notifications_today': self.notifications_today,
            'total_apps_count': self.total_apps_count,
            'calls_per_day': self.calls_per_day,
            'texts_per_day': self.texts_per_day,
            'unique_contacts': self.unique_contacts,
            'response_time_minutes': self.response_time_minutes,
            'daily_displacement_km': self.daily_displacement_km,
            'location_entropy': self.location_entropy,
            'home_time_ratio': self.home_time_ratio,
            'places_visited': self.places_visited,
            'wake_time_hour': self.wake_time_hour,
            'sleep_time_hour': self.sleep_time_hour,
            'sleep_duration_hours': self.sleep_duration_hours,
            'dark_duration_hours': self.dark_duration_hours,
            'charge_duration_hours': self.charge_duration_hours,
            'conversation_duration_hours': self.conversation_duration_hours,
            'conversation_frequency': self.conversation_frequency,
            'calendar_events_today': self.calendar_events_today,
            'upi_transactions_today': self.upi_transactions_today,
            'background_audio_hours': self.background_audio_hours,
            'storage_used_gb': self.storage_used_gb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float], variances: Optional[Dict[str, float]] = None) -> 'PersonalityVector':
        """Construct from a flat dict.  Missing keys default to 0.0."""
        feature_keys = list(cls().to_dict().keys())
        params = {k: d.get(k, 0.0) for k in feature_keys}
        pv = cls(**params)
        pv.variances = variances
        return pv


# ---------------------------------------------------------------------------
# L2 — AppDNA  (per-app behavioural fingerprint)
# ---------------------------------------------------------------------------

@dataclass
class AppDNA:
    """
    Per-app L2 baseline profile built from session events.
    Only constructed for apps with ≥3 sessions during baseline.
    """

    app_package: str = ''

    # Temporal DNA
    usage_heatmap: Optional[np.ndarray] = None          # shape (7, 24) — mean minutes per dow×hour
    primary_time_range: Tuple[int, int] = (0, 23)       # (hour_start, hour_end) containing 80 % usage
    time_concentration_ratio: float = 0.0
    time_concentration_std: float = 0.0

    # Session signature
    avg_session_minutes: float = 0.0
    std_session_minutes: float = 0.0
    p10_session_minutes: float = 0.0
    p90_session_minutes: float = 0.0
    abandon_rate: float = 0.0
    abandon_rate_std: float = 0.0

    # Trigger DNA
    self_open_ratio: float = 0.0
    notification_open_ratio: float = 0.0
    shortcut_open_ratio: float = 0.0
    notification_response_latency_median: float = 0.0
    notification_response_latency_std: float = 0.0

    # Sequence DNA
    pre_open_apps: Optional[Dict[str, float]] = None   # app_id → frequency
    post_open_apps: Optional[Dict[str, float]] = None

    # Engagement density
    interactions_per_minute_mean: float = 0.0
    interactions_per_minute_std: float = 0.0

    # Weekday / weekend split
    weekday_sessions_per_day: float = 0.0
    weekend_sessions_per_day: float = 0.0

    # Consistency
    daily_use_consistency: float = 0.0   # fraction of baseline days the app appears
    max_gap_days: int = 0                # longest consecutive absent streak


# ---------------------------------------------------------------------------
# L2 — PhoneDNA  (device-level behavioural fingerprint)
# ---------------------------------------------------------------------------

@dataclass
class PhoneDNA:
    """
    Device-level L2 baseline aggregated from all session and notification events.
    """

    first_pickup_hour_mean: float = 0.0
    first_pickup_hour_std: float = 0.0
    active_window_duration_mean: float = 0.0
    active_window_duration_std: float = 0.0

    pickups_per_hour_by_hour: Optional[np.ndarray] = None   # shape (24,)
    pickup_burst_rate: float = 0.0
    inter_pickup_interval_mean: float = 0.0
    inter_pickup_interval_std: float = 0.0

    # Session duration distribution — 5-bin histogram
    session_duration_distribution: Optional[np.ndarray] = None   # [<2m, 2-15m, 15-30m, 30-60m, 60+m]
    deep_session_ratio: float = 0.0      # fraction of sessions > 20 min
    micro_session_ratio: float = 0.0     # fraction of sessions < 2 min

    # App co-occurrence
    app_cooccurrence_matrix: Optional[np.ndarray] = None   # (N_apps × N_apps)

    # Notification relationship
    notification_open_rate: float = 0.0
    notification_dismiss_rate: float = 0.0
    notification_ignore_rate: float = 0.0

    # Rhythm
    daily_rhythm_regularity: float = 0.0   # autocorrelation of hourly pickup vector
    weekday_weekend_delta: float = 0.0     # L1 norm of weekday/weekend feature diff
    historically_active_hours: Optional[List[int]] = None   # hours with pickups > threshold


# ---------------------------------------------------------------------------
# L2 — ContextualTextureProfile  (per L1-archetype texture baseline)
# ---------------------------------------------------------------------------

@dataclass
class ContextualTextureProfile:
    """
    One profile per L1 DBSCAN archetype.
    L2 texture is always evaluated relative to the matched L1 context.
    """

    archetype_id: int = -1
    member_days: int = 0

    # Used when member_days ≥ 10  (K-means texture centroids)
    texture_centroids: Optional[np.ndarray] = None   # shape (K, 22)
    texture_radii: Optional[np.ndarray] = None       # shape (K,)

    # Fallback when member_days < 10  (mean/std per texture feature)
    texture_mean: Optional[np.ndarray] = None        # shape (22,)
    texture_std: Optional[np.ndarray] = None         # shape (22,)

    # Per-archetype tolerance scalar
    tolerance_factor: float = 1.0


# ---------------------------------------------------------------------------
# L1 DBSCAN cluster state
# ---------------------------------------------------------------------------

@dataclass
class L1ClusterState:
    """Output of the L1 DBSCAN clustering step."""

    n_clusters: int = 0
    centroids: Optional[np.ndarray] = None        # shape (K, 12)
    radii: Optional[np.ndarray] = None            # shape (K,)
    covariance_inv: Optional[np.ndarray] = None   # inverse covariance for Mahalanobis
    labels: Optional[np.ndarray] = None           # per-baseline-day cluster label
    outlier_indices: Optional[List[int]] = None
    feature_min: Optional[np.ndarray] = None      # for normalisation
    feature_max: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Evidence & Candidate state  (persisted across days)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceState:
    """Persistent state of the evidence accumulator."""

    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0
    max_evidence: float = 0.0
    max_sustained_days: int = 0
    max_anomaly_score: float = 0.0


@dataclass
class CandidateState:
    """State of the 7-day candidate-cluster evaluation window."""

    status: str = 'CLOSED'           # CLOSED | EVALUATING | PROMOTED | REJECTED
    open_day: int = 0
    days_elapsed: int = 0
    l1_buffer: List[Dict[str, float]] = field(default_factory=list)
    l2_buffer: List[Dict[str, float]] = field(default_factory=list)
    session_incoherence_history: List[float] = field(default_factory=list)
    held_evidence: List[float] = field(default_factory=list)   # effective scores held during window


# ---------------------------------------------------------------------------
# Output reports
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """Raw report for downstream systems — written per night."""

    timestamp: datetime = None
    overall_anomaly_score: float = 0.0           # L1 composite (pre-modifier)
    effective_score: float = 0.0                 # L1 × L2 modifier
    feature_deviations: Dict[str, float] = field(default_factory=dict)
    deviation_velocity: Dict[str, float] = field(default_factory=dict)
    l2_modifier: float = 1.0
    matched_context_id: int = -1
    coherence_score: float = 0.0
    rhythm_dissolution: float = 0.0
    session_incoherence: float = 0.0
    alert_level: str = 'green'
    flagged_features: List[str] = field(default_factory=list)
    pattern_type: str = 'stable'
    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0


@dataclass
class DailyReport:
    """Human-readable daily report for UI and clinicians."""

    day_number: int = 0
    date: datetime = None
    anomaly_score: float = 0.0                   # effective score
    alert_level: str = 'green'
    flagged_features: List[str] = field(default_factory=list)
    pattern_type: str = 'stable'
    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0
    top_deviations: Dict[str, float] = field(default_factory=dict)
    notes: str = ''
    l2_modifier: float = 1.0


@dataclass
class FinalPrediction:
    """Final retrospective prediction at end of monitoring or on clinical request."""

    patient_id: str = ''
    scenario: str = ''
    monitoring_days: int = 0
    baseline_vector: Optional[PersonalityVector] = None
    final_anomaly_score: float = 0.0
    sustained_anomaly_detected: bool = False
    confidence: float = 0.0
    pattern_identified: str = 'stable'
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ''

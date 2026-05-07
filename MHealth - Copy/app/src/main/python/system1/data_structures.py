"""
Core data structures for the L1 + L2 Behavioral Anomaly Detection pipeline.

Adapted for MHealth Android app — uses camelCase feature names matching
Android's PersonalityVector.toMap() keys exactly.

L1 PersonalityVector: 30 aggregate daily features (the baseline reference).
L2 Texture Vector: 22 session/notification-derived features (per-archetype texture).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# L1 — PersonalityVector  (30 features — camelCase matching Android)
# ---------------------------------------------------------------------------

@dataclass
class PersonalityVector:
    """
    Baseline personality profile — 30 aggregate daily behavioural features.
    Feature names match Android's PersonalityVector.toMap() exactly.
    """

    # ── Screen & App Activity ──────────────────────────────────────────────
    screenTimeHours: float = 0.0
    unlockCount: float = 0.0
    appLaunchCount: float = 0.0
    notificationsToday: float = 0.0
    socialAppRatio: float = 0.0

    # ── Communication ──────────────────────────────────────────────────────
    callsPerDay: float = 0.0
    callDurationMinutes: float = 0.0
    uniqueContacts: float = 0.0
    conversationFrequency: float = 0.0

    # ── Location & Movement ───────────────────────────────────────────────
    dailyDisplacementKm: float = 0.0
    locationEntropy: float = 0.0
    homeTimeRatio: float = 0.0

    # ── Sleep & Circadian ─────────────────────────────────────────────────
    wakeTimeHour: float = 0.0
    sleepTimeHour: float = 0.0
    sleepDurationHours: float = 0.0
    darkDurationHours: float = 0.0

    # ── System Usage ──────────────────────────────────────────────────────
    chargeDurationHours: float = 0.0
    memoryUsagePercent: float = 0.0
    networkWifiMB: float = 0.0
    networkMobileMB: float = 0.0
    storageUsedGB: float = 0.0

    # ── Behavioural Signals ───────────────────────────────────────────────
    totalAppsCount: float = 0.0
    upiTransactionsToday: float = 0.0
    appUninstallsToday: float = 0.0
    appInstallsToday: float = 0.0

    # ── Calendar & Engagement ─────────────────────────────────────────────
    calendarEventsToday: float = 0.0
    mediaCountToday: float = 0.0
    downloadsToday: float = 0.0
    backgroundAudioHours: float = 0.0

    # ── Internal: per-feature std deviation from baseline ─────────────────
    variances: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        """Return all feature values as an ordered dict."""
        return {
            "screenTimeHours": self.screenTimeHours,
            "unlockCount": self.unlockCount,
            "appLaunchCount": self.appLaunchCount,
            "notificationsToday": self.notificationsToday,
            "socialAppRatio": self.socialAppRatio,
            "callsPerDay": self.callsPerDay,
            "callDurationMinutes": self.callDurationMinutes,
            "uniqueContacts": self.uniqueContacts,
            "conversationFrequency": self.conversationFrequency,
            "dailyDisplacementKm": self.dailyDisplacementKm,
            "locationEntropy": self.locationEntropy,
            "homeTimeRatio": self.homeTimeRatio,
            "wakeTimeHour": self.wakeTimeHour,
            "sleepTimeHour": self.sleepTimeHour,
            "sleepDurationHours": self.sleepDurationHours,
            "darkDurationHours": self.darkDurationHours,
            "chargeDurationHours": self.chargeDurationHours,
            "memoryUsagePercent": self.memoryUsagePercent,
            "networkWifiMB": self.networkWifiMB,
            "networkMobileMB": self.networkMobileMB,
            "storageUsedGB": self.storageUsedGB,
            "totalAppsCount": self.totalAppsCount,
            "upiTransactionsToday": self.upiTransactionsToday,
            "appUninstallsToday": self.appUninstallsToday,
            "appInstallsToday": self.appInstallsToday,
            "calendarEventsToday": self.calendarEventsToday,
            "mediaCountToday": self.mediaCountToday,
            "downloadsToday": self.downloadsToday,
            "backgroundAudioHours": self.backgroundAudioHours,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float], variances: Optional[Dict[str, float]] = None) -> 'PersonalityVector':
        """Construct from a flat dict.  Missing keys default to 0.0."""
        feature_keys = list(cls().to_dict().keys())
        params = {k: float(d.get(k, 0.0)) for k in feature_keys}
        pv = cls(**params)
        pv.variances = variances
        return pv


# ---------------------------------------------------------------------------
# L2 — AppDNA  (per-app behavioural fingerprint)
# ---------------------------------------------------------------------------

@dataclass
class AppDNA:
    """Per-app L2 baseline profile built from session events."""

    app_package: str = ''

    # Temporal DNA
    usage_heatmap: Optional[np.ndarray] = None          # shape (7, 24)
    primary_time_range: Tuple[int, int] = (0, 23)
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
    pre_open_apps: Optional[Dict[str, float]] = None
    post_open_apps: Optional[Dict[str, float]] = None

    # Engagement density
    interactions_per_minute_mean: float = 0.0
    interactions_per_minute_std: float = 0.0

    # Weekday / weekend split
    weekday_sessions_per_day: float = 0.0
    weekend_sessions_per_day: float = 0.0

    # Consistency
    daily_use_consistency: float = 0.0
    max_gap_days: int = 0


# ---------------------------------------------------------------------------
# L2 — PhoneDNA  (device-level behavioural fingerprint)
# ---------------------------------------------------------------------------

@dataclass
class PhoneDNA:
    """Device-level L2 baseline aggregated from all session and notification events."""

    first_pickup_hour_mean: float = 0.0
    first_pickup_hour_std: float = 0.0
    active_window_duration_mean: float = 0.0
    active_window_duration_std: float = 0.0

    pickups_per_hour_by_hour: Optional[np.ndarray] = None   # shape (24,)
    pickup_burst_rate: float = 0.0
    inter_pickup_interval_mean: float = 0.0
    inter_pickup_interval_std: float = 0.0

    session_duration_distribution: Optional[np.ndarray] = None
    deep_session_ratio: float = 0.0
    micro_session_ratio: float = 0.0

    app_cooccurrence_matrix: Optional[np.ndarray] = None

    notification_open_rate: float = 0.0
    notification_dismiss_rate: float = 0.0
    notification_ignore_rate: float = 0.0

    daily_rhythm_regularity: float = 0.0
    weekday_weekend_delta: float = 0.0
    historically_active_hours: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# L2 — ContextualTextureProfile  (per L1-archetype texture baseline)
# ---------------------------------------------------------------------------

@dataclass
class ContextualTextureProfile:
    """One profile per L1 DBSCAN archetype."""

    archetype_id: int = -1
    member_days: int = 0

    texture_centroids: Optional[np.ndarray] = None
    texture_radii: Optional[np.ndarray] = None

    texture_mean: Optional[np.ndarray] = None
    texture_std: Optional[np.ndarray] = None

    tolerance_factor: float = 1.0


# ---------------------------------------------------------------------------
# L1 DBSCAN cluster state
# ---------------------------------------------------------------------------

@dataclass
class L1ClusterState:
    """Output of the L1 DBSCAN clustering step."""

    n_clusters: int = 0
    centroids: Optional[np.ndarray] = None
    radii: Optional[np.ndarray] = None
    covariance_inv: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    outlier_indices: Optional[List[int]] = None
    feature_min: Optional[np.ndarray] = None
    feature_max: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Bayesian Warm Start — Baseline Phase & Posterior State
# ---------------------------------------------------------------------------

class BaselinePhase(Enum):
    """Three-phase warm-start schedule for Bayesian baseline."""
    POPULATION_ANCHORED = "population_anchored"   # Day 0-13
    BLENDED = "blended"                           # Day 14-59
    IDIOGRAPHIC = "idiographic"                   # Day 60+


@dataclass
class FeaturePosterior:
    """Normal-Inverse-Gamma posterior for one feature's mean and variance."""

    mu_0: float = 0.0
    kappa_0: float = 14.0
    alpha_0: float = 2.0
    beta_0: float = 1.0

    mu_n: float = 0.0
    kappa_n: float = 14.0
    alpha_n: float = 2.0
    beta_n: float = 1.0

    n_observations: int = 0
    sum_observations: float = 0.0
    sum_sq_observations: float = 0.0


@dataclass
class BayesianState:
    """Full Bayesian warm-start state across all L1 features."""

    phase: BaselinePhase = BaselinePhase.POPULATION_ANCHORED
    day_number: int = 0
    personal_weight: float = 0.0
    confidence_score: float = 0.0
    feature_posteriors: Dict[str, FeaturePosterior] = field(default_factory=dict)

    effective_means: Dict[str, float] = field(default_factory=dict)
    effective_stds: Dict[str, float] = field(default_factory=dict)
    feature_confidences: Dict[str, float] = field(default_factory=dict)


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
    max_breadth: int = 0


@dataclass
class CandidateState:
    """State of the 7-day candidate-cluster evaluation window."""

    status: str = 'CLOSED'
    open_day: int = 0
    days_elapsed: int = 0
    l1_buffer: List[Dict[str, float]] = field(default_factory=list)
    l2_buffer: List[Dict[str, float]] = field(default_factory=list)
    session_incoherence_history: List[float] = field(default_factory=list)
    held_evidence: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Output reports
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """Raw report for downstream systems — written per night."""

    timestamp: datetime = None
    overall_anomaly_score: float = 0.0
    effective_score: float = 0.0
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
    baseline_phase: str = 'idiographic'
    baseline_confidence: float = 1.0


@dataclass
class DailyReport:
    """Human-readable daily report for UI and clinicians."""

    day_number: int = 0
    date: datetime = None
    anomaly_score: float = 0.0
    alert_level: str = 'green'
    flagged_features: List[str] = field(default_factory=list)
    pattern_type: str = 'stable'
    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0
    top_deviations: Dict[str, float] = field(default_factory=dict)
    notes: str = ''
    l2_modifier: float = 1.0
    baseline_phase: str = 'idiographic'
    baseline_confidence: float = 1.0
    baseline_label: str = ''


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
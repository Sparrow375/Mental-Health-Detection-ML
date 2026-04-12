"""
Core Data Structures for Behavioral Anomaly Detection System.
Defines PersonalityVector, AppDNA, PhoneDNA, ContextualTextureProfile,
and all report/intermediate dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Enums ──────────────────────────────────────────────────────────────────────

class ConfidenceTier(str, Enum):
    LOW = "LOW_CONFIDENCE"
    MEDIUM = "MEDIUM_CONFIDENCE"
    HIGH = "HIGH_CONFIDENCE"


class AlertLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class PatternType(str, Enum):
    STABLE = "stable"
    RAPID_CYCLING = "rapid_cycling"
    ACUTE_SPIKE = "acute_spike"
    GRADUAL_DRIFT = "gradual_drift"
    MIXED_PATTERN = "mixed_pattern"


class TriggerType(str, Enum):
    SELF = "SELF"
    NOTIFICATION = "NOTIFICATION"
    SHORTCUT = "SHORTCUT"
    WIDGET = "WIDGET"


class NotificationAction(str, Enum):
    TAP = "TAP"
    DISMISS = "DISMISS"
    IGNORE = "IGNORE"


class RecommendationTier(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    MONITOR = "MONITOR"
    REFER = "REFER"


# ── L1 Baseline ───────────────────────────────────────────────────────────────

@dataclass
class PersonalityVector:
    """29-feature baseline means and variances for a person."""
    means: Dict[str, float] = field(default_factory=dict)
    variances: Dict[str, float] = field(default_factory=dict)  # std-dev per feature
    confidence: ConfidenceTier = ConfidenceTier.LOW
    built_date: Optional[str] = None  # ISO date when baseline was built

    def to_dict(self) -> Dict[str, Any]:
        return {
            "means": self.means,
            "variances": self.variances,
            "confidence": self.confidence.value,
            "built_date": self.built_date,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonalityVector":
        return cls(
            means=d.get("means", {}),
            variances=d.get("variances", {}),
            confidence=ConfidenceTier(d.get("confidence", "LOW_CONFIDENCE")),
            built_date=d.get("built_date"),
        )


# ── L2 Per-App DNA ────────────────────────────────────────────────────────────

@dataclass
class AppDNA:
    """Per-app behavioral DNA built during baseline period."""
    app_id: str = ""
    usage_heatmap: Optional[np.ndarray] = None  # (7, 24) day-of-week x hour
    primary_time_range: Tuple[int, int] = (0, 0)  # hour_start, hour_end (80% usage)
    time_concentration_ratio: float = 0.0
    time_concentration_std: float = 0.0
    avg_session_minutes: float = 0.0
    std_session_minutes: float = 0.0
    p10_session_minutes: float = 0.0
    p90_session_minutes: float = 0.0
    abandon_rate: float = 0.0
    abandon_rate_std: float = 0.0
    self_open_ratio: float = 0.0
    notification_open_ratio: float = 0.0
    shortcut_open_ratio: float = 0.0
    notification_response_latency_median: float = 0.0
    notification_response_latency_std: float = 0.0
    pre_open_apps: Dict[str, float] = field(default_factory=dict)
    post_open_apps: Dict[str, float] = field(default_factory=dict)
    interactions_per_minute_mean: float = 0.0
    interactions_per_minute_std: float = 0.0
    weekday_sessions_per_day: float = 0.0
    weekend_sessions_per_day: float = 0.0
    daily_use_consistency: float = 0.0
    max_gap_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.usage_heatmap is not None:
            d["usage_heatmap"] = self.usage_heatmap.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppDNA":
        hm = d.get("usage_heatmap")
        obj = cls(
            app_id=d.get("app_id", ""),
            primary_time_range=tuple(d.get("primary_time_range", (0, 0))),
            time_concentration_ratio=d.get("time_concentration_ratio", 0.0),
            time_concentration_std=d.get("time_concentration_std", 0.0),
            avg_session_minutes=d.get("avg_session_minutes", 0.0),
            std_session_minutes=d.get("std_session_minutes", 0.0),
            p10_session_minutes=d.get("p10_session_minutes", 0.0),
            p90_session_minutes=d.get("p90_session_minutes", 0.0),
            abandon_rate=d.get("abandon_rate", 0.0),
            abandon_rate_std=d.get("abandon_rate_std", 0.0),
            self_open_ratio=d.get("self_open_ratio", 0.0),
            notification_open_ratio=d.get("notification_open_ratio", 0.0),
            shortcut_open_ratio=d.get("shortcut_open_ratio", 0.0),
            notification_response_latency_median=d.get("notification_response_latency_median", 0.0),
            notification_response_latency_std=d.get("notification_response_latency_std", 0.0),
            pre_open_apps=d.get("pre_open_apps", {}),
            post_open_apps=d.get("post_open_apps", {}),
            interactions_per_minute_mean=d.get("interactions_per_minute_mean", 0.0),
            interactions_per_minute_std=d.get("interactions_per_minute_std", 0.0),
            weekday_sessions_per_day=d.get("weekday_sessions_per_day", 0.0),
            weekend_sessions_per_day=d.get("weekend_sessions_per_day", 0.0),
            daily_use_consistency=d.get("daily_use_consistency", 0.0),
            max_gap_days=d.get("max_gap_days", 0),
        )
        if hm is not None:
            obj.usage_heatmap = np.array(hm)
        return obj


# ── L2 Device-Level DNA ───────────────────────────────────────────────────────

@dataclass
class PhoneDNA:
    """Device-level behavioral DNA built during baseline period."""
    first_pickup_hour_mean: float = 0.0
    first_pickup_hour_std: float = 0.0
    active_window_duration_mean: float = 0.0
    active_window_duration_std: float = 0.0
    pickups_per_hour_by_hour: Optional[np.ndarray] = None  # (24,)
    pickup_burst_rate: float = 0.0
    inter_pickup_interval_mean: float = 0.0
    inter_pickup_interval_std: float = 0.0
    session_duration_distribution: Optional[np.ndarray] = None  # 5-bin histogram
    deep_session_ratio: float = 0.0
    micro_session_ratio: float = 0.0
    app_cooccurrence_matrix: Optional[np.ndarray] = None  # (N_apps x N_apps)
    notification_open_rate: float = 0.0
    notification_dismiss_rate: float = 0.0
    notification_ignore_rate: float = 0.0
    daily_rhythm_regularity: float = 0.0
    weekday_weekend_delta: float = 0.0
    historically_active_hours: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.pickups_per_hour_by_hour is not None:
            d["pickups_per_hour_by_hour"] = self.pickups_per_hour_by_hour.tolist()
        if self.session_duration_distribution is not None:
            d["session_duration_distribution"] = self.session_duration_distribution.tolist()
        if self.app_cooccurrence_matrix is not None:
            d["app_cooccurrence_matrix"] = self.app_cooccurrence_matrix.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhoneDNA":
        obj = cls(
            first_pickup_hour_mean=d.get("first_pickup_hour_mean", 0.0),
            first_pickup_hour_std=d.get("first_pickup_hour_std", 0.0),
            active_window_duration_mean=d.get("active_window_duration_mean", 0.0),
            active_window_duration_std=d.get("active_window_duration_std", 0.0),
            pickup_burst_rate=d.get("pickup_burst_rate", 0.0),
            inter_pickup_interval_mean=d.get("inter_pickup_interval_mean", 0.0),
            inter_pickup_interval_std=d.get("inter_pickup_interval_std", 0.0),
            deep_session_ratio=d.get("deep_session_ratio", 0.0),
            micro_session_ratio=d.get("micro_session_ratio", 0.0),
            notification_open_rate=d.get("notification_open_rate", 0.0),
            notification_dismiss_rate=d.get("notification_dismiss_rate", 0.0),
            notification_ignore_rate=d.get("notification_ignore_rate", 0.0),
            daily_rhythm_regularity=d.get("daily_rhythm_regularity", 0.0),
            weekday_weekend_delta=d.get("weekday_weekend_delta", 0.0),
            historically_active_hours=d.get("historically_active_hours", []),
        )
        pph = d.get("pickups_per_hour_by_hour")
        if pph is not None:
            obj.pickups_per_hour_by_hour = np.array(pph)
        sdd = d.get("session_duration_distribution")
        if sdd is not None:
            obj.session_duration_distribution = np.array(sdd)
        aco = d.get("app_cooccurrence_matrix")
        if aco is not None:
            obj.app_cooccurrence_matrix = np.array(aco)
        return obj


# ── L1 Anchor Cluster ─────────────────────────────────────────────────────────

@dataclass
class AnchorCluster:
    """One DBSCAN archetype from baseline L1 daily vectors."""
    cluster_id: int = 0
    centroid: Optional[np.ndarray] = None  # 12-feature centroid
    radius: float = 0.0  # max intra-cluster Mahalanobis distance
    member_count: int = 0
    member_dates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.centroid is not None:
            d["centroid"] = self.centroid.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnchorCluster":
        obj = cls(
            cluster_id=d.get("cluster_id", 0),
            radius=d.get("radius", 0.0),
            member_count=d.get("member_count", 0),
            member_dates=d.get("member_dates", []),
        )
        c = d.get("centroid")
        if c is not None:
            obj.centroid = np.array(c)
        return obj


# ── L2 Contextual Texture Profile ─────────────────────────────────────────────

@dataclass
class ContextualTextureProfile:
    """Per-archetype L2 texture profile built during baseline."""
    archetype_id: int = 0
    member_days: int = 0
    texture_centroids: Optional[np.ndarray] = None  # (K, 22) K-means centroids
    texture_radii: Optional[np.ndarray] = None  # (K,) radii
    texture_mean: Optional[np.ndarray] = None  # (22,) fallback mean
    texture_std: Optional[np.ndarray] = None  # (22,) fallback std
    tolerance_factor: float = 1.0
    n_texture_clusters: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "archetype_id": self.archetype_id,
            "member_days": self.member_days,
            "tolerance_factor": self.tolerance_factor,
            "n_texture_clusters": self.n_texture_clusters,
        }
        if self.texture_centroids is not None:
            d["texture_centroids"] = self.texture_centroids.tolist()
        if self.texture_radii is not None:
            d["texture_radii"] = self.texture_radii.tolist()
        if self.texture_mean is not None:
            d["texture_mean"] = self.texture_mean.tolist()
        if self.texture_std is not None:
            d["texture_std"] = self.texture_std.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextualTextureProfile":
        obj = cls(
            archetype_id=d.get("archetype_id", 0),
            member_days=d.get("member_days", 0),
            tolerance_factor=d.get("tolerance_factor", 1.0),
            n_texture_clusters=d.get("n_texture_clusters", 0),
        )
        tc = d.get("texture_centroids")
        if tc is not None:
            obj.texture_centroids = np.array(tc)
        tr = d.get("texture_radii")
        if tr is not None:
            obj.texture_radii = np.array(tr)
        tm = d.get("texture_mean")
        if tm is not None:
            obj.texture_mean = np.array(tm)
        ts = d.get("texture_std")
        if ts is not None:
            obj.texture_std = np.array(ts)
        return obj


# ── Person Profile (aggregate) ────────────────────────────────────────────────

@dataclass
class PersonProfile:
    """Complete baseline profile for one person."""
    patient_id: str = ""
    personality_vector: PersonalityVector = field(default_factory=PersonalityVector)
    app_dnas: Dict[str, AppDNA] = field(default_factory=dict)
    phone_dna: PhoneDNA = field(default_factory=PhoneDNA)
    anchor_clusters: List[AnchorCluster] = field(default_factory=list)
    texture_profiles: List[ContextualTextureProfile] = field(default_factory=list)
    # Mahalanobis covariance inverse for L1 clustering
    l1_cov_inv: Optional[np.ndarray] = None
    l1_feature_mins: Optional[np.ndarray] = None
    l1_feature_maxs: Optional[np.ndarray] = None
    # Calibrated thresholds
    peak_evidence_threshold: float = 7.0
    peak_sustained_threshold_days: int = 10
    # Outlier day vectors from baseline
    outlier_vectors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "patient_id": self.patient_id,
            "personality_vector": self.personality_vector.to_dict(),
            "app_dnas": {k: v.to_dict() for k, v in self.app_dnas.items()},
            "phone_dna": self.phone_dna.to_dict(),
            "anchor_clusters": [c.to_dict() for c in self.anchor_clusters],
            "texture_profiles": [p.to_dict() for p in self.texture_profiles],
            "peak_evidence_threshold": self.peak_evidence_threshold,
            "peak_sustained_threshold_days": self.peak_sustained_threshold_days,
            "outlier_vectors": self.outlier_vectors,
        }
        if self.l1_cov_inv is not None:
            d["l1_cov_inv"] = self.l1_cov_inv.tolist()
        if self.l1_feature_mins is not None:
            d["l1_feature_mins"] = self.l1_feature_mins.tolist()
        if self.l1_feature_maxs is not None:
            d["l1_feature_maxs"] = self.l1_feature_maxs.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonProfile":
        pv_data = d.get("personality_vector", {})
        obj = cls(
            patient_id=d.get("patient_id", ""),
            personality_vector=PersonalityVector.from_dict(pv_data),
            app_dnas={k: AppDNA.from_dict(v) for k, v in d.get("app_dnas", {}).items()},
            phone_dna=PhoneDNA.from_dict(d.get("phone_dna", {})),
            anchor_clusters=[AnchorCluster.from_dict(c) for c in d.get("anchor_clusters", [])],
            texture_profiles=[ContextualTextureProfile.from_dict(p) for p in d.get("texture_profiles", [])],
            peak_evidence_threshold=d.get("peak_evidence_threshold", 7.0),
            peak_sustained_threshold_days=d.get("peak_sustained_threshold_days", 10),
            outlier_vectors=d.get("outlier_vectors", []),
        )
        ci = d.get("l1_cov_inv")
        if ci is not None:
            obj.l1_cov_inv = np.array(ci)
        fm = d.get("l1_feature_mins")
        if fm is not None:
            obj.l1_feature_mins = np.array(fm)
        fx = d.get("l1_feature_maxs")
        if fx is not None:
            obj.l1_feature_maxs = np.array(fx)
        return obj


# ── Intermediate Scoring Results ──────────────────────────────────────────────

@dataclass
class L1ScoreResult:
    """Output of L1 daily scoring."""
    weighted_z_scores: Dict[str, float] = field(default_factory=dict)
    velocity_slopes: Dict[str, float] = field(default_factory=dict)
    magnitude_score: float = 0.0
    velocity_score: float = 0.0
    composite_score: float = 0.0  # [0, 1]


@dataclass
class L2ScoreResult:
    """Output of L2 daily scoring."""
    coherence: float = 0.0  # [0, 1]
    matched_context_id: int = -1
    rhythm_dissolution: float = 0.0  # [0, 1]
    session_incoherence: float = 0.0  # [0, 1]
    modifier: float = 1.0  # [0.15, 2.0]
    candidate_flag: bool = False
    texture_vector: Optional[np.ndarray] = None  # 22-feature vector


# ── Evidence State (persistent) ───────────────────────────────────────────────

@dataclass
class EvidenceState:
    """Stateful evidence accumulator, persisted across days."""
    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0
    max_evidence: float = 0.0
    max_sustained_days: int = 0
    max_anomaly_score: float = 0.0
    effective_score: float = 0.0
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceState":
        return cls(
            sustained_deviation_days=d.get("sustained_deviation_days", 0),
            evidence_accumulated=d.get("evidence_accumulated", 0.0),
            max_evidence=d.get("max_evidence", 0.0),
            max_sustained_days=d.get("max_sustained_days", 0),
            max_anomaly_score=d.get("max_anomaly_score", 0.0),
            effective_score=d.get("effective_score", 0.0),
            last_updated=d.get("last_updated"),
        )


# ── Candidate Cluster State ──────────────────────────────────────────────────

@dataclass
class CandidateState:
    """7-day evaluation window state for candidate clusters."""
    status: str = "CLOSED"  # CLOSED | EVALUATING | PROMOTED | REJECTED_CLINICAL
    open_timestamp: Optional[str] = None
    close_timestamp: Optional[str] = None
    buffered_l1_vectors: List[Dict[str, float]] = field(default_factory=list)
    buffered_l2_vectors: List[Dict[str, float]] = field(default_factory=list)
    daily_texture_quality: List[float] = field(default_factory=list)
    daily_session_incoherence: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateState":
        return cls(
            status=d.get("status", "CLOSED"),
            open_timestamp=d.get("open_timestamp"),
            close_timestamp=d.get("close_timestamp"),
            buffered_l1_vectors=d.get("buffered_l1_vectors", []),
            buffered_l2_vectors=d.get("buffered_l2_vectors", []),
            daily_texture_quality=d.get("daily_texture_quality", []),
            daily_session_incoherence=d.get("daily_session_incoherence", []),
        )


# ── Output Reports ────────────────────────────────────────────────────────────

@dataclass
class AnomalyReport:
    """Raw anomaly report for downstream systems (System 2, Firestore)."""
    timestamp: str = ""
    overall_anomaly_score: float = 0.0  # L1 composite (pre-modifier)
    effective_score: float = 0.0  # L1 x L2 modifier
    feature_deviations: Dict[str, float] = field(default_factory=dict)
    deviation_velocity: Dict[str, float] = field(default_factory=dict)
    l2_modifier: float = 1.0
    matched_context_id: int = -1
    coherence_score: float = 0.0
    rhythm_dissolution: float = 0.0
    session_incoherence: float = 0.0
    alert_level: str = "green"
    flagged_features: List[str] = field(default_factory=list)
    pattern_type: str = "stable"
    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnomalyReport":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DailyReport:
    """Human-readable daily report for UI and clinicians."""
    day_number: int = 0
    date: str = ""
    anomaly_score: float = 0.0
    alert_level: str = "green"
    flagged_features: List[str] = field(default_factory=list)
    pattern_type: str = "stable"
    sustained_deviation_days: int = 0
    evidence_accumulated: float = 0.0
    top_deviations: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalPrediction:
    """End-of-period retrospective prediction."""
    patient_id: str = ""
    sustained_anomaly: bool = False
    confidence: float = 0.0
    pattern_identified: str = ""
    recommendation: str = "NORMAL"
    evidence_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Session / Notification Events ─────────────────────────────────────────────

@dataclass
class SessionEvent:
    """One app session event."""
    app_id: str = ""
    open_ts: float = 0.0  # timestamp ms
    close_ts: float = 0.0
    trigger: str = "SELF"
    interaction_count: int = 0

    @property
    def duration_minutes(self) -> float:
        return max(0, (self.close_ts - self.open_ts) / 60000.0)


@dataclass
class NotificationEvent:
    """One notification event."""
    app_id: str = ""
    arrival_ts: float = 0.0  # timestamp ms
    action: str = "IGNORE"  # TAP | DISMISS | IGNORE
    tap_latency_min: Optional[float] = None
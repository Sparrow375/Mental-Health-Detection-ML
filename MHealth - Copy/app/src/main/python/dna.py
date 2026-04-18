"""
Level 2 Behavioral DNA System — DNA data structures and builders.

This module defines the Behavioral DNA profile for a person based on
their app usage sessions over a 28-day baseline period. It provides:
  - AppDNA: per-app behavioral fingerprint
  - PersonDNA: whole-person behavioral profile with anchor clusters
  - build_app_dna: construct AppDNA from session data
  - build_daily_vector: fixed-length vector representing one day
  - build_person_dna: full DNA profile with K-means anchor clusters
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import calendar

EPS = 1e-9


@dataclass
class AppDNA:
    """Per-app behavioral fingerprint."""
    app_id: str
    usage_heatmap: np.ndarray        # shape (7, 24) — day_of_week x hour, value=avg_minutes
    avg_session_minutes: float
    session_duration_std: float
    sessions_per_active_day: float
    abandon_rate: float              # % sessions where interaction_count<3 AND duration<30s
    self_open_ratio: float           # fraction of sessions that are SELF-triggered
    temporal_anchor_std: float       # std of center-of-mass hour across days (low=consistent timing)

    def to_dict(self) -> dict:
        return {
            "app_id": self.app_id,
            "usage_heatmap": self.usage_heatmap.tolist(),
            "avg_session_minutes": self.avg_session_minutes,
            "session_duration_std": self.session_duration_std,
            "sessions_per_active_day": self.sessions_per_active_day,
            "abandon_rate": self.abandon_rate,
            "self_open_ratio": self.self_open_ratio,
            "temporal_anchor_std": self.temporal_anchor_std,
        }

    @staticmethod
    def from_dict(d: dict) -> "AppDNA":
        return AppDNA(
            app_id=d["app_id"],
            usage_heatmap=np.array(d["usage_heatmap"]),
            avg_session_minutes=d["avg_session_minutes"],
            session_duration_std=d["session_duration_std"],
            sessions_per_active_day=d["sessions_per_active_day"],
            abandon_rate=d["abandon_rate"],
            self_open_ratio=d["self_open_ratio"],
            temporal_anchor_std=d["temporal_anchor_std"],
        )


@dataclass
class CandidateCluster:
    """A candidate behavioral cluster discovered during monitoring."""
    centroid: np.ndarray
    days_observed: int
    texture_scores: List[float]
    held_evidence: float
    status: str  # "evaluating" | "promoted" | "rejected"

    def to_dict(self) -> dict:
        return {
            "centroid": self.centroid.tolist(),
            "days_observed": self.days_observed,
            "texture_scores": self.texture_scores,
            "held_evidence": self.held_evidence,
            "status": self.status,
        }

    @staticmethod
    def from_dict(d: dict) -> "CandidateCluster":
        return CandidateCluster(
            centroid=np.array(d["centroid"]),
            days_observed=d["days_observed"],
            texture_scores=d["texture_scores"],
            held_evidence=d["held_evidence"],
            status=d["status"],
        )


@dataclass
class PromotedCluster:
    """A promoted cluster that passed texture quality threshold."""
    centroid: np.ndarray
    radius: float
    texture_quality_mean: float

    def to_dict(self) -> dict:
        return {
            "centroid": self.centroid.tolist(),
            "radius": self.radius,
            "texture_quality_mean": self.texture_quality_mean,
        }

    @staticmethod
    def from_dict(d: dict) -> "PromotedCluster":
        return PromotedCluster(
            centroid=np.array(d["centroid"]),
            radius=d["radius"],
            texture_quality_mean=d["texture_quality_mean"],
        )


@dataclass
class PersonDNA:
    """Whole-person behavioral profile."""
    person_id: str
    app_profiles: Dict[str, AppDNA]            # package → AppDNA
    daily_session_count_mean: float
    daily_session_count_std: float
    app_switching_rate: float

    # Anchor clusters — FROZEN after baseline, never modified
    anchor_centroids: np.ndarray               # shape (K, n_features), K=3-5
    anchor_radii: np.ndarray                   # max intra-cluster distance per K
    anchor_k: int                              # number of anchor clusters

    # Rolling clusters — discovered during monitoring
    candidate_clusters: List[CandidateCluster]
    promoted_clusters: List[PromotedCluster]

    # Baseline normalization stats (for build_daily_vector)
    daily_vector_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_vector_std: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "app_profiles": {k: v.to_dict() for k, v in self.app_profiles.items()},
            "daily_session_count_mean": self.daily_session_count_mean,
            "daily_session_count_std": self.daily_session_count_std,
            "app_switching_rate": self.app_switching_rate,
            "anchor_centroids": self.anchor_centroids.tolist(),
            "anchor_radii": self.anchor_radii.tolist(),
            "anchor_k": self.anchor_k,
            "candidate_clusters": [c.to_dict() for c in self.candidate_clusters],
            "promoted_clusters": [p.to_dict() for p in self.promoted_clusters],
            "daily_vector_mean": self.daily_vector_mean.tolist(),
            "daily_vector_std": self.daily_vector_std.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "PersonDNA":
        return PersonDNA(
            person_id=d["person_id"],
            app_profiles={k: AppDNA.from_dict(v) for k, v in d.get("app_profiles", {}).items()},
            daily_session_count_mean=d.get("daily_session_count_mean", 0.0),
            daily_session_count_std=d.get("daily_session_count_std", 0.0),
            app_switching_rate=d.get("app_switching_rate", 0.0),
            anchor_centroids=np.array(d.get("anchor_centroids", [])),
            anchor_radii=np.array(d.get("anchor_radii", [])),
            anchor_k=d.get("anchor_k", 0),
            candidate_clusters=[CandidateCluster.from_dict(c) for c in d.get("candidate_clusters", [])],
            promoted_clusters=[PromotedCluster.from_dict(p) for p in d.get("promoted_clusters", [])],
            daily_vector_mean=np.array(d.get("daily_vector_mean", [])),
            daily_vector_std=np.array(d.get("daily_vector_std", [])),
        )


# ============================================================================
# BUILDERS
# ============================================================================

def _epoch_ms_to_day_hour(epoch_ms: int) -> Tuple[int, int]:
    """Convert epoch_ms to (day_of_week 0=Mon, hour 0-23)."""
    import datetime
    dt = datetime.datetime.fromtimestamp(epoch_ms / 1000.0)
    # Python: weekday() returns 0=Monday, 6=Sunday
    return dt.weekday(), dt.hour


def build_app_dna(sessions: List[dict], app_package: str) -> AppDNA:
    """
    Build AppDNA from a list of session dicts for a specific app.

    Session dict keys: app_package, open_timestamp, close_timestamp,
                       trigger, interaction_count
    """
    if not sessions:
        return AppDNA(
            app_id=app_package,
            usage_heatmap=np.zeros((7, 24)),
            avg_session_minutes=0.0,
            session_duration_std=0.0,
            sessions_per_active_day=0.0,
            abandon_rate=0.0,
            self_open_ratio=0.0,
            temporal_anchor_std=0.0,
        )

    # Compute durations in minutes
    durations_min = []
    for s in sessions:
        dur = (s["close_timestamp"] - s["open_timestamp"]) / 60_000.0
        durations_min.append(max(0.0, dur))

    avg_session = float(np.mean(durations_min)) if durations_min else 0.0
    session_std = float(np.std(durations_min)) if len(durations_min) > 1 else 0.0

    # Sessions per active day
    import datetime
    active_days = set()
    for s in sessions:
        dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
        active_days.add(dt.strftime("%Y-%m-%d"))
    sessions_per_active_day = len(sessions) / max(len(active_days), 1)

    # Abandon rate: sessions where interaction_count < 3 AND duration < 30s
    abandoned = sum(
        1 for i, s in enumerate(sessions)
        if s.get("interaction_count", 1) < 3 and durations_min[i] < 0.5
    )
    abandon_rate = abandoned / max(len(sessions), 1)

    # Self-open ratio
    self_opens = sum(1 for s in sessions if s.get("trigger", "SELF") == "SELF")
    self_open_ratio = self_opens / max(len(sessions), 1)

    # Build (7, 24) usage heatmap
    heatmap = np.zeros((7, 24))
    # Track per-day data for temporal_anchor_std
    # Collect per-day minute distributions
    day_hour_minutes = {}  # (date_str, dow, hour) → minutes

    for i, s in enumerate(sessions):
        open_dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
        close_dt = datetime.datetime.fromtimestamp(s["close_timestamp"] / 1000.0)
        dur_ms = s["close_timestamp"] - s["open_timestamp"]
        dur_min = dur_ms / 60_000.0

        dow = open_dt.weekday()
        open_hour = open_dt.hour

        # Distribute duration across occupied hours
        remaining_min = dur_min
        current_hour = open_hour
        current_dt = open_dt
        while remaining_min > 0 and current_dt <= close_dt:
            # Minutes in this hour slot
            next_hour_dt = current_dt.replace(minute=0, second=0, microsecond=0)
            next_hour_dt = next_hour_dt.replace(hour=current_dt.hour) 
            import datetime as _dt
            next_hour_dt = current_dt + _dt.timedelta(hours=1)
            slot_end = min(next_hour_dt, close_dt)
            slot_min = (slot_end - current_dt).total_seconds() / 60.0

            actual_min = min(slot_min, remaining_min)
            h = current_dt.hour
            d = current_dt.weekday()
            heatmap[d][h] += actual_min
            remaining_min -= actual_min
            current_dt = next_hour_dt
            if remaining_min > 0 and current_dt > close_dt:
                break

    # Average heatmap across the number of weeks in the data
    num_weeks = max(len(active_days) / 7.0, 1.0)
    heatmap = heatmap / num_weeks

    # Temporal anchor std: for each day compute center-of-mass hour
    daily_com_hours = []
    # Group sessions by date
    sessions_by_date = {}
    for s in sessions:
        dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        if date_str not in sessions_by_date:
            sessions_by_date[date_str] = []
        sessions_by_date[date_str].append(s)

    for date_str, day_sessions in sessions_by_date.items():
        total_weighted = 0.0
        total_minutes = 0.0
        for s in day_sessions:
            open_dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
            dur_min = (s["close_timestamp"] - s["open_timestamp"]) / 60_000.0
            hour = open_dt.hour + open_dt.minute / 60.0
            total_weighted += hour * dur_min
            total_minutes += dur_min
        if total_minutes > 0:
            com = total_weighted / total_minutes
            daily_com_hours.append(com)

    temporal_anchor_std = float(np.std(daily_com_hours)) if len(daily_com_hours) > 1 else 0.0

    return AppDNA(
        app_id=app_package,
        usage_heatmap=heatmap,
        avg_session_minutes=avg_session,
        session_duration_std=session_std,
        sessions_per_active_day=sessions_per_active_day,
        abandon_rate=abandon_rate,
        self_open_ratio=self_open_ratio,
        temporal_anchor_std=temporal_anchor_std,
    )


def build_daily_vector(
    sessions_today: List[dict],
    app_profiles: Dict[str, AppDNA],
    vector_mean: np.ndarray = None,
    vector_std: np.ndarray = None,
) -> np.ndarray:
    """
    Build a fixed-length vector representing one day of app usage.
    
    For each known app (from baseline): 
        [session_count, avg_duration, self_ratio, center_hour, abandon_rate]
    Cross-app: [total_sessions, switching_rate, active_hours_span]
    Missing apps get zeros.
    Normalize each dimension by baseline mean/std if provided.
    """
    import datetime

    # Per-app today stats
    app_sessions = {}
    for s in sessions_today:
        pkg = s["app_package"]
        if pkg not in app_sessions:
            app_sessions[pkg] = []
        app_sessions[pkg].append(s)

    # Build per-app features
    per_app_features = []
    sorted_apps = sorted(app_profiles.keys())
    
    for pkg in sorted_apps:
        today_sess = app_sessions.get(pkg, [])
        if not today_sess:
            per_app_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        count = len(today_sess)
        durations = [(s["close_timestamp"] - s["open_timestamp"]) / 60_000.0 for s in today_sess]
        avg_dur = float(np.mean(durations))
        self_count = sum(1 for s in today_sess if s.get("trigger", "SELF") == "SELF")
        self_ratio = self_count / max(count, 1)
        
        # Center-of-mass hour
        total_w = 0.0
        total_m = 0.0
        for s in today_sess:
            dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
            dur = (s["close_timestamp"] - s["open_timestamp"]) / 60_000.0
            h = dt.hour + dt.minute / 60.0
            total_w += h * dur
            total_m += dur
        center_hour = total_w / max(total_m, 1.0)

        abandon = sum(1 for i, s in enumerate(today_sess) 
                      if s.get("interaction_count", 1) < 3 and durations[i] < 0.5)
        abandon_rate = abandon / max(count, 1)

        per_app_features.extend([float(count), avg_dur, self_ratio, center_hour, abandon_rate])

    # Cross-app features
    total_sessions = len(sessions_today)
    
    # Switching rate: number of app switches / total sessions
    if len(sessions_today) > 1:
        sorted_sessions = sorted(sessions_today, key=lambda s: s["open_timestamp"])
        switches = sum(1 for i in range(1, len(sorted_sessions))
                       if sorted_sessions[i]["app_package"] != sorted_sessions[i-1]["app_package"])
        switching_rate = switches / max(total_sessions - 1, 1)
    else:
        switching_rate = 0.0

    # Active hours span
    if sessions_today:
        hours = [datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0).hour 
                 for s in sessions_today]
        active_span = max(hours) - min(hours)
    else:
        active_span = 0.0

    per_app_features.extend([float(total_sessions), switching_rate, float(active_span)])

    vector = np.array(per_app_features, dtype=np.float64)

    # Normalize by baseline stats if available
    if vector_mean is not None and vector_std is not None and len(vector_mean) == len(vector):
        std_safe = np.where(vector_std > EPS, vector_std, 1.0)
        vector = (vector - vector_mean) / std_safe

    return vector


def build_person_dna(sessions_28day: List[dict], person_id: str = "user") -> PersonDNA:
    """
    Build a full PersonDNA from 28 days of session data.
    
    1. Build AppDNA for every unique app_package
    2. Build 28 daily vectors
    3. Run K-means with K=3,4,5 — pick K by silhouette score
    4. Store centroids as anchor_centroids, compute anchor_radii
    5. candidate_clusters=[], promoted_clusters=[]
    """
    import datetime

    if not sessions_28day:
        # Return empty DNA
        return PersonDNA(
            person_id=person_id,
            app_profiles={},
            daily_session_count_mean=0.0,
            daily_session_count_std=0.0,
            app_switching_rate=0.0,
            anchor_centroids=np.array([]).reshape(0, 0),
            anchor_radii=np.array([]),
            anchor_k=0,
            candidate_clusters=[],
            promoted_clusters=[],
        )

    # 1. Build AppDNA for each unique package
    packages = set(s["app_package"] for s in sessions_28day)
    app_profiles = {}
    for pkg in packages:
        pkg_sessions = [s for s in sessions_28day if s["app_package"] == pkg]
        app_profiles[pkg] = build_app_dna(pkg_sessions, pkg)

    # 2. Group sessions by date
    sessions_by_date = {}
    for s in sessions_28day:
        dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        if date_str not in sessions_by_date:
            sessions_by_date[date_str] = []
        sessions_by_date[date_str].append(s)

    sorted_dates = sorted(sessions_by_date.keys())

    # Daily session count stats
    daily_counts = [len(sessions_by_date[d]) for d in sorted_dates]
    daily_session_mean = float(np.mean(daily_counts)) if daily_counts else 0.0
    daily_session_std = float(np.std(daily_counts)) if len(daily_counts) > 1 else 0.0

    # App switching rate (overall)
    total_switches = 0
    total_sessions_all = 0
    for date_str in sorted_dates:
        day_sess = sorted(sessions_by_date[date_str], key=lambda s: s["open_timestamp"])
        total_sessions_all += len(day_sess)
        switches = sum(1 for i in range(1, len(day_sess))
                       if day_sess[i]["app_package"] != day_sess[i-1]["app_package"])
        total_switches += switches
    app_switching_rate = total_switches / max(total_sessions_all - 1, 1)

    # Build daily vectors (unnormalized first to compute stats)
    daily_vectors_raw = []
    for date_str in sorted_dates:
        vec = build_daily_vector(sessions_by_date[date_str], app_profiles)
        daily_vectors_raw.append(vec)

    if not daily_vectors_raw:
        return PersonDNA(
            person_id=person_id,
            app_profiles=app_profiles,
            daily_session_count_mean=daily_session_mean,
            daily_session_count_std=daily_session_std,
            app_switching_rate=app_switching_rate,
            anchor_centroids=np.array([]).reshape(0, 0),
            anchor_radii=np.array([]),
            anchor_k=0,
            candidate_clusters=[],
            promoted_clusters=[],
        )

    # Pad vectors to same length if needed
    max_len = max(len(v) for v in daily_vectors_raw)
    padded = []
    for v in daily_vectors_raw:
        if len(v) < max_len:
            v = np.pad(v, (0, max_len - len(v)))
        padded.append(v)
    daily_matrix = np.array(padded)

    # Compute normalization stats
    vec_mean = np.mean(daily_matrix, axis=0)
    vec_std = np.std(daily_matrix, axis=0)

    # Normalize
    std_safe = np.where(vec_std > EPS, vec_std, 1.0)
    daily_matrix_norm = (daily_matrix - vec_mean) / std_safe

    # 3. K-means clustering
    best_k = 3
    best_score = -1.0
    best_centroids = None
    best_labels = None

    n_samples = len(daily_matrix_norm)
    max_k = min(5, n_samples - 1)
    min_k = min(3, n_samples - 1)

    if n_samples < 3:
        # Not enough data for clustering
        return PersonDNA(
            person_id=person_id,
            app_profiles=app_profiles,
            daily_session_count_mean=daily_session_mean,
            daily_session_count_std=daily_session_std,
            app_switching_rate=app_switching_rate,
            anchor_centroids=np.array([]).reshape(0, 0),
            anchor_radii=np.array([]),
            anchor_k=0,
            candidate_clusters=[],
            promoted_clusters=[],
            daily_vector_mean=vec_mean,
            daily_vector_std=vec_std,
        )

    for k in range(min_k, max_k + 1):
        centroids, labels, score = _kmeans_silhouette(daily_matrix_norm, k)
        if score > best_score:
            best_score = score
            best_k = k
            best_centroids = centroids
            best_labels = labels

    # 4. Compute anchor radii (max distance from centroid to any member)
    anchor_radii = np.zeros(best_k)
    if best_centroids is not None and best_labels is not None:
        for cluster_id in range(best_k):
            mask = best_labels == cluster_id
            if np.any(mask):
                members = daily_matrix_norm[mask]
                dists = np.linalg.norm(members - best_centroids[cluster_id], axis=1)
                anchor_radii[cluster_id] = np.max(dists) if len(dists) > 0 else 0.0

    return PersonDNA(
        person_id=person_id,
        app_profiles=app_profiles,
        daily_session_count_mean=daily_session_mean,
        daily_session_count_std=daily_session_std,
        app_switching_rate=app_switching_rate,
        anchor_centroids=best_centroids if best_centroids is not None else np.array([]).reshape(0, 0),
        anchor_radii=anchor_radii,
        anchor_k=best_k,
        candidate_clusters=[],
        promoted_clusters=[],
        daily_vector_mean=vec_mean,
        daily_vector_std=vec_std,
    )


def _kmeans_silhouette(
    data: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run K-means and compute silhouette score.
    Returns (centroids, labels, silhouette_score).
    Simple numpy-only implementation (no sklearn dependency needed).
    """
    n_samples, n_features = data.shape
    if n_samples <= k:
        # Assign each point as its own centroid
        centroids = data[:k]
        labels = np.arange(n_samples)
        return centroids, labels, 0.0

    best_inertia = float('inf')
    best_centroids = None
    best_labels = None

    rng = np.random.RandomState(42)
    for _ in range(n_init):
        # K-means++ initialization
        centroids = np.empty((k, n_features))
        centroids[0] = data[rng.randint(n_samples)]
        for c in range(1, k):
            dists = np.min([np.sum((data - centroids[j]) ** 2, axis=1) for j in range(c)], axis=0)
            probs = dists / dists.sum()
            centroids[c] = data[rng.choice(n_samples, p=probs)]

        # Iterate
        for _ in range(max_iter):
            # Assign
            dists = np.array([np.sum((data - c) ** 2, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            # Update
            new_centroids = np.array([
                data[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c]
                for c in range(k)
            ])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        inertia = sum(np.sum((data[labels == c] - centroids[c]) ** 2) for c in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    # Silhouette score
    sil_score = _silhouette_score(data, best_labels)

    return best_centroids, best_labels, sil_score


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score (simplified, no sklearn)."""
    n = len(data)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or n < 2:
        return 0.0

    scores = np.zeros(n)
    for i in range(n):
        same = data[labels == labels[i]]
        if len(same) > 1:
            a = np.mean(np.linalg.norm(data[i] - same, axis=1)) - 0  # exclude self dist=0 but close enough
        else:
            a = 0.0

        b = float('inf')
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other = data[labels == lbl]
            if len(other) > 0:
                mean_dist = np.mean(np.linalg.norm(data[i] - other, axis=1))
                b = min(b, mean_dist)

        if max(a, b) > 0:
            scores[i] = (b - a) / max(a, b)
        else:
            scores[i] = 0.0

    return float(np.mean(scores))
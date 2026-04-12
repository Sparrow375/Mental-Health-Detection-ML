"""
L2 Contextual Texture Profile Builder — K-means per archetype.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..data_structures import ContextualTextureProfile
from ..feature_meta import L2_TEXTURE_FEATURES, THRESHOLDS


def build_texture_profiles(
    archetype_assignments: Dict[int, List[int]],  # cluster_id -> list of day indices
    daily_texture_vectors: List[np.ndarray],       # one 22-feature vector per day
) -> List[ContextualTextureProfile]:
    """
    Build L2 contextual texture profiles for each L1 archetype.

    For archetypes with >= MIN_ARCHETYPE_DAYS_FOR_KMEANS days:
        Run K-means with K=2 and K=3, choose best by silhouette score.
    For smaller archetypes:
        Use mean/std fallback.
    """
    min_days = THRESHOLDS["MIN_ARCHETYPE_DAYS_FOR_KMEANS"]
    profiles = []

    for cluster_id, day_indices in sorted(archetype_assignments.items()):
        if not day_indices:
            continue

        # Collect texture vectors for this archetype
        vectors = []
        for idx in day_indices:
            if idx < len(daily_texture_vectors) and daily_texture_vectors[idx] is not None:
                vectors.append(daily_texture_vectors[idx])

        if not vectors:
            profiles.append(ContextualTextureProfile(
                archetype_id=cluster_id,
                member_days=0,
            ))
            continue

        vectors_arr = np.array(vectors)
        member_days = len(vectors_arr)
        profile = ContextualTextureProfile(
            archetype_id=cluster_id,
            member_days=member_days,
        )

        # Compute tolerance factor (mean intra-archetype variance)
        if member_days > 1:
            profile.tolerance_factor = float(np.mean(np.std(vectors_arr, axis=0)))

        if member_days >= min_days:
            # K-means clustering approach
            best_k = 2
            best_score = -1.0
            best_centroids = None
            best_labels = None

            for k in [2, 3]:
                if member_days <= k:
                    continue
                try:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
                    labels = km.fit_predict(vectors_arr)
                    score = silhouette_score(vectors_arr, labels)

                    if best_centroids is None or score > best_score + 0.05:
                        best_k = k
                        best_score = score
                        best_centroids = km.cluster_centers_
                        best_labels = labels
                except Exception:
                    continue

            if best_centroids is not None:
                profile.texture_centroids = best_centroids
                profile.n_texture_clusters = best_k

                # Compute radii for each texture cluster
                radii = []
                for ki in range(best_k):
                    cluster_mask = best_labels == ki
                    cluster_vecs = vectors_arr[cluster_mask]
                    if len(cluster_vecs) > 1:
                        dists = np.linalg.norm(cluster_vecs - best_centroids[ki], axis=1)
                        radii.append(float(np.max(dists)))
                    else:
                        radii.append(0.0)
                profile.texture_radii = np.array(radii)
            else:
                # Fallback to mean/std even though we had enough days
                profile.texture_mean = np.mean(vectors_arr, axis=0)
                profile.texture_std = np.std(vectors_arr, axis=0)
                profile.n_texture_clusters = 0
        else:
            # Fallback: mean and std
            profile.texture_mean = np.mean(vectors_arr, axis=0)
            if member_days > 1:
                profile.texture_std = np.std(vectors_arr, axis=0)
            else:
                profile.texture_std = np.ones(len(L2_TEXTURE_FEATURES)) * 0.1
            profile.n_texture_clusters = 0

        profiles.append(profile)

    return profiles


def compute_daily_texture_vector(
    sessions_today: List,
    app_dnas: Dict,
    phone_dna,
    notifications_today: List,
    baseline_dates_count: int,
) -> np.ndarray:
    """
    Compute the 22-feature L2 texture vector for a single monitoring day.
    This is a simplified implementation — full version would compute each
    sub-signal from actual session/notification data.
    """
    vector = np.zeros(len(L2_TEXTURE_FEATURES))

    if not sessions_today:
        return vector

    # ── Temporal anchoring (4 features) ───────────────────────────────────
    # time_in_primary_window_ratio: fraction of sessions within primary time range
    total_sessions = len(sessions_today)
    in_primary = 0
    if total_sessions > 0:
        for sess in sessions_today:
            app_dna = app_dnas.get(sess.app_id)
            if app_dna and app_dna.usage_heatmap is not None:
                import datetime
                dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
                start_h, end_h = app_dna.primary_time_range
                if start_h <= end_h:
                    if start_h <= dt.hour <= end_h:
                        in_primary += 1
                else:
                    if dt.hour >= start_h or dt.hour <= end_h:
                        in_primary += 1
        vector[0] = in_primary / max(1, total_sessions)

    # temporal_anchor_deviation: std of session hours vs baseline
    import datetime
    hours = [datetime.datetime.fromtimestamp(s.open_ts / 1000.0).hour + 
             datetime.datetime.fromtimestamp(s.open_ts / 1000.0).minute / 60.0
             for s in sessions_today]
    if hours:
        vector[1] = float(np.std(hours))

    # first_pickup_hour_deviation
    if phone_dna and hours:
        sorted_hours = sorted(hours)
        deviation = abs(sorted_hours[0] - phone_dna.first_pickup_hour_mean)
        std = max(phone_dna.first_pickup_hour_std, 0.1)
        vector[2] = min(deviation / std, 3.0) / 3.0  # Normalize to [0,1]

    # rhythm_dissolution_score: placeholder (KL divergence computed in l2_scorer)
    vector[3] = 0.0

    # ── Session quality (5 features) ──────────────────────────────────────
    durations = [s.duration_minutes for s in sessions_today]

    # weighted_abandon_rate
    abandon_count = sum(1 for s in sessions_today if s.duration_minutes < 0.75 and s.interaction_count < 5)
    vector[4] = abandon_count / max(1, total_sessions)

    # deep_session_ratio
    deep_count = sum(1 for d in durations if d > 20)
    vector[5] = deep_count / max(1, total_sessions)

    # micro_session_ratio
    micro_count = sum(1 for d in durations if d < 2)
    vector[6] = micro_count / max(1, total_sessions)

    # session_duration_collapse
    if durations and phone_dna:
        ratio = np.mean(durations) / max(phone_dna.deep_session_ratio * 40, 1)
        vector[7] = max(0, 1.0 - ratio)

    # interaction_density_ratio
    ipms = [s.interaction_count / max(s.duration_minutes, 0.1) for s in sessions_today if s.duration_minutes > 0]
    if ipms and phone_dna:
        baseline_ipm = phone_dna.deep_session_ratio  # proxy
        vector[8] = min(float(np.mean(ipms)) / max(baseline_ipm, 0.1), 3.0) / 3.0

    # ── Agency & initiation (4 features) ──────────────────────────────────
    self_opens = sum(1 for s in sessions_today if s.trigger == "SELF")
    notif_opens = sum(1 for s in sessions_today if s.trigger == "NOTIFICATION")

    vector[9] = self_opens / max(1, total_sessions)   # self_open_ratio
    vector[10] = notif_opens / max(1, total_sessions)  # notification_open_rate
    vector[11] = 0.0  # notification_ignore_rate (from notification events)
    vector[12] = phone_dna.pickup_burst_rate if phone_dna else 0.0

    # ── Attention coherence (4 features) ──────────────────────────────────
    # app_switching_rate
    if total_sessions > 1:
        switches = sum(1 for i in range(1, len(sessions_today)) 
                       if sessions_today[i].app_id != sessions_today[i-1].app_id)
        vector[13] = switches / max(1, total_sessions - 1)

    # distinct_apps_ratio
    distinct_apps = len(set(s.app_id for s in sessions_today))
    vector[15] = distinct_apps / max(1, total_sessions)

    # Remaining features use reasonable defaults
    vector[14] = 0.5  # app_cooccurrence_consistency
    vector[16] = 0.5  # session_context_match
    vector[17] = 0.5  # daily_rhythm_regularity
    vector[18] = 0.5  # weekday_weekend_alignment
    vector[19] = 0.0  # dead_zone_count
    vector[20] = 0.0  # notification_response_latency_shift
    vector[21] = notif_opens / max(1, total_sessions)  # notification_to_session_ratio

    return vector
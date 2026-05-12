"""
System 1 Profile Builder — builds PersonProfile from 28-day baseline data.

Adapted from system1/data_structures.py and system1/baseline/ to work
within the Chaquopy environment. Produces a rich JSON profile containing:
  - PersonalityVector (29-feature baseline means/variances)
  - AppDNA (per-app behavioral fingerprints)
  - PhoneDNA (device-level behavioral DNA)
  - AnchorClusters (Clinical-Weighted PCA + Mean-Shift archetypes from L1 daily vectors)
  - ContextualTextureProfiles (L2 texture per archetype)
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# ── Feature metadata ──────────────────────────────────────────────────────────

FEATURE_WEIGHTS = {
    "screenTimeHours": 1.4, "unlockCount": 1.2, "appLaunchCount": 0.9,
    "notificationsToday": 0.8, "socialAppRatio": 1.3,
    "callsPerDay": 1.3, "callDurationMinutes": 1.2, "uniqueContacts": 1.1,
    "conversationFrequency": 0.9,
    "dailyDisplacementKm": 1.5, "locationEntropy": 1.3, "homeTimeRatio": 1.2,
    "wakeTimeHour": 1.4, "sleepTimeHour": 1.3, "sleepDurationHours": 1.6,
    "darkDurationHours": 1.0,
    "chargeDurationHours": 0.8, "memoryUsagePercent": 0.5, "networkWifiMB": 0.6,
    "networkMobileMB": 0.6, "storageUsedGB": 0.4,
    "totalAppsCount": 0.8, "upiTransactionsToday": 1.1, "appUninstallsToday": 0.9,
    "appInstallsToday": 0.8,
    "calendarEventsToday": 0.9, "mediaCountToday": 0.7, "downloadsToday": 0.6,
    "musicTimeMinutes": 1.1,
}

ALL_L1_FEATURES = list(FEATURE_WEIGHTS.keys())

L1_CLUSTER_FEATURES = [
    "sleepDurationHours", "wakeTimeHour", "sleepTimeHour",
    "dailyDisplacementKm", "locationEntropy",
    "callsPerDay", "callDurationMinutes", "screenTimeHours",
    "unlockCount", "socialAppRatio", "darkDurationHours",
]


# ── Clinical-Weighted PCA (2D) + Mean-Shift clustering ──────────────────────

def _clinical_weighted_pca(data: np.ndarray, feature_weights: list, n_components: int = 2) -> tuple:
    """Apply clinical feature weights, then PCA to 2D via SVD."""
    W = np.diag(feature_weights)
    weighted = data @ W
    mean = weighted.mean(axis=0)
    centered = weighted - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    projected = centered @ components.T
    return projected, components, mean


def _meanshift(data: np.ndarray, bandwidth: float = None) -> list:
    """Mean-Shift clustering (pure numpy). Returns [(cluster_id, indices)]."""
    if len(data) == 0:
        return []
    n = len(data)
    # Auto-estimate bandwidth from pairwise distances
    if bandwidth is None:
        dists = np.linalg.norm(data[:, None] - data[None, :], axis=2)
        bandwidth = float(np.median(dists)) * 0.8
    if bandwidth <= 0:
        bandwidth = 1.0

    points = data.copy()
    # Shift each point toward mode
    for _ in range(50):
        shifted = np.zeros_like(points)
        for i in range(n):
            dists = np.linalg.norm(points - points[i], axis=1)
            mask = dists <= bandwidth
            if mask.any():
                shifted[i] = points[mask].mean(axis=0)
            else:
                shifted[i] = points[i]
        points = shifted

    # Merge converged points into clusters
    cluster_centers = []
    cluster_ids = [-1] * n
    merge_thresh = bandwidth * 0.3
    for i in range(n):
        merged = False
        for c_idx, center in enumerate(cluster_centers):
            if np.linalg.norm(points[i] - center) < merge_thresh:
                cluster_ids[i] = c_idx
                merged = True
                break
        if not merged:
            cluster_ids[i] = len(cluster_centers)
            cluster_centers.append(points[i].copy())

    clusters = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)
    return [(cid, indices) for cid, indices in sorted(clusters.items())]


# ── Profile Builder ───────────────────────────────────────────────────────────

def build_personality_vector(daily_features_list: list) -> dict:
    """Build 29-feature personality vector from daily features."""
    if not daily_features_list:
        return {"means": {}, "variances": {}, "confidence": "LOW", "feature_count": 0}

    means = {}
    variances = {}
    for feat in ALL_L1_FEATURES:
        values = [d.get(feat, 0.0) for d in daily_features_list if feat in d]
        if values:
            means[feat] = round(float(np.mean(values)), 4)
            variances[feat] = round(float(np.std(values)), 4)
        else:
            means[feat] = 0.0
            variances[feat] = 0.0

    n = len(daily_features_list)
    if n >= 21:
        confidence = "HIGH"
    elif n >= 14:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "means": means,
        "variances": variances,
        "confidence": confidence,
        "feature_count": len(means),
        "days_used": n,
    }


def build_app_dna_profiles(sessions: list) -> dict:
    """Build per-app DNA profiles from session data."""
    from dna import build_app_dna

    if not sessions:
        return {}

    packages = set(s.get("app_package", "") for s in sessions)
    profiles = {}
    for pkg in packages:
        if not pkg:
            continue
        pkg_sessions = [s for s in sessions if s.get("app_package") == pkg]
        try:
            app_dna = build_app_dna(pkg_sessions, pkg)
            profiles[pkg] = app_dna.to_dict()
        except Exception:
            # Fallback minimal profile
            profiles[pkg] = {
                "app_id": pkg,
                "avg_session_minutes": 0.0,
                "session_duration_std": 0.0,
                "sessions_per_active_day": float(len(pkg_sessions)),
                "abandon_rate": 0.0,
                "self_open_ratio": 0.0,
                "temporal_anchor_std": 0.0,
            }

    # Return top 15 apps by session count (keeps JSON manageable)
    pkg_counts = {}
    for s in sessions:
        pkg = s.get("app_package", "")
        pkg_counts[pkg] = pkg_counts.get(pkg, 0) + 1

    top_pkgs = sorted(pkg_counts.keys(), key=lambda p: pkg_counts[p], reverse=True)[:15]
    return {pkg: profiles[pkg] for pkg in top_pkgs if pkg in profiles}


def build_phone_dna(daily_features_list: list, sessions: list) -> dict:
    """Build device-level behavioral DNA with full metric computation."""
    import datetime

    if not daily_features_list and not sessions:
        return {}

    # First pickup hour from wake times
    wake_hours = [d.get("wakeTimeHour", 0.0) for d in daily_features_list if "wakeTimeHour" in d]
    first_pickup_mean = float(np.mean(wake_hours)) if wake_hours else 0.0
    first_pickup_std = float(np.std(wake_hours)) if len(wake_hours) > 1 else 0.0

    # Active window duration (proxy from sleep duration + screen time)
    screen_times = [d.get("screenTimeHours", 0.0) for d in daily_features_list if "screenTimeHours" in d]
    active_window_mean = float(np.mean(screen_times)) if screen_times else 0.0
    active_window_std = float(np.std(screen_times)) if len(screen_times) > 1 else 0.0

    # Session duration distribution (5-bin histogram matching phone_dna_builder.py)
    durations = []
    for s in sessions:
        # Prefer the pre-computed key written by Kotlin's sessionsToJson()
        if "duration_minutes" in s:
            dur = float(s["duration_minutes"])
        else:
            dur = (s.get("close_timestamp", 0) - s.get("open_timestamp", 0)) / 60000.0
        durations.append(max(0.0, dur))

    if durations:
        bins = [0, 2, 15, 30, 60, float('inf')]
        hist = [0] * 5
        for d in durations:
            for i in range(5):
                if bins[i] <= d < bins[i + 1]:
                    hist[i] += 1
                    break
        total = sum(hist)
        session_dist = [round(h / max(total, 1), 4) for h in hist]
    else:
        session_dist = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Deep/micro session ratios (matching phone_dna_builder.py thresholds)
    if durations:
        deep = sum(1 for d in durations if d > 20) / len(durations)
        micro = sum(1 for d in durations if d < 2) / len(durations)
    else:
        deep, micro = 0.0, 0.0

    # Pickup burst rate: fraction of sessions within 5 min of previous session
    burst_rate = 0.0
    inter_pickup_mean = 0.0
    inter_pickup_std = 0.0
    if sessions:
        timestamps = sorted(float(s.get("open_timestamp", 0)) for s in sessions if s.get("open_timestamp", 0) > 0)
        if len(timestamps) > 1:
            gaps = [(timestamps[i] - timestamps[i - 1]) / 60000.0 for i in range(1, len(timestamps))]
            burst_count = sum(1 for g in gaps if g < 5)
            burst_rate = round(burst_count / len(gaps), 4)
            # Inter-pickup interval (exclude gaps > 24h)
            valid_gaps = [g for g in gaps if 0 < g < 1440]
            if valid_gaps:
                inter_pickup_mean = round(float(np.mean(valid_gaps)), 2)
                inter_pickup_std = round(float(np.std(valid_gaps)), 2)

    # Notification rates from notification event data (passed in sessions as notification_events)
    notification_events = []
    # Check if notification_events were passed separately
    notif_data = []
    if isinstance(sessions, dict) and "notification_events" in sessions:
        notif_data = sessions["notification_events"]
    # Also try to get from daily features
    if not notif_data and daily_features_list:
        # Estimate from notification counts in daily features
        notif_counts = [d.get("notificationsToday", 0.0) for d in daily_features_list if "notificationsToday" in d]
        notif_rate = float(np.mean(notif_counts)) if notif_counts else 0.0
    else:
        notif_rate = 0.0

    # Compute notification open/dismiss/ignore rates from events
    notif_open_rate = 0.0
    notif_dismiss_rate = 0.0
    notif_ignore_rate = 0.0
    if notif_data:
        actions = [n.get("action", "") for n in notif_data]
        n_total = max(len(actions), 1)
        notif_open_rate = round(sum(1 for a in actions if a == "TAP") / n_total, 4)
        notif_dismiss_rate = round(sum(1 for a in actions if a == "DISMISS") / n_total, 4)
        notif_ignore_rate = round(sum(1 for a in actions if a == "IGNORE") / n_total, 4)

    # App cooccurrence (simplified: top 10 apps × top 10 apps)
    pkg_set = sorted(set(s.get("app_package", "") for s in sessions if isinstance(s, dict)))[:10]
    cooccurrence = [[0] * len(pkg_set) for _ in range(len(pkg_set))]
    if sessions:
        sessions_sorted = sorted(sessions, key=lambda s: s.get("open_timestamp", 0))
        for i in range(1, len(sessions_sorted)):
            p1 = sessions_sorted[i - 1].get("app_package", "")
            p2 = sessions_sorted[i].get("app_package", "")
            if p1 in pkg_set and p2 in pkg_set:
                idx1 = pkg_set.index(p1)
                idx2 = pkg_set.index(p2)
                cooccurrence[idx1][idx2] += 1

    # Daily rhythm regularity (std of screen time across days)
    rhythm_regularity = round(1.0 - min(active_window_std / max(active_window_mean, 0.1), 1.0), 4)

    # Weekday-weekend delta
    weekday_screen = []
    weekend_screen = []
    for d in daily_features_list:
        if "screenTimeHours" in d and "__day_of_week" in d:
            if d["__day_of_week"] < 5:
                weekday_screen.append(d["screenTimeHours"])
            else:
                weekend_screen.append(d["screenTimeHours"])
    wd_mean = float(np.mean(weekday_screen)) if weekday_screen else 0.0
    we_mean = float(np.mean(weekend_screen)) if weekend_screen else 0.0
    wd_we_delta = round(abs(wd_mean - we_mean), 4)

    return {
        "first_pickup_hour_mean": round(first_pickup_mean, 2),
        "first_pickup_hour_std": round(first_pickup_std, 2),
        "active_window_duration_mean": round(active_window_mean, 2),
        "active_window_duration_std": round(active_window_std, 2),
        "pickup_burst_rate": burst_rate,
        "inter_pickup_interval_mean": inter_pickup_mean,
        "inter_pickup_interval_std": inter_pickup_std,
        "session_duration_distribution": session_dist,
        "deep_session_ratio": round(deep, 4),
        "micro_session_ratio": round(micro, 4),
        "notification_open_rate": notif_open_rate,
        "notification_dismiss_rate": notif_dismiss_rate,
        "notification_ignore_rate": notif_ignore_rate,
        "daily_rhythm_regularity": rhythm_regularity,
        "weekday_weekend_delta": wd_we_delta,
        "app_cooccurrence_labels": pkg_set,
        "app_cooccurrence_matrix": cooccurrence,
    }


def build_anchor_clusters(daily_features_list: list) -> list:
    """Build L1 anchor clusters using Clinical-Weighted PCA (2D) + Mean-Shift."""
    if len(daily_features_list) < 3:
        return []

    # Extract L1 cluster feature vectors — use 0.0 for missing features rather than
    # dropping the entire day, so all history contributes to clustering.
    vectors = []
    dates = []
    for day in daily_features_list:
        vec = [float(day.get(feat, 0.0)) for feat in L1_CLUSTER_FEATURES]
        vectors.append(vec)
        dates.append(day.get("date", ""))

    if len(vectors) < 3:
        return []  # Should never hit given the guard above, but kept for safety

    matrix = np.array(vectors)

    # Normalize (z-score)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    matrix_norm = (matrix - means) / stds_safe

    # Build clinical weight vector for clustering features
    weights = [FEATURE_WEIGHTS.get(f, 1.0) for f in L1_CLUSTER_FEATURES]

    # Clinical-Weighted PCA → 2D projection
    projected, pca_components, pca_mean = _clinical_weighted_pca(matrix_norm, weights, n_components=2)

    # ── Projection params for future mismatch scoring ──────────────────────────
    # Store the parameters needed to project any new day into the same PCA space.
    # Serialised as plain Python lists so they survive JSON round-trips.
    _proj_params = {
        "features": L1_CLUSTER_FEATURES,
        "norm_means": [round(float(v), 6) for v in means],
        "norm_stds":  [round(float(v), 6) for v in stds_safe],
        "clinical_weights": [round(float(w), 4) for w in weights],
        "pca_mean": [round(float(v), 6) for v in pca_mean],
        "pca_components": [
            [round(float(v), 6) for v in row] for row in pca_components
        ],
    }

    # Mean-Shift clustering on 2D projection
    clusters = _meanshift(projected)

    # If Mean-Shift found no clusters or everything is one blob, that's OK
    if not clusters:
        # Single cluster fallback
        centroid = np.mean(projected, axis=0)
        return [{
            "cluster_id": 0,
            "centroid_features": {feat: round(float(means[i]), 4) for i, feat in enumerate(L1_CLUSTER_FEATURES)},
            "centroid_pca_2d": [round(float(centroid[0]), 4), round(float(centroid[1]), 4)],
            "radius": round(float(np.max(np.linalg.norm(projected - centroid, axis=1))), 4),
            "member_count": len(projected),
            "member_dates": dates,
            "method": "clinical_pca_meanshift",
            "_pca_projection": _proj_params,  # stored on first (only) cluster
        }]

    result = []
    for idx, (cid, indices) in enumerate(clusters):
        members_pca = projected[indices]
        centroid_pca = np.mean(members_pca, axis=0)

        # Radius in PCA space
        dists = np.linalg.norm(members_pca - centroid_pca, axis=1)
        radius = float(np.max(dists)) if len(dists) > 0 else 0.0

        # Denormalize centroid back to original feature space (approximate)
        members_norm = matrix_norm[indices]
        centroid_norm = np.mean(members_norm, axis=0)
        centroid_raw = centroid_norm * stds_safe + means

        cluster_dict = {
            "cluster_id": cid,
            "centroid_features": {feat: round(float(centroid_raw[i]), 4) for i, feat in enumerate(L1_CLUSTER_FEATURES)},
            "centroid_pca_2d": [round(float(centroid_pca[0]), 4), round(float(centroid_pca[1]), 4)],
            "radius": round(radius, 4),
            "member_count": len(indices),
            "member_dates": [dates[i] for i in indices if i < len(dates)],
            "method": "clinical_pca_meanshift",
        }
        # Attach projection params only to the first cluster (shared across all)
        if idx == 0:
            cluster_dict["_pca_projection"] = _proj_params
        result.append(cluster_dict)

    return result


def build_texture_profiles(daily_features_list: list, sessions: list, anchor_clusters: list = None) -> list:
    """
    Build simplified L2 texture profiles, one per anchor cluster archetype.

    If anchor_clusters is provided, each day is assigned to its nearest cluster
    and texture stats are computed per-cluster. Falls back to a single global
    profile when no cluster information is available.
    """
    if not daily_features_list or not sessions:
        return []

    import datetime as _dt

    # Group sessions by date
    sessions_by_date = {}
    for s in sessions:
        ts = s.get("open_timestamp", 0)
        try:
            dt = _dt.datetime.fromtimestamp(ts / 1000.0)
            date_str = dt.strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            date_str = ""
        if date_str:
            sessions_by_date.setdefault(date_str, []).append(s)

    def _texture_for_day(day_sessions: list) -> dict:
        """Compute texture feature dict for a single day's sessions."""
        durations = [
            (s.get("close_timestamp", 0) - s.get("open_timestamp", 0)) / 60000.0
            for s in day_sessions
        ]
        durations = [max(0.0, dur) for dur in durations]

        abandoned = sum(
            1 for s, dur in zip(day_sessions, durations)
            if s.get("interaction_count", 1) < 3 and dur < 0.5
        )
        abandon_rate = abandoned / max(len(day_sessions), 1)
        self_opens = sum(1 for s in day_sessions if s.get("trigger", "SELF") == "SELF")
        self_open_ratio = self_opens / max(len(day_sessions), 1)
        deep_ratio = sum(1 for dur in durations if dur >= 15) / max(len(durations), 1)
        micro_ratio = sum(1 for dur in durations if dur < 1) / max(len(durations), 1)

        sorted_sess = sorted(day_sessions, key=lambda s: s.get("open_timestamp", 0))
        switches = sum(
            1 for i in range(1, len(sorted_sess))
            if sorted_sess[i].get("app_package") != sorted_sess[i - 1].get("app_package")
        )
        switch_rate = switches / max(len(sorted_sess) - 1, 1)

        try:
            hours = [
                _dt.datetime.fromtimestamp(s.get("open_timestamp", 0) / 1000.0).hour
                for s in day_sessions
            ]
            active_span = (max(hours) - min(hours)) if hours else 0
        except (OSError, OverflowError, ValueError):
            active_span = 0

        return {
            "total_sessions": len(day_sessions),
            "abandon_rate": round(abandon_rate, 4),
            "self_open_ratio": round(self_open_ratio, 4),
            "deep_session_ratio": round(deep_ratio, 4),
            "micro_session_ratio": round(micro_ratio, 4),
            "app_switching_rate": round(switch_rate, 4),
            "active_hours_span": active_span,
            "avg_session_minutes": round(float(np.mean(durations)) if durations else 0.0, 2),
        }

    def _aggregate(texture_list: list) -> dict:
        """Aggregate a list of per-day texture dicts into summary stats."""
        if not texture_list:
            return {}
        keys = ["total_sessions", "abandon_rate", "self_open_ratio",
                "deep_session_ratio", "micro_session_ratio",
                "app_switching_rate", "avg_session_minutes"]
        return {
            "total_days_analyzed": len(texture_list),
            **{f"avg_{k}": round(float(np.mean([t[k] for t in texture_list])), 4)
               for k in keys},
        }

    # ── Build per-day texture rows ────────────────────────────────────────────
    # Map date → texture dict; also keep the cluster_id from anchor_clusters.
    date_to_cluster: dict = {}
    if anchor_clusters:
        for cluster in anchor_clusters:
            for date in cluster.get("member_dates", []):
                date_to_cluster[date] = cluster.get("cluster_id", 0)

    # Collect texture per day, tagged with cluster_id
    day_textures: list = []  # list of (cluster_id, texture_dict, date_str)
    for day in daily_features_list:
        date_str = day.get("date", "")
        day_sessions = sessions_by_date.get(date_str, [])
        if not day_sessions:
            continue
        tex = _texture_for_day(day_sessions)
        tex["date"] = date_str
        cluster_id = date_to_cluster.get(date_str, 0)
        day_textures.append((cluster_id, tex))

    if not day_textures:
        return []

    # ── If cluster info exists, produce one profile per cluster ───────────────
    if anchor_clusters and len(anchor_clusters) > 1:
        cluster_groups: dict = {}
        for cid, tex in day_textures:
            cluster_groups.setdefault(cid, []).append(tex)

        result = []
        for cluster in anchor_clusters:
            cid = cluster.get("cluster_id", 0)
            group = cluster_groups.get(cid, [])
            agg = _aggregate(group)
            agg["daily_breakdown"] = group[-14:]
            result.append({
                "archetype_id": cid,
                "member_days": len(group),
                "centroid_pca_2d": cluster.get("centroid_pca_2d", [0.0, 0.0]),
                "texture_summary": agg,
            })
        return result

    # ── Single global profile fallback ────────────────────────────────────────
    all_textures = [tex for _, tex in day_textures]
    agg = _aggregate(all_textures)
    agg["daily_breakdown"] = all_textures[-14:]
    return [{
        "archetype_id": 0,
        "member_days": len(all_textures),
        "texture_summary": agg,
    }]


def build_full_profile(
    daily_features_list: list,
    sessions: list,
    person_id: str = "user"
) -> dict:
    """
    Build a complete System 1 PersonProfile.

    Args:
        daily_features_list: List of daily feature dicts (from 28-day baseline)
        sessions: List of session dicts from baseline period
        person_id: User identifier

    Returns:
        Dict containing the full profile, serializable to JSON.
    """
    print(f"  [ProfileBuilder] Starting build for {person_id}")
    print(f"  [ProfileBuilder] Samples: {len(daily_features_list)} days, {len(sessions)} sessions")
    
    personality_vector = build_personality_vector(daily_features_list)
    print("  [ProfileBuilder] Personality vector built")
    
    app_dna_profiles = build_app_dna_profiles(sessions)
    print(f"  [ProfileBuilder] App DNA built: {len(app_dna_profiles)} apps")
    
    phone_dna = build_phone_dna(daily_features_list, sessions)
    print("  [ProfileBuilder] Phone DNA built")
    
    anchor_clusters = build_anchor_clusters(daily_features_list)
    print(f"  [ProfileBuilder] Anchor clusters built: {len(anchor_clusters)}")

    # Pass anchor_clusters so texture profiles are split per archetype
    texture_profiles = build_texture_profiles(daily_features_list, sessions, anchor_clusters)
    print(f"  [ProfileBuilder] Texture profiles built: {len(texture_profiles)}")

    # Weighted feature importance (for UI display)
    feature_importance = {}
    for feat in ALL_L1_FEATURES:
        mean_val = personality_vector["means"].get(feat, 0.0)
        std_val = personality_vector["variances"].get(feat, 0.0)
        weight = FEATURE_WEIGHTS.get(feat, 1.0)
        feature_importance[feat] = {
            "mean": mean_val,
            "std": std_val,
            "weight": weight,
            "group": _get_feature_group(feat),
        }

    # Group statistics
    groups = {}
    for feat, info in feature_importance.items():
        grp = info["group"]
        groups.setdefault(grp, []).append(feat)

    group_summaries = {}
    for grp, feats in groups.items():
        weights = [FEATURE_WEIGHTS.get(f, 1.0) for f in feats]
        group_summaries[grp] = {
            "features": feats,
            "avg_weight": round(float(np.mean(weights)), 2),
            "total_weight": round(float(sum(weights)), 2),
        }

    # Extract PCA projection params from the first anchor cluster (if available)
    pca_projection = None
    if anchor_clusters:
        pca_projection = anchor_clusters[0].get("_pca_projection", None)

    return {
        "person_id": person_id,
        "profile_version": "1.0",
        "built_at": datetime.utcnow().isoformat() + "Z",
        "days_of_data": len(daily_features_list),
        "personality_vector": personality_vector,
        "app_dna_profiles": app_dna_profiles,
        "phone_dna": phone_dna,
        "anchor_clusters": anchor_clusters,
        "texture_profiles": texture_profiles,
        "feature_importance": feature_importance,
        "group_summaries": group_summaries,
        "pca_projection": pca_projection,  # projection params for mismatch scoring
    }


def compute_cluster_mismatch(today_features: dict, profile_data: dict) -> float:
    """
    Project today's feature vector into the baseline PCA space and compute how
    far it falls from the nearest anchor cluster centroid, normalised by the
    cluster radius.

    Returns
    -------
    mismatch : float in [0.0, 1.0]
        0.0 → today sits well inside a known archetype (no penalty)
        1.0 → today is maximally distant from all known archetypes

    The score is intended to be used as an amplifier on the existing l2_modifier:
        l2_final = l2_modifier * (1.0 + MISMATCH_WEIGHT * mismatch)
    where MISMATCH_WEIGHT controls the maximum additional amplification.
    """
    try:
        proj_params = profile_data.get("pca_projection")
        anchor_clusters = profile_data.get("anchor_clusters", [])

        # Need at least projection params and one cluster with a centroid
        if not proj_params or not anchor_clusters:
            return 0.0

        features   = proj_params["features"]
        norm_means = np.array(proj_params["norm_means"], dtype=float)
        norm_stds  = np.array(proj_params["norm_stds"],  dtype=float)
        clin_w     = np.array(proj_params["clinical_weights"], dtype=float)
        pca_mean   = np.array(proj_params["pca_mean"],   dtype=float)
        pca_comps  = np.array(proj_params["pca_components"], dtype=float)  # shape (2, N)

        # Build today's raw feature vector using the same feature order
        raw_vec = np.array([float(today_features.get(f, 0.0)) for f in features])

        # Apply the same normalisation pipeline as during baseline
        z_vec = (raw_vec - norm_means) / norm_stds          # z-score normalise
        w_vec = z_vec * clin_w                              # clinical weighting
        today_pca = (w_vec - pca_mean) @ pca_comps.T       # project → 2D

        # Compute distance from today's projection to each cluster centroid
        best_excess = None
        for cluster in anchor_clusters:
            centroid = np.array(cluster.get("centroid_pca_2d", [0.0, 0.0]), dtype=float)
            radius   = float(cluster.get("radius", 0.0))

            dist = float(np.linalg.norm(today_pca - centroid))

            # How far beyond the cluster radius is today's point?
            # 0 if inside the cluster, positive if outside.
            excess = max(0.0, dist - radius)

            # Normalise: treat 2× the radius as the "full mismatch" distance
            # to avoid penalising mildly-out-of-cluster days too harshly.
            norm_radius = max(radius, 1e-6)
            norm_excess = excess / (2.0 * norm_radius)

            if best_excess is None or norm_excess < best_excess:
                best_excess = norm_excess

        # Clamp to [0, 1]
        return float(min(best_excess, 1.0)) if best_excess is not None else 0.0

    except Exception as e:
        print("  [ClusterMismatch] Failed to compute mismatch: {}".format(e))
        return 0.0


def _get_feature_group(feat: str) -> str:
    """Return the group name for a feature."""
    groups = {
        "screen_app": ["screenTimeHours", "unlockCount", "appLaunchCount",
                       "notificationsToday", "socialAppRatio"],
        "communication": ["callsPerDay", "callDurationMinutes", "uniqueContacts",
                          "conversationFrequency"],
        "location": ["dailyDisplacementKm", "locationEntropy", "homeTimeRatio"],
        "sleep": ["wakeTimeHour", "sleepTimeHour", "sleepDurationHours",
                  "darkDurationHours"],
        "system": ["chargeDurationHours", "memoryUsagePercent", "networkWifiMB",
                   "networkMobileMB", "storageUsedGB"],
        "behavioral": ["totalAppsCount", "upiTransactionsToday", "appUninstallsToday",
                       "appInstallsToday"],
        "engagement": ["calendarEventsToday", "mediaCountToday", "downloadsToday",
                       "musicTimeMinutes"],
    }
    for grp, feats in groups.items():
        if feat in feats:
            return grp
    return "unknown"

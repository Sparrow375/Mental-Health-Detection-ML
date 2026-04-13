"""
System 1 Profile Builder — builds PersonProfile from 28-day baseline data.

Adapted from system1/data_structures.py and system1/baseline/ to work
within the Chaquopy environment. Produces a rich JSON profile containing:
  - PersonalityVector (29-feature baseline means/variances)
  - AppDNA (per-app behavioral fingerprints)
  - PhoneDNA (device-level behavioral DNA)
  - AnchorClusters (DBSCAN archetypes from L1 daily vectors)
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
    "placesVisited": 1.1,
    "wakeTimeHour": 1.4, "sleepTimeHour": 1.3, "sleepDurationHours": 1.6,
    "darkDurationHours": 1.0,
    "chargeDurationHours": 0.8, "memoryUsagePercent": 0.5, "networkWifiMB": 0.6,
    "networkMobileMB": 0.6, "storageUsedGB": 0.4,
    "totalAppsCount": 0.8, "upiTransactionsToday": 1.1, "appUninstallsToday": 0.9,
    "appInstallsToday": 0.8,
    "calendarEventsToday": 0.9, "mediaCountToday": 0.7, "downloadsToday": 0.6,
    "backgroundAudioHours": 1.1,
}

ALL_L1_FEATURES = list(FEATURE_WEIGHTS.keys())

L1_CLUSTER_FEATURES = [
    "sleepDurationHours", "wakeTimeHour", "sleepTimeHour",
    "dailyDisplacementKm", "locationEntropy", "placesVisited",
    "callsPerDay", "callDurationMinutes", "screenTimeHours",
    "unlockCount", "socialAppRatio", "darkDurationHours",
]


# ── DBSCAN implementation ─────────────────────────────────────────────────────

def _dbscan(data: np.ndarray, eps: float = 2.0, min_samples: int = 3):
    """Simple DBSCAN clustering. Returns list of (cluster_id, indices)."""
    n = len(data)
    if n == 0:
        return []

    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(data[i] - data[j])
            dists[i][j] = d
            dists[j][i] = d

    # Find neighbors
    labels = [-1] * n  # -1 = noise
    cluster_id = 0

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = [j for j in range(n) if dists[i][j] <= eps]

        if len(neighbors) < min_samples:
            labels[i] = -1  # noise
        else:
            # Expand cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)
            seed_set = [s for s in seed_set if s != i]
            while seed_set:
                q = seed_set.pop(0)
                if labels[q] == -1:
                    labels[q] = cluster_id
                if visited[q]:
                    continue
                visited[q] = True
                labels[q] = cluster_id
                q_neighbors = [j for j in range(n) if dists[q][j] <= eps]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
            cluster_id += 1

    # Group by cluster
    clusters = {}
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            clusters.setdefault(lbl, []).append(i)

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
    """Build device-level behavioral DNA."""
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

    # Notification rates
    notif_counts = [d.get("notificationsToday", 0.0) for d in daily_features_list if "notificationsToday" in d]
    notif_rate = float(np.mean(notif_counts)) if notif_counts else 0.0

    # Session duration distribution (5-bin histogram)
    import datetime
    durations = []
    for s in sessions:
        dur = (s.get("close_timestamp", 0) - s.get("open_timestamp", 0)) / 60000.0
        durations.append(max(0.0, dur))

    if durations:
        bins = [0, 1, 5, 15, 60, float('inf')]
        hist = [0] * 5
        for d in durations:
            for i in range(5):
                if bins[i] <= d < bins[i + 1]:
                    hist[i] += 1
                    break
        total = sum(hist)
        session_dist = [round(h / max(total, 1), 4) for h in hist]
    else:
        session_dist = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Deep/micro session ratios
    if durations:
        deep = sum(1 for d in durations if d >= 15) / len(durations)
        micro = sum(1 for d in durations if d < 1) / len(durations)
    else:
        deep, micro = 0.0, 0.0

    # App cooccurrence (simplified: top 10 apps × top 10 apps)
    pkg_set = sorted(set(s.get("app_package", "") for s in sessions))[:10]
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
        "pickup_burst_rate": 0.0,
        "inter_pickup_interval_mean": 0.0,
        "session_duration_distribution": session_dist,
        "deep_session_ratio": round(deep, 4),
        "micro_session_ratio": round(micro, 4),
        "notification_open_rate": 0.0,
        "notification_dismiss_rate": 0.0,
        "notification_ignore_rate": 0.0,
        "daily_rhythm_regularity": rhythm_regularity,
        "weekday_weekend_delta": wd_we_delta,
        "app_cooccurrence_labels": pkg_set,
        "app_cooccurrence_matrix": cooccurrence,
    }


def build_anchor_clusters(daily_features_list: list) -> list:
    """Build L1 anchor clusters using DBSCAN on 12 cluster features."""
    if len(daily_features_list) < 5:
        return []

    # Extract L1 cluster feature vectors
    vectors = []
    dates = []
    for d in daily_features_list:
        vec = []
        valid = True
        for feat in L1_CLUSTER_FEATURES:
            if feat in d:
                vec.append(float(d[feat]))
            else:
                valid = False
                break
        if valid:
            vectors.append(vec)
            dates.append(d.get("date", ""))

    if len(vectors) < 5:
        return []

    matrix = np.array(vectors)

    # Normalize
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    matrix_norm = (matrix - means) / stds_safe

    # DBSCAN
    clusters = _dbscan(matrix_norm, eps=2.0, min_samples=3)

    result = []
    for cid, indices in clusters:
        members = matrix_norm[indices]
        centroid = np.mean(members, axis=0)

        # Radius = max distance from centroid
        dists = np.linalg.norm(members - centroid, axis=1)
        radius = float(np.max(dists)) if len(dists) > 0 else 0.0

        # Denormalize centroid to original feature space
        centroid_raw = centroid * stds_safe + means

        result.append({
            "cluster_id": cid,
            "centroid_features": {feat: round(float(centroid_raw[i]), 4) for i, feat in enumerate(L1_CLUSTER_FEATURES)},
            "centroid_normalized": [round(float(c), 4) for c in centroid],
            "radius": round(radius, 4),
            "member_count": len(indices),
            "member_dates": [dates[i] for i in indices if i < len(dates)],
        })

    return result


def build_texture_profiles(daily_features_list: list, sessions: list) -> list:
    """Build simplified L2 texture profiles per archetype."""
    if not daily_features_list or not sessions:
        return []

    import datetime

    # Group sessions by date
    sessions_by_date = {}
    for s in sessions:
        dt = datetime.datetime.fromtimestamp(s.get("open_timestamp", 0) / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        sessions_by_date.setdefault(date_str, []).append(s)

    # Compute daily texture vectors (simplified 22-feature)
    texture_data = []
    for d in daily_features_list:
        date_str = d.get("date", "")
        day_sessions = sessions_by_date.get(date_str, [])

        if not day_sessions:
            continue

        # Compute simplified texture features
        durations = [(s.get("close_timestamp", 0) - s.get("open_timestamp", 0)) / 60000.0
                     for s in day_sessions]

        # Abandon rate
        abandoned = sum(1 for s, dur in zip(day_sessions, durations)
                       if s.get("interaction_count", 1) < 3 and dur < 0.5)
        abandon_rate = abandoned / max(len(day_sessions), 1)

        # Self-open ratio
        self_opens = sum(1 for s in day_sessions if s.get("trigger", "SELF") == "SELF")
        self_open_ratio = self_opens / max(len(day_sessions), 1)

        # Deep/micro session ratios
        deep_ratio = sum(1 for d in durations if d >= 15) / max(len(durations), 1)
        micro_ratio = sum(1 for d in durations if d < 1) / max(len(durations), 1)

        # App switching
        sorted_sess = sorted(day_sessions, key=lambda s: s.get("open_timestamp", 0))
        switches = sum(1 for i in range(1, len(sorted_sess))
                      if sorted_sess[i].get("app_package") != sorted_sess[i-1].get("app_package"))
        switch_rate = switches / max(len(sorted_sess) - 1, 1)

        # Active hours span
        hours = [datetime.datetime.fromtimestamp(s.get("open_timestamp", 0) / 1000.0).hour
                for s in day_sessions]
        active_span = (max(hours) - min(hours)) if hours else 0

        texture_data.append({
            "date": date_str,
            "total_sessions": len(day_sessions),
            "abandon_rate": round(abandon_rate, 4),
            "self_open_ratio": round(self_open_ratio, 4),
            "deep_session_ratio": round(deep_ratio, 4),
            "micro_session_ratio": round(micro_ratio, 4),
            "app_switching_rate": round(switch_rate, 4),
            "active_hours_span": active_span,
            "avg_session_minutes": round(float(np.mean(durations)) if durations else 0, 2),
        })

    if not texture_data:
        return []

    # Compute aggregate texture stats
    n = len(texture_data)
    agg = {
        "total_days_analyzed": n,
        "avg_sessions_per_day": round(float(np.mean([t["total_sessions"] for t in texture_data])), 1),
        "avg_abandon_rate": round(float(np.mean([t["abandon_rate"] for t in texture_data])), 4),
        "avg_self_open_ratio": round(float(np.mean([t["self_open_ratio"] for t in texture_data])), 4),
        "avg_deep_session_ratio": round(float(np.mean([t["deep_session_ratio"] for t in texture_data])), 4),
        "avg_micro_session_ratio": round(float(np.mean([t["micro_session_ratio"] for t in texture_data])), 4),
        "avg_app_switching_rate": round(float(np.mean([t["app_switching_rate"] for t in texture_data])), 4),
        "avg_session_minutes": round(float(np.mean([t["avg_session_minutes"] for t in texture_data])), 2),
        "daily_breakdown": texture_data[-14:],  # Last 14 days
    }

    return [{
        "archetype_id": 0,
        "member_days": n,
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
    personality_vector = build_personality_vector(daily_features_list)
    app_dna_profiles = build_app_dna_profiles(sessions)
    phone_dna = build_phone_dna(daily_features_list, sessions)
    anchor_clusters = build_anchor_clusters(daily_features_list)
    texture_profiles = build_texture_profiles(daily_features_list, sessions)

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

    return {
        "person_id": person_id,
        "profile_version": "1.0",
        "built_at": datetime.now().isoformat(),
        "days_of_data": len(daily_features_list),
        "personality_vector": personality_vector,
        "app_dna_profiles": app_dna_profiles,
        "phone_dna": phone_dna,
        "anchor_clusters": anchor_clusters,
        "texture_profiles": texture_profiles,
        "feature_importance": feature_importance,
        "group_summaries": group_summaries,
    }


def _get_feature_group(feat: str) -> str:
    """Return the group name for a feature."""
    groups = {
        "screen_app": ["screenTimeHours", "unlockCount", "appLaunchCount",
                       "notificationsToday", "socialAppRatio"],
        "communication": ["callsPerDay", "callDurationMinutes", "uniqueContacts",
                          "conversationFrequency"],
        "location": ["dailyDisplacementKm", "locationEntropy", "homeTimeRatio",
                     "placesVisited"],
        "sleep": ["wakeTimeHour", "sleepTimeHour", "sleepDurationHours",
                  "darkDurationHours"],
        "system": ["chargeDurationHours", "memoryUsagePercent", "networkWifiMB",
                   "networkMobileMB", "storageUsedGB"],
        "behavioral": ["totalAppsCount", "upiTransactionsToday", "appUninstallsToday",
                       "appInstallsToday"],
        "engagement": ["calendarEventsToday", "mediaCountToday", "downloadsToday",
                       "backgroundAudioHours"],
    }
    for grp, feats in groups.items():
        if feat in feats:
            return grp
    return "unknown"

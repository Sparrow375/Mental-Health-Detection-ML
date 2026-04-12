"""
Baseline Builder — orchestrates all baseline construction steps.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data_structures import (
    PersonProfile, PersonalityVector, SessionEvent, NotificationEvent,
    ConfidenceTier,
)
from ..feature_meta import ALL_L1_FEATURES, L1_CLUSTER_FEATURES
from .app_dna_builder import build_all_app_dnas
from .phone_dna_builder import build_phone_dna
from .l1_clusterer import cluster_baseline_days
from .l2_texture_builder import build_texture_profiles, compute_daily_texture_vector
from .detector_calibration import calibrate_detector


def build_personality_vector(
    daily_features: List[Dict[str, float]],
) -> PersonalityVector:
    """Build PersonalityVector (means + std-devs) from baseline daily features."""
    if not daily_features:
        return PersonalityVector()

    means = {}
    variances = {}
    for feat in ALL_L1_FEATURES:
        values = [d.get(feat, 0.0) for d in daily_features if feat in d]
        if values:
            means[feat] = float(np.mean(values))
            variances[feat] = float(np.std(values)) if len(values) > 1 else 1.0
        else:
            means[feat] = 0.0
            variances[feat] = 1.0

    return PersonalityVector(means=means, variances=variances)


def build_baseline(
    patient_id: str,
    daily_features: List[Dict[str, float]],
    dates: List[str],
    sessions: List[SessionEvent],
    notifications: List[NotificationEvent],
    confidence_tier: ConfidenceTier = ConfidenceTier.LOW,
) -> PersonProfile:
    """
    Full baseline construction pipeline.

    Steps:
    1. PersonalityVector (29-feature means + std-devs)
    2. Per-app AppDNA
    3. PhoneDNA
    4. L1 DBSCAN clustering
    5. L2 texture profiles per archetype
    6. Detector calibration
    """
    profile = PersonProfile(patient_id=patient_id)

    # Step 1.1: PersonalityVector
    profile.personality_vector = build_personality_vector(daily_features)
    profile.personality_vector.confidence = confidence_tier
    profile.personality_vector.built_date = dates[-1] if dates else None

    # Step 1.2: Per-app AppDNA
    profile.app_dnas = build_all_app_dnas(sessions, dates)

    # Step 1.3: PhoneDNA
    profile.phone_dna = build_phone_dna(sessions, notifications, dates)

    # Step 1.4: L1 context clustering
    clusters, cov_inv, mins, maxs, outliers = cluster_baseline_days(daily_features, dates)
    profile.anchor_clusters = clusters
    profile.l1_cov_inv = cov_inv
    profile.l1_feature_mins = mins
    profile.l1_feature_maxs = maxs
    profile.outlier_vectors = outliers

    # Step 1.5: L2 texture profiles
    # Build archetype assignments from cluster member dates
    date_to_idx = {d: i for i, d in enumerate(dates)}
    archetype_assignments: Dict[int, List[int]] = {}
    for cluster in clusters:
        archetype_assignments[cluster.cluster_id] = []
        for d in cluster.member_dates:
            if d in date_to_idx:
                archetype_assignments[cluster.cluster_id].append(date_to_idx[d])

    # Compute texture vectors for each baseline day
    daily_texture_vectors = []
    for i, (date_str, day_feats) in enumerate(zip(dates, daily_features)):
        # Filter sessions for this date
        day_sessions = []
        for sess in sessions:
            dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
            if dt.strftime("%Y-%m-%d") == date_str:
                day_sessions.append(sess)
        day_notifs = []
        for n in notifications:
            dt = datetime.datetime.fromtimestamp(n.arrival_ts / 1000.0)
            if dt.strftime("%Y-%m-%d") == date_str:
                day_notifs.append(n)

        texture_vec = compute_daily_texture_vector(
            day_sessions, profile.app_dnas, profile.phone_dna,
            day_notifs, len(dates),
        )
        daily_texture_vectors.append(texture_vec)

    profile.texture_profiles = build_texture_profiles(archetype_assignments, daily_texture_vectors)

    # Step 1.6: Detector calibration (imported here to avoid circular dependency)
    from ..scoring.l1_scorer import score_l1_day
    profile = calibrate_detector(profile, daily_features)

    return profile
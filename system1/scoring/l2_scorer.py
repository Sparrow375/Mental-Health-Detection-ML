"""
L2 Scorer — coherence, rhythm dissolution, session incoherence, modifier.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as kl_divergence

from ..data_structures import (
    PersonProfile, L2ScoreResult, SessionEvent, NotificationEvent,
    AppDNA, PhoneDNA, AnchorCluster, ContextualTextureProfile,
)
from ..feature_meta import L1_CLUSTER_FEATURES, THRESHOLDS


def _compute_context_coherence(
    today_l1_vector: np.ndarray,
    anchor_clusters: List[AnchorCluster],
    cov_inv: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> Tuple[float, int]:
    """
    Compute context coherence as Mahalanobis distance to nearest anchor cluster.
    Returns (coherence, matched_context_id).
    """
    if not anchor_clusters:
        return 0.0, -1

    # Normalize today's vector using baseline min/max
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (today_l1_vector - mins) / ranges

    best_coherence = 0.0
    best_id = -1
    radius_factor = THRESHOLDS["L2_COHERENCE_MATCH_RADIUS_FACTOR"]

    for cluster in anchor_clusters:
        if cluster.centroid is None:
            continue
        diff = normalized - cluster.centroid
        try:
            dist = np.sqrt(diff @ cov_inv @ diff)
        except Exception:
            dist = np.linalg.norm(diff)

        radius = cluster.radius if cluster.radius > 0 else 1.0
        threshold = radius * radius_factor
        coherence = max(0.0, 1.0 - (dist / threshold))

        if coherence > best_coherence:
            best_coherence = coherence
            best_id = cluster.cluster_id

    return best_coherence, best_id


def _compute_rhythm_dissolution(
    sessions_today: List[SessionEvent],
    app_dnas: Dict[str, AppDNA],
    matched_context_id: int,
) -> float:
    """
    Compute rhythm dissolution via KL divergence between today's and baseline
    hourly usage distributions.
    """
    if not sessions_today or not app_dnas:
        return 0.0

    import datetime
    kl_scores = []
    weights = []

    # Group sessions by app
    app_sessions: Dict[str, List[SessionEvent]] = defaultdict(list)
    for sess in sessions_today:
        app_sessions[sess.app_id].append(sess)

    for app_id, sessions in app_sessions.items():
        dna = app_dnas.get(app_id)
        if dna is None or dna.usage_heatmap is None:
            continue

        # Build today's hourly distribution for this app
        today_dist = np.zeros(24)
        for sess in sessions:
            dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
            today_dist[dt.hour] += sess.duration_minutes

        # Normalize
        total_today = today_dist.sum()
        if total_today == 0:
            continue
        today_dist /= total_today

        # Get baseline distribution (average across all days of week)
        baseline_dist = dna.usage_heatmap.mean(axis=0)
        total_baseline = baseline_dist.sum()
        if total_baseline == 0:
            continue
        baseline_dist /= total_baseline

        # KL divergence with epsilon smoothing
        eps = 1e-9
        kl = kl_divergence(today_dist + eps, baseline_dist + eps)
        kl_scores.append(kl)

        # Weight by historical importance
        weight = dna.avg_session_minutes * max(dna.weekday_sessions_per_day, dna.weekend_sessions_per_day)
        weights.append(max(weight, 0.1))

    if not kl_scores:
        return 0.0

    weights_arr = np.array(weights)
    weights_arr /= weights_arr.sum()
    weighted_kl = float(np.average(kl_scores, weights=weights_arr))

    # Normalize: clip to [0, 1] using / 3.0
    return min(weighted_kl / 3.0, 1.0)


def _compute_session_incoherence(
    sessions_today: List[SessionEvent],
    app_dnas: Dict[str, AppDNA],
    notifications_today: Optional[List[NotificationEvent]] = None,
) -> float:
    """
    Compute session incoherence from three sub-signals:
    abandon spike, duration collapse, trigger shift.
    """
    if not sessions_today:
        return 0.0

    app_sessions: Dict[str, List[SessionEvent]] = defaultdict(list)
    for sess in sessions_today:
        app_sessions[sess.app_id].append(sess)

    abandon_deltas = []
    duration_collapses = []
    trigger_drops = []

    for app_id, sessions in app_sessions.items():
        dna = app_dnas.get(app_id)

        # Abandon spike
        today_abandon = sum(1 for s in sessions if s.duration_minutes < 0.75 and s.interaction_count < 5)
        today_abandon_rate = today_abandon / max(1, len(sessions))
        baseline_abandon = dna.abandon_rate if dna else 0.0
        abandon_deltas.append(max(0, today_abandon_rate - baseline_abandon))

        # Duration collapse
        if dna and dna.avg_session_minutes > 5:
            today_avg = np.mean([s.duration_minutes for s in sessions])
            ratio = today_avg / dna.avg_session_minutes
            duration_collapses.append(max(0, 1.0 - ratio))

        # Trigger shift
        if dna:
            today_self = sum(1 for s in sessions if s.trigger == "SELF") / max(1, len(sessions))
            trigger_drops.append(max(0, dna.self_open_ratio - today_self))

    mean_abandon = np.mean(abandon_deltas) if abandon_deltas else 0.0
    mean_duration = np.mean(duration_collapses) if duration_collapses else 0.0
    mean_trigger = np.mean(trigger_drops) if trigger_drops else 0.0

    return float(np.mean([mean_abandon, mean_duration, mean_trigger]))


def score_l2_day(
    today_features: Dict[str, float],
    sessions_today: List[SessionEvent],
    notifications_today: List[NotificationEvent],
    profile: PersonProfile,
) -> L2ScoreResult:
    """
    Compute L2 modifier for one monitoring day.
    """
    result = L2ScoreResult()

    # Build today's L1 daily vector for context matching
    today_l1_vec = np.array([today_features.get(f, 0.0) for f in L1_CLUSTER_FEATURES])

    # Step 3.1: Context coherence
    if (profile.l1_cov_inv is not None and
        profile.l1_feature_mins is not None and
        profile.l1_feature_maxs is not None and
        profile.anchor_clusters):

        result.coherence, result.matched_context_id = _compute_context_coherence(
            today_l1_vec,
            profile.anchor_clusters,
            profile.l1_cov_inv,
            profile.l1_feature_mins,
            profile.l1_feature_maxs,
        )
    else:
        result.coherence = 0.0
        result.matched_context_id = -1

    # Step 3.2: Rhythm dissolution
    result.rhythm_dissolution = _compute_rhythm_dissolution(
        sessions_today, profile.app_dnas, result.matched_context_id
    )

    # Step 3.3: Session incoherence
    result.session_incoherence = _compute_session_incoherence(
        sessions_today, profile.app_dnas, notifications_today
    )

    # Step 3.4: Candidate new pattern check
    if result.coherence < 0.25 and result.session_incoherence < 0.3:
        result.candidate_flag = True
    else:
        result.candidate_flag = False

    # Step 3.5: L2 modifier computation
    suppression = result.coherence * 0.85
    amplification = (result.rhythm_dissolution * 0.6 + result.session_incoherence * 0.4) * 1.5
    modifier = 1.0 - suppression + amplification
    result.modifier = float(np.clip(modifier, 0.15, 2.0))

    return result
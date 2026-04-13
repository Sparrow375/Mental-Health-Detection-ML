"""
Level 2 Behavioral DNA Engine — Computation functions.

Provides:
  - compute_context_coherence: how well today fits known clusters
  - compute_rhythm_integrity: KL-divergence of hourly usage patterns
  - compute_session_incoherence: abandon/duration/trigger anomaly scores
  - compute_texture_quality: overall behavioral texture health
  - compute_l2_modifier: multiplier for System 1 evidence scores
  - update_rolling_clusters: candidate cluster discovery/promotion/rejection
"""

import numpy as np
from typing import Tuple, List, Optional

from dna import PersonDNA, AppDNA, CandidateCluster, PromotedCluster

EPS = 1e-9


def compute_context_coherence(
    today_vector: np.ndarray, dna: PersonDNA
) -> Tuple[float, int]:
    """
    Check how well today's behavior fits known clusters.
    
    Checks anchor clusters first, then promoted clusters.
    Returns (coherence_score 0-1, matched_cluster_id or -1).
    coherence = max(0, 1 - nearest_dist / (nearest_radius * 1.5))
    matched = cluster_id if within 1.5x radius else -1
    """
    best_coherence = 0.0
    matched_id = -1

    # Check anchor clusters
    if dna.anchor_centroids is not None and len(dna.anchor_centroids) > 0:
        for i, centroid in enumerate(dna.anchor_centroids):
            if len(centroid) != len(today_vector):
                continue
            dist = float(np.linalg.norm(today_vector - centroid))
            radius = float(dna.anchor_radii[i]) if i < len(dna.anchor_radii) else 1.0
            radius_safe = max(radius, EPS)
            coherence = max(0.0, 1.0 - dist / (radius_safe * 1.5))

            if coherence > best_coherence:
                best_coherence = coherence
                matched_id = i if dist <= radius_safe * 1.5 else -1

    # Check promoted clusters (IDs start after anchor clusters)
    for j, promoted in enumerate(dna.promoted_clusters):
        if len(promoted.centroid) != len(today_vector):
            continue
        dist = float(np.linalg.norm(today_vector - promoted.centroid))
        radius_safe = max(promoted.radius, EPS)
        coherence = max(0.0, 1.0 - dist / (radius_safe * 1.5))

        if coherence > best_coherence:
            best_coherence = coherence
            matched_id = len(dna.anchor_centroids) + j if dist <= radius_safe * 1.5 else -1

    return best_coherence, matched_id


def compute_rhythm_integrity(
    sessions_today: List[dict], dna: PersonDNA
) -> float:
    """
    Compute rhythm dissolution score via KL divergence of hourly patterns.
    
    For each app active today that exists in baseline:
      - Build 24-bin hourly distribution from today's sessions
      - Get baseline heatmap row for today's day_of_week
      - Add epsilon 1e-9 to both, renormalize
      - Compute KL divergence = scipy.stats.entropy(today_dist, baseline_dist)
    Weighted average of KL scores (weight = app's share of baseline screen time)
    Return clipped to [0,1]: weighted_kl / 3.0
    
    Returns 0.0 if no sessions or no matching apps (perfect rhythm = no dissolution).
    """
    import datetime

    if not sessions_today or not dna.app_profiles:
        return 0.0

    # Determine today's day of week
    first_ts = sessions_today[0]["open_timestamp"]
    today_dow = datetime.datetime.fromtimestamp(first_ts / 1000.0).weekday()

    # Group today's sessions by app
    app_sessions_today = {}
    for s in sessions_today:
        pkg = s["app_package"]
        if pkg not in app_sessions_today:
            app_sessions_today[pkg] = []
        app_sessions_today[pkg].append(s)

    kl_scores = []
    weights = []

    # Compute baseline screen time for weighting
    baseline_total_minutes = 0.0
    for pkg, profile in dna.app_profiles.items():
        baseline_total_minutes += profile.usage_heatmap.sum()

    for pkg, today_sess in app_sessions_today.items():
        if pkg not in dna.app_profiles:
            continue

        profile = dna.app_profiles[pkg]

        # Build today's 24-bin hourly distribution
        hourly_minutes = np.zeros(24)
        for s in today_sess:
            open_dt = datetime.datetime.fromtimestamp(s["open_timestamp"] / 1000.0)
            dur_min = (s["close_timestamp"] - s["open_timestamp"]) / 60_000.0
            hourly_minutes[open_dt.hour] += max(0.0, dur_min)

        # Get baseline heatmap row for this day of week
        baseline_row = profile.usage_heatmap[today_dow].copy()

        # Add epsilon and renormalize
        today_dist = hourly_minutes + EPS
        baseline_dist = baseline_row + EPS
        today_dist = today_dist / today_dist.sum()
        baseline_dist = baseline_dist / baseline_dist.sum()

        # KL divergence (manual implementation: sum(p * log(p/q)))
        kl = float(np.sum(today_dist * np.log(today_dist / baseline_dist)))
        kl_scores.append(kl)

        # Weight: app's share of baseline screen time
        app_baseline_minutes = profile.usage_heatmap.sum()
        weight = app_baseline_minutes / max(baseline_total_minutes, EPS)
        weights.append(weight)

    if not kl_scores:
        return 0.0

    weights_arr = np.array(weights)
    weights_arr = weights_arr / max(weights_arr.sum(), EPS)

    weighted_kl = float(np.average(kl_scores, weights=weights_arr))
    # Clip to [0, 1]
    return min(max(weighted_kl / 3.0, 0.0), 1.0)


def compute_session_incoherence(
    sessions_today: List[dict], dna: PersonDNA
) -> float:
    """
    Three sub-scores, averaged:
      1. abandon_spike: mean(today_abandon_rate - baseline_abandon_rate) per app, floor 0
      2. duration_collapse: for apps with baseline avg_session > 5min only,
         mean(1 - today_avg_duration / baseline_avg_duration), floor 0
      3. trigger_drop: mean(baseline_self_ratio - today_self_ratio) per app, floor 0
    Return mean of three sub-scores, clipped [0,1]
    """
    if not sessions_today or not dna.app_profiles:
        return 0.0

    # Group today's sessions by app
    app_sessions_today = {}
    for s in sessions_today:
        pkg = s["app_package"]
        if pkg not in app_sessions_today:
            app_sessions_today[pkg] = []
        app_sessions_today[pkg].append(s)

    abandon_diffs = []
    duration_ratios = []
    trigger_diffs = []

    for pkg, today_sess in app_sessions_today.items():
        if pkg not in dna.app_profiles:
            continue
        profile = dna.app_profiles[pkg]

        # Today's stats
        durations = [(s["close_timestamp"] - s["open_timestamp"]) / 60_000.0 for s in today_sess]
        today_avg_dur = float(np.mean(durations)) if durations else 0.0
        
        abandoned = sum(1 for i, s in enumerate(today_sess)
                        if s.get("interaction_count", 1) < 3 and durations[i] < 0.5)
        today_abandon = abandoned / max(len(today_sess), 1)

        self_count = sum(1 for s in today_sess if s.get("trigger", "SELF") == "SELF")
        today_self_ratio = self_count / max(len(today_sess), 1)

        # 1. Abandon spike
        abandon_diffs.append(max(0.0, today_abandon - profile.abandon_rate))

        # 2. Duration collapse (only for apps with baseline > 5min)
        if profile.avg_session_minutes > 5.0:
            ratio = today_avg_dur / max(profile.avg_session_minutes, EPS)
            collapse = max(0.0, 1.0 - ratio)
            duration_ratios.append(collapse)

        # 3. Trigger drop
        trigger_diffs.append(max(0.0, profile.self_open_ratio - today_self_ratio))

    # Average sub-scores
    abandon_score = float(np.mean(abandon_diffs)) if abandon_diffs else 0.0
    duration_score = float(np.mean(duration_ratios)) if duration_ratios else 0.0
    trigger_score = float(np.mean(trigger_diffs)) if trigger_diffs else 0.0

    result = (abandon_score + duration_score + trigger_score) / 3.0
    return min(max(result, 0.0), 1.0)


def compute_texture_quality(
    coherence: float, rhythm_dissolution: float, session_incoherence: float
) -> float:
    """
    Quality = how healthy the behavioral texture is RIGHT NOW.
    quality = coherence * 0.4 + (1-rhythm_dissolution) * 0.3 + (1-session_incoherence) * 0.3
    Returns 0-1. Used to evaluate candidate clusters.
    """
    quality = (
        coherence * 0.4
        + (1.0 - rhythm_dissolution) * 0.3
        + (1.0 - session_incoherence) * 0.3
    )
    return min(max(quality, 0.0), 1.0)


def compute_l2_modifier(
    coherence: float, rhythm_dissolution: float, session_incoherence: float
) -> float:
    """
    suppression = coherence * 0.85
    amplification = (rhythm_dissolution * 0.6 + session_incoherence * 0.4) * 1.5
    modifier = 1.0 - suppression + amplification
    Return clipped [0.15, 2.0]
    """
    suppression = coherence * 0.85
    amplification = (rhythm_dissolution * 0.6 + session_incoherence * 0.4) * 1.5
    modifier = 1.0 - suppression + amplification
    return min(max(modifier, 0.15), 2.0)


# ============================================================================
# ROLLING CLUSTER DISCOVERY
# ============================================================================

def update_rolling_clusters(
    today_vector: np.ndarray,
    texture_quality: float,
    coherence: float,
    dna: PersonDNA,
    held_evidence_today: float,
) -> Tuple[PersonDNA, str]:
    """
    Rolling cluster discovery for recognizing new behavioral patterns.
    
    Only runs when coherence < 0.3 (today is outside all known clusters).
    
    Returns updated dna and action:
        "none" | "candidate_opened" | "promoted" | "rejected"
    
    CANDIDATE OPENING:
    - If no existing candidate: open new CandidateCluster
    - If existing candidate: check similarity to today_vector
      if distance < candidate_radius_estimate: update centroid (running mean)
      else: open second candidate (max 2 candidates at once)
    
    PROMOTION CHECK (run if candidate.days_observed >= 5):
    - mean_texture = mean(candidate.texture_scores)
    - If mean_texture >= 0.5: PROMOTE
    - If mean_texture < 0.5: REJECT
    
    CANDIDATE EXPIRY:
    - If candidate.days_observed >= 14 and not yet decided: force rejection
    """
    # Only run when coherence is low (unrecognized pattern)
    if coherence >= 0.3:
        return dna, "none"

    # Ensure vector is numpy
    today_vector = np.array(today_vector, dtype=np.float64)

    # Estimate candidate radius from anchor radii
    candidate_radius = 1.0  # default
    if dna.anchor_radii is not None and len(dna.anchor_radii) > 0:
        candidate_radius = float(np.mean(dna.anchor_radii)) * 1.5

    action = "none"

    # Clean up: force reject candidates that have been evaluating too long
    updated_candidates = []
    for cand in dna.candidate_clusters:
        if cand.status == "evaluating" and cand.days_observed >= 14:
            # Force rejection — release held evidence
            cand.status = "rejected"
            action = "rejected"
        updated_candidates.append(cand)
    dna.candidate_clusters = updated_candidates

    # Remove rejected candidates from the list
    active_candidates = [c for c in dna.candidate_clusters if c.status == "evaluating"]

    # PROMOTION / REJECTION CHECK for candidates with enough observations
    for i, cand in enumerate(active_candidates):
        if cand.days_observed >= 5:
            mean_texture = float(np.mean(cand.texture_scores))
            if mean_texture >= 0.5:
                # PROMOTE: add to promoted_clusters
                promoted = PromotedCluster(
                    centroid=cand.centroid,
                    radius=_compute_promoted_radius(cand.centroid, dna),
                    texture_quality_mean=mean_texture,
                )
                dna.promoted_clusters.append(promoted)
                cand.status = "promoted"
                # Clear held evidence (retroactively — evidence was noise, discard it)
                cand.held_evidence = 0.0
                action = "promoted"
            else:
                # REJECT: release held evidence
                cand.status = "rejected"
                action = "rejected"

    # Clean up non-evaluating candidates
    dna.candidate_clusters = [c for c in dna.candidate_clusters if c.status == "evaluating"]

    # Refresh active list
    active_candidates = [c for c in dna.candidate_clusters if c.status == "evaluating"]

    # If no action taken yet, try to open or update candidates
    if action == "none":
        matched_candidate = None
        for cand in active_candidates:
            if len(cand.centroid) == len(today_vector):
                dist = float(np.linalg.norm(today_vector - cand.centroid))
                if dist < candidate_radius:
                    matched_candidate = cand
                    break

        if matched_candidate is not None:
            # Update existing candidate (running mean centroid)
            n = matched_candidate.days_observed
            matched_candidate.centroid = (
                matched_candidate.centroid * n + today_vector
            ) / (n + 1)
            matched_candidate.days_observed += 1
            matched_candidate.texture_scores.append(texture_quality)
            matched_candidate.held_evidence += held_evidence_today
            action = "none"
        elif len(active_candidates) < 2:
            # Open new candidate
            new_candidate = CandidateCluster(
                centroid=today_vector.copy(),
                days_observed=1,
                texture_scores=[texture_quality],
                held_evidence=held_evidence_today,
                status="evaluating",
            )
            dna.candidate_clusters.append(new_candidate)
            action = "candidate_opened"

    return dna, action


def _compute_promoted_radius(centroid: np.ndarray, dna: PersonDNA) -> float:
    """Compute radius for a promoted cluster based on anchor cluster statistics."""
    if dna.anchor_radii is not None and len(dna.anchor_radii) > 0:
        return float(np.mean(dna.anchor_radii))
    return 1.0


def get_released_evidence(dna: PersonDNA) -> float:
    """
    Get total held evidence from rejected candidates.
    Called after update_rolling_clusters to find evidence to release.
    """
    total = 0.0
    for cand in dna.candidate_clusters:
        if cand.status == "rejected":
            total += cand.held_evidence
    return total


def clear_rejected_candidates(dna: PersonDNA) -> PersonDNA:
    """Remove rejected candidates after their evidence has been released."""
    dna.candidate_clusters = [
        c for c in dna.candidate_clusters if c.status != "rejected"
    ]
    return dna

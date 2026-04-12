"""
Candidate Cluster Evaluation — 7-day window for new behavioral archetypes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..data_structures import (
    CandidateState, EvidenceState, PersonProfile, AnchorCluster, L2ScoreResult,
)
from ..feature_meta import THRESHOLDS


def evaluate_candidate(
    candidate_state: CandidateState,
    l2_result: L2ScoreResult,
    today_l1_vector: Dict[str, float],
    evidence_state: EvidenceState,
    day_in_window: int,
) -> tuple:
    """
    Evaluate candidate cluster window. Returns (candidate_state, evidence_state, action).

    Actions:
        - "hold": still evaluating, pause evidence
        - "promote": new cluster confirmed, clear held evidence
        - "reject_clinical": texture degraded, release held evidence at full weight
    """
    window_length = THRESHOLDS["CANDIDATE_WINDOW_LENGTH"]
    texture_threshold = THRESHOLDS["L2_CANDIDATE_TEXTURE_THRESHOLD"]

    # Buffer today's data
    candidate_state.buffered_l1_vectors.append(today_l1_vector)
    candidate_state.daily_session_incoherence.append(l2_result.session_incoherence)
    candidate_state.daily_texture_quality.append(1.0 - l2_result.session_incoherence)

    # Days 1-3: Just observe
    if day_in_window <= 3:
        candidate_state.status = "EVALUATING"
        return candidate_state, evidence_state, "hold"

    # Days 4-7: Evaluate texture quality
    incoherences = candidate_state.daily_session_incoherence

    # Check for monotonic degradation (tiebreaker rule)
    monotonic_degradation = all(
        incoherences[i] >= incoherences[i - 1]
        for i in range(1, len(incoherences))
    )

    # Majority of days with high incoherence
    high_incoherence_days = sum(1 for si in incoherences if si > texture_threshold)
    majority_degraded = high_incoherence_days > len(incoherences) / 2

    if monotonic_degradation or majority_degraded:
        # Clinical onset — retroactively release all held evidence
        candidate_state.status = "REJECTED_CLINICAL"
        # Release held evidence at full weight
        for i, l1_vec in enumerate(candidate_state.buffered_l1_vectors):
            si = incoherences[i] if i < len(incoherences) else 0.5
            # Use a high modifier for clinical evidence release
            released_score = 0.5 * 2.0  # conservative effective score
            evidence_state.sustained_deviation_days += 1
            multiplier = 1.0 + evidence_state.sustained_deviation_days * THRESHOLDS["EVIDENCE_COMPOUND_FACTOR"]
            evidence_state.evidence_accumulated += released_score * multiplier
        return candidate_state, evidence_state, "reject_clinical"

    # If we've reached the end of the window and texture is still healthy
    if day_in_window >= window_length:
        # Promote to real cluster
        candidate_state.status = "PROMOTED"
        return candidate_state, evidence_state, "promote"

    # Still evaluating
    candidate_state.status = "EVALUATING"
    return candidate_state, evidence_state, "hold"


def open_candidate_window(date_str: str) -> CandidateState:
    """Open a new candidate evaluation window."""
    return CandidateState(
        status="EVALUATING",
        open_timestamp=date_str,
    )


def promote_to_anchor_cluster(
    candidate_state: CandidateState,
    profile: PersonProfile,
) -> PersonProfile:
    """Add candidate vectors as a new anchor cluster to the person profile."""
    if not candidate_state.buffered_l1_vectors:
        return profile

    from ..feature_meta import L1_CLUSTER_FEATURES
    vectors = []
    for vec_dict in candidate_state.buffered_l1_vectors:
        vec = [vec_dict.get(f, 0.0) for f in L1_CLUSTER_FEATURES]
        vectors.append(vec)

    vectors_arr = np.array(vectors)
    centroid = np.mean(vectors_arr, axis=0)

    # Compute radius
    if len(vectors_arr) > 1:
        dists = np.linalg.norm(vectors_arr - centroid, axis=1)
        radius = float(np.max(dists))
    else:
        radius = 0.5  # Default radius for single-vector cluster

    new_id = max((c.cluster_id for c in profile.anchor_clusters), default=-1) + 1
    new_cluster = AnchorCluster(
        cluster_id=new_id,
        centroid=centroid,
        radius=radius,
        member_count=len(vectors),
        member_dates=[candidate_state.open_timestamp] * len(vectors),
    )
    profile.anchor_clusters.append(new_cluster)
    return profile
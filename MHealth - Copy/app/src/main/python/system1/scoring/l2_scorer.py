"""
L2 Scorer: context coherence, rhythm dissolution, session incoherence,
and the combined L2 modifier that scales the L1 score.

The L2 modifier suppresses L1 evidence when the day matches a known healthy
context (high coherence, healthy texture) and amplifies it when the day is
unfamiliar with degraded texture (strongest clinical signal).

Modifier range:
    0.15 – 0.5   Strong suppress  (known context, healthy texture)
    0.5  – 0.9   Moderate suppress
    0.9  – 1.1   Neutral — mixed signals, L1 at face value
    1.1  – 1.5   Moderate amplify  (unfamiliar or dissolving rhythm)
    1.5  – 2.0   Strong amplify   (unknown + degraded — clinical)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from system1.data_structures import (
    AppDNA,
    PhoneDNA,
    ContextualTextureProfile,
    L1ClusterState,
)
from system1.feature_meta import L1_CLUSTERING_FEATURES, DEFAULT_THRESHOLDS

try:
    from scipy.stats import entropy as kl_divergence
    from scipy.spatial.distance import mahalanobis
except ImportError:
    kl_divergence = None
    mahalanobis = None


class L2Scorer:
    """
    Computes the L2 modifier for a single monitoring day.

    Requires:
        - L1 DBSCAN cluster state (centroids, radii, inverse covariance)
        - AppDNA dict  (may be empty if no session data)
        - PhoneDNA      (may be None if no session data)
        - L2 texture profiles per archetype (may be empty)
    """

    def __init__(
        self,
        cluster_state: Optional[L1ClusterState] = None,
        app_dna_dict: Optional[Dict[str, AppDNA]] = None,
        phone_dna: Optional[PhoneDNA] = None,
        texture_profiles: Optional[Dict[int, ContextualTextureProfile]] = None,
    ):
        self.cluster_state = cluster_state
        self.app_dna_dict = app_dna_dict or {}
        self.phone_dna = phone_dna
        self.texture_profiles = texture_profiles or {}
        self.radius_factor = DEFAULT_THRESHOLDS['COHERENCE_MATCH_RADIUS_FACTOR']

    # ------------------------------------------------------------------
    # Step 3.1 — Context coherence
    # ------------------------------------------------------------------

    def compute_coherence(
        self,
        today_l1_vector: Dict[str, float],
        baseline_dict: Dict[str, float],
        baseline_variances: Dict[str, float],
    ) -> Tuple[float, int]:
        """
        Compute Mahalanobis distance from today's 12-feature L1 vector to all
        anchor cluster centroids.

            coherence = max(0, 1.0 - nearest_distance / (radius * 1.5))

        Returns (coherence, matched_context_id).
            matched_context_id = -1  if no match within 1.5× radius.
        """
        if self.cluster_state is None or self.cluster_state.n_clusters == 0:
            return 0.0, -1

        # Build normalised 12-feature vector for today
        today_vec = self._build_clustering_vector(
            today_l1_vector, baseline_dict, baseline_variances
        )
        if today_vec is None:
            return 0.0, -1

        centroids = self.cluster_state.centroids
        radii = self.cluster_state.radii
        cov_inv = self.cluster_state.covariance_inv

        best_dist = float('inf')
        best_id = -1

        for idx in range(self.cluster_state.n_clusters):
            centroid = centroids[idx]
            if cov_inv is not None and mahalanobis is not None:
                try:
                    dist = mahalanobis(today_vec, centroid, cov_inv)
                except Exception:
                    dist = float(np.linalg.norm(today_vec - centroid))
            else:
                dist = float(np.linalg.norm(today_vec - centroid))

            if dist < best_dist:
                best_dist = dist
                best_id = idx

        if best_id >= 0 and radii is not None and best_dist <= radii[best_id] * self.radius_factor:
            coherence = max(0.0, 1.0 - best_dist / (radii[best_id] * self.radius_factor))
            return coherence, best_id
        else:
            # Outside all cluster radii
            return 0.0, -1

    # ------------------------------------------------------------------
    # Step 3.2 — Rhythm dissolution  (KL divergence)
    # ------------------------------------------------------------------

    def compute_rhythm_dissolution(
        self,
        today_session_events: Optional[List[Dict]] = None,
        matched_context_id: int = -1,
    ) -> float:
        """
        For each app active today with a baseline AppDNA:
            Build today's 24-bin hourly usage distribution.
            Compute KL divergence vs baseline heatmap row for today's day-of-week.
            Aggregate weighted by app importance.

        rhythm_dissolution = clip(weighted_mean_kl / 3.0, 0, 1)

        Returns 0.0 if no session data available.
        """
        if not today_session_events or not self.app_dna_dict or kl_divergence is None:
            return 0.0

        import datetime
        eps = 1e-9
        weighted_kl_sum = 0.0
        weight_sum = 0.0

        for app_pkg, dna in self.app_dna_dict.items():
            if dna.usage_heatmap is None:
                continue

            # Collect today's sessions for this app
            app_sessions = [s for s in today_session_events if s.get('app_package') == app_pkg]
            if not app_sessions:
                continue

            # Build 24-bin hourly usage distribution for today
            today_dist = np.zeros(24) + eps
            for sess in app_sessions:
                hour = int(sess.get('hour', 0))
                duration = float(sess.get('duration_minutes', 1.0))
                today_dist[hour % 24] += duration
            today_dist /= today_dist.sum()

            # Baseline distribution for today's day of week
            dow = int(sess.get('day_of_week', 0)) if app_sessions else 0
            baseline_dist = dna.usage_heatmap[dow % 7].copy()
            baseline_dist = np.maximum(baseline_dist, eps)
            baseline_dist /= baseline_dist.sum()

            kl = float(kl_divergence(today_dist, baseline_dist))

            # Weight by app importance
            importance = dna.avg_session_minutes * max(dna.weekday_sessions_per_day, 0.1)
            weighted_kl_sum += kl * importance
            weight_sum += importance

        if weight_sum > 0:
            weighted_mean_kl = weighted_kl_sum / weight_sum
            return float(np.clip(weighted_mean_kl / 3.0, 0.0, 1.0))
        return 0.0

    # ------------------------------------------------------------------
    # Step 3.3 — Session incoherence
    # ------------------------------------------------------------------

    def compute_session_incoherence(
        self,
        today_session_events: Optional[List[Dict]] = None,
    ) -> float:
        """
        Three sub-signals, averaged:
            1. Abandon spike: delta of today's abandon_rate vs baseline
            2. Duration collapse: 1 - (today_avg / baseline_avg) for long-session apps
            3. Trigger shift: baseline_self_ratio - today_self_ratio

        Returns 0.0 if no session data available.
        """
        if not today_session_events or not self.app_dna_dict:
            return 0.0

        abandon_deltas = []
        duration_collapses = []
        trigger_drops = []

        for app_pkg, dna in self.app_dna_dict.items():
            app_sessions = [s for s in today_session_events if s.get('app_package') == app_pkg]
            if not app_sessions:
                continue

            # --- Abandon spike ---
            short_sessions = sum(
                1 for s in app_sessions
                if float(s.get('duration_minutes', 0)) < 0.75  # <45s
                and int(s.get('interaction_count', 0)) < 5
            )
            today_abandon = short_sessions / max(len(app_sessions), 1)
            abandon_deltas.append(max(0.0, today_abandon - dna.abandon_rate))

            # --- Duration collapse ---
            if dna.avg_session_minutes > 5:
                today_avg = np.mean([float(s.get('duration_minutes', 0)) for s in app_sessions])
                ratio = today_avg / max(dna.avg_session_minutes, 0.01)
                duration_collapses.append(max(0.0, 1.0 - ratio))

            # --- Trigger shift ---
            self_opens = sum(1 for s in app_sessions if s.get('trigger') == 'SELF')
            today_self_ratio = self_opens / max(len(app_sessions), 1)
            trigger_drops.append(max(0.0, dna.self_open_ratio - today_self_ratio))

        components = []
        if abandon_deltas:
            components.append(np.mean(abandon_deltas))
        if duration_collapses:
            components.append(np.mean(duration_collapses))
        if trigger_drops:
            components.append(np.mean(trigger_drops))

        if components:
            return float(np.mean(components))
        return 0.0

    # ------------------------------------------------------------------
    # Step 3.5 — L2 modifier computation
    # ------------------------------------------------------------------

    def compute_modifier(
        self,
        coherence: float,
        rhythm_dissolution: float,
        session_incoherence: float,
    ) -> float:
        """
        suppression   = coherence × 0.85
        amplification = (rhythm_dissolution × 0.6 + session_incoherence × 0.4) × 1.5
        modifier      = clip(1.0 - suppression + amplification, 0.15, 2.0)
        """
        suppression = coherence * 0.85
        amplification = (rhythm_dissolution * 0.6 + session_incoherence * 0.4) * 1.5
        modifier = 1.0 - suppression + amplification
        return float(np.clip(modifier, 0.15, 2.0))

    # ------------------------------------------------------------------
    # Step 3.4 — New DNA pattern check
    # ------------------------------------------------------------------

    def check_candidate_flag(
        self,
        coherence: float,
        session_incoherence: float,
    ) -> bool:
        """
        If coherence < 0.25 AND session_incoherence < 0.3:
            candidate_new_pattern = True  → handled by candidate cluster evaluator
        If coherence < 0.25 AND session_incoherence >= 0.3:
            unfamiliar AND degraded → strongest clinical signal, no candidate needed
        """
        return coherence < 0.25 and session_incoherence < 0.3

    # ------------------------------------------------------------------
    # Full L2 pipeline for one day
    # ------------------------------------------------------------------

    def score_day(
        self,
        today_l1_vector: Dict[str, float],
        baseline_dict: Dict[str, float],
        baseline_variances: Dict[str, float],
        today_session_events: Optional[List[Dict]] = None,
        today_notification_events: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run the full L2 pipeline and return all intermediate values.

        Returns dict with:
            coherence, matched_context_id, rhythm_dissolution,
            session_incoherence, modifier, candidate_flag
        """
        coherence, matched_ctx = self.compute_coherence(
            today_l1_vector, baseline_dict, baseline_variances
        )
        rhythm_dissolution = self.compute_rhythm_dissolution(
            today_session_events, matched_ctx
        )
        session_incoherence = self.compute_session_incoherence(today_session_events)

        modifier = self.compute_modifier(coherence, rhythm_dissolution, session_incoherence)
        candidate_flag = self.check_candidate_flag(coherence, session_incoherence)

        return {
            'coherence': coherence,
            'matched_context_id': matched_ctx,
            'rhythm_dissolution': rhythm_dissolution,
            'session_incoherence': session_incoherence,
            'modifier': modifier,
            'candidate_flag': candidate_flag,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_clustering_vector(
        self,
        today_l1_vector: Dict[str, float],
        baseline_dict: Dict[str, float],
        baseline_variances: Dict[str, float],
    ) -> Optional[np.ndarray]:
        """
        Build the 12-feature normalised vector used for DBSCAN matching.
        Normalised to [0,1] using person's own baseline min/max stored in
        cluster_state.
        """
        if self.cluster_state is None:
            return None

        vec = []
        for feat in L1_CLUSTERING_FEATURES:
            val = today_l1_vector.get(feat, baseline_dict.get(feat, 0.0))
            # Normalise using stored min/max from baseline
            if (
                self.cluster_state.feature_min is not None
                and self.cluster_state.feature_max is not None
            ):
                idx = L1_CLUSTERING_FEATURES.index(feat)
                fmin = self.cluster_state.feature_min[idx]
                fmax = self.cluster_state.feature_max[idx]
                rng = fmax - fmin
                if rng > 0:
                    val = (val - fmin) / rng
                else:
                    val = 0.0
            vec.append(val)

        return np.array(vec, dtype=float)

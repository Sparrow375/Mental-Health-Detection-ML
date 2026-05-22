"""
L2 Texture Builder: build ContextualTextureProfile for each L1 archetype.

For each DBSCAN cluster (archetype):
    - Collect baseline days assigned to this archetype
    - Build 22-feature L2 texture vector for each member day
    - If member_days ≥ 10: K-means with K=2,3 (pick by silhouette)
    - If member_days < 10: fallback to mean/std
    - Compute tolerance_factor (mean intra-archetype variance)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from system1.data_structures import ContextualTextureProfile, AppDNA, PhoneDNA
from system1.feature_meta import L2_TEXTURE_FEATURES, DEFAULT_THRESHOLDS

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class L2TextureBuilder:
    """
    Builds per-archetype contextual texture profiles from session and
    notification event data.

    When no session data is available (e.g. StudentLife), returns empty
    profiles — the L2 scorer will use neutral modifiers.
    """

    def __init__(
        self,
        app_dna_dict: Optional[Dict[str, AppDNA]] = None,
        phone_dna: Optional[PhoneDNA] = None,
        min_days_for_kmeans: int | None = None,
    ):
        self.app_dna_dict = app_dna_dict or {}
        self.phone_dna = phone_dna
        self.min_days_for_kmeans = min_days_for_kmeans or DEFAULT_THRESHOLDS['MIN_ARCHETYPE_DAYS_FOR_KMEANS']

    def build_profiles(
        self,
        archetype_labels: np.ndarray,
        baseline_session_events: Optional[List[List[Dict]]] = None,
        baseline_notification_events: Optional[List[List[Dict]]] = None,
    ) -> Dict[int, ContextualTextureProfile]:
        """
        Build one ContextualTextureProfile per L1 archetype.

        Parameters
        ----------
        archetype_labels : array of cluster labels per baseline day
        baseline_session_events : list of daily session events (one entry per day)
        baseline_notification_events : list of daily notification events

        Returns
        -------
        Dict mapping archetype_id → ContextualTextureProfile
        """
        if archetype_labels is None or len(archetype_labels) == 0:
            return {}

        unique_labels = set(archetype_labels)
        unique_labels.discard(-1)  # skip noise

        profiles: Dict[int, ContextualTextureProfile] = {}

        for label in sorted(unique_labels):
            member_indices = np.where(archetype_labels == label)[0]
            member_days = len(member_indices)

            # Build texture vectors for member days
            texture_vectors = []
            for idx in member_indices:
                sess = None
                notif = None
                if baseline_session_events and idx < len(baseline_session_events):
                    sess = baseline_session_events[idx]
                if baseline_notification_events and idx < len(baseline_notification_events):
                    notif = baseline_notification_events[idx]

                tv = self._build_texture_vector(sess, notif)
                texture_vectors.append(tv)

            texture_matrix = np.array(texture_vectors)

            profile = ContextualTextureProfile(
                archetype_id=int(label),
                member_days=member_days,
            )

            if member_days >= self.min_days_for_kmeans and HAS_SKLEARN:
                profile = self._build_kmeans_profile(profile, texture_matrix)
            else:
                profile = self._build_fallback_profile(profile, texture_matrix)

            # Compute tolerance factor (mean intra-archetype variance)
            if texture_matrix.shape[0] > 1:
                profile.tolerance_factor = float(np.mean(np.var(texture_matrix, axis=0)))
            else:
                profile.tolerance_factor = 1.0

            profiles[int(label)] = profile

        return profiles

    def _build_kmeans_profile(
        self,
        profile: ContextualTextureProfile,
        texture_matrix: np.ndarray,
    ) -> ContextualTextureProfile:
        """K-means with K=2 and K=3, pick by silhouette."""
        best_score = -1.0
        best_k = 2

        for k in [2, 3]:
            if texture_matrix.shape[0] < k + 1:
                continue
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(texture_matrix)
                score = silhouette_score(texture_matrix, labels)
                if score > best_score + 0.05:  # need meaningful improvement
                    best_score = score
                    best_k = k
            except Exception:
                continue

        try:
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            km.fit(texture_matrix)
            profile.texture_centroids = km.cluster_centers_

            # Radius per texture cluster
            radii = []
            for ci in range(best_k):
                mask = km.labels_ == ci
                members = texture_matrix[mask]
                if len(members) > 0:
                    dists = np.linalg.norm(members - km.cluster_centers_[ci], axis=1)
                    radii.append(float(np.max(dists)) if len(dists) > 0 else 1.0)
                else:
                    radii.append(1.0)
            profile.texture_radii = np.array(radii)
        except Exception:
            return self._build_fallback_profile(profile, texture_matrix)

        return profile

    def _build_fallback_profile(
        self,
        profile: ContextualTextureProfile,
        texture_matrix: np.ndarray,
    ) -> ContextualTextureProfile:
        """Mean + std fallback when member_days < threshold."""
        if texture_matrix.shape[0] > 0:
            profile.texture_mean = np.mean(texture_matrix, axis=0)
            profile.texture_std = np.std(texture_matrix, axis=0) + 1e-6
        else:
            profile.texture_mean = np.zeros(len(L2_TEXTURE_FEATURES))
            profile.texture_std = np.ones(len(L2_TEXTURE_FEATURES))
        return profile

    def _build_texture_vector(
        self,
        session_events: Optional[List[Dict]],
        notification_events: Optional[List[Dict]],
    ) -> np.ndarray:
        """
        Build the 22-feature L2 texture vector for one day.
        Returns zeros if no session/notification data available.
        """
        vec = np.zeros(len(L2_TEXTURE_FEATURES))

        if not session_events and not notification_events:
            return vec

        sessions = session_events or []
        notifications = notification_events or []

        # --- Temporal anchoring (4) ---
        # time_in_primary_window_ratio
        if sessions and self.app_dna_dict:
            in_window = 0
            total = 0
            for s in sessions:
                pkg = s.get('app_package', '')
                dna = self.app_dna_dict.get(pkg)
                if dna and dna.primary_time_range:
                    hour = int(s.get('hour', 0))
                    start_h, end_h = dna.primary_time_range
                    if start_h <= hour <= end_h:
                        in_window += 1
                    total += 1
            vec[0] = in_window / max(total, 1)

        # temporal_anchor_deviation — std of session start hours
        if sessions:
            hours = [float(s.get('hour', 12)) for s in sessions]
            vec[1] = float(np.std(hours)) if len(hours) > 1 else 0.0

        # first_pickup_hour_deviation
        if sessions and self.phone_dna:
            first_hour = min(float(s.get('hour', 12)) for s in sessions)
            vec[2] = abs(first_hour - self.phone_dna.first_pickup_hour_mean)

        # rhythm_dissolution_score — populated by L2Scorer, leave as 0 here
        vec[3] = 0.0

        # --- Session quality (5) ---
        if sessions:
            durations = [float(s.get('duration_minutes', 0)) for s in sessions]
            interactions = [int(s.get('interaction_count', 0)) for s in sessions]

            # weighted_abandon_rate
            abandons = sum(1 for d, ic in zip(durations, interactions) if d < 0.75 and ic < 5)
            vec[4] = abandons / max(len(sessions), 1)

            # deep_session_ratio
            vec[5] = sum(1 for d in durations if d > 20) / max(len(sessions), 1)

            # micro_session_ratio
            vec[6] = sum(1 for d in durations if d < 2) / max(len(sessions), 1)

            # session_duration_collapse — ratio vs baseline avg
            if self.phone_dna and self.phone_dna.active_window_duration_mean > 0:
                today_avg = np.mean(durations)
                vec[7] = max(0, 1 - today_avg / self.phone_dna.active_window_duration_mean)

            # interaction_density_ratio
            if durations:
                densities = [ic / max(d, 0.01) for ic, d in zip(interactions, durations)]
                vec[8] = float(np.mean(densities))

        # --- Agency & initiation (4) ---
        if sessions:
            triggers = [s.get('trigger', 'SELF') for s in sessions]
            n = max(len(triggers), 1)
            vec[9] = sum(1 for t in triggers if t == 'SELF') / n           # self_open_ratio
            vec[10] = sum(1 for t in triggers if t == 'NOTIFICATION') / n  # notification_open_rate

        if notifications:
            actions = [n_evt.get('action', '') for n_evt in notifications]
            n = max(len(actions), 1)
            vec[11] = sum(1 for a in actions if a == 'IGNORE') / n         # notification_ignore_rate

        # pickup_burst_rate
        if sessions and self.phone_dna:
            vec[12] = self.phone_dna.pickup_burst_rate  # use baseline as reference

        # --- Attention coherence (4) ---
        if sessions:
            pkgs = [s.get('app_package', '') for s in sessions]
            unique_apps = len(set(pkgs))
            vec[13] = len(sessions) / max(unique_apps, 1)    # app_switching_rate
            vec[14] = 0.5  # app_cooccurrence_consistency (needs co-occurrence matrix)
            vec[15] = unique_apps / max(len(sessions), 1)   # distinct_apps_ratio
            vec[16] = 0.5  # session_context_match (placeholder)

        # --- Rhythm & structure (3) ---
        if self.phone_dna:
            vec[17] = self.phone_dna.daily_rhythm_regularity    # daily_rhythm_regularity
            vec[18] = 1.0 - self.phone_dna.weekday_weekend_delta  # weekday_weekend_alignment

        # dead_zone_count — hours with zero pickups that were previously active
        if sessions and self.phone_dna and self.phone_dna.historically_active_hours:
            active_today = set(int(s.get('hour', 0)) for s in sessions)
            dead_zones = sum(1 for h in self.phone_dna.historically_active_hours if h not in active_today)
            vec[19] = dead_zones

        # --- Notification relationship (2) ---
        if notifications:
            latencies = [float(n_evt.get('tap_latency_minutes', 0)) for n_evt in notifications if n_evt.get('tap_latency_minutes')]
            if latencies and self.phone_dna:
                vec[20] = float(np.mean(latencies))  # notification_response_latency_shift

            # notification_to_session_ratio
            vec[21] = len(notifications) / max(len(sessions), 1) if sessions else 0.0

        return vec

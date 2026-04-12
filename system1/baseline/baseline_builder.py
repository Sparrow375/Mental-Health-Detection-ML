"""
Baseline Builder: orchestrator for all baseline construction steps.

Phases (from docx):
    1.1  PersonalityVector construction (mean + std of 29 features)
    1.2  Per-app AppDNA construction  (from session events)
    1.3  PhoneDNA construction  (from all session events)
    1.4  L1 context clustering  (DBSCAN with Mahalanobis)
    1.5  L2 contextual texture profiles  (K-means per archetype)
    1.6  Detector calibration  (retroactive threshold adjustment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from system1.data_structures import (
    PersonalityVector,
    AppDNA,
    PhoneDNA,
    ContextualTextureProfile,
    L1ClusterState,
)
from system1.feature_meta import ALL_L1_FEATURES, DEFAULT_THRESHOLDS
from system1.baseline.l1_clusterer import L1Clusterer
from system1.baseline.l2_texture_builder import L2TextureBuilder
from system1.baseline.app_dna_builder import AppDNABuilder
from system1.baseline.phone_dna_builder import PhoneDNABuilder
from system1.baseline.detector_calibration import calibrate_thresholds


class BaselineProfile:
    """Container for all baseline outputs."""

    def __init__(self):
        self.personality_vector: Optional[PersonalityVector] = None
        self.app_dna_dict: Dict[str, AppDNA] = {}
        self.phone_dna: Optional[PhoneDNA] = None
        self.cluster_state: Optional[L1ClusterState] = None
        self.texture_profiles: Dict[int, ContextualTextureProfile] = {}
        self.thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        self.baseline_df: Optional[pd.DataFrame] = None


class BaselineBuilder:
    """
    Orchestrates all baseline construction steps.

    Usage:
        builder = BaselineBuilder()
        profile = builder.build(daily_features_df)
        # or with session data:
        profile = builder.build(daily_features_df, session_events, notification_events)
    """

    def build(
        self,
        daily_features_df: pd.DataFrame,
        session_events: Optional[List[List[Dict]]] = None,
        notification_events: Optional[List[List[Dict]]] = None,
        baseline_days: int = 28,
    ) -> BaselineProfile:
        """
        Build complete baseline profile.

        Parameters
        ----------
        daily_features_df : DataFrame with one row per day, columns matching
                            ALL_L1_FEATURES (plus 'date').
        session_events : list of daily session event lists (may be None)
        notification_events : list of daily notification event lists (may be None)
        baseline_days : number of days to use for baseline

        Returns
        -------
        BaselineProfile with all components populated.
        """
        profile = BaselineProfile()

        # Use first N days
        baseline_df = daily_features_df.head(baseline_days).copy()
        profile.baseline_df = baseline_df

        print(f"\n  Building baseline from {len(baseline_df)} days...")

        # Step 1.1 — PersonalityVector
        profile.personality_vector = self._build_personality_vector(baseline_df)
        print(f"  ✓ Step 1.1: PersonalityVector ({len(profile.personality_vector.to_dict())} features)")

        # Step 1.2 — AppDNA
        app_builder = AppDNABuilder()
        sess_for_baseline = session_events[:baseline_days] if session_events else None
        notif_for_baseline = notification_events[:baseline_days] if notification_events else None
        profile.app_dna_dict = app_builder.build(sess_for_baseline, notif_for_baseline, baseline_days)
        print(f"  ✓ Step 1.2: AppDNA ({len(profile.app_dna_dict)} apps)")

        # Step 1.3 — PhoneDNA
        phone_builder = PhoneDNABuilder()
        profile.phone_dna = phone_builder.build(sess_for_baseline, notif_for_baseline, baseline_days)
        print(f"  ✓ Step 1.3: PhoneDNA built")

        # Step 1.4 — L1 DBSCAN clustering
        clusterer = L1Clusterer()
        profile.cluster_state = clusterer.fit(baseline_df)
        print(f"  ✓ Step 1.4: {profile.cluster_state.n_clusters} L1 archetypes")

        # Step 1.5 — L2 texture profiles
        texture_builder = L2TextureBuilder(profile.app_dna_dict, profile.phone_dna)
        labels = profile.cluster_state.labels
        if labels is not None:
            profile.texture_profiles = texture_builder.build_profiles(
                labels, sess_for_baseline, notif_for_baseline
            )
        print(f"  ✓ Step 1.5: {len(profile.texture_profiles)} texture profiles")

        # Step 1.6 — Detector calibration
        profile.thresholds = calibrate_thresholds(
            baseline_df, profile.personality_vector
        )
        print(f"  ✓ Step 1.6: Thresholds calibrated")

        return profile

    def _build_personality_vector(self, baseline_df: pd.DataFrame) -> PersonalityVector:
        """Step 1.1: compute mean and std of each feature."""
        params = {}
        variances = {}

        for feat in ALL_L1_FEATURES:
            if feat in baseline_df.columns:
                values = baseline_df[feat].dropna()
                if len(values) >= 3:
                    params[feat] = float(values.mean())
                    variances[feat] = float(values.std()) + 0.01  # epsilon
                else:
                    params[feat] = _get_default_value(feat)
                    variances[feat] = params[feat] * 0.15
            else:
                params[feat] = _get_default_value(feat)
                variances[feat] = params[feat] * 0.15

        pv = PersonalityVector.from_dict(params, variances)
        return pv


def _get_default_value(feature_name: str) -> float:
    """Reasonable default values for features not available in data."""
    defaults = {
        'voice_pitch_mean': 180.0,
        'voice_pitch_std': 15.0,
        'voice_energy_mean': 0.65,
        'voice_speaking_rate': 3.5,
        'screen_time_hours': 4.5,
        'unlock_count': 50.0,
        'social_app_ratio': 0.30,
        'app_launch_count': 40.0,
        'notifications_today': 30.0,
        'total_apps_count': 25.0,
        'calls_per_day': 2.0,
        'texts_per_day': 25.0,
        'unique_contacts': 8.0,
        'response_time_minutes': 15.0,
        'daily_displacement_km': 5.0,
        'location_entropy': 2.0,
        'home_time_ratio': 0.65,
        'places_visited': 4.0,
        'wake_time_hour': 8.0,
        'sleep_time_hour': 23.5,
        'sleep_duration_hours': 7.5,
        'dark_duration_hours': 8.0,
        'charge_duration_hours': 6.0,
        'conversation_duration_hours': 1.5,
        'conversation_frequency': 10.0,
        'calendar_events_today': 2.0,
        'upi_transactions_today': 1.0,
        'background_audio_hours': 1.0,
        'storage_used_gb': 32.0,
    }
    return defaults.get(feature_name, 1.0)

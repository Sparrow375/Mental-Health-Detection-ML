"""
Baseline Builder: orchestrator for all baseline construction steps.
Adapted for MHealth app with camelCase feature names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

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
    """Orchestrates all baseline construction steps."""

    def build(
        self,
        daily_features_df: pd.DataFrame,
        session_events=None,
        notification_events=None,
        baseline_days: int = 28,
    ) -> BaselineProfile:
        profile = BaselineProfile()
        baseline_df = daily_features_df.head(baseline_days).copy()
        profile.baseline_df = baseline_df

        print(f"\n  Building baseline from {len(baseline_df)} days...")

        # Step 1.1 — PersonalityVector
        profile.personality_vector = self._build_personality_vector(baseline_df)
        print(f"  ✓ Step 1.1: PersonalityVector ({len(profile.personality_vector.to_dict())} features)")

        # Step 1.2 — AppDNA
        app_builder = AppDNABuilder()
        sess = session_events[:baseline_days] if session_events else None
        notif = notification_events[:baseline_days] if notification_events else None
        profile.app_dna_dict = app_builder.build(sess, notif, baseline_days)
        print(f"  ✓ Step 1.2: AppDNA ({len(profile.app_dna_dict)} apps)")

        # Step 1.3 — PhoneDNA
        phone_builder = PhoneDNABuilder()
        profile.phone_dna = phone_builder.build(sess, notif, baseline_days)
        print(f"  ✓ Step 1.3: PhoneDNA built")

        # Step 1.4 — L1 DBSCAN clustering
        clusterer = L1Clusterer()
        profile.cluster_state = clusterer.fit(baseline_df)
        print(f"  ✓ Step 1.4: {profile.cluster_state.n_clusters} L1 archetypes")

        # Step 1.5 — L2 texture profiles
        texture_builder = L2TextureBuilder(profile.app_dna_dict, profile.phone_dna)
        labels = profile.cluster_state.labels
        if labels is not None:
            profile.texture_profiles = texture_builder.build_profiles(labels, sess, notif)
        print(f"  ✓ Step 1.5: {len(profile.texture_profiles)} texture profiles")

        # Step 1.6 — Detector calibration
        profile.thresholds = calibrate_thresholds(baseline_df, profile.personality_vector)
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
                    variances[feat] = float(values.std()) + 0.01
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
        # Screen & App Activity
        "screenTimeHours": 4.5,
        "unlockCount": 50.0,
        "appLaunchCount": 40.0,
        "notificationsToday": 30.0,
        "socialAppRatio": 0.30,
        # Communication
        "callsPerDay": 2.0,
        "callDurationMinutes": 5.0,
        "uniqueContacts": 8.0,
        "conversationFrequency": 0.25,
        # Location & Movement
        "dailyDisplacementKm": 5.0,
        "locationEntropy": 2.0,
        "homeTimeRatio": 0.65,
        # Sleep & Circadian
        "wakeTimeHour": 8.0,
        "sleepTimeHour": 23.5,
        "sleepDurationHours": 7.5,
        "darkDurationHours": 8.0,
        # System Usage
        "chargeDurationHours": 6.0,
        "memoryUsagePercent": 65.0,
        "networkWifiMB": 500.0,
        "networkMobileMB": 200.0,
        "storageUsedGB": 32.0,
        # Behavioural Signals
        "totalAppsCount": 25.0,
        "upiTransactionsToday": 1.0,
        "appUninstallsToday": 0.0,
        "appInstallsToday": 0.0,
        # Calendar & Engagement
        "calendarEventsToday": 2.0,
        "mediaCountToday": 3.0,
        "downloadsToday": 1.0,
        "musicTimeMinutes": 1.0,
    }
    return defaults.get(feature_name, 1.0)

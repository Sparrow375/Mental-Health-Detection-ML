"""
L1 Scorer: deviation magnitude (weighted z-scores), deviation velocity (EWMA),
and composite L1 anomaly score.

Enhanced with Bayesian warm-start, adaptive weights, feature ceilings,
and lifestyle-adjusted weights.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from system1.data_structures import PersonalityVector, BayesianState
from system1.feature_meta import FEATURE_META, ALL_L1_FEATURES, DIRECTIONALITY_DAMPENING


class L1Scorer:
    """
    Stateful scorer that tracks feature history for velocity computation.

    Usage:
        scorer = L1Scorer(baseline)
        deviations = scorer.calculate_deviation_magnitude(current_data)
        velocities = scorer.calculate_deviation_velocity(current_data)
        score      = scorer.calculate_anomaly_score(deviations, velocities)
    """

    def __init__(self, baseline: PersonalityVector, history_window: int = 7):
        self.baseline = baseline
        self.baseline_dict = baseline.to_dict()
        self.feature_names = list(self.baseline_dict.keys())
        self.history_window = history_window
        self.bayesian_state: Optional[BayesianState] = None
        self._adaptive_weights: Dict[str, float] = {}

        # Rolling window for velocity computation
        self.feature_history: Dict[str, deque] = {
            feat: deque(maxlen=history_window) for feat in self.feature_names
        }

    def update_bayesian_state(self, state: BayesianState) -> None:
        """Replace the Bayesian state (called each day before scoring)."""
        self.bayesian_state = state

    def set_adaptive_weights(self, weights: Dict[str, float]) -> None:
        """Override feature weights with data-adaptive values."""
        self._adaptive_weights = weights

    def _get_weight(self, feature: str) -> float:
        """Return adaptive weight if set, else fall back to FEATURE_META."""
        if self._adaptive_weights:
            return self._adaptive_weights.get(feature, FEATURE_META.get(feature, {}).get('weight', 1.0))
        return FEATURE_META.get(feature, {}).get('weight', 1.0)

    # ------------------------------------------------------------------
    # Step 2.1 - Deviation magnitude  (weighted z-scores)
    # ------------------------------------------------------------------

    def calculate_deviation_magnitude(
        self,
        current_data: Dict[str, float],
        feature_ceilings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        For each feature:
            z_raw     = (today_value - effective_mean) / effective_std
            z_clamped = clip(z_raw, -ceiling, +ceiling)
            z_weighted = z_clamped * weight

        Uses Bayesian posterior mean/std when available, falls back to
        raw PersonalityVector baseline otherwise.

        Only includes features with effective_std > 0.05 (skip zero-variance).
        """
        deviations: Dict[str, float] = {}

        if self.bayesian_state is not None:
            effective_means = self.bayesian_state.effective_means
            effective_stds = self.bayesian_state.effective_stds
        else:
            effective_means = None
            effective_stds = None

        for feature in self.feature_names:
            if effective_means is not None and feature in effective_means:
                baseline_val = effective_means[feature]
                variance = effective_stds.get(feature, 1.0)
            else:
                baseline_val = self.baseline_dict[feature]
                variance = self.baseline.variances.get(feature, 1.0) if self.baseline.variances else 1.0

            # Skip features with near-zero variance
            if variance < 0.05:
                continue

            current_val = current_data.get(feature, baseline_val)
            z_raw = (current_val - baseline_val) / variance

            # Per-feature deviation ceiling (default +/-4)
            ceiling = feature_ceilings.get(feature, 4.0) if feature_ceilings else 4.0
            z_clamped = float(np.clip(z_raw, -ceiling, ceiling))

            # Asymmetric directionality: dampen non-significant direction
            directionality = FEATURE_META.get(feature, {}).get('directionality', 'both')
            if directionality == 'shrink_matters' and z_clamped > 0:
                z_clamped *= DIRECTIONALITY_DAMPENING
            elif directionality == 'grow_matters' and z_clamped < 0:
                z_clamped *= DIRECTIONALITY_DAMPENING

            weight = self._get_weight(feature)
            deviations[feature] = z_clamped * weight

        return deviations

    # ------------------------------------------------------------------
    # Step 2.2 - Deviation velocity  (EWMA)
    # ------------------------------------------------------------------

    def calculate_deviation_velocity(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """
        EWMA with alpha=0.4:
            ewma_t = 0.4 * current_value + 0.6 * previous_ewma
            slope  = (ewma_last - ewma_first) / window_length
            velocity = slope / baseline_mean   (normalised)

        Returns dict of normalised velocity slopes.
        """
        alpha = 0.4
        velocities: Dict[str, float] = {}

        # Append current values to history
        for feature in self.feature_names:
            val = current_data.get(feature, self.baseline_dict[feature])
            self.feature_history[feature].append(val)

        for feature in self.feature_names:
            history = list(self.feature_history[feature])

            if len(history) < 2:
                velocities[feature] = 0.0
            else:
                # EWMA computation
                ewma = history[0]
                ewma_values = [ewma]
                for val in history[1:]:
                    ewma = alpha * val + (1 - alpha) * ewma
                    ewma_values.append(ewma)

                slope = (ewma_values[-1] - ewma_values[0]) / len(ewma_values)
                baseline_val = self.baseline_dict[feature]
                if baseline_val > 0:
                    velocities[feature] = slope / baseline_val
                else:
                    velocities[feature] = 0.0

        return velocities

    # ------------------------------------------------------------------
    # Step 2.3 - Composite L1 score
    # ------------------------------------------------------------------

    def calculate_anomaly_score(
        self,
        deviations: Dict[str, float],
        velocities: Dict[str, float],
    ) -> float:
        """
        RMS computed only over features present in `deviations`
        (zero-variance features excluded upstream).

        magnitude_score = RMS(weighted z-scores) / 3.0 -> capped at 1.0
        velocity_score  = RMS(velocities for observed features) * 10.0 -> capped at 1.0
        L1_score = 0.7 * magnitude_score + 0.3 * velocity_score
        """
        observed_features = set(deviations.keys())

        dev_vals = list(deviations.values())
        if dev_vals:
            magnitude_score = float(np.sqrt(np.mean(np.square(dev_vals))))
        else:
            magnitude_score = 0.0
        magnitude_score = min(magnitude_score / 3.0, 1.0)

        # Only take velocity for features that were scored
        vel_vals = [v for k, v in velocities.items() if k in observed_features]
        if vel_vals:
            velocity_score = float(np.sqrt(np.mean(np.square(vel_vals))))
        else:
            velocity_score = 0.0
        velocity_score = min(velocity_score * 10.0, 1.0)

        return 0.7 * magnitude_score + 0.3 * velocity_score

    # ------------------------------------------------------------------
    # Lifestyle-adjusted weights
    # ------------------------------------------------------------------

    LIFESTYLE_FEATURE_MAP: Dict[str, List[str]] = {
        'screen': ['screenTimeHours', 'unlockCount', 'socialAppRatio',
                    'appLaunchCount', 'notificationsToday', 'totalAppsCount'],
        'communication': ['callsPerDay', 'callDurationMinutes', 'uniqueContacts',
                          'conversationFrequency'],
        'movement': ['dailyDisplacementKm', 'locationEntropy', 'homeTimeRatio'],
        'sleep': ['wakeTimeHour', 'sleepTimeHour', 'sleepDurationHours',
                  'darkDurationHours', 'chargeDurationHours'],
        'engagement': ['calendarEventsToday', 'musicTimeMinutes',
                       'upiTransactionsToday', 'mediaCountToday', 'downloadsToday'],
    }

    def apply_lifestyle_weights(self, user_profile) -> None:
        """
        Modulate feature weights by user's self-reported lifestyle scores.

        Formula: effective_weight = clinical_weight * (0.7 + 0.3 * score / 5.0)
        """
        lifestyle = user_profile.get_lifestyle_dict()
        behavioral_mult = (0.85 + 0.15 * lifestyle.get('behavioral', 3) / 5.0)

        adjusted = {}
        for feat in self.feature_names:
            base_weight = FEATURE_META.get(feat, {}).get('weight', 1.0)

            category_mult = 1.0
            for cat, feats in self.LIFESTYLE_FEATURE_MAP.items():
                if feat in feats:
                    score = lifestyle.get(cat, 3)
                    category_mult = 0.7 + 0.3 * score / 5.0
                    break

            adjusted[feat] = base_weight * category_mult * behavioral_mult

        self._adaptive_weights = adjusted

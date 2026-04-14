"""
L1 Scorer: deviation magnitude (weighted z-scores), deviation velocity (EWMA),
and composite L1 anomaly score.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from system1.data_structures import PersonalityVector
from system1.feature_meta import FEATURE_META, ALL_L1_FEATURES


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

        # Rolling window for velocity computation
        self.feature_history: Dict[str, deque] = {
            feat: deque(maxlen=history_window) for feat in self.feature_names
        }

    # ------------------------------------------------------------------
    # Step 2.1 — Deviation magnitude  (weighted z-scores)
    # ------------------------------------------------------------------

    def calculate_deviation_magnitude(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """
        For each of the 29 features:
            z_raw     = (today_value - baseline_mean) / baseline_std
            z_weighted = z_raw * FEATURE_META[feature]['weight']

        Returns dict of 29 weighted z-scores.
        """
        deviations: Dict[str, float] = {}

        for feature in self.feature_names:
            baseline_val = self.baseline_dict[feature]
            current_val = current_data.get(feature, baseline_val)
            variance = self.baseline.variances.get(feature, 1.0) if self.baseline.variances else 1.0

            if variance > 0:
                z_raw = (current_val - baseline_val) / variance
            else:
                z_raw = 0.0

            weight = FEATURE_META.get(feature, {}).get('weight', 1.0)
            deviations[feature] = z_raw * weight

        return deviations

    # ------------------------------------------------------------------
    # Step 2.2 — Deviation velocity  (EWMA)
    # ------------------------------------------------------------------

    def calculate_deviation_velocity(self, current_data: Dict[str, float]) -> Dict[str, float]:
        """
        EWMA with alpha=0.4:
            ewma_t = 0.4 * current_value + 0.6 * previous_ewma
            slope  = (ewma_last - ewma_first) / window_length
            velocity = slope / baseline_mean   (normalised)

        Returns dict of 29 normalised velocity slopes.
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
    # Step 2.3 — Composite L1 score
    # ------------------------------------------------------------------

    def calculate_anomaly_score(
        self,
        deviations: Dict[str, float],
        velocities: Dict[str, float],
    ) -> float:
        """
        magnitude_score = RMS(all weighted z-scores) / 3.0   → capped at 1.0
        velocity_score  = RMS(all velocities) * 10.0         → capped at 1.0
        L1_score = 0.7 * magnitude_score + 0.3 * velocity_score
        
        Root Mean Square squares the deviations, effectively ignoring stationary noise
        while exponentially penalizing localized, severe behavioral shifts.
        """
        dev_vals = list(deviations.values())
        if dev_vals:
            magnitude_score = float(np.sqrt(np.mean(np.square(dev_vals))))
        else:
            magnitude_score = 0.0
        magnitude_score = min(magnitude_score / 3.0, 1.0)

        vel_vals = list(velocities.values())
        if vel_vals:
            velocity_score = float(np.sqrt(np.mean(np.square(vel_vals))))
        else:
            velocity_score = 0.0
        velocity_score = min(velocity_score * 10.0, 1.0)

        return 0.7 * magnitude_score + 0.3 * velocity_score

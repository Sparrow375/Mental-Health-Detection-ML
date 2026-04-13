"""
Phase 3 — Temporal Shape Validator
====================================

After the distance scorer gives a tentative classification, this module
validates (or contradicts) it by analysing the *shape* of the anomaly
score time-series over a 14–28 day rolling window.

Detected shapes
---------------
  monotonic_drift   — steady decline (depression)
  oscillating       — regular cycling every 3-10 days (BPD)
  chaotic           — high variance, no autocorrelation (schizophrenia)
  episodic_spike    — single spike > 2 SD, recovers in 2 weeks (anxiety)
  phase_flip        — sustained low → sudden reversal (bipolar manic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import (
    TEMPORAL_SHAPES,
    SHAPE_DISORDER_MATRIX,
    TEMPORAL_BOOST,
    TEMPORAL_DOWNGRADE,
)
from prototype_matcher import ClassificationResult, ConfidenceTier


@dataclass
class AdjustedClassification:
    """Classification after temporal validation."""
    disorder: str
    original_score: float
    adjusted_score: float
    confidence: ConfidenceTier
    temporal_shape: str
    shape_supports: bool            # True = confirms, False = contradicts
    shape_neutral: bool             # True = no effect
    all_scores: Dict[str, float]
    frame_used: int


class TemporalValidator:
    """
    Validates a tentative classification against the temporal trajectory
    of the anomaly time-series.
    """

    # ── Shape Detection ─────────────────────────────────────────────

    def detect_shape(self, timeseries: List[float] | np.ndarray) -> str:
        """
        Classify the anomaly score time-series into a temporal shape.

        Parameters
        ----------
        timeseries : array-like
            Daily anomaly scores, most-recent last.  Length ≥ 14.

        Returns
        -------
        str  — one of the keys in TEMPORAL_SHAPES, or "unknown".
        """
        ts = np.asarray(timeseries, dtype=float)
        n = len(ts)

        if n < 7:
            return "unknown"

        # --- Monotonic drift ---
        if self._is_monotonic_drift(ts):
            return "monotonic_drift"

        # --- Phase flip ---
        if n >= 14 and self._is_phase_flip(ts):
            return "phase_flip"

        # --- Oscillating ---
        if n >= 14 and self._is_oscillating(ts):
            return "oscillating"

        # --- Episodic spike ---
        if self._is_episodic_spike(ts):
            return "episodic_spike"

        # --- Chaotic ---
        if self._is_chaotic(ts):
            return "chaotic"

        return "unknown"

    # ── Validation ──────────────────────────────────────────────────

    def validate(
        self,
        classification: ClassificationResult,
        timeseries: List[float] | np.ndarray,
    ) -> AdjustedClassification:
        """
        Validate a classification against the anomaly trajectory.

        Boosts confidence (×1.2) if shape confirms, downgrades (×0.6)
        if shape contradicts, no change if neutral.
        """
        shape = self.detect_shape(timeseries)
        disorder = classification.disorder

        # Look up compatibility
        compat = SHAPE_DISORDER_MATRIX.get(shape, {}).get(disorder, 0)

        if compat > 0:
            factor = TEMPORAL_BOOST
            supports = True
            neutral = False
        elif compat < 0:
            factor = TEMPORAL_DOWNGRADE
            supports = False
            neutral = False
        else:
            factor = 1.0
            supports = False
            neutral = True

        adjusted = classification.score * factor

        # Re-derive confidence tier
        from config import CONFIDENCE_THRESHOLDS
        if adjusted >= CONFIDENCE_THRESHOLDS["high"]:
            tier = ConfidenceTier.HIGH
        elif adjusted >= CONFIDENCE_THRESHOLDS["low"]:
            tier = ConfidenceTier.LOW
        else:
            tier = ConfidenceTier.UNCLASSIFIED

        return AdjustedClassification(
            disorder=disorder,
            original_score=classification.score,
            adjusted_score=adjusted,
            confidence=tier,
            temporal_shape=shape,
            shape_supports=supports,
            shape_neutral=neutral,
            all_scores=classification.all_scores,
            frame_used=classification.frame_used,
        )

    # ── Private shape detectors ─────────────────────────────────────

    def _is_monotonic_drift(self, ts: np.ndarray) -> bool:
        """Linear regression: slope < min_slope AND R² > threshold."""
        params = TEMPORAL_SHAPES["monotonic_drift"]
        n = len(ts)
        x = np.arange(n)
        coeffs = np.polyfit(x, ts, 1)
        slope = coeffs[0]

        # R²
        predicted = np.polyval(coeffs, x)
        ss_res = float(np.sum((ts - predicted) ** 2))
        ss_tot = float(np.sum((ts - np.mean(ts)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope < params["min_slope"] and r_squared > params["min_r_squared"]

    def _is_oscillating(self, ts: np.ndarray) -> bool:
        """Autocorrelation peak within 3-10 day lag band."""
        params = TEMPORAL_SHAPES["oscillating"]
        lag_min = params["autocorr_lag_min"]
        lag_max = min(params["autocorr_lag_max"], len(ts) // 2)
        threshold = params["autocorr_threshold"]

        if lag_max <= lag_min:
            return False

        ts_centered = ts - np.mean(ts)
        var = float(np.var(ts_centered))
        if var == 0:
            return False

        for lag in range(lag_min, lag_max + 1):
            corr = float(np.mean(ts_centered[:-lag] * ts_centered[lag:])) / var
            if corr > threshold:
                return True
        return False

    def _is_chaotic(self, ts: np.ndarray) -> bool:
        """High variance + low short-lag autocorrelation."""
        params = TEMPORAL_SHAPES["chaotic"]
        variance = float(np.var(ts))

        # Compare to expected variance (normalised — we use 1.0 as default)
        if variance < params["min_variance_ratio"]:
            return False

        # Check autocorrelation at lag 1
        ts_centered = ts - np.mean(ts)
        var_c = float(np.var(ts_centered))
        if var_c == 0:
            return False

        lag1_corr = float(np.mean(ts_centered[:-1] * ts_centered[1:])) / var_c
        return abs(lag1_corr) < params["max_autocorr"]

    def _is_episodic_spike(self, ts: np.ndarray) -> bool:
        """Single peak > 2 SD that returns within 14 days."""
        params = TEMPORAL_SHAPES["episodic_spike"]
        mean_val = float(np.mean(ts))
        std_val = float(np.std(ts))
        if std_val == 0:
            return False

        spike_idx = np.where((ts - mean_val) > params["spike_sd_threshold"] * std_val)[0]
        if len(spike_idx) == 0:
            return False

        # Check if the spike recovers within the window
        last_spike = int(spike_idx[-1])
        recovery_window = params["recovery_window_days"]
        window_end = min(last_spike + recovery_window, len(ts))

        if window_end >= len(ts):
            # Can't verify recovery yet
            return False

        post_spike = ts[last_spike + 1 : window_end]
        if len(post_spike) == 0:
            return False

        # "Recovered" = back within 1 SD of mean
        return bool(np.all(np.abs(post_spike - mean_val) < std_val))

    def _is_phase_flip(self, ts: np.ndarray) -> bool:
        """Sustained low → sudden reversal between consecutive weeks."""
        params = TEMPORAL_SHAPES["phase_flip"]
        n = len(ts)
        if n < 14:
            return False

        # Compare consecutive 7-day blocks
        n_weeks = n // 7
        weekly_means = [float(np.mean(ts[i * 7 : (i + 1) * 7])) for i in range(n_weeks)]

        if len(weekly_means) < 2:
            return False

        for i in range(1, len(weekly_means)):
            diff = abs(weekly_means[i] - weekly_means[i - 1])
            overall_std = float(np.std(ts))
            if overall_std == 0:
                continue
            if diff / overall_std > params["diff_sd_threshold"]:
                return True
        return False


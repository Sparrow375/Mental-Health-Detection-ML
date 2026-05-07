"""
Phase 2 — Distance Scoring Engine (Prototype Matcher)
======================================================

Compares a user's deviation pattern against all disorder prototypes and
returns a ranked classification with confidence tiers.

Scoring uses a weighted combination of:
  • Cosine Similarity  (shape — which features are high vs low)
  • Weighted Euclidean Distance (magnitude match)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME1,
    DISORDER_PROTOTYPES_FRAME2,
    FEATURE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
)


# ── Data classes ────────────────────────────────────────────────────────

class ConfidenceTier(Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass
class ClassificationResult:
    """Output of prototype matching."""
    disorder: str                             # top-match disorder name
    score: float                              # combined match score
    confidence: ConfidenceTier                # HIGH / LOW / UNCLASSIFIED
    all_scores: Dict[str, float] = field(default_factory=dict)
    frame_used: int = 2                       # 1 or 2


# ── Prototype Matcher ──────────────────────────────────────────────────

class PrototypeMatcher:
    """
    Distance-based disorder classifier.

    Parameters
    ----------
    prototypes_frame1 : dict, optional
    prototypes_frame2 : dict, optional
    feature_weights : dict, optional
    """

    def __init__(
        self,
        prototypes_frame1: Dict | None = None,
        prototypes_frame2: Dict | None = None,
        feature_weights: Dict | None = None,
    ):
        self.prototypes_f1 = prototypes_frame1 or DISORDER_PROTOTYPES_FRAME1
        self.prototypes_f2 = prototypes_frame2 or DISORDER_PROTOTYPES_FRAME2
        self.weights = feature_weights or FEATURE_WEIGHTS
        self.features = BEHAVIORAL_FEATURES
        self.norms = POPULATION_NORMS

    # ── Core distance functions ─────────────────────────────────────

    @staticmethod
    def weighted_euclidean(
        u: np.ndarray, p: np.ndarray, w: np.ndarray
    ) -> float:
        """
        Calculates weighted Euclidean distance between user vector and prototype.
        This provides perfect Nearest-Centroid classification when using empirical vectors.
        Adds a massive penalty (5x) for sign mismatches on significant deviations.
        """
        diff_sq = (u - p) ** 2
        
        # Apply sign penalty
        sign_mismatch = (np.sign(u) != np.sign(p)) & (np.abs(u) > 0.3) & (np.abs(p) > 0.3)
        penalized_diff_sq = np.where(sign_mismatch, diff_sq * 5.0, diff_sq)
        
        return float(np.sqrt(np.sum(w * penalized_diff_sq)))

    @staticmethod
    def match_score(dist: float, max_w_sum: float) -> float:
        """
        Pure Euclidean distance metric mapped to a [0, 1] probability score.
        Normalized by the weight sum so missing features don't skew the score.
        """
        normalized_dist = dist / (max_w_sum + 1e-6)
        return 1.0 / (1.0 + normalized_dist)

    # ── Classification ─────────────────────────────────────────────

    def classify(
        self,
        deviation_vector: Dict[str, float],
        frame: int = 2,
    ) -> ClassificationResult:
        """
        Score the user's deviation vector against all disorder prototypes.

        Parameters
        ----------
        deviation_vector : dict
            feature_name → value.
            Frame 2: z-scores from personal baseline.
            Frame 1: raw absolute values.
        frame : int
            Which reference frame to use (1 or 2).

        Returns
        -------
        ClassificationResult
        """
        prototypes = self.prototypes_f2 if frame == 2 else self.prototypes_f1

        scores: Dict[str, float] = {}
        for disorder, proto in prototypes.items():
            u_vec, p_vec, w_vec = self._build_vectors(
                deviation_vector, proto, frame
            )
            if len(u_vec) == 0:
                scores[disorder] = 0.0
                continue

            u = np.array(u_vec)
            p = np.array(p_vec)
            w = np.array(w_vec)
            w_sum = np.sum(w)

            dist = self.weighted_euclidean(u, p, w)
            scores[disorder] = self.match_score(dist, w_sum)

        # Rank
        top = max(scores, key=scores.get)
        top_score = scores[top]

        # Confidence tier
        if top_score >= CONFIDENCE_THRESHOLDS["high"]:
            tier = ConfidenceTier.HIGH
        elif top_score >= CONFIDENCE_THRESHOLDS["low"]:
            tier = ConfidenceTier.LOW
        else:
            tier = ConfidenceTier.UNCLASSIFIED

        return ClassificationResult(
            disorder=top,
            score=top_score,
            confidence=tier,
            all_scores=scores,
            frame_used=frame,
        )

    # ── Internal helpers ───────────────────────────────────────────

    def _build_vectors(
        self,
        user: Dict[str, float],
        proto: Dict[str, float],
        frame: int,
    ) -> tuple[List[float], List[float], List[float]]:
        """
        Align user and prototype into ordered vectors.

        Frame 1 raw values are normalised to population z-scores so that
        features are comparable across scales.
        Frame 2 values are already z-scores, used directly.
        Z-scores clamped to [-5, +5] to prevent outlier domination.
        """
        u_vec: List[float] = []
        p_vec: List[float] = []
        w_vec: List[float] = []

        for feat in self.features:
            if feat not in user or feat not in proto:
                continue

            if frame == 1:
                norm = self.norms[feat]
                if norm["std"] == 0:
                    continue
                u_val = (user[feat] - norm["mean"]) / norm["std"]
                p_val = (proto[feat] - norm["mean"]) / norm["std"]
            else:
                u_val = user[feat]
                p_val = proto[feat]
                # Clamp extreme z-scores to prevent outlier domination
                u_val = max(-5.0, min(5.0, u_val))

            u_vec.append(u_val)
            p_vec.append(p_val)
            w_vec.append(self.weights.get(feat, 0.5))

        return u_vec, p_vec, w_vec

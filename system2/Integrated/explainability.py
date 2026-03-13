"""
Phase 5 — Explainability & Output Formatting
==============================================

Generates human-readable output for each classification:
  1. Top contributing features driving the match
  2. Natural language narrative
  3. Radar / spider chart (matplotlib)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from system2.config import BEHAVIORAL_FEATURES, FEATURE_WEIGHTS

# Try to import matplotlib — graceful fallback if unavailable.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@dataclass
class Explanation:
    """Human-readable explanation of a classification."""
    narrative: str
    top_features: List[str]
    top_feature_values: Dict[str, float]   # feature → deviation magnitude
    radar_chart_path: Optional[str] = None


class ExplainabilityEngine:
    """
    Generates explanations for System 2 classification results.
    """

    def __init__(self, feature_weights: Dict[str, float] | None = None):
        self.weights = feature_weights or FEATURE_WEIGHTS
        self.features = BEHAVIORAL_FEATURES

    # ── Top Contributing Features ───────────────────────────────────

    def top_contributing_features(
        self,
        deviation_vector: Dict[str, float],
        prototype_vector: Dict[str, float],
        n: int = 3,
    ) -> List[str]:
        """
        Return the top-N features with highest weighted deviation
        agreement with the matched prototype.
        """
        scores: Dict[str, float] = {}
        for feat in self.features:
            if feat not in deviation_vector or feat not in prototype_vector:
                continue
            # Weighted magnitude of the contribution
            w = self.weights.get(feat, 0.5)
            # How much the user's deviation aligns (same sign & magnitude)
            user_dev = deviation_vector[feat]
            proto_dev = prototype_vector[feat]
            # Contribution = weight × |user_dev| × sign-agreement
            sign_agree = 1.0 if (user_dev * proto_dev) > 0 else -0.5
            scores[feat] = w * abs(user_dev) * sign_agree

        ranked = sorted(scores, key=lambda f: scores[f], reverse=True)
        return ranked[:n]

    # ── Natural Language Narrative ──────────────────────────────────

    def generate_narrative(
        self,
        disorder: str,
        score: float,
        top_features: List[str],
        days: int = 30,
    ) -> str:
        """
        Generate a patient-friendly narrative using a template.
        """
        # Prettify feature names for display
        pretty = [f.replace("_", " ") for f in top_features]

        if len(pretty) >= 3:
            feature_str = f"{pretty[0]}, {pretty[1]}, and {pretty[2]}"
        elif len(pretty) == 2:
            feature_str = f"{pretty[0]} and {pretty[1]}"
        elif pretty:
            feature_str = pretty[0]
        else:
            feature_str = "multiple behavioral metrics"

        disorder_display = disorder.replace("_", " ").title()
        confidence_pct = round(score * 100)

        narrative = (
            f"Over the past {days} days, your behavioral patterns show "
            f"significant changes in {feature_str}. This pattern is "
            f"consistent with {disorder_display}-like behavioral changes "
            f"(confidence: {confidence_pct}%)."
        )
        return narrative

    # ── Radar Chart Generator ───────────────────────────────────────

    def generate_radar_chart(
        self,
        user_profile: Dict[str, float],
        prototype: Dict[str, float],
        disorder_name: str,
        save_path: str,
        healthy_profile: Dict[str, float] | None = None,
    ) -> str:
        """
        Plot user profile vs matched disorder prototype on a spider chart.

        Parameters
        ----------
        user_profile : dict     feature → value  (z-scores or raw)
        prototype : dict        feature → value
        disorder_name : str     label for the prototype
        save_path : str         where to save the PNG
        healthy_profile : dict  optional healthy baseline for reference

        Returns the save_path on success.
        """
        if not _HAS_MPL:
            return ""

        # Select features present in both profiles
        feats = [f for f in self.features if f in user_profile and f in prototype]
        if not feats:
            return ""

        n = len(feats)
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
        angles += angles[:1]   # close the polygon

        user_vals = [user_profile[f] for f in feats] + [user_profile[feats[0]]]
        proto_vals = [prototype[f] for f in feats] + [prototype[feats[0]]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        labels = [f.replace("_", " ").title() for f in feats]
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)

        # Healthy baseline (grey)
        if healthy_profile:
            healthy_vals = [healthy_profile.get(f, 0) for f in feats]
            healthy_vals += [healthy_vals[0]]
            ax.fill(angles, healthy_vals, alpha=0.1, color="grey", label="Healthy")
            ax.plot(angles, healthy_vals, color="grey", linewidth=1, linestyle="--")

        # Prototype (dashed colour)
        ax.plot(angles, proto_vals, color="crimson", linewidth=2, linestyle="--",
                label=f"{disorder_name.replace('_', ' ').title()} Prototype")
        ax.fill(angles, proto_vals, alpha=0.05, color="crimson")

        # User profile (solid fill)
        ax.plot(angles, user_vals, color="dodgerblue", linewidth=2, label="User Profile")
        ax.fill(angles, user_vals, alpha=0.20, color="dodgerblue")

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title(f"Behavioral Profile vs {disorder_name.replace('_', ' ').title()}", pad=20)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return save_path

    # ── Full Explanation ────────────────────────────────────────────

    def explain(
        self,
        disorder: str,
        score: float,
        deviation_vector: Dict[str, float],
        prototype_vector: Dict[str, float],
        days: int = 30,
        chart_path: str | None = None,
        healthy_profile: Dict[str, float] | None = None,
    ) -> Explanation:
        """
        Generate a complete explanation: top features, narrative, and
        optionally a radar chart.
        """
        top_feats = self.top_contributing_features(deviation_vector, prototype_vector)

        top_vals = {
            f: deviation_vector.get(f, 0.0) for f in top_feats
        }

        narrative = self.generate_narrative(disorder, score, top_feats, days)

        chart_saved = None
        if chart_path:
            chart_saved = self.generate_radar_chart(
                deviation_vector, prototype_vector, disorder,
                chart_path, healthy_profile,
            )

        return Explanation(
            narrative=narrative,
            top_features=top_feats,
            top_feature_values=top_vals,
            radar_chart_path=chart_saved or None,
        )

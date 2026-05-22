"""
Bayesian Warm Start: Normal-Inverse-Gamma conjugate prior for per-feature
mean and variance estimation.

Insertion point:
    PersonalityVector → [BayesianBaseline] → z-score → CUSUM → Alert engine

Three-phase schedule:
    Population-anchored  (Day 0-13)  — population priors dominate
    Blended              (Day 14-59) — personal data grows, confidence shown
    Idiographic          (Day 60+)   — personal baseline dominates

Mathematical model (per feature):
    Prior:   sigma^2 ~ InvGamma(alpha_0, beta_0)
             mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0)

    Posterior after n observations with sample mean x_bar and SS:
        kappa_n = kappa_0 + n
        mu_n    = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        alpha_n = alpha_0 + n / 2
        beta_n  = beta_0 + SS / 2 + kappa_0 * n * (x_bar - mu_0)^2 / (2 * kappa_n)

    where SS = sum(x_i^2) - n * x_bar^2  (running sufficient statistics)

Adapted for MHealth: feature names are camelCase (matching Android).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from system1.data_structures import (
    BaselinePhase,
    BayesianState,
    FeaturePosterior,
)
from system1.feature_meta import FEATURE_META

# Population norms from StudentLife healthy cohort (N=27).
# {feature: {"mean": float, "std": float}}
# Imported lazily to avoid circular imports at module level.
_POPULATION_NORMS: Optional[Dict[str, Dict[str, float]]] = None

# Mapping: camelCase (System 1) → snake_case (System 2 config)
_CAMEL_TO_SNAKE: Dict[str, str] = {
    "screenTimeHours": "screen_time_hours",
    "unlockCount": "unlock_count",
    "socialAppRatio": "social_app_ratio",
    "callsPerDay": "calls_per_day",
    "uniqueContacts": "unique_contacts",
    "dailyDisplacementKm": "daily_displacement_km",
    "locationEntropy": "location_entropy",
    "homeTimeRatio": "home_time_ratio",
    "wakeTimeHour": "wake_time_hour",
    "sleepTimeHour": "sleep_time_hour",
    "sleepDurationHours": "sleep_duration_hours",
    "darkDurationHours": "dark_duration_hours",
    "chargeDurationHours": "charge_duration_hours",
    "conversationFrequency": "conversation_frequency",
    "musicTimeMinutes": "background_audio_hours",
    "calendarEventsToday": "calendar_events_today",
    "upiTransactionsToday": "upi_transactions_today",
}


def _get_population_norms() -> Dict[str, Dict[str, float]]:
    global _POPULATION_NORMS
    if _POPULATION_NORMS is None:
        from system2.config import POPULATION_NORMS
        # Store as-is (snake_case keys)
        _POPULATION_NORMS = POPULATION_NORMS
    return _POPULATION_NORMS


# Default values for features NOT in POPULATION_NORMS.
# Keys are camelCase matching MHealth PersonalityVector.
_FALLBACK_DEFAULTS: Dict[str, float] = {
    'appLaunchCount': 40.0,
    'notificationsToday': 30.0,
    'callDurationMinutes': 5.0,
    'memoryUsagePercent': 50.0,
    'networkWifiMB': 100.0,
    'networkMobileMB': 50.0,
    'totalAppsCount': 25.0,
    'appUninstallsToday': 0.5,
    'appInstallsToday': 0.5,
    'mediaCountToday': 3.0,
    'downloadsToday': 1.0,
    'storageUsedGB': 32.0,
}

# camelCase features that have corresponding snake_case entries in POPULATION_NORMS.
_POPULATED_FEATURES_SNAKE: Dict[str, str] = _CAMEL_TO_SNAKE


def _prior_for_feature(
    feature: str,
    population_norms: Dict[str, Dict[str, float]],
    kappa_0: float,
    alpha_0: float,
) -> FeaturePosterior:
    """Compute NIG prior parameters for a single feature.

    For features with POPULATION_NORMS (via snake_case mapping): use empirical mean and std.
    For others: use fallback defaults with 20% coefficient of variation.
    """
    # Try to find this camelCase feature in population norms (via snake_case mapping)
    snake_key = _CAMEL_TO_SNAKE.get(feature)

    if snake_key and snake_key in population_norms:
        mu_0 = population_norms[snake_key]['mean']
        std_0 = population_norms[snake_key]['std']
    elif feature in _FALLBACK_DEFAULTS:
        mu_0 = _FALLBACK_DEFAULTS[feature]
        std_0 = max(mu_0 * 0.20, 0.01)  # 20% CV, floored
    else:
        # Truly unknown feature — wide prior centred at 1.0
        mu_0 = 1.0
        std_0 = 1.0

    beta_0 = std_0 ** 2 * alpha_0

    return FeaturePosterior(
        mu_0=mu_0,
        kappa_0=kappa_0,
        alpha_0=alpha_0,
        beta_0=beta_0,
        mu_n=mu_0,
        kappa_n=kappa_0,
        alpha_n=alpha_0,
        beta_n=beta_0,
    )


class BayesianBaseline:
    """Bayesian warm-start baseline using NIG conjugate priors.

    Each call to ``update(day_data, day_number)`` produces a
    ``BayesianState`` containing posterior means, standard deviations,
    phase, and confidence — ready for L1 z-score computation.
    """

    def __init__(
        self,
        feature_names: List[str],
        population_norms: Optional[Dict[str, Dict[str, float]]] = None,
        kappa_0: float = 14.0,
        alpha_0: float = 2.0,
    ):
        self.feature_names = feature_names
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.norms = population_norms or _get_population_norms()

        self._posteriors: Dict[str, FeaturePosterior] = {}
        self._state = BayesianState()
        self._initialize_posteriors()

    def _initialize_posteriors(self) -> None:
        for feat in self.feature_names:
            self._posteriors[feat] = _prior_for_feature(
                feat, self.norms, self.kappa_0, self.alpha_0,
            )

    def update(self, day_data: Dict[str, float], day_number: int) -> BayesianState:
        """Incorporate one day of observations and return updated state."""
        # Step 1: update sufficient statistics and posteriors
        for feat in self.feature_names:
            p = self._posteriors[feat]
            value = day_data.get(feat, p.mu_n)  # impute missing with current posterior mean

            p.n_observations += 1
            p.sum_observations += value
            p.sum_sq_observations += value ** 2

            n = p.n_observations
            x_bar = p.sum_observations / n
            SS = p.sum_sq_observations - n * x_bar ** 2

            # NIG conjugate update
            p.kappa_n = p.kappa_0 + n
            p.mu_n = (p.kappa_0 * p.mu_0 + n * x_bar) / p.kappa_n
            p.alpha_n = p.alpha_0 + n / 2.0
            p.beta_n = (
                p.beta_0
                + 0.5 * SS
                + 0.5 * p.kappa_0 * n * (x_bar - p.mu_0) ** 2 / p.kappa_n
            )

        # Step 2: derive effective means and stds
        effective_means: Dict[str, float] = {}
        effective_stds: Dict[str, float] = {}
        feature_confidences: Dict[str, float] = {}

        for feat in self.feature_names:
            p = self._posteriors[feat]
            effective_means[feat] = p.mu_n
            posterior_var = p.beta_n / (p.alpha_n + 1)
            effective_stds[feat] = math.sqrt(max(posterior_var, 0.0)) + 0.05

            var_mu_posterior = p.beta_n / (p.alpha_n * p.kappa_n)
            var_mu_prior = p.beta_0 / (p.alpha_0 * p.kappa_0)
            if var_mu_prior > 0:
                shrinkage = 1.0 - math.sqrt(var_mu_posterior / var_mu_prior)
                feature_confidences[feat] = max(0.0, min(1.0, shrinkage))
            else:
                feature_confidences[feat] = 1.0

        # Step 3: aggregate confidence (feature-weighted)
        total_weight = 0.0
        weighted_sum = 0.0
        for feat in self.feature_names:
            w = FEATURE_META.get(feat, {}).get('weight', 1.0)
            weighted_sum += w * feature_confidences[feat]
            total_weight += w
        confidence_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Step 4: personal weight and phase
        personal_weight = day_number / (self.kappa_0 + day_number)

        if day_number <= 13:
            phase = BaselinePhase.POPULATION_ANCHORED
        elif day_number <= 59:
            phase = BaselinePhase.BLENDED
        else:
            phase = BaselinePhase.IDIOGRAPHIC

        # Step 5: update state
        self._state = BayesianState(
            phase=phase,
            day_number=day_number,
            personal_weight=personal_weight,
            confidence_score=confidence_score,
            feature_posteriors=dict(self._posteriors),
            effective_means=effective_means,
            effective_stds=effective_stds,
            feature_confidences=feature_confidences,
        )
        return self._state

    def get_state(self) -> BayesianState:
        return self._state

    def get_baseline_vector(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (effective_means, effective_stds) for L1 z-score computation."""
        return self._state.effective_means, self._state.effective_stds

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dict."""
        posteriors = {}
        for feat, p in self._posteriors.items():
            posteriors[feat] = {
                'mu_0': p.mu_0, 'kappa_0': p.kappa_0,
                'alpha_0': p.alpha_0, 'beta_0': p.beta_0,
                'mu_n': p.mu_n, 'kappa_n': p.kappa_n,
                'alpha_n': p.alpha_n, 'beta_n': p.beta_n,
                'n_observations': p.n_observations,
                'sum_observations': p.sum_observations,
                'sum_sq_observations': p.sum_sq_observations,
            }
        return {
            'feature_names': self.feature_names,
            'kappa_0': self.kappa_0,
            'alpha_0': self.alpha_0,
            'posteriors': posteriors,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BayesianBaseline':
        """Reconstruct from a serialized dict."""
        obj = cls.__new__(cls)
        obj.feature_names = data['feature_names']
        obj.kappa_0 = data['kappa_0']
        obj.alpha_0 = data['alpha_0']
        obj.norms = _get_population_norms()
        obj._posteriors = {}
        obj._state = BayesianState()

        for feat, pd in data['posteriors'].items():
            obj._posteriors[feat] = FeaturePosterior(
                mu_0=pd['mu_0'], kappa_0=pd['kappa_0'],
                alpha_0=pd['alpha_0'], beta_0=pd['beta_0'],
                mu_n=pd['mu_n'], kappa_n=pd['kappa_n'],
                alpha_n=pd['alpha_n'], beta_n=pd['beta_n'],
                n_observations=pd['n_observations'],
                sum_observations=pd['sum_observations'],
                sum_sq_observations=pd['sum_sq_observations'],
            )
        return obj

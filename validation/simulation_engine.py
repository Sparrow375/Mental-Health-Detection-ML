"""
Simulation Engine — Feeds StudentLife daily feature vectors through the
MHealth System 1 pipeline exactly as the Android app would, day by day.

Supports configurable baseline/monitoring splits:
  - Fixed split (e.g., first 14 days baseline, rest monitoring)
  - Hybrid approach (monitoring days also update baseline via Bayesian updates)
  - Sliding window approaches

For each student, produces:
  - Daily anomaly scores (L1 + effective)
  - Evidence accumulation trace
  - Final depression prediction (based on peak evidence/sustained days)
"""

from __future__ import annotations

import copy
import math
import sys
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add the Python engine to path so we can import the real system1 modules
ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "MHealth - Copy", "app", "src", "main", "python"
)
sys.path.insert(0, ENGINE_PATH)

from system1.data_structures import PersonalityVector, EvidenceState, BayesianState
from system1.scoring.l1_scorer import L1Scorer
from system1.engine.evidence_engine import EvidenceEngine
from system1.baseline.bayesian_baseline import BayesianBaseline
from system1.feature_meta import ALL_L1_FEATURES, FEATURE_META, DEFAULT_THRESHOLDS


# Global cache for PCA and MeanShift baseline parameters to optimize sweeps
CLUSTERING_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}


FEATURE_WEIGHTS = {
    "screenTimeHours": 1.4, "unlockCount": 1.2, "appLaunchCount": 0.9,
    "notificationsToday": 0.8, "socialAppRatio": 1.3,
    "callsPerDay": 1.3, "callDurationMinutes": 1.2, "uniqueContacts": 1.1,
    "conversationFrequency": 0.9,
    "dailyDisplacementKm": 1.5, "locationEntropy": 1.3, "homeTimeRatio": 1.2,
    "wakeTimeHour": 1.4, "sleepTimeHour": 1.3, "sleepDurationHours": 1.6,
    "dailyStepCount": 1.4, "activeMinutes": 1.2,
    "keystrokeSpeed": 1.3, "backspaceRatio": 1.2, "scrollVelocity": 1.1,
    "daylightExposureMinutes": 1.1, "chargeRegularity": 1.2,
    "chargeDurationHours": 0.8,
    "upiTransactionsToday": 1.1, "appUninstallsToday": 0.9, "appInstallsToday": 0.8,
    "calendarEventsToday": 0.9, "mediaCountToday": 0.7, "downloadsToday": 0.6,
    "musicTimeMinutes": 0.9,
}

L1_CLUSTER_FEATURES = [
    "sleepDurationHours", "wakeTimeHour", "sleepTimeHour",
    "dailyDisplacementKm", "locationEntropy",
    "callsPerDay", "conversationFrequency", "screenTimeHours",
    "unlockCount", "socialAppRatio", "dailyStepCount", "chargeRegularity"
]

L2_CLUSTER_FEATURES = [
    "total_sessions", "abandon_rate", "self_open_ratio",
    "deep_session_ratio", "micro_session_ratio", "app_switching_rate",
    "active_hours_span", "avg_session_minutes", "notifications",
]


def synthesize_l2_features(day_data: Dict[str, float]) -> Dict[str, float]:
    """Synthesize Layer 2 features from Layer 1 feature values."""
    app_launches = max(day_data.get("appLaunchCount", 1.0), 1.0)
    unlocks = max(day_data.get("unlockCount", 1.0), 1.0)
    screen_time = day_data.get("screenTimeHours", 2.0)
    notifications = day_data.get("notificationsToday", 10.0)
    
    avg_session = (screen_time * 60.0) / unlocks
    abandon = 0.05 + 0.10 * (unlocks / app_launches)
    abandon = min(max(abandon, 0.0), 1.0)
    
    deep_ratio = 0.1 + 0.3 * min(avg_session, 30.0) / 30.0
    micro_ratio = 0.8 - 0.5 * min(avg_session, 10.0) / 10.0
    
    switching = min(notifications / app_launches, 1.0)
    active_span = max(screen_time + 2.0, 16.0)
    
    return {
        "total_sessions": app_launches,
        "abandon_rate": abandon,
        "self_open_ratio": 0.85,
        "deep_session_ratio": deep_ratio,
        "micro_session_ratio": micro_ratio,
        "app_switching_rate": switching,
        "active_hours_span": active_span,
        "avg_session_minutes": avg_session,
        "notifications": notifications,
    }


def clinical_weighted_pca(data: np.ndarray, feature_weights: List[float], target_variance: float = 0.85) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply clinical weights, then PCA via SVD, capturing >= target_variance cumulative explained variance."""
    W = np.diag(feature_weights)
    weighted = data @ W
    mean = weighted.mean(axis=0)
    centered = weighted - mean
    
    if len(centered) < 2:
        # Fallback for single data point or degenerate variance
        n_comps = max(2, min(5, centered.shape[1]))
        components = np.eye(n_comps, centered.shape[1])
        projected = centered @ components.T
        return projected, components, mean
        
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    variances = S**2
    total_var = np.sum(variances)
    if total_var > 1e-9:
        explained_variance_ratio = variances / total_var
    else:
        explained_variance_ratio = np.ones_like(variances) / len(variances)
        
    cumulative_variance = np.cumsum(explained_variance_ratio)
    k = int(np.argmax(cumulative_variance >= target_variance) + 1)
    n_components = max(2, min(5, k, centered.shape[1]))
    
    components = Vt[:n_components]
    projected = centered @ components.T
    return projected, components, mean


def meanshift_clustering(data: np.ndarray, bandwidth: float = None) -> List[Tuple[int, List[int]]]:
    """Mean-Shift clustering (pure numpy) using 30% quantile for adaptive bandwidth."""
    if len(data) == 0:
        return []
    n = len(data)
    
    if bandwidth is None:
        n_neighbors = int(n * 0.3)
        if n_neighbors < 1:
            n_neighbors = 1
        
        pairwise = np.linalg.norm(data[:, None] - data[None, :], axis=2)
        sorted_dists = np.sort(pairwise, axis=1)
        kth_dists = sorted_dists[:, n_neighbors - 1]
        bandwidth = float(np.median(kth_dists))
    if bandwidth <= 0:
        bandwidth = 1.0
        
    points = data.copy()
    for iteration in range(50):
        shifted = np.zeros_like(points)
        max_shift = 0.0
        for i in range(n):
            dists = np.linalg.norm(data - points[i], axis=1)
            mask = dists <= bandwidth
            if mask.any():
                shifted[i] = data[mask].mean(axis=0)
            else:
                shifted[i] = points[i]
            max_shift = max(max_shift, np.linalg.norm(shifted[i] - points[i]))
        points = shifted
        if max_shift < 1e-6:
            break
            
    cluster_centers = []
    cluster_ids = [-1] * n
    merge_thresh = bandwidth * 0.5
    for i in range(n):
        merged = False
        for c_idx, center in enumerate(cluster_centers):
            if np.linalg.norm(points[i] - center) < merge_thresh:
                cluster_ids[i] = c_idx
                merged = True
                break
        if not merged:
            cluster_ids[i] = len(cluster_centers)
            cluster_centers.append(points[i].copy())
            
    clusters = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)
        
    return [(cid, indices) for cid, indices in sorted(clusters.items())]


class L2Scorer:
    """Stateful L2 Scorer that tracks feature history for velocity computation."""
    
    def __init__(self, baseline_means: Dict[str, float], baseline_stds: Dict[str, float], history_window: int = 7):
        self.baseline_means = baseline_means
        self.baseline_stds = baseline_stds
        self.feature_names = list(baseline_means.keys())
        self.history_window = history_window
        self.feature_history = {feat: deque(maxlen=history_window) for feat in self.feature_names}
        
    def calculate_deviation_magnitude(self, current_data: Dict[str, float]) -> Dict[str, float]:
        deviations = {}
        for feat in self.feature_names:
            mean = self.baseline_means.get(feat, 0.0)
            std = self.baseline_stds.get(feat, 1.0)
            val = current_data.get(feat, mean)
            z = (val - mean) / std
            deviations[feat] = float(np.clip(z, -4.0, 4.0))
        return deviations
        
    def calculate_deviation_velocity(self, current_data: Dict[str, float]) -> Dict[str, float]:
        alpha = 0.4
        velocities = {}
        for feat in self.feature_names:
            val = current_data.get(feat, self.baseline_means.get(feat, 0.0))
            self.feature_history[feat].append(val)
            
        for feat in self.feature_names:
            history = list(self.feature_history[feat])
            if len(history) < 2:
                velocities[feat] = 0.0
            else:
                ewma = history[0]
                ewma_values = [ewma]
                for val in history[1:]:
                    ewma = alpha * val + (1.0 - alpha) * ewma
                    ewma_values.append(ewma)
                slope = (ewma_values[-1] - ewma_values[0]) / len(ewma_values)
                mean = self.baseline_means.get(feat, 0.0)
                if mean > 0:
                    velocities[feat] = slope / mean
                else:
                    velocities[feat] = 0.0
        return velocities
        
    def calculate_anomaly_score(self, deviations: Dict[str, float], velocities: Dict[str, float]) -> float:
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


# ============================================================================
# Configuration Presets
# ============================================================================

class SimulationConfig:
    """Configuration for one simulation trial."""

    def __init__(
        self,
        name: str = "default",
        baseline_days: int = 14,
        hybrid_mode: bool = True,           # also feed monitoring days into baseline
        anomaly_threshold: float = 0.38,
        evidence_threshold: float = 2.0,
        sustained_threshold: int = 5,
        evidence_decay: float = 0.88,
        evidence_compounding: float = 0.15,
        magnitude_weight: float = 0.7,
        velocity_weight: float = 0.3,
        ewma_alpha: float = 0.4,
        deviation_ceiling: float = 4.0,
        l2_modifier_enabled: bool = True,    # Symmetrical dual-layer is now active by default
        min_data_days: int = 10,             # minimum days to consider student
        depression_cutoff: int = 10,         # PHQ-9 cutoff
        # Evidence-based prediction thresholds
        # Evidence-based prediction thresholds
        prediction_evidence_threshold: float = 40.0,
        prediction_sustained_threshold: int = 25,
        prediction_score_threshold: float = 0.50,
        prediction_strategy: str = "mean_anomaly",
        # Adaptive features
        use_bayesian_baseline: bool = True,
        kappa_0: float = 14.0,
        alpha_0: float = 2.0,
        # Geometric mean exponents
        l1_exponent: float = 0.6,
        l2_exponent: float = 0.4,
        # Compactness parameters
        compactness_N: int = 7,
        compactness_threshold: float = 1.2,
        clinical_overrides_enabled: bool = False,
    ):
        self.name = name
        self.baseline_days = baseline_days
        self.hybrid_mode = hybrid_mode
        self.anomaly_threshold = anomaly_threshold
        self.evidence_threshold = evidence_threshold
        self.sustained_threshold = sustained_threshold
        self.evidence_decay = evidence_decay
        self.evidence_compounding = evidence_compounding
        self.magnitude_weight = magnitude_weight
        self.velocity_weight = velocity_weight
        self.ewma_alpha = ewma_alpha
        self.deviation_ceiling = deviation_ceiling
        self.l2_modifier_enabled = l2_modifier_enabled
        self.min_data_days = min_data_days
        self.depression_cutoff = depression_cutoff
        self.prediction_evidence_threshold = prediction_evidence_threshold
        self.prediction_sustained_threshold = prediction_sustained_threshold
        self.prediction_score_threshold = prediction_score_threshold
        self.prediction_strategy = prediction_strategy
        self.use_bayesian_baseline = use_bayesian_baseline
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.l1_exponent = l1_exponent
        self.l2_exponent = l2_exponent
        self.compactness_N = compactness_N
        self.compactness_threshold = compactness_threshold
        self.clinical_overrides_enabled = clinical_overrides_enabled


# ============================================================================
# Student Simulation Result
# ============================================================================

class StudentResult:
    """Results from simulating one student through the system."""

    def __init__(self, uid: str, config: SimulationConfig):
        self.uid = uid
        self.config = config
        self.dates: List[str] = []
        self.daily_scores: List[float] = []          # L1 anomaly scores
        self.effective_scores: List[float] = []      # After L2 modifier
        self.evidence_trace: List[float] = []
        self.sustained_days_trace: List[int] = []
        self.deviations_trace: List[Dict[str, float]] = []
        self.flagged_features_trace: List[List[str]] = []
        self.baseline_phase_trace: List[str] = []

        # Summary metrics
        self.n_baseline_days: int = 0
        self.n_monitoring_days: int = 0
        self.peak_evidence: float = 0.0
        self.peak_sustained: int = 0
        self.peak_anomaly_score: float = 0.0
        self.mean_anomaly_score: float = 0.0
        self.median_anomaly_score: float = 0.0
        self.anomaly_days_count: int = 0
        self.anomaly_day_ratio: float = 0.0

        # Prediction
        self.predicted_depressed: bool = False
        self.prediction_confidence: float = 0.0

        # Ground truth
        self.phq9_pre: Optional[int] = None
        self.phq9_post: Optional[int] = None
        self.depressed_pre: Optional[bool] = None
        self.depressed_post: Optional[bool] = None

    def compute_summary(self):
        """Compute summary statistics from traces."""
        if not self.effective_scores:
            return

        monitoring_scores = self.effective_scores[self.n_baseline_days:]
        if not monitoring_scores:
            return

        self.peak_anomaly_score = max(monitoring_scores)
        self.mean_anomaly_score = np.mean(monitoring_scores)
        self.median_anomaly_score = np.median(monitoring_scores)
        self.anomaly_days_count = sum(1 for s in monitoring_scores
                                       if s > self.config.anomaly_threshold)
        self.anomaly_day_ratio = self.anomaly_days_count / len(monitoring_scores)

        if self.evidence_trace:
            evidence_monitoring = self.evidence_trace[self.n_baseline_days:]
            if evidence_monitoring:
                self.peak_evidence = max(evidence_monitoring)

        if self.sustained_days_trace:
            sustained_monitoring = self.sustained_days_trace[self.n_baseline_days:]
            if sustained_monitoring:
                self.peak_sustained = max(sustained_monitoring)


# ============================================================================
# Simulation Engine
# ============================================================================

class SimulationEngine:
    """
    Runs a student's daily data through the MHealth System 1 pipeline.

    Simulates the exact same flow as the Android app:
    1. Baseline phase: accumulate N days, build L1 and L2 baselines
    2. Run PCA and Mean-Shift clustering independently on L1 and L2 baseline matrices
    3. Monitoring phase: score each day symmetrically, apply Radial Proximity Decay
    4. Fuse layer scores using Weighted Geometric Mean
    5. Apply the Variance-Bounded Compactness Test (Post-Filter)
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def simulate_student(
        self,
        uid: str,
        daily_data: Dict[str, Dict[str, float]],
        phq9_scores: Optional[Dict[str, int]] = None,
        healthy_pop_means_l1: Optional[Dict[str, float]] = None,
        healthy_pop_stds_l1: Optional[Dict[str, float]] = None,
        healthy_pop_means_l2: Optional[Dict[str, float]] = None,
        healthy_pop_stds_l2: Optional[Dict[str, float]] = None,
    ) -> Optional[StudentResult]:
        """
        Run full simulation for one student.
        """
        sorted_dates = sorted(daily_data.keys())
        if len(sorted_dates) < self.config.min_data_days:
            return None

        result = StudentResult(uid, self.config)
        result.dates = sorted_dates

        if phq9_scores:
            result.phq9_pre = phq9_scores.get('pre')
            result.phq9_post = phq9_scores.get('post')
            result.depressed_pre = (result.phq9_pre is not None and
                                     result.phq9_pre >= self.config.depression_cutoff)
            result.depressed_post = (result.phq9_post is not None and
                                      result.phq9_post >= self.config.depression_cutoff)
        # Determine baseline vs monitoring split
        n_days = len(sorted_dates)
        n_baseline = min(self.config.baseline_days, n_days - 1)
        result.n_baseline_days = n_baseline
        result.n_monitoring_days = n_days - n_baseline

        cache_key = (uid, n_baseline)
        if cache_key in CLUSTERING_CACHE:
            c_data = CLUSTERING_CACHE[cache_key]
            baseline_means_l1 = c_data["baseline_means_l1"]
            baseline_stds_l1 = c_data["baseline_stds_l1"]
            baseline_means_l2 = c_data["baseline_means_l2"]
            baseline_stds_l2 = c_data["baseline_stds_l2"]
            baseline_pv = c_data["baseline_pv"]
            l1_means_baseline = c_data["l1_means_baseline"]
            l1_stds_baseline_safe = c_data["l1_stds_baseline_safe"]
            l1_weights = c_data["l1_weights"]
            pca_mean_l1 = c_data["pca_mean_l1"]
            components_l1 = c_data["components_l1"]
            l1_clusters = copy.deepcopy(c_data["l1_clusters"])
            l2_means_baseline = c_data["l2_means_baseline"]
            l2_stds_baseline_safe = c_data["l2_stds_baseline_safe"]
            l2_weights = c_data["l2_weights"]
            pca_mean_l2 = c_data["pca_mean_l2"]
            components_l2 = c_data["components_l2"]
            l2_clusters = copy.deepcopy(c_data["l2_clusters"])
            use_population_baseline = c_data["use_population_baseline"]
            available_features = c_data["available_features"]
        else:
            # ── Phase 1: Build baseline from first N days ──
            available_features = list(ALL_L1_FEATURES)

            # Compute baseline means + stds from own days first
            baseline_values_l1 = defaultdict(list)
            baseline_values_l2 = defaultdict(list)

            for d in sorted_dates[:n_baseline]:
                # L1 features
                for feat in available_features:
                    val = daily_data[d].get(feat, 0.0)
                    baseline_values_l1[feat].append(val)
                
                # L2 features
                l2_feats = synthesize_l2_features(daily_data[d])
                for feat, val in l2_feats.items():
                    baseline_values_l2[feat].append(val)

            # Check for baseline contamination:
            # Self-report gate: pre-study PHQ-9 >= 10.
            use_population_baseline = False
            if phq9_scores and phq9_scores.get('pre', 0) >= 10:
                use_population_baseline = True

            # Compute baseline L1 & L2 means/stds (use own or healthy fallback)
            baseline_means_l1 = {}
            baseline_stds_l1 = {}
            for feat in available_features:
                if use_population_baseline and healthy_pop_means_l1 and feat in healthy_pop_means_l1:
                    baseline_means_l1[feat] = healthy_pop_means_l1[feat]
                    baseline_stds_l1[feat] = healthy_pop_stds_l1[feat]
                else:
                    vals = baseline_values_l1[feat]
                    baseline_means_l1[feat] = np.mean(vals) if vals else 0.0
                    std = np.std(vals) if vals else 1.0
                    baseline_stds_l1[feat] = max(std, 0.05)

            baseline_means_l2 = {}
            baseline_stds_l2 = {}
            for feat in L2_CLUSTER_FEATURES:
                if use_population_baseline and healthy_pop_means_l2 and feat in healthy_pop_means_l2:
                    baseline_means_l2[feat] = healthy_pop_means_l2[feat]
                    baseline_stds_l2[feat] = healthy_pop_stds_l2[feat]
                else:
                    vals = baseline_values_l2[feat]
                    baseline_means_l2[feat] = np.mean(vals) if vals else 0.0
                    std = np.std(vals) if vals else 1.0
                    baseline_stds_l2[feat] = max(std, 0.05)

            # Create PersonalityVector baseline for L1Scorer
            baseline_pv = PersonalityVector.from_dict(baseline_means_l1, baseline_stds_l1)

            # ── PCA & Mean-Shift Clustering Discovery ──
            # L1 Clustering
            l1_matrix = []
            for d in sorted_dates[:n_baseline]:
                vec = [float(baseline_means_l1.get(feat, daily_data[d].get(feat, 0.0))) for feat in L1_CLUSTER_FEATURES]
                l1_matrix.append(vec)
            l1_matrix = np.array(l1_matrix)
            
            l1_means_baseline = np.mean(l1_matrix, axis=0)
            l1_stds_baseline = np.std(l1_matrix, axis=0)
            l1_stds_baseline_safe = np.where(l1_stds_baseline > 1e-9, l1_stds_baseline, 1.0)
            l1_matrix_norm = (l1_matrix - l1_means_baseline) / l1_stds_baseline_safe
            
            l1_weights = [FEATURE_WEIGHTS.get(f, 1.0) for f in L1_CLUSTER_FEATURES]
            proj_l1, components_l1, pca_mean_l1 = clinical_weighted_pca(l1_matrix_norm, l1_weights, target_variance=0.85)
            
            l1_clusters_raw = meanshift_clustering(proj_l1)
            l1_clusters = []
            if not l1_clusters_raw:
                centroid_l1 = np.mean(proj_l1, axis=0)
                radius_l1 = float(np.max(np.linalg.norm(proj_l1 - centroid_l1, axis=1))) if len(proj_l1) > 1 else 1.25
                l1_clusters.append({"centroid": centroid_l1, "radius": max(radius_l1, 0.5)})
            else:
                for cid, indices in l1_clusters_raw:
                    members = proj_l1[indices]
                    centroid_l1 = np.mean(members, axis=0)
                    radius_l1 = float(np.max(np.linalg.norm(members - centroid_l1, axis=1))) if len(members) > 1 else 0.5
                    l1_clusters.append({"centroid": centroid_l1, "radius": max(radius_l1, 0.5)})

            # L2 Clustering
            l2_matrix = []
            for d in sorted_dates[:n_baseline]:
                l2_feats = synthesize_l2_features(daily_data[d])
                vec = [float(baseline_means_l2.get(feat, l2_feats.get(feat, 0.0))) for feat in L2_CLUSTER_FEATURES]
                l2_matrix.append(vec)
            l2_matrix = np.array(l2_matrix)
            
            l2_means_baseline = np.mean(l2_matrix, axis=0)
            l2_stds_baseline = np.std(l2_matrix, axis=0)
            l2_stds_baseline_safe = np.where(l2_stds_baseline > 1e-9, l2_stds_baseline, 1.0)
            l2_matrix_norm = (l2_matrix - l2_means_baseline) / l2_stds_baseline_safe
            
            l2_weights = [1.0] * len(L2_CLUSTER_FEATURES)
            proj_l2, components_l2, pca_mean_l2 = clinical_weighted_pca(l2_matrix_norm, l2_weights, target_variance=0.85)
            
            l2_clusters_raw = meanshift_clustering(proj_l2)
            l2_clusters = []
            if not l2_clusters_raw:
                centroid_l2 = np.mean(proj_l2, axis=0)
                radius_l2 = float(np.max(np.linalg.norm(proj_l2 - centroid_l2, axis=1))) if len(proj_l2) > 1 else 1.25
                l2_clusters.append({"centroid": centroid_l2, "radius": max(radius_l2, 0.5)})
            else:
                for cid, indices in l2_clusters_raw:
                    members = proj_l2[indices]
                    centroid_l2 = np.mean(members, axis=0)
                    radius_l2 = float(np.max(np.linalg.norm(members - centroid_l2, axis=1))) if len(members) > 1 else 0.5
                    l2_clusters.append({"centroid": centroid_l2, "radius": max(radius_l2, 0.5)})

            # Save to cache
            CLUSTERING_CACHE[cache_key] = {
                "baseline_means_l1": baseline_means_l1,
                "baseline_stds_l1": baseline_stds_l1,
                "baseline_means_l2": baseline_means_l2,
                "baseline_stds_l2": baseline_stds_l2,
                "baseline_pv": baseline_pv,
                "l1_means_baseline": l1_means_baseline,
                "l1_stds_baseline_safe": l1_stds_baseline_safe,
                "l1_weights": l1_weights,
                "pca_mean_l1": pca_mean_l1,
                "components_l1": components_l1,
                "l1_clusters": copy.deepcopy(l1_clusters),
                "l2_means_baseline": l2_means_baseline,
                "l2_stds_baseline_safe": l2_stds_baseline_safe,
                "l2_weights": l2_weights,
                "pca_mean_l2": pca_mean_l2,
                "components_l2": components_l2,
                "l2_clusters": copy.deepcopy(l2_clusters),
                "use_population_baseline": use_population_baseline,
                "available_features": available_features,
            }

        # ── Initialize scorers ──
        scorer_l1 = L1Scorer(baseline_pv, history_window=7)
        scorer_l2 = L2Scorer(baseline_means_l2, baseline_stds_l2, history_window=7)

        # Initialize Bayesian baseline for L1 (only if own baseline is healthy)
        bayesian = None
        if self.config.use_bayesian_baseline and not use_population_baseline:
            bayesian = BayesianBaseline(
                available_features,
                kappa_0=self.config.kappa_0,
                alpha_0=self.config.alpha_0,
            )
            for day_idx, d in enumerate(sorted_dates[:n_baseline]):
                day_data = {feat: daily_data[d].get(feat, 0.0) for feat in available_features}
                bay_state = bayesian.update(day_data, day_idx + 1)
            scorer_l1.update_bayesian_state(bayesian.get_state())

        # Initialize evidence engine
        evidence_engine = EvidenceEngine(thresholds={
            'ANOMALY_SCORE_THRESHOLD': self.config.anomaly_threshold,
            'EVIDENCE_DECAY_RATE': self.config.evidence_decay,
            'EVIDENCE_COMPOUNDING': self.config.evidence_compounding,
        })

        feature_ceilings = {feat: self.config.deviation_ceiling for feat in available_features}

        # Compactness Post-Filter States
        consecutive_anomalous_days = 0
        anomalous_history_l1 = []
        anomalous_history_l2 = []
        anomalous_days_evidence = []

        # ── Phase 2: Score ALL days (baseline + monitoring) ──
        for day_idx, date in enumerate(sorted_dates):
            day_data = {feat: daily_data[date].get(feat, 0.0) for feat in available_features}
            today_l2 = synthesize_l2_features(daily_data[date])

            # 1. Compute Layer 1 Score
            deviations_l1 = scorer_l1.calculate_deviation_magnitude(day_data, feature_ceilings)
            velocities_l1 = scorer_l1.calculate_deviation_velocity(day_data)
            raw_l1 = scorer_l1.calculate_anomaly_score(deviations_l1, velocities_l1)

            # L1 PCA Projection
            l1_raw_vec = np.array([float(day_data.get(feat, baseline_means_l1.get(feat, 0.0))) for feat in L1_CLUSTER_FEATURES])
            l1_z = (l1_raw_vec - l1_means_baseline) / l1_stds_baseline_safe
            l1_w = l1_z * l1_weights
            projected_l1 = (l1_w - pca_mean_l1) @ components_l1.T

            # L1 Radial Proximity Decay
            best_dist_l1 = float('inf')
            best_radius_l1 = 1.0
            for c in l1_clusters:
                d = np.linalg.norm(projected_l1 - c["centroid"])
                if d < best_dist_l1:
                    best_dist_l1 = d
                    best_radius_l1 = c["radius"]
            
            if best_dist_l1 <= best_radius_l1 * 1.5:
                coherence_l1 = max(0.0, 1.0 - best_dist_l1 / (best_radius_l1 * 1.5))
                l1_score_adjusted = raw_l1 * (1.0 - coherence_l1 * 0.85)
            else:
                l1_score_adjusted = raw_l1

            # 2. Compute Layer 2 Score
            deviations_l2 = scorer_l2.calculate_deviation_magnitude(today_l2)
            velocities_l2 = scorer_l2.calculate_deviation_velocity(today_l2)
            raw_l2 = scorer_l2.calculate_anomaly_score(deviations_l2, velocities_l2)

            # L2 PCA Projection
            l2_raw_vec = np.array([float(today_l2.get(feat, baseline_means_l2.get(feat, 0.0))) for feat in L2_CLUSTER_FEATURES])
            l2_z = (l2_raw_vec - l2_means_baseline) / l2_stds_baseline_safe
            projected_l2 = (l2_z - pca_mean_l2) @ components_l2.T

            # L2 Radial Proximity Decay
            best_dist_l2 = float('inf')
            best_radius_l2 = 1.0
            for c in l2_clusters:
                d = np.linalg.norm(projected_l2 - c["centroid"])
                if d < best_dist_l2:
                    best_dist_l2 = d
                    best_radius_l2 = c["radius"]
            
            if best_dist_l2 <= best_radius_l2 * 1.5:
                coherence_l2 = max(0.0, 1.0 - best_dist_l2 / (best_radius_l2 * 1.5))
                l2_score_adjusted = raw_l2 * (1.0 - coherence_l2 * 0.85)
            else:
                l2_score_adjusted = raw_l2

            # 3. Fuse Scores using Weighted Geometric Mean
            effective_score = (l1_score_adjusted ** self.config.l1_exponent) * (l2_score_adjusted ** self.config.l2_exponent)

            # Breadth (L1 z-score count > 1.5)
            flagged = [f for f, d in deviations_l1.items() if abs(d) > 1.5]
            breadth = len(flagged)

            # 4. Evidence Accumulation & Compactness Post-Filter
            if day_idx >= n_baseline:
                ev_state = evidence_engine.update(effective_score, breadth)
                
                # Track anomalous days
                if effective_score > self.config.anomaly_threshold:
                    consecutive_anomalous_days += 1
                    anomalous_history_l1.append(projected_l1)
                    anomalous_history_l2.append(projected_l2)
                    
                    ev_increment = effective_score * (1.0 + (consecutive_anomalous_days - 1) * self.config.evidence_compounding)
                    anomalous_days_evidence.append(ev_increment)
                    
                    if consecutive_anomalous_days >= self.config.compactness_N:
                        # Compactness check on PCA points
                        pts_l1 = np.array(anomalous_history_l1)
                        pts_l2 = np.array(anomalous_history_l2)
                        
                        max_dist_l1 = 0.0
                        for i in range(len(pts_l1)):
                            for j in range(i + 1, len(pts_l1)):
                                max_dist_l1 = max(max_dist_l1, np.linalg.norm(pts_l1[i] - pts_l1[j]))
                                
                        max_dist_l2 = 0.0
                        for i in range(len(pts_l2)):
                            for j in range(i + 1, len(pts_l2)):
                                max_dist_l2 = max(max_dist_l2, np.linalg.norm(pts_l2[i] - pts_l2[j]))
                                
                        avg_radius_l1 = np.mean([c["radius"] for c in l1_clusters])
                        avg_radius_l2 = np.mean([c["radius"] for c in l2_clusters])
                        
                        if max_dist_l1 < avg_radius_l1 * self.config.compactness_threshold and \
                           max_dist_l2 < avg_radius_l2 * self.config.compactness_threshold:
                            # Promote as New Healthy Lifestyle Archetype Cluster!
                            l1_clusters.append({
                                "centroid": np.mean(pts_l1, axis=0),
                                "radius": max(max_dist_l1, 0.5)
                            })
                            l2_clusters.append({
                                "centroid": np.mean(pts_l2, axis=0),
                                "radius": max(max_dist_l2, 0.5)
                            })
                            
                            # Complete reset of stress evidence
                            evidence_engine.state.evidence_accumulated = 0.0
                            evidence_engine.state.sustained_deviation_days = 0
                            
                            # Reset tracking states
                            consecutive_anomalous_days = 0
                            anomalous_history_l1 = []
                            anomalous_history_l2 = []
                            anomalous_days_evidence = []
                else:
                    consecutive_anomalous_days = 0
                    anomalous_history_l1 = []
                    anomalous_history_l2 = []
                    anomalous_days_evidence = []
            else:
                ev_state = evidence_engine.get_state()

            # Hybrid mode: update baseline during monitoring (only if own baseline is healthy)
            if self.config.hybrid_mode and bayesian and day_idx >= n_baseline and not use_population_baseline:
                bay_state = bayesian.update(day_data, day_idx + 1)
                scorer_l1.update_bayesian_state(bay_state)

            # Record traces
            result.daily_scores.append(raw_l1)
            result.effective_scores.append(effective_score)
            result.evidence_trace.append(ev_state.evidence_accumulated)
            result.sustained_days_trace.append(ev_state.sustained_deviation_days)
            result.deviations_trace.append(deviations_l1)
            result.flagged_features_trace.append(flagged)

            if bayesian:
                result.baseline_phase_trace.append(bayesian.get_state().phase.value)
            else:
                result.baseline_phase_trace.append("fixed")

        # ── Phase 3: Make prediction ──
        result.compute_summary()

        # Support different prediction strategies
        strategy = self.config.prediction_strategy
        
        if strategy == "mean_anomaly":
            result.predicted_depressed = result.mean_anomaly_score >= self.config.prediction_score_threshold
            result.prediction_confidence = min(result.mean_anomaly_score, 1.0)
        elif strategy == "peak_evidence":
            result.predicted_depressed = result.peak_evidence >= self.config.prediction_evidence_threshold
            result.prediction_confidence = min(result.peak_evidence / 100.0, 1.0)
        elif strategy == "sustained_days":
            result.predicted_depressed = result.peak_sustained >= self.config.prediction_sustained_threshold
            result.prediction_confidence = min(result.peak_sustained / 30.0, 1.0)
        elif strategy == "anomaly_ratio":
            ratio_thresh = self.config.prediction_score_threshold if self.config.prediction_score_threshold <= 1.0 else 0.40
            result.predicted_depressed = result.anomaly_day_ratio >= ratio_thresh
            result.prediction_confidence = result.anomaly_day_ratio
        elif strategy == "idiographic_anomaly":
            n_baseline = result.n_baseline_days
            baseline_scores = result.effective_scores[:n_baseline]
            mu_base = np.mean(baseline_scores) if baseline_scores else 0.0
            std_base = np.std(baseline_scores) if baseline_scores else 0.05
            std_base_safe = max(std_base, 0.02)
            
            # prediction_score_threshold acts as the standard deviation multiplier (k)
            k = self.config.prediction_score_threshold
            thresh_id = mu_base + k * std_base_safe
            
            monitoring_scores = result.effective_scores[n_baseline:]
            anomaly_days_id = sum(1 for s in monitoring_scores if s > thresh_id)
            ratio_id = anomaly_days_id / len(monitoring_scores) if monitoring_scores else 0.0
            
            ratio_thresh = self.config.prediction_evidence_threshold
            if ratio_thresh > 1.0:
                ratio_thresh = ratio_thresh / 100.0  # Convert e.g. 25.0 -> 0.25
                
            result.predicted_depressed = ratio_id >= ratio_thresh
            result.prediction_confidence = ratio_id
        else:  # "multi_signal"
            # Multi-signal prediction using evidence, sustained days, and score patterns
            prediction_signals = 0
            confidence_sum = 0.0

            # Signal 1: Peak evidence exceeds threshold
            if result.peak_evidence >= self.config.prediction_evidence_threshold:
                prediction_signals += 1
                confidence_sum += min(result.peak_evidence / 5.0, 1.0)

            # Signal 2: Sustained deviation days exceed threshold
            if result.peak_sustained >= self.config.prediction_sustained_threshold:
                prediction_signals += 1
                confidence_sum += min(result.peak_sustained / 10.0, 1.0)

            # Signal 3: High anomaly day ratio
            if result.anomaly_day_ratio > 0.3:
                prediction_signals += 1
                confidence_sum += result.anomaly_day_ratio

            # Signal 4: Mean anomaly score is elevated
            if result.mean_anomaly_score > self.config.prediction_score_threshold:
                prediction_signals += 1
                confidence_sum += min(result.mean_anomaly_score, 1.0)

            # Require at least 2 concordant signals
            result.predicted_depressed = prediction_signals >= 2
            result.prediction_confidence = confidence_sum / max(prediction_signals, 1)

        # Layer clinical overrides if enabled
        if self.config.clinical_overrides_enabled:
            monitoring_traces = result.deviations_trace[n_baseline:]
            if monitoring_traces:
                # Helper to get average z-score of a feature
                def get_mean_z(feat):
                    vals = [t[feat] for t in monitoring_traces if feat in t]
                    return np.mean(vals) if vals else 0.0

                # 1. Social Withdrawal Override (Severe drop in comm, or comm + mobility)
                comm_drop = False
                for feat in ["callsPerDay", "uniqueContacts", "socialAppRatio"]:
                    if get_mean_z(feat) < -1.5:
                        comm_drop = True

                comm_mobility_collapse = False
                if get_mean_z("callsPerDay") < -1.2:
                    for feat in ["dailyStepCount", "dailyDisplacementKm"]:
                        if get_mean_z(feat) < -1.2:
                            comm_mobility_collapse = True

                # 2. Circadian Disorganization Override (Charging regularly or sleep collapse)
                charge_drop = get_mean_z("chargeRegularity") < -1.5
                sleep_drop = get_mean_z("sleepDurationHours") < -1.5

                if comm_drop or comm_mobility_collapse or charge_drop or sleep_drop:
                    result.predicted_depressed = True
                    result.prediction_confidence = max(result.prediction_confidence, 0.85)

        return result

    def simulate_all(
        self,
        all_data: Dict[str, Dict[str, Dict[str, float]]],
        phq9_scores: Dict[str, Dict[str, int]],
    ) -> List[StudentResult]:
        """
        Simulate all students and return results.
        """
        # Compute StudentLife healthy population norms for L1 and L2
        healthy_baseline_vals_l1 = defaultdict(list)
        healthy_baseline_vals_l2 = defaultdict(list)

        for uid, daily_data in all_data.items():
            pre_score = phq9_scores.get(uid, {}).get('pre', 0) if phq9_scores else 0
            if pre_score < 10:
                sorted_dates = sorted(daily_data.keys())
                n_baseline = min(self.config.baseline_days, len(sorted_dates) - 1)
                for d in sorted_dates[:n_baseline]:
                    # Layer 1
                    for feat in ALL_L1_FEATURES:
                        if feat in daily_data[d]:
                            healthy_baseline_vals_l1[feat].append(daily_data[d][feat])
                    # Layer 2
                    l2_feats = synthesize_l2_features(daily_data[d])
                    for feat, val in l2_feats.items():
                        healthy_baseline_vals_l2[feat].append(val)

        # Compute empirical L1 population mean and std
        healthy_pop_means_l1 = {}
        healthy_pop_stds_l1 = {}
        for feat in ALL_L1_FEATURES:
            vals = healthy_baseline_vals_l1[feat]
            healthy_pop_means_l1[feat] = np.mean(vals) if vals else 0.0
            std = np.std(vals) if vals else 1.0
            healthy_pop_stds_l1[feat] = max(std, 0.05)

        # Compute empirical L2 population mean and std
        healthy_pop_means_l2 = {}
        healthy_pop_stds_l2 = {}
        for feat in L2_CLUSTER_FEATURES:
            vals = healthy_baseline_vals_l2[feat]
            healthy_pop_means_l2[feat] = np.mean(vals) if vals else 0.0
            std = np.std(vals) if vals else 1.0
            healthy_pop_stds_l2[feat] = max(std, 0.05)

        results = []
        for uid in sorted(all_data.keys()):
            phq = phq9_scores.get(uid)
            result = self.simulate_student(
                uid, all_data[uid], phq,
                healthy_pop_means_l1=healthy_pop_means_l1,
                healthy_pop_stds_l1=healthy_pop_stds_l1,
                healthy_pop_means_l2=healthy_pop_means_l2,
                healthy_pop_stds_l2=healthy_pop_stds_l2
            )
            if result is not None:
                results.append(result)
        return results


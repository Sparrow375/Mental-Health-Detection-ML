"""
L1 Context Clustering — DBSCAN with Mahalanobis distance.
Discovers behavioral archetypes from baseline daily vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from ..data_structures import AnchorCluster
from ..feature_meta import L1_CLUSTER_FEATURES, THRESHOLDS


def _normalize_daily_vectors(
    vectors: np.ndarray,
    fit: bool = True,
    mins: Optional[np.ndarray] = None,
    maxs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize each feature to [0, 1] using person's own min/max."""
    if fit:
        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)
        # Avoid division by zero
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normalized = (vectors - mins) / ranges
        return normalized, mins, maxs
    else:
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normalized = (vectors - mins) / ranges
        return normalized, mins, maxs


def _compute_cov_inv(vectors: np.ndarray) -> np.ndarray:
    """Compute inverse covariance matrix for Mahalanobis distance."""
    if vectors.shape[0] < vectors.shape[1]:
        # Not enough samples; use identity
        return np.eye(vectors.shape[1])
    try:
        cov = np.cov(vectors.T)
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularize
        cov_inv = np.linalg.inv(cov)
        return cov_inv
    except np.linalg.LinAlgError:
        return np.eye(vectors.shape[1])


def _find_epsilon(normalized_vectors: np.ndarray, cov_inv: np.ndarray, k: int = 3) -> float:
    """Determine DBSCAN epsilon from k-distance graph elbow."""
    from scipy.spatial.distance import pdist, squareform

    # Compute pairwise Mahalanobis distances
    n = len(normalized_vectors)
    if n < k + 1:
        return 1.0

    # Use Euclidean with normalized data as approximation if too few points
    dist_matrix = squareform(pdist(normalized_vectors, metric='mahalanobis', VI=cov_inv))

    # Sort distances to k-th nearest neighbor for each point
    k_distances = np.sort(dist_matrix, axis=1)[:, k]
    k_distances.sort()

    # Find elbow: point of maximum curvature
    if len(k_distances) < 3:
        return float(np.median(k_distances))

    # Simple elbow detection: second derivative
    diffs = np.diff(k_distances)
    if len(diffs) < 2:
        return float(np.median(k_distances))

    second_diff = np.diff(diffs)
    elbow_idx = np.argmax(np.abs(second_diff)) + 1
    epsilon = float(k_distances[min(elbow_idx, len(k_distances) - 1)])

    # Ensure reasonable bounds
    epsilon = max(0.1, min(epsilon, 5.0))
    return epsilon


def cluster_baseline_days(
    daily_features: List[Dict[str, float]],
    dates: List[str],
) -> Tuple[List[AnchorCluster], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Run DBSCAN on baseline daily feature vectors.

    Returns:
        - List of AnchorCluster objects
        - Covariance inverse matrix
        - Feature mins for normalization
        - Feature maxs for normalization
        - List of outlier day vectors (noise points)
    """
    min_samples = THRESHOLDS["DBSCAN_MIN_SAMPLES"]

    # Build feature matrix using L1_CLUSTER_FEATURES
    vectors = []
    valid_dates = []
    for i, day_dict in enumerate(daily_features):
        vec = []
        valid = True
        for feat in L1_CLUSTER_FEATURES:
            val = day_dict.get(feat, None)
            if val is None or np.isnan(val) if val is not None else True:
                valid = False
                break
            vec.append(val)
        if valid:
            vectors.append(vec)
            valid_dates.append(dates[i])

    if len(vectors) < min_samples:
        # Not enough data for clustering; return single cluster
        vectors_arr = np.array(vectors) if vectors else np.zeros((1, len(L1_CLUSTER_FEATURES)))
        cluster = AnchorCluster(
            cluster_id=0,
            centroid=np.mean(vectors_arr, axis=0),
            radius=0.0,
            member_count=len(vectors),
            member_dates=valid_dates,
        )
        identity = np.eye(len(L1_CLUSTER_FEATURES))
        return [cluster], identity, np.zeros(len(L1_CLUSTER_FEATURES)), np.ones(len(L1_CLUSTER_FEATURES)), []

    vectors_arr = np.array(vectors)

    # Normalize to [0, 1]
    normalized, mins, maxs = _normalize_daily_vectors(vectors_arr, fit=True)

    # Compute Mahalanobis covariance inverse
    cov_inv = _compute_cov_inv(normalized)

    # Find epsilon
    epsilon = _find_epsilon(normalized, cov_inv, k=min_samples)

    # Run DBSCAN with Mahalanobis distance
    try:
        dist_matrix = pairwise_distances(normalized, metric='mahalanobis', VI=cov_inv)
        db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')
        labels = db.fit_predict(dist_matrix)
    except Exception:
        # Fallback to Euclidean
        db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
        labels = db.fit_predict(normalized)

    # Build anchor clusters
    clusters = []
    outlier_vectors = []
    unique_labels = set(labels) - {-1}  # Remove noise label

    for cluster_id in sorted(unique_labels):
        member_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        member_dates_cluster = [valid_dates[i] for i in member_indices]
        member_vectors = normalized[member_indices]

        centroid = np.mean(member_vectors, axis=0)

        # Compute radius (max intra-cluster Mahalanobis distance)
        if len(member_vectors) > 1:
            dists_to_centroid = []
            for mv in member_vectors:
                diff = mv - centroid
                d = np.sqrt(diff @ cov_inv @ diff)
                dists_to_centroid.append(d)
            radius = float(max(dists_to_centroid))
        else:
            radius = 0.0

        cluster = AnchorCluster(
            cluster_id=int(cluster_id),
            centroid=centroid,
            radius=radius,
            member_count=len(member_indices),
            member_dates=member_dates_cluster,
        )
        clusters.append(cluster)

    # Collect outlier vectors
    outlier_indices = [i for i, l in enumerate(labels) if l == -1]
    for idx in outlier_indices:
        outlier_vec = {}
        for j, feat in enumerate(L1_CLUSTER_FEATURES):
            outlier_vec[feat] = float(vectors_arr[idx, j])
        outlier_vectors.append(outlier_vec)

    # If no clusters found, create one from all data
    if not clusters:
        cluster = AnchorCluster(
            cluster_id=0,
            centroid=np.mean(normalized, axis=0),
            radius=0.0,
            member_count=len(normalized),
            member_dates=valid_dates,
        )
        clusters.append(cluster)

    return clusters, cov_inv, mins, maxs, outlier_vectors
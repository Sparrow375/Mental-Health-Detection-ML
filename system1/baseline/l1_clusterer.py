"""
L1 Clusterer: DBSCAN with Mahalanobis distance on 12-feature L1 daily vectors.

Determines behavioural archetypes from baseline data. These clusters
become the anchor reference for L2 coherence scoring — each monitoring
day is compared to these centroids to decide whether the person is in a
known behavioural context.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List

from system1.data_structures import L1ClusterState
from system1.feature_meta import L1_CLUSTERING_FEATURES, DEFAULT_THRESHOLDS

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class L1Clusterer:
    """
    DBSCAN clustering on the 12-feature L1 vector subspace.

    Step 1.4 from the implementation plan:
        1. Normalise features to [0,1] using person's baseline min/max
        2. Compute covariance matrix for Mahalanobis
        3. Auto-determine epsilon via k-distance elbow
        4. Run DBSCAN (min_samples=3, metric='mahalanobis')
        5. Extract centroids, radii, outlier indices
    """

    def __init__(self, min_samples: int | None = None):
        self.min_samples = min_samples or DEFAULT_THRESHOLDS['DBSCAN_MIN_SAMPLES']
        self.state = L1ClusterState()

    def fit(self, baseline_data: dict) -> L1ClusterState:
        """
        Fit DBSCAN clusters from baseline daily data.

        Parameters
        ----------
        baseline_data : dict
            Must contain keys for each day's features.  Typically a list of
            dicts, one per baseline day, each mapping feature-name → value.
            Or a pandas DataFrame.

        Returns
        -------
        L1ClusterState with centroids, radii, covariance_inv, etc.
        """
        import pandas as pd

        if isinstance(baseline_data, pd.DataFrame):
            df = baseline_data
        else:
            df = pd.DataFrame(baseline_data)

        # Build matrix of 12-feature vectors
        available = [f for f in L1_CLUSTERING_FEATURES if f in df.columns]
        if len(available) < 4:
            # Too few features — single cluster fallback
            return self._single_cluster_fallback(df, available)

        X_raw = df[available].values.astype(float)

        # Drop rows with all-NaN
        valid_mask = ~np.all(np.isnan(X_raw), axis=1)
        X_raw = X_raw[valid_mask]

        if len(X_raw) < self.min_samples + 1:
            return self._single_cluster_fallback(df, available)

        # Fill remaining NaN with column mean (or 0 if entirely NaN)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(X_raw, axis=0)
            col_means[np.isnan(col_means)] = 0.0

        for col_idx in range(X_raw.shape[1]):
            nan_mask = np.isnan(X_raw[:, col_idx])
            X_raw[nan_mask, col_idx] = col_means[col_idx]

        # Normalise to [0,1] per person's baseline
        fmin = X_raw.min(axis=0)
        fmax = X_raw.max(axis=0)
        rng = fmax - fmin
        rng[rng == 0] = 1.0  # avoid div by zero
        X_norm = (X_raw - fmin) / rng

        # Store normalisation params
        self.state.feature_min = fmin
        self.state.feature_max = fmax

        # Covariance for Mahalanobis
        try:
            cov = np.cov(X_norm.T)
            if cov.ndim < 2:
                cov = np.array([[cov]])
            # Regularise to avoid singularity
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)
            self.state.covariance_inv = cov_inv
        except np.linalg.LinAlgError:
            self.state.covariance_inv = np.eye(X_norm.shape[1])
            cov_inv = self.state.covariance_inv

        if not HAS_SKLEARN:
            return self._single_cluster_fallback(df, available)

        # Auto-determine epsilon via k-distance graph
        epsilon = self._compute_epsilon(X_norm, cov_inv)

        # Run DBSCAN
        try:
            # DBSCAN with precomputed distances for Mahalanobis
            from scipy.spatial.distance import pdist, squareform

            # Compute pairwise Mahalanobis distances
            n = X_norm.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    diff = X_norm[i] - X_norm[j]
                    d = np.sqrt(diff @ cov_inv @ diff)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            db = DBSCAN(
                eps=epsilon,
                min_samples=self.min_samples,
                metric='precomputed',
            ).fit(dist_matrix)
            labels = db.labels_
        except Exception:
            # Fallback to euclidean
            db = DBSCAN(
                eps=epsilon,
                min_samples=self.min_samples,
                metric='euclidean',
            ).fit(X_norm)
            labels = db.labels_

        self.state.labels = labels

        # Extract clusters
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            return self._single_cluster_fallback(df, available)

        centroids = []
        radii = []

        for label in sorted(unique_labels):
            member_mask = labels == label
            members = X_norm[member_mask]
            centroid = members.mean(axis=0)
            centroids.append(centroid)

            # Radius: max intra-cluster distance
            max_dist = 0.0
            for member in members:
                diff = member - centroid
                try:
                    d = np.sqrt(diff @ cov_inv @ diff)
                except Exception:
                    d = np.linalg.norm(diff)
                max_dist = max(max_dist, d)
            radii.append(max(max_dist, 0.1))  # min radius to avoid zero

        self.state.n_clusters = n_clusters
        self.state.centroids = np.array(centroids)
        self.state.radii = np.array(radii)
        self.state.outlier_indices = list(np.where(labels == -1)[0])

        print(f"  [L1 Cluster] {n_clusters} archetypes found, "
              f"{len(self.state.outlier_indices)} outlier days, "
              f"eps={epsilon:.3f}")

        return self.state

    def _compute_epsilon(self, X_norm: np.ndarray, cov_inv: np.ndarray) -> float:
        """
        Determine DBSCAN epsilon per person:
        Sort distances to k-th nearest neighbor (k=min_samples),
        find elbow of the k-distance graph.
        """
        k = self.min_samples
        n = X_norm.shape[0]

        # Compute k-th nearest neighbor distances using Mahalanobis
        kth_distances = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i == j:
                    continue
                diff = X_norm[i] - X_norm[j]
                try:
                    d = np.sqrt(diff @ cov_inv @ diff)
                except Exception:
                    d = np.linalg.norm(diff)
                distances.append(d)
            distances.sort()
            if len(distances) >= k:
                kth_distances.append(distances[k - 1])
            elif distances:
                kth_distances.append(distances[-1])

        kth_distances.sort()

        # Find elbow: point of maximum second derivative
        if len(kth_distances) < 3:
            return np.median(kth_distances) if kth_distances else 1.0

        diffs = np.diff(kth_distances)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            elbow_idx = int(np.argmax(diffs2)) + 1
            epsilon = kth_distances[min(elbow_idx, len(kth_distances) - 1)]
        else:
            epsilon = np.median(kth_distances)

        # Clamp to reasonable range
        if np.isnan(epsilon):
            return 1.0
        return float(np.clip(epsilon, 0.3, 3.5))

    def _single_cluster_fallback(self, df, available_features) -> L1ClusterState:
        """
        Fallback: entire baseline is one archetype.
        Used when data is too sparse for DBSCAN.
        """
        import pandas as pd

        if len(available_features) == 0:
            self.state.n_clusters = 1
            self.state.centroids = np.zeros((1, len(L1_CLUSTERING_FEATURES)))
            self.state.radii = np.array([2.0])
            self.state.covariance_inv = np.eye(len(L1_CLUSTERING_FEATURES))
            self.state.feature_min = np.zeros(len(L1_CLUSTERING_FEATURES))
            self.state.feature_max = np.ones(len(L1_CLUSTERING_FEATURES))
            print("  [L1 Cluster] Fallback: single cluster (no features)")
            return self.state

        X = df[available_features].values.astype(float)
        valid_mask = ~np.all(np.isnan(X), axis=1)
        X = X[valid_mask]

        col_means = np.nanmean(X, axis=0) if len(X) > 0 else np.zeros(len(available_features))
        for col_idx in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col_idx])
            X[nan_mask, col_idx] = col_means[col_idx]

        fmin = X.min(axis=0) if len(X) > 0 else np.zeros(len(available_features))
        fmax = X.max(axis=0) if len(X) > 0 else np.ones(len(available_features))
        rng = fmax - fmin
        rng[rng == 0] = 1.0
        X_norm = (X - fmin) / rng

        centroid = X_norm.mean(axis=0).reshape(1, -1) if len(X_norm) > 0 else np.zeros((1, len(available_features)))

        # Pad to full 12-feature width if needed
        if centroid.shape[1] < len(L1_CLUSTERING_FEATURES):
            padded = np.zeros((1, len(L1_CLUSTERING_FEATURES)))
            padded[0, :centroid.shape[1]] = centroid[0]
            centroid = padded
            fmin_padded = np.zeros(len(L1_CLUSTERING_FEATURES))
            fmax_padded = np.ones(len(L1_CLUSTERING_FEATURES))
            fmin_padded[:len(fmin)] = fmin
            fmax_padded[:len(fmax)] = fmax
            fmin = fmin_padded
            fmax = fmax_padded

        radius = 2.0  # Generous radius for single-cluster fallback

        self.state.n_clusters = 1
        self.state.centroids = centroid
        self.state.radii = np.array([radius])
        self.state.covariance_inv = np.eye(len(L1_CLUSTERING_FEATURES))
        self.state.feature_min = fmin
        self.state.feature_max = fmax
        self.state.labels = np.zeros(len(X), dtype=int) if len(X) > 0 else np.array([0])

        print(f"  [L1 Cluster] Fallback: single cluster ({len(available_features)} features)")
        return self.state

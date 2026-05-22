"""
Candidate Cluster Evaluator: 7-day evaluation window for new behavioural archetypes.

Triggered when L1 is anomalous but L2 texture is coherent — indicating a
potentially new healthy archetype (exam period, new job, travel) rather than
a clinical episode.

The window:
    Days 1–3 : hold and observe, no evidence accumulation
    Days 4–7 : evaluate texture quality
        → Promote:  texture healthy → new anchor cluster, clear held evidence
        → Reject:   texture degrades → retroactively release all held evidence
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from system1.data_structures import CandidateState, L1ClusterState
from system1.feature_meta import DEFAULT_THRESHOLDS, L1_CLUSTERING_FEATURES


class CandidateClusterEvaluator:
    """
    Manages the 7-day candidate evaluation window.

    Lifecycle:
        CLOSED → should_open_window() → EVALUATING → evaluate_day() ×7
        → promote() or reject() → CLOSED
    """

    def __init__(
        self,
        cluster_state: Optional[L1ClusterState] = None,
        thresholds: dict | None = None,
    ):
        self.cluster_state = cluster_state
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.WINDOW_DAYS = t['CANDIDATE_WINDOW_DAYS']
        self.TEXTURE_THRESHOLD = t['CANDIDATE_TEXTURE_THRESHOLD']

        self.state = CandidateState()

    @property
    def is_active(self) -> bool:
        return self.state.status == 'EVALUATING'

    # ------------------------------------------------------------------
    # Step 5.1 — Open candidate window
    # ------------------------------------------------------------------

    def should_open_window(self, candidate_flag: bool) -> bool:
        """Check whether to open a candidate evaluation window."""
        return candidate_flag and self.state.status == 'CLOSED'

    def open_window(self, day_number: int) -> CandidateState:
        """Pause evidence accumulation and start buffering."""
        self.state = CandidateState(
            status='EVALUATING',
            open_day=day_number,
            days_elapsed=0,
            l1_buffer=[],
            l2_buffer=[],
            session_incoherence_history=[],
            held_evidence=[],
        )
        return self.state

    # ------------------------------------------------------------------
    # Step 5.2–5.3 — Daily evaluation during window
    # ------------------------------------------------------------------

    def evaluate_day(
        self,
        l1_vector: Dict[str, float],
        l2_result: Dict,
        effective_score: float,
    ) -> str:
        """
        Buffer one day of data and decide whether to continue, promote, or reject.

        Returns: 'EVALUATING', 'PROMOTED', or 'REJECTED'
        """
        if self.state.status != 'EVALUATING':
            return self.state.status

        self.state.days_elapsed += 1
        self.state.l1_buffer.append(l1_vector)
        self.state.l2_buffer.append(l2_result)
        self.state.session_incoherence_history.append(
            l2_result.get('session_incoherence', 0.0)
        )
        self.state.held_evidence.append(effective_score)

        # Days 1–3: hold, do not evaluate yet
        if self.state.days_elapsed < 4:
            return 'EVALUATING'

        # Days 4–7: evaluate texture quality
        if self.state.days_elapsed >= 4:
            result = self._evaluate_texture_quality()
            if result is not None:
                return result

        # At end of window, force decision
        if self.state.days_elapsed >= self.WINDOW_DAYS:
            return self._force_decision()

        return 'EVALUATING'

    # ------------------------------------------------------------------
    # Promote → new anchor cluster
    # ------------------------------------------------------------------

    def promote(self) -> CandidateState:
        """
        Texture quality held — this is a new healthy archetype.
        Add new centroid to cluster state. Clear held evidence.
        """
        self.state.status = 'PROMOTED'

        # Compute new centroid from buffered L1 vectors
        if self.cluster_state is not None and self.state.l1_buffer:
            new_centroid = self._compute_centroid_from_buffer()
            if new_centroid is not None:
                self._add_cluster(new_centroid)

        held = list(self.state.held_evidence)
        self.state.held_evidence = []
        self.state = CandidateState()  # Reset to CLOSED
        return self.state

    # ------------------------------------------------------------------
    # Reject → retroactively release all held evidence
    # ------------------------------------------------------------------

    def reject(self) -> List[float]:
        """
        Texture degraded — this is clinical onset.
        Return all held effective scores for retroactive release.
        """
        self.state.status = 'REJECTED'
        held = list(self.state.held_evidence)
        self.state = CandidateState()  # Reset to CLOSED
        return held

    # ------------------------------------------------------------------
    # Internal evaluation logic
    # ------------------------------------------------------------------

    def _evaluate_texture_quality(self) -> Optional[str]:
        """
        Check texture quality conditions per the docx:
        - If session_incoherence < 0.35 on majority AND no monotonic degradation → promote
        - If session_incoherence trending up OR > 0.35 on majority → reject
        - Any monotonic degradation → reject (even if starting low)
        """
        history = self.state.session_incoherence_history
        if len(history) < 4:
            return None

        # Check majority condition
        above_threshold = sum(1 for v in history if v >= self.TEXTURE_THRESHOLD)
        majority_degraded = above_threshold > len(history) / 2

        # Check for monotonic degradation (tiebreaker rule)
        monotonic_degrading = all(
            history[i] <= history[i + 1] for i in range(len(history) - 1)
        ) and history[-1] > history[0] + 0.05  # meaningful increase

        if monotonic_degrading:
            return 'REJECTED'

        if majority_degraded:
            return 'REJECTED'

        # If we have enough evidence of healthy texture and near end of window
        if self.state.days_elapsed >= self.WINDOW_DAYS - 1 and not majority_degraded:
            return 'PROMOTED'

        return None

    def _force_decision(self) -> str:
        """End of window — force promote or reject."""
        history = self.state.session_incoherence_history
        above = sum(1 for v in history if v >= self.TEXTURE_THRESHOLD)

        if above > len(history) / 2:
            return 'REJECTED'
        else:
            return 'PROMOTED'

    def _compute_centroid_from_buffer(self) -> Optional[np.ndarray]:
        """Compute new cluster centroid from buffered L1 vectors."""
        vectors = []
        for l1_dict in self.state.l1_buffer:
            vec = [l1_dict.get(f, 0.0) for f in L1_CLUSTERING_FEATURES]
            vectors.append(vec)

        if vectors:
            return np.mean(vectors, axis=0)
        return None

    def _add_cluster(self, centroid: np.ndarray):
        """Add a new cluster to the L1 cluster state."""
        if self.cluster_state is None:
            return

        cs = self.cluster_state
        if cs.centroids is not None:
            cs.centroids = np.vstack([cs.centroids, centroid.reshape(1, -1)])
            # Compute radius as max distance from centroid to buffered points
            vecs = []
            for l1_dict in self.state.l1_buffer:
                vec = [l1_dict.get(f, 0.0) for f in L1_CLUSTERING_FEATURES]
                vecs.append(vec)
            distances = [np.linalg.norm(np.array(v) - centroid) for v in vecs]
            new_radius = max(distances) if distances else 1.0
            cs.radii = np.append(cs.radii, new_radius)
            cs.n_clusters += 1

"""
System 1: Behavioral Anomaly Detection Pipeline
L1 (aggregate deviation) + L2 (digital DNA texture) scoring

Public API:
    from system1 import PersonalityVector, AnomalyDetector
    from system1 import ImprovedAnomalyDetector  # backward-compatible alias
"""

from system1.data_structures import (
    PersonalityVector,
    AppDNA,
    PhoneDNA,
    ContextualTextureProfile,
    AnomalyReport,
    DailyReport,
    FinalPrediction,
    EvidenceState,
    CandidateState,
    L1ClusterState,
)
from system1.detector import AnomalyDetector

# Backward-compatible alias
ImprovedAnomalyDetector = AnomalyDetector

__all__ = [
    'PersonalityVector',
    'AppDNA',
    'PhoneDNA',
    'ContextualTextureProfile',
    'AnomalyReport',
    'DailyReport',
    'FinalPrediction',
    'EvidenceState',
    'CandidateState',
    'L1ClusterState',
    'AnomalyDetector',
    'ImprovedAnomalyDetector',
]

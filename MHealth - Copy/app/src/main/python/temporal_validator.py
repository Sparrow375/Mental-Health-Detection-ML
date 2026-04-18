from dataclasses import dataclass
from prototype_matcher import ConfidenceTier

@dataclass
class AdjustedClassification:
    disorder: str
    adjusted_score: float
    confidence: ConfidenceTier
    temporal_shape: str = "stable"

class TemporalValidator:
    def __init__(self):
        pass

    def validate(self, classification, timeseries):
        # Simply pass through the geometric classification result directly as a temporal result
        # Currently acting as a mock implementation since the temporal validation module is missing.
        return AdjustedClassification(
            disorder=classification.disorder,
            adjusted_score=classification.score,
            confidence=classification.confidence
        )

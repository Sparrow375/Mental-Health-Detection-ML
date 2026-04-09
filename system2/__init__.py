"""
System 2: Metric-Based Clinical Prototype Matching

A fully interpretable, metric-driven, training-data-free disorder
classification engine.  Compares a user's behavioral deviation pattern
against clinically-grounded disorder prototypes and outputs a ranked
match with confidence and explainability.
"""

from system2.config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME1,
    DISORDER_PROTOTYPES_FRAME2,
    FEATURE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
)
from system2.baseline_screener import BaselineScreener
from system2.prototype_matcher import PrototypeMatcher
from system2.temporal_validator import TemporalValidator
from system2.life_event_filter import LifeEventFilter
from system2.explainability import ExplainabilityEngine
from system2.pipeline import System2Pipeline

__all__ = [
    "BEHAVIORAL_FEATURES",
    "POPULATION_NORMS",
    "DISORDER_PROTOTYPES_FRAME1",
    "DISORDER_PROTOTYPES_FRAME2",
    "FEATURE_WEIGHTS",
    "CONFIDENCE_THRESHOLDS",
    "BaselineScreener",
    "PrototypeMatcher",
    "TemporalValidator",
    "LifeEventFilter",
    "ExplainabilityEngine",
    "System2Pipeline",
]

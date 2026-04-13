"""
System 2: Metric-Based Clinical Prototype Matching

A fully interpretable, metric-driven, training-data-free disorder
classification engine.  Compares a user's behavioral deviation pattern
against clinically-grounded disorder prototypes and outputs a ranked
match with confidence and explainability.
"""

from config import (
    BEHAVIORAL_FEATURES,
    POPULATION_NORMS,
    DISORDER_PROTOTYPES_FRAME1,
    DISORDER_PROTOTYPES_FRAME2,
    FEATURE_WEIGHTS,
    CONFIDENCE_THRESHOLDS,
)
from baseline_screener import BaselineScreener
from prototype_matcher import PrototypeMatcher
from temporal_validator import TemporalValidator
from life_event_filter import LifeEventFilter
from explainability import ExplainabilityEngine
from pipeline import System2Pipeline

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


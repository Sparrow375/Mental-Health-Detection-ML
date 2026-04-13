"""
S1 → S2 Adapter
================

Bridges System 1's output format to System 2's S1Input contract.
This module is the ONLY coupling point between the two systems.

Usage
-----
    from s1_s2_adapter import build_s1_input

    s1_input = build_s1_input(detector, baseline_df, latest_s1_report)
    s2_output = pipeline.classify(s1_input)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from life_event_filter import AnomalyReport as S2AnomalyReport
from pipeline import S1Input


# ── Baseline Data Builder ──────────────────────────────────────────────

def build_baseline_data(baseline_df: pd.DataFrame) -> Dict:
    """
    Convert S1's baseline DataFrame (28 days) into S2's expected format.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        28-day baseline period data from SyntheticDataGenerator or
        StudentLife feature extraction. Must contain columns for all
        18 behavioral features + a 'date' column.

    Returns
    -------
    dict with keys: "raw_7day", "weekly_windows" (3×), "raw_28day"
    """
    # Drop non-feature columns
    feature_cols = [c for c in baseline_df.columns if c != 'date']
    df = baseline_df[feature_cols]

    # raw_7day: first 7 days average
    raw_7day = df.iloc[:7].mean().to_dict()

    # weekly_windows: 3 weekly averages (days 0-6, 7-13, 14-20)
    # If baseline is shorter than 21 days, we use what we have
    weekly_windows = []
    for week_start in [0, 7, 14]:
        week_end = min(week_start + 7, len(df))
        if week_start < len(df):
            window = df.iloc[week_start:week_end].mean().to_dict()
            weekly_windows.append(window)

    # raw_28day: full 28-day average
    raw_28day = df.mean().to_dict()

    return {
        "raw_7day": raw_7day,
        "weekly_windows": weekly_windows,
        "raw_28day": raw_28day,
    }


# ── Anomaly Report Mapper ─────────────────────────────────────────────

def build_anomaly_report(
    s1_report,
    co_deviating_threshold: float = 1.0,
) -> S2AnomalyReport:
    """
    Map System 1's AnomalyReport to System 2's AnomalyReport format.

    Parameters
    ----------
    s1_report : system1.AnomalyReport
        The latest report from ImprovedAnomalyDetector.analyze().
    co_deviating_threshold : float
        SD threshold for counting co-deviating features.
        Default 1.0 SD — sub-1 SD fluctuations are within normal healthy
        daily variation and should not count as clinical co-deviations.

    Returns
    -------
    S2AnomalyReport
        Ready for System 2's LifeEventFilter and PrototypeMatcher.
    """
    # Count features deviating beyond threshold
    co_deviating_count = sum(
        1 for dev in s1_report.feature_deviations.values()
        if abs(dev) > co_deviating_threshold
    )

    return S2AnomalyReport(
        feature_deviations=s1_report.feature_deviations,
        days_sustained=s1_report.sustained_deviation_days,
        co_deviating_count=co_deviating_count,
        resolved=False,  # active anomalies are not resolved
        days_since_onset=s1_report.sustained_deviation_days,
    )


# ── Full S1Input Builder ──────────────────────────────────────────────

def build_s1_input(
    detector,
    baseline_df: pd.DataFrame,
    s1_report,
    timeseries_days: int = 28,
) -> S1Input:
    """
    Build the complete S1Input bundle from System 1 components.

    Parameters
    ----------
    detector : ImprovedAnomalyDetector
        The active detector instance (has full_anomaly_history).
    baseline_df : pd.DataFrame
        28-day baseline DataFrame.
    s1_report : system1.AnomalyReport
        Latest anomaly report from detector.analyze().
    timeseries_days : int
        Number of recent days for the anomaly timeseries (default 28).

    Returns
    -------
    S1Input
        Ready for System2Pipeline.classify().
    """
    # 1. Build baseline data from DataFrame
    baseline_data = build_baseline_data(baseline_df)

    # 2. Map anomaly report
    anomaly_report = build_anomaly_report(s1_report)

    # 3. Get anomaly timeseries (last N days, pad with 0 if shorter)
    history = detector.full_anomaly_history
    if len(history) >= timeseries_days:
        timeseries = history[-timeseries_days:]
    else:
        # Pad front with zeros if not enough history yet
        padding = [0.0] * (timeseries_days - len(history))
        timeseries = padding + list(history)

    return S1Input(
        baseline_data=baseline_data,
        anomaly_report=anomaly_report,
        anomaly_timeseries=timeseries,
    )


"""
AppDNA Builder — constructs per-app behavioral DNA from session events.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data_structures import AppDNA, SessionEvent
from ..feature_meta import THRESHOLDS


def build_app_dna(
    app_id: str,
    sessions: List[SessionEvent],
    all_sessions: List[SessionEvent],
    baseline_dates: List[str],
) -> Optional[AppDNA]:
    """
    Build AppDNA for a single app from its baseline sessions.
    Returns None if the app has fewer than MIN_APP_BASELINE_APPEARANCES sessions.
    """
    min_sessions = THRESHOLDS["MIN_APP_BASELINE_APPEARANCES"]
    if len(sessions) < min_sessions:
        return None

    dna = AppDNA(app_id=app_id)

    # ── Usage heatmap (7 x 24) ───────────────────────────────────────────
    heatmap = np.zeros((7, 24))
    for sess in sessions:
        import datetime
        dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
        dow = dt.weekday()  # 0=Mon, 6=Sun
        hour = dt.hour
        duration_hours = sess.duration_minutes / 60.0
        heatmap[dow, hour] += duration_hours

    # Average over number of weeks in baseline
    n_weeks = max(1, len(baseline_dates) / 7.0)
    heatmap /= n_weeks
    dna.usage_heatmap = heatmap

    # ── Session duration distribution ─────────────────────────────────────
    durations = np.array([s.duration_minutes for s in sessions])
    dna.avg_session_minutes = float(np.mean(durations))
    dna.std_session_minutes = float(np.std(durations)) if len(durations) > 1 else 0.0
    dna.p10_session_minutes = float(np.percentile(durations, 10))
    dna.p90_session_minutes = float(np.percentile(durations, 90))

    # ── Abandon rate ──────────────────────────────────────────────────────
    abandon_sessions = [s for s in sessions if s.duration_minutes < 0.75 and s.interaction_count < 5]
    dna.abandon_rate = len(abandon_sessions) / max(1, len(sessions))
    # Per-day abandon rate for std calculation
    daily_abandons: Dict[str, int] = defaultdict(int)
    daily_total: Dict[str, int] = defaultdict(int)
    for sess in sessions:
        import datetime
        dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        daily_total[date_str] += 1
        if sess.duration_minutes < 0.75 and sess.interaction_count < 5:
            daily_abandons[date_str] += 1
    daily_rates = [daily_abandons[d] / max(1, daily_total[d]) for d in daily_total]
    dna.abandon_rate_std = float(np.std(daily_rates)) if len(daily_rates) > 1 else 0.0

    # ── Primary time range (smallest contiguous window containing 80% usage) ──
    hourly_totals = heatmap.sum(axis=0)  # Sum across days of week
    total_usage = hourly_totals.sum()
    if total_usage > 0:
        best_start, best_end, best_span = 0, 23, 24
        for start in range(24):
            cumulative = 0.0
            for offset in range(24):
                hour = (start + offset) % 24
                cumulative += hourly_totals[hour]
                if cumulative >= 0.8 * total_usage:
                    span = offset + 1
                    if span < best_span:
                        best_start = start
                        best_end = (start + offset) % 24
                        best_span = span
                    break
        dna.primary_time_range = (best_start, best_end)

    # ── Time concentration ratio ──────────────────────────────────────────
    if total_usage > 0:
        start_h, end_h = dna.primary_time_range
        if start_h <= end_h:
            primary_total = hourly_totals[start_h:end_h + 1].sum()
        else:
            primary_total = np.concatenate([hourly_totals[start_h:], hourly_totals[:end_h + 1]]).sum()
        dna.time_concentration_ratio = primary_total / total_usage

        # Day-to-day std of concentration ratio
        daily_concentrations = []
        import datetime
        for sess in sessions:
            pass  # Simplified: use overall
        dna.time_concentration_std = 0.1  # Default estimate

    # ── Trigger DNA ───────────────────────────────────────────────────────
    trigger_counts: Dict[str, int] = defaultdict(int)
    for sess in sessions:
        trigger_counts[sess.trigger] += 1
    total_triggers = max(1, sum(trigger_counts.values()))
    dna.self_open_ratio = trigger_counts.get("SELF", 0) / total_triggers
    dna.notification_open_ratio = trigger_counts.get("NOTIFICATION", 0) / total_triggers
    dna.shortcut_open_ratio = (
        trigger_counts.get("SHORTCUT", 0) + trigger_counts.get("WIDGET", 0)
    ) / total_triggers

    # ── Interaction density ───────────────────────────────────────────────
    ipms = []
    for sess in sessions:
        if sess.duration_minutes > 0:
            ipms.append(sess.interaction_count / sess.duration_minutes)
    if ipms:
        dna.interactions_per_minute_mean = float(np.mean(ipms))
        dna.interactions_per_minute_std = float(np.std(ipms)) if len(ipms) > 1 else 0.0

    # ── Weekday / weekend split ───────────────────────────────────────────
    import datetime
    weekday_sessions = []
    weekend_sessions = []
    for sess in sessions:
        dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
        if dt.weekday() < 5:
            weekday_sessions.append(sess)
        else:
            weekend_sessions.append(sess)

    n_weekdays = max(1, len([d for d in baseline_dates
                              if datetime.datetime.strptime(d, "%Y-%m-%d").weekday() < 5]))
    n_weekends = max(1, len(baseline_dates) - n_weekdays)
    dna.weekday_sessions_per_day = len(weekday_sessions) / n_weekdays
    dna.weekend_sessions_per_day = len(weekend_sessions) / n_weekends

    # ── Pre/post open apps (behavioral grammar) ───────────────────────────
    pre_apps: Dict[str, int] = defaultdict(int)
    post_apps: Dict[str, int] = defaultdict(int)

    sorted_all = sorted(all_sessions, key=lambda s: s.open_ts)
    for i, sess in enumerate(sorted_all):
        if sess.app_id == app_id:
            if i > 0:
                pre_apps[sorted_all[i - 1].app_id] += 1
            if i < len(sorted_all) - 1:
                post_apps[sorted_all[i + 1].app_id] += 1

    total_pre = max(1, sum(pre_apps.values()))
    total_post = max(1, sum(post_apps.values()))
    dna.pre_open_apps = {k: v / total_pre for k, v in pre_apps.items()}
    dna.post_open_apps = {k: v / total_post for k, v in post_apps.items()}

    # ── Daily use consistency ─────────────────────────────────────────────
    active_dates = set()
    import datetime
    for sess in sessions:
        dt = datetime.datetime.fromtimestamp(sess.open_ts / 1000.0)
        active_dates.add(dt.strftime("%Y-%m-%d"))

    dna.daily_use_consistency = len(active_dates) / max(1, len(baseline_dates))

    # Max gap days
    if active_dates:
        sorted_dates = sorted(active_dates)
        max_gap = 0
        for i in range(1, len(sorted_dates)):
            d1 = datetime.datetime.strptime(sorted_dates[i - 1], "%Y-%m-%d")
            d2 = datetime.datetime.strptime(sorted_dates[i], "%Y-%m-%d")
            gap = (d2 - d1).days
            max_gap = max(max_gap, gap)
        dna.max_gap_days = max_gap

    return dna


def build_all_app_dnas(
    sessions: List[SessionEvent],
    baseline_dates: List[str],
) -> Dict[str, AppDNA]:
    """Build AppDNA for every app that meets the minimum session threshold."""
    app_sessions: Dict[str, List[SessionEvent]] = defaultdict(list)
    for sess in sessions:
        app_sessions[sess.app_id].append(sess)

    dnas = {}
    for app_id, sess_list in app_sessions.items():
        dna = build_app_dna(app_id, sess_list, sessions, baseline_dates)
        if dna is not None:
            dnas[app_id] = dna

    return dnas
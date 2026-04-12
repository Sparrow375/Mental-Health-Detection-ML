"""
AppDNA Builder: constructs per-app behavioural fingerprints from session events.

Each app with ≥3 sessions during baseline gets an AppDNA object containing:
    - Temporal DNA  (7×24 usage heatmap, primary time range)
    - Session signature  (duration distributions, abandon rate)
    - Trigger DNA  (self/notification/shortcut ratios, response latency)
    - Sequence DNA  (pre/post app transitions)
    - Engagement density  (interactions per minute)
    - Day-type split  (weekday vs weekend)
    - Consistency  (daily_use_consistency, max_gap_days)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime

from system1.data_structures import AppDNA
from system1.feature_meta import DEFAULT_THRESHOLDS


class AppDNABuilder:
    """
    Builds AppDNA profiles from baseline session and notification events.

    Returns empty dict if no session data is available (e.g. StudentLife).
    """

    def __init__(self, min_appearances: int | None = None):
        self.min_appearances = min_appearances or DEFAULT_THRESHOLDS['MIN_APP_BASELINE_APPEARANCES']

    def build(
        self,
        baseline_session_events: Optional[List[List[Dict]]] = None,
        baseline_notification_events: Optional[List[List[Dict]]] = None,
        baseline_days: int = 28,
    ) -> Dict[str, AppDNA]:
        """
        Build AppDNA for each app appearing ≥ min_appearances times.

        Parameters
        ----------
        baseline_session_events : list of daily session event lists
            Each inner list has dicts with keys: app_package, open_timestamp_ms,
            close_timestamp_ms, trigger, interaction_count, hour, day_of_week,
            duration_minutes.
        baseline_notification_events : list of daily notification event lists
        baseline_days : total number of baseline days

        Returns
        -------
        Dict mapping app_package → AppDNA
        """
        if not baseline_session_events:
            return {}

        # Flatten all sessions and group by app
        all_sessions: Dict[str, List[Dict]] = defaultdict(list)
        days_seen: Dict[str, set] = defaultdict(set)

        for day_idx, day_sessions in enumerate(baseline_session_events):
            if not day_sessions:
                continue
            for sess in day_sessions:
                pkg = sess.get('app_package', '')
                if pkg:
                    all_sessions[pkg].append({**sess, '_day_idx': day_idx})
                    days_seen[pkg].add(day_idx)

        # Build notification lookup
        all_notifications: Dict[str, List[Dict]] = defaultdict(list)
        if baseline_notification_events:
            for day_notifs in baseline_notification_events:
                if not day_notifs:
                    continue
                for notif in day_notifs:
                    pkg = notif.get('app_package', '')
                    if pkg:
                        all_notifications[pkg].append(notif)

        # Build AppDNA per qualifying app
        result: Dict[str, AppDNA] = {}

        for pkg, sessions in all_sessions.items():
            if len(sessions) < self.min_appearances:
                continue

            dna = AppDNA(app_package=pkg)

            # --- Temporal DNA ---
            heatmap = np.zeros((7, 24))
            day_counts = np.zeros((7, 24))
            for s in sessions:
                dow = int(s.get('day_of_week', 0)) % 7
                hour = int(s.get('hour', 0)) % 24
                dur = float(s.get('duration_minutes', 1.0))
                heatmap[dow, hour] += dur
                day_counts[dow, hour] += 1

            # Normalise by number of days that contributed
            for dow in range(7):
                for hour in range(24):
                    if day_counts[dow, hour] > 0:
                        heatmap[dow, hour] /= day_counts[dow, hour]
            dna.usage_heatmap = heatmap

            # Primary time range: smallest window containing 80% of usage
            hourly_total = heatmap.sum(axis=0)
            total_usage = hourly_total.sum()
            if total_usage > 0:
                best_window = (0, 23)
                best_size = 24
                for start in range(24):
                    cumulative = 0.0
                    for size in range(1, 25):
                        hour = (start + size - 1) % 24
                        cumulative += hourly_total[hour]
                        if cumulative >= total_usage * 0.8 and size < best_size:
                            best_size = size
                            best_window = (start, (start + size - 1) % 24)
                            break
                dna.primary_time_range = best_window

                # Time concentration ratio
                window_usage = sum(hourly_total[(best_window[0] + i) % 24]
                                   for i in range(best_size))
                dna.time_concentration_ratio = window_usage / max(total_usage, 1e-6)

            # --- Session signature ---
            durations = [float(s.get('duration_minutes', 0)) for s in sessions]
            if durations:
                dna.avg_session_minutes = float(np.mean(durations))
                dna.std_session_minutes = float(np.std(durations))
                dna.p10_session_minutes = float(np.percentile(durations, 10))
                dna.p90_session_minutes = float(np.percentile(durations, 90))

            # Abandon rate
            interactions = [int(s.get('interaction_count', 0)) for s in sessions]
            abandons = sum(1 for d, ic in zip(durations, interactions)
                          if d < 0.75 and ic < 5)
            dna.abandon_rate = abandons / max(len(sessions), 1)

            # --- Trigger DNA ---
            triggers = [s.get('trigger', 'SELF') for s in sessions]
            n = max(len(triggers), 1)
            dna.self_open_ratio = sum(1 for t in triggers if t == 'SELF') / n
            dna.notification_open_ratio = sum(1 for t in triggers if t == 'NOTIFICATION') / n
            dna.shortcut_open_ratio = sum(1 for t in triggers if t in ('SHORTCUT', 'WIDGET')) / n

            # Notification response latency
            notifs = all_notifications.get(pkg, [])
            latencies = [float(n_evt.get('tap_latency_minutes', 0))
                         for n_evt in notifs
                         if n_evt.get('tap_latency_minutes') is not None]
            if latencies:
                dna.notification_response_latency_median = float(np.median(latencies))
                dna.notification_response_latency_std = float(np.std(latencies))

            # --- Engagement density ---
            densities = [ic / max(d, 0.01)
                         for ic, d in zip(interactions, durations)]
            if densities:
                dna.interactions_per_minute_mean = float(np.mean(densities))
                dna.interactions_per_minute_std = float(np.std(densities))

            # --- Weekday/weekend split ---
            weekday_counts = defaultdict(int)
            for s in sessions:
                dow = int(s.get('day_of_week', 0)) % 7
                if dow < 5:
                    weekday_counts['weekday'] += 1
                else:
                    weekday_counts['weekend'] += 1

            weekday_days = min(baseline_days * 5 / 7, 1)
            weekend_days = min(baseline_days * 2 / 7, 1)
            dna.weekday_sessions_per_day = weekday_counts.get('weekday', 0) / max(weekday_days, 1)
            dna.weekend_sessions_per_day = weekday_counts.get('weekend', 0) / max(weekend_days, 1)

            # --- Consistency ---
            dna.daily_use_consistency = len(days_seen[pkg]) / max(baseline_days, 1)

            # Max gap days
            sorted_days = sorted(days_seen[pkg])
            max_gap = 0
            for i in range(1, len(sorted_days)):
                gap = sorted_days[i] - sorted_days[i - 1]
                max_gap = max(max_gap, gap)
            dna.max_gap_days = max_gap

            result[pkg] = dna

        print(f"  [AppDNA] Built profiles for {len(result)} apps")
        return result

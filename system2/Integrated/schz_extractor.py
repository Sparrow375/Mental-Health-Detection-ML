"""
Schizophrenia (CrossCheck/Eureka) Feature Extractor
=====================================================

Reads the CrossCheck_Daily_Data.csv and maps the pre-aggregated
sensor columns to the 18 behavioral features used by PersonalityVector.

The CrossCheck dataset has 5 time epochs (ep_0 to ep_4) per day.
We sum/average across epochs to get daily totals.

Usage
-----
    from schz_extractor import SchzExtractor

    ext = SchzExtractor()
    df = ext.extract_patient("u004")    # 1 row/day, 18 feature cols + date
    labels = ext.load_ema_labels()       # {uid: {avg_voices, avg_depression, ...}}
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "schz",
    "CrossCheck_Daily_Data.csv",
)

EPOCHS = [0, 1, 2, 3, 4]   # ep_0 … ep_4


class SchzExtractor:
    """
    Extract 18 PersonalityVector features from the CrossCheck daily data.
    """

    def __init__(self, data_path: str = DATA_PATH):
        self.path = data_path
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.path, low_memory=False)
            # Fix date: day column is int like 20150122 → datetime
            self._df["date"] = pd.to_datetime(
                self._df["day"].astype(str), format="%Y%m%d", errors="coerce"
            )
            self._df = self._df.dropna(subset=["date"])
        return self._df

    def get_patient_ids(self) -> List[str]:
        df = self._load()
        return sorted(df["eureka_id"].dropna().unique().tolist())

    # ── Public API ──────────────────────────────────────────────────

    def extract_patient(self, uid: str) -> pd.DataFrame:
        """
        Extract daily features for one patient. Returns DataFrame with
        columns: date + 18 feature columns.
        """
        df = self._load()
        patient = df[df["eureka_id"] == uid].copy()
        if len(patient) == 0:
            return pd.DataFrame()

        patient = patient.sort_values("date").reset_index(drop=True)

        rows = []
        for _, row in patient.iterrows():
            feat = {"date": row["date"]}

            # ── Screen time (unlock_duration in seconds across epochs) ──
            unlock_dur = sum(
                _safe(row, f"unlock_duration_ep_{e}") for e in EPOCHS
            )
            feat["screen_time_hours"] = unlock_dur / 3600.0

            # ── Unlock count ─────────────────────────────────────────
            feat["unlock_count"] = sum(
                _safe(row, f"unlock_num_ep_{e}") for e in EPOCHS
            )

            # ── Social app ratio ──────────────────────────────────────
            # CrossCheck has app_lists_num_apps_opened but not package breakdown
            # Use audio conversation time as proxy for social engagement ratio
            # (compared to total unlock time)
            convo_dur = sum(
                _safe(row, f"audio_convo_duration_ep_{e}") for e in EPOCHS
            )
            if unlock_dur > 0:
                feat["social_app_ratio"] = min(convo_dur / unlock_dur, 1.0)
            else:
                feat["social_app_ratio"] = 0.0

            # ── Calls per day ─────────────────────────────────────────
            calls_in = sum(_safe(row, f"call_in_num_ep_{e}") for e in EPOCHS)
            calls_out = sum(_safe(row, f"call_out_num_ep_{e}") for e in EPOCHS)
            feat["calls_per_day"] = calls_in + calls_out

            # ── Texts per day ─────────────────────────────────────────
            sms_in = sum(_safe(row, f"sms_in_num_ep_{e}") for e in EPOCHS)
            sms_out = sum(_safe(row, f"sms_out_num_ep_{e}") for e in EPOCHS)
            feat["texts_per_day"] = sms_in + sms_out

            # ── Unique contacts ───────────────────────────────────────
            # Not available in CrossCheck — use call count as proxy
            feat["unique_contacts"] = feat["calls_per_day"]  # fallback

            # ── Response time (minutes) ───────────────────────────────
            # EMA response time available as proxy
            feat["response_time_minutes"] = _safe(row, "ema_resp_time_median")

            # ── Daily displacement (km) ───────────────────────────────
            feat["daily_displacement_km"] = sum(
                _safe(row, f"loc_dist_ep_{e}") for e in EPOCHS
            ) / 1000.0   # dataset stores in meters

            # ── Location entropy ──────────────────────────────────────
            # Derive from visit counts across epochs (Shannon entropy)
            visits = [_safe(row, f"loc_visit_num_ep_{e}") for e in EPOCHS]
            total = sum(visits)
            if total > 0:
                probs = [v / total for v in visits if v > 0]
                feat["location_entropy"] = -sum(
                    p * np.log2(p) for p in probs
                )
            else:
                feat["location_entropy"] = 0.0

            # ── Home time ratio ───────────────────────────────────────
            # ep_0 = overnight/home period → fraction of unlock time
            home_dur = _safe(row, "unlock_duration_ep_0")
            feat["home_time_ratio"] = (
                home_dur / unlock_dur if unlock_dur > 0 else 0.5
            )

            # ── Places visited ────────────────────────────────────────
            feat["places_visited"] = sum(
                1 for e in EPOCHS if _safe(row, f"loc_visit_num_ep_{e}") > 0
            )

            # ── Wake time ─────────────────────────────────────────────
            # sleep_end = minutes after midnight; 88 min = 1:28am
            sleep_end_min = _safe(row, "sleep_end")
            if 0 < sleep_end_min < 1440:
                feat["wake_time_hour"] = sleep_end_min / 60.0
            else:
                feat["wake_time_hour"] = np.nan

            # ── Sleep time ────────────────────────────────────────────
            # sleep_start = minutes after midnight
            sleep_start_min = _safe(row, "sleep_start")
            if 0 <= sleep_start_min < 1440:
                feat["sleep_time_hour"] = sleep_start_min / 60.0
            else:
                feat["sleep_time_hour"] = np.nan

            # ── Sleep duration ──────────────────────────────────────
            # sleep_duration is already in hours (not seconds)
            sleep_dur_h = _safe(row, "sleep_duration")
            if 0 < sleep_dur_h < 14:   # filter artifacts (e.g. 23.75 = broken sensor)
                feat["sleep_duration_hours"] = sleep_dur_h
            else:
                feat["sleep_duration_hours"] = np.nan

            # ── Dark duration (proxy: overnight inactive time) ────────
            # Use ep_0 (midnight-6am) unlock duration as dark proxy
            feat["dark_duration_hours"] = (
                (86400 - unlock_dur) / 3600.0  # total non-screen time
                if unlock_dur > 0 else np.nan
            )

            # ── Charge duration ───────────────────────────────────────
            # Not in CrossCheck — estimate from screen-off time
            feat["charge_duration_hours"] = np.nan

            # ── Conversation duration ─────────────────────────────────
            feat["conversation_duration_hours"] = convo_dur / 3600.0

            # ── Conversation frequency ────────────────────────────────
            feat["conversation_frequency"] = sum(
                _safe(row, f"audio_convo_num_ep_{e}") for e in EPOCHS
            )

            rows.append(feat)

        result = pd.DataFrame(rows)
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)
        return result

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """Extract features for all 90 patients."""
        results = {}
        for uid in self.get_patient_ids():
            try:
                df = self.extract_patient(uid)
                if len(df) > 0:
                    results[uid] = df
            except Exception as e:
                warnings.warn(f"Failed to extract {uid}: {e}")
        return results

    def load_ema_labels(self) -> Dict[str, Dict]:
        """
        Compute per-patient EMA symptom averages.

        Returns {uid: {avg_voices, avg_seeing, avg_depressed, avg_neg_score,
                       avg_pos_score, schz_flag, depression_flag}}.

        schz_flag = True if avg(VOICES + SEEING_THINGS) > 0.5
        depression_flag = True if avg(DEPRESSED) >= 1.0
        """
        df = self._load()
        results = {}
        for uid, group in df.groupby("eureka_id"):
            ema = group[
                ["ema_VOICES", "ema_SEEING_THINGS", "ema_DEPRESSED",
                 "ema_neg_score", "ema_pos_score", "ema_score"]
            ].mean()

            avg_voices = float(ema.get("ema_VOICES", 0) or 0)
            avg_seeing = float(ema.get("ema_SEEING_THINGS", 0) or 0)
            avg_depressed = float(ema.get("ema_DEPRESSED", 0) or 0)

            results[uid] = {
                "avg_voices": round(avg_voices, 2),
                "avg_seeing": round(avg_seeing, 2),
                "avg_depressed": round(avg_depressed, 2),
                "avg_neg_score": round(float(ema.get("ema_neg_score", 0) or 0), 2),
                "avg_pos_score": round(float(ema.get("ema_pos_score", 0) or 0), 2),
                "avg_ema_score": round(float(ema.get("ema_score", 0) or 0), 2),
                # Clinical flags
                "schz_flag": (avg_voices + avg_seeing) > 0.5,
                "depression_flag": avg_depressed >= 1.0,
            }
        return results


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Return numeric value or default if missing/NaN."""
    val = row.get(col, default)
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Quick CLI test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ext = SchzExtractor()
    ids = ext.get_patient_ids()
    print(f"Patients found: {len(ids)}")

    uid = ids[0]
    print(f"\nExtracting {uid}...")
    df = ext.extract_patient(uid)
    print(f"  Days: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print()
    print(df.describe().round(2).to_string())

    # EMA labels
    labels = ext.load_ema_labels()
    print(f"\nEMA labels computed for {len(labels)} patients")

    # Show schz vs non-schz
    schz_pat = [u for u, v in labels.items() if v["schz_flag"]]
    non_schz = [u for u, v in labels.items() if not v["schz_flag"]]
    print(f"  Schz flag (VOICES+SEEING > 0.5): {len(schz_pat)} patients")
    print(f"  Non-schz: {len(non_schz)} patients")

    if ids[0] in labels:
        print(f"\n  {uid}: {labels[uid]}")

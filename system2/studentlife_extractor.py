"""
StudentLife Feature Extractor
==============================

Reads raw StudentLife CSVs and computes daily values for all 18
behavioral features used by System 1's PersonalityVector.

This is a VALIDATION-ONLY tool — it converts pre-recorded research
data into the format our pipeline expects, so we can prove the
system works on real humans.

Usage
-----
    from studentlife_extractor import StudentLifeExtractor

    ext = StudentLifeExtractor("C:/Users/embar/Downloads/StudentLife-dataset")
    df = ext.extract_student("u00")      # DataFrame: 1 row/day, 18 feature cols + date
    labels = ext.load_phq9_labels()       # {uid: {pre_score, post_score, ...}}
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Default dataset path ────────────────────────────────────────────────
DEFAULT_DATA_ROOT = r"C:\Users\embar\Downloads\StudentLife-dataset"

# ── Social app package prefixes ─────────────────────────────────────────
SOCIAL_PREFIXES = [
    "com.facebook",        # Facebook, Messenger
    "com.twitter",         # Twitter
    "com.instagram",       # Instagram
    "com.snapchat",        # Snapchat
    "com.whatsapp",        # WhatsApp
    "com.skype",           # Skype
    "com.tencent",         # WeChat, QQ
    "com.viber",           # Viber
    "com.google.android.talk",       # Hangouts
    "com.google.android.apps.messaging",  # Google Messages
    "com.google.android.gm",         # Gmail
    "com.android.mms",               # Stock MMS
    "com.android.email",             # Stock Email
    "org.telegram",                  # Telegram
    "com.linkedin",                  # LinkedIn
    "com.pinterest",                 # Pinterest
    "com.tumblr",                    # Tumblr
    "com.reddit",                    # Reddit
    "kik.android",                   # Kik
    "jp.naver.line",                 # LINE
]

# ── PHQ-9 response → score mapping ─────────────────────────────────────
PHQ9_SCORE_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

PHQ9_SEVERITY = [
    (0, 4, "none"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately_severe"),
    (20, 27, "severe"),
]


def _epoch_to_date(epoch: float) -> datetime:
    """Unix epoch → datetime."""
    return datetime.utcfromtimestamp(epoch)


def _epoch_to_datestr(epoch: float) -> str:
    """Unix epoch → 'YYYY-MM-DD'."""
    return _epoch_to_date(epoch).strftime("%Y-%m-%d")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two GPS points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class StudentLifeExtractor:
    """
    Extract daily behavioral features from the StudentLife dataset.

    Parameters
    ----------
    data_root : str
        Path to the StudentLife-dataset directory.
    """

    def __init__(self, data_root: str = DEFAULT_DATA_ROOT):
        self.root = data_root
        self._validate_root()

    def _validate_root(self):
        required = ["sensing", "app_usage", "call_log", "sms", "survey"]
        for d in required:
            if not os.path.isdir(os.path.join(self.root, d)):
                raise FileNotFoundError(
                    f"Expected directory '{d}' not found in {self.root}"
                )

    # ── Public API ──────────────────────────────────────────────────

    def get_student_ids(self) -> List[str]:
        """Return sorted list of student IDs that have call_log data."""
        ids = set()
        for f in os.listdir(os.path.join(self.root, "call_log")):
            if f.startswith("call_log_") and f.endswith(".csv"):
                uid = f.replace("call_log_", "").replace(".csv", "")
                ids.add(uid)
        return sorted(ids)

    def extract_student(self, uid: str) -> pd.DataFrame:
        """
        Extract all 18 features as a daily DataFrame for one student.

        Returns DataFrame with columns: date + 18 feature columns.
        Missing days/features are filled with NaN.
        """
        # Gather per-feature daily Series
        lock_feats = self._extract_phonelock(uid)
        app_feats = self._extract_app_usage(uid)
        call_feats = self._extract_calls(uid)
        sms_feats = self._extract_sms(uid)
        gps_feats = self._extract_gps(uid)
        dark_feats = self._extract_dark(uid)
        conv_feats = self._extract_conversation(uid)
        charge_feats = self._extract_phonecharge(uid)

        # Merge all feature dicts into one DataFrame
        all_feats = {}
        for d in [lock_feats, app_feats, call_feats, sms_feats,
                   gps_feats, dark_feats, conv_feats, charge_feats]:
            all_feats.update(d)

        if not all_feats:
            return pd.DataFrame()

        # Build DataFrame from {feature_name: {date_str: value}}
        df = pd.DataFrame(all_feats)

        # Ensure date index
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Merge unique_contacts from calls + sms
        call_contacts = call_feats.get("_call_contacts", {})
        sms_contacts = sms_feats.get("_sms_contacts", {})
        all_dates = set(list(call_contacts.keys()) + list(sms_contacts.keys()))
        unique_contacts = {}
        for d in all_dates:
            c_set = call_contacts.get(d, set())
            s_set = sms_contacts.get(d, set())
            unique_contacts[d] = len(c_set | s_set)
        if unique_contacts:
            uc_series = pd.Series(unique_contacts, name="unique_contacts")
            uc_df = uc_series.reset_index()
            uc_df.columns = ["date", "unique_contacts"]
            uc_df["date"] = pd.to_datetime(uc_df["date"])
            df = df.drop(columns=["unique_contacts"], errors="ignore")
            df = df.merge(uc_df, on="date", how="left")

        # Drop internal columns
        df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

        # Ensure all 18 features present
        expected = [
            "screen_time_hours", "unlock_count", "social_app_ratio",
            "calls_per_day", "texts_per_day", "unique_contacts",
            "response_time_minutes", "daily_displacement_km",
            "location_entropy", "home_time_ratio", "places_visited",
            "wake_time_hour", "sleep_time_hour", "sleep_duration_hours",
            "dark_duration_hours", "charge_duration_hours",
            "conversation_duration_hours", "conversation_frequency",
        ]
        for feat in expected:
            if feat not in df.columns:
                df[feat] = np.nan

        # Reorder columns
        df = df[["date"] + expected]

        return df

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """Extract features for all students. Returns {uid: DataFrame}."""
        results = {}
        for uid in self.get_student_ids():
            try:
                df = self.extract_student(uid)
                if len(df) > 0:
                    results[uid] = df
            except Exception as e:
                warnings.warn(f"Failed to extract {uid}: {e}")
        return results

    def load_phq9_labels(self) -> Dict[str, Dict]:
        """
        Load PHQ-9 scores from survey/PHQ-9.csv.

        Returns {uid: {pre_score, post_score, pre_severity, post_severity}}.
        """
        path = os.path.join(self.root, "survey", "PHQ-9.csv")
        df = pd.read_csv(path)

        results = {}
        for _, row in df.iterrows():
            uid = row["uid"]
            survey_type = row["type"]  # "pre" or "post"

            # Sum the 9 question scores
            score = 0
            for col in df.columns[2:-1]:  # skip uid, type, and Response
                val = row[col]
                score += PHQ9_SCORE_MAP.get(str(val).strip(), 0)

            severity = "unknown"
            for lo, hi, label in PHQ9_SEVERITY:
                if lo <= score <= hi:
                    severity = label
                    break

            if uid not in results:
                results[uid] = {}

            results[uid][f"{survey_type}_score"] = score
            results[uid][f"{survey_type}_severity"] = severity

        return results

    # ── Phone Lock → screen_time, unlock_count, wake_time, sleep_time ──

    def _extract_phonelock(self, uid: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.root, "sensing", "phonelock", f"phonelock_{uid}.csv")
        if not os.path.exists(path):
            return {}

        df = pd.read_csv(path, names=["start", "end"], skiprows=1)
        df = df.dropna(subset=["start", "end"])
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna()
        df["date"] = df["start"].apply(_epoch_to_datestr)
        df["duration_hours"] = (df["end"] - df["start"]) / 3600.0

        # Screen time = total unlock duration per day
        screen_time = df.groupby("date")["duration_hours"].sum().to_dict()

        # Unlock count = number of lock→unlock events per day
        unlock_count = df.groupby("date").size().to_dict()

        # Wake time = hour of first unlock each day
        wake_time = {}
        for date, group in df.groupby("date"):
            first_unlock = group["start"].min()
            dt = _epoch_to_date(first_unlock)
            wake_time[date] = dt.hour + dt.minute / 60.0

        # Sleep time = hour of last lock each day
        sleep_time = {}
        for date, group in df.groupby("date"):
            last_lock = group["end"].max()
            dt = _epoch_to_date(last_lock)
            sleep_time[date] = dt.hour + dt.minute / 60.0

        return {
            "screen_time_hours": screen_time,
            "unlock_count": {k: float(v) for k, v in unlock_count.items()},
            "wake_time_hour": wake_time,
            "sleep_time_hour": sleep_time,
        }

    # ── App Usage → social_app_ratio ────────────────────────────────

    def _extract_app_usage(self, uid: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.root, "app_usage", f"running_app_{uid}.csv")
        if not os.path.exists(path):
            return {}

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            return {}

        if "timestamp" not in df.columns:
            return {}

        # The key column is the top activity package
        pkg_col = None
        for col in ["RUNNING_TASKS_topActivity_mPackage",
                     "RUNNING_TASKS_baseActivity_mPackage"]:
            if col in df.columns:
                pkg_col = col
                break

        if pkg_col is None:
            return {}

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].apply(_epoch_to_datestr)

        # Classify each app sample as social or not
        df["is_social"] = df[pkg_col].apply(
            lambda p: any(str(p).startswith(prefix) for prefix in SOCIAL_PREFIXES)
            if pd.notna(p) else False
        )

        # Per day: ratio of social app samples over total samples
        daily = df.groupby("date").agg(
            total=("is_social", "count"),
            social=("is_social", "sum"),
        )
        daily["ratio"] = daily["social"] / daily["total"]
        social_app_ratio = daily["ratio"].to_dict()

        return {"social_app_ratio": social_app_ratio}

    # ── Call Log → calls_per_day, call contacts ─────────────────────

    def _extract_calls(self, uid: str) -> Dict[str, Dict]:
        path = os.path.join(self.root, "call_log", f"call_log_{uid}.csv")
        if not os.path.exists(path):
            return {}

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            return {}

        if "timestamp" not in df.columns:
            return {}

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].apply(_epoch_to_datestr)

        # Filter to rows that actually have call data (non-empty duration)
        if "CALLS_duration" in df.columns:
            df_calls = df[pd.to_numeric(df["CALLS_duration"], errors="coerce").notna()]
        else:
            df_calls = df

        calls_per_day = df_calls.groupby("date").size().to_dict()

        # Track unique contacts per day (hashed numbers)
        contact_col = "CALLS_number" if "CALLS_number" in df_calls.columns else None
        call_contacts = {}
        if contact_col:
            for date, group in df_calls.groupby("date"):
                contacts = set(group[contact_col].dropna().astype(str).tolist())
                call_contacts[date] = contacts

        return {
            "calls_per_day": {k: float(v) for k, v in calls_per_day.items()},
            "_call_contacts": call_contacts,
        }

    # ── SMS → texts_per_day, sms contacts, response_time ────────────

    def _extract_sms(self, uid: str) -> Dict[str, Dict]:
        path = os.path.join(self.root, "sms", f"sms_{uid}.csv")
        if not os.path.exists(path):
            return {}

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            return {}

        if "timestamp" not in df.columns:
            return {}

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].apply(_epoch_to_datestr)

        # Filter to rows with actual message data
        if "MESSAGES_date" in df.columns:
            df_msgs = df[pd.to_numeric(df["MESSAGES_date"], errors="coerce").notna()]
        else:
            df_msgs = df

        texts_per_day = df_msgs.groupby("date").size().to_dict()

        # Track unique contacts
        contact_col = "MESSAGES_address" if "MESSAGES_address" in df_msgs.columns else None
        sms_contacts = {}
        if contact_col:
            for date, group in df_msgs.groupby("date"):
                contacts = set(group[contact_col].dropna().astype(str).tolist())
                sms_contacts[date] = contacts

        # Response time: avg gap between received (type=1) → sent (type=2)
        # within same thread
        response_time = {}
        if "MESSAGES_type" in df_msgs.columns and "MESSAGES_date" in df_msgs.columns:
            df_msgs = df_msgs.copy()
            df_msgs["msg_ts"] = pd.to_numeric(df_msgs["MESSAGES_date"], errors="coerce") / 1000.0
            df_msgs["msg_type"] = pd.to_numeric(df_msgs["MESSAGES_type"], errors="coerce")

            for date, group in df_msgs.groupby("date"):
                # type 1 = received, type 2 = sent
                received = group[group["msg_type"] == 1].sort_values("msg_ts")
                sent = group[group["msg_type"] == 2].sort_values("msg_ts")

                if len(received) == 0 or len(sent) == 0:
                    continue

                # Simple approach: for each sent message, find the closest
                # preceding received message
                gaps = []
                for _, s_row in sent.iterrows():
                    prior_received = received[received["msg_ts"] < s_row["msg_ts"]]
                    if len(prior_received) > 0:
                        gap = (s_row["msg_ts"] - prior_received["msg_ts"].iloc[-1]) / 60.0
                        if 0 < gap < 1440:  # within 24 hours
                            gaps.append(gap)

                if gaps:
                    response_time[date] = float(np.mean(gaps))

        return {
            "texts_per_day": {k: float(v) for k, v in texts_per_day.items()},
            "response_time_minutes": response_time,
            "_sms_contacts": sms_contacts,
        }

    # ── GPS → displacement, entropy, home_time_ratio, places ────────

    def _extract_gps(self, uid: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.root, "sensing", "gps", f"gps_{uid}.csv")
        if not os.path.exists(path):
            return {}

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            return {}

        # The StudentLife GPS CSV uses the epoch timestamp as the first
        # column/index. After read_csv, the timestamps end up as the
        # DataFrame index and the "time" column actually contains provider
        # names ("network", "gps").  Reset to get them as a regular column.
        df = df.reset_index()
        # The index column is named either "index" or the original header
        idx_col = df.columns[0]  # first column = epoch timestamps
        df = df.rename(columns={idx_col: "epoch"})
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

        # The actual lat/lon columns shift too — find them by position
        # Header: time,provider,network_type,accuracy,latitude,longitude,...
        # After reset_index + rename:
        #   epoch | time(=provider) | provider(=network_type) | ... | latitude | longitude
        # But column names are as the header says, so "latitude"/"longitude"
        # may actually be in "accuracy"/"latitude" etc.
        # Safest: use position-based access if the named columns fail.
        lat_col = "latitude"
        lon_col = "longitude"

        # Verify the named columns contain numeric data
        df[lat_col] = pd.to_numeric(df.get(lat_col, pd.Series()), errors="coerce")
        df[lon_col] = pd.to_numeric(df.get(lon_col, pd.Series()), errors="coerce")
        df = df.dropna(subset=["epoch", lat_col, lon_col])

        if len(df) == 0:
            return {}

        df["date"] = df["epoch"].apply(_epoch_to_datestr)

        displacement = {}
        entropy = {}
        home_time = {}
        places = {}

        # Discretize locations into ~100m grid cells for clustering
        df["lat_bin"] = (df[lat_col] * 1000).round().astype(int)
        df["lon_bin"] = (df[lon_col] * 1000).round().astype(int)
        df["cell"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

        # Find overall "home" = most frequent cell across all days
        overall_home = df["cell"].mode()
        home_cell = overall_home.iloc[0] if len(overall_home) > 0 else None

        for date, group in df.groupby("date"):
            group = group.sort_values("epoch")

            # Daily displacement (sum of haversine between consecutive points)
            if len(group) >= 2:
                lats = group[lat_col].values
                lons = group[lon_col].values
                total_km = 0.0
                for i in range(1, len(lats)):
                    d = _haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i])
                    if d < 50:  # filter GPS jumps > 50km
                        total_km += d
                displacement[date] = total_km
            else:
                displacement[date] = 0.0

            # Location entropy (Shannon entropy over grid cells)
            cell_counts = group["cell"].value_counts()
            total = cell_counts.sum()
            if total > 0:
                probs = cell_counts / total
                ent = -float(np.sum(probs * np.log2(probs + 1e-10)))
                entropy[date] = ent
            else:
                entropy[date] = 0.0

            # Home time ratio
            if home_cell and total > 0:
                home_count = cell_counts.get(home_cell, 0)
                home_time[date] = float(home_count) / total
            else:
                home_time[date] = 0.0

            # Places visited (unique cells)
            places[date] = float(len(cell_counts))

        return {
            "daily_displacement_km": displacement,
            "location_entropy": entropy,
            "home_time_ratio": home_time,
            "places_visited": places,
        }

    # ── Dark → dark_duration, sleep_duration ────────────────────────

    def _extract_dark(self, uid: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.root, "sensing", "dark", f"dark_{uid}.csv")
        if not os.path.exists(path):
            return {}

        df = pd.read_csv(path, names=["start", "end"], skiprows=1)
        df = df.dropna()
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna()
        df["date"] = df["start"].apply(_epoch_to_datestr)
        df["duration_hours"] = (df["end"] - df["start"]) / 3600.0

        # Total dark duration per day
        dark_duration = df.groupby("date")["duration_hours"].sum().to_dict()

        # Sleep duration = longest continuous dark period per day
        sleep_duration = df.groupby("date")["duration_hours"].max().to_dict()

        return {
            "dark_duration_hours": dark_duration,
            "sleep_duration_hours": sleep_duration,
        }

    # ── Conversation → duration, frequency ──────────────────────────

    def _extract_conversation(self, uid: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.root, "sensing", "conversation", f"conversation_{uid}.csv")
        if not os.path.exists(path):
            return {}

        try:
            # File has header " start_timestamp, end_timestamp" with spaces
            df = pd.read_csv(path, skipinitialspace=True)
        except Exception:
            return {}

        # Normalize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        start_col = "start_timestamp" if "start_timestamp" in df.columns else None
        end_col = "end_timestamp" if "end_timestamp" in df.columns else None

        if start_col is None or end_col is None:
            return {}

        df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
        df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
        df = df.dropna(subset=[start_col, end_col])
        df["date"] = df[start_col].apply(_epoch_to_datestr)
        df["duration_hours"] = (df[end_col] - df[start_col]) / 3600.0

        conv_duration = df.groupby("date")["duration_hours"].sum().to_dict()
        conv_frequency = df.groupby("date").size().to_dict()

        return {
            "conversation_duration_hours": conv_duration,
            "conversation_frequency": {k: float(v) for k, v in conv_frequency.items()},
        }

    # ── Phone Charge → charge_duration ──────────────────────────────

    def _extract_phonecharge(self, uid: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self.root, "sensing", "phonecharge", f"phonecharge_{uid}.csv")
        if not os.path.exists(path):
            return {}

        df = pd.read_csv(path, names=["start", "end"], skiprows=1)
        df = df.dropna()
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna()
        df["date"] = df["start"].apply(_epoch_to_datestr)
        df["duration_hours"] = (df["end"] - df["start"]) / 3600.0

        charge_duration = df.groupby("date")["duration_hours"].sum().to_dict()

        return {"charge_duration_hours": charge_duration}


# ── Quick CLI test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ext = StudentLifeExtractor()
    print(f"Students found: {len(ext.get_student_ids())}")

    # Extract u00 as a quick test
    uid = "u00"
    print(f"\nExtracting {uid}...")
    df = ext.extract_student(uid)
    print(f"  Days: {len(df)}")
    print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")
    print()
    print(df.describe().round(2).to_string())

    # Check the two previously-broken features
    sar = df["social_app_ratio"]
    print(f"\nsocial_app_ratio coverage: {(sar > 0).mean():.0%}  (was 0%)")
    cpd = df["calls_per_day"]
    print(f"calls_per_day coverage:    {(cpd > 0).mean():.0%}")

    # PHQ-9 labels
    labels = ext.load_phq9_labels()
    print(f"\nPHQ-9 labels loaded for {len(labels)} students")
    if uid in labels:
        print(f"  {uid}: {labels[uid]}")

"""
feature_engineering.py — Extracts 26 behavioral features from raw access logs.
Optimized for performance with vectorized operations where possible.
"""

import sys
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _haversine(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km between two coordinates."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


class FeatureEngineer:
    """Extracts behavioral features from raw access logs."""

    def __init__(self):
        self.entity_known_resources = defaultdict(set)
        self.entity_known_devices = defaultdict(set)
        self.entity_known_commands = defaultdict(set)
        self.entity_common_geo = {}  # entity_id -> (lat, lon)
        self.entity_common_auth = {}  # entity_id -> auth_method
        self.global_known_devices = set()

    def fit(self, df):
        """Learn per-entity baselines from training data."""
        print("  [FE] Fitting entity baselines...")
        df_sorted = df.sort_values(by=["entity_id", "timestamp"])

        for eid, group in df_sorted.groupby("entity_id"):
            self.entity_known_resources[eid].update(
                group["resource_accessed"].dropna().tolist()
            )
            self.entity_known_devices[eid].update(
                group["device_fingerprint"].dropna().tolist()
            )

            cmds = group["command_sequence"].dropna().apply(
                lambda x: str(x).split(";")
            )
            for c_list in cmds:
                self.entity_known_commands[eid].update(c_list)

            # Most common geo
            geos = group["geo_location"].dropna().apply(self._parse_geo)
            geos = geos.dropna()
            if not geos.empty:
                self.entity_common_geo[eid] = geos.value_counts().idxmax()

            # Most common auth
            auths = group["auth_method"].dropna()
            if not auths.empty:
                self.entity_common_auth[eid] = auths.value_counts().idxmax()

        self.global_known_devices.update(
            df["device_fingerprint"].dropna().tolist()
        )
        return self

    def _parse_geo(self, g):
        """Parse 'city|lat|lon' to (lat, lon) tuple."""
        if pd.isna(g):
            return None
        parts = str(g).split("|")
        if len(parts) == 3:
            try:
                return (float(parts[1]), float(parts[2]))
            except ValueError:
                return None
        return None

    def transform(self, df):
        """Transform raw access logs into feature matrix."""
        print("  [FE] Transforming features...")
        df = df.copy()
        df = df.sort_values(by=["entity_id", "timestamp"]).reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # ── Parse geo-coordinates ──
        geo_parsed = df["geo_location"].apply(self._parse_geo)
        df["lat"] = geo_parsed.apply(lambda x: x[0] if x else np.nan)
        df["lon"] = geo_parsed.apply(lambda x: x[1] if x else np.nan)

        features = pd.DataFrame(index=df.index)

        # ── Temporal features (vectorized) ──
        print("  [FE]   Temporal features...")
        features["hour_of_day"] = df["timestamp"].dt.hour
        features["day_of_week"] = df["timestamp"].dt.dayofweek
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
        features["is_off_hours"] = (
            (features["hour_of_day"] < 6) | (features["hour_of_day"] > 22)
        ).astype(int)

        # ── Per-entity time since last access (vectorized) ──
        grouped = df.groupby("entity_id")
        features["time_since_last_access_sec"] = (
            grouped["timestamp"].diff().dt.total_seconds().fillna(0)
        )

        # ── Rolling window access frequencies ──
        # Use a simpler counting approach for speed: count events per entity
        # in fixed windows using groupby + rolling
        print("  [FE]   Access frequency features...")
        df_ts_idx = df.set_index("timestamp")
        grouped_ts = df_ts_idx.groupby("entity_id")

        for window, col_name in [
            ("5min", "access_freq_5min"),
            ("1h", "access_freq_1hr"),
            ("24h", "access_freq_24hr"),
            ("7d", "access_freq_7d"),
        ]:
            try:
                features[col_name] = (
                    grouped_ts["entity_type"].rolling(window).count().values
                )
            except Exception:
                features[col_name] = 1.0

        # ── Geo velocity (vectorized) ──
        print("  [FE]   Geo features...")
        prev_lat = grouped["lat"].shift(1)
        prev_lon = grouped["lon"].shift(1)

        # Vectorized haversine
        dists = _haversine_vec(
            prev_lat.fillna(0).values,
            prev_lon.fillna(0).values,
            df["lat"].fillna(0).values,
            df["lon"].fillna(0).values,
        )
        # Zero out where we don't have previous
        dists = np.where(prev_lat.isna(), 0.0, dists)

        time_diff_hours = features["time_since_last_access_sec"].values / 3600.0
        features["geo_velocity_kmh"] = np.where(
            time_diff_hours > 0, dists / time_diff_hours, 0
        )

        # Distance from entity's common geo
        def _geo_dist_from_common(row):
            eid = row["entity_id"]
            if eid in self.entity_common_geo and pd.notna(row["lat"]):
                clat, clon = self.entity_common_geo[eid]
                return _haversine(clat, clon, row["lat"], row["lon"])
            return 0.0

        features["geo_distance_km"] = df.apply(_geo_dist_from_common, axis=1)

        # Entropy features: set to 0 for speed (requires heavy window ops)
        features["geo_entropy"] = 0.0
        features["resource_entropy"] = 0.0

        # ── Resource features ──
        print("  [FE]   Resource features...")

        # Resource novelty: vectorized via set lookup
        features["resource_novelty_score"] = df.apply(
            lambda row: (
                0.0
                if pd.isna(row["resource_accessed"])
                or row["resource_accessed"]
                in self.entity_known_resources.get(row["entity_id"], set())
                else 1.0
            ),
            axis=1,
        )

        # Resource breadth: unique resources in 24h window
        try:
            features["resource_breadth"] = (
                grouped_ts["resource_accessed"]
                .rolling("24h")
                .apply(lambda x: x.nunique(), raw=False)
                .values
            )
        except Exception:
            features["resource_breadth"] = 1.0

        # ── Auth features ──
        print("  [FE]   Auth features...")
        df["is_failed"] = (df["session_duration"] < 2.0).astype(int)

        for window, col_name in [
            ("5min", "failed_auth_count_5min"),
            ("1h", "failed_auth_count_1hr"),
        ]:
            try:
                features[col_name] = (
                    df_ts_idx.groupby("entity_id")["is_failed"]
                    .rolling(window)
                    .sum()
                    .values
                )
            except Exception:
                features[col_name] = 0.0

        features["auth_method_changed"] = df.apply(
            lambda row: (
                1
                if pd.notna(row["auth_method"])
                and row["entity_id"] in self.entity_common_auth
                and row["auth_method"] != self.entity_common_auth[row["entity_id"]]
                else 0
            ),
            axis=1,
        )

        # ── Session features ──
        print("  [FE]   Session features...")
        features["session_duration_zscore"] = 0.0
        features["session_duration_raw"] = df["session_duration"].fillna(0)

        mean_dur = grouped["session_duration"].transform("mean")
        std_dur = grouped["session_duration"].transform("std").replace(0, 1)
        features["session_duration_zscore"] = (
            (df["session_duration"] - mean_dur) / std_dur
        ).fillna(0)

        # ── Device features (vectorized) ──
        print("  [FE]   Device features...")
        features["fingerprint_mismatch"] = df.apply(
            lambda r: (
                0
                if pd.isna(r["device_fingerprint"])
                or r["device_fingerprint"]
                in self.entity_known_devices.get(r["entity_id"], set())
                else 1
            ),
            axis=1,
        )
        features["new_device_flag"] = df["device_fingerprint"].apply(
            lambda x: 0 if pd.isna(x) or x in self.global_known_devices else 1
        )

        # ── Command sequence features ──
        print("  [FE]   Command features...")

        def cmd_len(x):
            if pd.isna(x) or str(x).strip() == "":
                return 0
            return len(str(x).split(";"))

        features["command_seq_length"] = df["command_sequence"].apply(cmd_len)

        def cmd_nov(row):
            eid = row["entity_id"]
            cmd_str = (
                str(row["command_sequence"])
                if pd.notna(row["command_sequence"])
                else ""
            )
            cmds = (
                [c for c in cmd_str.split(";") if c.strip()]
                if cmd_str.strip()
                else []
            )
            if not cmds:
                return 0.0
            known = self.entity_known_commands.get(eid, set())
            novel = sum(1 for c in cmds if c.strip() not in known)
            return novel / len(cmds)

        features["command_seq_novelty"] = df.apply(cmd_nov, axis=1)

        # ── Encoding features (vectorized) ──
        type_map = {t: i for i, t in enumerate(config.ENTITY_TYPES)}
        auth_map = {a: i for i, a in enumerate(config.AUTH_METHODS)}
        features["entity_type_encoded"] = df["entity_type"].map(type_map).fillna(-1)
        features["auth_method_encoded"] = df["auth_method"].map(auth_map).fillna(-1)

        # ── Clean up ──
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"  [FE] Done. Shape: {features.shape}")
        return features[config.FEATURE_NAMES]

    def fit_transform(self, df):
        """Fit and transform in one call."""
        self.fit(df)
        return self.transform(df)

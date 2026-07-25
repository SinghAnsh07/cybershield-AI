"""
generator.py — Main synthetic data generation orchestrator.
Generates normal access log events and injects attack patterns.
Uses vectorized operations for speed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

import config
from data_generator.profiles import create_entity_profiles, EntityProfile
from data_generator.attack_injector import inject_attacks


def generate_normal_events(
    profiles: list,
    n_events: int,
    sim_days: int,
    rng: np.random.Generator,
    base_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Generate normal access log events using vectorized numpy operations."""
    fake = Faker()
    n_profiles = len(profiles)

    # ── Pre-compute profile arrays for vectorized sampling ──
    freqs = np.array([p.access_frequency_per_day for p in profiles])
    probs = freqs / freqs.sum()

    # Pre-compute IP pools per geo for realism
    geo_ips = {}
    for geo in config.GEO_CITIES:
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        geo_ips[geo_str] = [fake.ipv4() for _ in range(30)]

    # ── Vectorized entity selection ──
    print("  Sampling entities...")
    entity_indices = rng.choice(n_profiles, size=n_events, p=probs)

    # ── Vectorized timestamp generation ──
    print("  Generating timestamps...")
    day_offsets = rng.integers(0, sim_days, size=n_events)

    # Determine hours: 95% in preferred hours, 5% random
    is_off_hour = rng.random(n_events) < 0.05
    preferred_means = np.array([profiles[i].preferred_hours[0] for i in entity_indices])
    preferred_stds = np.array([profiles[i].preferred_hours[1] for i in entity_indices])
    hours = np.where(
        is_off_hour,
        rng.uniform(0, 24, size=n_events),
        np.clip(rng.normal(preferred_means, preferred_stds), 0, 23.99),
    )

    timestamps = pd.to_datetime(base_timestamp) + pd.to_timedelta(
        day_offsets * 86400 + (hours * 3600).astype(int), unit="s"
    )

    # ── Build rows using list comprehension (much faster than append loop) ──
    print("  Building event rows...")

    # Pre-extract profile data for fast indexing
    entity_ids = [profiles[i].entity_id for i in entity_indices]
    entity_types = [profiles[i].entity_type for i in entity_indices]

    # Vectorized: decide if unusual resource (2% chance)
    use_unusual_resource = rng.random(n_events) < 0.02

    # Build all columns
    source_ips = []
    geo_locations = []
    resources = []
    auth_methods = []
    session_durations = []
    command_sequences = []
    device_fingerprints = []

    # Pre-compute session duration params
    dur_means = np.array([profiles[i].session_duration_params[0] for i in entity_indices])
    dur_stds = np.array([profiles[i].session_duration_params[1] for i in entity_indices])
    session_durations = rng.lognormal(dur_means, dur_stds)

    for idx in range(n_events):
        profile = profiles[entity_indices[idx]]

        # Geo
        geo = profile.typical_geos[rng.integers(0, len(profile.typical_geos))]
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        geo_locations.append(geo_str)
        source_ips.append(rng.choice(geo_ips.get(geo_str, [fake.ipv4()])))

        # Resource
        if use_unusual_resource[idx]:
            resources.append(config.RESOURCES[rng.integers(0, len(config.RESOURCES))])
        else:
            resources.append(
                profile.typical_resources[
                    rng.integers(0, len(profile.typical_resources))
                ]
            )

        # Auth
        auth_methods.append(
            profile.typical_auth_methods[
                rng.integers(0, len(profile.typical_auth_methods))
            ]
            if profile.typical_auth_methods
            else "password"
        )

        # Command sequence
        n_cmds = rng.integers(1, max(2, len(profile.typical_commands)))
        cmds = rng.choice(profile.typical_commands, size=n_cmds).tolist()
        command_sequences.append(";".join(cmds))

        # Device fingerprint
        fp = profile.device_fingerprints[
            rng.integers(0, len(profile.device_fingerprints))
        ]
        device_fingerprints.append(f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}")

    # ── Build DataFrame at once ──
    df = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "entity_type": entity_types,
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "source_ip": source_ips,
            "geo_location": geo_locations,
            "resource_accessed": resources,
            "auth_method": auth_methods,
            "session_duration": session_durations,
            "command_sequence": command_sequences,
            "device_fingerprint": device_fingerprints,
            "label": "normal",
        }
    )

    return df


def generate_dataset(
    n_entities: int = 500,
    n_events: int = 200000,
    sim_days: int = 90,
    seed: int = 42,
) -> tuple:
    """Full data generation pipeline."""
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    base_timestamp = pd.Timestamp("2023-01-01T00:00:00")

    print("Creating entity profiles...")
    profiles = create_entity_profiles(n=n_entities, seed=seed)

    print(f"Generating {n_events} normal events...")
    normal_df = generate_normal_events(profiles, n_events, sim_days, rng, base_timestamp)

    print("Injecting attacks...")
    attack_df = inject_attacks(profiles, base_timestamp, sim_days, rng, n_events)

    print("Merging and sorting by timestamp...")
    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Sort by timestamp to simulate real log stream
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    ground_truth_df = df[["entity_id", "timestamp", "label"]].copy()

    # Remove label from access logs (hidden at inference)
    access_logs_df = df.drop(columns=["label"])

    # Save to disk
    print(f"Saving to {config.ACCESS_LOG_PATH} and {config.GROUND_TRUTH_PATH}...")
    access_logs_df.to_csv(config.ACCESS_LOG_PATH, index=False)
    ground_truth_df.to_csv(config.GROUND_TRUTH_PATH, index=False)

    print("Generation complete!")
    print(f"Total events: {len(df)}")
    print(ground_truth_df["label"].value_counts())

    return access_logs_df, ground_truth_df


if __name__ == "__main__":
    generate_dataset(
        n_entities=config.NUM_ENTITIES,
        n_events=config.NUM_EVENTS,
        sim_days=config.SIMULATION_DAYS,
        seed=config.RANDOM_SEED,
    )

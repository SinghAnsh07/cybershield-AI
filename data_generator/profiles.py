import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from faker import Faker

import config

@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str
    preferred_hours: Tuple[float, float]
    typical_geos: List[Tuple[str, float, float]]
    typical_resources: List[str]
    typical_auth_methods: List[str]
    session_duration_params: Tuple[float, float]
    device_fingerprints: List[Dict[str, str]]
    typical_commands: List[str]
    access_frequency_per_day: float

def create_entity_profiles(n: int = 500, seed: int = 42) -> List[EntityProfile]:
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    profiles = []
    
    # Pre-define commands pool
    commands_pool = [
        "ls", "cd", "cat", "grep", "cp", "mv", "rm", "chmod", "chown", "ps", "top",
        "kill", "netstat", "ping", "curl", "wget", "ssh", "scp", "tar", "gzip",
        "mkdir", "rmdir", "find", "awk", "sed", "tail", "head", "less", "more"
    ]
    
    os_choices = ["Windows", "Linux", "macOS", "Android", "iOS"]
    protocol_choices = ["ssh", "https", "rdp", "api", "vpn"]

    for i in range(n):
        rand_val = rng.random()
        if rand_val < 0.7:
            entity_type = "user"
            entity_id = f"usr_{fake.uuid4()[:8]}"
            preferred_hours = (rng.uniform(7.0, 10.0), rng.uniform(1.0, 3.0)) # Starts around 7-10 AM
            num_geos = rng.integers(1, 3)
            typical_geos = [tuple(geo) for geo in rng.choice(config.GEO_CITIES, size=num_geos, replace=False)]
            num_res = rng.integers(3, 9)
            typical_resources = list(rng.choice(config.TYPICAL_RESOURCES[entity_type], size=min(num_res, len(config.TYPICAL_RESOURCES[entity_type])), replace=False))
            typical_auth_methods = list(rng.choice(["password", "token", "biometric"], size=rng.integers(1, 3), replace=False))
            session_duration_params = (rng.uniform(4.0, 6.0), rng.uniform(0.5, 1.0))
            num_devices = rng.integers(1, 4)
            access_frequency_per_day = rng.uniform(5, 20)
        elif rand_val < 0.9:
            entity_type = "service_account"
            entity_id = f"svc_{fake.uuid4()[:8]}"
            preferred_hours = (12.0, 12.0) # 24/7 hours, uniform across day
            typical_geos = [tuple(rng.choice(config.GEO_CITIES))]
            num_res = rng.integers(3, 9)
            typical_resources = list(rng.choice(config.TYPICAL_RESOURCES[entity_type], size=min(num_res, len(config.TYPICAL_RESOURCES[entity_type])), replace=False))
            typical_auth_methods = ["certificate", "token"][:rng.integers(1, 3)]
            session_duration_params = (rng.uniform(1.0, 3.0), rng.uniform(0.1, 0.5))
            num_devices = rng.integers(1, 3)
            access_frequency_per_day = rng.uniform(50, 200)
        else:
            entity_type = "edge_device"
            entity_id = f"dev_{fake.uuid4()[:8]}"
            preferred_hours = (12.0, 12.0) # 24/7 hours, regular intervals
            typical_geos = [tuple(rng.choice(config.GEO_CITIES))]
            num_res = rng.integers(3, 9)
            typical_resources = list(rng.choice(config.TYPICAL_RESOURCES[entity_type], size=min(num_res, len(config.TYPICAL_RESOURCES[entity_type])), replace=False))
            typical_auth_methods = ["certificate"]
            session_duration_params = (rng.uniform(2.0, 4.0), rng.uniform(0.1, 0.5))
            num_devices = 1
            access_frequency_per_day = rng.uniform(24, 144) # e.g. every hour to every 10 mins

        device_fingerprints = []
        for _ in range(num_devices):
            device_fingerprints.append({
                "os": str(rng.choice(os_choices)),
                "mac_address": fake.mac_address(),
                "protocol": str(rng.choice(protocol_choices))
            })

        num_cmds = rng.integers(3, 11)
        typical_commands = list(rng.choice(commands_pool, size=num_cmds, replace=False))
        
        # Adjust tuple elements from arrays properly
        if entity_type in ["service_account", "edge_device"]:
            # Make typical_geos list of tuples rather than nested arrays if rng.choice did something weird
            typical_geos = [(g[0], float(g[1]), float(g[2])) for g in typical_geos]
        else:
            typical_geos = [(g[0], float(g[1]), float(g[2])) for g in typical_geos]

        profile = EntityProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            preferred_hours=preferred_hours,
            typical_geos=typical_geos,
            typical_resources=typical_resources,
            typical_auth_methods=typical_auth_methods,
            session_duration_params=session_duration_params,
            device_fingerprints=device_fingerprints,
            typical_commands=typical_commands,
            access_frequency_per_day=access_frequency_per_day
        )
        profiles.append(profile)

    return profiles

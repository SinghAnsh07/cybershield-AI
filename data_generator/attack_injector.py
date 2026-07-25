import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from datetime import timedelta
import math
from typing import List, Dict, Any, Tuple
from faker import Faker
import config

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def _random_timestamp(base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator) -> pd.Timestamp:
    return base_timestamp + timedelta(seconds=int(rng.uniform(0, sim_days * 24 * 3600)))

def inject_brute_force(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        profile = rng.choice(profiles)
        start_time = _random_timestamp(base_timestamp, sim_days, rng)
        source_ip = fake.ipv4()
        geo = rng.choice(profile.typical_geos)
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        fp = rng.choice(profile.device_fingerprints)
        fp_str = f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}"
        res = rng.choice(profile.typical_resources)
        
        n_attempts = rng.integers(10, 50)
        curr_time = start_time
        for _ in range(n_attempts):
            rows.append({
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": curr_time.isoformat(),
                "source_ip": source_ip,
                "geo_location": geo_str,
                "resource_accessed": res,
                "auth_method": "password",
                "session_duration": rng.uniform(0.1, 1.0),
                "command_sequence": "",
                "device_fingerprint": fp_str,
                "label": "brute_force"
            })
            curr_time += timedelta(seconds=float(rng.uniform(2, 10)))
    return pd.DataFrame(rows)

def inject_impossible_travel(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        profile = rng.choice(profiles)
        start_time = _random_timestamp(base_timestamp, sim_days, rng)
        
        geo1 = rng.choice(profile.typical_geos)
        # Find a distant geo
        distant_geos = [g for g in config.GEO_CITIES if _haversine(float(geo1[1]), float(geo1[2]), float(g[1]), float(g[2])) > 5000]
        if not distant_geos:
            distant_geos = config.GEO_CITIES
        geo2 = rng.choice(distant_geos)
        
        geo1_str = f"{geo1[0]}|{geo1[1]}|{geo1[2]}"
        geo2_str = f"{geo2[0]}|{geo2[1]}|{geo2[2]}"
        
        fp = rng.choice(profile.device_fingerprints)
        fp_str = f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}"
        res = rng.choice(profile.typical_resources)
        auth = rng.choice(profile.typical_auth_methods) if profile.typical_auth_methods else "password"
        
        rows.append({
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": start_time.isoformat(),
            "source_ip": fake.ipv4(),
            "geo_location": geo1_str,
            "resource_accessed": res,
            "auth_method": auth,
            "session_duration": rng.lognormal(profile.session_duration_params[0], profile.session_duration_params[1]),
            "command_sequence": ";".join(rng.choice(profile.typical_commands, size=rng.integers(1, 4))),
            "device_fingerprint": fp_str,
            "label": "impossible_travel"
        })
        
        time2 = start_time + timedelta(minutes=float(rng.uniform(5, 30)))
        rows.append({
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": time2.isoformat(),
            "source_ip": fake.ipv4(),
            "geo_location": geo2_str,
            "resource_accessed": res,
            "auth_method": auth,
            "session_duration": rng.lognormal(profile.session_duration_params[0], profile.session_duration_params[1]),
            "command_sequence": ";".join(rng.choice(profile.typical_commands, size=rng.integers(1, 4))),
            "device_fingerprint": fp_str,
            "label": "impossible_travel"
        })
    return pd.DataFrame(rows)

def inject_credential_stuffing(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        source_ips = [fake.ipv4() for _ in range(rng.integers(1, 4))]
        start_time = _random_timestamp(base_timestamp, sim_days, rng)
        n_targets = rng.integers(20, 100)
        targets = rng.choice(profiles, size=n_targets)
        
        curr_time = start_time
        for target in targets:
            geo = rng.choice(target.typical_geos)
            geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
            fp = rng.choice(target.device_fingerprints)
            fp_str = f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}"
            res = rng.choice(target.typical_resources)
            
            rows.append({
                "entity_id": target.entity_id,
                "entity_type": target.entity_type,
                "timestamp": curr_time.isoformat(),
                "source_ip": rng.choice(source_ips),
                "geo_location": geo_str,
                "resource_accessed": res,
                "auth_method": "password",
                "session_duration": rng.uniform(0.1, 1.0),
                "command_sequence": "",
                "device_fingerprint": fp_str,
                "label": "credential_stuffing"
            })
            curr_time += timedelta(seconds=float(rng.uniform(1, 5)))
    return pd.DataFrame(rows)

def inject_lateral_movement(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        profile = rng.choice(profiles)
        start_time = _random_timestamp(base_timestamp, sim_days, rng)
        geo = rng.choice(profile.typical_geos)
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        fp = rng.choice(profile.device_fingerprints)
        fp_str = f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}"
        auth = rng.choice(profile.typical_auth_methods) if profile.typical_auth_methods else "password"
        
        unusual_resources = [r for r in config.RESOURCES if r not in profile.typical_resources]
        if not unusual_resources:
            unusual_resources = config.RESOURCES
            
        n_steps = rng.integers(5, 15)
        curr_time = start_time
        for _ in range(n_steps):
            res = rng.choice(unusual_resources)
            rows.append({
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": curr_time.isoformat(),
                "source_ip": fake.ipv4(),
                "geo_location": geo_str,
                "resource_accessed": res,
                "auth_method": auth,
                "session_duration": rng.lognormal(profile.session_duration_params[0], profile.session_duration_params[1]),
                "command_sequence": ";".join(rng.choice(profile.typical_commands, size=rng.integers(2, 6))),
                "device_fingerprint": fp_str,
                "label": "lateral_movement"
            })
            curr_time += timedelta(minutes=float(rng.uniform(1, 5)))
    return pd.DataFrame(rows)

def inject_device_spoofing(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        profile = rng.choice(profiles)
        start_time = _random_timestamp(base_timestamp, sim_days, rng)
        geo = rng.choice(profile.typical_geos)
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        auth = rng.choice(profile.typical_auth_methods) if profile.typical_auth_methods else "password"
        res = rng.choice(profile.typical_resources)
        
        # Completely different device
        fake_os = rng.choice(["TempleOS", "Symbian", "BlackBerry OS", "FreeBSD"])
        fake_mac = fake.mac_address()
        fake_protocol = "telnet"
        fp_str = f"{fake_os}|{fake_mac}|{fake_protocol}"
        
        rows.append({
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": start_time.isoformat(),
            "source_ip": fake.ipv4(),
            "geo_location": geo_str,
            "resource_accessed": res,
            "auth_method": auth,
            "session_duration": rng.lognormal(profile.session_duration_params[0], profile.session_duration_params[1]),
            "command_sequence": ";".join(rng.choice(profile.typical_commands, size=rng.integers(1, 4))),
            "device_fingerprint": fp_str,
            "label": "device_spoofing"
        })
    return pd.DataFrame(rows)

def inject_low_and_slow_exfiltration(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        profile = rng.choice(profiles)
        # Choose a start day leaving enough room
        start_day_offset = rng.integers(0, max(1, sim_days - 14))
        start_time = base_timestamp + timedelta(days=float(start_day_offset))
        geo = rng.choice(profile.typical_geos)
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        fp = rng.choice(profile.device_fingerprints)
        fp_str = f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}"
        auth = rng.choice(profile.typical_auth_methods) if profile.typical_auth_methods else "password"
        
        sensitive_res = [r for r in config.RESOURCES if "file://" in r]
        if not sensitive_res:
            sensitive_res = config.RESOURCES
            
        n_days = rng.integers(7, 15)
        for day in range(n_days):
            night_events = rng.integers(1, 4)
            for _ in range(night_events):
                event_time = start_time + timedelta(days=day, hours=float(rng.uniform(0, 5)))
                res = rng.choice(sensitive_res)
                rows.append({
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": event_time.isoformat(),
                    "source_ip": fake.ipv4(),
                    "geo_location": geo_str,
                    "resource_accessed": res,
                    "auth_method": auth,
                    "session_duration": rng.lognormal(profile.session_duration_params[0], profile.session_duration_params[1]),
                    "command_sequence": ";".join(rng.choice(profile.typical_commands, size=rng.integers(1, 4))),
                    "device_fingerprint": fp_str,
                    "label": "low_and_slow_exfiltration"
                })
    return pd.DataFrame(rows)

def inject_insider_drift(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, n_attacks: int) -> pd.DataFrame:
    fake = Faker()
    rows = []
    for _ in range(n_attacks):
        profile = rng.choice(profiles)
        start_day_offset = rng.integers(0, max(1, sim_days - 30))
        start_time = base_timestamp + timedelta(days=float(start_day_offset))
        geo = rng.choice(profile.typical_geos)
        geo_str = f"{geo[0]}|{geo[1]}|{geo[2]}"
        fp = rng.choice(profile.device_fingerprints)
        fp_str = f"{fp['os']}|{fp['mac_address']}|{fp['protocol']}"
        auth = rng.choice(profile.typical_auth_methods) if profile.typical_auth_methods else "password"
        
        unusual_resources = [r for r in config.RESOURCES if r not in profile.typical_resources]
        if not unusual_resources:
            unusual_resources = config.RESOURCES
            
        n_days = rng.integers(14, 31)
        drift_resources = []
        for day in range(n_days):
            if day % 7 == 0 and len(unusual_resources) > 0:
                drift_resources.append(rng.choice(unusual_resources))
                
            event_time = start_time + timedelta(days=day, hours=float(profile.preferred_hours[0] + rng.uniform(-2, 2)))
            res = rng.choice(drift_resources) if drift_resources and rng.random() < 0.5 else rng.choice(profile.typical_resources)
            
            rows.append({
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": event_time.isoformat(),
                "source_ip": fake.ipv4(),
                "geo_location": geo_str,
                "resource_accessed": res,
                "auth_method": auth,
                "session_duration": rng.lognormal(profile.session_duration_params[0], profile.session_duration_params[1]),
                "command_sequence": ";".join(rng.choice(profile.typical_commands, size=rng.integers(1, 4))),
                "device_fingerprint": fp_str,
                "label": "insider_drift"
            })
    return pd.DataFrame(rows)

def inject_attacks(profiles: List[Any], base_timestamp: pd.Timestamp, sim_days: int, rng: np.random.Generator, total_events: int) -> pd.DataFrame:
    all_attacks = []
    
    for attack_type, rate in config.ATTACK_RATES.items():
        n_attacks = max(1, int(total_events * rate / 10)) # approximate events per attack
        if attack_type == "brute_force":
            df = inject_brute_force(profiles, base_timestamp, sim_days, rng, n_attacks)
        elif attack_type == "impossible_travel":
            df = inject_impossible_travel(profiles, base_timestamp, sim_days, rng, n_attacks)
        elif attack_type == "credential_stuffing":
            df = inject_credential_stuffing(profiles, base_timestamp, sim_days, rng, n_attacks)
        elif attack_type == "lateral_movement":
            df = inject_lateral_movement(profiles, base_timestamp, sim_days, rng, n_attacks)
        elif attack_type == "device_spoofing":
            df = inject_device_spoofing(profiles, base_timestamp, sim_days, rng, n_attacks)
        elif attack_type == "low_and_slow_exfiltration":
            df = inject_low_and_slow_exfiltration(profiles, base_timestamp, sim_days, rng, n_attacks)
        elif attack_type == "insider_drift":
            df = inject_insider_drift(profiles, base_timestamp, sim_days, rng, n_attacks)
        else:
            continue
            
        all_attacks.append(df)
        
    if all_attacks:
        return pd.concat(all_attacks, ignore_index=True)
    return pd.DataFrame()

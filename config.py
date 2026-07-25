"""
Central configuration for AI-Powered Behavioral Anomaly Detection System.
All constants, hyperparameters, paths, and schema definitions in one place.
"""

import os
from pathlib import Path

# ─────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"

for d in [DATA_DIR, MODEL_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ACCESS_LOG_PATH = DATA_DIR / "access_logs.csv"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"
METRICS_PATH = DATA_DIR / "metrics.json"
REPORT_PATH = PROJECT_ROOT / "report.pdf"

# ─────────────────────────── Schema ───────────────────────────
SCHEMA_FIELDS = [
    "entity_id",
    "entity_type",
    "timestamp",
    "source_ip",
    "geo_location",
    "resource_accessed",
    "auth_method",
    "session_duration",
    "command_sequence",
    "device_fingerprint",
]

ENTITY_TYPES = ["user", "service_account", "edge_device"]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]

# ─────────────────────────── Attack Taxonomy ──────────────────
LABEL_NORMAL = "normal"
ATTACK_TYPES = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
    "insider_drift",
]
ALL_LABELS = [LABEL_NORMAL] + ATTACK_TYPES
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES = len(ALL_LABELS)

# ─────────────────────────── Data Generation ──────────────────
NUM_ENTITIES = 500
NUM_EVENTS = 200_000
SIMULATION_DAYS = 90
ANOMALY_RATE_MIN = 0.005  # 0.5%
ANOMALY_RATE_MAX = 0.03   # 3.0%
RANDOM_SEED = 42

# Attack injection rates (fraction of total events)
ATTACK_RATES = {
    "brute_force":              0.005,
    "impossible_travel":        0.005,
    "credential_stuffing":      0.005,
    "lateral_movement":         0.005,
    "device_spoofing":          0.005,
    "low_and_slow_exfiltration": 0.003,
    "insider_drift":            0.002,
}

# ─────────────────────────── Geo-locations ────────────────────
GEO_CITIES = [
    ("New York", 40.7128, -74.0060),
    ("London", 51.5074, -0.1278),
    ("Tokyo", 35.6762, 139.6503),
    ("Sydney", -33.8688, 151.2093),
    ("Mumbai", 19.0760, 72.8777),
    ("Berlin", 52.5200, 13.4050),
    ("São Paulo", -23.5505, -46.6333),
    ("Toronto", 43.6532, -79.3832),
    ("Dubai", 25.2048, 55.2708),
    ("Singapore", 1.3521, 103.8198),
    ("San Francisco", 37.7749, -122.4194),
    ("Chicago", 41.8781, -87.6298),
    ("Paris", 48.8566, 2.3522),
    ("Seoul", 37.5665, 126.9780),
    ("Shanghai", 31.2304, 121.4737),
]

# ─────────────────────────── Resources ────────────────────────
RESOURCES = [
    # Files
    "file://docs/quarterly_report.xlsx",
    "file://docs/employee_records.csv",
    "file://code/main_app.py",
    "file://secrets/api_keys.json",
    "file://finance/invoices_2025.pdf",
    "file://hr/salary_data.xlsx",
    "file://devops/deployment_config.yaml",
    "file://legal/contracts.docx",
    # Endpoints
    "api://auth/login",
    "api://auth/reset_password",
    "api://users/admin",
    "api://data/export",
    "api://config/update",
    "api://logs/audit",
    # Ports / Services
    "port://22/ssh",
    "port://443/https",
    "port://3389/rdp",
    "port://5432/postgres",
    "port://27017/mongodb",
    # Device functions (for edge_device)
    "device://plc/firmware_update",
    "device://sensor/read_telemetry",
    "device://actuator/control",
    "device://gateway/config",
]

# Per entity_type typical resources (subset indices into RESOURCES)
TYPICAL_RESOURCES = {
    "user": RESOURCES[:8] + RESOURCES[8:14],        # files + APIs
    "service_account": RESOURCES[8:14] + RESOURCES[14:19],  # APIs + ports
    "edge_device": RESOURCES[19:],                  # device functions
}

# ─────────────────────────── Feature Engineering ──────────────
WINDOW_SIZES_MINUTES = [5, 60, 1440, 10080]  # 5min, 1hr, 24hr, 7 days
SEQUENCE_LENGTH = 10   # Events per sequence for LSTM
FEATURE_NAMES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    "time_since_last_access_sec",
    "access_freq_5min",
    "access_freq_1hr",
    "access_freq_24hr",
    "access_freq_7d",
    "geo_velocity_kmh",
    "geo_distance_km",
    "geo_entropy",
    "resource_novelty_score",
    "resource_breadth",
    "resource_entropy",
    "failed_auth_count_5min",
    "failed_auth_count_1hr",
    "auth_method_changed",
    "session_duration_zscore",
    "session_duration_raw",
    "fingerprint_mismatch",
    "new_device_flag",
    "command_seq_length",
    "command_seq_novelty",
    "entity_type_encoded",
    "auth_method_encoded",
]
NUM_FEATURES = len(FEATURE_NAMES)

# ─────────────────────────── Model Hyperparameters ────────────
# Autoencoder (Baseline Profiler)
AE_INPUT_DIM = NUM_FEATURES
AE_HIDDEN_DIMS = [64, 32, 16]
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 25
AE_BATCH_SIZE = 256
AE_CONTAMINATION = 0.03  # Expected anomaly fraction

# LSTM (Detection Model)
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LEARNING_RATE = 1e-3
LSTM_EPOCHS = 15
LSTM_BATCH_SIZE = 128
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.75  # Weight for positive (anomaly) class

# XGBoost (Anomaly Classifier)
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": RANDOM_SEED,
}

# ─────────────────────────── Concept Drift ────────────────────
DRIFT_WINDOW_DAYS = 30       # Sliding window for retraining
DRIFT_DECAY_FACTOR = 0.95    # Exponential decay for older samples

# ─────────────────────────── Cold Start ───────────────────────
COLD_START_THRESHOLD = 10    # Min events before entity-specific model kicks in
COLD_START_BLEND_EVENTS = 50 # Full blend-in over this many events

# ─────────────────────────── Explainability ───────────────────
TOP_K_FEATURES = 3  # Number of features to show in human-readable explanation

FEATURE_REASON_MAP = {
    "geo_velocity_kmh": "impossible geo-velocity ({val:.0f} km/h between consecutive logins)",
    "geo_distance_km": "large geographic distance ({val:.0f} km) from usual location",
    "fingerprint_mismatch": "device fingerprint does not match historical profile",
    "new_device_flag": "access from a previously unseen device",
    "failed_auth_count_5min": "{val:.0f} failed authentication attempts in the last 5 minutes",
    "failed_auth_count_1hr": "{val:.0f} failed authentication attempts in the last hour",
    "resource_novelty_score": "accessed {val:.0f}% novel resources never seen for this entity",
    "resource_breadth": "unusual breadth of resources accessed ({val:.0f} distinct resources)",
    "session_duration_zscore": "session duration {val:.1f} standard deviations from normal",
    "is_off_hours": "access occurred during off-hours",
    "access_freq_5min": "abnormally high access frequency ({val:.0f} events in 5 min)",
    "access_freq_1hr": "elevated access frequency ({val:.0f} events in 1 hour)",
    "auth_method_changed": "authentication method changed from the usual method",
    "command_seq_novelty": "unusual command sequence pattern detected",
    "command_seq_length": "abnormally long command sequence ({val:.0f} commands)",
    "resource_entropy": "chaotic resource access pattern (high entropy)",
    "geo_entropy": "access from highly varied geographic locations",
    "time_since_last_access_sec": "unusually {desc} time since last access ({val:.0f}s)",
    "hour_of_day": "access at unusual hour ({val:.0f}:00)",
    "day_of_week": "access on unusual day of week",
    "is_weekend": "access occurred on weekend",
    "access_freq_24hr": "elevated 24-hour access volume ({val:.0f} events)",
    "access_freq_7d": "elevated 7-day access volume ({val:.0f} events)",
    "session_duration_raw": "session duration of {val:.0f} seconds",
    "entity_type_encoded": "entity type factor",
    "auth_method_encoded": "authentication method factor",
}

# ─────────────────────────── Dashboard ────────────────────────
DASHBOARD_PAGE_SIZE = 50      # Alerts per page
DASHBOARD_REFRESH_SEC = 30    # Auto-refresh interval

# ─────────────────────────── Evaluation ───────────────────────
ALERT_BUDGET_PERCENTILE = 0.01   # Top 1% alert budget
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.1

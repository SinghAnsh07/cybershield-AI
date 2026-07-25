# AI-Powered Behavioral Anomaly Detection for Cybersecurity

A complete ML-driven cybersecurity system that models normal user/device
behaviour, detects anomalies in near real-time, classifies attack types,
and provides explainable risk scores through an analyst dashboard.

---

## 🏗 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌────────────────────┐
│  Synthetic Data  │────▷│   Feature        │────▷│  Baseline Profiler │
│  Generator       │     │   Engineering    │     │  (Autoencoder)     │
└────────┬────────┘     └────────┬────────┘     └─────────┬──────────┘
         │                       │                         │
         │                       ▼                         ▼
         │              ┌─────────────────┐     ┌────────────────────┐
         │              │  LSTM Detection  │────▷│ XGBoost Anomaly    │
         │              │  Model           │     │ Classifier         │
         │              └────────┬────────┘     └─────────┬──────────┘
         │                       │                         │
         │                       ▼                         ▼
         │              ┌─────────────────┐     ┌────────────────────┐
         │              │  SHAP            │────▷│ Streamlit          │
         │              │  Explainability  │     │ Dashboard          │
         │              └─────────────────┘     └────────────────────┘
         │
         ▼
   ┌───────────┐
   │   PDF     │
   │   Report  │
   └───────────┘
```

## 📦 Project Structure

```
honeywell/
├── config.py                    # Central configuration
├── pipeline.py                  # End-to-end training pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data_generator/              # Synthetic data generation
│   ├── profiles.py              # Entity behavioral profiles
│   ├── attack_injector.py       # 7 attack pattern injectors
│   └── generator.py             # Main generator orchestrator
│
├── models/                      # ML models
│   ├── feature_engineering.py   # Feature extraction pipeline
│   ├── baseline_profiler.py     # Statistical + Autoencoder profiler
│   ├── cold_start.py            # Cold-start entity handling
│   ├── detection_model.py       # BiLSTM anomaly detector
│   └── anomaly_classifier.py   # Multi-class XGBoost classifier
│
├── explainability/              # Explainability layer
│   └── explainer.py             # SHAP + human-readable reasons
│
├── dashboard/                   # Analyst dashboard
│   └── app.py                   # Streamlit application
│
├── evaluation/                  # Metrics & evaluation
│   └── evaluate.py              # F1, AUC-ROC, precision, recall
│
├── report/                      # PDF report generation
│   └── generate_report.py       # FPDF2-based report builder
│
├── streaming/                   # Real-time architecture
│   └── architecture_notes.md    # Kafka/Flink integration design
│
└── data/                        # Generated data (auto-created)
    ├── access_logs.csv
    ├── ground_truth.csv
    ├── models/                  # Saved model weights
    └── plots/                   # Evaluation charts
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python pipeline.py
```

This single command executes:
1. **Data generation** — 200K synthetic events, 500 entities, 7 attack types
2. **Feature engineering** — 26 behavioral features per event
3. **Model training** — Autoencoder baseline, BiLSTM detector, XGBoost classifier
4. **Evaluation** — F1, AUC-ROC, confusion matrix on imbalanced test set
5. **Report generation** — PDF with metrics, charts, and analysis

### 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens an analyst-facing dashboard at `http://localhost:8501` with:
- Ranked alert queue with risk scores
- Entity investigation view
- SHAP-powered explanations
- Attack type distribution analytics

## 🎯 Attack Taxonomy

| # | Attack Type | Description |
|---|-------------|-------------|
| 1 | Brute Force | Rapid repeated failed-auth attempts from one source |
| 2 | Impossible Travel | Geographically distant logins within implausible time |
| 3 | Credential Stuffing | Many entities, few IPs, high failure rate |
| 4 | Lateral Movement | Unusual breadth/sequence of resource access |
| 5 | Device Spoofing | Device ID with mismatched fingerprint |
| 6 | Low-and-Slow Exfil | Gradual off-hours resource access over days |
| 7 | Insider Drift | Slow privilege/resource expansion (edge case) |

## 📊 Data Schema

| Field | Description |
|-------|-------------|
| `entity_id` | User ID or device ID |
| `entity_type` | user / service_account / edge_device |
| `timestamp` | Access or connection time (ISO 8601) |
| `source_ip` | Origin IP address |
| `geo_location` | City\|latitude\|longitude |
| `resource_accessed` | File, endpoint, port, or device function |
| `auth_method` | password / token / certificate / biometric |
| `session_duration` | Length of connection in seconds |
| `command_sequence` | Ordered list of actions (semicolon-separated) |
| `device_fingerprint` | OS\|MAC address\|protocol |

## 🧠 Models

### Baseline Profiler
- **Statistical Profiler**: Per-entity mean/std/percentiles for all features
- **Autoencoder**: 26→64→32→16→32→64→26 architecture, trained on normal data

### Detection Model
- **Bidirectional LSTM**: Processes sequences of 10 events per entity
- **Focal Loss**: Handles extreme class imbalance (γ=2, α=0.75)
- **Concept Drift**: Exponential decay weighting on older training samples

### Anomaly Classifier
- **XGBoost**: Multi-class classification into 7 attack types + normal
- **SMOTE**: Synthetic oversampling for minority attack classes

### Cold-Start Handling
- New entities with < 10 events scored against entity-type cohort baseline
- Progressive blending toward entity-specific model over 50 events

## 🔍 Explainability

Every alert includes:
- **SHAP values**: Feature-level attribution from TreeExplainer
- **Human-readable reasons**: e.g., "Flagged due to impossible geo-velocity
  (8000 km/h) + access from a previously unseen device"
- **Top 3 contributing factors** ranked by importance

## 🌊 Streaming Feasibility

See `streaming/architecture_notes.md` for a detailed Kafka/Flink design:
- **Target latency**: 20–50ms end-to-end
- **Model serving**: ONNX Runtime in Flink for sub-10ms inference
- **State management**: RocksDB with per-entity keyed state
- **Throughput**: 10K–50K events/sec per task slot

## ⚠️ Known Limitations

1. Synthetic data may not capture all real-world behavioral complexity
2. Concept drift handling uses window-based retraining, not fully online
3. Cold-start relies on population baselines (limited personalization)
4. No graph-based entity-resource relationship modeling
5. Streaming integration requires dedicated Kafka/Flink infrastructure
6. SHAP explanations add latency; may need caching for real-time use

## 📝 License

Built for the Honeywell Hackathon 2026.

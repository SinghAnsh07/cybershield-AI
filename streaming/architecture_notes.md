# Kafka / Flink Streaming Integration — Architecture Notes

## Overview

This document outlines how the batch-trained anomaly detection system can be
adapted for **near real-time streaming** inference using Apache Kafka and
Apache Flink.

---

## Proposed Streaming Architecture

```
┌─────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌────────────┐
│ Auth / Edge  │───▷│  Kafka Topic  │───▷│   Flink Job      │───▷│ Alert Topic│
│ Log Sources  │    │ (raw-events)  │    │ (feature compute │    │ (alerts)   │
└─────────────┘    └───────────────┘    │  + inference)    │    └─────┬──────┘
                                        └──────────────────┘          │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  Dashboard / │
                                                               │  SIEM / SOAR │
                                                               └──────────────┘
```

## Components

### 1. Ingestion Layer — Apache Kafka
- **Topic `raw-access-events`**: All access/connection events land here in
  Avro or JSON schema matching our 10-field schema.
- **Partitioning**: Partition by `entity_id` so all events for a single entity
  land on the same partition, preserving ordering for sequence models.
- **Retention**: 30-day retention for replay and reprocessing.
- **Schema Registry**: Confluent Schema Registry enforces schema evolution.

### 2. Feature Computation — Apache Flink
Flink is ideal because it provides:
- **Event-time processing** with watermarks (handles late-arriving events)
- **Keyed state** per entity for rolling aggregations
- **Multiple window types**: tumbling, sliding, session windows

#### Flink Job Design
```
KafkaSource(raw-access-events)
  ─▷ KeyBy(entity_id)
  ─▷ ProcessFunction: per-entity stateful feature computation
       ├─ Maintains rolling windows (5min, 1hr, 24hr, 7d) in Flink state
       ├─ Tracks entity history (geos, resources, devices) in RocksDB state
       ├─ Computes all 26 features per event
       └─ Emits feature vector
  ─▷ ProcessFunction: model inference
       ├─ Loads ONNX-exported LSTM + XGBoost models
       ├─ Maintains per-entity sequence buffer (last 10 events)
       ├─ Runs inference, produces risk_score + attack_type
       └─ Emits alerts above threshold
  ─▷ KafkaSink(alerts)
```

#### State Management
- **RocksDB state backend**: Handles large per-entity state
- **Checkpointing**: Every 60 seconds for exactly-once semantics
- **State TTL**: 90-day TTL on entity history to bound memory

### 3. Model Serving
Two options for serving the PyTorch LSTM and XGBoost models inside Flink:

| Approach | Pros | Cons |
|----------|------|------|
| **ONNX Runtime in Flink** | Low latency (<5ms), no network hop | Model reload requires job restart |
| **TorchServe / Triton sidecar** | Hot model reload, GPU support | Network latency (~10-20ms), extra infra |

**Recommendation**: ONNX Runtime embedded in Flink for sub-10ms latency.
Export PyTorch LSTM via `torch.onnx.export()` and XGBoost via
`xgboost2onnx` or `treelite`.

### 4. Model Update / Concept Drift
- **Periodic retraining**: Nightly batch job on last 30 days of data
- **Model versioning**: MLflow or simple S3/GCS versioned artifacts
- **Hot swap**: Flink reads model version from a broadcast stream; on new
  version event, each operator reloads the ONNX model from shared storage
- **A/B testing**: Run old and new models in parallel via Flink side outputs;
  compare alert overlap before promoting new model

### 5. Alert Delivery
- **Kafka topic `alerts`**: Downstream consumers include:
  - Streamlit dashboard (polls or uses WebSocket)
  - SIEM integration (Splunk, Elastic SIEM via Kafka Connect)
  - SOAR playbook trigger (PagerDuty, TheHive)
- **Alert enrichment**: Flink async I/O to enrich alerts with entity metadata
  from a lookup database (Redis / DynamoDB)

## Latency Budget

| Stage | Expected Latency |
|-------|-----------------|
| Kafka ingest | < 10 ms |
| Flink feature computation | 5–20 ms |
| ONNX model inference | 2–5 ms |
| Kafka alert publish | < 10 ms |
| **End-to-end** | **~20–50 ms** |

This meets the "near real-time" requirement with sub-second detection.

## Scalability Considerations

- **Horizontal scaling**: Flink parallelism = Kafka partition count.
  Start with 8–16 partitions; scale linearly.
- **Throughput**: Expected ~10,000–50,000 events/sec per Flink task slot.
- **Backpressure**: Flink's credit-based flow control prevents overload.
- **Entity cardinality**: RocksDB state supports millions of entities;
  TTL prevents unbounded growth.

## Implementation Roadmap

1. **Phase 1** (Current): Batch training + batch/micro-batch inference
2. **Phase 2**: Kafka ingestion + Flink feature computation + batch models
3. **Phase 3**: ONNX model serving in Flink + real-time alerts
4. **Phase 4**: Online learning / incremental model updates

## Dependencies

- Apache Kafka 3.x + Confluent Schema Registry
- Apache Flink 1.18+ with RocksDB state backend
- ONNX Runtime 1.16+
- Docker / Kubernetes for orchestration

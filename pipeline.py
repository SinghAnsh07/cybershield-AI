"""
pipeline.py — End-to-end training, evaluation, and report generation pipeline.

Usage:
    python pipeline.py              # Full pipeline
    python pipeline.py --skip-gen   # Skip data generation (use existing data)
    python pipeline.py --data-only  # Only generate data
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Project imports
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ────────────────────────────────────────────────────────────────
def step_generate_data():
    """Generate synthetic access logs and ground truth labels."""
    log.info("=" * 60)
    log.info("STEP 1: Generating synthetic data")
    log.info("=" * 60)

    from data_generator.generator import generate_dataset

    access_df, truth_df = generate_dataset(
        n_entities=config.NUM_ENTITIES,
        n_events=config.NUM_EVENTS,
        sim_days=config.SIMULATION_DAYS,
        seed=config.RANDOM_SEED,
    )

    log.info(f"  Access logs shape: {access_df.shape}")
    log.info(f"  Ground truth shape: {truth_df.shape}")
    log.info(f"  Label distribution:\n{truth_df['label'].value_counts().to_string()}")
    return access_df, truth_df


def step_load_data():
    """Load existing data from disk."""
    log.info("=" * 60)
    log.info("STEP 1: Loading existing data from disk")
    log.info("=" * 60)

    access_df = pd.read_csv(config.ACCESS_LOG_PATH)
    truth_df = pd.read_csv(config.GROUND_TRUTH_PATH)

    log.info(f"  Access logs shape: {access_df.shape}")
    log.info(f"  Ground truth shape: {truth_df.shape}")
    return access_df, truth_df


# ────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────
def step_feature_engineering(access_df):
    """Extract features from raw access logs."""
    log.info("=" * 60)
    log.info("STEP 2: Feature engineering")
    log.info("=" * 60)

    from models.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    features_df = fe.fit_transform(access_df)

    log.info(f"  Features shape: {features_df.shape}")
    log.info(f"  Feature columns: {list(features_df.columns)}")
    log.info(f"  NaN count: {features_df.isna().sum().sum()}")
    return fe, features_df


# ────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────
def step_split_data(features_df, truth_df, access_df):
    """Time-based train/test split to respect temporal ordering."""
    log.info("=" * 60)
    log.info("STEP 3: Train / test split")
    log.info("=" * 60)

    # Merge labels onto features — keep as DataFrames for the models
    feature_cols = config.FEATURE_NAMES

    # Create a combined DataFrame for easy splitting
    n = len(features_df)
    split_idx = int(n * (1 - config.TEST_SPLIT_RATIO))

    # Train features and labels
    train_features = features_df.iloc[:split_idx].reset_index(drop=True)
    test_features = features_df.iloc[split_idx:].reset_index(drop=True)

    train_labels = truth_df.iloc[:split_idx]["label"].reset_index(drop=True)
    test_labels = truth_df.iloc[split_idx:]["label"].reset_index(drop=True)

    train_entity_ids = access_df.iloc[:split_idx]["entity_id"].reset_index(drop=True)
    test_entity_ids = access_df.iloc[split_idx:]["entity_id"].reset_index(drop=True)

    train_entity_types = access_df.iloc[:split_idx]["entity_type"].reset_index(drop=True)

    train_timestamps = access_df.iloc[:split_idx]["timestamp"].reset_index(drop=True)
    test_timestamps = access_df.iloc[split_idx:]["timestamp"].reset_index(drop=True)

    # Binary labels
    train_is_anomaly = (train_labels != config.LABEL_NORMAL).astype(int)
    test_is_anomaly = (test_labels != config.LABEL_NORMAL).astype(int)

    log.info(f"  Train size: {len(train_features)}, Test size: {len(test_features)}")
    log.info(f"  Train anomaly rate: {train_is_anomaly.mean():.4f}")
    log.info(f"  Test anomaly rate: {test_is_anomaly.mean():.4f}")

    split = {
        "train_features": train_features,
        "test_features": test_features,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_entity_ids": train_entity_ids,
        "test_entity_ids": test_entity_ids,
        "train_entity_types": train_entity_types,
        "train_is_anomaly": train_is_anomaly,
        "test_is_anomaly": test_is_anomaly,
        "train_timestamps": train_timestamps,
        "test_timestamps": test_timestamps,
        "feature_cols": feature_cols,
    }
    return split


# ────────────────────────────────────────────────────────────────
# 4. BASELINE PROFILER
# ────────────────────────────────────────────────────────────────
def step_baseline_profiler(split):
    """Train the baseline profiler (statistical + autoencoder)."""
    log.info("=" * 60)
    log.info("STEP 4: Training baseline profiler")
    log.info("=" * 60)

    from models.baseline_profiler import BaselineProfiler

    profiler = BaselineProfiler()

    # Build a mock df with entity_id for the statistical profiler
    train_df = pd.DataFrame({"entity_id": split["train_entity_ids"]})
    profiler.fit(train_df, split["train_features"], split["train_labels"])
    profiler.save(config.MODEL_DIR / "baseline_profiler")

    log.info("  Baseline profiler trained and saved")
    return profiler


# ────────────────────────────────────────────────────────────────
# 5. DETECTION MODEL (LSTM)
# ────────────────────────────────────────────────────────────────
def step_detection_model(split):
    """Train the BiLSTM detection model."""
    log.info("=" * 60)
    log.info("STEP 5: Training LSTM detection model")
    log.info("=" * 60)

    from models.detection_model import DetectionModel

    detector = DetectionModel()
    detector.fit(
        split["train_features"],
        split["train_labels"],
        split["train_entity_ids"],
    )
    detector.save(config.MODEL_DIR / "detection_model")

    # Predict on test set
    anomaly_flags, risk_scores = detector.predict(
        split["test_features"], split["test_entity_ids"]
    )

    log.info(f"  Flagged {anomaly_flags.sum()} anomalies in test set")
    log.info(f"  Mean risk score: {risk_scores.mean():.4f}")
    return detector, anomaly_flags, risk_scores


# ────────────────────────────────────────────────────────────────
# 6. ANOMALY CLASSIFIER
# ────────────────────────────────────────────────────────────────
def step_anomaly_classifier(split):
    """Train the multi-class anomaly classifier."""
    log.info("=" * 60)
    log.info("STEP 6: Training anomaly classifier (XGBoost)")
    log.info("=" * 60)

    from models.anomaly_classifier import AnomalyClassifier

    classifier = AnomalyClassifier()
    classifier.fit(split["train_features"], split["train_labels"])
    classifier.save(config.MODEL_DIR / "anomaly_classifier")

    # Predict on test set
    pred_labels, pred_confidences = classifier.predict(split["test_features"])
    pred_proba = classifier.predict_proba(split["test_features"])

    log.info(f"  Predicted label distribution:")
    unique, counts = np.unique(pred_labels, return_counts=True)
    for label, count in zip(unique, counts):
        log.info(f"    {label}: {count}")

    return classifier, pred_labels, pred_confidences, pred_proba


# ────────────────────────────────────────────────────────────────
# 7. EXPLAINABILITY
# ────────────────────────────────────────────────────────────────
def step_explainability(classifier, split, risk_scores, pred_labels):
    """Generate SHAP explanations for test alerts."""
    log.info("=" * 60)
    log.info("STEP 7: Generating SHAP explanations")
    log.info("=" * 60)

    from explainability.explainer import AnomalyExplainer

    explainer = AnomalyExplainer(classifier.model)

    test_features_df = split["test_features"]
    feature_cols = split["feature_cols"]
    explanations = explainer.explain(test_features_df, feature_names=feature_cols)

    # Build alert DataFrame for dashboard
    alert_df = pd.DataFrame()
    alert_df["entity_id"] = split["test_entity_ids"].values
    alert_df["timestamp"] = split["test_timestamps"].values
    alert_df["true_label"] = split["test_labels"].values
    alert_df["risk_score"] = risk_scores
    alert_df["predicted_attack_type"] = pred_labels
    alert_df["top_reason"] = [e["reason_string"] for e in explanations]

    # Add feature values for dashboard
    for col in feature_cols:
        alert_df[col] = test_features_df[col].values

    alert_df.to_csv(config.DATA_DIR / "alerts.csv", index=False)
    log.info(f"  Generated {len(explanations)} explanations")
    log.info(f"  Alerts saved to {config.DATA_DIR / 'alerts.csv'}")

    # Show sample explanations for anomalous events
    anomalous_mask = alert_df["true_label"] != config.LABEL_NORMAL
    if anomalous_mask.any():
        sample_indices = alert_df[anomalous_mask].head(3).index
        for idx in sample_indices:
            log.info(f"\n  Sample alert (true={alert_df.loc[idx, 'true_label']}):")
            log.info(f"    {explanations[idx]['reason_string']}")

    return explainer, explanations, alert_df


# ────────────────────────────────────────────────────────────────
# 8. EVALUATION
# ────────────────────────────────────────────────────────────────
def step_evaluation(split, risk_scores, anomaly_flags, pred_labels, pred_proba):
    """Evaluate all models with comprehensive metrics."""
    log.info("=" * 60)
    log.info("STEP 8: Evaluation")
    log.info("=" * 60)

    from evaluation.evaluate import Evaluator

    evaluator = Evaluator()

    y_true_binary = split["test_is_anomaly"].values
    y_true_multi = split["test_labels"].values

    # Binary detection metrics
    detection_metrics = evaluator.evaluate_detection(
        y_true_binary, risk_scores, anomaly_flags
    )
    log.info("\n  Binary Detection Metrics:")
    for key, val in detection_metrics.items():
        if isinstance(val, float):
            log.info(f"    {key}: {val:.4f}")

    # Multi-class classification metrics
    classification_metrics = evaluator.evaluate_classification(
        y_true_multi, pred_labels, pred_proba
    )
    log.info("\n  Multi-class Classification Metrics:")
    for key, val in classification_metrics.items():
        if isinstance(val, (float, int)):
            log.info(
                f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}"
            )

    # Generate plots
    results = {
        "y_true_binary": y_true_binary,
        "y_scores": risk_scores,
        "y_pred_binary": anomaly_flags,
        "y_true_multi": y_true_multi,
        "y_pred_multi": pred_labels,
        "labels": config.ALL_LABELS,
    }
    evaluator.generate_all_plots(results, str(config.PLOTS_DIR))
    log.info(f"\n  Plots saved to {config.PLOTS_DIR}")

    # Combine metrics
    all_metrics = {
        "detection": detection_metrics,
        "classification": classification_metrics,
    }
    evaluator.save_metrics(all_metrics, str(config.METRICS_PATH))
    log.info(f"  Metrics saved to {config.METRICS_PATH}")

    return all_metrics


# ────────────────────────────────────────────────────────────────
# 9. REPORT GENERATION
# ────────────────────────────────────────────────────────────────
def step_report(all_metrics):
    """Generate PDF report."""
    log.info("=" * 60)
    log.info("STEP 9: Generating PDF report")
    log.info("=" * 60)

    from report.generate_report import ReportGenerator

    generator = ReportGenerator()
    generator.generate(all_metrics, str(config.PLOTS_DIR), str(config.REPORT_PATH))

    log.info(f"  Report saved to {config.REPORT_PATH}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Behavioral Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Skip data generation, use existing data files",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only generate synthetic data, skip model training",
    )
    args = parser.parse_args()

    total_start = time.time()
    log.info("Starting AI-Powered Behavioral Anomaly Detection Pipeline")
    log.info(f"   Project root: {config.PROJECT_ROOT}")
    log.info(f"   Data directory: {config.DATA_DIR}")

    # Step 1: Data
    if args.skip_gen:
        access_df, truth_df = step_load_data()
    else:
        access_df, truth_df = step_generate_data()

    if args.data_only:
        log.info("Data generation complete. Exiting (--data-only).")
        return

    # Step 2: Features
    fe, features_df = step_feature_engineering(access_df)

    # Step 3: Split
    split = step_split_data(features_df, truth_df, access_df)

    # Step 4: Baseline profiler
    profiler = step_baseline_profiler(split)

    # Step 5: Detection model
    detector, anomaly_flags, risk_scores = step_detection_model(split)

    # Step 6: Anomaly classifier
    classifier, pred_labels, pred_confidences, pred_proba = step_anomaly_classifier(
        split
    )

    # Step 7: Explainability
    explainer, explanations, alert_df = step_explainability(
        classifier, split, risk_scores, pred_labels
    )

    # Step 8: Evaluation
    all_metrics = step_evaluation(
        split, risk_scores, anomaly_flags, pred_labels, pred_proba
    )

    # Step 9: Report
    step_report(all_metrics)

    elapsed = time.time() - total_start
    log.info("=" * 60)
    log.info(f"Pipeline complete in {elapsed:.1f} seconds")
    log.info(f"   Dashboard: streamlit run dashboard/app.py")
    log.info(f"   Report:    {config.REPORT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train ML Models from features_dump.csv
=======================================

Trains BOTH models in order:
1. Logistic Regression (fast, interpretable)
2. XGBoost (better accuracy)

USAGE:
    python train_models.py

OUTPUT:
    models/logistic_risk_model.pkl
    models/xgboost_risk_model.json
    models/training_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from brain.feature_aggregator import FeatureAggregator
from brain.logistic_regression_scorer import LogisticRegressionScorer
from brain.xgboost_trainer import XGBoostTrainer


def aggregate_features(csv_path='features_dump.csv'):
    """Load CSV and aggregate per-frame features."""

    print("=" * 70)
    print("STEP 1: Loading and Aggregating Features")
    print("=" * 70)

    # Load CSV
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=[
        'timestamp', 'camera_id', 'track_id',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'torso_angle', 'head_pitch', 'speed', 'dist_to_edge'
    ])

    print(f"✓ Loaded {len(df)} frame records")
    print(f"  Unique tracks: {df['track_id'].nunique()}")

    # Aggregate
    aggregator = FeatureAggregator(window_seconds=4.0)
    aggregated_data = []

    for track_id, track_df in df.groupby('track_id'):
        track_df = track_df.sort_values('timestamp')

        for _, row in track_df.iterrows():
            cx = (row['bbox_x1'] + row['bbox_x2']) / 2
            cy = (row['bbox_y1'] + row['bbox_y2']) / 2

            aggregator.add_frame_features(
                track_id=track_id,
                timestamp=row['timestamp'],
                features={
                    'torso_angle': row['torso_angle'],
                    'head_pitch': row['head_pitch'] if pd.notna(row['head_pitch']) else None,
                    'speed': row['speed'],
                    'dist_to_edge': row['dist_to_edge'],
                    'center': (cx, cy),
                }
            )

        agg = aggregator.get_aggregated_features(track_id)
        if agg is not None:
            aggregated_data.append(agg)

    result_df = pd.DataFrame(aggregated_data)
    print(f"✓ Aggregated {len(result_df)} tracks")

    return result_df


def label_data(df):
    """Label samples as suspicious (1) or normal (0)."""

    print("\n" + "=" * 70)
    print("STEP 2: Labeling Data")
    print("=" * 70)

    labels = []

    for _, row in df.iterrows():
        score = 0

        # Rule-based labeling
        if pd.notna(row.get('min_dist_to_edge')):
            if row['min_dist_to_edge'] < 50:
                score += 30
            elif row['min_dist_to_edge'] < 100:
                score += 15

        if pd.notna(row.get('dwell_time_near_edge')):
            if row['dwell_time_near_edge'] > 8:
                score += 30
            elif row['dwell_time_near_edge'] > 5:
                score += 15

        if pd.notna(row.get('max_speed')):
            if row['max_speed'] > 350:
                score += 20

        if pd.notna(row.get('direction_changes')):
            if row['direction_changes'] > 8:
                score += 15

        if pd.notna(row.get('edge_transgression_count')):
            if row['edge_transgression_count'] >= 3:
                score += 35

        labels.append(1 if score > 50 else 0)

    df['label'] = labels

    normal = (df['label'] == 0).sum()
    suspicious = (df['label'] == 1).sum()

    print(f"\n✓ Labeled {len(df)} samples:")
    print(f"  Normal: {normal} ({normal/len(df)*100:.1f}%)")
    print(f"  Suspicious: {suspicious} ({suspicious/len(df)*100:.1f}%)")

    return df


def prepare_training_data(df):
    """Prepare features for training."""

    print("\n" + "=" * 70)
    print("STEP 3: Preparing Training Data")
    print("=" * 70)

    # Logistic Regression features (extended set)
    lr_features = [
        'min_dist_to_edge',
        'mean_dist_to_edge',
        'dwell_time_near_edge',
        'mean_torso_angle',
        'std_torso_angle',
        'mean_speed',
        'max_speed',
        'direction_changes',
        'mean_head_pitch',
        'time_looking_down',
        'max_acceleration',
        'acceleration_spikes',
        'edge_transgression_count',
    ]

    # XGBoost features (basic set)
    xgb_features = [
        'mean_torso_angle',
        'max_torso_angle',
        'std_torso_angle',
        'mean_speed',
        'max_speed',
        'min_dist_to_edge',
        'dwell_time_near_edge',
        'direction_changes'
    ]

    # Create dataframes
    lr_df = pd.DataFrame()
    for feat in lr_features:
        lr_df[feat] = df[feat] if feat in df.columns else 0
    lr_df['label'] = df['label']
    lr_df = lr_df.fillna(0)

    xgb_df = pd.DataFrame()
    for feat in xgb_features:
        xgb_df[feat] = df[feat] if feat in df.columns else 0
    xgb_df['label'] = df['label']
    xgb_df = xgb_df.fillna(0)

    print(f"✓ Logistic Regression: {len(lr_features)} features")
    print(f"✓ XGBoost: {len(xgb_features)} features")

    return lr_df, xgb_df


def train_logistic_regression(lr_df):
    """Train Logistic Regression model."""

    print("\n" + "=" * 70)
    print("STEP 4: Training Logistic Regression")
    print("=" * 70)

    scorer = LogisticRegressionScorer(use_advanced_features=True)

    X = lr_df.drop('label', axis=1).values
    y = lr_df['label'].values

    metrics = scorer.train(X, y, test_size=0.25)

    # Save model
    os.makedirs('models', exist_ok=True)
    scorer.save_model('models/logistic_risk_model.pkl')

    print(f"\n✓ Logistic Regression trained!")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.1%}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")

    return scorer, metrics


def train_xgboost(xgb_df):
    """Train XGBoost model."""

    print("\n" + "=" * 70)
    print("STEP 5: Training XGBoost")
    print("=" * 70)

    trainer = XGBoostTrainer()
    model, metrics = trainer.train(xgb_df, test_size=0.25)

    # Save model
    os.makedirs('models', exist_ok=True)
    trainer.save_model(model, 'models/xgboost_risk_model.json')

    print(f"\n✓ XGBoost trained!")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.1%}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")

    return trainer, model, metrics


def compare_models(lr_metrics, xgb_metrics):
    """Compare both models."""

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Logistic Reg':<15} {'XGBoost':<15}")
    print("-" * 50)
    print(f"{'Test Accuracy':<20} {lr_metrics['test_accuracy']:>8.1%}      {xgb_metrics['test_accuracy']:>8.1%}")
    print(f"{'ROC-AUC':<20} {lr_metrics['roc_auc']:>8.3f}      {xgb_metrics['roc_auc']:>8.3f}")
    print(f"{'Precision':<20} {lr_metrics['precision']:>8.1%}      {xgb_metrics['precision']:>8.1%}")
    print(f"{'Recall':<20} {lr_metrics['recall']:>8.1%}      {xgb_metrics['recall']:>8.1%}")
    print(f"{'F1-Score':<20} {lr_metrics['f1_score']:>8.1%}      {xgb_metrics['f1_score']:>8.1%}")

    # Save report
    with open('models/training_report.txt', 'w') as f:
        f.write("MODEL COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Logistic Regression:\n")
        f.write(f"  Accuracy: {lr_metrics['test_accuracy']:.1%}\n")
        f.write(f"  ROC-AUC: {lr_metrics['roc_auc']:.3f}\n\n")
        f.write(f"XGBoost:\n")
        f.write(f"  Accuracy: {xgb_metrics['test_accuracy']:.1%}\n")
        f.write(f"  ROC-AUC: {xgb_metrics['roc_auc']:.3f}\n")

    print("\n✓ Report saved to: models/training_report.txt")


def main():
    print("\n" + "=" * 70)
    print("TRAIN LOGISTIC REGRESSION + XGBOOST MODELS")
    print("=" * 70)

    # Step 1: Aggregate features
    agg_df = aggregate_features('features_dump.csv')

    # Step 2: Label data
    labeled_df = label_data(agg_df)

    # Step 3: Prepare data
    lr_df, xgb_df = prepare_training_data(labeled_df)

    # Save training data
    os.makedirs('models', exist_ok=True)
    lr_df.to_csv('models/training_data.csv', index=False)
    print(f"\n✓ Saved: models/training_data.csv")

    # Step 4: Train Logistic Regression FIRST
    lr_scorer, lr_metrics = train_logistic_regression(lr_df)

    # Step 5: Train XGBoost SECOND
    xgb_trainer, xgb_model, xgb_metrics = train_xgboost(xgb_df)

    # Step 6: Compare
    compare_models(lr_metrics, xgb_metrics)

    # Test predictions
    print("\n" + "=" * 70)
    print("TESTING PREDICTIONS")
    print("=" * 70)

    test_features = {
        'min_dist_to_edge': 42,
        'dwell_time_near_edge': 9.5,
        'max_speed': 380,
        'mean_torso_angle': 70,
        'direction_changes': 8,
        'edge_transgression_count': 3,
    }

    print(f"\nTest: Suspicious behavior (dist=42px, dwell=9.5s, transgressions=3)")

    # Logistic
    lr_prob, lr_level = lr_scorer.predict(test_features)
    print(f"\n  Logistic Regression: {lr_prob:.1%} risk ({lr_level})")

    # XGBoost
    xgb_pred, xgb_conf = xgb_trainer.predict(xgb_model, test_features)
    print(f"  XGBoost: {'SUSPICIOUS' if xgb_pred == 1 else 'NORMAL'} ({xgb_conf:.1%} confident)")

    # Summary
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Models saved:")
    print(f"   • models/logistic_risk_model.pkl")
    print(f"   • models/xgboost_risk_model.json")
    print(f"   • models/training_data.csv")
    print(f"   • models/training_report.txt")

    print(f"\n🚀 Usage:")
    print(f"""
# Logistic Regression
from brain.logistic_regression_scorer import LogisticRegressionScorer
scorer = LogisticRegressionScorer()
scorer.load_model('models/logistic_risk_model.pkl')
prob, level = scorer.predict(features)

# XGBoost
from brain.xgboost_trainer import XGBoostTrainer
trainer = XGBoostTrainer()
model = trainer.load_model('models/xgboost_risk_model.json')
prediction, confidence = trainer.predict(model, features)
    """)

    print("\n" + "=" * 70)
    print("🎉 DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

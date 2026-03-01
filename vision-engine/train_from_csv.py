#!/usr/bin/env python3
"""
Complete Pipeline: CSV → Training Data → Trained Model
=======================================================

ONE COMMAND to go from features_dump.csv to trained XGBoost model!

USAGE:
    python train_from_csv.py --input features_dump.csv

This will:
1. Load per-frame features
2. Aggregate into time windows
3. Label samples (suspicious/normal)
4. Train XGBoost model
5. Save model + report

OUTPUT:
    - models/xgboost_risk_model.json (trained model)
    - models/training_data.csv (aggregated features)
    - models/training_report.txt (metrics)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our scripts
from process_features_to_training_data import (
    load_per_frame_features,
    aggregate_features_per_track,
    label_aggregated_features,
    prepare_for_xgboost
)
from brain.xgboost_trainer import XGBoostTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: CSV to trained model"
    )
    parser.add_argument('--input', type=str, default='features_dump.csv',
                        help='Input CSV with per-frame features')
    parser.add_argument('--window', type=float, default=4.0,
                        help='Sliding window size (seconds)')
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Test set fraction (default: 0.25)')

    args = parser.parse_args()

    print("=" * 70)
    print("COMPLETE PIPELINE: CSV → TRAINED MODEL")
    print("=" * 70)
    print()

    # STEP 1: Load per-frame features
    print("STEP 1: Loading per-frame features...")
    print("-" * 70)
    per_frame_df = load_per_frame_features(args.input)

    # STEP 2: Aggregate features
    print("\nSTEP 2: Aggregating features...")
    print("-" * 70)
    aggregated_df = aggregate_features_per_track(per_frame_df, window_seconds=args.window)

    # STEP 3: Label samples
    print("\nSTEP 3: Labeling samples...")
    print("-" * 70)
    labeled_df = label_aggregated_features(aggregated_df)

    # STEP 4: Prepare for XGBoost
    print("\nSTEP 4: Preparing for XGBoost...")
    print("-" * 70)
    training_df = prepare_for_xgboost(labeled_df)

    # Save training data
    import os
    os.makedirs('models', exist_ok=True)
    training_df.to_csv('models/training_data.csv', index=False)
    print(f"✓ Saved training data to: models/training_data.csv")

    # STEP 5: Train XGBoost model
    print("\nSTEP 5: Training XGBoost model...")
    print("-" * 70)

    trainer = XGBoostTrainer()
    model, metrics = trainer.train(training_df, test_size=args.test_size)

    # Save model
    model_path = 'models/xgboost_risk_model.json'
    trainer.save_model(model, model_path)

    # Save training report
    report_path = 'models/training_report.txt'
    with open(report_path, 'w') as f:
        f.write("XGBOOST RISK DETECTION MODEL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Data: {args.input}\n")
        f.write(f"Total Samples: {len(training_df)}\n")
        f.write(f"  Normal: {(training_df['label']==0).sum()}\n")
        f.write(f"  Suspicious: {(training_df['label']==1).sum()}\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Test Accuracy:  {metrics['test_accuracy']:.1%}\n")
        f.write(f"  ROC-AUC:        {metrics['roc_auc']:.3f}\n")
        f.write(f"  Precision:      {metrics['precision']:.1%}\n")
        f.write(f"  Recall:         {metrics['recall']:.1%}\n")
        f.write(f"  F1-Score:       {metrics['f1_score']:.1%}\n")

    print(f"\n📄 Training report saved to: {report_path}")

    # Test prediction
    print("\n" + "=" * 70)
    print("TESTING MODEL")
    print("=" * 70)

    test_features = {
        'mean_torso_angle': 70,
        'max_torso_angle': 65,
        'std_torso_angle': 6.0,
        'mean_speed': 220,
        'max_speed': 380,
        'min_dist_to_edge': 45,
        'dwell_time_near_edge': 9.5,
        'direction_changes': 8
    }

    prediction, confidence = trainer.predict(model, test_features)
    print(f"\nTest Case (Suspicious Behavior):")
    print(f"  Features: {test_features}")
    print(f"  → Prediction: {'SUSPICIOUS' if prediction == 1 else 'NORMAL'}")
    print(f"  → Confidence: {confidence:.1%}")

    # Final summary
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Model saved to: {model_path}")
    print(f"\n📊 Performance:")
    print(f"   Test Accuracy:  {metrics['test_accuracy']:.1%}")
    print(f"   ROC-AUC:        {metrics['roc_auc']:.3f}")
    print(f"   Precision:      {metrics['precision']:.1%}")
    print(f"   Recall:         {metrics['recall']:.1%}")

    print(f"\n📝 How to use in your code:")
    print(f"""
from brain.xgboost_trainer import XGBoostTrainer

# Load model
trainer = XGBoostTrainer()
model = trainer.load_model('{model_path}')

# Predict
features = {{
    'mean_torso_angle': 85,
    'max_speed': 150,
    'min_dist_to_edge': 120,
    # ... other features
}}
prediction, confidence = trainer.predict(model, features)

if prediction == 1:
    print(f"ALERT: Suspicious behavior ({{confidence:.0%}} confident)")
    """)

    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Your model is ready to deploy!")
    print("=" * 70)


if __name__ == "__main__":
    main()

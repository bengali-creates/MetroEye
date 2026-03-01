#!/usr/bin/env python3
"""
Quick ML Model Training Script
===============================

PURPOSE:
Train logistic regression model for risk detection in 1 command.

USAGE:
    # Option 1: Use synthetic data (fastest, for demo)
    python train_ml_model.py --synthetic

    # Option 2: Use your labeled CSV
    python train_ml_model.py --csv labeled_data.csv

    # Option 3: Auto-label existing features
    python train_ml_model.py --auto-label features_dump.csv

OUTPUT:
    - Trained model saved to: models/risk_model.pkl
    - Training report: models/training_report.txt
    - Feature importance plot: models/feature_importance.png
"""

import argparse
import os
import sys
from pathlib import Path

# Add brain directory to path
sys.path.insert(0, str(Path(__file__).parent))

from brain.logistic_regression_scorer import LogisticRegressionScorer
import pandas as pd
import numpy as np


def train_with_synthetic_data():
    """Generate synthetic data and train model (fastest option)."""
    print("=" * 70)
    print("TRAINING WITH SYNTHETIC DATA")
    print("=" * 70)
    print("\n⚠️  NOTE: This uses fake data for demo purposes.")
    print("   For production, use real labeled data!\n")

    # Generate synthetic data
    print("Step 1: Generating synthetic training data...")
    np.random.seed(42)

    n_normal = 150
    n_suspicious = 75

    # Normal behaviors
    normal_data = {
        'min_dist_to_edge': np.random.normal(180, 40, n_normal),
        'mean_dist_to_edge': np.random.normal(200, 35, n_normal),
        'dwell_time_near_edge': np.random.uniform(0, 2, n_normal),
        'mean_torso_angle': np.random.normal(88, 3, n_normal),
        'std_torso_angle': np.random.uniform(1, 3, n_normal),
        'mean_speed': np.random.normal(80, 20, n_normal),
        'max_speed': np.random.normal(120, 25, n_normal),
        'direction_changes': np.random.poisson(1.5, n_normal),
        'mean_head_pitch': np.random.normal(5, 5, n_normal),
        'time_looking_down': np.random.uniform(0, 1, n_normal),
        'max_acceleration': np.random.uniform(50, 150, n_normal),
        'acceleration_spikes': np.random.poisson(0.5, n_normal),
        'edge_transgression_count': np.random.poisson(0.2, n_normal),
        'shoulder_hunch_index': np.random.normal(0, 0.1, n_normal),
        'hand_to_face_distance': np.random.normal(300, 50, n_normal),
        'head_yaw_angle': np.random.normal(0, 15, n_normal),
        'weight_shifting_variance': np.random.uniform(10, 30, n_normal),
        'label': np.zeros(n_normal)
    }

    # Suspicious behaviors
    suspicious_data = {
        'min_dist_to_edge': np.random.normal(60, 25, n_suspicious),
        'mean_dist_to_edge': np.random.normal(80, 30, n_suspicious),
        'dwell_time_near_edge': np.random.uniform(5, 12, n_suspicious),
        'mean_torso_angle': np.random.normal(72, 6, n_suspicious),
        'std_torso_angle': np.random.uniform(4, 8, n_suspicious),
        'mean_speed': np.random.normal(180, 40, n_suspicious),
        'max_speed': np.random.normal(320, 50, n_suspicious),
        'direction_changes': np.random.poisson(6, n_suspicious),
        'mean_head_pitch': np.random.normal(35, 10, n_suspicious),
        'time_looking_down': np.random.uniform(2, 5, n_suspicious),
        'max_acceleration': np.random.uniform(200, 400, n_suspicious),
        'acceleration_spikes': np.random.poisson(3, n_suspicious),
        'edge_transgression_count': np.random.poisson(2, n_suspicious),
        'shoulder_hunch_index': np.random.normal(-0.2, 0.15, n_suspicious),
        'hand_to_face_distance': np.random.normal(80, 30, n_suspicious),
        'head_yaw_angle': np.random.normal(50, 20, n_suspicious),
        'weight_shifting_variance': np.random.uniform(40, 80, n_suspicious),
        'label': np.ones(n_suspicious)
    }

    df_normal = pd.DataFrame(normal_data)
    df_suspicious = pd.DataFrame(suspicious_data)
    df = pd.concat([df_normal, df_suspicious], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✓ Generated {len(df)} samples ({n_normal} normal, {n_suspicious} suspicious)")

    # Save for reference
    os.makedirs('models', exist_ok=True)
    df.to_csv('models/synthetic_training_data.csv', index=False)
    print(f"✓ Saved to: models/synthetic_training_data.csv")

    return train_model(df)


def train_with_csv(csv_path):
    """Train model from existing labeled CSV."""
    print("=" * 70)
    print(f"TRAINING WITH CSV: {csv_path}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"\n✗ ERROR: File not found: {csv_path}")
        return False

    print(f"\nStep 1: Loading data from CSV...")
    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        print("\n✗ ERROR: CSV must have a 'label' column (0=normal, 1=suspicious)")
        return False

    print(f"✓ Loaded {len(df)} samples")

    return train_model(df)


def train_with_auto_label(features_csv):
    """Auto-label features CSV using rules, then train."""
    print("=" * 70)
    print(f"AUTO-LABELING: {features_csv}")
    print("=" * 70)

    if not os.path.exists(features_csv):
        print(f"\n✗ ERROR: File not found: {features_csv}")
        return False

    print(f"\nStep 1: Loading features...")
    df = pd.read_csv(features_csv)

    print(f"Step 2: Auto-labeling with rules...")
    # Simple labeling logic
    labels = []
    for _, row in df.iterrows():
        score = 0

        # Distance to edge
        if row.get('min_dist_to_edge', 999) < 50:
            score += 30
        elif row.get('min_dist_to_edge', 999) < 100:
            score += 15

        # Dwell time
        if row.get('dwell_time_near_edge', 0) > 8:
            score += 30
        elif row.get('dwell_time_near_edge', 0) > 5:
            score += 15

        # Speed
        if row.get('max_speed', 0) > 350:
            score += 20

        # Direction changes
        if row.get('direction_changes', 0) > 8:
            score += 15

        # Edge transgressions
        if row.get('edge_transgression_count', 0) >= 3:
            score += 35

        # Label: suspicious if score > 50
        labels.append(1 if score > 50 else 0)

    df['label'] = labels

    print(f"✓ Labeled {len(df)} samples")
    print(f"  Normal: {(df['label']==0).sum()}")
    print(f"  Suspicious: {(df['label']==1).sum()}")

    # Save labeled data
    os.makedirs('models', exist_ok=True)
    labeled_path = 'models/auto_labeled_data.csv'
    df.to_csv(labeled_path, index=False)
    print(f"✓ Saved labeled data to: {labeled_path}")

    return train_model(df)


def train_model(df):
    """Train logistic regression model."""

    print(f"\nStep 3: Training Logistic Regression Model...")
    print("-" * 70)

    # Create scorer
    scorer = LogisticRegressionScorer(use_advanced_features=True)

    # Train
    metrics = scorer.train_from_csv('models/synthetic_training_data.csv' if 'models/synthetic_training_data.csv' in os.listdir('models') else 'models/auto_labeled_data.csv', test_size=0.25)

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/risk_model.pkl'
    scorer.save_model(model_path)

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Model saved to: {model_path}")
    print(f"\n📊 Performance Metrics:")
    print(f"   Test Accuracy:  {metrics['test_accuracy']:.1%}")
    print(f"   ROC-AUC:        {metrics['roc_auc']:.3f}")
    print(f"   Precision:      {metrics['precision']:.1%}")
    print(f"   Recall:         {metrics['recall']:.1%}")
    print(f"   F1-Score:       {metrics['f1_score']:.1%}")
    print(f"   CV AUC:         {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

    # Save training report
    report_path = 'models/training_report.txt'
    with open(report_path, 'w') as f:
        f.write("LOGISTIC REGRESSION MODEL TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy:  {metrics['test_accuracy']:.1%}\n")
        f.write(f"ROC-AUC:        {metrics['roc_auc']:.3f}\n")
        f.write(f"Precision:      {metrics['precision']:.1%}\n")
        f.write(f"Recall:         {metrics['recall']:.1%}\n")
        f.write(f"F1-Score:       {metrics['f1_score']:.1%}\n")
        f.write(f"CV AUC:         {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}\n")

    print(f"\n📄 Training report saved to: {report_path}")

    # Test prediction
    print(f"\n🧪 Testing prediction on sample data...")
    test_features = {
        'min_dist_to_edge': 45,
        'mean_dist_to_edge': 65,
        'dwell_time_near_edge': 9.5,
        'mean_torso_angle': 70,
        'std_torso_angle': 6.2,
        'mean_speed': 220,
        'max_speed': 380,
        'direction_changes': 8,
        'edge_transgression_count': 3
    }

    prob, level = scorer.predict(test_features)
    print(f"\n   Suspicious behavior test:")
    print(f"   → Risk Probability: {prob:.1%}")
    print(f"   → Risk Level: {level.upper()}")

    print(f"\n✅ Model is ready to use!")
    print(f"\n📝 Next steps:")
    print(f"   1. Integrate into your detection code:")
    print(f"      from brain.logistic_regression_scorer import LogisticRegressionScorer")
    print(f"      scorer = LogisticRegressionScorer()")
    print(f"      scorer.load_model('models/risk_model.pkl')")
    print(f"      prob, level = scorer.predict(features)")
    print(f"\n   2. See UPGRADE_TO_ML_SCORING.md for full integration guide")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model for risk detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use synthetic data (fastest, for demo)
  python train_ml_model.py --synthetic

  # Use labeled CSV
  python train_ml_model.py --csv labeled_data.csv

  # Auto-label existing features
  python train_ml_model.py --auto-label features_dump.csv
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic', action='store_true',
                       help='Generate and use synthetic training data')
    group.add_argument('--csv', type=str,
                       help='Path to labeled CSV file')
    group.add_argument('--auto-label', type=str,
                       help='Path to features CSV (will auto-label using rules)')

    args = parser.parse_args()

    # Train based on option
    if args.synthetic:
        success = train_with_synthetic_data()
    elif args.csv:
        success = train_with_csv(args.csv)
    elif args.auto_label:
        success = train_with_auto_label(args.auto_label)

    if success:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Your ML model is ready to deploy!")
        print("=" * 70)
        return 0
    else:
        print("\n✗ Training failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
StockSense AI — Phase 4/5/6: RELIANCE.NS Sanity Experiment Runner
Runs:
1. Model Architecture Inspection
2. CNN Filter & Activation Extraction
3. Baseline 1 (Naive Persistence)
4. Baseline 2 (Pure LSTM — 5 epochs)
5. Hybrid CNN-LSTM (5 epochs)
6. Comparative Metric Evaluation & Error Analysis
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.experiment_lab import DeepLearningExperimentLab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sanity_runner")


def main():
    print("\n" + "=" * 80)
    print("  STOCKSENSE AI — PHASE 4/5/6: DEEP LEARNING MODEL LAB")
    print("  TARGET: RELIANCE.NS (5-Epoch Sanity Verification)")
    print("=" * 80 + "\n")

    lab = DeepLearningExperimentLab(
        symbol="RELIANCE.NS",
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        conv_filters=64,
        kernel_size=3,
        lstm_units=50,
        dense_units=32,
        dropout=0.2,
        loss="mse",
        random_seed=42
    )

    results = lab.run_sanity_experiment()

    print("\n" + "=" * 80)
    print("  SANITY EXPERIMENT RESULTS & COMPARATIVE EVALUATION")
    print("=" * 80)

    print("\n[1] Architecture & Parameters:")
    insp = results["inspection"]
    print(f"    - Input Shape: {insp['input_shape']}")
    print(f"    - Total Parameters: {insp['total_params']:,}")
    print(f"    - Trainable Parameters: {insp['trainable_params']:,}")
    print(f"    - Non-Trainable Parameters: {insp['non_trainable_params']:,}")
    print(f"    - Model Summary Saved: {insp['report_path']}")

    print("\n[2] Comparative Test Metrics (Held-Out Test Set):")
    m = results["metrics"]
    print(f"    {'Model':<22} {'MAE (INR)':<12} {'RMSE (INR)':<12} {'MAPE (%)':<12} {'Directional Acc (%)'}")
    print("    " + "-" * 70)
    for model_key, label in [("naive_persistence", "Naive Persistence"), ("pure_lstm", "Pure LSTM (5 ep)"), ("cnn_lstm", "CNN-LSTM (5 ep)")]:
        metrics = m[model_key]
        print(f"    {label:<22} INR {metrics['mae']:<8.2f} INR {metrics['rmse']:<8.2f} {metrics['mape']:<10.2f}% {metrics['directional_accuracy']:<8.2f}%")

    print("\n[3] CNN Mathematical Instrumentation:")
    act = results["activations"]
    print(f"    - Activation Map Shape: {act['activation_shape']} (30 timesteps x 64 filters)")
    print(f"    - Top 3 Strongest Activations:")
    for top in act["top_filters"][:3]:
        print(f"      * Filter #{top['filter_index']}: peak {top['max_activation']} at {top['correlated_date']} (mean={top['mean_activation']})")

    print("\n[4] Training Artifacts Generated:")
    print(f"    - Run Directory: {results['run_dir']}")
    print(f"    - Checkpoint: {results['cnn_lstm']['best_model_path']}")
    print(f"    - History CSV: {results['cnn_lstm']['history_csv']}")
    print(f"    - Loss Plot: {results['cnn_lstm']['loss_plot']}")
    print(f"    - Actual vs Predicted Plot: {results['cnn_lstm']['actual_vs_pred_plot']}")

    print("\n" + "=" * 80)
    print("  SANITY RUN SUCCESSFUL — LOCAL VERIFICATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

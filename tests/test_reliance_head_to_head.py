"""
Unit and regression tests for Reliance Head-to-Head Model Evaluation.
Verifies data provenance, mathematical price reconstruction,
leakage constraints, and metric reproducibility.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "research" / "EXP-05" / "evaluation"
CSV_PATH = EVAL_DIR / "predictions_head_to_head_RELIANCE_NS.csv"
JSON_PATH = EVAL_DIR / "metrics_head_to_head_summary.json"
REPORT_PATH = EVAL_DIR / "HEAD_TO_HEAD_REPORT.md"


def test_evaluation_artifacts_exist():
    """Ensure all required evaluation artifacts were generated."""
    assert CSV_PATH.exists(), f"Missing {CSV_PATH}"
    assert JSON_PATH.exists(), f"Missing {JSON_PATH}"
    assert REPORT_PATH.exists(), f"Missing {REPORT_PATH}"


def test_prediction_csv_schema_and_counts():
    """Check row count, model configs, and schema integrity."""
    df = pd.read_csv(CSV_PATH)
    # 184 sessions * 4 models = 736 records
    assert len(df) == 736
    
    expected_models = {
        "Production CNN-LSTM",
        "Colab EXP-05 Fold 3 (Calibrated)",
        "Colab EXP-05 Fold 3 (Uncalibrated)",
        "Naive Persistence"
    }
    assert set(df["model_name"].unique()) == expected_models

    for model in expected_models:
        sub = df[df["model_name"] == model]
        assert len(sub) == 184

    required_cols = [
        "target_date", "input_window_end", "forecast_timestamp",
        "model_name", "model_hash", "training_cutoff", "data_source",
        "previous_close", "predicted_return", "predicted_close",
        "actual_close", "actual_return", "predicted_direction",
        "actual_direction", "direction_correct"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_chronological_integrity_and_no_lookahead():
    """Ensure input cutoff date is strictly before target forecast date."""
    df = pd.read_csv(CSV_PATH)
    t_in = pd.to_datetime(df["input_window_end"])
    t_tgt = pd.to_datetime(df["target_date"])
    assert (t_in < t_tgt).all(), "Found lookahead date alignment violation!"


def test_price_reconstruction_formula():
    """Verify: Predicted Close = Previous Close * (1 + Predicted Return)."""
    df = pd.read_csv(CSV_PATH)
    computed_close = df["previous_close"] * (1.0 + df["predicted_return"])
    diff = np.abs(computed_close - df["predicted_close"])
    # Allowed rounding error from 2 decimal precision in export
    assert (diff <= 0.05).all(), f"Price reconstruction formula mismatch! Max diff: {diff.max()}"


def test_metrics_summary_consistency():
    """Verify metrics JSON contains expected models, hashes, and non-trivial calculations."""
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["symbol"] == "RELIANCE.NS"
    assert data["evaluation_block"]["sample_count"] == 184

    models = data["models"]
    assert "production_cnn_lstm" in models
    assert "colab_exp05_fold3_calibrated" in models
    assert "colab_exp05_fold3_uncalibrated" in models
    assert "naive_persistence" in models

    # Check price MAEs
    naive_mae = models["naive_persistence"]["metrics"]["price_mae_inr"]
    colab_mae = models["colab_exp05_fold3_calibrated"]["metrics"]["price_mae_inr"]
    assert 13.0 <= naive_mae <= 15.0
    assert 13.0 <= colab_mae <= 15.0

    # Ensure Colab fold 3 hash matches audited hash
    assert models["colab_exp05_fold3_calibrated"]["hash"] == "78741c67240f2626b2366443f3e4305961edc2fa424f9c7b3872f0acfa18ce97"
    assert models["production_cnn_lstm"]["hash"] == "a4715e84cdd7a16d1b334936c9a39b35249bb594aec5c987a4f6ee62d50d4f11"

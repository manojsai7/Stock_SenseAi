"""
Tests for EXP-01: Offline Directional Benchmark & Metric Audit.
Verifies date alignment, baseline prediction validity, metric calculations,
and strict prevention of test-label leakage.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.analysis.directional_benchmark_audit import (
    sign_from_difference,
    compute_comprehensive_metrics,
    DirectionalBenchmarkAuditor,
    DIRECTION_CLASSES,
    DIRECTION_NAMES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_sign_from_difference_mapping():
    """Verify 3-class mapping with tolerance, never mapping equal to UP."""
    diffs = np.array([10.0, -5.0, 0.0, 1e-6, -1e-6, 1e-4, -1e-4])
    signs = sign_from_difference(diffs, atol=1e-5)
    # 10.0 -> +1 (UP)
    # -5.0 -> -1 (DOWN)
    # 0.0 -> 0 (FLAT)
    # 1e-6 -> 0 (FLAT)
    # -1e-6 -> 0 (FLAT)
    # 1e-4 -> +1 (UP)
    # -1e-4 -> -1 (DOWN)
    expected = np.array([1, -1, 0, 0, 0, 1, -1])
    np.testing.assert_array_equal(signs, expected)


def test_metric_calculations_controlled_synthetic():
    """Verify metric calculations against hand-calculated ground truth."""
    # Actual: 4 UP (+1), 4 DOWN (-1), 2 FLAT (0) -> N = 10, Non-flat N = 8
    actual = np.array([1, 1, 1, 1, -1, -1, -1, -1, 0, 0])
    # Predicted: 3 correct UP, 1 UP pred DOWN, 2 correct DOWN, 2 DOWN pred UP, 1 correct FLAT, 1 FLAT pred DOWN
    # Predictions: [1, 1, 1, -1, -1, -1, 1, 1, 0, -1]
    pred = np.array([1, 1, 1, -1, -1, -1, 1, 1, 0, -1])

    actual_prices = np.array([10.0] * 10)
    pred_prices = np.array([11.0] * 10)

    m = compute_comprehensive_metrics(
        actual_signs=actual,
        predicted_signs=pred,
        actual_prices=actual_prices,
        predicted_prices=pred_prices
    )

    # Correct count: (1==1): 3, (-1==-1): 2, (0==0): 1 -> Total = 6 / 10 = 60.0%
    assert m["sample_count"] == 10
    assert m["three_class_accuracy"] == 60.0

    # Non-flat actual sessions: indices 0..7 (N=8).
    # Correct among first 8: 3 + 2 = 5 / 8 = 62.5%
    assert m["non_flat_denominator"] == 8
    assert m["flat_sample_count"] == 2
    assert m["non_flat_accuracy"] == 62.5

    # Confusion matrix checks: sum equals sample_count
    cm = np.array(m["confusion_matrix"]["matrix"])
    assert cm.shape == (3, 3)
    assert cm.sum() == 10

    # Price metrics
    assert m["mae"] == 1.0
    assert m["rmse"] == 1.0
    assert m["mape"] == 10.0


def test_date_alignment_and_price_matching():
    """Verifies date alignment on actual RELIANCE.NS dataset."""
    auditor = DirectionalBenchmarkAuditor(symbol="RELIANCE.NS")
    alignment = auditor.verify_alignment()

    assert alignment["is_valid"] is True, f"Alignment errors: {alignment['mismatches']}"
    assert alignment["mismatch_count"] == 0
    assert alignment["sample_count"] == 184

    # Check that every target date is strictly 1 row ahead of input date in processed df
    df = auditor.df_proc
    for pair in alignment["aligned_pairs"][:10]:
        idx_in = df.index[df["Date"] == pair["input_date"]][0]
        idx_tgt = df.index[df["Date"] == pair["target_date"]][0]
        assert idx_tgt == idx_in + 1, f"Target date not exactly t+1: {pair}"


def test_baseline_predictions_and_leakage_prevention():
    """Verifies that baseline fitting uses training data only, without test peeking."""
    auditor = DirectionalBenchmarkAuditor(symbol="RELIANCE.NS")

    # Training signs calculation
    train_diff = auditor.actual_target_train - auditor.prev_close_train
    train_signs = sign_from_difference(train_diff)

    up_count = np.sum(train_signs == 1)
    down_count = np.sum(train_signs == -1)
    flat_count = np.sum(train_signs == 0)

    # In RELIANCE.NS training data, UP is the dominant class (422 vs 402)
    assert up_count > down_count
    assert up_count == 422
    assert down_count == 402
    assert flat_count == 1

    # In contrast, test data has DOWN as majority (95 vs 83)
    test_diff = auditor.actual_target_test - auditor.prev_close_test
    test_signs = sign_from_difference(test_diff)
    assert np.sum(test_signs == -1) == 95
    assert np.sum(test_signs == 1) == 83

    # Generate predictions
    _, predictors = auditor.generate_all_predictions()

    # Training majority must predict UP (+1), NOT DOWN (-1)
    train_maj_pred = predictors["training_majority"]["predicted_signs"]
    assert np.all(train_maj_pred == 1), "Training majority must predict UP (1)"

    # Always-DOWN diagnostic must predict DOWN (-1)
    always_down_pred = predictors["always_down_diagnostic"]["predicted_signs"]
    assert np.all(always_down_pred == -1), "Always-DOWN must predict -1"

    # Naive persistence must predict FLAT (0)
    naive_pred = predictors["naive_persistence"]["predicted_signs"]
    assert np.all(naive_pred == 0), "Naive persistence must predict FLAT (0)"


def test_audit_end_to_end_execution(tmp_path):
    """Executes full audit and verifies all artifacts are generated cleanly."""
    auditor = DirectionalBenchmarkAuditor(symbol="RELIANCE.NS")
    res = auditor.run_audit()

    out_dir = Path(res["output_dir"])
    assert out_dir.exists()

    csv_file = Path(res["csv_path"])
    json_file = Path(res["json_path"])
    report_file = Path(res["report_path"])

    assert csv_file.exists()
    assert json_file.exists()
    assert report_file.exists()

    # Check CSV contents
    df_pred = pd.read_csv(csv_file)
    assert len(df_pred) == 184
    assert "input_date" in df_pred.columns
    assert "target_date" in df_pred.columns
    assert "previous_close" in df_pred.columns
    assert "actual_close" in df_pred.columns
    assert "actual_direction" in df_pred.columns

    # Check JSON contents
    with open(json_file, "r") as f:
        summary = json.load(f)

    assert summary["symbol"] == "RELIANCE.NS"
    assert summary["sample_counts"]["total_test_samples"] == 184
    assert summary["sample_counts"]["actual_up_samples"] == 83
    assert summary["sample_counts"]["actual_down_samples"] == 95
    assert summary["sample_counts"]["actual_flat_samples"] == 6

    # Verify key metrics are present
    metrics = summary["metrics_by_predictor"]
    assert "training_majority" in metrics
    assert "always_down_diagnostic" in metrics
    assert "previous_day_momentum" in metrics
    assert "sma20_trend" in metrics
    assert "naive_persistence" in metrics
    assert "saved_lstm" in metrics
    assert "saved_cnn_lstm" in metrics

    # Training majority accuracy is 83 / 184 = 45.11%
    assert metrics["training_majority"]["metrics"]["three_class_accuracy"] == 45.11
    # Always DOWN is 95 / 184 = 51.63%
    assert metrics["always_down_diagnostic"]["metrics"]["three_class_accuracy"] == 51.63
    # Naive is 6 / 184 = 3.26%
    assert metrics["naive_persistence"]["metrics"]["three_class_accuracy"] == 3.26


def test_multi_stock_auditor_execution():
    """Verify MultiStockDirectionalAuditor runs and generates cross-sectional outputs."""
    from ml.analysis.directional_benchmark_audit import MultiStockDirectionalAuditor

    # Test with 2 stocks for speed in unit test
    multi_auditor = MultiStockDirectionalAuditor(symbols=["INFY.NS", "HDFCBANK.NS"])
    res = multi_auditor.run_multi_audit()

    assert Path(res["output_dir"]).exists()
    assert Path(res["csv_path"]).exists()
    assert Path(res["json_path"]).exists()
    assert Path(res["report_path"]).exists()

    df_comb = pd.read_csv(res["csv_path"])
    assert "symbol" in df_comb.columns
    assert set(df_comb["symbol"].unique()) == {"INFY.NS", "HDFCBANK.NS"}

    macro_df = res["macro_df"]
    assert len(macro_df) > 0
    assert "Macro Raw Acc (%)" in macro_df.columns
    assert "Macro Balanced Acc (%)" in macro_df.columns
    assert "Macro MCC" in macro_df.columns


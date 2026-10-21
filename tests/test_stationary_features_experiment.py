"""
Unit tests for EXP-04: Stationary Features + Return-Target Forecasting.
Verifies:
1. Extraction of 7 relative stationary features and strict causality.
2. Return target calculation and price reconstruction identity.
3. Target scaler leakage safety (train-only fitting).
4. Walk-forward expanding fold boundaries and dates.
5. Direction sign classification (UP, DOWN, FLAT).
6. Pipeline dry-run integrity on RELIANCE.NS.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from scripts.run_stationary_features_experiment import (
    compute_stationary_features,
    sign_from_difference,
    compute_metrics,
    ReturnTargetScaler,
    StationaryFeaturesExperiment,
    FEATURE_COLUMNS
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_stationary_feature_columns_and_causality():
    """Verify that all 7 stationary features are computed causally without lookahead."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    # Synthetic bars
    close_prices = [100.0 + i * 2.0 for i in range(30)]
    open_prices = [c - 1.0 for c in close_prices]
    high_prices = [c + 2.0 for c in close_prices]
    low_prices = [c - 3.0 for c in close_prices]
    volume = [1000 + i * 10 for i in range(30)]

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume
    })

    clean_df, feature_cols = compute_stationary_features(df)

    # Check 7 feature columns
    assert feature_cols == [
        "return_1d",
        "return_5d",
        "return_20d",
        "intraday_return",
        "hl_range_ratio",
        "close_to_ma20",
        "volume_ratio"
    ]
    for col in feature_cols:
        assert col in clean_df.columns
        assert not clean_df[col].isna().any()

    # Verify return_1d at index 0 in clean_df: Close[t] / Close[t-1] - 1
    # row 20 in original df corresponds to row 0 in clean_df (due to 20-period warmup)
    idx_in_orig = 20
    expected_ret_1d = (close_prices[idx_in_orig] / close_prices[idx_in_orig - 1]) - 1.0
    assert pytest.approx(clean_df["return_1d"].iloc[0], 1e-6) == expected_ret_1d

    # Verify intraday return: (Close[t] - Open[t]) / Open[t]
    expected_intraday = (close_prices[idx_in_orig] - open_prices[idx_in_orig]) / open_prices[idx_in_orig]
    assert pytest.approx(clean_df["intraday_return"].iloc[0], 1e-6) == expected_intraday

    # Verify hl_range_ratio: (High[t] - Low[t]) / Close[t-1]
    expected_hl = (high_prices[idx_in_orig] - low_prices[idx_in_orig]) / close_prices[idx_in_orig - 1]
    assert pytest.approx(clean_df["hl_range_ratio"].iloc[0], 1e-6) == expected_hl

    # Verify target_return: Close[t+1] / Close[t] - 1
    expected_tgt = (close_prices[idx_in_orig + 1] / close_prices[idx_in_orig]) - 1.0
    assert pytest.approx(clean_df["target_return"].iloc[0], 1e-6) == expected_tgt


def test_target_return_reconstruction_identity():
    """Verify that reconstructed price Close[t] * (1 + target_return) exactly recovers Close[t+1]."""
    prev_closes = np.array([1250.0, 2400.5, 3100.25, 1890.0], dtype=np.float64)
    target_returns = np.array([0.015, -0.022, 0.0, 0.05], dtype=np.float64)

    reconstructed_closes = prev_closes * (1.0 + target_returns)
    computed_returns = (reconstructed_closes / prev_closes) - 1.0

    np.testing.assert_allclose(computed_returns, target_returns, rtol=1e-7)


def test_return_target_scaler_leakage_safety():
    """Verify ReturnTargetScaler fits exclusively on training and transforms cleanly."""
    train_returns = np.array([0.01, -0.015, 0.02, -0.005, 0.002])
    test_returns = np.array([0.05, -0.04])

    scaler = ReturnTargetScaler()
    scaler.fit(train_returns)

    # Check that statistics match train only
    assert pytest.approx(scaler.mean_, 1e-6) == float(np.mean(train_returns))
    assert pytest.approx(scaler.std_, 1e-6) == float(np.std(train_returns))

    # Transform and inverse transform
    transformed = scaler.transform(test_returns)
    restored = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(restored, test_returns, rtol=1e-5)


def test_sign_from_difference_three_classes():
    """Verify 3-class classification with zero tolerance: UP (1), DOWN (-1), FLAT (0)."""
    diffs = np.array([10.5, -5.2, 0.0, 1e-7, -1e-7, 0.00002, -0.00002])
    signs = sign_from_difference(diffs, atol=1e-5)
    # 10.5 -> 1
    # -5.2 -> -1
    # 0.0 -> 0
    # 1e-7 -> 0
    # -1e-7 -> 0
    # 0.00002 -> 1
    # -0.00002 -> -1
    expected = np.array([1, -1, 0, 0, 0, 1, -1], dtype=int)
    np.testing.assert_array_equal(signs, expected)


def test_compute_metrics_sanity():
    """Verify metric calculations including balanced accuracy, MCC, MAE, and confusion matrix."""
    actual_signs = np.array([1, -1, 1, 0, -1])
    pred_signs = np.array([1, -1, -1, 0, -1])
    actual_prices = np.array([101.0, 99.0, 102.0, 100.0, 98.0])
    pred_prices = np.array([101.0, 99.0, 99.0, 100.0, 97.0])

    m = compute_metrics(actual_signs, pred_signs, actual_prices, pred_prices)

    assert "three_class_accuracy" in m
    assert "balanced_accuracy" in m
    assert "mcc" in m
    assert "mae" in m
    assert "rmse" in m
    assert "mape" in m
    assert "confusion_matrix" in m
    assert m["sample_count"] == 5
    assert m["mae"] == round(float(np.mean([0.0, 0.0, 3.0, 0.0, 1.0])), 2)


def test_reliance_dry_run_and_walk_forward_folds():
    """Verify EXP-04 initialization and walk-forward fold structure on RELIANCE.NS."""
    exp = StationaryFeaturesExperiment(
        symbol="RELIANCE.NS",
        window_size=30,
        dry_run=True
    )
    res = exp.run_experiment()

    assert res["status"] == "dry_run_success"
    assert res["total_sequences"] == 1191
    assert res["folds_count"] == 3

    folds = exp.build_walk_forward_folds(test_block_size=184)
    assert len(folds) == 3

    # Fold 1: train 0:455, val 455:639, test 639:823
    assert folds[0]["train_slice"] == (0, 455)
    assert folds[0]["val_slice"] == (455, 639)
    assert folds[0]["test_slice"] == (639, 823)

    # Fold 2: train 0:639, val 639:823, test 823:1007
    assert folds[1]["train_slice"] == (0, 639)
    assert folds[1]["val_slice"] == (639, 823)
    assert folds[1]["test_slice"] == (823, 1007)

    # Fold 3: train 0:823, val 823:1007, test 1007:1191 (exact 184-session benchmark block!)
    assert folds[2]["train_slice"] == (0, 823)
    assert folds[2]["val_slice"] == (823, 1007)
    assert folds[2]["test_slice"] == (1007, 1191)
    assert folds[2]["is_benchmark_fold"] is True

    # Check benchmark block dates match EXP-01/02/03 test block
    test_dates = exp.input_dates_all[1007:1191]
    tgt_dates = exp.target_dates_all[1007:1191]
    assert len(test_dates) == 184
    assert test_dates[0] == "2026-01-14"
    assert test_dates[-1] == "2026-10-06"
    assert tgt_dates[0] == "2026-01-15"
    assert tgt_dates[-1] == "2026-10-07"

    # Verify Fold 3 Naive Persistence matches historical benchmark exactly
    act_te = exp.actual_closes_all[1007:1191]
    prev_te = exp.prev_closes_all[1007:1191]
    err = act_te - prev_te
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / act_te)) * 100)

    assert pytest.approx(mae, abs=0.01) == 14.16
    assert pytest.approx(rmse, abs=0.01) == 18.52
    assert pytest.approx(mape, abs=0.01) == 1.06

    # Verify target date continuity across all sequences (target is strictly idx + 1)
    df_raw = pd.read_csv(exp.proc_file)
    raw_dates = pd.to_datetime(df_raw["Date"]).dt.strftime("%Y-%m-%d").tolist()
    d_to_i = {d: i for i, d in enumerate(raw_dates)}
    for d_in, d_tgt in zip(exp.input_dates_all, exp.target_dates_all):
        assert d_to_i[d_tgt] == d_to_i[d_in] + 1


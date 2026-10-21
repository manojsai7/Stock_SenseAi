"""
Unit tests for the Canonical Stationary-Feature Pipeline (Phase 1).

Verifies:
1. Feature column ordering strictly matches canonical specification.
2. Exact mathematical formulas for all 7 stationary features and target return.
3. No future leakage: perturbing future bars does NOT affect past or current feature values.
4. Missing required bar columns fail explicitly.
5. NaN/warmup rows are handled and cleaned safely.
6. Sequence construction produces shape (N, 30, 7) without NaNs.
7. Feature scaler is training-only (test data cannot shift fitted training parameters).
8. Target scaler is training-only (test target distribution cannot shift fitted scaler).
9. Return-to-price reconstruction identity: predicted_close = last_close * (1 + predicted_return).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.features.stationary import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_stationary_features,
    prepare_inference_features,
    validate_raw_bars,
    validate_feature_dataframe,
    validate_sequence_tensor,
    reconstruct_price_from_return,
    WINDOW_SIZE
)
from ml.preprocessing.stationary_scaler import (
    StationaryFeatureScaler,
    ReturnTargetScaler
)


def _create_synthetic_ohlcv(num_bars: int = 50, start_date: str = "2025-01-01") -> pd.DataFrame:
    """Helper to generate deterministic, non-degenerate price bars."""
    dates = pd.date_range(start_date, periods=num_bars, freq="B")
    closes = [100.0 + i * 1.5 + (2.0 if i % 2 == 0 else -1.0) for i in range(num_bars)]
    opens = [c - 0.5 for c in closes]
    highs = [max(o, c) + 1.2 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 1.0 for o, c in zip(opens, closes)]
    volumes = [10000 + i * 50 for i in range(num_bars)]

    return pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    })


def test_feature_column_order():
    """1. Verify feature column count and exact column ordering."""
    expected_order = [
        "return_1d",
        "return_5d",
        "return_20d",
        "intraday_return",
        "hl_range_ratio",
        "close_to_ma20",
        "volume_ratio",
    ]
    assert len(FEATURE_COLUMNS) == 7
    assert FEATURE_COLUMNS == expected_order


def test_stationary_feature_formulas():
    """2. Verify exact mathematical definitions for all 7 features and target return."""
    df = _create_synthetic_ohlcv(num_bars=35)
    clean_df, feature_cols = compute_stationary_features(df)

    assert len(clean_df) > 0
    assert feature_cols == FEATURE_COLUMNS

    # Pick an arbitrary row k in clean_df and trace back its original index in df
    row_date = clean_df.iloc[5]["Date"]
    orig_idx = df.index[df["Date"] == row_date][0]

    # Formula 1: return_1d = Close[t] / Close[t-1] - 1
    expected_ret1 = (df["Close"].iloc[orig_idx] / df["Close"].iloc[orig_idx - 1]) - 1.0
    assert np.isclose(clean_df.iloc[5]["return_1d"], expected_ret1, atol=1e-6)

    # Formula 2: return_5d = Close[t] / Close[t-5] - 1
    expected_ret5 = (df["Close"].iloc[orig_idx] / df["Close"].iloc[orig_idx - 5]) - 1.0
    assert np.isclose(clean_df.iloc[5]["return_5d"], expected_ret5, atol=1e-6)

    # Formula 3: return_20d = Close[t] / Close[t-20] - 1
    expected_ret20 = (df["Close"].iloc[orig_idx] / df["Close"].iloc[orig_idx - 20]) - 1.0
    assert np.isclose(clean_df.iloc[5]["return_20d"], expected_ret20, atol=1e-6)

    # Formula 4: intraday_return = (Close[t] - Open[t]) / Open[t]
    expected_intra = (df["Close"].iloc[orig_idx] - df["Open"].iloc[orig_idx]) / df["Open"].iloc[orig_idx]
    assert np.isclose(clean_df.iloc[5]["intraday_return"], expected_intra, atol=1e-6)

    # Formula 5: hl_range_ratio = (High[t] - Low[t]) / Close[t-1]
    expected_hl = (df["High"].iloc[orig_idx] - df["Low"].iloc[orig_idx]) / df["Close"].iloc[orig_idx - 1]
    assert np.isclose(clean_df.iloc[5]["hl_range_ratio"], expected_hl, atol=1e-6)

    # Formula 6: close_to_ma20 = Close[t] / MA20[t] - 1
    expected_ma20 = df["Close"].iloc[orig_idx - 19 : orig_idx + 1].mean()
    expected_ma_ratio = (df["Close"].iloc[orig_idx] / expected_ma20) - 1.0
    assert np.isclose(clean_df.iloc[5]["close_to_ma20"], expected_ma_ratio, atol=1e-6)

    # Formula 7: volume_ratio = Volume[t] / Mean(Volume[t-19...t])
    expected_vol_ma = df["Volume"].iloc[orig_idx - 19 : orig_idx + 1].mean()
    expected_vol_ratio = df["Volume"].iloc[orig_idx] / (expected_vol_ma + 1e-8)
    assert np.isclose(clean_df.iloc[5]["volume_ratio"], expected_vol_ratio, atol=1e-6)

    # Target: target_return = Close[t+1] / Close[t] - 1
    expected_target_ret = (df["Close"].iloc[orig_idx + 1] / df["Close"].iloc[orig_idx]) - 1.0
    assert np.isclose(clean_df.iloc[5][TARGET_COLUMN], expected_target_ret, atol=1e-6)


def test_no_future_leakage():
    """3. Verify strict causality: modifying future sessions never alters past or current features."""
    df_base = _create_synthetic_ohlcv(num_bars=40)
    clean_base, _ = compute_stationary_features(df_base)

    # Create modified dataframe where future bars (from index 30 onwards) are drastically changed
    df_modified = df_base.copy()
    df_modified.loc[30:, "Close"] = df_modified.loc[30:, "Close"] * 5.0
    df_modified.loc[30:, "High"] = df_modified.loc[30:, "High"] * 5.0
    df_modified.loc[30:, "Volume"] = df_modified.loc[30:, "Volume"] * 10.0

    clean_mod, _ = compute_stationary_features(df_modified)

    # Find the row corresponding to date at original index 28 (strictly before modification)
    date_28 = df_base["Date"].iloc[28]
    row_base_28 = clean_base[clean_base["Date"] == date_28][FEATURE_COLUMNS].to_numpy()
    row_mod_28 = clean_mod[clean_mod["Date"] == date_28][FEATURE_COLUMNS].to_numpy()

    # The 7 stationary features at t=28 must be IDENTICAL despite massive future changes
    np.testing.assert_allclose(row_base_28, row_mod_28, rtol=1e-9, atol=1e-9)


def test_missing_columns_fail():
    """4. Verify that missing required raw columns raises ValueError."""
    df = _create_synthetic_ohlcv(num_bars=30)

    # Drop Volume
    df_no_vol = df.drop(columns=["Volume"])
    with pytest.raises(ValueError, match="missing required bar columns"):
        compute_stationary_features(df_no_vol)

    # Drop Date
    df_no_date = df.drop(columns=["Date"])
    with pytest.raises(ValueError, match="missing required bar columns"):
        compute_stationary_features(df_no_date)


def test_nan_rows_are_handled():
    """5. Verify that initial warmup rows with NaN lookbacks are filtered out safely."""
    df = _create_synthetic_ohlcv(num_bars=35)
    clean_df, _ = compute_stationary_features(df)

    # Clean df must have zero NaNs across all 7 features and target
    assert clean_df[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert clean_df[TARGET_COLUMN].isna().sum() == 0
    # Because 20-day returns require 20 bars and target requires t+1, 21 rows are dropped
    assert len(clean_df) == len(df) - 21


def test_sequence_shape_is_30_by_7():
    """6. Verify sequence generation produces shape (N, 30, 7) without NaNs or Infs."""
    df = _create_synthetic_ohlcv(num_bars=65)
    clean_df, _ = compute_stationary_features(df)

    L = 30
    features_matrix = clean_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    assert len(features_matrix) >= L

    sequences = []
    for i in range(len(features_matrix) - L + 1):
        window = features_matrix[i : i + L]
        sequences.append(window)

    seq_tensor = np.array(sequences, dtype=np.float32)
    assert seq_tensor.ndim == 3
    assert seq_tensor.shape[1] == 30
    assert seq_tensor.shape[2] == 7
    assert not np.isnan(seq_tensor).any()
    assert not np.isinf(seq_tensor).any()

    # Verify tensor validation function passes
    validate_sequence_tensor(seq_tensor, expected_window=30)


def test_feature_scaler_is_training_only():
    """7. Verify StationaryFeatureScaler is fitted strictly on training data."""
    df = _create_synthetic_ohlcv(num_bars=70)
    clean_df, _ = compute_stationary_features(df)
    features = clean_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Split 70% train, 30% test
    train_size = int(len(features) * 0.7)
    train_feat = features[:train_size]
    test_feat = features[train_size:]

    scaler = StationaryFeatureScaler()
    assert not scaler.is_fitted

    # Attempting to transform before fitting must raise ValueError
    with pytest.raises(ValueError, match="not fitted yet"):
        scaler.transform(test_feat)

    scaler.fit(train_feat)
    assert scaler.is_fitted

    # Fitted parameters must reflect train_feat only
    expected_mean = np.mean(train_feat, axis=0)
    np.testing.assert_allclose(scaler.scaler.mean_, expected_mean, rtol=1e-5)

    # Test features transformed without altering fitted mean
    saved_mean = np.copy(scaler.scaler.mean_)
    scaled_test = scaler.transform(test_feat)
    np.testing.assert_allclose(scaler.scaler.mean_, saved_mean)
    assert scaled_test.shape == test_feat.shape


def test_target_scaler_is_training_only():
    """8. Verify ReturnTargetScaler standardizes using training distribution only."""
    train_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.03], dtype=np.float64)
    test_returns = np.array([0.15, -0.12], dtype=np.float64)  # Extreme test shift

    scaler = ReturnTargetScaler()
    assert not scaler.is_fitted

    with pytest.raises(ValueError, match="not fitted yet"):
        scaler.transform(test_returns)

    scaler.fit(train_returns)
    assert scaler.is_fitted
    assert np.isclose(scaler.mean_, np.mean(train_returns))
    assert np.isclose(scaler.std_, np.std(train_returns))

    # Transforming and inverse-transforming test returns recovers exact values
    scaled_test = scaler.transform(test_returns)
    reconstructed_test = scaler.inverse_transform(scaled_test)
    np.testing.assert_allclose(reconstructed_test, test_returns, rtol=1e-6)


def test_return_to_price_reconstruction():
    """9. Verify price reconstruction identity: predicted_close = last_close * (1 + predicted_return)."""
    # Positive return case: 100 * (1 + 0.05) == 105
    last_close = 100.0
    pred_return = 0.05
    assert reconstruct_price_from_return(last_close, pred_return) == 105.0

    # Negative return case: 2500 * (1 - 0.02) == 2450
    last_close_2 = 2500.0
    pred_return_neg = -0.02
    assert np.isclose(reconstruct_price_from_return(last_close_2, pred_return_neg), 2450.0)

    # Zero return case: 1350 * (1 + 0.0) == 1350
    assert reconstruct_price_from_return(1350.0, 0.0) == 1350.0

    # Non-positive last_close must raise ValueError
    with pytest.raises(ValueError, match="positive"):
        reconstruct_price_from_return(-50.0, 0.05)

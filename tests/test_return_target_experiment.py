"""
Unit tests for EXP-03 Return-Target Forecasting.
Verifies return-target calculation, inverse transformation, price reconstruction,
and date alignment with zero data leakage.
"""

from pathlib import Path
import numpy as np
import pytest

from scripts.run_return_target_experiment import (
    ReturnTargetScaler,
    calculate_simple_returns,
    reconstruct_prices_from_returns,
    ReturnTargetExperiment
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_return_target_calculation_synthetic():
    """Verify simple return calculation on controlled values."""
    actual_targets = np.array([105.0, 95.0, 100.0, 200.0])
    prev_closes = np.array([100.0, 100.0, 100.0, 100.0])

    returns = calculate_simple_returns(actual_targets, prev_closes)
    expected = np.array([0.05, -0.05, 0.0, 1.0])

    np.testing.assert_allclose(returns, expected, rtol=1e-5)


def test_return_target_calculation_invalid_price():
    """Verify exception on non-positive previous close."""
    with pytest.raises(ValueError):
        calculate_simple_returns(np.array([100.0]), np.array([0.0]))


def test_return_target_scaler_leakage_safety():
    """Verify ReturnTargetScaler fits exclusively on train and round-trips cleanly."""
    train_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.0])
    test_returns = np.array([0.05, -0.04])

    scaler = ReturnTargetScaler()
    scaler.fit(train_returns)

    # Check fitted values
    expected_mean = float(np.mean(train_returns))
    expected_std = float(np.std(train_returns))
    assert pytest.approx(scaler.mean_, 1e-6) == expected_mean
    assert pytest.approx(scaler.std_, 1e-6) == expected_std

    # Check round-trip on test returns
    scaled_test = scaler.transform(test_returns)
    inversed_test = scaler.inverse_transform(scaled_test)
    np.testing.assert_allclose(inversed_test, test_returns, rtol=1e-5)

    # Leakage check: test values should not change fitted mean or std
    assert scaler.mean_ == expected_mean
    assert scaler.std_ == expected_std


def test_price_reconstruction():
    """Verify price reconstruction formula Close[t] * (1 + r[t+1]) equals target."""
    prev_closes = np.array([1200.0, 1450.5, 980.25])
    returns = np.array([0.025, -0.012, 0.0])

    reconstructed = reconstruct_prices_from_returns(returns, prev_closes)
    expected = prev_closes * (1.0 + returns)

    np.testing.assert_allclose(reconstructed, expected, rtol=1e-6)


def test_direction_sign_equivalence():
    """
    Mathematical verification: For any positive price P[t] > 0,
    sign(P[t]*(1 + r) - P[t]) == sign(P[t]*r) == sign(r).
    """
    prev_closes = np.array([1000.0, 1500.0, 2000.0, 500.0])
    returns = np.array([0.015, -0.02, 0.0, 1e-6])

    pred_closes = reconstruct_prices_from_returns(returns, prev_closes)
    diff = pred_closes - prev_closes

    diff_signs = np.where(np.abs(diff) <= 1e-5, 0, np.sign(diff))
    return_signs = np.where(np.abs(returns * prev_closes) <= 1e-5, 0, np.sign(returns))

    np.testing.assert_array_equal(diff_signs, return_signs)


def test_reliance_data_alignment_and_dry_run():
    """Verify RELIANCE.NS returns and alignment without running full training."""
    exp = ReturnTargetExperiment(symbol="RELIANCE.NS", epochs=1, dry_run=True)
    y_train_sc, y_val_sc, y_test_sc, scaler = exp.prepare_targets()

    # Verify lengths match prepared sequences
    assert len(y_train_sc) == len(exp.X_train) == 825
    assert len(y_val_sc) == len(exp.X_val) == 183
    assert len(y_test_sc) == len(exp.X_test) == 184

    # Verify scaler is strictly fitted on training
    assert pytest.approx(scaler.mean_, 1e-6) == float(np.mean(exp.returns_train))
    assert pytest.approx(scaler.std_, 1e-6) == float(np.std(exp.returns_train))

    # Verify reconstruction identity on full test set
    reconstructed_test = reconstruct_prices_from_returns(exp.returns_test, exp.prev_close_test)
    np.testing.assert_allclose(reconstructed_test, exp.actual_target_test, rtol=1e-4)

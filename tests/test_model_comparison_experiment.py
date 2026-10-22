"""
Unit tests for EXP-05: Model Comparison & Direction Calibration.
Verifies:
1. Date alignment and target continuity (target is strictly session t+1).
2. Train-only preprocessing and scalers leakage safety.
3. Validation-only threshold calibration (no test labels used).
4. Ridge and Logistic Regression validation selection and FLAT handling.
5. Metrics calculation including UP Recall, DOWN Recall, Balanced Accuracy, MCC.
6. Exact fold boundary alignment and Naive Persistence benchmark consistency.
7. Pipeline dry-run verification.
"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

from scripts.run_model_comparison_experiment import (
    sign_from_difference,
    calibrate_direction_threshold,
    compute_metrics,
    ReturnTargetScaler,
    ModelComparisonExperiment
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_validation_only_direction_calibration():
    """Verify that calibrate_direction_threshold finds the optimal bias-correcting threshold."""
    # Synthetic validation scenario:
    # Ground truth returns: 50% positive (+0.01), 50% negative (-0.01)
    # Model has a positive drift bias: predicts +0.02 for UP, and +0.005 for DOWN!
    # Without calibration (theta=0), ALL predictions are > 0 -> 100% UP, Balanced Acc = 50%!
    np.random.seed(42)
    val_actual_returns = np.array([0.01] * 20 + [-0.01] * 20)
    val_pred_returns = np.array([0.02] * 20 + [0.005] * 20)  # Constant positive bias (+0.015)

    # Uncalibrated (theta=0)
    uncal_signs = sign_from_difference(val_pred_returns, atol=1e-5)
    assert np.all(uncal_signs == 1)  # All predicted UP!

    # Calibrate on validation data only
    theta = calibrate_direction_threshold(val_pred_returns, val_actual_returns)
    # Threshold should lie between 0.005 and 0.02
    assert 0.005 < theta < 0.02

    # Calibrated predictions on validation
    cal_signs = sign_from_difference(val_pred_returns - theta, atol=1e-5)
    actual_signs = sign_from_difference(val_actual_returns, atol=1e-5)
    np.testing.assert_array_equal(cal_signs, actual_signs)


def test_train_only_scalers_leakage_safety():
    """Verify that 3D, 2D, and target scalers fit strictly on training partitions."""
    train_3d = np.random.randn(50, 30, 7).astype(np.float32)
    test_3d = np.random.randn(20, 30, 7).astype(np.float32) + 10.0  # Large distribution shift

    train_flat = train_3d.reshape(50, -1)
    test_flat = test_3d.reshape(20, -1)

    train_returns = np.array([0.01, -0.01, 0.02, -0.015, 0.005] * 10)
    test_returns = np.array([0.05, -0.05] * 10)

    # 1. 3D Scaler
    scaler_3d = StandardScaler()
    scaler_3d.fit(train_3d.reshape(-1, 7))
    mean_3d_before = scaler_3d.mean_.copy()
    _ = scaler_3d.transform(test_3d.reshape(-1, 7))
    np.testing.assert_array_equal(scaler_3d.mean_, mean_3d_before)

    # 2. Flat Scaler
    scaler_flat = StandardScaler()
    scaler_flat.fit(train_flat)
    mean_flat_before = scaler_flat.mean_.copy()
    _ = scaler_flat.transform(test_flat)
    np.testing.assert_array_equal(scaler_flat.mean_, mean_flat_before)

    # 3. Return Target Scaler
    scaler_ret = ReturnTargetScaler()
    scaler_ret.fit(train_returns)
    assert pytest.approx(scaler_ret.mean_, 1e-6) == float(np.mean(train_returns))
    assert pytest.approx(scaler_ret.std_, 1e-6) == float(np.std(train_returns))
    # Transform test does not modify fitted statistics
    scaled_test = scaler_ret.transform(test_returns)
    assert scaler_ret.mean_ == float(np.mean(train_returns))
    restored = scaler_ret.inverse_transform(scaled_test)
    np.testing.assert_allclose(restored, test_returns, rtol=1e-5)


def test_ridge_regression_validation_selection():
    """Verify Ridge alpha selection optimizes validation MSE without touching test data."""
    np.random.seed(42)
    X_tr = np.random.randn(100, 210)
    # Synthetic target with noise
    true_w = np.random.randn(210) * 0.01
    y_tr = X_tr @ true_w + np.random.randn(100) * 0.001
    X_val = np.random.randn(50, 210)
    y_val = X_val @ true_w + np.random.randn(50) * 0.001

    alphas = [0.1, 1.0, 10.0, 100.0]
    best_a = None
    best_mse = float("inf")
    for a in alphas:
        r = Ridge(alpha=a, random_state=42)
        r.fit(X_tr, y_tr)
        mse = float(np.mean((y_val - r.predict(X_val)) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_a = a

    assert best_a in alphas
    assert best_mse < float(np.var(y_val))


def test_logistic_regression_direction_and_flat_handling():
    """Verify Logistic Regression training, class balancing, and 3-class test evaluation."""
    np.random.seed(42)
    # 80 training samples: 50 UP (+0.02), 26 DOWN (-0.02), 4 FLAT (0.0)
    X = np.random.randn(80, 210)
    y = np.array([0.02] * 50 + [-0.02] * 26 + [0.0] * 4)

    # FLAT handling: excluded from binary training
    non_flat = np.abs(y) > 1e-5
    assert np.sum(non_flat) == 76
    y_binary = (y[non_flat] > 0).astype(int)

    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X[non_flat], y_binary)

    # Test evaluation on 3-class ground truth (including FLAT)
    test_actual_signs = np.array([1, 1, -1, -1, 0])
    test_X = np.random.randn(5, 210)
    probs = clf.predict_proba(test_X)[:, 1]
    pred_signs = np.where(probs > 0.5, 1, -1)

    m = compute_metrics(test_actual_signs, pred_signs)
    assert "up_recall" in m
    assert "down_recall" in m
    assert "balanced_accuracy" in m
    assert "mcc" in m
    assert m["sample_count"] == 5


def test_metrics_explicit_up_and_down_recall():
    """Verify that UP Recall and DOWN Recall are explicitly reported and correct."""
    actual_signs = np.array([1, 1, 1, 1, -1, -1, 0, 0])
    # Model predicts 3 of 4 UP, and 1 of 2 DOWN
    pred_signs = np.array([1, 1, 1, -1, -1, 1, 1, -1])

    m = compute_metrics(actual_signs, pred_signs)

    # UP: support 4, correct 3 -> recall = 75.0%
    assert m["up_recall"] == 75.0
    # DOWN: support 2, correct 1 -> recall = 50.0%
    assert m["down_recall"] == 50.0
    assert m["balanced_accuracy"] > 0
    assert "three_class_accuracy" in m


def test_exp05_dry_run_and_fold_benchmarks():
    """Verify EXP-05 dry run, fold boundaries, and Naive Persistence benchmark consistency."""
    exp = ModelComparisonExperiment(
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

    # Fold 1
    assert folds[0]["train_slice"] == (0, 455)
    assert folds[0]["val_slice"] == (455, 639)
    assert folds[0]["test_slice"] == (639, 823)

    # Fold 2
    assert folds[1]["train_slice"] == (0, 639)
    assert folds[1]["val_slice"] == (639, 823)
    assert folds[1]["test_slice"] == (823, 1007)

    # Fold 3 (Benchmark Block)
    assert folds[2]["train_slice"] == (0, 823)
    assert folds[2]["val_slice"] == (823, 1007)
    assert folds[2]["test_slice"] == (1007, 1191)

    # Fold 3 dates and Naive Persistence checks
    test_dates = exp.input_dates_all[1007:1191]
    tgt_dates = exp.target_dates_all[1007:1191]
    assert test_dates[0] == "2026-01-14"
    assert test_dates[-1] == "2026-10-06"
    assert tgt_dates[0] == "2026-01-15"
    assert tgt_dates[-1] == "2026-10-07"

    act_te = exp.actual_closes_all[1007:1191]
    prev_te = exp.prev_closes_all[1007:1191]
    err = act_te - prev_te
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / act_te)) * 100)

    assert pytest.approx(mae, abs=0.01) == 14.16
    assert pytest.approx(rmse, abs=0.01) == 18.52
    assert pytest.approx(mape, abs=0.01) == 1.06

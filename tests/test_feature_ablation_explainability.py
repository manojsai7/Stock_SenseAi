"""Unit tests for pure feature-ablation helpers (no TensorFlow/model files needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_feature_ablation_explainability import (
    neutralize_feature,
    price_predictions,
    regression_metrics,
)


def test_neutralize_feature_changes_only_selected_channel():
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    result = neutralize_feature(source, 2)

    assert result.shape == source.shape
    assert np.all(result[:, :, 2] == 0.0)
    np.testing.assert_array_equal(result[:, :, [0, 1, 3]], source[:, :, [0, 1, 3]])
    # The original input is not mutated.
    assert np.any(source[:, :, 2] != 0.0)


def test_price_reconstruction_uses_simple_return():
    origin = np.array([100.0, 200.0])
    returns = np.array([0.05, -0.10])
    np.testing.assert_allclose(price_predictions(origin, returns), [105.0, 180.0])


def test_regression_metrics_are_correct():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["mae_inr"] == pytest.approx(15.0)
    assert metrics["rmse_inr"] == pytest.approx(np.sqrt(250.0))
    assert metrics["mape_pct"] == pytest.approx(10.0)


def test_regression_metrics_reject_empty_inputs():
    with pytest.raises(ValueError):
        regression_metrics(np.array([]), np.array([]))


def test_price_reconstruction_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        price_predictions(np.array([100.0]), np.array([0.01, 0.02]))


def test_scale_windows_with_sklearn_scaler_preserves_3d_shape():
    from sklearn.preprocessing import StandardScaler
    from scripts.run_feature_ablation_explainability import scale_windows, FEATURE_COLUMNS, WINDOW_SIZE

    rng = np.random.default_rng(7)
    fit_data = rng.normal(size=(200, len(FEATURE_COLUMNS)))
    scaler = StandardScaler().fit(fit_data)
    windows = rng.normal(size=(4, WINDOW_SIZE, len(FEATURE_COLUMNS))).astype(np.float32)
    scaled = scale_windows(scaler, windows)
    assert scaled.shape == windows.shape
    assert scaled.dtype == np.float32
    assert np.isfinite(scaled).all()


def test_ablation_summary_returns_one_row_per_feature_and_method():
    import pandas as pd
    from scripts.run_feature_ablation_explainability import (
        FEATURE_COLUMNS,
        summarize_historical_ablation,
    )

    class IdentityTargetScaler:
        @staticmethod
        def inverse_transform(values):
            return np.asarray(values, dtype=float)

    class TinyDeterministicModel:
        @staticmethod
        def predict(windows, batch_size=64, verbose=0):
            # Depend on feature channel zero only, making the helper deterministic.
            return np.mean(windows[:, :, 0], axis=1, keepdims=True)

    rng = np.random.default_rng(19)
    scaled_windows = rng.normal(size=(6, 30, len(FEATURE_COLUMNS))).astype(np.float32)
    origin_close = np.array([100, 101, 99, 102, 98, 100], dtype=float)
    actual_close = np.array([101, 99, 100, 103, 97, 101], dtype=float)
    samples = pd.DataFrame({"Close": origin_close, "target_close": actual_close})

    neutralization, permutation = summarize_historical_ablation(
        TinyDeterministicModel(), IdentityTargetScaler(), scaled_windows, samples, permutations=3, seed=5
    )
    assert len(neutralization) == len(FEATURE_COLUMNS)
    assert len(permutation) == len(FEATURE_COLUMNS)
    assert set(neutralization["method"]) == {"mean_neutralization"}
    assert set(permutation["method"]) == {"window_permutation"}
    assert "delta_mae_inr" in neutralization.columns
    assert "mean_delta_mae_inr" in permutation.columns
    # Model uses feature zero, so neutralizing that channel changes its predictions.
    row = neutralization.loc[neutralization["feature"] == FEATURE_COLUMNS[0]].iloc[0]
    assert row["mean_abs_predicted_return_shift"] > 0

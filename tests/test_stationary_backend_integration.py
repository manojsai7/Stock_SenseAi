"""
Tests for Phase 3: Versioned Model Registry & Production Inference (Stationary-Return Pipeline).

Covers all 16 required test scenarios:
1. Stationary artifact loading and caching.
2. Rejection of mismatched feature order.
3. Rejection of a corrupt or missing artifact.
4. Legacy model loading remains unchanged.
5. Latest completed-session filtering.
6. Exact input shape (1, 30, 7).
7. Feature and target scaler usage.
8. Correct inverse-transform and return-to-price reconstruction.
9. Positive, negative, and zero predicted returns.
10. Non-finite model output fails safely.
11. Unsupported Ather Energy stationary prediction returns a clear error.
12. Pipeline version is respected by prediction dispatch.
13. Legacy prediction responses remain compatible.
14. Stationary response satisfies PredictionResponse.
15. Cache keys differ between pipelines.
16. Validation metrics are not presented as unseen-test benchmarks.
"""

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.core.cache import cache
from backend.core.config import settings
from backend.schemas.prediction import PredictionResponse
from backend.services.model_registry import ModelRegistry, model_registry
from backend.services.prediction_service import PredictionService
from ml.features.stationary import FEATURE_COLUMNS, reconstruct_price_from_return


def generate_synthetic_ohlcv(
    end_date: date = date(2026, 10, 7),
    num_days: int = 120,
    base_price: float = 3000.0,
    include_future_candle: bool = False
) -> pd.DataFrame:
    """Generates synthetic daily OHLCV bars ending on end_date."""
    dates = []
    curr = end_date
    while len(dates) < num_days:
        if curr.weekday() < 5:  # Monday to Friday
            dates.append(curr)
        curr -= timedelta(days=1)
    dates.reverse()

    if include_future_candle:
        # Add an incomplete intraday candle after end_date
        next_day = end_date + timedelta(days=1)
        dates.append(next_day)

    np.random.seed(42)
    rows = []
    price = base_price
    for d in dates:
        ret = np.random.normal(0.0005, 0.015)
        close = price * (1.0 + ret)
        high = max(price, close) * (1.0 + abs(np.random.normal(0.005, 0.003)))
        low = min(price, close) * (1.0 - abs(np.random.normal(0.005, 0.003)))
        open_p = price
        volume = int(np.random.uniform(500000, 2000000))
        rows.append({
            "Date": pd.Timestamp(d),
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })
        price = close

    return pd.DataFrame(rows)


# =============================================================================
# 1. Stationary artifact loading and caching
# =============================================================================
def test_stationary_artifact_loading_and_caching():
    test_symbol = "TCS.NS"
    assert model_registry.has_stationary_artifacts(test_symbol)

    # First load
    model_1 = model_registry.get_stationary_model(test_symbol)
    scaler_1 = model_registry.get_stationary_feature_scaler(test_symbol)
    target_scaler_1 = model_registry.get_stationary_target_scaler(test_symbol)
    metadata_1 = model_registry.get_stationary_metadata(test_symbol)
    manifest_1 = model_registry.get_stationary_manifest(test_symbol)

    assert model_1 is not None
    assert scaler_1.is_fitted
    assert target_scaler_1.is_fitted
    assert metadata_1["pipeline_version"] == "stationary-return-v1"
    assert manifest_1["pipeline_version"] == "stationary-return-v1"

    # Second load must return identical cached objects without reloading from disk
    model_2 = model_registry.get_stationary_model(test_symbol)
    assert model_1 is model_2
    assert scaler_1 is model_registry.get_stationary_feature_scaler(test_symbol)


# =============================================================================
# 2. Rejection of mismatched feature order
# =============================================================================
def test_rejection_of_mismatched_feature_order(tmp_path):
    # Create a corrupted temporary stationary directory with mismatched feature order
    symbol = "TCS.NS"
    src_dir = model_registry._get_stationary_paths(symbol)["dir"]
    temp_stationary_dir = tmp_path / "models_storage_stationary"
    temp_stock_dir = temp_stationary_dir / "TCS_NS"
    shutil.copytree(src_dir, temp_stock_dir)

    # Tamper with metadata.json feature order
    meta_path = temp_stock_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["features"] = list(reversed(FEATURE_COLUMNS))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Update manifest to match new metadata sha256 so it reaches feature validation
    manifest_path = temp_stock_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    import hashlib
    hasher = hashlib.sha256()
    with open(meta_path, "rb") as f:
        hasher.update(f.read())
    manifest["artifacts"]["metadata"]["sha256"] = hasher.hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    registry = ModelRegistry(stationary_models_dir=temp_stationary_dir)
    with pytest.raises(ValueError, match="Mismatched feature columns or order"):
        registry.load_stationary_artifacts(symbol)


# =============================================================================
# 3. Rejection of a corrupt or missing artifact
# =============================================================================
def test_rejection_of_corrupt_artifact_checksum(tmp_path):
    symbol = "TCS.NS"
    src_dir = model_registry._get_stationary_paths(symbol)["dir"]
    temp_stationary_dir = tmp_path / "models_storage_stationary"
    temp_stock_dir = temp_stationary_dir / "TCS_NS"
    shutil.copytree(src_dir, temp_stock_dir)

    # Corrupt model file contents
    model_path = temp_stock_dir / "model_cnn_lstm_stationary.keras"
    with open(model_path, "ab") as f:
        f.write(b"CORRUPTION_BYTES")

    registry = ModelRegistry(stationary_models_dir=temp_stationary_dir)
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        registry.load_stationary_artifacts(symbol)


def test_rejection_of_missing_artifact(tmp_path):
    symbol = "TCS.NS"
    src_dir = model_registry._get_stationary_paths(symbol)["dir"]
    temp_stationary_dir = tmp_path / "models_storage_stationary"
    temp_stock_dir = temp_stationary_dir / "TCS_NS"
    shutil.copytree(src_dir, temp_stock_dir)

    # Delete return_target_scaler.pkl
    (temp_stock_dir / "return_target_scaler.pkl").unlink()

    registry = ModelRegistry(stationary_models_dir=temp_stationary_dir)
    with pytest.raises(FileNotFoundError, match="Missing stationary model artifacts"):
        registry.load_stationary_artifacts(symbol)


# =============================================================================
# 4. Legacy model loading remains unchanged
# =============================================================================
def test_legacy_model_loading_unchanged():
    legacy_symbol = "RELIANCE.NS"
    assert model_registry.has_artifacts(legacy_symbol)
    model = model_registry.get_model(legacy_symbol)
    scaler = model_registry.get_scaler(legacy_symbol)
    metadata = model_registry.get_metadata(legacy_symbol)

    assert model is not None
    assert scaler.is_fitted
    assert len(metadata["features"]) == 6
    assert metadata["symbol"] == legacy_symbol


# =============================================================================
# 5. Latest completed-session filtering
# =============================================================================
def test_latest_completed_session_filtering():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)

    # Synthetic data includes an incomplete candle dated 2026-10-08
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80, include_future_candle=True)
    assert raw_df["Date"].max().date() > completed_date

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):

        cache.clear()
        result = service.get_prediction(symbol, pipeline_version="stationary-return-v1")
        assert result["last_observed_date"] == completed_date.strftime("%Y-%m-%d")
        assert result["latest_completed_session"] == completed_date.strftime("%Y-%m-%d")


# =============================================================================
# 6. Exact input shape (1, 30, 7)
# =============================================================================
def test_exact_input_shape_to_model():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    captured_inputs = []
    real_model = model_registry.get_stationary_model(symbol)

    def spy_predict(tensor, verbose=0):
        captured_inputs.append(tensor)
        return np.array([[0.05]], dtype=np.float32)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date), \
         patch.object(real_model, "predict", side_effect=spy_predict):

        cache.clear()
        service.get_prediction(symbol, pipeline_version="stationary-return-v1")

    assert len(captured_inputs) == 1
    input_tensor = captured_inputs[0]
    assert input_tensor.shape == (1, 30, 7)
    assert not np.isnan(input_tensor).any()


# =============================================================================
# 7. Feature and target scaler usage
# =============================================================================
def test_scalers_used_without_refitting():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    feat_scaler = model_registry.get_stationary_feature_scaler(symbol)
    tgt_scaler = model_registry.get_stationary_target_scaler(symbol)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date), \
         patch.object(feat_scaler, "transform", wraps=feat_scaler.transform) as spy_f_transform, \
         patch.object(feat_scaler, "fit") as mock_f_fit, \
         patch.object(tgt_scaler, "inverse_transform", wraps=tgt_scaler.inverse_transform) as spy_t_inv, \
         patch.object(tgt_scaler, "fit") as mock_t_fit:

        cache.clear()
        service.get_prediction(symbol, pipeline_version="stationary-return-v1")

        assert spy_f_transform.called
        assert spy_t_inv.called
        mock_f_fit.assert_not_called()
        mock_t_fit.assert_not_called()


# =============================================================================
# 8. Correct inverse-transform and return-to-price reconstruction
# =============================================================================
def test_return_to_price_reconstruction():
    last_close = 3500.00
    pred_return = 0.0125  # +1.25%
    expected_price = 3500.00 * (1.0 + 0.0125)
    reconstructed = reconstruct_price_from_return(last_close, pred_return)
    assert abs(reconstructed - expected_price) < 1e-6
    assert round(reconstructed, 2) == 3543.75


# =============================================================================
# 9. Positive, negative, and zero predicted returns
# =============================================================================
@pytest.mark.parametrize("return_val,expected_dir", [
    (0.015, "BULLISH"),
    (-0.015, "BEARISH"),
    (0.00001, "FLAT"),
])
def test_directional_labeling_for_positive_negative_zero_returns(return_val, expected_dir):
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80, base_price=3000.0)

    tgt_scaler = model_registry.get_stationary_target_scaler(symbol)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date), \
         patch.object(tgt_scaler, "inverse_transform", return_value=np.array([return_val])):

        cache.clear()
        res = service.get_prediction(symbol, pipeline_version="stationary-return-v1")
        assert res["direction"] == expected_dir


# =============================================================================
# 10. Non-finite model output fails safely
# =============================================================================
def test_non_finite_model_output_fails_safely():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    tgt_scaler = model_registry.get_stationary_target_scaler(symbol)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date), \
         patch.object(tgt_scaler, "inverse_transform", return_value=np.array([np.nan])):

        cache.clear()
        with pytest.raises(RuntimeError, match="Non-finite return prediction"):
            service.get_prediction(symbol, pipeline_version="stationary-return-v1")


# =============================================================================
# 11. Unsupported Ather Energy stationary prediction returns a clear error
# =============================================================================
def test_ather_energy_stationary_prediction_unsupported():
    service = PredictionService()
    cache.clear()
    with pytest.raises(FileNotFoundError, match="Ather Energy has insufficient historical trading sessions"):
        service.get_prediction("ATHERENERG.NS", pipeline_version="stationary-return-v1")


# =============================================================================
# 12. Pipeline version is respected by prediction dispatch
# =============================================================================
def test_pipeline_version_dispatch():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):

        cache.clear()
        pred_stat = service.get_prediction(symbol, pipeline_version="stationary-return-v1")
        assert pred_stat["model_version"] == "cnn_lstm_stationary_v1"
        assert len(pred_stat["features_used"]) == 7

        cache.clear()
        pred_leg = service.get_prediction(symbol, pipeline_version="legacy")
        assert pred_leg["model_version"] != "cnn_lstm_stationary_v1"
        assert len(pred_leg["features_used"]) == 6


# =============================================================================
# 13. Legacy prediction responses remain compatible
# =============================================================================
def test_legacy_prediction_response_compatibility():
    service = PredictionService()
    symbol = "RELIANCE.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):

        cache.clear()
        legacy_pred = service.get_prediction(symbol, pipeline_version="legacy")
        parsed = PredictionResponse(**legacy_pred)
        assert parsed.symbol == symbol
        assert len(parsed.features_used) == 6


# =============================================================================
# 14. Stationary response satisfies PredictionResponse
# =============================================================================
def test_stationary_response_satisfies_prediction_response():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):

        cache.clear()
        stationary_pred = service.get_prediction(symbol, pipeline_version="stationary-return-v1")
        parsed = PredictionResponse(**stationary_pred)
        assert parsed.symbol == symbol
        assert parsed.model_version == "cnn_lstm_stationary_v1"
        assert parsed.features_used == FEATURE_COLUMNS
        assert parsed.predicted_close > 0
        assert parsed.direction in ["BULLISH", "BEARISH", "FLAT"]


# =============================================================================
# 15. Cache keys differ between pipelines
# =============================================================================
def test_cache_keys_differ_between_pipelines():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):

        cache.clear()
        service.get_prediction(symbol, pipeline_version="legacy")
        service.get_prediction(symbol, pipeline_version="stationary-return-v1")

        legacy_key = f"pred:legacy:{symbol}:{completed_date.strftime('%Y-%m-%d')}"
        stationary_key = f"pred:stationary-return-v1:{symbol}:{completed_date.strftime('%Y-%m-%d')}"

        assert cache.get(legacy_key) is not None
        assert cache.get(stationary_key) is not None
        assert cache.get(legacy_key)["model_version"] != cache.get(stationary_key)["model_version"]


# =============================================================================
# 16. Validation metrics are not presented as unseen-test benchmarks
# =============================================================================
def test_validation_metrics_not_presented_as_unseen_benchmarks():
    service = PredictionService()
    symbol = "TCS.NS"
    completed_date = date(2026, 10, 7)
    raw_df = generate_synthetic_ohlcv(end_date=completed_date, num_days=80)

    with patch.object(service.provider, "fetch_historical_ohlcv", return_value=raw_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):

        cache.clear()
        res = service.get_prediction(symbol, pipeline_version="stationary-return-v1")
        assert res["historical_directional_accuracy"] is None
        assert res["directional_accuracy_benchmark"] is None
        assert res["mae_benchmark"] is None
        assert res["rmse_benchmark"] is None
        assert res["mape_benchmark"] is None

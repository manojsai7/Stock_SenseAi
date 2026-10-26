import numpy as np
import pandas as pd
import pytest

from backend.services import explainability_service as module
from backend.services.explainability_service import ExplainabilityService
from ml.features.stationary import FEATURE_COLUMNS


class FakeProvider:
    def __init__(self, frame):
        self.frame = frame

    def fetch_historical_ohlcv(self, symbol, period, interval):
        assert symbol == "RELIANCE.NS"
        assert period == "1y"
        assert interval == "1d"
        return self.frame.copy()


class FakeFeatureScaler:
    def transform(self, frame):
        return frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


class FakeTargetScaler:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=float).reshape(-1)


class FakeModel:
    def predict(self, tensor, verbose=0):
        # Depend on every feature so neutralization produces deterministic shifts.
        weights = np.arange(1, 8, dtype=float).reshape(1, 1, 7)
        score = float(np.mean(tensor * weights))
        return np.asarray([[score * 1e-4]], dtype=float)


def make_bars():
    dates = pd.bdate_range("2026-01-01", periods=100)
    close = 100 + np.arange(len(dates)) * 0.2 + np.sin(np.arange(len(dates)) / 3)
    frame = pd.DataFrame({
        "Date": dates,
        "Open": close * (1 - 0.002),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": 100_000 + np.arange(len(dates)) * 250,
    })
    frame.attrs["data_source"] = "unit-test-provider"
    frame.attrs["source_status"] = "ONLINE"
    return frame


def test_live_sensitivity_returns_all_features_and_window_provenance(monkeypatch):
    frame = make_bars()
    latest_date = frame["Date"].iloc[-1].date()
    service = ExplainabilityService()
    service.provider = FakeProvider(frame)

    monkeypatch.setattr(module.market_status_service, "get_market_status", lambda: "CLOSED")
    monkeypatch.setattr(module.market_status_service, "get_latest_completed_trading_day", lambda: latest_date)
    monkeypatch.setattr(module.market_status_service, "get_next_trading_day", lambda _: latest_date + pd.Timedelta(days=1))
    monkeypatch.setattr(module.model_registry, "is_stationary_supported", lambda _: True)
    monkeypatch.setattr(module.model_registry, "has_stationary_artifacts", lambda _: True)
    monkeypatch.setattr(module.model_registry, "get_stationary_model", lambda _: FakeModel())
    monkeypatch.setattr(module.model_registry, "get_stationary_feature_scaler", lambda _: FakeFeatureScaler())
    monkeypatch.setattr(module.model_registry, "get_stationary_target_scaler", lambda _: FakeTargetScaler())
    monkeypatch.setattr(module.model_registry, "get_stationary_metadata", lambda _: {
        "model_version": "fake-v1",
        "stage_b_final_deployment_refit": {"training_data_end": "2025-12-31"},
    })

    result = service.get_stationary_feature_sensitivity("reliance.ns")

    assert result["symbol"] == "RELIANCE.NS"
    assert result["window_size"] == 30
    assert result["input_window_end"] == latest_date.strftime("%Y-%m-%d")
    assert result["data_status"] == "ONLINE"
    assert len(result["feature_sensitivities"]) == 7
    assert {item["feature"] for item in result["feature_sensitivities"]} == set(FEATURE_COLUMNS)
    assert all(np.isfinite(item["delta_return_bps"]) for item in result["feature_sensitivities"])
    assert "not causal" in result["disclaimer"].lower() or "not causal attribution" in result["disclaimer"].lower()


def test_live_sensitivity_rejects_unsupported_symbol(monkeypatch):
    service = ExplainabilityService()
    monkeypatch.setattr(module.model_registry, "is_stationary_supported", lambda _: False)
    monkeypatch.setattr(module.model_registry, "has_stationary_artifacts", lambda _: False)
    with pytest.raises(FileNotFoundError):
        service.get_stationary_feature_sensitivity("ATHERENERG.NS")

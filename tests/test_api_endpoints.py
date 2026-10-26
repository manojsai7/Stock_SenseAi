from datetime import date
from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from backend.api.prediction_routes import pred_service
from backend.core.cache import cache

client = TestClient(app)


def _mock_ohlcv(last_day: str = "2026-10-26", periods: int = 60, base_price: float = 1200.0, source_status: str = "ONLINE") -> pd.DataFrame:
    dates = pd.bdate_range(end=last_day, periods=periods)
    df = pd.DataFrame({
        "Date": dates,
        "Open": [base_price * 0.99] * len(dates),
        "High": [base_price * 1.02] * len(dates),
        "Low": [base_price * 0.98] * len(dates),
        "Close": [base_price] * len(dates),
        "Volume": [1000000] * len(dates)
    })
    df.attrs["source_status"] = source_status
    df.attrs["data_source"] = "Mocked Yahoo Finance Data Feed"
    return df

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "market_status" in data
    assert "services" in data
    assert "models" in data
    assert "cache" in data

def test_ready_endpoint():
    response = client.get("/api/system/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ready", "not_ready"]
    assert "models_accessible" in data
    assert data["models_accessible"] is True


def test_stocks_search():
    response = client.get("/api/stocks/search?q=TCS")
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert any(s["symbol"] == "TCS.NS" for s in results)

def test_models_list():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 5
    symbols = [m["symbol"] for m in models]
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols

def test_benchmark_experiments():
    response = client.get("/api/benchmark/experiments")
    assert response.status_code == 200
    benchmarks = response.json()
    assert isinstance(benchmarks, list)
    assert len(benchmarks) > 0
    # Verify row columns
    row = benchmarks[0]
    assert "model" in row
    assert "directional_accuracy" in row

def test_quote_endpoint():
    response = client.get("/api/stocks/RELIANCE.NS/quote")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert "price" in data
    assert "change" in data
    assert "market_status" in data
    assert "data_status" in data
    assert data["source"] == "Yahoo Finance"

def test_history_endpoint():
    response = client.get("/api/stocks/RELIANCE.NS/history?period=1mo")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert data["count"] > 0
    assert "data" in data
    candle = data["data"][0]
    assert "open" in candle
    assert "close" in candle

def test_predict_endpoint_contract_fresh_data():
    """Verify that predictions succeed with HTTP 200 on fresh, valid completed market data."""
    completed_date = date(2026, 10, 26)
    fresh_df = _mock_ohlcv(last_day="2026-10-26", periods=60)
    cache.clear()

    with patch.object(pred_service.provider, "fetch_historical_ohlcv", return_value=fresh_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):
        response = client.get("/api/predict/RELIANCE.NS")

    assert response.status_code == 200
    pred = response.json()
    assert pred["symbol"] == "RELIANCE.NS"
    assert pred["pipeline_version"] == "stationary-return-v1"
    assert "predicted_close" in pred
    assert "last_close" in pred
    assert pred["predicted_close"] > 0
    assert pred["direction"] in ["BULLISH", "BEARISH", "FLAT"]
    assert "target_date" in pred
    assert "latest_completed_session" in pred
    assert pred["input_window_size"] == 30
    assert len(pred["features_used"]) == 7
    assert pred["data_source"] == "Yahoo Finance"
    assert pred["data_status"] == "ONLINE"


def test_predict_endpoint_default_pipeline_is_stationary_and_preserves_legacy():
    """Regression test: default pipeline is stationary-return-v1, and legacy remains explicitly available."""
    completed_date = date(2026, 10, 26)
    fresh_df = _mock_ohlcv(last_day="2026-10-26", periods=60)
    cache.clear()

    with patch.object(pred_service.provider, "fetch_historical_ohlcv", return_value=fresh_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):
        # 1. Default without parameter -> stationary-return-v1
        resp_default = client.get("/api/predict/RELIANCE.NS")
        assert resp_default.status_code == 200
        assert resp_default.json()["pipeline_version"] == "stationary-return-v1"

        cache.clear()
        # 2. Explicit legacy query param -> legacy
        resp_legacy = client.get("/api/predict/RELIANCE.NS?pipeline_version=legacy")
        assert resp_legacy.status_code == 200
        assert resp_legacy.json()["pipeline_version"] == "legacy"


def test_predict_endpoint_stale_data_returns_503():
    """Verify that the freshness guard correctly fails closed with HTTP 503 when market data is stale."""
    completed_date = date(2026, 10, 26)
    # Stale data ending earlier than the exchange completed date
    stale_df = _mock_ohlcv(last_day="2026-10-20", periods=60)
    cache.clear()

    with patch.object(pred_service.provider, "fetch_historical_ohlcv", return_value=stale_df), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=completed_date):
        response = client.get("/api/predict/RELIANCE.NS")

    assert response.status_code == 503
    assert "Market history is stale" in response.json()["detail"]


def test_unsupported_symbol_prediction_error():
    response = client.get("/api/predict/INVALID_EQUITY_ABC.NS")
    assert response.status_code in [400, 404]

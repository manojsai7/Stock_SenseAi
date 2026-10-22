from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

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

def test_predict_endpoint_contract():
    response = client.get("/api/predict/RELIANCE.NS")
    assert response.status_code == 200
    pred = response.json()
    assert pred["symbol"] == "RELIANCE.NS"
    assert "predicted_close" in pred
    assert "last_close" in pred
    assert pred["predicted_close"] > 0
    assert pred["direction"] in ["BULLISH", "BEARISH", "FLAT"]
    assert "target_date" in pred
    assert "latest_completed_session" in pred
    assert pred["input_window_size"] == 30
    assert "historical_directional_accuracy" in pred
    assert pred["data_source"] == "Yahoo Finance"

def test_unsupported_symbol_prediction_error():
    response = client.get("/api/predict/INVALID_EQUITY_ABC.NS")
    assert response.status_code in [400, 404]

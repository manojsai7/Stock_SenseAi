import pytest
import math
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.historical_predictions_service import (
    HistoricalPredictionsService,
    safe_float,
    calculate_mcc,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_coverage_matrix():
    """Verify multi-stock coverage matrix returns all supported equities with appropriate flags."""
    service = HistoricalPredictionsService()
    matrix = service.get_model_coverage_matrix()
    
    assert isinstance(matrix, list)
    symbols = {item["symbol"] for item in matrix}
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols
    assert "INFY.NS" in symbols
    assert "HDFCBANK.NS" in symbols
    assert "ICICIBANK.NS" in symbols
    
    reliance_item = next(item for item in matrix if item["symbol"] == "RELIANCE.NS")
    assert reliance_item["gru_v2_challenger"] is True
    assert reliance_item["total_archived_predictions"] > 0


def test_load_reliance_exp06_predictions():
    """Verify loading EXP-06 predictions for RELIANCE.NS across 3 folds."""
    service = HistoricalPredictionsService()
    res = service.get_historical_predictions(symbol="RELIANCE.NS", experiment="EXP-06")
    
    assert res["supported"] is True
    assert len(res["records"]) > 0
    
    models = res["available_models"]
    assert "GRU V2 Challenger" in models or "candidate_1_seed_42" in models
    
    folds = res["available_folds"]
    assert any("Fold 1" in f for f in folds)
    assert any("Fold 2" in f for f in folds)
    assert any("Fold 3" in f for f in folds)


def test_load_multi_stock_v1_audit_predictions():
    """Verify loading V1 Audit predictions for non-Reliance stocks (e.g. TCS.NS, INFY.NS)."""
    service = HistoricalPredictionsService()
    
    tcs_res = service.get_historical_predictions("TCS.NS")
    assert tcs_res["supported"] is True
    assert len(tcs_res["records"]) > 0
    assert all(r["symbol"] == "TCS.NS" for r in tcs_res["records"])
    
    infy_res = service.get_historical_predictions("INFY.NS")
    assert infy_res["supported"] is True
    assert len(infy_res["records"]) > 0
    assert all(r["symbol"] == "INFY.NS" for r in infy_res["records"])


def test_compute_predicting_capability_score():
    """Verify the 0-100 capability score formula, bounds, and tier labeling."""
    service = HistoricalPredictionsService()
    
    # Superior scenario
    mock_metrics = {
        "sample_size": 300,
        "price_mae": 13.23,
        "directional_accuracy_pct": 55.0,
        "mcc": 0.12,
        "up_recall_pct": 60.0,
        "down_recall_pct": 50.0,
    }
    mock_naive = {
        "price_mae": 13.34,
    }
    
    score_res = service.compute_predicting_capability_score(
        metrics=mock_metrics,
        naive_metrics=mock_naive,
        symbol="RELIANCE.NS",
        model_name="GRU V2 Challenger",
    )
    assert 0 <= score_res["score"] <= 100
    assert "components" in score_res
    assert score_res["tier"] in [
        "Exceptional Alpha",
        "High Relative Capability",
        "Moderate Predictive Capability",
        "Marginal Edge",
        "Sub-Baseline / Directionally Asymmetric",
    ]
    assert "directional_skill_score" in score_res["components"]
    assert "mcc_correlation_score" in score_res["components"]
    assert "price_precision_vs_naive_score" in score_res["components"]
    assert "regime_balance_score" in score_res["components"]

    # Underperforming / Zero sample scenario
    zero_res = service.compute_predicting_capability_score(
        metrics={"sample_size": 0},
        naive_metrics=None,
    )
    assert zero_res["score"] is None
    assert zero_res["badge"] == "PRELIMINARY"


def test_compute_slice_metrics_and_residuals():
    """Verify metrics calculation and residual distribution generation."""
    service = HistoricalPredictionsService()
    records = [
        {
            "record_id": "test_1",
            "symbol": "RELIANCE.NS",
            "target_date": "2026-01-01",
            "model_name": "GRU V2 Challenger",
            "fold": "Fold 1",
            "actual_close": 100.0,
            "previous_close": 98.0,
            "predicted_close": 102.0,
            "absolute_error": 2.0,
            "percentage_error": 2.0,
            "predicted_return": 0.04,
            "actual_return": 0.02,
            "actual_direction": "UP",
            "predicted_direction": "UP",
            "direction_correct_bool": True,
        },
        {
            "record_id": "test_2",
            "symbol": "RELIANCE.NS",
            "target_date": "2026-01-02",
            "model_name": "GRU V2 Challenger",
            "fold": "Fold 1",
            "actual_close": 105.0,
            "previous_close": 100.0,
            "predicted_close": 103.0,
            "absolute_error": 2.0,
            "percentage_error": 1.9,
            "predicted_return": -0.01,
            "actual_return": 0.05,
            "actual_direction": "UP",
            "predicted_direction": "DOWN",
            "direction_correct_bool": False,
        },
        {
            "record_id": "test_3",
            "symbol": "RELIANCE.NS",
            "target_date": "2026-01-03",
            "model_name": "GRU V2 Challenger",
            "fold": "Fold 1",
            "actual_close": 98.0,
            "previous_close": 105.0,
            "predicted_close": 97.0,
            "absolute_error": 1.0,
            "percentage_error": 1.02,
            "predicted_return": -0.02,
            "actual_return": -0.06,
            "actual_direction": "DOWN",
            "predicted_direction": "DOWN",
            "direction_correct_bool": True,
        },
    ]
    
    metrics = service.compute_slice_metrics(records)
    assert metrics["sample_size"] == 3
    assert pytest.approx(metrics["price_mae"], 0.01) == 1.67
    assert metrics["confusion"]["tp"] == 1
    assert metrics["confusion"]["tn"] == 1
    assert metrics["confusion"]["fn"] == 1
    assert metrics["confusion"]["fp"] == 0
    assert "residuals" in metrics


def test_api_historical_predictions_endpoint(client):
    """Verify GET /api/research/historical-predictions returns valid structure for RELIANCE.NS."""
    resp = client.get("/api/research/historical-predictions?symbol=RELIANCE.NS")
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["symbol"] == "RELIANCE.NS"
    assert data["supported"] is True
    assert "predicting_capability_scores" in data
    assert "metrics" in data
    assert "chart_data" in data
    assert "model_coverage_matrix" in data
    assert "available_models" in data
    assert "available_folds" in data
    assert len(data["chart_data"]) > 0


def test_api_historical_predictions_filtering(client):
    """Verify filtering by fold and model works via query params."""
    resp = client.get(
        "/api/research/historical-predictions?symbol=RELIANCE.NS&experiment=EXP-06&fold=Fold%203&model_name=GRU%20V2%20Challenger"
    )
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["supported"] is True
    assert data["selected_fold"] == "Fold 3"
    assert "GRU" in data["selected_model"]
    assert len(data["records"]) > 0
    # Every prediction returned should be Fold 3
    for p in data["records"]:
        assert "Fold 3" in p["fold"]


def test_api_historical_predictions_unsupported_symbol(client):
    """Verify unknown or un-audited symbol returns structured supported=False response."""
    resp = client.get("/api/research/historical-predictions?symbol=INVALID.NS")
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is False
    assert "model_coverage_matrix" in data

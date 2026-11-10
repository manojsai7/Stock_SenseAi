import pytest
import math
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.forecast_accuracy_service import (
    ForecastAccuracyService,
    calculate_mcc,
    assess_sample_size,
    normalize_model_family,
    safe_float,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_calculate_mcc_handles_edge_cases():
    """Verify MCC handles perfect, inverse, and zero-denominator states."""
    # Perfect correlation
    assert calculate_mcc(tp=10, tn=10, fp=0, fn=0) == 1.0

    # Inverse correlation
    assert calculate_mcc(tp=0, tn=0, fp=10, fn=10) == -1.0

    # Zero denominator (e.g. all true instances are UP) returns None for degenerate class variance
    assert calculate_mcc(tp=5, tn=0, fp=5, fn=0) is None

    # Mixed
    mcc = calculate_mcc(tp=8, tn=4, fp=2, fn=2)
    assert 0.0 < mcc < 1.0


def test_assess_sample_size_tiers():
    """Verify sample size interpretation categories."""
    assert assess_sample_size(0)[0] == "No Resolved Observations"
    assert "Very Limited Evidence" in assess_sample_size(1)[0]
    assert "Very Limited Evidence" in assess_sample_size(19)[0]
    assert "Preliminary Prospective Evaluation" in assess_sample_size(20)[0]
    assert "Developing Prospective Evidence" in assess_sample_size(60)[0]
    assert "Substantial Prospective Sample" in assess_sample_size(100)[0]


def test_normalize_model_family():
    """Verify canonical model family mapping."""
    assert normalize_model_family("GRU V2 Challenger") == "GRU V2 Challenger"
    assert normalize_model_family("Production CNN-LSTM") == "Production CNN-LSTM V1"
    assert normalize_model_family("Production CNN-LSTM V1") == "Production CNN-LSTM V1"
    assert normalize_model_family("Naive Persistence") == "Naive Persistence Baseline"
    assert normalize_model_family("Naive Persistence Benchmark") == "Naive Persistence Baseline"
    assert normalize_model_family("Colab EXP-05 Fold 3 (Calibrated)") == "Colab EXP-05 Candidate"


def test_compute_metrics_for_group_calculations(tmp_path):
    """Test MAE, RMSE, MAPE, Return BPS, DA, and MCC with synthetic rows."""
    service = ForecastAccuracyService()

    df = pd.DataFrame([
        {
            "target_date": "2026-11-01",
            "predicted_close": 102.0,
            "actual_close": 100.0,
            "price_error_abs": 2.0,
            "predicted_return": 0.02,
            "actual_return": 0.01,
            "predicted_direction": "UP",
            "actual_direction": "UP",
            "direction_correct_bool": True,
        },
        {
            "target_date": "2026-11-02",
            "predicted_close": 98.0,
            "actual_close": 100.0,
            "price_error_abs": 2.0,
            "predicted_return": -0.02,
            "actual_return": -0.01,
            "predicted_direction": "DOWN",
            "actual_direction": "DOWN",
            "direction_correct_bool": True,
        },
        {
            "target_date": "2026-11-03",
            "predicted_close": 105.0,
            "actual_close": 100.0,
            "price_error_abs": 5.0,
            "predicted_return": 0.05,
            "actual_return": -0.01,
            "predicted_direction": "UP",
            "actual_direction": "DOWN",
            "direction_correct_bool": False,
        },
    ])

    metrics = service._compute_metrics_for_group(df)
    assert metrics["sample_size"] == 3
    # MAE = (2 + 2 + 5) / 3 = 3.0
    assert metrics["price_mae"] == 3.0
    # RMSE = sqrt((4 + 4 + 25) / 3) = sqrt(33/3) = sqrt(11) = 3.32
    assert metrics["price_rmse"] == 3.32
    # MAPE = (2% + 2% + 5%) / 3 = 3.0%
    assert metrics["price_mape"] == 3.0
    # Return BPS = (100 bps + 100 bps + 600 bps) / 3 = 266.7 bps
    assert metrics["return_mae_bps"] == 266.7
    # Directional Acc = 2/3 = 66.7%
    assert metrics["directional_accuracy_pct"] == 66.7
    assert metrics["confusion"]["tp"] == 1
    assert metrics["confusion"]["tn"] == 1
    assert metrics["confusion"]["fp"] == 1
    assert metrics["confusion"]["fn"] == 0


def test_forecast_accuracy_service_unsupported_symbol():
    """Verify non-RELIANCE tickers return supported=False."""
    service = ForecastAccuracyService()
    payload = service.get_accuracy_payload("TCS.NS")
    assert payload["supported"] is False
    assert payload["symbol"] == "TCS.NS"
    assert "exclusively to RELIANCE.NS" in payload["message"]


def test_forecast_accuracy_api_endpoint(client):
    """Test GET /api/research/forecast-accuracy endpoint."""
    res = client.get("/api/research/forecast-accuracy?symbol=RELIANCE.NS")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert data["supported"] is True
    assert "per_family_metrics" in data
    assert "common_date_paired_metrics" in data
    assert "sample_size_assessment" in data
    assert "resolved_records" in data
    assert "pending_records" in data
    assert "chart_data" in data


def test_forecast_accuracy_api_unsupported_symbol(client):
    """Test GET /api/research/forecast-accuracy/INFY.NS returns supported=False."""
    res = client.get("/api/research/forecast-accuracy/INFY.NS")
    assert res.status_code == 200
    data = res.json()
    assert data["supported"] is False
    assert data["symbol"] == "INFY.NS"


def test_common_date_paired_metrics_isolation(tmp_path):
    """Test that paired metrics calculate only on shared dates."""
    exp06_csv = tmp_path / "exp06.csv"
    exp06_csv.write_text(
        "forecast_id,forecast_timestamp,symbol,input_window_end,target_date,model_name,model_hash,training_cutoff,previous_close,predicted_return,predicted_close,predicted_direction,actual_close,actual_return,actual_direction,direction_correct,price_error_abs,status,resolution_timestamp,data_source\n"
        "id1,ts,RELIANCE.NS,2026-11-06,2026-11-09,GRU V2 Challenger,h,c,1320.0,0.002,1322.64,UP,1324.4,0.0033,UP,True,1.76,RESOLVED,ts2,src\n"
        "id2,ts,RELIANCE.NS,2026-11-06,2026-11-09,Production CNN-LSTM V1,h,c,1320.0,-0.001,1318.68,DOWN,1324.4,0.0033,UP,False,5.72,RESOLVED,ts2,src\n"
        "id3,ts,RELIANCE.NS,2026-11-06,2026-11-09,Naive Persistence Benchmark,h,c,1320.0,0.0,1320.0,FLAT,1324.4,0.0033,UP,False,4.40,RESOLVED,ts2,src\n"
    )

    service = ForecastAccuracyService(exp06_log_path=exp06_csv, exp05_log_path=tmp_path / "nonexistent.csv")
    payload = service.get_accuracy_payload("RELIANCE.NS")
    paired = payload["common_date_paired_metrics"]

    assert paired["common_dates_count"] == 1
    assert paired["common_dates"] == ["2026-11-09"]
    assert "GRU V2 Challenger" in paired["metrics_per_model"]
    assert "Production CNN-LSTM V1" in paired["metrics_per_model"]
    assert "Naive Persistence Baseline" in paired["metrics_per_model"]
    assert paired["metrics_per_model"]["GRU V2 Challenger"]["price_mae"] == 1.76
    assert paired["metrics_per_model"]["Production CNN-LSTM V1"]["price_mae"] == 5.72


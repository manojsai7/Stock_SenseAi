import pytest
import os
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.market_status_service import market_status_service
from backend.services.prospective_v2_service import ProspectiveV2Service
from scripts.run_daily_shadow_evaluation import run_daily_shadow_evaluation


@pytest.fixture
def client():
    return TestClient(app)


def test_get_latest_forecast_endpoint(client):
    """Verify GET /api/research/latest-forecast returns valid structure with synchronized models."""
    resp = client.get("/api/research/latest-forecast?symbol=RELIANCE.NS")
    assert resp.status_code == 200
    data = resp.json()

    assert data["symbol"] == "RELIANCE.NS"
    assert data["supported"] is True
    assert "target_trading_date" in data
    assert "latest_completed_input_date" in data
    assert "market_status" in data
    assert "models" in data

    models = data["models"]
    assert "production_v1" in models
    assert "gru_v2_challenger" in models
    assert "naive_baseline" in models

    v1 = models["production_v1"]
    v2 = models["gru_v2_challenger"]
    naive = models["naive_baseline"]

    assert v1["is_experimental"] is False
    assert v2["is_experimental"] is True
    assert naive["is_experimental"] is False
    assert v1["target_trading_date"] == v2["target_trading_date"] == naive["target_trading_date"]
    assert v1["reference_close"] == v2["reference_close"] == naive["reference_close"]


def test_get_latest_forecast_unsupported_symbol(client):
    """Verify non-Reliance symbol returns structured supported=False."""
    resp = client.get("/api/research/latest-forecast?symbol=TCS.NS")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "TCS.NS"
    assert data["supported"] is False


def test_get_challenger_prediction_has_no_logging_side_effects(client, tmp_path):
    """
    CRITICAL INVARIANT: GET /api/research/challenger-prediction must be purely read-only
    and never mutate prospective evaluation logs.
    """
    log_path = Path("research/EXP-06/evaluation/prospective_predictions_log.csv")
    initial_mtime = log_path.stat().st_mtime if log_path.exists() else 0
    initial_rows = len(pd.read_csv(log_path)) if log_path.exists() else 0

    # Execute GET request
    resp = client.get("/api/research/challenger-prediction/RELIANCE.NS")
    # Response can be 200 or 503 (if stale/closed), but must NOT touch the log
    final_mtime = log_path.stat().st_mtime if log_path.exists() else 0
    final_rows = len(pd.read_csv(log_path)) if log_path.exists() else 0

    assert initial_rows == final_rows, "GET request mutated prospective prediction log!"
    assert initial_mtime == final_mtime, "GET request touched prospective prediction log mtime!"


def test_shadow_evaluation_idempotency_and_trio(tmp_path):
    """
    Verify daily shadow evaluation script creates synchronized trio and is strictly idempotent.
    """
    mock_log = tmp_path / "mock_prospective_log.csv"
    mock_registry = Path("research/EXP-06/evaluation/locked_evaluation_registry_v2.json")

    service = ProspectiveV2Service(registry_path=mock_registry, log_path=mock_log)

    # 1. Record first trio
    trio1 = service.record_daily_forecasts(
        symbol="RELIANCE.NS",
        target_date="2026-11-20",
        input_window_end="2026-11-19",
        previous_close=1350.0,
        v1_pred={"predicted_close": 1348.0, "predicted_return": -0.00148},
        v2_pred={"predicted_next_session_close": 1353.5, "predicted_simple_return": 0.00259, "data_provider": "TestProvider"},
    )
    assert len(trio1) == 3

    df1 = service._load_log()
    assert len(df1) == 3
    models_logged = set(df1["model_name"])
    assert "Production CNN-LSTM V1" in models_logged
    assert "GRU V2 Challenger" in models_logged
    assert "Naive Persistence Benchmark" in models_logged

    # 2. Record again with identical target date -> MUST BE IDEMPOTENT (no duplicate rows)
    trio2 = service.record_daily_forecasts(
        symbol="RELIANCE.NS",
        target_date="2026-11-20",
        input_window_end="2026-11-19",
        previous_close=1350.0,
        v1_pred={"predicted_close": 1348.0, "predicted_return": -0.00148},
        v2_pred={"predicted_next_session_close": 1353.5, "predicted_simple_return": 0.00259, "data_provider": "TestProvider"},
    )
    df2 = service._load_log()
    assert len(df2) == 3, "Idempotency failed: duplicate entries created for the same target date!"


def test_pending_outcomes_resolution(tmp_path):
    """Verify pending forecasts resolve accurately against historical closing prices."""
    mock_log = tmp_path / "mock_log.csv"
    mock_registry = Path("research/EXP-06/evaluation/locked_evaluation_registry_v2.json")
    service = ProspectiveV2Service(registry_path=mock_registry, log_path=mock_log)

    # Seed pending records
    service.record_daily_forecasts(
        symbol="RELIANCE.NS",
        target_date="2026-11-19",
        input_window_end="2026-11-18",
        previous_close=1000.0,
        v1_pred={"predicted_close": 1010.0, "predicted_return": 0.01},
        v2_pred={"predicted_next_session_close": 1005.0, "predicted_simple_return": 0.005},
    )

    # Historical dataframe where 2026-11-19 is now completed
    hist_df = pd.DataFrame([
        {"Date": "2026-11-18", "Close": 1000.0},
        {"Date": "2026-11-19", "Close": 1020.0}, # Actual close is 1020 (UP)
    ])

    resolved_count = service.resolve_pending_forecasts("RELIANCE.NS", hist_df)
    assert resolved_count == 3

    resolved_df = service._load_log()
    assert (resolved_df["status"] == "RESOLVED").all()
    assert (resolved_df["actual_close"] == 1020.0).all()
    assert (resolved_df["actual_direction"] == "UP").all()

    # V1 predicted 1010 (UP) -> correct
    v1_row = resolved_df[resolved_df["model_name"] == "Production CNN-LSTM V1"].iloc[0]
    assert bool(v1_row["direction_correct"]) is True
    assert v1_row["price_error_abs"] == 10.0

    # Naive predicted 1000 (FLAT) -> incorrect
    naive_row = resolved_df[resolved_df["model_name"] == "Naive Persistence Benchmark"].iloc[0]
    assert bool(naive_row["direction_correct"]) is False
    assert naive_row["price_error_abs"] == 20.0


def test_dry_run_daily_shadow_script():
    """Verify dry run executes cleanly without writing files."""
    code = run_daily_shadow_evaluation(symbol="RELIANCE.NS", dry_run=True)
    # Exit code should be 0 (success) or 2 (skipped if market is active)
    assert code in {0, 2}

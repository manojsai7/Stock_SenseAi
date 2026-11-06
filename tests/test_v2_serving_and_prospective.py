import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.research_v2_registry import ResearchV2Registry, sha256
from backend.services.challenger_prediction_service import ChallengerPredictionService
from backend.services.prediction_service import StaleMarketDataError
from backend.services.prospective_v2_service import ProspectiveV2Service
from ml.features.stationary import FEATURE_COLUMNS, reconstruct_price_from_return


@pytest.fixture
def client():
    return TestClient(app)


def test_v2_artifact_manifest_integrity():
    """Verify that models_storage_v2/RELIANCE_NS is complete and digests match."""
    root = Path("models_storage_v2/RELIANCE_NS")
    assert root.is_dir()
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for filename, info in manifest["files"].items():
        fpath = root / filename
        assert fpath.is_file(), f"Missing file: {filename}"
        assert sha256(fpath) == info["sha256"], f"Digest mismatch on {filename}"


def test_v2_registry_loads_real_model():
    """Verify that ResearchV2Registry successfully validates and loads the real V2 GRU model."""
    registry = ResearchV2Registry()
    manifest, model, feature_scaler, target_scaler = registry.load("RELIANCE.NS")
    assert manifest["architecture"] == "GRU"
    assert getattr(feature_scaler, "feature_columns", None) == list(FEATURE_COLUMNS)
    assert getattr(feature_scaler, "is_fitted", False) is True
    assert getattr(target_scaler, "is_fitted", False) is True


def test_v2_loader_rejects_unsupported_symbols():
    """Verify that ResearchV2Registry rejects any non-RELIANCE symbol."""
    registry = ResearchV2Registry()
    with pytest.raises(FileNotFoundError, match="V2 unavailable"):
        registry.validate_manifest("TCS.NS")
    with pytest.raises(FileNotFoundError, match="V2 unavailable"):
        registry.load("INFY.NS")


def test_v2_loader_rejects_tampered_bundle(tmp_path):
    """Verify that checksum mismatch causes bundle rejection."""
    dir_v2 = tmp_path / "RELIANCE_NS"
    dir_v2.mkdir()
    model_file = dir_v2 / "model.keras"
    model_file.write_bytes(b"dummy")
    fs_file = dir_v2 / "feature_scaler.pkl"
    fs_file.write_bytes(b"dummy")
    ts_file = dir_v2 / "target_scaler.pkl"
    ts_file.write_bytes(b"dummy")
    meta_file = dir_v2 / "metadata.json"
    meta_file.write_bytes(b"{}")

    manifest = {
        "model_id": "stocksense-v2-gru-challenger",
        "symbol": "RELIANCE.NS",
        "features": list(FEATURE_COLUMNS),
        "feature_version": "stationary-seven-v1",
        "input_shape": [1, 30, 7],
        "status": "experimental-shadow-candidate",
        "files": {
            "model.keras": {"path": "model.keras", "sha256": "bad_hash"},
            "feature_scaler.pkl": {"path": "feature_scaler.pkl", "sha256": sha256(fs_file)},
            "target_scaler.pkl": {"path": "target_scaler.pkl", "sha256": sha256(ts_file)},
            "metadata.json": {"path": "metadata.json", "sha256": sha256(meta_file)},
        }
    }
    (dir_v2 / "manifest.json").write_text(json.dumps(manifest))

    reg = ResearchV2Registry(tmp_path)
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        reg.validate_manifest("RELIANCE.NS")


def test_challenger_prediction_service_unsupported_symbol():
    """Verify that ChallengerPredictionService rejects non-RELIANCE symbols."""
    service = ChallengerPredictionService()
    with pytest.raises(ValueError, match="exclusively for RELIANCE.NS"):
        service.get_prediction("TCS.NS")


def test_challenger_stale_data_rejection():
    """Verify that ChallengerPredictionService rejects stale data with StaleMarketDataError."""
    service = ChallengerPredictionService()
    dates = pd.bdate_range("2026-01-01", periods=60)
    stale_df = pd.DataFrame({
        "Date": dates,
        "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000000
    })
    stale_df.attrs["data_source"] = "Synthetic Test"
    stale_df.attrs["source_status"] = "OFFLINE"

    expected_session = dates[-1].date() + timedelta(days=5) # Far in future -> stale
    with pytest.raises(StaleMarketDataError, match="Market history is stale"):
        service._market_data_provenance(stale_df, stale_df, expected_session)


def test_challenger_prediction_api_endpoint(client, monkeypatch):
    """Test GET /api/research/challenger-prediction/{symbol} returns expected schema."""
    dates = pd.bdate_range("2026-01-01", periods=60)
    mock_df = pd.DataFrame({
        "Date": dates,
        "Open": 1350.0 + np.sin(np.arange(60)),
        "High": 1360.0 + np.sin(np.arange(60)),
        "Low": 1340.0 + np.sin(np.arange(60)),
        "Close": 1355.0 + np.sin(np.arange(60)),
        "Volume": 5000000 + np.arange(60) * 1000,
    })
    mock_df.attrs["data_source"] = "Yahoo Finance (Mock)"
    mock_df.attrs["source_status"] = "COMPLETED"

    from backend.services.challenger_prediction_service import challenger_prediction_service
    from backend.services.market_status_service import market_status_service

    monkeypatch.setattr(challenger_prediction_service.provider, "fetch_historical_ohlcv", lambda *args, **kwargs: mock_df)
    monkeypatch.setattr(market_status_service, "get_latest_completed_trading_day", lambda: dates[-1].date())
    monkeypatch.setattr(market_status_service, "get_next_trading_day", lambda d: d + timedelta(days=1))

    res = client.get("/api/research/challenger-prediction/RELIANCE.NS")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert data["model_architecture"] == "GRU"
    assert data["research_status"] == "EXPERIMENTAL — SHADOW EVALUATION"
    assert data["predicted_next_session_close"] > 0
    assert np.isfinite(data["predicted_simple_return"])
    assert data["training_cutoff"] == "2025-12-11"
    assert data["artifact_sha256"] == "6f44d5782514af8223eb65564da7bef23ed85fbd049b79b2dc43ea1a568bbf45"


def test_challenger_prediction_api_rejects_unsupported(client):
    """Test GET /api/research/challenger-prediction/TCS.NS returns 400 with explanation."""
    res = client.get("/api/research/challenger-prediction/TCS.NS")
    assert res.status_code == 400
    assert "exclusively for RELIANCE.NS" in res.json()["detail"]


def test_prospective_v2_tracker_idempotency(tmp_path):
    """Test that prospective tracker enforces idempotency and creates no duplicate rows."""
    registry_file = tmp_path / "reg.json"
    reg_content = {
        "status": "LOCKED_PROSPECTIVE",
        "symbol": "RELIANCE.NS",
        "models": {
            "production_cnn_lstm": {"name": "Prod V1", "hash": "h1"},
            "gru_v2_challenger": {"name": "GRU V2", "hash": "h2"},
            "naive_persistence": {"name": "Naive", "hash": "h3"}
        }
    }
    registry_file.write_text(json.dumps(reg_content))
    log_file = tmp_path / "log.csv"

    tracker = ProspectiveV2Service(registry_path=registry_file, log_path=log_file)
    v1_mock = {"predicted_return": 0.001, "predicted_close": 1360.0}
    v2_mock = {"predicted_simple_return": -0.001, "predicted_next_session_close": 1357.0}

    # First log
    first_rows = tracker.record_daily_forecasts(
        symbol="RELIANCE.NS",
        target_date="2026-11-10",
        input_window_end="2026-11-09",
        previous_close=1358.0,
        v1_pred=v1_mock,
        v2_pred=v2_mock,
    )
    assert len(first_rows) == 3
    df1 = pd.read_csv(log_file)
    assert len(df1) == 3

    # Attempt duplicate log for identical target_date
    second_rows = tracker.record_daily_forecasts(
        symbol="RELIANCE.NS",
        target_date="2026-11-10",
        input_window_end="2026-11-09",
        previous_close=1358.0,
        v1_pred=v1_mock,
        v2_pred=v2_mock,
    )
    df2 = pd.read_csv(log_file)
    assert len(df2) == 3 # Still exactly 3 rows, no duplicate!


def test_prospective_v2_tracker_resolution_against_same_actual_close(tmp_path):
    """Test that resolution scores all 3 models against identical actual close."""
    registry_file = tmp_path / "reg.json"
    registry_file.write_text(json.dumps({
        "status": "LOCKED",
        "symbol": "RELIANCE.NS",
        "models": {
            "production_cnn_lstm": {"name": "Prod V1", "hash": "h1"},
            "gru_v2_challenger": {"name": "GRU V2", "hash": "h2"},
            "naive_persistence": {"name": "Naive", "hash": "h3"}
        }
    }))
    log_file = tmp_path / "log.csv"
    tracker = ProspectiveV2Service(registry_path=registry_file, log_path=log_file)

    tracker.record_daily_forecasts(
        symbol="RELIANCE.NS",
        target_date="2026-11-10",
        input_window_end="2026-11-09",
        previous_close=1350.0,
        v1_pred={"predicted_return": 0.005, "predicted_close": 1356.75},
        v2_pred={"predicted_simple_return": 0.002, "predicted_next_session_close": 1352.70},
    )

    # Market finalizes 2026-11-10 with close = 1355.0
    hist_df = pd.DataFrame([
        {"Date": pd.Timestamp("2026-11-09"), "Close": 1350.0},
        {"Date": pd.Timestamp("2026-11-10"), "Close": 1355.0},
    ])

    resolved = tracker.resolve_pending_forecasts("RELIANCE.NS", hist_df)
    assert resolved == 3

    df = pd.read_csv(log_file)
    assert (df["status"] == "RESOLVED").all()
    # All models must be evaluated against the exact same actual_close
    assert (df["actual_close"] == 1355.0).all()
    assert (df["actual_direction"] == "UP").all()


def test_exp05_registry_remains_untouched():
    """Verify that Sprint 4 changes did not alter EXP-05 locked registry."""
    exp05_path = Path("research/EXP-05/evaluation/locked_evaluation_registry.json")
    assert exp05_path.is_file()
    reg = json.loads(exp05_path.read_text(encoding="utf-8"))
    assert reg["status"] == "LOCKED"
    assert "colab_exp05_fold3" in reg["models"]
    assert "production_cnn_lstm" in reg["models"]


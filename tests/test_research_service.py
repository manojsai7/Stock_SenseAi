import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.research_service import ResearchService, research_service

client = TestClient(app)


def test_research_model_comparison_success_for_reliance():
    res = research_service.get_model_comparison("RELIANCE.NS")
    assert res["symbol"] == "RELIANCE.NS"
    assert res["supported"] is True
    assert res["status"] == "LOCKED"
    assert "scientific_disclaimer" in res
    assert "N=1" in res["scientific_disclaimer"]

    # Sample counts
    counts = res["sample_counts"]
    assert counts["total_predictions"] == 6
    assert counts["total_sessions"] == 2
    assert counts["resolved_sessions"] == 1
    assert counts["pending_sessions"] == 1
    assert counts["resolved_predictions"] == 3
    assert counts["pending_predictions"] == 3

    # Models
    models = res["models"]
    assert len(models) == 3
    model_keys = [m["key"] for m in models]
    assert "production_cnn_lstm" in model_keys
    assert "colab_exp05_fold3" in model_keys
    assert "naive_persistence" in model_keys

    # Check that summary metrics are computed for each model
    for m in models:
        metrics = m["summary_metrics"]
        assert metrics["resolved_count"] == 1
        assert metrics["price_mae"] is not None
        assert metrics["price_mae"] > 0
        assert metrics["price_rmse"] is not None

    # Check resolved session
    assert len(res["resolved_observations"]) == 1
    resolved_session = res["resolved_observations"][0]
    assert resolved_session["target_date"] == "2026-10-26"
    assert resolved_session["actual_close"] == 1201.8
    assert resolved_session["previous_close"] == 1190.0
    assert resolved_session["actual_direction"] == "UP"
    assert len(resolved_session["predictions"]) == 3

    # Check pending session
    assert len(res["pending_forecasts"]) == 1
    pending_session = res["pending_forecasts"][0]
    assert pending_session["target_date"] == "2026-10-27"
    assert pending_session["previous_close"] == 1201.8
    assert pending_session["status"] == "PENDING_OUTCOME"
    assert len(pending_session["predictions"]) == 3


def test_pending_vs_resolved_strict_isolation():
    res = research_service.get_model_comparison("RELIANCE.NS")
    pending = res["pending_forecasts"][0]

    # Verify pending predictions do not contain resolved fields
    for m_name, pred_data in pending["predictions"].items():
        assert "actual_close" not in pred_data
        assert "price_error_abs" not in pred_data
        assert "direction_correct" not in pred_data
        assert pred_data["predicted_close"] is not None

    # Verify resolved predictions contain error calculations
    resolved = res["resolved_observations"][0]
    for m_name, pred_data in resolved["predictions"].items():
        assert "price_error_abs" in pred_data
        assert pred_data["price_error_abs"] is not None
        assert "direction_correct" in pred_data


def test_sample_size_calculations_exactness():
    res = research_service.get_model_comparison("RELIANCE.NS")
    models_by_key = {m["key"]: m for m in res["models"]}

    # Production CNN-LSTM (Oct 26: Pred 1190.23, Actual 1201.80 -> Error 11.57, Dir UP vs UP -> True)
    prod = models_by_key["production_cnn_lstm"]
    assert prod["summary_metrics"]["price_mae"] == 11.57
    assert prod["summary_metrics"]["directional_accuracy_pct"] == 100.0
    assert prod["summary_metrics"]["correct_directions"] == 1
    assert prod["summary_metrics"]["resolved_count"] == 1

    # Naive Persistence (Oct 26: Pred 1190.00, Actual 1201.80 -> Error 11.80, Dir FLAT vs UP -> False)
    naive = models_by_key["naive_persistence"]
    assert naive["summary_metrics"]["price_mae"] == 11.80
    assert naive["summary_metrics"]["directional_accuracy_pct"] == 0.0
    assert naive["summary_metrics"]["resolved_count"] == 1

    # Colab Fold 3 (Oct 26: Pred 1189.31, Actual 1201.80 -> Error 12.49, Dir DOWN vs UP -> False)
    colab = models_by_key["colab_exp05_fold3"]
    assert colab["summary_metrics"]["price_mae"] == 12.49
    assert colab["summary_metrics"]["directional_accuracy_pct"] == 0.0
    assert colab["summary_metrics"]["resolved_count"] == 1


def test_unsupported_symbol_safe_response():
    # Unsupported symbols must not fail or crash, but return supported=False
    for sym in ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL"]:
        res = research_service.get_model_comparison(sym)
        assert res["supported"] is False
        assert res["symbol"] == sym.strip().upper()
        assert "scoped exclusively to RELIANCE.NS" in res["message"]
        assert res["sample_counts"]["total_predictions"] == 0
        assert res["models"] == []


def test_no_historical_prospective_mixing():
    res = research_service.get_model_comparison("RELIANCE.NS")
    # Verify that historical EXP-05 metrics (such as ₹14.09 MAE from 184 test sessions) are NOT present in prospective summary
    for m in res["models"]:
        assert m["summary_metrics"]["resolved_count"] == 1
        assert m["summary_metrics"]["price_mae"] != 14.09
        assert m["summary_metrics"]["price_mae"] != 14.16


def test_missing_registry_or_log_raises():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_reg = Path(tmp_dir) / "non_existent_registry.json"
        fake_log = Path(tmp_dir) / "non_existent_log.csv"

        # Missing registry
        srv_no_reg = ResearchService(registry_path=fake_reg, log_path=fake_log)
        with pytest.raises(FileNotFoundError, match="Locked evaluation registry not found"):
            srv_no_reg.get_model_comparison("RELIANCE.NS")

        # Create dummy registry
        fake_reg.write_text('{"symbol": "RELIANCE.NS", "status": "LOCKED"}', encoding="utf-8")
        srv_no_log = ResearchService(registry_path=fake_reg, log_path=fake_log)
        with pytest.raises(FileNotFoundError, match="Prospective predictions log not found"):
            srv_no_log.get_model_comparison("RELIANCE.NS")


def test_empty_log_handling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        reg_file = Path(tmp_dir) / "registry.json"
        reg_file.write_text('{"symbol": "RELIANCE.NS", "status": "LOCKED"}', encoding="utf-8")

        log_file = Path(tmp_dir) / "empty_log.csv"
        log_file.write_text("forecast_id,forecast_timestamp,symbol,input_window_end,target_date,model_name,model_hash,training_cutoff,previous_close,predicted_return,predicted_close,predicted_direction,actual_close,actual_return,actual_direction,direction_correct,price_error_abs,status,resolution_timestamp\n", encoding="utf-8")

        srv = ResearchService(registry_path=reg_file, log_path=log_file)
        res = srv.get_model_comparison("RELIANCE.NS")
        assert res["status"] == "EMPTY_LOG"
        assert res["sample_counts"]["total_predictions"] == 0


def test_api_endpoints_research_comparison():
    # 1. Query parameter
    r_query = client.get("/api/research/model-comparison?symbol=RELIANCE.NS")
    assert r_query.status_code == 200
    data = r_query.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert data["supported"] is True
    assert len(data["models"]) == 3

    # 2. Path parameter
    r_path = client.get("/api/research/model-comparison/RELIANCE.NS")
    assert r_path.status_code == 200
    assert r_path.json()["symbol"] == "RELIANCE.NS"

    # 3. Unsupported symbol via endpoint
    r_tcs = client.get("/api/research/model-comparison?symbol=TCS.NS")
    assert r_tcs.status_code == 200
    assert r_tcs.json()["supported"] is False

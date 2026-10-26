"""
Unit tests for the Prospective Forward Evaluation Logger (Phase 3).
Verifies registry schema, completed-candle filtering, and strict freshness guards.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
import pytest
import pandas as pd

from scripts.forward_evaluation_logger import (
    lock_evaluation,
    fetch_and_validate_market_data,
    REGISTRY_FILE,
    LOG_CSV,
)
from backend.services.market_status_service import market_status_service


def test_registry_locking(tmp_path, monkeypatch):
    test_eval_dir = tmp_path / "evaluation"
    test_reg_file = test_eval_dir / "locked_evaluation_registry.json"
    test_log_csv = test_eval_dir / "prospective_predictions_log.csv"

    import scripts.forward_evaluation_logger as fel
    monkeypatch.setattr(fel, "EVAL_DIR", test_eval_dir)
    monkeypatch.setattr(fel, "REGISTRY_FILE", test_reg_file)
    monkeypatch.setattr(fel, "LOG_CSV", test_log_csv)

    registry = fel.lock_evaluation("RELIANCE.NS")
    assert registry["status"] == "LOCKED"
    assert registry["symbol"] == "RELIANCE.NS"
    assert "production_cnn_lstm" in registry["models"]
    assert "colab_exp05_fold3" in registry["models"]
    assert "naive_persistence" in registry["models"]
    assert len(registry["models"]["production_cnn_lstm"]["hash"]) == 64
    assert len(registry["models"]["colab_exp05_fold3"]["hash"]) == 64
    assert test_reg_file.exists()
    assert test_log_csv.exists()


def test_freshness_guard_rejects_stale_data(monkeypatch):
    """
    Ensures that if the data provider returns data older than the exchange calendar's
    latest completed trading session, a RuntimeError is raised.
    """
    # Create synthetic stale data ending on 2026-10-01
    dates = pd.date_range("2026-08-01", "2026-10-01", freq="B")
    df_stale = pd.DataFrame({
        "Date": dates,
        "Open": 1000.0,
        "High": 1010.0,
        "Low": 990.0,
        "Close": 1005.0,
        "Volume": 1000000
    })

    # Mock fetch_raw_ohlcv to return stale data
    class MockAcquisition:
        def fetch_raw_ohlcv(self, symbol, period="5y"):
            return df_stale.copy()

    import scripts.forward_evaluation_logger as fel
    monkeypatch.setattr(fel, "ResearchDataAcquisition", MockAcquisition)

    # Latest completed trading day is guaranteed to be after 2026-10-01
    with pytest.raises(RuntimeError, match="Data Freshness Error: Stale market data"):
        fel.fetch_and_validate_market_data("RELIANCE.NS", save_cache=False)


def test_completed_candle_rule_filters_future_intraday_candles(monkeypatch):
    """
    Ensures that any unfinalized candle timestamped after latest_completed_day is stripped.
    """
    latest_completed = market_status_service.get_latest_completed_trading_day()
    next_day = market_status_service.get_next_trading_day(latest_completed)

    # Create dates up to next_day (simulating an open intraday bar)
    dates = [
        pd.Timestamp(latest_completed) - pd.Timedelta(days=2),
        pd.Timestamp(latest_completed) - pd.Timedelta(days=1),
        pd.Timestamp(latest_completed),
        pd.Timestamp(next_day)  # Incomplete future candle
    ]
    df_with_intraday = pd.DataFrame({
        "Date": dates,
        "Open": [1000.0, 1005.0, 1010.0, 1015.0],
        "High": [1010.0, 1015.0, 1020.0, 1025.0],
        "Low": [995.0, 1000.0, 1005.0, 1010.0],
        "Close": [1005.0, 1010.0, 1015.0, 1020.0],
        "Volume": [1000000, 1000000, 1000000, 500000]
    })

    class MockAcquisition:
        def fetch_raw_ohlcv(self, symbol, period="5y"):
            return df_with_intraday.copy()

    import scripts.forward_evaluation_logger as fel
    monkeypatch.setattr(fel, "ResearchDataAcquisition", MockAcquisition)

    df_clean, comp_day, next_tgt = fel.fetch_and_validate_market_data("RELIANCE.NS", save_cache=False)

    # Incomplete candle dated next_day MUST be filtered out
    assert df_clean["Date"].dt.date.max() == latest_completed
    assert comp_day == latest_completed
    assert next_tgt == next_day

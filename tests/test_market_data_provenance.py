from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app
from backend.services.prediction_service import cache
from backend.services.prediction_service import (
    PredictionService,
    StaleMarketDataError,
)


def _daily_frame(last_day: str, source_status: str = "ONLINE") -> pd.DataFrame:
    dates = pd.bdate_range(end=last_day, periods=40)
    frame = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 40,
        "High": [105.0] * 40,
        "Low": [95.0] * 40,
        "Close": [100.0] * 40,
        "Volume": [1000000] * 40
    })
    frame.attrs["source_status"] = source_status
    frame.attrs["data_source"] = "test market feed"
    return frame


def test_market_data_provenance_reports_actual_online_source():
    frame = _daily_frame("2026-10-22")

    result = PredictionService._market_data_provenance(
        frame, frame, date(2026, 10, 22)
    )

    assert result == {
        "data_status": "ONLINE",
        "data_provider_source": "test market feed",
        "data_as_of_date": "2026-10-22",
    }


def test_market_data_provenance_keeps_fresh_offline_data_labelled_offline():
    frame = _daily_frame("2026-10-22", source_status="OFFLINE")

    result = PredictionService._market_data_provenance(
        frame, frame, date(2026, 10, 22)
    )

    assert result["data_status"] == "OFFLINE"
    assert result["data_provider_source"] == "test market feed"


def test_market_data_provenance_rejects_old_history():
    frame = _daily_frame("2026-10-07", source_status="OFFLINE")

    with pytest.raises(StaleMarketDataError, match="expected at least 2026-10-22"):
        PredictionService._market_data_provenance(
            frame, frame, date(2026, 10, 22)
        )


def test_prediction_route_returns_503_for_stale_fallback_history():
    from backend.api.prediction_routes import pred_service

    stale = _daily_frame("2026-10-07", source_status="OFFLINE")
    client = TestClient(app)
    cache.clear()

    with patch.object(pred_service.provider, "fetch_historical_ohlcv", return_value=stale), \
         patch("backend.services.prediction_service.market_status_service.get_latest_completed_trading_day", return_value=date(2026, 10, 22)):
        response = client.get("/api/predictions/TCS.NS/latest?pipeline_version=legacy")

    assert response.status_code == 503
    assert "Market history is stale" in response.json()["detail"]

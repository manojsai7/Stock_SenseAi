from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    symbol: str
    last_observed_date: str
    latest_completed_session: str
    target_date: str
    last_close: float
    predicted_close: float
    baseline_close: float
    expected_change: float
    expected_change_pct: float
    direction: str  # "BULLISH", "BEARISH", or "FLAT"
    directional_signal: str  # "BULLISH", "BEARISH", or "FLAT"
    prediction_horizon: str = "Next Trading Day"
    model_version: str
    architecture: str
    input_window_size: int = 30
    features_used: List[str]
    historical_directional_accuracy: Optional[float] = None
    directional_accuracy_benchmark: Optional[float] = None
    mae_benchmark: Optional[float] = None
    rmse_benchmark: Optional[float] = None
    mape_benchmark: Optional[float] = None
    data_status: str = "LIVE"
    data_source: str = "Yahoo Finance"
    market_status: str = "CLOSED"
    recent_input_candles: Optional[List[Dict[str, Any]]] = None


class ModelMetricsResponse(BaseModel):
    symbol: str
    trained_at: str
    model_version: str
    window_size: int
    prediction_horizon: str
    features: List[str]
    architecture: Dict[str, Any]
    metrics_comparison: Dict[str, Dict[str, float]]
    test_trajectory: List[Dict[str, Any]]

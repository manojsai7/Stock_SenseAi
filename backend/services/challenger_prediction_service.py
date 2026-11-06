import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from backend.core.cache import cache
from backend.services.market_status_service import market_status_service
from backend.services.prediction_service import StaleMarketDataError
from backend.services.research_v2_registry import ResearchV2Registry
from ml.data.providers.yahoo_provider import YahooFinanceProvider
from ml.features.stationary import (
    prepare_inference_features,
    reconstruct_price_from_return,
)

logger = logging.getLogger(__name__)


class ChallengerPredictionService:
    """
    Independent V2 GRU Challenger Prediction Service.
    - Operates in shadow mode exclusively for RELIANCE.NS.
    - Completely isolated from V1 production registry.
    - Enforces completed-candle rule and rejects stale data with StaleMarketDataError.
    - Uses exact matching V2 scalers and 30x7 stationary features.
    """

    def __init__(self, registry: Optional[ResearchV2Registry] = None):
        self.registry = registry or ResearchV2Registry()
        self.provider = YahooFinanceProvider()

    def _market_data_provenance(self, raw_df: pd.DataFrame, completed_df: pd.DataFrame, expected_session) -> Dict[str, Any]:
        if completed_df.empty:
            raise StaleMarketDataError("No completed daily bars are available for the expected session.")

        latest_bar = completed_df["Date"].max().date()
        if latest_bar < expected_session:
            source = raw_df.attrs.get("data_source", "Yahoo Finance")
            raise StaleMarketDataError(
                f"Market history is stale: latest available completed bar is {latest_bar}; "
                f"expected at least {expected_session}. Source: {source}."
            )

        source_status = raw_df.attrs.get("source_status")
        if source_status not in {"ONLINE", "OFFLINE"}:
            legacy_status = raw_df.attrs.get("data_status")
            source_status = "OFFLINE" if legacy_status in {"FALLBACK", "OFFLINE"} else (
                "ONLINE" if legacy_status == "LIVE" else "COMPLETED"
            )

        return {
            "data_status": source_status,
            "data_provider_source": raw_df.attrs.get("data_source", "Yahoo Finance (Daily OHLCV)"),
            "data_as_of_date": latest_bar.strftime("%Y-%m-%d"),
        }

    def get_prediction(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()
        if clean_symbol != "RELIANCE.NS":
            raise ValueError(
                f"V2 GRU Challenger model is trained and authorized exclusively for RELIANCE.NS "
                f"under protocol EXP-06. Symbol '{clean_symbol}' is not supported in this research phase."
            )

        # 1. Check registry availability
        status_info = self.registry.status(clean_symbol)
        if not status_info.get("available", False):
            raise RuntimeError(
                f"V2 Challenger model artifacts unavailable for {clean_symbol}: "
                f"{status_info.get('reason', 'Artifact verification failed')}"
            )

        # 2. Determine latest completed trading day
        latest_completed_date = market_status_service.get_latest_completed_trading_day()
        completed_session_str = latest_completed_date.strftime("%Y-%m-%d")

        # 3. Check session-keyed cache
        cache_key = f"challenger-pred:v2-gru:{clean_symbol}:{completed_session_str}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"[CHALLENGER] Cache HIT for {clean_symbol} on session {completed_session_str}")
            return cached

        logger.info(f"[CHALLENGER] Computing V2 GRU prediction for {clean_symbol} on session <= {completed_session_str}")

        # 4. Fetch 1y OHLCV
        df = self.provider.fetch_historical_ohlcv(clean_symbol, period="1y", interval="1d")
        if df.empty or len(df) < 55:
            raise ValueError(
                f"Insufficient historical data points returned for {clean_symbol} ({len(df)} rows). "
                f"At least 55 trading sessions required for stationary warm-up."
            )

        # 5. Filter completed candles only
        df["SessionDate"] = df["Date"].dt.date
        completed_df = df[df["SessionDate"] <= latest_completed_date].copy()
        completed_df = completed_df.sort_values("Date").reset_index(drop=True)
        provenance = self._market_data_provenance(df, completed_df, latest_completed_date)

        if len(completed_df) < 50:
            raise ValueError(
                f"Only {len(completed_df)} completed daily candles available for {clean_symbol}. "
                f"Minimum 50 required for 20-day stationary lookback + 30-session window."
            )

        # 6. Compute stationary features
        clean_features_df, _ = prepare_inference_features(completed_df)
        if len(clean_features_df) < 30:
            raise ValueError(
                f"Insufficient valid feature rows ({len(clean_features_df)}) after stationary calculation. "
                f"Minimum required is 30."
            )

        window_df = clean_features_df.tail(30).copy()
        input_window_start = window_df["Date"].iloc[0].strftime("%Y-%m-%d")
        input_window_end = window_df["Date"].iloc[-1].strftime("%Y-%m-%d")
        last_close = round(float(window_df["Close"].iloc[-1]), 2)
        target_trading_date = market_status_service.get_next_trading_day(window_df["Date"].iloc[-1].date())
        target_date_str = target_trading_date.strftime("%Y-%m-%d")

        # 7. Load V2 model and scalers
        manifest, model, feature_scaler, target_scaler = self.registry.load(clean_symbol)

        # 8. Leakage-safe feature scaling & inference
        scaled_features = feature_scaler.transform(window_df)
        input_tensor = np.expand_dims(scaled_features, axis=0) # (1, 30, 7)

        raw_pred = model(input_tensor, training=False)
        pred_return_raw = float(target_scaler.inverse_transform(np.asarray(raw_pred).reshape(-1))[0])

        if not np.isfinite(pred_return_raw):
            raise RuntimeError(f"Non-finite return prediction ({pred_return_raw}) generated by V2 GRU for {clean_symbol}.")

        pred_close_raw = reconstruct_price_from_return(last_close, pred_return_raw)
        pred_close = round(pred_close_raw, 2)

        if not np.isfinite(pred_close) or pred_close <= 0:
            raise RuntimeError(f"Invalid reconstructed closing price ({pred_close}) generated by V2 GRU for {clean_symbol}.")

        pred_return = round(pred_return_raw, 6)
        pred_return_pct = round(pred_return_raw * 100, 4)

        if pred_return > 0.0005:
            direction = "BULLISH"
        elif pred_return < -0.0005:
            direction = "BEARISH"
        else:
            direction = "FLAT"

        model_hash = manifest["files"].get("model.keras", manifest["files"].get("model", {})).get("sha256", "")

        response = {
            "symbol": clean_symbol,
            "model_architecture": manifest.get("architecture", "GRU"),
            "model_version": manifest.get("model_id", "stocksense-v2-gru-challenger"),
            "research_status": "EXPERIMENTAL — SHADOW EVALUATION",
            "latest_completed_close": last_close,
            "latest_completed_input_date": input_window_end,
            "input_window_start": input_window_start,
            "input_window_end": input_window_end,
            "predicted_simple_return": pred_return,
            "predicted_return_pct": pred_return_pct,
            "predicted_next_session_close": pred_close,
            "target_trading_date": target_date_str,
            "direction": direction,
            "data_provider": provenance.get("data_provider_source", "Yahoo Finance"),
            "data_freshness": provenance.get("data_status", "COMPLETED"),
            "data_as_of_date": provenance.get("data_as_of_date", input_window_end),
            "artifact_sha256": model_hash,
            "training_cutoff": manifest.get("training_cutoff", "2025-12-11"),
            "feature_version": manifest.get("feature_version", "stationary-seven-v1"),
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Session-aware cache
        cache.set(cache_key, response, ttl=300)
        return response


challenger_prediction_service = ChallengerPredictionService()

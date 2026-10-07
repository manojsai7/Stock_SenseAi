import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from ml.data.providers.yahoo_provider import YahooFinanceProvider
from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS
from ml.preprocessing.sequences import SequenceGenerator
from backend.core.cache import cache
from backend.core.config import settings
from backend.services.market_status_service import market_status_service
from backend.services.model_registry import model_registry, MODELS_DIR

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Production-grade CNN-LSTM Prediction Service:
    - Preloads and reuses in-memory models/scalers via ModelRegistry (no retraining, no disk reload per request).
    - Enforces completed daily candle rule: incomplete intraday candles are strictly excluded from inference.
    - Accurately computes next trading day target date via MarketStatusService.
    - Caches predictions by symbol and completed session date (invalidates automatically when session rolls over).
    """

    def __init__(self):
        self.provider = YahooFinanceProvider()
        self.feature_extractor = TechnicalFeatureExtractor(ma_window=20)
        self.seq_generator = SequenceGenerator(window_size=settings.MODEL_WINDOW)

    def get_prediction(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()

        # Validate symbol is supported
        if not model_registry.is_symbol_supported(clean_symbol):
            raise ValueError(
                f"Symbol '{clean_symbol}' is not supported for deep learning inference. "
                f"Supported stocks: {', '.join(model_registry.supported_symbols)}"
            )

        if not model_registry.has_artifacts(clean_symbol):
            raise FileNotFoundError(
                f"Model weights or scaler not found for '{clean_symbol}'. Offline training must be performed first."
            )

        # 1. Determine current market status and latest completed session date
        market_status = market_status_service.get_market_status()
        latest_completed_date = market_status_service.get_latest_completed_trading_day()
        completed_session_str = latest_completed_date.strftime("%Y-%m-%d")

        # 2. Check cache keyed by symbol + completed session date
        cache_key = f"pred:{clean_symbol}:{completed_session_str}"
        cached_pred = cache.get(cache_key)
        if cached_pred:
            logger.info(f"[PREDICTION] Cache HIT for {clean_symbol} on session {completed_session_str}")
            # Refresh live market status in response
            cached_pred["market_status"] = market_status
            return cached_pred

        logger.info(f"[PREDICTION] Cache MISS for {clean_symbol}. Computing inference on session <= {completed_session_str}...")

        # 3. Retrieve daily historical OHLCV data (fetch 6 months for ample MA20 and 30-day window)
        df = self.provider.fetch_historical_ohlcv(clean_symbol, period="6mo", interval="1d")
        if df.empty or len(df) < 35:
            raise ValueError(f"Insufficient historical data points returned for {clean_symbol} ({len(df)} rows)")

        # 4. Strict Completed Candle Rule:
        # Exclude any row with Date > latest_completed_date (e.g. today's live/incomplete session)
        df["SessionDate"] = df["Date"].dt.date
        completed_df = df[df["SessionDate"] <= latest_completed_date].copy()

        if len(completed_df) < 30:
            raise ValueError(
                f"Only {len(completed_df)} completed daily candles available for {clean_symbol}. "
                f"Minimum required is {settings.MODEL_WINDOW}."
            )

        # 5. Compute technical features (MA20) identically to offline training
        featured_df = self.feature_extractor.prepare_inference_features(completed_df)
        if len(featured_df) < settings.MODEL_WINDOW:
            raise ValueError(f"Insufficient feature rows ({len(featured_df)}) after computing MA20.")

        # 6. Extract the latest 30 COMPLETED trading days
        window_df = featured_df.tail(settings.MODEL_WINDOW).copy()
        last_observed_date = window_df["Date"].iloc[-1].strftime("%Y-%m-%d")
        last_close = round(float(window_df["Close"].iloc[-1]), 2)

        # 7. Compute next trading day target date
        target_trading_date = market_status_service.get_next_trading_day(window_df["Date"].iloc[-1].date())
        target_date_str = target_trading_date.strftime("%Y-%m-%d")

        # 8. Load model, fitted scaler, and metadata from in-memory ModelRegistry
        model = model_registry.get_model(clean_symbol)
        scaler = model_registry.get_scaler(clean_symbol)
        metadata = model_registry.get_metadata(clean_symbol)

        # 9. Leakage-safe feature scaling using training scaler
        scaled_features = scaler.transform_features_only(window_df)
        input_tensor = self.seq_generator.create_inference_sequence(scaled_features)

        # 10. Execute CNN-LSTM neural network inference
        raw_pred = model.predict(input_tensor, verbose=0)
        predicted_close_raw = float(scaler.inverse_transform_target(raw_pred)[0])
        predicted_close = round(predicted_close_raw, 2)

        # 11. Delta, percentage change, and directional signal
        expected_change = round(predicted_close - last_close, 2)
        expected_change_pct = round(((predicted_close - last_close) / last_close) * 100, 2)

        if expected_change > 0.05:
            direction = "BULLISH"
        elif expected_change < -0.05:
            direction = "BEARISH"
        else:
            direction = "FLAT"

        # 12. Retrieve academic test metrics from metadata
        metrics = metadata.get("metrics_comparison", {}).get("cnn_lstm", {})
        dir_acc = metrics.get("directional_accuracy", None)
        mae = metrics.get("mae", None)
        rmse = metrics.get("rmse", None)
        mape = metrics.get("mape", None)

        # Recent 30 input candles for audit / UI inspection
        recent_candles = []
        for _, row in window_df.iterrows():
            recent_candles.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "ma20": round(float(row["MA20"]), 2),
            })

        response = {
            "symbol": clean_symbol,
            "last_observed_date": last_observed_date,
            "latest_completed_session": completed_session_str,
            "target_date": target_date_str,
            "last_close": last_close,
            "predicted_close": predicted_close,
            "baseline_close": last_close,
            "expected_change": expected_change,
            "expected_change_pct": expected_change_pct,
            "direction": direction,
            "directional_signal": direction,
            "prediction_horizon": "Next Trading Day",
            "model_version": metadata.get("model_version", "v1.0 Defensible"),
            "architecture": metadata.get("architecture", {}).get("name", "CNN-LSTM Hybrid (TensorFlow/Keras)"),
            "input_window_size": settings.MODEL_WINDOW,
            "features_used": FEATURE_COLUMNS,
            "historical_directional_accuracy": dir_acc,
            "directional_accuracy_benchmark": dir_acc,
            "mae_benchmark": mae,
            "rmse_benchmark": rmse,
            "mape_benchmark": mape,
            "data_status": "LIVE",
            "data_source": settings.PROVIDER_NAME,
            "market_status": market_status,
            "recent_input_candles": recent_candles,
        }

        # Cache result using session-aware TTL
        cache.set(cache_key, response, ttl=settings.PREDICTION_CACHE_TTL)
        return response

    def get_model_metrics(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()
        if not model_registry.is_symbol_supported(clean_symbol):
            # Fallback for display
            clean_symbol = "TCS.NS"
        return model_registry.get_metadata(clean_symbol)

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for sym in model_registry.supported_symbols:
            if model_registry.has_artifacts(sym):
                try:
                    meta = model_registry.get_metadata(sym)
                    models.append({
                        "symbol": sym,
                        "version": meta.get("model_version", "v1.0"),
                        "architecture": meta.get("architecture", {}).get("name", "CNN-LSTM Hybrid"),
                        "trained_at": meta.get("trained_at"),
                        "metrics": meta.get("metrics_comparison", {})
                    })
                except Exception as e:
                    logger.error(f"Error reading metadata for {sym}: {e}")
        return models


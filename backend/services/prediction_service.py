import logging
from typing import Dict, Any, List, Optional
import numpy as np
from ml.data.providers.yahoo_provider import YahooFinanceProvider
from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS as LEGACY_FEATURE_COLUMNS
from ml.features.stationary import (
    prepare_inference_features,
    FEATURE_COLUMNS as STATIONARY_FEATURE_COLUMNS,
    validate_sequence_tensor,
    reconstruct_price_from_return,
)
from ml.preprocessing.sequences import SequenceGenerator
from backend.core.cache import cache
from backend.core.config import settings
from backend.services.market_status_service import market_status_service
from backend.services.model_registry import model_registry

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Production-grade CNN-LSTM Prediction Service:
    - Supports two distinct pipelines:
        * 'legacy': 6-feature price-level target CNN-LSTM.
        * 'stationary-return-v1': 7-feature stationary return-target CNN-LSTM.
    - Preloads and reuses in-memory models/scalers via ModelRegistry (no retraining, no disk reload per request).
    - Enforces completed daily candle rule: incomplete intraday candles are strictly excluded from inference.
    - Accurately computes next trading day target date via MarketStatusService.
    - Caches predictions by pipeline version, symbol, and completed session date.
    """

    def __init__(self):
        self.provider = YahooFinanceProvider()
        self.feature_extractor = TechnicalFeatureExtractor(ma_window=20)
        self.seq_generator = SequenceGenerator(window_size=settings.MODEL_WINDOW)

    def get_prediction(self, symbol: str, pipeline_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Dispatches inference to the appropriate pipeline based on pipeline_version
        or settings.PREDICTION_PIPELINE_VERSION.
        """
        clean_symbol = symbol.strip().upper()
        version = (pipeline_version or settings.PREDICTION_PIPELINE_VERSION).strip().lower()

        if version not in settings.VALID_PIPELINE_VERSIONS:
            raise ValueError(
                f"Invalid prediction pipeline version '{version}'. "
                f"Supported versions: {settings.VALID_PIPELINE_VERSIONS}"
            )

        if version == settings.PIPELINE_STATIONARY_V1:
            return self._get_stationary_prediction(clean_symbol)
        else:
            return self._get_legacy_prediction(clean_symbol)

    def _get_legacy_prediction(self, clean_symbol: str) -> Dict[str, Any]:
        """
        Legacy 6-feature price-level CNN-LSTM inference.
        """
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

        # 2. Check cache keyed by version + symbol + completed session date
        cache_key = f"pred:legacy:{clean_symbol}:{completed_session_str}"
        cached_pred = cache.get(cache_key)
        if cached_pred:
            logger.info(f"[PREDICTION] Cache HIT for {clean_symbol} (legacy) on session {completed_session_str}")
            cached_pred["market_status"] = market_status
            return cached_pred

        logger.info(f"[PREDICTION] Cache MISS for {clean_symbol} (legacy). Computing inference on session <= {completed_session_str}...")

        # 3. Retrieve daily historical OHLCV data (fetch 6 months for ample MA20 and 30-day window)
        df = self.provider.fetch_historical_ohlcv(clean_symbol, period="6mo", interval="1d")
        if df.empty or len(df) < 35:
            raise ValueError(f"Insufficient historical data points returned for {clean_symbol} ({len(df)} rows)")

        # 4. Strict Completed Candle Rule:
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
            "pipeline_version": "legacy",
            "predicted_return": round((predicted_close - last_close) / last_close, 6),
            "input_window_size": settings.MODEL_WINDOW,
            "features_used": LEGACY_FEATURE_COLUMNS,
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

    def _get_stationary_prediction(self, clean_symbol: str) -> Dict[str, Any]:
        """
        Stationary-return CNN-LSTM inference pipeline (stationary-return-v1).
        Uses 7 causal relative features, standardizes features and targets with fitted scalers,
        predicts standardized next-day return, inverse-transforms to simple return, and reconstructs
        the forecast price from the reference completed session close.
        """
        # 1. Validate that the requested ticker has stationary artifacts
        if not model_registry.is_stationary_supported(clean_symbol) or not model_registry.has_stationary_artifacts(clean_symbol):
            if clean_symbol == "ATHERENERG.NS":
                raise FileNotFoundError(
                    f"Stationary inference pipeline 'stationary-return-v1' is not available for '{clean_symbol}'. "
                    f"Ather Energy has insufficient historical trading sessions for stationary return modeling."
                )
            raise FileNotFoundError(
                f"Stationary model artifacts not found for '{clean_symbol}' under pipeline 'stationary-return-v1'."
            )

        # 2. Determine latest completed trading session using existing market-status logic
        market_status = market_status_service.get_market_status()
        latest_completed_date = market_status_service.get_latest_completed_trading_day()
        completed_session_str = latest_completed_date.strftime("%Y-%m-%d")

        # Check cache keyed by version + symbol + completed session date
        cache_key = f"pred:stationary-return-v1:{clean_symbol}:{completed_session_str}"
        cached_pred = cache.get(cache_key)
        if cached_pred:
            logger.info(f"[PREDICTION] Cache HIT for {clean_symbol} (stationary-return-v1) on session {completed_session_str}")
            cached_pred["market_status"] = market_status
            return cached_pred

        logger.info(f"[PREDICTION] Cache MISS for {clean_symbol} (stationary-return-v1). Computing inference on session <= {completed_session_str}...")

        # 3. Fetch sufficient daily OHLCV history (1 year fetch for rolling warm-up and 30-session window)
        df = self.provider.fetch_historical_ohlcv(clean_symbol, period="1y", interval="1d")
        if df.empty or len(df) < 55:
            raise ValueError(
                f"Insufficient historical data points returned for {clean_symbol} ({len(df)} rows). "
                f"At least 55 trading sessions required for stationary warm-up."
            )

        # 4. Normalize dates, sort chronologically, exclude incomplete sessions
        df["SessionDate"] = df["Date"].dt.date
        completed_df = df[df["SessionDate"] <= latest_completed_date].copy()
        completed_df = completed_df.sort_values("Date").reset_index(drop=True)

        if len(completed_df) < 50:
            raise ValueError(
                f"Only {len(completed_df)} completed daily candles available for {clean_symbol}. "
                f"Minimum 50 required for 20-day stationary lookback + 30-session window."
            )

        # 5. Call canonical prepare_inference_features implementation (validates all 7 ordered features)
        clean_features_df, feature_cols = prepare_inference_features(completed_df)

        # 6. Extract the latest 30 valid feature rows
        if len(clean_features_df) < 30:
            raise ValueError(
                f"Insufficient valid feature rows ({len(clean_features_df)}) after stationary calculation. "
                f"Minimum required is 30."
            )
        window_df = clean_features_df.tail(30).copy()

        # 7. Extract reference dates and reference close
        last_observed_date = window_df["Date"].iloc[-1].strftime("%Y-%m-%d")
        last_close = round(float(window_df["Close"].iloc[-1]), 2)
        target_trading_date = market_status_service.get_next_trading_day(window_df["Date"].iloc[-1].date())
        target_date_str = target_trading_date.strftime("%Y-%m-%d")

        # 8. Load stationary model, scalers, and metadata from in-memory ModelRegistry
        model = model_registry.get_stationary_model(clean_symbol)
        feature_scaler = model_registry.get_stationary_feature_scaler(clean_symbol)
        target_scaler = model_registry.get_stationary_target_scaler(clean_symbol)
        metadata = model_registry.get_stationary_metadata(clean_symbol)

        # 9. Apply persisted StationaryFeatureScaler without refitting
        scaled_features = feature_scaler.transform(window_df)

        # 10. Construct input tensor of shape (1, 30, 7)
        input_tensor = np.expand_dims(scaled_features, axis=0)
        validate_sequence_tensor(input_tensor, expected_window=30)

        # 11. Predict one standardized return with saved CNN-LSTM
        raw_pred = model.predict(input_tensor, verbose=0)

        # 12. Inverse-transform output using persisted ReturnTargetScaler
        pred_return = float(target_scaler.inverse_transform(raw_pred)[0])

        # 13. Validate that predicted return is finite
        if not np.isfinite(pred_return):
            raise RuntimeError(
                f"Non-finite return prediction ({pred_return}) generated by stationary CNN-LSTM for {clean_symbol}."
            )

        # 14. Reconstruct predicted closing price: predicted_close = last_close * (1 + predicted_return)
        predicted_close_unrounded = reconstruct_price_from_return(last_close, pred_return)
        predicted_close = round(predicted_close_unrounded, 2)

        # 15. Validate that final predicted price is positive and finite
        if not np.isfinite(predicted_close) or predicted_close <= 0:
            raise RuntimeError(
                f"Invalid reconstructed closing price ({predicted_close}) for {clean_symbol}."
            )

        # 16. Calculate absolute change, percentage change, and directional label consistently
        expected_change = round(predicted_close - last_close, 2)
        expected_change_pct = round(((predicted_close - last_close) / last_close) * 100, 2)

        if expected_change > 0.05:
            direction = "BULLISH"
        elif expected_change < -0.05:
            direction = "BEARISH"
        else:
            direction = "FLAT"

        # 17. Return result using existing PredictionResponse contract
        # Benchmark fields are null because Stage A metrics are training-duration selection metrics,
        # not held-out test benchmarks.
        recent_candles = []
        for _, row in window_df.iterrows():
            recent_candles.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "ma20": round(float(row["Close"] / (1.0 + row["close_to_ma20"])), 2) if "close_to_ma20" in row else None,
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
            "model_version": metadata.get("model_version", "cnn_lstm_stationary_v1"),
            "architecture": metadata.get("architecture", "CNN-LSTM (Conv1D + LSTM)"),
            "pipeline_version": "stationary-return-v1",
            "predicted_return": round(pred_return, 6),
            "input_window_size": 30,
            "features_used": metadata.get("features", STATIONARY_FEATURE_COLUMNS),
            "historical_directional_accuracy": None,
            "directional_accuracy_benchmark": None,
            "mae_benchmark": None,
            "rmse_benchmark": None,
            "mape_benchmark": None,
            "data_status": "LIVE",
            "data_source": settings.PROVIDER_NAME,
            "market_status": market_status,
            "recent_input_candles": recent_candles,
        }

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


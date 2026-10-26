"""Live, point-in-time feature-sensitivity diagnostics for the stationary pipeline.

This service does not retrain models or alter saved weights. For each feature, it
neutralizes that feature across the entire 30-session standardized input window
(set to 0, the training-mean reference under StandardScaler) and measures how the
frozen model's forecast changes. This is local model sensitivity, not causality.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.services.market_status_service import market_status_service
from backend.services.model_registry import model_registry
from backend.services.prediction_service import PredictionService
from ml.features.stationary import (
    FEATURE_COLUMNS as STATIONARY_FEATURE_COLUMNS,
    prepare_inference_features,
    reconstruct_price_from_return,
    validate_sequence_tensor,
)


class ExplainabilityService:
    """Compute current-window feature neutralization sensitivity without retraining."""

    def __init__(self):
        # Reuse the existing provider and the canonical market-data provenance guard.
        self.prediction_service = PredictionService()
        self.provider = self.prediction_service.provider

    @staticmethod
    def _direction(last_close: float, predicted_return: float) -> str:
        predicted_close = round(reconstruct_price_from_return(last_close, predicted_return), 2)
        change = round(predicted_close - last_close, 2)
        if change > 0.05:
            return "BULLISH"
        if change < -0.05:
            return "BEARISH"
        return "FLAT"

    def get_stationary_feature_sensitivity(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()
        if (
            not model_registry.is_stationary_supported(clean_symbol)
            or not model_registry.has_stationary_artifacts(clean_symbol)
        ):
            raise FileNotFoundError(
                f"Live explainability is available only for symbols with stationary-return-v1 model artifacts; "
                f"no eligible package was found for '{clean_symbol}'."
            )

        market_status = market_status_service.get_market_status()
        latest_completed_date = market_status_service.get_latest_completed_trading_day()

        # Fetch freshly on each request. Do not use stale cached sensitivity values.
        raw_df = self.provider.fetch_historical_ohlcv(clean_symbol, period="1y", interval="1d")
        if raw_df.empty or len(raw_df) < 55:
            raise ValueError(f"Insufficient daily market history for {clean_symbol} to compute explainability.")

        df = raw_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["SessionDate"] = df["Date"].dt.date
        completed_df = df[df["SessionDate"] <= latest_completed_date].copy()
        completed_df = completed_df.sort_values("Date").reset_index(drop=True)
        provenance = self.prediction_service._market_data_provenance(
            raw_df, completed_df, latest_completed_date
        )

        if len(completed_df) < 50:
            raise ValueError(f"Insufficient completed sessions for 30-session stationary input window: {clean_symbol}.")

        features_df, feature_columns = prepare_inference_features(completed_df)
        if len(features_df) < 30:
            raise ValueError(f"Insufficient valid feature rows to create a 30-session input for {clean_symbol}.")
        if list(feature_columns) != list(STATIONARY_FEATURE_COLUMNS):
            raise RuntimeError("Feature order from canonical pipeline does not match the deployed stationary model contract.")

        window_df = features_df.tail(30).copy()
        last_close = float(window_df["Close"].iloc[-1])
        window_start = window_df["Date"].iloc[0].strftime("%Y-%m-%d")
        window_end = window_df["Date"].iloc[-1].strftime("%Y-%m-%d")
        target_date = market_status_service.get_next_trading_day(window_df["Date"].iloc[-1].date()).strftime("%Y-%m-%d")

        model = model_registry.get_stationary_model(clean_symbol)
        feature_scaler = model_registry.get_stationary_feature_scaler(clean_symbol)
        target_scaler = model_registry.get_stationary_target_scaler(clean_symbol)
        metadata = model_registry.get_stationary_metadata(clean_symbol)

        scaled_features = feature_scaler.transform(window_df)
        baseline_tensor = np.expand_dims(scaled_features, axis=0)
        validate_sequence_tensor(baseline_tensor, expected_window=30)
        baseline_scaled_return = model.predict(baseline_tensor, verbose=0)
        baseline_return = float(target_scaler.inverse_transform(baseline_scaled_return)[0])
        if not np.isfinite(baseline_return):
            raise RuntimeError("The deployed model returned a non-finite baseline prediction; sensitivity is unavailable.")

        baseline_close = round(reconstruct_price_from_return(last_close, baseline_return), 2)
        baseline_direction = self._direction(last_close, baseline_return)
        sensitivities: List[Dict[str, Any]] = []

        for index, feature_name in enumerate(STATIONARY_FEATURE_COLUMNS):
            perturbed_features = np.array(scaled_features, copy=True)
            # StandardScaler's zero is the training-mean reference. Neutralize this
            # feature across all 30 steps, preserving the other six feature channels.
            perturbed_features[:, index] = 0.0
            perturbed_tensor = np.expand_dims(perturbed_features, axis=0)
            validate_sequence_tensor(perturbed_tensor, expected_window=30)
            perturbed_scaled_return = model.predict(perturbed_tensor, verbose=0)
            perturbed_return = float(target_scaler.inverse_transform(perturbed_scaled_return)[0])
            if not np.isfinite(perturbed_return):
                raise RuntimeError(f"Non-finite sensitivity prediction for feature '{feature_name}'.")

            perturbed_close = round(reconstruct_price_from_return(last_close, perturbed_return), 2)
            perturbed_direction = self._direction(last_close, perturbed_return)
            sensitivities.append({
                "feature": feature_name,
                "latest_session_value": float(window_df[feature_name].iloc[-1]),
                "neutralized_predicted_return": perturbed_return,
                "delta_return_bps": (perturbed_return - baseline_return) * 10_000.0,
                "neutralized_predicted_close": perturbed_close,
                "delta_predicted_close": round(perturbed_close - baseline_close, 2),
                "direction_after_neutralization": perturbed_direction,
                "direction_flipped": perturbed_direction != baseline_direction,
            })

        sensitivities.sort(key=lambda item: abs(item["delta_return_bps"]), reverse=True)
        training_cutoff = (
            metadata.get("stage_b_final_deployment_refit", {}).get("training_data_end")
            or metadata.get("source_data_range", {}).get("last_completed_session")
        )

        return {
            "symbol": clean_symbol,
            "pipeline_version": "stationary-return-v1",
            "model_version": metadata.get("model_version", "cnn_lstm_stationary_v1"),
            "model_training_cutoff": training_cutoff,
            "method": "30-session feature-trajectory mean neutralization",
            "reference_definition": "One standardized feature channel is set to 0 across the 30 input sessions; for StandardScaler this is the training-mean reference.",
            "input_window_start": window_start,
            "input_window_end": window_end,
            "target_date": target_date,
            "last_close": round(last_close, 2),
            "baseline_predicted_return": baseline_return,
            "baseline_predicted_close": baseline_close,
            "baseline_direction": baseline_direction,
            "data_status": provenance["data_status"],
            "data_provider_source": provenance["data_provider_source"],
            "data_as_of_date": provenance["data_as_of_date"],
            "market_status": market_status,
            "window_size": 30,
            "feature_sensitivities": sensitivities,
            "disclaimer": "Local model sensitivity, not causal attribution or a guarantee of future price movement.",
        }


explainability_service = ExplainabilityService()

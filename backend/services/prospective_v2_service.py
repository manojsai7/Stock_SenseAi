import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_V2_PATH = PROJECT_ROOT / "research" / "EXP-06" / "evaluation" / "locked_evaluation_registry_v2.json"
DEFAULT_LOG_V2_PATH = PROJECT_ROOT / "research" / "EXP-06" / "evaluation" / "prospective_predictions_log.csv"


class ProspectiveV2Service:
    """
    Independent Prospective Evaluation Service for EXP-06 (GRU V2 Challenger).
    - Preserves EXP-05 evaluation log and registry strictly untouched.
    - Manages dedicated EXP-06 prospective evaluation stream for RELIANCE.NS.
    - Idempotent: enforces single daily entry per model for each target date.
    - Scored strictly after session outcome is finalized by exchange close.
    - All models evaluated against identical reference and actual closing prices.
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_V2_PATH,
        log_path: Path = DEFAULT_LOG_V2_PATH,
    ):
        self.registry_path = Path(registry_path)
        self.log_path = Path(log_path)

    def _load_registry(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"EXP-06 locked registry not found at '{self.registry_path}'")
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _load_log(self) -> pd.DataFrame:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            cols = {
                "forecast_id": "object", "forecast_timestamp": "object", "symbol": "object",
                "input_window_end": "object", "target_date": "object", "model_name": "object",
                "model_hash": "object", "training_cutoff": "object", "previous_close": "float64",
                "predicted_return": "float64", "predicted_close": "float64", "predicted_direction": "object",
                "actual_close": "float64", "actual_return": "float64", "actual_direction": "object",
                "direction_correct": "object", "price_error_abs": "float64", "status": "object",
                "resolution_timestamp": "object", "data_source": "object"
            }
            return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in cols.items()})
        df = pd.read_csv(self.log_path)
        obj_cols = ["actual_direction", "direction_correct", "resolution_timestamp", "status"]
        for c in obj_cols:
            if c in df.columns and df[c].dtype == "float64":
                df[c] = df[c].astype("object")
        return df

    def _save_log(self, df: pd.DataFrame) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.log_path, index=False)

    def record_daily_forecasts(
        self,
        symbol: str,
        target_date: str,
        input_window_end: str,
        previous_close: float,
        v1_pred: Dict[str, Any],
        v2_pred: Dict[str, Any],
        data_source: str = "Yahoo Finance"
    ) -> List[Dict[str, Any]]:
        """
        Logs a synchronized trio of forecasts (V1, V2, Naive) with idempotency protection.
        """
        clean_symbol = symbol.strip().upper()
        df_log = self._load_log()

        # Idempotency check: if target_date already recorded for this symbol, skip re-insertion
        existing = df_log[(df_log["symbol"] == clean_symbol) & (df_log["target_date"] == target_date)]
        if not existing.empty:
            logger.info(f"[EXP-06 LOG] Forecast already logged for {clean_symbol} on {target_date}. Skipping duplicate.")
            return existing.to_dict(orient="records")

        timestamp = datetime.now(timezone.utc).isoformat()
        registry = self._load_registry()
        models_meta = registry.get("models", {})

        new_rows = []
        # 1. Production CNN-LSTM V1
        v1_meta = models_meta.get("production_cnn_lstm", {})
        v1_return = float(v1_pred.get("predicted_return", 0.0))
        v1_close = float(v1_pred.get("predicted_close", previous_close))
        v1_dir = "UP" if v1_return > 0 else ("DOWN" if v1_return < 0 else "FLAT")
        new_rows.append({
            "forecast_id": f"{clean_symbol}_{target_date}_v1_prod",
            "forecast_timestamp": timestamp,
            "symbol": clean_symbol,
            "input_window_end": input_window_end,
            "target_date": target_date,
            "model_name": "Production CNN-LSTM V1",
            "model_hash": v1_meta.get("hash", "a4715e84cdd7a16d1b334936c9a39b35249bb594aec5c987a4f6ee62d50d4f11"),
            "training_cutoff": v1_meta.get("training_cutoff", "2026-10-06"),
            "previous_close": previous_close,
            "predicted_return": round(v1_return, 6),
            "predicted_close": round(v1_close, 2),
            "predicted_direction": v1_dir,
            "actual_close": np.nan,
            "actual_return": np.nan,
            "actual_direction": None,
            "direction_correct": None,
            "price_error_abs": np.nan,
            "status": "PENDING_OUTCOME",
            "resolution_timestamp": None,
            "data_source": data_source,
        })

        # 2. GRU V2 Challenger
        v2_meta = models_meta.get("gru_v2_challenger", {})
        v2_return = float(v2_pred.get("predicted_simple_return", 0.0))
        v2_close = float(v2_pred.get("predicted_next_session_close", previous_close))
        v2_dir = "UP" if v2_return > 0.0005 else ("DOWN" if v2_return < -0.0005 else "FLAT")
        new_rows.append({
            "forecast_id": f"{clean_symbol}_{target_date}_v2_gru",
            "forecast_timestamp": timestamp,
            "symbol": clean_symbol,
            "input_window_end": input_window_end,
            "target_date": target_date,
            "model_name": "GRU V2 Challenger",
            "model_hash": v2_meta.get("hash", "6f44d5782514af8223eb65564da7bef23ed85fbd049b79b2dc43ea1a568bbf45"),
            "training_cutoff": v2_meta.get("training_cutoff", "2025-12-11"),
            "previous_close": previous_close,
            "predicted_return": round(v2_return, 6),
            "predicted_close": round(v2_close, 2),
            "predicted_direction": v2_dir,
            "actual_close": np.nan,
            "actual_return": np.nan,
            "actual_direction": None,
            "direction_correct": None,
            "price_error_abs": np.nan,
            "status": "PENDING_OUTCOME",
            "resolution_timestamp": None,
            "data_source": data_source,
        })

        # 3. Naive Persistence Benchmark
        naive_meta = models_meta.get("naive_persistence", {})
        new_rows.append({
            "forecast_id": f"{clean_symbol}_{target_date}_naive",
            "forecast_timestamp": timestamp,
            "symbol": clean_symbol,
            "input_window_end": input_window_end,
            "target_date": target_date,
            "model_name": "Naive Persistence Benchmark",
            "model_hash": naive_meta.get("hash", "zero_parameter_baseline"),
            "training_cutoff": "None",
            "previous_close": previous_close,
            "predicted_return": 0.0,
            "predicted_close": round(previous_close, 2),
            "predicted_direction": "FLAT",
            "actual_close": np.nan,
            "actual_return": np.nan,
            "actual_direction": None,
            "direction_correct": None,
            "price_error_abs": np.nan,
            "status": "PENDING_OUTCOME",
            "resolution_timestamp": None,
            "data_source": data_source,
        })

        appended_df = pd.concat([df_log, pd.DataFrame(new_rows)], ignore_index=True)
        self._save_log(appended_df)
        logger.info(f"[EXP-06 LOG] Successfully logged 3 prospective forecasts for {clean_symbol} on target {target_date}")
        return new_rows

    def resolve_pending_forecasts(self, symbol: str, historical_df: pd.DataFrame) -> int:
        """
        Scans pending forecasts and resolves any target_date that has finalized exchange close.
        Returns number of resolved forecasts.
        """
        clean_symbol = symbol.strip().upper()
        df_log = self._load_log()
        if df_log.empty:
            return 0

        pending_mask = (df_log["symbol"] == clean_symbol) & (df_log["status"] == "PENDING_OUTCOME")
        if not pending_mask.any():
            return 0

        # Build date to close map from completed historical dataframe
        hist = historical_df.copy()
        hist["DateStr"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
        price_map = dict(zip(hist["DateStr"], hist["Close"]))

        resolved_count = 0
        now_ts = datetime.now(timezone.utc).isoformat()

        for idx in df_log[pending_mask].index:
            t_date = df_log.loc[idx, "target_date"]
            if t_date in price_map:
                actual_close = float(price_map[t_date])
                prev_close = float(df_log.loc[idx, "previous_close"])
                pred_close = float(df_log.loc[idx, "predicted_close"])
                pred_dir = df_log.loc[idx, "predicted_direction"]

                actual_return = (actual_close - prev_close) / prev_close
                actual_dir = "UP" if actual_return > 0 else ("DOWN" if actual_return < 0 else "FLAT")
                price_err = abs(pred_close - actual_close)
                is_correct = (pred_dir == actual_dir)

                df_log.loc[idx, "actual_close"] = round(actual_close, 2)
                df_log.loc[idx, "actual_return"] = round(actual_return, 6)
                df_log.loc[idx, "actual_direction"] = actual_dir
                df_log.loc[idx, "direction_correct"] = is_correct
                df_log.loc[idx, "price_error_abs"] = round(price_err, 2)
                df_log.loc[idx, "status"] = "RESOLVED"
                df_log.loc[idx, "resolution_timestamp"] = now_ts
                resolved_count += 1

        if resolved_count > 0:
            self._save_log(df_log)
            logger.info(f"[EXP-06 LOG] Resolved {resolved_count} pending forecasts for {clean_symbol}")

        return resolved_count

    def get_exp06_comparison(self, symbol: str) -> Dict[str, Any]:
        """
        Returns full comparative payload for frontend dashboard:
        - Models metadata
        - Sample counts (pending vs resolved)
        - Forecast streams
        - Chart data
        """
        clean_symbol = symbol.strip().upper()
        registry = self._load_registry()
        authorized_symbol = registry.get("symbol", "RELIANCE.NS").strip().upper()

        if clean_symbol != authorized_symbol:
            return {
                "symbol": clean_symbol,
                "supported": False,
                "message": (
                    f"Prospective shadow evaluation (EXP-06) is currently authorized exclusively for {authorized_symbol}. "
                    f"No prospective evaluation records exist for '{clean_symbol}'."
                ),
                "authorized_symbol": authorized_symbol,
                "sample_counts": {
                    "total_predictions": 0,
                    "resolved_sessions": 0,
                    "pending_sessions": 0,
                    "resolved_predictions": 0,
                    "pending_predictions": 0,
                },
                "models": [],
                "resolved_observations": [],
                "pending_forecasts": [],
            }

        df_log = self._load_log()
        symbol_log = df_log[df_log["symbol"] == clean_symbol] if not df_log.empty else pd.DataFrame()

        resolved_df = symbol_log[symbol_log["status"] == "RESOLVED"] if not symbol_log.empty else pd.DataFrame()
        pending_df = symbol_log[symbol_log["status"] == "PENDING_OUTCOME"] if not symbol_log.empty else pd.DataFrame()

        resolved_sessions = resolved_df["target_date"].nunique() if not resolved_df.empty else 0
        pending_sessions = pending_df["target_date"].nunique() if not pending_df.empty else 0

        # Model summary metrics
        models_config = registry.get("models", {})
        models_list = []

        for m_key, m_info in models_config.items():
            m_name = m_info["name"]
            m_resolved = resolved_df[resolved_df["model_name"] == m_name] if not resolved_df.empty else pd.DataFrame()
            n_obs = len(m_resolved)

            price_mae = round(float(m_resolved["price_error_abs"].mean()), 2) if n_obs > 0 else None
            dir_acc = round(float(m_resolved["direction_correct"].mean() * 100), 1) if n_obs > 0 else None

            models_list.append({
                "key": m_key,
                "name": m_name,
                "role": m_info.get("role", "Candidate"),
                "hash": m_info.get("hash", ""),
                "training_cutoff": m_info.get("training_cutoff", "Unknown"),
                "decision_rule": m_info.get("decision_rule", ""),
                "summary_metrics": {
                    "sample_size": n_obs,
                    "price_mae": price_mae,
                    "directional_accuracy_pct": dir_acc,
                }
            })

        # Group pending forecasts by target_date
        pending_forecasts = []
        if not pending_df.empty:
            for t_date, group in pending_df.groupby("target_date"):
                prev_close = float(group["previous_close"].iloc[0])
                input_end = str(group["input_window_end"].iloc[0])
                preds_map = {}
                for _, row in group.iterrows():
                    preds_map[row["model_name"]] = {
                        "predicted_close": float(row["predicted_close"]),
                        "predicted_return": float(row["predicted_return"]),
                        "predicted_direction": row["predicted_direction"],
                    }
                pending_forecasts.append({
                    "target_date": t_date,
                    "input_window_end": input_end,
                    "previous_close": prev_close,
                    "predictions": preds_map,
                })

        # Group resolved observations by target_date
        resolved_observations = []
        if not resolved_df.empty:
            for t_date, group in resolved_df.groupby("target_date"):
                actual_close = float(group["actual_close"].iloc[0])
                prev_close = float(group["previous_close"].iloc[0])
                input_end = str(group["input_window_end"].iloc[0])
                actual_dir = group["actual_direction"].iloc[0]
                preds_map = {}
                for _, row in group.iterrows():
                    preds_map[row["model_name"]] = {
                        "predicted_close": float(row["predicted_close"]),
                        "predicted_return": float(row["predicted_return"]),
                        "predicted_direction": row["predicted_direction"],
                        "price_error_abs": float(row["price_error_abs"]),
                        "direction_correct": bool(row["direction_correct"]),
                    }
                resolved_observations.append({
                    "target_date": t_date,
                    "input_window_end": input_end,
                    "previous_close": prev_close,
                    "actual_close": actual_close,
                    "actual_direction": actual_dir,
                    "predictions": preds_map,
                })

        return {
            "symbol": clean_symbol,
            "supported": True,
            "protocol": registry.get("protocol", "EXP-06-PROSPECTIVE-SHADOW"),
            "status": registry.get("status", "LOCKED_PROSPECTIVE"),
            "target_sample_size": registry.get("target_sample_size", "60–100 Prospective Sessions"),
            "sample_counts": {
                "total_predictions": len(symbol_log),
                "resolved_sessions": resolved_sessions,
                "pending_sessions": pending_sessions,
                "resolved_predictions": len(resolved_df),
                "pending_predictions": len(pending_df),
            },
            "models": models_list,
            "pending_forecasts": pending_forecasts,
            "resolved_observations": resolved_observations,
            "evaluation_gates": registry.get("evaluation_gates", {}),
        }


prospective_v2_service = ProspectiveV2Service()

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "research" / "EXP-05" / "evaluation" / "locked_evaluation_registry.json"
if not DEFAULT_REGISTRY_PATH.exists():
    DEFAULT_REGISTRY_PATH = Path("research/EXP-05/evaluation/locked_evaluation_registry.json")

DEFAULT_LOG_PATH = PROJECT_ROOT / "research" / "EXP-05" / "evaluation" / "prospective_predictions_log.csv"
if not DEFAULT_LOG_PATH.exists():
    DEFAULT_LOG_PATH = Path("research/EXP-05/evaluation/prospective_predictions_log.csv")



class ResearchService:
    """
    Read-only service for prospective model comparison.
    Loads locked evaluation registry metadata and parses prospective prediction logs.
    Strictly isolates resolved outcomes from pending forecasts.
    Guarantees no mixing of historical offline folds with prospective forward metrics.
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        log_path: Path = DEFAULT_LOG_PATH
    ):
        self.registry_path = Path(registry_path)
        self.log_path = Path(log_path)

    def get_model_comparison(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()

        # 1. Load registry and verify symbol authorization
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Locked evaluation registry not found at '{self.registry_path}'"
            )

        with open(self.registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        authorized_symbol = registry.get("symbol", "RELIANCE.NS").strip().upper()
        if clean_symbol != authorized_symbol:
            return {
                "symbol": clean_symbol,
                "supported": False,
                "message": (
                    f"Prospective model comparison is currently scoped exclusively to {authorized_symbol} "
                    f"under locked protocol EXP-05. No prospective evaluation logs exist for '{clean_symbol}'."
                ),
                "authorized_symbol": authorized_symbol,
                "sample_counts": {
                    "total_predictions": 0,
                    "total_sessions": 0,
                    "resolved_sessions": 0,
                    "pending_sessions": 0,
                },
                "models": [],
                "resolved_observations": [],
                "pending_forecasts": [],
            }

        # 2. Load prospective predictions log
        if not self.log_path.exists():
            raise FileNotFoundError(
                f"Prospective predictions log not found at '{self.log_path}'"
            )

        try:
            df = pd.read_csv(self.log_path)
        except Exception as e:
            raise ValueError(f"Failed to parse prospective predictions log: {e}")

        if df.empty:
            return {
                "symbol": clean_symbol,
                "supported": True,
                "status": "EMPTY_LOG",
                "message": "Prospective predictions log exists but contains no records.",
                "sample_counts": {
                    "total_predictions": 0,
                    "total_sessions": 0,
                    "resolved_sessions": 0,
                    "pending_sessions": 0,
                },
                "models": [],
                "resolved_observations": [],
                "pending_forecasts": [],
            }

        # Clean symbol filter
        df = df[df["symbol"].str.strip().str.upper() == clean_symbol].copy()

        # 3. Categorize into resolved and pending
        resolved_mask = df["status"].str.upper() == "RESOLVED"
        pending_mask = df["status"].str.upper() == "PENDING_OUTCOME"

        resolved_df = df[resolved_mask].copy()
        pending_df = df[pending_mask].copy()

        total_predictions = len(df)
        total_sessions = int(df["target_date"].nunique()) if "target_date" in df else 0
        resolved_sessions = int(resolved_df["target_date"].nunique()) if not resolved_df.empty else 0
        pending_sessions = int(pending_df["target_date"].nunique()) if not pending_df.empty else 0

        # 4. Compute model-specific summary metrics strictly on RESOLVED rows
        model_configs = registry.get("models", {})
        model_cards = []

        # Canonical mapping of model keys to model names in log
        canonical_models = [
            {
                "key": "production_cnn_lstm",
                "display_name": "Production CNN-LSTM",
                "log_name_match": "Production CNN-LSTM",
                "role": "Production Model",
                "badge": "Active Production Checkpoint",
                "cutoff": model_configs.get("production_cnn_lstm", {}).get("training_cutoff", "2026-10-06"),
                "hash": model_configs.get("production_cnn_lstm", {}).get("hash", ""),
                "decision_rule": model_configs.get("production_cnn_lstm", {}).get("decision_rule", "return > 0 -> UP, return < 0 -> DOWN, else FLAT"),
            },
            {
                "key": "colab_exp05_fold3",
                "display_name": "Colab EXP-05 Fold 3 (Calibrated)",
                "log_name_match": "Colab EXP-05 Fold 3 (Calibrated)",
                "role": "Research Challenger",
                "badge": "Challenger (EXP-05 Fold 3)",
                "cutoff": model_configs.get("colab_exp05_fold3", {}).get("training_cutoff", "2025-04-21 (val cutoff: 2026-01-13)"),
                "hash": model_configs.get("colab_exp05_fold3", {}).get("hash", ""),
                "decision_rule": model_configs.get("colab_exp05_fold3", {}).get("decision_rule", "return > theta -> UP, return < theta -> DOWN, else FLAT"),
            },
            {
                "key": "naive_persistence",
                "display_name": "Naive Persistence",
                "log_name_match": "Naive Persistence",
                "role": "Reference Baseline",
                "badge": "Zero-Parameter Martingale Baseline",
                "cutoff": "none",
                "hash": "zero_parameter_baseline",
                "decision_rule": model_configs.get("naive_persistence", {}).get("decision_rule", "predicted_return = 0.0, predicted_close = latest_close"),
            },
        ]

        for m_info in canonical_models:
            m_resolved = resolved_df[resolved_df["model_name"] == m_info["log_name_match"]]
            resolved_count = len(m_resolved)

            if resolved_count > 0:
                errors = m_resolved["price_error_abs"].dropna().astype(float)
                mae = round(float(errors.mean()), 2) if not errors.empty else None
                rmse = round(float(np.sqrt((errors ** 2).mean())), 2) if not errors.empty else None

                # Directional accuracy
                if "direction_correct" in m_resolved:
                    dir_series = m_resolved["direction_correct"].astype(str).str.lower() == "true"
                    correct_count = int(dir_series.sum())
                    dir_acc = round(float(correct_count / resolved_count * 100), 2)
                else:
                    correct_count = 0
                    dir_acc = None

                metrics_summary = {
                    "resolved_count": resolved_count,
                    "price_mae": mae,
                    "price_rmse": rmse,
                    "directional_accuracy_pct": dir_acc,
                    "correct_directions": correct_count,
                    "total_evaluated": resolved_count,
                }
            else:
                metrics_summary = {
                    "resolved_count": 0,
                    "price_mae": None,
                    "price_rmse": None,
                    "directional_accuracy_pct": None,
                    "correct_directions": 0,
                    "total_evaluated": 0,
                }

            model_cards.append({
                "key": m_info["key"],
                "name": m_info["display_name"],
                "role": m_info["role"],
                "badge": m_info["badge"],
                "training_cutoff": m_info["cutoff"],
                "hash": m_info["hash"],
                "decision_rule": m_info["decision_rule"],
                "summary_metrics": metrics_summary,
            })

        # 5. Format resolved sessions
        resolved_sessions_list = []
        if not resolved_df.empty:
            for target_date, group in resolved_df.groupby("target_date", sort=True):
                first_row = group.iloc[0]
                session_item = {
                    "target_date": str(target_date),
                    "input_window_end": str(first_row.get("input_window_end", "")),
                    "previous_close": float(first_row["previous_close"]) if pd.notnull(first_row.get("previous_close")) else None,
                    "actual_close": float(first_row["actual_close"]) if pd.notnull(first_row.get("actual_close")) else None,
                    "actual_return": round(float(first_row["actual_return"]), 6) if pd.notnull(first_row.get("actual_return")) else None,
                    "actual_direction": str(first_row.get("actual_direction", "")),
                    "resolution_timestamp": str(first_row.get("resolution_timestamp", "")),
                    "status": "RESOLVED",
                    "predictions": {},
                }
                for _, row in group.iterrows():
                    m_name = row["model_name"]
                    session_item["predictions"][m_name] = {
                        "model_name": m_name,
                        "predicted_close": float(row["predicted_close"]) if pd.notnull(row.get("predicted_close")) else None,
                        "predicted_return": round(float(row["predicted_return"]), 6) if pd.notnull(row.get("predicted_return")) else None,
                        "predicted_direction": str(row.get("predicted_direction", "")),
                        "price_error_abs": round(float(row["price_error_abs"]), 2) if pd.notnull(row.get("price_error_abs")) else None,
                        "direction_correct": bool(str(row.get("direction_correct", "")).lower() == "true") if pd.notnull(row.get("direction_correct")) else None,
                    }
                resolved_sessions_list.append(session_item)

        # 6. Format pending forecasts
        pending_forecasts_list = []
        if not pending_df.empty:
            for target_date, group in pending_df.groupby("target_date", sort=True):
                first_row = group.iloc[0]
                session_item = {
                    "target_date": str(target_date),
                    "input_window_end": str(first_row.get("input_window_end", "")),
                    "previous_close": float(first_row["previous_close"]) if pd.notnull(first_row.get("previous_close")) else None,
                    "forecast_timestamp": str(first_row.get("forecast_timestamp", "")),
                    "status": "PENDING_OUTCOME",
                    "predictions": {},
                }
                for _, row in group.iterrows():
                    m_name = row["model_name"]
                    session_item["predictions"][m_name] = {
                        "model_name": m_name,
                        "predicted_close": float(row["predicted_close"]) if pd.notnull(row.get("predicted_close")) else None,
                        "predicted_return": round(float(row["predicted_return"]), 6) if pd.notnull(row.get("predicted_return")) else None,
                        "predicted_direction": str(row.get("predicted_direction", "")),
                    }
                pending_forecasts_list.append(session_item)

        return {
            "symbol": clean_symbol,
            "supported": True,
            "status": registry.get("status", "LOCKED"),
            "lock_timestamp": registry.get("lock_timestamp", ""),
            "target_sample_size": registry.get("target_sample_size", "60-100 prospective sessions"),
            "evaluation_gates": registry.get("evaluation_gates", {}),
            "scientific_disclaimer": (
                "Metrics are preliminary observations based on N=1 resolved prospective session. "
                "One successful forecast does not establish predictive superiority or statistical significance. "
                "Accumulating 60–100 prospective sessions is mandatory before running hypothesis tests."
            ),
            "sample_counts": {
                "total_predictions": total_predictions,
                "total_sessions": total_sessions,
                "resolved_sessions": resolved_sessions,
                "pending_sessions": pending_sessions,
                "resolved_predictions": len(resolved_df),
                "pending_predictions": len(pending_df),
            },
            "models": model_cards,
            "resolved_observations": resolved_sessions_list,
            "pending_forecasts": pending_forecasts_list,
        }


research_service = ResearchService()

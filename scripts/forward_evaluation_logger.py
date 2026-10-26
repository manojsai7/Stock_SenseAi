"""
StockSense AI — Prospective Forward Evaluation Logger (Phase 3)
Tracks immutable out-of-sample forward predictions committed BEFORE next-session closes are known.

Commands:
  lock:      Locks and registers model hashes, cutoffs, features, and evaluation rules.
  predict:   Generates and commits next-session forecast for a completed market session.
  resolve:   Attaches actual outcome once target market session completes.
  status:    Displays logged sample size, pending forecasts, and running metrics.
"""

import os
import sys
import json
import hashlib
import zipfile
import tempfile
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Project roots
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.preprocessing.stationary_scaler import ReturnTargetScaler, StationaryFeatureScaler
from ml.features.stationary import compute_stationary_features, FEATURE_COLUMNS
from scripts.run_reliance_head_to_head_evaluation import (
    compute_sha256,
    SafeCustomUnpickler,
    load_keras_model_safely,
    sign_from_diff
)

EVAL_DIR = PROJECT_ROOT / "research" / "EXP-05" / "evaluation"
REGISTRY_FILE = EVAL_DIR / "locked_evaluation_registry.json"
LOG_CSV = EVAL_DIR / "prospective_predictions_log.csv"


def lock_evaluation(symbol: str = "RELIANCE.NS"):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    clean_sym = symbol.replace(".", "_")

    prod_dir = PROJECT_ROOT / "models_storage_stationary" / clean_sym
    exp05_dir = PROJECT_ROOT / "research" / "EXP-05" / "extracted" / "EXP-05_Model_Comparison_20261021_100324"

    prod_model_path = prod_dir / "model_cnn_lstm_stationary.keras"
    colab_model_path = exp05_dir / "best_cnn_lstm_fold3.keras"

    if not prod_model_path.exists() or not colab_model_path.exists():
        raise FileNotFoundError("Required model checkpoints not found.")

    prod_hash = compute_sha256(prod_model_path)
    colab_hash = compute_sha256(colab_model_path)

    registry = {
        "status": "LOCKED",
        "lock_timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "input_window_size": 30,
        "features": list(FEATURE_COLUMNS),
        "target": "next_session_simple_return",
        "models": {
            "production_cnn_lstm": {
                "name": "Production CNN-LSTM",
                "hash": prod_hash,
                "training_cutoff": "2026-10-06",
                "feature_scaler": "stationary_feature_scaler.pkl",
                "target_scaler": "return_target_scaler.pkl",
                "decision_rule": "return > 0 -> UP, return < 0 -> DOWN, else FLAT"
            },
            "colab_exp05_fold3": {
                "name": "Colab EXP-05 Fold 3",
                "hash": colab_hash,
                "training_cutoff": "2025-04-21 (val cutoff: 2026-01-13)",
                "feature_scaler": "feature_scaler_3d_fold3.pkl",
                "target_scaler": "return_scaler_fold3.pkl",
                "calibrated_theta": 0.000307,
                "decision_rule": "return > theta -> UP, return < theta -> DOWN, else FLAT"
            },
            "naive_persistence": {
                "name": "Naive Persistence Benchmark",
                "hash": "zero_parameter_baseline",
                "training_cutoff": "none",
                "decision_rule": "predicted_return = 0.0, predicted_close = latest_close"
            }
        },
        "target_sample_size": "60-100 prospective sessions",
        "evaluation_gates": {
            "price_gate": "Candidate return MAE < Naive persistence MAE",
            "direction_gate": "Macro MCC > 0 with both UP Recall and DOWN Recall > 20%",
            "significance_gate": "Block bootstrap or Diebold-Mariano test p < 0.05"
        }
    }

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    # Initialize CSV if not present
    if not LOG_CSV.exists():
        columns = [
            "forecast_id", "forecast_timestamp", "symbol", "input_window_end",
            "target_date", "model_name", "model_hash", "training_cutoff",
            "previous_close", "predicted_return", "predicted_close",
            "predicted_direction", "actual_close", "actual_return",
            "actual_direction", "direction_correct", "price_error_abs",
            "status", "resolution_timestamp"
        ]
        pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False)

    print(f"Evaluation registry LOCKED successfully at {REGISTRY_FILE}")
    print(f"Production Model Hash:  {prod_hash}")
    print(f"Colab Fold 3 Model Hash: {colab_hash}")
    return registry


def generate_prospective_forecast(symbol: str = "RELIANCE.NS"):
    if not REGISTRY_FILE.exists():
        lock_evaluation(symbol)

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    clean_sym = symbol.replace(".", "_")
    proc_csv = PROJECT_ROOT / "data" / "processed" / f"{clean_sym}_processed.csv"
    if not proc_csv.exists():
        proc_csv = PROJECT_ROOT / "data" / "raw" / f"{clean_sym}.csv"

    df_raw = pd.read_csv(proc_csv)
    clean_df, feature_cols = compute_stationary_features(df_raw)

    L = 30
    if len(clean_df) < L:
        raise ValueError(f"Insufficient historical data ({len(clean_df)}) for window {L}")

    # Use the most recent completed window
    last_window = clean_df[feature_cols].iloc[-L:].values
    last_row = clean_df.iloc[-1]
    input_date = str(last_row["Date"])[:10]
    prev_close = float(last_row["Close"])

    # Load existing logs
    df_log = pd.read_csv(LOG_CSV)
    existing = df_log[df_log["input_window_end"] == input_date]
    if len(existing) > 0:
        print(f"Notice: Forecast for input date {input_date} already recorded in log ({len(existing)} rows).")
        return

    # Load models & scalers
    prod_dir = PROJECT_ROOT / "models_storage_stationary" / clean_sym
    exp05_dir = PROJECT_ROOT / "research" / "EXP-05" / "extracted" / "EXP-05_Model_Comparison_20261021_100324"

    prod_model = load_keras_model_safely(prod_dir / "model_cnn_lstm_stationary.keras")
    colab_model = load_keras_model_safely(exp05_dir / "best_cnn_lstm_fold3.keras")

    with open(prod_dir / "stationary_feature_scaler.pkl", "rb") as f:
        prod_feat_scaler = pickle.load(f)
    with open(prod_dir / "return_target_scaler.pkl", "rb") as f:
        prod_target_scaler = SafeCustomUnpickler(f).load()

    with open(exp05_dir / "feature_scaler_3d_fold3.pkl", "rb") as f:
        colab_feat_scaler = pickle.load(f)
    with open(exp05_dir / "return_scaler_fold3.pkl", "rb") as f:
        colab_target_scaler = SafeCustomUnpickler(f).load()

    X_in = np.expand_dims(last_window, axis=0).astype(np.float32)

    # Production prediction
    X_p_sc = prod_feat_scaler.transform(X_in)
    p_ret_sc = prod_model.predict(X_p_sc, verbose=0).flatten()[0]
    prod_pred_ret = float(prod_target_scaler.inverse_transform(p_ret_sc))
    prod_pred_close = float(prev_close * (1.0 + prod_pred_ret))
    prod_dir = "UP" if prod_pred_ret > 1e-5 else ("DOWN" if prod_pred_ret < -1e-5 else "FLAT")

    # Colab Fold 3 prediction
    X_c_sc = colab_feat_scaler.transform(X_in.reshape(-1, 7)).reshape(1, 30, 7)
    c_ret_sc = colab_model.predict(X_c_sc, verbose=0).flatten()[0]
    colab_pred_ret = float(colab_target_scaler.inverse_transform(c_ret_sc))
    colab_pred_close = float(prev_close * (1.0 + colab_pred_ret))
    theta = registry["models"]["colab_exp05_fold3"]["calibrated_theta"]
    colab_dir_cal = "UP" if (colab_pred_ret - theta) > 1e-5 else ("DOWN" if (colab_pred_ret - theta) < -1e-5 else "FLAT")

    # Naive persistence
    naive_pred_ret = 0.0
    naive_pred_close = float(prev_close)
    naive_dir = "FLAT"

    ts = datetime.now(timezone.utc).isoformat()
    target_date_est = "next_trading_session"

    new_rows = [
        {
            "forecast_id": f"{symbol}_{input_date}_prod",
            "forecast_timestamp": ts,
            "symbol": symbol,
            "input_window_end": input_date,
            "target_date": target_date_est,
            "model_name": "Production CNN-LSTM",
            "model_hash": registry["models"]["production_cnn_lstm"]["hash"],
            "training_cutoff": registry["models"]["production_cnn_lstm"]["training_cutoff"],
            "previous_close": round(prev_close, 2),
            "predicted_return": round(prod_pred_ret, 6),
            "predicted_close": round(prod_pred_close, 2),
            "predicted_direction": prod_dir,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
            "direction_correct": None,
            "price_error_abs": None,
            "status": "PENDING_OUTCOME",
            "resolution_timestamp": None
        },
        {
            "forecast_id": f"{symbol}_{input_date}_colab_cal",
            "forecast_timestamp": ts,
            "symbol": symbol,
            "input_window_end": input_date,
            "target_date": target_date_est,
            "model_name": "Colab EXP-05 Fold 3 (Calibrated)",
            "model_hash": registry["models"]["colab_exp05_fold3"]["hash"],
            "training_cutoff": registry["models"]["colab_exp05_fold3"]["training_cutoff"],
            "previous_close": round(prev_close, 2),
            "predicted_return": round(colab_pred_ret, 6),
            "predicted_close": round(colab_pred_close, 2),
            "predicted_direction": colab_dir_cal,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
            "direction_correct": None,
            "price_error_abs": None,
            "status": "PENDING_OUTCOME",
            "resolution_timestamp": None
        },
        {
            "forecast_id": f"{symbol}_{input_date}_naive",
            "forecast_timestamp": ts,
            "symbol": symbol,
            "input_window_end": input_date,
            "target_date": target_date_est,
            "model_name": "Naive Persistence",
            "model_hash": "zero_parameter_baseline",
            "training_cutoff": "none",
            "previous_close": round(prev_close, 2),
            "predicted_return": 0.0,
            "predicted_close": round(naive_pred_close, 2),
            "predicted_direction": naive_dir,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
            "direction_correct": None,
            "price_error_abs": None,
            "status": "PENDING_OUTCOME",
            "resolution_timestamp": None
        }
    ]

    df_updated = pd.concat([df_log, pd.DataFrame(new_rows)], ignore_index=True)
    df_updated.to_csv(LOG_CSV, index=False)
    print(f"Logged {len(new_rows)} prospective forecasts for {symbol} (Input Cutoff: {input_date})")


def print_status():
    if not REGISTRY_FILE.exists():
        print("Evaluation registry is NOT locked yet. Run: python scripts/forward_evaluation_logger.py lock")
        return

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    print("=" * 60)
    print("StockSense AI — Prospective Evaluation Status")
    print("=" * 60)
    print(f"Status:             {registry['status']}")
    print(f"Locked At:          {registry['lock_timestamp']}")
    print(f"Target Equity:      {registry['symbol']}")
    print(f"Target Horizon:     {registry['target_sample_size']}")
    print(f"Production Model:   {registry['models']['production_cnn_lstm']['hash'][:16]}... (Cutoff: {registry['models']['production_cnn_lstm']['training_cutoff']})")
    print(f"Colab Model:        {registry['models']['colab_exp05_fold3']['hash'][:16]}... (Cutoff: {registry['models']['colab_exp05_fold3']['training_cutoff']})")

    if LOG_CSV.exists():
        df = pd.read_csv(LOG_CSV)
        total = len(df)
        resolved = len(df[df["status"] == "RESOLVED"])
        pending = len(df[df["status"] == "PENDING_OUTCOME"])
        unique_sessions = df["input_window_end"].nunique()
        print(f"\nLogged Records:     {total} ({unique_sessions} trading sessions)")
        print(f"Resolved Sessions:  {resolved // 3 if total else 0}")
        print(f"Pending Sessions:   {pending // 3 if total else 0}")
    else:
        print("\nLogged Records:     0 (Log file not created)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSense AI Forward Evaluation Logger")
    parser.add_argument("command", choices=["lock", "predict", "status"], help="Action to execute")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Ticker symbol")
    args = parser.parse_args()

    if args.command == "lock":
        lock_evaluation(args.symbol)
    elif args.command == "predict":
        generate_prospective_forecast(args.symbol)
    elif args.command == "status":
        print_status()

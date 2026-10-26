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
import pickle
import hashlib
import zipfile
import tempfile
import argparse
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Dict, Any, List, Optional, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import yfinance as yf

# Project roots
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.preprocessing.stationary_scaler import ReturnTargetScaler, StationaryFeatureScaler
from ml.features.stationary import FEATURE_COLUMNS, prepare_inference_features
from backend.services.market_status_service import market_status_service
from ml.data.acquisition import ResearchDataAcquisition
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

    with open(REGISTRY_FILE, "w", encoding="utf-8", newline="\n") as f:
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
        pd.DataFrame(columns=columns).to_csv(LOG_CSV, index=False, lineterminator="\n")

    print(f"Evaluation registry LOCKED successfully at {REGISTRY_FILE}")
    print(f"Production Model Hash:  {prod_hash}")
    print(f"Colab Fold 3 Model Hash: {colab_hash}")
    return registry


def fetch_and_validate_market_data(
    symbol: str = "RELIANCE.NS",
    save_cache: bool = True
) -> Tuple[pd.DataFrame, date, date]:
    """
    Acquires market data and applies the Daily Candle Inference Rule & Strict Freshness Guard.
    Discards any incomplete intraday candle (e.g. today's bar if market is still open).
    Rejects stale data if the latest available completed candle is older than the latest
    completed trading session on the exchange calendar.
    """
    clean_sym = symbol.replace(".", "_")

    # 1. Determine official exchange session calendar
    latest_completed_day = market_status_service.get_latest_completed_trading_day()
    next_target_day = market_status_service.get_next_trading_day(latest_completed_day)

    # 2. Ingest fresh data
    df_raw = None
    try:
        acq = ResearchDataAcquisition()
        df_raw = acq.fetch_raw_ohlcv(symbol, period="5y")
    except Exception as e:
        print(f"Warning: Live acquisition via yfinance failed ({e}). Checking local cache...")

    if df_raw is None or df_raw.empty:
        proc_csv = PROJECT_ROOT / "data" / "processed" / f"{clean_sym}_processed.csv"
        if not proc_csv.exists():
            proc_csv = PROJECT_ROOT / "data" / "raw" / f"{clean_sym}.csv"
        if not proc_csv.exists():
            raise FileNotFoundError(f"No local data found for {symbol} and live fetch failed.")
        df_raw = pd.read_csv(proc_csv)

    # 3. Clean date column
    df_raw["Date"] = pd.to_datetime(df_raw["Date"]).dt.tz_localize(None)

    # 4. Filter strictly to COMPLETED trading sessions (The Daily Candle Inference Rule)
    # Discard any incomplete intraday candle where date > latest_completed_day
    df_completed = df_raw[df_raw["Date"].dt.date <= latest_completed_day].copy()
    df_completed = df_completed.sort_values("Date").reset_index(drop=True)

    if df_completed.empty:
        raise RuntimeError(f"No completed trading data available for {symbol}.")

    last_candle_date = df_completed["Date"].dt.date.iloc[-1]

    # 5. Strict Freshness Guard: Reject stale data
    if last_candle_date < latest_completed_day:
        raise RuntimeError(
            f"Data Freshness Error: Stale market data for {symbol}.\n"
            f"Latest candle available:          {last_candle_date}\n"
            f"Expected completed market session: {latest_completed_day}\n"
            f"Data provider returned outdated data. Please update data provider before predicting.\n"
            f"Never bypass the freshness guard or use incomplete candles."
        )

    # Save fresh validated cache if requested
    if save_cache:
        proc_csv = PROJECT_ROOT / "data" / "processed" / f"{clean_sym}_processed.csv"
        raw_csv = PROJECT_ROOT / "data" / "raw" / f"{clean_sym}.csv"
        proc_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df_completed.to_csv(raw_csv, index=False, lineterminator="\n")

        # Compute MA20 for processed
        df_proc = df_completed.copy()
        df_proc["MA20"] = df_proc["Close"].rolling(window=20, min_periods=20).mean()
        df_proc.to_csv(proc_csv, index=False, lineterminator="\n")

    return df_completed, latest_completed_day, next_target_day


def generate_prospective_forecast(symbol: str = "RELIANCE.NS"):
    if not REGISTRY_FILE.exists():
        lock_evaluation(symbol)

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Ingest and validate data freshness
    df_completed, latest_completed_day, next_target_day = fetch_and_validate_market_data(symbol)

    clean_df, feature_cols = prepare_inference_features(df_completed)

    L = 30
    if len(clean_df) < L:
        raise ValueError(f"Insufficient historical data ({len(clean_df)}) for window {L}")

    # Use the most recent completed window
    last_window = clean_df[feature_cols].iloc[-L:].values
    last_row = clean_df.iloc[-1]
    input_date = str(last_row["Date"])[:10]
    prev_close = float(last_row["Close"])

    if not LOG_CSV.exists():
        lock_evaluation(symbol)

    # Load existing logs
    df_log = pd.read_csv(LOG_CSV)
    existing = df_log[df_log["input_window_end"] == input_date]
    if len(existing) > 0:
        print(f"Notice: Prospective forecast for completed session {input_date} already logged ({len(existing)} rows).")
        return

    # Load models & scalers
    clean_sym = symbol.replace(".", "_")
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
    target_date_str = str(next_target_day)

    new_rows = [
        {
            "forecast_id": f"{symbol}_{input_date}_prod",
            "forecast_timestamp": ts,
            "symbol": symbol,
            "input_window_end": input_date,
            "target_date": target_date_str,
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
            "target_date": target_date_str,
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
            "target_date": target_date_str,
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
    df_updated.to_csv(LOG_CSV, index=False, lineterminator="\n")
    print(f"\n[SUCCESS] Committed {len(new_rows)} prospective forecasts for {symbol}:")
    print(f"  Input Cutoff (Completed Session): {input_date} (Close: INR {prev_close:.2f})")
    print(f"  Target Trading Session:           {target_date_str}")
    print(f"  Production CNN-LSTM:              INR {prod_pred_close:.2f} ({prod_pred_ret:+.4%}, {prod_dir})")
    print(f"  Colab EXP-05 Fold 3 (Calibrated): INR {colab_pred_close:.2f} ({colab_pred_ret:+.4%}, {colab_dir_cal})")
    print(f"  Naive Persistence Benchmark:      INR {naive_pred_close:.2f} (+0.00%, FLAT)")
    print(f"  Log File:                         {LOG_CSV}")


def resolve_prospective_forecast(symbol: str = "RELIANCE.NS"):
    if not LOG_CSV.exists():
        print("No prospective log file found.")
        return

    df_log = pd.read_csv(LOG_CSV)
    pending_mask = (df_log["status"] == "PENDING_OUTCOME") & (df_log["symbol"] == symbol)
    if not pending_mask.any():
        print(f"No pending prospective forecasts to resolve for {symbol}.")
        return

    # Ingest latest completed data
    df_completed, latest_completed_day, _ = fetch_and_validate_market_data(symbol)
    df_completed["DateStr"] = df_completed["Date"].dt.date.astype(str)

    resolved_count = 0
    now_ts = datetime.now(timezone.utc).isoformat()

    for idx, row in df_log[pending_mask].iterrows():
        tgt_date = str(row["target_date"])
        match = df_completed[df_completed["DateStr"] == tgt_date]
        if not match.empty:
            actual_close = float(match.iloc[0]["Close"])
            prev_close = float(row["previous_close"])
            actual_ret = (actual_close - prev_close) / prev_close
            actual_dir = "UP" if actual_ret > 1e-5 else ("DOWN" if actual_ret < -1e-5 else "FLAT")

            pred_dir = row["predicted_direction"]
            dir_correct = (pred_dir == actual_dir) if pred_dir != "FLAT" else False
            price_err = abs(float(row["predicted_close"]) - actual_close)

            df_log.at[idx, "actual_close"] = round(actual_close, 2)
            df_log.at[idx, "actual_return"] = round(actual_ret, 6)
            df_log.at[idx, "actual_direction"] = actual_dir
            df_log.at[idx, "direction_correct"] = bool(dir_correct)
            df_log.at[idx, "price_error_abs"] = round(price_err, 2)
            df_log.at[idx, "status"] = "RESOLVED"
            df_log.at[idx, "resolution_timestamp"] = now_ts
            resolved_count += 1

    if resolved_count > 0:
        df_log.to_csv(LOG_CSV, index=False, lineterminator="\n")
        print(f"[RESOLVED] Attached actual market outcomes for {resolved_count} predictions.")
    else:
        print(f"Pending forecasts exist, but target market session outcome is not yet completed in market data.")


def print_status():
    if not REGISTRY_FILE.exists():
        print("Evaluation registry is NOT locked yet. Run: python scripts/forward_evaluation_logger.py lock")
        return

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)

    latest_completed = market_status_service.get_latest_completed_trading_day()
    mkt_status = market_status_service.get_market_status()

    print("=" * 66)
    print("StockSense AI — Prospective Evaluation Status")
    print("=" * 66)
    print(f"Registry Status:          {registry['status']}")
    print(f"Locked At:               {registry['lock_timestamp']}")
    print(f"Target Equity:           {registry['symbol']}")
    print(f"Market Status:           {mkt_status} (Latest Completed Session: {latest_completed})")
    print(f"Target Horizon:          {registry['target_sample_size']}")
    print(f"Production Model Hash:   {registry['models']['production_cnn_lstm']['hash'][:16]}... (Cutoff: {registry['models']['production_cnn_lstm']['training_cutoff']})")
    print(f"Colab Model Hash:        {registry['models']['colab_exp05_fold3']['hash'][:16]}... (Cutoff: {registry['models']['colab_exp05_fold3']['training_cutoff']})")

    if LOG_CSV.exists():
        df = pd.read_csv(LOG_CSV)
        total = len(df)
        resolved = df[df["status"] == "RESOLVED"]
        pending = df[df["status"] == "PENDING_OUTCOME"]
        unique_sessions = df["input_window_end"].nunique()
        print(f"\nLogged Records:          {total} ({unique_sessions} trading sessions)")
        print(f"Resolved Records:        {len(resolved)} ({resolved['input_window_end'].nunique()} sessions)")
        print(f"Pending Records:         {len(pending)} ({pending['input_window_end'].nunique()} sessions)")

        if len(pending) > 0:
            print("\nPending Predictions Awaiting Next-Session Close:")
            for s_date, grp in pending.groupby("input_window_end"):
                tgt = grp['target_date'].iloc[0]
                prev_c = grp['previous_close'].iloc[0]
                print(f"  * Input Cutoff: {s_date} (Close: INR {prev_c:.2f}) -> Target: {tgt}")
                for _, r in grp.iterrows():
                    print(f"    - {r['model_name']:32}: INR {r['predicted_close']:.2f} ({r['predicted_return']:+.4%}, {r['predicted_direction']})")

        if len(resolved) > 0:
            print("\nRunning Evaluation Metrics:")
            for m_name, grp in resolved.groupby("model_name"):
                mae = grp["price_error_abs"].mean()
                acc = grp["direction_correct"].mean() * 100.0 if "direction_correct" in grp else 0.0
                print(f"  * {m_name:32}: Price MAE = INR {mae:.2f} | Dir Acc = {acc:.1f}% (N={len(grp)})")
    else:
        print("\nLogged Records:          0 (Log file not created)")
    print("=" * 66)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSense AI Forward Evaluation Logger")
    parser.add_argument("command", choices=["lock", "predict", "resolve", "status"], help="Action to execute")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Ticker symbol")
    args = parser.parse_args()

    if args.command == "lock":
        lock_evaluation(args.symbol)
    elif args.command == "predict":
        generate_prospective_forecast(args.symbol)
    elif args.command == "resolve":
        resolve_prospective_forecast(args.symbol)
    elif args.command == "status":
        print_status()

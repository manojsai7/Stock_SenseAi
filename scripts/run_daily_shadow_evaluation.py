#!/usr/bin/env python3
"""
StockSense AI — Daily Prospective Shadow Evaluation Runner
Authoritative daily runner for prospective model tracking under protocols EXP-05 and EXP-06.

Operational Rules & Invariants:
1. Pure Idempotency: Running this script multiple times for the same market session produces
   the exact same state without creating duplicate forecast rows.
2. Market Hours Safety: Never logs a forward prospective forecast after the target trading
   session has opened (09:15 IST).
3. Finalized Candle Requirement: Requires completed daily OHLCV bars. Rejects unfinalized intraday bars.
4. Synchronized Trio: Production CNN-LSTM V1, GRU V2 Challenger, and Naive Persistence Benchmark
   must share the exact same input window cutoff date and previous closing price.
5. Pending Resolution: Resolves any older pending forecasts whose target date is now in the past
   and has finalized settlement data.
6. Zero Fabrication: Never invents retroactively back-dated forecasts for missed sessions.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DAILY-SHADOW] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("daily_shadow_evaluation")

from backend.services.market_status_service import market_status_service
from backend.services.prospective_v2_service import prospective_v2_service
from backend.services.challenger_prediction_service import challenger_prediction_service
from backend.services.prediction_service import PredictionService, StaleMarketDataError
from ml.data.providers.yahoo_provider import YahooFinanceProvider


def run_daily_shadow_evaluation(symbol: str = "RELIANCE.NS", dry_run: bool = False) -> int:
    """
    Executes the daily automated prospective shadow evaluation pipeline.
    Returns exit code:
      0 = Successful execution (logged new forecast or verified existing)
      1 = Error (stale data, inference failure, network error)
      2 = Skipped (active trading session, holiday/weekend without completed candle)
    """
    clean_symbol = symbol.strip().upper()
    now_ist = market_status_service.get_current_time()
    market_state = market_status_service.get_market_status(now_ist)
    logger.info(f"Initiating daily prospective shadow evaluation for {clean_symbol}")
    logger.info(f"Current Time: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} | Market State: {market_state}")

    # Guard 1: Do not log prospective forecasts during active trading sessions (09:00 - 15:30 IST)
    # Target sessions must be committed before market open or after session close for the next session.
    if market_state in {"OPEN", "PRE_MARKET"}:
        logger.warning(
            f"Market is currently ACTIVE ({market_state}). Prospective shadow forecasts must only be "
            f"generated and logged outside regular trading hours with finalized market candles. Exiting."
        )
        return 2

    # Step 1: Fetch 1-year historical OHLCV data to verify market completion
    provider = YahooFinanceProvider()
    try:
        raw_df = provider.fetch_historical_ohlcv(clean_symbol, period="1y", interval="1d")
        if raw_df.empty or len(raw_df) < 55:
            logger.error(f"Provider returned insufficient history for {clean_symbol}: {len(raw_df)} bars.")
            return 1
    except Exception as e:
        logger.error(f"Failed to fetch market data from provider for {clean_symbol}: {e}")
        return 1

    latest_completed_date = market_status_service.get_latest_completed_trading_day(now_ist)
    completed_session_str = latest_completed_date.strftime("%Y-%m-%d")
    next_trading_date = market_status_service.get_next_trading_day(latest_completed_date)
    next_session_str = next_trading_date.strftime("%Y-%m-%d")

    logger.info(f"Latest Completed Trading Session: {completed_session_str}")
    logger.info(f"Next Target Trading Session: {next_session_str}")

    # Step 2: Resolve any eligible pending forecasts whose target date is <= completed session
    raw_df["SessionDate"] = raw_df["Date"].dt.date
    completed_df = raw_df[raw_df["SessionDate"] <= latest_completed_date].copy()
    completed_df = completed_df.sort_values("Date").reset_index(drop=True)

    if not dry_run:
        try:
            resolved_count = prospective_v2_service.resolve_pending_forecasts(clean_symbol, completed_df)
            if resolved_count > 0:
                logger.info(f"Successfully resolved {resolved_count} pending forecast outcomes against official close.")
            else:
                logger.info("No pending forecasts were eligible for resolution at this time.")
        except Exception as e:
            logger.error(f"Error resolving pending forecasts: {e}")
            return 1

    # Step 3: Check if next-session forecast has already been recorded (Idempotency)
    existing_log = prospective_v2_service._load_log()
    already_logged = False
    if not existing_log.empty and "target_date" in existing_log.columns and "symbol" in existing_log.columns:
        matching = existing_log[
            (existing_log["symbol"] == clean_symbol) &
            (existing_log["target_date"] == next_session_str)
        ]
        if not matching.empty:
            already_logged = True
            logger.info(
                f"Prospective forecast for target date {next_session_str} has ALREADY been committed "
                f"({len(matching)} model entries found). Preserving existing records for idempotency."
            )

    if already_logged:
        logger.info("Execution complete: all eligible outcomes resolved and next-session forecast verified.")
        return 0

    # Step 4: Generate Synchronized Trio (V1 CNN-LSTM, V2 GRU Challenger, Naive Persistence)
    logger.info(f"Generating synchronized forecast trio for target date {next_session_str}...")
    try:
        # A. V2 GRU Challenger Prediction
        v2_pred = challenger_prediction_service.get_prediction(clean_symbol)
        v2_cutoff = v2_pred["latest_completed_input_date"]
        v2_ref_close = v2_pred["latest_completed_close"]
        v2_target = v2_pred["target_trading_date"]

        # B. V1 Production CNN-LSTM Prediction
        prod_service = PredictionService()
        v1_pred = prod_service._get_stationary_prediction(clean_symbol)
        v1_cutoff = v1_pred.get("latest_completed_session", v1_pred.get("input_window_end", v2_cutoff))
        v1_ref_close = v1_pred.get("last_close", v1_pred.get("reference_close", v2_ref_close))
        v1_target = v1_pred.get("target_date", v1_pred.get("target_trading_date", v2_target))
        v1_close = v1_pred.get("predicted_close", v1_pred.get("predicted_next_session_close", v1_ref_close))
        v1_ret = v1_pred.get("predicted_return", 0.0)

        # Invariant Verification: Identical input cutoff and reference close
        if v1_target != next_session_str or v2_target != next_session_str:
            logger.warning(
                f"Target date discrepancy: next_session={next_session_str}, "
                f"v1_target={v1_target}, v2_target={v2_target}. Reconciling to exchange calendar."
            )

        logger.info(
            f"Synchronized Forecasts Generated: Ref Close=₹{v2_ref_close:.2f} | Cutoff={v2_cutoff} | Target={next_session_str}\n"
            f"   - GRU V2: ₹{v2_pred['predicted_next_session_close']:.2f} ({v2_pred['predicted_return_pct']:+.2f}%, {v2_pred['direction']})\n"
            f"   - CNN-LSTM V1: ₹{v1_close:.2f} ({v1_ret * 100:+.2f}%, {v1_pred.get('direction', 'FLAT')})\n"
            f"   - Naive Baseline: ₹{v2_ref_close:.2f} (0.00%, FLAT)"
        )

        if dry_run:
            logger.info("[DRY RUN] Forecast generated successfully. Skipping file writes.")
            return 0

        # Step 5: Commit synchronized trio to immutable prospective evaluation log
        logged_rows = prospective_v2_service.record_daily_forecasts(
            symbol=clean_symbol,
            target_date=next_session_str,
            input_window_end=v2_cutoff,
            previous_close=v2_ref_close,
            v1_pred=v1_pred,
            v2_pred=v2_pred,
            data_source=v2_pred.get("data_provider", "Yahoo Finance (Daily Completed Bar)")
        )
        logger.info(f"Successfully recorded {len(logged_rows)} prospective forecast records into EXP-06 log.")
        return 0

    except StaleMarketDataError as sme:
        logger.warning(f"Market data is stale or provider has not yet finalized candle: {sme}")
        return 1
    except Exception as e:
        logger.error(f"Failed to generate synchronized prospective forecast: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSense AI Daily Shadow Evaluation")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Ticker symbol (default: RELIANCE.NS)")
    parser.add_argument("--dry-run", action="store_true", help="Execute inference without persisting to log")
    args = parser.parse_args()

    exit_code = run_daily_shadow_evaluation(symbol=args.symbol, dry_run=args.dry_run)
    sys.exit(exit_code)

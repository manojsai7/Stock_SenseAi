#!/usr/bin/env python3
"""
StockSense AI — Research Data Foundation & EDA Pipeline Runner
Executes:
1. Raw market data ingestion & validation (Yahoo Finance via yfinance)
2. Raw immutable CSV snapshots under data/raw/
3. Provenance manifest generation under data/manifests/
4. Chronological feature generation (OHLCV + MA20) under data/processed/
5. Academic Exploratory Data Analysis (EDA) suite under experiments/eda/<SYMBOL>/
6. Master machine-readable summary report at experiments/eda/summary.json
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.acquisition import ResearchDataAcquisition, RESEARCH_SYMBOLS
from ml.analysis.eda_pipeline import StockEDAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("research_data_runner")


def main():
    print("\n" + "=" * 70)
    print("  STOCKSENSE AI — RESEARCH DATA FOUNDATION & EDA PIPELINE")
    print("=" * 70)

    # 1. Acquisition & Feature Engineering
    print("\n[PHASE 1] Acquiring & Validating Raw Market Datasets...")
    acq = ResearchDataAcquisition()
    acq_results = acq.acquire_all(RESEARCH_SYMBOLS, force=False)

    print("\n--- Ingestion & Validation Summary ---")
    for sym, res in acq_results.items():
        if res.get("is_valid"):
            print(f"  [VALID] {sym:<15} | Rows: {res['rows']:<5} | Range: {res['start_date']} to {res['end_date']}")
        else:
            print(f"  [ERROR] {sym:<15} | Error: {res.get('error', 'Validation failed')}")

    # 2. EDA Pipeline
    print("\n[PHASE 2] Executing Academic Exploratory Data Analysis (EDA)...")
    eda = StockEDAPipeline()
    eda_summary = eda.run_all(RESEARCH_SYMBOLS)

    print("\n--- EDA Summary Matrix ---")
    print(f"  {'Symbol':<15} {'Start Date':<12} {'End Date':<12} {'Rows':<6} {'Mean Ret':<10} {'Std Vol':<10} {'Price Range (INR)'}")
    print("  " + "-" * 78)
    for row in eda_summary:
        price_range = f"{row['price_min']:.1f} - {row['price_max']:.1f}"
        print(f"  {row['symbol']:<15} {row['first_date']:<12} {row['last_date']:<12} {row['row_count']:<6} {row['mean_return']*100:>6.3f}%   {row['std_return']*100:>6.3f}%   {price_range}")

    print("\n" + "=" * 70)
    print("  RESEARCH PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

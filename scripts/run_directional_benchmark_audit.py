"""
Runner script for EXP-01: Offline Directional Benchmark & Metric Audit.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analysis.directional_benchmark_audit import DirectionalBenchmarkAuditor, MultiStockDirectionalAuditor


def main():
    args = sys.argv[1:]
    if len(args) == 0 or (len(args) == 1 and args[0] in ["--exp02", "exp02"]):
        symbols = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    elif len(args) == 1 and args[0] in ["--all", "all"]:
        symbols = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
    else:
        symbols = args

    if len(symbols) == 1:
        symbol = symbols[0]
        print(f"Running Directional Benchmark Audit for {symbol}...")
        auditor = DirectionalBenchmarkAuditor(symbol=symbol)
        res = auditor.run_audit()
        print("\n" + "=" * 80)
        print(f"BENCHMARK COMPARISON TABLE ({symbol})")
        print("=" * 80)
        print(res["comparison_df"].to_string(index=False))
        print("\nSaved Artifacts:")
        print(f"  CSV:    {res['csv_path']}")
        print(f"  JSON:   {res['json_path']}")
        print(f"  Report: {res['report_path']}")
    else:
        print(f"Running EXP-02 Multi-Stock Cross-Sectional Audit for: {symbols}...")
        multi_auditor = MultiStockDirectionalAuditor(symbols=symbols)
        res = multi_auditor.run_multi_audit()
        print("\n" + "=" * 80)
        print("EXP-02 CROSS-SECTIONAL MACRO-AVERAGE TABLE")
        print("=" * 80)
        print(res["macro_df"].to_string(index=False))
        print("\nSaved Artifacts:")
        print(f"  Combined CSV: {res['csv_path']}")
        print(f"  Summary JSON: {res['json_path']}")
        print(f"  Report MD:    {res['report_path']}")


if __name__ == "__main__":
    main()

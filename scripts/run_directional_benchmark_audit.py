"""
Runner script for EXP-01: Offline Directional Benchmark & Metric Audit.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analysis.directional_benchmark_audit import DirectionalBenchmarkAuditor


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print(f"Running EXP-01 Directional Benchmark Audit for {symbol}...")
    auditor = DirectionalBenchmarkAuditor(symbol=symbol)
    res = auditor.run_audit()
    print("\n" + "=" * 80)
    print("EXP-01 COMPARISON TABLE")
    print("=" * 80)
    print(res["comparison_df"].to_string(index=False))
    print("\nSaved Artifacts:")
    print(f"  CSV:    {res['csv_path']}")
    print(f"  JSON:   {res['json_path']}")
    print(f"  Report: {res['report_path']}")


if __name__ == "__main__":
    main()

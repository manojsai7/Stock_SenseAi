#!/usr/bin/env python3
"""
StockSense AI — Phase 3 Supervised Temporal Dataset & Sequence Generation Runner
Generates:
1. (N, 30, 6) feature tensors and (N, 1) target tensors for all 6 equities.
2. Compressed NumPy arrays under data/sequences/<SYMBOL>/
3. Auditable sequence example tables and reports under experiments/dataset/<SYMBOL>/
4. Consolidated dataset summary at experiments/dataset/summary.json
"""

import sys
import logging
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.dataset_builder import SupervisedDatasetBuilder, RESEARCH_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sequence_runner")


def main():
    print("\n" + "=" * 80)
    print("  STOCKSENSE AI — PHASE 3: SUPERVISED TEMPORAL DATASET BUILDER")
    print("=" * 80)

    builder = SupervisedDatasetBuilder(window_size=30, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    summaries = builder.build_all(RESEARCH_SYMBOLS)

    print("\n" + "-" * 80)
    print("  SUMMARY: SUPERVISED TENSOR POPULATION ACROSS 6 EQUITIES")
    print("-" * 80)
    print(f"  {'Symbol':<15} {'Raw':<6} {'Usable':<7} {'Total Seq':<10} {'X_train Shape':<16} {'X_val Shape':<14} {'X_test Shape':<14}")
    print("  " + "-" * 78)

    for s in summaries:
        xt_shape = str(tuple(s["shapes"]["X_train"]))
        xv_shape = str(tuple(s["shapes"]["X_val"]))
        xte_shape = str(tuple(s["shapes"]["X_test"]))
        print(f"  {s['symbol']:<15} {s['raw_observations']:<6} {s['usable_observations']:<7} {s['total_sequences']:<10} {xt_shape:<16} {xv_shape:<14} {xte_shape:<14}")

    print("\n" + "-" * 80)
    print("  TARGET HORIZON & SCALER PARAMETERS AUDIT")
    print("-" * 80)
    for s in summaries:
        print(f"\n  [{s['symbol']}] ({'SHORT-HISTORY EV' if s['is_short_history'] else 'MATURE 5-YEAR'})")
        print(f"    - Target Horizon: {s['first_target_date']} to {s['last_target_date']}")
        print(f"    - Sequences: Train={s['train_sequences']}, Val={s['validation_sequences']}, Test={s['test_sequences']} (Total: {s['total_sequences']})")
        print(f"    - Scaler Target Min/Max (TRAIN ONLY): INR {s['scaler_parameters']['target_min']:.2f} to INR {s['scaler_parameters']['target_max']:.2f}")
        print(f"    - Note: {s['history_note']}")

    # -------------------------------------------------------------------
    # Mathematical Alignment Demonstration
    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  MATHEMATICAL TARGET ALIGNMENT PROOF (TCS.NS & ATHERENERG.NS)")
    print("=" * 80)

    for sym in ["TCS.NS", "ATHERENERG.NS"]:
        clean = sym.replace(".", "_")
        npz_file = PROJECT_ROOT / "data" / "sequences" / clean / "sequences.npz"
        data = np.load(npz_file, allow_pickle=True)
        X_tr = data["X_train"]
        y_tr = data["y_train"]
        act_tr = data["actual_target_train"]
        prev_tr = data["prev_close_train"]
        dates_tr = data["dates_train"]

        print(f"\n  Equity: {sym} — Sample #0 Verification:")
        print(f"    - Input Window Shape: {X_tr[0].shape} (30 trading days x 6 features)")
        print(f"    - Last Input Step Scaled Close: {X_tr[0, -1, 3]:.5f}")
        print(f"    - Last Input Step Unscaled Close (Day t): INR {prev_tr[0]:.2f}")
        print(f"    - Scaled Target y (Day t+1): {y_tr[0]:.5f}")
        print(f"    - Actual Target Close (Day t+1): INR {act_tr[0]:.2f}")
        print(f"    - Target Date: {dates_tr[0]}")
        print(f"    - Ground Truth Movement: INR {act_tr[0] - prev_tr[0]:+.2f} ({'UP' if act_tr[0] >= prev_tr[0] else 'DOWN'})")

    print("\n" + "=" * 80)
    print("  PHASE 3 EXECUTION COMPLETE — READY FOR UNIT TESTING")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

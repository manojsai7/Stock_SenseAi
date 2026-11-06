"""Execute the 18-fit bounded exploratory development experiments."""
import json
import os
import sys
import time
from pathlib import Path

# Suppress TF protobuf / oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from ml.research_v2.runner import run


def main():
    config_path = ROOT / "research/EXP-06/config/exploratory_config.json"
    dataset_path = ROOT / "research/EXP-06/data/RELIANCE_NS_verified_v2.csv"
    output_dir = ROOT / "research/EXP-06/artifacts/exploratory_run"

    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"Starting exploratory development run: {config['experiment_id']}")
    print(f"Dataset: {dataset_path.name} | sha256: {config['dataset_sha256']}")
    print(f"Candidates: {[c['architecture'] for c in config['candidates']]}")
    print(f"Seeds: {config['seeds']} | Folds: {config['splits']['folds']}")

    t0 = time.time()
    results = run(config, dataset_path, output_dir)
    elapsed = time.time() - t0

    print(f"\nRun completed in {elapsed:.2f} seconds!")
    print(f"Results recorded: {len(results)}")

    # Load comparison summary
    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    summary = []
    for r in comparison:
        m = r["metrics"]
        summary.append({
            "fold": r["fold"],
            "model": r["model"],
            "seed": r.get("seed"),
            "price_mae": round(m["price_mae"], 4),
            "price_rmse": round(m["price_rmse"], 4),
            "mape_pct": round(m["mape_pct"], 4),
            "return_mae_bps": round(m["return_mae_bps"], 2),
            "directional_accuracy": round(m["directional_accuracy"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "mcc": round(m["mcc"], 4),
            "down_recall": round(m["recall"][0], 4) if len(m["recall"]) > 0 else None,
            "up_recall": round(m["recall"][2], 4) if len(m["recall"]) > 2 else None,
            "volatility_ratio": round(m["volatility_ratio"], 4)
        })
    df_summary = pd.DataFrame(summary)
    print("\n=== Model Comparison Leaderboard ===")
    print(df_summary.to_string(index=False))

    # Save summary table
    df_summary.to_csv(output_dir / "leaderboard_summary.csv", index=False)


if __name__ == "__main__":
    main()

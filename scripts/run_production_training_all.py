"""
Batch Runner & Smoke Tester for Production Stationary Models (Phase 2).

Executes:
1. Two-stage production training for TCS.NS, RELIANCE.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS.
2. Artifact integrity & smoke verification:
   - Reload model from disk
   - Reload both scalers
   - Recompute latest 30-session inference input via canonical prepare_inference_features
   - Verify shape (1, 30, 7) and feature ordering
   - Validate finite output, return inverse-transform, and price reconstruction
   - Verify SHA-256 checksums against manifest.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_production_stationary_model import (
    ProductionStationaryTrainer,
    compute_sha256
)
from ml.features.stationary import (
    FEATURE_COLUMNS,
    prepare_inference_features,
    reconstruct_price_from_return,
    WINDOW_SIZE
)
from ml.preprocessing.stationary_scaler import (
    StationaryFeatureScaler,
    ReturnTargetScaler
)

PRIMARY_SYMBOLS = [
    "TCS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]


def smoke_test_artifact(
    symbol: str,
    artifact_dir: Path,
    data_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive artifact integrity and inference smoke verification.
    """
    clean_sym = symbol.strip().upper().replace(".", "_")
    model_path = artifact_dir / "model_cnn_lstm_stationary.keras"
    feat_scaler_path = artifact_dir / "stationary_feature_scaler.pkl"
    target_scaler_path = artifact_dir / "return_target_scaler.pkl"
    metadata_path = artifact_dir / "metadata.json"
    manifest_path = artifact_dir / "manifest.json"

    # 1. Check file existence
    for p in [model_path, feat_scaler_path, target_scaler_path, metadata_path, manifest_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    # 2. Verify Manifest SHA-256 Checksums
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    artifacts_manifest = manifest_data["artifacts"]
    assert artifacts_manifest["model"]["sha256"] == compute_sha256(model_path), "Model SHA256 mismatch"
    assert artifacts_manifest["feature_scaler"]["sha256"] == compute_sha256(feat_scaler_path), "Feature scaler SHA256 mismatch"
    assert artifacts_manifest["target_scaler"]["sha256"] == compute_sha256(target_scaler_path), "Target scaler SHA256 mismatch"
    assert artifacts_manifest["metadata"]["sha256"] == compute_sha256(metadata_path), "Metadata SHA256 mismatch"

    # 3. Reload Model and Scalers from Disk
    reloaded_model = tf.keras.models.load_model(str(model_path), compile=False)
    reloaded_feat_scaler = StationaryFeatureScaler.load(feat_scaler_path)
    reloaded_target_scaler = ReturnTargetScaler.load(target_scaler_path)

    # 4. Load raw source data and build latest inference sequence
    if data_file is None:
        data_file = PROJECT_ROOT / "data" / "processed" / f"{clean_sym}_processed.csv"
        if not data_file.exists():
            data_file = PROJECT_ROOT / "data" / "raw" / f"{clean_sym}.csv"

    df_raw = pd.read_csv(data_file)
    clean_inf_df, feature_cols = prepare_inference_features(df_raw)

    assert feature_cols == FEATURE_COLUMNS, f"Feature column order mismatch: {feature_cols} != {FEATURE_COLUMNS}"
    assert len(clean_inf_df) >= WINDOW_SIZE, f"Insufficient rows ({len(clean_inf_df)}) for inference window {WINDOW_SIZE}"

    # Extract last 30 completed sessions
    window_df = clean_inf_df.iloc[-WINDOW_SIZE:]
    last_close = float(window_df["Close"].iloc[-1])
    last_date = str(window_df["Date"].iloc[-1])[:10]

    # Transform features using reloaded feature scaler
    scaled_window = reloaded_feat_scaler.transform(window_df)
    assert scaled_window.shape == (30, 7), f"Expected (30, 7), got {scaled_window.shape}"
    assert not np.isnan(scaled_window).any(), "Scaled window contains NaNs"
    assert not np.isinf(scaled_window).any(), "Scaled window contains Infs"

    # 5. Format 3D sequence tensor (1, 30, 7)
    input_tensor = np.expand_dims(scaled_window, axis=0).astype(np.float32)
    assert input_tensor.shape == (1, 30, 7), f"Expected input tensor shape (1, 30, 7), got {input_tensor.shape}"

    # 6. Execute forward pass
    raw_pred = reloaded_model.predict(input_tensor, verbose=0).flatten()
    assert len(raw_pred) == 1, f"Expected 1 scalar output, got {len(raw_pred)}"
    assert not np.isnan(raw_pred[0]), "Model prediction is NaN"
    assert not np.isinf(raw_pred[0]), "Model prediction is Inf"

    # 7. Inverse-transform predicted return
    pred_return = float(reloaded_target_scaler.inverse_transform(raw_pred[0]))
    assert not np.isnan(pred_return), "Inverse-transformed return is NaN"

    # 8. Reconstruct price
    pred_close = reconstruct_price_from_return(last_close, pred_return)
    assert pred_close > 0, f"Predicted close must be positive, got {pred_close}"
    assert not np.isnan(pred_close), "Predicted close is NaN"

    return {
        "symbol": symbol,
        "last_completed_date": last_date,
        "last_close": round(last_close, 2),
        "input_tensor_shape": list(input_tensor.shape),
        "predicted_standardized_output": round(float(raw_pred[0]), 6),
        "predicted_return": round(pred_return, 6),
        "predicted_close": round(pred_close, 2),
        "manifest_sha256_verified": True,
        "model_file_size": model_path.stat().st_size
    }


def main():
    print(f"[{datetime.now().isoformat()}] Starting Phase 2 Production Training across 5 NSE Large Caps...")
    results = {}

    for sym in PRIMARY_SYMBOLS:
        print("\n========================================================")
        print(f"TRAINING FINAL DEPLOYMENT ARTIFACTS: {sym}")
        print("========================================================")
        trainer = ProductionStationaryTrainer(
            symbol=sym,
            val_ratio=0.15,
            min_samples=400,
            max_epochs=40,
            batch_size=32,
            learning_rate=0.001,
            patience=8,
            random_seed=42,
            dry_run=False
        )

        train_res = trainer.run()

        # Run Smoke Test
        print(f"\n[SMOKE TEST] Running artifact reload & inference smoke test for {sym}...")
        smoke_res = smoke_test_artifact(sym, trainer.output_dir)
        print(f"[SMOKE TEST PASSED] {sym} -> Predicted Next Close: INR {smoke_res['predicted_close']} (Return: {smoke_res['predicted_return']*100:.2f}%)")

        results[sym] = {
            "training": train_res,
            "smoke_test": smoke_res
        }

    # Summary Report File
    summary_path = PROJECT_ROOT / "models_storage_stationary" / "phase2_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{datetime.now().isoformat()}] All 5 primary stocks trained, verified, and saved.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()

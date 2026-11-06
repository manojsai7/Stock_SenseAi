"""Export best justified V2 candidate and execute independent inference dry run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "research/EXP-06/artifacts/exploratory_run/fold_3/candidate_1_seed_42"
DEST_DIR = ROOT / "models_storage_v2/RELIANCE_NS"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_v2_artifacts():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "model.keras",
        "feature_scaler.pkl",
        "target_scaler.pkl",
        "history.csv",
        "predictions.csv",
        "model_metadata.json",
        "MODEL_CARD.md"
    ]

    for fname in files_to_copy:
        src = SOURCE_DIR / fname
        dst = DEST_DIR / fname
        shutil.copy2(src, dst)

    # Load metadata and enhance with V2 Challenger details
    metadata = json.loads((DEST_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    metadata.update({
        "model_role": "V2 Research Challenger (Shadow Candidate)",
        "model_architecture": "GRU",
        "units": 32,
        "seed": 42,
        "exploratory_track": True,
        "status": "experimental-v2-challenger",
        "dataset_verified_version": "v2",
        "dataset_sha256": metadata.get("dataset_sha256"),
        "training_cutoff": metadata.get("training_cutoff"),
        "validation_cutoff": metadata.get("validation_cutoff"),
        "test_cutoff": metadata.get("test_cutoff"),
        "checksums": {fname: fingerprint(DEST_DIR / fname) for fname in files_to_copy}
    })
    (DEST_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Generate formal V2 manifest
    manifest = {
        "model_id": "stocksense-v2-gru-challenger",
        "symbol": "RELIANCE.NS",
        "architecture": "GRU",
        "status": "experimental-shadow-candidate",
        "feature_version": metadata["feature_version"],
        "features": metadata["features"],
        "input_shape": [1, 30, 7],
        "training_cutoff": metadata["training_cutoff"],
        "model_checksum": fingerprint(DEST_DIR / "model.keras"),
        "files": {
            fname: {"path": fname, "sha256": fingerprint(DEST_DIR / fname)}
            for fname in files_to_copy
        }
    }
    (DEST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Exported V2 artifacts to: {DEST_DIR}")
    print(f"Model sha256: {manifest['model_checksum']}")


def dry_run_inference():
    print("\n=== Executing V2 Independent Inference Dry Run ===")

    # 1. Load Model & Scalers
    from ml.preprocessing.stationary_scaler import ReturnTargetScaler, StationaryFeatureScaler
    model_path = DEST_DIR / "model.keras"
    feature_scaler = StationaryFeatureScaler.load(DEST_DIR / "feature_scaler.pkl")
    target_scaler = ReturnTargetScaler.load(DEST_DIR / "target_scaler.pkl")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Model & scalers successfully loaded independently.")

    # 2. Prepare Sample Input
    # Load dataset sequences
    from ml.research_v2.pipeline import sequences
    dataset_csv = ROOT / "research/EXP-06/data/RELIANCE_NS_verified_v2.csv"
    data = sequences(pd.read_csv(dataset_csv), 30)

    # Select the very first test observation from fold 3 (index 1084)
    sample_idx = 1084
    sample_raw_x = data.x[sample_idx]  # Shape [30, 7]
    target_date = str(data.dates.iloc[sample_idx].target_date.date())
    prev_close = data.dates.iloc[sample_idx].previous_close
    actual_close = data.dates.iloc[sample_idx].actual_close
    actual_return = data.y[sample_idx]

    # Verify input tensor shape
    tensor_input = feature_scaler.transform(sample_raw_x)[np.newaxis, ...]  # Shape [1, 30, 7]
    assert tensor_input.shape == (1, 30, 7), f"Unexpected shape {tensor_input.shape}"
    print(f"Input tensor verified: shape = {tensor_input.shape}")

    # 3. Predict Raw Output
    raw_pred = model(tensor_input, training=False).numpy().reshape(-1)

    # 4. Inverse Transform Return & Price Reconstruction
    predicted_return = float(target_scaler.inverse_transform(raw_pred)[0])
    predicted_close = prev_close * (1.0 + predicted_return)

    print(f"\nDry Run Sample Prediction for Target Date: {target_date}")
    print(f"  Previous Close:   {prev_close:.2f} INR")
    print(f"  Predicted Return: {predicted_return * 100:.4f}% ({predicted_return * 10000:.1f} bps)")
    print(f"  Predicted Close:  {predicted_close:.2f} INR")
    print(f"  Actual Close:     {actual_close:.2f} INR")
    print(f"  Absolute Error:   {abs(predicted_close - actual_close):.2f} INR")

    # 5. Verify Repeatability against Saved predictions.csv
    saved_preds = pd.read_csv(DEST_DIR / "predictions.csv")
    match = saved_preds[saved_preds.target_date.str.startswith(target_date)].iloc[0]

    np.testing.assert_allclose(predicted_return, match.predicted_return, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(predicted_close, match.predicted_close, rtol=1e-6, atol=1e-8)
    print("\nRepeatability Check: EXACT MATCH with offline training predictions.csv!")

    # 6. Checksum validation
    manifest = json.loads((DEST_DIR / "manifest.json").read_text(encoding="utf-8"))
    for fname, info in manifest["files"].items():
        assert fingerprint(DEST_DIR / fname) == info["sha256"], f"Checksum mismatch on {fname}"
    print("Checksum Validation: All 7 artifact digests verified successfully.")


if __name__ == "__main__":
    export_v2_artifacts()
    dry_run_inference()

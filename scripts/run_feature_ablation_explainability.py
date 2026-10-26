"""Post-hoc feature-ablation diagnostics for StockSense AI's stationary CNN-LSTM.

This script never trains or writes to production model artifacts.

Modes
-----
fold3-test:
    Evaluates the EXP-05 Fold 3 checkpoint on its saved historical test block,
    verifies that the loaded model reproduces the archived predictions, and
    measures one-feature neutralization and window-wise permutation importance.
    This historical block has already been inspected; results are exploratory,
    not an untouched independent test.

input-sensitivity:
    Explains how the model's prediction for one latest available input window
    changes when one standardized feature channel is neutralized to zero (the
    training-mean reference for a centered StandardScaler). This is local model
    sensitivity, not evidence that the feature improves forecasting accuracy.

Run from the repository root. See research/EXP-05/evaluation/FEATURE_ABLATION_README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep pure metric/perturbation helpers importable without TensorFlow or the repository
# package. Canonical pipeline functions are imported lazily inside main() and checked
# against these names before any model evaluation.
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "intraday_return",
    "hl_range_ratio",
    "close_to_ma20",
    "volume_ratio",
]
WINDOW_SIZE = 30

PRODUCTION_DIR = PROJECT_ROOT / "models_storage_stationary" / "RELIANCE_NS"
EXP05_DIR = (
    PROJECT_ROOT
    / "research"
    / "EXP-05"
    / "extracted"
    / "EXP-05_Model_Comparison_20261021_100324"
)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "RELIANCE_NS_processed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research" / "EXP-05" / "evaluation" / "feature_ablation"
EXPECTED_FOLD3_SAMPLES = 184
DEFAULT_PERMUTATIONS = 25
DEFAULT_SEED = 20261026


class SafeCustomUnpickler(pickle.Unpickler):
    """Load the EXP-05 scaler even when it was pickled from Colab's __main__."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "ReturnTargetScaler":
            from ml.preprocessing.stationary_scaler import ReturnTargetScaler
            return ReturnTargetScaler
        return super().find_class(module, name)


def safe_pickle_load(path: Path) -> Any:
    with path.open("rb") as stream:
        return SafeCustomUnpickler(stream).load()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_compat(model_path: Path) -> Any:
    """Load a .keras model normally; fall back to the canonical architecture + weights.

    The fallback handles Colab Keras archives that serialize a null
    ``quantization_config`` field unsupported by an older local Keras version.
    """
    direct_error_text = ""
    try:
        import tensorflow as tf  # imported lazily so pure helpers can be tested

        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as direct_error:
            direct_error_text = f"{type(direct_error).__name__}: {direct_error}"
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to load model checkpoints. Activate the project's "
            "Python environment and install the repository's TensorFlow dependency."
        ) from exc

    from ml.models.cnn_lstm import build_cnn_lstm_model

    model = build_cnn_lstm_model(
        input_shape=(WINDOW_SIZE, len(FEATURE_COLUMNS)),
        conv_filters=64,
        kernel_size=3,
        lstm_units=50,
        dense_units=32,
        dropout=0.2,
        learning_rate=0.001,
        loss="mse",
    )
    temp_path: Path | None = None
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            candidates = [n for n in archive.namelist() if n.endswith("model.weights.h5")]
            if not candidates:
                raise RuntimeError(
                    f"Could not load {model_path}. Standard Keras load failed ({direct_error_text}) "
                    "and the archive contains no model.weights.h5 fallback."
                )
            with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as temp:
                temp.write(archive.read(candidates[0]))
                temp_path = Path(temp.name)
        model.load_weights(temp_path)
        return model
    except Exception as fallback_error:
        raise RuntimeError(
            f"Could not load model {model_path}. Direct Keras loading failed: "
            f"{direct_error_text}. Weight-only fallback failed: {type(fallback_error).__name__}: "
            f"{fallback_error}"
        ) from fallback_error
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def load_assets(model_name: str) -> Tuple[Any, Any, Any, Path, Dict[str, Any]]:
    if model_name == "production":
        model_dir = PRODUCTION_DIR
        model_path = model_dir / "model_cnn_lstm_stationary.keras"
        feature_scaler_path = model_dir / "stationary_feature_scaler.pkl"
        target_scaler_path = model_dir / "return_target_scaler.pkl"
        metadata_path = model_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    elif model_name == "exp05-fold3":
        model_dir = EXP05_DIR
        model_path = model_dir / "best_cnn_lstm_fold3.keras"
        feature_scaler_path = model_dir / "feature_scaler_3d_fold3.pkl"
        target_scaler_path = model_dir / "return_scaler_fold3.pkl"
        metadata_path = model_dir / "config.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    else:  # defensive; argparse prevents this
        raise ValueError(f"Unsupported model: {model_name}")

    for path in (model_path, feature_scaler_path, target_scaler_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}\n"
                "Check the extracted EXP-05 folder and production artifact paths."
            )

    model = load_model_compat(model_path)
    feature_scaler = safe_pickle_load(feature_scaler_path)
    target_scaler = safe_pickle_load(target_scaler_path)
    if not hasattr(target_scaler, "inverse_transform"):
        raise TypeError(f"Target scaler {target_scaler_path} has no inverse_transform().")

    details = {
        "model_name": model_name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256_file(model_path),
        "feature_scaler_path": str(feature_scaler_path.relative_to(PROJECT_ROOT)),
        "target_scaler_path": str(target_scaler_path.relative_to(PROJECT_ROOT)),
        "features": list(FEATURE_COLUMNS),
        "window_size": WINDOW_SIZE,
        "metadata": metadata,
    }
    return model, feature_scaler, target_scaler, model_path, details


def scale_windows(feature_scaler: Any, windows: np.ndarray) -> np.ndarray:
    """Transform (N, window, feature) arrays through either project or sklearn scaler."""
    windows = np.asarray(windows, dtype=np.float32)
    expected = (windows.shape[0], windows.shape[1], len(FEATURE_COLUMNS))
    if windows.ndim != 3 or windows.shape != expected:
        raise ValueError(f"Expected windows with shape (N, {WINDOW_SIZE}, {len(FEATURE_COLUMNS)}); got {windows.shape}")

    # The production StationaryFeatureScaler exposes .scaler and accepts 3-D arrays.
    if hasattr(feature_scaler, "scaler") and callable(getattr(feature_scaler, "transform", None)):
        transformed = feature_scaler.transform(windows)
    else:
        # EXP-05 serialized a sklearn scaler fitted over flattened window rows.
        flattened = windows.reshape(-1, windows.shape[-1])
        transformed = feature_scaler.transform(flattened).reshape(windows.shape)

    transformed = np.asarray(transformed, dtype=np.float32)
    if transformed.shape != windows.shape:
        raise ValueError(f"Scaler returned unexpected shape {transformed.shape}; expected {windows.shape}.")
    underlying = getattr(feature_scaler, "scaler", feature_scaler)
    if getattr(underlying, "with_mean", True) is False:
        raise ValueError(
            "This runner neutralizes a standardized feature to zero, which represents the "
            "training mean only for a centered scaler (with_mean=True)."
        )
    if not np.isfinite(transformed).all():
        raise ValueError("Scaled input windows contain non-finite values.")
    return transformed


def make_supervised_windows(clean_df: pd.DataFrame, starts: Iterable[int]) -> Tuple[np.ndarray, pd.DataFrame]:
    starts = list(starts)
    if not starts:
        raise ValueError("No supervised windows selected.")
    feature_matrices = [
        clean_df.iloc[start : start + WINDOW_SIZE][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        for start in starts
    ]
    if any(len(item) != WINDOW_SIZE for item in feature_matrices):
        raise ValueError("A selected window is shorter than the required window size.")
    end_rows = [start + WINDOW_SIZE - 1 for start in starts]
    samples = clean_df.iloc[end_rows].copy().reset_index(drop=True)
    samples["input_date"] = pd.to_datetime(samples["Date"]).dt.strftime("%Y-%m-%d")
    samples["target_date"] = samples["target_date"].astype(str).str[:10]
    return np.stack(feature_matrices).astype(np.float32), samples


def predict_returns(model: Any, target_scaler: Any, scaled_windows: np.ndarray) -> np.ndarray:
    scaled_predictions = np.asarray(
        model.predict(scaled_windows, batch_size=64, verbose=0), dtype=np.float64
    ).reshape(-1)
    actual_scale = np.asarray(target_scaler.inverse_transform(scaled_predictions), dtype=np.float64).reshape(-1)
    if len(actual_scale) != len(scaled_windows) or not np.isfinite(actual_scale).all():
        raise ValueError("Model produced invalid predicted returns.")
    return actual_scale


def regression_metrics(actual_close: np.ndarray, predicted_close: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual_close, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_close, dtype=np.float64).reshape(-1)
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("Metric arrays must have equal non-zero length.")
    errors = predicted - actual
    return {
        "mae_inr": float(np.mean(np.abs(errors))),
        "rmse_inr": float(np.sqrt(np.mean(np.square(errors)))),
        "mape_pct": float(np.mean(np.abs(errors / np.maximum(np.abs(actual), 1e-12))) * 100.0),
    }


def neutralize_feature(windows_scaled: np.ndarray, feature_index: int) -> np.ndarray:
    """Return a copy with one standardized channel set to the training-mean reference 0."""
    result = np.array(windows_scaled, copy=True)
    result[:, :, feature_index] = 0.0
    return result


def price_predictions(origin_close: np.ndarray, predicted_returns: np.ndarray) -> np.ndarray:
    close = np.asarray(origin_close, dtype=np.float64).reshape(-1)
    returns = np.asarray(predicted_returns, dtype=np.float64).reshape(-1)
    if len(close) != len(returns):
        raise ValueError("Origin closes and predicted returns must have equal length.")
    return close * (1.0 + returns)


def align_fold3_test(
    clean_df: pd.DataFrame,
    feature_scaler: Any,
    target_scaler: Any,
    model: Any,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    saved_path = EXP05_DIR / "predictions_per_date_fold3.csv"
    if not saved_path.exists():
        raise FileNotFoundError(f"Expected archived predictions not found: {saved_path}")
    saved = pd.read_csv(saved_path)
    if len(saved) != EXPECTED_FOLD3_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_FOLD3_SAMPLES} archived Fold 3 test samples; found {len(saved)}. "
            "Do not silently redefine the historical test block."
        )

    for col in ("input_date", "target_date", "pred_return_cnn"):
        if col not in saved.columns:
            raise ValueError(f"Archived prediction CSV is missing required column '{col}'.")
    saved_input_dates = pd.to_datetime(saved["input_date"]).dt.strftime("%Y-%m-%d").tolist()
    saved_target_dates = pd.to_datetime(saved["target_date"]).dt.strftime("%Y-%m-%d").tolist()

    # Locate the historical rows by exact date pairs. This remains correct if the source CSV
    # later grows to include newer sessions; the runner will not shift the test block forward.
    clean_input_dates = pd.to_datetime(clean_df["Date"]).dt.strftime("%Y-%m-%d").tolist()
    clean_target_dates = clean_df["target_date"].astype(str).str[:10].tolist()
    row_lookup = {
        (input_date, target_date): row_idx
        for row_idx, (input_date, target_date) in enumerate(zip(clean_input_dates, clean_target_dates))
    }
    starts: List[int] = []
    for input_date, target_date in zip(saved_input_dates, saved_target_dates):
        row_idx = row_lookup.get((input_date, target_date))
        if row_idx is None:
            raise ValueError(
                f"Source data does not contain the archived input/target pair {input_date} -> {target_date}."
            )
        start_idx = row_idx - WINDOW_SIZE + 1
        if start_idx < 0:
            raise ValueError(f"Not enough feature history before input date {input_date}.")
        starts.append(start_idx)
    raw_windows, samples = make_supervised_windows(clean_df, starts)
    if saved_input_dates != samples["input_date"].tolist():
        raise ValueError("Reconstructed Fold 3 input dates do not match the archived prediction CSV.")
    if saved_target_dates != samples["target_date"].tolist():
        raise ValueError("Reconstructed Fold 3 target dates do not match the archived prediction CSV.")

    scaled_windows = scale_windows(feature_scaler, raw_windows)
    reproduced_returns = predict_returns(model, target_scaler, scaled_windows)
    max_abs_diff = float(np.max(np.abs(reproduced_returns - saved["pred_return_cnn"].to_numpy(dtype=float))))
    if max_abs_diff > 1e-4:
        raise ValueError(
            f"Checkpoint did not reproduce saved Fold 3 predictions (max abs difference {max_abs_diff:.8g}). "
            "No attribution outputs were written. Check the checkpoint, scalers, feature order, and preprocessing."
        )
    return raw_windows, scaled_windows, samples, max_abs_diff


def summarize_historical_ablation(
    model: Any,
    target_scaler: Any,
    scaled_windows: np.ndarray,
    samples: pd.DataFrame,
    permutations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    actual_close = samples["target_close"].to_numpy(dtype=float)
    origin_close = samples["Close"].to_numpy(dtype=float)
    baseline_returns = predict_returns(model, target_scaler, scaled_windows)
    baseline_prices = price_predictions(origin_close, baseline_returns)
    baseline = regression_metrics(actual_close, baseline_prices)

    rows: List[Dict[str, Any]] = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        altered = neutralize_feature(scaled_windows, idx)
        altered_returns = predict_returns(model, target_scaler, altered)
        altered_prices = price_predictions(origin_close, altered_returns)
        altered_metrics = regression_metrics(actual_close, altered_prices)
        rows.append({
            "method": "mean_neutralization",
            "feature": feature,
            "sample_count": len(samples),
            "baseline_mae_inr": baseline["mae_inr"],
            "altered_mae_inr": altered_metrics["mae_inr"],
            "delta_mae_inr": altered_metrics["mae_inr"] - baseline["mae_inr"],
            "baseline_rmse_inr": baseline["rmse_inr"],
            "altered_rmse_inr": altered_metrics["rmse_inr"],
            "delta_rmse_inr": altered_metrics["rmse_inr"] - baseline["rmse_inr"],
            "baseline_mape_pct": baseline["mape_pct"],
            "altered_mape_pct": altered_metrics["mape_pct"],
            "delta_mape_pct": altered_metrics["mape_pct"] - baseline["mape_pct"],
            "mean_abs_predicted_return_shift": float(np.mean(np.abs(altered_returns - baseline_returns))),
            "direction_flip_rate_pct": float(np.mean(np.sign(altered_returns) != np.sign(baseline_returns)) * 100),
        })

    rng = np.random.default_rng(seed)
    perm_rows: List[Dict[str, Any]] = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        deltas_mae: List[float] = []
        deltas_rmse: List[float] = []
        deltas_mape: List[float] = []
        for _ in range(permutations):
            order = rng.permutation(len(scaled_windows))
            permuted = np.array(scaled_windows, copy=True)
            # Permute whole 30-step feature trajectories across samples, not individual time steps.
            permuted[:, :, idx] = scaled_windows[order, :, idx]
            perm_returns = predict_returns(model, target_scaler, permuted)
            perm_prices = price_predictions(origin_close, perm_returns)
            perm_metrics = regression_metrics(actual_close, perm_prices)
            deltas_mae.append(perm_metrics["mae_inr"] - baseline["mae_inr"])
            deltas_rmse.append(perm_metrics["rmse_inr"] - baseline["rmse_inr"])
            deltas_mape.append(perm_metrics["mape_pct"] - baseline["mape_pct"])

        perm_rows.append({
            "method": "window_permutation",
            "feature": feature,
            "sample_count": len(samples),
            "permutations": permutations,
            "baseline_mae_inr": baseline["mae_inr"],
            "mean_delta_mae_inr": float(np.mean(deltas_mae)),
            "delta_mae_perm_q025_inr": float(np.quantile(deltas_mae, 0.025)),
            "delta_mae_perm_q975_inr": float(np.quantile(deltas_mae, 0.975)),
            "std_delta_mae_inr": float(np.std(deltas_mae, ddof=1)) if permutations > 1 else 0.0,
            "baseline_rmse_inr": baseline["rmse_inr"],
            "mean_delta_rmse_inr": float(np.mean(deltas_rmse)),
            "delta_rmse_perm_q025_inr": float(np.quantile(deltas_rmse, 0.025)),
            "delta_rmse_perm_q975_inr": float(np.quantile(deltas_rmse, 0.975)),
            "baseline_mape_pct": baseline["mape_pct"],
            "mean_delta_mape_pct": float(np.mean(deltas_mape)),
            "delta_mape_perm_q025_pct": float(np.quantile(deltas_mape, 0.025)),
            "delta_mape_perm_q975_pct": float(np.quantile(deltas_mape, 0.975)),
        })

    return pd.DataFrame(rows), pd.DataFrame(perm_rows)


def run_input_sensitivity(
    model: Any,
    feature_scaler: Any,
    target_scaler: Any,
    raw_bars: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from ml.features.stationary import prepare_inference_features

    inference_df, _ = prepare_inference_features(raw_bars)
    valid = inference_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(valid) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} valid feature rows; found {len(valid)}.")
    latest = valid.iloc[-WINDOW_SIZE:]
    raw_window = latest[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[None, :, :]
    scaled_window = scale_windows(feature_scaler, raw_window)
    last_close = float(latest.iloc[-1]["Close"])
    base_return = float(predict_returns(model, target_scaler, scaled_window)[0])
    base_close = float(price_predictions(np.array([last_close]), np.array([base_return]))[0])

    rows: List[Dict[str, Any]] = []
    for idx, feature in enumerate(FEATURE_COLUMNS):
        altered = neutralize_feature(scaled_window, idx)
        new_return = float(predict_returns(model, target_scaler, altered)[0])
        new_close = float(price_predictions(np.array([last_close]), np.array([new_return]))[0])
        rows.append({
            "feature": feature,
            "input_window_start": pd.to_datetime(latest.iloc[0]["Date"]).strftime("%Y-%m-%d"),
            "input_window_end": pd.to_datetime(latest.iloc[-1]["Date"]).strftime("%Y-%m-%d"),
            "baseline_predicted_return": base_return,
            "neutralized_predicted_return": new_return,
            "delta_predicted_return": new_return - base_return,
            "delta_return_basis_points": (new_return - base_return) * 10000.0,
            "baseline_predicted_close_inr": base_close,
            "neutralized_predicted_close_inr": new_close,
            "delta_predicted_close_inr": new_close - base_close,
            "direction_flipped": bool(np.sign(new_return) != np.sign(base_return)),
        })

    info = {
        "input_window_start": pd.to_datetime(latest.iloc[0]["Date"]).strftime("%Y-%m-%d"),
        "input_window_end": pd.to_datetime(latest.iloc[-1]["Date"]).strftime("%Y-%m-%d"),
        "last_close_inr": last_close,
        "baseline_predicted_return": base_return,
        "baseline_predicted_close_inr": base_close,
        "sample_count": 1,
        "interpretation": (
            "Single-window model sensitivity only. This does not quantify out-of-sample feature importance "
            "or prove that a feature improves forecast accuracy. Confirm the input date is the latest completed "
            "market session before presenting it as a current explanation."
        ),
    }
    return pd.DataFrame(rows), info


def write_markdown_report(path: Path, details: Dict[str, Any], mode: str, summary: Dict[str, Any], notes: List[str]) -> None:
    lines = [
        "# StockSense AI — Feature Ablation / Sensitivity Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Mode: `{mode}`",
        f"- Model: `{details['model_name']}`",
        f"- Model file: `{details['model_path']}`",
        f"- Model SHA-256: `{details['model_sha256']}`",
        f"- Window: {WINDOW_SIZE} sessions",
        f"- Features: {', '.join(FEATURE_COLUMNS)}",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- **{key.replace('_', ' ')}:** {value}")
    lines += ["", "## Interpretation and limitations", ""]
    lines.extend([f"- {note}" for note in notes])
    lines += [
        "- A positive delta MAE/RMSE after neutralizing or permuting a feature means the model's error increased on this dataset under that perturbation. It is evidence of model reliance on this sample, not causality.",
        "- Correlated features can substitute for one another, so a small importance score does not prove that a feature is useless.",
        "- Historical Fold 3 has already been inspected. Do not present it as a fresh, untouched test or use this analysis to tune the locked prospective evaluation.",
        "- This script does not retrain models, mutate model files, or change the live model registry.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["production", "exp05-fold3"], required=True)
    parser.add_argument(
        "--mode",
        choices=["fold3-test", "input-sensitivity"],
        default="fold3-test",
        help="fold3-test is allowed only for EXP-05 Fold 3. Use input-sensitivity for one input window.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.permutations < 2:
        raise SystemExit("--permutations must be at least 2 to report a variability interval.")
    if args.mode == "fold3-test" and args.model != "exp05-fold3":
        raise SystemExit(
            "Refusing production-model metrics on the Fold 3 historical test block: the production model "
            "was refit through 2026-10-07, overlapping that block. Use --model production --mode input-sensitivity "
            "for local prediction sensitivity instead."
        )
    if not args.data_path.exists():
        raise SystemExit(f"Data CSV not found: {args.data_path}")

    from ml.features.stationary import (
        FEATURE_COLUMNS as CANONICAL_FEATURE_COLUMNS,
        WINDOW_SIZE as CANONICAL_WINDOW_SIZE,
        compute_stationary_features,
        prepare_inference_features,
    )
    if list(CANONICAL_FEATURE_COLUMNS) != FEATURE_COLUMNS or CANONICAL_WINDOW_SIZE != WINDOW_SIZE:
        raise SystemExit(
            "This runner's feature/window constants differ from ml.features.stationary. "
            "Update the runner to match the canonical production pipeline before proceeding."
        )

    raw_bars = pd.read_csv(args.data_path)
    model, feature_scaler, target_scaler, model_path, details = load_assets(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes: List[str] = [
        "Mean-neutralization sets one standardized feature channel to zero, which corresponds to the training mean for the centered StandardScaler used by this pipeline.",
        "Window permutation shuffles the complete 30-step trajectory of one feature across test samples while leaving other channels unchanged.",
    ]

    model_slug = args.model.replace("-", "_")

    if args.mode == "fold3-test":
        clean_df, _ = compute_stationary_features(raw_bars)
        raw_windows, scaled_windows, samples, max_diff = align_fold3_test(clean_df, feature_scaler, target_scaler, model)
        neutralization_df, permutation_df = summarize_historical_ablation(
            model, target_scaler, scaled_windows, samples, args.permutations, args.seed
        )
        neutralization_path = args.output_dir / "fold3_mean_neutralization.csv"
        permutation_path = args.output_dir / "fold3_window_permutation.csv"
        neutralization_df.sort_values("delta_mae_inr", ascending=False).to_csv(neutralization_path, index=False)
        permutation_df.sort_values("mean_delta_mae_inr", ascending=False).to_csv(permutation_path, index=False)
        base_returns = predict_returns(model, target_scaler, scaled_windows)
        base_metrics = regression_metrics(
            samples["target_close"].to_numpy(dtype=float),
            price_predictions(samples["Close"].to_numpy(dtype=float), base_returns),
        )
        summary = {
            "evaluation_block": f"{samples['target_date'].iloc[0]} to {samples['target_date'].iloc[-1]}",
            "test_sample_count": len(samples),
            "max_abs_prediction_diff_vs_archived_csv": f"{max_diff:.8g}",
            "baseline_mae_inr": f"{base_metrics['mae_inr']:.4f}",
            "baseline_rmse_inr": f"{base_metrics['rmse_inr']:.4f}",
            "baseline_mape_pct": f"{base_metrics['mape_pct']:.4f}",
            "mean_neutralization_csv": str(neutralization_path.relative_to(PROJECT_ROOT)) if neutralization_path.is_relative_to(PROJECT_ROOT) else str(neutralization_path),
            "window_permutation_csv": str(permutation_path.relative_to(PROJECT_ROOT)) if permutation_path.is_relative_to(PROJECT_ROOT) else str(permutation_path),
        }
        notes += [
            "This is a retrospective, post-hoc analysis of the EXP-05 Fold 3 test block. The dates were already examined in earlier project reviews.",
            f"The runner checked the original predictions first; maximum absolute difference from the archived Colab prediction CSV was {max_diff:.8g}.",
        ]
    else:
        attribution_df, summary = run_input_sensitivity(model, feature_scaler, target_scaler, raw_bars)
        attribution_path = args.output_dir / f"{model_slug}_input_sensitivity.csv"
        attribution_df.sort_values("delta_predicted_close_inr", key=lambda s: s.abs(), ascending=False).to_csv(attribution_path, index=False)
        summary["sensitivity_csv"] = str(attribution_path.relative_to(PROJECT_ROOT)) if attribution_path.is_relative_to(PROJECT_ROOT) else str(attribution_path)
        notes += [
            "Input-sensitivity mode uses one 30-session window and has no scored outcomes; it is not a test-set feature-importance estimate.",
            "The source is the CSV supplied via --data-path. Check the reported input-window end date and confirm that its last bar is completed and current before describing it as a live explanation.",
        ]

    details["data_path"] = str(args.data_path)
    details["data_last_date_in_source_csv"] = pd.to_datetime(raw_bars["Date"]).max().strftime("%Y-%m-%d") if "Date" in raw_bars else "unknown"
    report_path = args.output_dir / f"{model_slug}_{args.mode}_report.md"
    write_markdown_report(report_path, details, args.mode, summary, notes)
    details_path = args.output_dir / f"{model_slug}_{args.mode}_metadata.json"
    details_path.write_text(
        json.dumps({"model": details, "mode": args.mode, "summary": summary, "random_seed": args.seed, "permutations": args.permutations}, indent=2, default=str),
        encoding="utf-8",
    )

    print("Feature ablation/sensitivity run complete.")
    print(f"Model: {args.model} | SHA-256: {details['model_sha256']}")
    print(f"Source CSV latest date: {details['data_last_date_in_source_csv']}")
    print(f"Report: {report_path}")
    print(f"Metadata: {details_path}")
    if args.mode == "fold3-test":
        print(f"Mean-neutralization CSV: {args.output_dir / 'fold3_mean_neutralization.csv'}")
        print(f"Window-permutation CSV: {args.output_dir / 'fold3_window_permutation.csv'}")
    else:
        print(f"Sensitivity CSV: {args.output_dir / f'{model_slug}_input_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

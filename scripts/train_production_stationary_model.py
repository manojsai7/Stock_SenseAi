"""
Production Training Pipeline: Canonical Stationary-Feature CNN-LSTM.

Dedicated production training foundation for generating deployment artifacts:
- 7 Causal relative stationary features
- Standardized next-day simple return target
- Two-Stage Protocol:
    Stage A: Chronological train/val split (85/15) for training-duration selection (early stopping).
             Evaluates model vs. Naive Persistence baseline on identical validation dates.
    Stage B: Final deployment refit on 100% of eligible labelled sequences for the selected fixed epochs.
             Refits both scalers on all eligible labelled data.
- Generates metadata.json with Stage A metrics and Stage B deployment details.
- Generates manifest.json with SHA-256 artifact verification.

Does NOT modify existing 6-feature production artifacts in models_storage/.
"""

import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.stationary import (
    TARGET_COLUMN,
    WINDOW_SIZE,
    compute_stationary_features,
    validate_sequence_tensor
)
from ml.preprocessing.stationary_scaler import (
    StationaryFeatureScaler,
    ReturnTargetScaler
)
from ml.models.cnn_lstm import build_cnn_lstm_model

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainProductionStationary")


def compute_sha256(file_path: Path) -> str:
    """Computes SHA256 hex digest of a saved artifact file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_movement_sign(diff: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    """Computes directional sign (-1, 0, 1) with small floating tolerance."""
    diff = np.asarray(diff, dtype=np.float64)
    signs = np.zeros(diff.shape, dtype=int)
    signs[diff > atol] = 1
    signs[diff < -atol] = -1
    return signs


class ProductionStationaryTrainer:
    """
    Orchestrates the two-stage production training pipeline:
    Stage A: Epoch selection & comparative validation against Naive baseline.
    Stage B: Final refit on all eligible labelled data using selected epochs.
    """

    def __init__(
        self,
        symbol: str = "RELIANCE.NS",
        window_size: int = WINDOW_SIZE,
        val_ratio: float = 0.15,
        min_samples: int = 400,
        max_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
        random_seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        data_file: Optional[Union[str, Path]] = None,
        dry_run: bool = False
    ):
        self.symbol = symbol.strip().upper()
        self.clean_symbol = self.symbol.replace(".", "_")
        self.window_size = window_size
        self.val_ratio = val_ratio
        self.min_samples = min_samples
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_seed = random_seed
        self.dry_run = dry_run

        # Dedicated stationary storage directory (isolated from legacy 6-feature storage)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = PROJECT_ROOT / "models_storage_stationary" / self.clean_symbol

        # Locate raw / processed data file
        self.data_file = Path(data_file) if data_file else (
            PROJECT_ROOT / "data" / "processed" / f"{self.clean_symbol}_processed.csv"
        )
        if not data_file and not self.data_file.exists():
            self.data_file = PROJECT_ROOT / "data" / "raw" / f"{self.clean_symbol}.csv"

        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Historical bar data not found for '{self.symbol}'. Checked: {self.data_file}"
            )

    def prepare_data(self) -> Dict[str, Any]:
        """
        Loads bars, computes features, validates history sufficiency,
        and constructs chronological sequences.
        """
        logger.info(f"Loading historical market data for {self.symbol} from {self.data_file}...")
        df_raw = pd.read_csv(self.data_file)

        raw_count = len(df_raw)
        if raw_count < self.min_samples:
            raise ValueError(
                f"Insufficient historical data points for '{self.symbol}': found {raw_count} bars, "
                f"minimum required for robust deep-learning training is {self.min_samples}. "
                f"Ather Energy and recently listed assets cannot be trained until sufficient trading history accumulates."
            )

        clean_df, feature_cols = compute_stationary_features(df_raw)
        usable_rows = len(clean_df)

        if usable_rows < self.window_size + 20:
            raise ValueError(
                f"Only {usable_rows} usable feature rows after preprocessing for window={self.window_size}"
            )

        X, y, prev_closes, actual_closes, input_dates, target_dates = [], [], [], [], [], []

        for i in range(usable_rows - self.window_size + 1):
            window = clean_df[feature_cols].iloc[i : i + self.window_size].to_numpy(dtype=np.float32)
            last_row = clean_df.iloc[i + self.window_size - 1]

            c_t = float(last_row["Close"])
            d_t = str(last_row["Date"])[:10]
            c_target = float(last_row["target_close"])
            d_target = str(last_row["target_date"])[:10]
            ret_t1 = float(last_row[TARGET_COLUMN])

            X.append(window)
            y.append(ret_t1)
            prev_closes.append(c_t)
            actual_closes.append(c_target)
            input_dates.append(d_t)
            target_dates.append(d_target)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)

        validate_sequence_tensor(X_arr, expected_window=self.window_size)

        # Chronological split: (1 - val_ratio) train, val_ratio validation
        N = len(X_arr)
        split_idx = int(N * (1.0 - self.val_ratio))

        X_train, y_train = X_arr[:split_idx], y_arr[:split_idx]
        X_val, y_val = X_arr[split_idx:], y_arr[split_idx:]
        prev_closes_val = np.array(prev_closes[split_idx:], dtype=np.float64)
        actual_closes_val = np.array(actual_closes[split_idx:], dtype=np.float64)

        logger.info(
            f"Prepared {N} sequences of shape (30, {len(feature_cols)}). "
            f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples."
        )

        return {
            "X_all": X_arr,
            "y_all": y_arr,
            "prev_closes_all": np.array(prev_closes, dtype=np.float64),
            "actual_closes_all": np.array(actual_closes, dtype=np.float64),
            "input_dates_all": input_dates,
            "target_dates_all": target_dates,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "prev_closes_val": prev_closes_val,
            "actual_closes_val": actual_closes_val,
            "raw_start_date": str(df_raw["Date"].iloc[0])[:10],
            "raw_end_date": str(df_raw["Date"].iloc[-1])[:10],
            "training_data_start": input_dates[0],
            "training_data_end": input_dates[split_idx - 1],
            "validation_data_end": input_dates[-1],
            "last_target_date": target_dates[-1],
            "feature_columns": feature_cols,
            "total_sequences": N
        }

    def _evaluate_predictions(
        self,
        actual_prices: np.ndarray,
        predicted_prices: np.ndarray,
        prev_prices: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates regression and directional metrics on price level."""
        mae = float(mean_absolute_error(actual_prices, predicted_prices))
        rmse = float(np.sqrt(mean_squared_error(actual_prices, predicted_prices)))
        mape = float(np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100)

        actual_signs = compute_movement_sign(actual_prices - prev_prices)
        pred_signs = compute_movement_sign(predicted_prices - prev_prices)

        acc = float(accuracy_score(actual_signs, pred_signs) * 100)
        try:
            bal_acc = float(balanced_accuracy_score(actual_signs, pred_signs) * 100)
        except Exception:
            bal_acc = acc
        try:
            mcc = float(matthews_corrcoef(actual_signs, pred_signs))
        except Exception:
            mcc = 0.0

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "directional_accuracy": round(acc, 2),
            "balanced_accuracy": round(bal_acc, 2),
            "mcc": round(mcc, 4),
            "sample_count": len(actual_prices)
        }

    def run(self) -> Dict[str, Any]:
        """
        Executes data preparation, Stage A validation training,
        and Stage B final deployment refit.
        """
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        data = self.prepare_data()

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]

        # -------------------------------------------------------------
        # Dry-run validation check
        # -------------------------------------------------------------
        if self.dry_run:
            feat_scaler_dry = StationaryFeatureScaler(data["feature_columns"])
            feat_scaler_dry.fit(X_train)
            ret_scaler_dry = ReturnTargetScaler()
            ret_scaler_dry.fit(y_train)

            logger.info("========== DRY-RUN COMPLETED SUCCESSFULLY ==========")
            logger.info(f"Symbol: {self.symbol}")
            logger.info(f"Raw source range: {data['raw_start_date']} to {data['raw_end_date']}")
            logger.info(f"Eligible sequences: {data['total_sequences']} of shape (30, 7)")
            logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
            logger.info(f"Feature order: {data['feature_columns']}")
            logger.info("All data validation checks and scaler fittings verified without error.")
            return {
                "dry_run": True,
                "symbol": self.symbol,
                "source_data_range": f"{data['raw_start_date']} to {data['raw_end_date']}",
                "total_sequences": data["total_sequences"],
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "feature_shape": list(X_train.shape),
                "target_scaler_mean": ret_scaler_dry.mean_,
                "target_scaler_std": ret_scaler_dry.std_
            }

        # -------------------------------------------------------------
        # STAGE A: Validation & Training-Duration Selection
        # -------------------------------------------------------------
        logger.info(f"========== STAGE A: Training-Duration Selection for {self.symbol} ==========")
        feat_scaler_a = StationaryFeatureScaler(data["feature_columns"])
        feat_scaler_a.fit(X_train)
        X_train_a = feat_scaler_a.transform(X_train)
        X_val_a = feat_scaler_a.transform(X_val)

        ret_scaler_a = ReturnTargetScaler()
        ret_scaler_a.fit(y_train)
        y_train_a = ret_scaler_a.transform(y_train)
        y_val_a = ret_scaler_a.transform(y_val)

        model_a = build_cnn_lstm_model(
            input_shape=(self.window_size, len(data["feature_columns"])),
            conv_filters=64,
            kernel_size=3,
            lstm_units=50,
            dense_units=32,
            dropout=0.2,
            learning_rate=self.learning_rate,
            loss="mse"
        )

        early_stop_a = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )

        history_a = model_a.fit(
            X_train_a,
            y_train_a,
            validation_data=(X_val_a, y_val_a),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop_a],
            verbose=1
        )

        val_losses = history_a.history.get("val_loss", [])
        best_epoch = int(np.argmin(val_losses) + 1)
        best_val_loss = float(val_losses[best_epoch - 1])
        logger.info(f"[STAGE A] Selected best epoch: {best_epoch}/{len(val_losses)} (val_loss={best_val_loss:.6f})")

        # Evaluate Stage A model on validation set
        val_pred_scaled = model_a.predict(X_val_a, verbose=0).flatten()
        val_pred_returns = ret_scaler_a.inverse_transform(val_pred_scaled)

        prev_closes_val = data["prev_closes_val"]
        actual_closes_val = data["actual_closes_val"]
        val_pred_closes = prev_closes_val * (1.0 + val_pred_returns)

        stage_a_model_metrics = self._evaluate_predictions(
            actual_closes_val, val_pred_closes, prev_closes_val
        )

        # Evaluate Naive Persistence on EXACTLY the same validation dates
        naive_val_closes = np.copy(prev_closes_val)  # y_hat[t+1] = Close[t]
        stage_a_naive_metrics = self._evaluate_predictions(
            actual_closes_val, naive_val_closes, prev_closes_val
        )

        logger.info(
            f"[STAGE A Comparison] Model MAE: INR {stage_a_model_metrics['mae']} vs. "
            f"Naive MAE: INR {stage_a_naive_metrics['mae']} | "
            f"Model DirAcc: {stage_a_model_metrics['directional_accuracy']}% vs. "
            f"Naive DirAcc: {stage_a_naive_metrics['directional_accuracy']}%"
        )

        # -------------------------------------------------------------
        # STAGE B: Final Deployment Refit on 100% of Data
        # -------------------------------------------------------------
        logger.info(f"========== STAGE B: Final Deployment Refit for {self.symbol} ==========")
        X_all = data["X_all"]
        y_all = data["y_all"]

        # Refit scalers on all eligible labelled history
        final_feat_scaler = StationaryFeatureScaler(data["feature_columns"])
        final_feat_scaler.fit(X_all)
        X_all_scaled = final_feat_scaler.transform(X_all)

        final_ret_scaler = ReturnTargetScaler()
        final_ret_scaler.fit(y_all)
        y_all_scaled = final_ret_scaler.transform(y_all)

        # Fresh model instance
        final_model = build_cnn_lstm_model(
            input_shape=(self.window_size, len(data["feature_columns"])),
            conv_filters=64,
            kernel_size=3,
            lstm_units=50,
            dense_units=32,
            dropout=0.2,
            learning_rate=self.learning_rate,
            loss="mse"
        )

        logger.info(
            f"[STAGE B] Refitting fresh model on all {len(X_all)} sequences "
            f"for {best_epoch} fixed epochs..."
        )
        final_model.fit(
            X_all_scaled,
            y_all_scaled,
            epochs=best_epoch,
            batch_size=self.batch_size,
            verbose=1
        )

        # -------------------------------------------------------------
        # Save Final Deployment Artifacts
        # -------------------------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.output_dir / "model_cnn_lstm_stationary.keras"
        feat_scaler_path = self.output_dir / "stationary_feature_scaler.pkl"
        target_scaler_path = self.output_dir / "return_target_scaler.pkl"
        metadata_path = self.output_dir / "metadata.json"
        manifest_path = self.output_dir / "manifest.json"

        final_model.save(str(model_path))
        final_feat_scaler.save(feat_scaler_path)
        final_ret_scaler.save(target_scaler_path)

        metadata: Dict[str, Any] = {
            "pipeline_version": "stationary-return-v1",
            "model_version": "cnn_lstm_stationary_v1",
            "symbol": self.symbol,
            "architecture": "CNN-LSTM (Conv1D + LSTM)",
            "window_size": self.window_size,
            "target": "next_day_return",
            "features": list(data["feature_columns"]),
            "model_file": "model_cnn_lstm_stationary.keras",
            "feature_scaler": "stationary_feature_scaler.pkl",
            "target_scaler": "return_target_scaler.pkl",
            "trained_at": datetime.now().isoformat(),
            "source_data_range": {
                "start_date": data["raw_start_date"],
                "last_completed_session": data["raw_end_date"],
                "last_target_date": data["last_target_date"]
            },
            "stage_a_training_duration_selection": {
                "best_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 6),
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
                "validation_data_end": data["validation_data_end"],
                "validation_metrics": stage_a_model_metrics,
                "naive_persistence_metrics": stage_a_naive_metrics
            },
            "stage_b_final_deployment_refit": {
                "fixed_epochs": best_epoch,
                "total_training_samples": len(X_all),
                "training_data_end": data["raw_end_date"],
                "note": "Refit on 100% eligible labelled sequences. No validation callback used during Stage B."
            },
            "disclaimer": (
                "Validation metrics reflect Stage A early-stopping selection on the chronological validation fold. "
                "They are documented for model selection evidence, not as an untouched future test. "
                "Statistical estimate for academic research; not financial advice."
            )
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Compute SHA256 checksums only after all files are fully written
        manifest = {
            "symbol": self.symbol,
            "pipeline_version": "stationary-return-v1",
            "created_at": datetime.now().isoformat(),
            "artifacts": {
                "model": {
                    "file": "model_cnn_lstm_stationary.keras",
                    "sha256": compute_sha256(model_path),
                    "size_bytes": model_path.stat().st_size
                },
                "feature_scaler": {
                    "file": "stationary_feature_scaler.pkl",
                    "sha256": compute_sha256(feat_scaler_path),
                    "size_bytes": feat_scaler_path.stat().st_size
                },
                "target_scaler": {
                    "file": "return_target_scaler.pkl",
                    "sha256": compute_sha256(target_scaler_path),
                    "size_bytes": target_scaler_path.stat().st_size
                },
                "metadata": {
                    "file": "metadata.json",
                    "sha256": compute_sha256(metadata_path),
                    "size_bytes": metadata_path.stat().st_size
                }
            }
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Successfully generated final deployment artifacts for {self.symbol} at {self.output_dir}:")
        logger.info(f"  - Model: {model_path} ({model_path.stat().st_size} bytes)")
        logger.info(f"  - Feature Scaler: {feat_scaler_path} ({feat_scaler_path.stat().st_size} bytes)")
        logger.info(f"  - Target Scaler: {target_scaler_path} ({target_scaler_path.stat().st_size} bytes)")
        logger.info(f"  - Metadata: {metadata_path} ({metadata_path.stat().st_size} bytes)")
        logger.info(f"  - Manifest: {manifest_path} ({manifest_path.stat().st_size} bytes)")

        return {
            "symbol": self.symbol,
            "output_dir": str(self.output_dir),
            "stage_a_best_epoch": best_epoch,
            "stage_a_val_metrics": stage_a_model_metrics,
            "stage_a_naive_metrics": stage_a_naive_metrics,
            "stage_b_samples": len(X_all),
            "manifest": manifest
        }


def main():
    parser = argparse.ArgumentParser(description="Train production stationary-feature CNN-LSTM model.")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Stock symbol")
    parser.add_argument("--dry-run", action="store_true", help="Validate data preparation without training")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs for Stage A")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--min-samples", type=int, default=400, help="Minimum required historical bars")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    trainer = ProductionStationaryTrainer(
        symbol=args.symbol,
        val_ratio=args.val_ratio,
        min_samples=args.min_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )

    try:
        res = trainer.run()
        print(json.dumps(res, indent=2))
    except Exception as e:
        logger.error(f"Production training pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

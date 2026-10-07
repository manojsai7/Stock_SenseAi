import os
import json
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from ml.data.ingestion import DataIngestionService, DEFAULT_TICKERS
from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS, TARGET_COLUMN
from ml.preprocessing.scaler import StockDataScaler
from ml.preprocessing.sequences import SequenceGenerator
from ml.models.baseline import NaivePersistenceModel
from ml.models.lstm import build_lstm_model
from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.evaluation.metrics import calculate_metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
PROCESSED_DATA_DIR = Path("data/processed")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"

class ExperimentRunner:
    """
    Academic Experimentation & Evaluation Framework:
    Conducts systematic ablation studies, parameter tests, and baseline comparisons.
    Ensures strict chronological isolation (Train / Validation / Test) and zero data leakage.
    Logs structured outputs to experiments/results.csv.
    """

    def __init__(self):
        self.ingestion = DataIngestionService()
        self.fe = TechnicalFeatureExtractor(ma_window=20)
        self._init_csv()

    def _init_csv(self):
        if not RESULTS_CSV.exists():
            with open(RESULTS_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "experiment_id",
                    "stock",
                    "model",
                    "window_size",
                    "features",
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "mae",
                    "rmse",
                    "mape",
                    "directional_accuracy",
                    "timestamp"
                ])

    def log_result(self, row: dict):
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("experiment_id"),
                row.get("stock"),
                row.get("model"),
                row.get("window_size"),
                row.get("features"),
                row.get("epochs"),
                row.get("batch_size"),
                row.get("learning_rate"),
                row.get("mae"),
                row.get("rmse"),
                row.get("mape"),
                row.get("directional_accuracy"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def prepare_stock_splits(self, symbol: str, period: str = "5y"):
        raw_df = self.ingestion.get_stock_data(symbol, period=period)
        featured_df = self.fe.prepare_training_features(raw_df)

        # Save processed dataset for reproducibility
        clean_sym = symbol.replace(".", "_")
        proc_file = PROCESSED_DATA_DIR / f"{clean_sym}_processed.csv"
        featured_df.to_csv(proc_file, index=False)

        total = len(featured_df)
        train_idx = int(total * 0.70)
        val_idx = int(total * 0.85)

        train_df = featured_df.iloc[:train_idx].copy()
        val_df = featured_df.iloc[train_idx:val_idx].copy()
        test_df = featured_df.iloc[val_idx:].copy()

        # Scaler fitted EXCLUSIVELY on train_df
        scaler = StockDataScaler()
        scaler.fit(train_df)

        scaled_features = scaler.transform_features_only(featured_df)
        scaled_targets = scaler.target_scaler.transform(featured_df[[TARGET_COLUMN]].values)

        return {
            "featured_df": featured_df,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "scaler": scaler,
            "scaled_features": scaled_features,
            "scaled_targets": scaled_targets,
        }

    def run_suite(self, symbols: List[str] = DEFAULT_TICKERS):
        logger.info("Starting Systematic Experimentation Suite across all 5 benchmark stocks...")

        for sym in symbols:
            logger.info(f"=== Running Experiments for {sym} ===")
            splits = self.prepare_stock_splits(sym)

            featured_df = splits["featured_df"]
            scaler = splits["scaler"]

            # -------------------------------------------------------------
            # Exp 001: Naive Persistence Baseline (window=30)
            # -------------------------------------------------------------
            seq_gen_30 = SequenceGenerator(window_size=30)
            (X_tr30, y_tr30), (X_val30, y_val30), (X_te30, y_te30), test_indices = seq_gen_30.create_split_sequences(
                splits["scaled_features"], splits["scaled_targets"], splits["train_idx"], splits["val_idx"]
            )

            # Actual test targets in native ₹
            actual_test_prices = scaler.inverse_transform_target(y_te30)

            # In featured_df, target index idx has previous close at idx
            # (since Next_Close at row idx is the close of row idx + 1)
            test_prev_close = featured_df["Close"].iloc[test_indices].values
            prior_indices = [max(0, i - 1) for i in test_indices]
            prior_trends = test_prev_close - featured_df["Close"].iloc[prior_indices].values

            # Naive Model evaluation (predicts previous close)
            naive_pred_prices = test_prev_close
            naive_metrics = calculate_metrics(actual_test_prices, naive_pred_prices, test_prev_close, prior_trends)
            self.log_result({
                "experiment_id": "exp_001_baseline",
                "stock": sym,
                "model": "Naive Persistence",
                "window_size": 30,
                "features": "Close",
                "epochs": 0,
                "batch_size": 0,
                "learning_rate": 0,
                "mae": naive_metrics["mae"],
                "rmse": naive_metrics["rmse"],
                "mape": naive_metrics["mape"],
                "directional_accuracy": naive_metrics["directional_accuracy"]
            })
            logger.info(f"[{sym}] Naive Metrics: {naive_metrics}")

            # -------------------------------------------------------------
            # Exp 002: Standard LSTM (window=30, units=50)
            # -------------------------------------------------------------
            input_shape_30 = (30, len(FEATURE_COLUMNS))
            lstm = build_lstm_model(input_shape=input_shape_30, units=50, learning_rate=0.001)
            lstm.fit(
                X_tr30, y_tr30,
                validation_data=(X_val30, y_val30),
                epochs=35,
                batch_size=32,
                callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
                verbose=0
            )
            lstm_preds = scaler.inverse_transform_target(lstm.predict(X_te30, verbose=0).flatten())
            lstm_metrics = calculate_metrics(actual_test_prices, lstm_preds, test_prev_close)
            self.log_result({
                "experiment_id": "exp_002_lstm",
                "stock": sym,
                "model": "Standard LSTM (TensorFlow)",
                "window_size": 30,
                "features": ",".join(FEATURE_COLUMNS),
                "epochs": 35,
                "batch_size": 32,
                "learning_rate": 0.001,
                "mae": lstm_metrics["mae"],
                "rmse": lstm_metrics["rmse"],
                "mape": lstm_metrics["mape"],
                "directional_accuracy": lstm_metrics["directional_accuracy"]
            })
            logger.info(f"[{sym}] LSTM Metrics: {lstm_metrics}")

            # -------------------------------------------------------------
            # Exp 003: Proposed CNN-LSTM Hybrid (window=30, conv=64, lstm=50)
            # -------------------------------------------------------------
            cnn_lstm = build_cnn_lstm_model(input_shape=input_shape_30, conv_filters=64, lstm_units=50, learning_rate=0.001)
            cnn_lstm.fit(
                X_tr30, y_tr30,
                validation_data=(X_val30, y_val30),
                epochs=35,
                batch_size=32,
                callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
                verbose=0
            )
            cnn_preds = scaler.inverse_transform_target(cnn_lstm.predict(X_te30, verbose=0).flatten())
            cnn_metrics = calculate_metrics(actual_test_prices, cnn_preds, test_prev_close)
            self.log_result({
                "experiment_id": "exp_003_cnn_lstm",
                "stock": sym,
                "model": "CNN-LSTM Hybrid (TensorFlow)",
                "window_size": 30,
                "features": ",".join(FEATURE_COLUMNS),
                "epochs": 35,
                "batch_size": 32,
                "learning_rate": 0.001,
                "mae": cnn_metrics["mae"],
                "rmse": cnn_metrics["rmse"],
                "mape": cnn_metrics["mape"],
                "directional_accuracy": cnn_metrics["directional_accuracy"]
            })
            logger.info(f"[{sym}] CNN-LSTM Metrics: {cnn_metrics}")

            # -------------------------------------------------------------
            # Exp 004: Window Size Ablation (window=20)
            # -------------------------------------------------------------
            seq_gen_20 = SequenceGenerator(window_size=20)
            (X_tr20, y_tr20), (X_val20, y_val20), (X_te20, y_te20), test_indices20 = seq_gen_20.create_split_sequences(
                splits["scaled_features"], splits["scaled_targets"], splits["train_idx"], splits["val_idx"]
            )
            cnn_20 = build_cnn_lstm_model(input_shape=(20, len(FEATURE_COLUMNS)), conv_filters=64, lstm_units=50)
            cnn_20.fit(X_tr20, y_tr20, validation_data=(X_val20, y_val20), epochs=25, batch_size=32, verbose=0)
            cnn20_preds = scaler.inverse_transform_target(cnn_20.predict(X_te20, verbose=0).flatten())
            m20 = calculate_metrics(actual_test_prices, cnn20_preds, test_prev_close)
            self.log_result({
                "experiment_id": "exp_004_window20",
                "stock": sym,
                "model": "CNN-LSTM (Window=20)",
                "window_size": 20,
                "features": ",".join(FEATURE_COLUMNS),
                "epochs": 25,
                "batch_size": 32,
                "learning_rate": 0.001,
                "mae": m20["mae"],
                "rmse": m20["rmse"],
                "mape": m20["mape"],
                "directional_accuracy": m20["directional_accuracy"]
            })

            # -------------------------------------------------------------
            # Exp 005: Window Size Ablation (window=60)
            # -------------------------------------------------------------
            seq_gen_60 = SequenceGenerator(window_size=60)
            (X_tr60, y_tr60), (X_val60, y_val60), (X_te60, y_te60), test_indices60 = seq_gen_60.create_split_sequences(
                splits["scaled_features"], splits["scaled_targets"], splits["train_idx"], splits["val_idx"]
            )
            cnn_60 = build_cnn_lstm_model(input_shape=(60, len(FEATURE_COLUMNS)), conv_filters=64, lstm_units=50)
            cnn_60.fit(X_tr60, y_tr60, validation_data=(X_val60, y_val60), epochs=25, batch_size=32, verbose=0)
            cnn60_preds = scaler.inverse_transform_target(cnn_60.predict(X_te60, verbose=0).flatten())
            m60 = calculate_metrics(actual_test_prices, cnn60_preds, test_prev_close)
            self.log_result({
                "experiment_id": "exp_005_window60",
                "stock": sym,
                "model": "CNN-LSTM (Window=60)",
                "window_size": 60,
                "features": ",".join(FEATURE_COLUMNS),
                "epochs": 25,
                "batch_size": 32,
                "learning_rate": 0.001,
                "mae": m60["mae"],
                "rmse": m60["rmse"],
                "mape": m60["mape"],
                "directional_accuracy": m60["directional_accuracy"]
            })

            # Save production model artifacts for this stock
            clean_sym = sym.replace(".", "_")
            stock_dir = Path("models_storage") / clean_sym
            stock_dir.mkdir(parents=True, exist_ok=True)
            cnn_lstm.save(stock_dir / "model_cnn_lstm.keras")
            lstm.save(stock_dir / "model_lstm.keras")
            scaler.save(stock_dir / "scaler.pkl")

            # Trajectory for dashboard
            test_dates = featured_df["Date"].iloc[test_indices].dt.strftime("%Y-%m-%d").tolist()
            sample_slice = slice(-60, None) if len(actual_test_prices) > 60 else slice(0, None)
            trajectory = []
            for d, act, n_p, l_p, c_p in zip(
                test_dates[sample_slice],
                actual_test_prices[sample_slice],
                naive_pred_prices[sample_slice],
                lstm_preds[sample_slice],
                cnn_preds[sample_slice]
            ):
                trajectory.append({
                    "date": d,
                    "actual": round(float(act), 2),
                    "naive": round(float(n_p), 2),
                    "lstm": round(float(l_p), 2),
                    "cnn_lstm": round(float(c_p), 2)
                })

            metadata = {
                "symbol": sym,
                "trained_at": datetime.now().isoformat(),
                "deep_learning_framework": "TensorFlow 2.17+ with Keras API",
                "model_version": "v2.0 Defensible",
                "window_size": 30,
                "prediction_horizon": "Next Trading Day",
                "features": FEATURE_COLUMNS,
                "architecture": {
                    "framework": "TensorFlow / Keras",
                    "layers": [
                        "Input(shape=(30, 6))",
                        "Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')",
                        "MaxPooling1D(pool_size=2)",
                        "LSTM(units=50, return_sequences=False)",
                        "Dropout(rate=0.2)",
                        "Dense(units=32)",
                        "LeakyReLU(negative_slope=0.1)",
                        "Dense(units=1, activation='linear')"
                    ]
                },
                "dataset_summary": {
                    "total_samples": len(featured_df),
                    "train_samples": len(splits["train_df"]),
                    "val_samples": len(splits["val_df"]),
                    "test_samples": len(splits["test_df"])
                },
                "metrics_comparison": {
                    "naive": naive_metrics,
                    "lstm": lstm_metrics,
                    "cnn_lstm": cnn_metrics
                },
                "test_trajectory": trajectory
            }
            with open(stock_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Completed experiment suite for {sym}. Saved models and metadata.")

        logger.info(f"All experiments completed! Results logged to {RESULTS_CSV}")

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_suite()

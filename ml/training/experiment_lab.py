import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import sklearn

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from ml.models.baseline import NaivePersistenceModel
from ml.models.lstm import build_lstm_model
from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.evaluation.metrics import calculate_metrics
from ml.preprocessing.scaler import StockDataScaler
from ml.analysis.model_inspector import inspect_model_architecture
from ml.analysis.cnn_instrumentation import CNNInstrumentation

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MODELS_STORAGE_DIR = PROJECT_ROOT / "models_storage"


class DeepLearningExperimentLab:
    """
    Controlled Deep Learning Experiment Laboratory for StockSense AI.
    
    Orchestrates:
    1. Baseline 1: Naive Persistence Model evaluation
    2. Baseline 2: Pure LSTM controlled comparison
    3. Hybrid Model: CNN-LSTM training with callbacks and early stopping
    4. Mathematical Model Inspection & CNN filter/activation instrumentation
    5. Test set evaluation with actual vs predicted curves and error analysis
    """

    def __init__(
        self,
        symbol: str = "RELIANCE.NS",
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        conv_filters: int = 64,
        kernel_size: int = 3,
        lstm_units: int = 50,
        dense_units: int = 32,
        dropout: float = 0.2,
        loss: str = "mse",
        random_seed: int = 42,
        run_id: Optional[str] = None
    ):
        self.symbol = symbol
        self.clean_symbol = symbol.replace(".", "_")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.loss = loss
        self.random_seed = random_seed

        # Set reproducible seeds
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"run_{self.clean_symbol}_{timestamp}"
        self.run_dir = RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.seq_dir = SEQUENCES_DIR / self.clean_symbol
        if not self.seq_dir.exists():
            raise FileNotFoundError(f"Sequences directory not found for {self.symbol} at {self.seq_dir}")

        self.scaler = StockDataScaler.load(self.seq_dir / "scaler.pkl")
        self.cnn_instrumentor = CNNInstrumentation()

    def load_tensors(self) -> Dict[str, np.ndarray]:
        """Loads prepared supervised tensors from sequences.npz."""
        npz_file = self.seq_dir / "sequences.npz"
        data = np.load(npz_file, allow_pickle=True)
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "dates_train": data["dates_train"],
            "dates_val": data["dates_val"],
            "dates_test": data["dates_test"],
            "prev_close_test": data["prev_close_test"],
            "actual_target_test": data["actual_target_test"]
        }

    def get_test_window_dates(self, sample_index: int = 0) -> List[str]:
        """Maps a test sequence back to its exact 30 chronological trading dates."""
        proc_csv = PROJECT_ROOT / "data" / "processed" / f"{self.clean_symbol}_processed.csv"
        if proc_csv.exists():
            df = pd.read_csv(proc_csv)
            tensors = self.load_tensors()
            target_date = str(tensors["dates_test"][sample_index])
            match_idx = df.index[df["Date"] == target_date].tolist()
            if match_idx:
                k = match_idx[0]
                start_k = max(0, k - 29)
                return df["Date"].iloc[start_k : k + 1].tolist()
        return [f"Day_{i+1}" for i in range(30)]

    def evaluate_naive_baseline(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluates Baseline 1: Naive Persistence (prediction(t+1) = Close(t))."""
        logger.info(f"[{self.symbol}] Evaluating Baseline 1: Naive Persistence...")
        X_test = tensors["X_test"]
        actual_test_prices = tensors["actual_target_test"]
        prev_close_prices = tensors["prev_close_test"]

        # Close column index in [Open, High, Low, Close, Volume, MA20] is 3
        naive = NaivePersistenceModel(close_col_index=3)
        naive_pred_sc = naive.predict(X_test)
        naive_pred_prices = self.scaler.target_scaler.inverse_transform(naive_pred_sc.reshape(-1, 1)).flatten()

        metrics = calculate_metrics(
            y_true=actual_test_prices,
            y_pred=naive_pred_prices,
            previous_close=prev_close_prices
        )

        return {
            "model_name": "Naive_Persistence",
            "metrics": metrics,
            "predictions": naive_pred_prices
        }

    def train_pure_lstm(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Trains and evaluates Baseline 2: Pure LSTM Model."""
        logger.info(f"[{self.symbol}] Training Baseline 2: Pure LSTM Model ({self.epochs} epochs)...")
        input_shape = (tensors["X_train"].shape[1], tensors["X_train"].shape[2])

        lstm_model = build_lstm_model(
            input_shape=input_shape,
            units=self.lstm_units,
            dense_units=self.dense_units,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            loss=self.loss
        )

        lstm_ckpt_path = self.run_dir / "best_lstm_model.keras"
        callbacks = [
            ModelCheckpoint(filepath=str(lstm_ckpt_path), monitor="val_loss", save_best_only=True, verbose=0),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
        ]

        history = lstm_model.fit(
            tensors["X_train"],
            tensors["y_train"],
            validation_data=(tensors["X_val"], tensors["y_val"]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            shuffle=False,  # Preserve temporal sequence order
            verbose=1
        )

        # Test evaluation
        y_pred_sc = lstm_model.predict(tensors["X_test"], verbose=0).flatten()
        pred_prices = self.scaler.inverse_transform_target(y_pred_sc)

        metrics = calculate_metrics(
            y_true=tensors["actual_target_test"],
            y_pred=pred_prices,
            previous_close=tensors["prev_close_test"]
        )

        return {
            "model_name": "Pure_LSTM",
            "model": lstm_model,
            "metrics": metrics,
            "predictions": pred_prices,
            "history": history.history
        }

    def train_cnn_lstm(self, tensors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Trains and evaluates the Hybrid CNN-LSTM Model."""
        logger.info(f"[{self.symbol}] Training Hybrid CNN-LSTM Model ({self.epochs} epochs)...")
        input_shape = (tensors["X_train"].shape[1], tensors["X_train"].shape[2])

        model = build_cnn_lstm_model(
            input_shape=input_shape,
            conv_filters=self.conv_filters,
            kernel_size=self.kernel_size,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            loss=self.loss
        )

        best_model_path = self.run_dir / "best_cnn_lstm.keras"
        final_model_path = self.run_dir / "final_cnn_lstm.keras"
        history_csv_path = self.run_dir / "training_history.csv"

        callbacks = [
            ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            CSVLogger(filename=str(history_csv_path))
        ]

        # Fit on training data with explicit validation data
        history = model.fit(
            tensors["X_train"],
            tensors["y_train"],
            validation_data=(tensors["X_val"], tensors["y_val"]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            shuffle=False,  # Explicitly do NOT shuffle temporal order
            verbose=1
        )

        model.save(final_model_path)

        # -------------------------------------------------------------------
        # Render Loss Curves (Train Loss vs Val Loss)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        epochs_arr = range(1, len(history.history["loss"]) + 1)
        ax.plot(epochs_arr, history.history["loss"], "o-", color="#2563eb", linewidth=1.8, label="Train Loss (MSE)")
        ax.plot(epochs_arr, history.history["val_loss"], "s--", color="#ef4444", linewidth=1.8, label="Val Loss (MSE)")
        ax.set_title(f"{self.symbol} — CNN-LSTM Training & Validation Loss", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Mean Squared Error (Scaled)", fontsize=10)
        ax.legend(loc="upper right")
        fig.tight_layout()
        loss_plot_path = self.run_dir / "loss_curve.png"
        fig.savefig(loss_plot_path)
        plt.close(fig)

        # -------------------------------------------------------------------
        # Predict on Test Set & Calculate Evaluation Metrics
        # -------------------------------------------------------------------
        y_pred_sc = model.predict(tensors["X_test"], verbose=0).flatten()
        pred_prices = self.scaler.inverse_transform_target(y_pred_sc)
        actual_prices = tensors["actual_target_test"]
        prev_closes = tensors["prev_close_test"]
        dates_test = tensors["dates_test"]

        metrics = calculate_metrics(
            y_true=actual_prices,
            y_pred=pred_prices,
            previous_close=prev_closes
        )

        # -------------------------------------------------------------------
        # Actual vs Predicted CSV & Error Analysis
        # -------------------------------------------------------------------
        errors = pred_prices - actual_prices
        abs_errors = np.abs(errors)

        actual_dirs = np.where(actual_prices >= prev_closes, "UP", "DOWN")
        pred_dirs = np.where(pred_prices >= prev_closes, "UP", "DOWN")
        dir_corrects = (actual_dirs == pred_dirs)

        df_eval = pd.DataFrame({
            "date": dates_test,
            "actual_close": actual_prices,
            "predicted_close": np.round(pred_prices, 2),
            "previous_close": prev_closes,
            "error": np.round(errors, 2),
            "absolute_error": np.round(abs_errors, 2),
            "direction_actual": actual_dirs,
            "direction_predicted": pred_dirs,
            "direction_correct": dir_corrects
        })

        actual_vs_pred_csv = self.run_dir / "actual_vs_predicted.csv"
        df_eval.to_csv(actual_vs_pred_csv, index=False)

        # Error analysis: top 20 largest absolute errors
        df_error = df_eval.sort_values("absolute_error", ascending=False).reset_index(drop=True)
        error_analysis_csv = self.run_dir / "error_analysis.csv"
        df_error.to_csv(error_analysis_csv, index=False)

        # -------------------------------------------------------------------
        # Actual vs Predicted Trajectory Plot
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        x_indices = range(len(actual_prices))
        ax.plot(x_indices, actual_prices, color="#1e293b", linewidth=1.8, label="Actual Close (INR)")
        ax.plot(x_indices, pred_prices, color="#10b981", linewidth=1.5, linestyle="--", label="CNN-LSTM Prediction")
        ax.set_title(f"{self.symbol} — Test Horizon: Actual vs. Predicted Close Price", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Test Session Index", fontsize=10)
        ax.set_ylabel("Price (INR)", fontsize=10)
        ax.legend(loc="upper left")
        fig.tight_layout()
        pred_plot_path = self.run_dir / "actual_vs_predicted.png"
        fig.savefig(pred_plot_path)
        plt.close(fig)

        return {
            "model_name": "CNN_LSTM",
            "model": model,
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
            "metrics": metrics,
            "predictions": pred_prices,
            "history": history.history,
            "history_csv": str(history_csv_path),
            "loss_plot": str(loss_plot_path),
            "actual_vs_pred_plot": str(pred_plot_path),
            "eval_df": df_eval,
            "top_20_errors": df_error.head(20).to_dict(orient="records")
        }

    def run_sanity_experiment(self) -> Dict[str, Any]:
        """
        Executes a complete controlled experiment on the configured stock:
        1. Model Inspection Tool
        2. CNN Mathematical Instrumentation (Weights & Activations)
        3. Baseline 1 (Naive Persistence)
        4. Baseline 2 (Pure LSTM)
        5. Hybrid CNN-LSTM (3-5 epochs)
        6. Comparative Evaluation Metrics
        """
        logger.info(f"========== Starting Experiment Lab for {self.symbol} ==========")
        tensors = self.load_tensors()

        # 1. Model Inspection Tool
        cnn_lstm_temp = build_cnn_lstm_model(input_shape=(30, 6))
        inspection = inspect_model_architecture(cnn_lstm_temp, symbol=self.symbol)

        # 2. Extract initial weights and activations
        test_window_dates = self.get_test_window_dates(sample_index=0)
        self.cnn_instrumentor.extract_conv1d_weights(cnn_lstm_temp, symbol=self.symbol)
        act_info = self.cnn_instrumentor.extract_activations(
            model=cnn_lstm_temp,
            X_sample=tensors["X_test"][0],
            symbol=self.symbol,
            input_dates=test_window_dates,
            sample_index=0
        )

        # 3. Baseline 1: Naive Persistence
        naive_result = self.evaluate_naive_baseline(tensors)

        # 4. Baseline 2: Pure LSTM
        lstm_result = self.train_pure_lstm(tensors)

        # 5. Hybrid CNN-LSTM
        cnn_lstm_result = self.train_cnn_lstm(tensors)

        # 6. Extract learned Conv1D kernels and activations post-training
        self.cnn_instrumentor.extract_conv1d_weights(cnn_lstm_result["model"], symbol=self.symbol)
        trained_act_info = self.cnn_instrumentor.extract_activations(
            model=cnn_lstm_result["model"],
            X_sample=tensors["X_test"][0],
            symbol=self.symbol,
            input_dates=test_window_dates,
            sample_index=0
        )

        # 7. Experiment Configuration & Metadata Archive
        config = {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "clean_symbol": self.clean_symbol,
            "window_size": 30,
            "features": ["Open", "High", "Low", "Close", "Volume", "MA20"],
            "target": "Close(t+1)",
            "conv_filters": self.conv_filters,
            "kernel_size": self.kernel_size,
            "pool_size": 2,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "optimizer": "adam",
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "random_seed": self.random_seed,
                "python_version": sys.version,
                "tensorflow_version": tf.__version__,
                "keras_version": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "3.11.2",
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
                "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")]
            },
            "timestamp": datetime.now().astimezone().isoformat()
        }

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Comparative Metrics Matrix
        metrics_comparison = {
            "symbol": self.symbol,
            "run_id": self.run_id,
            "epochs": self.epochs,
            "naive_persistence": naive_result["metrics"],
            "pure_lstm": lstm_result["metrics"],
            "cnn_lstm": cnn_lstm_result["metrics"]
        }

        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics_comparison, f, indent=2)

        with open(RESULTS_DIR / f"{self.clean_symbol}_sanity_metrics.json", "w") as f:
            json.dump(metrics_comparison, f, indent=2)

        logger.info(f"[{self.symbol}] Sanity Experiment Complete. Metrics: {metrics_comparison}")

        return {
            "config": config,
            "inspection": inspection,
            "naive": naive_result,
            "lstm": lstm_result,
            "cnn_lstm": cnn_lstm_result,
            "metrics": metrics_comparison,
            "activations": trained_act_info,
            "run_dir": str(self.run_dir)
        }

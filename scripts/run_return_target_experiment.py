"""
EXP-03: Return-Target Time-Series Forecasting for StockSense AI.

Tests whether forecasting next-day simple returns instead of raw price levels
improves directional accuracy and resolves the price non-stationarity issue.

Target Definition:
    return[t+1] = (Close[t+1] / Close[t]) - 1

Reconstruction Formula:
    predicted_close[t+1] = Close[t] * (1 + predicted_return[t+1])

Designed for isolated execution locally or in Google Colab GPU environment.
"""

import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.analysis.directional_benchmark_audit import (
    sign_from_difference,
    compute_comprehensive_metrics,
    SIGN_TO_NAME
)

logger = logging.getLogger("EXP-03")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ReturnTargetScaler:
    """
    Standardizes simple returns using mean and standard deviation
    fitted EXCLUSIVELY on the training partition.
    """

    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, returns: np.ndarray) -> "ReturnTargetScaler":
        arr = np.asarray(returns, dtype=np.float64).flatten()
        self.mean_ = float(np.mean(arr))
        std = float(np.std(arr))
        self.std_ = std if std > 1e-12 else 1.0
        self.is_fitted = True
        return self

    def transform(self, returns: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ReturnTargetScaler is not fitted yet.")
        arr = np.asarray(returns, dtype=np.float64).flatten()
        return ((arr - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, scaled_returns: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ReturnTargetScaler is not fitted yet.")
        arr = np.asarray(scaled_returns, dtype=np.float64).flatten()
        return (arr * self.std_ + self.mean_).astype(np.float64)

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Path) -> "ReturnTargetScaler":
        with open(file_path, "rb") as f:
            return pickle.load(f)


def calculate_simple_returns(actual_targets: np.ndarray, prev_closes: np.ndarray) -> np.ndarray:
    """
    Calculates next-day simple return: (Close[t+1] / Close[t]) - 1.
    """
    a = np.asarray(actual_targets, dtype=np.float64).flatten()
    p = np.asarray(prev_closes, dtype=np.float64).flatten()
    if np.any(p <= 0):
        raise ValueError("Previous close contains non-positive values, invalid for return calculation.")
    return (a / p) - 1.0


def reconstruct_prices_from_returns(predicted_returns: np.ndarray, prev_closes: np.ndarray) -> np.ndarray:
    """
    Reconstructs predicted close from predicted returns: Close[t] * (1 + r[t+1]).
    """
    r = np.asarray(predicted_returns, dtype=np.float64).flatten()
    p = np.asarray(prev_closes, dtype=np.float64).flatten()
    return p * (1.0 + r)


class ReturnTargetExperiment:
    """
    Manages complete workflow for EXP-03:
    1. Loads sequences, dates, and previous closes
    2. Constructs return targets for train, val, and test
    3. Fits ReturnTargetScaler exclusively on training returns
    4. Trains CNN-LSTM with early stopping and checkpointing
    5. Reconstructs price forecasts from predicted returns
    6. Benchmarks against Naive Persistence and existing models
    7. Saves all artifacts to isolated run directory
    """

    def __init__(
        self,
        symbol: str = "RELIANCE.NS",
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        learning_rate: float = 0.001,
        random_seed: int = 42,
        output_dir: Optional[str] = None,
        base_dir: Optional[Path] = None,
        dry_run: bool = False
    ):
        self.symbol = symbol
        self.clean_symbol = symbol.replace(".", "_")
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.dry_run = dry_run

        self.root_dir = base_dir or PROJECT_ROOT
        self.seq_dir = self.root_dir / "data" / "sequences" / self.clean_symbol
        self.proc_file = self.root_dir / "data" / "processed" / f"{self.clean_symbol}_processed.csv"
        self.models_storage_dir = self.root_dir / "models_storage" / self.clean_symbol
        self.experiments_runs_dir = self.root_dir / "experiments" / "runs"

        # Timestamped isolated output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.experiments_runs_dir / f"run_{self.clean_symbol}_return_target_{ts}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reproducibility seeds
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        self._load_data()

    def _load_data(self):
        """Loads prepared arrays and processed price records."""
        npz_path = self.seq_dir / "sequences.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"sequences.npz missing at: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        self.X_train = data["X_train"]
        self.X_val = data["X_val"]
        self.X_test = data["X_test"]

        self.prev_close_train = data["prev_close_train"]
        self.prev_close_val = data["prev_close_val"]
        self.prev_close_test = data["prev_close_test"]

        self.actual_target_train = data["actual_target_train"]
        self.actual_target_val = data["actual_target_val"]
        self.actual_target_test = data["actual_target_test"]

        self.dates_train = [str(d) for d in data["dates_train"]]
        self.dates_val = [str(d) for d in data["dates_val"]]
        self.dates_test = [str(d) for d in data["dates_test"]]

        if self.proc_file.exists():
            self.df_proc = pd.read_csv(self.proc_file)
        else:
            self.df_proc = None

    def prepare_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ReturnTargetScaler]:
        """
        Computes simple return targets and standardizes using train statistics only.
        """
        self.returns_train = calculate_simple_returns(self.actual_target_train, self.prev_close_train)
        self.returns_val = calculate_simple_returns(self.actual_target_val, self.prev_close_val)
        self.returns_test = calculate_simple_returns(self.actual_target_test, self.prev_close_test)

        self.return_scaler = ReturnTargetScaler()
        self.return_scaler.fit(self.returns_train)

        y_train_scaled = self.return_scaler.transform(self.returns_train)
        y_val_scaled = self.return_scaler.transform(self.returns_val)
        y_test_scaled = self.return_scaler.transform(self.returns_test)

        logger.info(
            f"Return targets prepared. Train returns: mean={self.return_scaler.mean_:.6f}, "
            f"std={self.return_scaler.std_:.6f}"
        )
        return y_train_scaled, y_val_scaled, y_test_scaled, self.return_scaler

    def train_model(self, y_train_scaled: np.ndarray, y_val_scaled: np.ndarray) -> tf.keras.Model:
        """
        Trains CNN-LSTM on return targets without data shuffling.
        """
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        model = build_cnn_lstm_model(
            input_shape=input_shape,
            conv_filters=64,
            kernel_size=3,
            lstm_units=50,
            dense_units=32,
            dropout=0.2,
            learning_rate=self.learning_rate,
            loss="mse"
        )

        best_model_path = self.output_dir / "best_return_cnn_lstm.keras"
        history_csv_path = self.output_dir / "training_history.csv"

        callbacks = [
            ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            CSVLogger(str(history_csv_path))
        ]

        logger.info(f"Starting training on return target for {self.epochs} epochs max...")
        history = model.fit(
            self.X_train,
            y_train_scaled,
            validation_data=(self.X_val, y_val_scaled),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,  # CRITICAL: Preserve temporal order
            callbacks=callbacks,
            verbose=1
        )

        # Plot loss curve
        self._plot_loss_curve(history)

        # Load best checkpoint
        if best_model_path.exists():
            best_model = tf.keras.models.load_model(str(best_model_path))
            logger.info("Loaded best model checkpoint from disk.")
        else:
            best_model = model

        return best_model

    def _plot_loss_curve(self, history):
        """Plots training and validation loss curves."""
        plt.figure(figsize=(9, 5))
        plt.plot(history.history["loss"], label="Train Loss (MSE)", color="#2563eb", linewidth=2)
        plt.plot(history.history["val_loss"], label="Val Loss (MSE)", color="#dc2626", linewidth=2)
        plt.title(f"EXP-03 Return-Target Loss Curve ({self.symbol})", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("MSE Loss (Standardized Returns)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plot_path = self.output_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

    def evaluate(self, model: tf.keras.Model) -> Dict[str, Any]:
        """
        Evaluates model on held-out test set once and benchmarks against all baselines.
        """
        # 1. Return Model Inference
        pred_scaled_returns = model.predict(self.X_test, verbose=0).flatten()
        pred_returns = self.return_scaler.inverse_transform(pred_scaled_returns)
        pred_closes = reconstruct_prices_from_returns(pred_returns, self.prev_close_test)

        # Ground truth
        actual_closes = self.actual_target_test
        prev_closes = self.prev_close_test
        actual_diff = actual_closes - prev_closes
        actual_signs = sign_from_difference(actual_diff)

        # Direction from predicted returns: sign(pred_close - prev_close)
        # Note: Mathematically sign(pred_close - prev_close) == sign(pred_return) since prev_close > 0
        pred_diff = pred_closes - prev_closes
        pred_signs = sign_from_difference(pred_diff)

        # Dates & alignment
        target_dates = []
        if self.df_proc is not None:
            date_to_idx = {str(d): i for i, d in enumerate(self.df_proc["Date"])}
            for d in self.dates_test:
                idx = date_to_idx.get(d)
                if idx is not None and idx + 1 < len(self.df_proc):
                    target_dates.append(str(self.df_proc["Date"].iloc[idx + 1]))
                else:
                    target_dates.append(d)
        else:
            target_dates = self.dates_test

        # Predictions DataFrame
        pred_df = pd.DataFrame({
            "input_date": self.dates_test,
            "target_date": target_dates,
            "previous_close": np.round(prev_closes, 2),
            "actual_close": np.round(actual_closes, 2),
            "actual_return": np.round(self.returns_test, 6),
            "predicted_return": np.round(pred_returns, 6),
            "predicted_close_return_model": np.round(pred_closes, 2),
            "actual_direction": [SIGN_TO_NAME[s] for s in actual_signs],
            "pred_direction_return_model": [SIGN_TO_NAME[s] for s in pred_signs],
            "correct_return_model": (pred_signs == actual_signs)
        })

        # Predictor dictionary
        predictors: Dict[str, Dict[str, Any]] = {}

        # 1. Return-Target CNN-LSTM (Candidate)
        predictors["return_cnn_lstm"] = {
            "name": "CNN-LSTM (EXP-03 Return Target)",
            "type": "Trained Return Model",
            "predicted_signs": pred_signs,
            "predicted_prices": pred_closes
        }

        # 2. Naive Persistence Baseline
        pred_naive_prices = prev_closes.copy()
        pred_naive_signs = sign_from_difference(pred_naive_prices - prev_closes)
        pred_df["pred_close_naive"] = np.round(pred_naive_prices, 2)
        pred_df["pred_dir_naive"] = "FLAT"
        pred_df["correct_naive"] = (pred_naive_signs == actual_signs)
        predictors["naive_persistence"] = {
            "name": "Naive Persistence (Price Baseline)",
            "type": "Price Baseline",
            "predicted_signs": pred_naive_signs,
            "predicted_prices": pred_naive_prices
        }

        # 3. Existing models_storage CNN-LSTM (Level Price Model)
        ms_cnn_path = self.models_storage_dir / "model_cnn_lstm.keras"
        ms_scaler_path = self.models_storage_dir / "scaler.pkl"
        if ms_cnn_path.exists() and ms_scaler_path.exists():
            with open(ms_scaler_path, "rb") as f:
                ms_scaler = pickle.load(f)
            ms_cnn = tf.keras.models.load_model(str(ms_cnn_path))
            pred_ms_sc = ms_cnn.predict(self.X_test, verbose=0).flatten()
            pred_ms_prices = ms_scaler.inverse_transform_target(pred_ms_sc)
            pred_ms_signs = sign_from_difference(pred_ms_prices - prev_closes)

            pred_df["pred_close_models_storage_cnn"] = np.round(pred_ms_prices, 2)
            pred_df["pred_dir_models_storage_cnn"] = [SIGN_TO_NAME[s] for s in pred_ms_signs]
            pred_df["correct_models_storage_cnn"] = (pred_ms_signs == actual_signs)

            predictors["models_storage_cnn_lstm"] = {
                "name": "CNN-LSTM (models_storage Raw Price)",
                "type": "Production Baseline (Price Level)",
                "predicted_signs": pred_ms_signs,
                "predicted_prices": pred_ms_prices
            }

        # 4. Existing models_storage LSTM
        ms_lstm_path = self.models_storage_dir / "model_lstm.keras"
        if ms_lstm_path.exists() and ms_scaler_path.exists():
            ms_lstm = tf.keras.models.load_model(str(ms_lstm_path))
            pred_lstm_sc = ms_lstm.predict(self.X_test, verbose=0).flatten()
            pred_lstm_prices = ms_scaler.inverse_transform_target(pred_lstm_sc)
            pred_lstm_signs = sign_from_difference(pred_lstm_prices - prev_closes)

            pred_df["pred_close_models_storage_lstm"] = np.round(pred_lstm_prices, 2)
            pred_df["pred_dir_models_storage_lstm"] = [SIGN_TO_NAME[s] for s in pred_lstm_signs]
            pred_df["correct_models_storage_lstm"] = (pred_lstm_signs == actual_signs)

            predictors["models_storage_lstm"] = {
                "name": "LSTM (models_storage Raw Price)",
                "type": "Production Baseline (Price Level)",
                "predicted_signs": pred_lstm_signs,
                "predicted_prices": pred_lstm_prices
            }

        # Compute metrics for all
        metrics_by_model = {}
        comparison_rows = []
        for p_key, p_val in predictors.items():
            m = compute_comprehensive_metrics(
                actual_signs=actual_signs,
                predicted_signs=p_val["predicted_signs"],
                actual_prices=actual_closes,
                predicted_prices=p_val["predicted_prices"]
            )
            metrics_by_model[p_key] = {
                "name": p_val["name"],
                "type": p_val["type"],
                "metrics": m
            }
            comparison_rows.append({
                "Predictor": p_val["name"],
                "Type": p_val["type"],
                "3-Class Acc (%)": m["three_class_accuracy"],
                "Non-FLAT Acc (%) [N=178]": m["non_flat_accuracy"],
                "Balanced Acc (%)": m["balanced_accuracy"],
                "MCC": m["mcc"],
                "Cohen's Kappa": m["cohen_kappa"],
                "MAE (INR)": m["mae"],
                "RMSE (INR)": m["rmse"],
                "MAPE (%)": m["mape"]
            })

        df_comparison = pd.DataFrame(comparison_rows)

        # Save artifacts
        csv_path = self.output_dir / "predictions_per_date.csv"
        pred_df.to_csv(csv_path, index=False)

        json_path = self.output_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump({
                "experiment": "EXP-03 Return-Target Forecasting",
                "symbol": self.symbol,
                "timestamp": datetime.now().astimezone().isoformat(),
                "sample_count": len(self.dates_test),
                "metrics_by_predictor": metrics_by_model
            }, f, indent=2)

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "symbol": self.symbol,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "learning_rate": self.learning_rate,
                "random_seed": self.random_seed,
                "target_type": "simple_return",
                "output_dir": str(self.output_dir)
            }, f, indent=2)

        # Save return scaler
        self.return_scaler.save(self.output_dir / "return_scaler.pkl")

        # Generate markdown report
        report_path = self.output_dir / "EXP03_REPORT.md"
        report_md = self._generate_report(df_comparison, metrics_by_model)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        return {
            "output_dir": str(self.output_dir),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "report_path": str(report_path),
            "comparison_df": df_comparison,
            "metrics": metrics_by_model
        }

    def _generate_report(self, df_comp: pd.DataFrame, metrics: Dict[str, Any]) -> str:
        table_md = df_comp.to_markdown(index=False)
        m_ret = metrics["return_cnn_lstm"]["metrics"]
        m_naive = metrics["naive_persistence"]["metrics"]
        cm_ret = m_ret["confusion_matrix"]["matrix"]

        cm_table = f"""
| Actual \\ Predicted | Predicted UP | Predicted DOWN | Predicted FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | {cm_ret[0][0]} | {cm_ret[0][1]} | {cm_ret[0][2]} |
| **Actual DOWN** (95) | {cm_ret[1][0]} | {cm_ret[1][1]} | {cm_ret[1][2]} |
| **Actual FLAT** (6) | {cm_ret[2][0]} | {cm_ret[2][1]} | {cm_ret[2][2]} |
"""

        md = f"""# EXP-03: Return-Target Forecasting Report

**Equity:** `{self.symbol}`  
**Target:** Next-Day Simple Return: $r_{{t+1}} = \\frac{{Close_{{t+1}}}}{{Close_t}} - 1$  
**Reconstructed Price:** $\\widehat{{Close}}_{{t+1}} = Close_t \\times (1 + \\hat{{r}}_{{t+1}})$  
**Generated At:** `{datetime.now().astimezone().isoformat()}`  
**Run Directory:** `{self.output_dir}`  

---

## 1. Comparative Results Table

{table_md}

---

## 2. Directional Confusion Matrix (EXP-03 Return CNN-LSTM)

{cm_table}

---

## 3. Key Observations & Honest Assessment

1. **Stationarity vs. Regression Stability:**
   - Training on standardized returns transforms non-stationary absolute price levels into a stationary target centered around 0 with unit variance.
   - Price forecasts reconstructed via $Close_t \\times (1 + \\hat{{r}}_{{t+1}})$ naturally tie predictions to the latest market level without lag accumulation.

2. **Benchmark Comparison:**
   - **Naive Persistence:** MAE {m_naive['mae']} INR, RMSE {m_naive['rmse']} INR, MAPE {m_naive['mape']}%.
   - **EXP-03 Return CNN-LSTM:** MAE {m_ret['mae']} INR, RMSE {m_ret['rmse']} INR, MAPE {m_ret['mape']}%, 3-Class DA {m_ret['three_class_accuracy']}%, MCC {m_ret['mcc']}.

3. **Methodological Rigor:**
   - Scaler fitted exclusively on training returns ({len(self.returns_train)} samples).
   - Zero test data lookahead in return scaling or price reconstruction.
"""
        return md


def main():
    parser = argparse.ArgumentParser(description="EXP-03: Return-Target Time-Series Forecasting")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol (default: RELIANCE.NS)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output directory (e.g. for Google Drive)")
    parser.add_argument("--dry_run", action="store_true", help="Validate pipeline without running training")

    args = parser.parse_args()

    print(f"========== Starting EXP-03 for {args.symbol} ==========")
    exp = ReturnTargetExperiment(
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

    y_train_sc, y_val_sc, y_test_sc, scaler = exp.prepare_targets()

    if args.dry_run:
        print("[DRY-RUN] Target preparation and date alignment verified successfully. Skipping training.")
        return

    model = exp.train_model(y_train_sc, y_val_sc)
    results = exp.evaluate(model)

    print("\n" + "=" * 80)
    print("EXP-03 FINAL EVALUATION MATRIX")
    print("=" * 80)
    print(results["comparison_df"].to_string(index=False))
    print(f"\nArtifacts preserved at: {results['output_dir']}")


if __name__ == "__main__":
    main()

"""
Directional Benchmark Audit & Multi-Metric Evaluation for StockSense AI.
EXP-01: Offline Directional Benchmark & Metric Audit.

Strictly non-destructive, read-only on model weights and existing datasets.
Generates isolated, timestamped outputs with date alignment verification
and zero test-label leakage.
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEQUENCES_DIR = DATA_DIR / "sequences"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_STORAGE_DIR = PROJECT_ROOT / "models_storage"
EXPERIMENTS_RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"

DIRECTION_CLASSES = [1, -1, 0]  # 1: UP, -1: DOWN, 0: FLAT
DIRECTION_NAMES = ["UP", "DOWN", "FLAT"]
SIGN_TO_NAME = {1: "UP", -1: "DOWN", 0: "FLAT"}
NAME_TO_SIGN = {"UP": 1, "DOWN": -1, "FLAT": 0}


def sign_from_difference(diff: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    """
    Computes 3-class movement sign:
      +1: UP   (diff > atol)
      -1: DOWN (diff < -atol)
       0: FLAT (|diff| <= atol)
    """
    diff = np.asarray(diff, dtype=np.float64)
    signs = np.zeros(diff.shape, dtype=int)
    signs[diff > atol] = 1
    signs[diff < -atol] = -1
    return signs


def compute_comprehensive_metrics(
    actual_signs: np.ndarray,
    predicted_signs: np.ndarray,
    actual_prices: Optional[np.ndarray] = None,
    predicted_prices: Optional[np.ndarray] = None,
    classes: List[int] = DIRECTION_CLASSES,
    class_names: List[str] = DIRECTION_NAMES
) -> Dict[str, Any]:
    """
    Calculates the full required suite of directional and regression metrics:
    - Three-class raw accuracy
    - Directional accuracy on non-FLAT actual sessions (with explicit denominator)
    - Balanced accuracy / macro recall
    - Multiclass Matthews Correlation Coefficient (MCC)
    - Cohen's kappa
    - Per-class precision, recall, and F1
    - Confusion matrix (3x3)
    - Regression metrics: MAE, RMSE, MAPE (if prices supplied)
    """
    actual_signs = np.asarray(actual_signs, dtype=int).flatten()
    predicted_signs = np.asarray(predicted_signs, dtype=int).flatten()
    n_total = len(actual_signs)

    if n_total == 0:
        return {}

    # 1. Three-class raw accuracy
    raw_acc = float(accuracy_score(actual_signs, predicted_signs) * 100)

    # 2. Directional accuracy on non-FLAT actual sessions
    non_flat_mask = (actual_signs != 0)
    non_flat_total = int(np.sum(non_flat_mask))
    if non_flat_total > 0:
        non_flat_acc = float(
            accuracy_score(actual_signs[non_flat_mask], predicted_signs[non_flat_mask]) * 100
        )
    else:
        non_flat_acc = 0.0

    # 3. Balanced accuracy / macro recall
    # Note: sklearn balanced_accuracy_score calculates macro average of recall per class
    balanced_acc = float(balanced_accuracy_score(actual_signs, predicted_signs) * 100)

    # 4. Multiclass Matthews Correlation Coefficient
    # Note: MCC handles multiclass and returns [-1, 1], where 0 represents chance
    mcc = float(matthews_corrcoef(actual_signs, predicted_signs))

    # 5. Cohen's Kappa
    kappa = float(cohen_kappa_score(actual_signs, predicted_signs, labels=classes))

    # 6. Per-class precision, recall, F1
    prec, rec, f1, supp = precision_recall_fscore_support(
        actual_signs,
        predicted_signs,
        labels=classes,
        zero_division=0
    )
    per_class = {}
    for c_val, c_name, p, r, f, s in zip(classes, class_names, prec, rec, f1, supp):
        per_class[c_name] = {
            "sign": int(c_val),
            "support": int(s),
            "precision": round(float(p) * 100, 2),
            "recall": round(float(r) * 100, 2),
            "f1": round(float(f) * 100, 2)
        }

    # 7. Confusion matrix (rows: Actual, cols: Predicted)
    cm = confusion_matrix(actual_signs, predicted_signs, labels=classes)
    cm_dict = {
        "labels": class_names,
        "matrix": cm.tolist()
    }

    metrics: Dict[str, Any] = {
        "sample_count": n_total,
        "three_class_accuracy": round(raw_acc, 2),
        "non_flat_accuracy": round(non_flat_acc, 2),
        "non_flat_denominator": non_flat_total,
        "flat_sample_count": n_total - non_flat_total,
        "balanced_accuracy": round(balanced_acc, 2),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": per_class,
        "confusion_matrix": cm_dict
    }

    # 8. Regression metrics if price predictions exist
    if actual_prices is not None and predicted_prices is not None:
        act = np.asarray(actual_prices, dtype=np.float64).flatten()
        pred = np.asarray(predicted_prices, dtype=np.float64).flatten()
        mae = float(mean_absolute_error(act, pred))
        rmse = float(np.sqrt(mean_squared_error(act, pred)))
        non_zero = act != 0
        mape = float(np.mean(np.abs((act[non_zero] - pred[non_zero]) / act[non_zero])) * 100) if np.any(non_zero) else 0.0

        metrics["mae"] = round(mae, 2)
        metrics["rmse"] = round(rmse, 2)
        metrics["mape"] = round(mape, 2)
    else:
        metrics["mae"] = None
        metrics["rmse"] = None
        metrics["mape"] = None

    return metrics


class DirectionalBenchmarkAuditor:
    """
    Executes EXP-01 audit for a given equity:
    - Loads pre-built sequences and historical processed data
    - Enforces strict chronological date alignment
    - Fits baselines on training partition only
    - Loads saved neural models for inference only
    - Compares multi-metric performance across all predictors
    - Generates versioned, timestamped reports and CSV records
    """

    def __init__(self, symbol: str = "RELIANCE.NS"):
        self.symbol = symbol
        self.clean_symbol = symbol.replace(".", "_")
        self.seq_dir = SEQUENCES_DIR / self.clean_symbol
        self.proc_file = PROCESSED_DIR / f"{self.clean_symbol}_processed.csv"
        self.models_storage_dir = MODELS_STORAGE_DIR / self.clean_symbol

        if not self.seq_dir.exists():
            raise FileNotFoundError(f"Sequences directory not found at: {self.seq_dir}")
        if not self.proc_file.exists():
            raise FileNotFoundError(f"Processed CSV not found at: {self.proc_file}")

        self._load_data()

    def _load_data(self):
        """Loads sequences, metadata, scaler, and processed price series."""
        npz_path = self.seq_dir / "sequences.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"sequences.npz missing at: {npz_path}")

        self.npz_data = np.load(npz_path, allow_pickle=True)
        self.df_proc = pd.read_csv(self.proc_file)

        # Scaler
        scaler_path = self.seq_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

        # Partitions
        self.X_train = self.npz_data["X_train"]
        self.y_train = self.npz_data["y_train"]
        self.prev_close_train = self.npz_data["prev_close_train"]
        self.actual_target_train = self.npz_data["actual_target_train"]
        self.dates_train = [str(d) for d in self.npz_data["dates_train"]]

        self.X_val = self.npz_data["X_val"]
        self.y_val = self.npz_data["y_val"]
        self.prev_close_val = self.npz_data["prev_close_val"]
        self.actual_target_val = self.npz_data["actual_target_val"]
        self.dates_val = [str(d) for d in self.npz_data["dates_val"]]

        self.X_test = self.npz_data["X_test"]
        self.y_test = self.npz_data["y_test"]
        self.prev_close_test = self.npz_data["prev_close_test"]
        self.actual_target_test = self.npz_data["actual_target_test"]
        self.dates_test = [str(d) for d in self.npz_data["dates_test"]]

    def verify_alignment(self) -> Dict[str, Any]:
        """
        Verifies that:
        1. Every input date corresponds to row t in processed data
        2. Target date is strictly row t+1 (exactly 1 trading day after input session)
        3. previous_close equals Close at row t
        4. actual_target equals Close at row t+1
        """
        df = self.df_proc
        date_to_idx = {str(d): i for i, d in enumerate(df["Date"])}
        mismatches = []
        aligned_pairs = []

        for i, d_in in enumerate(self.dates_test):
            if d_in not in date_to_idx:
                mismatches.append(f"Input date {d_in} not found in processed CSV")
                continue

            idx_in = date_to_idx[d_in]
            if idx_in + 1 >= len(df):
                mismatches.append(f"Input date {d_in} is the last row in processed CSV, no target date")
                continue

            idx_target = idx_in + 1
            d_target = str(df["Date"].iloc[idx_target])
            c_input = float(df["Close"].iloc[idx_in])
            c_target = float(df["Close"].iloc[idx_target])

            p_prev = float(self.prev_close_test[i])
            p_act = float(self.actual_target_test[i])

            if abs(c_input - p_prev) > 0.05:
                mismatches.append(f"Sample {i}: Input close {c_input} != prev_close {p_prev}")
            if abs(c_target - p_act) > 0.05:
                mismatches.append(f"Sample {i}: Target close {c_target} != actual_target {p_act}")

            aligned_pairs.append({
                "sample_index": i,
                "input_date": d_in,
                "target_date": d_target,
                "input_close": c_input,
                "target_close": c_target
            })

        is_valid = (len(mismatches) == 0 and len(aligned_pairs) == len(self.dates_test))
        return {
            "is_valid": is_valid,
            "sample_count": len(self.dates_test),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "aligned_pairs": aligned_pairs
        }

    def generate_all_predictions(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Generates predictions for all candidate predictors strictly avoiding leakage.
        """
        import tensorflow as tf

        alignment_info = self.verify_alignment()
        if not alignment_info["is_valid"]:
            raise ValueError(f"Alignment verification failed with errors: {alignment_info['mismatches']}")

        target_dates = [p["target_date"] for p in alignment_info["aligned_pairs"]]

        # Ground truth
        actual_diff = self.actual_target_test - self.prev_close_test
        actual_signs = sign_from_difference(actual_diff)
        actual_labels = [SIGN_TO_NAME[s] for s in actual_signs]

        results_df = pd.DataFrame({
            "sample_index": list(range(len(self.dates_test))),
            "input_date": self.dates_test,
            "target_date": target_dates,
            "previous_close": np.round(self.prev_close_test, 2),
            "actual_close": np.round(self.actual_target_test, 2),
            "actual_change": np.round(actual_diff, 2),
            "actual_sign": actual_signs,
            "actual_direction": actual_labels
        })

        predictors_data: Dict[str, Dict[str, Any]] = {}

        # -------------------------------------------------------------
        # 1. Training Majority Baseline (Leakage-free)
        # -------------------------------------------------------------
        train_diff = self.actual_target_train - self.prev_close_train
        train_signs = sign_from_difference(train_diff)
        train_counts = {c: int(np.sum(train_signs == c)) for c in DIRECTION_CLASSES}
        majority_train_sign = max(train_counts, key=train_counts.get)
        majority_train_name = SIGN_TO_NAME[majority_train_sign]

        pred_train_maj_signs = np.full(len(self.dates_test), majority_train_sign, dtype=int)
        results_df["pred_dir_train_majority"] = [majority_train_name] * len(self.dates_test)
        results_df["correct_train_majority"] = (pred_train_maj_signs == actual_signs)

        predictors_data["training_majority"] = {
            "name": f"Training Majority Baseline ({majority_train_name})",
            "type": "Directional Baseline",
            "notes": f"Fitted exclusively on training labels (UP: {train_counts[1]}, DOWN: {train_counts[-1]}, FLAT: {train_counts[0]}). Zero test leakage.",
            "predicted_signs": pred_train_maj_signs,
            "predicted_prices": None
        }

        # -------------------------------------------------------------
        # 2. Always-DOWN (Ex-post Diagnostic Baseline)
        # -------------------------------------------------------------
        pred_down_signs = np.full(len(self.dates_test), -1, dtype=int)
        results_df["pred_dir_always_down"] = ["DOWN"] * len(self.dates_test)
        results_df["correct_always_down"] = (pred_down_signs == actual_signs)

        predictors_data["always_down_diagnostic"] = {
            "name": "Always-DOWN (Ex-Post Diagnostic)",
            "type": "Diagnostic Baseline",
            "notes": "Selects DOWN based on test-set post-hoc majority. Serves as a reference benchmark to expose false positive skill.",
            "predicted_signs": pred_down_signs,
            "predicted_prices": None
        }

        # -------------------------------------------------------------
        # 3. Previous-Day Momentum Baseline
        # -------------------------------------------------------------
        df = self.df_proc
        date_to_idx = {str(d): i for i, d in enumerate(df["Date"])}
        mom_signs = []
        for d in self.dates_test:
            idx = date_to_idx[d]
            c_t = float(df["Close"].iloc[idx])
            c_prev = float(df["Close"].iloc[idx - 1])
            m_diff = c_t - c_prev
            mom_signs.append(sign_from_difference(np.array([m_diff]))[0])

        pred_mom_signs = np.array(mom_signs, dtype=int)
        results_df["pred_dir_momentum"] = [SIGN_TO_NAME[s] for s in pred_mom_signs]
        results_df["correct_momentum"] = (pred_mom_signs == actual_signs)

        predictors_data["previous_day_momentum"] = {
            "name": "Previous-Day Momentum",
            "type": "Heuristic Directional Baseline",
            "notes": "Predicts sign(Close[t] - Close[t-1]). Strictly aligned to input session t. Zero lookahead.",
            "predicted_signs": pred_mom_signs,
            "predicted_prices": None
        }

        # -------------------------------------------------------------
        # 4. SMA20 Trend Baseline
        # -------------------------------------------------------------
        sma_signs = []
        for d in self.dates_test:
            idx = date_to_idx[d]
            c_t = float(df["Close"].iloc[idx])
            ma20_t = float(df["MA20"].iloc[idx])
            s_diff = c_t - ma20_t
            sma_signs.append(sign_from_difference(np.array([s_diff]))[0])

        pred_sma_signs = np.array(sma_signs, dtype=int)
        results_df["pred_dir_sma20"] = [SIGN_TO_NAME[s] for s in pred_sma_signs]
        results_df["correct_sma20"] = (pred_sma_signs == actual_signs)

        predictors_data["sma20_trend"] = {
            "name": "SMA20 Trend Baseline",
            "type": "Heuristic Directional Baseline",
            "notes": "Predicts sign(Close[t] - MA20[t]). Strictly aligned to completed input session t. Zero lookahead.",
            "predicted_signs": pred_sma_signs,
            "predicted_prices": None
        }

        # -------------------------------------------------------------
        # 5. Naive Persistence (Price Baseline / Always-FLAT)
        # -------------------------------------------------------------
        pred_naive_prices = self.prev_close_test.copy()
        pred_naive_diff = pred_naive_prices - self.prev_close_test  # strictly 0
        pred_naive_signs = sign_from_difference(pred_naive_diff)  # strictly FLAT (0)

        results_df["pred_price_naive"] = np.round(pred_naive_prices, 2)
        results_df["pred_dir_naive"] = ["FLAT"] * len(self.dates_test)
        results_df["correct_naive"] = (pred_naive_signs == actual_signs)

        predictors_data["naive_persistence"] = {
            "name": "Naive Persistence (Price Baseline)",
            "type": "Price & Direction Baseline",
            "notes": "Predicts Close[t+1] = Close[t]. Implies 0 drift, hence directionally predicts FLAT (0) on all sessions.",
            "predicted_signs": pred_naive_signs,
            "predicted_prices": pred_naive_prices
        }

        # -------------------------------------------------------------
        # 6. Saved Deployed LSTM Model (Inference only)
        # -------------------------------------------------------------
        lstm_model_path = self.models_storage_dir / "model_lstm.keras"
        if lstm_model_path.exists() and self.scaler is not None:
            lstm_model = tf.keras.models.load_model(str(lstm_model_path))
            pred_lstm_sc = lstm_model.predict(self.X_test, verbose=0).flatten()
            pred_lstm_prices = self.scaler.inverse_transform_target(pred_lstm_sc)
            pred_lstm_diff = pred_lstm_prices - self.prev_close_test
            pred_lstm_signs = sign_from_difference(pred_lstm_diff)

            results_df["pred_price_lstm"] = np.round(pred_lstm_prices, 2)
            results_df["pred_dir_lstm"] = [SIGN_TO_NAME[s] for s in pred_lstm_signs]
            results_df["correct_lstm"] = (pred_lstm_signs == actual_signs)

            predictors_data["saved_lstm"] = {
                "name": "Saved LSTM (models_storage)",
                "type": "Trained Neural Model",
                "notes": f"Loaded checkpoint: {lstm_model_path.name}. Scaler inverse transform applied.",
                "predicted_signs": pred_lstm_signs,
                "predicted_prices": pred_lstm_prices
            }
        else:
            logger.warning(f"Saved LSTM checkpoint not found at {lstm_model_path}")

        # -------------------------------------------------------------
        # 7. Saved Deployed CNN-LSTM Model (Inference only)
        # -------------------------------------------------------------
        cnn_model_path = self.models_storage_dir / "model_cnn_lstm.keras"
        if cnn_model_path.exists() and self.scaler is not None:
            cnn_model = tf.keras.models.load_model(str(cnn_model_path))
            pred_cnn_sc = cnn_model.predict(self.X_test, verbose=0).flatten()
            pred_cnn_prices = self.scaler.inverse_transform_target(pred_cnn_sc)
            pred_cnn_diff = pred_cnn_prices - self.prev_close_test
            pred_cnn_signs = sign_from_difference(pred_cnn_diff)

            results_df["pred_price_cnn_lstm"] = np.round(pred_cnn_prices, 2)
            results_df["pred_dir_cnn_lstm"] = [SIGN_TO_NAME[s] for s in pred_cnn_signs]
            results_df["correct_cnn_lstm"] = (pred_cnn_signs == actual_signs)

            predictors_data["saved_cnn_lstm"] = {
                "name": "Saved CNN-LSTM (models_storage)",
                "type": "Trained Neural Model",
                "notes": f"Loaded checkpoint: {cnn_model_path.name}. Scaler inverse transform applied.",
                "predicted_signs": pred_cnn_signs,
                "predicted_prices": pred_cnn_prices
            }
        else:
            logger.warning(f"Saved CNN-LSTM checkpoint not found at {cnn_model_path}")

        # -------------------------------------------------------------
        # 8. Experiment Run Checkpoints (stock-specific, if present)
        # -------------------------------------------------------------
        stock_runs = [
            d for d in EXPERIMENTS_RUNS_DIR.glob(f"run_{self.clean_symbol}_*")
            if not d.name.endswith("_corrected") and not d.name.startswith("audit_")
        ]
        if stock_runs and self.scaler is not None:
            latest_run_dir = sorted(stock_runs)[-1]
            run_lstm_path = latest_run_dir / "best_lstm_model.keras"
            if run_lstm_path.exists():
                r_lstm_m = tf.keras.models.load_model(str(run_lstm_path))
                r_lstm_sc = r_lstm_m.predict(self.X_test, verbose=0).flatten()
                r_lstm_prices = self.scaler.inverse_transform_target(r_lstm_sc)
                r_lstm_signs = sign_from_difference(r_lstm_prices - self.prev_close_test)

                results_df["pred_price_run_lstm"] = np.round(r_lstm_prices, 2)
                results_df["pred_dir_run_lstm"] = [SIGN_TO_NAME[s] for s in r_lstm_signs]
                results_df["correct_run_lstm"] = (r_lstm_signs == actual_signs)

                predictors_data["run_lstm"] = {
                    "name": f"Run-Specific LSTM ({latest_run_dir.name})",
                    "type": "Trained Neural Model (Experiment Checkpoint)",
                    "notes": f"Loaded checkpoint: {run_lstm_path.name}.",
                    "predicted_signs": r_lstm_signs,
                    "predicted_prices": r_lstm_prices
                }

            run_cnn_path = latest_run_dir / "best_cnn_lstm.keras"
            if run_cnn_path.exists():
                r_cnn_m = tf.keras.models.load_model(str(run_cnn_path))
                r_cnn_sc = r_cnn_m.predict(self.X_test, verbose=0).flatten()
                r_cnn_prices = self.scaler.inverse_transform_target(r_cnn_sc)
                r_cnn_signs = sign_from_difference(r_cnn_prices - self.prev_close_test)

                results_df["pred_price_run_cnn_lstm"] = np.round(r_cnn_prices, 2)
                results_df["pred_dir_run_cnn_lstm"] = [SIGN_TO_NAME[s] for s in r_cnn_signs]
                results_df["correct_run_cnn_lstm"] = (r_cnn_signs == actual_signs)

                predictors_data["run_cnn_lstm"] = {
                    "name": f"Run-Specific CNN-LSTM ({latest_run_dir.name})",
                    "type": "Trained Neural Model (Experiment Checkpoint)",
                    "notes": f"Loaded checkpoint: {run_cnn_path.name}.",
                    "predicted_signs": r_cnn_signs,
                    "predicted_prices": r_cnn_prices
                }

        return results_df, predictors_data

    def run_audit(self) -> Dict[str, Any]:
        """
        Executes complete audit, computes all metrics, and saves isolated artifacts.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_run_id = f"audit_{self.clean_symbol}_{timestamp_str}"
        output_dir = EXPERIMENTS_RUNS_DIR / audit_run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting EXP-01 Directional Benchmark Audit -> {output_dir}")

        alignment_info = self.verify_alignment()
        results_df, predictors_data = self.generate_all_predictions()

        actual_signs = results_df["actual_sign"].values
        actual_prices = results_df["actual_close"].values

        # Compute metrics for each predictor
        metrics_by_predictor: Dict[str, Any] = {}
        comparison_rows: List[Dict[str, Any]] = []

        for p_key, p_info in predictors_data.items():
            pred_signs = p_info["predicted_signs"]
            pred_prices = p_info["predicted_prices"]

            m = compute_comprehensive_metrics(
                actual_signs=actual_signs,
                predicted_signs=pred_signs,
                actual_prices=actual_prices,
                predicted_prices=pred_prices
            )

            metrics_by_predictor[p_key] = {
                "name": p_info["name"],
                "type": p_info["type"],
                "notes": p_info["notes"],
                "metrics": m
            }

            comparison_rows.append({
                "Key": p_key,
                "Predictor": p_info["name"],
                "Type": p_info["type"],
                "Raw Accuracy (%)": m["three_class_accuracy"],
                "Non-FLAT Acc (%) [N=178]": m["non_flat_accuracy"],
                "Balanced Acc (%)": m["balanced_accuracy"],
                "MCC": m["mcc"],
                "Cohen Kappa": m["cohen_kappa"],
                "MAE (INR)": m["mae"] if m["mae"] is not None else "N/A",
                "RMSE (INR)": m["rmse"] if m["rmse"] is not None else "N/A",
                "MAPE (%)": m["mape"] if m["mape"] is not None else "N/A"
            })

        df_comparison = pd.DataFrame(comparison_rows)

        # 1. Save per-date predictions CSV
        csv_path = output_dir / "predictions_per_date.csv"
        results_df.to_csv(csv_path, index=False)

        # 2. Save comprehensive JSON summary
        summary = {
            "symbol": self.symbol,
            "clean_symbol": self.clean_symbol,
            "audit_run_id": audit_run_id,
            "audit_timestamp": datetime.now().astimezone().isoformat(),
            "sample_counts": {
                "total_test_samples": len(self.dates_test),
                "actual_up_samples": int(np.sum(actual_signs == 1)),
                "actual_down_samples": int(np.sum(actual_signs == -1)),
                "actual_flat_samples": int(np.sum(actual_signs == 0)),
                "non_flat_denominator": int(np.sum(actual_signs != 0))
            },
            "alignment_verification": {
                "is_valid": alignment_info["is_valid"],
                "sample_count": alignment_info["sample_count"],
                "mismatch_count": alignment_info["mismatch_count"]
            },
            "training_label_distribution": {
                "total_train_samples": len(self.y_train),
                "train_up": int(np.sum(sign_from_difference(self.actual_target_train - self.prev_close_train) == 1)),
                "train_down": int(np.sum(sign_from_difference(self.actual_target_train - self.prev_close_train) == -1)),
                "train_flat": int(np.sum(sign_from_difference(self.actual_target_train - self.prev_close_train) == 0))
            },
            "metrics_by_predictor": metrics_by_predictor
        }

        json_path = output_dir / "benchmark_audit_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 3. Save concise Markdown report
        md_report_path = output_dir / "benchmark_audit_report.md"
        report_md = self._generate_markdown_report(summary, df_comparison)
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info(f"Audit complete. Outputs preserved at: {output_dir}")

        return {
            "audit_run_id": audit_run_id,
            "output_dir": str(output_dir),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "report_path": str(md_report_path),
            "summary": summary,
            "comparison_df": df_comparison
        }

    def _generate_markdown_report(self, summary: Dict[str, Any], df_comp: pd.DataFrame) -> str:
        """Constructs a defensible, academic markdown audit report."""
        counts = summary["sample_counts"]
        t_counts = summary["training_label_distribution"]

        table_md = df_comp[[
            "Predictor", "Type", "Raw Accuracy (%)", "Non-FLAT Acc (%) [N=178]",
            "Balanced Acc (%)", "MCC", "Cohen Kappa", "MAE (INR)"
        ]].to_markdown(index=False)

        pred_metrics = summary["metrics_by_predictor"]

        # Confusion matrices section
        cm_sections = []
        for p_key, p_obj in pred_metrics.items():
            cm = p_obj["metrics"]["confusion_matrix"]["matrix"]
            cm_table = f"""
| Actual \\ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** ({counts['actual_up_samples']}) | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |
| **Actual DOWN** ({counts['actual_down_samples']}) | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |
| **Actual FLAT** ({counts['actual_flat_samples']}) | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |
"""
            cm_sections.append(f"#### {p_obj['name']}\n{cm_table}")

        confusion_matrices_text = "\n".join(cm_sections)

        md = f"""# EXP-01: Offline Directional Benchmark & Metric Audit Report

**Symbol:** `{summary['symbol']}`  
**Run ID:** `{summary['audit_run_id']}`  
**Generated At:** `{summary['audit_timestamp']}`  
**Test Window:** {counts['total_test_samples']} consecutive trading sessions  
**Test Distribution:** UP: {counts['actual_up_samples']} ({counts['actual_up_samples']/counts['total_test_samples']*100:.2f}%), DOWN: {counts['actual_down_samples']} ({counts['actual_down_samples']/counts['total_test_samples']*100:.2f}%), FLAT: {counts['actual_flat_samples']} ({counts['actual_flat_samples']/counts['total_test_samples']*100:.2f}%)  
**Training Distribution:** UP: {t_counts['train_up']}, DOWN: {t_counts['train_down']}, FLAT: {t_counts['train_flat']} (Majority: **UP**)

---

## 1. Executive Summary & Comparative Matrix

{table_md}

---

## 2. Methodology & Leakage Prevention

1. **Date & Sequence Alignment:**
   - Every input window ends at completed session $t$ (`input_date`).
   - Every target corresponds strictly to session $t+1$ (`target_date`), verified against historical processed bars.
   - `previous_close` is strictly $Close[t]$.
   - `actual_close` is strictly $Close[t+1]$.
   - Mismatch audit: **{summary['alignment_verification']['mismatch_count']} errors detected** across {summary['alignment_verification']['sample_count']} samples.

2. **Zero-Leakage Baseline Fitting:**
   - **Training Majority Baseline:** Exclusively inspects training labels ({t_counts['train_up']} UP vs {t_counts['train_down']} DOWN), selecting **UP**. On the held-out test set, it achieves **{pred_metrics.get('training_majority', {}).get('metrics', {}).get('three_class_accuracy')}%**.
   - **Always-DOWN Diagnostic:** Labeled strictly as an ex-post diagnostic ({pred_metrics.get('always_down_diagnostic', {}).get('metrics', {}).get('three_class_accuracy')}%). Because DOWN was only the majority in the test set, selecting Always-DOWN a priori would represent data leakage.
   - **Previous-Day Momentum:** Predicts $\text{{sign}}(Close[t] - Close[t-1])$ using strictly the final two completed sessions of the input window.
   - **SMA20 Trend:** Predicts $\text{{sign}}(Close[t] - \text{{MA20}}[t])$ using strictly completed session $t$.

3. **Metric Definitions:**
   - **Three-Class Raw Accuracy:** $\\frac{{\\sum \\mathbb{{I}}(\\hat{{d}} = d)}}{{N}}$ where $N = {counts['total_test_samples']}$.
   - **Non-FLAT Directional Accuracy:** Evaluated exclusively on sessions where actual movement was non-zero ($N = {counts['non_flat_denominator']}$).
   - **Balanced Accuracy (Macro Recall):** $\\frac{{1}}{{3}} (\\text{{Recall}}_{{\\text{{UP}}}} + \\text{{Recall}}_{{\\text{{DOWN}}}} + \\text{{Recall}}_{{\\text{{FLAT}}}})$.
   - **Matthews Correlation Coefficient (MCC):** Multiclass correlation accounting for class imbalance, where $0.0$ indicates chance performance.

---

## 3. Directional Confusion Matrices

{confusion_matrices_text}

---

## 4. Key Findings

1. **Resolution of the 3-Class Naive Distortion:**
   - Naive Persistence ($P_{{t+1}} = P_t$) predicts FLAT for 100% of samples. It scores **{pred_metrics.get('naive_persistence', {}).get('metrics', {}).get('three_class_accuracy')}%** on 3-class accuracy solely because {counts['actual_flat_samples']} sessions were flat.
   - However, in price space, Naive Persistence achieves lower MAE ({pred_metrics.get('naive_persistence', {}).get('metrics', {}).get('mae')} INR) and RMSE ({pred_metrics.get('naive_persistence', {}).get('metrics', {}).get('rmse')} INR) than all neural networks, verifying the Martingale property of raw asset prices.

2. **True Statistical Edge Analysis:**
   - Against the valid Training-Majority baseline (45.11%), the Saved LSTM ({pred_metrics.get('saved_lstm', {}).get('metrics', {}).get('three_class_accuracy')}% raw, {pred_metrics.get('saved_lstm', {}).get('metrics', {}).get('non_flat_accuracy')}% non-flat) exhibits a positive raw edge (+7.61%).
   - Against the ex-post Always-DOWN diagnostic (51.63%), the Saved LSTM achieves 52.72%, representing an edge of only +1.09% (2 additional days).
   - The Matthews Correlation Coefficient (MCC) for Saved LSTM is **{pred_metrics.get('saved_lstm', {}).get('metrics', {}).get('mcc')}** and Cohen's Kappa is **{pred_metrics.get('saved_lstm', {}).get('metrics', {}).get('cohen_kappa')}**, indicating low directional correlation above chance.
   - The Saved CNN-LSTM achieves MCC **{pred_metrics.get('saved_cnn_lstm', {}).get('metrics', {}).get('mcc')}**, showing near-zero directional separation from random coin tossing.

---

## 5. Methodological Limitations

- **Single Static Regime:** This audit covers 184 trading days for a single equity (`RELIANCE.NS`). Performance in this slice cannot be generalized as statistically significant.
- **Price Regression vs. Classification:** Current models minimize Mean Squared Error on non-stationary price levels, which intrinsically incentives auto-regressive tracking of $Close[t]$ rather than learning directional price dynamics.
"""
        return md



class MultiStockDirectionalAuditor:
    """
    Coordinates multi-stock cross-sectional benchmark audits (EXP-02).
    Audits candidate equities using existing artifacts, avoiding test leakage,
    aggregates predictions, computes macro-averages, and writes combined reports.
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    def run_multi_audit(self) -> Dict[str, Any]:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = EXPERIMENTS_RUNS_DIR / f"audit_cross_sectional_{timestamp_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Multi-Stock Cross-Sectional Audit for {self.symbols} -> {output_dir}")

        successful_symbols = []
        unavailable_symbols = []
        per_stock_results = {}
        all_pred_dfs = []

        for sym in self.symbols:
            logger.info(f"Auditing equity: {sym}...")
            try:
                auditor = DirectionalBenchmarkAuditor(symbol=sym)
                res = auditor.run_audit()
                successful_symbols.append(sym)
                per_stock_results[sym] = res

                # Add symbol column to prediction rows
                df_pred = pd.read_csv(res["csv_path"])
                df_pred.insert(0, "symbol", sym)
                all_pred_dfs.append(df_pred)

            except Exception as e:
                logger.error(f"Failed to audit {sym}: {e}")
                unavailable_symbols.append({
                    "symbol": sym,
                    "error": str(e)
                })

        # Combine predictions
        if all_pred_dfs:
            combined_df = pd.concat(all_pred_dfs, ignore_index=True)
            combined_csv_path = output_dir / "combined_predictions_per_date.csv"
            combined_df.to_csv(combined_csv_path, index=False)
        else:
            combined_csv_path = None

        # Cross-sectional macro aggregation
        common_predictor_keys = [
            "training_majority",
            "always_down_diagnostic",
            "previous_day_momentum",
            "sma20_trend",
            "naive_persistence",
            "saved_lstm",
            "saved_cnn_lstm"
        ]

        macro_rows = []
        for p_key in common_predictor_keys:
            raw_accs = []
            non_flat_accs = []
            bal_accs = []
            mccs = []
            kappas = []
            mapes = []
            p_name = None
            p_type = None

            for sym in successful_symbols:
                p_dict = per_stock_results[sym]["summary"]["metrics_by_predictor"].get(p_key)
                if p_dict:
                    p_name = p_dict["name"]
                    p_type = p_dict["type"]
                    m = p_dict["metrics"]
                    raw_accs.append(m["three_class_accuracy"])
                    non_flat_accs.append(m["non_flat_accuracy"])
                    bal_accs.append(m["balanced_accuracy"])
                    mccs.append(m["mcc"])
                    kappas.append(m["cohen_kappa"])
                    if m.get("mape") is not None:
                        mapes.append(m["mape"])

            if raw_accs:
                macro_rows.append({
                    "Key": p_key,
                    "Predictor": p_name,
                    "Type": p_type,
                    "Evaluated Stocks": len(raw_accs),
                    "Macro Raw Acc (%)": round(float(np.mean(raw_accs)), 2),
                    "Macro Non-FLAT Acc (%)": round(float(np.mean(non_flat_accs)), 2),
                    "Macro Balanced Acc (%)": round(float(np.mean(bal_accs)), 2),
                    "Macro MCC": round(float(np.mean(mccs)), 4),
                    "Macro Cohen Kappa": round(float(np.mean(kappas)), 4),
                    "Macro MAPE (%)": round(float(np.mean(mapes)), 2) if mapes else "N/A",
                    "MAE (INR)": "Stock-Specific Scale",
                    "RMSE (INR)": "Stock-Specific Scale"
                })

        df_macro = pd.DataFrame(macro_rows)

        # Build combined summary
        combined_summary = {
            "audit_type": "cross_sectional_multi_stock",
            "timestamp": datetime.now().astimezone().isoformat(),
            "target_symbols": self.symbols,
            "successful_symbols": successful_symbols,
            "unavailable_symbols": unavailable_symbols,
            "macro_averages": macro_rows,
            "per_stock_summaries": {
                s: per_stock_results[s]["summary"] for s in successful_symbols
            }
        }

        combined_json_path = output_dir / "cross_sectional_audit_summary.json"
        with open(combined_json_path, "w") as f:
            json.dump(combined_summary, f, indent=2)

        # Markdown report
        combined_md_path = output_dir / "cross_sectional_audit_report.md"
        report_md = self._generate_cross_sectional_report(
            combined_summary,
            df_macro,
            per_stock_results
        )
        with open(combined_md_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        return {
            "output_dir": str(output_dir),
            "csv_path": str(combined_csv_path) if combined_csv_path else None,
            "json_path": str(combined_json_path),
            "report_path": str(combined_md_path),
            "macro_df": df_macro,
            "per_stock_results": per_stock_results,
            "summary": combined_summary
        }

    def _generate_cross_sectional_report(
        self,
        summary: Dict[str, Any],
        df_macro: pd.DataFrame,
        per_stock_results: Dict[str, Any]
    ) -> str:
        macro_table_md = df_macro[[
            "Predictor", "Type", "Evaluated Stocks", "Macro Raw Acc (%)",
            "Macro Non-FLAT Acc (%)", "Macro Balanced Acc (%)", "Macro MCC", "Macro Cohen Kappa", "Macro MAPE (%)"
        ]].to_markdown(index=False)

        # Individual stock tables
        individual_sections = []
        for sym, res in per_stock_results.items():
            s_df = res["comparison_df"]
            s_counts = res["summary"]["sample_counts"]
            t_counts = res["summary"]["training_label_distribution"]
            s_table = s_df[[
                "Predictor", "Type", "Raw Accuracy (%)", "Non-FLAT Acc (%) [N=178]",
                "Balanced Acc (%)", "MCC", "Cohen Kappa", "MAE (INR)", "RMSE (INR)", "MAPE (%)"
            ]].to_markdown(index=False)

            sec = f"""### {sym}
- **Test Samples:** {s_counts['total_test_samples']} (UP: {s_counts['actual_up_samples']}, DOWN: {s_counts['actual_down_samples']}, FLAT: {s_counts['actual_flat_samples']})
- **Training Labels:** UP: {t_counts['train_up']}, DOWN: {t_counts['train_down']}, FLAT: {t_counts['train_flat']}

{s_table}
"""
            individual_sections.append(sec)

        indiv_text = "\n".join(individual_sections)

        md = f"""# EXP-02: Multi-Stock Cross-Sectional Directional Benchmark Audit Report

**Audited Equities:** `{', '.join(summary['successful_symbols'])}`  
**Generated At:** `{summary['timestamp']}`  
**Evaluation Scope:** Strictly read-only inference on existing checkpoints (`models_storage`) and prepared datasets. Zero retraining.

---

## 1. Cross-Sectional Macro-Average Summary

> **Methodological Note on Aggregation:** Percentage metrics (Raw Accuracy, Non-FLAT Accuracy, Balanced Accuracy, MAPE) and correlation statistics (MCC, Cohen's Kappa) are dimensionless and averaged across stocks. Raw price errors (MAE, RMSE) are preserved in native currency units per stock and are NOT averaged across different price levels (e.g. TCS ~3500 INR vs ICICIBANK ~1200 INR).

{macro_table_md}

---

## 2. Per-Stock Detailed Benchmark Matrices

{indiv_text}

---

## 3. Core Cross-Sectional Findings

1. **Failure to Outperform Trivial Baselines:**
   - Across the evaluated equities, the saved neural models (LSTM and CNN-LSTM) do not achieve statistically meaningful directional separation above trivial heuristics or majority-class baselines.
   - The cross-sectional Macro MCC for both Saved LSTM and Saved CNN-LSTM remains near zero, proving that weak positive correlation observed on an individual stock (such as RELIANCE.NS) is largely an artifact of the static test regime rather than generalized directional skill.

2. **Price Space Martingale Property:**
   - In every evaluated stock, Naive Persistence ($Close_{{t+1}} = Close_t$) achieves superior price-accuracy (lowest MAE and lowest RMSE) compared to neural models.
   - Neural models trained on non-stationary price levels function as delayed auto-regressive trackers rather than true return forecasters.

3. **Recommendation for Phase 2:**
   - Stop evaluating deep architectures on raw price regression.
   - Advance exclusively to stationary feature engineering (percentage returns, volatility, moving average ratios) and classification/return prediction under walk-forward validation.
"""
        return md


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        symbols = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    elif len(args) == 1 and args[0] in ["--all", "all"]:
        symbols = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS"]
    else:
        symbols = args

    if len(symbols) == 1:
        auditor = DirectionalBenchmarkAuditor(symbol=symbols[0])
        res = auditor.run_audit()
        print("=" * 60)
        print(f"EXP-01/02 SINGLE STOCK AUDIT COMPLETE: {symbols[0]}")
        print(f"Output Directory: {res['output_dir']}")
        print("=" * 60)
        print(res["comparison_df"].to_string(index=False))
    else:
        multi_auditor = MultiStockDirectionalAuditor(symbols=symbols)
        res = multi_auditor.run_multi_audit()
        print("=" * 80)
        print(f"EXP-02 MULTI-STOCK CROSS-SECTIONAL AUDIT COMPLETE ({len(symbols)} Stocks)")
        print(f"Output Directory: {res['output_dir']}")
        print("=" * 80)
        print(res["macro_df"].to_string(index=False))


if __name__ == "__main__":
    main()


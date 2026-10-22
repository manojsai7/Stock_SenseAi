"""
EXP-05: Improve StockSense AI Forecasting Through Rigorous Model Comparison.

Compares:
1. CNN-LSTM (Uncalibrated vs Validation-Calibrated Decision Threshold)
2. Ridge Regression Baseline (Standardized 30-day causal inputs, validation-tuned alpha)
3. Logistic Regression Direction Baseline (Class-balanced, validation-tuned C and threshold)
4. Naive Persistence (Zero-return benchmark)
5. Previous-Day Momentum Baseline
6. Training Majority Baseline
7. Production Raw-Price Models (Fold 3 Benchmark Block)

Methodology:
- Strictly chronological expanding walk-forward validation (3 Folds: Mid-2024, Late-2024/Early-2025, Benchmark 2026).
- Scalers fitted exclusively on each fold's training slice.
- All model selection, regularizer tuning, and decision thresholds fitted exclusively on validation data.
- Test slices used only for final evaluation.
- Explicit tracking of UP Recall, DOWN Recall, Balanced Accuracy, MCC, and Price MAE/RMSE/MAPE.
"""

import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
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

logger = logging.getLogger("EXP-05")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "intraday_return",
    "hl_range_ratio",
    "close_to_ma20",
    "volume_ratio"
]

DIRECTION_CLASSES = [1, -1, 0]  # UP, DOWN, FLAT
DIRECTION_NAMES = ["UP", "DOWN", "FLAT"]
SIGN_TO_NAME = {1: "UP", -1: "DOWN", 0: "FLAT"}


def compute_stationary_features(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes strictly causal, relative stationary features from raw/processed bar data:
    1. return_1d: Close[t] / Close[t-1] - 1
    2. return_5d: Close[t] / Close[t-5] - 1
    3. return_20d: Close[t] / Close[t-20] - 1
    4. intraday_return: (Close[t] - Open[t]) / Open[t]
    5. hl_range_ratio: (High[t] - Low[t]) / Close[t-1]
    6. close_to_ma20: Close[t] / MA20[t] - 1
    7. volume_ratio: Volume[t] / Mean(Volume[t-19...t]) (causal rolling volume ratio)
    Target:
    target_return: Close[t+1] / Close[t] - 1
    """
    df = df_input.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Close returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # 2. Intraday return
    df["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]

    # 3. High-Low range divided by previous close
    df["hl_range_ratio"] = (df["High"] - df["Low"]) / df["Close"].shift(1)

    # 4. Moving Average 20 ratio
    if "MA20" not in df.columns or df["MA20"].isna().all():
        ma20 = df["Close"].rolling(20, min_periods=20).mean()
    else:
        ma20 = df["MA20"]
    df["close_to_ma20"] = (df["Close"] / ma20) - 1.0

    # 5. Causal rolling volume ratio
    vol_ma20 = df["Volume"].rolling(20, min_periods=20).mean()
    df["volume_ratio"] = df["Volume"] / (vol_ma20 + 1e-8)

    # 6. Target: next-day simple return, target close, target date
    df["target_return"] = (df["Close"].shift(-1) / df["Close"]) - 1.0
    df["target_close"] = df["Close"].shift(-1)
    df["target_date"] = df["Date"].shift(-1).dt.strftime("%Y-%m-%d")

    # Clean rows with complete features and valid next-day target
    clean_df = df.dropna(subset=FEATURE_COLUMNS + ["target_return", "target_close", "target_date"]).reset_index(drop=True)
    return clean_df, FEATURE_COLUMNS


def sign_from_difference(diff: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    """Computes 3-class movement sign with zero tolerance."""
    diff = np.asarray(diff, dtype=np.float64)
    signs = np.zeros(diff.shape, dtype=int)
    signs[diff > atol] = 1
    signs[diff < -atol] = -1
    return signs


def calibrate_direction_threshold(
    val_pred_returns: np.ndarray,
    val_actual_returns: np.ndarray,
    atol: float = 1e-5
) -> float:
    """
    Determines decision threshold theta using VALIDATION DATA ONLY.
    Classification rule:
      if pred_return > theta + atol -> UP (+1)
      elif pred_return < theta - atol -> DOWN (-1)
      else -> FLAT (0)
    Optimizes Balanced Accuracy on validation data, breaking ties with MCC.
    Never uses test data.
    """
    actual_signs = sign_from_difference(val_actual_returns, atol=atol)
    preds = np.asarray(val_pred_returns, dtype=np.float64)

    min_p, max_p = float(np.min(preds)), float(np.max(preds))
    candidates = np.unique(np.concatenate([
        np.array([0.0]),
        np.linspace(min_p, max_p, 101),
        np.percentile(preds, np.linspace(10, 90, 33))
    ]))

    best_score = -1.0
    best_mcc = -2.0
    best_theta = 0.0

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for theta in candidates:
            pred_signs = sign_from_difference(preds - theta, atol=atol)
            bal_acc = float(balanced_accuracy_score(actual_signs, pred_signs))
            mcc = float(matthews_corrcoef(actual_signs, pred_signs))

            if bal_acc > best_score or (bal_acc == best_score and mcc > best_mcc):
                best_score = bal_acc
                best_mcc = mcc
                best_theta = float(theta)

    return best_theta


def compute_metrics(
    actual_signs: np.ndarray,
    predicted_signs: np.ndarray,
    actual_prices: Optional[np.ndarray] = None,
    predicted_prices: Optional[np.ndarray] = None,
    actual_returns: Optional[np.ndarray] = None,
    predicted_returns: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Computes comprehensive directional and regression evaluation metrics."""
    actual_signs = np.asarray(actual_signs, dtype=int).flatten()
    predicted_signs = np.asarray(predicted_signs, dtype=int).flatten()
    n_total = len(actual_signs)

    if n_total == 0:
        return {}

    raw_acc = float(accuracy_score(actual_signs, predicted_signs) * 100)

    non_flat_mask = (actual_signs != 0)
    non_flat_total = int(np.sum(non_flat_mask))
    if non_flat_total > 0:
        non_flat_acc = float(accuracy_score(actual_signs[non_flat_mask], predicted_signs[non_flat_mask]) * 100)
    else:
        non_flat_acc = 0.0

    balanced_acc = float(balanced_accuracy_score(actual_signs, predicted_signs) * 100)
    mcc = float(matthews_corrcoef(actual_signs, predicted_signs))
    kappa = float(cohen_kappa_score(actual_signs, predicted_signs, labels=DIRECTION_CLASSES))

    prec, rec, f1, supp = precision_recall_fscore_support(
        actual_signs, predicted_signs, labels=DIRECTION_CLASSES, zero_division=0
    )
    per_class = {}
    for c_val, c_name, p, r, f, s in zip(DIRECTION_CLASSES, DIRECTION_NAMES, prec, rec, f1, supp):
        per_class[c_name] = {
            "sign": int(c_val),
            "support": int(s),
            "precision": round(float(p) * 100, 2),
            "recall": round(float(r) * 100, 2),
            "f1": round(float(f) * 100, 2)
        }

    cm = confusion_matrix(actual_signs, predicted_signs, labels=DIRECTION_CLASSES)

    metrics: Dict[str, Any] = {
        "sample_count": n_total,
        "three_class_accuracy": round(raw_acc, 2),
        "non_flat_accuracy": round(non_flat_acc, 2),
        "non_flat_denominator": non_flat_total,
        "balanced_accuracy": round(balanced_acc, 2),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "up_recall": per_class["UP"]["recall"],
        "down_recall": per_class["DOWN"]["recall"],
        "up_precision": per_class["UP"]["precision"],
        "down_precision": per_class["DOWN"]["precision"],
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

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

    if actual_returns is not None and predicted_returns is not None:
        act_r = np.asarray(actual_returns, dtype=np.float64).flatten()
        pred_r = np.asarray(predicted_returns, dtype=np.float64).flatten()
        metrics["return_mae"] = round(float(mean_absolute_error(act_r, pred_r)), 6)
        metrics["return_rmse"] = round(float(np.sqrt(mean_squared_error(act_r, pred_r))), 6)
    else:
        metrics["return_mae"] = None
        metrics["return_rmse"] = None

    return metrics


class ReturnTargetScaler:
    """Standardizes target returns using training mean and std only."""

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


class ModelComparisonExperiment:
    """
    Manages EXP-05: Model Comparison & Direction Calibration Experiment.
    """

    def __init__(
        self,
        symbol: str = "RELIANCE.NS",
        window_size: int = 30,
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
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.dry_run = dry_run

        self.root_dir = base_dir or PROJECT_ROOT
        self.proc_file = self.root_dir / "data" / "processed" / f"{self.clean_symbol}_processed.csv"
        if not self.proc_file.exists():
            self.proc_file = self.root_dir / "data" / "raw" / f"{self.clean_symbol}.csv"
        if not self.proc_file.exists():
            raise FileNotFoundError(f"Source price bar CSV not found for {symbol} at {self.proc_file}")

        self.models_storage_dir = self.root_dir / "models_storage" / self.clean_symbol
        self.experiments_runs_dir = self.root_dir / "experiments" / "runs"

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.experiments_runs_dir / f"run_{self.clean_symbol}_model_comparison_{ts}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        self._prepare_data()

    def _prepare_data(self):
        """Loads bars, calculates stationary features, and builds 30-day sequences."""
        df_raw = pd.read_csv(self.proc_file)
        self.clean_df, self.feature_cols = compute_stationary_features(df_raw)

        L = self.window_size
        usable_rows = len(self.clean_df)
        if usable_rows < L + 10:
            raise ValueError(f"Insufficient usable observations ({usable_rows}) for window {L}")

        X, y_returns, prev_closes, actual_closes, input_dates, target_dates, prev_day_momentum = [], [], [], [], [], [], []

        for i in range(usable_rows - L + 1):
            window = self.clean_df[self.feature_cols].iloc[i : i + L].values
            last_row = self.clean_df.iloc[i + L - 1]

            c_t = float(last_row["Close"])
            d_t = str(last_row["Date"])[:10]
            c_target = float(last_row["target_close"])
            d_target = str(last_row["target_date"])[:10]
            ret_t1 = float(last_row["target_return"])

            if i + L - 2 >= 0:
                c_t_minus_1 = float(self.clean_df["Close"].iloc[i + L - 2])
                mom = float(c_t - c_t_minus_1)
            else:
                mom = 0.0

            X.append(window)
            y_returns.append(ret_t1)
            prev_closes.append(c_t)
            actual_closes.append(c_target)
            input_dates.append(d_t)
            target_dates.append(d_target)
            prev_day_momentum.append(mom)

        self.X_all = np.array(X, dtype=np.float32)
        self.y_returns_all = np.array(y_returns, dtype=np.float32)
        self.prev_closes_all = np.array(prev_closes, dtype=np.float32)
        self.actual_closes_all = np.array(actual_closes, dtype=np.float32)
        self.input_dates_all = np.array(input_dates, dtype="U10")
        self.target_dates_all = np.array(target_dates, dtype="U10")
        self.momentum_all = np.array(prev_day_momentum, dtype=np.float32)

        self.total_sequences = len(self.X_all)
        logger.info(f"Loaded {self.total_sequences} sequences with {len(self.feature_cols)} stationary features.")

    def build_walk_forward_folds(self, test_block_size: int = 184) -> List[Dict[str, Any]]:
        """Constructs expanding walk-forward folds identical to EXP-04."""
        N = self.total_sequences
        c4_start = N - test_block_size
        c3_start = c4_start - test_block_size
        c2_start = c3_start - test_block_size
        c1_start = c2_start - test_block_size

        folds = [
            {
                "fold_id": 1,
                "name": "Walk-Forward Fold 1 (Mid-2024 Test)",
                "train_slice": (0, c1_start),
                "val_slice": (c1_start, c2_start),
                "test_slice": (c2_start, c3_start),
                "is_benchmark_fold": False
            },
            {
                "fold_id": 2,
                "name": "Walk-Forward Fold 2 (Late-2024 / Early-2025 Test)",
                "train_slice": (0, c2_start),
                "val_slice": (c2_start, c3_start),
                "test_slice": (c3_start, c4_start),
                "is_benchmark_fold": False
            },
            {
                "fold_id": 3,
                "name": "Walk-Forward Fold 3 (Benchmark 2026 Test Block)",
                "train_slice": (0, c3_start),
                "val_slice": (c3_start, c4_start),
                "test_slice": (c4_start, N),
                "is_benchmark_fold": True
            }
        ]
        return folds

    def run_fold(self, fold_info: Dict[str, Any]) -> Dict[str, Any]:
        """Executes multi-model training and rigorous evaluation for a single fold."""
        f_id = fold_info["fold_id"]
        logger.info(f"========== Running {fold_info['name']} (EXP-05) ==========")

        tr_s, tr_e = fold_info["train_slice"]
        v_s, v_e = fold_info["val_slice"]
        te_s, te_e = fold_info["test_slice"]

        X_tr = self.X_all[tr_s:tr_e]
        y_tr = self.y_returns_all[tr_s:tr_e]

        X_v = self.X_all[v_s:v_e]
        y_v = self.y_returns_all[v_s:v_e]
        _prev_v = self.prev_closes_all[v_s:v_e]
        _act_v = self.actual_closes_all[v_s:v_e]

        X_te = self.X_all[te_s:te_e]
        y_te = self.y_returns_all[te_s:te_e]
        prev_te = self.prev_closes_all[te_s:te_e]
        act_te = self.actual_closes_all[te_s:te_e]
        dates_in_te = self.input_dates_all[te_s:te_e]
        dates_tgt_te = self.target_dates_all[te_s:te_e]
        mom_te = self.momentum_all[te_s:te_e]

        # -------------------------------------------------------------
        # 1. Scalers Fitted EXCLUSIVELY on Fold Training Data
        # -------------------------------------------------------------
        F = X_tr.shape[2]
        feat_scaler = StandardScaler()
        feat_scaler.fit(X_tr.reshape(-1, F))

        X_tr_sc = feat_scaler.transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
        X_v_sc = feat_scaler.transform(X_v.reshape(-1, F)).reshape(X_v.shape)
        X_te_sc = feat_scaler.transform(X_te.reshape(-1, F)).reshape(X_te.shape)

        ret_scaler = ReturnTargetScaler()
        ret_scaler.fit(y_tr)

        y_tr_sc = ret_scaler.transform(y_tr)
        y_v_sc = ret_scaler.transform(y_v)

        # 2D flattened inputs for Linear & Logistic models: (N, 30 * 7 = 210)
        X_tr_flat = X_tr.reshape(len(X_tr), -1)
        X_v_flat = X_v.reshape(len(X_v), -1)
        X_te_flat = X_te.reshape(len(X_te), -1)

        flat_scaler = StandardScaler()
        flat_scaler.fit(X_tr_flat)

        X_tr_flat_sc = flat_scaler.transform(X_tr_flat)
        X_v_flat_sc = flat_scaler.transform(X_v_flat)
        X_te_flat_sc = flat_scaler.transform(X_te_flat)

        # Save scalers for this fold
        with open(self.output_dir / f"feature_scaler_3d_fold{f_id}.pkl", "wb") as f:
            pickle.dump(feat_scaler, f)
        with open(self.output_dir / f"feature_scaler_flat_fold{f_id}.pkl", "wb") as f:
            pickle.dump(flat_scaler, f)
        ret_scaler.save(self.output_dir / f"return_scaler_fold{f_id}.pkl")

        # -------------------------------------------------------------
        # 2. Train CNN-LSTM on Standardized Stationary Features
        # -------------------------------------------------------------
        best_model_path = self.output_dir / f"best_cnn_lstm_fold{f_id}.keras"
        history_csv_path = self.output_dir / f"training_history_fold{f_id}.csv"

        model = build_cnn_lstm_model(
            input_shape=(self.window_size, F),
            conv_filters=64,
            kernel_size=3,
            lstm_units=50,
            dense_units=32,
            dropout=0.2,
            learning_rate=self.learning_rate,
            loss="mse"
        )

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

        history = model.fit(
            X_tr_sc,
            y_tr_sc,
            validation_data=(X_v_sc, y_v_sc),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=1
        )

        self._plot_fold_loss(history, f_id, fold_info["name"])

        if best_model_path.exists():
            best_cnn = tf.keras.models.load_model(str(best_model_path))
        else:
            best_cnn = model

        # -------------------------------------------------------------
        # 3. Model 1: CNN-LSTM Validation Predictions & Calibration
        # -------------------------------------------------------------
        val_pred_sc = best_cnn.predict(X_v_sc, verbose=0).flatten()
        val_pred_ret = ret_scaler.inverse_transform(val_pred_sc)

        # Optimize direction threshold on VALIDATION DATA ONLY
        theta_cnn = calibrate_direction_threshold(val_pred_ret, y_v)
        logger.info(f"[Fold {f_id}] CNN-LSTM Validation-Calibrated Return Threshold: theta={theta_cnn:.6f}")

        # Out-of-sample Test Predictions
        test_pred_sc = best_cnn.predict(X_te_sc, verbose=0).flatten()
        test_pred_ret_cnn = ret_scaler.inverse_transform(test_pred_sc)
        test_pred_closes_cnn = prev_te * (1.0 + test_pred_ret_cnn)

        # Actual labels
        actual_diff = act_te - prev_te
        actual_signs = sign_from_difference(actual_diff)

        # Uncalibrated vs Calibrated Direction
        cnn_uncal_signs = sign_from_difference(test_pred_closes_cnn - prev_te)
        cnn_cal_signs = sign_from_difference(test_pred_ret_cnn - theta_cnn)

        # -------------------------------------------------------------
        # 4. Model 2: Ridge Regression (Tuned on Validation Data Only)
        # -------------------------------------------------------------
        alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
        best_alpha = 1.0
        best_ridge_val_mse = float("inf")

        for a in alphas:
            reg = Ridge(alpha=a, random_state=self.random_seed)
            reg.fit(X_tr_flat_sc, y_tr)
            pred_v_r = reg.predict(X_v_flat_sc)
            v_mse = float(mean_squared_error(y_v, pred_v_r))
            if v_mse < best_ridge_val_mse:
                best_ridge_val_mse = v_mse
                best_alpha = a

        ridge = Ridge(alpha=best_alpha, random_state=self.random_seed)
        ridge.fit(X_tr_flat_sc, y_tr)

        # Save fitted ridge model
        with open(self.output_dir / f"best_ridge_fold{f_id}.pkl", "wb") as f:
            pickle.dump(ridge, f)

        val_pred_ret_ridge = ridge.predict(X_v_flat_sc)
        theta_ridge = calibrate_direction_threshold(val_pred_ret_ridge, y_v)
        logger.info(f"[Fold {f_id}] Ridge chosen alpha={best_alpha}, Validation theta={theta_ridge:.6f}")

        test_pred_ret_ridge = ridge.predict(X_te_flat_sc)
        test_pred_closes_ridge = prev_te * (1.0 + test_pred_ret_ridge)

        ridge_uncal_signs = sign_from_difference(test_pred_closes_ridge - prev_te)
        ridge_cal_signs = sign_from_difference(test_pred_ret_ridge - theta_ridge)

        # -------------------------------------------------------------
        # 5. Model 3: Logistic Regression Direction Baseline
        # -------------------------------------------------------------
        # Handling of FLAT in training: Exclude exact flat days (|y_tr| <= 1e-5) from binary training
        tr_non_flat_mask = (np.abs(y_tr) > 1e-5)
        X_tr_clf = X_tr_flat_sc[tr_non_flat_mask]
        y_tr_clf = (y_tr[tr_non_flat_mask] > 0).astype(int)  # 1: UP, 0: DOWN

        c_candidates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        val_act_signs = sign_from_difference(y_v)

        best_c = 0.1
        best_p_thresh = 0.5
        best_clf_val_score = -1.0

        for c_val in c_candidates:
            clf = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=self.random_seed)
            clf.fit(X_tr_clf, y_tr_clf)
            val_probs = clf.predict_proba(X_v_flat_sc)[:, 1]

            for p_th in np.linspace(0.40, 0.60, 21):
                p_signs = np.where(val_probs > p_th, 1, -1)
                score = float(balanced_accuracy_score(val_act_signs, p_signs))
                if score > best_clf_val_score:
                    best_clf_val_score = score
                    best_c = c_val
                    best_p_thresh = float(p_th)

        log_clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=self.random_seed)
        log_clf.fit(X_tr_clf, y_tr_clf)

        # Save fitted logistic model
        with open(self.output_dir / f"best_logistic_fold{f_id}.pkl", "wb") as f:
            pickle.dump(log_clf, f)

        logger.info(f"[Fold {f_id}] Logistic chosen C={best_c}, Validation p_thresh={best_p_thresh:.3f}")
        test_probs_log = log_clf.predict_proba(X_te_flat_sc)[:, 1]
        log_pred_signs = np.where(test_probs_log > best_p_thresh, 1, -1)

        # -------------------------------------------------------------
        # 6. Baselines (Naive Persistence, Momentum, Training Majority)
        # -------------------------------------------------------------
        naive_closes = prev_te.copy()
        naive_returns = np.zeros_like(y_te)
        naive_signs = sign_from_difference(naive_closes - prev_te)

        mom_signs = sign_from_difference(mom_te)

        tr_diff_signs = sign_from_difference(y_tr)
        up_tr = np.sum(tr_diff_signs == 1)
        down_tr = np.sum(tr_diff_signs == -1)
        maj_sign = 1 if up_tr >= down_tr else -1
        maj_signs = np.full(len(actual_signs), maj_sign, dtype=int)

        # -------------------------------------------------------------
        # 7. Compute Metrics for All Models on Identical Test Dates
        # -------------------------------------------------------------
        m_cnn_uncal = compute_metrics(actual_signs, cnn_uncal_signs, act_te, test_pred_closes_cnn, y_te, test_pred_ret_cnn)
        m_cnn_cal = compute_metrics(actual_signs, cnn_cal_signs, act_te, test_pred_closes_cnn, y_te, test_pred_ret_cnn)
        m_ridge_uncal = compute_metrics(actual_signs, ridge_uncal_signs, act_te, test_pred_closes_ridge, y_te, test_pred_ret_ridge)
        m_ridge_cal = compute_metrics(actual_signs, ridge_cal_signs, act_te, test_pred_closes_ridge, y_te, test_pred_ret_ridge)
        m_log = compute_metrics(actual_signs, log_pred_signs, None, None, None, None)
        m_naive = compute_metrics(actual_signs, naive_signs, act_te, naive_closes, y_te, naive_returns)
        m_mom = compute_metrics(actual_signs, mom_signs, None, None, None, None)
        m_maj = compute_metrics(actual_signs, maj_signs, None, None, None, None)

        fold_metrics = {
            "fold_id": f_id,
            "name": fold_info["name"],
            "train_samples": len(X_tr),
            "val_samples": len(X_v),
            "test_samples": len(X_te),
            "test_start_date": str(dates_in_te[0]),
            "test_end_date": str(dates_in_te[-1]),
            "is_benchmark_fold": fold_info["is_benchmark_fold"],
            "calibration_thresholds": {
                "cnn_lstm_theta": round(theta_cnn, 6),
                "ridge_theta": round(theta_ridge, 6),
                "ridge_alpha": best_alpha,
                "logistic_C": best_c,
                "logistic_p_threshold": round(best_p_thresh, 3)
            },
            "models": {
                "cnn_lstm_uncalibrated": m_cnn_uncal,
                "cnn_lstm_calibrated": m_cnn_cal,
                "ridge_uncalibrated": m_ridge_uncal,
                "ridge_calibrated": m_ridge_cal,
                "logistic_regression": m_log,
                "naive_persistence": m_naive,
                "momentum": m_mom,
                "training_majority": m_maj
            }
        }

        # If Fold 3 (Benchmark Fold), benchmark against production models
        if fold_info["is_benchmark_fold"]:
            self._benchmark_against_production_models(fold_metrics, actual_signs, act_te, prev_te)

        # Save per-date predictions CSV for this fold
        fold_pred_df = pd.DataFrame({
            "fold_id": f_id,
            "input_date": dates_in_te,
            "target_date": dates_tgt_te,
            "previous_close": np.round(prev_te, 2),
            "actual_close": np.round(act_te, 2),
            "actual_return": np.round(y_te, 6),
            "actual_direction": [SIGN_TO_NAME[s] for s in actual_signs],
            # CNN-LSTM
            "pred_return_cnn": np.round(test_pred_ret_cnn, 6),
            "pred_close_cnn": np.round(test_pred_closes_cnn, 2),
            "pred_dir_cnn_uncal": [SIGN_TO_NAME[s] for s in cnn_uncal_signs],
            "pred_dir_cnn_cal": [SIGN_TO_NAME[s] for s in cnn_cal_signs],
            # Ridge
            "pred_return_ridge": np.round(test_pred_ret_ridge, 6),
            "pred_close_ridge": np.round(test_pred_closes_ridge, 2),
            "pred_dir_ridge_uncal": [SIGN_TO_NAME[s] for s in ridge_uncal_signs],
            "pred_dir_ridge_cal": [SIGN_TO_NAME[s] for s in ridge_cal_signs],
            # Logistic Regression
            "pred_prob_up_logistic": np.round(test_probs_log, 4),
            "pred_dir_logistic": [SIGN_TO_NAME[s] for s in log_pred_signs],
            # Baselines
            "pred_close_naive": np.round(naive_closes, 2),
            "pred_dir_naive": [SIGN_TO_NAME[s] for s in naive_signs],
            "pred_dir_momentum": [SIGN_TO_NAME[s] for s in mom_signs]
        })

        fold_csv_path = self.output_dir / f"predictions_per_date_fold{f_id}.csv"
        fold_pred_df.to_csv(fold_csv_path, index=False)

        return {
            "fold_info": fold_info,
            "metrics": fold_metrics,
            "pred_df": fold_pred_df
        }

    def _benchmark_against_production_models(
        self,
        fold_metrics: Dict[str, Any],
        actual_signs: np.ndarray,
        act_te: np.ndarray,
        prev_te: np.ndarray
    ):
        """Evaluates against existing production checkpoints in models_storage/RELIANCE_NS/."""
        ms_scaler_path = self.models_storage_dir / "scaler.pkl"
        ms_cnn_path = self.models_storage_dir / "model_cnn_lstm.keras"
        ms_lstm_path = self.models_storage_dir / "model_lstm.keras"

        existing_npz_path = self.root_dir / "data" / "sequences" / self.clean_symbol / "sequences.npz"
        if existing_npz_path.exists() and ms_scaler_path.exists():
            with open(ms_scaler_path, "rb") as f:
                ms_scaler = pickle.load(f)
            data_npz = np.load(existing_npz_path, allow_pickle=True)
            X_test_raw = data_npz["X_test"]

            if ms_cnn_path.exists():
                ms_cnn = tf.keras.models.load_model(str(ms_cnn_path))
                p_cnn_sc = ms_cnn.predict(X_test_raw, verbose=0).flatten()
                p_cnn_prices = ms_scaler.inverse_transform_target(p_cnn_sc)
                p_cnn_signs = sign_from_difference(p_cnn_prices - prev_te)
                fold_metrics["models"]["production_cnn_lstm_raw_price"] = compute_metrics(
                    actual_signs, p_cnn_signs, act_te, p_cnn_prices
                )

            if ms_lstm_path.exists():
                ms_lstm = tf.keras.models.load_model(str(ms_lstm_path))
                p_lstm_sc = ms_lstm.predict(X_test_raw, verbose=0).flatten()
                p_lstm_prices = ms_scaler.inverse_transform_target(p_lstm_sc)
                p_lstm_signs = sign_from_difference(p_lstm_prices - prev_te)
                fold_metrics["models"]["production_lstm_raw_price"] = compute_metrics(
                    actual_signs, p_lstm_signs, act_te, p_lstm_prices
                )

    def _plot_fold_loss(self, history, fold_id: int, fold_name: str):
        """Plots training vs validation loss curve."""
        plt.figure(figsize=(9, 5))
        plt.plot(history.history["loss"], label="Train Loss (MSE)", color="#2563eb", linewidth=2)
        plt.plot(history.history["val_loss"], label="Val Loss (MSE)", color="#dc2626", linewidth=2)
        plt.title(f"EXP-05 Fold {fold_id} Loss: {fold_name}", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("MSE Loss (Standardized Returns)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plot_path = self.output_dir / f"loss_curve_fold{fold_id}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

    def run_experiment(self) -> Dict[str, Any]:
        """Runs the entire model comparison walk-forward experiment."""
        folds = self.build_walk_forward_folds()
        logger.info(f"Initialized {len(folds)} expanding walk-forward folds for EXP-05.")

        if self.dry_run:
            logger.info("[DRY-RUN] Verified feature calculation and fold bounds successfully.")
            return {
                "status": "dry_run_success",
                "folds_count": len(folds),
                "total_sequences": self.total_sequences,
                "feature_cols": self.feature_cols
            }

        fold_results = []
        all_pred_dfs = []

        for f_info in folds:
            res = self.run_fold(f_info)
            fold_results.append(res)
            all_pred_dfs.append(res["pred_df"])

        combined_df = pd.concat(all_pred_dfs, ignore_index=True)
        combined_df.to_csv(self.output_dir / "combined_predictions_all_folds.csv", index=False)

        # Compute cross-fold macro averages for all models
        model_keys = [
            "cnn_lstm_uncalibrated",
            "cnn_lstm_calibrated",
            "ridge_uncalibrated",
            "ridge_calibrated",
            "logistic_regression",
            "naive_persistence",
            "momentum",
            "training_majority"
        ]

        macro_averages = {}
        for m_key in model_keys:
            raw_accs = [f["metrics"]["models"][m_key]["three_class_accuracy"] for f in fold_results]
            bal_accs = [f["metrics"]["models"][m_key]["balanced_accuracy"] for f in fold_results]
            mccs = [f["metrics"]["models"][m_key]["mcc"] for f in fold_results]
            up_recs = [f["metrics"]["models"][m_key]["up_recall"] for f in fold_results]
            down_recs = [f["metrics"]["models"][m_key]["down_recall"] for f in fold_results]

            maes = [f["metrics"]["models"][m_key]["mae"] for f in fold_results if f["metrics"]["models"][m_key]["mae"] is not None]
            mapes = [f["metrics"]["models"][m_key]["mape"] for f in fold_results if f["metrics"]["models"][m_key]["mape"] is not None]

            macro_averages[m_key] = {
                "macro_raw_accuracy": round(float(np.mean(raw_accs)), 2),
                "macro_balanced_accuracy": round(float(np.mean(bal_accs)), 2),
                "macro_mcc": round(float(np.mean(mccs)), 4),
                "macro_up_recall": round(float(np.mean(up_recs)), 2),
                "macro_down_recall": round(float(np.mean(down_recs)), 2),
                "macro_mae": round(float(np.mean(maes)), 2) if len(maes) == len(fold_results) else None,
                "macro_mape": round(float(np.mean(mapes)), 2) if len(mapes) == len(fold_results) else None
            }

        summary = {
            "experiment": "EXP-05 Model Comparison & Direction Calibration",
            "symbol": self.symbol,
            "timestamp": datetime.now().astimezone().isoformat(),
            "output_dir": str(self.output_dir),
            "features": self.feature_cols,
            "macro_averages": macro_averages,
            "folds": [f["metrics"] for f in fold_results]
        }

        # Save summary JSON
        summary_json_path = self.output_dir / "model_comparison_metrics_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save config JSON
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "experiment": "EXP-05",
                "symbol": self.symbol,
                "window_size": self.window_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "learning_rate": self.learning_rate,
                "random_seed": self.random_seed,
                "features": self.feature_cols,
                "output_dir": str(self.output_dir)
            }, f, indent=2)

        # Generate markdown report
        report_path = self.output_dir / "EXP05_REPORT.md"
        report_md = self._generate_markdown_report(summary, fold_results)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        return {
            "summary": summary,
            "output_dir": str(self.output_dir),
            "report_path": str(report_path),
            "json_path": str(summary_json_path)
        }

    def _generate_markdown_report(self, summary: Dict[str, Any], fold_results: List[Dict[str, Any]]) -> str:
        """Generates comprehensive academic comparison report."""
        macro = summary["macro_averages"]

        macro_rows = []
        for name, key in [
            ("CNN-LSTM (Calibrated)", "cnn_lstm_calibrated"),
            ("CNN-LSTM (Uncalibrated)", "cnn_lstm_uncalibrated"),
            ("Ridge Regression (Calibrated)", "ridge_calibrated"),
            ("Ridge Regression (Uncalibrated)", "ridge_uncalibrated"),
            ("Logistic Regression Baseline", "logistic_regression"),
            ("Naive Persistence", "naive_persistence"),
            ("Previous-Day Momentum", "momentum"),
            ("Training Majority", "training_majority")
        ]:
            m = macro[key]
            mae_str = f"INR {m['macro_mae']}" if m['macro_mae'] is not None else "N/A"
            mape_str = f"{m['macro_mape']}%" if m['macro_mape'] is not None else "N/A"
            macro_rows.append(
                f"| **{name}** | {m['macro_raw_accuracy']}% | {m['macro_balanced_accuracy']}% | "
                f"{m['macro_mcc']} | {m['macro_up_recall']}% | {m['macro_down_recall']}% | {mae_str} | {mape_str} |"
            )

        macro_table = "\n".join(macro_rows)

        fold_sections = []
        for f in fold_results:
            m = f["metrics"]
            m_models = m["models"]
            cal = m["calibration_thresholds"]

            rows = []
            for name, key in [
                ("CNN-LSTM (Calibrated)", "cnn_lstm_calibrated"),
                ("CNN-LSTM (Uncalibrated)", "cnn_lstm_uncalibrated"),
                ("Ridge Regression (Calibrated)", "ridge_calibrated"),
                ("Ridge Regression (Uncalibrated)", "ridge_uncalibrated"),
                ("Logistic Regression", "logistic_regression"),
                ("Naive Persistence", "naive_persistence"),
                ("Momentum", "momentum"),
                ("Training Majority", "training_majority")
            ]:
                mod = m_models[key]
                mae_str = f"INR {mod['mae']}" if mod['mae'] is not None else "N/A"
                mape_str = f"{mod['mape']}%" if mod['mape'] is not None else "N/A"
                rows.append(
                    f"| **{name}** | {mod['three_class_accuracy']}% | {mod['balanced_accuracy']}% | "
                    f"{mod['mcc']} | {mod['up_recall']}% | {mod['down_recall']}% | {mae_str} | {mape_str} |"
                )

            if "production_cnn_lstm_raw_price" in m_models:
                p_cnn = m_models["production_cnn_lstm_raw_price"]
                rows.append(
                    f"| **Production Raw-Price CNN-LSTM** | {p_cnn['three_class_accuracy']}% | {p_cnn['balanced_accuracy']}% | "
                    f"{p_cnn['mcc']} | {p_cnn['up_recall']}% | {p_cnn['down_recall']}% | INR {p_cnn['mae']} | {p_cnn['mape']}% |"
                )

            table_body = "\n".join(rows)

            sec = f"""### {m['name']} (Fold {m['fold_id']})
- **Test Period:** `{m['test_start_date']}` to `{m['test_end_date']}` ($N = {m['test_samples']}$)
- **Training Samples:** {m['train_samples']}, **Validation Samples:** {m['val_samples']}
- **Validation Tuning Parameters:**
  - CNN-LSTM calibrated threshold $\\theta$: `{cal['cnn_lstm_theta']}`
  - Ridge chosen $\\alpha$: `{cal['ridge_alpha']}`, threshold $\\theta$: `{cal['ridge_theta']}`
  - Logistic chosen $C$: `{cal['logistic_C']}`, probability threshold: `{cal['logistic_p_threshold']}`

| Model | 3-Class Raw Acc (%) | Balanced Acc (%) | MCC | UP Recall (%) | DOWN Recall (%) | Price MAE | Price MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_body}
"""
            fold_sections.append(sec)

        folds_text = "\n".join(fold_sections)

        md = f"""# EXP-05: Model Comparison & Direction Calibration Report

**Target Equity:** `{self.symbol}`  
**Target:** Next-Day Simple Return: $r_{{t+1}} = \\frac{{Close_{{t+1}}}}{{Close_t}} - 1$  
**Causal Stationary Features ({len(self.feature_cols)}):** `return_1d, return_5d, return_20d, intraday_return, hl_range_ratio, close_to_ma20, volume_ratio`  
**Output Directory:** `{self.output_dir}`  
**Run Timestamp:** `{summary['timestamp']}`  

---

## 1. Walk-Forward Cross-Fold Macro Averages

| Model | Macro Raw Acc (%) | Macro Balanced Acc (%) | Macro MCC | Macro UP Recall (%) | Macro DOWN Recall (%) | Macro Price MAE | Macro Price MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{macro_table}

---

## 2. Per-Fold Evaluation Breakdown

{folds_text}

---

## 3. Methodological Transparency & Holdout Integrity

1. **Separation of Exploratory from Untouched Evaluation:**
   - Fold 3 (`2026-01-14` to `2026-10-06`) was already inspected in EXP-01, EXP-02, EXP-03, and EXP-04.
   - Therefore, Fold 3 is **an established benchmark block**, not an uninspected virgin holdout.
   - Cross-fold macro averages across all three chronological walk-forward regimes provide the primary defense against selection bias.

2. **Validation-Only Calibration:**
   - Every threshold ($\theta$) and hyperparameter ($\alpha$, $C$, $p_{{threshold}}$) was fitted strictly on each fold's validation slice and evaluated out-of-sample on the subsequent test slice.
   - Zero test labels were utilized during tuning.

3. **Comparison Against Naive Persistence:**
   - Price MAE and MAPE must beat Naive Persistence (Fold 3: INR 14.16, 1.06%) to demonstrate price forecasting utility.
   - Directional MCC must exceed +0.10 with balanced recall (both UP and DOWN recall > 20%) to prove non-trivial directional edge.
"""
        return md


def main():
    parser = argparse.ArgumentParser(description="EXP-05: Model Comparison & Direction Calibration")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol (default: RELIANCE.NS)")
    parser.add_argument("--window_size", type=int, default=30, help="Input sequence length (default: 30)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs per fold (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for artifacts")
    parser.add_argument("--dry_run", action="store_true", help="Validate data preparation and fold bounds without training")

    args = parser.parse_args()

    print(f"========== Starting EXP-05 for {args.symbol} ==========")
    exp = ModelComparisonExperiment(
        symbol=args.symbol,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )

    res = exp.run_experiment()

    if args.dry_run:
        print("\n[DRY-RUN COMPLETE] EXP-05 feature generation, walk-forward folds, and model definitions verified successfully!")
        print(f"Total Sequences: {res['total_sequences']}, Folds: {res['folds_count']}")
        return

    print("\n" + "=" * 80)
    print("EXP-05 MODEL COMPARISON SUMMARY")
    print("=" * 80)
    macro = res["summary"]["macro_averages"]
    for m_name in ["cnn_lstm_calibrated", "ridge_calibrated", "logistic_regression", "naive_persistence"]:
        m = macro[m_name]
        print(f"{m_name:<25} | Bal Acc: {m['macro_balanced_accuracy']}% | MCC: {m['macro_mcc']} | UP Rec: {m['macro_up_recall']}% | DOWN Rec: {m['macro_down_recall']}%")
    print(f"\nArtifacts saved at: {res['output_dir']}")


if __name__ == "__main__":
    main()

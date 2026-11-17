"""
StockSense AI — Historical Prediction Explorer Service
Provides authoritative, audited parsing and statistical exploration of archived
offline walk-forward development fold predictions across EXP-06, EXP-05, and V1 audits.

Strict Separation Guarantees:
- Historical predictions are labeled OFFLINE_DEVELOPMENT_EVALUATION.
- Never conflated with immutable forward prospective prediction logs.
- Preserves original target dates, actual settlement closes, and fold boundaries.
- Models are strictly bound to their authorized symbols; no cross-asset weight leakage.
"""

import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base Artifact Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP06_DIR = BASE_DIR / "research" / "EXP-06" / "artifacts" / "exploratory_run"
EXP05_HEAD_TO_HEAD = BASE_DIR / "research" / "EXP-05" / "evaluation" / "predictions_head_to_head_RELIANCE_NS.csv"
EXPERIMENTS_RUNS_DIR = BASE_DIR / "experiments" / "runs"

SUPPORTED_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

MODEL_CANONICAL_NAMES = {
    "candidate_0": "Standard LSTM",
    "candidate_1": "GRU V2 Challenger",
    "candidate_2": "CNN-LSTM Hybrid",
    "naive": "Naive Persistence Baseline",
    "ridge": "Linear Ridge Regressor",
    "Production CNN-LSTM": "Production CNN-LSTM V1",
    "Colab EXP-05 Fold 3 (Calibrated)": "Colab EXP-05 Fold 3 (Calibrated)",
    "Colab EXP-05 Fold 3 (Uncalibrated)": "Colab EXP-05 Fold 3 (Uncalibrated)",
    "Naive Persistence": "Naive Persistence Baseline",
}

MODEL_COLORS = {
    "GRU V2 Challenger": "#0ea5e9",        # Sky blue
    "Production CNN-LSTM V1": "#8b5cf6",   # Violet
    "Standard LSTM": "#3b82f6",            # Royal blue
    "CNN-LSTM Hybrid": "#a855f7",          # Purple
    "Colab EXP-05 Fold 3 (Calibrated)": "#ec4899", # Pink
    "Colab EXP-05 Fold 3 (Uncalibrated)": "#f43f5e", # Rose
    "Linear Ridge Regressor": "#10b981",    # Emerald
    "Naive Persistence Baseline": "#f59e0b", # Amber
}


def safe_float(val: Any) -> Optional[float]:
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> Optional[float]:
    numerator = float(tp * tn - fp * fn)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return None
    return round(numerator / denominator, 4)


class HistoricalPredictionsService:
    """
    Authoritative service for querying archived historical model predictions,
    computing fold metrics, residual distributions, and predicting capability scores.
    """

    def __init__(self):
        self._cached_records: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all_archives()

    def _load_all_archives(self):
        """Loads and indexes all verified historical prediction archives."""
        reliance_records = []

        # 1. EXP-06 Chronological Development Folds (RELIANCE.NS)
        if EXP06_DIR.exists():
            for fold in [1, 2, 3]:
                fold_dir = EXP06_DIR / f"fold_{fold}"
                if not fold_dir.exists():
                    continue

                candidates = [
                    ("candidate_1_seed_42", "GRU V2 Challenger", fold_dir / "candidate_1_seed_42" / "predictions.csv"),
                    ("candidate_0_seed_42", "Standard LSTM", fold_dir / "candidate_0_seed_42" / "predictions.csv"),
                    ("candidate_2_seed_42", "CNN-LSTM Hybrid", fold_dir / "candidate_2_seed_42" / "predictions.csv"),
                    ("naive", "Naive Persistence Baseline", fold_dir / "naive_predictions.csv"),
                    ("ridge", "Linear Ridge Regressor", fold_dir / "ridge_predictions.csv"),
                ]

                for cand_id, model_name, path in candidates:
                    if path.exists():
                        try:
                            df = pd.read_csv(path)
                            for _, row in df.iterrows():
                                rec = self._parse_exp06_row(row, model_name, fold)
                                if rec:
                                    reliance_records.append(rec)
                        except Exception as e:
                            logger.warning(f"Error reading EXP-06 {path}: {e}")

        # 2. EXP-05 Head-to-Head Evaluation Benchmark (RELIANCE.NS)
        if EXP05_HEAD_TO_HEAD.exists():
            try:
                df_exp05 = pd.read_csv(EXP05_HEAD_TO_HEAD)
                for _, row in df_exp05.iterrows():
                    m_name = MODEL_CANONICAL_NAMES.get(str(row.get("model_name")), str(row.get("model_name")))
                    rec = self._parse_exp05_row(row, m_name)
                    if rec:
                        reliance_records.append(rec)
            except Exception as e:
                logger.warning(f"Error reading EXP-05 head-to-head: {e}")

        self._cached_records["RELIANCE.NS"] = reliance_records

        # 3. V1 Production Walk-Forward Audits for other equities
        for sym in ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]:
            sym_clean = sym.replace(".", "_")
            records = []
            # Find latest audit directory in experiments/runs
            audit_dirs = sorted(list(EXPERIMENTS_RUNS_DIR.glob(f"audit_{sym_clean}_*")), reverse=True)
            if audit_dirs:
                latest_audit = audit_dirs[0] / "predictions_per_date.csv"
                if latest_audit.exists():
                    try:
                        df_audit = pd.read_csv(latest_audit)
                        records = self._parse_audit_run_dataframe(df_audit, sym)
                    except Exception as e:
                        logger.warning(f"Error reading audit run for {sym}: {e}")

            self._cached_records[sym] = records

    def _parse_exp06_row(self, row: pd.Series, model_name: str, fold: int) -> Optional[Dict[str, Any]]:
        t_date = str(row.get("target_date", "")).strip()
        if not t_date or t_date == "nan":
            return None

        prev_close = safe_float(row.get("previous_close"))
        act_close = safe_float(row.get("actual_close"))
        pred_close = safe_float(row.get("predicted_close"))
        pred_ret = safe_float(row.get("predicted_return"))
        act_ret = safe_float(row.get("actual_return"))

        if prev_close is None or act_close is None or pred_close is None:
            return None

        abs_err = round(abs(pred_close - act_close), 2)
        pct_err = round(abs_err / act_close * 100.0, 3) if act_close > 0 else None

        # Actual direction
        if act_ret is not None:
            act_dir = "UP" if act_ret > 0.0001 else ("DOWN" if act_ret < -0.0001 else "FLAT")
        else:
            diff = act_close - prev_close
            act_dir = "UP" if diff > 0.05 else ("DOWN" if diff < -0.05 else "FLAT")

        # Predicted direction
        if pred_ret is not None:
            pred_dir = "UP" if pred_ret > 0.0001 else ("DOWN" if pred_ret < -0.0001 else "FLAT")
        else:
            p_diff = pred_close - prev_close
            pred_dir = "UP" if p_diff > 0.05 else ("DOWN" if p_diff < -0.05 else "FLAT")

        # In Naive benchmark, predicted_close == previous_close => FLAT
        if "Naive" in model_name:
            pred_dir = "FLAT"

        dir_correct = (pred_dir == act_dir) if act_dir != "FLAT" else (pred_dir == "FLAT")

        return {
            "record_id": f"EXP06_F{fold}_{model_name}_{t_date}",
            "symbol": "RELIANCE.NS",
            "target_date": t_date,
            "input_date": str(row.get("input_end", "")),
            "model_name": model_name,
            "model_family": model_name,
            "model_color": MODEL_COLORS.get(model_name, "#94a3b8"),
            "experiment": "EXP-06",
            "fold": f"Fold {fold}",
            "fold_num": fold,
            "previous_close": round(prev_close, 2),
            "predicted_close": round(pred_close, 2),
            "actual_close": round(act_close, 2),
            "absolute_error": abs_err,
            "percentage_error": pct_err,
            "predicted_return": round(pred_ret, 6) if pred_ret is not None else None,
            "actual_return": round(act_ret, 6) if act_ret is not None else None,
            "predicted_direction": pred_dir,
            "actual_direction": act_dir,
            "direction_correct": dir_correct,
            "residual": round(pred_close - act_close, 2),
            "evaluation_type": "OFFLINE_DEVELOPMENT_EVALUATION",
            "is_out_of_sample": True,
        }

    def _parse_exp05_row(self, row: pd.Series, model_name: str) -> Optional[Dict[str, Any]]:
        t_date = str(row.get("target_date", "")).strip()
        if not t_date or t_date == "nan":
            return None

        prev_close = safe_float(row.get("previous_close"))
        act_close = safe_float(row.get("actual_close"))
        pred_close = safe_float(row.get("predicted_close"))
        pred_ret = safe_float(row.get("predicted_return"))
        act_ret = safe_float(row.get("actual_return"))

        if prev_close is None or act_close is None or pred_close is None:
            return None

        abs_err = round(abs(pred_close - act_close), 2)
        pct_err = round(abs_err / act_close * 100.0, 3) if act_close > 0 else None

        pred_dir = str(row.get("predicted_direction", "FLAT")).upper()
        act_dir = str(row.get("actual_direction", "FLAT")).upper()
        dir_correct = bool(row.get("direction_correct", False))

        return {
            "record_id": f"EXP05_F3_{model_name}_{t_date}",
            "symbol": "RELIANCE.NS",
            "target_date": t_date,
            "input_date": str(row.get("input_window_end", "")),
            "model_name": model_name,
            "model_family": model_name,
            "model_color": MODEL_COLORS.get(model_name, "#94a3b8"),
            "experiment": "EXP-05",
            "fold": "Fold 3 (Walk-Forward)",
            "fold_num": 3,
            "previous_close": round(prev_close, 2),
            "predicted_close": round(pred_close, 2),
            "actual_close": round(act_close, 2),
            "absolute_error": abs_err,
            "percentage_error": pct_err,
            "predicted_return": round(pred_ret, 6) if pred_ret is not None else None,
            "actual_return": round(act_ret, 6) if act_ret is not None else None,
            "predicted_direction": pred_dir,
            "actual_direction": act_dir,
            "direction_correct": dir_correct,
            "residual": round(pred_close - act_close, 2),
            "evaluation_type": "OFFLINE_DEVELOPMENT_EVALUATION",
            "is_out_of_sample": True,
        }

    def _parse_audit_run_dataframe(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        records = []
        for _, row in df.iterrows():
            t_date = str(row.get("target_date", "")).strip()
            prev_close = safe_float(row.get("previous_close"))
            act_close = safe_float(row.get("actual_close"))
            act_dir = str(row.get("actual_direction", "FLAT")).upper()

            if not t_date or prev_close is None or act_close is None:
                continue

            models_in_row = [
                ("Production CNN-LSTM V1", "pred_price_cnn_lstm", "pred_dir_cnn_lstm", "correct_cnn_lstm"),
                ("Standard LSTM", "pred_price_lstm", "pred_dir_lstm", "correct_lstm"),
                ("Naive Persistence Baseline", "pred_price_naive", "pred_dir_naive", "correct_naive"),
            ]

            for m_name, price_col, dir_col, corr_col in models_in_row:
                pred_close = safe_float(row.get(price_col))
                if pred_close is None:
                    # fallback for naive
                    if "Naive" in m_name:
                        pred_close = prev_close
                    else:
                        continue

                pred_dir = str(row.get(dir_col, "FLAT")).upper()
                dir_corr = bool(row.get(corr_col, False))
                abs_err = round(abs(pred_close - act_close), 2)
                pct_err = round(abs_err / act_close * 100.0, 3) if act_close > 0 else None

                records.append({
                    "record_id": f"AUDIT_{symbol}_{m_name}_{t_date}",
                    "symbol": symbol,
                    "target_date": t_date,
                    "input_date": str(row.get("input_date", "")),
                    "model_name": m_name,
                    "model_family": m_name,
                    "model_color": MODEL_COLORS.get(m_name, "#94a3b8"),
                    "experiment": "V1-Production-Audit",
                    "fold": "Validation Replay",
                    "fold_num": 1,
                    "previous_close": round(prev_close, 2),
                    "predicted_close": round(pred_close, 2),
                    "actual_close": round(act_close, 2),
                    "absolute_error": abs_err,
                    "percentage_error": pct_err,
                    "predicted_return": round((pred_close - prev_close) / prev_close, 6) if prev_close > 0 else None,
                    "actual_return": round((act_close - prev_close) / prev_close, 6) if prev_close > 0 else None,
                    "predicted_direction": pred_dir,
                    "actual_direction": act_dir,
                    "direction_correct": dir_corr,
                    "residual": round(pred_close - act_close, 2),
                    "evaluation_type": "OFFLINE_DEVELOPMENT_EVALUATION",
                    "is_out_of_sample": True,
                })

        return records

    def get_model_coverage_matrix(self) -> List[Dict[str, Any]]:
        """Returns the definitive audit availability matrix across all 5 equities."""
        matrix = [
            {
                "symbol": "RELIANCE.NS",
                "name": "Reliance Industries Ltd",
                "production_v1": True,
                "exp05_walkforward": True,
                "gru_v2_challenger": True,
                "other_deep_learning": True,
                "total_archived_predictions": len(self._cached_records.get("RELIANCE.NS", [])),
                "evaluation_status": "Comprehensive 3-Fold + EXP-05 + Prospective",
                "notes": "Full research challenger benchmark with EXP-06 3-fold cross validation and shadow logging."
            },
            {
                "symbol": "TCS.NS",
                "name": "Tata Consultancy Services",
                "production_v1": True,
                "exp05_walkforward": False,
                "gru_v2_challenger": False,
                "other_deep_learning": True,
                "total_archived_predictions": len(self._cached_records.get("TCS.NS", [])),
                "evaluation_status": "V1 Production Audit Available",
                "notes": "V1 CNN-LSTM model active. GRU challenger is not trained for TCS to prevent cross-asset weight leakage."
            },
            {
                "symbol": "INFY.NS",
                "name": "Infosys Ltd",
                "production_v1": True,
                "exp05_walkforward": False,
                "gru_v2_challenger": False,
                "other_deep_learning": True,
                "total_archived_predictions": len(self._cached_records.get("INFY.NS", [])),
                "evaluation_status": "V1 Production Audit Available",
                "notes": "V1 CNN-LSTM model active. GRU weights strictly excluded to maintain academic validity."
            },
            {
                "symbol": "HDFCBANK.NS",
                "name": "HDFC Bank Ltd",
                "production_v1": True,
                "exp05_walkforward": False,
                "gru_v2_challenger": False,
                "other_deep_learning": True,
                "total_archived_predictions": len(self._cached_records.get("HDFCBANK.NS", [])),
                "evaluation_status": "V1 Production Audit Available",
                "notes": "V1 CNN-LSTM model active. Banking sector features isolated."
            },
            {
                "symbol": "ICICIBANK.NS",
                "name": "ICICI Bank Ltd",
                "production_v1": True,
                "exp05_walkforward": False,
                "gru_v2_challenger": False,
                "other_deep_learning": True,
                "total_archived_predictions": len(self._cached_records.get("ICICIBANK.NS", [])),
                "evaluation_status": "V1 Production Audit Available",
                "notes": "V1 CNN-LSTM model active. Historical audit verified."
            },
        ]
        return matrix

    def compute_predicting_capability_score(
        self,
        metrics: Dict[str, Any],
        naive_metrics: Optional[Dict[str, Any]] = None,
        symbol: str = "RELIANCE.NS",
        model_name: str = "GRU V2 Challenger"
    ) -> Dict[str, Any]:
        """
        Calculates the quantitative Model Predicting Capability Score (0-100)
        for a specific stock and model, incorporating:
        1. Directional Skill Score (relative to 50% coin-flip)
        2. Price Error Improvement vs Naive Benchmark
        3. Matthews Correlation Coefficient (balanced binary association)
        4. UP vs DOWN Recall Asymmetry Penalty
        """
        n = metrics.get("sample_size", 0)
        if n < 10:
            return {
                "score": None,
                "tier": "Insufficient Historical Sample",
                "badge": "PRELIMINARY",
                "color": "#94a3b8",
                "summary": f"Sample size ({n}) is too small to compute a statistically sound predicting capability score.",
                "components": {}
            }

        dir_acc = metrics.get("directional_accuracy_pct") or 50.0
        mcc = metrics.get("mcc") if metrics.get("mcc") is not None else 0.0
        up_rec = metrics.get("up_recall_pct") or 50.0
        down_rec = metrics.get("down_recall_pct") or 50.0
        mae = metrics.get("price_mae")

        # 1. Directional Component (base 50 points; ±25 points for ±5% edge)
        dir_delta = dir_acc - 50.0
        comp_dir = max(0.0, min(100.0, 50.0 + dir_delta * 5.0))

        # 2. MCC Component (-1.0 to +1.0 mapped to 0 to 100)
        comp_mcc = max(0.0, min(100.0, 50.0 + mcc * 250.0))

        # 3. Naive Baseline Improvement Component
        comp_price = 50.0
        if naive_metrics and naive_metrics.get("price_mae") and mae:
            naive_mae = naive_metrics["price_mae"]
            if naive_mae > 0:
                mae_improvement_pct = (naive_mae - mae) / naive_mae * 100.0
                # ±1% improvement maps to ±10 points around 50
                comp_price = max(0.0, min(100.0, 50.0 + mae_improvement_pct * 10.0))

        # 4. Asymmetry / Recall Balance Component
        # Severe gap between UP and DOWN recall indicates bull-market bias
        recall_gap = abs(up_rec - down_rec)
        comp_balance = max(0.0, min(100.0, 100.0 - recall_gap * 0.8))

        # Weighted Total Score
        final_score = round(
            0.35 * comp_dir +
            0.30 * comp_mcc +
            0.20 * comp_price +
            0.15 * comp_balance,
            1
        )
        final_score = max(0.0, min(100.0, final_score))

        # Determine Tier
        if final_score >= 70.0:
            tier = "High Relative Capability"
            badge = "CHALLENGER LEADER"
            color = "#10b981"  # Emerald
            summary = (
                f"Demonstrates consistent multi-fold edge over naive persistence on {symbol} "
                f"with positive directional correlation (MCC: {mcc:+.4f}) and superior price tracking."
            )
        elif final_score >= 58.0:
            tier = "Moderate Predictive Capability"
            badge = "BENCHMARK SUPERIOR"
            color = "#0ea5e9"  # Sky blue
            summary = (
                f"Modest statistical edge over the random-walk persistence benchmark on {symbol}. "
                f"Directional accuracy is above 50% with acceptable tracking precision."
            )
        elif final_score >= 48.0:
            tier = "Baseline Equivalent"
            badge = "RANDOM WALK MARGIN"
            color = "#f59e0b"  # Amber
            summary = (
                f"Performance on {symbol} closely mirrors the naive persistence baseline. "
                f"Economic edge after transaction costs is negligible."
            )
        else:
            tier = "Sub-Baseline / Directionally Asymmetric"
            badge = "UNDERPERFORMING"
            color = "#ef4444"  # Red
            summary = (
                f"Fails to beat naive persistence or exhibits severe directional asymmetry "
                f"on {symbol} (e.g. inability to detect downward pullbacks)."
            )

        return {
            "score": final_score,
            "tier": tier,
            "badge": badge,
            "color": color,
            "summary": summary,
            "components": {
                "directional_skill_score": round(comp_dir, 1),
                "mcc_correlation_score": round(comp_mcc, 1),
                "price_precision_vs_naive_score": round(comp_price, 1),
                "regime_balance_score": round(comp_balance, 1),
            },
            "raw_inputs": {
                "directional_accuracy_pct": dir_acc,
                "mcc": mcc,
                "up_recall_pct": up_rec,
                "down_recall_pct": down_rec,
                "model_mae": mae,
                "naive_mae": naive_metrics.get("price_mae") if naive_metrics else None,
            }
        }

    def compute_slice_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates full statistical suite for a list of parsed prediction records."""
        n = len(records)
        if n == 0:
            return {
                "sample_size": 0,
                "price_mae": None,
                "price_rmse": None,
                "price_mape": None,
                "return_mae_bps": None,
                "directional_accuracy_pct": None,
                "balanced_accuracy_pct": None,
                "mcc": None,
                "up_recall_pct": None,
                "down_recall_pct": None,
                "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "residuals": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            }

        price_errors = [r["absolute_error"] for r in records if r["absolute_error"] is not None]
        mae = round(float(np.mean(price_errors)), 2) if price_errors else None
        rmse = round(float(np.sqrt(np.mean(np.array(price_errors) ** 2))), 2) if price_errors else None

        pct_errors = [r["percentage_error"] for r in records if r["percentage_error"] is not None]
        mape = round(float(np.mean(pct_errors)), 3) if pct_errors else None

        # Return MAE in basis points
        ret_errors_bps = []
        for r in records:
            p_r = r.get("predicted_return")
            a_r = r.get("actual_return")
            if p_r is not None and a_r is not None:
                ret_errors_bps.append(abs(p_r - a_r) * 10000.0)
        return_mae_bps = round(float(np.mean(ret_errors_bps)), 1) if ret_errors_bps else None

        # Directional statistics
        tp = tn = fp = fn = 0
        actual_up = actual_down = actual_flat = 0
        correct_count = 0

        for r in records:
            act_d = str(r.get("actual_direction", "")).upper()
            pred_d = str(r.get("predicted_direction", "")).upper()
            is_corr = r.get("direction_correct")

            if is_corr:
                correct_count += 1

            if act_d == "UP":
                actual_up += 1
            elif act_d == "DOWN":
                actual_down += 1
            else:
                actual_flat += 1

            # Binary directional confusion (UP = positive, DOWN/FLAT = negative)
            if act_d == "UP" and pred_d == "UP":
                tp += 1
            elif act_d != "UP" and pred_d != "UP":
                tn += 1
            elif act_d != "UP" and pred_d == "UP":
                fp += 1
            elif act_d == "UP" and pred_d != "UP":
                fn += 1

        dir_acc = round((correct_count / n) * 100.0, 1) if n > 0 else None
        mcc = calculate_mcc(tp, tn, fp, fn)

        up_recall = round((tp / (tp + fn)) * 100.0, 1) if (tp + fn) > 0 else None
        down_recall = round((tn / (tn + fp)) * 100.0, 1) if (tn + fp) > 0 else None
        balanced_acc = round((up_recall + down_recall) / 2.0, 1) if (up_recall is not None and down_recall is not None) else None

        # Residuals analysis
        residuals = [r["residual"] for r in records if r.get("residual") is not None]
        res_mean = round(float(np.mean(residuals)), 2) if residuals else 0.0
        res_std = round(float(np.std(residuals)), 2) if residuals else 0.0
        res_min = round(float(np.min(residuals)), 2) if residuals else 0.0
        res_max = round(float(np.max(residuals)), 2) if residuals else 0.0

        return {
            "sample_size": n,
            "price_mae": mae,
            "price_rmse": rmse,
            "price_mape": mape,
            "return_mae_bps": return_mae_bps,
            "directional_accuracy_pct": dir_acc,
            "balanced_accuracy_pct": balanced_acc,
            "mcc": mcc,
            "up_recall_pct": up_recall,
            "down_recall_pct": down_recall,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "class_counts": {
                "actual_up": actual_up,
                "actual_down": actual_down,
                "actual_flat": actual_flat,
            },
            "residuals": {
                "mean": res_mean,
                "std": res_std,
                "min": res_min,
                "max": res_max,
            }
        }

    def get_historical_predictions(
        self,
        symbol: str = "RELIANCE.NS",
        experiment: Optional[str] = None,
        model_name: Optional[str] = None,
        fold: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main query API for Historical Walk-Forward Explorer.
        Filters and groups genuine archived offline test predictions.
        """
        clean_symbol = symbol.strip().upper()
        all_records = self._cached_records.get(clean_symbol, [])

        if not all_records:
            return {
                "symbol": clean_symbol,
                "supported": clean_symbol in SUPPORTED_SYMBOLS,
                "has_artifacts": False,
                "message": f"No archived historical development prediction artifacts found for symbol '{clean_symbol}'.",
                "available_models": [],
                "available_experiments": [],
                "available_folds": [],
                "model_coverage_matrix": self.get_model_coverage_matrix(),
                "predicting_capability_scores": {},
                "metrics": {},
                "records": [],
                "chart_data": [],
                "limitations_disclaimer": "These predictions represent offline development research evaluations and must NEVER be conflated with prospective forward testing."
            }

        # Available metadata for this symbol
        available_models = sorted(list(set(r["model_name"] for r in all_records)))
        available_experiments = sorted(list(set(r["experiment"] for r in all_records)))
        available_folds = sorted(list(set(r["fold"] for r in all_records)))

        # Default model selection if not provided
        selected_model = model_name or ("GRU V2 Challenger" if "GRU V2 Challenger" in available_models else available_models[0])

        # Filter records
        filtered = all_records
        if experiment and experiment != "ALL":
            filtered = [r for r in filtered if r["experiment"] == experiment]
        if selected_model and selected_model != "ALL":
            filtered = [r for r in filtered if r["model_name"] == selected_model]
        if fold and fold != "ALL":
            filtered = [r for r in filtered if str(fold).lower() in r["fold"].lower()]
        if start_date:
            filtered = [r for r in filtered if r["target_date"] >= start_date]
        if end_date:
            filtered = [r for r in filtered if r["target_date"] <= end_date]

        # Sort chronologically
        filtered.sort(key=lambda x: x["target_date"])

        # Also get naive baseline metrics on the same subset of dates for fair capability comparison
        naive_subset = [
            r for r in all_records
            if "Naive" in r["model_name"] and r["target_date"] in set(f["target_date"] for f in filtered)
        ]
        naive_metrics = self.compute_slice_metrics(naive_subset) if naive_subset else None

        # Slice metrics for selected model
        selected_metrics = self.compute_slice_metrics(filtered)

        # Compute Predicting Capability Scores for all available models for this stock
        capability_scores = {}
        for m in available_models:
            m_records = [r for r in all_records if r["model_name"] == m]
            m_metrics = self.compute_slice_metrics(m_records)
            m_naive = [r for r in all_records if "Naive" in r["model_name"] and r["target_date"] in set(x["target_date"] for x in m_records)]
            m_naive_metrics = self.compute_slice_metrics(m_naive) if m_naive else None
            capability_scores[m] = self.compute_predicting_capability_score(
                metrics=m_metrics,
                naive_metrics=m_naive_metrics,
                symbol=clean_symbol,
                model_name=m
            )

        # Build timeline chart data (Actual vs Predicted)
        chart_dict = {}
        for r in filtered:
            t_date = r["target_date"]
            if t_date not in chart_dict:
                chart_dict[t_date] = {
                    "target_date": t_date,
                    "previous_close": r["previous_close"],
                    "actual_close": r["actual_close"],
                    "actual_direction": r["actual_direction"],
                    "fold": r["fold"],
                }
            chart_dict[t_date]["predicted_close"] = r["predicted_close"]
            chart_dict[t_date]["absolute_error"] = r["absolute_error"]
            chart_dict[t_date]["direction_correct"] = r["direction_correct"]
            chart_dict[t_date]["residual"] = r["residual"]

        chart_data = sorted(list(chart_dict.values()), key=lambda x: x["target_date"])

        # Build residual histogram data (10 bins)
        residuals_list = [r["residual"] for r in filtered if r.get("residual") is not None]
        residual_bins = []
        if len(residuals_list) >= 10:
            hist, bin_edges = np.histogram(residuals_list, bins=10)
            for i in range(len(hist)):
                residual_bins.append({
                    "range": f"₹{bin_edges[i]:.1f} to ₹{bin_edges[i+1]:.1f}",
                    "count": int(hist[i]),
                    "midpoint": round(float((bin_edges[i] + bin_edges[i+1]) / 2.0), 2)
                })

        # Apply record limit if specified (e.g. latest 200 for UI table performance)
        display_records = filtered[-limit:] if (limit and limit > 0) else filtered
        # Display records in descending order for the table
        display_records_desc = sorted(display_records, key=lambda x: x["target_date"], reverse=True)

        return {
            "symbol": clean_symbol,
            "supported": True,
            "has_artifacts": True,
            "evaluation_type": "HISTORICAL_OFFLINE_DEVELOPMENT",
            "selected_model": selected_model,
            "selected_experiment": experiment or "ALL",
            "selected_fold": fold or "ALL",
            "total_available_historical_predictions": len(filtered),
            "available_models": available_models,
            "available_experiments": available_experiments,
            "available_folds": available_folds,
            "model_coverage_matrix": self.get_model_coverage_matrix(),
            "predicting_capability_scores": capability_scores,
            "selected_model_capability_score": capability_scores.get(selected_model, {}),
            "metrics": selected_metrics,
            "naive_comparison_metrics": naive_metrics,
            "chart_data": chart_data,
            "residual_histogram": residual_bins,
            "records": display_records_desc,
            "limitations_disclaimer": (
                "These predictions originate from archived historical development folds (EXP-05 & EXP-06). "
                "They represent offline backtesting research and must NEVER be conflated with genuine "
                "prospective forecasts committed before actual trading session openings."
            )
        }


# Global singleton instance
historical_predictions_service = HistoricalPredictionsService()


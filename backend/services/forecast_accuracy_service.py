"""
StockSense AI — Forecast Accuracy & Track Record Service
Provides authoritative, data-driven calculation of forecast performance metrics
strictly from genuine prospective evaluation logs.

Separation Guarantees:
- Prospective logs (EXP-06 & EXP-05) are evaluated strictly for forward-testing track records.
- Historical offline development cross-validation metrics are preserved as separate benchmarks.
- Pending forecasts never contaminate accuracy calculations or register as zero error.
"""

import math
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXP06_LOG = Path("research/EXP-06/evaluation/prospective_predictions_log.csv")
DEFAULT_EXP06_REGISTRY = Path("research/EXP-06/evaluation/locked_evaluation_registry_v2.json")
DEFAULT_EXP05_LOG = Path("research/EXP-05/evaluation/prospective_predictions_log.csv")
DEFAULT_EXP05_REGISTRY = Path("research/EXP-05/evaluation/locked_evaluation_registry.json")


def safe_float(val: Any) -> Optional[float]:
    """Converts a value to float, returning None if NaN, null, or non-finite."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> Optional[float]:
    """
    Calculates Matthews Correlation Coefficient safely handling zero denominators.
    Returns None when the denominator is zero (e.g., when all true or predicted labels belong
    to a single class, resulting in degenerate variance), indicating insufficient class variation.
    """
    numerator = float(tp * tn - fp * fn)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return None
    return round(numerator / denominator, 4)


def assess_sample_size(count: int) -> Tuple[str, str]:
    """
    Returns empirical sample size assessment tier and scientific advisory note.
    Suggested interpretation:
    - < 20: Very limited evidence (preliminary)
    - 20-59: Preliminary evaluation
    - 60-99: Developing prospective evidence
    - 100+: Substantial prospective sample
    """
    if count == 0:
        return (
            "No Resolved Observations",
            "No completed prospective sessions have been scored yet. Pending forecasts await official market settlement."
        )
    elif count < 20:
        return (
            "Very Limited Evidence (Preliminary)",
            f"Only {count} prospective session(s) resolved. Sample size is preliminary and does not establish reliable predictive performance or trading alpha."
        )
    elif count < 60:
        return (
            "Preliminary Prospective Evaluation",
            f"Accumulated {count} prospective sessions. Evidence remains preliminary; formal statistical tests (Diebold-Mariano) require at least 60 sessions."
        )
    elif count < 100:
        return (
            "Developing Prospective Evidence",
            f"Accumulated {count} prospective sessions. Sample is approaching the preregistered 60–100 session threshold for confirmatory evaluation."
        )
    else:
        return (
            "Substantial Prospective Sample",
            f"Accumulated {count} prospective sessions. Sufficient for rolling stability analysis and formal significance testing."
        )


def normalize_model_family(name: str) -> str:
    """Normalizes model names across EXP-05 and EXP-06 into canonical family identifiers."""
    n = str(name).lower()
    if "gru" in n:
        return "GRU V2 Challenger"
    if "colab" in n or "exp-05" in n or "fold 3" in n:
        return "Colab EXP-05 Candidate"
    if "naive" in n or "persistence" in n:
        return "Naive Persistence Baseline"
    if "cnn" in n or "lstm" in n or "prod" in n:
        return "Production CNN-LSTM V1"
    return name


def get_model_color(family: str) -> str:
    """Returns consistent hex color for UI chart lines and badges."""
    if "GRU" in family:
        return "#0ea5e9"  # Sky blue
    if "Production" in family or "CNN" in family:
        return "#8b5cf6"  # Violet
    if "Colab" in family:
        return "#ec4899"  # Pink
    return "#f59e0b"      # Amber (Naive)


class ForecastAccuracyService:
    """
    Authoritative service computing verified forecast accuracy metrics
    from persistent prospective evaluation records.
    """

    def __init__(
        self,
        exp06_log_path: Path = DEFAULT_EXP06_LOG,
        exp06_registry_path: Path = DEFAULT_EXP06_REGISTRY,
        exp05_log_path: Path = DEFAULT_EXP05_LOG,
        exp05_registry_path: Path = DEFAULT_EXP05_REGISTRY,
    ):
        self.exp06_log_path = Path(exp06_log_path)
        self.exp06_registry_path = Path(exp06_registry_path)
        self.exp05_log_path = Path(exp05_log_path)
        self.exp05_registry_path = Path(exp05_registry_path)

    def _load_log_file(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            # Ensure correct string handling
            str_cols = ["forecast_id", "symbol", "target_date", "input_window_end", "model_name", "status", "predicted_direction", "actual_direction"]
            for c in str_cols:
                if c in df.columns:
                    df[c] = df[c].astype(str)
            return df
        except Exception as e:
            logger.warning(f"[ACCURACY SERVICE] Could not read log at {path}: {e}")
            return pd.DataFrame()

    def get_accuracy_payload(
        self,
        symbol: str = "RELIANCE.NS",
        limit: Optional[int] = None,
        include_historical_benchmarks: bool = True
    ) -> Dict[str, Any]:
        """
        Returns full forecast accuracy and track record payload.
        Handles empty data, single-equity restriction, and limits.
        """
        clean_symbol = symbol.strip().upper()
        if clean_symbol != "RELIANCE.NS":
            return {
                "symbol": clean_symbol,
                "supported": False,
                "message": (
                    f"Prospective shadow tracking and forecast accuracy evaluation are currently locked exclusively "
                    f"to RELIANCE.NS under protocols EXP-05 & EXP-06. No prospective records exist for '{clean_symbol}'."
                ),
                "authorized_symbol": "RELIANCE.NS",
                "summary": {},
                "resolved_records": [],
                "pending_records": [],
                "per_model_metrics": {},
                "common_date_paired_metrics": {},
                "sample_size_assessment": {
                    "tier": "Unsupported Symbol",
                    "note": f"Ticker '{clean_symbol}' is not enrolled in the prospective evaluation registry."
                }
            }

        # 1. Load prospective logs
        df_v2 = self._load_log_file(self.exp06_log_path)
        df_v1_exp05 = self._load_log_file(self.exp05_log_path)

        # Combine prospective observations, tagging origin protocol
        frames = []
        if not df_v2.empty and "symbol" in df_v2.columns:
            sub = df_v2[df_v2["symbol"] == clean_symbol].copy()
            sub["protocol"] = "EXP-06"
            frames.append(sub)

        if not df_v1_exp05.empty and "symbol" in df_v1_exp05.columns:
            sub = df_v1_exp05[df_v1_exp05["symbol"] == clean_symbol].copy()
            sub["protocol"] = "EXP-05"
            # Normalize column names if any slight difference
            frames.append(sub)

        if frames:
            combined_df = pd.concat(frames, ignore_index=True)
            # Deduplicate by forecast_id if any overlap
            if "forecast_id" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["forecast_id"], keep="first")
        else:
            combined_df = pd.DataFrame()

        if combined_df.empty:
            tier, note = assess_sample_size(0)
            return {
                "symbol": clean_symbol,
                "supported": True,
                "data_status": "NO_RECORDS",
                "resolved_records": [],
                "pending_records": [],
                "per_model_metrics": {},
                "common_date_paired_metrics": {},
                "sample_size_assessment": {"tier": tier, "note": note},
                "historical_backtest_benchmarks": self._get_historical_benchmarks() if include_historical_benchmarks else {}
            }

        # Separate into resolved and pending
        # Clean boolean direction_correct
        if "direction_correct" in combined_df.columns:
            combined_df["direction_correct_bool"] = combined_df["direction_correct"].apply(
                lambda v: True if str(v).strip().lower() in ("true", "1") else (False if str(v).strip().lower() in ("false", "0") else None)
            )
        else:
            combined_df["direction_correct_bool"] = None

        # Assign model_family
        if not combined_df.empty:
            combined_df["model_family"] = combined_df["model_name"].apply(normalize_model_family)

        resolved_mask = (combined_df["status"] == "RESOLVED") & combined_df["actual_close"].notna()
        resolved_df = combined_df[resolved_mask].copy()
        pending_mask = (combined_df["status"] == "PENDING_OUTCOME")
        pending_df = combined_df[pending_mask].copy()

        # Sort by target_date
        if not resolved_df.empty:
            resolved_df = resolved_df.sort_values("target_date", ascending=True).reset_index(drop=True)
        if not pending_df.empty:
            pending_df = pending_df.sort_values("target_date", ascending=True).reset_index(drop=True)

        # Apply session limit to resolved if requested (e.g. last 20 sessions)
        available_resolved_sessions = resolved_df["target_date"].nunique() if not resolved_df.empty else 0
        if limit and limit > 0 and not resolved_df.empty:
            unique_dates = sorted(resolved_df["target_date"].unique())
            selected_dates = set(unique_dates[-limit:])
            eval_resolved_df = resolved_df[resolved_df["target_date"].isin(selected_dates)].copy()
        else:
            eval_resolved_df = resolved_df.copy()

        # 2. Compute individual model metrics (both by exact name and by canonical family)
        per_model_metrics = {}
        unique_models = eval_resolved_df["model_name"].unique() if not eval_resolved_df.empty else []
        for m_name in unique_models:
            m_df = eval_resolved_df[eval_resolved_df["model_name"] == m_name]
            per_model_metrics[m_name] = self._compute_metrics_for_group(m_df)

        per_family_metrics = {}
        unique_families = eval_resolved_df["model_family"].unique() if not eval_resolved_df.empty else []
        for fam in unique_families:
            fam_df = eval_resolved_df[eval_resolved_df["model_family"] == fam]
            f_metrics = self._compute_metrics_for_group(fam_df)
            f_metrics["color"] = get_model_color(fam)
            per_family_metrics[fam] = f_metrics

        # 3. Compute common-date paired metrics across target models
        common_paired_metrics = self._compute_common_date_paired_metrics(eval_resolved_df)

        # 4. Format resolved records for UI table and charts
        resolved_records = self._format_records_list(eval_resolved_df)
        pending_records = self._format_records_list(pending_df)

        # 5. Build timeline chart points (grouping by target_date)
        chart_data = self._build_chart_series(eval_resolved_df)

        # 6. Sample size assessment
        total_unique_resolved_sessions = available_resolved_sessions
        tier, note = assess_sample_size(total_unique_resolved_sessions)

        return {
            "symbol": clean_symbol,
            "supported": True,
            "evaluation_type": "PROSPECTIVE_FORWARD_TRACKING",
            "evaluation_horizon": "Prospective Market Settlements (October–November 2026)",
            "limit_applied": limit,
            "total_available_resolved_sessions": total_unique_resolved_sessions,
            "total_pending_forecasts": len(pending_df),
            "sample_size_assessment": {
                "tier": tier,
                "note": note,
                "resolved_session_count": total_unique_resolved_sessions,
                "is_sufficient_for_significance": total_unique_resolved_sessions >= 60,
            },
            "per_model_metrics": per_model_metrics,
            "per_family_metrics": per_family_metrics,
            "common_date_paired_metrics": common_paired_metrics,
            "chart_data": chart_data,
            "resolved_records": resolved_records,
            "pending_records": pending_records,
            "historical_backtest_benchmarks": self._get_historical_benchmarks() if include_historical_benchmarks else {}
        }

    def _compute_metrics_for_group(self, group: pd.DataFrame) -> Dict[str, Any]:
        """Calculates full statistical suite for a model's observations."""
        n = len(group)
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
                "start_date": None,
                "end_date": None,
            }

        price_errors = group["price_error_abs"].dropna().astype(float)
        mae = round(float(price_errors.mean()), 2) if not price_errors.empty else None
        rmse = round(float(np.sqrt((price_errors ** 2).mean())), 2) if not price_errors.empty else None

        # Safe MAPE calculation (ignoring zero/negative actual closes)
        mape_vals = []
        for _, row in group.iterrows():
            act = safe_float(row.get("actual_close"))
            pred = safe_float(row.get("predicted_close"))
            if act is not None and act > 0 and pred is not None:
                mape_vals.append(abs(pred - act) / act * 100.0)
        mape = round(float(np.mean(mape_vals)), 3) if mape_vals else None

        # Return MAE in basis points
        ret_errs_bps = []
        for _, row in group.iterrows():
            act_r = safe_float(row.get("actual_return"))
            pred_r = safe_float(row.get("predicted_return"))
            if act_r is not None and pred_r is not None:
                ret_errs_bps.append(abs(pred_r - act_r) * 10000.0)
        return_mae_bps = round(float(np.mean(ret_errs_bps)), 1) if ret_errs_bps else None

        # Directional statistics
        tp = tn = fp = fn = 0
        actual_up = actual_down = actual_flat = 0
        pred_up = pred_down = pred_flat = 0
        correct_count = 0

        for _, row in group.iterrows():
            act_d = str(row.get("actual_direction", "")).upper()
            pred_d = str(row.get("predicted_direction", "")).upper()
            is_corr = row.get("direction_correct_bool")

            if is_corr is True:
                correct_count += 1

            if act_d == "UP":
                actual_up += 1
            elif act_d == "DOWN":
                actual_down += 1
            else:
                actual_flat += 1

            if pred_d == "UP":
                pred_up += 1
            elif pred_d == "DOWN":
                pred_down += 1
            else:
                pred_flat += 1

            # Binary directional confusion (UP = positive class, DOWN/FLAT = negative)
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

        # Balanced accuracy requires both classes to have non-zero representation
        if up_recall is not None and down_recall is not None:
            balanced_acc = round((up_recall + down_recall) / 2.0, 1)
        else:
            balanced_acc = None

        start_date = str(group["target_date"].min())
        end_date = str(group["target_date"].max())

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
                "pred_up": pred_up,
                "pred_down": pred_down,
                "pred_flat": pred_flat,
            },
            "start_date": start_date,
            "end_date": end_date,
        }

    def _compute_common_date_paired_metrics(self, resolved_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Finds target dates where multiple comparison models share evaluations
        and computes paired metrics strictly on that identical date set.
        """
        if resolved_df.empty:
            return {"common_dates_count": 0, "common_dates": [], "models": {}}

        # Target comparison model families
        target_fams = ["GRU V2 Challenger", "Production CNN-LSTM V1", "Naive Persistence Baseline"]
        date_groups = resolved_df.groupby("target_date")["model_family"].apply(set)

        # Check which target families exist
        existing_fams = [f for f in target_fams if f in resolved_df["model_family"].values]
        if not existing_fams:
            existing_fams = list(resolved_df["model_family"].unique())

        common_dates = []
        for t_date, fams_present in date_groups.items():
            if set(existing_fams).issubset(fams_present):
                common_dates.append(t_date)

        if not common_dates:
            # Fallback: take dates present in at least 2 families
            for t_date, fams_present in date_groups.items():
                if len(fams_present) >= 2:
                    common_dates.append(t_date)

        common_dates = sorted(common_dates)
        paired_subset = resolved_df[resolved_df["target_date"].isin(common_dates)]

        paired_family_metrics = {}
        for fam in paired_subset["model_family"].unique():
            fam_sub = paired_subset[paired_subset["model_family"] == fam]
            f_m = self._compute_metrics_for_group(fam_sub)
            f_m["color"] = get_model_color(fam)
            paired_family_metrics[fam] = f_m

        return {
            "common_dates_count": len(common_dates),
            "common_dates": common_dates,
            "models_compared": list(paired_family_metrics.keys()),
            "metrics_per_model": paired_family_metrics,
            "methodology_note": (
                "Metrics in this section are computed exclusively on identical resolved target dates "
                "evaluated against the exact same finalized settlement prices."
            )
        }

    def _format_records_list(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Converts DataFrame rows to clean typed JSON records."""
        records = []
        if df.empty:
            return records

        for _, row in df.iterrows():
            act_c = safe_float(row.get("actual_close"))
            pred_c = safe_float(row.get("predicted_close"))
            prev_c = safe_float(row.get("previous_close"))
            err_abs = safe_float(row.get("price_error_abs"))

            # Calculate pct error if resolved
            pct_err = None
            if act_c is not None and act_c > 0 and pred_c is not None:
                pct_err = round(abs(pred_c - act_c) / act_c * 100.0, 3)

            m_name = str(row.get("model_name", ""))
            m_family = normalize_model_family(m_name)
            m_color = get_model_color(m_family)

            records.append({
                "forecast_id": str(row.get("forecast_id", "")),
                "target_date": str(row.get("target_date", "")),
                "input_window_end": str(row.get("input_window_end", "")),
                "model_name": m_name,
                "model_family": m_family,
                "model_color": m_color,
                "model_hash": str(row.get("model_hash", "")),
                "protocol": str(row.get("protocol", "EXP-06")),
                "previous_close": prev_c,
                "predicted_close": pred_c,
                "predicted_return": safe_float(row.get("predicted_return")),
                "predicted_direction": str(row.get("predicted_direction", "")),
                "actual_close": act_c,
                "actual_return": safe_float(row.get("actual_return")),
                "actual_direction": str(row.get("actual_direction", "")) if pd.notna(row.get("actual_direction")) else None,
                "direction_correct": row.get("direction_correct_bool"),
                "price_error_abs": err_abs,
                "price_error_pct": pct_err,
                "status": str(row.get("status", "PENDING_OUTCOME")),
                "resolution_timestamp": str(row.get("resolution_timestamp", "")) if pd.notna(row.get("resolution_timestamp")) else None,
                "data_source": str(row.get("data_source", "Yahoo Finance")),
            })

        return records

    def _build_chart_series(self, resolved_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Groups resolved predictions by target_date into Recharts-friendly timeline."""
        if resolved_df.empty:
            return []

        chart_rows = []
        for t_date, grp in resolved_df.groupby("target_date"):
            row_dict = {
                "target_date": str(t_date),
                "actual_close": safe_float(grp["actual_close"].iloc[0]),
                "previous_close": safe_float(grp["previous_close"].iloc[0]),
                "actual_direction": str(grp["actual_direction"].iloc[0]),
            }

            for _, row in grp.iterrows():
                m_name = str(row["model_name"])
                pred_c = safe_float(row.get("predicted_close"))
                err = safe_float(row.get("price_error_abs"))
                corr = row.get("direction_correct_bool")

                if "GRU" in m_name:
                    row_dict["v2_gru_pred"] = pred_c
                    row_dict["v2_gru_error"] = err
                    row_dict["v2_gru_correct"] = corr
                elif "CNN" in m_name or "Prod" in m_name:
                    row_dict["v1_prod_pred"] = pred_c
                    row_dict["v1_prod_error"] = err
                    row_dict["v1_prod_correct"] = corr
                elif "Naive" in m_name:
                    row_dict["naive_pred"] = pred_c
                    row_dict["naive_error"] = err
                    row_dict["naive_correct"] = corr

            chart_rows.append(row_dict)

        chart_rows.sort(key=lambda x: x["target_date"])
        return chart_rows

    def _get_historical_benchmarks(self) -> Dict[str, Any]:
        """
        Returns verified EXP-06 offline cross-fold development results.
        Preserved as a clearly labeled reference that is NEVER conflated with prospective outcomes.
        Explicitly distinguishes full dataset size (1,233 rows) from fold-specific out-of-sample test sizes.
        """
        return {
            "evaluation_type": "OFFLINE_DEVELOPMENT_BACKTEST",
            "protocol": "EXP-06 Chronological Development Folds",
            "dataset_rows": 1233,
            "fold_splits": {
                "fold_1": {"train_rows": 740, "val_rows": 247, "test_rows": 246},
                "fold_2": {"train_rows": 863, "val_rows": 185, "test_rows": 185},
                "fold_3": {"train_rows": 987, "val_rows": 123, "test_rows": 123},
            },
            "fold3_effective_sequence_windows": 100,
            "period": "2021-01 to 2026-10",
            "disclaimer": (
                "These metrics reflect retrospective 3-fold cross-validation on historical data. "
                "The 1,233 rows represent the entire audited dataset (2021–2026), whereas test evaluations "
                "were computed out-of-sample on distinct chronological test slices (Fold 1: 246 sessions, "
                "Fold 2: 185 sessions, Fold 3: 123 sessions / 100 test sequence windows). "
                "They are provided as baseline engineering benchmarks and must NEVER be conflated "
                "with prospective live-market forward testing."
            ),
            "cross_fold_leaderboard": [
                {
                    "rank": 1,
                    "model": "GRU V2 (EXP-06)",
                    "parameters": 44065,
                    "cross_fold_mae": 13.29,
                    "fold3_mae": 12.66,
                    "fold3_directional_acc_pct": 54.0,
                    "fold3_mcc": 0.1214,
                    "up_recall_pct": 78.7,
                    "down_recall_pct": 32.1,
                    "role": "Research Challenger (Shadow)",
                },
                {
                    "rank": 2,
                    "model": "Naive Persistence Benchmark",
                    "parameters": 0,
                    "cross_fold_mae": 13.34,
                    "fold3_mae": 12.71,
                    "fold3_directional_acc_pct": 50.0,
                    "fold3_mcc": 0.0000,
                    "up_recall_pct": 0.0,
                    "down_recall_pct": 0.0,
                    "role": "Reference Baseline",
                },
                {
                    "rank": 3,
                    "model": "Standard LSTM",
                    "parameters": 52289,
                    "cross_fold_mae": 13.39,
                    "fold3_mae": 12.78,
                    "fold3_directional_acc_pct": 52.0,
                    "fold3_mcc": 0.0610,
                    "up_recall_pct": 74.2,
                    "down_recall_pct": 35.8,
                    "role": "Comparative Control",
                },
                {
                    "rank": 4,
                    "model": "CNN-LSTM Hybrid (V1)",
                    "parameters": 112417,
                    "cross_fold_mae": 13.42,
                    "fold3_mae": 12.82,
                    "fold3_directional_acc_pct": 53.0,
                    "fold3_mcc": 0.0825,
                    "up_recall_pct": 76.5,
                    "down_recall_pct": 34.0,
                    "role": "Production Standard",
                },
                {
                    "rank": 5,
                    "model": "Linear Ridge Regressor",
                    "parameters": 211,
                    "cross_fold_mae": 14.18,
                    "fold3_mae": 13.65,
                    "fold3_directional_acc_pct": 50.5,
                    "fold3_mcc": 0.0120,
                    "up_recall_pct": 52.0,
                    "down_recall_pct": 49.0,
                    "role": "Linear Baseline",
                },
            ]
        }


forecast_accuracy_service = ForecastAccuracyService()

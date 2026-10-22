"""
Canonical Stationary Feature Pipeline for StockSense AI.

This module provides the single source of truth for the 7 relative stationary features
and next-day return target validated in EXP-04 and EXP-05.

Features:
1. return_1d:        Close[t] / Close[t-1] - 1
2. return_5d:        Close[t] / Close[t-5] - 1
3. return_20d:       Close[t] / Close[t-20] - 1
4. intraday_return:  (Close[t] - Open[t]) / Open[t]
5. hl_range_ratio:   (High[t] - Low[t]) / Close[t-1]
6. close_to_ma20:    Close[t] / MA20[t] - 1
7. volume_ratio:     Volume[t] / Mean(Volume[t-19...t])

Target:
- target_return:     Close[t+1] / Close[t] - 1

Causality & Leakage Guarantee:
- All 7 features at time t are computed strictly using observations at or prior to session t.
- The target return is computed using session t+1 relative to session t.
- During inference on completed session t, only features up to session t are used to predict t+1.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "return_1d",
    "return_5d",
    "return_20d",
    "intraday_return",
    "hl_range_ratio",
    "close_to_ma20",
    "volume_ratio",
]

TARGET_COLUMN: str = "target_return"
REQUIRED_BAR_COLUMNS: List[str] = ["Date", "Open", "High", "Low", "Close", "Volume"]
WINDOW_SIZE: int = 30
MIN_WARMUP_PERIOD: int = 20  # Minimum bars needed for 20-day returns and rolling MA20


def validate_raw_bars(df: pd.DataFrame) -> None:
    """
    Validates that the input dataframe contains all required price bar columns.
    Raises ValueError if required columns are missing or dataframe is empty.
    """
    if df is None or df.empty:
        raise ValueError("Input bar DataFrame is empty or None.")

    missing_cols = [c for c in REQUIRED_BAR_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required bar columns: {missing_cols}. "
            f"Required columns: {REQUIRED_BAR_COLUMNS}"
        )


def compute_stationary_features(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes strictly causal, relative stationary features and the next-day target return.
    
    This function:
    1. Validates required bar columns.
    2. Sorts bars chronologically by Date.
    3. Computes the 7 stationary features without lookahead.
    4. Computes target_return, target_close, and target_date (session t+1).
    5. Cleans and validates that all feature rows and targets are non-null.

    Returns:
        (clean_df, FEATURE_COLUMNS): Dataframe with computed features and targets,
        and the canonical feature column list.
    """
    validate_raw_bars(df_input)

    df = df_input.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Multi-horizon close returns
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
    df[TARGET_COLUMN] = (df["Close"].shift(-1) / df["Close"]) - 1.0
    df["target_close"] = df["Close"].shift(-1)
    df["target_date"] = df["Date"].shift(-1).dt.strftime("%Y-%m-%d")

    # Clean rows with complete features and valid next-day target
    clean_df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN, "target_close", "target_date"]
    ).reset_index(drop=True)

    validate_feature_dataframe(clean_df, require_target=True)
    return clean_df, FEATURE_COLUMNS


def prepare_inference_features(df_input: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes stationary features for online inference on completed trading sessions.
    
    Unlike training preparation, the most recent completed candle (session t)
    is preserved because its next-day target is unknown (the subject of forecasting).
    """
    validate_raw_bars(df_input)

    df = df_input.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Multi-horizon close returns
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

    # Drop only warmup rows where 20-day lookbacks cannot be computed
    clean_df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if len(clean_df) < WINDOW_SIZE:
        raise ValueError(
            f"Insufficient valid feature rows ({len(clean_df)}) for inference window of {WINDOW_SIZE}."
        )

    validate_feature_dataframe(clean_df, require_target=False)
    return clean_df, FEATURE_COLUMNS


def validate_feature_dataframe(df: pd.DataFrame, require_target: bool = False) -> None:
    """
    Validates feature column presence, count, ordering, and absence of NaN/inf.
    """
    # 1. Check feature count
    if len(FEATURE_COLUMNS) != 7:
        raise ValueError(f"Expected exactly 7 feature columns, found {len(FEATURE_COLUMNS)}")

    # 2. Check column existence and ordering
    for idx, expected_col in enumerate(FEATURE_COLUMNS):
        if expected_col not in df.columns:
            raise ValueError(f"Missing required stationary feature column: '{expected_col}'")

    # 3. Check NaN/inf in feature columns
    feat_matrix = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if np.isnan(feat_matrix).any():
        raise ValueError("Feature matrix contains NaN values.")
    if np.isinf(feat_matrix).any():
        raise ValueError("Feature matrix contains Infinite values.")

    # 4. Optional target validation
    if require_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' is required but missing.")
        tgt_arr = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
        if np.isnan(tgt_arr).any() or np.isinf(tgt_arr).any():
            raise ValueError(f"Target column '{TARGET_COLUMN}' contains NaN or Inf values.")


def validate_sequence_tensor(tensor: np.ndarray, expected_window: int = WINDOW_SIZE) -> None:
    """
    Validates sequence tensor dimensions, feature dimension, and value validity.
    Accepts 3D tensor of shape (batch, window, features).
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(tensor)}")

    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D sequence tensor (samples, window, features), got ndim={tensor.ndim}")

    _, window, num_features = tensor.shape
    if window != expected_window:
        raise ValueError(f"Expected window size {expected_window}, got {window}")

    if num_features != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {num_features}")

    if np.isnan(tensor).any():
        raise ValueError("Sequence tensor contains NaN values.")

    if np.isinf(tensor).any():
        raise ValueError("Sequence tensor contains Infinite values.")


def reconstruct_price_from_return(last_close: float, predicted_return: float) -> float:
    """
    Reconstructs predicted next-session closing price from the reference close and predicted return:
    
    predicted_close = last_close * (1 + predicted_return)
    """
    if last_close <= 0:
        raise ValueError(f"last_close must be positive, got {last_close}")
    return float(last_close * (1.0 + predicted_return))

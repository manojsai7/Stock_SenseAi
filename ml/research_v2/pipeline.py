"""Strict validation around the shared V1 causal features."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
from ml.features.stationary import FEATURE_COLUMNS, compute_stationary_features, validate_raw_bars
from ml.preprocessing.stationary_scaler import StationaryFeatureScaler, ReturnTargetScaler

FEATURE_VERSION = 'stationary-seven-v1'

def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def validate_bars(raw):
    validate_raw_bars(raw)
    df = raw.copy()
    dates = pd.to_datetime(df.Date, errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique, chronological and non-null')
    if not (dates == dates.dt.normalize()).all():
        raise ValueError('Expected daily session dates')
    values = df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] <= 0).any():
        raise ValueError('OHLCV must be finite and positive; resolve zero-volume rows explicitly')
    if ((df.High < df[['Open', 'Close', 'Low']].max(axis=1)) | (df.Low > df[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC bounds')
    # Shared V1 code trusts supplied MA20; reject noncausal/precomputed disagreement.
    expected = df.Close.rolling(20, min_periods=20).mean()
    if 'MA20' in df and not np.allclose(df.MA20, expected, equal_nan=True, rtol=1e-6, atol=1e-6):
        raise ValueError('Supplied MA20 differs from causal rolling mean')
    df.Date = dates
    return df

@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    dates: pd.DataFrame

def sequences(raw, window=30):
    if window != 30:
        raise ValueError('EXP-06 initial contract requires 30 sessions')
    frame, columns = compute_stationary_features(validate_bars(raw))
    if columns != FEATURE_COLUMNS or len(frame) < window:
        raise ValueError('Invalid feature contract or insufficient history')
    x, y, records = [], [], []
    for end in range(window-1, len(frame)):
        row = frame.iloc[end]
        start = frame.iloc[end-window+1].Date
        target = pd.Timestamp(row.target_date)
        if not start <= row.Date < target:
            raise ValueError('Input/target leakage')
        x.append(frame[columns].iloc[end-window+1:end+1].to_numpy())
        y.append(row.target_return)
        records.append(dict(input_start=start, input_end=row.Date, target_date=target,
                            previous_close=float(row.Close), actual_close=float(row.target_close)))
    dates = pd.DataFrame(records)
    if dates.target_date.duplicated().any():
        raise ValueError('Duplicate target dates')
    return Samples(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float64), dates)

def chronological_splits(samples, *, holdout_start, min_train, validation_size, test_size, folds):
    if min(min_train, validation_size, test_size, folds) <= 0:
        raise ValueError('Split sizes must be positive')
    cutoff = pd.Timestamp(holdout_start)
    if cutoff <= pd.Timestamp('2026-10-07'):
        raise ValueError('Holdout overlaps previously inspected EXP-05 dates')
    n = int((samples.dates.target_date < cutoff).sum())
    if len(samples.y)-n < 60:
        raise ValueError('Dataset requires at least 60 reserved holdout observations')
    if min_train + validation_size + test_size * folds > n:
        raise ValueError('Insufficient pre-holdout samples for requested folds')
    first = n - validation_size - test_size * folds
    result = []
    for fold in range(folds):
        train_end = first + fold * test_size
        val_end = train_end + validation_size
        result.append((np.arange(train_end), np.arange(train_end, val_end), np.arange(val_end, val_end+test_size)))
    return result, np.arange(n, len(samples.y))

def fit_fold_scalers(samples, train_indices):
    indices = np.asarray(train_indices)
    if not np.array_equal(indices, np.arange(len(indices))) or not len(indices):
        raise ValueError('Training partition must be a nonempty chronological prefix')
    return (StationaryFeatureScaler().fit(samples.x[indices]),
            ReturnTargetScaler().fit(samples.y[indices]))

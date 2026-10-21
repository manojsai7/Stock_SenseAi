"""
Scalers for the Canonical Stationary-Feature and Return-Target Pipeline.

Provides dedicated, leakage-free scalers:
1. StationaryFeatureScaler: Standardizes the 7 stationary features using training statistics only.
2. ReturnTargetScaler: Standardizes next-day simple returns using training mean and standard deviation.

Strict Constraints:
- Scalers must be fitted EXCLUSIVELY on training partitions.
- Testing, validation, and real-time inference data are strictly transformed without re-fitting.
"""

from pathlib import Path
from typing import Optional, Union, List
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.features.stationary import FEATURE_COLUMNS


class StationaryFeatureScaler:
    """
    Standardizes 7 stationary features across (samples, window, features) or (N, features).
    Fitted exclusively on training data.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.scaler = StandardScaler()
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "StationaryFeatureScaler":
        """
        Fits StandardScaler on 2D or 3D training features.
        """
        arr = self._to_2d_array(X)
        if arr.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {arr.shape[1]}"
            )
        self.scaler.fit(arr)
        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transforms features using fitted statistics.
        Preserves 3D shape if input is 3D.
        """
        if not self.is_fitted:
            raise ValueError("StationaryFeatureScaler is not fitted yet. Call fit() on training data first.")

        if isinstance(X, pd.DataFrame):
            arr_2d = X[self.feature_columns].to_numpy(dtype=np.float32)
            scaled = self.scaler.transform(arr_2d)
            return scaled.astype(np.float32)

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            return self.scaler.transform(arr).astype(np.float32)
        elif arr.ndim == 3:
            N, W, F = arr.shape
            if F != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {F}")
            scaled_2d = self.scaler.transform(arr.reshape(-1, F))
            return scaled_2d.reshape(N, W, F).astype(np.float32)
        else:
            raise ValueError(f"Expected 2D or 3D array, got ndim={arr.ndim}")

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)

    def _to_2d_array(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_columns].to_numpy(dtype=np.float64)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2:
            return arr
        elif arr.ndim == 3:
            return arr.reshape(-1, arr.shape[-1])
        raise ValueError(f"Expected 2D or 3D array, got ndim={arr.ndim}")

    def save(self, file_path: Union[str, Path]) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "StationaryFeatureScaler":
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Feature scaler artifact not found: {p}")
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}, got {type(obj)}")
        return obj


class ReturnTargetScaler:
    """
    Standardizes scalar target returns: y_scaled = (y - mean) / std.
    Fitted exclusively on training targets.
    """

    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, returns: Union[np.ndarray, pd.Series, List[float]]) -> "ReturnTargetScaler":
        arr = np.asarray(returns, dtype=np.float64).flatten()
        if len(arr) == 0:
            raise ValueError("Cannot fit ReturnTargetScaler on empty array.")
        if np.isnan(arr).any():
            raise ValueError("Training returns contain NaN values.")

        self.mean_ = float(np.mean(arr))
        std = float(np.std(arr))
        self.std_ = std if std > 1e-12 else 1.0
        self.is_fitted = True
        return self

    def transform(self, returns: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ReturnTargetScaler is not fitted yet. Call fit() on training targets first.")
        arr = np.asarray(returns, dtype=np.float64).flatten()
        return ((arr - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, scaled_returns: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if not self.is_fitted:
            raise ValueError("ReturnTargetScaler is not fitted yet.")

        if isinstance(scaled_returns, (float, int, np.floating)):
            return float(scaled_returns * self.std_ + self.mean_)

        arr = np.asarray(scaled_returns, dtype=np.float64).flatten()
        return (arr * self.std_ + self.mean_).astype(np.float64)

    def fit_transform(self, returns: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
        return self.fit(returns).transform(returns)

    def save(self, file_path: Union[str, Path]) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ReturnTargetScaler":
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Return scaler artifact not found: {p}")
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}, got {type(obj)}")
        return obj

import pickle
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ml.features.technical import FEATURE_COLUMNS, TARGET_COLUMN

class StockDataScaler:
    """
    Leakage-safe scaler for stock features and target.
    Fitted exclusively on the training partition.
    """

    def __init__(self, feature_columns: List[str] = FEATURE_COLUMNS):
        self.feature_columns = feature_columns
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "StockDataScaler":
        features = train_df[self.feature_columns].values
        targets = train_df[[TARGET_COLUMN]].values

        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet! Call fit() on training data first.")

        scaled_features = self.feature_scaler.transform(df[self.feature_columns].values)
        if TARGET_COLUMN in df.columns and not df[TARGET_COLUMN].isna().all():
            scaled_targets = self.target_scaler.transform(df[[TARGET_COLUMN]].values)
        else:
            scaled_targets = None
        return scaled_features, scaled_targets

    def transform_features_only(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet!")
        return self.feature_scaler.transform(df[self.feature_columns].values)

    def inverse_transform_target(self, scaled_predictions: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet!")
        if scaled_predictions.ndim == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled_predictions).flatten()

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Path) -> "StockDataScaler":
        with open(file_path, "rb") as f:
            return pickle.load(f)

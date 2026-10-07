import pandas as pd
import numpy as np

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "MA20"]
TARGET_COLUMN = "Next_Close"

class TechnicalFeatureExtractor:
    """
    Extracts features for deep learning stock prediction.
    Features specified in system design:
    - Open, High, Low, Close, Volume
    - MA20 (20-day Simple Moving Average)
    Target:
    - Next_Close: Closing price at day t+1
    """

    def __init__(self, ma_window: int = 20):
        self.ma_window = ma_window

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = data.sort_values("Date").reset_index(drop=True)

        # 20-day Simple Moving Average on Close
        data["MA20"] = data["Close"].rolling(window=self.ma_window, min_periods=self.ma_window).mean()

        # Target: Next day's Close
        data[TARGET_COLUMN] = data["Close"].shift(-1)

        # Drop rows where MA20 cannot be computed (first 19 rows)
        # Note: Do not drop the last row if we are preparing for inference where Next_Close is unknown!
        return data

    def prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares dataframe for training, dropping rows with NaN in features or target.
        """
        data = self.add_features(df)
        data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
        return data

    def prepare_inference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares features for inference on recent data where next close is unknown.
        """
        data = self.add_features(df)
        data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        return data

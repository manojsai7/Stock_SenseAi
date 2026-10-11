import numpy as np

class NaivePersistenceModel:
    """
    Naive Baseline Model:
    Predicts the next day's closing price will equal the most recent observed closing price.
    Acts as the fundamental baseline for time series forecasting.
    """

    def __init__(self, close_col_index: int = 3):
        # In FEATURE_COLUMNS: ["Open", "High", "Low", "Close", "Volume", "MA20"] -> index 3 is Close
        self.close_col_index = close_col_index

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (N, window_size, num_features)
        Returns the Close value of the last timestep in the window: X[:, -1, close_col_index]
        """
        return X[:, -1, self.close_col_index]

    def predict_direction(self, X: np.ndarray) -> np.ndarray:
        """
        Since naive persistence forecasts PredictedClose[t+1] = Close[t],
        PredictedClose[t+1] - Close[t] == 0 identically.
        Returns 'FLAT' for all samples. Equal prices must never be classified as 'UP'.
        """
        return np.full(shape=(len(X),), fill_value="FLAT", dtype=object)

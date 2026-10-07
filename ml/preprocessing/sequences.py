from typing import Tuple, Optional
import numpy as np

class SequenceGenerator:
    """
    Constructs sliding window sequences for temporal neural networks (LSTM, CNN-LSTM).
    Guarantees strict chronological ordering without lookahead leakage.
    Window size: N trading days (default 30).
    Input shape per sample: (window_size, num_features)
    Target: Next trading day's closing price.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def create_sequences(
        self, scaled_features: np.ndarray, scaled_targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Takes scaled_features of shape (N, features) and scaled_targets of shape (N, 1).
        Sample i has input window: scaled_features[i - window_size : i]
        Target for sample i is: scaled_targets[i - 1] (which represents Next_Close of row i-1)
        """
        num_samples = len(scaled_features)
        if num_samples < self.window_size:
            raise ValueError(
                f"Insufficient data points ({num_samples}) for window size of {self.window_size}."
            )

        X = []
        y = []

        for i in range(self.window_size, num_samples + 1):
            X.append(scaled_features[i - self.window_size : i])
            if scaled_targets is not None:
                y.append(scaled_targets[i - 1])

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32).flatten() if scaled_targets is not None else None
        return X_arr, y_arr

    def create_split_sequences(
        self,
        scaled_features: np.ndarray,
        scaled_targets: np.ndarray,
        train_idx: int,
        val_idx: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Constructs sequences for Train, Validation, and Test partitions.
        Uses full preceding sequence history for input windows, while strictly partitioning
        targets by chronological index boundaries:
        - Train targets: index < train_idx
        - Val targets: train_idx <= index < val_idx
        - Test targets: index >= val_idx
        Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), test_orig_indices
        """
        total_len = len(scaled_features)
        X, y = [], []

        for i in range(self.window_size, total_len + 1):
            X.append(scaled_features[i - self.window_size : i])
            y.append(scaled_targets[i - 1])

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32).flatten()

        target_orig_indices = np.arange(self.window_size - 1, total_len)

        train_mask = target_orig_indices < train_idx
        val_mask = (target_orig_indices >= train_idx) & (target_orig_indices < val_idx)
        test_mask = target_orig_indices >= val_idx

        X_train, y_train = X_arr[train_mask], y_arr[train_mask]
        X_val, y_val = X_arr[val_mask], y_arr[val_mask]
        X_test, y_test = X_arr[test_mask], y_arr[test_mask]

        test_indices = target_orig_indices[test_mask]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), test_indices

    def create_inference_sequence(self, scaled_features: np.ndarray) -> np.ndarray:
        """
        Creates single sequence of shape (1, window_size, features) from the most recent window.
        """
        if len(scaled_features) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} candles for inference sequence.")
        recent_window = scaled_features[-self.window_size:]
        return np.expand_dims(recent_window, axis=0).astype(np.float32)

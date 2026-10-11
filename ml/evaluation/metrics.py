from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_direction_signs(
    prices: np.ndarray,
    reference_prices: np.ndarray,
    atol: float = 1e-5
) -> np.ndarray:
    """
    Computes directional movement signs with three discrete classes:
      +1: UP   (prices - reference_prices > atol)
      -1: DOWN (prices - reference_prices < -atol)
       0: FLAT (|prices - reference_prices| <= atol)
    Equal or near-equal prices are strictly mapped to 0 (FLAT) and NEVER +1 (UP).
    """
    prices = np.asarray(prices, dtype=np.float64).flatten()
    reference_prices = np.asarray(reference_prices, dtype=np.float64).flatten()
    diff = prices - reference_prices
    if atol > 0:
        diff = np.where(np.abs(diff) <= atol, 0.0, diff)
    return np.sign(diff).astype(int)

def compute_direction_labels(
    prices: np.ndarray,
    reference_prices: np.ndarray,
    atol: float = 1e-5
) -> np.ndarray:
    """
    Computes three-class string direction labels: "UP", "DOWN", "FLAT".
    Equal and unchanged prices are never classified as "UP".
    """
    signs = compute_direction_signs(prices, reference_prices, atol=atol)
    labels = np.full(signs.shape, "FLAT", dtype=object)
    labels[signs == 1] = "UP"
    labels[signs == -1] = "DOWN"
    return labels

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray,
    prior_trend: Optional[np.ndarray] = None,
    atol: float = 1e-5
) -> Dict[str, float]:
    """
    Computes rigorous academic and financial evaluation metrics:
    - MAE: Mean Absolute Error (in native currency, INR)
    - RMSE: Root Mean Squared Error (in native currency)
    - MAPE: Mean Absolute Percentage Error (%)
    - Directional Accuracy: Percentage of correctly predicted price movement directions
      using three classes (UP, DOWN, FLAT):
        actual_dir = sign(Close[t+1] - Close[t])
        pred_dir   = sign(PredictedClose[t+1] - Close[t])
        DA = count(actual_dir == pred_dir) / total * 100
      Equal prices must never be classified as UP.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    previous_close = np.asarray(previous_close, dtype=np.float64).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE: avoid division by zero
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
    else:
        mape = 0.0

    # Three-Class Directional Accuracy: UP, DOWN, FLAT
    actual_dir = compute_direction_signs(y_true, previous_close, atol=atol)
    pred_dir = compute_direction_signs(y_pred, previous_close, atol=atol)

    correct_directions = np.sum(actual_dir == pred_dir)
    directional_accuracy = float((correct_directions / len(actual_dir)) * 100) if len(actual_dir) > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "directional_accuracy": round(directional_accuracy, 2)
    }


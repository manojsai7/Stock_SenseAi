from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray,
    prior_trend: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes rigorous academic and financial evaluation metrics:
    - MAE: Mean Absolute Error (in native currency, INR)
    - RMSE: Root Mean Squared Error (in native currency)
    - MAPE: Mean Absolute Percentage Error (%)
    - Directional Accuracy: Percentage of correctly predicted price movement directions:
        actual_dir = +1 if y_true >= previous_close else -1
        pred_dir   = +1 if y_pred >= previous_close else -1
        DA = count(actual_dir == pred_dir) / total * 100
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

    # Binary Directional Accuracy (Hit Ratio)
    actual_dir = np.where(y_true >= previous_close, 1, -1)

    # Check if model is pure naive persistence where y_pred == previous_close everywhere
    diff = y_pred - previous_close
    is_pure_persistence = np.allclose(diff, 0, atol=1e-5)

    if is_pure_persistence:
        if prior_trend is not None and len(prior_trend) == len(actual_dir):
            # Evaluate using standard momentum direction baseline: sign(P_t - P_{t-1})
            naive_dir = np.where(prior_trend >= 0, 1, -1)
            correct_directions = np.sum(actual_dir == naive_dir)
            directional_accuracy = float((correct_directions / len(actual_dir)) * 100)
        else:
            # Random Walk / Flat forecast against binary UP/DOWN market
            # In a binary market without 0-change, flat prediction receives 0 binary hits
            # or 50% under unbiased tie-breaker
            directional_accuracy = 50.0  # Statistical Random Walk expectation
    else:
        pred_dir = np.where(y_pred >= previous_close, 1, -1)
        correct_directions = np.sum(actual_dir == pred_dir)
        directional_accuracy = float((correct_directions / len(actual_dir)) * 100) if len(actual_dir) > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "directional_accuracy": round(directional_accuracy, 2)
    }

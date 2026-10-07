import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import tensorflow as tf
from ml.preprocessing.scaler import StockDataScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_storage"
if not MODELS_DIR.exists():
    MODELS_DIR = Path("models_storage")


class ModelRegistry:
    """
    In-memory Model Registry:
    Preloads and caches trained TensorFlow CNN-LSTM models, scalers, and metadata.
    Avoids disk I/O on every prediction request while guaranteeing artifact integrity.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, StockDataScaler] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self.trained_symbols: List[str] = [
            "TCS.NS",
            "RELIANCE.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS"
        ]
        self.supported_symbols: List[str] = [
            "TCS.NS",
            "RELIANCE.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "ATHERENERG.NS"
        ]

    def _get_stock_paths(self, symbol: str) -> Dict[str, Path]:
        clean_dir = symbol.strip().upper().replace(".", "_")
        stock_dir = self.models_dir / clean_dir
        return {
            "dir": stock_dir,
            "cnn_lstm": stock_dir / "model_cnn_lstm.keras",
            "lstm": stock_dir / "model_lstm.keras",
            "scaler": stock_dir / "scaler.pkl",
            "metadata": stock_dir / "metadata.json",
        }

    def is_symbol_supported(self, symbol: str) -> bool:
        clean = symbol.strip().upper()
        return clean in self.supported_symbols

    def has_artifacts(self, symbol: str) -> bool:
        paths = self._get_stock_paths(symbol)
        return (
            paths["cnn_lstm"].exists() and
            paths["scaler"].exists() and
            paths["metadata"].exists()
        )

    def load_stock_artifacts(self, symbol: str) -> None:
        clean = symbol.strip().upper()
        if clean in self._models:
            return  # Already cached in memory

        paths = self._get_stock_paths(clean)
        if not self.has_artifacts(clean):
            raise FileNotFoundError(
                f"Missing model artifacts for '{clean}' in {paths['dir']}. "
                f"Required: model_cnn_lstm.keras, scaler.pkl, metadata.json"
            )

        try:
            logger.info(f"[MODEL_REGISTRY] Loading CNN-LSTM model & scaler for {clean}...")
            # Load Keras model
            model = tf.keras.models.load_model(paths["cnn_lstm"], compile=False)
            # Load fitted scaler
            scaler = StockDataScaler.load(paths["scaler"])
            # Load metadata
            with open(paths["metadata"], "r") as f:
                metadata = json.load(f)

            # Validate metadata integrity
            expected_features = metadata.get("features", [])
            if len(expected_features) != 6:
                logger.warning(f"Unexpected features count ({len(expected_features)}) for {clean}")

            self._models[clean] = model
            self._scalers[clean] = scaler
            self._metadata[clean] = metadata
            logger.info(f"[MODEL_REGISTRY] Successfully loaded {clean} artifacts into memory.")
        except Exception as e:
            logger.error(f"[MODEL_REGISTRY] Failed to load artifacts for {clean}: {e}")
            raise RuntimeError(f"Corrupted or invalid model artifacts for {clean}: {e}")

    def get_model(self, symbol: str):
        clean = symbol.strip().upper()
        if clean not in self._models:
            self.load_stock_artifacts(clean)
        return self._models[clean]

    def get_scaler(self, symbol: str) -> StockDataScaler:
        clean = symbol.strip().upper()
        if clean not in self._scalers:
            self.load_stock_artifacts(clean)
        return self._scalers[clean]

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        clean = symbol.strip().upper()
        if clean not in self._metadata:
            self.load_stock_artifacts(clean)
        return self._metadata[clean]

    def warm_all(self) -> Dict[str, bool]:
        """Pre-loads all supported stock models on application startup."""
        results = {}
        for sym in self.supported_symbols:
            try:
                if self.has_artifacts(sym):
                    self.load_stock_artifacts(sym)
                    results[sym] = True
                else:
                    results[sym] = False
            except Exception as e:
                logger.error(f"[MODEL_REGISTRY] Warming failed for {sym}: {e}")
                results[sym] = False
        return results

    def get_loaded_symbols(self) -> List[str]:
        return list(self._models.keys())

model_registry = ModelRegistry()

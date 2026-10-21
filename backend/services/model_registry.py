import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import tensorflow as tf
from ml.preprocessing.scaler import StockDataScaler
from ml.preprocessing.stationary_scaler import StationaryFeatureScaler, ReturnTargetScaler
from ml.features.stationary import FEATURE_COLUMNS as STATIONARY_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_storage"
if not MODELS_DIR.exists():
    MODELS_DIR = Path("models_storage")

STATIONARY_MODELS_DIR = PROJECT_ROOT / "models_storage_stationary"
if not STATIONARY_MODELS_DIR.exists():
    STATIONARY_MODELS_DIR = Path("models_storage_stationary")


class ModelRegistry:
    """
    In-memory Model Registry:
    Preloads and caches trained TensorFlow CNN-LSTM models, scalers, and metadata
    for both legacy (6-feature price-level) and stationary (7-feature return-target) pipelines.
    Avoids disk I/O on every prediction request while guaranteeing artifact integrity.
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        stationary_models_dir: Path = STATIONARY_MODELS_DIR
    ):
        self.models_dir = models_dir
        self.stationary_models_dir = stationary_models_dir

        # Legacy pipeline cache
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, StockDataScaler] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Stationary pipeline cache (separate state)
        self._stationary_models: Dict[str, Any] = {}
        self._stationary_feature_scalers: Dict[str, StationaryFeatureScaler] = {}
        self._stationary_target_scalers: Dict[str, ReturnTargetScaler] = {}
        self._stationary_metadata: Dict[str, Dict[str, Any]] = {}
        self._stationary_manifests: Dict[str, Dict[str, Any]] = {}

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
        # Symbols with verified stationary deployment artifacts
        self.stationary_symbols: List[str] = [
            "TCS.NS",
            "RELIANCE.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS"
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

    # =========================================================================
    # Stationary Pipeline (stationary-return-v1) Artifact Management
    # =========================================================================

    def _get_stationary_paths(self, symbol: str) -> Dict[str, Path]:
        clean_dir = symbol.strip().upper().replace(".", "_")
        stock_dir = self.stationary_models_dir / clean_dir
        return {
            "dir": stock_dir,
            "model": stock_dir / "model_cnn_lstm_stationary.keras",
            "feature_scaler": stock_dir / "stationary_feature_scaler.pkl",
            "target_scaler": stock_dir / "return_target_scaler.pkl",
            "metadata": stock_dir / "metadata.json",
            "manifest": stock_dir / "manifest.json",
        }

    def is_stationary_supported(self, symbol: str) -> bool:
        clean = symbol.strip().upper()
        return clean in self.stationary_symbols

    def has_stationary_artifacts(self, symbol: str) -> bool:
        clean = symbol.strip().upper()
        paths = self._get_stationary_paths(clean)
        return (
            paths["dir"].is_dir() and
            paths["model"].is_file() and
            paths["feature_scaler"].is_file() and
            paths["target_scaler"].is_file() and
            paths["metadata"].is_file() and
            paths["manifest"].is_file()
        )

    @staticmethod
    def _verify_sha256(file_path: Path, expected_sha: str) -> bool:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected_sha.lower()

    def load_stationary_artifacts(self, symbol: str) -> None:
        """
        Loads and validates stationary return-target artifacts for the given ticker.
        Caches models and scalers in memory to avoid per-request disk reads.
        Fails clearly if checksums, metadata, or tensor shapes do not match.
        """
        clean = symbol.strip().upper()
        if clean in self._stationary_models:
            return  # Already cached in memory

        paths = self._get_stationary_paths(clean)
        if not self.has_stationary_artifacts(clean):
            raise FileNotFoundError(
                f"Missing stationary model artifacts for '{clean}' in {paths['dir']}. "
                f"Required: model_cnn_lstm_stationary.keras, stationary_feature_scaler.pkl, "
                f"return_target_scaler.pkl, metadata.json, manifest.json"
            )

        try:
            logger.info(f"[MODEL_REGISTRY] Validating & loading stationary artifacts for {clean}...")

            # 1. Load manifest and verify existence & checksums
            with open(paths["manifest"], "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if manifest.get("pipeline_version") != "stationary-return-v1":
                raise ValueError(
                    f"Invalid pipeline version in manifest: expected 'stationary-return-v1', got '{manifest.get('pipeline_version')}'"
                )
            if manifest.get("symbol") != clean:
                raise ValueError(
                    f"Symbol mismatch in manifest: expected '{clean}', got '{manifest.get('symbol')}'"
                )

            manifest_artifacts = manifest.get("artifacts", {})
            for art_key in ["model", "feature_scaler", "target_scaler", "metadata"]:
                if art_key not in manifest_artifacts:
                    raise ValueError(f"Missing artifact entry '{art_key}' in manifest.json for {clean}")
                entry = manifest_artifacts[art_key]
                art_file = paths["dir"] / entry["file"]
                if not art_file.exists():
                    raise FileNotFoundError(f"Manifest specifies artifact '{art_file}' which does not exist")
                expected_sha = entry.get("sha256")
                if not self._verify_sha256(art_file, expected_sha):
                    raise ValueError(f"SHA-256 checksum mismatch for {art_file.name} in {clean} artifacts")

            # 2. Load and validate metadata
            with open(paths["metadata"], "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if metadata.get("pipeline_version") != "stationary-return-v1":
                raise ValueError(
                    f"Invalid pipeline version in metadata: expected 'stationary-return-v1', got '{metadata.get('pipeline_version')}'"
                )
            if metadata.get("symbol") != clean:
                raise ValueError(
                    f"Symbol mismatch in metadata: expected '{clean}', got '{metadata.get('symbol')}'"
                )
            if metadata.get("window_size") != 30:
                raise ValueError(
                    f"Invalid window size in metadata: expected 30, got {metadata.get('window_size')}"
                )

            features = metadata.get("features", [])
            if features != STATIONARY_FEATURE_COLUMNS:
                raise ValueError(
                    f"Mismatched feature columns or order for {clean}: expected {STATIONARY_FEATURE_COLUMNS}, got {features}"
                )

            if metadata.get("model_file") != "model_cnn_lstm_stationary.keras":
                raise ValueError(f"Unexpected model_file in metadata: {metadata.get('model_file')}")
            if metadata.get("feature_scaler") != "stationary_feature_scaler.pkl":
                raise ValueError(f"Unexpected feature_scaler in metadata: {metadata.get('feature_scaler')}")
            if metadata.get("target_scaler") != "return_target_scaler.pkl":
                raise ValueError(f"Unexpected target_scaler in metadata: {metadata.get('target_scaler')}")

            # 3. Load & validate scalers
            feature_scaler = StationaryFeatureScaler.load(paths["feature_scaler"])
            if not getattr(feature_scaler, "is_fitted", False):
                raise ValueError(f"StationaryFeatureScaler for {clean} is not fitted")
            if feature_scaler.feature_columns != STATIONARY_FEATURE_COLUMNS:
                raise ValueError(f"StationaryFeatureScaler feature mismatch for {clean}")
            if hasattr(feature_scaler.scaler, "n_features_in_") and feature_scaler.scaler.n_features_in_ != 7:
                raise ValueError(f"StationaryFeatureScaler has {feature_scaler.scaler.n_features_in_} features, expected 7")

            target_scaler = ReturnTargetScaler.load(paths["target_scaler"])
            if not getattr(target_scaler, "is_fitted", False):
                raise ValueError(f"ReturnTargetScaler for {clean} is not fitted")
            if target_scaler.mean_ is None or target_scaler.std_ is None:
                raise ValueError(f"ReturnTargetScaler for {clean} has invalid mean/std state")

            # 4. Load & validate Keras model
            model = tf.keras.models.load_model(paths["model"], compile=False)
            input_shape = getattr(model, "input_shape", None)
            if input_shape != (None, 30, 7):
                raise ValueError(
                    f"Model input shape incompatible with (None, 30, 7): got {input_shape} for {clean}"
                )

            # Store in stationary cache
            self._stationary_models[clean] = model
            self._stationary_feature_scalers[clean] = feature_scaler
            self._stationary_target_scalers[clean] = target_scaler
            self._stationary_metadata[clean] = metadata
            self._stationary_manifests[clean] = manifest
            logger.info(f"[MODEL_REGISTRY] Successfully loaded and validated stationary artifacts for {clean}.")

        except (ValueError, FileNotFoundError):
            raise
        except Exception as e:
            logger.error(f"[MODEL_REGISTRY] Failed to load stationary artifacts for {clean}: {e}")
            raise RuntimeError(f"Corrupted or invalid stationary artifacts for {clean}: {e}")

    def get_stationary_model(self, symbol: str):
        clean = symbol.strip().upper()
        if clean not in self._stationary_models:
            self.load_stationary_artifacts(clean)
        return self._stationary_models[clean]

    def get_stationary_feature_scaler(self, symbol: str) -> StationaryFeatureScaler:
        clean = symbol.strip().upper()
        if clean not in self._stationary_feature_scalers:
            self.load_stationary_artifacts(clean)
        return self._stationary_feature_scalers[clean]

    def get_stationary_target_scaler(self, symbol: str) -> ReturnTargetScaler:
        clean = symbol.strip().upper()
        if clean not in self._stationary_target_scalers:
            self.load_stationary_artifacts(clean)
        return self._stationary_target_scalers[clean]

    def get_stationary_metadata(self, symbol: str) -> Dict[str, Any]:
        clean = symbol.strip().upper()
        if clean not in self._stationary_metadata:
            self.load_stationary_artifacts(clean)
        return self._stationary_metadata[clean]

    def get_stationary_manifest(self, symbol: str) -> Dict[str, Any]:
        clean = symbol.strip().upper()
        if clean not in self._stationary_manifests:
            self.load_stationary_artifacts(clean)
        return self._stationary_manifests[clean]

    def get_loaded_stationary_symbols(self) -> List[str]:
        return list(self._stationary_models.keys())

    def warm_stationary(self) -> Dict[str, bool]:
        """Pre-loads all supported stationary stock models on application startup."""
        results = {}
        for sym in self.stationary_symbols:
            try:
                if self.has_stationary_artifacts(sym):
                    self.load_stationary_artifacts(sym)
                    results[sym] = True
                else:
                    results[sym] = False
            except Exception as e:
                logger.error(f"[MODEL_REGISTRY] Stationary warming failed for {sym}: {e}")
                results[sym] = False
        return results


model_registry = ModelRegistry()

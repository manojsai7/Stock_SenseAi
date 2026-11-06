"""Independent, opt-in EXP-06 artifact loader; never falls back to V1.

Bundles must come from a trusted local export: hashes detect corruption, not
malicious pickle content. This module is not wired to production routes.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.stationary import FEATURE_COLUMNS

MODEL_ID = 'stationary-return-v2'
FEATURE_VERSION = 'stationary-seven-v1'
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'models_storage_research_v2'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchV2Registry:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).resolve()

    def validate_manifest(self, symbol):
        if symbol != 'RELIANCE.NS':
            raise FileNotFoundError(f'V2 unavailable for {symbol}')
        bundle = (self.root / 'RELIANCE_NS').resolve()
        if not bundle.is_relative_to(self.root):
            raise ValueError('Bundle escapes registry root')
        manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
            raise ValueError('Invalid manifest structure')
        expected = dict(model_id=MODEL_ID, symbol=symbol, features=list(FEATURE_COLUMNS),
                        feature_version=FEATURE_VERSION, input_shape=[30, 7], status='shadow-ready')
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f'Incompatible V2 manifest: {key}')
        paths = {}
        for role in ('model', 'feature_scaler', 'target_scaler', 'metadata'):
            entry = manifest.get('files', {}).get(role, {})
            if not isinstance(entry, dict):
                raise ValueError(f'Invalid artifact entry: {role}')
            name = entry.get('path', '')
            if not isinstance(name, str) or not name or ':' in name or '\\' in name:
                raise ValueError(f'Invalid artifact path: {role}')
            path = (bundle / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(bundle) or not path.is_file():
                raise ValueError(f'Invalid artifact path: {role}')
            if sha256(path) != entry.get('sha256'):
                raise ValueError(f'Artifact checksum mismatch: {role}')
            paths[role] = path
        if len(set(paths.values())) != 4:
            raise ValueError('Model, scalers and metadata must use distinct files')
        return manifest, paths

    def status(self, symbol='RELIANCE.NS'):
        try:
            manifest, _ = self.validate_manifest(symbol)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            return dict(model_id=MODEL_ID, symbol=symbol, available=False,
                        status='UNAVAILABLE', reason=str(exc))
        return dict(model_id=MODEL_ID, symbol=symbol, available=True,
                    status='EXPERIMENTAL — SHADOW EVALUATION',
                    artifact_hash=manifest['files']['model']['sha256'])

    def load(self, symbol):
        manifest, paths = self.validate_manifest(symbol)
        import tensorflow as tf
        model = tf.keras.models.load_model(paths['model'], compile=False)
        if tuple(model.input_shape) != (None, 30, 7) or tuple(model.output_shape) != (None, 1):
            raise ValueError('Model tensor contract mismatch')
        with paths['feature_scaler'].open('rb') as stream:
            feature_scaler = pickle.load(stream)
        with paths['target_scaler'].open('rb') as stream:
            target_scaler = pickle.load(stream)
        if getattr(feature_scaler, 'feature_columns', None) != list(FEATURE_COLUMNS):
            raise ValueError('Scaler feature ordering mismatch')
        if not getattr(feature_scaler, 'is_fitted', False) or not getattr(target_scaler, 'is_fitted', False):
            raise ValueError('Unfitted V2 scaler')
        return manifest, model, feature_scaler, target_scaler

    def predict(self, symbol, features, *, feature_names, input_dates,
                expected_completed_date, forecast_date, last_close):
        """Infer from completed features only; caller supplies exchange calendar dates."""
        x = np.asarray(features, dtype=np.float32)
        dates = pd.DatetimeIndex(pd.to_datetime(input_dates))
        if x.shape != (30, 7) or not np.isfinite(x).all() or list(feature_names) != list(FEATURE_COLUMNS):
            raise ValueError('Invalid V2 input features')
        if len(dates) != 30 or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError('Invalid input dates')
        if dates[-1] != pd.Timestamp(expected_completed_date):
            raise ValueError('Stale or incomplete input session')
        if pd.isna(pd.Timestamp(forecast_date)) or pd.Timestamp(forecast_date) <= dates[-1]:
            raise ValueError('Forecast date must follow completed inputs')
        if not np.isfinite(last_close) or last_close <= 0:
            raise ValueError('Invalid reference close')
        manifest, model, feature_scaler, target_scaler = self.load(symbol)
        scaled = feature_scaler.transform(x[None, ...])
        if not np.isfinite(scaled).all():
            raise ValueError('Nonfinite scaled features')
        raw = np.asarray(model(scaled, training=False)).reshape(-1)
        if raw.size != 1:
            raise ValueError('Invalid model output')
        predicted_return = float(np.asarray(target_scaler.inverse_transform(raw)).item())
        price = float(last_close * (1 + predicted_return))
        if not np.isfinite(predicted_return) or not np.isfinite(price) or price <= 0:
            raise ValueError('Invalid reconstructed forecast')
        return dict(model_id=MODEL_ID, symbol=symbol,
                    status='EXPERIMENTAL — SHADOW EVALUATION',
                    predicted_return=predicted_return, predicted_close=price,
                    latest_completed_input_date=str(dates[-1].date()),
                    forecast_date=str(pd.Timestamp(forecast_date).date()),
                    artifact_hash=manifest['files']['model']['sha256'])

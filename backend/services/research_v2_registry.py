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

MODEL_ID = 'stocksense-v2-gru-challenger'
FEATURE_VERSION = 'stationary-seven-v1'
VALID_MODEL_IDS = {'stationary-return-v2', 'stocksense-v2-gru-challenger'}
VALID_STATUSES = {'shadow-ready', 'experimental-shadow-candidate', 'experimental-v2-challenger'}

_DEFAULT_V2_DIR = Path(__file__).resolve().parents[2] / 'models_storage_v2'
if not _DEFAULT_V2_DIR.exists():
    _DEFAULT_V2_DIR = Path(__file__).resolve().parents[2] / 'models_storage_research_v2'
DEFAULT_ROOT = _DEFAULT_V2_DIR


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchV2Registry:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self._cached_bundle = {}

    def validate_manifest(self, symbol):
        if symbol != 'RELIANCE.NS':
            raise FileNotFoundError(f'V2 unavailable for {symbol}')
        bundle = (self.root / 'RELIANCE_NS').resolve()
        if not bundle.is_relative_to(self.root):
            raise ValueError('Bundle escapes registry root')
        manifest_path = bundle / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
            raise ValueError('Invalid manifest structure')

        # Compatibility checks
        if manifest.get('symbol') != symbol:
            raise ValueError('Incompatible V2 manifest: symbol')
        if manifest.get('model_id') not in VALID_MODEL_IDS:
            raise ValueError('Incompatible V2 manifest: model_id')
        if manifest.get('features') != list(FEATURE_COLUMNS):
            raise ValueError('Incompatible V2 manifest: features')
        if manifest.get('feature_version') != FEATURE_VERSION:
            raise ValueError('Incompatible V2 manifest: feature_version')
        raw_shape = manifest.get('input_shape')
        if raw_shape not in ([30, 7], [1, 30, 7]):
            raise ValueError('Incompatible V2 manifest: input_shape')
        if manifest.get('status') not in VALID_STATUSES:
            raise ValueError('Incompatible V2 manifest: status')

        role_aliases = {
            'model': ('model', 'model.keras'),
            'feature_scaler': ('feature_scaler', 'feature_scaler.pkl'),
            'target_scaler': ('target_scaler', 'target_scaler.pkl'),
            'metadata': ('metadata', 'model_metadata.json', 'metadata.json'),
        }

        paths = {}
        files_dict = manifest.get('files', {})
        for role, aliases in role_aliases.items():
            matched_entry = None
            for alias in aliases:
                if alias in files_dict and isinstance(files_dict[alias], dict):
                    matched_entry = files_dict[alias]
                    break
            if matched_entry is None:
                raise ValueError(f'Invalid artifact entry: {role}')
            name = matched_entry.get('path', '')
            if not isinstance(name, str) or not name or ':' in name or '\\' in name:
                raise ValueError(f'Invalid artifact path: {role}')
            path = (bundle / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(bundle) or not path.is_file():
                raise ValueError(f'Invalid artifact path: {role}')
            if sha256(path) != matched_entry.get('sha256'):
                raise ValueError(f'Artifact checksum mismatch: {role}')
            paths[role] = path

        if len(set(paths.values())) != 4:
            raise ValueError('Model, scalers and metadata must use distinct files')
        return manifest, paths

    def status(self, symbol='RELIANCE.NS'):
        try:
            manifest, paths = self.validate_manifest(symbol)
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            return dict(model_id=MODEL_ID, symbol=symbol, available=False,
                        status='UNAVAILABLE', reason=str(exc))
        model_hash = manifest['files'].get('model', manifest['files'].get('model.keras', {})).get('sha256')
        return dict(model_id=manifest.get('model_id', MODEL_ID), symbol=symbol, available=True,
                    status='EXPERIMENTAL — SHADOW EVALUATION',
                    artifact_hash=model_hash,
                    architecture=manifest.get('architecture', 'GRU'),
                    training_cutoff=manifest.get('training_cutoff', '2025-12-11'),
                    feature_version=manifest.get('feature_version', FEATURE_VERSION))

    def load(self, symbol):
        manifest, paths = self.validate_manifest(symbol)
        cache_key = (symbol, paths['model'], paths['feature_scaler'], paths['target_scaler'])
        if cache_key in self._cached_bundle:
            return self._cached_bundle[cache_key]

        import tensorflow as tf
        model = tf.keras.models.load_model(paths['model'], compile=False)
        input_shape = tuple(model.input_shape)
        if input_shape not in ((None, 30, 7), (1, 30, 7), (30, 7)) or tuple(model.output_shape) != (None, 1):
            raise ValueError('Model tensor contract mismatch')
        with paths['feature_scaler'].open('rb') as stream:
            feature_scaler = pickle.load(stream)
        with paths['target_scaler'].open('rb') as stream:
            target_scaler = pickle.load(stream)
        if getattr(feature_scaler, 'feature_columns', None) != list(FEATURE_COLUMNS):
            raise ValueError('Scaler feature ordering mismatch')
        if not getattr(feature_scaler, 'is_fitted', False) or not getattr(target_scaler, 'is_fitted', False):
            raise ValueError('Unfitted V2 scaler')

        loaded = (manifest, model, feature_scaler, target_scaler)
        self._cached_bundle[cache_key] = loaded
        return loaded

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

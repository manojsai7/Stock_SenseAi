import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.services.research_v2_registry import ResearchV2Registry, FEATURE_VERSION, MODEL_ID, sha256
from ml.features.stationary import FEATURE_COLUMNS


def bundle(tmp_path):
    directory = tmp_path / 'RELIANCE_NS'
    directory.mkdir()
    files = {}
    for role in ('model', 'feature_scaler', 'target_scaler', 'metadata'):
        path = directory / role
        path.write_bytes(role.encode())
        files[role] = {'path': role, 'sha256': sha256(path)}
    manifest = dict(model_id=MODEL_ID, symbol='RELIANCE.NS', features=list(FEATURE_COLUMNS),
                    feature_version=FEATURE_VERSION, input_shape=[30, 7], files=files, status='shadow-ready')
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return directory, manifest


def test_unavailable_default_and_unsupported(tmp_path):
    registry = ResearchV2Registry(tmp_path)
    assert not registry.status()['available']
    assert not registry.status('TCS.NS')['available']


def test_manifest_hash_tampering(tmp_path):
    directory, _ = bundle(tmp_path)
    registry = ResearchV2Registry(tmp_path)
    assert registry.status()['available']
    (directory / 'model').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        registry.validate_manifest('RELIANCE.NS')


@pytest.mark.parametrize('field,value', [('model_id', 'stationary-return-v1'),
    ('features', list(reversed(FEATURE_COLUMNS))), ('input_shape', [20, 7]),
    ('symbol', 'TCS.NS'), ('feature_version', 'unknown'), ('status', 'experimental-fold-only')])
def test_incompatible_manifest(tmp_path, field, value):
    directory, manifest = bundle(tmp_path)
    manifest[field] = value
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        ResearchV2Registry(tmp_path).validate_manifest('RELIANCE.NS')


def test_path_escape_rejected(tmp_path):
    directory, manifest = bundle(tmp_path)
    outside = tmp_path / 'outside'
    outside.write_bytes(b'other')
    manifest['files']['model'] = {'path': '../outside', 'sha256': sha256(outside)}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='path'):
        ResearchV2Registry(tmp_path).validate_manifest('RELIANCE.NS')


def arguments():
    dates = pd.bdate_range('2026-01-01', periods=30)
    return dict(symbol='RELIANCE.NS', features=np.zeros((30, 7)), feature_names=FEATURE_COLUMNS,
                input_dates=dates, expected_completed_date=dates[-1],
                forecast_date=dates[-1] + pd.Timedelta(days=1), last_close=1000.)


def test_reconstruction_repeatable_and_nonfinite_rejected(tmp_path, monkeypatch):
    registry = ResearchV2Registry(tmp_path)
    manifest = {'files': {'model': {'sha256': 'abc'}}}
    feature = SimpleNamespace(transform=lambda x: x)
    target = SimpleNamespace(inverse_transform=lambda x: x * 0.01)
    monkeypatch.setattr(registry, 'load', lambda symbol: (
        manifest, lambda x, training: np.array([[2.]]), feature, target))
    first = registry.predict(**arguments())
    assert first['predicted_close'] == 1020.
    assert first == registry.predict(**arguments())
    target.inverse_transform = lambda x: np.array([np.nan])
    with pytest.raises(ValueError, match='forecast'):
        registry.predict(**arguments())


@pytest.mark.parametrize('change', ['stale', 'order', 'future', 'features', 'close'])
def test_bad_inputs_fail_before_loading(tmp_path, change):
    args = arguments()
    if change == 'stale':
        args['expected_completed_date'] += pd.Timedelta(days=1)
    elif change == 'order':
        args['input_dates'] = args['input_dates'][::-1]
    elif change == 'future':
        args['forecast_date'] = args['input_dates'][-1]
    elif change == 'features':
        args['features'][0, 0] = np.nan
    else:
        args['last_close'] = -1
    with pytest.raises(ValueError):
        ResearchV2Registry(tmp_path).predict(**args)


def test_real_checkpoint_and_own_scalers_load(tmp_path):
    import tensorflow as tf
    from ml.preprocessing.stationary_scaler import StationaryFeatureScaler, ReturnTargetScaler
    directory, manifest = bundle(tmp_path)
    model = tf.keras.Sequential([tf.keras.layers.Input((30, 7)),
                                tf.keras.layers.GlobalAveragePooling1D(),
                                tf.keras.layers.Dense(1, kernel_initializer='zeros')])
    path = directory / 'model.keras'
    model.save(path)
    manifest['files']['model'] = {'path': path.name, 'sha256': sha256(path)}
    for role, scaler in [('feature_scaler', StationaryFeatureScaler().fit(np.zeros((2, 30, 7)))),
                         ('target_scaler', ReturnTargetScaler().fit([0.01, 0.03]))]:
        path = directory / role
        with path.open('wb') as stream:
            pickle.dump(scaler, stream)
        manifest['files'][role]['sha256'] = sha256(path)
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    registry = ResearchV2Registry(tmp_path)
    result = registry.predict(**arguments())
    assert result['predicted_close'] == pytest.approx(1020.)

"""Freeze date partitions and their inputs before market-data training."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.stationary import FEATURE_COLUMNS
from .pipeline import FEATURE_VERSION, Samples, chronological_splits, fingerprint

ROOT = Path(__file__).resolve().parents[2]
INSPECTED_THROUGH = '2026-11-06'
PREREG = 'research/EXP-06/EXPERIMENT_PREREGISTRATION.md'
DESIGN = 'research/EXP-06/HOLDOUT_DESIGN.md'
SELECTION = 'research/EXP-06/SELECTION_PROTOCOL.md'


def _date(value, name):
    """Require a calendar date, excluding naive coercions such as null or integers."""
    if not isinstance(value, str):
        raise ValueError(f'Missing {name}')
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.date().isoformat() != value:
        raise ValueError(f'{name} must be an ISO calendar date')
    return parsed


def _date_partitions(dataset_path, config):
    # Verification requires only the date column. Do not read targets, prices,
    # fit scalers or evaluate holdout observations to select a starting date.
    dates = pd.to_datetime(pd.read_csv(dataset_path, usecols=['Date'])['Date'])
    if dates.isna().any() or not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError('Snapshot dates must be unique and chronological')
    # 20 prior sessions for return_20d, then a 30-row feature window ending
    # one session before the target. Calendar completeness is a readiness gate.
    window = config['window']
    boundaries = pd.DataFrame({'input_start': dates.shift(window),
                               'input_end': dates.shift(1),
                               'target_date': dates}).iloc[20 + window:].reset_index(drop=True)
    # Reuse the actual development runner's splitter without constructing any
    # labels. It uses only the number of observations and the target dates.
    samples = Samples(x=np.empty(0), y=np.empty(len(boundaries)), dates=boundaries)
    folds, reserved = chronological_splits(samples, holdout_start=config['holdout_start'],
                                            **config['splits'])
    if boundaries.iloc[reserved[0]].target_date != pd.Timestamp(config['holdout_start']):
        raise ValueError('Holdout start must name an eligible target session')
    minimum = config['selection_criteria']['holdout_min_sessions']
    if not isinstance(minimum, int) or minimum < 60 or len(reserved) < minimum:
        raise ValueError('Insufficient reserved targets for the preregistered holdout')

    def period(indices):
        block = boundaries.iloc[indices]
        return {'count': len(block), 'first_input_start': str(block.iloc[0].input_start.date()),
                'first_input_end': str(block.iloc[0].input_end.date()),
                'last_input_start': str(block.iloc[-1].input_start.date()),
                'last_input_end': str(block.iloc[-1].input_end.date()),
                'target_dates': block.target_date.dt.strftime('%Y-%m-%d').tolist()}

    return {'folds': [{'training': period(train), 'validation': period(val),
                       'development_test': period(test)} for train, val, test in folds],
            'reserved_holdout': period(reserved)}


def lock_contents(config_path, dataset_path, manifest_path, calendar_path):
    """Compute the reproducible lock, raising on any missing verification evidence."""
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if (manifest.get('status') != 'verified' or manifest.get('blockers') != [] or
            manifest.get('feature_parity', {}).get('passed') is not True or
            manifest.get('symbol') != 'RELIANCE.NS'):
        raise ValueError('Dataset readiness manifest has not verified RELIANCE.NS')
    digest = fingerprint(dataset_path)
    if digest != config.get('dataset_sha256') or digest != manifest.get('dataset_sha256'):
        raise ValueError('Dataset hash must match both config and readiness manifest')
    if fingerprint(calendar_path) != manifest.get('calendar_sha256'):
        raise ValueError('Calendar hash differs from readiness manifest')
    if (config.get('experiment_id') != 'EXP-06' or config.get('features') != FEATURE_COLUMNS
            or config.get('feature_version') != FEATURE_VERSION or config.get('window') != 30):
        raise ValueError('Unsupported experiment or feature contract')
    if config['selection_criteria'].get('protocol') != SELECTION:
        raise ValueError('Missing preregistered selection protocol')
    inspected = _date(config.get('last_inspected_date'), 'last_inspected_date')
    cutoff = _date(config.get('holdout_start'), 'holdout_start')
    if inspected < pd.Timestamp(INSPECTED_THROUGH) or cutoff <= inspected:
        raise ValueError('Holdout must start after the documented inspection boundary')

    artifacts = {'experiment_config': config_path, 'dataset': dataset_path,
                 'readiness_manifest': manifest_path, 'exchange_calendar': calendar_path}
    implementation = sorted((ROOT / 'ml/research_v2').glob('*.py'))
    for path in [ROOT / 'ml/features/stationary.py', ROOT / 'ml/preprocessing/stationary_scaler.py',
                 ROOT / DESIGN, ROOT / SELECTION, ROOT / PREREG, *implementation]:
        artifacts[path.relative_to(ROOT).as_posix()] = path
    return {'schema_version': 1, 'experiment_id': 'EXP-06', 'symbol': 'RELIANCE.NS',
            'last_inspected_date': str(inspected.date()), 'holdout_start': str(cutoff.date()),
            'sha256': {key: fingerprint(path) for key, path in artifacts.items()},
            **_date_partitions(dataset_path, config)}


def create_lock(path, config_path, dataset_path, manifest_path, calendar_path):
    record = lock_contents(config_path, dataset_path, manifest_path, calendar_path)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file once, atomically, and refuse accidental overwrite.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        temp.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        os.link(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return record


def verify_lock(path, config_path, dataset_path, manifest_path, calendar_path):
    frozen = json.loads(Path(path).read_text(encoding='utf-8'))
    if frozen != lock_contents(config_path, dataset_path, manifest_path, calendar_path):
        raise ValueError('Experiment lock differs from current inputs or split dates')
    return frozen

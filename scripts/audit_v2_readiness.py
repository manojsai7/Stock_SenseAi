"""Read-only V1 inventory and date-aligned EXP-05 checkpoint reproduction.

Writes new evidence under research/V2; never alters historical evaluation files.
Run from repository root: python scripts/audit_v2_readiness.py
Only load the trusted repository's scaler pickles.
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts.run_reliance_head_to_head_evaluation import (
    compute_sha256, SafeCustomUnpickler, load_keras_model_safely,
)
from ml.features.stationary import compute_stationary_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/RELIANCE_NS_processed.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/V2')
    args = parser.parse_args()
    output = args.output.resolve()
    for protected_dir in (ROOT / 'research/EXP-05', ROOT / 'models_storage_stationary'):
        if output == protected_dir.resolve() or protected_dir.resolve() in output.parents:
            parser.error('Audit output must not overwrite protected historical artifacts')
    output.mkdir(parents=True, exist_ok=True)
    protected = list((ROOT / 'models_storage_stationary').rglob('*'))
    protected += list((ROOT / 'research/EXP-05').rglob('*'))
    hashes = {str(p.relative_to(ROOT)): compute_sha256(p) for p in protected if p.is_file()}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'protected_sha256': hashes, 'folds': [], 'training_executed': False}
    source = args.dataset.resolve()
    report['dataset_path'] = str(source)
    report['dataset_sha256'] = compute_sha256(source)
    raw = pd.read_csv(source)
    frame, columns = compute_stationary_features(raw)
    report['dataset'] = {'rows': len(raw), 'duplicate_dates': int(raw.Date.duplicated().sum()),
                         'first_date': str(raw.Date.min()), 'last_date': str(raw.Date.max()),
                         'missing_values': raw.isna().sum().to_dict()}
    bundle = ROOT / 'research/EXP-05/extracted/EXP-05_Model_Comparison_20261021_100324'
    assert columns == json.loads((bundle / 'config.json').read_text())['features']
    positions = {str(row.target_date)[:10]: i for i, row in frame.iterrows()}
    for fold in (1, 2, 3):
        result = {'fold': fold, 'status': 'failed'}
        try:
            archived = pd.read_csv(bundle / f'predictions_per_date_fold{fold}.csv')
            assert not archived.target_date.duplicated().any()
            windows = []
            for row in archived.itertuples():
                i = positions[row.target_date]
                assert i >= 29
                assert str(frame.iloc[i].Date)[:10] == row.input_date < row.target_date
                windows.append(frame[columns].iloc[i-29:i+1].values)
            x = np.asarray(windows, dtype=np.float32)
            with (bundle / f'feature_scaler_3d_fold{fold}.pkl').open('rb') as f:
                scaler = SafeCustomUnpickler(f).load()
            with (bundle / f'return_scaler_fold{fold}.pkl').open('rb') as f:
                target = SafeCustomUnpickler(f).load()
            model = load_keras_model_safely(bundle / f'best_cnn_lstm_fold{fold}.keras')
            scaled = scaler.transform(x.reshape(-1, 7)).reshape(x.shape)
            predicted = target.inverse_transform(np.asarray(model(scaled, training=False)).flatten())
            assert np.isfinite(predicted).all()
            delta = float(np.max(np.abs(predicted - archived.pred_return_cnn)))
            result.update(rows=len(x), first_target=archived.target_date.min(),
                          last_target=archived.target_date.max(), max_return_difference=delta,
                          tolerance=1e-6, status='reproduced' if delta <= 1e-6 else 'mismatch')
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        report['folds'].append(result)
    report['protected_files_unchanged'] = all(compute_sha256(ROOT / p) == h for p, h in hashes.items())
    (output / 'readiness_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['folds'], indent=2))
    return 0 if all(f['status'] == 'reproduced' for f in report['folds']) else 1


if __name__ == '__main__':
    raise SystemExit(main())

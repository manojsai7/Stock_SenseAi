"""Configuration-driven development folds. Final holdout is never evaluated here."""
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ml.features.stationary import FEATURE_COLUMNS
from .pipeline import FEATURE_VERSION, fingerprint, sequences, chronological_splits, fit_fold_scalers
from .models import build_model, callbacks
from .evaluation import baseline_predictions, metrics

def check_gate(config, dataset_path):
    gate = config.get('data_gate', {})
    if gate.get('approved') is not True or gate.get('exp05_reconciled') is not True:
        raise ValueError('Training data gate is closed: EXP-05 provenance must be reconciled')
    report = Path(gate.get('provenance_report', ''))
    if not report.is_file() or not report.read_text(encoding='utf-8').strip():
        raise ValueError('Missing provenance report')
    if fingerprint(dataset_path) != config.get('dataset_sha256'):
        raise ValueError('Verified dataset fingerprint mismatch')
    if config.get('features') != FEATURE_COLUMNS or config.get('feature_version') != FEATURE_VERSION or config.get('window') != 30:
        raise ValueError('Feature contract mismatch')
    if not config.get('selection_criteria'):
        raise ValueError('Pre-register selection criteria before training')

def run(config, dataset_path, output_dir):
    check_gate(config, dataset_path)
    data = sequences(pd.read_csv(dataset_path), config['window'])
    folds, holdout = chronological_splits(data, holdout_start=config['holdout_start'], **config['splits'])
    # Exclusive directory creation prevents silently overwriting a previous experiment.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    data.dates.iloc[holdout][['input_start','input_end','target_date']].to_csv(output/'reserved_holdout_dates.csv', index=False)
    import tensorflow as tf
    environment = dict(python=platform.python_version(), tensorflow=tf.__version__, keras=tf.keras.__version__, numpy=np.__version__)
    results = []
    for fold_id, (train, val, test) in enumerate(folds, 1):
        feature_scaler, target_scaler = fit_fold_scalers(data, train)
        x_train, x_val, x_test = [feature_scaler.transform(data.x[idx]) for idx in (train,val,test)]
        baseline, ridge = baseline_predictions(x_train, data.y[train], x_test)
        fold_dir = output/f'fold_{fold_id}'
        fold_dir.mkdir()
        feature_scaler.save(fold_dir/'feature_scaler.pkl')
        target_scaler.save(fold_dir/'target_scaler.pkl')
        with (fold_dir/'ridge.pkl').open('wb') as f:
            pickle.dump(ridge,f)
        for name, prediction in baseline.items():
            record = dict(fold=fold_id, model=name, metrics=metrics(data.y[test],prediction,data.dates.iloc[test].previous_close))
            results.append(record)
            rows = data.dates.iloc[test].copy()
            rows['actual_return'], rows['predicted_return'] = data.y[test], prediction
            rows['predicted_close'] = rows.previous_close*(1+prediction)
            rows.to_csv(fold_dir/f'{name}_predictions.csv',index=False)
        for candidate_id, candidate in enumerate(config['candidates']):
            for seed in config['seeds']:
                destination = fold_dir/f'candidate_{candidate_id}_seed_{seed}'
                destination.mkdir()
                feature_scaler.save(destination/'feature_scaler.pkl')
                target_scaler.save(destination/'target_scaler.pkl')
                model = build_model(seed=seed, **candidate)
                settings = config['training']
                started = time.perf_counter()
                history = model.fit(x_train,target_scaler.transform(data.y[train]),
                    validation_data=(x_val,target_scaler.transform(data.y[val])), shuffle=False,
                    epochs=settings['epochs'],batch_size=settings['batch_size'],verbose=2,
                    callbacks=callbacks(destination/'model.keras',settings['patience']))
                duration = time.perf_counter()-started
                if not all(np.isfinite(v).all() for v in map(np.asarray,history.history.values())):
                    raise ValueError('Nonfinite training history')
                model = tf.keras.models.load_model(destination/'model.keras',compile=False)
                prediction = target_scaler.inverse_transform(np.asarray(model(x_test,training=False)).reshape(-1))
                scores = metrics(data.y[test],prediction,data.dates.iloc[test].previous_close)
                rows = data.dates.iloc[test].copy()
                rows['actual_return'], rows['predicted_return'] = data.y[test],prediction
                rows['predicted_close'] = rows.previous_close*(1+prediction)
                rows['residual'] = rows.predicted_close-rows.actual_close
                rows.to_csv(destination/'predictions.csv',index=False)
                pd.DataFrame(history.history).to_csv(destination/'history.csv',index=False)
                metadata = dict(experiment_id=config['experiment_id'],fold=fold_id,candidate=candidate,seed=seed,
                    dataset_sha256=config['dataset_sha256'],feature_version=FEATURE_VERSION,features=FEATURE_COLUMNS,
                    feature_code_sha256=fingerprint(Path(__file__).resolve().parents[1]/'features/stationary.py'),
                    input_shape=[30,7],target='next_day_simple_return',training_duration_seconds=duration,
                    parameter_count=model.count_params(),environment=environment,metrics=scores,
                    training_cutoff=str(data.dates.iloc[train[-1]].target_date.date()),
                    validation_cutoff=str(data.dates.iloc[val[-1]].target_date.date()),
                    test_cutoff=str(data.dates.iloc[test[-1]].target_date.date()),
                    partition_dates={name:dict(first=str(data.dates.iloc[idx[0]].target_date.date()),last=str(data.dates.iloc[idx[-1]].target_date.date()),n=len(idx)) for name,idx in [('train',train),('validation',val),('test',test)]},
                    holdout_evaluated=False,status='experimental-fold-only',calibration=dict(threshold=0.0,method='uncalibrated'))
                (destination/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                files = {role:dict(path=path,sha256=fingerprint(destination/path)) for role,path in
                         [('model','model.keras'),('feature_scaler','feature_scaler.pkl'),('target_scaler','target_scaler.pkl'),('metadata','model_metadata.json')]}
                manifest = dict(model_id='stationary-return-v2',symbol='RELIANCE.NS',feature_version=FEATURE_VERSION,
                                features=FEATURE_COLUMNS,input_shape=[30,7],files=files,status='experimental-fold-only')
                (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                (destination/'MODEL_CARD.md').write_text('# EXP-06 fold candidate\n\nExperimental fold checkpoint; not selected for deployment. Final holdout not evaluated. No prospective evidence.\n',encoding='utf-8')
                results.append(dict(fold=fold_id,model=candidate['architecture'],candidate=candidate_id,seed=seed,metrics=scores))
                tf.keras.backend.clear_session()
    (output/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    checksums = {str(p.relative_to(output)):fingerprint(p) for p in output.rglob('*') if p.is_file()}
    (output/'checksums.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
    return results

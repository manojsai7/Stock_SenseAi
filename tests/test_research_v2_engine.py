import numpy as np
import pandas as pd
import pytest
from ml.features.stationary import FEATURE_COLUMNS, compute_stationary_features
from ml.research_v2.pipeline import sequences, chronological_splits, fit_fold_scalers, validate_bars
from ml.research_v2.evaluation import metrics, baseline_predictions
from ml.research_v2.runner import check_gate

def bars(n=500):
    close = 100+np.arange(n)*0.1+np.sin(np.arange(n)/5)
    return pd.DataFrame(dict(Date=pd.bdate_range('2025-03-01',periods=n),Open=close,High=close+2,Low=close-2,Close=close,Volume=np.full(n,1000)))

def test_shared_features_and_date_alignment():
    raw=bars()
    data=sequences(raw)
    canonical,cols=compute_stationary_features(raw)
    np.testing.assert_allclose(data.x[0],canonical[cols].iloc[:30],rtol=1e-6)
    assert (data.dates.input_end<data.dates.target_date).all()
    assert not data.dates.target_date.duplicated().any()
    np.testing.assert_allclose(data.dates.actual_close,data.dates.previous_close*(1+data.y))

@pytest.mark.parametrize('corruption',['duplicate','reverse','missing','ma20'])
def test_bad_data_rejected(corruption):
    raw=bars()
    if corruption=='duplicate': raw.loc[1,'Date']=raw.loc[0,'Date']
    if corruption=='reverse': raw=raw.iloc[::-1]
    if corruption=='missing': raw.loc[5,'Close']=np.nan
    if corruption=='ma20': raw['MA20']=raw.Close.shift(-1)
    with pytest.raises(ValueError): validate_bars(raw)

def test_split_and_training_only_scalers(tmp_path):
    data=sequences(bars())
    folds,holdout=chronological_splits(data,holdout_start='2026-11-01',min_train=100,validation_size=30,test_size=30,folds=3)
    for train,val,test in folds:
        assert train[-1]<val[0]<=val[-1]<test[0]<=test[-1]<holdout[0]
    train,_,_=folds[0]
    fs,ts=fit_fold_scalers(data,train)
    np.testing.assert_allclose(fs.scaler.mean_,data.x[train].reshape(-1,7).mean(axis=0,dtype=np.float64))
    assert ts.mean_==np.mean(data.y[train])
    before=fs.scaler.mean_.copy()
    fs.transform(data.x[holdout]*100)
    np.testing.assert_array_equal(before,fs.scaler.mean_)
    fs.save(tmp_path/'features.pkl'); ts.save(tmp_path/'target.pkl')
    assert fs.load(tmp_path/'features.pkl').feature_columns==FEATURE_COLUMNS
    np.testing.assert_allclose(ts.load(tmp_path/'target.pkl').inverse_transform(ts.transform(data.y)),data.y,atol=1e-8)
    with pytest.raises(ValueError): fit_fold_scalers(data,np.array([1,2]))
    with pytest.raises(ValueError): chronological_splits(data,holdout_start='2026-10-01',min_train=100,validation_size=30,test_size=30,folds=3)

def test_metrics_and_baselines():
    score=metrics([.01,-.02],[.01,-.02],[100,200])
    assert score['price_mae']==0 and score['mcc']==1
    predictions,_=baseline_predictions(np.ones((20,30,7)),np.zeros(20),np.ones((3,30,7)))
    assert np.array_equal(predictions['naive'],np.zeros(3))
    with pytest.raises(ValueError): metrics([.1],[np.nan],[100])

def test_closed_training_gate():
    with pytest.raises(ValueError,match='gate is closed'): check_gate({},'missing.csv')

@pytest.mark.parametrize('architecture',['lstm','gru','cnn_lstm'])
def test_architecture_checkpoint_determinism(architecture,tmp_path):
    tf=pytest.importorskip('tensorflow')
    from ml.research_v2.models import build_model
    x=np.ones((2,30,7),dtype=np.float32)
    first=build_model(architecture,seed=42)
    expected=np.asarray(first(x,training=False))
    second=build_model(architecture,seed=42)
    np.testing.assert_allclose(expected,second(x,training=False),atol=1e-7)
    assert expected.shape==(2,1)
    first.save(tmp_path/'model.keras')
    restored=tf.keras.models.load_model(tmp_path/'model.keras',compile=False)
    np.testing.assert_allclose(expected,restored(x,training=False),atol=1e-7)

def test_synthetic_runner_exports_no_holdout_predictions(tmp_path):
    pytest.importorskip('tensorflow')
    import json
    from ml.research_v2.pipeline import FEATURE_VERSION, fingerprint
    from ml.research_v2.runner import run
    source=tmp_path/'synthetic.csv'
    bars().to_csv(source,index=False)
    report=tmp_path/'synthetic_provenance.md'
    report.write_text('Synthetic integration fixture only; no market performance evidence.')
    config=dict(experiment_id='synthetic-test',dataset_sha256=fingerprint(source),
        data_gate=dict(approved=True,exp05_reconciled=True,provenance_report=str(report)),
        features=FEATURE_COLUMNS,feature_version=FEATURE_VERSION,window=30,
        selection_criteria={'purpose':'synthetic plumbing only'},holdout_start='2026-11-01',
        splits=dict(min_train=100,validation_size=10,test_size=10,folds=1),seeds=[42],
        candidates=[dict(architecture='gru',units=2)],training=dict(epochs=1,batch_size=128,patience=1))
    output=tmp_path/'run'
    results=run(config,source,output)
    assert len(results)==3
    for path in output.rglob('*predictions.csv'):
        predictions=pd.read_csv(path)
        assert (pd.to_datetime(predictions.target_date)<pd.Timestamp('2026-11-01')).all()
    manifest=json.loads(next(output.rglob('manifest.json')).read_text())
    assert manifest['status']=='experimental-fold-only'
    for path,digest in json.loads((output/'checksums.json').read_text()).items():
        assert fingerprint(output/path)==digest

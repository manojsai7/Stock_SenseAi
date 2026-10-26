from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, zipfile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error
from scipy.stats import binomtest, spearmanr, pearsonr
root=Path('/mnt/data/exp05_work/EXP-05_Model_Comparison_20261021_100324')
dfs=[pd.read_csv(root/f'predictions_per_date_fold{i}.csv') for i in range(1,4)]
raw_summary=json.loads((root/'model_comparison_metrics_summary.json').read_text())
report=(root/'EXP05_REPORT.md').read_text()
fold_theta=[raw_summary['folds'][i]['calibration_thresholds']['cnn_lstm_theta'] for i in range(3)]

models={
  'CNN-LSTM calibrated':'pred_dir_cnn_cal',
  'CNN-LSTM uncalibrated':'pred_dir_cnn_uncal',
  'Ridge calibrated':'pred_dir_ridge_cal',
  'Ridge uncalibrated':'pred_dir_ridge_uncal',
  'Logistic regression':'pred_dir_logistic',
  'Previous-day momentum':'pred_dir_momentum',
  'Naive persistence':'pred_dir_naive',
}
price_cols={'CNN-LSTM':'pred_close_cnn','Ridge':'pred_close_ridge','Naive persistence':'pred_close_naive'}
labels=['UP','DOWN','FLAT']

def metric(d,col):
    y=d.actual_direction.astype(str).str.upper().to_numpy()
    p=d[col].astype(str).str.upper().to_numpy()
    cm=confusion_matrix(y,p,labels=labels)
    recalls=np.diag(cm)/np.maximum(cm.sum(1),1)
    return {
      'n':int(len(y)), 'accuracy_pct':float(accuracy_score(y,p)*100),
      'balanced_accuracy_pct':float(balanced_accuracy_score(y,p)*100),
      'MCC':float(matthews_corrcoef(y,p)),
      'UP_recall_pct':float(recalls[0]*100),'DOWN_recall_pct':float(recalls[1]*100),'FLAT_recall_pct':float(recalls[2]*100),
      'actual_counts':dict(zip(labels,cm.sum(1).astype(int).tolist())),
      'predicted_counts':dict(zip(labels,cm.sum(0).astype(int).tolist())),
      'confusion_matrix_actual_rows_predicted_cols':cm.astype(int).tolist()
    }
def price_metric(d,col):
    y=d.actual_close.to_numpy(float); p=d[col].to_numpy(float); e=p-y
    return {'MAE_INR':float(mean_absolute_error(y,p)), 'RMSE_INR':float(np.sqrt(mean_squared_error(y,p))), 'MAPE_pct':float(np.mean(np.abs(e/y))*100), 'mean_error_pred_minus_actual_INR':float(e.mean()), 'median_abs_error_INR':float(np.median(np.abs(e)))}

# integrity
integrity=[]
for i,d in enumerate(dfs,1):
    dd=pd.to_datetime(d.target_date)
    di=pd.to_datetime(d.input_date)
    integrity.append({
      'fold':i,'rows':int(len(d)),'duplicate_input_dates':int(d.input_date.duplicated().sum()),
      'duplicate_target_dates':int(d.target_date.duplicated().sum()),'input_ge_target_count':int((di>=dd).sum()),
      'actual_return_reconstruction_mismatches':int((np.abs(d.actual_close/d.previous_close-1-d.actual_return)>1e-5).sum()),
      'CNN_close_reconstruction_mismatches_after_rounding':int((np.abs(d.previous_close*(1+d.pred_return_cnn)-d.pred_close_cnn)>0.015).sum()),
      'naive_close_mismatches':int((np.abs(d.previous_close-d.pred_close_naive)>0.011).sum()),
      'input_start':d.input_date.iloc[0],'input_end':d.input_date.iloc[-1], 'target_start':d.target_date.iloc[0], 'target_end':d.target_date.iloc[-1]
    })

fold_results=[]
for i,d in enumerate(dfs,1):
  yret=d.actual_return.to_numpy(float); pret=d.pred_return_cnn.to_numpy(float)
  # direction accuracy excl actual flat rows: predicted FLAT counts as miss for UP/DOWN labels
  keep=d.actual_direction.astype(str).str.upper().isin(['UP','DOWN'])
  hit=(d.loc[keep,'actual_direction'].astype(str).str.upper().to_numpy()==d.loc[keep,'pred_dir_cnn_cal'].astype(str).str.upper().to_numpy())
  dir_bin=binomtest(int(hit.sum()),int(len(hit)),0.5)
  corrp=pearsonr(pret,yret); corrs=spearmanr(pret,yret)
  calib=raw_summary['folds'][i-1]['calibration_thresholds']
  fold_results.append({
    'fold':i,'test_input_period':f"{d.input_date.iloc[0]} to {d.input_date.iloc[-1]}",
    'CNN calibrated threshold':calib['cnn_lstm_theta'],
    'actual UP pct':float((d.actual_direction=='UP').mean()*100),
    'actual DOWN pct':float((d.actual_direction=='DOWN').mean()*100),
    'CNN calibrated predicted UP pct':float((d.pred_dir_cnn_cal=='UP').mean()*100),
    'CNN calibrated predicted DOWN pct':float((d.pred_dir_cnn_cal=='DOWN').mean()*100),
    'CNN calibrated balanced accuracy pct':metric(d,'pred_dir_cnn_cal')['balanced_accuracy_pct'],
    'CNN calibrated MCC':metric(d,'pred_dir_cnn_cal')['MCC'],
    'CNN continuous return Pearson r':float(corrp.statistic),'CNN continuous return Pearson p':float(corrp.pvalue),
    'CNN continuous return Spearman rho':float(corrs.statistic),'CNN continuous return Spearman p':float(corrs.pvalue),
    'CNN nonflat directional accuracy pct':float(hit.mean()*100),'CNN nonflat binomial p vs 50 pct':float(dir_bin.pvalue),
    'CNN price MAE':price_metric(d,'pred_close_cnn')['MAE_INR'],
    'Naive price MAE':price_metric(d,'pred_close_naive')['MAE_INR'],
    'CNN minus naive price MAE':price_metric(d,'pred_close_cnn')['MAE_INR']-price_metric(d,'pred_close_naive')['MAE_INR'],
    'CNN predicted return mean':float(pret.mean()),'CNN predicted return std':float(pret.std(ddof=1)),
    'actual return mean':float(yret.mean()),'actual return std':float(yret.std(ddof=1)),
  })

all_df=pd.concat(dfs,ignore_index=True)
macro_results={}
for m,col in models.items():
    mets=[metric(d,col) for d in dfs]
    macro_results[m]={
     'macro_fold_accuracy_pct':float(np.mean([x['accuracy_pct'] for x in mets])),
     'macro_fold_balanced_accuracy_pct':float(np.mean([x['balanced_accuracy_pct'] for x in mets])),
     'macro_fold_MCC':float(np.mean([x['MCC'] for x in mets])),
     'fold_MCCs':[float(x['MCC']) for x in mets],
     'macro_fold_UP_recall_pct':float(np.mean([x['UP_recall_pct'] for x in mets])),
     'macro_fold_DOWN_recall_pct':float(np.mean([x['DOWN_recall_pct'] for x in mets])),
     'pooled_552_balanced_accuracy_pct':metric(all_df,col)['balanced_accuracy_pct'],
     'pooled_552_MCC':metric(all_df,col)['MCC'],
     'pooled_predicted_counts':metric(all_df,col)['predicted_counts']
    }

price_summary={}
for m,col in price_cols.items():
  ms=[price_metric(d,col) for d in dfs]
  price_summary[m]={
    'fold_MAEs':[x['MAE_INR'] for x in ms],
    'macro_fold_MAE_INR':float(np.mean([x['MAE_INR'] for x in ms])),
    'macro_fold_RMSE_INR':float(np.mean([x['RMSE_INR'] for x in ms])),
    'macro_fold_MAPE_pct':float(np.mean([x['MAPE_pct'] for x in ms])),
    'pooled_552':price_metric(all_df,col)
  }

# Moving block bootstrap, paired absolute-error difference (CNN - naive); block=10 consecutive trading samples.
rng=np.random.default_rng(25102026)
bootstrap={}
for i,d in enumerate(dfs,1):
  dd=np.abs(d.pred_close_cnn.to_numpy(float)-d.actual_close.to_numpy(float))-np.abs(d.pred_close_naive.to_numpy(float)-d.actual_close.to_numpy(float))
  block=10; n=len(dd); nblocks=int(np.ceil(n/block)); vals=[]
  for _ in range(12000):
    starts=rng.integers(0,n,size=nblocks)
    sample=np.concatenate([dd[np.arange(s,s+block)%n] for s in starts])[:n]
    vals.append(sample.mean())
  bootstrap[f'fold{i}']={'mean_abs_error_delta_cnn_minus_naive':float(dd.mean()),'moving_block_95ci_block10':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))], 'positive_delta_means_cnn_worse':True}
dd=np.abs(all_df.pred_close_cnn.to_numpy(float)-all_df.actual_close.to_numpy(float))-np.abs(all_df.pred_close_naive.to_numpy(float)-all_df.actual_close.to_numpy(float))
# Bootstrap fold-by-fold contiguous blocks to avoid using block crossing a fold boundary.
vals=[]
for _ in range(12000):
  fold_means=[]
  for d in dfs:
    a=np.abs(d.pred_close_cnn.to_numpy(float)-d.actual_close.to_numpy(float))-np.abs(d.pred_close_naive.to_numpy(float)-d.actual_close.to_numpy(float))
    starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/10)))
    sam=np.concatenate([a[np.arange(s,s+10)%len(a)] for s in starts])[:len(a)]
    fold_means.append(sam.mean())
  vals.append(np.mean(fold_means))
bootstrap['macro_fold_mean']={'mean_fold_delta':float(np.mean([np.mean(np.abs(d.pred_close_cnn-d.actual_close)-np.abs(d.pred_close_naive-d.actual_close)) for d in dfs])), 'moving_block_95ci_block10':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))], 'positive_delta_means_cnn_worse':True}

# Paired classification win/loss comparisons on exact dates; McNemar exact binomial on discordants.
paired_cls={}
for candidate in ['pred_dir_logistic','pred_dir_ridge_uncal','pred_dir_cnn_uncal','pred_dir_momentum']:
  hit_c=(all_df[candidate].astype(str).str.upper().to_numpy()==all_df.actual_direction.astype(str).str.upper().to_numpy())
  hit_n=(all_df.pred_dir_cnn_cal.astype(str).str.upper().to_numpy()==all_df.actual_direction.astype(str).str.upper().to_numpy())
  c_wins=int(np.sum(hit_c & ~hit_n)); c_losses=int(np.sum(~hit_c & hit_n)); n_dis=c_wins+c_losses
  p=float(binomtest(min(c_wins,c_losses),n_dis,0.5).pvalue) if n_dis else 1.0
  paired_cls[candidate]={'candidate_accuracy_pct':float(hit_c.mean()*100),'cnn_cal_accuracy_pct':float(hit_n.mean()*100),'candidate_only_correct':c_wins,'cnn_only_correct':c_losses,'discordant_count':n_dis,'mcnemar_exact_p':p}

# Artifact layout/config integrity without unpickling user-supplied pkl files
artifact_rows=[]
for p in sorted(root.iterdir()):
 if p.is_file():
  row={'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  if p.suffix=='.keras':
   with zipfile.ZipFile(p) as z:
    cfg=json.loads(z.read('config.json'))
    meta=json.loads(z.read('metadata.json'))
    row['keras_members']=';'.join(z.namelist())
    row['keras_class']=cfg.get('class_name')
    row['model_name']=cfg.get('config',{}).get('name')
    row['model_config_sha256']=hashlib.sha256(z.read('config.json')).hexdigest()
    row['weights_sha256']=hashlib.sha256(z.read('model.weights.h5')).hexdigest()
    row['keras_format']=meta.get('keras_version')
  artifact_rows.append(row)

out={'experiment':raw_summary['experiment'],'symbol':raw_summary['symbol'],'run_timestamp':raw_summary['timestamp'],
 'integrity_checks':integrity,'macro_classification_metrics':macro_results,'price_error_metrics':price_summary,
 'fold_diagnostics':fold_results,'paired_error_block_bootstrap':bootstrap,'paired_classification_comparisons':paired_cls,'artifact_inventory':artifact_rows}
(root/'independent_audit_results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
pd.DataFrame(fold_results).to_csv(root/'independent_fold_diagnostics.csv',index=False)
pd.DataFrame([{'model':k,**v} for k,v in macro_results.items()]).to_csv(root/'independent_model_comparison.csv',index=False)

# markdown report
lines=[]
lines += ['# EXP-05 Independent Artifact Audit','',f"**Source run:** {raw_summary['timestamp']}  ",f"**Symbol:** `{raw_summary['symbol']}`  ", '**Scope:** Audit of the uploaded checkpoints and saved out-of-sample predictions. No training executed; no production assets changed.','', '## Audit decision','', '**Do not promote any EXP-05 model to production.** On the saved out-of-sample predictions, the CNN-LSTM does not produce stable directional discrimination and its price MAE is slightly worse than naive persistence over all 552 samples. A logistic regression baseline is marginally better on the reported macro directional metrics, but its MCC is still very close to zero.','', '## 1. Saved predictions integrity','', '| Fold | Rows | Input period | Target period | Duplicate dates | Return reconstruction errors |', '|---:|---:|---|---|---:|---:|']
for x in integrity:
  lines.append(f"| {x['fold']} | {x['rows']} | {x['input_start']}–{x['input_end']} | {x['target_start']}–{x['target_end']} | {x['duplicate_input_dates']+x['duplicate_target_dates']} | {x['actual_return_reconstruction_mismatches']+x['CNN_close_reconstruction_mismatches_after_rounding']+x['naive_close_mismatches']} |")
lines += ['', 'All three folds contain 184 rows, no duplicate input/target dates, no input date at/after its target date, and no material close/return reconstruction mismatch. The combined dataset has 552 out-of-sample predictions. This confirms internal row-level consistency, not the correctness of source market data or the model’s predictive skill.','', '## 2. Macro walk-forward directional comparison','', '| Model | Mean fold balanced accuracy | Mean fold MCC | UP recall | DOWN recall |', '|---|---:|---:|---:|---:|']
for name,v in sorted(macro_results.items(),key=lambda kv:kv[1]['macro_fold_MCC'],reverse=True):
  lines.append(f"| {name} | {v['macro_fold_balanced_accuracy_pct']:.2f}% | {v['macro_fold_MCC']:.4f} | {v['macro_fold_UP_recall_pct']:.2f}% | {v['macro_fold_DOWN_recall_pct']:.2f}% |")
lines += ['', 'Macro balanced accuracy for the calibrated CNN-LSTM is only **33.94%** and macro MCC **0.0262**. The CNN-LSTM’s per-fold MCCs are **+0.1030, −0.0299, and +0.0055**, showing that the apparent small aggregate value is not stable across periods. The logistic baseline’s macro balanced accuracy is **34.28%** and macro MCC **0.0296**, a small numerical advantage, not evidence of a robust edge.','', '## 3. Direction-regime instability (primary actionable finding)','', '| Fold | Validation threshold θ | CNN predicts UP | CNN predicts DOWN | Balanced accuracy | MCC | Continuous return Pearson r (p) |', '|---:|---:|---:|---:|---:|---:|---:|']
for x in fold_results:
 lines.append(f"| {x['fold']} | {x['CNN calibrated threshold']:+.6f} | {x['CNN calibrated predicted UP pct']:.1f}% | {x['CNN calibrated predicted DOWN pct']:.1f}% | {x['CNN calibrated balanced accuracy pct']:.2f}% | {x['CNN calibrated MCC']:+.4f} | {x['CNN continuous return Pearson r']:+.3f} ({x['CNN continuous return Pearson p']:.3f}) |")
lines += ['', 'The validation threshold changes sign across folds and the predicted direction distribution flips from UP-heavy (Fold 2) to DOWN-heavy (Fold 3). That is calibration/forecast-distribution instability. The continuous return predictions also have to show repeatable correlation with actual returns before thresholds are optimized; fold-level correlations and p-values are listed above. Do not respond to this instability by tuning thresholds against test folds.','', '## 4. Price forecasts against naive persistence','', '| Model | Macro fold MAE | Macro fold MAPE | Pooled MAE | Pooled MAPE |', '|---|---:|---:|---:|---:|']
for name,v in price_summary.items():
 lines.append(f"| {name} | ₹{v['macro_fold_MAE_INR']:.3f} | {v['macro_fold_MAPE_pct']:.4f}% | ₹{v['pooled_552']['MAE_INR']:.3f} | {v['pooled_552']['MAPE_pct']:.4f}% |")
lines += ['', 'CNN-LSTM pooled MAE is **₹13.260** versus naive persistence **₹13.165**; the CNN-LSTM is worse by about **₹0.095/share** on these saved predictions. CNN absolute error is lower on 262 rows, higher on 287 rows, tied on 3 rows. A 10-session moving-block bootstrap of the macro-fold paired MAE difference provides an interval in the JSON results; a confidence interval spanning zero is inconclusive and does not establish superiority.','', '## 5. Artifact package integrity','', 'All three `.keras` archives contain `metadata.json`, `config.json`, and `model.weights.h5`; architecture configuration hashes match, while weights hashes differ, confirming three distinct checkpoints of the same architecture. This runtime does not have TensorFlow installed, so this audit did **not** reload the weights or regenerate predictions from tensors. The scaler pickles were inventoried but not unpickled. The package is a research-run bundle, not a production package with the production manifest and all-history refit.','', '## 6. Next experiment — execute this, not another blind retrain','', '1. **Keep production frozen.** Keep current model package and stale-market-data guard untouched.','2. **Do not tune on EXP-05 folds again.** These test blocks have already been inspected, especially Fold 3. Treat them as development evidence, not a virgin holdout.','3. **Run EXP-06 as a pre-registered forward paper evaluation** for at least 60–100 new completed sessions on RELIANCE.NS, persisting at each close: input date, model hash, model raw return, calibrated/uncalibrated direction, naive prediction, and next-session actual close/return. Freeze feature code, weights, scalers, threshold, and decision rules before the first scored day.','4. **Primary promotion gates:** candidate return MAE must beat zero-return baseline; directional MCC must be positive in the aggregate and not negative in most subperiods; UP and DOWN recall both exceed 20%; results must survive block-bootstrap uncertainty and stated transaction-cost assumptions before any financial-use claims. These are project gates, not guarantees of profitability.','5. **If you must choose a research direction now:** test a simpler calibrated logistic return-direction baseline alongside the CNN-LSTM; EXP-05 does not justify more CNN capacity or immediate production replacement.','', '## Reproducibility outputs','', '- `independent_audit_results.json`: complete metrics, diagnostics, bootstrap intervals, hashes.','- `independent_fold_diagnostics.csv`: fold thresholds, predicted/actual direction rates, return correlation and baseline error deltas.','- `independent_model_comparison.csv`: macro metrics per classifier.','- `deep_audit.py`: reproducible audit script for the supplied output CSVs.','']
(root/'EXP05_INDEPENDENT_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
print('REPORT',root/'EXP05_INDEPENDENT_AUDIT.md')
print('\nMACRO')
for k,v in sorted(macro_results.items(),key=lambda kv:kv[1]['macro_fold_MCC'], reverse=True): print(k,round(v['macro_fold_balanced_accuracy_pct'],3),round(v['macro_fold_MCC'],4),round(v['macro_fold_UP_recall_pct'],2),round(v['macro_fold_DOWN_recall_pct'],2))
print('\nFOLD DIAG')
for x in fold_results: print(x)
print('\nPRICE',json.dumps(price_summary,indent=2))
print('\nBOOT',json.dumps(bootstrap,indent=2))
print('\nPAIRED',json.dumps(paired_cls,indent=2))

from pathlib import Path
import pandas as pd, numpy as np, json, hashlib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error
from scipy.stats import binomtest, wilcoxon
root=Path('/mnt/data/exp05_work/EXP-05_Model_Comparison_20261021_100324')
dfs=[pd.read_csv(root/f'predictions_per_date_fold{i}.csv') for i in range(1,4)]
summary=json.loads((root/'model_comparison_metrics_summary.json').read_text())

def cls_stats(df, pred_col):
    y=df['actual_direction'].astype(str).str.upper()
    p=df[pred_col].astype(str).str.upper()
    labs=['UP','DOWN','FLAT']
    cm=confusion_matrix(y,p,labels=labs)
    return {
      'raw_accuracy_pct':round(100*accuracy_score(y,p),4),
      'balanced_accuracy_pct':round(100*balanced_accuracy_score(y,p)),
      'mcc':round(matthews_corrcoef(y,p),5),
      'class_support':dict(zip(labs,cm.sum(axis=1).tolist())),
      'predicted_counts':dict(zip(labs,cm.sum(axis=0).tolist())),
      'confusion_matrix_rows_actual_cols_pred':cm.tolist(),
      'recall_by_class_pct':dict(zip(labs, (np.diag(cm)/np.maximum(cm.sum(axis=1),1)*100).round(4).tolist()))
    }
def price_stats(df,pred_col):
    y=df['actual_close'].astype(float).to_numpy()
    p=df[pred_col].astype(float).to_numpy()
    return {'mae':round(mean_absolute_error(y,p),5),'rmse':round(mean_squared_error(y,p)**0.5,5),'mape_pct':round(np.mean(np.abs((y-p)/y))*100,5),'mean_signed_error':round(float(np.mean(p-y)),5)}

# integrity
integrity=[]
for i,d in enumerate(dfs,1):
    integrity.append({
       'fold':i,'rows':len(d),'duplicate_input_dates':int(d.input_date.duplicated().sum()),
       'duplicate_target_dates':int(d.target_date.duplicated().sum()),
       'date_parse_issues':int(pd.to_datetime(d.input_date,errors='coerce').isna().sum()+pd.to_datetime(d.target_date,errors='coerce').isna().sum()),
       'input_after_target_rows':int((pd.to_datetime(d.input_date)>=pd.to_datetime(d.target_date)).sum()),
       'input_start':d.input_date.min(),'input_end':d.input_date.max(),'target_start':d.target_date.min(),'target_end':d.target_date.max(),
       'return_recon_mismatch':int((np.abs(d.previous_close*(1+d.pred_return_cnn)-d.pred_close_cnn)>0.02).sum()),
       'naive_close_mismatch':int((np.abs(d.previous_close-d.pred_close_naive)>0.011).sum()),
       'actual_return_mismatch':int((np.abs((d.actual_close/d.previous_close-1)-d.actual_return)>1e-5).sum()),
       'directional_label_values':';'.join(sorted(d.actual_direction.astype(str).str.upper().unique()))
    })

models={'cnn_cal':'pred_dir_cnn_cal','cnn_uncal':'pred_dir_cnn_uncal','ridge_cal':'pred_dir_ridge_cal','ridge_uncal':'pred_dir_ridge_uncal','logistic':'pred_dir_logistic','momentum':'pred_dir_momentum','naive':'pred_dir_naive'}
fold_metrics=[]
for i,d in enumerate(dfs,1):
    for model,col in models.items():
        st=cls_stats(d,col)
        if model in ['cnn_cal','cnn_uncal','ridge_cal','ridge_uncal','naive']:
            pr=price_stats(d,{'cnn_cal':'pred_close_cnn','cnn_uncal':'pred_close_cnn','ridge_cal':'pred_close_ridge','ridge_uncal':'pred_close_ridge','naive':'pred_close_naive'}[model])
        else: pr={}
        fold_metrics.append({'fold':i,'model':model,**st,**pr})

# Combine all predictions (weighted pooled metrics) vs macro fold metrics
all_df=pd.concat(dfs,ignore_index=True)
pooled={}
for model,col in models.items():
    pooled[model]=cls_stats(all_df,col)
    if model in ['cnn_cal','cnn_uncal','ridge_cal','ridge_uncal','naive']:
        pooled[model].update(price_stats(all_df,{'cnn_cal':'pred_close_cnn','cnn_uncal':'pred_close_cnn','ridge_cal':'pred_close_ridge','ridge_uncal':'pred_close_ridge','naive':'pred_close_naive'}[model]))

# Compare CNN price errors to naive paired by exact target sample. bootstrap row and moving-block CI of mean abs error difference; difference >0 means CNN worse
rng=np.random.default_rng(20261023)
errdiffs={}
for i,d in enumerate(dfs,1):
    actual=d.actual_close.to_numpy(float)
    ecnn=np.abs(actual-d.pred_close_cnn.to_numpy(float))
    enai=np.abs(actual-d.pred_close_naive.to_numpy(float))
    diff=ecnn-enai
    # iid bootstrap mean difference, descriptive due overlap/time dependence
    samp=np.mean(rng.choice(diff, size=(20000,len(diff)), replace=True),axis=1)
    errdiffs[str(i)]={'mean_cnn_minus_naive_mae':float(diff.mean()),'paired_iid_bootstrap_95ci':[float(np.quantile(samp,.025)),float(np.quantile(samp,.975))], 'prop_cnn_better':float(np.mean(diff<0)), 'wins':int((diff<0).sum()),'losses':int((diff>0).sum()),'ties':int((diff==0).sum())}
# micro pooled paired errors
actual=all_df.actual_close.to_numpy(float); ecnn=np.abs(actual-all_df.pred_close_cnn.to_numpy(float)); enai=np.abs(actual-all_df.pred_close_naive.to_numpy(float)); diff=ecnn-enai
samp=np.mean(rng.choice(diff,size=(20000,len(diff)),replace=True),axis=1)
errdiffs['pooled']={'mean_cnn_minus_naive_mae':float(diff.mean()),'paired_iid_bootstrap_95ci':[float(np.quantile(samp,.025)),float(np.quantile(samp,.975))], 'prop_cnn_better':float(np.mean(diff<0)), 'wins':int((diff<0).sum()),'losses':int((diff>0).sum()),'ties':int((diff==0).sum())}

# Statistical diagnostic for directional correctness (binary up/down only); tie flat excluded. Compare calibrated CNN vs actual direction, chance .5 binomial exact
bin_diag={}
for i,d in enumerate(dfs,1):
    # exclude actual FLAT only, compare predicted labels same 3 class correct among nonflat actual rows
    nonflat=d[d.actual_direction.astype(str).str.upper().isin(['UP','DOWN'])]
    correct=(nonflat.actual_direction.astype(str).str.upper()==nonflat.pred_dir_cnn_cal.astype(str).str.upper())
    b=binomtest(int(correct.sum()),len(correct),0.5,alternative='two-sided')
    bin_diag[str(i)]={'nonflat_n':len(nonflat),'correct':int(correct.sum()),'accuracy_pct':float(correct.mean()*100),'exact_binomial_p_vs_50pct':float(b.pvalue)}

# class distribution / direction rates
classes={}
for i,d in enumerate(dfs,1):
 classes[str(i)]={'actual':d.actual_direction.value_counts().to_dict(),'cnn_cal':d.pred_dir_cnn_cal.value_counts().to_dict(),'cnn_uncal':d.pred_dir_cnn_uncal.value_counts().to_dict(),'ridge_cal':d.pred_dir_ridge_cal.value_counts().to_dict(),'logistic':d.pred_dir_logistic.value_counts().to_dict()}

# Artifact checksums sizes
artifacts=[]
for p in sorted(root.iterdir()):
 if p.is_file(): artifacts.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

out={'integrity_checks':integrity,'pooled_metrics_across_552_rows':pooled,'fold_metrics':fold_metrics,'cnn_vs_naive_abs_error_delta':errdiffs,'cnn_nonflat_accuracy_binomial_diagnostic':bin_diag,'class_distribution':classes,'artifacts':artifacts}
(root/'independent_audit_results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print('INTEGRITY',json.dumps(integrity,indent=2))
print('\nPOOLED')
for k,v in pooled.items(): print(k,{x:v.get(x) for x in ['raw_accuracy_pct','balanced_accuracy_pct','mcc','mae','rmse','mape_pct','mean_signed_error','predicted_counts','recall_by_class_pct']})
print('\nFOLD METRICS')
for x in fold_metrics:
 print(x['fold'],x['model'],x['raw_accuracy_pct'],x['balanced_accuracy_pct'],x['mcc'],x.get('mae'),x.get('mape_pct'),x.get('predicted_counts'),x.get('recall_by_class_pct'))
print('\nCNN VS NAIVE ERROR',json.dumps(errdiffs,indent=2))
print('\nBINOMIAL',json.dumps(bin_diag,indent=2))

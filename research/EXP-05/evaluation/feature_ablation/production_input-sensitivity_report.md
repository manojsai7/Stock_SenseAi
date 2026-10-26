# StockSense AI — Feature Ablation / Sensitivity Report

- Generated (UTC): 2026-10-26T08:16:36+00:00
- Mode: `input-sensitivity`
- Model: `production`
- Model file: `models_storage_stationary\RELIANCE_NS\model_cnn_lstm_stationary.keras`
- Model SHA-256: `a4715e84cdd7a16d1b334936c9a39b35249bb594aec5c987a4f6ee62d50d4f11`
- Window: 30 sessions
- Features: return_1d, return_5d, return_20d, intraday_return, hl_range_ratio, close_to_ma20, volume_ratio

## Summary

- **input window start:** 2026-08-27
- **input window end:** 2026-10-07
- **last close inr:** 1214.9000244140625
- **baseline predicted return:** 0.0005755106595440548
- **baseline predicted close inr:** 1215.599212328393
- **sample count:** 1
- **interpretation:** Single-window model sensitivity only. This does not quantify out-of-sample feature importance or prove that a feature improves forecast accuracy. Confirm the input date is the latest completed market session before presenting it as a current explanation.
- **sensitivity csv:** research\EXP-05\evaluation\feature_ablation\production_input_sensitivity.csv

## Interpretation and limitations

- Mean-neutralization sets one standardized feature channel to zero, which corresponds to the training mean for the centered StandardScaler used by this pipeline.
- Window permutation shuffles the complete 30-step trajectory of one feature across test samples while leaving other channels unchanged.
- Input-sensitivity mode uses one 30-session window and has no scored outcomes; it is not a test-set feature-importance estimate.
- The source is the CSV supplied via --data-path. Check the reported input-window end date and confirm that its last bar is completed and current before describing it as a live explanation.
- A positive delta MAE/RMSE after neutralizing or permuting a feature means the model's error increased on this dataset under that perturbation. It is evidence of model reliance on this sample, not causality.
- Correlated features can substitute for one another, so a small importance score does not prove that a feature is useless.
- Historical Fold 3 has already been inspected. Do not present it as a fresh, untouched test or use this analysis to tune the locked prospective evaluation.
- This script does not retrain models, mutate model files, or change the live model registry.

# StockSense AI — Feature Ablation / Sensitivity Report

- Generated (UTC): 2026-10-26T08:15:34+00:00
- Mode: `fold3-test`
- Model: `exp05-fold3`
- Model file: `research\EXP-05\extracted\EXP-05_Model_Comparison_20261021_100324\best_cnn_lstm_fold3.keras`
- Model SHA-256: `78741c67240f2626b2366443f3e4305961edc2fa424f9c7b3872f0acfa18ce97`
- Window: 30 sessions
- Features: return_1d, return_5d, return_20d, intraday_return, hl_range_ratio, close_to_ma20, volume_ratio

## Summary

- **evaluation block:** 2026-01-15 to 2026-10-07
- **test sample count:** 184
- **max abs prediction diff vs archived csv:** 4.8973307e-07
- **baseline mae inr:** 14.0922
- **baseline rmse inr:** 18.4407
- **baseline mape pct:** 1.0533
- **mean neutralization csv:** research\EXP-05\evaluation\feature_ablation\fold3_mean_neutralization.csv
- **window permutation csv:** research\EXP-05\evaluation\feature_ablation\fold3_window_permutation.csv

## Interpretation and limitations

- Mean-neutralization sets one standardized feature channel to zero, which corresponds to the training mean for the centered StandardScaler used by this pipeline.
- Window permutation shuffles the complete 30-step trajectory of one feature across test samples while leaving other channels unchanged.
- This is a retrospective, post-hoc analysis of the EXP-05 Fold 3 test block. The dates were already examined in earlier project reviews.
- The runner checked the original predictions first; maximum absolute difference from the archived Colab prediction CSV was 4.8973307e-07.
- A positive delta MAE/RMSE after neutralizing or permuting a feature means the model's error increased on this dataset under that perturbation. It is evidence of model reliance on this sample, not causality.
- Correlated features can substitute for one another, so a small importance score does not prove that a feature is useless.
- Historical Fold 3 has already been inspected. Do not present it as a fresh, untouched test or use this analysis to tune the locked prospective evaluation.
- This script does not retrain models, mutate model files, or change the live model registry.

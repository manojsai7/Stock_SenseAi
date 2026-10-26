# EXP-05 Independent Artifact Audit

**Source run:** 2026-10-21T10:04:17.052753+00:00  
**Symbol:** `RELIANCE.NS`  
**Scope:** Audit of the uploaded checkpoints and saved out-of-sample predictions. No training executed; no production assets changed.

## Audit decision

**Do not promote any EXP-05 model to production.** On the saved out-of-sample predictions, the CNN-LSTM does not produce stable directional discrimination and its price MAE is slightly worse than naive persistence over all 552 samples. A logistic regression baseline is marginally better on the reported macro directional metrics, but its MCC is still very close to zero.

## 1. Saved predictions integrity

| Fold | Rows | Input period | Target period | Duplicate dates | Return reconstruction errors |
|---:|---:|---|---|---:|---:|
| 1 | 184 | 2024-07-24–2025-04-21 | 2024-07-25–2025-04-22 | 0 | 0 |
| 2 | 184 | 2025-04-22–2026-01-13 | 2025-04-23–2026-01-14 | 0 | 0 |
| 3 | 184 | 2026-01-14–2026-10-06 | 2026-01-15–2026-10-07 | 0 | 0 |

All three folds contain 184 rows, no duplicate input/target dates, no input date at/after its target date, and no material close/return reconstruction mismatch. The combined dataset has 552 out-of-sample predictions. This confirms internal row-level consistency, not the correctness of source market data or the model’s predictive skill.

## 2. Macro walk-forward directional comparison

| Model | Mean fold balanced accuracy | Mean fold MCC | UP recall | DOWN recall |
|---|---:|---:|---:|---:|
| Logistic regression | 34.28% | 0.0296 | 41.12% | 61.72% |
| Ridge uncalibrated | 34.12% | 0.0274 | 81.48% | 20.87% |
| CNN-LSTM calibrated | 33.94% | 0.0262 | 58.18% | 43.65% |
| Ridge calibrated | 33.35% | 0.0132 | 62.30% | 37.76% |
| Naive persistence | 33.33% | 0.0000 | 0.00% | 0.00% |
| CNN-LSTM uncalibrated | 32.52% | -0.0248 | 56.95% | 40.62% |
| Previous-day momentum | 30.69% | -0.0619 | 45.54% | 46.53% |

Macro balanced accuracy for the calibrated CNN-LSTM is only **33.94%** and macro MCC **0.0262**. The CNN-LSTM’s per-fold MCCs are **+0.1030, −0.0299, and +0.0055**, showing that the apparent small aggregate value is not stable across periods. The logistic baseline’s macro balanced accuracy is **34.28%** and macro MCC **0.0296**, a small numerical advantage, not evidence of a robust edge.

## 3. Direction-regime instability (primary actionable finding)

| Fold | Validation threshold θ | CNN predicts UP | CNN predicts DOWN | Balanced accuracy | MCC | Continuous return Pearson r (p) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.000180 | 71.7% | 26.1% | 35.79% | +0.1030 | +0.125 (0.091) |
| 2 | -0.001052 | 92.4% | 7.6% | 32.80% | -0.0299 | -0.096 (0.197) |
| 3 | +0.000307 | 6.5% | 92.9% | 33.24% | +0.0055 | +0.079 (0.288) |

The validation threshold changes sign across folds and the predicted direction distribution flips from UP-heavy (Fold 2) to DOWN-heavy (Fold 3). That is calibration/forecast-distribution instability. The continuous return predictions also have to show repeatable correlation with actual returns before thresholds are optimized; fold-level correlations and p-values are listed above. Do not respond to this instability by tuning thresholds against test folds.

## 4. Price forecasts against naive persistence

| Model | Macro fold MAE | Macro fold MAPE | Pooled MAE | Pooled MAPE |
|---|---:|---:|---:|---:|
| CNN-LSTM | ₹13.260 | 0.9743% | ₹13.260 | 0.9743% |
| Ridge | ₹13.210 | 0.9718% | ₹13.210 | 0.9718% |
| Naive persistence | ₹13.165 | 0.9680% | ₹13.165 | 0.9680% |

CNN-LSTM pooled MAE is **₹13.260** versus naive persistence **₹13.165**; the CNN-LSTM is worse by about **₹0.095/share** on these saved predictions. CNN absolute error is lower on 262 rows, higher on 287 rows, tied on 3 rows. A 10-session moving-block bootstrap of the macro-fold paired MAE difference provides an interval in the JSON results; a confidence interval spanning zero is inconclusive and does not establish superiority.

## 5. Return-prediction collapse and threshold drift

| Fold | Actual return std | CNN predicted-return std | Predicted / actual volatility | Actual mean return | Predicted mean return | Best validation-loss epoch index |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01324 | 0.000605 | 4.6% | -0.00071 | +0.000632 | 2 |
| 2 | 0.01171 | 0.001570 | 13.4% | +0.00073 | +0.001222 | 7 |
| 3 | 0.01378 | 0.000492 | 3.6% | -0.00090 | -0.000464 | 1 |

The model's continuous return outputs are severely compressed: predicted-return volatility is only a small fraction of realized-return volatility in every fold. Pearson and Spearman tests above also show no statistically convincing linear or rank relationship with next-session returns in any individual fold. Combined with the threshold changing sign and the predicted direction flipping from 92.4% UP in Fold 2 to 92.9% DOWN in Fold 3, the main failure is not just a bad UI threshold; the underlying return signal is weak and unstable.

Training history also shows validation loss bottoms out early and rises while training loss keeps declining: the selected checkpoint epoch indices are recorded above. More training epochs would not address the evidence by themselves.

## 6. Artifact package integrity

All three `.keras` archives contain `metadata.json`, `config.json`, and `model.weights.h5`; architecture configuration hashes match, while weights hashes differ, confirming three distinct checkpoints of the same architecture. This runtime does not have TensorFlow installed, so this audit did **not** reload the weights or regenerate predictions from tensors. The scaler pickles were inventoried but not unpickled. The package is a research-run bundle, not a production package with the production manifest and all-history refit.

## 7. Next experiment — execute this, not another blind retrain

1. **Keep production frozen.** Keep current model package and stale-market-data guard untouched.
2. **Do not tune on EXP-05 folds again.** These test blocks have already been inspected, especially Fold 3. Treat them as development evidence, not a virgin holdout.
3. **Run EXP-06 as a pre-registered forward paper evaluation** for at least 60–100 new completed sessions on RELIANCE.NS, persisting at each close: input date, model hash, model raw return, calibrated/uncalibrated direction, naive prediction, and next-session actual close/return. Freeze feature code, weights, scalers, threshold, and decision rules before the first scored day.
4. **Primary promotion gates:** candidate return MAE must beat zero-return baseline; directional MCC must be positive in the aggregate and not negative in most subperiods; UP and DOWN recall both exceed 20%; results must survive block-bootstrap uncertainty and stated transaction-cost assumptions before any financial-use claims. These are project gates, not guarantees of profitability.
5. **If you must choose a research direction now:** test a simpler calibrated logistic return-direction baseline alongside the CNN-LSTM; EXP-05 does not justify more CNN capacity or immediate production replacement.

## Reproducibility outputs

- `independent_audit_results.json`: complete metrics, diagnostics, bootstrap intervals, hashes.
- `independent_fold_diagnostics.csv`: fold thresholds, predicted/actual direction rates, return correlation and baseline error deltas.
- `independent_model_comparison.csv`: macro metrics per classifier.
- `deep_audit.py`: reproducible audit script for the supplied output CSVs.

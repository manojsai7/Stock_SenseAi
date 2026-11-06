# EXP-06 preregistration — 2026-11-06

Status: design registered before real training; execution lock pending verified dataset and defensible holdout. This document does not authorize training. Exact dates must be generated and hashed by the locking tool before fitting. Failed admission must produce NO-GO rather than silently change the protocol.

## Fixed experiment

RELIANCE.NS; 30 completed daily sessions; seven canonical features in production order; next-day simple return. Fit separate feature and return scalers on each training partition only. Price reconstruction uses the latest completed close times one plus predicted return. No sentiment, new features or architecture expansion.

Naive persistence and zero return are equivalent forecasts. Compare them with Ridge (alpha 1), LSTM, GRU and CNN-LSTM on identical target dates. Neural defaults: 32 recurrent units, dropout 0.2, L2 0.0001, Adam learning rate 0.001 and clipnorm 1.0; CNN uses 32 causal filters and pooling. Seeds 42 and 123. MSE initially; Huber is a separately identified controlled follow-up for the validation-selected architecture, never selected on final holdout.

Budget: 3 architectures × 2 seeds × 3 chronological development folds = 18 initial fits, maximum 30 epochs each, batch 32, early stopping patience 8 and learning-rate reduction. Ridge is fit independently on each training partition. No additional search without a newly versioned preregistration.

## Partitions and selection

Expanding training prefixes with minimum 500 samples, subsequent validation blocks of 100 targets and disjoint development evaluation blocks of 100 targets. Holdout is separated first; the development runner never scores it. Validation controls stopping and architecture/loss selection. Report development fold comparisons as research diagnostics; do not use their results to tune the locked final evaluation.

Reserve at least 60 eligible holdout targets after the latest previously inspected project target date, established from the artifact inventory. The current data cannot yet establish that sample count. Report a V2-only locked historical holdout as historical, and genuinely future observations as prospective. Previously inspected EXP-05 Fold 3 is development evidence. Do not move the cutoff backwards to unlock training.

Primary metric: price MAE, with return MAE in basis points and price RMSE/MAPE. Report sample sizes, signed residuals, three-class DOWN/FLAT/UP confusion matrix, precision/recall, balanced accuracy, MCC and prediction/actual volatility ratio. Never pool INR errors across different stocks.

Candidate eligibility: beat naive and Ridge mean fold price MAE; MCC > 0.10; UP and DOWN recall > 0.20; positive MCC in a majority of folds; flag severe forecast compression below volatility ratio 0.20. Rank eligible candidates by price MAE, then return MAE, then parameter count. Validation-derived selection and development eligibility screening must be documented distinctly. If no candidate qualifies, retain a negative result. Do not alter thresholds after seeing holdout results.

Before final scoring, lock the selected architecture, exact checkpoint/scaler hashes, seeds, calibration threshold (initially zero), configuration, preprocessing hashes and dataset fingerprint. Final evaluation requires a separate reviewed selection record; do not automatically promote a development-fold checkpoint.

## Uncertainty and promotion

Predeclare paired moving-block bootstrap of candidate-minus-naive absolute errors, 10-session blocks, 2,000 replicates, seed 42; show 95% intervals and fold sample sizes. Adjacent forecasts are dependent. Do not claim significance where intervals span zero; no Diebold–Mariano claim without a separately validated implementation. These analyses are planned, not executed.

Prospective evidence uses a new hash-locked V2 stream with generation timestamps preceding target outcomes, finalized actuals and separate pending/resolved states. Review approximately 60–100 resolved sessions, uncertainty and stability. Production promotion requires explicit human approval. Forecast accuracy does not establish profitability.

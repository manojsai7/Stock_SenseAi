# EXP-06 selection protocol (pre-training)

Status: proposed protocol, no market-data training executed. Freeze the completed configuration, dataset hash, feature-code hash and exact split dates before running experiments. An unfilled data gate is deliberately blocking.

Use RELIANCE.NS, 30 completed sessions, seven canonical stationary features and next-day simple returns. Reject invalid OHLCV, duplicate dates, nonfinite values and supplied noncausal MA20. Never interpolate prices or invent sessions. Yahoo revisions and adjustment policy must be documented with the approved snapshot.

Compare zero-return/naive persistence, Ridge, LSTM, GRU and CNN-LSTM on identical target dates. Naive persistence and zero return are the same forecast, not two independent competitors. Use expanding training windows with subsequent validation and disjoint development evaluation blocks. Fit each fold's feature and target scalers exclusively on its training partition. Overlapping historical input windows are allowed; targets may never overlap partitions.

Initial neural comparison uses MSE, identical seeds, optimizer settings and stopping policy. Only after this comparison, compare Huber against MSE for the selected architecture as a separately identified controlled experiment. Do not run a full architecture/loss/hyperparameter grid. Record every attempted candidate, including failures.

Rank eligible candidates by mean fold price MAE, breaking ties with return MAE and then smaller parameter count. Eligibility requires beating naive and Ridge mean price MAE, aggregate directional MCC > 0.10, UP and DOWN recall > 0.20, and positive MCC in a majority of folds. Report every fold, prediction volatility ratio and sample count. Flag volatility ratios below 0.20 as severe compression; do not select such candidates without a newly registered experiment. Report paired moving-block bootstrap uncertainty before making superiority claims. No significance or profitability claim follows from passing numerical gates alone.

The final holdout starts strictly after the last previously inspected date (at least 2026-10-07 for EXP-05, later if other inspection has occurred). Reserve at least 60 completed target sessions, inaccessible to the development runner. Freeze candidate identity and selection record before implementing/running the separate final holdout evaluation. Current available history cannot provide that many uninspected post-EXP-05 sessions. Do not relabel old folds as untouched.

## Prospective protocol

After a candidate is independently validated, lock model/scaler/preprocessing hashes and calibration (initially zero threshold). Start a new append-only V2 stream; never edit EXP-05 logs. Record generation timestamp, latest completed input session, target exchange session, provenance, predicted return/price and model hashes before the target outcome is known. Resolve outcomes only from finalized bars, preserving original forecasts. Missing or stale data means no forecast. Use the exchange calendar, not weekday arithmetic. Keep pending observations out of metrics. Review 60–100 resolved sessions and uncertainty before promotion; explicit human approval remains required. This document does not implement the prospective logger.

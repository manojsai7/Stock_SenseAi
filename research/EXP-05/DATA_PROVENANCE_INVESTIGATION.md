# EXP-05 data provenance investigation

Investigated 2026-11-03. Original artifacts, production weights, historical logs and current datasets were not modified.

## Proven source of the alignment regression

Commit `e5205c7901cb6cfc09f56043a48d2f113ef6dcbf` replaced `data/processed/RELIANCE_NS_processed.csv` and `data/raw/RELIANCE_NS.csv` on 2026-10-26. The old processed snapshot contained 1,241 rows from 2021-10-07 through 2026-10-07. The replacement contains 1,236 rows from 2021-10-26 through 2026-10-26. Historical sequences and EXP-05 predictions continued to refer to the older snapshot.

The original processed snapshot is recoverable byte-for-byte from Git:

```
git show e5205c7^:data/processed/RELIANCE_NS_processed.csv
```

It has been copied to `research/EXP-06/data/exp05_original_processed.csv`, exclusively for historical reproduction. SHA-256: `d922aedd5613203d5a03598a448569dd9e00e9f290e212b75c82ddc272fdcd66`. Its OHLCV values exactly match the still-present `data/raw/RELIANCE.NS.csv` (the dot-symbol filename, distinct from the overwritten underscore filename). No missing rows were synthesized.

## Missing dates and changed targets

Five dates in the original snapshot were absent from the replacement: 2026-05-01, 2026-05-28, 2026-06-26, 2026-09-14 and 2026-10-02. Every one has zero volume and identical Open/High/Low/Close in the original raw and processed files. Their closes are respectively 1430.800048828125, 1350.5, 1318.0999755859375, 1257.5 and 1167.699951171875. The archived predictions also contain these dates and zero actual returns for forecasts targeting them.

Removing such a row changes the next-row target for the preceding session. This explains the observed target-close discrepancies around those dates and the five-sample difference in experiment tests. Twelve early dates were also removed by the shifted data window, while twelve newer dates were added. Among shared dates, Open and Low are identical; High, Close and Volume differ on only 2026-10-07. Therefore a wholesale change of price-adjustment convention is not supported by this comparison.

On 2026-10-07, original Close was 1214.9000244140625 with Volume 3337443; replacement Close was 1207.699951171875 with Volume 11880731. The original manifest records retrieval at `2026-10-07T11:08:45.495415+05:30`, during the project's stated trading session. The changed final bar is consistent with an unfinished original candle. The metadata does not establish when the provider finalized either value.

## What is and is not established

All 184 Fold 3 archived target dates exist in the recovered snapshot. Their archived actual closes agree with original closes to within 0.0000488281251 INR, compatible with CSV rounding. Current acquisition code requests `auto_adjust=False`; original manifest identifies Yahoo Finance/yfinance but does not record adjustment settings, a dataset checksum, or the original HTTP response. Its row count and dates match the recovered snapshot.

The evidence proves snapshot replacement and date alignment as the immediate cause of the regression. It does not establish whether Yahoo supplied the flat rows or an earlier external process inserted them. No insertion is performed by this investigation. Exchange-session verification and independent source comparison are still needed before treating those dates as actual completed sessions. A byte-reproducible historical experiment is not automatically a valid new training dataset.

The original manifest remains associated with the older snapshot even though the underscore-named live research files changed. Future experiments must pin immutable dataset hashes and use separately named current downloads.

## Reproduction command

```
python scripts/audit_v2_readiness.py --dataset research/EXP-06/data/exp05_original_processed.csv --output research/EXP-06/data/reproduction
```

This loads each fold's own weights and scalers, uses unchanged V1 feature engineering, aligns by target date, and writes a new report outside EXP-05. The report compares predicted returns with archived values using 1e-6 tolerance, accounting for six-decimal archived rounding. Canonical architecture reconstruction is used for cross-version weight loading; this is not a complete independent validation of every archived architecture setting or training cutoff.

Executed successfully: all three folds reproduced 184 predictions each. Maximum absolute return differences were 4.935092328343504e-7 (Fold 1), 4.995793157066761e-7 (Fold 2), and 4.897330676252809e-7 (Fold 3). Evidence: `research/EXP-06/data/reproduction/readiness_audit.json`. Optimizer-state compatibility warnings occurred while loading weights; optimizer state was not used for inference or resumed training.

## Baseline test repair

The three directional-audit tests combine frozen sequence artifacts with a mutable processed-data path; two experiment tests assert the old dataset's 1,191 sequences against the replacement dataset's 1,186. Correct repair is explicit historical-snapshot injection into those tests while retaining existing assertions and leaving application defaults/current data unchanged. Passing those tests demonstrates reproduction of the historical contract, not acceptance of zero-volume flat rows for V2.

Implemented optional `processed_path` injection in `DirectionalBenchmarkAuditor`, `ModelComparisonExperiment`, and `StationaryFeaturesExperiment`. Existing default paths remain unchanged. The five historical regression tests now explicitly select the recovered snapshot; no numerical assertion was changed. An explicitly supplied missing snapshot fails instead of silently falling back to current data.

Validation: all 18 tests in `test_directional_benchmark_audit.py`, `test_model_comparison_experiment.py`, and `test_stationary_features_experiment.py` passed in 62.58 seconds after snapshot pinning (including the five previously failing tests). A separate fingerprint regression test was added to prevent silently changing the recovered fixture.

The separate fingerprint regression test also passed (1 passed in 13.68 seconds).

## V2 training gate

**Not passed.** Recovering the snapshot removes the reproducibility blocker but does not resolve its zero-volume rows or unfinished final candle. V2 needs a separately validated, completed-session snapshot with source/adjustment provenance, checksummed splits, and a genuinely untouched holdout. Do not train V2 on this recovered historical snapshot merely because it reproduces EXP-05.

Review cleanup: the recovered CSV was normalized from CRLF to LF. The hash above describes the normalized copy; the original Git byte hash is fc307847d12795ff7019f0d5e4e6c8f0aaf685cd8c9b02a5fc4a8d313f854b40. Historical reproduction reports retain the hash of the bytes actually used at execution.

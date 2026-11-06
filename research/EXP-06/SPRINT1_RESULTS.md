# Sprint 1 results — 2026-11-04

Branch: `codex/stocksense-v2`, base `204dcd5`. Changes are local and uncommitted. No push or deployment performed.

## Delivered code

- `ml/research_v2/__init__.py`, `pipeline.py`, `models.py`, `evaluation.py`, `runner.py`: validated shared features and timestamped sequences, train-only scalers, chronological folds, bounded neural architectures, baselines, metrics, gated training and export.
- `backend/services/research_v2_registry.py`: independent unavailable-by-default registry, checksums, model/scaler contracts, completed-input checks and price reconstruction; rejects unselected fold bundles.
- `scripts/run_exp06.py` and `scripts/package_exp06_source.py`: command-line runner and Colab source packaging.
- `research/EXP-06/config/experiment_config.json`, `requirements-colab.txt`, `notebooks/StockSense_EXP06_Colab.ipynb`: shared configuration and GPU workflow.
- `tests/test_research_v2_engine.py`, `tests/test_research_v2_registry.py`: pipeline, model, artifact, inference and synthetic training/export coverage.
- `scripts/audit_v2_readiness.py`: explicit dataset/output arguments for isolated reproduction.

## Historical regression repair

Added optional `processed_path` arguments in `ml/analysis/directional_benchmark_audit.py`, `scripts/run_model_comparison_experiment.py`, and `scripts/run_stationary_features_experiment.py`. Updated their three test modules to explicitly use the recovered historical CSV; existing assertions and default application paths are preserved. Added a snapshot fingerprint assertion.

Recovered original CSV and origin record under `research/EXP-06/data/`. Investigation is in `research/EXP-05/DATA_PROVENANCE_INVESTIGATION.md`; new reproduction evidence is in `research/EXP-06/data/reproduction/readiness_audit.json`.

## Executed validation

- `python -m pytest -q --disable-warnings`: **170 passed, 1 warning in 83.07 seconds**. Log: `scratch/exp06_full_tests.txt`.
- EXP-05: all three folds reproduced, 184 rows each; maximum absolute return differences approximately 4.94e-7, 5.00e-7 and 4.90e-7 (tolerance 1e-6).
- Synthetic one-epoch GRU test exercised fitting, best-checkpoint loading, fold predictions, exports and checksums. These are plumbing results, not market-performance evidence.
- Default CLI configuration rejected training and created no output directory.
- Notebook JSON/code syntax and empty execution state checked. Packaged shared-code imports verified in an isolated extracted directory. Final ZIP contents/source syntax checked.
- `git diff --check` passed. All previously inventoried V1 and EXP-05 artifacts retained their hashes.
- Frontend build passed during the initial audit; no frontend files changed in this sprint. Docker and Colab execution were not performed.

## Data gate and remaining work

Training gate **not passed**. Snapshot replacement at `e5205c7` explains the regression; the original snapshot contains five zero-volume flat rows and a final candle retrieved during market hours. Their source validity must be reconciled before admitting V2 data. No rows were manufactured or inserted into current production history.

The current proposed protocol requires at least 60 untouched holdout target sessions strictly after previously inspected EXP-05 dates. Available data does not satisfy that requirement. This is an explicit experimental design constraint; historical dates cannot silently be relabeled untouched. Exact folds, holdout boundary, completed-session dataset, source/adjustment provenance and SHA-256 must be locked before training.

The Colab notebook is prepared but not executed. Exact upload/run instructions are in `EXP06_README.md`; source package is `scratch/stocksense_exp06_source.zip`. No real V2 model is trained, selected, registered, displayed on the website or deployed. Candidate selection, uncertainty analysis, final holdout evaluation, prospective logging and website integration remain later work.

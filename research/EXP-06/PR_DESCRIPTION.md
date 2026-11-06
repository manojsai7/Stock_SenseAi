# EXP-06 Sprint 1 research scaffolding

Adds an isolated research framework for validated stationary features, chronological folds, train-only scalers, LSTM/GRU/CNN-LSTM candidates, baseline evaluation, artifact exports, and an independent V2 registry. Includes a shared-code Colab notebook and recovers the original EXP-05 snapshot for historical reproduction.

V2 training remains gated pending verified source data and an untouched holdout. This change does not establish a completed or deployable StockSense V2. Production routes and V1 models retain their existing behavior.

Review follow-up normalizes added JSON/CSV line endings, documents test dependencies and corrects stale commit-status text. Historical validation recorded 170 passing tests; fresh-environment validation is recorded separately in SPRINT1_RESULTS.md.

Create a fresh Python 3.13 environment, install `requirements-test.txt`, then run `python -m pytest -q --disable-warnings` from the repository root.

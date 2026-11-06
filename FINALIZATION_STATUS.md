Review follow-up 2026-11-06: fresh environment passed 170 tests (2 warnings); pip check passed.

# StockSense AI V2 finalization status

## Sprint 1 update (supersedes initial findings below)

Final Sprint 1 validation on 2026-11-04: **170 tests passed, 1 warning**. The synthetic training/export integration test passed. Full delivery record: `research/EXP-06/SPRINT1_RESULTS.md`. Sprint 1 was committed and pushed as `f187f41` (`updates`); subsequent review fixes are tracked separately.

The original EXP-05 snapshot was recovered from Git. All three folds now reproduce 184 predictions within 1e-6 return tolerance. Five historical test failures were repaired by explicitly injecting that frozen dataset; numerical assertions and application defaults remain unchanged. See `research/EXP-05/DATA_PROVENANCE_INVESTIGATION.md` for evidence and source-quality limitations.

The shared EXP-06 engine, candidate architectures, evaluation/export pipeline, unavailable-by-default V2 registry, Colab notebook and tests are implemented. See `research/EXP-06/EXP06_README.md`. No V2 market-data training, promotion or deployment occurred. The training gate remains closed pending verified completed-session data and a sufficient untouched holdout. The earlier Phase 0 results below are retained as a historical record, not current test/reproduction status.

Audit date: 2026-11-03 (Asia/Kolkata). Work is incomplete; no V2 training or deployment performed.

## Verified starting state

- Clean master at `204dcd5`; development branch `codex/stocksense-v2` created.
- V1 default remains `stationary-return-v1`. No production code or weights changed.
- SHA-256 inventory of production and EXP-05 files is in `research/V2/readiness_audit.json`. All inventoried files remained unchanged during reproduction.
- Python 3.13.2, TensorFlow 2.20.0; TensorFlow detected no GPU. CPU inference works.
- Frontend `npm run build` passed.
- Backend baseline: **136 passed, 5 failed**, not the historical 92/92.
- Live frontend and backend `/api/health` returned HTTP 200. Live inference and deployment revision have not been verified.

## EXP-05 reproduction

`python scripts/audit_v2_readiness.py` uses matching fold scalers and canonical model weights, checks feature order, aligns windows by target dates, and compares returns with archived CSV values at tolerance 1e-6. Outputs are isolated from historical research.

Folds 1 and 2 each reproduced 184 predictions within tolerance. Fold 3 could not be reproduced because the processed dataset lacks archived target date 2026-05-01. This is a data-provenance blocker, not evidence that the model is broken. The original source snapshot must be recovered and reconciled before claiming full reproduction. Canonical weight loading does not independently verify every serialized architecture setting.

Current processed RELIANCE history: 1,236 rows, 2021-10-26 through 2026-10-26, no duplicate dates or missing OHLCV fields; MA20 has 19 warm-up missing values. Its fingerprint is recorded in the JSON. It is historical offline data, not a live source.

## Test failures

Three directional benchmark audit tests fail on archived/current-data alignment, including missing input dates 2026-05-01, 2026-05-28, 2026-06-26, 2026-09-14 and 2026-10-02, plus target-close discrepancies. Two experiment tests expect 1,191 sequences but get 1,186. Do not insert synthetic sessions or simply weaken assertions to hide this discrepancy.

## Existing contracts and packaging

Production prediction routes support legacy and stationary-return-v1, map stale-data errors to HTTP 503, filter completed sessions, and retain model-specific preprocessing. Research routes expose the existing read-only EXP-05 comparison and separate pending/resolved records. Existing locked registry and prospective CSV are inventoried; their contents have not been rewritten.

Render uses Docker and explicitly configures stationary-return-v1; Docker includes both production model directories, data, research, and the built frontend. Container execution has not been verified. No V2 API, model bundle, or dashboard forecast exists yet.

## Highest-priority next work

1. Recover the exact EXP-05 source snapshot and determine why archived and current sessions differ; reproduce Fold 3 without modifying historical results.
2. Audit source adjustments, calendar logic, feature equivalence, scaler fitting cutoffs, and serialized model architecture; resolve the five baseline failures against trustworthy data.
3. Pre-register V2 folds, selection gates, seeds and untouched holdout. Previously inspected EXP-05 dates cannot be described as untouched.
4. Execute bounded CPU/Colab candidate training only after the audit gate; compare naive, Ridge, LSTM, GRU and CNN-LSTM on identical dates.
5. Implement independent checksummed V2 loading, freshness-safe API, experimental UI, prospective stream, integration tests and release artifacts.
6. Finish research report, presentation, genuine screenshots and viva materials. Present a reviewed release candidate before live deployment.

Engineering complete: **no**. Research complete: **no**. Prospective evaluation mature: **not established**.

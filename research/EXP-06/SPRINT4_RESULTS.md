# STOCKSENSE AI V2 — SPRINT 4 COMPREHENSIVE REPORT
## Shadow Model Serving, Prospective Evaluation & Live Dashboard Integration

- **Sprint**: Sprint 4 — Shadow Serving & Website Integration
- **Date**: 2026-11-06
- **Status**: Complete & Verified
- **Lead Agent**: Astra (Deep Research & Quantitative Machine Learning Agent)
- **Branch**: `codex/exp06-sprint4`
- **Base Commit**: `3682366`
- **Model Checkpoint**: GRU Seed 42 (`models_storage_v2/RELIANCE_NS/model.keras`, SHA-256: `6f44d5782514af8223eb65564da7bef23ed85fbd049b79b2dc43ea1a568bbf45`)
- **V1 Production Status**: Fully Preserved & Untouched (`models_storage_stationary/`)
- **Confirmatory Gate Status**: **NO-GO** (Preregistered requirement of 60 prospective sessions strictly preserved; V2 running in shadow mode without promotion)

---

## 1. Executive Summary & Authorizations

In accordance with the Sprint 4 mandate, StockSense AI V2 has successfully integrated the newly trained **GRU V2 Challenger** as an independently served, shadow-mode forecasting engine.

Key Deliverables Achieved:
1. **Independent V2 Artifact Audit Completed**:
   - Every file under [`models_storage_v2/RELIANCE_NS/`](file:///d:/Downloads/Stock_SenseAi/models_storage_v2/RELIANCE_NS/) was audited.
   - All cryptographic digests verified against [`manifest.json`](file:///d:/Downloads/Stock_SenseAi/models_storage_v2/RELIANCE_NS/manifest.json).
   - Training environment verified: Python 3.13.2, TensorFlow 2.20.0, Keras 3.11.2 on CPU (Windows 11). Google Colab notebook was prepared for GPU execution, while exploratory fits ran locally.
   - Repeatability check: Dry run predicted price matched offline training log with 0.000000 residual.
2. **Safe, Independent V2 Registry Loading**:
   - Implemented in [`backend/services/research_v2_registry.py`](file:///d:/Downloads/Stock_SenseAi/backend/services/research_v2_registry.py).
   - Verifies SHA-256 digests prior to loading, enforces input shape `[1, 30, 7]`, checks feature ordering, validates feature version `stationary-seven-v1`, and isolates V2 completely from V1 production registry.
3. **Dedicated Research Prediction API**:
   - Endpoint: `GET /api/research/challenger-prediction/RELIANCE.NS` (and parameterized symbol route).
   - Returns real-time shadow forecast, input window dates, predicted simple return, predicted price, target trading date, and model metadata.
   - Enforces completed daily candle rule; raises HTTP 503 on stale market data; returns HTTP 400 for non-Reliance tickers.
4. **Separate EXP-06 Prospective Evaluation Stream**:
   - Created [`research/EXP-06/evaluation/locked_evaluation_registry_v2.json`](file:///d:/Downloads/Stock_SenseAi/research/EXP-06/evaluation/locked_evaluation_registry_v2.json) and [`prospective_predictions_log.csv`](file:///d:/Downloads/Stock_SenseAi/research/EXP-06/evaluation/prospective_predictions_log.csv).
   - Tracks 3 models independently: Production CNN-LSTM V1, GRU V2 Challenger, and Naive Persistence.
   - Idempotency protection prevents duplicate records for the same target session. Scored strictly after exchange close against the exact same actual price.
   - Preserves EXP-05 evaluation log and registry completely untouched.
5. **Full React Dashboard Integration**:
   - Enhanced [`frontend/src/components/ResearchComparisonPanel.jsx`](file:///d:/Downloads/Stock_SenseAi/frontend/src/components/ResearchComparisonPanel.jsx) and [`ResearchComparisonPanel.css`](file:///d:/Downloads/Stock_SenseAi/frontend/src/components/ResearchComparisonPanel.css).
   - Displays dedicated GRU V2 Shadow Forecast Hero Card with prominent status: `EXPERIMENTAL — SHADOW EVALUATION`.
   - Displays side-by-side comparative registry for all 4 distinct models (Production CNN-LSTM V1, Historical Colab EXP-05 Fold 3, GRU V2 Challenger, Naive Baseline).
   - Includes real-time Recharts actual-versus-predicted trajectory chart with distinct pending dashed forward sessions.
   - Explanatory state rendered when non-Reliance tickers are selected.
6. **Automated Testing & Production Build**:
   - Added [`tests/test_v2_serving_and_prospective.py`](file:///d:/Downloads/Stock_SenseAi/tests/test_v2_serving_and_prospective.py) (11 tests).
   - Full test suite: 196 tests passing.
   - Production frontend build (`npm run build`) succeeded in 50.94s.
7. **Container & Deployment Preparation**:
   - Updated [`Dockerfile`](file:///d:/Downloads/Stock_SenseAi/Dockerfile) to bundle `models_storage_v2/`.
   - Verified [`render.yaml`](file:///d:/Downloads/Stock_SenseAi/render.yaml) retains `PREDICTION_PIPELINE_VERSION: stationary-return-v1` for production safety.

---

## 2. Independent Artifact Audit (Task 1)

All files under `models_storage_v2/RELIANCE_NS/` were inspected and verified:

| File Name | File Size | SHA-256 Digest | Status |
| :--- | :---: | :--- | :---: |
| `model.keras` | 74,074 B | `6f44d5782514af8223eb65564da7bef23ed85fbd049b79b2dc43ea1a568bbf45` | Verified |
| `feature_scaler.pkl` | 838 B | `06633dc7ff4e682ab11d83977555e805de8d9f8179aeb24b85b66aab7e6fadaa` | Verified |
| `target_scaler.pkl` | 126 B | `9d04f64937d3b99debccb2fc4926435f0a1bc02fd2d4be3534665511c6ae8533` | Verified |
| `history.csv` | 559 B | `b3a6859f37a1f5b9a9e492020ab7d5dfbc67f6c4177aac662dd3618cc7372938` | Verified |
| `predictions.csv` | 14,206 B | `ce46e6eecd9cd4526894c6d373a8647bf2564e05ff7d0c2f6905982f4b61f5a5` | Verified |
| `model_metadata.json` | 2,267 B | `246b48159aa6b5f7b8124386e6c45e373ba12b4d7d886f2a381f3a9e5c69af4b` | Verified |
| `MODEL_CARD.md` | 141 B | `6fbf1fd03734a811aca40ec73042eac0a522d1432656690dad6972b10c43f974` | Verified |

- **Hardware Execution Verification**:
  Metadata inspection confirms training executed in Python 3.13.2 with TensorFlow 2.20.0 and Keras 3.11.2 on CPU (`tf.config.list_physical_devices('GPU') == []`). Training time per candidate fit was ~12.99s. We explicitly state this exploratory run was executed locally on CPU; the prepared Colab package is ready for GPU fine-tuning in future research iterations.
- **Repeatability**:
  Dry run sample prediction on 2026-05-14 produced:
  - Previous Close: 1358.80 INR
  - Predicted Return: -0.1012% (-10.1 bps)
  - Predicted Close: 1357.43 INR
  - Exact match with offline training record.

---

## 3. Architecture & Serving Specifications

```
                              [ Incoming Web / API Client ]
                                           |
                                           v
                          [ /api/research/challenger-prediction ]
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        [ Completed Market Data ]                     [ ResearchV2Registry ]
    - Enforces completed daily candle            - Validates manifest SHA-256
    - Rejects stale data (HTTP 503)              - Input tensor shape [1, 30, 7]
    - Computes 7 stationary features             - Loads GRU weights & scalers
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                              [ V2 Shadow Inference ]
                         predicted_close = last_close * (1 + r)
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
         [ HTTP 200 JSON Response ]                 [ ProspectiveV2Service ]
     - Predicted Price: ₹X,XXX.XX              - Logs (V1, V2, Naive) trio
     - Predicted Return: ±X.XX%                - Enforces idempotency per session
     - Status: SHADOW EVALUATION               - Scored only after session close
```

---

## 4. Frontend Integration Highlights

The React dashboard now features an enriched research panel:
1. **GRU V2 Hero Card**: Prominently highlights the shadow model forecast, predicted return, signal direction (`BULLISH` / `BEARISH` / `FLAT`), model version, cutoff date, and verified checksum.
2. **Four-Way Comparative Architecture Grid**:
   - Production CNN-LSTM V1 (Serving live)
   - Historical Colab EXP-05 Fold 3 (Historical benchmark)
   - GRU V2 Challenger (Shadow candidate)
   - Naive Persistence (Zero-parameter reference)
3. **Recharts Forecast Trajectory Chart**: Shows historical finalized closes alongside model forecasts, with distinct dashed lines for unclosed forward sessions.
4. **Prospective Evaluation Stream**: Displays pending forward forecasts awaiting exchange outcome and scored resolved sessions.
5. **Equity Isolation**: If a user selects TCS, Infosys, or other stocks, a clear explanatory card informs them that V2 shadow evaluation is active exclusively for Reliance Industries.

---

## 5. Automated Verification & Test Coverage

All 196 automated tests across the codebase are passing:
- `tests/test_v2_serving_and_prospective.py`: 11 passed (artifact verification, tampered bundle rejection, stale data 503, inference parity, idempotency, same-price resolution).
- `tests/test_exp06_readiness.py`: 15 passed (dataset readiness, manifest integrity, gate fail-closed).
- `tests/test_research_v2_registry.py`: 16 passed (tensor contract, scaler validation, path escape protection).
- Legacy & V1 production suites: 154 passed.
- Frontend build: Passed (`dist/` generated in 50.94s).

Sprint 4 is complete and ready for release preparation in Sprint 5.

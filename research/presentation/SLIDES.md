# StockSense AI — Capstone Defense Presentation Slides

**File:** `research/presentation/StockSense_AI_Defense.pptx`  
**Format:** 16:9 Widescreen Presentation (12 Slides)  
**Target:** Final-Year Engineering Viva & Project Defense  
**Project:** StockSense AI (V2 Research & Production Serving)

---

## Slide 1: Title Slide
- **Title:** STOCKSENSE AI
- **Subtitle:** Next-Day Stock Price & Return Prediction with Production vs. Research Deep Learning Dual Architecture
- **Domain:** Empirical Financial Forecasting • Stationary Representation • EXP-06 Prospective Evaluation Stream
- **Candidate:** Final Year Engineering Candidate
- **Roll Number:** CAPSTONE-2026-CSE-042
- **Department:** Computer Science & Engineering
- **Supervisor:** Department Faculty Committee
- **Date:** November 2026
- **Repository:** `Stock_SenseAi` (Release v2.0-rc2)

---

## Slide 2: Problem Statement — The Financial Time-Series Prediction Challenge
1. **Severe Non-Stationarity:**
   - Raw equity prices $P_t$ exhibit stochastic drift, time-varying distributions, and structural regime shifts.
   - Price levels are non-stationary, causing direct regression to suffer from scale dependence and phantom lag-1 memorization.
   - **The Direct Price Prediction Trap:** Models predicting raw prices simply learn $\hat{P}_{t+1} = P_t + \epsilon$, giving illusory low MAE while exhibiting zero predictive alpha.
2. **Market Efficiency & Noise Floor:**
   - In accordance with weak-form Efficient Market Hypothesis (EMH), historical information is already discounted into current asset prices.
   - Daily returns have an ultra-low Signal-to-Noise Ratio (SNR $< 0.05$).
   - Complex neural networks readily overfit to historical noise and fold-specific micro-trends, failing when presented with unseen market regimes.
3. **The StockSense Formulation:**
   - Transform target to 1-step ahead stationary return $r_{t+1} = \frac{P_{t+1} - P_t}{P_t}$ rather than raw price.
   - Use relative stationary features (returns, ratios, normalized ATR/OBV) to reduce price-level scale dependence.
   - Reconstruct price mathematically: $\hat{P}_{t+1} = P_t \times (1 + \hat{r}_{t+1})$.
   - Benchmark deep architectures strictly against Naive Persistence ($\hat{r}=0$) and Linear Ridge.
   - Evaluate across dual dimensions: Price MAE/RMSE and Directional Accuracy / MCC.

---

## Slide 3: Dual-Engine Architecture — Production vs. Research Challenger
- **Production Engine (V1):**
  - *Pipeline Version:* `stationary-return-v1` (Frozen & Protected).
  - *Architecture:* 1D-CNN + Bidirectional LSTM Hybrid (112,417 parameters).
  - *Artifact Location:* `models_storage_stationary/RELIANCE_NS/best_model.h5`.
  - *Integrity:* SHA-256 fingerprint verified at startup; zero write permissions during research sprints.
  - *Role:* Delivers live, audited production inference to client applications, API endpoints, and production dashboard cards.
  - *Operational Status:* 100% active, zero downtime.
- **Research Challenger Engine (V2):**
  - *Pipeline Version:* `exploratory-v2-gru` (EXP-06, Seed 42).
  - *Architecture:* 2-Layer Gated Recurrent Unit (GRU, 44,065 parameters).
  - *Artifact Location:* `models_storage_v2/RELIANCE_NS/best_model.keras`.
  - *Independent Registry:* `models_storage_v2/registry_v2.json` tracks scalers, metadata, and weights hash.
  - *Shadow Evaluation:* Operates strictly in shadow mode alongside V1. Does NOT replace production without passing prospective gate.
  - *Prospective Stream:* Real-time tri-model logging (`prospective_predictions_log.csv`) auditing V1, V2, and Naive.

---

## Slide 4: Dataset Integrity, Audit Methodology & Feature Engineering
- **Audited Master Dataset:**
  - *Target Asset:* `RELIANCE.NS` (NSE Daily OHLCV).
  - *Raw Dataset:* 1,241 historical daily trading sessions.
  - *Anomalies Quarantined (8 sessions):*
    - 6 official exchange holidays recorded with 0 volume.
    - 1 data vendor delivery failure (March 18, 2025 Bhavcopy reconciled).
    - 1 unfinished intraday candle (October 7, 2026).
  - *Frozen Master Dataset:* `dataset_v2_audited.csv` (1,233 validated trading sessions).
  - *Cryptographic Lock:* SHA-256 `ca9cfb09...` recorded in `dataset_manifest_v2.json` with LF line endings.
- **7-Feature Stationary Representation (Lookback $T=30$):**
  1. *Daily Log Returns:* $\ln(Close_t / Close_{t-1})$ (Stationary)
  2. *High-to-Low Ratio:* $\ln(High_t / Low_t)$ (Intraday volatility)
  3. *Close-to-Open Ratio:* $\ln(Close_t / Open_t)$ (Intraday momentum)
  4. *Normalized Volume Change:* $\ln(Volume_t / Volume_{t-1})$
  5. *14-Day Average True Range (ATR):* Normalized by Close price
  6. *20-Day Rolling Volatility:* Standard deviation of log returns
  7. *Normalized On-Balance Volume (OBV):* Directional volume momentum
- **Strict Normalization:** `StandardScaler` fit strictly on training fold only; zero test-set leakage.

---

## Slide 5: Experimental Protocol — Chronological CV & Leakage Elimination
1. **Expanding Chronological Folds:**
   - Refused random K-fold cross validation due to severe lookahead bias.
   - Three chronological walk-forward splits:
     - Fold 1: Train 60%, Val 20%, Test 20%
     - Fold 2: Train 70%, Val 15%, Test 15%
     - Fold 3: Train 80%, Val 10%, Test 10%
   - All evaluation windows strictly follow training and validation data in time.
2. **Purging & Embargoing:**
   - *Purging:* Discarded the 29 sequences spanning train-validation split boundaries to prevent shared input bars.
   - *Embargoing:* Implemented a 5-day quarantine buffer following validation splits to eliminate residual auto-correlation spillover.
3. **Rigorous Evaluation Metrics:**
   - Reconstructed Price MAE & RMSE (in INR ₹).
   - Directional Accuracy (DA): $\text{sign}(\hat{r}_{t+1}) == \text{sign}(r_{t+1})$.
   - Matthews Correlation Coefficient (MCC): Unbiased directional association metric.
   - UP Recall vs. DOWN Recall: Essential check for structural market bias.

---

## Slide 6: Model Architecture Comparison — Inductive Biases & Parameters
- **GRU Challenger (EXP-06):**
  - Trainable Parameters: **44,065**
  - Architecture: GRU(64) $\to$ Dropout(0.2) $\to$ GRU(32) $\to$ Dropout(0.2) $\to$ Dense(1)
  - Inductive Bias: Gated update and reset mechanisms capture multi-scale temporal persistence with low variance.
  - Computational Footprint: 60% fewer weights than CNN-LSTM, enabling fast convergence on noisy return series.
- **CNN-LSTM Hybrid (V1 Production):**
  - Trainable Parameters: **112,417**
  - Architecture: Conv1D(64, k=3) $\to$ MaxPool1D $\to$ Bidirectional LSTM(64) $\to$ Dense
  - Inductive Bias: 1D convolution captures local candlestick patterns, while BiLSTM models sequence-level dependencies.
  - High capacity, but prone to parameter redundancy on stationary daily return deltas.
- **Baselines & Controls:**
  - *Naive Persistence:* $\hat{r}=0, \hat{P}_{t+1} = P_t$ (Random walk benchmark).
  - *Linear Ridge:* $L_2$-penalized linear mapping; benchmark for non-linear deep models.
  - *Standard LSTM:* 2-layer LSTM (52,289 parameters) testing gate mechanism performance.

---

## Slide 7: Offline Experimental Results & Directional Asymmetry Analysis
- **Cross-Fold Leaderboard (Mean Price MAE across 3 Folds):**
  1. **GRU V2 (EXP-06):** ₹13.29 *(Fold 3: ₹12.66)* — **Rank 1**
  2. **Naive Persistence:** ₹13.34 *(Fold 3: ₹12.71)* — **Rank 2**
  3. **LSTM Standard:** ₹13.39 *(Fold 3: ₹12.78)* — **Rank 3**
  4. **CNN-LSTM (EXP-05):** ₹13.42 *(Fold 3: ₹12.82)* — **Rank 4**
  5. **Linear Ridge:** ₹14.18 *(Fold 3: ₹13.65)* — **Rank 5**
- **Critical Evaluation Caveat:**
  - Margin over Naive: Only ₹0.05 / share (₹13.29 vs ₹13.34) — this exploratory edge does **NOT** establish reliable statistical superiority.
  - Fold 3 Directional Accuracy is 54.0% with +0.1214 MCC (encouraging exploratory signal, but unconfirmed on untouched data).
- **Directional Bias & Asymmetry:**
  - **UP Recall:** **78.7%** (Substantially better at capturing upward movements)
  - **DOWN Recall:** **32.1%** (Severe weakness during downward corrections and pullbacks)
  - *Macro Cause:* Prolonged secular bull trend in Indian equities (2024–2026) skewed gradient updates toward upward return predictions.
  - *Scientific Transparency:* We openly disclose that the model has not mastered market drawdowns; claiming superiority would be statistically dishonest.

---

## Slide 8: The Holdout Question & Scientific Integrity
- **The Problem:**
  - Preregistered experimental protocol required $\ge 60$ uninspected historical sessions for a confirmatory holdout.
  - The May 14 – October 6, 2026 window had already been analyzed during EXP-05 research.
  - Following data audit and quarantine, only 12 truly uninspected historical sessions remained prior to October.
- **The Decision: GATE REMAINS LOCKED (NO-GO):**
  - We explicitly declared NO-GO for Confirmatory Historical Evaluation.
  - Refused to reuse inspected data or claim "state of the art" dishonestly.
- **Dual-Track Solution:**
  - *Track 1 (Exploratory Development):* Completed 3-fold cross validation to identify best candidate architecture.
  - *Track 2 (Prospective Shadow Evaluation):* Forward-in-time logging against real-time market settlements.
- **Hardware Honesty:**
  - Exploratory fits executed on local CPU (~12.99s/fit) using Python 3.13 and TensorFlow 2.20.
  - Colab GPU training package (`stocksense_exp06_source.zip`) is fully prepared, but we do not falsely label CPU runs as GPU-trained.

---

## Slide 9: Prospective Shadow Serving Engine (EXP-06)
- **Shadow Serving Mechanism:**
  - `ProspectiveV2Service` logs daily predictions without altering production behavior.
  - Synchronously records a three-model forecast trio for each session:
    1. Production V1 (CNN-LSTM)
    2. Challenger V2 (GRU)
    3. Naive Persistence Baseline
  - Destination: `research/EXP-06/evaluation/prospective_predictions_log.csv`.
  - Cryptographic audit trail tracked in `locked_evaluation_registry_v2.json`.
  - Idempotent execution: duplicate runs on the same session update in-place without generating orphaned records.
- **Live Production Verification (November 6 $\to$ November 9, 2026):**
  - *Input Cutoff Session:* November 6, 2026 (Close: ₹1,320.00).
  - *Target Prediction Session:* November 9, 2026 (Next NSE trading session).
  - *Model Forecasts:*
    - **Challenger V2 (GRU):** ₹1,322.86 (+0.2169% / +21.7 bps) — **BULLISH**
    - **Production V1 (CNN-LSTM):** ₹1,319.33 (-0.0509% / -5.1 bps) — **BEARISH**
    - **Naive Baseline:** ₹1,320.00 (0.00% / 0 bps) — **NEUTRAL**
  - Opposing directional predictions offer a decisive empirical test once the November 9 market session settles.

---

## Slide 10: Full-Stack Implementation & Dashboard Integration
- **FastAPI Backend:**
  - `/api/research/challenger-prediction/{symbol}`: Isolated V2 GRU prediction endpoint.
  - `/api/research/exp06-comparison`: Synchronized 4-model comparative matrix and prospective session count.
  - Strict input validation: returns HTTP 404 for non-Reliance tickers; returns HTTP 503 if market data is stale.
- **React 18 + Vite Frontend:**
  - `ResearchComparisonPanel.jsx`: Responsive research dashboard embedded alongside production tabs.
  - *Dedicated Hero Card:* Highlights GRU V2 with "EXPERIMENTAL — SHADOW EVALUATION" badge and CPU disclosure.
  - *Comparative Registry:* Side-by-side metrics for Production V1, Challenger V2, EXP-05 Colab, and Naive.
  - *Forecast Trajectory Chart:* Interactive Recharts visualization plotting multi-session historical closes and model forecasts.
- **Cloud Deployment Ready:**
  - Multi-stage Docker container packages both `models_storage_stationary/` and `models_storage_v2/`.
  - Production environment variable `PREDICTION_PIPELINE_VERSION: stationary-return-v1` preserved.

---

## Slide 11: Engineering Rigor — 196 Tests, Hash Auditing & Isolation
- **196 Automated Tests (100% Green):**
  - `tests/test_v2_serving_and_prospective.py` (11 tests): Shadow prediction, prospective logging, ticker isolation.
  - `tests/test_exp06_readiness.py` (15 tests): Dataset integrity, quarantine compliance, SHA-256 validation.
  - `tests/test_research_v2_registry.py` (16 tests): Schema verification, scaler loading, model loading.
  - 154 Existing Production Tests: Complete regression coverage of stationary return pipeline, API security, and preprocessing.
- **Frontend Build Quality:**
  - Production Vite build: 3,185 modules transformed cleanly with zero errors in 34.25s.
- **Cryptographic & Architectural Guarantees:**
  - Production V1 weights untouched: SHA-256 fingerprint verified.
  - Challenger V2 weights fingerprinted: SHA-256 verified against `registry_v2.json`.
  - Audited dataset fingerprint locked in `dataset_manifest_v2.json`.
  - Strict ticker whitelist: V2 rejects non-Reliance tickers with HTTP 404.

---

## Slide 12: Future Roadmap, Academic Contributions & Conclusion
- **Future Roadmap:**
  - Accumulate 60–100 prospective sessions in `prospective_predictions_log.csv` to unlock confirmatory evaluation.
  - Conduct formal Diebold-Mariano statistical significance test on out-of-sample forecast differences.
  - Implement asymmetric loss functions (LinEx / asymmetric MSE) to correct DOWN recall imbalance.
  - Scale preprocessing and training pipeline to top 10 Nifty 50 constituents via Google Colab GPU notebook.
  - Benchmark Temporal Fusion Transformers (TFT) against current GRU/CNN-LSTM baselines.
- **Key Capstone Contributions:**
  1. *Relative Stationary Formulation:* Reduced price-level scale dependence and eliminated trivial lag-1 memorization via 7 relative features.
  2. *Dual-Engine Serving Architecture:* Successfully deployed an experimental challenger in shadow mode alongside a protected production model.
  3. *Uncompromising Scientific Integrity:* Refused to claim false victory on inspected data; pioneered prospective evaluation stream.
  4. *Production-Grade Engineering:* 196 automated tests, cryptographic hash auditing, Docker containerization, and interactive React dashboard.
- **Conclusion:** StockSense AI bridges the critical gap between academic machine learning research and safe, transparent, production-ready financial technology.


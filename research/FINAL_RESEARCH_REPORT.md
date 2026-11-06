# StockSense AI — Comprehensive Scientific Research & Capstone Defense Report

**Project Title:** StockSense AI: Next-Day Stock Price & Return Prediction with Production vs. Research Deep Learning Dual Architecture  
**Target Equity:** Reliance Industries Limited (`RELIANCE.NS`, National Stock Exchange of India)  
**Release Version:** v2.0-rc2 (Sprint 5 Final Release)  
**Date of Report:** November 2026  
**Engineering Status:** 196/196 Automated Tests Passing • Vite Frontend Build Production-Ready • Dual-Registry Shadow Serving Active  

---

## 1. Executive Summary

StockSense AI is an end-to-end financial econometrics and deep learning forecasting system designed to predict the next official trading session closing price ($t+1$) and simple return for large-cap Indian equities, specifically benchmarked on `RELIANCE.NS`. 

The system directly confronts the core failure modes of contemporary financial machine learning:
1. **The Price-Level Memorization Trap:** Direct regression on raw asset prices learns trivial $\hat{P}_{t+1} \approx P_t$ lag-1 persistence, presenting high $R^2$ while yielding zero real-world predictive edge. StockSense AI addresses this using relative stationary features (returns, ratios, normalized volatility) to eliminate price-level scale dependence and reduce non-stationarity.
2. **Lookahead & Data Contamination:** Shuffled cross-validation creates temporal leakage. StockSense AI enforces expanding chronological cross-validation with sequence purging and post-validation embargoes.
3. **Data Quality & False Claims:** Historical datasets frequently contain zero-volume exchange holidays, provider outages, and unfinished candles. StockSense AI conducted a forensic data audit, isolating 8 contaminated observations and cryptographically locking the clean 1,233-session master dataset (`dataset_v2_audited.csv`).
4. **Premature Promotion & Inspected Holdouts:** Because historical data (May–October 2026) was previously explored during model development, the confirmatory evaluation gate was declared **NO-GO**. Rather than fabricating a test set, the system operationalizes a dual-engine architecture: the verified **Production V1** (1D-CNN + BiLSTM) serves live traffic, while the **Research Challenger V2** (GRU, EXP-06) serves in real-time **shadow evaluation mode**, accumulating an immutable, prospective forward-testing audit trail (`prospective_predictions_log.csv`).

---

## 2. Problem Formulation & Econometric Foundations

### 2.1 Horizon & Inference Contract
- **Forecast Horizon:** Next official NSE trading session close ($t+1$).
- **Inference Mode:** The network predicts the standardized 1-step ahead stationary simple return:
  $$r_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$
- **Price Reconstruction:** Closing price is reconstructed deterministically from predicted return and the verified completed session close $P_t$:
  $$\hat{P}_{t+1} = P_t \times (1 + \hat{r}_{t+1})$$
- **Input Horizon:** Exactly $T = 30$ consecutive, completed daily trading sessions:
  $$X_t = [x_{t-29}, x_{t-28}, \dots, x_t] \in \mathbb{R}^{30 \times 7}$$

### 2.2 Efficient Market Hypothesis & Signal-to-Noise Floor
In accordance with weak-form Efficient Market Hypothesis (EMH), historical price and volume information is rapidly discounted into market quotes. Daily equity returns possess an ultra-low Signal-to-Noise Ratio ($\text{SNR} < 0.05$). Deep models attempting to forecast raw levels inevitably overfit noise. Formulating the target as stationary relative returns bounds the conditional expectation near zero and forces the network to learn genuine momentum and volatility dynamics.

---

## 3. Data Engineering & Integrity Audit

### 3.1 Forensic Ingestion & Quarantine Pipeline
During Sprint 2, an exhaustive audit was performed on 1,241 raw OHLCV rows ingested from Yahoo Finance against official National Stock Exchange of India (NSE) Bhavcopy circulars and holiday calendars. Eight contaminated rows were identified and quarantined:

1. **Exchange Holidays (6 rows):** 2021-11-05 (Diwali Balipratipada), 2022-03-01 (Mahashivratri), 2022-10-26 (Diwali Laxmi Pujan next day), 2023-01-26 (Republic Day), 2023-03-07 (Holi), and 2024-01-22 (Ayodhya Ram Mandir Pran Pratishtha). The vendor emitted static previous-day closes with 0 volume.
2. **Provider Delivery Outage (1 row - 2025-03-18):** Trading occurred normally on the NSE (>4.2M shares traded), but vendor delivery dropped volume to 0.
3. **Unfinished Daily Candle (1 row - 2026-10-07):** Ingested mid-market at 11:42 AM IST with partial volume and unfinalized price.

All 8 observations were quarantined into `quarantine_records.csv`, yielding the clean, frozen master dataset `dataset_v2_audited.csv` with exactly 1,233 valid trading sessions.

### 3.2 Cryptographic Lock & Line-Ending Normalization
The audited dataset is locked via SHA-256 in `dataset_manifest_v2.json`:
- Canonical SHA-256 (LF): `ca9cfb09228815f9b4c06b29f00030589a1c87e83e6022e06180302b1ff58019`
- `.gitattributes` enforces `*.csv text eol=lf`, guaranteeing identical checksums across Windows, Linux, and cloud containers.

---

## 4. Feature Engineering & Stationarity Transforms

To ensure zero price-level memorization, seven causal, stationary features are computed over lookback $T=30$:

1. **`log_return`**: $\ln(Close_t / Close_{t-1})$ (Stationary price delta)
2. **`hl_ratio`**: $\ln(High_t / Low_t)$ (Intraday volatility spread)
3. **`co_ratio`**: $\ln(Close_t / Open_t)$ (Intraday bar momentum)
4. **`vol_change`**: $\ln(Volume_t / Volume_{t-1})$ (Volume acceleration)
5. **`atr_norm`**: $\text{ATR}_{14, t} / Close_t$ (Price-normalized true range)
6. **`rolling_vol`**: 20-session sample standard deviation of log returns
7. **`obv_norm`**: 20-session standardized On-Balance Volume oscillator:
   $$OBV\_Norm_t = \frac{OBV_t - \mu_{OBV, 20}}{\sigma_{OBV, 20} + \epsilon}$$

All features are normalized using `StandardScaler` fit strictly on the training partition of each fold, ensuring zero forward leakage.

---

## 5. Model Architecture & Inductive Biases

StockSense AI benchmarks five distinct model architectures:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                STOCKSENSE AI MODEL SUITE                                │
├─────────────────────────┬──────────────┬──────────────┬────────────────────────────────┤
│ Architecture            │ Parameters   │ Role         │ Primary Inductive Bias         │
├─────────────────────────┼──────────────┼──────────────┼────────────────────────────────┤
│ GRU V2 (EXP-06)         │ 44,065       │ Challenger   │ Gated memory, low variance     │
│ CNN-LSTM Hybrid (V1)    │ 112,417      │ Production   │ 1D spatial patterns + BiLSTM   │
│ Standard LSTM           │ 52,289       │ Baseline     │ Separate cell/hidden gating    │
│ Linear Ridge            │ 211          │ Benchmark    │ L2 regularized linear baseline │
│ Naive Persistence       │ 0            │ Baseline     │ r=0, Random walk persistence   │
└─────────────────────────┴──────────────┴──────────────┴────────────────────────────────┘
```

### GRU V2 Challenger Architecture Detail
- **Layer 1:** `GRU(units=64, return_sequences=True)` + `Dropout(0.2)`
- **Layer 2:** `GRU(units=32, return_sequences=False)` + `Dropout(0.2)`
- **Output:** `Dense(1, activation='linear')` predicting standardized return $\hat{z}_{t+1}$.
- **Optimizer:** Adam ($\alpha = 10^{-3}$, decay on plateau).
- **Execution Runtime:** ~12.99s per training fit on local CPU runtime.

---

## 6. Experimental Protocol & Leakage Controls

### 6.1 Expanding Chronological Folds
To prevent lookahead bias, we implement 3 strictly chronological walk-forward splits:
- **Fold 1:** Train: 0–739 (60%), Val: 740–986 (20%), Test: 987–1232 (20%)
- **Fold 2:** Train: 0–862 (70%), Val: 863–1047 (15%), Test: 1048–1232 (15%)
- **Fold 3:** Train: 0–986 (80%), Val: 987–1109 (10%), Test: 1110–1232 (10%, ~100 sessions)

### 6.2 Purging & Embargoing
- **Purging:** The 29 sequences spanning train-validation split boundaries are dropped to ensure no overlap in input lookback tensors.
- **Embargo:** A 5-session quarantine window follows validation splits before test evaluation begins.

---

## 7. Offline Cross-Fold Experimental Results

### 7.1 Cross-Fold Performance Summary
Across the three expanding chronological folds:

| Rank | Model Architecture | Cross-Fold Mean Price MAE | Fold 3 Price MAE | Fold 3 Directional Acc | Fold 3 Directional MCC |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **GRU V2 (EXP-06)** | **₹13.29** | **₹12.66** | **54.0%** | **+0.1214** |
| 2 | Naive Persistence | ₹13.34 | ₹12.71 | 50.0% | 0.0000 |
| 3 | LSTM Standard | ₹13.39 | ₹12.78 | 52.0% | +0.0610 |
| 4 | CNN-LSTM (EXP-05) | ₹13.42 | ₹12.82 | 53.0% | +0.0825 |
| 5 | Linear Ridge | ₹14.18 | ₹13.65 | 50.5% | +0.0120 |

> **Critical Evaluation Caveat:** The GRU's cross-fold mean MAE is ₹13.29 versus ₹13.34 for naive persistence—an exploratory difference of only **₹0.05 / share** (0.37% relative reduction). While the Fold 3 Directional MCC of **+0.1214** is encouraging as an exploratory signal, these development results do **NOT** establish reliable statistical superiority over naive persistence. Firm claims of superiority cannot be made until prospective evaluation is completed.

### 7.2 Transparent Directional Bias & Asymmetry Analysis
While GRU achieved the lowest overall MAE and positive MCC (+0.1214), granular confusion matrix analysis revealed pronounced directional asymmetry:
- **UP Recall:** **78.7%** (Substantially better at capturing upward movements)
- **DOWN Recall:** **32.1%** (Severe weakness during downward corrections and pullbacks)

*Root Cause:* The 2024–2026 Indian equity market exhibited a prolonged secular bull trend, imparting upward gradient drift during model optimization. The model learned to default to positive return forecasts, performing well on up-days while missing sudden pullbacks. Rather than concealing this asymmetry or claiming comprehensive market mastery, StockSense AI transparently documents that the model has not mastered downward regimes, and highlights this as a priority for asymmetric loss functions in future research.

---

## 8. The Holdout Question & Scientific Integrity

### 8.1 Preregistered Holdout Criteria
Preregistration required $\ge 60$ uninspected historical trading sessions. Following data audit and quarantine, only 12 historical sessions prior to October remained genuinely uninspected. The May–October 2026 window had already been analyzed during EXP-05 research.

### 8.2 The Decision: Gate Remains Locked (NO-GO)
StockSense AI explicitly declared **NO-GO** for confirmatory historical evaluation. Reusing inspected data to claim "state-of-the-art" was rejected. The project adopted a dual-track strategy:
1. **Exploratory Development:** Historical 3-fold cross validation identifies best challenger architecture (GRU).
2. **Prospective Shadow Evaluation:** The frozen GRU model is deployed in shadow mode, evaluated strictly forward-in-time.

---

## 9. Prospective Shadow Serving Engine (EXP-06)

### 9.1 Synchronized Tri-Model Logging
The shadow evaluation engine (`ProspectiveV2Service`) logs synchronized forecasts for every completed market session without modifying production behavior:
1. **Production V1 (CNN-LSTM)**
2. **Challenger V2 (GRU)**
3. **Naive Persistence Baseline**

Logs are committed to `research/EXP-06/evaluation/prospective_predictions_log.csv` with cryptographic hash tracking in `locked_evaluation_registry_v2.json`.

### 9.2 Live Session Verification (November 6 $\to$ November 9, 2026)
- **Cutoff Completed Session:** November 6, 2026 (NSE Close: ₹1,320.00)
- **Target Forecast Session:** November 9, 2026 (Next NSE trading day)
- **Model Forecasts:**
  - **Challenger V2 (GRU):** ₹1,322.86 (+0.2169% / +21.7 bps) — **BULLISH**
  - **Production V1 (CNN-LSTM):** ₹1,319.33 (-0.0509% / -5.1 bps) — **BEARISH**
  - **Naive Baseline:** ₹1,320.00 (0.00% / 0 bps) — **NEUTRAL**

*Empirical Significance:* The two deep models take opposing directional stances, providing a definitive prospective test when the market settles on November 9, 2026.

---

## 10. Full-Stack Production Implementation

- **FastAPI High-Performance Backend:**
  - `/api/research/challenger-prediction/{symbol}`: Returns isolated GRU V2 prediction with shape validation `(1, 30, 7)`.
  - `/api/research/exp06-comparison`: Synchronized 4-model registry response and prospective evaluation metadata.
  - Returns HTTP 404 for non-Reliance tickers; returns HTTP 503 if market data is stale.
- **React 18 + Vite Frontend:**
  - `ResearchComparisonPanel.jsx`: Embedded research dashboard.
  - Dedicated **GRU V2 Hero Card** (`EXPERIMENTAL — SHADOW EVALUATION`) disclosing CPU runtime.
  - **4-Model Comparative Registry** displaying V1, V2, EXP-05 Colab, and Naive in parallel.
  - **Interactive Recharts Trajectory:** Visualizes price path and predicted horizons.
- **Docker & Cloud Architecture:**
  - Multi-stage Dockerfile packaging `models_storage_stationary/` and `models_storage_v2/`.
  - Environment variable `PREDICTION_PIPELINE_VERSION: stationary-return-v1` preserved for production safety.

---

## 11. Verification, Testing & Release Engineering

- **196 Automated Tests (100% Passing):**
  - `tests/test_v2_serving_and_prospective.py`: 11 tests covering shadow prediction, prospective logging, symbol isolation, and outcome resolution.
  - `tests/test_exp06_readiness.py`: 15 tests covering audited dataset integrity, SHA-256 fingerprint verification, and line ending normalization.
  - `tests/test_research_v2_registry.py`: 16 tests covering registry schema, tensor loading, and scaler isolation.
  - 154 core tests covering stationary pipeline, data verification, security headers, and API endpoints.
- **Frontend Build Quality:**
  - Vite production build transformed 3,185 modules cleanly with 0 errors in 34.25s.

---

## 12. Critical Limitations & Future Research

1. **₹0.05 Margin over Naive Persistence:** The GRU model's ₹0.05 edge on daily price prediction does not achieve statistical significance under a Diebold-Mariano test on historical sample sizes. Prospective evaluation across 60–100 sessions is necessary.
2. **Asymmetric DOWN Recall (32.1%):** Future work must incorporate asymmetric loss functions (LinEx loss, directional penalty terms) to overcome bull-market bias.
3. **Transaction Cost Bounding:** A ₹0.05 edge is insufficient to cover standard retail trading friction (~₹0.65 to ₹1.98 per share); the system must be utilized for directional allocation rather than high-frequency execution.
4. **Multi-Asset Scaling:** The audited pipeline is ready to be scaled to the Nifty 50 constituent universe using the packaged Google Colab GPU notebook (`stocksense_exp06_source.zip`).

---

## 13. Academic Viva Defense Reference

For full viva and examination preparation, see:
- [VIVA_QUESTION_BANK.md](file:///d:/Downloads/Stock_SenseAi/research/VIVA_QUESTION_BANK.md) (52 detailed technical questions & answers)
- [StockSense_AI_Defense.pptx](file:///d:/Downloads/Stock_SenseAi/research/presentation/StockSense_AI_Defense.pptx) (12-slide widescreen presentation)
- [SLIDES.md](file:///d:/Downloads/Stock_SenseAi/research/presentation/SLIDES.md) (Full presentation slide transcript)

---

*Certified Sprint 5 Final Release • StockSense AI Engineering & Research Team*

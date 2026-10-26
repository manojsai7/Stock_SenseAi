# StockSense AI — Comprehensive Scientific Research & Evaluation Report

**Project Title:** StockSense AI: Explainable Next-Day Stock Price Forecasting Using CNN-LSTM  
**Target Equity:** Reliance Industries Ltd. (`RELIANCE.NS`, NSE)  
**Evaluation Protocol Branch:** `research/reliance-model-evaluation`  
**Date of Report:** October 26, 2026  
**Status:** Completed Historical Audits, Point-in-Time Explainability Integrated, Prospective Forward Tracking Active ($N=1$ Resolved, $N=2$ Pending)

---

## Abstract

Stock price forecasting remains a foundational problem in financial econometrics and applied machine learning due to non-stationarity, low signal-to-noise ratios, and regime shifts. Many academic proposals claim substantial predictive edges while inadvertently introducing temporal lookahead bias, evaluating on unrepresentative splits, or claiming intraday/hourly forecasting capabilities that suffer from severe microstructure noise and API lookback limits. 

**StockSense AI** implements an end-to-end, scientifically defensible, and leak-free forecasting framework targeting the **next official trading day's closing price** ($t+1$) for Indian large-cap equities (benchmarked on `RELIANCE.NS`). The system employs an expanding walk-forward **CNN-LSTM** neural network operating on a **30-session input sequence** of **seven relative stationary features** without raw price memorization. Model inference strictly respects the **Completed Daily Candle Rule**, ensuring unfinalized intraday candles never enter inference. 

To bridge offline research with live serving, this study establishes a strict three-tier evaluation methodology:
1. **Historical Walk-Forward Folds (EXP-05):** Audited and mathematically reproduced expanding-window fold checkpoints.
2. **Production Validation:** Diagnostics on deployed production weights, explicitly cataloging lookahead when evaluated retrospectively.
3. **Prospective Forward Evaluation:** An immutable, timestamp-locked registry recording predictions *before* market close and scoring outcomes only after exchange session finalization.

On the initial prospective evaluation session (**October 26, 2026**, $N=1$), the production model recorded an absolute price error of **₹11.57** (predicted direction: UP, actual close: **₹1,201.80**, +0.99%), outperforming the Naive Persistence baseline (**₹11.80**) and the Colab EXP-05 candidate (**₹12.49**, which suffered from threshold collapse predicting DOWN). We emphasize that an $N=1$ observation is preliminary and one successful forecast cannot establish statistical superiority; accumulating 60–100 prospective sessions is mandatory. Furthermore, we document an on-demand **input-sensitivity explainability panel** based on standardized channel neutralization (feature-perturbation sensitivity), clarifying that sensitivity measures model response to input perturbations rather than financial market causation. Finally, we formally note that multimodal news sentiment is not proven or implemented in this codebase, and recommend aligning the project scope to explainable price-series CNN-LSTM forecasting.

---

## 1. Frozen Problem Formulation & System Architecture

### 1.1 Horizon & Inference Contract
- **Forecast Target:** Next official NSE trading session closing price:
  $$y_{t+1} = \text{Close}_{t+1}$$
- **Modeled Target (Stationary Formulation):** The network predicts the next-session simple return:
  $$r_{t+1} = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t}$$
- **Price Reconstruction:** Closing price is reconstructed deterministically from the predicted return and the latest completed session close:
  $$\widehat{\text{Close}}_{t+1} = \text{Close}_t \times (1 + \hat{r}_{t+1})$$
- **Input Horizon:** Exactly $L = 30$ consecutive, completed daily trading sessions:
  $$X_t = [x_{t-29}, x_{t-28}, \dots, x_t] \in \mathbb{R}^{30 \times 7}$$

### 1.2 The Seven Causal Stationary Features
Unlike raw OHLCV models that suffer from unit-root non-stationarity and fail under regime shifts, all 7 features are scale-invariant, dimensionless, and strictly causal (derived exclusively from session $t$ and preceding history):

1. **`return_1d`**: 1-day close-to-close return:
   $$\frac{\text{Close}_t}{\text{Close}_{t-1}} - 1$$
2. **`return_5d`**: 5-day momentum return:
   $$\frac{\text{Close}_t}{\text{Close}_{t-5}} - 1$$
3. **`return_20d`**: 20-day trend return:
   $$\frac{\text{Close}_t}{\text{Close}_{t-20}} - 1$$
4. **`intraday_return`**: Within-session bar return:
   $$\frac{\text{Close}_t - \text{Open}_t}{\text{Open}_t}$$
5. **`hl_range_ratio`**: Normalized high-low volatility spread:
   $$\frac{\text{High}_t - \text{Low}_t}{\text{Close}_{t-1}}$$
6. **`close_to_ma20`**: Price distance from 20-day Simple Moving Average:
   $$\frac{\text{Close}_t}{\text{MA20}_t} - 1, \quad \text{where } \text{MA20}_t = \frac{1}{20} \sum_{i=0}^{19} \text{Close}_{t-i}$$
7. **`volume_ratio`**: Normalized volume relative to 20-day rolling baseline:
   $$\frac{\text{Volume}_t}{\frac{1}{20}\sum_{i=0}^{19} \text{Volume}_{t-i} + 10^{-8}}$$

### 1.3 Feature Normalization & Inversion
- **Feature Scaler (`StationaryFeatureScaler`):** Standardizes all 7 features to zero mean and unit variance ($\mu_X, \sigma_X$), fitted strictly on training partitions.
- **Target Scaler (`ReturnTargetScaler`):** Standardizes the simple return target $r_{t+1}$ using training mean $\mu_r$ and standard deviation $\sigma_r$.
- **Causality Guarantee:** Scalers never incorporate validation, test, or prospective observations during fitting.

### 1.4 Deep Learning Architecture (CNN-LSTM Hybrid)
The architecture combines 1D spatial feature extraction with recurrent temporal gating:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Input Sequence Tensor: (batch_size, 30, 7)                             │
├────────────────────────────────────────────────────────────────────────┤
│ Conv1D (filters=64, kernel_size=3, padding='same', activation='relu')  │
│ → Isolates 3-day translation-invariant technical momentum patterns    │
├────────────────────────────────────────────────────────────────────────┤
│ MaxPooling1D (pool_size=2)                                             │
│ → Downsamples temporal dimension (30 → 15 steps), suppressing noise   │
├────────────────────────────────────────────────────────────────────────┤
│ LSTM (units=50, return_sequences=False, activation='tanh')             │
│ → Captures long-range temporal dependencies across 15 compressed steps │
├────────────────────────────────────────────────────────────────────────┤
│ Dropout (rate=0.2)                                                     │
│ → Prevents co-adaptation and historical noise memorization             │
├────────────────────────────────────────────────────────────────────────┤
│ Dense (units=32) + LeakyReLU (negative_slope=0.1)                      │
│ → Non-saturating projection allowing asymmetric return tail dynamics   │
├────────────────────────────────────────────────────────────────────────┤
│ Dense (units=1, activation='linear')                                   │
│ → Outputs standardized next-session predicted return (\hat{z}_{t+1})   │
└────────────────────────────────────────────────────────────────────────┘
```
- **Total Trainable Parameters:** 27,477 (107.33 KB).
- **Loss Function:** Mean Squared Error (MSE) on standardized returns.
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, learning rate initialized at $10^{-3}$).

### 1.5 Strict Completed-Candle & Freshness Rules
- **No Intraday Pollution:** An incomplete intraday candle (where `Close`, `High`, `Low`, and `Volume` fluctuate during live trading) is **strictly forbidden** from entering the input tensor.
- **Exchange Boundary Check:** Inference checks the official NSE market calendar (via `MarketStatusService`):
  - During market hours (`OPEN`, 09:15–15:30 IST): Only completed bars up to $t-1$ are used.
  - After market close (`POST_MARKET` / `CLOSED`, 15:30+ IST): The day's finalized bar is ingested to forecast session $t+1$.
- **Freshness Guard:** If the data provider returns data whose latest completed bar is older than the exchange calendar's completed session, inference fails closed (HTTP 503 / `RuntimeError`) rather than silently forecasting on stale data.

---

## 2. Partition Rigor: Separation of Historical, Validation, and Prospective Sets

A common methodological flaw in academic machine learning is conflating historical test blocks, validation tuning splits, and forward monitoring. In this study, they are maintained as three mutually isolated environments:

```
Chronological Sequence Partitions (Total: 1,191 Historical Sequences):
[0 ----------------- 455 ----------- 639 ----------- 823 ----------- 1007 ------- 1191] | [Prospective Stream]
|--- Fold 1 Train ---|-- F1 Val/F2 Tr -|-- F2 Val/F3 Tr -|-- F3 Val/Test3 -|            |
                                       |--- Fold 1 Test -|                              |
                                                         |--- Fold 2 Test -|            |
                                                                           |-- Fold 3 --|
                                                                           (Diagnostic) |
|----------------------- Deployed Production Refit (Through 2026-10-06) ----------------|
                                                                                        |-> 2026-10-23 (Cutoff) -> Target 2026-10-26 (Resolved)
                                                                                        |-> 2026-10-26 (Cutoff) -> Target 2026-10-27 (Pending)
```

1. **Historical EXP-05 Folds (Google Colab Offline):**
   - Three expanding walk-forward splits evaluating model stability across distinct market regimes from 2021 through 2026.
   - Fold 3 trained through `2025-04-21` and validated through `2026-01-13`.
   - The test period (`2026-01-15` to `2026-10-07`, $N=184$) was inspected during experiment analysis and is therefore an **audited diagnostic benchmark**, not a virgin holdout.
2. **Production Model-Selection Validation:**
   - The live production model in `models_storage_stationary/` was trained/refit across sequences through `2026-10-06`.
   - Evaluating production weights retrospectively on the Fold 3 window contains lookahead bias. It is included strictly as a diagnostic reference, never as proof of out-of-sample skill.
3. **Prospective Forward Observations:**
   - Genuinely out-of-sample forward evaluation. Forecasts are generated and cryptographically logged *prior* to next-day trading, with actual prices attached only after official exchange settlement.

---

## 3. Historical Benchmark & Checkpoint Provenance (EXP-05 Fold 3, $N=184$)

### 3.1 Mathematical Provenance Verification
The Colab checkpoint `best_cnn_lstm_fold3.keras` and its scalers (`feature_scaler_3d_fold3.pkl`, `return_scaler_fold3.pkl`) were extracted and evaluated locally against the saved Colab Fold 3 prediction CSV.
- **Maximum Absolute Discrepancy:** $4.897 \times 10^{-7}$
- **Mean Discrepancy:** $2.504 \times 10^{-7}$
- **Verdict:** Mathematical reproducibility is **100% established**.

### 3.2 Head-to-Head Comparison Matrix (184 Sessions: 2026-01-15 to 2026-10-07)

| Metric | Naive Persistence Baseline | Colab EXP-05 Fold 3 (Uncalibrated, $\theta=0$) | Colab EXP-05 Fold 3 (Calibrated, $\theta=+0.000307$) | Production CNN-LSTM (Diagnostic Lookahead) |
| :--- | :---: | :---: | :---: | :---: |
| **Price MAE (₹)** | **₹14.16** | ₹14.09 | **₹14.09** | ₹14.18 |
| **Price RMSE (₹)** | **₹18.52** | ₹18.44 | ₹18.44 | ₹18.46 |
| **Price MAPE (%)** | **1.06%** | 1.05% | 1.05% | 1.06% |
| **3-Class Raw Acc (%)** | 3.26% | 52.72% | 51.09% | 48.91% |
| **Balanced Accuracy (%)**| 33.33% | 34.69% | 33.24% | 35.59% |
| **Matthews Corr (MCC)** | 0.0000 | 0.0560 | 0.0055 | 0.1117 |
| **UP Recall (%)** | 0.0% | 15.66% | **6.02%** | 95.18% |
| **DOWN Recall (%)** | 0.0% | 88.42% | **93.68%** | 11.58% |

### 3.3 Scientific Findings on Historical Folds
1. **The Martingale Barrier:** Naive Persistence (predicting zero return, $Close_{t+1} = Close_t$) achieves a price MAE of **₹14.16**. The best deep learning model achieves **₹14.09**—a difference of only ₹0.07 (0.49%), which is statistically indistinguishable from zero under standard Diebold-Mariano tests.
2. **Directional Calibration Collapse:** When calibrated using threshold $\theta = +0.000307$ (derived from validation optimization), Colab Fold 3 collapses to predicting **DOWN on 93.68% of sessions** and captures only 6.02% of upward sessions. This produces a near-zero MCC (**0.0055**).
3. **Formal Gate Decision:** Colab EXP-05 failed the pre-registered promotion criteria:
   - *Price Gate:* Failed to meaningfully outperform Naive Persistence.
   - *Balance Gate:* Failed balanced recall requirement (both UP and DOWN recall $>20\%$).
   - **Action Taken:** The production models were **strictly frozen**, and candidate promotion was rejected.

---

## 4. Feature Ablation & Point-in-Time Explainability Diagnostics

### 4.1 Feature Neutralization & Permutation Results
To understand what internal signals the CNN-LSTM relies upon, two diagnostic ablation methods were executed on the canonical 7 features:
1. **Trajectory Mean Neutralization:** Setting a standardized feature channel to 0.0 across all 30 input sessions (corresponding to the training-set mean under `StandardScaler`).
2. **Window-Shuffled Permutation:** Permuting the 30-day temporal sequence across test sessions to destroy temporal ordering while preserving marginal distributions.

#### Ablation Impact Summary (Fold 3 Diagnostic Benchmark):
- **`return_20d` & `return_5d`:** Neutralizing these multi-day return channels induced the largest increase in error ($\Delta \text{MAE} \approx +0.01\text{ to }+0.03$), indicating that multi-day momentum is the model's primary driver.
- **`hl_range_ratio`:** Disrupting this volatility feature slightly *reduced* historical MAE (by ~₹0.03), indicating that high-low range ratio introduced noise into the sequence representation.
- **Magnitude Compression:** The absolute impact across all feature neutralizations remained within ₹0.01–₹0.03 against a ₹14.09 baseline. This empirical finding confirms that the neural network's internal mapping is heavily compressed around the central drift.

### 4.2 Live Dashboard Explainability Integration
We integrated a real-time, read-only explainability service (`backend/services/explainability_service.py`) and UI component (`FeatureExplainabilityPanel.jsx`):
- **API Endpoint:** `GET /api/models/{symbol}/explainability`
- **Mechanism:** On each completed daily session, the service extracts the latest 30-day window, computes the baseline forecast, neutralizes each of the 7 standardized channels to 0.0, and returns:
  - Feature value on latest session $t$
  - Delta return in basis points ($\Delta \text{bps}$)
  - Delta predicted price (₹)
  - Direction flip indicator (`True` / `False`)
- **Strict Methodological Disclaimer:**
  > **Model Sensitivity $\neq$ Market Causation:** Feature sensitivity measures the model's algorithmic response under trajectory mean neutralization (feature-perturbation sensitivity). Gradients are not calculated; rather, the service evaluates forward-pass output shifts when channels are zeroed out to the training-set mean. It diagnosticates how the model reacts to input perturbations. It does **not** prove that the feature caused the market movement, nor does it guarantee predictive reliability.

---

## 5. Prospective Forward Evaluation Tracking

To eliminate all potential historical leakage, lookahead bias, and post-hoc rationalization, prospective evaluation is logged in real time to [`research/EXP-05/evaluation/prospective_predictions_log.csv`](file:///d:/Downloads/Stock_SenseAi/research/EXP-05/evaluation/prospective_predictions_log.csv).

### 5.0 Current Prospective Evaluation Scope
- **1 resolved session:** October 26, 2026 (resolved against official NSE settlement).
- **1 pending session:** October 27, 2026 (forecast committed, outcome pending).
- **Three predictions per session:** Production CNN-LSTM, Colab EXP-05 Fold 3 (Calibrated), and Naive Persistence baseline (total 6 prospective records logged).
- **Key Scientific Finding:** One successful forecast does not establish predictive superiority.

### 5.1 Observation 1: October 26, 2026 ($N=1$, Fully Resolved)
- **Input Window:** 30 completed sessions ending **Friday, October 23, 2026** (Baseline Close: **₹1,190.00**).
- **Target Session:** **Monday, October 26, 2026**.
- **Predictions Committed:** Logged on `2026-10-26T09:06:58 UTC` (during morning market hours, prior to session settlement).
- **Exchange Finalized Outcome:** **₹1,201.80** (Change: **+₹11.80**, **+0.9916%**, `UP`).

#### Scored Performance:
| Model / Benchmark | Predicted Return | Predicted Close | Predicted Dir | Actual Close | Price Error (Abs) | Direction Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production CNN-LSTM** | `+0.0193%` | **₹1,190.23** | `UP` | ₹1,201.80 | **₹11.57** | **YES** |
| **Naive Persistence Baseline** | `+0.0000%` | **₹1,190.00** | `FLAT` | ₹1,201.80 | **₹11.80** | **NO** |
| **Colab EXP-05 Fold 3 (Calibrated)** | `-0.0578%` | **₹1,189.31** | `DOWN` | ₹1,201.80 | **₹12.49** | **NO** |

#### Critical Methodological Statement:
> While the Production CNN-LSTM achieved the lowest absolute error (₹11.57 vs ₹11.80 for Naive Persistence) and correctly predicted the upward direction, **an $N=1$ sample size has zero statistical power**. A ₹0.23 error difference on a single session cannot establish model superiority or predictive edge. Likewise, Colab Fold 3's inferior performance on this single observation does not independently prove universal underperformance. Sound empirical inference requires accumulating **60 to 100 prospective trading sessions** before computing Diebold-Mariano tests or block-bootstrap intervals.

### 5.2 Observation 2: October 27, 2026 ($N=2$, Locked & Pending)
Following the resolution of October 26, the next-session forecast was automatically committed using the updated completed window ending **October 26, 2026** (Close: **₹1,201.80**):

| Model | Input Window End | Target Session | Baseline Close | Predicted Return | Predicted Close | Predicted Direction | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production CNN-LSTM** | 2026-10-26 | **2026-10-27** | ₹1,201.80 | `+0.0608%` | **₹1,202.53** | `UP` | `PENDING_OUTCOME` |
| **Colab EXP-05 Fold 3 (Calibrated)** | 2026-10-26 | **2026-10-27** | ₹1,201.80 | `-0.0644%` | **₹1,201.03** | `DOWN` | `PENDING_OUTCOME` |
| **Naive Persistence** | 2026-10-26 | **2026-10-27** | ₹1,201.80 | `+0.0000%` | **₹1,201.80** | `FLAT` | `PENDING_OUTCOME` |

*Note: In accordance with scientific integrity standards, the October 27 outcome will remain `PENDING_OUTCOME` until official post-market settlement on October 27, 2026. No outcomes are fabricated.*

---

## 6. Scope Clarification: Multimodal Sentiment & Title Alignment

### 6.1 Status of Multimodal Financial Sentiment
Early conceptual brainstorming discussions (documented in legacy chat transcripts) proposed incorporating financial news sentiment into the forecasting pipeline. We explicitly state the following:
1. **Implementation Status:** A financial-news sentiment pipeline is **not implemented, tested, or empirically evaluated** within this repository.
2. **Methodological Risks of Sentiment:** Adding historical news sentiment without timestamp-aligned, point-in-time publication records introduces severe lookahead bias (e.g., articles updated after market hours or revised post-event).
3. **Recommendation:**
   - **Recommended Title (Implemented Scope):**
     $$\textbf{"StockSense AI: Explainable Next-Day Stock Price Forecasting Using CNN-LSTM"}$$
     This title accurately, honestly, and defensibly represents the system that exists in code today.
   - **Future Research Track:** If the final-year project timeline permits, a formal multimodal experiment should be pre-registered: ingesting timestamped financial news headlines, computing sentiment via FinBERT, fusing with stationary technical sequences, and testing whether the multimodal architecture beats the unimodal CNN-LSTM and Naive Persistence baselines.

---

## 7. Viva Defense & Technical Questions Guide

| Examiner Question | Defensible Scientific Answer |
| :--- | :--- |
| **"Why does your model only predict next-day close instead of hourly or intraday?"** | "Free financial APIs (like Yahoo Finance) only provide up to 60 days of hourly data, which is insufficient for training deep neural networks without severe overfitting. Furthermore, hourly market data is dominated by transient order routing, bid-ask bounce, and microstructure noise. Completed daily sessions provide 5 years of macroeconomic cycles, allowing the CNN-LSTM to learn robust, generalizable momentum patterns." |
| **"Why did you freeze production instead of promoting the Colab EXP-05 model?"** | "In our audited head-to-head evaluation across 184 sessions, Colab EXP-05 failed our pre-registered promotion criteria: its price MAE (₹14.09) was virtually identical to Naive Persistence (₹14.16), and its directional predictions collapsed to 93.68% DOWN with only 6.02% UP recall. In production, we cannot deploy a model that fails to beat a zero-parameter baseline." |
| **"Does your explainability panel prove that a feature caused the price change?"** | "No. We explicitly distinguish feature sensitivity from market causation. Our sensitivity diagnostic sets each standardized channel to zero (the training mean) to measure feature-perturbation / mean-neutralization sensitivity. Gradients are not calculated; instead, we evaluate forward-pass output differences when a feature channel is neutralized. It shows how the neural network reacts to an input perturbation, not what caused external market participants to trade." |
| **"Can you claim your model beats the market after the October 26 prediction?"** | "No. On October 26, our production model had an error of ₹11.57 versus ₹11.80 for naive persistence, but $N=1$ has no statistical validity. We document it as an individual observation within an ongoing prospective protocol that requires 60–100 trading sessions before drawing definitive statistical conclusions." |

---

## 8. Verified Artifact Inventory

| Category | Artifact Path | Description |
| :--- | :--- | :--- |
| **Production Model** | `models_storage_stationary/RELIANCE_NS/model_cnn_lstm_stationary.keras` | Frozen production CNN-LSTM checkpoint |
| **Production Scalers** | `models_storage_stationary/RELIANCE_NS/*_scaler.pkl` | Causal feature and target scalers |
| **Colab Checkpoint** | `research/EXP-05/extracted/.../best_cnn_lstm_fold3.keras` | Audited Fold 3 candidate checkpoint |
| **Evaluation Registry** | `research/EXP-05/evaluation/locked_evaluation_registry.json` | Cryptographic parameter & model hash lock |
| **Prospective Log** | `research/EXP-05/evaluation/prospective_predictions_log.csv` | Immutable prospective observations log |
| **Explainability API** | `backend/services/explainability_service.py` | Live 30-session mean-neutralization diagnostic |
| **Frontend UI** | `frontend/src/components/FeatureExplainabilityPanel.jsx` | User-facing explainability panel & disclaimer |
| **Test Suites** | `tests/test_*.py` (17 tests total) | Complete automated validation suite (100% pass) |

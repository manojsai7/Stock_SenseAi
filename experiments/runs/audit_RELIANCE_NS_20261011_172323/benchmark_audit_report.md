# EXP-01: Offline Directional Benchmark & Metric Audit Report

**Symbol:** `RELIANCE.NS`  
**Run ID:** `audit_RELIANCE_NS_20261011_172323`  
**Generated At:** `2026-10-11T17:23:26.721539+05:30`  
**Test Window:** 184 consecutive trading sessions  
**Test Distribution:** UP: 83 (45.11%), DOWN: 95 (51.63%), FLAT: 6 (3.26%)  
**Training Distribution:** UP: 422, DOWN: 402, FLAT: 1 (Majority: **UP**)

---

## 1. Executive Summary & Comparative Matrix

| Predictor                                               | Type                                         |   Raw Accuracy (%) |   Non-FLAT Acc (%) [N=178] |   Balanced Acc (%) |     MCC |   Cohen Kappa | MAE (INR)   |
|:--------------------------------------------------------|:---------------------------------------------|-------------------:|---------------------------:|-------------------:|--------:|--------------:|:------------|
| Training Majority Baseline (UP)                         | Directional Baseline                         |              45.11 |                      46.63 |              33.33 |  0      |        0      | N/A         |
| Always-DOWN (Ex-Post Diagnostic)                        | Diagnostic Baseline                          |              51.63 |                      53.37 |              33.33 |  0      |        0      | N/A         |
| Previous-Day Momentum                                   | Heuristic Directional Baseline               |              41.3  |                      42.7  |              28.34 | -0.1091 |       -0.1091 | N/A         |
| SMA20 Trend Baseline                                    | Heuristic Directional Baseline               |              45.65 |                      47.19 |              30.79 | -0.0747 |       -0.0725 | N/A         |
| Naive Persistence (Price Baseline)                      | Price & Direction Baseline                   |               3.26 |                       0    |              33.33 |  0      |        0      | 14.16       |
| Saved LSTM (models_storage)                             | Trained Neural Model                         |              52.72 |                      54.49 |              36.22 |  0.0812 |        0.0811 | 18.37       |
| Saved CNN-LSTM (models_storage)                         | Trained Neural Model                         |              50.54 |                      52.25 |              35.17 |  0.0525 |        0.0512 | 19.32       |
| Run-Specific LSTM (run_RELIANCE_NS_20261007_120035)     | Trained Neural Model (Experiment Checkpoint) |              48.37 |                      50    |              32.7  | -0.0179 |       -0.0175 | 26.39       |
| Run-Specific CNN-LSTM (run_RELIANCE_NS_20261007_120035) | Trained Neural Model (Experiment Checkpoint) |              48.37 |                      50    |              34.68 |  0.0473 |        0.0364 | 30.87       |

---

## 2. Methodology & Leakage Prevention

1. **Date & Sequence Alignment:**
   - Every input window ends at completed session $t$ (`input_date`).
   - Every target corresponds strictly to session $t+1$ (`target_date`), verified against historical processed bars.
   - `previous_close` is strictly $Close[t]$.
   - `actual_close` is strictly $Close[t+1]$.
   - Mismatch audit: **0 errors detected** across 184 samples.

2. **Zero-Leakage Baseline Fitting:**
   - **Training Majority Baseline:** Exclusively inspects training labels (422 UP vs 402 DOWN), selecting **UP**. On the held-out test set, it achieves **45.11%**.
   - **Always-DOWN Diagnostic:** Labeled strictly as an ex-post diagnostic (51.63%). Because DOWN was only the majority in the test set, selecting Always-DOWN a priori would represent data leakage.
   - **Previous-Day Momentum:** Predicts $	ext{sign}(Close[t] - Close[t-1])$ using strictly the final two completed sessions of the input window.
   - **SMA20 Trend:** Predicts $	ext{sign}(Close[t] - 	ext{MA20}[t])$ using strictly completed session $t$.

3. **Metric Definitions:**
   - **Three-Class Raw Accuracy:** $\frac{\sum \mathbb{I}(\hat{d} = d)}{N}$ where $N = 184$.
   - **Non-FLAT Directional Accuracy:** Evaluated exclusively on sessions where actual movement was non-zero ($N = 178$).
   - **Balanced Accuracy (Macro Recall):** $\frac{1}{3} (\text{Recall}_{\text{UP}} + \text{Recall}_{\text{DOWN}} + \text{Recall}_{\text{FLAT}})$.
   - **Matthews Correlation Coefficient (MCC):** Multiclass correlation accounting for class imbalance, where $0.0$ indicates chance performance.

---

## 3. Directional Confusion Matrices

#### Training Majority Baseline (UP)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 83 | 0 | 0 |
| **Actual DOWN** (95) | 95 | 0 | 0 |
| **Actual FLAT** (6) | 6 | 0 | 0 |

#### Always-DOWN (Ex-Post Diagnostic)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 0 | 83 | 0 |
| **Actual DOWN** (95) | 0 | 95 | 0 |
| **Actual FLAT** (6) | 0 | 6 | 0 |

#### Previous-Day Momentum

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 33 | 48 | 2 |
| **Actual DOWN** (95) | 48 | 43 | 4 |
| **Actual FLAT** (6) | 3 | 3 | 0 |

#### SMA20 Trend Baseline

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 26 | 57 | 0 |
| **Actual DOWN** (95) | 37 | 58 | 0 |
| **Actual FLAT** (6) | 2 | 4 | 0 |

#### Naive Persistence (Price Baseline)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 0 | 0 | 83 |
| **Actual DOWN** (95) | 0 | 0 | 95 |
| **Actual FLAT** (6) | 0 | 0 | 6 |

#### Saved LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 43 | 40 | 0 |
| **Actual DOWN** (95) | 41 | 54 | 0 |
| **Actual FLAT** (6) | 3 | 3 | 0 |

#### Saved CNN-LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 50 | 33 | 0 |
| **Actual DOWN** (95) | 52 | 43 | 0 |
| **Actual FLAT** (6) | 4 | 2 | 0 |

#### Run-Specific LSTM (run_RELIANCE_NS_20261007_120035)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 29 | 54 | 0 |
| **Actual DOWN** (95) | 35 | 60 | 0 |
| **Actual FLAT** (6) | 3 | 3 | 0 |

#### Run-Specific CNN-LSTM (run_RELIANCE_NS_20261007_120035)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (83) | 68 | 15 | 0 |
| **Actual DOWN** (95) | 74 | 21 | 0 |
| **Actual FLAT** (6) | 5 | 1 | 0 |


---

## 4. Key Findings

1. **Resolution of the 3-Class Naive Distortion:**
   - Naive Persistence ($P_{t+1} = P_t$) predicts FLAT for 100% of samples. It scores **3.26%** on 3-class accuracy solely because 6 sessions were flat.
   - However, in price space, Naive Persistence achieves lower MAE (14.16 INR) and RMSE (18.52 INR) than all neural networks, verifying the Martingale property of raw asset prices.

2. **True Statistical Edge Analysis:**
   - Against the valid Training-Majority baseline (45.11%), the Saved LSTM (52.72% raw, 54.49% non-flat) exhibits a positive raw edge (+7.61%).
   - Against the ex-post Always-DOWN diagnostic (51.63%), the Saved LSTM achieves 52.72%, representing an edge of only +1.09% (2 additional days).
   - The Matthews Correlation Coefficient (MCC) for Saved LSTM is **0.0812** and Cohen's Kappa is **0.0811**, indicating low directional correlation above chance.
   - The Saved CNN-LSTM achieves MCC **0.0525**, showing near-zero directional separation from random coin tossing.

---

## 5. Methodological Limitations

- **Single Static Regime:** This audit covers 184 trading days for a single equity (`RELIANCE.NS`). Performance in this slice cannot be generalized as statistically significant.
- **Price Regression vs. Classification:** Current models minimize Mean Squared Error on non-stationary price levels, which intrinsically incentives auto-regressive tracking of $Close[t]$ rather than learning directional price dynamics.

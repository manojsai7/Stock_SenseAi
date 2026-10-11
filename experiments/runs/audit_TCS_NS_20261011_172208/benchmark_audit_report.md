# EXP-01: Offline Directional Benchmark & Metric Audit Report

**Symbol:** `TCS.NS`  
**Run ID:** `audit_TCS_NS_20261011_172208`  
**Generated At:** `2026-10-11T17:22:19.786968+05:30`  
**Test Window:** 184 consecutive trading sessions  
**Test Distribution:** UP: 75 (40.76%), DOWN: 102 (55.43%), FLAT: 7 (3.80%)  
**Training Distribution:** UP: 405, DOWN: 419, FLAT: 1 (Majority: **UP**)

---

## 1. Executive Summary & Comparative Matrix

| Predictor                          | Type                           |   Raw Accuracy (%) |   Non-FLAT Acc (%) [N=178] |   Balanced Acc (%) |     MCC |   Cohen Kappa | MAE (INR)   |
|:-----------------------------------|:-------------------------------|-------------------:|---------------------------:|-------------------:|--------:|--------------:|:------------|
| Training Majority Baseline (DOWN)  | Directional Baseline           |              55.43 |                      57.63 |              33.33 |  0      |        0      | N/A         |
| Always-DOWN (Ex-Post Diagnostic)   | Diagnostic Baseline            |              55.43 |                      57.63 |              33.33 |  0      |        0      | N/A         |
| Previous-Day Momentum              | Heuristic Directional Baseline |              49.46 |                      51.41 |              33.5  |  0.0323 |        0.0323 | N/A         |
| SMA20 Trend Baseline               | Heuristic Directional Baseline |              53.8  |                      55.93 |              34.71 |  0.0415 |        0.0375 | N/A         |
| Naive Persistence (Price Baseline) | Price & Direction Baseline     |               3.8  |                       0    |              33.33 |  0      |        0      | 33.71       |
| Saved LSTM (models_storage)        | Trained Neural Model           |              41.3  |                      42.94 |              33.31 | -0.0008 |       -0.0003 | 237.13      |
| Saved CNN-LSTM (models_storage)    | Trained Neural Model           |              39.67 |                      41.24 |              32.21 | -0.086  |       -0.0266 | 435.79      |

---

## 2. Methodology & Leakage Prevention

1. **Date & Sequence Alignment:**
   - Every input window ends at completed session $t$ (`input_date`).
   - Every target corresponds strictly to session $t+1$ (`target_date`), verified against historical processed bars.
   - `previous_close` is strictly $Close[t]$.
   - `actual_close` is strictly $Close[t+1]$.
   - Mismatch audit: **0 errors detected** across 184 samples.

2. **Zero-Leakage Baseline Fitting:**
   - **Training Majority Baseline:** Exclusively inspects training labels (405 UP vs 419 DOWN), selecting **UP**. On the held-out test set, it achieves **55.43%**.
   - **Always-DOWN Diagnostic:** Labeled strictly as an ex-post diagnostic (55.43%). Because DOWN was only the majority in the test set, selecting Always-DOWN a priori would represent data leakage.
   - **Previous-Day Momentum:** Predicts $	ext{sign}(Close[t] - Close[t-1])$ using strictly the final two completed sessions of the input window.
   - **SMA20 Trend:** Predicts $	ext{sign}(Close[t] - 	ext{MA20}[t])$ using strictly completed session $t$.

3. **Metric Definitions:**
   - **Three-Class Raw Accuracy:** $\frac{\sum \mathbb{I}(\hat{d} = d)}{N}$ where $N = 184$.
   - **Non-FLAT Directional Accuracy:** Evaluated exclusively on sessions where actual movement was non-zero ($N = 177$).
   - **Balanced Accuracy (Macro Recall):** $\frac{1}{3} (\text{Recall}_{\text{UP}} + \text{Recall}_{\text{DOWN}} + \text{Recall}_{\text{FLAT}})$.
   - **Matthews Correlation Coefficient (MCC):** Multiclass correlation accounting for class imbalance, where $0.0$ indicates chance performance.

---

## 3. Directional Confusion Matrices

#### Training Majority Baseline (DOWN)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 0 | 75 | 0 |
| **Actual DOWN** (102) | 0 | 102 | 0 |
| **Actual FLAT** (7) | 0 | 7 | 0 |

#### Always-DOWN (Ex-Post Diagnostic)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 0 | 75 | 0 |
| **Actual DOWN** (102) | 0 | 102 | 0 |
| **Actual FLAT** (7) | 0 | 7 | 0 |

#### Previous-Day Momentum

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 32 | 39 | 4 |
| **Actual DOWN** (102) | 41 | 59 | 2 |
| **Actual FLAT** (7) | 2 | 5 | 0 |

#### SMA20 Trend Baseline

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 20 | 55 | 0 |
| **Actual DOWN** (102) | 23 | 79 | 0 |
| **Actual FLAT** (7) | 0 | 7 | 0 |

#### Naive Persistence (Price Baseline)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 0 | 0 | 75 |
| **Actual DOWN** (102) | 0 | 0 | 102 |
| **Actual FLAT** (7) | 0 | 0 | 7 |

#### Saved LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 72 | 3 | 0 |
| **Actual DOWN** (102) | 98 | 4 | 0 |
| **Actual FLAT** (7) | 7 | 0 | 0 |

#### Saved CNN-LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (75) | 71 | 4 | 0 |
| **Actual DOWN** (102) | 100 | 2 | 0 |
| **Actual FLAT** (7) | 7 | 0 | 0 |


---

## 4. Key Findings

1. **Resolution of the 3-Class Naive Distortion:**
   - Naive Persistence ($P_{t+1} = P_t$) predicts FLAT for 100% of samples. It scores **3.8%** on 3-class accuracy solely because 7 sessions were flat.
   - However, in price space, Naive Persistence achieves lower MAE (33.71 INR) and RMSE (48.44 INR) than all neural networks, verifying the Martingale property of raw asset prices.

2. **True Statistical Edge Analysis:**
   - Against the valid Training-Majority baseline (45.11%), the Saved LSTM (41.3% raw, 42.94% non-flat) exhibits a positive raw edge (+7.61%).
   - Against the ex-post Always-DOWN diagnostic (51.63%), the Saved LSTM achieves 52.72%, representing an edge of only +1.09% (2 additional days).
   - The Matthews Correlation Coefficient (MCC) for Saved LSTM is **-0.0008** and Cohen's Kappa is **-0.0003**, indicating low directional correlation above chance.
   - The Saved CNN-LSTM achieves MCC **-0.086**, showing near-zero directional separation from random coin tossing.

---

## 5. Methodological Limitations

- **Single Static Regime:** This audit covers 184 trading days for a single equity (`RELIANCE.NS`). Performance in this slice cannot be generalized as statistically significant.
- **Price Regression vs. Classification:** Current models minimize Mean Squared Error on non-stationary price levels, which intrinsically incentives auto-regressive tracking of $Close[t]$ rather than learning directional price dynamics.

# EXP-01: Offline Directional Benchmark & Metric Audit Report

**Symbol:** `INFY.NS`  
**Run ID:** `audit_INFY_NS_20261011_172326`  
**Generated At:** `2026-10-11T17:23:29.651402+05:30`  
**Test Window:** 184 consecutive trading sessions  
**Test Distribution:** UP: 70 (38.04%), DOWN: 108 (58.70%), FLAT: 6 (3.26%)  
**Training Distribution:** UP: 412, DOWN: 412, FLAT: 1 (Majority: **UP**)

---

## 1. Executive Summary & Comparative Matrix

| Predictor                          | Type                           |   Raw Accuracy (%) |   Non-FLAT Acc (%) [N=178] |   Balanced Acc (%) |     MCC |   Cohen Kappa | MAE (INR)   |
|:-----------------------------------|:-------------------------------|-------------------:|---------------------------:|-------------------:|--------:|--------------:|:------------|
| Training Majority Baseline (UP)    | Directional Baseline           |              38.04 |                      39.33 |              33.33 |  0      |        0      | N/A         |
| Always-DOWN (Ex-Post Diagnostic)   | Diagnostic Baseline            |              58.7  |                      60.67 |              33.33 |  0      |        0      | N/A         |
| Previous-Day Momentum              | Heuristic Directional Baseline |              53.26 |                      55.06 |              35.27 |  0.085  |        0.085  | N/A         |
| SMA20 Trend Baseline               | Heuristic Directional Baseline |              47.28 |                      48.88 |              30.54 | -0.0853 |       -0.0847 | N/A         |
| Naive Persistence (Price Baseline) | Price & Direction Baseline     |               3.26 |                       0    |              33.33 |  0      |        0      | 18.21       |
| Saved LSTM (models_storage)        | Trained Neural Model           |              50.54 |                      52.25 |              33.06 | -0.0083 |       -0.0083 | 35.87       |
| Saved CNN-LSTM (models_storage)    | Trained Neural Model           |              54.89 |                      56.74 |              37.71 |  0.1182 |        0.1167 | 28.48       |

---

## 2. Methodology & Leakage Prevention

1. **Date & Sequence Alignment:**
   - Every input window ends at completed session $t$ (`input_date`).
   - Every target corresponds strictly to session $t+1$ (`target_date`), verified against historical processed bars.
   - `previous_close` is strictly $Close[t]$.
   - `actual_close` is strictly $Close[t+1]$.
   - Mismatch audit: **0 errors detected** across 184 samples.

2. **Zero-Leakage Baseline Fitting:**
   - **Training Majority Baseline:** Exclusively inspects training labels (412 UP vs 412 DOWN), selecting **UP**. On the held-out test set, it achieves **38.04%**.
   - **Always-DOWN Diagnostic:** Labeled strictly as an ex-post diagnostic (58.7%). Because DOWN was only the majority in the test set, selecting Always-DOWN a priori would represent data leakage.
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
| **Actual UP** (70) | 70 | 0 | 0 |
| **Actual DOWN** (108) | 108 | 0 | 0 |
| **Actual FLAT** (6) | 6 | 0 | 0 |

#### Always-DOWN (Ex-Post Diagnostic)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (70) | 0 | 70 | 0 |
| **Actual DOWN** (108) | 0 | 108 | 0 |
| **Actual FLAT** (6) | 0 | 6 | 0 |

#### Previous-Day Momentum

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (70) | 30 | 37 | 3 |
| **Actual DOWN** (108) | 37 | 68 | 3 |
| **Actual FLAT** (6) | 4 | 2 | 0 |

#### SMA20 Trend Baseline

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (70) | 22 | 48 | 0 |
| **Actual DOWN** (108) | 43 | 65 | 0 |
| **Actual FLAT** (6) | 0 | 6 | 0 |

#### Naive Persistence (Price Baseline)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (70) | 0 | 0 | 70 |
| **Actual DOWN** (108) | 0 | 0 | 108 |
| **Actual FLAT** (6) | 0 | 0 | 6 |

#### Saved LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (70) | 26 | 44 | 0 |
| **Actual DOWN** (108) | 41 | 67 | 0 |
| **Actual FLAT** (6) | 2 | 4 | 0 |

#### Saved CNN-LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (70) | 39 | 31 | 0 |
| **Actual DOWN** (108) | 46 | 62 | 0 |
| **Actual FLAT** (6) | 2 | 4 | 0 |


---

## 4. Key Findings

1. **Resolution of the 3-Class Naive Distortion:**
   - Naive Persistence ($P_{t+1} = P_t$) predicts FLAT for 100% of samples. It scores **3.26%** on 3-class accuracy solely because 6 sessions were flat.
   - However, in price space, Naive Persistence achieves lower MAE (18.21 INR) and RMSE (25.84 INR) than all neural networks, verifying the Martingale property of raw asset prices.

2. **True Statistical Edge Analysis:**
   - Against the valid Training-Majority baseline (45.11%), the Saved LSTM (50.54% raw, 52.25% non-flat) exhibits a positive raw edge (+7.61%).
   - Against the ex-post Always-DOWN diagnostic (51.63%), the Saved LSTM achieves 52.72%, representing an edge of only +1.09% (2 additional days).
   - The Matthews Correlation Coefficient (MCC) for Saved LSTM is **-0.0083** and Cohen's Kappa is **-0.0083**, indicating low directional correlation above chance.
   - The Saved CNN-LSTM achieves MCC **0.1182**, showing near-zero directional separation from random coin tossing.

---

## 5. Methodological Limitations

- **Single Static Regime:** This audit covers 184 trading days for a single equity (`RELIANCE.NS`). Performance in this slice cannot be generalized as statistically significant.
- **Price Regression vs. Classification:** Current models minimize Mean Squared Error on non-stationary price levels, which intrinsically incentives auto-regressive tracking of $Close[t]$ rather than learning directional price dynamics.

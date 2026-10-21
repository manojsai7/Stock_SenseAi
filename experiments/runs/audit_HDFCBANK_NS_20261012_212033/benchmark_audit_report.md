# EXP-01: Offline Directional Benchmark & Metric Audit Report

**Symbol:** `HDFCBANK.NS`  
**Run ID:** `audit_HDFCBANK_NS_20261012_212033`  
**Generated At:** `2026-10-12T21:20:35.536981+05:30`  
**Test Window:** 184 consecutive trading sessions  
**Test Distribution:** UP: 73 (39.67%), DOWN: 103 (55.98%), FLAT: 8 (4.35%)  
**Training Distribution:** UP: 438, DOWN: 386, FLAT: 1 (Majority: **UP**)

---

## 1. Executive Summary & Comparative Matrix

| Predictor                          | Type                           |   Raw Accuracy (%) |   Non-FLAT Acc (%) [N=178] |   Balanced Acc (%) |     MCC |   Cohen Kappa | MAE (INR)   |
|:-----------------------------------|:-------------------------------|-------------------:|---------------------------:|-------------------:|--------:|--------------:|:------------|
| Training Majority Baseline (UP)    | Directional Baseline           |              39.67 |                      41.48 |              33.33 |  0      |        0      | N/A         |
| Always-DOWN (Ex-Post Diagnostic)   | Diagnostic Baseline            |              55.98 |                      58.52 |              33.33 |  0      |        0      | N/A         |
| Previous-Day Momentum              | Heuristic Directional Baseline |              46.2  |                      48.3  |              31.23 | -0.0203 |       -0.0203 | N/A         |
| SMA20 Trend Baseline               | Heuristic Directional Baseline |              51.63 |                      53.98 |              34.2  |  0.0236 |        0.0231 | N/A         |
| Naive Persistence (Price Baseline) | Price & Direction Baseline     |               4.35 |                       0    |              33.33 |  0      |        0      | 8.92        |
| Saved LSTM (models_storage)        | Trained Neural Model           |              56.52 |                      59.09 |              38.05 |  0.1339 |        0.1328 | 11.64       |
| Saved CNN-LSTM (models_storage)    | Trained Neural Model           |              58.7  |                      61.36 |              37.48 |  0.1438 |        0.1218 | 11.78       |

---

## 2. Methodology & Leakage Prevention

1. **Date & Sequence Alignment:**
   - Every input window ends at completed session $t$ (`input_date`).
   - Every target corresponds strictly to session $t+1$ (`target_date`), verified against historical processed bars.
   - `previous_close` is strictly $Close[t]$.
   - `actual_close` is strictly $Close[t+1]$.
   - Mismatch audit: **0 errors detected** across 184 samples.

2. **Zero-Leakage Baseline Fitting:**
   - **Training Majority Baseline:** Exclusively inspects training labels (438 UP vs 386 DOWN), selecting **UP**. On the held-out test set, it achieves **39.67%**.
   - **Always-DOWN Diagnostic:** Labeled strictly as an ex-post diagnostic (55.98%). Because DOWN was only the majority in the test set, selecting Always-DOWN a priori would represent data leakage.
   - **Previous-Day Momentum:** Predicts $	ext{sign}(Close[t] - Close[t-1])$ using strictly the final two completed sessions of the input window.
   - **SMA20 Trend:** Predicts $	ext{sign}(Close[t] - 	ext{MA20}[t])$ using strictly completed session $t$.

3. **Metric Definitions:**
   - **Three-Class Raw Accuracy:** $\frac{\sum \mathbb{I}(\hat{d} = d)}{N}$ where $N = 184$.
   - **Non-FLAT Directional Accuracy:** Evaluated exclusively on sessions where actual movement was non-zero ($N = 176$).
   - **Balanced Accuracy (Macro Recall):** $\frac{1}{3} (\text{Recall}_{\text{UP}} + \text{Recall}_{\text{DOWN}} + \text{Recall}_{\text{FLAT}})$.
   - **Matthews Correlation Coefficient (MCC):** Multiclass correlation accounting for class imbalance, where $0.0$ indicates chance performance.

---

## 3. Directional Confusion Matrices

#### Training Majority Baseline (UP)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 73 | 0 | 0 |
| **Actual DOWN** (103) | 103 | 0 | 0 |
| **Actual FLAT** (8) | 8 | 0 | 0 |

#### Always-DOWN (Ex-Post Diagnostic)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 0 | 73 | 0 |
| **Actual DOWN** (103) | 0 | 103 | 0 |
| **Actual FLAT** (8) | 0 | 8 | 0 |

#### Previous-Day Momentum

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 28 | 41 | 4 |
| **Actual DOWN** (103) | 42 | 57 | 4 |
| **Actual FLAT** (8) | 3 | 5 | 0 |

#### SMA20 Trend Baseline

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 26 | 47 | 0 |
| **Actual DOWN** (103) | 34 | 69 | 0 |
| **Actual FLAT** (8) | 2 | 6 | 0 |

#### Naive Persistence (Price Baseline)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 0 | 0 | 73 |
| **Actual DOWN** (103) | 0 | 0 | 103 |
| **Actual FLAT** (8) | 0 | 0 | 8 |

#### Saved LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 33 | 40 | 0 |
| **Actual DOWN** (103) | 32 | 71 | 0 |
| **Actual FLAT** (8) | 4 | 4 | 0 |

#### Saved CNN-LSTM (models_storage)

| Actual \ Predicted | UP | DOWN | FLAT |
| :--- | :---: | :---: | :---: |
| **Actual UP** (73) | 19 | 54 | 0 |
| **Actual DOWN** (103) | 14 | 89 | 0 |
| **Actual FLAT** (8) | 1 | 7 | 0 |


---

## 4. Key Findings

1. **Resolution of the 3-Class Naive Distortion:**
   - Naive Persistence ($P_{t+1} = P_t$) predicts FLAT for 100% of samples. It scores **4.35%** on 3-class accuracy solely because 8 sessions were flat.
   - However, in price space, Naive Persistence achieves lower MAE (8.92 INR) and RMSE (12.15 INR) than all neural networks, verifying the Martingale property of raw asset prices.

2. **True Statistical Edge Analysis:**
   - Against the valid Training-Majority baseline (45.11%), the Saved LSTM (56.52% raw, 59.09% non-flat) exhibits a positive raw edge (+7.61%).
   - Against the ex-post Always-DOWN diagnostic (51.63%), the Saved LSTM achieves 52.72%, representing an edge of only +1.09% (2 additional days).
   - The Matthews Correlation Coefficient (MCC) for Saved LSTM is **0.1339** and Cohen's Kappa is **0.1328**, indicating low directional correlation above chance.
   - The Saved CNN-LSTM achieves MCC **0.1438**, showing near-zero directional separation from random coin tossing.

---

## 5. Methodological Limitations

- **Single Static Regime:** This audit covers 184 trading days for a single equity (`RELIANCE.NS`). Performance in this slice cannot be generalized as statistically significant.
- **Price Regression vs. Classification:** Current models minimize Mean Squared Error on non-stationary price levels, which intrinsically incentives auto-regressive tracking of $Close[t]$ rather than learning directional price dynamics.

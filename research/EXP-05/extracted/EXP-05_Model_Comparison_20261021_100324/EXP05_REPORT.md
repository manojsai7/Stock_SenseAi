# EXP-05: Model Comparison & Direction Calibration Report

**Target Equity:** `RELIANCE.NS`  
**Target:** Next-Day Simple Return: $r_{t+1} = \frac{Close_{t+1}}{Close_t} - 1$  
**Causal Stationary Features (7):** `return_1d, return_5d, return_20d, intraday_return, hl_range_ratio, close_to_ma20, volume_ratio`  
**Output Directory:** `/content/drive/MyDrive/StockSense_Experiments/EXP-05_Model_Comparison_20261021_100324`  
**Run Timestamp:** `2026-10-21T10:04:17.052753+00:00`  

---

## 1. Walk-Forward Cross-Fold Macro Averages

| Model | Macro Raw Acc (%) | Macro Balanced Acc (%) | Macro MCC | Macro UP Recall (%) | Macro DOWN Recall (%) | Macro Price MAE | Macro Price MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM (Calibrated)** | 51.63% | 33.94% | 0.0262 | 58.17% | 43.64% | INR 13.26 | 0.97% |
| **CNN-LSTM (Uncalibrated)** | 49.1% | 32.52% | -0.0248 | 56.95% | 40.62% | INR 13.26 | 0.97% |
| **Ridge Regression (Calibrated)** | 48.01% | 33.35% | 0.0132 | 62.3% | 37.76% | INR 13.21 | 0.97% |
| **Ridge Regression (Uncalibrated)** | 50.0% | 34.11% | 0.0274 | 81.48% | 20.87% | INR 13.21 | 0.97% |
| **Logistic Regression Baseline** | 50.9% | 34.28% | 0.0296 | 41.12% | 61.72% | N/A | N/A |
| **Naive Persistence** | 1.45% | 33.33% | 0.0 | 0.0% | 0.0% | INR 13.17 | 0.97% |
| **Previous-Day Momentum** | 45.47% | 30.69% | -0.0619 | 45.54% | 46.53% | N/A | N/A |
| **Training Majority** | 48.73% | 33.33% | 0.0 | 100.0% | 0.0% | N/A | N/A |

---

## 2. Per-Fold Evaluation Breakdown

### Walk-Forward Fold 1 (Mid-2024 Test) (Fold 1)
- **Test Period:** `2024-07-24` to `2025-04-21` ($N = 184$)
- **Training Samples:** 455, **Validation Samples:** 184
- **Validation Tuning Parameters:**
  - CNN-LSTM calibrated threshold $\theta$: `0.00018`
  - Ridge chosen $\alpha$: `10000.0`, threshold $\theta$: `-0.001128`
  - Logistic chosen $C$: `0.001`, probability threshold: `0.54`

| Model | 3-Class Raw Acc (%) | Balanced Acc (%) | MCC | UP Recall (%) | DOWN Recall (%) | Price MAE | Price MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM (Calibrated)** | 53.26% | 35.79% | 0.103 | 76.92% | 30.43% | INR 13.26 | 1.01% |
| **CNN-LSTM (Uncalibrated)** | 49.46% | 33.28% | -0.0022 | 85.71% | 14.13% | INR 13.26 | 1.01% |
| **Ridge Regression (Calibrated)** | 49.46% | 33.33% | 0.0 | 100.0% | 0.0% | INR 13.34 | 1.02% |
| **Ridge Regression (Uncalibrated)** | 50.0% | 33.64% | 0.0125 | 84.62% | 16.3% | INR 13.34 | 1.02% |
| **Logistic Regression** | 51.63% | 34.52% | 0.0398 | 28.57% | 75.0% | N/A | N/A |
| **Naive Persistence** | 0.54% | 33.33% | 0.0 | 0.0% | 0.0% | INR 13.32 | 1.02% |
| **Momentum** | 50.0% | 33.52% | 0.0107 | 50.55% | 50.0% | N/A | N/A |
| **Training Majority** | 49.46% | 33.33% | 0.0 | 100.0% | 0.0% | N/A | N/A |

### Walk-Forward Fold 2 (Late-2024 / Early-2025 Test) (Fold 2)
- **Test Period:** `2025-04-22` to `2026-01-13` ($N = 184$)
- **Training Samples:** 639, **Validation Samples:** 184
- **Validation Tuning Parameters:**
  - CNN-LSTM calibrated threshold $\theta$: `-0.001052`
  - Ridge chosen $\alpha$: `10000.0`, threshold $\theta$: `0.001337`
  - Logistic chosen $C$: `0.1`, probability threshold: `0.47`

| Model | 3-Class Raw Acc (%) | Balanced Acc (%) | MCC | UP Recall (%) | DOWN Recall (%) | Price MAE | Price MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM (Calibrated)** | 50.54% | 32.8% | -0.0299 | 91.58% | 6.82% | INR 12.43 | 0.86% |
| **CNN-LSTM (Uncalibrated)** | 45.11% | 29.6% | -0.1281 | 69.47% | 19.32% | INR 12.43 | 0.86% |
| **Ridge Regression (Calibrated)** | 48.91% | 33.9% | 0.0338 | 7.37% | 94.32% | INR 12.02 | 0.83% |
| **Ridge Regression (Uncalibrated)** | 53.8% | 35.43% | 0.0718 | 77.89% | 28.41% | INR 12.02 | 0.83% |
| **Logistic Regression** | 52.17% | 34.97% | 0.0487 | 52.63% | 52.27% | N/A | N/A |
| **Naive Persistence** | 0.54% | 33.33% | 0.0 | 0.0% | 0.0% | INR 12.02 | 0.83% |
| **Momentum** | 45.11% | 30.21% | -0.0872 | 46.32% | 44.32% | N/A | N/A |
| **Training Majority** | 51.63% | 33.33% | 0.0 | 100.0% | 0.0% | N/A | N/A |

### Walk-Forward Fold 3 (Benchmark 2026 Test Block) (Fold 3)
- **Test Period:** `2026-01-14` to `2026-10-06` ($N = 184$)
- **Training Samples:** 823, **Validation Samples:** 184
- **Validation Tuning Parameters:**
  - CNN-LSTM calibrated threshold $\theta$: `0.000307`
  - Ridge chosen $\alpha$: `10000.0`, threshold $\theta$: `4.5e-05`
  - Logistic chosen $C$: `10.0`, probability threshold: `0.56`

| Model | 3-Class Raw Acc (%) | Balanced Acc (%) | MCC | UP Recall (%) | DOWN Recall (%) | Price MAE | Price MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM (Calibrated)** | 51.09% | 33.24% | 0.0055 | 6.02% | 93.68% | INR 14.09 | 1.05% |
| **CNN-LSTM (Uncalibrated)** | 52.72% | 34.69% | 0.056 | 15.66% | 88.42% | INR 14.09 | 1.05% |
| **Ridge Regression (Calibrated)** | 45.65% | 32.82% | 0.0059 | 79.52% | 18.95% | INR 14.27 | 1.07% |
| **Ridge Regression (Uncalibrated)** | 46.2% | 33.27% | -0.0021 | 81.93% | 17.89% | INR 14.27 | 1.07% |
| **Logistic Regression** | 48.91% | 33.35% | 0.0002 | 42.17% | 57.89% | N/A | N/A |
| **Naive Persistence** | 3.26% | 33.33% | 0.0 | 0.0% | 0.0% | INR 14.16 | 1.06% |
| **Momentum** | 41.3% | 28.34% | -0.1091 | 39.76% | 45.26% | N/A | N/A |
| **Training Majority** | 45.11% | 33.33% | 0.0 | 100.0% | 0.0% | N/A | N/A |
| **Production Raw-Price CNN-LSTM** | 50.54% | 35.17% | 0.0525 | 60.24% | 45.26% | INR 19.32 | 1.44% |


---

## 3. Methodological Transparency & Holdout Integrity

1. **Separation of Exploratory from Untouched Evaluation:**
   - Fold 3 (`2026-01-14` to `2026-10-06`) was already inspected in EXP-01, EXP-02, EXP-03, and EXP-04.
   - Therefore, Fold 3 is **an established benchmark block**, not an uninspected virgin holdout.
   - Cross-fold macro averages across all three chronological walk-forward regimes provide the primary defense against selection bias.

2. **Validation-Only Calibration:**
   - Every threshold ($	heta$) and hyperparameter ($lpha$, $C$, $p_{threshold}$) was fitted strictly on each fold's validation slice and evaluated out-of-sample on the subsequent test slice.
   - Zero test labels were utilized during tuning.

3. **Comparison Against Naive Persistence:**
   - Price MAE and MAPE must beat Naive Persistence (Fold 3: INR 14.16, 1.06%) to demonstrate price forecasting utility.
   - Directional MCC must exceed +0.10 with balanced recall (both UP and DOWN recall > 20%) to prove non-trivial directional edge.

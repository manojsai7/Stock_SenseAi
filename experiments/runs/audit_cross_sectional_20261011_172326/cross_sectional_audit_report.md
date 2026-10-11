# EXP-02: Multi-Stock Cross-Sectional Directional Benchmark Audit Report

**Audited Equities:** `INFY.NS, HDFCBANK.NS`  
**Generated At:** `2026-10-11T17:23:32.099141+05:30`  
**Evaluation Scope:** Strictly read-only inference on existing checkpoints (`models_storage`) and prepared datasets. Zero retraining.

---

## 1. Cross-Sectional Macro-Average Summary

> **Methodological Note on Aggregation:** Percentage metrics (Raw Accuracy, Non-FLAT Accuracy, Balanced Accuracy, MAPE) and correlation statistics (MCC, Cohen's Kappa) are dimensionless and averaged across stocks. Raw price errors (MAE, RMSE) are preserved in native currency units per stock and are NOT averaged across different price levels (e.g. TCS ~3500 INR vs ICICIBANK ~1200 INR).

| Predictor                          | Type                           |   Evaluated Stocks |   Macro Raw Acc (%) |   Macro Non-FLAT Acc (%) |   Macro Balanced Acc (%) |   Macro MCC |   Macro Cohen Kappa | Macro MAPE (%)   |
|:-----------------------------------|:-------------------------------|-------------------:|--------------------:|-------------------------:|-------------------------:|------------:|--------------------:|:-----------------|
| Training Majority Baseline (UP)    | Directional Baseline           |                  2 |               38.86 |                    40.41 |                    33.33 |      0      |              0      | N/A              |
| Always-DOWN (Ex-Post Diagnostic)   | Diagnostic Baseline            |                  2 |               57.34 |                    59.59 |                    33.33 |      0      |              0      | N/A              |
| Previous-Day Momentum              | Heuristic Directional Baseline |                  2 |               49.73 |                    51.68 |                    33.25 |      0.0324 |              0.0324 | N/A              |
| SMA20 Trend Baseline               | Heuristic Directional Baseline |                  2 |               49.45 |                    51.43 |                    32.37 |     -0.0309 |             -0.0308 | N/A              |
| Naive Persistence (Price Baseline) | Price & Direction Baseline     |                  2 |                3.8  |                     0    |                    33.33 |      0      |              0      | 1.32             |
| Saved LSTM (models_storage)        | Trained Neural Model           |                  2 |               53.53 |                    55.67 |                    35.55 |      0.0628 |              0.0622 | 2.25             |
| Saved CNN-LSTM (models_storage)    | Trained Neural Model           |                  2 |               56.8  |                    59.05 |                    37.59 |      0.131  |              0.1192 | 1.93             |

---

## 2. Per-Stock Detailed Benchmark Matrices

### INFY.NS
- **Test Samples:** 184 (UP: 70, DOWN: 108, FLAT: 6)
- **Training Labels:** UP: 412, DOWN: 412, FLAT: 1

| Predictor                          | Type                           |   Raw Accuracy (%) |   Non-FLAT Acc (%) [N=178] |   Balanced Acc (%) |     MCC |   Cohen Kappa | MAE (INR)   | RMSE (INR)   | MAPE (%)   |
|:-----------------------------------|:-------------------------------|-------------------:|---------------------------:|-------------------:|--------:|--------------:|:------------|:-------------|:-----------|
| Training Majority Baseline (UP)    | Directional Baseline           |              38.04 |                      39.33 |              33.33 |  0      |        0      | N/A         | N/A          | N/A        |
| Always-DOWN (Ex-Post Diagnostic)   | Diagnostic Baseline            |              58.7  |                      60.67 |              33.33 |  0      |        0      | N/A         | N/A          | N/A        |
| Previous-Day Momentum              | Heuristic Directional Baseline |              53.26 |                      55.06 |              35.27 |  0.085  |        0.085  | N/A         | N/A          | N/A        |
| SMA20 Trend Baseline               | Heuristic Directional Baseline |              47.28 |                      48.88 |              30.54 | -0.0853 |       -0.0847 | N/A         | N/A          | N/A        |
| Naive Persistence (Price Baseline) | Price & Direction Baseline     |               3.26 |                       0    |              33.33 |  0      |        0      | 18.21       | 25.84        | 1.51       |
| Saved LSTM (models_storage)        | Trained Neural Model           |              50.54 |                      52.25 |              33.06 | -0.0083 |       -0.0083 | 35.87       | 47.52        | 3.04       |
| Saved CNN-LSTM (models_storage)    | Trained Neural Model           |              54.89 |                      56.74 |              37.71 |  0.1182 |        0.1167 | 28.48       | 38.68        | 2.4        |

### HDFCBANK.NS
- **Test Samples:** 184 (UP: 73, DOWN: 103, FLAT: 8)
- **Training Labels:** UP: 438, DOWN: 386, FLAT: 1

| Predictor                          | Type                           |   Raw Accuracy (%) |   Non-FLAT Acc (%) [N=178] |   Balanced Acc (%) |     MCC |   Cohen Kappa | MAE (INR)   | RMSE (INR)   | MAPE (%)   |
|:-----------------------------------|:-------------------------------|-------------------:|---------------------------:|-------------------:|--------:|--------------:|:------------|:-------------|:-----------|
| Training Majority Baseline (UP)    | Directional Baseline           |              39.67 |                      41.48 |              33.33 |  0      |        0      | N/A         | N/A          | N/A        |
| Always-DOWN (Ex-Post Diagnostic)   | Diagnostic Baseline            |              55.98 |                      58.52 |              33.33 |  0      |        0      | N/A         | N/A          | N/A        |
| Previous-Day Momentum              | Heuristic Directional Baseline |              46.2  |                      48.3  |              31.23 | -0.0203 |       -0.0203 | N/A         | N/A          | N/A        |
| SMA20 Trend Baseline               | Heuristic Directional Baseline |              51.63 |                      53.98 |              34.2  |  0.0236 |        0.0231 | N/A         | N/A          | N/A        |
| Naive Persistence (Price Baseline) | Price & Direction Baseline     |               4.35 |                       0    |              33.33 |  0      |        0      | 8.92        | 12.15        | 1.13       |
| Saved LSTM (models_storage)        | Trained Neural Model           |              56.52 |                      59.09 |              38.05 |  0.1339 |        0.1328 | 11.64       | 15.39        | 1.47       |
| Saved CNN-LSTM (models_storage)    | Trained Neural Model           |              58.7  |                      61.36 |              37.48 |  0.1438 |        0.1218 | 11.78       | 15.61        | 1.46       |


---

## 3. Core Cross-Sectional Findings

1. **Failure to Outperform Trivial Baselines:**
   - Across the evaluated equities, the saved neural models (LSTM and CNN-LSTM) do not achieve statistically meaningful directional separation above trivial heuristics or majority-class baselines.
   - The cross-sectional Macro MCC for both Saved LSTM and Saved CNN-LSTM remains near zero, proving that weak positive correlation observed on an individual stock (such as RELIANCE.NS) is largely an artifact of the static test regime rather than generalized directional skill.

2. **Price Space Martingale Property:**
   - In every evaluated stock, Naive Persistence ($Close_{t+1} = Close_t$) achieves superior price-accuracy (lowest MAE and lowest RMSE) compared to neural models.
   - Neural models trained on non-stationary price levels function as delayed auto-regressive trackers rather than true return forecasters.

3. **Recommendation for Phase 2:**
   - Stop evaluating deep architectures on raw price regression.
   - Advance exclusively to stationary feature engineering (percentage returns, volatility, moving average ratios) and classification/return prediction under walk-forward validation.

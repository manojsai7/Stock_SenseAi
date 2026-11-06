# StockSense AI V2 — EXP-06 Exploratory Development Addendum

**Date:** 2026-11-06  
**Addendum To:** [`research/EXP-06/EXPERIMENT_PREREGISTRATION.md`](file:///d:/Downloads/Stock_SenseAi/research/EXP-06/EXPERIMENT_PREREGISTRATION.md)  
**Status:** Authorized Controlled Research Track  
**Scope:** Chronological Development Fold Training & Model Selection  
**Confirmatory Training Gate Status:** `NO-GO` (Held Intact)

---

## 1. Purpose and Authority

Following the Sprint 2 quality audit, the confirmatory training gate was determined to be `NO-GO — TRAINING BLOCKED` because the available historical dataset contains only 12 uninspected sessions after EXP-05, failing the preregistered requirement of at least 60 untouched observations.

This Addendum authorizes a strictly delineated **EXPLORATORY DEVELOPMENT TRACK**. The objective is to evaluate model architectures, train development candidates across expanding temporal folds, establish baseline comparisons against Naive and Ridge benchmarks, and select a frozen V2 challenger without compromising scientific integrity.

---

## 2. Governance Principles & Strict Boundaries

1. **Exploratory Development Designation:**
   All model training and evaluations authorized under this Addendum are **exploratory development benchmarks**. They measure model learning dynamics, convergence stability, and cross-fold temporal robustness.
2. **No Claim of Independent Virgin Evidence:**
   Target dates from `2024-07-25` through `2026-10-06` overlap with previously inspected historical development cycles (including EXP-05 Folds 1, 2, and 3). Under no circumstances may performance metrics obtained on these development test folds be represented as independent, untouched holdout evidence.
3. **Preservation of the Confirmatory Holdout Requirement:**
   The original requirement of a minimum 60-session untouched holdout remains fully active and non-negotiable. Passing development benchmarks does NOT unlock production deployment.
4. **V1 Production Protection:**
   No exploratory model trained under this addendum may automatically replace the V1 production model (`models_storage_stationary/RELIANCE_NS/`). All V2 artifacts are isolated under `models_storage_v2/` or `research/EXP-06/artifacts/`.
5. **Future Prospective Confirmation:**
   Final validation for production candidacy requires forward-logging 60–100 finalized prospective trading sessions in shadow mode, evaluated under the preregistered evaluation protocol with paired moving-block bootstrap intervals.

---

## 3. Permitted Data & Chronological Development Partitions

### Dataset Version
- **Dataset:** `research/EXP-06/data/RELIANCE_NS_verified_v2.parquet` / `.csv` (1,234 confirmed records, 2021-10-07 to 2026-10-06).
- **March 18, 2025 Reconciliation:** Verified and safely restored from official NSE Bhavcopy (`sec_bhavdata_full_18032025.csv`, adjustment factor 1.0000, 15,745,877 shares).
- **Quarantine Retained:** 6 exchange holidays (`2026-01-15`, `2026-05-01`, `2026-05-28`, `2026-06-26`, `2026-09-14`, `2026-10-02`) and 1 incomplete candle (`2026-10-07`).

### 3-Fold Chronological Split Structure
Total Supervised Samples: 1,184 ($t-29$ to $t$ input window of 30 sessions, with 20 warmup sessions for `return_20d`).

| Fold | Training Partition | Validation Partition | Development Test Partition |
| :---: | :---: | :---: | :---: |
| **Fold 1** | 0 to 784 (784 samples)<br/>2021-12-21 to 2025-02-18 | 784 to 884 (100 samples)<br/>2025-02-19 to 2025-07-17 | 884 to 984 (100 samples)<br/>2025-07-18 to 2025-12-11 |
| **Fold 2** | 0 to 884 (884 samples)<br/>2021-12-21 to 2025-07-17 | 884 to 984 (100 samples)<br/>2025-07-18 to 2025-12-11 | 984 to 1084 (100 samples)<br/>2025-12-12 to 2026-05-13 |
| **Fold 3** | 0 to 984 (984 samples)<br/>2021-12-21 to 2025-12-11 | 984 to 1084 (100 samples)<br/>2025-12-12 to 2026-05-13 | 1084 to 1184 (100 samples)<br/>2026-05-14 to 2026-10-06 |

- **Strict Isolation:** Scalers (`feature_scaler.pkl` and `target_scaler.pkl`) are fitted strictly per-fold on the training slice only. Validation and test slices are transformed using the training scaler parameters.

---

## 4. Candidate Architecture Matrix & Search Budget

The bounded experiment budget is strictly capped at **18 total neural fits**:
- **Architectures:**
  1. `LSTM`: 32 units, Dropout 0.2, L2 0.0001, MSE loss
  2. `GRU`: 32 units, Dropout 0.2, L2 0.0001, MSE loss
  3. `CNN-LSTM`: Conv1D (32 filters, kernel 3, causal, ReLU), MaxPooling1D (pool 2), LSTM (32 units), Dropout 0.2, L2 0.0001, MSE loss
- **Random Seeds:** 42 and 123
- **Optimizer:** Adam (learning rate 0.001, clipnorm 1.0)
- **Training Constraints:** Max 30 epochs, batch size 32, early stopping patience 8 with `restore_best_weights=True`.
- **Classical Baselines:** Naive persistence ($\Delta = 0$, $P_{t+1} = P_t$), Zero-return baseline, Ridge regression ($\alpha = 1.0$).

---

## 5. Candidate Selection Protocol

The best development candidate will be selected based on:
1. Outperforming Naive persistence and Ridge on mean test fold Price MAE.
2. Aggregate directional MCC $> 0.10$.
3. Directional UP and DOWN recall $> 0.20$.
4. Prediction volatility ratio $\frac{\sigma(\hat{y})}{\sigma(y)} \ge 0.20$ (ruling out severe forecast collapse).
5. Positive MCC in a majority of development folds.

# StockSense AI V2 — EXP-06 Training Gate Decision Report

**Date:** 2026-11-06  
**Symbol:** `RELIANCE.NS`  
**Evaluation Scope:** Sprint 2 Quality Assurance, Data Readiness, Holdout Feasibility & Training Gate Authorization  
**Author:** Quantitative Research & Architecture Integrity (StockSense AI V2 Team)

---

## 1. Context and Mission

Sprint 2 established a strict quantitative gate prior to authorizing genuine Google Colab GPU training for StockSense AI V2. Following the recovery of the historical dataset and reproduction of EXP-05 folds in Sprint 1, two major concerns were uncovered:
1. Historical dataset observations with zero volume (`Volume = 0`).
2. An unfinished daily candle captured during active market hours.
3. The lack of a defensible, untouched holdout of sufficient statistical power.

This report documents the rigorous evaluation of all ten predeclared training gate criteria.

---

## 2. Evaluation of Training Gate Criteria

| # | Gate Condition | Verification Evidence | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Raw Data Provenance Documented** | All raw, live, processed, model, scaler, and prediction log files are cataloged with cryptographic SHA-256 digests in `research/EXP-06/data/source_inventory.json`. Reference Bhavcopies and circulars are cataloged in `research/EXP-06/data/reference/sources.json`. | **PASSED** |
| **2** | **Unfinished Candles Excluded** | The final row in `exp05_original_processed.csv` (`2026-10-07`) was verified against NSE Bhavcopy as an intraday in-progress snapshot (volume 3.33M vs 11.88M, high 1216 vs 1220.8, close 1214.9 vs 1207.7). It is quarantined under `latest_bar_finalization_unconfirmed`. Final accepted session is `2026-10-06`. `2026-10-26` in the current processed data is similarly quarantined. | **PASSED** |
| **3** | **Zero-Volume Rows Resolved / Quarantined** | All seven zero-volume rows were investigated individually against NSE Bhavcopy and circulars. Row 852 (`2025-03-18`) was an active trading session corrupted by provider data glitch. Row 1058 (`2026-01-15`) was an unscheduled holiday (Maharashtra Municipal Elections per CMTR72260). Rows 1128, 1147, 1168, 1224, 1238 were scheduled holidays (CMTR71775). None are imputed or forward-filled; all are isolated in `quarantine.csv`. | **PASSED** |
| **4** | **Trading-Session Alignment Valid** | Official NSE calendar schedule, 15 scheduled holidays for 2026, unscheduled election holiday, and special Sunday session (2026-02-01 Union Budget per CMTR72349) reconciled. No synthetic trading sessions fabricated. | **PASSED** |
| **5** | **Feature Construction Reproducible** | Independent causal recomputation of all seven stationary features and target return across 1,212 samples achieved maximum absolute error of **0.000000000000** against production implementation (`ml/research_v2/readiness.py`). | **PASSED** |
| **6** | **No Target Leakage Exists** | Input feature window terminates at session $t-1$ for target session $t$. All features are strictly backward-looking. | **PASSED** |
| **7** | **Scalers Fitted Only on Training Data** | Standardized in `ml/research_v2/pipeline.py` and preregistered in `EXPERIMENT_PREREGISTRATION.md`. Scalers are fitted exclusively per-fold on training slices. | **PASSED** |
| **8** | **Defensible Holdout Locked** | Detailed timeline audit reveals all historical targets from 2024-07-25 to 2026-10-07 were repeatedly inspected during EXP-05 Fold 1, 2, 3, head-to-head comparisons, and feature ablation studies. The available post-EXP-05 dataset provides only 12 trading sessions (2026-10-08 to 2026-10-23). The preregistered requirement mandates minimum 60 untouched eligible targets. Adequate uninspected historical observations do NOT exist. | **UNMET / FAILED** |
| **9** | **Model Selection Criteria Pre-registered** | Preregistered in `EXPERIMENT_PREREGISTRATION.md` and `SELECTION_PROTOCOL.md`: Naive, Zero-return, Ridge ($\alpha=1$), LSTM, GRU, CNN-LSTM; price MAE primary, return MAE (bps), MCC > 0.10, recall > 0.20, paired moving-block bootstrap (10 sessions, 2000 replicates). | **PASSED** |
| **10** | **Dataset Fingerprint and Config Frozen** | Candidate dataset (`RELIANCE_NS_verified_v1.parquet` / `.csv`) and `dataset_manifest.json` are generated and fingerprinted. However, full experiment execution lock cannot be issued without a defensible, uninspected holdout partition. | **UNMET / FAILED** |

---

## 3. List of Unmet Conditions

### 1. Condition 8 — Defensible Untouched Holdout
- **Deficiency:** The project inspection boundary is established at `2026-11-06`. All historical data up through `2026-10-07` (including EXP-05 Fold 3) was heavily inspected during model selection, ablation studies, and evaluation benchmarking. The candidate dataset terminates at `2026-10-23`, leaving only **12 uninspected trading sessions**.
- **Impact:** 12 sessions is statistically insufficient (preregistered minimum: **60 eligible sessions**). Re-labeling Fold 3 as virgin data would introduce severe snooping bias.
- **Remediation Required:** Acquisition of a fresh, verified trading stream of at least 60 finalized sessions past the inspection boundary, or execution of genuine forward prospective logging over 60–100 sessions.

### 2. Condition 10 — Experiment Lock Authorization
- **Deficiency:** Because Condition 8 is unmet, the experiment locking tool (`ml/research_v2/experiment_lock.py`) correctly raises `ValueError: Insufficient reserved targets for the preregistered holdout` and refuses to issue an execution lock.
- **Impact:** In accordance with the preregistered protocol, training on unverified holdout partitions is strictly prohibited.

---

## 4. Final Training Gate Decision

The data audit, provenance cataloging, feature equivalence recomputation, and anomaly quarantine have succeeded with zero tolerance for data snooping or fabrication. However, because a statistically defensible, untouched holdout of $\ge 60$ sessions does not yet exist in the available historical snapshot, quantitative integrity requires that the training gate remain locked.

NO-GO — TRAINING BLOCKED

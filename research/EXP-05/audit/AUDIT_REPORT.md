# StockSense AI — Google Colab EXP-05 Audit & Provenance Report

**Date of Audit:** 2026-10-26  
**Audited Target Equity:** `RELIANCE.NS`  
**Git Branch:** `research/reliance-model-evaluation`  
**Scope:** Verification of offline Colab EXP-05 artifacts, training boundaries, scaler parameters, model architectures, serialization quirks, and data-leakage boundaries against production models.

---

## 1. Executive Summary & Verification Decision

1. **Artifact Integrity:** All EXP-05 artifacts downloaded from Google Colab were successfully inventoried, extracted, and mathematically verified. The raw ZIP package is safely archived in `research/EXP-05/raw/`.
2. **Mathematical Reproducibility:** Using the extracted weights from `best_cnn_lstm_fold3.keras` and its paired scalers (`feature_scaler_3d_fold3.pkl`, `return_scaler_fold3.pkl`), we reproduced the 184 saved out-of-sample predictions with a maximum absolute deviation of **$4.897 \times 10^{-7}$** (mean difference $2.504 \times 10^{-7}$). Provenance and numerical consistency are **100% verified**.
3. **Eligible Checkpoint Selection:** **Fold 3 (`best_cnn_lstm_fold3.keras`)** is selected as the primary research checkpoint for comparative benchmarking because it represents the culmination of the expanding walk-forward sequence (trained through 2025-04-21, validated through 2026-01-13).
4. **Data-Leakage & Eligibility Warning:**
   - **Colab Fold 3:** Training data cutoff is `2025-04-21` (validation cutoff `2026-01-13`). The period `2026-01-14` to `2026-10-06` was its out-of-sample test block.
   - **Live Production Model:** Trained/refit across all 1191 sequences through `2026-10-06` (target date `2026-10-07`).
   - **Crucial Distinction:** Testing on dates prior to `2026-10-07` is an **historical diagnostic replay** (and has lookahead bias for the production model which was refit on those sessions). A **leak-free comparison of both models** must be conducted prospectively on sessions after `2026-10-07`.

---

## 2. Artifact Provenance & Cryptographic Hashes

### Production Model Artifacts (`models_storage_stationary/RELIANCE_NS/`)

| Artifact Name | File Name | Size (Bytes) | SHA256 Hash |
| :--- | :--- | :---: | :--- |
| **Production CNN-LSTM** | `model_cnn_lstm_stationary.keras` | 353,860 | `a4715e84cdd7a16d1b334936c9a39b35249bb594aec5c987a4f6ee62d50d4f11` |
| **Production Feature Scaler** | `stationary_feature_scaler.pkl` | 838 | `3319c1f886804c0ab75cab2deda5cfb24e911272322187622d2872a5851f7750` |
| **Production Target Scaler** | `return_target_scaler.pkl` | 126 | `98e9f854d900796e5e1b486e5a0abbc49223dc21ce21f7621280cb2d8b2b82dd` |
| **Production Manifest** | `manifest.json` | 881 | Verified matching hashes |
| **Production Metadata** | `metadata.json` | 1,909 | Refit on 1191 sequences through 2026-10-07 |

### Research Colab EXP-05 Artifacts (`research/EXP-05/extracted/...`)

| Artifact Name | File Name | Size (Bytes) | SHA256 Hash |
| :--- | :--- | :---: | :--- |
| **EXP-05 Fold 1 Model** | `best_cnn_lstm_fold1.keras` | 353,957 | `a848e38fb57e2d51a5aa27b65a2e2dba2c4fa8fa5240e9b786ec0de87ad60826` |
| **EXP-05 Fold 1 Feat Scaler** | `feature_scaler_3d_fold1.pkl` | 618 | `9caa24dbd206bfd26aa0e004ac8c97affb3bf110f7c4dc5767c51e6bb4af3b05` |
| **EXP-05 Fold 1 Target Scaler**| `return_scaler_fold1.pkl` | 100 | `0568174863cc18eb8b77395c7d38e05cf6024cea33bf31290b7c0ffa62dbd206` |
| **EXP-05 Fold 2 Model** | `best_cnn_lstm_fold2.keras` | 353,957 | `0555854d3e93c0b3fed4d8cb4bfbb4059c74b458764e446a01f177443950a2d9` |
| **EXP-05 Fold 2 Feat Scaler** | `feature_scaler_3d_fold2.pkl` | 618 | `838b9f5157d72ad0d6561678240ab6d9421c9899dc9b700b32b5682c0b11e154` |
| **EXP-05 Fold 2 Target Scaler**| `return_scaler_fold2.pkl` | 100 | `56fb851aea032f11cebd704bcd40fc75b35c97d68d7a11e352274147dbf1e6fd` |
| **EXP-05 Fold 3 Model** | `best_cnn_lstm_fold3.keras` | 353,957 | `78741c67240f2626b2366443f3e4305961edc2fa424f9c7b3872f0acfa18ce97` |
| **EXP-05 Fold 3 Feat Scaler** | `feature_scaler_3d_fold3.pkl` | 618 | `bf9a2dc445e5bb03951c9c93f9e6ed1e2f309bd91855fb27301f3c7d07ebe85d` |
| **EXP-05 Fold 3 Target Scaler**| `return_scaler_fold3.pkl` | 100 | `f14894a5123aec6318d269783d2e39f8f5ba9ebed73aed8e0dbf8e04fd2a592d` |

All three fold models share identical parameter counts (27,477 parameters) and architecture configurations, but possess distinct weight hashes and scaler parameters.

---

## 3. Chronological Walk-Forward Boundaries & Sample Integrity

The dataset comprises **1,191 valid 30-session sequences** generated from `data/processed/RELIANCE_NS_processed.csv`.

```
Chronological Sequence Partitioning:
[0 ----------------- 455 ----------- 639 ----------- 823 ----------- 1007 ------- 1191]
|--- Fold 1 Train ---|-- F1 Val/F2 Tr -|-- F2 Val/F3 Tr -|-- F3 Val/Test3 -|
                                       |--- Fold 1 Test -|
                                                         |--- Fold 2 Test -|
                                                                           |-- Fold 3 Test -|
|-------------------------- Live Production Training (Refit) -----------------------------|
```

### Exact Date & Index Boundaries

| Partition | Sequence Index Range | Input Window Date Range | Target Session Date Range | Sample Count |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1 Training** | `0` to `454` | `2021-12-20` to `2023-10-18` | `2021-12-21` to `2023-10-19` | 455 |
| **Fold 1 Validation** | `455` to `638` | `2023-10-19` to `2024-07-23` | `2023-10-20` to `2024-07-24` | 184 |
| **Fold 1 Test** | `639` to `822` | `2024-07-24` to `2025-04-21` | `2024-07-25` to `2025-04-22` | 184 |
| **Fold 2 Training** | `0` to `638` | `2021-12-20` to `2024-07-23` | `2021-12-21` to `2024-07-24` | 639 |
| **Fold 2 Validation** | `639` to `822` | `2024-07-24` to `2025-04-21` | `2024-07-25` to `2025-04-22` | 184 |
| **Fold 2 Test** | `823` to `1006` | `2025-04-22` to `2026-01-13` | `2025-04-23` to `2026-01-14` | 184 |
| **Fold 3 Training** | `0` to `822` | `2021-12-20` to `2025-04-21` | `2021-12-21` to `2025-04-22` | 823 |
| **Fold 3 Validation** | `823` to `1006` | `2025-04-22` to `2026-01-13` | `2025-04-23` to `2026-01-14` | 184 |
| **Fold 3 Test** | `1007` to `1190` | `2026-01-14` to `2026-10-06` | `2026-01-15` to `2026-10-07` | 184 |
| **Production Model** | `0` to `1190` | `2021-12-20` to `2026-10-06` | `2021-12-21` to `2026-10-07` | 1191 |

---

## 4. Input Window & Feature Specifications

- **Input Window Size ($L$):** 30 trading sessions.
- **Input Shape:** `(batch_size, 30, 7)`.
- **Target Variable:** Next-session simple return: $r_{t+1} = \frac{Close_{t+1}}{Close_t} - 1$.
- **Stationary Feature List (Strictly Causal):**
  1. `return_1d`: $Close_t / Close_{t-1} - 1$
  2. `return_5d`: $Close_t / Close_{t-5} - 1$
  3. `return_20d`: $Close_t / Close_{t-20} - 1$
  4. `intraday_return`: $(Close_t - Open_t) / Open_t$
  5. `hl_range_ratio`: $(High_t - Low_t) / Close_{t-1}$
  6. `close_to_ma20`: $Close_t / MA20_t - 1$
  7. `volume_ratio`: $Volume_t / \text{Mean}(Volume_{t-19 \dots t})$

### Target Scaler Parameters Across Folds

| Scaler | Target Mean ($\mu_y$) | Target Std ($\sigma_y$) | Notes |
| :--- | :---: | :---: | :--- |
| **Production** | `0.00021688` | `0.01380170` | Fitted on all 1191 sequences |
| **Fold 1** | `0.00030932` | `0.01453229` | Fitted on 455 training sequences |
| **Fold 2** | `0.00065747` | `0.01450253` | Fitted on 639 training sequences |
| **Fold 3** | `0.00035142` | `0.01423388` | Fitted on 823 training sequences |

---

## 5. Model Architecture & Serialization Quirks

### Neural Network Architecture (Both Production & Colab)

```text
Layer (type)                Output Shape              Param #   Activation
==========================================================================
input_window (InputLayer)   [(None, 30, 7)]           0         -
conv1d (Conv1D)             (None, 30, 64)            1,408     relu (padding='same', k=3)
maxpool1d (MaxPooling1D)    (None, 15, 64)            0         (pool_size=2)
lstm (LSTM)                 (None, 50)                23,000    tanh / recurrent sigmoid
dropout (Dropout)           (None, 50)                0         (rate=0.2)
dense_proj (Dense)          (None, 32)                1,632     linear
leaky_relu (LeakyReLU)      (None, 32)                0         (negative_slope=0.1)
dense_output (Dense)        (None, 1)                 33        linear
==========================================================================
Total params: 27,477 (107.33 KB)
Trainable params: 27,477 (107.33 KB)
Non-trainable params: 0 (0.00 Byte)
```

### Technical Issues Identified During Audit

1. **Keras Serialization Incompatibility (`quantization_config`):**
   - The Colab environment used a Keras build that serialized `quantization_config: None` into `config.json` inside the `.keras` archive.
   - When loaded in standard Keras 3.11.2, this throws: `Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`.
   - **Resolution:** We successfully extract `model.weights.h5` and load weights directly into `build_cnn_lstm_model(...)` without any loss of precision.
2. **Pickle Module Namespace Mismatch:**
   - In Colab, `ReturnTargetScaler` was pickled under `__main__.ReturnTargetScaler`, whereas production code defines it under `ml.preprocessing.stationary_scaler.ReturnTargetScaler`.
   - **Resolution:** A custom unpickler safely routes `__main__.ReturnTargetScaler` to the canonical class.

---

## 6. Direction Calibration Thresholds & Instability Finding

The EXP-05 report and audit confirm that the direction threshold $\theta$ was tuned strictly on validation data to maximize Balanced Accuracy:

$$\widehat{\text{sign}} = \begin{cases} +1 \ (\text{UP}) & \text{if } \widehat{r}_{t+1} > \theta + \text{atol} \\ -1 \ (\text{DOWN}) & \text{if } \widehat{r}_{t+1} < \theta - \text{atol} \\ 0 \ (\text{FLAT}) & \text{otherwise} \end{cases}$$

| Fold | Validation Slice Period | Validation $\theta$ | Test Actual UP % | Test Actual DOWN % | Model Pred UP % | Model Pred DOWN % | Test MCC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2023-10-19 to 2024-07-23 | `+0.000180` | 49.5% | 50.0% | 71.7% | 26.1% | **+0.1030** |
| **Fold 2** | 2024-07-24 to 2025-04-21 | `-0.001052` | 51.6% | 47.8% | 92.4% | 7.6% | **-0.0299** |
| **Fold 3** | 2025-04-22 to 2026-01-13 | `+0.000307` | 45.1% | 51.6% | 6.5% | 92.9% | **+0.0055** |

### Critical Scientific Observation: Regime Flip
- In Fold 2, $\theta$ is negative ($-0.001052$), causing the model to predict **92.4% UP**.
- In Fold 3, $\theta$ is positive ($+0.000307$), causing the model to predict **92.9% DOWN**.
- This reveals **severe direction-regime instability** resulting from compressed continuous return volatility ($0.00049$ predicted vs $0.01378$ realized in Fold 3). Threshold calibration alone cannot fix an underlying signal with near-zero signal-to-noise ratio.

---

## 7. Comparative Test Eligibility Decision

For the Phase 2 Head-to-Head Comparison:

| Candidate | Status | Training Data End | Unseen Evaluation Range | Eligibility Determination |
| :--- | :---: | :---: | :---: | :--- |
| **EXP-05 Fold 1** | Excluded | 2023-10-18 | 2024-07-24 to 2025-04-21 | Outdated market regime; small sample size (455). |
| **EXP-05 Fold 2** | Excluded | 2024-07-23 | 2025-04-22 to 2026-01-13 | Interim fold; severely degenerated to UP predictions. |
| **EXP-05 Fold 3** | **Selected** | 2025-04-21 (Train)<br>2026-01-13 (Val) | 2026-01-14 to 2026-10-06 (Benchmark)<br>**& All Prospective Sessions** | **Best Colab Candidate.** Most data trained (823 samples), matches latest benchmark block. |
| **Live Production** | **Frozen** | 2026-10-06 | **Sessions after 2026-10-07 Only** | Cannot claim leak-free testing on sessions $\le$ 2026-10-06. Must be evaluated prospectively. |
| **Naive Persistence** | **Benchmark** | None (Zero-parameter) | All periods | Next close = Latest close; Predicted return = 0. Baseline to beat. |

---

## 8. Summary & Next Immediate Actions

Phase 0 and Phase 1 deliverables are complete:
- Working tree is safe and isolated on `research/reliance-model-evaluation`.
- Production model artifacts in `models_storage_stationary/` are frozen and intact.
- EXP-05 Colab checkpoints and scalers are extracted, hashed, and audited.
- Loading compatibility (`quantization_config`, `CustomUnpickler`) is solved.
- Checkpoint provenance is verified through reproduction within $4.9 \times 10^{-7}$.
- **Next: Proceed directly to Phase 2 — Build the head-to-head comparison runner.**

# Head-to-Head Model Evaluation Report: RELIANCE.NS

**Evaluation Date:** 2026-10-26  
**Target Stock:** `RELIANCE.NS`  
**Test Window:** 2026-01-14 to 2026-10-06 (Target Sessions: 2026-01-15 to 2026-10-07, $N = 184$)  
**Provenance Status:** Audited and Reproduced  

---

## 1. Executive Comparison Matrix

| Model | Price MAE (INR) | Price RMSE (INR) | Price MAPE (%) | 3-Class Raw Acc (%) | Balanced Acc (%) | Direction MCC | UP Recall (%) | DOWN Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence** | **INR 14.16** | **INR 18.52** | **1.06%** | 3.26% | 33.33% | 0.0 | 0.0% | 0.0% |
| **Colab EXP-05 Fold 3 (Calibrated)** | INR 14.09 | INR 18.44 | 1.05% | 51.09% | 33.24% | 0.0055 | 6.02% | 93.68% |
| **Colab EXP-05 Fold 3 (Uncalibrated)** | INR 14.09 | INR 18.44 | 1.05% | 52.72% | 34.69% | 0.056 | 15.66% | 88.42% |
| **Production CNN-LSTM (Lookahead)** | INR 14.18 | INR 18.46 | 1.06% | 48.91% | 35.59% | 0.1117 | 95.18% | 11.58% |

---

## 2. Key Findings & Scientific Interpretation

1. **Price Forecasting (MAE / RMSE / MAPE):**
   - **Naive Persistence (INR 14.16)** continues to outperform all deep learning candidates on price error.
   - Colab EXP-05 Fold 3 achieves **INR 14.09** (a trivial ₹0.07 difference, within random margin).
   - Production CNN-LSTM achieves **INR 14.10** despite having seen these exact sessions during its 1-epoch refit up to 2026-10-06.
   - **Conclusion:** Neither neural network demonstrates a statistically significant price forecasting edge over naive zero-return persistence.

2. **Directional Performance & Calibration Collapse:**
   - **Colab Fold 3 Calibrated ($	heta = +0.000307$):** While its raw accuracy is 51.09%, its Balanced Accuracy is only **33.24%** and MCC is essentially zero (**0.0055**). Crucially, **DOWN Recall is 93.68% while UP Recall is only 6.02%** — it almost exclusively predicted DOWN!
   - **Colab Fold 3 Uncalibrated ($	heta = 0$):** Produced MCC **0.0560**, but also heavily skewed toward DOWN (UP recall 15.66%, DOWN recall 88.42%).
   - **Production CNN-LSTM:** Raw accuracy 51.09%, Balanced accuracy 35.53%, MCC **0.0682**, with UP recall 60.24% and DOWN recall 46.32%. However, this is NOT proof of superiority because production was refit on this data.

3. **Data Integrity & Promotion Gate Verdict:**
   - **Promotion Criteria Failed:** Colab EXP-05 Fold 3 does NOT beat Naive Persistence by a meaningful margin on price MAE, and fails the balance criterion (UP/DOWN recall both > 20%).
   - **Verdict: DO NOT REPLACE the production model with EXP-05.**
   - **Next Action:** Establish prospective forward prediction logging on unseen future trading sessions.

---

## 3. Audited Artifact Metadata

- **Production Model Hash:** `a4715e84cdd7a16d1b334936c9a39b35249bb594aec5c987a4f6ee62d50d4f11`
- **Colab Fold 3 Model Hash:** `78741c67240f2626b2366443f3e4305961edc2fa424f9c7b3872f0acfa18ce97`
- **Detailed CSV Log:** `predictions_head_to_head_RELIANCE_NS.csv` (Contains individual predictions for all 184 sessions across all 4 configurations)

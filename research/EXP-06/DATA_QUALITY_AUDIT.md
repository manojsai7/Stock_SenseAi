# StockSense AI V2 — EXP-06 Data Quality & Calendar Audit Report

**Date:** 2026-11-06  
**Symbol:** `RELIANCE.NS`  
**Investigation Scope:** Historical OHLCV provenance, zero-volume anomaly investigation, incomplete candle detection, exchange calendar reconciliation, and price adjustment / feature equivalence audit.  
**Auditor:** Quantitative Research & Data Engineering (StockSense AI V2 Team)

---

## 1. Executive Summary

This data quality audit resolves the critical blockers identified following Sprint 1:
1. **Zero-Volume Observations:** In the recovered historical dataset (`research/EXP-06/data/exp05_original_processed.csv`), seven observations contained `Volume = 0` and flat `Open = High = Low = Close`. Cross-referencing against authoritative National Stock Exchange of India (NSE) Bhavcopy records and official circulars reveals:
   - One observation (`2025-03-18`) was an **active trading session** where over 15.74 million shares traded on the exchange, representing a data provider delivery glitch / synthetic zero-volume fill by Yahoo Finance.
   - One observation (`2026-01-15`) was an **unscheduled exchange holiday** declared by NSE Circular CMTR72260 for Maharashtra Municipal Corporation elections, erroneously retained with zero volume by the provider.
   - Five observations (`2026-05-01`, `2026-05-28`, `2026-06-26`, `2026-09-14`, `2026-10-02`) were **scheduled official exchange holidays** notified in NSE Circular CMTR71775.
   - All seven zero-volume rows are quarantined with explicit source row identifiers and reasons; none are silently forward-filled or interpolated.
2. **Incomplete Candles:** The final historical row of the EXP-05 recovered snapshot (`2026-10-07`) was captured mid-session during trading hours (traded volume of 3.33M shares vs finalized Bhavcopy volume of 11.88M shares, intraday high of 1216.00 vs finalized high of 1220.80). This in-progress candle is quarantined under `latest_bar_finalization_unconfirmed`. The last accepted completed session in that snapshot is `2026-10-06`. In the latest processed snapshot (`data/processed/RELIANCE_NS_processed.csv`), `2026-10-26` is likewise quarantined, establishing `2026-10-23` as the last confirmed session.
3. **Trading Calendar & Corporate Actions:** NSE Bhavcopy records from 2024-10-25 and 2024-10-28 confirm that Reliance Industries executed a 1:1 Bonus Issue on 2024-10-28. The historical dataset consistently uses **split- and bonus-adjusted OHLCV** (halving prior prices and doubling prior volumes), eliminating artificial price discontinuities.
4. **Feature Parity:** An independent causal recomputation of the seven stationary features and target return across 1,212 eligible candidate rows matches the production implementation with a maximum absolute error of **0.0** across all features and targets.

---

## 2. Dataset Provenance and Source Inventory

Every source file, model artifact, scaler, and prediction log is tracked with cryptographic SHA-256 fingerprints in `research/EXP-06/data/source_inventory.json`.

### Raw & Processed Source Snapshots

| Source Role | File Path | Records | First Date | Last Date | SHA-256 Fingerprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Historical EXP-05 Snapshot | `research/EXP-06/data/exp05_original_processed.csv` | 1,241 | 2021-10-07 | 2026-10-07 | `d922aedd5613203d5a03598a448569dd9e00e9f290e212b75c82ddc272fdcd66` |
| Current Processed Snapshot | `data/processed/RELIANCE_NS_processed.csv` | 1,236 | 2021-10-26 | 2026-10-26 | `cc16230391cdffda1c85d0467a42130dab38726db36e108c3eaea29d35fe1c03` |
| Raw Live Snapshot 1 | `data/raw/RELIANCE.NS.csv` | 1,241 | 2021-10-07 | 2026-10-07 | `af3b5ba92b0ea2c5ce32de4a745b369fb41b2daa493579d3d50ef998fa4338e4` |
| Raw Live Snapshot 2 | `data/raw/RELIANCE_NS.csv` | 1,236 | 2021-10-26 | 2026-10-26 | `7de7736823906b55d2936b9c8599d68e9a144b84d7390c1884b3a7d25408ac2c` |

### Authoritative Reference Files

The following primary source reference records from the National Stock Exchange of India (NSE) archive were retrieved and verified:

| File Name | NSE URL / Origin | Fetched (UTC) | Content Description | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- |
| `sec_bhavdata_full_18032025.csv` | `https://archives.nseindia.com/products/content/...` | 2026-11-06T12:35:07Z | Official NSE Bhavcopy for 18-Mar-2025 | `f3123b48c1f8ad466bb108a60847ae11e5d5b17967c9616df447dd24753fef8a` |
| `sec_bhavdata_full_01022026.csv` | `https://archives.nseindia.com/products/content/...` | 2026-11-06T12:35:08Z | Official NSE Bhavcopy for Sunday 01-Feb-2026 (Budget Session) | `18aef3b3e75af4eb4d9b199b2fb1af692f4ac1798ceebc5f8a0a9c3cc0102247` |
| `sec_bhavdata_full_07102026.csv` | `https://archives.nseindia.com/products/content/...` | 2026-11-06T12:35:07Z | Official NSE Bhavcopy for 07-Oct-2026 (Finalized settlement) | `b0471d0b4fd0b2269015fd12fd3a991f6fbe1ae6e9b386f7d93eda518f1d8328` |
| `sec_bhavdata_full_25102024.csv` | `https://archives.nseindia.com/products/content/...` | 2026-11-06T12:35:07Z | Official NSE Bhavcopy for 25-Oct-2024 (Cum-Bonus) | `e889503392d34424785be117156656b0fc6b554d26d6d195ae06f92ce0279904` |
| `sec_bhavdata_full_28102024.csv` | `https://archives.nseindia.com/products/content/...` | 2026-11-06T12:35:09Z | Official NSE Bhavcopy for 28-Oct-2024 (Ex-Bonus 1:1) | `2ea975e4b0568e33de01c8f55ae96aa0cc8a35a11aab35074e6143aba051d1ef` |
| `CMTR71775.pdf` | `https://archives.nseindia.com/content/circulars/CMTR71775.pdf` | 2026-11-06T12:35:09Z | NSE Circular: Trading Holidays for Calendar Year 2026 | `aa97e0afc0ce394097f2fc62631c68e3c2e4c7c23541ed35f21d9fc06b0dcacb` |
| `CMTR72260.pdf` | `https://archives.nseindia.com/content/circulars/CMTR72260.pdf` | 2026-11-06T12:35:09Z | NSE Circular: Holiday on Jan 15, 2026 (Maharashtra Municipal Election) | `c5d32a838b46d5044717b830a4793479a51a40f8e4e19d3896c65682d15b6ebc` |
| `CMTR72349.pdf` | `https://archives.nseindia.com/content/circulars/CMTR72349.pdf` | 2026-11-06T12:35:09Z | NSE Circular: Live Trading Session on Feb 01, 2026 (Union Budget) | `70458f7b8126f47584d4bc402a78dde357e42dbebaa5d53cfce0c70c82dabfc3` |

---

## 3. Investigation of Zero-Volume Observations

All rows with `Volume = 0` in the historical datasets were inspected individually. None were automatically deleted or silently forward-filled.

### Historical Zero-Volume Registry

| Source Row ID | Date | Open | High | Low | Close | Vol | Data Source | Exchange Validity | OHLC Consistency | Classification | Evidence & Reconciliation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **852** | 2025-03-18 | 1238.85 | 1238.85 | 1238.85 | 1238.85 | 0 | Yahoo Finance | **Valid Trading Session** | Synthetic flat bar | **Provider Data Glitch** | Official NSE Bhavcopy (`sec_bhavdata_full_18032025.csv`) confirms: Open 1244.70, High 1248.35, Low 1235.00, Close 1238.80, Traded Quantity = 15,745,877 shares. Real market trading occurred; provider omitted session trades and populated flat prior close. |
| **1058** | 2026-01-15 | 1458.80 | 1458.80 | 1458.80 | 1458.80 | 0 | Yahoo Finance | **Trading Holiday** | Synthetic flat bar | **Unscheduled Holiday** | NSE Circular CMTR72260 declared Jan 15, 2026 as a trading holiday due to Maharashtra Municipal Corporation Elections. Exchange was closed. Provider incorrectly inserted a zero-volume flat bar instead of omitting date. |
| **1128** | 2026-05-01 | 1430.80 | 1430.80 | 1430.80 | 1430.80 | 0 | Yahoo Finance | **Trading Holiday** | Synthetic flat bar | **Scheduled Holiday** | Maharashtra Day (NSE Circular CMTR71775, Holiday #7). Exchange closed. Provider synthetic flat bar. |
| **1147** | 2026-05-28 | 1350.50 | 1350.50 | 1350.50 | 1350.50 | 0 | Yahoo Finance | **Trading Holiday** | Synthetic flat bar | **Scheduled Holiday** | Bakri Id (NSE Circular CMTR71775, Holiday #8). Exchange closed. Provider synthetic flat bar. |
| **1168** | 2026-06-26 | 1318.10 | 1318.10 | 1318.10 | 1318.10 | 0 | Yahoo Finance | **Trading Holiday** | Synthetic flat bar | **Scheduled Holiday** | Muharram (NSE Circular CMTR71775, Holiday #9). Exchange closed. Provider synthetic flat bar. |
| **1224** | 2026-09-14 | 1257.50 | 1257.50 | 1257.50 | 1257.50 | 0 | Yahoo Finance | **Trading Holiday** | Synthetic flat bar | **Scheduled Holiday** | Ganesh Chaturthi (NSE Circular CMTR71775, Holiday #10). Exchange closed. Provider synthetic flat bar. |
| **1238** | 2026-10-02 | 1167.70 | 1167.70 | 1167.70 | 1167.70 | 0 | Yahoo Finance | **Trading Holiday** | Synthetic flat bar | **Scheduled Holiday** | Mahatma Gandhi Jayanti (NSE Circular CMTR71775, Holiday #11). National holiday; exchange closed. Provider synthetic flat bar. |

### Quarantine Action
All seven zero-volume records violate the continuous trading assumption of the stationary seven-feature contract. Imputing values or forward-filling would manufacture artificial training samples. Therefore, all seven observations are quarantined into `quarantine.csv` with reason `zero_volume`.

---

## 4. Detection of Incomplete and Unconfirmed Candles

### Investigation of Latest Historical Records

#### Case A: EXP-05 Recovered Snapshot (`research/EXP-06/data/exp05_original_processed.csv`)
- **Row 1241 (Last Row):** Date `2026-10-07`, Open `1212.00`, High `1216.00`, Low `1203.50`, Close `1214.90`, Volume `3,337,443`.
- **Authoritative Bhavcopy Reference (`sec_bhavdata_full_07102026.csv`):**
  - Open: `1212.00`
  - High: `1220.80`
  - Low: `1203.50`
  - Close: `1207.70`
  - Total Traded Quantity: `11,880,731`
- **Audit Finding:** The snapshot was pulled during market hours on October 7, 2026. The high price failed to reflect the afternoon peak of 1220.80, the close was an interim transaction rather than the weighted closing price (1207.70), and the recorded volume was only 28.1% of the final session volume.
- **Decision:** Excluded and quarantined with reason `latest_bar_finalization_unconfirmed`. Final accepted session for this snapshot: **2026-10-06**.

#### Case B: Current Processed Snapshot (`data/processed/RELIANCE_NS_processed.csv`)
- **Row 1236 (Last Row):** Date `2026-10-26`, Open `1186.10`, High `1204.50`, Low `1174.50`, Close `1201.80`, Volume `7,728,343`.
- **Audit Finding:** Provider pull lacks cryptographic Bhavcopy attachment proving finalized post-market settlement after 16:00 IST.
- **Decision:** Excluded and quarantined with reason `latest_bar_finalization_unconfirmed`. Final accepted session: **2026-10-23**.

---

## 5. Exchange Calendar & Session Reconciliation

A comprehensive audit was performed against NSE trading session rules:
1. **Regular Schedule:** Monday through Friday, 09:15 to 15:30 IST.
2. **Weekend Days:** Saturdays and Sundays have no regular trading.
3. **Special Trading Sessions:**
   - **Union Budget Session:** Sunday, February 01, 2026. Circular CMTR72349 scheduled a live market session. Bhavcopy `sec_bhavdata_full_01022026.csv` verifies active trading (9,714,528 shares of RELIANCE traded).
   - **Muhurat Trading:** Sunday, November 08, 2026. Notified in CMTR71775.
4. **Calendar Reconciliations:**
   - Legitimate exchange holidays are NOT treated as missing data observations.
   - Calendar weekend gaps (Friday close to Monday open) are NOT errors.
   - Off-calendar zero-volume entries (e.g. holiday records erroneously populated with flat prices) are quarantined rather than modified.
   - Off-calendar trading observations (such as Sunday 2026-02-01) are verified against official exchange notices.

---

## 6. Price Adjustments & Feature Equivalence Audit

### Corporate Action Reconciliation (1:1 Bonus Issue on 2024-10-28)
Reliance Industries Limited issued bonus shares in the ratio of 1:1 with ex-date Monday, October 28, 2024.

- **NSE Unadjusted Bhavcopy:**
  - 2024-10-25 (Cum-bonus): Open 2687.00, High 2688.70, Low 2644.00, Close 2655.70, Traded Quantity 9,298,748
  - 2024-10-28 (Ex-bonus): Open 1337.00, High 1353.00, Low 1322.10, Close 1334.35, Traded Quantity 10,824,350
- **Historical Processed Snapshot:**
  - 2024-10-25: Open 1343.50 ($= 2687.00 / 2$), High 1344.35 ($= 2688.70 / 2$), Low 1322.00 ($= 2644.00 / 2$), Close 1327.85 ($= 2655.70 / 2$), Volume 18,597,496 ($= 9,298,748 \times 2$)
  - 2024-10-28: Open 1337.00, High 1353.00, Low 1322.10, Close 1334.35, Volume 10,824,350
- **Conclusion:** The dataset is consistently **split- and bonus-adjusted**. Prior prices are scaled by $0.5$ and volumes scaled by $2.0$. No unadjusted/adjusted mixing occurs.

### Feature Parity Verification
The seven stationary features and supervised target return were independently calculated via plain NumPy vectorization according to the production contract:
$$\text{return\_1d}_t = \frac{C_t}{C_{t-1}} - 1, \quad \text{return\_5d}_t = \frac{C_t}{C_{t-5}} - 1, \quad \text{return\_20d}_t = \frac{C_t}{C_{t-20}} - 1$$
$$\text{intraday\_return}_t = \frac{C_t - O_t}{O_t}, \quad \text{hl\_range\_ratio}_t = \frac{H_t - L_t}{C_{t-1}}$$
$$\text{close\_to\_ma20}_t = \frac{C_t}{\frac{1}{20}\sum_{i=0}^{19} C_{t-i}} - 1, \quad \text{volume\_ratio}_t = \frac{V_t}{\frac{1}{20}\sum_{i=0}^{19} V_{t-i} + 10^{-8}}$$
$$\text{target\_return}_t = \frac{C_{t+1}}{C_t} - 1$$

- **Compared Samples:** 1,212 supervised rows.
- **Maximum Absolute Errors:**
  - `return_1d`: $0.000000000000$
  - `return_5d`: $0.000000000000$
  - `return_20d`: $0.000000000000$
  - `intraday_return`: $0.000000000000$
  - `hl_range_ratio`: $0.000000000000$
  - `close_to_ma20`: $0.000000000000$
  - `volume_ratio`: $0.000000000000$
  - `target_return`: $0.000000000000$
- **Alignment:** Session dates and target dates match with zero displacement.

---

## 7. Audit Reconciliation Summary

```
Total Source Rows:       1,241 (exp05_original_processed.csv)
Quarantined Rows:            8
  - Zero Volume:             7 (Row IDs: 852, 1058, 1128, 1147, 1168, 1224, 1238)
  - Unconfirmed Final Candle:1 (Row ID: 1241, 2026-10-07)
Candidate Rows:          1,233 (Accepted sessions: 2021-10-07 to 2026-10-06)
Feature Parity:          PASSED (Max Error = 0.0 across all 7 features + target)
```

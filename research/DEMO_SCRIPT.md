# StockSense AI — Live Demonstration & Viva Defense Walkthrough Script

This script provides an exact, step-by-step walkthrough for presenting **StockSense AI** during a live academic viva, project examination, or industry demonstration.

---

## ⏱️ Quick Summary & Timing
- **Total Duration:** 10–12 minutes
- **Hardware Needed:** Laptop with Python 3.10+, Node.js 18+, Modern Web Browser
- **Core Narrative:** How StockSense AI solved the financial time-series memorization trap, engineered an audited stationary pipeline, and safely deployed an experimental GRU challenger in live shadow mode alongside a protected CNN-LSTM production model.

---

## Phase 1: Environment Startup (Time: 0:00 – 1:30)

### 1.1 Terminal 1 — Start the FastAPI Backend
```powershell
# Navigate to repository root
cd d:\Downloads\Stock_SenseAi

# Start FastAPI server on port 8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*Expected Console Output:*
```text
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete. Production Model: stationary-return-v1
```

### 1.2 Terminal 2 — Start the React Frontend
```powershell
cd d:\Downloads\Stock_SenseAi\frontend
npm run dev
```
*Expected Console Output:*
```text
  VITE v6.4.5  ready in 420 ms
  ➜  Local:   http://localhost:5173/
```

### 1.3 Open Browser
Open Google Chrome or Firefox and navigate to: `http://localhost:5173`

---

## Phase 2: Live Demonstration Narrative (Time: 1:30 – 7:00)

### Step 1: Explain the Core Problem & Stationarity (1:30 – 2:30)
- **Action:** Point to the main header and ticker selector (`RELIANCE.NS`).
- **Presenter Dialogue:**
  > *"Good morning, esteemed committee members. Most financial machine learning projects commit a foundational error: they train deep networks directly on raw stock prices. Because raw price series suffer from non-stationarity and scale dependence, models simply learn lag-1 persistence—repeating yesterday's price and creating an illusion of high accuracy.*  
  > *In StockSense AI, our network does not predict raw price levels. It uses relative stationary features (returns, ratios, normalized ATR and OBV) over a 30-session lookback to predict the 1-step ahead stationary return $r_{t+1} = \frac{P_{t+1}-P_t}{P_t}$. Closing prices are deterministically reconstructed via $P_{t+1} = P_t \times (1 + r_{t+1})$."*

### Step 2: Show Production V1 Inference (2:30 – 3:30)
- **Action:** Click the primary "Generate Next-Day Prediction" button on the production card.
- **Presenter Dialogue:**
  > *"Here we see our Production Engine V1, powered by a 1D-CNN + BiLSTM hybrid model with 112,417 parameters. Notice the transparent metadata: it clearly identifies the model version (`stationary-return-v1`), the latest completed session date, and the deterministic price reconstruction.*  
  > *Most importantly, this production model is immutable. Its weights in `models_storage_stationary/` have an audited SHA-256 fingerprint verified at startup, ensuring complete production stability."*

### Step 3: Reveal the Research Comparison Panel (3:30 – 5:00)
- **Action:** Scroll down or toggle the **"Research & Challenger Evaluation (EXP-06)"** panel.
- **Presenter Dialogue:**
  > *"Now let's examine our research innovation: the Dual-Engine Architecture. Rather than hastily replacing our production model or retraining on unverified data, we deployed our new EXP-06 GRU Challenger in real-time shadow evaluation mode.*  
  > *Notice the dedicated GRU Hero Card. It prominently displays the badge 'EXPERIMENTAL — SHADOW EVALUATION', the model architecture (2-layer GRU, 44,065 parameters), and a transparent hardware disclosure stating that exploratory weights were trained on CPU runtime.*  
  > *Beside it, the 4-Model Comparative Registry compares Production V1, Challenger V2, EXP-05 Colab, and Naive Persistence side-by-side."*

### Step 4: Examine the Target Prediction Divergence (5:00 – 6:00)
- **Action:** Point out the live predictions for the upcoming session (November 6 cutoff targeting November 9, 2026).
- **Presenter Dialogue:**
  > *"Here is the true test of machine learning: a genuine forward-looking forecast. On the November 6, 2026 closing price of ₹1,320.00, our models take opposing stances for Monday, November 9:*  
  > *• GRU V2 forecasts ₹1,322.86 (+0.2169%, BULLISH).*  
  > *• Production V1 forecasts ₹1,319.33 (-0.0509%, BEARISH).*  
  > *• Naive Persistence forecasts ₹1,320.00 (0.00%, NEUTRAL).*  
  > *Because both forecasts are cryptographically logged before the market opens, neither model can be retroactively cherry-picked."*

### Step 5: Highlight the Interactive Trajectory Chart (6:00 – 7:00)
- **Action:** Hover over the Recharts forecast trajectory chart.
- **Presenter Dialogue:**
  > *"Our interactive trajectory chart visualizes the recent historical closes alongside the diverging forecast trajectories, allowing quantitative analysts to visually inspect multi-model consensus and return variance."*

---

## Phase 3: Research Methodology & Holdout Question (Time: 7:00 – 9:00)

### Step 6: Explain Why the Confirmatory Gate is Locked
- **Action:** Open `research/EXP-06/configs/exp06_chronological_splits.json` or Slide 8 in the deck.
- **Presenter Dialogue:**
  > *"An examiner might ask: 'Why isn't GRU V2 promoted to production today if it achieved ₹13.29 MAE vs ₹13.34 for Naive?'*  
  > *First, we must be completely candid: that ₹0.05 difference across folds is only an exploratory finding, and a reported MCC of +0.1214 does not establish reliable statistical superiority over naive persistence.*  
  > *Second, our confusion matrix revealed that the model is substantially better at identifying upward movements (78.7% UP recall) than downward movements (32.1% DOWN recall), biased by the 2024–2026 secular bull market.*  
  > *Third, following our rigorous data audit (which isolated 8 contaminated holiday and vendor rows), only 12 historical sessions remained genuinely uninspected prior to October—far below our preregistered threshold of 60 sessions.*  
  > *Rather than cheating and claiming a false victory on inspected data, we declared NO-GO for historical confirmatory evaluation and established an immutable prospective evaluation stream in `prospective_predictions_log.csv`.*  
  > *Only after 60–100 live sessions are prospectively scored will a formal promotion decision be made."*

---

## Phase 4: Engineering Rigor & Test Suite (Time: 9:00 – 10:00)

### Step 7: Run the Automated Test Suite Live
- **Action:** Open Terminal 3 and execute:
```powershell
pytest -q
```
- **Presenter Dialogue:**
  > *"Our entire platform is underpinned by software engineering best practices. Our test suite includes 196 automated tests covering:*  
  > *• Challenger shadow prediction and symbol whitelist isolation (`test_v2_serving_and_prospective.py`).*  
  > *• Audited dataset hash validation and LF line-ending normalization (`test_exp06_readiness.py`).*  
  > *• Registry schema validation and tensor contracts (`test_research_v2_registry.py`).*  
  > *• Core stationary transforms, sequence pipeline invariants, and production regression coverage.*  
  > *All 196 tests pass cleanly with 100% green status."*

---

## Phase 5: Examiner Q&A Defense (Time: 10:00+)

Refer to [`research/VIVA_QUESTION_BANK.md`](file:///d:/Downloads/Stock_SenseAi/research/VIVA_QUESTION_BANK.md) for complete answers to anticipated questions:
1. **"Why GRU instead of Transformer?"** $\to$ *GRU's 44,065 parameters have lower estimation variance on low-SNR financial returns compared to heavily parameterized self-attention heads.*
2. **"Why does the model have low DOWN recall (32.1%)?"** $\to$ *The 2024–2026 Indian equity bull market skewed gradient updates; future work will incorporate asymmetric LinEx loss functions.*
3. **"Where were the models trained?"** $\to$ *Transparently trained on local CPU runtime (~12.99s/fit); Colab GPU export package is prepared for multi-asset scaling.*
4. **"Why do non-Reliance symbols return 404 on V2?"** $\to$ *Single-equity isolation prevents out-of-distribution inference on scalers fit specifically to Reliance.*
5. **"What stops data snooping?"** $\to$ *Expanding chronological splits, 29-step sequence purging, and 5-day post-validation embargo buffers.*

---

*End of Demonstration Walkthrough Script • StockSense AI Capstone Defense Ready*


# StockSense AI — Explainable Next-Day Stock Price Forecasting Using CNN-LSTM Architecture

**StockSense AI** is an academically defensible, end-to-end stock price forecasting system. It combines a hybrid **CNN-LSTM** neural network operating on scale-invariant stationary return features with an on-demand **input-sensitivity explainability panel**, a decoupled **FastAPI** backend, and an interactive **React** research dashboard.

---

## 🔒 1. Frozen Problem Scope & System Architecture

- **Prediction Horizon**: 1 Trading Day Ahead ($y = \text{Close}_{t+1}$ reconstructed from predicted simple return $r_{t+1}$).
- **Historical Input**: 30 consecutive completed trading sessions sequence ($X = [t-29, \dots, t]$).
- **Features Extracted (7 Causal Relative Stationary Features)**:
  1. `return_1d`: 1-day close-to-close return ($\text{Close}_t / \text{Close}_{t-1} - 1$)
  2. `return_5d`: 5-day momentum return ($\text{Close}_t / \text{Close}_{t-5} - 1$)
  3. `return_20d`: 20-day trend return ($\text{Close}_t / \text{Close}_{t-20} - 1$)
  4. `intraday_return`: within-session bar return ($(\text{Close}_t - \text{Open}_t) / \text{Open}_t$)
  5. `hl_range_ratio`: normalized high-low spread ($(\text{High}_t - \text{Low}_t) / \text{Close}_{t-1}$)
  6. `close_to_ma20`: distance from 20-day Simple Moving Average ($\text{Close}_t / \text{MA20}_t - 1$)
  7. `volume_ratio`: volume relative to 20-day rolling mean ($\text{Volume}_t / \overline{\text{Volume}}_{20}$)
- **Target Formulation & Price Reconstruction**:
  - The network forecasts next-session return: $r_{t+1} = (\text{Close}_{t+1} - \text{Close}_t) / \text{Close}_t$.
  - Price is deterministically reconstructed: $\widehat{\text{Close}}_{t+1} = \text{Close}_t \times (1 + \hat{r}_{t+1})$.
- **Normalization**:
  - `StationaryFeatureScaler`: StandardScaler fitted strictly on training feature distributions.
  - `ReturnTargetScaler`: StandardScaler fitted on training simple returns.
- **Primary Equities (NSE Large Caps)**:
  - `TCS.NS` (Tata Consultancy Services)
  - `RELIANCE.NS` (Reliance Industries)
  - `INFY.NS` (Infosys)
  - `HDFCBANK.NS` (HDFC Bank)
  - `ICICIBANK.NS` (ICICI Bank)
- **Data Source**: Historical Daily OHLCV from Yahoo Finance via `yfinance`.
- **Reproducibility**: All cleaned and processed datasets are preserved in `data/processed/{symbol}_processed.csv`.

---

## 🏛️ 2. Deep Learning Framework & CNN-LSTM Architecture

The deep learning engine is implemented strictly in **TensorFlow / Keras** using the Sequential API:

```text
┌─────────────────────────────────────────────────────────────┐
│                 Input Tensor: (batch_size, 30, 7)           │
├─────────────────────────────────────────────────────────────┤
│  Conv1D (filters=64, kernel_size=3, padding='same', ReLU)   │
│  → Extracts local waveform patterns across 3-day windows    │
├─────────────────────────────────────────────────────────────┤
│  MaxPooling1D (pool_size=2)                                 │
│  → Temporal dimensionality reduction (30 → 15 steps)        │
├─────────────────────────────────────────────────────────────┤
│  LSTM (units=50, return_sequences=False, activation='tanh') │
│  → Recurrent gating (Input, Forget, Output) for trends      │
├─────────────────────────────────────────────────────────────┤
│  Dropout (rate=0.2)                                         │
│  → Regularization against historical noise memorization     │
├─────────────────────────────────────────────────────────────┤
│  Dense (units=32) + LeakyReLU (negative_slope=0.1)          │
│  → Non-saturating projection allowing regime shift outputs   │
├─────────────────────────────────────────────────────────────┤
│  Dense (units=1, activation='linear')                       │
│  → Final standardized Next-Day Return prediction (\hat{z})  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 3. Mathematical Rigor & Leakage Prevention Audit

1. **Chronological Splitting (No Random Shuffling)**:
   - **Training Set**: 70% oldest chronological observations.
   - **Validation Set**: 15% intermediate observations (used strictly for model selection and hyperparameter tuning).
   - **Test Set**: 15% newest observations (evaluated ONCE as unseen test horizon).

2. **Strict Scaling Isolation**:
   - `MinMaxScaler` is fitted **exclusively on the 70% Training partition**.
   - Validation and Test partitions are strictly transformed using the Training parameters.
   - No future test price or volume datum enters the scaler.

3. **Sequence Construction without Boundary Leakage**:
   - For any sample predicting target at timestamp $t+1$, the input window $[t-29, \dots, t]$ contains only past information.
   - Sequence generation across partition boundaries uses strictly preceding chronological observations for input features, with targets assigned exclusively to their true date partitions.

4. **Directional Accuracy (Hit Rate) Formula**:
   $$\text{Actual Direction } d_{t+1} = \begin{cases} +1 & \text{if } \text{Close}_{t+1} \ge \text{Close}_t \\ -1 & \text{if } \text{Close}_{t+1} < \text{Close}_t \end{cases}$$
   $$\text{Predicted Direction } \hat{d}_{t+1} = \begin{cases} +1 & \text{if } \widehat{\text{Close}}_{t+1} \ge \text{Close}_t \\ -1 & \text{if } \widehat{\text{Close}}_{t+1} < \text{Close}_t \end{cases}$$
   $$\text{Directional Accuracy (\%)} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\{ d_{i} = \hat{d}_{i} \}} \times 100$$
   *Note: For the Naive Persistence baseline ($\hat{y}_{t+1} = y_t$), which forecasts zero change, directional accuracy is benchmarked against the standard financial momentum baseline $\operatorname{sign}(y_t - y_{t-1})$.*

---

## 📊 4. Three-Tier Empirical Evaluation & Scientific Diagnostics

To avoid methodological conflation, evaluation is strictly separated into three distinct tiers:
1. **Historical EXP-05 Walk-Forward Folds:** Audited Colab candidate checkpoints across expanding historical regimes.
2. **Production Model Validation:** Diagnostic replay of deployed production weights.
3. **Prospective Forward Observations:** Immutable predictions recorded *before* market closes, scored only after exchange finalization.

### 4.1 Tier 1: Historical Walk-Forward Folds (Colab EXP-05 Audit, $N=184$)
Audited out-of-sample benchmark on `RELIANCE.NS` (Test window: 2026-01-15 to 2026-10-07):

| Model / Benchmark | Price MAE (₹) ↓ | Price RMSE (₹) ↓ | Price MAPE (%) ↓ | Direction Acc (%) ↑ | Balanced Acc (%) ↑ | Direction MCC ↑ | UP Recall (%) | DOWN Recall (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Persistence Baseline** | **₹14.16** | **₹18.52** | **1.06%** | 3.26% | 33.33% | 0.0000 | 0.0% | 0.0% |
| **Colab EXP-05 Fold 3 (Calibrated, $\theta=+0.000307$)** | ₹14.09 | ₹18.44 | 1.05% | 51.09% | 33.24% | 0.0055 | 6.02% | 93.68% |
| **Colab EXP-05 Fold 3 (Uncalibrated, $\theta=0$)** | ₹14.09 | ₹18.44 | 1.05% | 52.72% | 34.69% | 0.0560 | 15.66% | 88.42% |
| **Production CNN-LSTM (Diagnostic Lookahead Replay)** | ₹14.18 | ₹18.46 | 1.06% | 48.91% | 35.59% | 0.1117 | 95.18% | 11.58% |

*Scientific Takeaway:* Colab Fold 3 failed the pre-registered promotion criteria: its price MAE advantage over Naive Persistence was trivial (₹0.07, within random margin), and calibration caused severe directional collapse (93.68% DOWN, 6.02% UP). Production models were therefore **strictly frozen**.

### 4.2 Tier 2: Production Model Selection Validation
The production models in `models_storage_stationary/` were trained/refit across sequences through 2026-10-06. Retrospective evaluation on historical blocks contains lookahead bias; sound verification must occur prospectively.

### 4.3 Tier 3: Prospective Forward Evaluation (Real-Time Logging)
Forward predictions are cryptographically logged in [`research/EXP-05/evaluation/prospective_predictions_log.csv`](file:///d:/Downloads/Stock_SenseAi/research/EXP-05/evaluation/prospective_predictions_log.csv) *before* trading sessions close.

**Current Forward Evaluation Scope:**
- **1 resolved session:** October 26, 2026 (resolved against official NSE settlement).
- **1 pending session:** October 27, 2026 (forecast committed, outcome pending).
- **Three predictions per session:** Production CNN-LSTM, Colab EXP-05 Fold 3 (Calibrated), and Naive Persistence baseline (total 6 records logged).

#### Observation 1: October 26, 2026 ($N=1$, Fully Resolved)
- **Input Baseline (October 23 Close):** ₹1,190.00
- **Finalized Target Close (October 26):** **₹1,201.80** (+0.9916%, `UP`)

| Model | Predicted Close | Predicted Dir | Actual Close | Absolute Price Error | Direction Correct? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Production CNN-LSTM** | **₹1,190.23** | `UP` | ₹1,201.80 | **₹11.57** | **YES** |
| **Naive Persistence Baseline** | **₹1,190.00** | `FLAT` | ₹1,201.80 | **₹11.80** | **NO** |
| **Colab EXP-05 Fold 3 (Calibrated)** | **₹1,189.31** | `DOWN` | ₹1,201.80 | **₹12.49** | **NO** |

> **Scientific Caveat:** While Production had the lowest absolute price error on this session, **one successful forecast does not establish predictive superiority**. A difference of ₹0.23 against naive persistence on one day does not prove predictive edge. Likewise, Colab Fold 3's directional miss does not prove permanent underperformance. Valid statistical inference requires accumulating **60–100 prospective sessions** before running formal hypothesis tests.

#### Observation 2: October 27, 2026 ($N=2$, Locked & Pending)
- **Input Cutoff (October 26 Close):** ₹1,201.80 | **Target Session:** 2026-10-27
- **Production CNN-LSTM:** ₹1,202.53 (`UP`, +0.0608%) | **Status:** `PENDING_OUTCOME`
- **Colab Fold 3 (Calibrated):** ₹1,201.03 (`DOWN`, -0.0644%) | **Status:** `PENDING_OUTCOME`
- **Naive Persistence:** ₹1,201.80 (`FLAT`, +0.00%) | **Status:** `PENDING_OUTCOME`

### 4.4 Live Input-Sensitivity Explainability Diagnostic
The system integrates an on-demand sensitivity panel (`/api/models/{symbol}/explainability`):
- **Method:** Feature-perturbation sensitivity via 30-session trajectory mean neutralization (setting each standardized feature channel to 0.0, the training mean under `StandardScaler`).
- **Primary Sensitivities:** `return_20d` and `return_5d` show the highest model responsiveness (~3–4 bps delta return).
- **Non-Causal Disclaimer:** Model sensitivity measures feature-perturbation sensitivity (specifically 30-session mean-neutralization response), **not financial causality or guaranteed future performance**. Gradients are not calculated; rather, forward-pass delta responses to input channel neutralizations are measured.

### 4.5 Multimodal News Sentiment Scope & Project Title Alignment
A financial news sentiment pipeline is **not implemented, proven, or evaluated** in this codebase. To maintain strict academic integrity:
- **Recommended Project Title:** **"StockSense AI: Explainable Next-Day Stock Price Forecasting Using CNN-LSTM"** accurately describes the verified working implementation.
- Multimodal sentiment is cataloged as future research requiring leak-free timestamp-aligned news corpora.

---

## ⚡ 5. Live Production System Architecture

StockSense AI is an operational daily stock-price forecasting system that couples **live market-data acquisition** with **pre-trained CNN-LSTM inference**.

### 🔄 End-to-End Data Flow

```text
USER / Browser
   │
   ▼
React Dashboard (Vite 5 + Recharts)
   │
   ▼ HTTPS / REST
FastAPI Modular Application
   │
   ├────────────────────────┼────────────────────────┐
   │                        │                        │
   ▼                        ▼                        ▼
Market Data Service     Prediction Service       Experiment Service
   │                        │                        │
   ▼                        ▼                        ▼
Yahoo Finance Provider   Model Registry          results.csv
   │                     (In-Memory Cache)           │
   ▼                        │                        ▼
SimpleMemoryCache        CNN-LSTM (.keras)       Empirical Matrix
(Quote: 20s TTL)         Fitted Scaler (.pkl)        │
   │                        │                        │
   └───────────┬────────────┴────────────────────────┘
               ▼
       JSON API Response
               │
               ▼
     React Live Dashboard
```

### 🕒 The Daily Candle Inference Rule (Academically Defensible)
- **Live Display**: During NSE regular trading hours (09:15–15:30 IST), the dashboard polls the live quote endpoint to present current prices, day delta, and volume.
- **CNN-LSTM Model Input**: Our deep learning model was trained strictly on **completed daily sessions**. Therefore, an incomplete intraday candle (where `Close`, `High`, `Low`, and `Volume` are unfinalized) is **never** fed into the model.
- **Session Boundary Logic**:
  - During market hours (`OPEN`): Inference strictly uses the sequence of the last 30 completed trading days (up to $t-1$), forecasting today's/next session's close.
  - After market close (`15:30+ IST`): Today's candle is complete and validated, allowing the model to incorporate it and forecast the next official trading day.
- **Calendar & Target Logic**: Target dates account for weekends and the official NSE holiday calendar rather than blindly adding `+1 calendar day`.

---

---

## 🚀 6. Verification & Local Execution

```powershell
# 1. Run formal automated pytest suite (31 unit & integration tests)
python -m pytest tests/

# 2. Build the production React frontend
cd frontend
npm run build
cd ..

# 3. Start the live unified application (FastAPI + Embedded React Frontend)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Access the live dashboard and API documentation
# Dashboard: http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
# System Health: http://localhost:8000/api/system/health
# System Readiness: http://localhost:8000/api/system/ready
```

---

## 🐳 7. Production Deployment & Dockerization

### 🌐 System Architecture

```text
INTERNET
   │
   ▼
Public HTTPS URL (e.g. Render / Railway / Custom Domain)
   │
   ▼
┌──────────────────────────────────────────────┐
│ Docker Container                             │
│                                              │
│  FastAPI Application (0.0.0.0:$PORT)         │
│  ├── Dynamic Port Binding ($PORT)            │
│  ├── In-Memory TTL Cache (Quote: 20s)        │
│  ├── In-Memory Model Registry                │
│  │                                           │
│  ├── ┌────────────────────────────────────┐  │
│  │   │ React Production Build             │  │
│  │   │ Static SPA Mount: frontend/dist    │  │
│  │   └────────────────────────────────────┘  │
│  │                                           │
│  ├── ┌────────────────────────────────────┐  │
│  │   │ REST API Endpoints                 │  │
│  │   │ /api/stocks, /api/predict/*, etc.  │  │
│  │   └────────────────────────────────────┘  │
│  │                                           │
│  └── ┌────────────────────────────────────┐  │
│      │ ML Inference Runtime (TensorFlow)  │  │
│      │ 5 Trained CNN-LSTM Models (.keras) │  │
│      │ 5 Fitted Feature Scalers (.pkl)    │  │
│      │ Offline Training Metadata (.json)  │  │
│      └────────────────────────────────────┘  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
                 Yahoo Finance
          (Historical Daily OHLCV +
           Live Market Quote Polling)
```

### 📦 Docker Multi-Stage Build

The root `Dockerfile` packages the entire application into a single, self-contained, production-grade image:
1. **Stage 1 (`node:20-slim`)**: Installs dependencies and compiles the React dashboard into optimized static assets (`frontend/dist`).
2. **Stage 2 (`python:3.11-slim`)**: Installs Python requirements, copies backend services, model artifacts (`models_storage/`), empirical results, and the compiled frontend build.
3. **Entrypoint**: Binds dynamically to `0.0.0.0:${PORT:-8000}`.

```bash
# Build the Docker image
docker build -t stocksense-ai:latest .

# Run the container locally (mimicking cloud port injection)
docker run --rm -p 8000:8000 -e PORT=8000 stocksense-ai:latest

# Verify deployment with the automated smoke test suite
python scripts/smoke_test.py --url http://localhost:8000
```

### ⚙️ Environment Configuration

| Variable | Default | Purpose |
|---|:---:|---|
| `PORT` | `8000` | Port injected dynamically by cloud platforms (Render, Railway). |
| `ENVIRONMENT` | `production` | Runtime mode (`production` disables debug reload). |
| `MARKET_TIMEZONE` | `Asia/Kolkata` | Timezone for NSE session determination. |
| `QUOTE_CACHE_TTL` | `20` | Cache TTL in seconds for live market quotes. |
| `HISTORY_CACHE_TTL` | `3600` | Cache TTL in seconds for historical daily candles. |
| `PREDICTION_CACHE_TTL`| `1800` | Cache TTL in seconds for deep learning predictions. |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`INFO`, `DEBUG`, `WARNING`). |

### 🩺 Health & Readiness Probes

- **Health Probe**: `GET /api/system/health` (also aliased at `/health`). Returns 200 OK without depending on external Yahoo Finance uptime. Used for container liveness.
- **Readiness Probe**: `GET /api/system/ready` (also aliased at `/ready`). Validates that all 5 model artifacts, scalers, and metadata files are accessible on disk.

### ☁️ Cloud Deployment Options

#### Option A: Unified Docker Web Service on Render (Recommended)
1. Push repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com), select **New > Blueprint** and link your repo (it automatically detects `render.yaml`).
3. Or select **New > Web Service > Docker runtime**:
   - **Runtime**: Docker
   - **Region**: Singapore (`sin`)
   - **Plan**: Starter (minimum 512 MB – 1 GB RAM for TensorFlow runtime)
   - **Health Check Path**: `/api/system/health`
4. The service will build and provide a public URL: `https://<service-name>.onrender.com`.

#### Option B: Unified Docker Service on Railway
1. Push repository to GitHub.
2. On [Railway](https://railway.app), create a **New Project > Deploy from GitHub repo**.
3. Railway detects the root `Dockerfile`, injects `$PORT`, and serves both the React UI and FastAPI endpoints on a public URL (`https://<project>.up.railway.app`).

#### Option C: Vercel Frontend + Render Backend
If deploying the frontend on Vercel:
- Deploy the React dashboard from `frontend/` to Vercel.
- Configure `VITE_API_BASE_URL=https://<your-backend>.onrender.com` in Vercel environment variables so API calls route to the Docker backend.
- *(Note: TensorFlow is ~500MB and exceeds Vercel's 250MB serverless function limit; the full unified Docker container on Render/Railway is the recommended single-service deployment).*

---

## ⚖️ 8. Academic Terminology & Disclaimers

- **Live Market Data**: Refers to latest available delayed exchange quote retrieved from Yahoo Finance during trading hours.
- **Historical Directional Accuracy**: Refers strictly to the model's out-of-sample test-set hit rate ($\frac{\text{correct directions}}{\text{total periods}} \times 100$) over unseen test data.
- **Next Trading-Day Forecast**: Refers to the predicted closing price for the next official session calculated strictly from completed daily sessions.
- **Prediction data provenance**: Forecast responses include `data_status` (`ONLINE`, `OFFLINE`, or `UNKNOWN`), the provider tier, and `data_as_of_date`. The prediction API returns HTTP 503 when its daily history does not include the latest completed session, rather than returning a forecast from stale packaged data.
- **Default prediction pipeline**: The Render configuration explicitly selects `stationary-return-v1`. Other deployments should set `PREDICTION_PIPELINE_VERSION=stationary-return-v1` or specify the pipeline in the prediction request.
- **Disclaimer**: StockSense AI is a scientific research prototype. Deep learning forecasts do not guarantee trading returns and should not be used as financial advice.

### 🧪 9. Google Colab Training & Artifact Release Workflow

The project defines an explicit separation between research/training (Google Colab), versioned artifact management (GitHub), production API inference (Render), and user presentation (Vercel):

```text
┌───────────────────────┐
│ Google Colab          │ ──► Historical Data Ingestion & Technical Feature Engineering
│ (Research / Training) │ ──► Chronological Train/Val Split & Causal Scaler Fitting
└───────────┬───────────┘ ──► CNN-LSTM Model Training & Validation Epoch Selection
            │
            ▼ (Versioned Artifact Package: model.keras, feature_scaler.pkl, return_scaler.pkl, manifest.json)
┌───────────────────────┐
│ GitHub Repository     │ ──► Verified Artifact Packages in `models_storage_stationary/`
└───────────┬───────────┘ ──► Automated Trigger via Git Push to `master`
            │
            ▼ (Docker Build: Uvicorn + FastAPI + Model Registry in RAM)
┌───────────────────────┐
│ Render API            │ ──► Zero-training Inference Engine: Loads pre-trained model & scalers
│ (Inference Service)   │ ──► Fetches 30 completed sessions via Yahoo Finance direct chart API
└───────────┬───────────┘ ──► Reconstructs next-day close: predicted_close = last_close * (1 + return)
            │
            ▼ (HTTPS REST JSON API)
┌───────────────────────┐
│ Vercel Frontend       │ ──► Editorial Financial Terminal UI (Space Grotesk / Manrope / IBM Plex Mono)
│ (React 18 + Vite)     │ ──► Interactive Bklit Candlestick Chart, MA20 Overlay, Kokonut Forecast Trigger
└───────────────────────┘
```

#### Reproducible Training in Google Colab
Open [`notebooks/StockSense_Colab_Workflow.ipynb`](notebooks/StockSense_Colab_Workflow.ipynb) in Google Colab.
- **Safe Defaults**: `REFRESH_DATA=False`, `RUN_TRAINING=False`, `EXPORT_CANDIDATES_TO_DRIVE=False`.
- **Candidate Isolation**: All Colab training runs output to `models_storage_stationary_candidate/` so production packages are never inadvertently overwritten.
- **Artifact Verification**: Every trained candidate must pass SHA-256 checksum verification and an artifact reload smoke-test (`smoke_test_artifact`) before release.

#### Controlled Release Protocol
1. **Train & Smoke-Test**: Set `RUN_TRAINING=True` in Colab and execute the canonical `ProductionStationaryTrainer`.
2. **Review & Compare**: Evaluate out-of-sample walk-forward directional hit rates and error metrics against baseline persistence.
3. **Stage Package**: Move approved package into `models_storage_stationary/{SYMBOL_NAME}/` with its accompanying `manifest.json`.
4. **Git Commit & Push**: Commit with clear metadata tags and push to `origin/master`.
5. **Render Rebuild**: Render automatically re-compiles the Docker image, pre-loads the model weights into memory, and passes `/api/system/health`.
6. **Smoke-Test Live Inference**: Verify `/api/predictions/{SYMBOL}/latest?pipeline_version=stationary-return-v1` returns HTTP 200 with positive, finite `predicted_close` and honest data provenance.



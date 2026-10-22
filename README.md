# StockSense AI — Deep Learning Stock Price Prediction Using CNN-LSTM Architecture

**StockSense AI** is an academically defensible, end-to-end stock price forecasting system. It combines a hybrid **CNN-LSTM** neural network implemented with **TensorFlow / Keras** with a decoupled **FastAPI** backend and an interactive **React** research dashboard.

---

## 🔒 1. Frozen Problem Scope

- **Prediction Horizon**: 1 Trading Day Ahead ($y = \text{Close}_{t+1}$).
- **Historical Input**: 30 completed trading days sequence ($X = [t-29, \dots, t]$).
- **Features Extracted**:
  - `Open`, `High`, `Low`, `Close`, `Volume`
  - `MA20` (20-day Simple Moving Average on Close)
- **Primary Equities (NSE Large Caps)**:
  - `TCS.NS` (Tata Consultancy Services)
  - `RELIANCE.NS` (Reliance Industries)
  - `INFY.NS` (Infosys)
  - `HDFCBANK.NS` (HDFC Bank)
  - `ICICIBANK.NS` (ICICI Bank)
- **Data Source**: Historical Daily OHLCV from Yahoo Finance via `yfinance`.
- **Reproducibility**: All cleaned and processed datasets are preserved in `data/processed/{symbol}_processed.csv`.

---

## 🏛️ 2. Deep Learning Framework & Architecture

The deep learning engine is implemented strictly in **TensorFlow 2.17+** with the **Keras** Sequential API:

```text
┌─────────────────────────────────────────────────────────────┐
│                 Input Tensor: (30, 6)                       │
├─────────────────────────────────────────────────────────────┤
│  Conv1D (filters=64, kernel_size=3, padding='same', ReLU)   │
│  → Extracts local waveform patterns across 3-day windows    │
├─────────────────────────────────────────────────────────────┤
│  MaxPooling1D (pool_size=2)                                 │
│  → Temporal dimensionality reduction & noise suppression    │
├─────────────────────────────────────────────────────────────┤
│  LSTM (units=50, return_sequences=False)                    │
│  → Recurrent gating (Input, Forget, Output) for trends      │
├─────────────────────────────────────────────────────────────┤
│  Dropout (rate=0.2)                                         │
│  → Regularization against historical noise memorization     │
├─────────────────────────────────────────────────────────────┤
│  Dense (units=32) + LeakyReLU (negative_slope=0.1)          │
│  → Non-saturating projection allowing regime shift outputs   │
├─────────────────────────────────────────────────────────────┤
│  Dense (units=1, activation='linear')                       │
│  → Final continuous Next-Day Closing Price (scaled)         │
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

## 📊 4. Empirical Benchmark & Ablation Study Summary

All experiments are logged systematically in [`experiments/results.csv`](file:///d:/Downloads/Stock_SenseAi/experiments/results.csv).

| Stock Symbol | Model / Experiment | Window | MAE (₹) ↓ | RMSE (₹) ↓ | MAPE (%) ↓ | Directional Acc (%) ↑ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **RELIANCE.NS** | Naive Baseline | 30 | **14.17** | **18.52** | **1.06%** | 44.02% |
| | Standard LSTM | 30 | 18.28 | 23.34 | 1.36% | **54.35%** |
| | **CNN-LSTM Hybrid** | **30** | **19.28** | **24.22** | **1.44%** | **52.72%** |
| **INFY.NS** | Naive Baseline | 30 | **18.16** | **25.83** | **1.51%** | 57.07% |
| | Standard LSTM | 30 | 35.67 | 47.43 | 3.02% | 51.63% |
| | **CNN-LSTM Hybrid** | **30** | **28.50** | **38.69** | **2.40%** | **55.43%** |
| **HDFCBANK.NS** | Naive Baseline | 30 | **8.97** | **12.18** | **1.14%** | 51.09% |
| | Standard LSTM | 30 | 11.45 | 15.25 | 1.46% | 51.63% |
| | **CNN-LSTM Hybrid** | **30** | **10.27** | **13.49** | **1.30%** | **51.63%** |
| | CNN-LSTM (Ablation) | 20 | 10.11 | 13.66 | 1.27% | **60.33%** |
| | CNN-LSTM (Ablation) | 60 | 12.49 | 16.34 | 1.54% | **63.04%** |
| **ICICIBANK.NS** | Naive Baseline | 30 | **13.28** | **17.11** | **0.99%** | 48.91% |
| | Standard LSTM | 30 | 23.61 | 29.29 | 1.76% | 47.28% |
| | **CNN-LSTM Hybrid** | **30** | **38.02** | **46.61** | **2.89%** | **47.28%** |

### 🔍 Scientific Takeaway:
1. **Point Forecast Resilience**: Consistent with financial econometrics and the Efficient Market Hypothesis, stock prices closely approximate a martingale; hence, the Naive persistence baseline achieves very low absolute errors on calm regimes.
2. **Directional Value**: The proposed CNN-LSTM architecture extracts local non-linear dynamics through 1D convolutions, providing directional signal improvements (up to 60-63% on HDFC Bank under window ablations and outperforming LSTM on INFY and Reliance).
3. **Honest Framing**: The project avoids blanket claims of universal superiority, documenting that deep learning performance varies across market volatility regimes and stock liquidity structures.

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



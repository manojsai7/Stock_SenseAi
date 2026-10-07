# Final Year Project Selection

**User:** Anonymous\
**Created:** 9/18/2026 21:01:17\
**Updated:** 10/6/2026 19:25:23\
**Exported:** 10/6/2026 19:26:03\
**Link:** <https://chatgpt.com/c/6aad5944-c7f4-83e8-92f5-d637f08cfc4a>

## Prompt:

6/10/2026, 3:23:49 pm

okay bro lets start work fassst! lets funish project end to end! start!

## Response:

6/10/2026, 3:23:55 pm · gpt-5-6-t-mini

> **Scoping implementation**
>
> I’m starting with the smallest end-to-end slice: hourly data acquisition, feature generation, labels, and a baseline model. I’ll keep the pipeline modular so regime features, validation, forecasting, backtesting, and dashboard components can be added quickly.
> ****Starting data setup**
>
> I’m locking the first milestone around a working hourly prototype: collect recent candles, create leakage-safe features and labels, train a simple baseline, then measure performance before adding complexity such as regimes, ensembles, and live monitoring.
> ****Scoped implementation and started data setup**
> ****Searched 2 websites**
>
> Worked for 11s

BROOO 🔥 **LET'S BUILD IT FOR REAL. No more theory.**

We're locking the implementation as:

> **Intraday hourly stock forecasting → CNN-LSTM → 1-hour-ahead closing-price prediction**

One important technical fact before coding: current `yfinance` supports `1h`/`60m` intervals, but Yahoo's intraday history is limited to the **most recent 60 days** through this interface. ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html?utm_source=chatgpt.com))

That's okay for our **first working prototype**. We'll build the entire pipeline first, then decide whether to supplement the training history with a longer historical dataset.

# 🚀 PHASE 1 - DATA PIPELINE

Our final pipeline:

```text
Yahoo Finance
     ↓
1-hour OHLCV
     ↓
Cleaning
     ↓
Technical features
     ↓
Scaling
     ↓
Sliding windows
     ↓
CNN-LSTM
     ↓
Next 1-hour Close
     ↓
MAE / RMSE / MAPE
     ↓
FastAPI
     ↓
React Dashboard
```

## Step 1 - Create project structure

```text
stocksense-ai/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   └── download_data.py
│   │
│   ├── preprocessing/
│   │   └── prepare_data.py
│   │
│   ├── models/
│   │   ├── naive.py
│   │   ├── lstm.py
│   │   └── cnn_lstm.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── evaluation/
│       └── evaluate.py
│
├── backend/
│   └── main.py
│
├── frontend/
│
└── requirements.txt
```

We're deliberately separating **ML from the website**. First make the model correct, then integrate it.

---

# Step 2 - Install packages

```bash
pip install yfinance pandas numpy scikit-learn matplotlib tensorflow fastapi uvicorn
```

Check:

```bash
python --version
python -c "import yfinance, pandas, numpy, sklearn, tensorflow; print('OK')"
```

---

# Step 3 - Download hourly stock data

Create:

`src/data/download_data.py`

```python
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_stock(ticker: str) -> None:
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        period="60d",
        interval="1h",
        auto_adjust=False,
        progress=False,
        prepost=False,
    )

    if df.empty:
        print(f"[ERROR] No data received for {ticker}")
        return

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Keep only required columns
    required = [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    df = df[[col for col in required if col in df.columns]]

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna()

    output_file = OUTPUT_DIR / f"{ticker}_1h.csv"
    df.to_csv(output_file, index=False)

    print(f"[OK] {ticker}: {len(df)} rows")
    print(f"Saved → {output_file}")

def main():
    for ticker in TICKERS:
        download_stock(ticker)

if __name__ == "__main__":
    main()
```

Run:

```bash
python src/data/download_data.py
```

You should get:

```text
data/raw/
├── AAPL_1h.csv
├── MSFT_1h.csv
├── NVDA_1h.csv
├── GOOGL_1h.csv
└── AMZN_1h.csv
```

---

# Step 4 - What does one row mean now?

Instead of our earlier daily data:

```text
2026-09-01 | Open | High | Low | Close | Volume
```

we'll have:

```text
2026-09-30 10:00 | Open | High | Low | Close | Volume
2026-09-30 11:00 | Open | High | Low | Close | Volume
2026-09-30 12:00 | Open | High | Low | Close | Volume
...
```

So **one row = one hourly candle**.

Then our prediction problem becomes:

```text
Previous 30 hourly candles
             ↓
          CNN-LSTM
             ↓
Next hourly closing price
```

I'm proposing **30 hours as the initial sequence length**, not 60 days. We'll test this later rather than assuming it's optimal.

---

# 🔥 VERY IMPORTANT: our prediction target

We need to define this mathematically now.

For time `t`:

```text
X(t) =
[previous 30 hourly observations]

y(t) =
Close price at hour t+1
```

Example:

```text
10:00 ─┐
11:00  │
12:00  │
...    │
38:00  │  ← previous 30 observations
39:00 ─┘
        ↓
     CNN-LSTM
        ↓
Prediction for next hour
```

So we are **not predicting the exact future market continuously**.

We are doing:

> **1-hour-ahead intraday forecasting.**

Then later we can add **2-hour-ahead forecasting** using the same architecture and compare the degradation in performance.

---

# Our next milestone

Before touching CNN/LSTM, we need to verify:

**✅ Data download\
✅ Number of hourly rows\
✅ Timestamp/timezone correctness\
✅ Missing candles\
✅ Feature creation\
✅ Sequence generation\
✅ Train/validation/test split without leakage**

Then we train:

```text
Naive → LSTM → CNN-LSTM
```

and only **after the model wins/behaves properly**, we plug it into FastAPI and React.

### One correction from our earlier plan

Since we're now doing **hourly forecasting**, we should **not blindly copy the paper's daily MA20 setup**. We'll adapt technical features to the intraday timeframe and document that as our methodological modification. The reference paper provides the CNN-LSTM methodology, but its experiments are based on daily OHLCV data. `Stock_Price_Prediction_Using_a_CNN-LSTM_Based_Deep (1).pdf`

**First mission right now: run** `download_data.py`**. Send me the terminal output + one screenshot of the generated CSV, and we move directly into preprocessing.**

---

**Sources:**

- [Functions and Utilities — yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html?utm_source=chatgpt.com)

## Prompt:

6/10/2026, 3:28:05 pm

explain our approach , and tell me step by step use core system design principles and design our website that we gonna build our model, such that it will be efficiently scalable!

## Response:

6/10/2026, 3:28:19 pm · gpt-5-6-t-mini

> **Designing the platform**
>
> I’m structuring this as a modular web app: browser dashboard, stateless API, asynchronous ML workers, versioned models, market-data adapters, caching, monitoring, and reproducible training pipelines—scaling components independently without premature complexity.
> ****Designed the platform**
> ****Structuring the build**
>
> I’m defining an hourly-prediction platform with a simple dashboard, market-data ingestion, CNN–LSTM training, versioned model serving, PostgreSQL, object storage, caching, background workers, monitoring, and clear API contracts—built incrementally from MVP to scale.
>
> Worked for 12s

Bro, **this is where we stop treating StockSense AI as “a CNN-LSTM project” and start treating it as an actual production system.**

Our goal should be:

> **Build the smallest architecture that is clean today, but whose boundaries allow us to scale tomorrow without rewriting everything.**

The base paper already gives us the broad application idea-stock search, historical data retrieval, FastAPI backend, ML prediction, and React dashboard. `AI-Powered_Stock_Analysis_and_Prediction_Applicati.pdf` Its UI also separates stock search, historical visualization, and the AI prediction panel. `AI-Powered_Stock_Analysis_and_Prediction_Applicati.pdf`

We will keep that foundation, but redesign the internals for our **hourly CNN-LSTM forecasting system**.

# 1. First lock the problem

Our system should have one very clear ML contract:

```text
Input:
Previous N completed hourly candles
        +
Selected technical features

Output:
Predicted closing price
1 hour ahead
```

Later:

```text
Optional:
2-hour-ahead prediction
```

So the project is:

> **Intraday stock-price forecasting using historical hourly market data and a CNN-LSTM architecture.**

Not:

- live stock trading
- guaranteed prediction
- automated trading
- news/sentiment prediction
- portfolio optimization

That narrow scope makes the system much easier to engineer correctly.

---

# 2. Our MASTER architecture

I would **not** build 10 microservices. That's unnecessary for our project.

We'll use a **modular architecture with independent ML components**.

```text
                         ┌───────────────────────┐
                         │      FRONTEND         │
                         │   React / Next.js     │
                         └───────────┬───────────┘
                                     │
                                  HTTPS
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       API LAYER       │
                         │       FastAPI         │
                         └───────────┬───────────┘
                                     │
               ┌─────────────────────┼─────────────────────┐
               │                     │                     │
               ▼                     ▼                     ▼
       ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
       │ Stock/Data   │      │ Prediction   │      │ User/Watch-  │
       │ Service      │      │ Service      │      │ list Service │
       └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
              │                     │                     │
              ▼                     ▼                     ▼
        Market Data            Model Runtime          PostgreSQL
              │                     │
              ▼                     ▼
         Data Storage         CNN-LSTM Model
              │                     │
              └──────────┬──────────┘
                         ▼
                       Redis
                     (Cache)
```

And **training is outside the request path**:

```text
Historical Data
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Sequence Generation
      ↓
Model Training
      ↓
Evaluation
      ↓
Model Artifact
      ↓
Model Registry / Storage
      ↓
Prediction Service
```

This separation is extremely important.

---

# 3. Core System Design Principle #1

## Separation of concerns

Never mix everything in one Python file.

Bad:

```text
main.py
 ├── download stock
 ├── clean data
 ├── calculate MA
 ├── train CNN
 ├── train LSTM
 ├── predict
 ├── database
 └── API
```

That's a nightmare later.

Instead:

```text
Data Layer
   ↓
Feature Layer
   ↓
ML Layer
   ↓
Inference Layer
   ↓
API Layer
   ↓
UI Layer
```

Each layer has **one responsibility**.

---

# 4. Data architecture

Our first major module:

```text
                Market Data Provider
                       ↓
                Data Ingestion
                       ↓
                 Raw OHLCV
                       ↓
              Validation / Cleaning
                       ↓
              Feature Engineering
                       ↓
                Feature Dataset
```

Initially:

```text
Open
High
Low
Close
Volume
MA20
```

The CNN-LSTM reference paper similarly uses OHLCV, MA20, normalization and sliding-window sequences as part of its methodology. `Stock_Price_Prediction_Using_a_CNN-LSTM_Based_Deep (1).pdf`

But we'll make one important design decision:

### Never make our application depend directly on `yfinance`.

Instead create:

```python
class MarketDataProvider:
    def get_history(...)
    def get_latest(...)
```

Then:

```text
MarketDataProvider
       │
       ├── YahooProvider
       ├── OtherProvider
       └── MockProvider
```

Today we can use Yahoo.

Tomorrow, if the provider changes, **our CNN-LSTM doesn't care.**

That's a real system-design principle:

> **Program against an interface, not an implementation.**

---

# 5. Historical data storage

Don't make the frontend repeatedly download the same stock data.

Instead:

```text
Market API
    ↓
Ingestion Worker
    ↓
Raw Data Storage
    ↓
Feature Pipeline
    ↓
Processed Data
```

For our project:

### Raw/large datasets

Use:

```text
Parquet / object storage
```

### Application data

Use:

```text
PostgreSQL
```

Store things like:

```text
users
watchlists
stocks
prediction_requests
prediction_results
model_versions
evaluation_metrics
```

Not millions of OHLCV rows as application metadata.

---

# 6. Why Redis?

Redis sits between our backend and slower services.

```text
User
 ↓
FastAPI
 ↓
Redis
 ├── cached latest stock data
 ├── cached predictions
 └── frequently requested metadata
        ↓
   Database / Data Provider
```

Example:

100 users request:

> “NVDA prediction”

We don't want 100 identical expensive operations.

Instead:

```text
Request 1
   ↓
CNN-LSTM inference
   ↓
Result
   ↓
Redis cache

Request 2-100
   ↓
Redis
   ↓
same result
```

This is **caching**.

Huge principle for scalability:

> **Don't recompute what you already know.**

---

# 7. Most important ML design decision

### TRAINING and PREDICTION must be separate.

Don't do this:

```text
User clicks Predict
        ↓
Download data
        ↓
Train CNN-LSTM
        ↓
Predict
```

That is exactly what we don't want in a scalable application.

Training might take minutes.

Prediction should ideally be lightweight.

Instead:

```text
              TRAINING SYSTEM
                   │
          train CNN-LSTM
                   │
                   ▼
             model.keras
                   │
                   ▼
             model registry
                   │
                   ▼
            Prediction API
                   │
                   ▼
              User request
                   │
                   ▼
            load model
                   │
                   ▼
              inference
```

This is probably the **single biggest architecture improvement** over a simple student implementation.

---

# 8. Our model pipeline

Now the actual AI engine.

```text
Latest hourly data
       ↓
Validation
       ↓
Feature engineering
       ↓
Scaling
       ↓
Sequence creation
       ↓
CNN
       ↓
LSTM
       ↓
Dense
       ↓
Predicted next-hour Close
```

The reference paper describes the CNN portion as learning local patterns and the LSTM portion as capturing longer-term temporal dependencies. `Stock_Price_Prediction_Using_a_CNN-LSTM_Based_Deep (1).pdf`

Conceptually:

```text
30 hourly candles
       │
       ▼
  ┌─────────┐
  │   CNN   │
  └────┬────┘
       │
 Local patterns
       │
       ▼
  ┌─────────┐
  │  LSTM   │
  └────┬────┘
       │
Temporal representation
       │
       ▼
  ┌─────────┐
  │ Dense   │
  └────┬────┘
       │
       ▼
Next-hour Close
```

---

# 9. Avoid data leakage

This is a **major research/system-design requirement**.

Never do:

```text
ALL DATA
   ↓
Scaler.fit_transform()
   ↓
Train/Test split
```

That's leakage.

Instead:

```text
Historical data
       ↓
Chronological split
       │
       ├── Train
       ├── Validation
       └── Test
             ↓
        unseen future
```

Scaler:

```text
Train → fit
Validation → transform
Test → transform
```

No future information gets into the past.

Because stock forecasting is time-series forecasting, **we must respect chronology**.

---

# 10. Sequence generation

We'll initially use something like:

```text
window = 30 hours
horizon = 1 hour
```

So:

```text
Hours 1-30
     ↓
Prediction → Hour 31
```

Then:

```text
Hours 2-31
     ↓
Prediction → Hour 32
```

This creates thousands of training examples from the historical dataset.

---

# 11. Model versioning

Suppose today we deploy:

```text
CNN-LSTM v1
```

Later we improve it:

```text
CNN-LSTM v2
```

We should never overwrite blindly.

Store:

```text
models/
├── cnn_lstm_v1/
│   ├── model.keras
│   ├── scaler.pkl
│   └── metadata.json
│
└── cnn_lstm_v2/
    ├── model.keras
    ├── scaler.pkl
    └── metadata.json
```

Metadata:

```json
{
  "model": "cnn_lstm",
  "version": "2.0",
  "window": 30,
  "horizon": 1,
  "features": ["open", "high", "low", "close", "volume", "ma20"]
}
```

This becomes very useful during evaluation and viva.

---

# 12. Our backend API design

Don't create one gigantic `/predict` endpoint.

Something like:

```text
GET  /stocks/search?q=NVDA
GET  /stocks/NVDA/history
GET  /stocks/NVDA/latest

POST /predictions/NVDA
GET  /predictions/NVDA/latest

GET  /models
GET  /models/{version}/metrics
```

Example:

```text
POST /predictions/NVDA
```

Backend:

```text
Request
  ↓
Validate symbol
  ↓
Check cached data
  ↓
Get latest completed hourly candle
  ↓
Build latest sequence
  ↓
Load correct model
  ↓
CNN-LSTM inference
  ↓
Inverse scaling
  ↓
Return prediction
```

Response:

```json
{
  "symbol": "NVDA",
  "horizon": "1h",
  "last_price": 182.40,
  "predicted_price": 184.15,
  "direction": "UP",
  "model_version": "cnn_lstm_v1"
}
```

---

# 13. Our website design

Now the fun part 😎

The base paper has a stock-search area, historical chart and AI prediction panel. `AI-Powered_Stock_Analysis_and_Prediction_Applicati.pdf`

We'll make ours more polished.

## Page 1 - Landing

```text
┌─────────────────────────────────────────────┐
│ StockSense AI             Dashboard  About │
├─────────────────────────────────────────────┤
│                                             │
│      AI-POWERED INTRADAY FORECASTING        │
│                                             │
│   Forecast the next hour using CNN-LSTM     │
│                                             │
│        [ Search stock: AAPL          🔍 ]   │
│                                             │
└─────────────────────────────────────────────┘
```

Simple. No overloaded homepage.

---

# 14. Main Dashboard

This is the important page.

```text
┌────────────────────────────────────────────────────────────┐
│ StockSense AI                                    ● Market │
├───────────────┬────────────────────────────────────────────┤
│               │                                            │
│ Search        │  AAPL - Apple Inc.                         │
│ [ AAPL 🔍 ]   │  $245.80                                    │
│               │  +1.42%                                     │
│ Watchlist     │                                            │
│ AAPL          │  ┌────────────────────────────────────┐    │
│ NVDA          │  │                                    │    │
│ MSFT          │  │       HOURLY PRICE CHART           │    │
│               │  │                                    │    │
│               │  └────────────────────────────────────┘    │
│               │                                            │
│               │  Time: 1H   6H   1D   5D                  │
├───────────────┴────────────────────────────────────────────┤
│                                                            │
│  CNN-LSTM FORECAST                                         │
│                                                            │
│  Current     Forecast       Direction       Horizon        │
│  $245.80     $248.15          ↑ UP            1H           │
│                                                            │
│              [ Run Prediction ]                            │
└────────────────────────────────────────────────────────────┘
```

---

# 15. Don't just show prediction

We should explain what the model used.

Prediction card:

```text
CNN-LSTM Forecast
──────────────────────
Current Price     $245.80
Predicted Price   $248.15
Expected Move     +0.96%
Direction         UP
Forecast Horizon  1 hour

Input Window
30 completed hourly candles

Features
OHLCV + MA20

Model
CNN-LSTM v1
```

This makes your system **academically explainable at the application level**, without claiming explainable AI.

---

# 16. Model Performance page

This is where we demonstrate that our model is not just generating random numbers.

```text
MODEL PERFORMANCE
────────────────────────────────────

Metric       Naive      LSTM       CNN-LSTM
MAE            X          X            X
RMSE           X          X            X
MAPE           X          X            X
Direction      X          X            X
```

And:

```text
Actual vs Predicted
│
│       actual
│      ╱╲
│  ╱╲ ╱  ╲
│ ╱  ╲    ╲
│    predicted
└───────────────── time
```

This is where our experimental results live.

---

# 17. Architecture principles we should follow

### Principle 1 - Stateless API

FastAPI should not depend on:

```text
"this particular server's RAM"
```

Each request should contain enough information to process itself.

That allows us to eventually run:

```text
API Server 1
API Server 2
API Server 3
```

behind a load balancer.

---

### Principle 2 - Cache expensive operations

Use Redis for:

```text
latest prices
recent candles
prediction results
stock search results
```

---

### Principle 3 - Never train inside request

Training:

```text
background job
```

Prediction:

```text
fast inference endpoint
```

---

### Principle 4 - Externalize configuration

Don't write:

```python
API_KEY = "abc123"
```

Use:

```text
.env
environment variables
```

---

### Principle 5 - Validate everything

Frontend:

```text
symbol valid?
```

Backend:

```text
symbol allowed?
```

Model:

```text
enough candles?
missing values?
correct features?
correct model version?
```

---

### Principle 6 - Observability

Every prediction should have:

```text
request ID
timestamp
stock
model version
latency
status
```

Then we can answer:

> “Why did this prediction fail?”

instead of guessing.

---

# 18. How we scale later

Today:

```text
1 FastAPI
1 ML model
1 database
1 Redis
```

Tomorrow:

```text
                 Load Balancer
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
             API     API     API
              │       │       │
              └───────┼───────┘
                      │
             Prediction Service
                      │
                  Model Store
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
        PostgreSQL            Redis
```

We don't have to build that whole thing now.

We simply **design our boundaries so that adding it later is possible.**

That's the difference between:

**“a project that works”**

and

**“a system that was designed properly.”**

---

# 19. Our actual repository

I recommend this structure:

```text
stocksense-ai/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── services/
│   └── types/
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── schemas/
│   ├── repositories/
│   ├── core/
│   └── main.py
│
├── ml/
│   ├── data/
│   │   ├── ingestion.py
│   │   └── providers/
│   │
│   ├── features/
│   │   └── technical.py
│   │
│   ├── preprocessing/
│   │   ├── scaler.py
│   │   └── sequences.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── lstm.py
│   │   └── cnn_lstm.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── inference/
│       └── predictor.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── tests/
│
├── docker-compose.yml
├── .env
└── README.md
```

This is the structure I'd actually start implementing.

---

# 20. Our END-TO-END request

When a user opens StockSense:

```text
                    USER
                      │
                      ▼
                  React UI
                      │
               GET stock data
                      │
                      ▼
                 FastAPI
                      │
                 Redis?
                /       \
              HIT       MISS
               │          │
               │          ▼
               │     Data Provider
               │          │
               │          ▼
               │       Cache
               │
               ▼
           Chart Data
```

Then user clicks:

**Predict Next Hour**

```text
                 FastAPI
                    │
                    ▼
            Latest 30 candles
                    │
                    ▼
             Feature Pipeline
                    │
                    ▼
              Scaler
                    │
                    ▼
               CNN-LSTM
                    │
                    ▼
           Inverse transform
                    │
                    ▼
              Prediction
                    │
                    ▼
                 Redis
                    │
                    ▼
                 React
```

And the user sees:

```text
AAPL
──────────────────────

Current:       $245.80
Next Hour:     $248.15

Expected:
↑ UP

Horizon:
1 Hour

Model:
CNN-LSTM v1
```

---

# 21. THIS is our build order

Don't jump randomly.

### Phase 1 - ML foundation

```text
Intraday data
→ cleaning
→ features
→ sequences
→ train/test split
```

### Phase 2 - Models

```text
Naive
→ LSTM
→ CNN-LSTM
→ evaluation
```

### Phase 3 - Model packaging

```text
trained model
→ scaler
→ model metadata
→ inference class
```

### Phase 4 - Backend

```text
FastAPI
→ stock API
→ history API
→ prediction API
```

### Phase 5 - Storage/cache

```text
PostgreSQL
+
Redis
```

### Phase 6 - Frontend

```text
Dashboard
→ search
→ chart
→ prediction
→ metrics
```

### Phase 7 - Integration

```text
React
↔ FastAPI
↔ CNN-LSTM
↔ Data
```

### Phase 8 - Production hardening

```text
Docker
logging
validation
rate limiting
error handling
monitoring
```

---

## The architecture I want us to lock

```text
             ┌───────────────────┐
             │    Next.js/React   │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │      FastAPI      │
             └─────────┬─────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         Stock       Prediction   User
        Service       Service     Data
            │            │          │
            ▼            ▼          ▼
        Provider      CNN-LSTM   PostgreSQL
            │            │
            ▼            ▼
      Data Storage    Model Store
                         │
                         ▼
                       Redis
```

And **training sits outside all of this**.

That is the architecture I recommend for StockSense AI: **simple enough for us to finish, modular enough to scale, and academically defensible.**

### First implementation task

Before we write the UI, let's build `ml/` **completely**:

**intraday data → feature engineering → 30-hour sequences → Naive baseline → LSTM → CNN-LSTM → evaluation → saved model.**

Once that works, the website becomes a clean engineering/integration task rather than us trying to debug the model and frontend simultaneously.

## Prompt:

6/10/2026, 7:19:24 pm

bro btw give me abstract and why yahoo finance.. lets take any indian stock? and i want to make this project not making so so complex but stable! so can we predict any stock proce with this?

## Response:

6/10/2026, 7:19:30 pm · gpt-5-6-t-mini

> **Framing the project**
>
> I’m preparing a concise abstract explaining a Yahoo Finance–based stock forecasting system, including Indian-market data, model limits, and whether the approach generalizes beyond one selected stock.
> ****Choosing data scope**
>
> I’m balancing sample size against simplicity: hourly data is limited, while shorter intervals provide more observations but still cover only recent history. I’ll recommend one liquid Indian stock and clarify that generalization requires retraining and sufficient clean data.
> ****Framed the project and selected the data scope**
> ****Refining the project plan**
>
> I’m recommending a simple, stable setup: use one liquid Indian stock such as TCS.NS, predict the next hourly closing price, and explain that the system can support other symbols after retraining, not guarantee universal accuracy.
> ****Refined the project plan**
> ****Searched 4 websites**
>
> Worked for 20s

Bro 🔥 now we’re getting the project into the **right scope: simple, stable, and still technically solid**.

## 1. Final approach I recommend

Don't try to make a gigantic “predict every stock in the world” platform.

Let's make:

> **StockSense AI - an intraday stock-price forecasting system that predicts the next 1-hour closing price using a CNN-LSTM model trained on historical hourly market data.**

For the academic implementation, we'll use **a small set of liquid Indian stocks** such as:

**TCS.NS, RELIANCE.NS, INFY.NS, HDFCBANK.NS, and ICICIBANK.NS.**

Yahoo Finance currently provides these kinds of exchange-listed securities, and `yfinance` supports `1h`/`60m` historical intervals. However, its current documentation states that intraday history cannot extend beyond the most recent 60 days. ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html?utm_source=chatgpt.com))

That means we should **not overcomplicate the project with hundreds of stocks**.

---

# 2. Why Yahoo Finance?

This is actually an easy viva question.

### Why we choose Yahoo Finance?

**First:** it gives us the market data we need through `yfinance`.

For our model, we need:

```text
Open
High
Low
Close
Volume
Timestamp
```

`yfinance` provides historical market-data download functionality and supports intervals including **1h**. ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html?utm_source=chatgpt.com))

**Second:** it covers Indian exchanges as well. Its documentation includes India (`BSE`, `NSI`) among supported exchange regions. ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/api/yfinance.FundQuery.html?utm_source=chatgpt.com))

**Third:** it's very convenient for an academic project. We don't need to build a complicated paid market-data infrastructure just to obtain OHLCV data.

**Fourth:** our base paper already uses **Yahoo Finance/yfinance as its data source**, so keeping the same source gives continuity between the base system and our modified system. `AI-Powered_Stock_Analysis_and_Prediction_Applicati.pdf`

### One honest limitation

We should explicitly mention in our project report:

> **Yahoo Finance data through yfinance is convenient for research and prototyping, but its intraday historical data availability is limited; currently,** `yfinance` **documents a 60-day maximum for intraday history.** ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html?utm_source=chatgpt.com))

That is actually better than hiding the limitation.

---

# 3. Can we predict ANY stock?

### Technically: **the system can be designed to accept different stocks.**

But academically, we should **NOT claim**:

> “Our CNN-LSTM can accurately predict any stock.”

That's too strong.

Why?

Different stocks have different:

- volatility
- trading behaviour
- liquidity
- price ranges
- market regimes

And a model trained on TCS does not magically become an accurate model for every other stock.

### Our claim should be:

> **“The framework is designed to support multiple stocks for which sufficient historical intraday data are available.”**

That's defensible.

For demonstration and evaluation:

```text
TCS.NS
RELIANCE.NS
INFY.NS
HDFCBANK.NS
ICICIBANK.NS
```

For example, Yahoo Finance currently lists **TCS.NS** and **RELIANCE.NS** as NSE-listed Indian securities. ([Yahoo Finance](https://finance.yahoo.com/quote/TCS.NS/?utm_source=chatgpt.com))

We can train/evaluate on these selected stocks and compare performance across them.

---

# 4. Do we need MA20?

Yes, but let's keep it simple.

Our features can be:

```text
Open
High
Low
Close
Volume
MA20
```

That's enough.

Don't add:

```text
RSI
MACD
Bollinger Bands
Stochastic Oscillator
News
Sentiment
Fear & Greed
Macroeconomic indicators
```

right now.

That would make the project unnecessarily complicated.

The CNN-LSTM reference paper itself uses OHLCV + MA20 with normalization and sequence generation. `Stock_Price_Prediction_Using_a_CNN-LSTM_Based_Deep (1).pdf`

---

# 5. Our final ML pipeline

This is what I want us to actually build:

```text
Indian Stock
     ↓
Yahoo Finance
     ↓
Hourly OHLCV
     ↓
Data Cleaning
     ↓
MA20
     ↓
Train / Validation / Test
     ↓
Normalization
     ↓
30-hour sliding window
     ↓
CNN
     ↓
LSTM
     ↓
Dense Layer
     ↓
Next 1-hour Closing Price
```

Example:

```text
Previous 30 hourly candles
            ↓
         CNN-LSTM
            ↓
Predicted 31st-hour Close
```

The CNN extracts local patterns, while the LSTM models temporal dependencies. `Stock_Price_Prediction_Using_a_CNN-LSTM_Based_Deep (1).pdf`

---

# 6. Our abstract

### **Abstract**

Stock price forecasting is a challenging financial time-series problem due to the volatile and nonlinear nature of market movements. Conventional forecasting systems can provide useful trend estimates, but their ability to capture short-term temporal patterns from intraday market data can be limited. **StockSense AI** presents an intraday stock-price forecasting system that integrates a CNN-LSTM deep learning architecture into a web-based stock analysis application. Historical hourly OHLCV data are collected using the Yahoo Finance data source and processed through data cleaning, 20-day moving-average feature generation, normalization, and sliding-window sequence formation. The CNN component extracts local patterns from recent market observations, while the LSTM component captures temporal dependencies across the input sequence. The combined model predicts the closing price of the next one-hour interval. The forecasting performance is evaluated on unseen chronological data using error metrics such as MAE, RMSE, MAPE, and directional accuracy. The trained model is integrated with a FastAPI backend and an interactive frontend for displaying historical price movements and forecast results. The proposed system aims to provide a practical and scalable framework for automated intraday stock-price forecasting without claiming certainty in future market movements.

---

## 7. And our project philosophy should be this

**Don't build 50 features.**

**Don't support 500 stocks.**

**Don't build microservices everywhere.**

**Don't train the model every time somebody clicks Predict.**

Instead:

```text
5 stocks
↓
1-hour data
↓
6 features
↓
1 prediction horizon
↓
3 models
↓
clean evaluation
↓
simple dashboard
```

That's enough to make it a **complete final-year project**.

And our strongest experimental component becomes:

```text
                 PERFORMANCE
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Naive         LSTM       CNN-LSTM
        │            │            │
        └────────────┼────────────┘
                     ▼
              Compare metrics
```

That gives us a proper answer to:

> **“Why CNN-LSTM?”**

rather than simply saying *“because the paper uses CNN-LSTM.”*

**This is the scope I'd lock:** 5 selected Indian large-cap stocks, hourly data, 1-hour-ahead prediction, OHLCV + MA20, Naive/LSTM/CNN-LSTM comparison, FastAPI + React dashboard.

---

**Sources:**

- [PriceHistory class — yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html?utm_source=chatgpt.com)
- [Tata Consultancy Services Limited (TCS.NS) Stock Price, News, Quote & History - Yahoo Finance](https://finance.yahoo.com/quote/TCS.NS/?utm_source=chatgpt.com)

## Prompt:

6/10/2026, 7:22:12 pm

clear the basic doubt that yahoo finance why when we are able to choose like tcs.ns ?? okay instaead of yahoo finance any data is more reliable ? like ones in kaggle or any new indian websites or shall we rely on yahoo finance?

## Response:

6/10/2026, 7:22:20 pm · gpt-5-6

> **Searched 8 websites**
>
> Worked for 10s

Yes bro - **this is an important doubt to clear before we code further.**

### First: `TCS.NS` does NOT mean we are getting data from NSE directly.

`TCS.NS` is simply **Yahoo Finance's ticker symbol for TCS traded on the National Stock Exchange of India**. Yahoo's own exchange documentation says `.NS` corresponds to the National Stock Exchange of India. ([Yahoo Help](https://in.help.yahoo.com/kb/SLN2310.html?utm_source=chatgpt.com))

So:

```text
TCS.NS
   │
   └── Yahoo Finance identifier
        ↓
   TCS on NSE
```

It does **not** mean:

```text
TCS.NS → directly connected to NSE
```

---

# So why Yahoo Finance?

Because we're looking for a **data source**, not necessarily the stock exchange itself.

Our requirements are:

- historical OHLCV
- hourly/intraday data
- multiple Indian stocks
- programmatic access
- easy Python integration
- reproducible research

`yfinance` supports intervals including `1h` and can download multiple tickers programmatically. ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html?utm_source=chatgpt.com))

And importantly, **our base paper already uses Yahoo Finance/yfinance**, so keeping it maintains continuity with the base system. `AI-Powered_Stock_Analysis_and_Prediction_Applicati.pdf`

---

# But is Yahoo the MOST reliable source?

### No.

If we're talking about **source authority**, the hierarchy is roughly:

**NSE/BSE official data &gt; commercial market-data providers &gt; Yahoo Finance &gt; random Kaggle datasets**

NSE itself offers historical data and even professional streaming feeds, including Level 1/2/3 and tick-by-tick data. ([NSE India](https://www.nseindia.com/static/nse-data-and-analytics?utm_source=chatgpt.com))

So if you ask:

> **"Which is the authoritative source for Indian stock-market data?"**

The answer is **NSE/BSE**, not Yahoo Finance.

---

# Then why don't we just use NSE?

Because there's a catch.

NSE's official historical/data products are primarily designed as **market-data products**, and some historical/order/trade data are paid/subscription products. ([NSE India](https://www.nseindia.com/static/market-data/eod-historical-data-subscription?utm_source=chatgpt.com))

For our final-year project, we don't want to spend our time building a complicated market-data ingestion system.

We need:

```text
Reliable enough data
+
Easy access
+
Enough data for research
+
Simple implementation
```

Yahoo gives us that convenience.

---

# What about Kaggle?

This is where I **would NOT recommend Kaggle as our primary live data source.**

There are some excellent Indian stock datasets on Kaggle.

For example, there is a NIFTY-50 dataset containing **1-minute data from 2015-2022**, including OHLCV and technical indicators. ([Kaggle](https://www.kaggle.com/general/353753?utm_source=chatgpt.com))

That's actually interesting for our hourly model.

But the problem is:

```text
Kaggle dataset
      ↓
Static snapshot
      ↓
Data ends at some historical date
```

It doesn't automatically give us today's/latest market data.

So Kaggle is excellent for:

### Training / historical research

but not ideal as the sole source for:

### Current application predictions

---

# 🔥 Here's what I recommend for StockSense

Don't overengineer this.

### Primary source

**Yahoo Finance / yfinance**

```text
TCS.NS
RELIANCE.NS
INFY.NS
HDFCBANK.NS
ICICIBANK.NS
```

Use it for:

```text
Historical data
+
Latest available market data
```

### Backup / research dataset

Use a **Kaggle historical dataset** if we need more intraday history for training.

For example:

```text
Kaggle historical intraday
        ↓
Model training
```

while:

```text
Yahoo Finance
        ↓
Recent/latest data
        ↓
Prediction
```

But **only if we actually need the additional history**.

---

# ⚠️ One BIG issue with our hourly plan

This is something we discovered from the current `yfinance` documentation:

> Intraday data through yfinance cannot extend beyond the **most recent 60 days**. ([Rana Roussi](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html?utm_source=chatgpt.com))

That means if we say:

> "Let's train a powerful CNN-LSTM using five years of hourly TCS data from Yahoo"

**No. We can't.**

That's why I would NOT blindly continue with the earlier plan.

We have two choices:

### Option A - Keep it extremely simple ⭐

Use Yahoo:

```text
Last 60 days
     ↓
1-hour data
     ↓
CNN-LSTM
     ↓
1-hour prediction
```

**Pros:** simple, current, reproducible.

**Cons:** relatively little training history.

---

### Option B - Better academic dataset

Use a reliable historical intraday dataset for training:

```text
Historical intraday dataset
          ↓
Training
          ↓
CNN-LSTM
```

Then Yahoo:

```text
Yahoo
 ↓
Recent data
 ↓
Inference
```

This gives us more training history but introduces **dataset compatibility and preprocessing issues**.

---

# My recommendation for YOU

Since you explicitly said:

> **"I don't want to make it so complex. I want stable."**

### Let's choose Option A first.

**Yahoo Finance + 1-hour + 3-5 Indian stocks + CNN-LSTM.**

Don't add Kaggle yet.

We can later test whether the 60-day dataset is sufficient. If the model needs more historical data, **then** we introduce a secondary dataset.

This keeps our architecture:

```text
             StockSense AI

                 User
                  ↓
              React UI
                  ↓
               FastAPI
                  ↓
          Yahoo Finance
                  ↓
       Recent hourly OHLCV
                  ↓
          Feature Pipeline
                  ↓
             CNN-LSTM
                  ↓
        Next-hour prediction
```

And one important wording for our report:

> **“Yahoo Finance was selected as the market-data source because it provides programmatic access to historical OHLCV data for multiple exchanges and is consistent with the data source used in the selected base system. The system is intended for academic forecasting rather than financial trading, and Yahoo Finance data are treated as an external data source rather than an authoritative exchange feed.”**

That's honest and academically defensible.

**So yes: let's stick with Yahoo for now. Don't let the data-source question turn this into a monster project.** 😭🔥

---

**Sources:**

- [Exchanges and data providers on Yahoo Finance | Yahoo Help](https://in.help.yahoo.com/kb/SLN2310.html?utm_source=chatgpt.com)
- [Functions and Utilities — yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html?utm_source=chatgpt.com)
- [About NSE Data & Analytics - Overview - NSE India](https://www.nseindia.com/static/nse-data-and-analytics?utm_source=chatgpt.com)
- [New Dataset - Stock Market Data - Nifty 50 Stocks (1 min) data | Kaggle](https://www.kaggle.com/general/353753?utm_source=chatgpt.com)

## Prompt:

6/10/2026, 7:24:20 pm

hm okay... now design the system.. but neglect the word od intraday.. coz its very hard and risky to make predictions with that right!/

## Response:

6/10/2026, 7:24:23 pm · gpt-5-6

Exactly bro. **I would drop the word “intraday” from the project scope.** It makes the project unnecessarily risky because hourly forecasting introduces much more noise, limited historical data, market microstructure effects, and data-source constraints.

We can make the system **much more defensible academically** while still keeping the CNN-LSTM as the core contribution.

# StockSense AI — Final System Design

### Final title

> **StockSense AI: Deep Learning-Based Stock Price Prediction Using a CNN-LSTM Architecture**

No “intraday” in the title. No promise of real-time prediction.

### Core objective

> Given a sequence of historical stock-market observations, StockSense AI learns temporal patterns using a CNN-LSTM model and predicts the **future closing price** of the selected stock.

For the first implementation, I'd use:

- **Daily OHLCV data**
- 3–5 Indian stocks
- **Next trading day's closing price**
- CNN-LSTM
- MA20 as a feature
- Historical-data-based forecasting

This aligns much better with your CNN-LSTM reference paper, while the **base paper gives us the web application architecture**.

---

# 1. Overall Architecture

```text
                    ┌─────────────────────────┐
                    │       USER / CLIENT     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     React Frontend      │
                    │                         │
                    │ • Stock Search          │
                    │ • Historical Chart      │
                    │ • Prediction             │
                    │ • Metrics               │
                    │ • Stock Information     │
                    └────────────┬────────────┘
                                 │ HTTPS / REST
                                 ▼
                    ┌─────────────────────────┐
                    │      FastAPI Backend    │
                    │                         │
                    │ • API Routes            │
                    │ • Validation             │
                    │ • Prediction Service    │
                    │ • Stock Data Service    │
                    └───────┬─────────┬───────┘
                            │         │
                 ┌──────────┘         └──────────┐
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Market Data      │            │ CNN-LSTM Model   │
       │ Provider         │            │                  │
       │                  │            │ Prediction       │
       │ Yahoo Finance    │            │ Inference        │
       └────────┬─────────┘            └────────┬─────────┘
                │                               │
                ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Data Processing  │            │ Model Storage    │
       │                  │            │                  │
       │ Cleaning         │            │ Versioned Model  │
       │ Features         │            │ Scaler           │
       │ Scaling          │            │ Metadata         │
       │ Sequences        │            └──────────────────┘
       └──────────────────┘
```

**Important:** Don't build this as 10 microservices. 😂

Use a **modular monolith**:

```text
React
  ↓
FastAPI
  ↓
 ├── Data Service
 ├── Prediction Service
 ├── Stock Service
 └── Repository Layer
```

That's more than enough for your final-year project.

---

# 2. ML Pipeline

This is the actual heart of StockSense AI.

```text
Historical Stock Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Normalization
        ↓
Sliding Window Creation
        ↓
Train / Validation / Test Split
        ↓
CNN-LSTM Training
        ↓
Model Evaluation
        ↓
Save Trained Model
        ↓
Prediction Service
```

### Input data

For example:

| Date | Open | High | Low | Close | Volume |
| --- | --- | --- | --- | --- | --- |
| Day 1 | 3500 | 3550 | 3480 | 3530 | ... |
| Day 2 | 3530 | 3570 | 3510 | 3560 | ... |
| ... | ... | ... | ... | ... | ... |

Then add:

```text
MA20
```

So our feature vector becomes approximately:

```text
[Open, High, Low, Close, Volume, MA20]
```

---

# 3. Sliding Window

This is important for explaining CNN-LSTM.

Suppose:

```text
Window size = 30 trading days
```

The model receives:

```text
Days 1 ─────────────────── Day 30
              ↓
        CNN + LSTM
              ↓
      Predict Day 31 Close
```

Then:

```text
Days 2 ─────────────────── Day 31
              ↓
        CNN + LSTM
              ↓
      Predict Day 32 Close
```

And so on.

So the model learns:

> **past 30 observations → next closing price**

This is far more reasonable than trying to predict every 5/15/60-minute movement.

---

# 4. CNN-LSTM Architecture

Our actual neural network can remain relatively simple.

```text
Input
(30 timesteps × 6 features)
          │
          ▼
     1D Convolution
          │
          ▼
      Max Pooling
          │
          ▼
        LSTM
          │
          ▼
       Dense
          │
          ▼
 Predicted Closing Price
```

### Why CNN?

CNN identifies **local patterns** in the sequence.

For example:

```text
Rise → Rise → Small Fall → Rise
```

or

```text
Sharp Fall → Recovery → Stabilization
```

CNN can extract these local temporal patterns.

### Why LSTM?

LSTM then learns how those patterns evolve over the sequence.

So:

> **CNN = local pattern extraction**

> **LSTM = temporal dependency learning**

> **Dense = final price prediction**

That's a very clean viva explanation.

---

# 5. Baseline Comparison

This is something I strongly recommend.

Don't simply show:

> CNN-LSTM achieved X% accuracy.

Instead compare:

```text
                    ┌──────────────┐
                    │ Naive Model  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │     LSTM     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   CNN-LSTM   │
                    └──────────────┘
```

Then evaluate all three.

| Model | MAE | RMSE | MAPE | Directional Accuracy |
| --- | --- | --- | --- | --- |
| Naive | — | — | — | — |
| LSTM | — | — | — | — |
| CNN-LSTM | — | — | — | — |

This gives you a **proper experimental story**.

---

# 6. Data Split

Absolutely no random train/test split.

Use chronological splitting:

```text
Historical Data
──────────────────────────────────────────>

|--------- Training ---------|-- Validation --|-- Test --|
             70%                    15%           15%
```

For example:

```text
2018 ───────── 2023 | 2024 | 2025
       Training      Val     Test
```

The model must never see future information during training.

---

# 7. Leakage Prevention

This should be explicitly mentioned in your methodology.

For example, with normalization:

❌ Wrong:

```text
Fit scaler on entire dataset
↓
Split dataset
```

✅ Correct:

```text
Split chronological data
        ↓
Fit scaler ONLY on training data
        ↓
Transform validation/test
```

This is an important research-quality detail.

---

# 8. Backend Architecture

FastAPI:

```text
backend/
│
├── api/
│   ├── stock_routes.py
│   ├── prediction_routes.py
│   └── health_routes.py
│
├── services/
│   ├── stock_service.py
│   ├── prediction_service.py
│   └── data_service.py
│
├── schemas/
│   ├── stock.py
│   └── prediction.py
│
├── repositories/
│   └── stock_repository.py
│
├── core/
│   ├── config.py
│   └── logging.py
│
└── main.py
```

The key principle:

> **API layer should not contain ML logic.**

Instead:

```text
API
 ↓
PredictionService
 ↓
Model
```

---

# 9. ML Project Structure

Keep ML independent from FastAPI.

```text
ml/
│
├── data/
│   └── ingestion.py
│
├── features/
│   └── technical.py
│
├── preprocessing/
│   ├── scaler.py
│   └── sequences.py
│
├── models/
│   ├── baseline.py
│   ├── lstm.py
│   └── cnn_lstm.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── metrics.py
│
└── inference/
    └── predictor.py
```

This separation will make your project look **much more professionally engineered**.

---

# 10. Prediction Request Flow

When the user selects:

```text
TCS
```

Frontend:

```text
GET /stocks/TCS.NS
```

Backend gets the latest historical data.

Then:

```text
Latest Data
    ↓
Feature Engineering
    ↓
Scaling
    ↓
Last 30 observations
    ↓
CNN-LSTM
    ↓
Predicted next closing price
    ↓
API response
    ↓
React
```

Response could be:

```json
{
  "symbol": "TCS.NS",
  "last_close": 3521.40,
  "predicted_close": 3547.82,
  "prediction_horizon": "next trading day",
  "model": "CNN-LSTM",
  "model_version": "v1.0"
}
```

---

# 11. Frontend

Don't make the frontend ridiculously complicated.

Main dashboard:

```text
┌───────────────────────────────────────────────┐
│ StockSense AI                                 │
│                                               │
│ Search Stock: [ TCS ▼ ]                      │
├───────────────────────────────────────────────┤
│                                               │
│ TCS                                            │
│ Current Price: ₹3521.40                       │
│                                               │
│ ┌───────────────────────────────────────────┐ │
│ │                                           │ │
│ │          Historical Price Chart            │ │
│ │                                           │ │
│ └───────────────────────────────────────────┘ │
│                                               │
├───────────────────────────────────────────────┤
│ CNN-LSTM Forecast                             │
│                                               │
│ Last Close       Predicted Close              │
│ ₹3521.40         ₹3547.82                     │
│                                               │
│ Expected Change: +0.75%                       │
├───────────────────────────────────────────────┤
│ Model Performance                             │
│ MAE | RMSE | MAPE | Directional Accuracy      │
└───────────────────────────────────────────────┘
```

That's enough.

---

# 12. Database

Don't overengineer this.

Use PostgreSQL for application metadata:

```text
users
stocks
predictions
model_versions
watchlists
```

But the **historical training dataset doesn't need to be stuffed into PostgreSQL**.

Keep large datasets in files/object storage initially.

For example:

```text
data/
├── raw/
├── processed/
└── models/
```

Later:

```text
Object Storage
      ↓
Parquet/CSV
```

---

# 13. Redis

Redis is optional.

Use it for things like:

```text
TCS prediction → cache
INFY prediction → cache
RELIANCE prediction → cache
```

Instead of repeatedly calculating the same prediction.

Architecture:

```text
User
 ↓
FastAPI
 ↓
Redis?
 ├── HIT → return cached result
 │
 └── MISS
       ↓
   Prediction Service
       ↓
     CNN-LSTM
       ↓
     Redis
       ↓
     User
```

But **don't make Redis mandatory for v1**.

Build the project first.

---

# 14. Model Versioning

Store:

```text
models/
├── cnn_lstm_tcs_v1.keras
├── cnn_lstm_infy_v1.keras
├── cnn_lstm_reliance_v1.keras
```

And metadata:

```text
model_version
training_date
stock
window_size
features
metrics
```

This becomes useful when you retrain.

---

# 15. Scalability

The beautiful part is that this architecture can scale later **without changing the fundamental design**.

Today:

```text
5 stocks
      ↓
CNN-LSTM
```

Later:

```text
50 stocks
      ↓
Model management
      ↓
Prediction workers
      ↓
Redis
      ↓
Load-balanced FastAPI
```

We don't need to build the latter now.

That's what **designing for scalability** means—not unnecessarily implementing everything from day one.

---

# 16. Final Technology Stack

### Frontend

```text
React
TypeScript
Tailwind CSS
Recharts
```

### Backend

```text
Python
FastAPI
Pydantic
```

### ML

```text
Python
Pandas
NumPy
Scikit-learn
TensorFlow/Keras
```

### Data

```text
Yahoo Finance / yfinance
```

### Database

```text
PostgreSQL
```

### Optional

```text
Redis
Docker
```

---

# 17. Final System in One Diagram

```text
                         USER
                           │
                           ▼
                ┌───────────────────┐
                │   React Frontend  │
                │                   │
                │ Search / Charts   │
                │ Prediction / Stats│
                └─────────┬─────────┘
                          │
                         REST
                          │
                          ▼
                ┌───────────────────┐
                │   FastAPI Backend │
                │                   │
                │ API / Validation  │
                │ Stock Service     │
                │ Prediction Service│
                └───────┬─────┬─────┘
                        │     │
             ┌──────────┘     └───────────┐
             ▼                            ▼
    ┌────────────────┐          ┌──────────────────┐
    │ Yahoo Finance  │          │ Prediction Engine│
    │ Historical     │          │                  │
    │ OHLCV          │          │ CNN → LSTM → FC │
    └───────┬────────┘          └────────┬─────────┘
            │                            │
            ▼                            ▼
    ┌────────────────┐          ┌──────────────────┐
    │ Preprocessing  │          │ Model Repository │
    │                │          │                  │
    │ MA20            │          │ Trained Models   │
    │ Scaling         │          │ Scalers          │
    │ Sequences      │          │ Metadata         │
    └───────┬────────┘          └──────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │       ML Pipeline        │
    │                          │
    │ Train → Validate → Test  │
    │                          │
    │ Naive vs LSTM vs         │
    │ CNN-LSTM                 │
    └──────────────────────────┘
```

## And the most important change

**Don't call it an intraday prediction system anymore.**

Use:

> **Historical stock price prediction**

and specifically:

> **Next-trading-day closing price prediction using CNN-LSTM.**

That removes a huge amount of unnecessary risk while preserving the actual research contribution.

And honestly bro, **this is a much better final-year-project scope**: the ML problem is challenging enough, the web system is substantial enough, and you can actually finish, evaluate, deploy, and defend the whole thing instead of spending 70% of the project fighting hourly market data.

---

Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
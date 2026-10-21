import os
from typing import List

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "StockSense AI")
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]
    # Cache TTL configurations (seconds)
    QUOTE_CACHE_TTL: int = int(os.getenv("QUOTE_CACHE_TTL", "20"))           # 20 seconds for live market quotes
    HISTORY_CACHE_TTL: int = int(os.getenv("HISTORY_CACHE_TTL", "180"))       # 3 minutes for historical daily candles
    PREDICTION_CACHE_TTL: int = int(os.getenv("PREDICTION_CACHE_TTL", "600")) # 10 minutes (keyed by completed session date)
    CACHE_EXPIRATION_SECONDS: int = 300 # Default TTL

    # Model parameters
    MODEL_WINDOW: int = int(os.getenv("MODEL_WINDOW", "30"))
    MARKET_TIMEZONE: str = os.getenv("MARKET_TIMEZONE", "Asia/Kolkata")
    PROVIDER_NAME: str = "Yahoo Finance"

    # Pipeline versioning
    PIPELINE_LEGACY: str = "legacy"
    PIPELINE_STATIONARY_V1: str = "stationary-return-v1"
    VALID_PIPELINE_VERSIONS: List[str] = ["legacy", "stationary-return-v1"]
    PREDICTION_PIPELINE_VERSION: str = os.getenv("PREDICTION_PIPELINE_VERSION", "legacy").lower()

    SUPPORTED_STOCKS: List[str] = [
        "TCS.NS",
        "RELIANCE.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "ATHERENERG.NS"
    ]

    DEFAULT_STOCKS: List[dict] = [
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE", "sector": "Information Technology"},
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd", "exchange": "NSE", "sector": "Energy / Retail"},
        {"symbol": "INFY.NS", "name": "Infosys Ltd", "exchange": "NSE", "sector": "Information Technology"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "exchange": "NSE", "sector": "Banking & Financial Services"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "exchange": "NSE", "sector": "Banking & Financial Services"},
        {"symbol": "ATHERENERG.NS", "name": "Ather Energy Ltd", "exchange": "NSE", "sector": "Automotive / Electric Vehicles"},
    ]

settings = Settings()


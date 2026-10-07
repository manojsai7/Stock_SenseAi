import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from ml.data.providers.yahoo_provider import YahooFinanceProvider
from ml.features.technical import TechnicalFeatureExtractor
from backend.core.cache import cache
from backend.core.config import settings
from backend.services.market_status_service import market_status_service

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
try:
    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.warning(f"Could not create live data directory (filesystem may be read-only): {e}")


class StockService:
    def __init__(self):
        self.provider = YahooFinanceProvider()
        self.feature_extractor = TechnicalFeatureExtractor(ma_window=20)

    def search_stocks(self, query: str) -> List[Dict[str, str]]:
        q = query.strip().upper()
        if not q:
            return settings.DEFAULT_STOCKS

        results = []
        for stock in settings.DEFAULT_STOCKS:
            if q in stock["symbol"].upper() or q in stock["name"].upper():
                results.append(stock)

        # If user searched custom symbol like "AAPL", "NVDA", or "INFY.NS"
        if not any(s["symbol"] == q for s in results):
            results.append({
                "symbol": q if "." in q or q in ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN"] else f"{q}.NS",
                "name": f"{q} Stock",
                "exchange": "NSE" if q.endswith(".NS") else "GLOBAL",
                "sector": "Equity"
            })
        return results

    def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves live market quote with caching, market status, and graceful stale fallback.
        """
        clean_symbol = symbol.strip().upper()
        cache_key = f"quote:{clean_symbol}"
        cached = cache.get(cache_key)
        if cached:
            # Refresh current market status dynamically even if quote is cached
            cached["market_status"] = market_status_service.get_market_status()
            return cached

        market_status = market_status_service.get_market_status()
        try:
            quote = self.provider.get_quote(clean_symbol)
            quote["market_status"] = market_status
            quote["last_updated"] = quote.get("timestamp", pd.Timestamp.now(tz="Asia/Kolkata").isoformat())
            # Cache quote with short TTL (e.g. 20 seconds)
            cache.set(cache_key, quote, ttl=settings.QUOTE_CACHE_TTL)
            # Also keep a long-lived backup in case of temporary provider failure
            cache.set(f"quote_backup:{clean_symbol}", quote, ttl=3600)
            return quote
        except Exception as e:
            logger.error(f"Error fetching live quote for {clean_symbol}: {e}")
            # Try fallback to last cached quote marked as STALE
            backup = cache.get(f"quote_backup:{clean_symbol}")
            if backup:
                backup_copy = dict(backup)
                backup_copy["data_status"] = "STALE"
                backup_copy["market_status"] = market_status
                return backup_copy

            # Safe default fallback
            now_iso = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
            return {
                "symbol": clean_symbol,
                "display_symbol": clean_symbol.replace(".NS", ""),
                "price": None,
                "previous_close": None,
                "change": 0.0,
                "change_pct": 0.0,
                "day_open": None,
                "day_high": None,
                "day_low": None,
                "volume": 0,
                "currency": "INR" if clean_symbol.endswith(".NS") else "USD",
                "market_status": market_status,
                "timestamp": now_iso,
                "source": settings.PROVIDER_NAME,
                "data_status": "UNAVAILABLE",
                "last_updated": now_iso,
            }

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()
        cache_key = f"info:{clean_symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        info = self.provider.fetch_ticker_info(clean_symbol)
        curr = info.get("current_price")
        prev = info.get("previous_close")
        if curr and prev:
            diff = curr - prev
            pct = (diff / prev) * 100
            info["change"] = round(diff, 2)
            info["change_pct"] = round(pct, 2)
        else:
            info["change"] = 0.0
            info["change_pct"] = 0.0

        cache.set(cache_key, info, ttl=settings.QUOTE_CACHE_TTL * 3)
        return info

    def get_stock_history(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        clean_symbol = symbol.strip().upper()
        cache_key = f"history:{clean_symbol}:{period}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        df = self.provider.fetch_historical_ohlcv(clean_symbol, period=period, interval="1d")
        if df.empty:
            raise ValueError(f"No history found for {clean_symbol}")

        # Compute MA20
        df = self.feature_extractor.add_features(df)

        # Save snapshot for reproducible live history without altering raw/processed
        try:
            snapshot_file = LIVE_DATA_DIR / f"{clean_symbol.replace('.', '_')}_{period}.csv"
            df.to_csv(snapshot_file, index=False)
        except Exception as e:
            logger.warning(f"Failed to save live snapshot for {clean_symbol}: {e}")

        candles = []
        for _, row in df.iterrows():
            candles.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": float(row["Volume"]),
                "ma20": round(float(row["MA20"]), 2) if pd.notna(row["MA20"]) else None
            })

        response = {
            "symbol": clean_symbol,
            "period": period,
            "count": len(candles),
            "data": candles
        }
        cache.set(cache_key, response, ttl=settings.HISTORY_CACHE_TTL)
        return response


import logging
from pathlib import Path
from typing import Optional, Dict, Any
import httpx
import pandas as pd
import yfinance as yf
from ml.data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
if not RAW_DATA_DIR.exists():
    RAW_DATA_DIR = Path("data/raw")


class YahooFinanceProvider(BaseMarketDataProvider):
    """
    Yahoo Finance Market Data Provider using resilient multi-tier fallback:
    1. Primary: yfinance library.
    2. Fallback: Direct Yahoo Finance Chart API (query1/query2) with browser impersonation headers.
       (Bypasses cloud datacenter IP blocks on Render/AWS/GCP).
    3. Final Fallback: Offline historical packaged datasets in data/raw.
    Supports NSE/BSE tickers (.NS, .BO) and global tickers.
    """

    def _fetch_direct_chart_ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Direct HTTP query to Yahoo Finance Chart API with browser headers.
        Bypasses cloud IP scraper blocks on Render/AWS/GCP.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        endpoints = [
            "https://query1.finance.yahoo.com",
            "https://query2.finance.yahoo.com",
        ]
        range_param = period if period in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] else "1y"

        for base_url in endpoints:
            try:
                url = f"{base_url}/v8/finance/chart/{symbol}?range={range_param}&interval={interval}"
                with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        chart = data.get("chart", {})
                        results = chart.get("result")
                        if results and len(results) > 0:
                            res = results[0]
                            timestamps = res.get("timestamp", [])
                            indicators = res.get("indicators", {})
                            quotes = indicators.get("quote", [{}])[0]
                            if timestamps and quotes.get("close"):
                                df = pd.DataFrame({
                                    "Date": pd.to_datetime(timestamps, unit="s"),
                                    "Open": quotes.get("open"),
                                    "High": quotes.get("high"),
                                    "Low": quotes.get("low"),
                                    "Close": quotes.get("close"),
                                    "Volume": quotes.get("volume"),
                                }).dropna()
                                if not df.empty:
                                    logger.info(f"Successfully fetched {len(df)} rows for {symbol} via direct Yahoo Chart API ({base_url})")
                                    return df
            except Exception as e:
                logger.warning(f"Direct chart fetch error on {base_url} for {symbol}: {e}")
        return None

    def _fetch_offline_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fallback to packaged historical dataset in data/raw when live network fails.
        """
        clean = symbol.strip().upper()
        candidates = [
            RAW_DATA_DIR / f"{clean}.csv",
            RAW_DATA_DIR / f"{clean.replace('.', '_')}.csv",
        ]
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                        df = df.dropna().sort_values("Date").reset_index(drop=True)
                        df.attrs["data_status"] = "OFFLINE"
                        df.attrs["data_source"] = f"Packaged Dataset ({p.name})"
                        logger.info(f"Loaded {len(df)} offline historical rows for {symbol} from {p.name}")
                        return df
                except Exception as e:
                    logger.warning(f"Error reading offline data from {p}: {e}")
        return None

    def _fetch_direct_chart_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        endpoints = [
            "https://query1.finance.yahoo.com",
            "https://query2.finance.yahoo.com",
        ]
        for base_url in endpoints:
            try:
                url = f"{base_url}/v8/finance/chart/{symbol}?range=5d&interval=1d"
                with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        chart = data.get("chart", {})
                        results = chart.get("result")
                        if results and len(results) > 0:
                            res = results[0]
                            meta = res.get("meta", {})
                            quotes = res.get("indicators", {}).get("quote", [{}])[0]
                            closes = [c for c in quotes.get("close", []) if c is not None]
                            price = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
                            prev_close = meta.get("chartPreviousClose") or (closes[-2] if len(closes) > 1 else None)
                            open_p = meta.get("regularMarketDayOpen") or (quotes.get("open", [None])[-1] if quotes.get("open") else None)
                            highs = [h for h in quotes.get("high", []) if h is not None]
                            lows = [l for l in quotes.get("low", []) if l is not None]
                            day_h = meta.get("regularMarketDayHigh") or (max(highs) if highs else None)
                            day_l = meta.get("regularMarketDayLow") or (min(lows) if lows else None)
                            vol = meta.get("regularMarketVolume") or (quotes.get("volume", [0])[-1] if quotes.get("volume") else 0)
                            return {
                                "price": round(float(price), 2) if price is not None else None,
                                "prev_close": round(float(prev_close), 2) if prev_close is not None else None,
                                "open": round(float(open_p), 2) if open_p is not None else None,
                                "high": round(float(day_h), 2) if day_h is not None else None,
                                "low": round(float(day_l), 2) if day_l is not None else None,
                                "volume": int(vol) if vol else 0,
                                "currency": meta.get("currency", "INR" if symbol.endswith(".NS") else "USD"),
                            }
            except Exception as e:
                logger.warning(f"Direct quote fetch error on {base_url} for {symbol}: {e}")
        return None

    def fetch_historical_ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        logger.info(f"Fetching {period} ({interval}) data for {symbol} via Yahoo Finance...")
        df = None
        data_source = None
        source_status = None
        try:
            # Tier 1: Try yfinance library
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval, auto_adjust=False)
                if df is not None and not df.empty:
                    data_source = "Yahoo Finance via yfinance history"
                    source_status = "ONLINE"
            except Exception as e:
                logger.warning(f"yfinance ticker.history failed for {symbol}: {e}")
                df = None

            if df is None or df.empty:
                # Fallback to yf.download
                try:
                    df = yf.download(
                        symbol,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        progress=False
                    )
                    if df is not None and not df.empty:
                        data_source = "Yahoo Finance via yfinance download"
                        source_status = "ONLINE"
                except Exception as e:
                    logger.warning(f"yf.download failed for {symbol}: {e}")
                    df = None

            # Tier 2: Direct Yahoo Finance Chart API (bypasses cloud IP rate limiting)
            if df is None or df.empty:
                logger.info(f"Falling back to direct Yahoo Chart API for {symbol}...")
                df = self._fetch_direct_chart_ohlcv(symbol, period=period, interval=interval)
                if df is not None and not df.empty:
                    data_source = "Yahoo Finance Chart API"
                    source_status = "ONLINE"

            # Tier 3: Packaged historical CSV fallback
            if df is None or df.empty:
                logger.warning(f"Falling back to packaged offline CSV for {symbol}...")
                df = self._fetch_offline_ohlcv(symbol)
                if df is not None and not df.empty:
                    data_source = "Packaged historical CSV"
                    source_status = "OFFLINE"

            if df is None or df.empty:
                raise ValueError(f"No market data returned for symbol '{symbol}'")

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Normalize column names
            date_col = None
            for candidate in ["Date", "Datetime", "index"]:
                if candidate in df.columns:
                    date_col = candidate
                    break
            
            if date_col is None:
                date_col = df.columns[0]

            df = df.rename(columns={date_col: "Date"})

            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            present_cols = [col for col in required_cols if col in df.columns]
            df = df[present_cols]

            # Format Date column
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

            # Ensure numeric types
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna().drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            # DataFrame attrs survive normal filtering/copy operations and let the
            # prediction service distinguish online retrieval from an offline cache.
            df.attrs["data_source"] = data_source or "Unknown market-data source"
            df.attrs["source_status"] = source_status or "UNKNOWN"
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise e

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves the latest market quote for symbol from Yahoo Finance.
        Handles fast_info, ticker.info, direct chart quote, and offline fallback.
        """
        clean_symbol = symbol.strip().upper()
        display_symbol = clean_symbol.replace(".NS", "").replace(".BO", "")
        currency = "INR" if clean_symbol.endswith(".NS") or clean_symbol.endswith(".BO") else "USD"

        try:
            price = None
            prev_close = None
            open_price = None
            day_high = None
            day_low = None
            volume = None

            # Tier 1: Try yfinance fast_info / info
            try:
                ticker = yf.Ticker(clean_symbol)
                fast_info = getattr(ticker, "fast_info", None)
                info = ticker.info or {}

                if fast_info:
                    try:
                        price = getattr(fast_info, "last_price", None)
                        prev_close = getattr(fast_info, "previous_close", None)
                        open_price = getattr(fast_info, "open", None)
                        day_high = getattr(fast_info, "day_high", None)
                        day_low = getattr(fast_info, "day_low", None)
                        volume = getattr(fast_info, "last_volume", None)
                    except Exception:
                        pass

                if price is None:
                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                if prev_close is None:
                    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
                if open_price is None:
                    open_price = info.get("regularMarketOpen") or info.get("open")
                if day_high is None:
                    day_high = info.get("regularMarketDayHigh") or info.get("dayHigh")
                if day_low is None:
                    day_low = info.get("regularMarketDayLow") or info.get("dayLow")
                if volume is None:
                    volume = info.get("regularMarketVolume") or info.get("volume")
            except Exception as e:
                logger.warning(f"yfinance quote lookup failed for {clean_symbol}: {e}")

            # Tier 2: Direct Yahoo Chart API (resilient against cloud IP blocking)
            if price is None or prev_close is None:
                direct_quote = self._fetch_direct_chart_quote(clean_symbol)
                if direct_quote:
                    price = price or direct_quote.get("price")
                    prev_close = prev_close or direct_quote.get("prev_close")
                    open_price = open_price or direct_quote.get("open")
                    day_high = day_high or direct_quote.get("high")
                    day_low = day_low or direct_quote.get("low")
                    volume = volume or direct_quote.get("volume")
                    currency = direct_quote.get("currency") or currency

            # Tier 3: Packaged CSV fallback
            if price is None or prev_close is None:
                offline_df = self._fetch_offline_ohlcv(clean_symbol)
                if offline_df is not None and not offline_df.empty:
                    last_row = offline_df.iloc[-1]
                    price = float(last_row["Close"])
                    open_price = open_price or float(last_row["Open"])
                    day_high = day_high or float(last_row["High"])
                    day_low = day_low or float(last_row["Low"])
                    volume = volume or float(last_row["Volume"])
                    if len(offline_df) > 1:
                        prev_close = float(offline_df.iloc[-2]["Close"])
                    else:
                        prev_close = open_price

            # Calculate change & percentage safely
            change = 0.0
            change_pct = 0.0
            if price is not None and prev_close is not None and prev_close > 0:
                change = round(float(price) - float(prev_close), 2)
                change_pct = round((change / float(prev_close)) * 100, 2)

            now_iso = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()

            return {
                "symbol": clean_symbol,
                "display_symbol": display_symbol,
                "price": round(float(price), 2) if price is not None else None,
                "previous_close": round(float(prev_close), 2) if prev_close is not None else None,
                "change": change,
                "change_pct": change_pct,
                "day_open": round(float(open_price), 2) if open_price is not None else None,
                "day_high": round(float(day_high), 2) if day_high is not None else None,
                "day_low": round(float(day_low), 2) if day_low is not None else None,
                "volume": int(volume) if volume is not None and not pd.isna(volume) else 0,
                "currency": currency,
                "timestamp": now_iso,
                "source": "Yahoo Finance",
                "data_status": "LIVE" if price is not None else "UNAVAILABLE",
            }
        except Exception as e:
            logger.error(f"Error fetching quote for {clean_symbol}: {e}")
            now_iso = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
            return {
                "symbol": clean_symbol,
                "display_symbol": display_symbol,
                "price": None,
                "previous_close": None,
                "change": 0.0,
                "change_pct": 0.0,
                "day_open": None,
                "day_high": None,
                "day_low": None,
                "volume": 0,
                "currency": currency,
                "timestamp": now_iso,
                "source": "Yahoo Finance",
                "data_status": "UNAVAILABLE",
            }

    def fetch_ticker_info(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            fast_info = getattr(ticker, "fast_info", None)

            current_price = (
                info.get("currentPrice") or
                info.get("regularMarketPrice") or
                (getattr(fast_info, "last_price", None) if fast_info else None)
            )
            previous_close = (
                info.get("previousClose") or
                info.get("regularMarketPreviousClose") or
                (getattr(fast_info, "previous_close", None) if fast_info else None)
            )

            # Fallback to get_quote if price still None
            if current_price is None or previous_close is None:
                q = self.get_quote(symbol)
                current_price = current_price or q.get("price")
                previous_close = previous_close or q.get("previous_close")

            name = info.get("longName") or info.get("shortName") or symbol
            currency = info.get("currency", "INR" if symbol.endswith(".NS") or symbol.endswith(".BO") else "USD")

            return {
                "symbol": symbol,
                "name": name,
                "current_price": current_price,
                "previous_close": previous_close,
                "currency": currency,
                "exchange": info.get("exchange", "NSE" if symbol.endswith(".NS") else "NASDAQ"),
                "sector": info.get("sector", "Financial Technology / Conglomerate"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "market_cap": info.get("marketCap"),
                "volume": info.get("volume"),
                "pe_ratio": info.get("trailingPE"),
            }
        except Exception as e:
            logger.warning(f"Error retrieving info for {symbol}: {e}")
            return {
                "symbol": symbol,
                "name": symbol,
                "currency": "INR" if symbol.endswith(".NS") else "USD",
                "current_price": None,
                "previous_close": None,
            }

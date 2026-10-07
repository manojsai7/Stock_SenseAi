import logging
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from ml.data.providers.base import BaseMarketDataProvider

logger = logging.getLogger(__name__)

class YahooFinanceProvider(BaseMarketDataProvider):
    """
    Yahoo Finance Market Data Provider using yfinance.
    Supports NSE/BSE tickers (.NS, .BO) and global tickers.
    """

    def fetch_historical_ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        logger.info(f"Fetching {period} ({interval}) data for {symbol} via Yahoo Finance...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)

            if df is None or df.empty:
                # Fallback to yf.download
                df = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False
                )

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
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise e

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves the latest market quote for symbol from Yahoo Finance.
        Handles fast_info, ticker.info, and fallback to latest history candle.
        """
        clean_symbol = symbol.strip().upper()
        display_symbol = clean_symbol.replace(".NS", "").replace(".BO", "")
        currency = "INR" if clean_symbol.endswith(".NS") or clean_symbol.endswith(".BO") else "USD"

        try:
            ticker = yf.Ticker(clean_symbol)
            fast_info = getattr(ticker, "fast_info", None)
            info = ticker.info or {}

            price = None
            prev_close = None
            open_price = None
            day_high = None
            day_low = None
            volume = None

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

            # Fallback: if price still None, fetch 1-day or 5-day history
            if price is None or prev_close is None:
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    last_row = hist.iloc[-1]
                    price = float(last_row["Close"])
                    open_price = open_price or float(last_row["Open"])
                    day_high = day_high or float(last_row["High"])
                    day_low = day_low or float(last_row["Low"])
                    volume = volume or float(last_row["Volume"])
                    if len(hist) > 1:
                        prev_close = float(hist.iloc[-2]["Close"])
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


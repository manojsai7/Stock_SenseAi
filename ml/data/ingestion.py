import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from ml.data.providers.yahoo_provider import YahooFinanceProvider

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
RAW_DATA_DIR = Path("data/raw")

class DataIngestionService:
    def __init__(self, provider=None, output_dir: Path = RAW_DATA_DIR):
        self.provider = provider or YahooFinanceProvider()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_and_save(self, symbol: str, period: str = "5y") -> Path:
        df = self.provider.fetch_historical_ohlcv(symbol, period=period)
        clean_symbol = symbol.replace(".", "_")
        file_path = self.output_dir / f"{clean_symbol}_daily.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows for {symbol} to {file_path}")
        return file_path

    def download_all_defaults(self, tickers: Optional[List[str]] = None, period: str = "5y") -> List[Path]:
        symbols = tickers or DEFAULT_TICKERS
        paths = []
        for sym in symbols:
            try:
                p = self.download_and_save(sym, period=period)
                paths.append(p)
            except Exception as e:
                logger.error(f"Failed to ingest data for {sym}: {e}")
        return paths

    def get_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Gets stock data either from raw cache or by fetching live.
        """
        clean_symbol = symbol.replace(".", "_")
        file_path = self.output_dir / f"{clean_symbol}_daily.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        return self.provider.fetch_historical_ohlcv(symbol, period=period)

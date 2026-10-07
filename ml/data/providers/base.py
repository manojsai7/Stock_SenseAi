from abc import ABC, abstractmethod
import pandas as pd

class BaseMarketDataProvider(ABC):
    """
    Abstract Interface for market data ingestion.
    Allows decoupling backend and models from any specific data vendor.
    """

    @abstractmethod
    def fetch_historical_ohlcv(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        Returns a DataFrame with columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        pass

    @abstractmethod
    def fetch_ticker_info(self, symbol: str) -> dict:
        """
        Fetch metadata regarding the stock (name, currency, market cap, etc.).
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """
        Fetch latest available market quote for the stock.
        """
        pass

from typing import List, Optional
from pydantic import BaseModel

class StockOHLCV(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ma20: Optional[float] = None

class StockHistoryResponse(BaseModel):
    symbol: str
    period: str
    count: int
    data: List[StockOHLCV]

class StockInfoResponse(BaseModel):
    symbol: str
    name: str
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    currency: str = "INR"
    exchange: Optional[str] = None
    sector: Optional[str] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

class StockSearchItem(BaseModel):
    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None

class StockQuoteResponse(BaseModel):
    symbol: str
    display_symbol: str
    price: Optional[float] = None
    previous_close: Optional[float] = None
    change: float = 0.0
    change_pct: float = 0.0
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: int = 0
    currency: str = "INR"
    market_status: str  # PRE_MARKET, OPEN, POST_MARKET, CLOSED, WEEKEND, HOLIDAY
    timestamp: str
    source: str = "Yahoo Finance"
    data_status: str = "LIVE"  # LIVE, STALE, UNAVAILABLE
    last_updated: str


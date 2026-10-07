from fastapi import APIRouter, HTTPException, Query
from typing import List
from backend.services.stock_service import StockService
from backend.schemas.stock import StockSearchItem, StockInfoResponse, StockHistoryResponse, StockQuoteResponse

router = APIRouter(prefix="/stocks", tags=["Stocks"])
stock_service = StockService()

@router.get("", response_model=List[StockSearchItem], summary="List Supported Equities")
@router.get("/search", response_model=List[StockSearchItem], summary="Search Supported Equities")
def search_stocks(q: str = Query("", description="Query string for stock symbol or name")):
    try:
        return stock_service.search_stocks(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/quote", response_model=StockQuoteResponse, summary="Get Live Market Quote")
def get_stock_quote(symbol: str):
    """
    Returns latest available market quote, day delta, volume, session status (OPEN/CLOSED),
    and last updated timestamp.
    """
    try:
        return stock_service.get_stock_quote(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote for '{symbol}': {str(e)}")

@router.get("/{symbol}/info", response_model=StockInfoResponse, summary="Get Stock Profile & Fundamentals")
def get_stock_info(symbol: str):
    try:
        return stock_service.get_stock_info(symbol)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Stock info for '{symbol}' not found: {str(e)}")

@router.get("/{symbol}/history", response_model=StockHistoryResponse, summary="Get Historical OHLCV + MA20")
def get_stock_history(symbol: str, period: str = Query("1y", pattern="^(1mo|3mo|6mo|1y|2y|5y)$")):
    try:
        return stock_service.get_stock_history(symbol, period=period)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching history for '{symbol}': {str(e)}")


from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from backend.services.research_service import research_service

router = APIRouter(prefix="/research", tags=["Research & Evaluation"])


@router.get("/model-comparison", summary="Get Prospective Research Model Comparison")
def get_model_comparison(symbol: str = Query("RELIANCE.NS", description="Equity symbol (e.g. RELIANCE.NS)")) -> Dict[str, Any]:
    """
    Returns the real-time prospective forward evaluation data for authorized equities.
    Compares Production CNN-LSTM vs Colab EXP-05 Fold 3 (Challenger) vs Naive Persistence Baseline.
    Strictly isolates resolved outcomes from pending forecasts.
    Does not mix historical offline fold results with forward prospective metrics.
    """
    try:
        return research_service.get_model_comparison(symbol)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load research comparison for '{symbol}': {str(e)}")


@router.get("/model-comparison/{symbol}", summary="Get Prospective Research Model Comparison by Symbol Path")
def get_model_comparison_by_symbol(symbol: str) -> Dict[str, Any]:
    """
    Path-parameter variant for /api/research/model-comparison/{symbol}.
    """
    return get_model_comparison(symbol=symbol)

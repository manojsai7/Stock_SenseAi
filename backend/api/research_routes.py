from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from backend.services.research_service import research_service
from backend.services.challenger_prediction_service import challenger_prediction_service
from backend.services.prediction_service import StaleMarketDataError, PredictionService
from backend.services.prospective_v2_service import prospective_v2_service

router = APIRouter(prefix="/research", tags=["Research & Evaluation"])


@router.get("/model-comparison", summary="Get Prospective Research Model Comparison (EXP-05)")
def get_model_comparison(symbol: str = Query("RELIANCE.NS", description="Equity symbol (e.g. RELIANCE.NS)")) -> Dict[str, Any]:
    """
    Returns the real-time prospective forward evaluation data for authorized equities (EXP-05).
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


@router.get("/model-comparison/{symbol}", summary="Get Prospective Research Model Comparison by Symbol Path (EXP-05)")
def get_model_comparison_by_symbol(symbol: str) -> Dict[str, Any]:
    """
    Path-parameter variant for /api/research/model-comparison/{symbol}.
    """
    return get_model_comparison(symbol=symbol)


@router.get("/challenger-prediction", summary="Get V2 GRU Challenger Prediction")
@router.get("/challenger-prediction/{symbol}", summary="Get V2 GRU Challenger Prediction by Symbol")
def get_challenger_prediction(symbol: str = "RELIANCE.NS") -> Dict[str, Any]:
    """
    Returns real-time shadow inference forecast for the V2 GRU Challenger model (EXP-06).
    Exclusively authorized for RELIANCE.NS.
    Enforces completed-candle rules and rejects stale data with HTTP 503.
    """
    clean_symbol = symbol.strip().upper()
    if clean_symbol != "RELIANCE.NS":
        raise HTTPException(
            status_code=400,
            detail=f"The V2 GRU challenger model is currently trained and authorized exclusively for RELIANCE.NS under protocol EXP-06. Symbol '{clean_symbol}' is not supported in this research phase."
        )

    try:
        pred = challenger_prediction_service.get_prediction(clean_symbol)

        # Synchronize prospective evaluation stream
        try:
            prod_service = PredictionService()
            v1_pred = prod_service._get_stationary_prediction(clean_symbol)
            prospective_v2_service.record_daily_forecasts(
                symbol=clean_symbol,
                target_date=pred["target_trading_date"],
                input_window_end=pred["latest_completed_input_date"],
                previous_close=pred["latest_completed_close"],
                v1_pred=v1_pred,
                v2_pred=pred,
                data_source=pred.get("data_provider", "Yahoo Finance")
            )
        except Exception as sync_exc:
            # Failure to record shadow prospective log should not crash inference endpoint
            pass

        return pred
    except StaleMarketDataError as sme:
        raise HTTPException(status_code=503, detail=f"Market data is stale: {str(sme)}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate V2 GRU challenger prediction: {str(e)}")


@router.get("/exp06-comparison", summary="Get EXP-06 V2 Prospective Model Comparison")
@router.get("/exp06-comparison/{symbol}", summary="Get EXP-06 V2 Prospective Model Comparison by Symbol")
def get_exp06_comparison(symbol: str = "RELIANCE.NS") -> Dict[str, Any]:
    """
    Returns dedicated EXP-06 prospective evaluation stream:
    Compares Production CNN-LSTM V1 vs GRU V2 Challenger vs Naive Persistence Benchmark.
    Maintains strict separation between resolved market outcomes and pending forward forecasts.
    """
    try:
        return prospective_v2_service.get_exp06_comparison(symbol)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load EXP-06 prospective comparison: {str(e)}")


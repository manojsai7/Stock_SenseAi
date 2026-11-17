from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from backend.services.research_service import research_service
from backend.services.challenger_prediction_service import challenger_prediction_service
from backend.services.prediction_service import StaleMarketDataError, PredictionService
from backend.services.prospective_v2_service import prospective_v2_service
from backend.services.forecast_accuracy_service import forecast_accuracy_service
from backend.services.historical_predictions_service import historical_predictions_service
from backend.services.market_status_service import market_status_service
from backend.services.research_v2_registry import ResearchV2Registry

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
    Pure read-only endpoint with zero prospective logging side effects.
    """
    clean_symbol = symbol.strip().upper()
    if clean_symbol != "RELIANCE.NS":
        raise HTTPException(
            status_code=400,
            detail=f"The V2 GRU challenger model is currently trained and authorized exclusively for RELIANCE.NS under protocol EXP-06. Symbol '{clean_symbol}' is not supported in this research phase."
        )

    try:
        return challenger_prediction_service.get_prediction(clean_symbol)
    except StaleMarketDataError as sme:
        raise HTTPException(status_code=503, detail=f"Market data is stale: {str(sme)}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate V2 GRU challenger prediction: {str(e)}")


@router.get("/latest-forecast", summary="Get Latest Next-Session Synchronized Forecast Trio")
@router.get("/latest-forecast/{symbol}", summary="Get Latest Next-Session Forecast by Symbol")
def get_latest_forecast(symbol: str = "RELIANCE.NS") -> Dict[str, Any]:
    """
    Returns the authoritative next-session forecast comparison trio (Production V1, GRU V2, Naive Baseline).
    Pure read-only endpoint with zero prospective logging side effects.
    Includes target date, input cutoff, predicted prices, returns, model versions, hashes, and data freshness.
    """
    clean_symbol = symbol.strip().upper()
    if clean_symbol != "RELIANCE.NS":
        return {
            "symbol": clean_symbol,
            "supported": False,
            "message": f"Synchronized multi-model next-session forecasts are currently authorized exclusively for RELIANCE.NS. Symbol '{clean_symbol}' is evaluated via standard V1 production serving.",
            "authorized_symbol": "RELIANCE.NS",
            "models": {},
            "is_stale": False,
        }

    now_ist = market_status_service.get_current_time()
    market_state = market_status_service.get_market_status(now_ist)
    latest_completed = market_status_service.get_latest_completed_trading_day(now_ist)
    next_trading_day = market_status_service.get_next_trading_day(latest_completed)

    try:
        # 1. Fetch V2 GRU prediction
        v2 = challenger_prediction_service.get_prediction(clean_symbol)
        ref_close = v2["latest_completed_close"]
        input_cutoff = v2["latest_completed_input_date"]
        target_date = v2["target_trading_date"]

        # 2. Fetch V1 Production prediction
        prod_service = PredictionService()
        v1 = prod_service._get_stationary_prediction(clean_symbol)

        # 3. Model metadata from registry
        registry = ResearchV2Registry()
        reg_manifest, _, _, _ = registry.load(clean_symbol)
        gru_hash = reg_manifest["files"].get("model.keras", reg_manifest["files"].get("model", {})).get("sha256", "")
        prod_hash = "a4715e84cdd7a16d1b334936c9a39b35249bb594aec5c987a4f6ee62d50d4f11"

        v1_close = float(v1.get("predicted_close", v1.get("predicted_next_session_close", ref_close)))
        v1_ret = float(v1.get("predicted_return", 0.0))
        v1_dir = str(v1.get("direction", "FLAT")).upper()

        v2_close = float(v2["predicted_next_session_close"])
        v2_ret = float(v2["predicted_simple_return"])
        v2_dir = str(v2["direction"]).upper()

        return {
            "symbol": clean_symbol,
            "supported": True,
            "target_trading_date": target_date,
            "latest_completed_input_date": input_cutoff,
            "reference_close": ref_close,
            "market_status": market_state,
            "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            "data_provider": v2.get("data_provider", "Yahoo Finance (Daily OHLCV)"),
            "data_freshness": v2.get("data_freshness", "COMPLETED"),
            "is_stale": False,
            "stale_reason": None,
            "models": {
                "production_v1": {
                    "name": "Production CNN-LSTM V1",
                    "role": "Production Baseline",
                    "status_badge": "PRODUCTION",
                    "is_experimental": False,
                    "predicted_next_session_close": round(v1_close, 2),
                    "predicted_return": round(v1_ret, 6),
                    "predicted_return_pct": round(v1_ret * 100, 2),
                    "predicted_direction": v1_dir,
                    "reference_close": ref_close,
                    "input_cutoff_date": input_cutoff,
                    "target_trading_date": target_date,
                    "model_version": "stationary-return-v1",
                    "model_hash": prod_hash,
                    "training_cutoff": "2026-10-06",
                },
                "gru_v2_challenger": {
                    "name": "GRU V2 Challenger",
                    "role": "Research Challenger (Shadow)",
                    "status_badge": "EXPERIMENTAL",
                    "is_experimental": True,
                    "predicted_next_session_close": round(v2_close, 2),
                    "predicted_return": round(v2_ret, 6),
                    "predicted_return_pct": round(v2_ret * 100, 2),
                    "predicted_direction": v2_dir,
                    "reference_close": ref_close,
                    "input_cutoff_date": input_cutoff,
                    "target_trading_date": target_date,
                    "model_version": "stocksense-v2-gru-challenger",
                    "model_hash": gru_hash,
                    "training_cutoff": "2025-12-11",
                },
                "naive_baseline": {
                    "name": "Naive Persistence Benchmark",
                    "role": "Reference Benchmark",
                    "status_badge": "BASELINE",
                    "is_experimental": False,
                    "predicted_next_session_close": round(ref_close, 2),
                    "predicted_return": 0.0,
                    "predicted_return_pct": 0.0,
                    "predicted_direction": "FLAT",
                    "reference_close": ref_close,
                    "input_cutoff_date": input_cutoff,
                    "target_trading_date": target_date,
                    "model_version": "zero-parameter-persistence",
                    "model_hash": "zero_parameter_baseline",
                    "training_cutoff": "N/A",
                }
            }
        }
    except StaleMarketDataError as sme:
        return {
            "symbol": clean_symbol,
            "supported": True,
            "target_trading_date": next_trading_day.strftime("%Y-%m-%d"),
            "latest_completed_input_date": latest_completed.strftime("%Y-%m-%d"),
            "reference_close": None,
            "market_status": market_state,
            "is_stale": True,
            "stale_reason": str(sme),
            "models": {}
        }
    except Exception as e:
        return {
            "symbol": clean_symbol,
            "supported": True,
            "target_trading_date": next_trading_day.strftime("%Y-%m-%d"),
            "latest_completed_input_date": latest_completed.strftime("%Y-%m-%d"),
            "reference_close": None,
            "market_status": market_state,
            "is_stale": True,
            "stale_reason": f"Forecast currently unavailable: {str(e)}",
            "models": {}
        }


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


@router.get("/forecast-accuracy", summary="Get Historical Forecast Accuracy & Track Record")
@router.get("/forecast-accuracy/{symbol}", summary="Get Historical Forecast Accuracy & Track Record by Symbol")
def get_forecast_accuracy(
    symbol: str = "RELIANCE.NS",
    limit: Optional[int] = Query(None, description="Limit to last N resolved sessions (e.g. 20, 60)"),
    include_historical: bool = Query(True, description="Include separate historical development backtest benchmarks")
) -> Dict[str, Any]:
    """
    Returns data-driven forecast accuracy metrics and historical track record:
    - Overall model error (MAE, RMSE, MAPE in INR and %)
    - Directional accuracy, balanced accuracy, MCC, UP/DOWN recall
    - Common-date paired head-to-head comparison
    - Recharts-ready predicted vs actual price timeline
    - Chronological table of resolved outcomes and pending forecasts
    - Strict decoupling between prospective track records and historical offline backtests
    """
    try:
        return forecast_accuracy_service.get_accuracy_payload(
            symbol=symbol,
            limit=limit,
            include_historical_benchmarks=include_historical
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load forecast accuracy track record: {str(e)}")


@router.get("/historical-predictions", summary="Get Historical Walk-Forward Predictions & Explorer Data")
@router.get("/historical-predictions/{symbol}", summary="Get Historical Walk-Forward Predictions by Symbol")
def get_historical_predictions(
    symbol: str = "RELIANCE.NS",
    experiment: Optional[str] = Query(None, description="Experiment filter: EXP-06, EXP-05, V1-Production-Audit, or ALL"),
    model_name: Optional[str] = Query(None, description="Model name filter"),
    fold: Optional[str] = Query(None, description="Fold filter (e.g. 1, 2, 3, or ALL)"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: Optional[int] = Query(None, description="Limit returned prediction rows for UI performance (e.g. 100, 300)")
) -> Dict[str, Any]:
    """
    Returns verified archived historical walk-forward predictions, error distributions,
    model predicting capability scores, and model coverage matrices.
    Clearly designated as OFFLINE_DEVELOPMENT_EVALUATION.
    """
    try:
        return historical_predictions_service.get_historical_predictions(
            symbol=symbol,
            experiment=experiment,
            model_name=model_name,
            fold=fold,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query historical predictions for '{symbol}': {str(e)}")




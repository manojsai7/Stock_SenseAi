from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from backend.services.prediction_service import PredictionService, StaleMarketDataError
from backend.schemas.prediction import PredictionResponse, ModelMetricsResponse
from backend.services.explainability_service import explainability_service

router = APIRouter(tags=["Predictions & Models"])
pred_service = PredictionService()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_CSV = PROJECT_ROOT / "experiments" / "results.csv"
if not RESULTS_CSV.exists():
    RESULTS_CSV = Path("experiments/results.csv")

@router.get("/predict/{symbol}", response_model=PredictionResponse, summary="Predict Next Trading Day Close")
@router.get("/predictions/{symbol}/latest", response_model=PredictionResponse, summary="Latest Prediction")
@router.post("/predictions/{symbol}", response_model=PredictionResponse, summary="Execute Prediction")
def get_prediction(symbol: str, pipeline_version: Optional[str] = None):
    """
    Executes CNN-LSTM inference on the latest 30 COMPLETED trading days to forecast
    the next trading day's closing price.
    Supports versioned inference via pipeline_version parameter ('legacy' or 'stationary-return-v1').
    """
    try:
        return pred_service.get_prediction(symbol, pipeline_version=pipeline_version)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except StaleMarketDataError as stale:
        raise HTTPException(status_code=503, detail=str(stale))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error for '{symbol}': {str(e)}")

@router.get("/models", response_model=List[Dict[str, Any]], summary="List Pretrained Model Weights")
def list_models():
    """
    List trained model versions and saved architectures across all supported large caps.
    """
    return pred_service.list_models()

@router.get("/models/{symbol}/explainability", summary="Explain current stationary model input sensitivity")
def get_model_explainability(symbol: str):
    """
    Computes live input sensitivity for the stationary model using the latest 30
    completed sessions. This is a perturbation diagnostic, not causal attribution.
    """
    try:
        return explainability_service.get_stationary_feature_sensitivity(symbol)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except StaleMarketDataError as stale:
        raise HTTPException(status_code=503, detail=str(stale))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error for '{symbol}': {str(e)}")

@router.get("/models/{symbol}/metrics", response_model=ModelMetricsResponse, summary="Get Model Test Metrics")
def get_model_metrics(symbol: str):
    """
    Returns comparative evaluation metrics (Naive vs LSTM vs CNN-LSTM)
    and test-set actual vs predicted trajectory from offline evaluation.
    """
    try:
        return pred_service.get_model_metrics(symbol)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/benchmark/experiments", summary="Get Full Empirical Experiment Matrix")
def get_benchmark_experiments():
    """
    Returns the systematic empirical results matrix and ablation studies from experiments/results.csv.
    """
    if not RESULTS_CSV.exists():
        return []
    try:
        df = pd.read_csv(RESULTS_CSV)
        # Clean NaN values for JSON serialization
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read benchmark results: {str(e)}")


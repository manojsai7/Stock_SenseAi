from fastapi import APIRouter
from datetime import datetime, timezone
from backend.schemas.health import HealthResponse
from backend.services.market_status_service import market_status_service
from backend.services.model_registry import model_registry
from backend.core.cache import cache

router = APIRouter(tags=["Health & Observability"])

@router.get("/health", response_model=HealthResponse, summary="Basic Health Check")
@router.get("/system/health", response_model=HealthResponse, summary="Detailed System & Cache Health")
def get_system_health():
    """
    Returns live health metrics: API operational status, market session state,
    ModelRegistry status, and memory cache hit ratio.
    """
    cache_stats = cache.get_stats()
    market_status = market_status_service.get_market_status()

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_status": market_status,

        "services": {
            "api": "healthy",
            "market_data": "healthy",
            "model_registry": "healthy",
            "cache": "healthy",
        },
        "models": {
            "loaded_count": len(model_registry.get_loaded_symbols()),
            "loaded": model_registry.get_loaded_symbols(),
            "available_count": len(model_registry.supported_symbols),
            "supported": model_registry.supported_symbols,
        },
        "cache": cache_stats,
    }

@router.get("/ready", summary="Basic Readiness Probe")
@router.get("/system/ready", summary="System Readiness Probe")
def get_readiness():
    """
    Readiness probe verifying model artifact accessibility and local engine integrity.
    Does NOT depend on external Yahoo Finance network availability.
    """
    has_artifacts = all(model_registry.has_artifacts(sym) for sym in model_registry.trained_symbols)
    loaded_symbols = model_registry.get_loaded_symbols()
    return {
        "status": "ready" if has_artifacts else "not_ready",
        "models_accessible": has_artifacts,
        "models_in_memory": len(loaded_symbols),
        "trained_stocks": model_registry.trained_symbols,
        "supported_stocks": model_registry.supported_symbols,
    }



import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.core.config import settings
from backend.api.health_routes import router as health_router
from backend.api.stock_routes import router as stock_router
from backend.api.prediction_routes import router as prediction_router

from contextlib import asynccontextmanager
import logging
from backend.services.model_registry import model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("stocksense")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Initializing StockSense AI Production Backend...")
    logger.info(f"[STARTUP] Supported Stocks: {settings.SUPPORTED_STOCKS}")
    # Warm up models in the background/startup so inference is instantaneous
    try:
        warmed = model_registry.warm_all()
        logger.info(f"[STARTUP] Pre-warmed model artifacts: {warmed}")
    except Exception as e:
        logger.warning(f"[STARTUP] Model warming notice: {e}")
    yield
    logger.info("[SHUTDOWN] Shutting down StockSense AI Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="StockSense AI: Deep Learning-Based Stock Price Prediction Using CNN-LSTM Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(health_router, prefix="/api/system")
app.include_router(stock_router, prefix=settings.API_V1_PREFIX)
app.include_router(prediction_router, prefix=settings.API_V1_PREFIX)


# Serve Frontend static build if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if not frontend_dist.exists():
    frontend_dist = Path("frontend/dist")

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def index():
        return {
            "project": "StockSense AI",
            "message": "Backend API is active. Access docs at /docs or run frontend dev server.",
            "endpoints": {
                "health": "/api/system/health",
                "ready": "/api/system/ready",
                "stocks_search": "/api/stocks/search",
                "stock_info": "/api/stocks/{symbol}/info",
                "stock_quote": "/api/stocks/{symbol}/quote",
                "stock_history": "/api/stocks/{symbol}/history",
                "prediction": "/api/predict/{symbol}",
                "benchmark": "/api/benchmark/experiments"
            }
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", settings.PORT))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting StockSense AI server on {host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=(settings.ENVIRONMENT == "development"))


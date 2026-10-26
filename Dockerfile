# ============================================================
# Stage 1: Build Frontend Assets
# ============================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install frontend dependencies
COPY frontend/package.json ./
RUN npm install

# Build static React production bundle
COPY frontend/ ./
RUN npm run build

# ============================================================
# Stage 2: Python Production Runtime
# ============================================================
FROM python:3.11-slim AS runtime

# Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install system dependencies (curl for container healthcheck, ca-certificates for TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML and API dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY models_storage/ ./models_storage/
COPY models_storage_stationary/ ./models_storage_stationary/
COPY experiments/ ./experiments/
COPY data/ ./data/
COPY research/ ./research/


# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (default 8000, platforms like Render/Railway inject $PORT)
EXPOSE 8000

# Container Healthcheck (polls local /api/system/health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/system/health || exit 1

# Start Uvicorn process listening on 0.0.0.0 and dynamic $PORT
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

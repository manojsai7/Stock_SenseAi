from typing import Dict, Any
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    market_status: str
    services: Dict[str, str]
    models: Dict[str, Any]
    cache: Dict[str, Any]

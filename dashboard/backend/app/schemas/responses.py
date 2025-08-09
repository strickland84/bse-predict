from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


class SystemStatus(BaseModel):
    status: str  # healthy, warning, error
    database: bool
    services: Dict[str, str]
    last_data_fetch: Optional[datetime]
    last_prediction: Optional[datetime]
    last_model_training: Optional[datetime]


class PredictionData(BaseModel):
    symbol: str
    timestamp: datetime
    target_1pct: float
    target_2pct: float
    target_5pct: float
    confidence_1pct: float
    confidence_2pct: float
    confidence_5pct: float
    current_price: float


class ModelPerformance(BaseModel):
    symbol: str
    target: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_trained: datetime
    training_samples: int


class DataHealth(BaseModel):
    symbol: str
    total_candles: int
    latest_candle: datetime
    oldest_candle: datetime
    gaps_detected: int
    coverage_percentage: float
    status: str  # healthy, gaps_detected, critical


class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime
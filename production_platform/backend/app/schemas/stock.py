from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class StockHistoryRequest(BaseModel):
    ticker: str
    period: str = "1y"

class StockIndicator(BaseModel):
    date: str
    close: float
    volume: int
    sma_20: Optional[float]
    sma_50: Optional[float]
    ema_20: Optional[float]
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    bb_high: Optional[float]
    bb_low: Optional[float]

class StockPredictRequest(BaseModel):
    ticker: str
    timeframe: str = Field(..., pattern="^(1 Day|1 Week|1 Month)$")

class PredictionPoint(BaseModel):
    date: str
    price: float

class StockPredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predictions: List[PredictionPoint]
    metrics: Dict[str, float]

from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.stock import StockHistoryRequest, StockPredictionResponse, StockPredictRequest, StockIndicator, PredictionPoint
from app.services.stock_service import StockPredictor
from app.services.stock_analysis import calculate_indicators
from app.utils.market_data import MarketDataService
import pandas as pd
import numpy as np
from typing import List

router = APIRouter()

@router.get("/history", response_model=List[StockIndicator])
def get_stock_history(ticker: str, period: str = "1y"):
    data = MarketDataService.fetch_history(ticker, period)
    if data is None or data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    # Calculate indicators
    df_analyzed = calculate_indicators(data)
    
    # Prepare details for JSON response
    results = []
    for index, row in df_analyzed.iterrows():
        # Handle NaN values safely for JSON
        def safe_float(val):
            return val if not pd.isna(val) else None
            
        results.append({
            "date": index.strftime("%Y-%m-%d"),
            "close": safe_float(row['Close']),
            "volume": int(row['Volume']),
            "sma_20": safe_float(row.get('SMA_20')),
            "sma_50": safe_float(row.get('SMA_50')),
            "ema_20": safe_float(row.get('EMA_20')),
            "rsi": safe_float(row.get('RSI')),
            "macd": safe_float(row.get('MACD')),
            "macd_signal": safe_float(row.get('MACD_Signal')),
            "bb_high": safe_float(row.get('BB_High')),
            "bb_low": safe_float(row.get('BB_Low')),
        })
    
    return results

@router.post("/predict", response_model=StockPredictionResponse)
def predict_stock_price(request: StockPredictRequest):
    # Map timeframe to days
    timeframe_map = {"1 Day": 1, "1 Week": 7, "1 Month": 30}
    days = timeframe_map.get(request.timeframe, 1)
    
    # 1. Fetch Data
    data = MarketDataService.fetch_history(request.ticker, period="5y") # Need enough history for LSTM
    if data is None or data.empty:
         raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
         
    try:
        # 2. Initialize and Train Model (On-the-fly as per requirements)
        # Note: In a real/scaled system, this would be a background task or separate worker.
        # But per "preserve logic" constraint, we do it here.
        predictor = StockPredictor(sequence_length=60)
        history = predictor.train(data, epochs=50) # Matching original defaults
        
        # 3. Predict
        predictions = predictor.predict(data, days=days)
        
        # 4. Format Content
        last_date = data.index[-1]
        pred_points = []
        for i, price in enumerate(predictions):
            future_date = last_date + pd.Timedelta(days=i+1)
            pred_points.append(PredictionPoint(
                date=future_date.strftime("%Y-%m-%d"),
                price=float(price)
            ))
            
        # Extract metrics safely (Handle both Keras History object and Mock dict)
        if isinstance(history, dict):
            loss = history.get('loss', 0.0)
            val_loss = history.get('val_loss', 0.0)
        else:
            # Assume it's a Keras History object
            loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0.0

        final_val_loss = val_loss
        
        return {
            "ticker": request.ticker,
            "current_price": float(data['Close'].iloc[-1]),
            "predictions": pred_points,
            "metrics": {
                "loss": float(loss),
                "val_loss": float(final_val_loss),
                "confidence": max(0.0, (1.0 - final_val_loss) * 100.0)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

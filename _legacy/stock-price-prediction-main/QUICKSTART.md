# Quick Start Guide

## Installation

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
```

2. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Historical Analysis Page
1. Enter a stock ticker (e.g., AAPL, GOOGL, MSFT, TSLA)
2. Select time period (1 month to 2 years)
3. View interactive charts:
   - Candlestick chart with moving averages
   - Bollinger Bands
   - Trading volume
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Price distribution

### Price Prediction Page
1. Select prediction timeframe (1 day, 1 week, or 1 month)
2. Click "Generate Prediction"
3. Wait for model training (30-60 seconds)
4. View:
   - Training metrics
   - Price forecast chart
   - Detailed predictions table
   - Trend analysis

### Session Log Page
1. View all activities performed in the current session
2. See summary metrics (total activities, unique stocks, etc.)
3. Filter by activity type or stock ticker
4. Export log as CSV for record keeping
5. Clear log when starting a new session

## Model Details

**Architecture**: Bidirectional LSTM with Attention Mechanism

**Layers**:
- Bidirectional LSTM (128 units) with return sequences
- Dropout (0.2)
- Bidirectional LSTM (64 units) with return sequences
- Dropout (0.2)
- Custom Attention Layer
- Dense (32 units, ReLU)
- Dropout (0.2)
- Dense (16 units, ReLU)
- Output Dense (1 unit)

**Features Used**:
- Close Price
- Volume
- SMA (10, 20 days)
- EMA (10 days)
- RSI (14 days)
- MACD & Signal
- Price Change %
- Volume Change %

**Training**:
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Early Stopping: Patience=10
- Validation Split: 20%

## Tips

- Use well-known stock tickers for best results
- Longer historical data provides better predictions
- Model training takes 30-60 seconds depending on your hardware
- Predictions are more accurate for shorter timeframes (1 day vs 1 month)

## Troubleshooting

**Issue**: "Could not load data for ticker"
- **Solution**: Check if the ticker symbol is correct and the stock is actively traded

**Issue**: Model training is slow
- **Solution**: Reduce epochs in `model/predictor.py` (default is 50)

**Issue**: Predictions seem unrealistic
- **Solution**: Stock markets are inherently unpredictable. Use predictions as one of many tools for analysis.

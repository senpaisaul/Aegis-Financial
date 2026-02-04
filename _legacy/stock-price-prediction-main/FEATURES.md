# Features Documentation

## 📊 Historical Analysis

### Overview
Comprehensive technical analysis of stock price history with interactive visualizations.

### Key Components
- **Candlestick Chart**: OHLC data with moving averages (SMA 20/50, EMA 20)
- **Bollinger Bands**: Volatility indicator with upper/lower bands
- **Volume Analysis**: Color-coded volume bars (red for down days, green for up days)
- **RSI Indicator**: Momentum oscillator with overbought (70) and oversold (30) levels
- **MACD**: Trend-following momentum indicator with signal line and histogram
- **Price Distribution**: Histogram showing price frequency distribution

### Summary Statistics
- Current price
- Period change ($ and %)
- Highest price in period
- Lowest price in period

### Time Periods
- 1 Month
- 3 Months
- 6 Months
- 1 Year
- 2 Years
- All available data

---

## 🔮 Price Prediction

### Overview
Advanced time series forecasting using deep learning with Bidirectional LSTM and Attention mechanism.

### Prediction Timeframes
- **1 Day**: Next trading day forecast
- **1 Week**: 7-day forecast
- **1 Month**: 30-day forecast

### Model Architecture
```
Input Layer (60 timesteps × 10 features)
    ↓
Bidirectional LSTM (128 units) + Dropout (0.2)
    ↓
Bidirectional LSTM (64 units) + Dropout (0.2)
    ↓
Custom Attention Layer
    ↓
Dense (32 units, ReLU) + Dropout (0.2)
    ↓
Dense (16 units, ReLU)
    ↓
Output (1 unit)
```

### Features Used
1. Close Price
2. Volume
3. SMA 10-day
4. SMA 20-day
5. EMA 10-day
6. RSI (14-day)
7. MACD
8. MACD Signal
9. Price Change %
10. Volume Change %

### Training Details
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Validation Split**: 20%
- **Early Stopping**: Patience of 10 epochs
- **Typical Training Time**: 30-60 seconds

### Prediction Output
- Training loss and validation loss metrics
- Model confidence score
- Price forecast chart with confidence intervals
- Detailed predictions table
- Expected change ($ and %)
- Trend indicator (Bullish/Bearish)

---

## 📋 Session Log

### Overview
Comprehensive activity tracking for all analyses and predictions performed during the current session.

### Features

#### Summary Dashboard
- **Total Activities**: Count of all logged activities
- **Unique Stocks**: Number of different stocks analyzed
- **Analyses Count**: Number of historical analyses performed
- **Predictions Count**: Number of predictions generated

#### Visualizations
- **Activity Type Distribution**: Pie chart showing breakdown of analyses vs predictions
- **Stocks Analyzed**: Bar chart showing most frequently analyzed stocks

#### Activity Details
Each logged activity includes:

**For Analysis:**
- Timestamp
- Stock ticker
- Time period analyzed
- Current price
- Price range (high/low)

**For Predictions:**
- Timestamp
- Stock ticker
- Prediction timeframe (1 day/week/month)
- Current price
- Predicted price
- Expected change ($ and %)
- Model confidence score
- Training and validation loss

#### Filtering Options
- Filter by activity type (Analysis/Prediction)
- Filter by stock ticker
- Sort by newest or oldest first

#### Export & Management
- **Export to CSV**: Download complete log with all details
- **Clear Log**: Reset session log (requires confirmation)

#### Real-time Updates
- Activity counter in sidebar
- Last activity preview in sidebar
- Toast notifications when activities are logged

### Use Cases
1. **Session Review**: Review all stocks analyzed in current session
2. **Performance Tracking**: Compare predictions across different stocks
3. **Record Keeping**: Export logs for future reference
4. **Pattern Recognition**: Identify frequently analyzed stocks

---

## 🎨 User Interface

### Design Principles
- **Dark Theme**: Easy on the eyes for extended use
- **Interactive Charts**: Plotly-powered visualizations with zoom, pan, and hover details
- **Responsive Layout**: Adapts to different screen sizes
- **Clear Navigation**: Simple sidebar navigation between pages
- **Real-time Feedback**: Toast notifications and loading indicators

### Color Scheme
- Primary: Cyan/Blue tones
- Success: Green
- Warning: Orange
- Error: Red
- Background: Dark theme

### Accessibility
- Clear labels and descriptions
- Hover tooltips on charts
- Color-blind friendly chart colors
- Responsive design for mobile devices

---

## 🔒 Data & Privacy

### Data Sources
- **Stock Data**: Yahoo Finance (via yfinance library)
- **Real-time Updates**: Data cached for 1 hour to reduce API calls

### Session Data
- All logs stored in browser session only
- No data sent to external servers
- Logs cleared when browser session ends
- Export feature for local storage

### Model Training
- Models trained locally on your machine
- No data sharing or cloud processing
- Training data derived from public stock data only

---

## 💡 Tips & Best Practices

### For Analysis
1. Start with longer time periods (1-2 years) for better context
2. Use multiple technical indicators together for confirmation
3. Compare volume patterns with price movements
4. Look for RSI divergences as potential reversal signals

### For Predictions
1. Shorter timeframes (1 day) generally more accurate than longer ones
2. Higher model confidence (>80%) indicates better training
3. Use predictions as one tool among many, not sole decision factor
4. Compare predictions across multiple runs for consistency
5. Consider market conditions and news events alongside predictions

### For Session Log
1. Export logs regularly for record keeping
2. Review past predictions to evaluate model performance
3. Use filters to focus on specific stocks or activity types
4. Clear log at start of new trading day for fresh tracking

---

## 🚀 Future Enhancements

Potential features for future versions:
- Multiple stock comparison
- Portfolio tracking
- Alert system for price targets
- Historical prediction accuracy tracking
- Custom model parameters
- Additional technical indicators
- News sentiment integration
- Real-time price updates

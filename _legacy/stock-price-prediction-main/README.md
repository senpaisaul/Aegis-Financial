<div align="center">

# 📈 Stock Price Analysis & Prediction

### AI-Powered Stock Market Analysis with Deep Learning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.16+-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 📊 Historical Analysis
- **Interactive Candlestick Charts** with moving averages
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Volume Analysis** with color-coded bars
- **Price Distribution** visualization
- **Multiple timeframes**: 1M to 2Y

</td>
<td width="50%">

### 🔮 AI Predictions
- **Bidirectional LSTM** with Attention mechanism
- **Multiple horizons**: 1 day, 1 week, 1 month
- **Confidence intervals** on predictions
- **Real-time training** with progress tracking
- **Model performance** metrics

</td>
</tr>
<tr>
<td width="50%">

### 📋 Session Logging
- **Activity tracking** for all analyses
- **Export to CSV** for record keeping
- **Visual dashboards** with charts
- **Filter & search** capabilities

</td>
<td width="50%">

### 🎨 User Experience
- **Dark theme** interface
- **Responsive design** for all devices
- **Interactive charts** with Plotly
- **No API keys required** - completely free!

</td>
</tr>
</table>

---

## 🎬 Demo

> **Note**: Add screenshots here after running the app

```bash
# Run the app and take screenshots of:
# 1. Historical Analysis page
# 2. Prediction page with results
# 3. Session Log page
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```


## 💻 Usage

### 1️⃣ Historical Analysis

1. Enter a stock ticker (e.g., `AAPL`, `GOOGL`, `TSLA`)
2. Select time period (1 month to 2 years)
3. Explore interactive charts:
   - Candlestick with moving averages
   - Bollinger Bands
   - RSI & MACD indicators
   - Volume analysis

### 2️⃣ Price Prediction

1. Select prediction timeframe (1 day / 1 week / 1 month)
2. Click "Generate Prediction"
3. Wait 30-60 seconds for model training
4. View forecast with confidence intervals

### 3️⃣ Session Log

1. Review all activities in current session
2. Filter by stock or activity type
3. Export data as CSV
4. Clear log when done

---

## 🧠 Model Architecture

### Bidirectional LSTM with Attention Mechanism

```
Input Layer (60 timesteps × 10 features)
         ↓
Bidirectional LSTM (128 units) + Dropout (0.2)
         ↓
Bidirectional LSTM (64 units) + Dropout (0.2)
         ↓
    Attention Layer
         ↓
Dense (32 units, ReLU) + Dropout (0.2)
         ↓
Dense (16 units, ReLU)
         ↓
    Output (1 unit)
```

### Features Used
- Close Price
- Volume
- SMA (10, 20 days)
- EMA (10 days)
- RSI (14 days)
- MACD & Signal
- Price Change %
- Volume Change %

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Mean Squared Error
- **Validation Split**: 20%
- **Early Stopping**: Patience=10
- **Typical Epochs**: 20-50

---

## 📁 Project Structure

```
stock-price-prediction/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── QUICKSTART.md              # Quick start guide
├── FEATURES.md                # Detailed features documentation
├── .gitignore                 # Git ignore rules
│
├── pages/                     # Streamlit pages
│   ├── __init__.py
│   ├── analysis.py           # Historical analysis page
│   ├── prediction.py         # Prediction page
│   └── session_log.py        # Session log page
│
└── model/                     # ML model
    ├── __init__.py
    └── predictor.py          # LSTM model with attention
```

---

## 🛠️ Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/) - Interactive web interface
- **Data**: [yfinance](https://github.com/ranaroussi/yfinance) - Stock market data
- **ML Framework**: [TensorFlow](https://www.tensorflow.org/) - Deep learning
- **Visualization**: [Plotly](https://plotly.com/) - Interactive charts
- **Technical Analysis**: [TA-Lib](https://github.com/bukosabino/ta) - Indicators
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)

---

## 📊 Performance

- **Training Time**: 30-60 seconds per model
- **Prediction Accuracy**: Varies by stock and timeframe
- **Best Results**: Short-term predictions (1 day)
- **Data Source**: Yahoo Finance (free, no API key)

---

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

- Predictions are generated by machine learning models and should NOT be considered financial advice
- Stock markets are inherently unpredictable and influenced by many factors
- Past performance does not guarantee future results
- Always do your own research and consult with financial advisors before making investment decisions
- The developers are not responsible for any financial losses incurred from using this application

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more technical indicators
- Implement portfolio tracking
- Add real-time price updates
- Improve model architecture
- Add more visualization options
- Write tests

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing free stock data
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [TensorFlow](https://www.tensorflow.org/) team for the ML framework
- All open-source contributors

---

## 📧 Contact

Shatakshi - shatakshiguha@gmail.com

Project Link: [https://github.com/Shatakshi2204/stock-price-prediction](https://github.com/Shatakshi2204/stock-price-prediction)

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

Made with ❤️ and Python

</div>
"# stock-price-prediction" 

import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Stock Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit navigation
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("📈 Stock Price Analysis & Prediction")

# Initialize session state
if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigate", ["📊 Historical Analysis", "🔮 Price Prediction", "📋 Session Log"])

# Show activity count in sidebar
if st.session_state.activity_log:
    st.sidebar.markdown("---")
    st.sidebar.info(f"📋 **Session Activities:** {len(st.session_state.activity_log)}")
    
    # Show last activity
    last_activity = st.session_state.activity_log[-1]
    st.sidebar.caption(f"Last: {last_activity['activity_type']} - {last_activity['ticker']}")

st.sidebar.markdown("---")

# Load data
@st.cache_data(ttl=3600)
def load_stock_data(ticker, period="2y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
data = load_stock_data(ticker)

if data is None or data.empty:
    st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    st.stop()

# Display stock info
try:
    stock_info = yf.Ticker(ticker).info
    company_name = stock_info.get('longName', ticker)
    st.sidebar.success(f"**{company_name}**")
    st.sidebar.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
except:
    st.sidebar.success(f"**{ticker}**")

# Route to appropriate page
if page == "📊 Historical Analysis":
    from pages import analysis
    analysis.show(data, ticker)
elif page == "🔮 Price Prediction":
    from pages import prediction
    prediction.show(data, ticker)
else:
    from pages import session_log
    session_log.show()

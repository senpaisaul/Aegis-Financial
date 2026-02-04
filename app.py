import streamlit as st

# Page Configuration (Must be first)
st.set_page_config(
    page_title="Integrated AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# Import Modules
import modules.credit_risk.ui as credit_risk
import modules.stock_prediction.ui as stock_prediction
import modules.option_pricing.ui as option_pricing

def main():
    # Sidebar Navigation
    st.sidebar.title("🤖 AI Platform")
    st.sidebar.markdown("Integrated Financial & Risk Tools")
    
    module = st.sidebar.radio(
        "Select Module",
        ["🏠 Home", "💳 Credit Risk Analysis", "📈 Stock Prediction", "🧮 Option Pricing"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 | Integrated Architecture")

    # Routing
    if module == "🏠 Home":
        show_home()
    elif module == "💳 Credit Risk Analysis":
        credit_risk.render()
    elif module == "📈 Stock Prediction":
        stock_prediction.render()
    elif module == "🧮 Option Pricing":
        option_pricing.render()

def show_home():
    st.title("Welcome to the Integrated AI Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 💳 Credit Risk\nPredict creditworthiness using Random Forest models.\n\n[Launch Module](?module=credit)")
    
    with col2:
        st.info("### 📈 Stock Prediction\nAnalyze history and predict prices with LSTM + Attention.\n\n[Launch Module](?module=stock)")
        
    with col3:
        st.info("### 🧮 Option Pricing\nCalculate European/American options via BS, Monte Carlo, and Binomial trees.\n\n[Launch Module](?module=option)")
    
    st.markdown("---")
    st.markdown("### 🛠️ Architecture Highlights")
    st.markdown("""
    - **Modular Monolith**: Separated concerns into UI (`modules`), Logic (`services`), and Data (`assets`).
    - **Unified Utilities**: Shared market data fetching and caching.
    - **Optimized Performance**: Lazy loading of heavy models and dependencies.
    """)

if __name__ == "__main__":
    main()

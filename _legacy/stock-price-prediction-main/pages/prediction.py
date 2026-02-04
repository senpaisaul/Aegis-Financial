import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model.predictor import StockPredictor

def show(data, ticker):
    st.header(f"🔮 Price Prediction - {ticker}")
    
    st.info("🤖 Using Bidirectional LSTM with Attention Mechanism for predictions")
    
    # Prediction timeframe selector
    col1, col2 = st.columns([3, 1])
    with col1:
        timeframe = st.selectbox(
            "Select Prediction Timeframe",
            ["1 Day", "1 Week", "1 Month"],
            index=0
        )
    
    with col2:
        st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Generate Prediction", type="primary", use_container_width=True):
            st.session_state.generate_prediction = True
    
    # Map timeframe to days
    timeframe_days = {
        "1 Day": 1,
        "1 Week": 7,
        "1 Month": 30
    }
    days = timeframe_days[timeframe]
    
    # Generate prediction
    if st.session_state.get('generate_prediction', False):
        with st.spinner(f"Training model and generating {timeframe.lower()} prediction..."):
            try:
                # Initialize predictor
                predictor = StockPredictor(sequence_length=60)
                
                # Train model
                history = predictor.train(data)
                
                # Make predictions
                predictions = predictor.predict(data, days=days)
                
                # Log this prediction activity
                from pages.session_log import log_activity
                current_price = data['Close'].iloc[-1]
                predicted_price = predictions[-1]
                final_val_loss = history.history['val_loss'][-1]
                model_confidence = max(0, (1 - final_val_loss) * 100)
                
                log_activity(
                    activity_type='Prediction',
                    ticker=ticker,
                    timeframe=timeframe,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    model_confidence=model_confidence,
                    training_loss=history.history['loss'][-1],
                    validation_loss=final_val_loss
                )
                
                st.toast(f"✅ Prediction logged for {ticker}", icon="🔮")
                
                # Display results
                display_predictions(data, predictions, ticker, timeframe, history)
                
            except Exception as e:
                st.error(f"Error generating prediction: {e}")
                st.exception(e)
    else:
        st.info("👆 Click 'Generate Prediction' to start the forecasting process")
        
        # Show model architecture
        st.subheader("🧠 Model Architecture")
        st.markdown("""
        **Bidirectional LSTM with Attention Mechanism**
        
        - **Input Layer**: Historical price data with technical indicators
        - **Bidirectional LSTM**: Captures patterns from both past and future contexts
        - **Attention Layer**: Focuses on the most relevant time steps
        - **Dense Layers**: Final prediction layers
        - **Output**: Future price predictions
        
        **Features Used**:
        - Close Price
        - Volume
        - Moving Averages (SMA, EMA)
        - RSI (Relative Strength Index)
        - MACD
        """)

def display_predictions(data, predictions, ticker, timeframe, history):
    """Display prediction results with charts"""
    
    st.success(f"✅ Prediction generated successfully!")
    
    # Training metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    with col1:
        st.metric("Training Loss", f"{final_loss:.6f}")
    with col2:
        st.metric("Validation Loss", f"{final_val_loss:.6f}")
    with col3:
        accuracy = max(0, (1 - final_val_loss) * 100)
        st.metric("Model Confidence", f"{accuracy:.1f}%")
    
    # Training history chart
    st.subheader("📈 Training History")
    fig_history = go.Figure()
    
    fig_history.add_trace(go.Scatter(
        y=history.history['loss'],
        name='Training Loss',
        line=dict(color='blue', width=2)
    ))
    fig_history.add_trace(go.Scatter(
        y=history.history['val_loss'],
        name='Validation Loss',
        line=dict(color='orange', width=2)
    ))
    
    fig_history.update_layout(
        title="Model Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=300,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_history, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction results
    st.subheader(f"🎯 {timeframe} Price Forecast")
    
    # Create prediction dataframe
    last_date = data.index[-1]
    pred_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                periods=len(predictions), freq='D')
    pred_df = pd.DataFrame({
        'Date': pred_dates,
        'Predicted_Price': predictions
    })
    
    # Display prediction metrics
    current_price = data['Close'].iloc[-1]
    predicted_price = predictions[-1]
    price_change = predicted_price - current_price
    pct_change = (price_change / current_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Predicted Price", f"${predicted_price:.2f}")
    with col3:
        st.metric("Expected Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
    with col4:
        trend = "📈 Bullish" if price_change > 0 else "📉 Bearish"
        st.metric("Trend", trend)
    
    # Prediction chart
    st.subheader("📊 Price Forecast Visualization")
    
    # Combine historical and predicted data
    historical_last_30 = data.tail(60)
    
    fig_pred = go.Figure()
    
    # Historical prices
    fig_pred.add_trace(go.Scatter(
        x=historical_last_30.index,
        y=historical_last_30['Close'],
        name='Historical Price',
        line=dict(color='cyan', width=2)
    ))
    
    # Predicted prices
    fig_pred.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted_Price'],
        name='Predicted Price',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Add confidence interval (simple estimation)
    std_dev = np.std(predictions) * 0.5
    fig_pred.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted_Price'] + std_dev,
        name='Upper Bound',
        line=dict(color='rgba(255,165,0,0.3)', width=1),
        showlegend=False
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted_Price'] - std_dev,
        name='Lower Bound',
        line=dict(color='rgba(255,165,0,0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.2)',
        showlegend=False
    ))
    
    fig_pred.update_layout(
        title=f"{ticker} - {timeframe} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Prediction table
    st.subheader("📋 Detailed Predictions")
    
    display_df = pred_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"${x:.2f}")
    display_df['Change_from_Current'] = [
        f"${p - current_price:.2f} ({((p - current_price) / current_price * 100):.2f}%)"
        for p in predictions
    ]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: These predictions are generated by a machine learning model and should not be considered as financial advice. 
    Stock prices are influenced by many factors and actual results may vary significantly. Always do your own research and 
    consult with financial advisors before making investment decisions.
    """)

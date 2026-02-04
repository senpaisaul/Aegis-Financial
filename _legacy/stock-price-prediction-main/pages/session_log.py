import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def show():
    st.header("📋 Session Activity Log")
    
    # Initialize session state for logs if not exists
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    
    if not st.session_state.activity_log:
        st.info("No activities logged yet. Start analyzing or predicting stocks to see them here!")
        return
    
    # Convert log to dataframe
    log_df = pd.DataFrame(st.session_state.activity_log)
    
    # Summary metrics
    st.subheader("📊 Session Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_activities = len(log_df)
        st.metric("Total Activities", total_activities)
    
    with col2:
        unique_stocks = log_df['ticker'].nunique()
        st.metric("Unique Stocks", unique_stocks)
    
    with col3:
        analysis_count = len(log_df[log_df['activity_type'] == 'Analysis'])
        st.metric("Analyses", analysis_count)
    
    with col4:
        prediction_count = len(log_df[log_df['activity_type'] == 'Prediction'])
        st.metric("Predictions", prediction_count)
    
    st.markdown("---")
    
    # Activity breakdown chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Activity Type Distribution")
        activity_counts = log_df['activity_type'].value_counts()
        fig1 = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        fig1.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Stocks Analyzed")
        stock_counts = log_df['ticker'].value_counts().head(10)
        fig2 = px.bar(
            x=stock_counts.index,
            y=stock_counts.values,
            labels={'x': 'Stock Ticker', 'y': 'Count'},
            color=stock_counts.values,
            color_continuous_scale='Blues'
        )
        fig2.update_layout(template='plotly_dark', height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Filter options
    st.subheader("🔍 Filter Activities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.multiselect(
            "Activity Type",
            options=log_df['activity_type'].unique(),
            default=log_df['activity_type'].unique()
        )
    
    with col2:
        filter_ticker = st.multiselect(
            "Stock Ticker",
            options=sorted(log_df['ticker'].unique()),
            default=sorted(log_df['ticker'].unique())
        )
    
    with col3:
        sort_order = st.selectbox(
            "Sort By",
            ["Newest First", "Oldest First"],
            index=0
        )
    
    # Apply filters
    filtered_df = log_df[
        (log_df['activity_type'].isin(filter_type)) &
        (log_df['ticker'].isin(filter_ticker))
    ]
    
    if sort_order == "Oldest First":
        filtered_df = filtered_df.sort_values('timestamp')
    else:
        filtered_df = filtered_df.sort_values('timestamp', ascending=False)
    
    st.markdown("---")
    
    # Display detailed log
    st.subheader("📝 Detailed Activity Log")
    
    if filtered_df.empty:
        st.warning("No activities match the selected filters.")
    else:
        for idx, row in filtered_df.iterrows():
            with st.expander(
                f"{'📊' if row['activity_type'] == 'Analysis' else '🔮'} "
                f"{row['ticker']} - {row['activity_type']} - {row['timestamp']}"
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Ticker:** {row['ticker']}")
                    st.markdown(f"**Type:** {row['activity_type']}")
                    st.markdown(f"**Time:** {row['timestamp']}")
                
                with col2:
                    if row['activity_type'] == 'Analysis':
                        st.markdown(f"**Period:** {row.get('period', 'N/A')}")
                        st.markdown(f"**Price Range:** ${row.get('price_low', 0):.2f} - ${row.get('price_high', 0):.2f}")
                        st.markdown(f"**Current Price:** ${row.get('current_price', 0):.2f}")
                    else:  # Prediction
                        st.markdown(f"**Timeframe:** {row.get('timeframe', 'N/A')}")
                        st.markdown(f"**Current Price:** ${row.get('current_price', 0):.2f}")
                        st.markdown(f"**Predicted Price:** ${row.get('predicted_price', 0):.2f}")
                        change = row.get('predicted_price', 0) - row.get('current_price', 0)
                        pct_change = (change / row.get('current_price', 1)) * 100
                        st.markdown(f"**Expected Change:** ${change:.2f} ({pct_change:.2f}%)")
                        st.markdown(f"**Model Confidence:** {row.get('model_confidence', 0):.1f}%")
    
    st.markdown("---")
    
    # Export and clear options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("📥 Export Log", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"stock_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🗑️ Clear Log", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                st.session_state.activity_log = []
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all logs")

def log_activity(activity_type, ticker, **kwargs):
    """Helper function to log an activity"""
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'activity_type': activity_type,
        'ticker': ticker,
        **kwargs
    }
    
    st.session_state.activity_log.append(log_entry)

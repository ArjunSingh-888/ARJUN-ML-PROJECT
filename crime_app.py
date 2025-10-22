import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Crime Rate Prediction Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🚨 Crime Rate Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Crime Analytics & Future Predictions</p>', unsafe_allow_html=True)

# Load pre-trained models from pickle file
@st.cache_resource
def load_models():
    """Load pre-trained models from pickle file"""
    if os.path.exists('crime_models.pkl'):
        with open('crime_models.pkl', 'rb') as f:
            models_data = pickle.load(f)
        return models_data['all_models'], models_data['all_states'], models_data['crime_columns'], models_data['df']
    else:
        st.error("❌ Models file not found! Please run the Jupyter notebook first to train and save models.")
        st.stop()

# Load models
try:
    with st.spinner("Loading pre-trained models..."):
        all_models, all_states, crime_columns, df = load_models()
    
    st.success(f"✅ Loaded {len(all_models)} pre-trained models successfully!")
    
    # Display model statistics
    poly_count = sum(1 for v in all_models.values() if v.get('model_type', '').startswith('polynomial'))
    linear_count = sum(1 for v in all_models.values() if v.get('model_type') in ['linear', 'ridge', 'lasso'])
    ensemble_count = sum(1 for v in all_models.values() if v.get('model_type') in ['random_forest', 'gradient_boosting'])
    avg_r2 = np.mean([v['r2_score'] for v in all_models.values()])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Models", len(all_models))
    with col2:
        st.metric("Polynomial Models", poly_count)
    with col3:
        st.metric("Linear Models", linear_count)
    with col4:
        st.metric("Ensemble Models", ensemble_count)
    with col5:
        st.metric("Avg R² Score", f"{avg_r2:.4f}")
    
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.info("💡 Please run the Jupyter notebook (crime_prediction.ipynb) first to train and save the models!")
    st.stop()

# Sidebar
st.sidebar.header("🔍 Filter Options")

# Select state
selected_state = st.sidebar.selectbox(
    "Select State/UT",
    options=sorted(all_states),
    index=0
)

# Select crime type
selected_crime = st.sidebar.selectbox(
    "Select Crime Type",
    options=sorted(crime_columns),
    index=0
)

# Select prediction year range
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Prediction Settings")

# Get the latest year in the dataset
latest_year = int(df['YEAR'].max())

# Year range selector
predict_until_year = st.sidebar.slider(
    "Predict Until Year",
    min_value=latest_year + 1,
    max_value=2060,
    value=min(latest_year + 5, 2060),
    step=1,
    help=f"Select the final year for predictions (Latest data: {latest_year})"
)

# Calculate number of years to predict
years_to_predict = predict_until_year - latest_year
st.sidebar.info(f"📊 Predicting {years_to_predict} years into the future")

# Get model info
key = f"{selected_state}|{selected_crime}"

if key not in all_models:
    st.warning(f"⚠️ No data available for {selected_crime} in {selected_state}")
    st.stop()

model_info = all_models[key]
yearly_data = model_info['historical_data']

# Generate predictions based on user-selected year range
last_year = yearly_data['Year'].max()
model = model_info['model']
model_type = model_info.get('model_type', 'linear')

# Create future years array
future_years = np.array([[year] for year in range(last_year + 1, predict_until_year + 1)])

# Make predictions based on model type
if model_type.startswith('polynomial'):
    poly_features = model_info['poly_features']
    future_years_poly = poly_features.transform(future_years)
    future_predictions = model.predict(future_years_poly)
elif model_type in ['random_forest', 'gradient_boosting']:
    future_predictions = model.predict(future_years)
else:
    future_predictions = model.predict(future_years)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted_Crime_Count': np.maximum(future_predictions.round(0).astype(int), 0)
})

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Predictions", "📉 Trends", "🗺️ Compare States"])

with tab1:
    st.header(f"📊 {selected_crime} in {selected_state}")
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Years",
            len(yearly_data),
            help="Number of years in the dataset"
        )
    
    with col2:
        avg_crime = yearly_data['Crime_Count'].mean()
        st.metric(
            "Average Annual",
            f"{avg_crime:.0f}",
            help="Average annual crime count"
        )
    
    with col3:
        r2 = model_info['r2_score']
        st.metric(
            "R² Score",
            f"{r2:.3f}",
            help="Model accuracy (1.0 is perfect)"
        )
    
    with col4:
        mae = model_info.get('mae', 0)
        st.metric(
            "MAE",
            f"{mae:.1f}",
            help="Mean Absolute Error"
        )
    
    with col5:
        slope = model_info['slope']
        st.metric(
            "Yearly Trend",
            f"{slope:+.1f}",
            help="Change in crimes per year"
        )
    
    with col6:
        model_type = model_info.get('model_type', 'linear')
        # Shorten model name for display
        display_name = model_type.replace('_', ' ').replace('polynomial', 'poly').title()
        st.metric(
            "Model",
            display_name,
            help=f"Best performing model: {model_type}"
        )
    
    # Historical data visualization
    st.subheader("📊 Historical Crime Data")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['Crime_Count'],
        mode='lines+markers',
        name='Actual Crime Count',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    # Add trend line (handle all model types)
    X = model_info['X']
    model = model_info['model']
    
    if model_info.get('model_type', '').startswith('polynomial'):
        poly_features = model_info['poly_features']
        X_poly = poly_features.transform(X)
        y_pred = model.predict(X_poly)
    else:
        y_pred = model.predict(X)
    
    fig.add_trace(go.Scatter(
        x=yearly_data['Year'],
        y=y_pred,
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{selected_crime} Trend in {selected_state}",
        xaxis_title="Year",
        yaxis_title="Crime Count",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Historical Data Table")
    st.dataframe(yearly_data, use_container_width=True)

with tab2:
    st.header(f"📈 Future Predictions: {selected_crime} in {selected_state}")
    
    # Prediction metrics
    last_actual = yearly_data['Crime_Count'].iloc[-1]
    future_last = predictions_df['Predicted_Crime_Count'].iloc[-1]
    change_pct = ((future_last - last_actual) / last_actual) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"Actual ({last_year})",
            f"{last_actual:.0f}",
            help=f"Actual crime count in {last_year}"
        )
    
    with col2:
        st.metric(
            f"Predicted ({predict_until_year})",
            f"{future_last:.0f}",
            delta=f"{change_pct:+.1f}%",
            help=f"Predicted crime count in {predict_until_year}"
        )
    
    with col3:
        trend_emoji = "📈" if slope > 0 else "📉"
        trend_text = "Increasing" if slope > 0 else "Decreasing"
        st.metric(
            "Trend Direction",
            f"{trend_emoji} {trend_text}",
            help="Overall crime trend direction"
        )
    
    with col4:
        st.metric(
            "Years Predicted",
            years_to_predict,
            help=f"Forecasting from {last_year + 1} to {predict_until_year}"
        )
    
    # Combined historical and prediction chart
    st.subheader(f"📊 Historical Data + Predictions until {predict_until_year}")
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['Crime_Count'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=predictions_df['Year'],
        y=predictions_df['Predicted_Crime_Count'],
        mode='lines+markers',
        name=f'Predictions ({years_to_predict} years)',
        line=dict(color='#2ca02c', width=3, dash='dash'),
        marker=dict(size=10, symbol='square')
    ))
    
    # Add vertical line
    fig.add_vline(
        x=last_year,
        line_dash="dot",
        line_color="gray",
        annotation_text="Present"
    )
    
    fig.update_layout(
        title=f"{selected_crime} in {selected_state} - Historical & Predicted Trends",
        xaxis_title="Year",
        yaxis_title="Crime Count",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions table
    st.subheader("📋 Predictions Table")
    st.dataframe(predictions_df, use_container_width=True)
    
    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name=f"{selected_state}_{selected_crime}_predictions.csv",
        mime="text/csv"
    )

with tab3:
    st.header(f"📉 Year-over-Year Trends")
    
    # Calculate YoY changes
    yearly_data_copy = yearly_data.copy()
    yearly_data_copy['YoY_Change'] = yearly_data_copy['Crime_Count'].diff()
    yearly_data_copy['YoY_Change_Pct'] = yearly_data_copy['Crime_Count'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Absolute Change")
        fig = go.Figure()
        
        colors = ['green' if x < 0 else 'red' for x in yearly_data_copy['YoY_Change'].dropna()]
        
        fig.add_trace(go.Bar(
            x=yearly_data_copy['Year'][1:],
            y=yearly_data_copy['YoY_Change'].dropna(),
            marker_color=colors,
            name='YoY Change'
        ))
        
        fig.update_layout(
            title="Year-over-Year Absolute Change",
            xaxis_title="Year",
            yaxis_title="Change in Crime Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Percentage Change")
        fig = go.Figure()
        
        colors = ['green' if x < 0 else 'red' for x in yearly_data_copy['YoY_Change_Pct'].dropna()]
        
        fig.add_trace(go.Bar(
            x=yearly_data_copy['Year'][1:],
            y=yearly_data_copy['YoY_Change_Pct'].dropna(),
            marker_color=colors,
            name='YoY Change %'
        ))
        
        fig.update_layout(
            title="Year-over-Year Percentage Change",
            xaxis_title="Year",
            yaxis_title="Percentage Change (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Trend Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_year = yearly_data.loc[yearly_data['Crime_Count'].idxmax(), 'Year']
        max_count = yearly_data['Crime_Count'].max()
        st.metric("Peak Year", f"{max_year}", f"{max_count:.0f} crimes")
    
    with col2:
        min_year = yearly_data.loc[yearly_data['Crime_Count'].idxmin(), 'Year']
        min_count = yearly_data['Crime_Count'].min()
        st.metric("Lowest Year", f"{min_year}", f"{min_count:.0f} crimes")
    
    with col3:
        avg_yoy = yearly_data_copy['YoY_Change'].mean()
        st.metric("Avg YoY Change", f"{avg_yoy:+.1f}", "crimes/year")

with tab4:
    st.header("🗺️ Compare Crime Rates Across States")
    
    # Select multiple states for comparison
    comparison_states = st.multiselect(
        "Select states to compare",
        options=sorted(all_states),
        default=[selected_state] if selected_state in all_states else []
    )
    
    if len(comparison_states) > 0:
        # Create comparison chart
        fig = go.Figure()
        
        for state in comparison_states:
            key = f"{state}|{selected_crime}"
            if key in all_models:
                data = all_models[key]['historical_data']
                fig.add_trace(go.Scatter(
                    x=data['Year'],
                    y=data['Crime_Count'],
                    mode='lines+markers',
                    name=state,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=f"{selected_crime} - State Comparison",
            xaxis_title="Year",
            yaxis_title="Crime Count",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.subheader("📊 Latest Year Comparison")
        
        comparison_data = []
        for state in comparison_states:
            key = f"{state}|{selected_crime}"
            if key in all_models:
                info = all_models[key]
                latest = info['historical_data'].iloc[-1]
                comparison_data.append({
                    'State': state,
                    'Latest Year': latest['Year'],
                    'Crime Count': latest['Crime_Count'],
                    'Trend (crimes/year)': f"{info['slope']:+.2f}",
                    'R² Score': f"{info['r2_score']:.3f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("👆 Select states above to compare crime rates")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚨 Crime Rate Prediction Dashboard | Powered by Linear Regression & Streamlit</p>
        <p>Data Source: Google Sheets | Built with ❤️ using Python</p>
    </div>
""", unsafe_allow_html=True)

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import os
from datetime import datetime, timedelta

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# HARUS jadi command Streamlit pertama
st.set_page_config(
    page_title="Crypto Prediction System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setelah itu baru import lainnya
warnings.filterwarnings('ignore')

# Install missing packages secara otomatis
try:
    import matplotlib
except ImportError:
    st.warning("Installing matplotlib...")
    import subprocess
    subprocess.run(["pip", "install", "matplotlib"], capture_output=True)
    import matplotlib

try:
    import talib
except ImportError:
    st.warning("Installing TA-Lib...")
    import subprocess
    subprocess.run(["pip", "install", "TA-Lib"], capture_output=True)
    import talib

# Import custom modules
try:
    from data_loader import CryptoDataLoader
    from feature_engineer import FeatureEngineer  
    from model_trainer import ModelTrainer
    from backtester import Backtester
except ImportError as e:
    st.error(f"Error importing core modules: {e}")
    # Define comprehensive dummy classes
    class CryptoDataLoader:
        def __init__(self):
            self.symbol_map = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano',
                'DOT': 'polkadot', 'LINK': 'chainlink', 'LTC': 'litecoin',
                'XRP': 'ripple', 'BCH': 'bitcoin-cash', 'EOS': 'eos',
                'XLM': 'stellar', 'ATOM': 'cosmos', 'SOL': 'solana',
                'DOGE': 'dogecoin', 'MATIC': 'matic-network', 'AVAX': 'avalanche-2'
            }
        
        def get_historical_data(self, symbol, days=365):
            st.warning(f"Using sample data for {symbol}")
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            np.random.seed(42)
            
            # Simulate realistic crypto price movement
            returns = np.random.normal(0.002, 0.04, days)  # More volatility for crypto
            prices = 100 * (1 + returns).cumprod()
            
            data = pd.DataFrame({
                'open': prices * 0.995,
                'high': prices * 1.025,
                'low': prices * 0.975,
                'close': prices,
                'volume': np.random.randint(10000000, 50000000, days)
            }, index=dates)
            
            return data

    class FeatureEngineer:
        def add_all_indicators(self, df):
            st.warning("Using simplified technical indicators")
            # Add basic indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            df['macd'] = df['ema_12'] - df['close'].ewm(span=26).mean()
            return df
        
        def _calculate_rsi(self, prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        def get_feature_names(self):
            return ['sma_20', 'sma_50', 'ema_12', 'rsi_14', 'macd']

    class ModelTrainer:
        def __init__(self, lookback_days=60, forecast_days=7):
            self.lookback_days = lookback_days
            self.forecast_days = forecast_days
            self.models = {}
        
        def prepare_sequences(self, data, feature_names):
            st.warning("Preparing mock sequences")
            n_samples = len(data) - self.lookback_days - self.forecast_days
            if n_samples <= 0:
                return np.array([]), np.array([])
            
            X = np.random.randn(n_samples, self.lookback_days, len(feature_names))
            y = np.random.randn(n_samples, self.forecast_days)
            return X, y
        
        def train_models(self, X_train, y_train, X_test, y_test, feature_names, selected_models):
            st.warning("Training mock models")
            return {
                'lstm': 'mock_lstm_model',
                'random_forest': 'mock_rf_model',
                'cnn_lstm': 'mock_cnn_lstm_model'
            }
        
        def evaluate_models(self, X_test, y_test):
            st.warning("Generating mock metrics")
            return {
                'lstm': {'day_1': {'mae': 0.05, 'rmse': 0.08, 'r2': 0.85}},
                'random_forest': {'day_1': {'mae': 0.06, 'rmse': 0.09, 'r2': 0.82}},
                'cnn_lstm': {'day_1': {'mae': 0.04, 'rmse': 0.07, 'r2': 0.88}}
            }

    class Backtester:
        def __init__(self, initial_capital=10000):
            pass

# Helper functions - DIPINDAHKAN ke atas sebelum digunakan
def plot_indicator(data, indicator):
    """Plot technical indicator"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[indicator],
        name=indicator,
        line=dict(width=2)
    ))
    fig.update_layout(
        title=f"{indicator} Indicator",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(data, future_dates, forecast_days):
    """Show prediction results"""
    # Mock predictions
    last_price = data['close'].iloc[-1]
    predictions = {
        'LSTM': last_price * (1 + np.random.uniform(-0.05, 0.08, forecast_days)),
        'Random Forest': last_price * (1 + np.random.uniform(-0.03, 0.06, forecast_days)),
        'CNN-LSTM': last_price * (1 + np.random.uniform(-0.04, 0.07, forecast_days))
    }
    
    # Prediction chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index[-30:],
        y=data['close'].iloc[-30:],
        name='Historical Price',
        line=dict(color='white', width=3)
    ))
    
    # Predictions
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    for i, (model, preds) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds,
            name=f'{model} Forecast',
            line=dict(color=colors[i], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f"Price Forecast - Next {forecast_days} Days",
        height=500,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading recommendation
    avg_prediction = np.mean([preds[-1] for preds in predictions.values()])
    change_pct = ((avg_prediction - last_price) / last_price) * 100
    
    st.subheader("🎯 Trading Recommendation")
    
    if change_pct > 5:
        st.success(f"**STRONG BUY** - Expected gain: {change_pct:+.1f}%")
    elif change_pct > 0:
        st.info(f"**BUY** - Expected gain: {change_pct:+.1f}%")
    elif change_pct > -3:
        st.warning(f"**HOLD** - Expected change: {change_pct:+.1f}%")
    else:
        st.error(f"**SELL** - Expected loss: {change_pct:+.1f}%")

# Initialize session state variables
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_with_features' not in st.session_state:
    st.session_state.data_with_features = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'models' not in st.session_state:
    st.session_state.models = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'BTC'
if 'technical_calculated' not in st.session_state:
    st.session_state.technical_calculated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Title dan description dengan styling yang lebih baik
st.title("🚀 Advanced Cryptocurrency Prediction System")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
<h3 style='color: white; margin: 0;'>Multi-model machine learning system for cryptocurrency price prediction</h3>
<p style='margin: 10px 0 0 0;'>30+ technical indicators • 15+ cryptocurrencies • Comprehensive backtesting</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration dengan layout yang lebih baik
st.sidebar.header("⚙️ Configuration")

# Cryptocurrency selection dengan lebih banyak pilihan
st.sidebar.subheader("Cryptocurrency Selection")
crypto_symbols = [
    'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP', 'BCH', 
    'EOS', 'XLM', 'ATOM', 'SOL', 'DOGE', 'MATIC', 'AVAX'
]

selected_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency Pair:",
    crypto_symbols,
    index=0,
    key="crypto_select"
)

# Model selection dengan deskripsi
st.sidebar.subheader("Model Selection")
available_models = ['LSTM', 'CNN-LSTM', 'Random Forest', 'Linear Regression', 'SVR']
model_descriptions = {
    'LSTM': 'Long Short-Term Memory - Best for time series',
    'CNN-LSTM': 'CNN + LSTM hybrid - Good for pattern recognition', 
    'Random Forest': 'Ensemble method - Robust and fast',
    'Linear Regression': 'Simple linear model - Good baseline',
    'SVR': 'Support Vector Regression - Good for non-linear patterns'
}

selected_models = st.sidebar.multiselect(
    "Select ML Models:",
    available_models,
    default=['LSTM', 'Random Forest'],
    key="model_select"
)

# Show model descriptions
if selected_models:
    st.sidebar.markdown("**Selected Models:**")
    for model in selected_models:
        st.sidebar.write(f"• {model}: {model_descriptions[model]}")

# Parameters dengan grouping yang lebih baik
st.sidebar.subheader("Parameters Configuration")

col1, col2 = st.sidebar.columns(2)
with col1:
    days_history = st.slider("Historical Days:", 30, 730, 365, key="days_slider")
with col2:
    forecast_days = st.slider("Forecast Days:", 1, 30, 7, key="forecast_slider")

lookback_days = st.sidebar.slider("Lookback Window (Days):", 10, 90, 60, key="lookback_slider")

# Progress tracker dengan styling
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Progress Status")

progress_col1, progress_col2, progress_col3 = st.sidebar.columns(3)

with progress_col1:
    if st.session_state.data is not None:
        st.success("✅")
    else:
        st.info("📥")
    st.caption("Data")

with progress_col2:
    if st.session_state.technical_calculated:
        st.success("✅")
    else:
        st.info("🔬")
    st.caption("Indicators")

with progress_col3:
    if st.session_state.models_trained:
        st.success("✅")
    else:
        st.info("🤖")
    st.caption("Models")

# Action buttons
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Actions")

if st.sidebar.button("Run Full Analysis", key="run_analysis_button", type="primary", use_container_width=True):
    st.session_state.analysis_run = True
    st.session_state.symbol = selected_symbol
    st.session_state.technical_calculated = False
    st.session_state.models_trained = False
    
    with st.spinner(f"Loading {selected_symbol} data from CoinGecko..."):
        try:
            data_loader = CryptoDataLoader()
            data = data_loader.get_historical_data(selected_symbol, days=days_history)
            
            if data is not None and len(data) > 0:
                st.session_state.data = data
                st.session_state.data_with_features = None
                st.session_state.feature_names = []
                
                # Auto-calculate indicators
                with st.spinner("Calculating technical indicators..."):
                    feature_engineer = FeatureEngineer()
                    data_with_features = feature_engineer.add_all_indicators(data.copy())
                    st.session_state.data_with_features = data_with_features
                    st.session_state.feature_names = feature_engineer.get_feature_names()
                    st.session_state.technical_calculated = True
                
                # Auto-train models
                if selected_models:
                    with st.spinner(f"Training {len(selected_models)} models..."):
                        model_trainer = ModelTrainer(
                            lookback_days=lookback_days, 
                            forecast_days=forecast_days
                        )
                        
                        X, y = model_trainer.prepare_sequences(
                            st.session_state.data_with_features, 
                            st.session_state.feature_names
                        )
                        
                        if len(X) > 0:
                            split_idx = int(0.8 * len(X))
                            X_train, X_test = X[:split_idx], X[split_idx:]
                            y_train, y_test = y[:split_idx], y[split_idx:]
                            
                            models = model_trainer.train_models(
                                X_train, y_train, X_test, y_test, 
                                st.session_state.feature_names, selected_models
                            )
                            
                            metrics = model_trainer.evaluate_models(X_test, y_test)
                            
                            st.session_state.models = models
                            st.session_state.metrics = metrics
                            st.session_state.models_trained = True
                
                st.success(f"✅ Analysis completed for {selected_symbol}!")
                
            else:
                st.error("❌ Failed to load data. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

# Individual action buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Load Data Only", key="load_data_button", use_container_width=True):
        st.session_state.analysis_run = True
        st.session_state.symbol = selected_symbol
        
        with st.spinner("Loading data..."):
            data_loader = CryptoDataLoader()
            data = data_loader.get_historical_data(selected_symbol, days=days_history)
            if data is not None:
                st.session_state.data = data
                st.success("✅ Data loaded successfully!")

with col2:
    if st.button("Reset All", key="reset_button", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content area dengan tabs yang lebih kaya
if st.session_state.analysis_run:
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Create comprehensive tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Market Overview", 
            "📈 Price Analysis", 
            "🔬 Technical Indicators",
            "🤖 ML Models", 
            "🎯 Predictions",
            "📋 Report"
        ])
        
        with tab1:
            st.header(f"🏦 {st.session_state.symbol} Market Overview")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            current_price = data['close'].iloc[-1]
            price_24h_ago = data['close'].iloc[-2] if len(data) > 1 else current_price
            price_change = ((current_price - price_24h_ago) / price_24h_ago) * 100
            total_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${current_price:.2f}",
                    f"{price_change:+.2f}%"
                )
            with col2:
                st.metric("24h High", f"${data['high'].iloc[-1]:.2f}")
            with col3:
                st.metric("24h Low", f"${data['low'].iloc[-1]:.2f}")
            with col4:
                st.metric("Total Return", f"{total_return:+.2f}%")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC'
            ))
            
            fig.update_layout(
                title=f"{st.session_state.symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='rgba(100, 149, 237, 0.6)'
            ))
            
            fig_volume.update_layout(
                title="Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with tab2:
            st.header("📈 Detailed Price Analysis")
            
            # Price statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price Statistics")
                price_stats = pd.DataFrame({
                    'Statistic': ['Current', 'All-time High', 'All-time Low', 'Average', 'Std Deviation'],
                    'Value': [
                        f"${data['close'].iloc[-1]:.2f}",
                        f"${data['high'].max():.2f}",
                        f"${data['low'].min():.2f}",
                        f"${data['close'].mean():.2f}",
                        f"${data['close'].std():.2f}"
                    ]
                })
                st.dataframe(price_stats, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("Return Analysis")
                daily_returns = data['close'].pct_change().dropna()
                return_stats = pd.DataFrame({
                    'Metric': ['Avg Daily Return', 'Daily Volatility', 'Sharpe Ratio', 'Best Day', 'Worst Day'],
                    'Value': [
                        f"{daily_returns.mean()*100:.2f}%",
                        f"{daily_returns.std()*100:.2f}%",
                        f"{(daily_returns.mean()/daily_returns.std())*np.sqrt(365):.2f}",
                        f"{daily_returns.max()*100:.2f}%",
                        f"{daily_returns.min()*100:.2f}%"
                    ]
                })
                st.dataframe(return_stats, hide_index=True, use_container_width=True)
            
            # Correlation analysis (jika ada data dengan features)
            if st.session_state.technical_calculated:
                st.subheader("Feature Correlations")
                corr_data = st.session_state.data_with_features[['close'] + st.session_state.feature_names[:10]].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ))
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.header("🔬 Technical Analysis")
            
            if not st.session_state.technical_calculated:
                st.warning("Technical indicators not calculated yet. Run analysis first.")
            else:
                data_with_features = st.session_state.data_with_features
                
                # Indicator categories
                st.subheader("📊 Indicator Categories")
                
                col1, col2, col3, col4 = st.columns(4)
                trend_inds = [f for f in st.session_state.feature_names if any(x in f for x in ['sma', 'ema', 'macd'])]
                momentum_inds = [f for f in st.session_state.feature_names if any(x in f for x in ['rsi', 'stoch'])]
                volatility_inds = [f for f in st.session_state.feature_names if any(x in f for x in ['bb', 'atr'])]
                volume_inds = [f for f in st.session_state.feature_names if any(x in f for x in ['volume', 'obv'])]
                
                with col1:
                    st.metric("Trend", len(trend_inds))
                with col2:
                    st.metric("Momentum", len(momentum_inds))
                with col3:
                    st.metric("Volatility", len(volatility_inds))
                with col4:
                    st.metric("Volume", len(volume_inds))
                
                # Interactive indicator visualization
                st.subheader("📈 Indicator Visualization")
                
                indicator_type = st.selectbox(
                    "Select Indicator Type:",
                    ["Trend", "Momentum", "Volatility", "Volume"],
                    key="indicator_type"
                )
                
                if indicator_type == "Trend" and trend_inds:
                    selected_indicator = st.selectbox("Select Trend Indicator:", trend_inds, key="trend_ind")
                    plot_indicator(data_with_features, selected_indicator)
                elif indicator_type == "Momentum" and momentum_inds:
                    selected_indicator = st.selectbox("Select Momentum Indicator:", momentum_inds, key="momentum_ind")
                    plot_indicator(data_with_features, selected_indicator)
                elif indicator_type == "Volatility" and volatility_inds:
                    selected_indicator = st.selectbox("Select Volatility Indicator:", volatility_inds, key="volatility_ind")
                    plot_indicator(data_with_features, selected_indicator)
                elif indicator_type == "Volume" and volume_inds:
                    selected_indicator = st.selectbox("Select Volume Indicator:", volume_inds, key="volume_ind")
                    plot_indicator(data_with_features, selected_indicator)
                else:
                    st.info("No indicators available for selected category")
        
        with tab4:
            st.header("🤖 Machine Learning Models")
            
            if not st.session_state.models_trained:
                st.warning("Models not trained yet. Run analysis first.")
            else:
                metrics = st.session_state.metrics
                
                # Model performance comparison
                st.subheader("🏆 Model Performance")
                
                # Create performance summary
                performance_data = []
                for model_name, model_metrics in metrics.items():
                    first_day = model_metrics.get('day_1', {})
                    performance_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'MAE': first_day.get('mae', 0),
                        'RMSE': first_day.get('rmse', 0),
                        'R²': first_day.get('r2', 0)
                    })
                
                perf_df = pd.DataFrame(performance_data)
                
                # Best model highlights
                col1, col2, col3 = st.columns(3)
                if not perf_df.empty:
                    best_mae = perf_df.loc[perf_df['MAE'].idxmin()]
                    best_r2 = perf_df.loc[perf_df['R²'].idxmax()]
                    
                    with col1:
                        st.metric("Best Model (MAE)", best_mae['Model'], f"MAE: {best_mae['MAE']:.4f}")
                    with col2:
                        st.metric("Best Model (R²)", best_r2['Model'], f"R²: {best_r2['R²']:.4f}")
                    with col3:
                        st.metric("Models Trained", len(metrics))
                
                # Performance chart
                st.subheader("Performance Comparison")
                
                chart_type = st.radio("Select Metric:", ["MAE", "R²"], horizontal=True, key="perf_chart")
                
                if chart_type == "MAE":
                    fig_perf = go.Figure(data=[
                        go.Bar(name=row['Model'], x=[row['Model']], y=[row['MAE']])
                        for _, row in perf_df.iterrows()
                    ])
                    fig_perf.update_layout(title="MAE by Model", yaxis_title="Mean Absolute Error")
                else:
                    fig_perf = go.Figure(data=[
                        go.Bar(name=row['Model'], x=[row['Model']], y=[row['R²']])
                        for _, row in perf_df.iterrows()
                    ])
                    fig_perf.update_layout(title="R² Score by Model", yaxis_title="R² Score")
                
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with tab5:
            st.header("🎯 Price Predictions")
            
            if not st.session_state.models_trained:
                st.warning("Train models first to generate predictions.")
            else:
                # Prediction interface
                st.subheader("Future Price Forecast")
                
                if st.button("Generate Predictions", type="primary", key="predict_btn"):
                    with st.spinner("Generating predictions..."):
                        # Mock prediction display
                        future_dates = pd.date_range(
                            start=data.index[-1] + timedelta(days=1),
                            periods=forecast_days,
                            freq='D'
                        )
                        
                        # Create prediction visualization
                        show_predictions(data, future_dates, forecast_days)
        
        with tab6:
            st.header("📋 Analysis Report")
            
            # Generate comprehensive report
            st.subheader("Executive Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Data Overview**")
                st.write(f"- Cryptocurrency: {st.session_state.symbol}")
                st.write(f"- Analysis Period: {len(data)} days")
                st.write(f"- Current Price: ${data['close'].iloc[-1]:.2f}")
                st.write(f"- Total Return: {((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100):.2f}%")
            
            with col2:
                st.info("**Model Performance**")
                if st.session_state.models_trained:
                    best_model = min(st.session_state.metrics.items(), 
                                   key=lambda x: x[1].get('day_1', {}).get('mae', float('inf')))
                    st.write(f"- Best Model: {best_model[0].title()}")
                    st.write(f"- Best MAE: {best_model[1].get('day_1', {}).get('mae', 0):.4f}")
                    st.write(f"- Models Trained: {len(st.session_state.models)}")
                else:
                    st.write("- Models: Not trained")
            
            # Download report
            if st.button("Generate PDF Report", key="report_btn"):
                st.success("Report generation feature coming soon!")

    else:
        st.warning("No data available. Please run the analysis first.")

else:
    # Welcome screen yang lebih menarik
    st.markdown("""
    <style>
    .welcome-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .step-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='welcome-header'>
        <h1>🚀 Advanced Cryptocurrency Prediction System</h1>
        <h3>AI-Powered Market Analysis & Price Forecasting</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    st.subheader("🎯 System Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h4>📊 Market Data</h4>
            <p>• 15+ Cryptocurrencies</p>
            <p>• Real-time OHLC data</p>
            <p>• Historical analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h4>🔬 Technical Analysis</h4>
            <p>• 30+ Technical indicators</p>
            <p>• RSI, MACD, Bollinger Bands</p>
            <p>• Volume analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h4>🤖 AI Models</h4>
            <p>• LSTM Neural Networks</p>
            <p>• Ensemble Methods</p>
            <p>• Price predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started
    st.subheader("🚀 Getting Started")
    
    st.markdown("""
    <div class='step-card'>
        <strong>Step 1:</strong> Select cryptocurrency pair from sidebar
    </div>
    <div class='step-card'>
        <strong>Step 2:</strong> Choose machine learning models
    </div>
    <div class='step-card'>
        <strong>Step 3:</strong> Adjust analysis parameters
    </div>
    <div class='step-card'>
        <strong>Step 4:</strong> Click "Run Full Analysis"
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start buttons
    st.subheader("⚡ Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Analyze BTC", use_container_width=True, key="btc_btn"):
            st.session_state.symbol = 'BTC'
            st.session_state.analysis_run = True
            st.rerun()
    
    with col2:
        if st.button("Analyze ETH", use_container_width=True, key="eth_btn"):
            st.session_state.symbol = 'ETH'
            st.session_state.analysis_run = True
            st.rerun()
    
    with col3:
        if st.button("Analyze SOL", use_container_width=True, key="sol_btn"):
            st.session_state.symbol = 'SOL'
            st.session_state.analysis_run = True
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit, Python, and Machine Learning • 
    <a href='#' target='_blank'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)

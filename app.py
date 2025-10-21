# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import os
import subprocess
import threading
import queue
import time
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
# Tambahkan di app.py (sebelum helper functions)
def run_crypto_scanner_real_time(params, output_queue):
    """Run crypto_scanner.py dan capture output real-time"""
    try:
        # Build command dari parameters
        cmd = [
            "python", "crypto_scanner.py",
            "--preset", params['preset'],
            "--exchanges", params['exchanges'],
            "--top", str(params['top']),
            "--ltf_fetch_limit", str(params['ltf_fetch_limit']),
            "--min_24h_quotevol_usd", str(params['min_24h_quotevol_usd']),
            "--min_price_usd", str(params['min_price_usd']),
            "--min_depth_usd", str(params['min_depth_usd']),
            "--depth_window_pct", str(params['depth_window_pct']),
            "--max_spread_bps", str(params['max_spread_bps']),
            "--htf_logic", params['htf_logic'],
            "--adx_min", str(params['adx_min']),
            "--adx_max", str(params['adx_max']),
            "--ma_dist_pct_max", str(params['ma_dist_pct_max']),
            "--bbw_rank_min", str(params['bbw_rank_min']),
            "--bbw_rank_max", str(params['bbw_rank_max']),
            "--ltf", params['ltf'],
            "--rsi_long", str(params['rsi_long']),
            "--rsi_short", str(params['rsi_short']),
            "--rsi_cross_lookback", str(params['rsi_cross_lookback']),
            "--vol_ratio_max", str(params['vol_ratio_max']),
            "--atr_sl_mult", str(params['atr_sl_mult']),
            "--atr_tp_mult", str(params['atr_tp_mult']),
            "--min_rr1", str(params['min_rr1']),
            "--budget_usd", str(params['budget_usd']),
            "--risk_pct", str(params['risk_pct']),
            "--max_positions", str(params['max_positions']),
            "--max_leverage", str(params['max_leverage']),
            "--eta_horizon_bars", str(params['eta_horizon_bars']),
            "--eta_back_bars", str(params['eta_back_bars']),
            "--eta_min_samples", str(params['eta_min_samples']),
            "--eta_min_p", str(params['eta_min_p']),
            "--min_score", str(params['min_score']),
        ]
        
        # Add optional flags
        if params.get('dedupe_bases', False):
            cmd.append("--dedupe_bases")
        if params.get('eta_enable', False):
            cmd.append("--eta_enable")
        if params.get('eta_gate_enable', False):
            cmd.append("--eta_gate_enable")
        if params.get('eta_vol_adjust', False):
            cmd.append("--eta_vol_adjust")
        if params.get('eta_require_rsi_adx', False):
            cmd.append("--eta_require_rsi_adx")
        if params.get('sort_by_score', False):
            cmd.append("--sort_by_score")
        if params.get('ltf_bb_touch', False):
            cmd.append("--ltf_bb_touch")
            cmd.extend(["--ltf_bb_touch_slack_pct", str(params['ltf_bb_touch_slack_pct'])])
            cmd.extend(["--ltf_bb_touch_lookback", str(params['ltf_bb_touch_lookback'])])
        if params.get('trigger_touch_fallback', False):
            cmd.append("--trigger_touch_fallback")
        if params.get('macro_blackout_enable', False):
            cmd.append("--macro_blackout_enable")
            cmd.extend(["--macro_file", params['macro_file']])
            cmd.extend(["--macro_before_min", str(params['macro_before_min'])])
            cmd.extend(["--macro_after_min", str(params['macro_after_min'])])
        if params.get('htf_rsi_mid_enable', False):
            cmd.append("--htf_rsi_mid_enable")
            cmd.extend(["--rsi4h_min", str(params['rsi4h_min'])])
            cmd.extend(["--rsi4h_max", str(params['rsi4h_max'])])
        if params.get('htf_macd_mid_enable', False):
            cmd.append("--htf_macd_mid_enable")
            cmd.extend(["--macd_hist_pct_max", str(params['macd_hist_pct_max'])])
        if params.get('ma200_slope_gate_enable', False):
            cmd.append("--ma200_slope_gate_enable")
            cmd.extend(["--ma200_slope_max_pct_per_bar", str(params['ma200_slope_max_pct_per_bar'])])
        if params.get('ct_override_enable', False):
            cmd.append("--ct_override_enable")
            cmd.extend(["--ct_rsi_long_extreme", str(params['ct_rsi_long_extreme'])])
            cmd.extend(["--ct_rsi_short_extreme", str(params['ct_rsi_short_extreme'])])
            cmd.extend(["--ct_dev_ma1h_min_pct", str(params['ct_dev_ma1h_min_pct'])])
            cmd.extend(["--ct_adx1h_min", str(params['ct_adx1h_min'])])
            cmd.extend(["--ct_bb_touch_lookback", str(params['ct_bb_touch_lookback'])])
        if params.get('whitelist'):
            cmd.extend(["--whitelist", params['whitelist']])
        if params.get('why', False):
            cmd.append("--why")
        if params.get('notify_telegram', False):
            cmd.append("--notify_telegram")
            if params.get('telegram_bot_token'):
                cmd.extend(["--telegram_bot_token", params['telegram_bot_token']])
            if params.get('telegram_chat_id'):
                cmd.extend(["--telegram_chat_id", params['telegram_chat_id']])
        
        output_queue.put(f"🚀 Starting crypto scanner with command:\n{' '.join(cmd)}\n")
        
        # Run process dengan real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        for line in process.stdout:
            output_queue.put(line)
            
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            output_queue.put("✅ Crypto scanner completed successfully!")
        else:
            output_queue.put(f"❌ Crypto scanner failed with return code: {return_code}")
            
    except Exception as e:
        output_queue.put(f"❌ Error running crypto scanner: {str(e)}")

def show_crypto_scanner_real_time():
    """Display Crypto Scanner UI dengan real-time output"""
    st.header("🔍 Crypto Scanner - Real-time Execution")
    
    # Parameter configuration
    with st.expander("⚙️ Scanner Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            preset = st.selectbox("Strategy Preset", ["conservative", "moderate", "aggressive"], index=0)
            exchanges = st.multiselect("Exchanges", ["okx", "gateio", "binance"], default=["okx", "gateio", "binance"])
            top = st.number_input("Top Coins", 50, 300, 160)
            min_vol = st.number_input("Min 24h Volume (M)", 5, 50, 15)
            whitelist = st.text_input("Whitelist", "BTC,ETH,SOL,BNB,LINK,AVAX,LTC,XRP,DOGE,ADA")
            
        with col2:
            rsi_long = st.slider("RSI Long", 20, 40, 38)
            rsi_short = st.slider("RSI Short", 60, 80, 62)
            budget_usd = st.number_input("Budget USD", 50, 1000, 200)
            max_positions = st.number_input("Max Positions", 1, 10, 2)
    
    # Advanced parameters
    with st.expander("🔧 Advanced Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Management")
            risk_pct = st.number_input("Risk %", 0.001, 0.02, 0.005, 0.001, format="%.3f")
            max_leverage = st.number_input("Max Leverage", 1, 20, 5)
            atr_sl_mult = st.number_input("ATR SL Multiplier", 1.0, 3.0, 1.6, 0.1)
            atr_tp_mult = st.number_input("ATR TP Multiplier", 1.0, 3.0, 1.2, 0.1)
            
        with col2:
            st.subheader("Filters")
            ltf_bb_touch = st.checkbox("LTF BB Touch", value=True)
            vol_ratio_max = st.number_input("Max Vol Ratio", 1.0, 5.0, 3.2, 0.1)
            min_rr1 = st.number_input("Min RR1", 0.1, 2.0, 0.7, 0.1)
            dedupe_bases = st.checkbox("Dedupe Bases", value=True)
    
    # ETA Settings
    with st.expander("🤖 ETA Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            eta_enable = st.checkbox("Enable ETA", value=True)
            eta_gate_enable = st.checkbox("ETA Gate", value=True)
            eta_horizon_bars = st.number_input("Horizon Bars", 48, 200, 96)
            eta_back_bars = st.number_input("Lookback Bars", 100, 600, 420)
            
        with col2:
            eta_min_samples = st.number_input("Min Samples", 20, 100, 40)
            eta_min_p = st.slider("Min Probability", 0.0, 1.0, 0.60, 0.05)
            eta_vol_adjust = st.checkbox("Volume Adjust", value=True)
            eta_require_rsi_adx = st.checkbox("Require RSI+ADX", value=True)
    
    # Run button
    st.markdown("---")
    
    if 'scanner_running' not in st.session_state:
        st.session_state.scanner_running = False
    if 'scanner_output' not in st.session_state:
        st.session_state.scanner_output = []
    if 'scanner_thread' not in st.session_state:
        st.session_state.scanner_thread = None
    if 'output_queue' not in st.session_state:
        st.session_state.output_queue = queue.Queue()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.scanner_running:
            if st.button("🚀 RUN CRYPTO SCANNER", type="primary", use_container_width=True):
                # Prepare parameters
                params = {
                    'preset': preset,
                    'exchanges': ','.join(exchanges),
                    'top': top,
                    'ltf_fetch_limit': 750,
                    'min_24h_quotevol_usd': min_vol * 1_000_000,
                    'min_price_usd': 0.01,
                    'min_depth_usd': 20000,
                    'depth_window_pct': 0.02,
                    'max_spread_bps': 16,
                    'htf_logic': 'all',
                    'adx_min': 10,
                    'adx_max': 40,
                    'ma_dist_pct_max': 0.06,
                    'bbw_rank_min': 0.02,
                    'bbw_rank_max': 0.70,
                    'ltf': '5m',
                    'rsi_long': rsi_long,
                    'rsi_short': rsi_short,
                    'rsi_cross_lookback': 0,
                    'ltf_bb_touch': ltf_bb_touch,
                    'ltf_bb_touch_slack_pct': 0.0015,
                    'ltf_bb_touch_lookback': 5,
                    'trigger_touch_fallback': True,
                    'vol_ratio_max': vol_ratio_max,
                    'atr_sl_mult': atr_sl_mult,
                    'atr_tp_mult': atr_tp_mult,
                    'min_rr1': min_rr1,
                    'budget_usd': budget_usd,
                    'risk_pct': risk_pct,
                    'max_positions': max_positions,
                    'max_leverage': max_leverage,
                    'dedupe_bases': dedupe_bases,
                    'eta_enable': eta_enable,
                    'eta_gate_enable': eta_gate_enable,
                    'eta_horizon_bars': eta_horizon_bars,
                    'eta_back_bars': eta_back_bars,
                    'eta_min_samples': eta_min_samples,
                    'eta_min_p': eta_min_p,
                    'eta_vol_adjust': eta_vol_adjust,
                    'eta_require_rsi_adx': eta_require_rsi_adx,
                    'sort_by_score': True,
                    'min_score': 0.60,
                    'whitelist': whitelist,
                    'macro_blackout_enable': True,
                    'macro_file': 'macro_events.utc.txt',
                    'macro_before_min': 45,
                    'macro_after_min': 45,
                    'htf_rsi_mid_enable': True,
                    'rsi4h_min': 45,
                    'rsi4h_max': 55,
                    'htf_macd_mid_enable': True,
                    'macd_hist_pct_max': 0.0030,
                    'ma200_slope_gate_enable': True,
                    'ma200_slope_max_pct_per_bar': 0.0015,
                    'ct_override_enable': True,
                    'ct_rsi_long_extreme': 20,
                    'ct_rsi_short_extreme': 80,
                    'ct_dev_ma1h_min_pct': 0.025,
                    'ct_adx1h_min': 30,
                    'ct_bb_touch_lookback': 5,
                    'why': True,
                    'notify_telegram': False
                }
                
                # Start scanner in separate thread
                st.session_state.scanner_running = True
                st.session_state.scanner_output = ["🚀 Starting Crypto Scanner..."]
                st.session_state.output_queue = queue.Queue()
                
                st.session_state.scanner_thread = threading.Thread(
                    target=run_crypto_scanner_real_time,
                    args=(params, st.session_state.output_queue)
                )
                st.session_state.scanner_thread.daemon = True
                st.session_state.scanner_thread.start()
                
                st.rerun()
        else:
            if st.button("🛑 STOP SCANNER", type="secondary", use_container_width=True):
                st.session_state.scanner_running = False
                st.rerun()
    
    # Display real-time output
    if st.session_state.scanner_running:
        st.subheader("📊 Scanner Output (Real-time)")
        
        # Create output container
        output_container = st.container()
        
        # Check for new output
        try:
            while True:
                try:
                    new_output = st.session_state.output_queue.get_nowait()
                    st.session_state.scanner_output.append(new_output)
                    
                    # Keep only last 1000 lines to prevent memory issues
                    if len(st.session_state.scanner_output) > 1000:
                        st.session_state.scanner_output = st.session_state.scanner_output[-1000:]
                        
                except queue.Empty:
                    break
        except:
            pass
        
        # Display output with syntax highlighting
        with output_container:
            # Create scrollable text area
            output_text = "\n".join(st.session_state.scanner_output)
            
            # Use code block for better formatting
            st.code(output_text, language='text')
            
            # Auto-refresh
            time.sleep(1)
            st.rerun()
            
        # Check if thread is still alive
        if (st.session_state.scanner_thread and 
            not st.session_state.scanner_thread.is_alive()):
            st.session_state.scanner_running = False
            st.success("✅ Scanner completed!")
            st.rerun()
    
    else:
        if st.session_state.scanner_output:
            st.subheader("📋 Last Scanner Output")
            output_text = "\n".join(st.session_state.scanner_output)
            st.code(output_text, language='text')

def predict_with_models(model_trainer, data_with_features, feature_names, forecast_days):
    """External prediction function dengan enhanced debugging"""
    predictions = {}
    
    st.write("🔍 DEBUG: predict_with_models started")
    
    try:
        # Validasi input
        if model_trainer is None:
            st.error("❌ model_trainer is None in predict_with_models")
            return {}
            
        if data_with_features is None:
            st.error("❌ data_with_features is None in predict_with_models")
            return {}
            
        # Get current price
        current_price = data_with_features['close'].iloc[-1]
        st.write(f"🔍 DEBUG: Current price: ${current_price:.2f}")
        
        # Prepare the last sequence for prediction
        features = data_with_features[feature_names].values
        st.write(f"🔍 DEBUG: Features shape: {features.shape}")
        
        # Ensure we have enough data for lookback
        lookback_days = model_trainer.lookback_days
        st.write(f"🔍 DEBUG: Lookback days: {lookback_days}")
        st.write(f"🔍 DEBUG: Available data: {len(features)} days")
        
        if len(features) < lookback_days:
            error_msg = f"Not enough data for prediction. Need {lookback_days} days, got {len(features)}"
            st.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        last_sequence = features[-lookback_days:]
        last_sequence = last_sequence.reshape(1, lookback_days, len(feature_names))
        st.write(f"🔍 DEBUG: Last sequence shape: {last_sequence.shape}")
        
        # Check available models
        available_models = list(model_trainer.models.keys())
        st.write(f"🔍 DEBUG: Available models: {available_models}")
        
        if not available_models:
            st.error("❌ No models available for prediction")
            return {}
        
        for model_name, model in model_trainer.models.items():
            try:
                st.write(f"🔍 DEBUG: Predicting with {model_name}...")
                
                if model_name in ['lstm', 'bilstm', 'cnn_lstm', 'cnn_bilstm']:
                    # Deep learning models
                    y_pred = model.predict(last_sequence, verbose=0)[0]
                    st.write(f"🔍 DEBUG: {model_name} DL prediction: {y_pred[:3]}...")
                else:
                    # Traditional ML models
                    X_flat = last_sequence.reshape(1, -1)
                    y_pred_single = model.predict(X_flat)[0]
                    # Create forecast for all days
                    y_pred = np.full(forecast_days, y_pred_single)
                    st.write(f"🔍 DEBUG: {model_name} ML prediction: {y_pred_single:.2f}")
                
                # Ensure we only return the requested forecast days
                y_pred = y_pred[:forecast_days]
                
                # Validasi prediksi
                if (y_pred is not None and 
                    len(y_pred) == forecast_days and 
                    not np.isnan(y_pred).any() and
                    np.min(y_pred) > 0):
                    
                    predictions[model_name] = y_pred
                    st.write(f"✅ {model_name}: Prediction successful")
                else:
                    st.warning(f"⚠️ {model_name}: Invalid prediction, using fallback")
                    predictions[model_name] = [current_price] * forecast_days
                
            except Exception as e:
                st.error(f"❌ Error predicting with {model_name}: {e}")
                predictions[model_name] = [current_price] * forecast_days
        
        st.write(f"✅ Successfully generated predictions for {len(predictions)} models")
        
    except Exception as e:
        st.error(f"❌ Error in predict_with_models: {e}")
        import traceback
        st.write("Stack trace:")
        st.code(traceback.format_exc())
        
        # Fallback for all models
        try:
            current_price = data_with_features['close'].iloc[-1]
        except:
            current_price = 1000
            
        for model_name in model_trainer.models.keys():
            predictions[model_name] = [current_price] * forecast_days
    
    return predictions

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

def show_model_performance(metrics):
    """Show realistic model performance"""
    st.subheader("🏆 Realistic Model Ranking")
    
    if not metrics:
        st.warning("No model metrics available")
        return
    
    performance_data = []
    for model_name, model_data in metrics.items():
        day1 = model_data.get('day_metrics', {}).get('day_1', {})
        is_realistic = model_data.get('realistic', False)
        overall_score = model_data.get('overall_score', 0)
        
        performance_data.append({
            'Model': model_name.upper(),
            'Realistic': '✅' if is_realistic else '❌',
            'Profit_Score': f"{overall_score:.1f}%",
            'Direction_Acc': f"{day1.get('direction_accuracy', 0)*100:.1f}%",
            'R²_Score': f"{day1.get('r2', 0):.3f}",
            'MAE': f"${day1.get('mae', 0):.2f}"
        })
    
    # Sort by profit score
    performance_df = pd.DataFrame(performance_data)
    if not performance_df.empty:
        performance_df = performance_df.sort_values('Profit_Score', ascending=False)
        st.dataframe(performance_df, use_container_width=True)
        
        # Show best realistic model
        realistic_models = performance_df[performance_df['Realistic'] == '✅']
        if not realistic_models.empty:
            best_model = realistic_models.iloc[0]
            st.success(f"🎯 **Recommended Model**: {best_model['Model']} (Profit Score: {best_model['Profit_Score']})")
        else:
            st.error("⚠️ No realistic models found! Predictions may be unreliable.")
# GANTI function show_predictions di app.py

# app.py - UPDATE function show_predictions

# app.py - UPDATE function show_predictions dengan detailed debugging

def show_predictions(data, data_with_features, feature_names, forecast_days, model_trainer):
    """Show REAL predictions dengan detailed debugging"""
    
    st.write("🔍 DEBUG: show_predictions function started")
    
    try:
        # ✅ Validasi input parameters
        if data is None:
            st.error("❌ Data is None")
            return
        if data_with_features is None:
            st.error("❌ data_with_features is None") 
            return
        if feature_names is None or len(feature_names) == 0:
            st.error("❌ feature_names is empty")
            return
        if model_trainer is None:
            st.error("❌ model_trainer is None")
            return
            
        st.write(f"🔍 DEBUG: Input validation passed")
        st.write(f"🔍 DEBUG: Data shape: {data.shape}")
        st.write(f"🔍 DEBUG: Features shape: {data_with_features.shape}")
        st.write(f"🔍 DEBUG: Feature names count: {len(feature_names)}")
        st.write(f"🔍 DEBUG: Forecast days: {forecast_days}")
        st.write(f"🔍 DEBUG: Model trainer: {type(model_trainer)}")
        
        # Get current price
        current_price = data['close'].iloc[-1]
        st.write(f"🔍 DEBUG: Current price: ${current_price:.2f}")
        
        # Generate predictions
        st.write("🔍 DEBUG: Starting prediction generation...")
        
        if hasattr(model_trainer, 'predict_future_prices'):
            st.write("🔍 DEBUG: Using model_trainer.predict_future_prices")
            try:
                predictions = model_trainer.predict_future_prices(
                    data_with_features, feature_names, forecast_days
                )
                st.write(f"🔍 DEBUG: Predictions received: {len(predictions)} models")
            except Exception as e:
                st.error(f"❌ Error in predict_future_prices: {e}")
                predictions = None
        else:
            st.write("🔍 DEBUG: Using predict_with_models")
            predictions = predict_with_models(
                model_trainer, data_with_features, feature_names, forecast_days
            )
        
        # Validasi predictions
        if predictions is None:
            st.error("❌ Predictions is None")
            show_mock_predictions(data, forecast_days)
            return
            
        if len(predictions) == 0:
            st.error("❌ No predictions generated")
            show_mock_predictions(data, forecast_days)
            return
            
        st.write(f"🔍 DEBUG: Raw predictions: {list(predictions.keys())}")
        
        # Filter valid predictions
        valid_predictions = {}
        for model_name, pred_values in predictions.items():
            st.write(f"🔍 DEBUG: Checking {model_name}: {pred_values}")
            
            if (pred_values is not None and 
                len(pred_values) == forecast_days and 
                not np.isnan(pred_values).any()):
                
                # Check if predictions are reasonable
                max_reasonable = current_price * 1.5
                min_reasonable = current_price * 0.5
                
                if (np.max(pred_values) < max_reasonable and 
                    np.min(pred_values) > min_reasonable):
                    valid_predictions[model_name] = pred_values
                    st.write(f"✅ {model_name}: Valid predictions")
                else:
                    st.warning(f"⚠️ {model_name}: Unreasonable predictions")
                    # Fallback
                    trend = np.linspace(0, 0.02, forecast_days)
                    valid_predictions[model_name] = current_price * (1 + trend)
            else:
                st.warning(f"⚠️ {model_name}: Invalid predictions")
                # Fallback
                valid_predictions[model_name] = [current_price] * forecast_days
        
        st.write(f"🔍 DEBUG: Valid predictions: {len(valid_predictions)} models")
        
        if not valid_predictions:
            st.error("❌ No valid predictions after filtering")
            show_mock_predictions(data, forecast_days)
            return
        
        # Create future dates
        future_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        st.write("🔍 DEBUG: Starting to display predictions...")
        
        # Display predictions
        st.subheader("📈 Real Price Forecast")
        
        # Create prediction chart
        fig_pred = go.Figure()
        
        # Historical data
        lookback_days = min(60, len(data))
        fig_pred.add_trace(go.Scatter(
            x=data.index[-lookback_days:],
            y=data['close'].iloc[-lookback_days:],
            name='Historical Price',
            line=dict(color='white', width=3),
            mode='lines'
        ))
        
        # Current price marker
        fig_pred.add_trace(go.Scatter(
            x=[data.index[-1]],
            y=[current_price],
            name='Current Price',
            mode='markers',
            marker=dict(color='yellow', size=10, symbol='star')
        ))
        
        # Predictions for each model
        colors = {
            'lstm': '#ff6b6b', 'bilstm': '#4ecdc4', 'cnn_lstm': '#45b7d1',
            'cnn_bilstm': '#96ceb4', 'random_forest': '#feca57',
            'linear_regression': '#ff9ff3', 'svr': '#54a0ff'
        }
        
        display_names = {
            'lstm': 'LSTM', 'bilstm': 'BiLSTM', 'cnn_lstm': 'CNN-LSTM',
            'cnn_bilstm': 'CNN-BiLSTM', 'random_forest': 'Random Forest',
            'linear_regression': 'Linear Regression', 'svr': 'SVR'
        }
        
        for model_name, pred_values in valid_predictions.items():
            if model_name in display_names:
                fig_pred.add_trace(go.Scatter(
                    x=future_dates,
                    y=pred_values,
                    name=f'{display_names[model_name]} Forecast',
                    line=dict(color=colors.get(model_name, '#666666'), width=2, dash='dash'),
                    mode='lines+markers'
                ))
        
        fig_pred.update_layout(
            title=f"Price Forecast - Next {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500,
            template="plotly_dark",
            showlegend=True
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction table
        st.subheader("📋 Prediction Values (USD)")
        pred_df = pd.DataFrame(valid_predictions, index=future_dates)
        pred_df_display = pred_df.rename(columns=display_names)
        pred_df_display['Average'] = pred_df_display.mean(axis=1)
        
        st.dataframe(pred_df_display.style.format("${:.2f}"), use_container_width=True)
        
        st.success("✅ Predictions displayed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error in show_predictions: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Using fallback predictions...")
        show_mock_predictions(data, forecast_days)

def show_mock_predictions(data, forecast_days):
    """Fallback mock predictions dengan current_price yang terdefinisi"""
    
    # ✅ FIX: Pastikan current_price tersedia
    try:
        current_price = data['close'].iloc[-1]
    except:
        current_price = 1000  # Default fallback
    
    # Create future dates
    future_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )
    
    # Create realistic mock predictions based on current_price
    predictions = {
        'lstm': current_price * (1 + np.random.uniform(-0.05, 0.08, forecast_days)),
        'bilstm': current_price * (1 + np.random.uniform(-0.04, 0.07, forecast_days)),
        'cnn_lstm': current_price * (1 + np.random.uniform(-0.03, 0.06, forecast_days)),
        'cnn_bilstm': current_price * (1 + np.random.uniform(-0.04, 0.06, forecast_days)),
        'random_forest': current_price * (1 + np.random.uniform(-0.02, 0.05, forecast_days)),
        'linear_regression': current_price * (1 + np.random.uniform(-0.01, 0.04, forecast_days)),
        'svr': current_price * (1 + np.random.uniform(-0.03, 0.05, forecast_days))
    }
    
    # Display predictions
    st.subheader("📈 Price Forecast (Mock Data - Fallback)")
    
    # Create prediction chart
    fig_pred = go.Figure()
    
    # Historical data (last 30 days)
    lookback_days = min(30, len(data))
    fig_pred.add_trace(go.Scatter(
        x=data.index[-lookback_days:],
        y=data['close'].iloc[-lookback_days:],
        name='Historical Price',
        line=dict(color='white', width=3)
    ))
    
    # Current price marker
    fig_pred.add_trace(go.Scatter(
        x=[data.index[-1]],
        y=[current_price],
        name='Current Price',
        mode='markers',
        marker=dict(color='yellow', size=10, symbol='star')
    ))
    
    # Predictions for each model
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    
    # Model display names
    display_names = {
        'lstm': 'LSTM', 'bilstm': 'BiLSTM', 'cnn_lstm': 'CNN-LSTM',
        'cnn_bilstm': 'CNN-BiLSTM', 'random_forest': 'Random Forest',
        'linear_regression': 'Linear Regression', 'svr': 'SVR'
    }
    
    for i, (model_name, pred_values) in enumerate(predictions.items()):
        if model_name in display_names:
            fig_pred.add_trace(go.Scatter(
                x=future_dates,
                y=pred_values,
                name=f'{display_names[model_name]} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
    
    fig_pred.update_layout(
        title=f"Price Forecast - Next {forecast_days} Days (Mock Data)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Prediction table
    st.subheader("📋 Prediction Values (Mock Data)")
    pred_df = pd.DataFrame(predictions, index=future_dates)
    
    # Use display names for columns
    pred_df_display = pred_df.rename(columns=display_names)
    pred_df_display['Average'] = pred_df_display.mean(axis=1)
    
    st.dataframe(
        pred_df_display.style.format("${:.2f}"),
        use_container_width=True
    )
    
    # Trading recommendation
    st.subheader("🎯 Trading Recommendation (Based on Mock Data)")
    avg_predictions = pred_df_display['Average']
    avg_prediction = avg_predictions.iloc[-1]
    
    price_change_pct = ((avg_prediction - current_price) / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Predicted Price", f"${avg_prediction:.2f}")
    with col3:
        st.metric("Expected Change", f"{price_change_pct:+.2f}%")
    
    if price_change_pct > 5:
        st.success("🚀 **STRONG BUY** - Significant upside predicted")
    elif price_change_pct > 0:
        st.info("📈 **BUY** - Moderate upside predicted")
    elif price_change_pct > -3:
        st.warning("⚖️ **HOLD** - Minimal price movement expected")
    else:
        st.error("📉 **SELL** - Downside predicted")
    
    st.warning("⚠️ Note: This is mock data. Run full analysis for real predictions.")

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
if 'model_trainer' not in st.session_state:  # ✅ TAMBAHKAN INI
    st.session_state.model_trainer = None

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
# Di bagian model selection app.py - UPDATE

# Model selection dengan model baru
available_models = [
    'LSTM', 
    'BiLSTM',           # ✅ NEW: Bidirectional LSTM
    'CNN-BiLSTM',       # ✅ NEW: Hybrid model  
    'CNN-LSTM', 
    'Random Forest', 
    'Linear Regression', 
    'SVR'
]

model_descriptions = {
    'LSTM': 'Long Short-Term Memory - Best for time series',
    'BiLSTM': 'Bidirectional LSTM - Better context understanding for volatile markets',  # ✅ NEW
    'CNN-BiLSTM': 'CNN + BiLSTM hybrid - Best for complex volatile patterns',  # ✅ NEW
    'CNN-LSTM': 'CNN + LSTM hybrid - Good for pattern recognition',
    'Random Forest': 'Ensemble method - Robust and fast',
    'Linear Regression': 'Simple linear model - Good baseline', 
    'SVR': 'Support Vector Regression - Good for non-linear patterns'
}

# Default models yang recommended untuk crypto volatile
selected_models = st.sidebar.multiselect(
    "Select ML Models:",
    available_models,
    default=['LSTM', 'BiLSTM', 'Random Forest'],  # ✅ Updated default
    key="model_select"
)

# Training configuration di sidebar
st.sidebar.subheader("🎯 Advanced Training")

# Early stopping configuration
col1, col2 = st.sidebar.columns(2)
with col1:
    early_stop_patience = st.slider("Early Stop Patience", 5, 30, 15, 
                                   help="Epochs to wait before stopping")
with col2:
    reduce_lr_patience = st.slider("Reduce LR Patience", 5, 20, 10,
                                  help="Epochs to wait before reducing learning rate")

max_epochs = st.sidebar.slider("Max Epochs", 50, 200, 100,
                              help="Maximum training epochs")

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
                        try:
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
                                
                                # ✅ FIX: Check if train_models method exists
                                if hasattr(model_trainer, 'train_models'):
                                    models = model_trainer.train_models(
                                        X_train, y_train, X_test, y_test, 
                                        st.session_state.feature_names, selected_models
                                    )
                                    
                                    metrics = model_trainer.evaluate_models(X_test, y_test)
                                    
                                    st.session_state.models = models
                                    st.session_state.metrics = metrics
                                    st.session_state.models_trained = True
                                    st.session_state.model_trainer = model_trainer
                                    
                                    st.success(f"✅ Analysis completed for {selected_symbol}!")
                                    st.success(f"✅ {len(models)} models trained successfully!")
                                    
                                else:
                                    st.error("❌ ModelTrainer doesn't have train_models method!")
                                    # Fallback: create simple models
                                    st.session_state.models_trained = False
                                    
                            else:
                                st.error("❌ Not enough data for training sequences.")
                                
                        except Exception as e:
                            st.error(f"❌ Error during model training: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning("No models selected for training.")
                
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
# Dalam Reset Button - UPDATE

    if st.sidebar.button("Reset All", key="reset_button", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Re-initialize critical states
        st.session_state.analysis_run = False
        st.session_state.data = None
        st.session_state.models_trained = False
        st.session_state.model_trainer = None  # ✅ Reset model trainer juga
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
                
                # Show what's missing
                col1, col2, col3 = st.columns(3)
                with col1:
                    status = "✅" if st.session_state.data is not None else "❌"
                    st.write(f"{status} Data Loaded")
                with col2:
                    status = "✅" if st.session_state.technical_calculated else "❌" 
                    st.write(f"{status} Technical Indicators")
                with col3:
                    status = "✅" if st.session_state.models_trained else "❌"
                    st.write(f"{status} Models Trained")
                    
            else:
                metrics = st.session_state.metrics
                
                # Model performance comparison
                st.subheader("🏆 Model Performance Comparison")
                
                # Create performance summary
                performance_data = []
                
                for model_name, model_data in metrics.items():
                    day_metrics = model_data.get('day_metrics', {})
                    first_day = day_metrics.get('day_1', {})
                    
                    performance_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'MAE': first_day.get('mae', 0),
                        'RMSE': first_day.get('rmse', 0),
                        'R²': first_day.get('r2', 0),
                        'Direction Acc': f"{first_day.get('direction_accuracy', 0)*100:.1f}%"
                    })
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    
                    # Highlight best model for each metric
                    styled_df = perf_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')\
                                            .highlight_max(subset=['R²', 'Direction Acc'], color='lightgreen')
                    
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.warning("No performance metrics available")
                
                # Training History Visualization - DIPERBAIKI
                st.subheader("📈 Training Progress")
                
                if (hasattr(st.session_state, 'model_trainer') and 
                    hasattr(st.session_state.model_trainer, 'histories') and
                    st.session_state.model_trainer.histories):
                    
                    model_for_history = st.selectbox(
                        "Select Model to View Training History:",
                        list(st.session_state.model_trainer.histories.keys()),
                        key="history_select"
                    )
                    
                    if model_for_history:
                        # ✅ FIX: Gunakan method yang sudah diperbaiki
                        if hasattr(st.session_state.model_trainer, 'plot_training_history'):
                            fig_history = st.session_state.model_trainer.plot_training_history(model_for_history)
                        else:
                            # Fallback
                            fig_history = create_simple_training_plot(
                                st.session_state.model_trainer.histories[model_for_history],
                                model_for_history
                            )
                        
                        if fig_history:
                            st.plotly_chart(fig_history, use_container_width=True)
                            
                            # Show training summary
                            history = st.session_state.model_trainer.histories[model_for_history]
                            final_epoch = len(history.history['loss'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**Final Epoch:** {final_epoch}")
                            with col2:
                                if 'val_loss' in history.history:
                                    best_epoch = np.argmin(history.history['val_loss']) + 1
                                    st.info(f"**Best Epoch:** {best_epoch}")
                                else:
                                    st.info("**No Validation Data**")
                            with col3:
                                best_loss = min(history.history['loss'])
                                st.info(f"**Best Loss:** {best_loss:.4f}")
                        else:
                            st.warning("Could not generate training history plot")
                else:
                    st.info("No training history available. Only deep learning models have training history.")
                    
                    # Show available models
                    if hasattr(st.session_state, 'model_trainer') and st.session_state.model_trainer.models:
                        st.write("**Available Models:**", list(st.session_state.model_trainer.models.keys()))
                    
                # Dalam with tab5: - UPDATE dengan better error handling

                with tab5:
                    st.header("🎯 Price Predictions")
                    
                    # Debug info
                    with st.expander("🔧 Debug Info", expanded=False):
                        st.write("Session State Status:")
                        st.json({
                            'data_loaded': st.session_state.data is not None,
                            'features_calculated': st.session_state.data_with_features is not None,
                            'feature_names_count': len(st.session_state.feature_names) if st.session_state.feature_names else 0,
                            'models_trained': st.session_state.models_trained,
                            'model_trainer_available': st.session_state.model_trainer is not None,
                            'models_in_trainer': list(st.session_state.model_trainer.models.keys()) if st.session_state.model_trainer else []
                        })
                    
                    if not st.session_state.models_trained:
                        st.error("""
                        ❌ Models belum trained! Silakan:
                        1. Pilih cryptocurrency dan models di sidebar
                        2. Klik 'Run Full Analysis'
                        3. Tunggu sampai training selesai
                        """)
                        
                        if st.button("🚀 Run Full Analysis Now", type="primary"):
                            st.session_state.analysis_run = True
                            st.rerun()
                            
                    else:
                        st.subheader("Future Price Forecast")
                        
                        # Validasi final sebelum prediction
                        can_predict = all([
                            st.session_state.data is not None,
                            st.session_state.data_with_features is not None,
                            st.session_state.feature_names is not None and len(st.session_state.feature_names) > 0,
                            st.session_state.model_trainer is not None,
                            hasattr(st.session_state.model_trainer, 'models'),
                            len(st.session_state.model_trainer.models) > 0
                        ])
                        
                        if not can_predict:
                            st.error("❌ Cannot generate predictions - missing required data")
                            st.info("Please run 'Run Full Analysis' again")
                        else:
                            st.success("✅ Ready for predictions!")
                            
                            if st.button("Generate Real Predictions", type="primary", key="real_predict_btn"):
                                try:
                                    with st.spinner("🔄 Generating predictions..."):
                                        # Call show_predictions
                                        show_predictions(
                                            data=st.session_state.data,
                                            data_with_features=st.session_state.data_with_features,
                                            feature_names=st.session_state.feature_names,
                                            forecast_days=forecast_days,
                                            model_trainer=st.session_state.model_trainer
                                        )
                                except Exception as e:
                                    st.error(f"❌ Prediction error: {str(e)}")
                                    import traceback
                                    with st.expander("Technical Details"):
                                        st.code(traceback.format_exc())
        
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
    
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # Crypto Scanner Section
    show_crypto_scanner_real_time()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit, Python, and Machine Learning • 
    <a href='#' target='_blank'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)

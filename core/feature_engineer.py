# core/feature_engineer.py
import pandas as pd
import numpy as np
import talib as ta

class FeatureEngineer:
    def __init__(self):
        self.indicators = []
    
    def add_all_indicators(self, df):
        """Add comprehensive technical indicators menggunakan TA-Lib"""
        try:
            # Convert to numpy arrays for TA-Lib
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # ===== ENHANCEMENT: Tambahkan lebih banyak indicators =====
            
            # Trend Indicators
            df['sma_10'] = ta.SMA(close, timeperiod=10)
            df['sma_20'] = ta.SMA(close, timeperiod=20)
            df['sma_50'] = ta.SMA(close, timeperiod=50)
            df['ema_12'] = ta.EMA(close, timeperiod=12)
            df['ema_26'] = ta.EMA(close, timeperiod=26)
            
            # MACD dengan lebih banyak features
            macd, macd_signal, macd_hist = ta.MACD(close)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            df['macd_diff'] = macd - macd_signal  # ENHANCEMENT: Tambahkan difference
            
            # Multiple RSI periods
            df['rsi_7'] = ta.RSI(close, timeperiod=7)   # ENHANCEMENT: Short-term RSI
            df['rsi_14'] = ta.RSI(close, timeperiod=14)
            df['rsi_21'] = ta.RSI(close, timeperiod=21) # ENHANCEMENT: Long-term RSI
            
            # Bollinger Bands dengan multiple deviations
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # ENHANCEMENT: Position in BB
            
            # Stochastic dengan multiple timeframes
            slowk, slowd = ta.STOCH(high, low, close)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            df['stoch_diff'] = slowk - slowd  # ENHANCEMENT: Difference
            
            # Williams %R
            df['williams_r'] = ta.WILLR(high, low, close, timeperiod=14)
            
            # CMO - Chande Momentum Oscillator
            df['cmo'] = ta.CMO(close, timeperiod=14)
            
            # ROC - Rate of Change
            df['roc_10'] = ta.ROC(close, timeperiod=10)
            df['roc_21'] = ta.ROC(close, timeperiod=21)  # ENHANCEMENT: Multiple ROC
            
            # ATR - Average True Range
            df['atr'] = ta.ATR(high, low, close, timeperiod=14)
            
            # ADX - Average Directional Index
            df['adx'] = ta.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = ta.PLUS_DI(high, low, close, timeperiod=14)  # ENHANCEMENT: Directional indicators
            df['minus_di'] = ta.MINUS_DI(high, low, close, timeperiod=14)
            
            # Volume Indicators dengan enhancement
            df['obv'] = ta.OBV(close, volume)
            df['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)
            df['volume_sma_10'] = ta.SMA(volume, timeperiod=10)  # ENHANCEMENT: Volume MA
            df['volume_sma_20'] = ta.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma_20']  # ENHANCEMENT: Volume ratio
            
            # Additional custom indicators - ENHANCEMENT
            df['price_rate_of_change'] = ta.ROC(close, timeperiod=10)
            df['high_low_ratio'] = high / low
            df['close_open_ratio'] = close / df['open']
            
            # Price position relative to moving averages - ENHANCEMENT
            df['price_vs_sma10'] = close / df['sma_10']
            df['price_vs_sma20'] = close / df['sma_20']
            df['price_vs_sma50'] = close / df['sma_50']
            df['sma_cross'] = df['sma_10'] / df['sma_50']  # ENHANCEMENT: MA crossover signal
            
            # Volatility indicators - ENHANCEMENT
            df['true_range'] = ta.TRANGE(high, low, close)
            df['volatility_10'] = df['close'].rolling(10).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            
            # Momentum indicators - ENHANCEMENT
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            
            # Drop NaN values
            df = df.dropna()
            
            self.indicators = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            print(f"✅ Enhanced: Added {len(self.indicators)} technical indicators")
            
            return df
            
        except Exception as e:
            print(f"Error in enhanced feature engineering: {e}")
            # Fallback to original implementation
            return self._add_simple_indicators(df)
    
    def _add_simple_indicators(self, df):
        """Simple indicators sebagai fallback"""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        self.indicators = ['sma_20', 'sma_50', 'rsi_14', 'volume_sma']
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_feature_names(self):
        return self.indicators

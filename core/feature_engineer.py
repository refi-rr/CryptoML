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
            
            # Trend Indicators
            df['sma_20'] = ta.SMA(close, timeperiod=20)
            df['sma_50'] = ta.SMA(close, timeperiod=50)
            df['ema_12'] = ta.EMA(close, timeperiod=12)
            df['ema_26'] = ta.EMA(close, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(close)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # RSI
            df['rsi_14'] = ta.RSI(close, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic
            slowk, slowd = ta.STOCH(high, low, close)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Williams %R
            df['williams_r'] = ta.WILLR(high, low, close, timeperiod=14)
            
            # CMO
            df['cmo'] = ta.CMO(close, timeperiod=14)
            
            # ROC
            df['roc'] = ta.ROC(close, timeperiod=10)
            
            # ATR
            df['atr'] = ta.ATR(high, low, close, timeperiod=14)
            
            # ADX
            df['adx'] = ta.ADX(high, low, close, timeperiod=14)
            
            # Volume Indicators
            df['obv'] = ta.OBV(close, volume)
            df['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)
            
            # Additional custom indicators
            df['price_rate_of_change'] = ta.ROC(close, timeperiod=10)
            df['volume_sma'] = ta.SMA(volume, timeperiod=20)
            df['high_low_ratio'] = high / low
            
            # Price position relative to moving averages
            df['price_vs_sma20'] = close / df['sma_20']
            df['price_vs_sma50'] = close / df['sma_50']
            
            # Drop NaN values
            df = df.dropna()
            
            self.indicators = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            print(f"Added {len(self.indicators)} technical indicators")
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            # Fallback to simple indicators
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

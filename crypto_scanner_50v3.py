import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
warnings.filterwarnings('ignore')

class CryptoScanner:
    def __init__(self, use_backup=False):
        """
        Inisialisasi scanner dengan Binance Public API
        use_backup: gunakan Bybit sebagai alternatif jika Binance bermasalah
        """
        self.use_backup = use_backup
        
        if use_backup:
            self.base_url = 'https://api.bybit.com'
            self.exchange_name = 'Bybit'
        else:
            self.base_url = 'https://fapi.binance.com'
            self.exchange_name = 'Binance'
        
        # Setup session dengan retry mechanism
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def get_top_coins_binance(self, limit=50):
        """Ambil top coins dari Binance Futures"""
        try:
            url = f'{self.base_url}/fapi/v1/ticker/24hr'
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            usdt_pairs = [
                item for item in data 
                if item['symbol'].endswith('USDT') and 
                float(item['quoteVolume']) > 0 and
                item['symbol'] not in ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT']
            ]
            
            sorted_pairs = sorted(usdt_pairs, 
                                key=lambda x: float(x['quoteVolume']), 
                                reverse=True)
            
            return [pair['symbol'] for pair in sorted_pairs[:limit]]
        
        except Exception as e:
            print(f"❌ Error getting coins from Binance: {e}")
            return None
    
    def get_top_coins_bybit(self, limit=50):
        """Ambil top coins dari Bybit (backup)"""
        try:
            url = f'{self.base_url}/v5/market/tickers'
            params = {'category': 'linear'}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['retCode'] != 0:
                return None
            
            usdt_pairs = [
                item for item in data['result']['list']
                if item['symbol'].endswith('USDT') and
                float(item['turnover24h']) > 0
            ]
            
            sorted_pairs = sorted(usdt_pairs,
                                key=lambda x: float(x['turnover24h']),
                                reverse=True)
            
            return [pair['symbol'] for pair in sorted_pairs[:limit]]
        
        except Exception as e:
            print(f"❌ Error getting coins from Bybit: {e}")
            return None
    
    def get_top_coins(self, limit=50):
        """Ambil top coins dengan fallback mechanism"""
        if self.use_backup:
            coins = self.get_top_coins_bybit(limit)
        else:
            coins = self.get_top_coins_binance(limit)
            
            if coins is None:
                print("\n⚠️  Binance timeout, switching to Bybit...\n")
                self.use_backup = True
                self.base_url = 'https://api.bybit.com'
                self.exchange_name = 'Bybit'
                coins = self.get_top_coins_bybit(limit)
        
        return coins if coins else []
    
    def fetch_ohlcv_binance(self, symbol, interval='4h', limit=500):
        """Ambil data OHLCV dari Binance"""
        try:
            url = f'{self.base_url}/fapi/v1/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            return None
    
    def fetch_ohlcv_bybit(self, symbol, interval='240', limit=500):
        """Ambil data OHLCV dari Bybit"""
        try:
            url = f'{self.base_url}/v5/market/kline'
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['retCode'] != 0:
                return None
            
            klines = data['result']['list']
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.iloc[::-1].reset_index(drop=True)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            return None
    
    def fetch_ohlcv(self, symbol, interval='4h', limit=500):
        """Fetch OHLCV dengan auto-switch exchange"""
        interval_map = {
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }
        
        if self.use_backup:
            bybit_interval = interval_map.get(interval, '240')
            return self.fetch_ohlcv_bybit(symbol, bybit_interval, limit)
        else:
            return self.fetch_ohlcv_binance(symbol, interval, limit)
    
    def calculate_rsi(self, df, period=14):
        """Hitung RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stoch_rsi(self, df, period=14, smooth_k=3, smooth_d=3):
        """Hitung Stochastic RSI"""
        rsi = self.calculate_rsi(df, period)
        stoch_rsi = (rsi - rsi.rolling(window=period).min()) / \
                    (rsi.rolling(window=period).max() - rsi.rolling(window=period).min()) * 100
        k = stoch_rsi.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        return k, d
    
    def calculate_ema(self, df, periods=[21, 50, 200]):
        """Hitung EMA"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Hitung MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self, df, period=20, std=2):
        """Hitung Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    def calculate_adx(self, df, period=14):
        """Hitung ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_atr(self, df, period=14):
        """Hitung Average True Range untuk volatility"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def find_support_resistance(self, df, lookback=50, num_levels=3):
        """
        Deteksi Support dan Resistance levels
        Menggunakan pivot points dan clustering price levels
        """
        recent_data = df.tail(lookback)
        
        # Find pivot highs (resistance candidates)
        pivot_highs = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                pivot_highs.append(recent_data['high'].iloc[i])
        
        # Find pivot lows (support candidates)
        pivot_lows = []
        for i in range(2, len(recent_data) - 2):
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                pivot_lows.append(recent_data['low'].iloc[i])
        
        # Cluster similar levels
        def cluster_levels(levels, tolerance=0.02):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        resistance_levels = cluster_levels(pivot_highs)
        support_levels = cluster_levels(pivot_lows)
        
        # Get closest levels to current price
        current_price = df['close'].iloc[-1]
        
        resistance_levels = [r for r in resistance_levels if r > current_price]
        support_levels = [s for s in support_levels if s < current_price]
        
        resistance_levels = sorted(resistance_levels)[:num_levels]
        support_levels = sorted(support_levels, reverse=True)[:num_levels]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'current_price': current_price
        }
    
    def calculate_fibonacci_levels(self, df, lookback=100):
        """
        Hitung Fibonacci Retracement levels
        """
        recent_data = df.tail(lookback)
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        diff = high - low
        
        fib_levels = {
            '0.0': high,
            '0.236': high - (diff * 0.236),
            '0.382': high - (diff * 0.382),
            '0.5': high - (diff * 0.5),
            '0.618': high - (diff * 0.618),
            '0.786': high - (diff * 0.786),
            '1.0': low
        }
        
        return fib_levels
    
    def calculate_volume_trend(self, df, period=20):
        """Deteksi peningkatan volume"""
        avg_volume = df['volume'].rolling(window=period).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 0
        return volume_ratio
    
    def estimate_entry_time(self, df, signal_data, sr_levels):
        """
        Estimasi waktu entry berdasarkan analisis teknikal
        Return: urgency_level, estimated_time, reason
        """
        current_price = df['close'].iloc[-1]
        rsi = signal_data['rsi']
        stoch_k = signal_data['stoch_k']
        stoch_d = signal_data['stoch_d']
        volume_ratio = signal_data['volume_ratio']
        trend = signal_data['trend']
        
        # Analisis kondisi saat ini untuk estimasi waktu
        if 'LONG' in trend:
            # Untuk LONG position
            if sr_levels['support']:
                nearest_support = sr_levels['support'][0]
                distance_to_support = abs(current_price - nearest_support) / current_price * 100
                
                if distance_to_support <= 0.5:
                    if rsi < 35 and stoch_k < 25:
                        return "SEGERA", "0-4 jam", f"Harga dekat support (${nearest_support:.6f}), RSI oversold ({rsi:.1f})"
                    else:
                        return "SIAP", "4-12 jam", f"Harga di area support, tunggu konfirmasi bullish"
                elif distance_to_support <= 2:
                    return "TUNGGU", "12-24 jam", f"Menunggu harga mendekati support (${nearest_support:.6f})"
                else:
                    return "PANTAU", "24+ jam", f"Support jauh di ${nearest_support:.6f}, tunggu koreksi"
            
            else:
                if rsi < 30 and volume_ratio > 1.5:
                    return "SEGERA", "0-4 jam", f"RSI oversold ({rsi:.1f}) dengan volume tinggi ({volume_ratio:.1f}x)"
                elif rsi < 35:
                    return "SIAP", "4-12 jam", f"RSI oversold ({rsi:.1f}), tunggu konfirmasi"
                else:
                    return "TUNGGU", "12-24 jam", f"Menunggu kondisi oversold (RSI: {rsi:.1f})"
        
        elif 'SHORT' in trend:
            # Untuk SHORT position
            if sr_levels['resistance']:
                nearest_resistance = sr_levels['resistance'][0]
                distance_to_resistance = abs(current_price - nearest_resistance) / current_price * 100
                
                if distance_to_resistance <= 0.5:
                    if rsi > 65 and stoch_k > 75:
                        return "SEGERA", "0-4 jam", f"Harga dekat resistance (${nearest_resistance:.6f}), RSI overbought ({rsi:.1f})"
                    else:
                        return "SIAP", "4-12 jam", f"Harga di area resistance, tunggu konfirmasi bearish"
                elif distance_to_resistance <= 2:
                    return "TUNGGU", "12-24 jam", f"Menunggu harga mendekati resistance (${nearest_resistance:.6f})"
                else:
                    return "PANTAU", "24+ jam", f"Resistance jauh di ${nearest_resistance:.6f}, tunggu rally"
            
            else:
                if rsi > 70 and volume_ratio > 1.5:
                    return "SEGERA", "0-4 jam", f"RSI overbought ({rsi:.1f}) dengan volume tinggi ({volume_ratio:.1f}x)"
                elif rsi > 65:
                    return "SIAP", "4-12 jam", f"RSI overbought ({rsi:.1f}), tunggu konfirmasi"
                else:
                    return "TUNGGU", "12-24 jam", f"Menunggu kondisi overbought (RSI: {rsi:.1f})"
        
        return "PANTAU", "24+ jam", "Kondisi netral, tunggu sinyal lebih kuat"
    
    def calculate_price_momentum(self, df):
        """Hitung momentum harga untuk konfirmasi trend"""
        # Price change dalam berbagai timeframe
        price_1h = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100
        price_4h = (df['close'].iloc[-1] - df['close'].iloc[-16]) / df['close'].iloc[-16] * 100
        price_1d = (df['close'].iloc[-1] - df['close'].iloc[-96]) / df['close'].iloc[-96] * 100
        
        return {
            '1h_change': price_1h,
            '4h_change': price_4h,
            '1d_change': price_1d
        }
    
    def generate_trading_advice(self, df, signal_data, sr_levels):
        """
        Generate Entry, Stop Loss, dan Take Profit recommendations
        Risk:Reward = 1:1, 1:1.5, 1:2 untuk TP1, TP2, TP3
        """
        current_price = df['close'].iloc[-1]
        atr = self.calculate_atr(df).iloc[-1]
        trend = signal_data['trend']
        
        # Estimasi waktu entry
        urgency, estimated_time, reason = self.estimate_entry_time(df, signal_data, sr_levels)
        
        # Momentum analysis
        momentum = self.calculate_price_momentum(df)
        
        advice = {
            'entry_zone': [],
            'stop_loss': None,
            'take_profits': [],
            'risk_reward': '1:1, 1:1.5, 1:2',
            'position_size_advice': None,
            'entry_urgency': urgency,
            'estimated_entry_time': estimated_time,
            'entry_reason': reason,
            'price_momentum': momentum
        }
        
        if 'LONG' in trend:
            # LONG POSITION
            # Entry: near support or current price
            if sr_levels['support']:
                nearest_support = sr_levels['support'][0]
                entry_low = max(nearest_support * 0.995, current_price * 0.98)
                entry_high = min(nearest_support * 1.005, current_price * 1.01)
            else:
                entry_low = current_price * 0.98
                entry_high = current_price * 1.005
            
            advice['entry_zone'] = [entry_low, entry_high]
            
            # Stop Loss: below support or using ATR
            if sr_levels['support']:
                sl = sr_levels['support'][0] * 0.985  # 1.5% below support
            else:
                sl = current_price - (atr * 1.5)
            
            advice['stop_loss'] = sl
            
            # Calculate risk
            risk = entry_high - sl
            
            # Take Profits dengan Fibonacci extensions
            tp1 = entry_high + (risk * 1.0)   # 1:1
            tp2 = entry_high + (risk * 1.5)   # 1:1.5
            tp3 = entry_high + (risk * 2.0)   # 1:2
            
            # Adjust TP berdasarkan resistance levels
            if sr_levels['resistance']:
                for resistance in sr_levels['resistance']:
                    if resistance > tp1 and resistance < tp3:
                        tp2 = min(tp2, resistance)
            
            advice['take_profits'] = [
                {'level': 'TP1', 'price': tp1, 'rr': '1:1'},
                {'level': 'TP2', 'price': tp2, 'rr': '1:1.5'},
                {'level': 'TP3', 'price': tp3, 'rr': '1:2'}
            ]
            
        elif 'SHORT' in trend:
            # SHORT POSITION
            # Entry: near resistance or current price
            if sr_levels['resistance']:
                nearest_resistance = sr_levels['resistance'][0]
                entry_low = max(nearest_resistance * 0.995, current_price * 0.99)
                entry_high = min(nearest_resistance * 1.005, current_price * 1.02)
            else:
                entry_low = current_price * 0.995
                entry_high = current_price * 1.02
            
            advice['entry_zone'] = [entry_low, entry_high]
            
            # Stop Loss: above resistance or using ATR
            if sr_levels['resistance']:
                sl = sr_levels['resistance'][0] * 1.015  # 1.5% above resistance
            else:
                sl = current_price + (atr * 1.5)
            
            advice['stop_loss'] = sl
            
            # Calculate risk
            risk = sl - entry_low
            
            # Take Profits
            tp1 = entry_low - (risk * 1.0)   # 1:1
            tp2 = entry_low - (risk * 1.5)   # 1:1.5
            tp3 = entry_low - (risk * 2.0)   # 1:2
            
            # Adjust TP berdasarkan support levels
            if sr_levels['support']:
                for support in sr_levels['support']:
                    if support < tp1 and support > tp3:
                        tp2 = max(tp2, support)
            
            advice['take_profits'] = [
                {'level': 'TP1', 'price': tp1, 'rr': '1:1'},
                {'level': 'TP2', 'price': tp2, 'rr': '1:1.5'},
                {'level': 'TP3', 'price': tp3, 'rr': '1:2'}
            ]
        
        # Position size advice based on risk
        if advice['stop_loss']:
            risk_percent = abs((advice['entry_zone'][1] - advice['stop_loss']) / advice['entry_zone'][1] * 100)
            if risk_percent > 5:
                position_advice = f"Risk tinggi: {risk_percent:.2f}% | Gunakan 0.5-1% dari capital"
            elif risk_percent > 3:
                position_advice = f"Risk medium: {risk_percent:.2f}% | Gunakan 1-2% dari capital"
            else:
                position_advice = f"Risk rendah: {risk_percent:.2f}% | Gunakan 2-3% dari capital"
            
            advice['position_size_advice'] = position_advice
        
        return advice
    
    def analyze_signal(self, df, timeframe):
        """Analisis sinyal trading dengan S/R dan trading advice"""
        if df is None or len(df) < 200:
            return None
        
        df = self.calculate_ema(df)
        rsi = self.calculate_rsi(df)
        stoch_k, stoch_d = self.calculate_stoch_rsi(df)
        macd, signal, histogram = self.calculate_macd(df)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(df)
        adx, plus_di, minus_di = self.calculate_adx(df)
        volume_ratio = self.calculate_volume_trend(df)
        
        # Support & Resistance
        sr_levels = self.find_support_resistance(df)
        fib_levels = self.calculate_fibonacci_levels(df)
        
        last_close = df['close'].iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_stoch_k = stoch_k.iloc[-1]
        last_stoch_d = stoch_d.iloc[-1]
        last_macd = macd.iloc[-1]
        last_signal = signal.iloc[-1]
        last_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2]
        last_adx = adx.iloc[-1]
        
        ema_21 = df['ema_21'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1]
        prev_ema_21 = df['ema_21'].iloc[-2]
        prev_ema_50 = df['ema_50'].iloc[-2]
        
        signals = {
            'timeframe': timeframe,
            'price': last_close,
            'rsi': last_rsi,
            'stoch_k': last_stoch_k,
            'stoch_d': last_stoch_d,
            'macd': last_macd,
            'signal_line': last_signal,
            'adx': last_adx,
            'volume_ratio': volume_ratio,
            'support_resistance': sr_levels,
            'fibonacci': fib_levels,
            'trend': None,
            'signal_strength': 0,
            'signals': []
        }
        
        score = 0
        
        # RSI Analysis
        if last_rsi < 30:
            signals['signals'].append('RSI Oversold (Long)')
            score += 2
        elif last_rsi > 70:
            signals['signals'].append('RSI Overbought (Short)')
            score -= 2
        
        # Stoch RSI Analysis
        if last_stoch_k < 20 and last_stoch_d < 20:
            signals['signals'].append('Stoch RSI Oversold (Long)')
            score += 2
        elif last_stoch_k > 80 and last_stoch_d > 80:
            signals['signals'].append('Stoch RSI Overbought (Short)')
            score -= 2
        
        # Stoch Cross
        if last_stoch_k > last_stoch_d and stoch_k.iloc[-2] <= stoch_d.iloc[-2]:
            signals['signals'].append('Stoch RSI Golden Cross (Long)')
            score += 3
        elif last_stoch_k < last_stoch_d and stoch_k.iloc[-2] >= stoch_d.iloc[-2]:
            signals['signals'].append('Stoch RSI Death Cross (Short)')
            score -= 3
        
        # EMA Analysis
        if ema_21 > ema_50 > ema_200:
            signals['signals'].append('EMA Bullish Alignment')
            score += 2
        elif ema_21 < ema_50 < ema_200:
            signals['signals'].append('EMA Bearish Alignment')
            score -= 2
        
        # EMA Cross
        if prev_ema_21 <= prev_ema_50 and ema_21 > ema_50:
            signals['signals'].append('EMA 21/50 Golden Cross (Long)')
            score += 3
        elif prev_ema_21 >= prev_ema_50 and ema_21 < ema_50:
            signals['signals'].append('EMA 21/50 Death Cross (Short)')
            score -= 3
        
        # MACD Analysis
        if last_macd > last_signal and prev_histogram < 0 and last_histogram > 0:
            signals['signals'].append('MACD Bullish Cross (Long)')
            score += 3
        elif last_macd < last_signal and prev_histogram > 0 and last_histogram < 0:
            signals['signals'].append('MACD Bearish Cross (Short)')
            score -= 3
        
        # Volume Analysis
        if volume_ratio > 1.5:
            signals['signals'].append(f'High Volume ({volume_ratio:.2f}x)')
            score += 1
        
        # ADX Trend Strength
        if last_adx > 25:
            signals['signals'].append(f'Strong Trend (ADX: {last_adx:.1f})')
            score += 1
        
        # Bollinger Bands
        if last_close < lower_bb.iloc[-1]:
            signals['signals'].append('Price Below Lower BB (Long)')
            score += 1
        elif last_close > upper_bb.iloc[-1]:
            signals['signals'].append('Price Above Upper BB (Short)')
            score -= 1
        
        # Support/Resistance proximity
        if sr_levels['support']:
            nearest_support = sr_levels['support'][0]
            if abs(last_close - nearest_support) / last_close < 0.01:  # within 1%
                signals['signals'].append('Near Support Level (Long)')
                score += 2
        
        if sr_levels['resistance']:
            nearest_resistance = sr_levels['resistance'][0]
            if abs(last_close - nearest_resistance) / last_close < 0.01:  # within 1%
                signals['signals'].append('Near Resistance Level (Short)')
                score -= 2
        
        # Determine trend
        signals['signal_strength'] = score
        if score >= 5:
            signals['trend'] = 'STRONG LONG'
        elif score >= 3:
            signals['trend'] = 'LONG'
        elif score <= -5:
            signals['trend'] = 'STRONG SHORT'
        elif score <= -3:
            signals['trend'] = 'SHORT'
        else:
            signals['trend'] = 'NEUTRAL'
        
        # Generate trading advice
        if signals['trend'] != 'NEUTRAL':
            signals['trading_advice'] = self.generate_trading_advice(df, signals, sr_levels)
        
        return signals
    
    def scan_markets(self, timeframes=['4h', '1h'], top_n=50):
        """Scan markets for trading opportunities"""
        print(f"🔍 Memulai scanning top {top_n} crypto coins...")
        print(f"⏰ Timeframes: {', '.join(timeframes)}")
        print(f"🌐 Exchange: {self.exchange_name} Public API")
        print("="*80)
        
        coins = self.get_top_coins(top_n)
        if not coins:
            print("❌ Failed to get coin list. Please check your internet connection.")
            return []
        
        print(f"✅ Found {len(coins)} USDT futures pairs\n")
        
        results = []
        
        for i, symbol in enumerate(coins, 1):
            print(f"[{i}/{len(coins)}] Analyzing {symbol}...", end='\r')
            
            try:
                signals_all_tf = {}
                
                for tf in timeframes:
                    df = self.fetch_ohlcv(symbol, tf)
                    signal = self.analyze_signal(df, tf)
                    if signal:
                        signals_all_tf[tf] = signal
                    time.sleep(0.3)
                
                if len(signals_all_tf) == len(timeframes):
                    trends = [s['trend'] for s in signals_all_tf.values()]
                    
                    if 'NEUTRAL' not in trends[0]:
                        is_bullish = all('LONG' in t for t in trends)
                        is_bearish = all('SHORT' in t for t in trends)
                        
                        if is_bullish or is_bearish:
                            results.append({
                                'symbol': symbol,
                                'signals': signals_all_tf
                            })
                
            except Exception as e:
                continue
        
        print("\n" + "="*80)
        return results
    
    def display_results(self, results):
        """Display scan results with trading advice"""
        if not results:
            print("❌ Tidak ada sinyal trading yang ditemukan.")
            return
        
        print(f"\n✅ Ditemukan {len(results)} peluang trading!\n")
        
        long_signals = [r for r in results if 'LONG' in list(r['signals'].values())[0]['trend']]
        short_signals = [r for r in results if 'SHORT' in list(r['signals'].values())[0]['trend']]
        
        long_signals.sort(key=lambda x: list(x['signals'].values())[0]['signal_strength'], reverse=True)
        short_signals.sort(key=lambda x: abs(list(x['signals'].values())[0]['signal_strength']), reverse=True)
        
        if long_signals:
            print("\n🟢 " + "="*35 + " LONG SIGNALS " + "="*35)
            for result in long_signals:
                self._print_signal_detail(result)
        
        if short_signals:
            print("\n🔴 " + "="*35 + " SHORT SIGNALS " + "="*35)
            for result in short_signals:
                self._print_signal_detail(result)
    
    def _print_signal_detail(self, result):
        """Print detailed signal information"""
        symbol = result['symbol']
        signals = result['signals']
        
        print(f"\n📊 {symbol}")
        print("=" * 80)
        
        for tf, data in signals.items():
            print(f"\n  ⏱  Timeframe: {tf.upper()}")
            print(f"  💰 Price: ${data['price']:.6f}")
            print(f"  📈 Trend: {data['trend']} (Score: {data['signal_strength']})")
            print(f"  📊 RSI: {data['rsi']:.2f} | Stoch K/D: {data['stoch_k']:.2f}/{data['stoch_d']:.2f}")
            print(f"  📉 MACD: {data['macd']:.6f} | Signal: {data['signal_line']:.6f}")
            print(f"  💪 ADX: {data['adx']:.2f} | Volume: {data['volume_ratio']:.2f}x")
            
            # Support & Resistance
            sr = data['support_resistance']
            if sr['support']:
                print(f"  🟢 Support: {', '.join([f'${s:.6f}' for s in sr['support']])}")
            else:
                print(f"  🟢 Support: No clear level detected")
            
            if sr['resistance']:
                print(f"  🔴 Resistance: {', '.join([f'${r:.6f}' for r in sr['resistance']])}")
            else:
                print(f"  🔴 Resistance: No clear level detected")
            
            print(f"  🎯 Signals: {', '.join(data['signals'])}")
            
            # Trading Advice (only for primary timeframe)
            if tf == '4h' and 'trading_advice' in data:
                advice = data['trading_advice']
                print(f"\n  {'='*76}")
                print(f"  💡 TRADING ADVICE & TIMING")
                print(f"  {'='*76}")
                
                # Entry timing information
                urgency_icon = "🚨" if advice['entry_urgency'] == "SEGERA" else "⚠️" if advice['entry_urgency'] == "SIAP" else "⏳"
                print(f"  {urgency_icon} Entry Urgency: {advice['entry_urgency']}")
                print(f"  ⏰ Estimated Time: {advice['estimated_entry_time']}")
                print(f"  📝 Reason: {advice['entry_reason']}")
                
                # Price momentum
                momentum = advice['price_momentum']
                print(f"  📊 Momentum - 1h: {momentum['1h_change']:+.2f}% | 4h: {momentum['4h_change']:+.2f}% | 1d: {momentum['1d_change']:+.2f}%")
                
                entry_low, entry_high = advice['entry_zone']
                print(f"  📍 Entry Zone: ${entry_low:.6f} - ${entry_high:.6f}")
                print(f"  🛑 Stop Loss: ${advice['stop_loss']:.6f}")
                print(f"  🎯 Take Profits:")
                
                for tp in advice['take_profits']:
                    print(f"     • {tp['level']}: ${tp['price']:.6f} (R:R {tp['rr']})")
                
                if advice['position_size_advice']:
                    print(f"  ⚖️  {advice['position_size_advice']}")
                
                # Calculate potential profit percentages
                entry_mid = (entry_low + entry_high) / 2
                for tp in advice['take_profits']:
                    profit_pct = abs((tp['price'] - entry_mid) / entry_mid * 100)
                    print(f"     {tp['level']} Profit: {profit_pct:.2f}%")
        
        print("=" * 80)


def main():
    print("\n" + "="*80)
    print("🚀 ADVANCED CRYPTO FUTURES SCANNER - TOP 50 COINS")
    print("📡 Using Public API (No Authentication Required)")
    print("✨ Features: S/R Detection, Entry/SL/TP Recommendations, Entry Timing")
    print("="*80 + "\n")
    
    # Start with Binance, auto-switch to Bybit if needed
    scanner = CryptoScanner(use_backup=False)
    
    results = scanner.scan_markets(timeframes=['4h', '1h'], top_n=50)
    scanner.display_results(results)
    
    # Save results to file
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_scan_advanced_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Advanced Crypto Scan Results - {datetime.now()}\n")
            f.write(f"Exchange: {scanner.exchange_name}\n")
            f.write("="*80 + "\n\n")
            
            for result in results:
                symbol = result['symbol']
                signals = result['signals']
                
                f.write(f"\n{'='*80}\n")
                f.write(f"Symbol: {symbol}\n")
                f.write(f"{'='*80}\n\n")
                
                for tf, data in signals.items():
                    f.write(f"Timeframe: {tf.upper()}\n")
                    f.write(f"Price: ${data['price']:.6f}\n")
                    f.write(f"Trend: {data['trend']} (Score: {data['signal_strength']})\n")
                    f.write(f"RSI: {data['rsi']:.2f} | Stoch K/D: {data['stoch_k']:.2f}/{data['stoch_d']:.2f}\n")
                    f.write(f"MACD: {data['macd']:.6f} | Signal: {data['signal_line']:.6f}\n")
                    f.write(f"ADX: {data['adx']:.2f} | Volume: {data['volume_ratio']:.2f}x\n")
                    
                    # Support & Resistance
                    sr = data['support_resistance']
                    if sr['support']:
                        f.write(f"Support: {', '.join([f'${s:.6f}' for s in sr['support']])}\n")
                    if sr['resistance']:
                        f.write(f"Resistance: {', '.join([f'${r:.6f}' for r in sr['resistance']])}\n")
                    
                    f.write(f"Signals: {', '.join(data['signals'])}\n")
                    
                    # Trading Advice
                    if tf == '4h' and 'trading_advice' in data:
                        advice = data['trading_advice']
                        f.write(f"\nTRADING ADVICE & TIMING:\n")
                        f.write(f"Entry Urgency: {advice['entry_urgency']}\n")
                        f.write(f"Estimated Time: {advice['estimated_entry_time']}\n")
                        f.write(f"Reason: {advice['entry_reason']}\n")
                        f.write(f"Entry Zone: ${advice['entry_zone'][0]:.6f} - ${advice['entry_zone'][1]:.6f}\n")
                        f.write(f"Stop Loss: ${advice['stop_loss']:.6f}\n")
                        f.write(f"Take Profits:\n")
                        for tp in advice['take_profits']:
                            f.write(f"  {tp['level']}: ${tp['price']:.6f} (R:R {tp['rr']})\n")
                        f.write(f"{advice['position_size_advice']}\n")
                    
                    f.write("\n")
        
        print(f"\n💾 Results saved to: {filename}")
    
    print("\n" + "="*80)
    print("📝 TRADING TIPS:")
    print("   • SEGERA: Entry dalam 0-4 jam - kondisi sudah matang")
    print("   • SIAP: Entry dalam 4-12 jam - tunggu konfirmasi terakhir")
    print("   • TUNGGU: Entry dalam 12-24 jam - tunggu harga mencapai level optimal")
    print("   • PANTAU: Entry 24+ jam - kondisi belum optimal, pantau terus")
    print("   • Always use proper risk management (1-2% per trade)")
    print("   • Wait for confirmation on multiple timeframes")
    print("   • Take partial profits at TP1, let winners run to TP2/TP3")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

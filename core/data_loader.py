# core/data_loader.py
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import time

class CryptoDataLoader:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.symbol_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano',
            'DOT': 'polkadot', 'LINK': 'chainlink', 'LTC': 'litecoin',
            'XRP': 'ripple', 'BCH': 'bitcoin-cash', 'EOS': 'eos',
            'XLM': 'stellar', 'ATOM': 'cosmos', 'SOL': 'solana',
            'DOGE': 'dogecoin', 'MATIC': 'matic-network', 'AVAX': 'avalanche-2'
        }
    
    def get_historical_data(self, symbol, days=365):
        """Get real OHLC data from CoinGecko dengan data terbaru"""
        coin_id = self.symbol_map.get(symbol.upper(), symbol.lower())
        
        try:
            # Get data sampai hari ini
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
            
            # Get market chart data
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id, 
                vs_currency='usd', 
                days=days,
                interval='daily'
            )
            
            if not data or 'prices' not in data:
                raise Exception("No data returned from CoinGecko")
            
            # Process the data
            prices = data['prices']
            market_caps = data.get('market_caps', [])
            total_volumes = data.get('total_volumes', [])
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume
            if total_volumes:
                df['volume'] = [vol[1] for vol in total_volumes[:len(df)]]
            else:
                df['volume'] = 0
            
            # Create proper OHLC data (CoinGecko hanya provide close, jadi kita estimasi OHLC)
            df['open'] = df['close'].shift(1)
            df['high'] = df[['open', 'close']].max(axis=1)
            df['low'] = df[['open', 'close']].min(axis=1)
            
            # Handle first row
            if len(df) > 0:
                df.iloc[0, df.columns.get_loc('open')] = df.iloc[0]['close']
                df.iloc[0, df.columns.get_loc('high')] = df.iloc[0]['close']
                df.iloc[0, df.columns.get_loc('low')] = df.iloc[0]['close']
            
            # Reorder columns to standard OHLCV format
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.dropna()
            
            print(f"Successfully loaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            # Return sample data sebagai fallback
            return self._get_sample_data(days)
    
    def _get_sample_data(self, days=365):
        """Generate sample data sebagai fallback"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        # Simulate realistic price movement
        returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
        prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': prices * 0.998,
            'high': prices * 1.015,
            'low': prices * 0.985,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, days)
        }, index=dates)
        
        return data
    
    def get_multiple_pairs(self, symbols, days=365):
        """Get data for multiple cryptocurrency pairs"""
        data_dict = {}
        for symbol in symbols:
            print(f"Fetching {symbol} data...")
            data = self.get_historical_data(symbol, days)
            if data is not None:
                data_dict[symbol] = data
            time.sleep(1.2)  # Rate limiting
        return data_dict
    
    # ENHANCEMENT: Tambahkan method untuk data quality check
    def validate_data_quality(self, df):
        """Validate data quality sebelum digunakan"""
        if df is None or len(df) == 0:
            return False
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.1:  # More than 10% missing
            print(f"⚠️ High missing data ratio: {missing_ratio:.2%}")
            return False
        
        # Check for zero or negative prices
        if (df['close'] <= 0).any():
            print("⚠️ Invalid price data detected")
            return False
        
        # Check for sufficient data points
        if len(df) < 100:
            print("⚠️ Insufficient data points")
            return False
        
        print("✅ Data quality check passed")
        return True

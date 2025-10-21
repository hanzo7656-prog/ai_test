# data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataProcessor:
    """پردازش و پاک‌سازی داده‌های بازار"""
    
    def __init__(self):
        self.processed_data = {}
    
    def clean_coin_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """پاک‌سازی داده‌های خام کوین‌ها"""
        if not raw_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # تبدیل انواع داده
        numeric_columns = ['price', 'priceBtc', 'volume', 'marketCap', 
                          'availableSupply', 'totalSupply', 'fullyDilutedValuation',
                          'priceChange1h', 'priceChange1d', 'priceChange1w']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # حذف داده‌های نامعتبر
        df = df.dropna(subset=['price', 'marketCap'])
        
        return df
    
    def process_chart_data(self, chart_data: List[Dict]) -> pd.DataFrame:
        """پردازش داده‌های چارت"""
        if not chart_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(chart_data)
        
        # تبدیل timestamp به datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('datetime')
        
        # اطمینان از numeric بودن قیمت و حجم
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        return df.dropna()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبه اندیکاتورهای تکنیکال"""
        if df.empty:
            return df
        
        # Moving Averages
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['price'])
        
        # MACD
        df['macd'] = self.calculate_macd(df['price'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['price'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """محاسبه MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """محاسبه باندهای بولینگر"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

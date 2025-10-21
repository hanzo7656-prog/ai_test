# data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import pandas_ta as ta  # ✅ اضافه شده

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
        """محاسبه اندیکاتورهای تکنیکال با pandas-ta"""
        if df.empty or 'price' not in df.columns:
            return df
        
        try:
            # استفاده از pandas-ta برای اندیکاتورهای پیشرفته
            # RSI
            df['rsi'] = ta.rsi(df['price'], length=14)
            
            # MACD
            macd = ta.macd(df['price'])
            if macd is not None:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_histogram'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bollinger = ta.bbands(df['price'], length=20)
            if bollinger is not None:
                df['bb_upper'] = bollinger['BBU_20_2.0']
                df['bb_middle'] = bollinger['BBM_20_2.0'] 
                df['bb_lower'] = bollinger['BBL_20_2.0']
            
            # Moving Averages
            df['sma_20'] = ta.sma(df['price'], length=20)
            df['sma_50'] = ta.sma(df['price'], length=50)
            df['ema_12'] = ta.ema(df['price'], length=12)
            
            # Stochastic
            stoch = ta.stoch(df['high'] if 'high' in df.columns else df['price'], 
                           df['low'] if 'low' in df.columns else df['price'], 
                           df['price'])
            if stoch is not None:
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            return df
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورها: {e}")
            # Fallback به محاسبات ساده
            return self._calculate_basic_indicators(df)
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """محاسبات پایه اندیکاتورها (fallback)"""
        # Moving Averages
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['price'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه RSI (fallback)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

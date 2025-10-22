# 📁 src/core/technical_analysis/classic_indicators.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import talib
from ...utils.memory_monitor import MemoryMonitor

class ClassicIndicators:
    """اندیکاتورهای کلاسیک بهینه‌شده"""
    
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI با مدیریت حافظه"""
        with self.memory_monitor.track("rsi_calculation"):
            return talib.RSI(prices, timeperiod=period)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD با خروجی کامل"""
        with self.memory_monitor.track("macd_calculation"):
            macd, macd_signal, macd_hist = talib.MACD(prices, fastperiod=fast, 
                                                     slowperiod=slow, signalperiod=signal)
            return {
                'macd': macd,
                'signal': macd_signal, 
                'histogram': macd_hist
            }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Dict:
        """باندهای بولینگر"""
        with self.memory_monitor.track("bollinger_calculation"):
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period, 
                                              nbdevup=std, nbdevdn=std)
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'band_width': (upper - lower) / middle,
                '%b': (prices - lower) / (upper - lower)  # موقعیت قیمت در باند
            }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict:
        """استوکاستیک"""
        with self.memory_monitor.track("stochastic_calculation"):
            slowk, slowd = talib.STOCH(high, low, close, 
                                     fastk_period=k_period, 
                                     slowk_period=d_period, 
                                     slowd_period=d_period)
            return {
                'k': slowk,
                'd': slowd,
                'oversold': slowk < 20,
                'overbought': slowk > 80
            }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        with self.memory_monitor.track("atr_calculation"):
            return talib.ATR(high, low, close, timeperiod=period)
    
    def get_indicators_summary(self) -> Dict:
        """خلاصه عملکرد اندیکاتورها"""
        return self.memory_monitor.get_usage_stats()

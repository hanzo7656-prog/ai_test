# 📁 src/core/technical_analysis/pattern_recognition.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib

class PatternRecognition:
    """تشخیص الگوهای قیمتی"""
    
    def __init__(self):
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'engulfing': self._detect_engulfing,
            'evening_star': self._detect_evening_star,
            'morning_star': self._detect_morning_star
        }
    
    def detect_candlestick_patterns(self, open_prices: pd.Series, high: pd.Series, 
                                  low: pd.Series, close: pd.Series) -> Dict:
        """تشخیص الگوهای شمعی"""
        patterns_detected = {}
        
        for pattern_name, pattern_func in self.patterns.items():
            detection = pattern_func(open_prices, high, low, close)
            patterns_detected[pattern_name] = detection
        
        return patterns_detected
    
    def _detect_doji(self, open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """تشخیص الگوی دوجی"""
        body_size = np.abs(close - open_p)
        total_range = high - low
        doji_condition = (body_size / total_range) < 0.1
        return doji_condition
    
    def _detect_hammer(self, open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """تشخیص الگوی چکش"""
        body_size = np.abs(close - open_p)
        lower_shadow = np.minimum(open_p, close) - low
        upper_shadow = high - np.maximum(open_p, close)
        
        hammer_condition = (
            (lower_shadow > 2 * body_size) &  # سایه پایینی بلند
            (upper_shadow < body_size * 0.5)   # سایه بالایی کوتاه
        )
        return hammer_condition
    
    def _detect_engulfing(self, open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """تشخیص الگوی انگالفینگ"""
        bullish_engulfing = (
            (close > open_p) &  # شمع فعلی صعودی
            (close.shift(1) < open_p.shift(1)) &  # شمع قبلی نزولی
            (open_p < close.shift(1)) &  # open فعلی کمتر از close قبلی
            (close > open_p.shift(1))    # close فعلی بیشتر از open قبلی
        )
        
        bearish_engulfing = (
            (close < open_p) &  # شمع فعلی نزولی
            (close.shift(1) > open_p.shift(1)) &  # شمع قبلی صعودی
            (open_p > close.shift(1)) &  # open فعلی بیشتر از close قبلی
            (close < open_p.shift(1))    # close فعلی کمتر از open قبلی
        )
        
        engulfing = pd.Series(index=open_p.index, data=False)
        engulfing[bullish_engulfing] = 'BULLISH_ENGULFING'
        engulfing[bearish_engulfing] = 'BEARISH_ENGULFING'
        
        return engulfing
    
    def _detect_evening_star(self, open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """تشخیص الگوی ستاره عصرگاهی"""
        # پیاده‌سازی ساده‌شده
        evening_star = (
            (close.shift(2) > open_p.shift(2)) &  # شمع ۲ دوره قبل صعودی
            (high.shift(1) > high.shift(2)) &     # شمع قبلی high بالاتر
            (close < open_p.shift(2))             # شمع فعلی پایین‌تر بسته شود
        )
        return evening_star
    
    def _detect_morning_star(self, open_p: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """تشخیص الگوی ستاره صبحگاهی"""
        morning_star = (
            (close.shift(2) < open_p.shift(2)) &  # شمع ۲ دوره قبل نزولی
            (low.shift(1) < low.shift(2)) &       # شمع قبلی low پایین‌تر
            (close > open_p.shift(2))             # شمع فعلی بالاتر بسته شود
        )
        return morning_star

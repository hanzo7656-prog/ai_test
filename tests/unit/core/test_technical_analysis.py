# 📁 tests/unit/core/test_technical_analysis.py

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.core.technical_analysis.signal_engine import IntelligentSignalEngine, SignalType
from src.core.technical_analysis.classic_indicators import ClassicIndicators
from src.core.technical_analysis.advanced_indicators import AdvancedIndicators

class TestSignalEngine:
    """تست موتور سیگنال"""
    
    def setup_method(self):
        self.signal_engine = IntelligentSignalEngine()
        self.sample_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50500, 50600, 50700, 50800, 50900],
            'low': [49500, 49600, 49700, 49800, 49900],
            'close': [50200, 50300, 50400, 50500, 50600],
            'volume': [1000, 1200, 1100, 1300, 1400]
        })
    
    def test_generate_signals(self):
        """تست تولید سیگنال‌ها"""
        market_data = {'BTC/USDT': self.sample_data}
        indicators = {
            'BTC/USDT': {
                'rsi': pd.Series([45, 50, 55, 60, 65]),
                'macd': pd.Series([10, 12, 15, 18, 20]),
                'macd_signal': pd.Series([8, 10, 12, 14, 16])
            }
        }
        
        signals = self.signal_engine.generate_signals(market_data, indicators)
        
        assert isinstance(signals, list)
    
    def test_calculate_momentum_score(self):
        """تست محاسبه امتیاز مومنتوم"""
        score = self.signal_engine._calculate_momentum_score(
            self.sample_data, 
            {'rsi': pd.Series([25, 30, 35]), 'macd': pd.Series([1, 2, 3]), 'macd_signal': pd.Series([0.5, 1, 1.5])}
        )
        
        assert isinstance(score, float)
        assert -1 <= score <= 1

class TestClassicIndicators:
    """تست اندیکاتورهای کلاسیک"""
    
    def setup_method(self):
        self.indicators = ClassicIndicators()
        self.prices = pd.Series([100, 102, 101, 105, 107, 106, 108, 110, 109, 111])
    
    def test_calculate_rsi(self):
        """تست محاسبه RSI"""
        rsi = self.indicators.calculate_rsi(self.prices, period=6)
        
        assert rsi is not None
        assert len(rsi) == len(self.prices)
    
    def test_calculate_macd(self):
        """تست محاسبه MACD"""
        macd_data = self.indicators.calculate_macd(self.prices)
        
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data

class TestAdvancedIndicators:
    """تست اندیکاتورهای پیشرفته"""
    
    def setup_method(self):
        self.adv_indicators = AdvancedIndicators()
        self.prices = pd.Series(np.random.normal(100, 5, 100))
    
    def test_adaptive_rsi(self):
        """تست RSI تطبیقی"""
        result = self.adv_indicators.adaptive_rsi(self.prices)
        
        assert 'individual_rsis' in result
        assert 'combined_rsi' in result
        assert 'signal' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

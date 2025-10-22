# 📁 src/core/technical_analysis/advanced_indicators.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .classic_indicators import ClassicIndicators

class AdvancedIndicators:
    """اندیکاتورهای پیشرفته و ترکیبی"""
    
    def __init__(self):
        self.classic = ClassicIndicators()
        
    def adaptive_rsi(self, prices: pd.Series, lookback_periods: List[int] = [14, 21, 28]) -> Dict:
        """RSI تطبیقی با چند دوره lookback"""
        adaptive_results = {}
        
        for period in lookback_periods:
            rsi = self.classic.calculate_rsi(prices, period)
            adaptive_results[f'rsi_{period}'] = rsi
            
        # ترکیب RSI‌های مختلف
        combined_rsi = pd.DataFrame(adaptive_results).mean(axis=1)
        
        return {
            'individual_rsis': adaptive_results,
            'combined_rsi': combined_rsi,
            'signal': self._generate_rsi_signal(combined_rsi)
        }
    
    def multi_timeframe_macd(self, prices_dict: Dict[str, pd.Series]) -> Dict:
        """MACD چندزمانه"""
        macd_results = {}
        
        for timeframe, prices in prices_dict.items():
            macd_data = self.classic.calculate_macd(prices)
            macd_results[timeframe] = {
                'macd': macd_data['macd'],
                'signal': macd_data['signal'],
                'histogram': macd_data['histogram'],
                'cross_signal': self._detect_macd_cross(macd_data['macd'], macd_data['signal'])
            }
        
        # اجماع سیگنال‌های چندزمانه
        consensus = self._calculate_macd_consensus(macd_results)
        
        return {
            'timeframe_signals': macd_results,
            'consensus_signal': consensus
        }
    
    def volume_weighted_moving_average(self, prices: pd.Series, volume: pd.Series, 
                                     period: int = 20) -> pd.Series:
        """VWMA - میانگین متحرک وزنی با حجم"""
        vwma = (prices * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwma
    
    def momentum_oscillator(self, prices: pd.Series, short_period: int = 10, 
                          long_period: int = 30) -> pd.Series:
        """اسیلاتور مومنتوم ترکیبی"""
        short_ma = prices.rolling(window=short_period).mean()
        long_ma = prices.rolling(window=long_period).mean()
        
        momentum = (short_ma - long_ma) / long_ma * 100
        return momentum
    
    def _generate_rsi_signal(self, rsi: pd.Series) -> pd.Series:
        """تولید سیگنال از RSI"""
        signals = pd.Series(index=rsi.index, data='HOLD')
        signals[rsi < 30] = 'OVERSOLD'
        signals[rsi > 70] = 'OVERBOUGHT'
        return signals
    
    def _detect_macd_cross(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """تشخیص تقاطع MACD"""
        cross_signals = pd.Series(index=macd.index, data='NO_CROSS')
        
        # تقاطع صعودی
        bullish_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        cross_signals[bullish_cross] = 'BULLISH_CROSS'
        
        # تقاطع نزولی
        bearish_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        cross_signals[bearish_cross] = 'BEARISH_CROSS'
        
        return cross_signals
    
    def _calculate_macd_consensus(self, macd_results: Dict) -> str:
        """محاسبه اجماع سیگنال‌های MACD"""
        bullish_count = 0
        bearish_count = 0
        
        for tf_data in macd_results.values():
            latest_signal = tf_data['cross_signal'].iloc[-1] if not tf_data['cross_signal'].empty else 'NO_CROSS'
            if latest_signal == 'BULLISH_CROSS':
                bullish_count += 1
            elif latest_signal == 'BEARISH_CROSS':
                bearish_count += 1
        
        if bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

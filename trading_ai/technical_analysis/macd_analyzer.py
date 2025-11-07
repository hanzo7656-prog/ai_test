# تحلیل‌گر MACD برای تحلیل تکنیکال
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MACDAnalyzer:
    """تحلیل‌گر MACD (Moving Average Convergence Divergence)"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        logger.info(f"✅ MACD Analyzer initialized: {fast_period}/{slow_period}/{signal_period}")
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """محاسبه میانگین متحرک نمایی (EMA)"""
        try:
            if len(prices) < period:
                return []
            
            ema = [prices[0]]  # اولین مقدار
            
            multiplier = 2 / (period + 1)
            
            for i in range(1, len(prices)):
                ema_value = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
                ema.append(ema_value)
            
            return ema
            
        except Exception as e:
            logger.error(f"خطا در محاسبه EMA: {e}")
            return []
    
    def calculate_macd(self, prices: List[float]) -> Dict[str, List[float]]:
        """محاسبه MACD"""
        try:
            if len(prices) < self.slow_period:
                return {'macd_line': [], 'signal_line': [], 'histogram': []}
            
            # محاسبه EMAهای سریع و کند
            ema_fast = self.calculate_ema(prices, self.fast_period)
            ema_slow = self.calculate_ema(prices, self.slow_period)
            
            if len(ema_fast) < self.slow_period or len(ema_slow) < self.slow_period:
                return {'macd_line': [], 'signal_line': [], 'histogram': []}
            
            # خط MACD
            macd_line = []
            for i in range(len(ema_slow)):
                if i < len(ema_fast):
                    macd_line.append(ema_fast[i] - ema_slow[i])
                else:
                    macd_line.append(0.0)
            
            # خط سیگنال
            signal_line = self.calculate_ema(macd_line, self.signal_period)
            
            # هیستوگرام
            histogram = []
            for i in range(len(signal_line)):
                if i < len(macd_line):
                    histogram.append(macd_line[i] - signal_line[i])
                else:
                    histogram.append(0.0)
            
            return {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"خطا در محاسبه MACD: {e}")
            return {'macd_line': [], 'signal_line': [], 'histogram': []}
    
    def analyze(self, prices: List[float], current_price: float) -> Dict[str, Any]:
        """تحلیل کامل MACD"""
        try:
            macd_data = self.calculate_macd(prices)
            
            if not macd_data['macd_line'] or not macd_data['signal_line']:
                return self._get_default_analysis()
            
            # مقادیر فعلی
            current_macd = macd_data['macd_line'][-1]
            current_signal = macd_data['signal_line'][-1]
            current_histogram = macd_data['histogram'][-1] if macd_data['histogram'] else 0
            
            # مقادیر قبلی برای تشخیص تغییر روند
            prev_macd = macd_data['macd_line'][-2] if len(macd_data['macd_line']) > 1 else current_macd
            prev_signal = macd_data['signal_line'][-2] if len(macd_data['signal_line']) > 1 else current_signal
            
            # تحلیل سیگنال
            signal, confidence = self._generate_signal(
                current_macd, current_signal, current_histogram,
                prev_macd, prev_signal
            )
            
            return {
                'macd_line': float(current_macd),
                'signal_line': float(current_signal),
                'histogram': float(current_histogram),
                'signal': signal,
                'confidence': confidence,
                'crossing': self._check_crossing(current_macd, current_signal, prev_macd, prev_signal),
                'momentum': self._assess_momentum(current_histogram),
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'signal_period': self.signal_period
                }
            }
            
        except Exception as e:
            logger.error(f"خطا در تحلیل MACD: {e}")
            return self._get_default_analysis()
    
    def _generate_signal(self, macd: float, signal: float, histogram: float, 
                        prev_macd: float, prev_signal: float) -> tuple:
        """تولید سیگنال بر اساس MACD"""
        try:
            # بررسی تقاطع
            if prev_macd <= prev_signal and macd > signal:
                # تقاطع صعودی
                return "BULLISH_CROSS", 0.7
            elif prev_macd >= prev_signal and macd < signal:
                # تقاطع نزولی
                return "BEARISH_CROSS", 0.7
            
            # بررسی موقعیت نسبت به خط صفر
            if macd > 0 and signal > 0:
                if histogram > 0 and histogram > prev_macd - prev_signal:
                    return "BULLISH", 0.6
                else:
                    return "BULLISH_WEAK", 0.4
            elif macd < 0 and signal < 0:
                if histogram < 0 and histogram < prev_macd - prev_signal:
                    return "BEARISH", 0.6
                else:
                    return "BEARISH_WEAK", 0.4
            else:
                return "NEUTRAL", 0.3
                
        except:
            return "NEUTRAL", 0.3
    
    def _check_crossing(self, macd: float, signal: float, prev_macd: float, prev_signal: float) -> str:
        """بررسی تقاطع خطوط"""
        try:
            if prev_macd <= prev_signal and macd > signal:
                return "BULLISH_CROSS"
            elif prev_macd >= prev_signal and macd < signal:
                return "BEARISH_CROSS"
            else:
                return "NO_CROSS"
        except:
            return "UNKNOWN"
    
    def _assess_momentum(self, histogram: float) -> str:
        """ارزیابی مومنتوم"""
        try:
            if histogram > 0:
                return "BULLISH_MOMENTUM"
            elif histogram < 0:
                return "BEARISH_MOMENTUM"
            else:
                return "NEUTRAL_MOMENTUM"
        except:
            return "UNKNOWN"
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """تحلیل پیش‌فرض در صورت خطا"""
        return {
            'macd_line': 0.0,
            'signal_line': 0.0,
            'histogram': 0.0,
            'signal': 'NEUTRAL',
            'confidence': 0.3,
            'crossing': 'NO_CROSS',
            'momentum': 'NEUTRAL_MOMENTUM',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period
            },
            'error': True
        }

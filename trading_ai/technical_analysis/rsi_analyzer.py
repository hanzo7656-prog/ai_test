# تحلیل‌گر RSI برای تحلیل تکنیکال
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class RSIAnalyzer:
    """تحلیل‌گر شاخص قدرت نسبی (RSI)"""
    
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        logger.info(f"✅ RSI Analyzer initialized: period={period}")
    
    def calculate_rsi(self, prices: List[float]) -> float:
        """محاسبه RSI از لیست قیمت‌ها"""
        try:
            if len(prices) < self.period + 1:
                return 50.0  # مقدار پیش‌فرض
            
            # محاسبه تغییرات
            deltas = np.diff(prices)
            
            # جدا کردن gains و losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # محاسبه میانگین gains و losses
            avg_gains = np.mean(gains[-self.period:])
            avg_losses = np.mean(losses[-self.period:])
            
            # جلوگیری از تقسیم بر صفر
            if avg_losses == 0:
                return 100.0
            
            # محاسبه RS و RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.error(f"خطا در محاسبه RSI: {e}")
            return 50.0
    
    def analyze(self, prices: List[float], current_price: float) -> Dict[str, Any]:
        """تحلیل کامل RSI"""
        try:
            rsi = self.calculate_rsi(prices)
            
            # تحلیل وضعیت
            if rsi >= self.overbought:
                signal = "OVERBOUGHT"
                strength = "STRONG"
            elif rsi <= self.oversold:
                signal = "OVERSOLD" 
                strength = "STRONG"
            elif rsi > 60:
                signal = "BULLISH"
                strength = "WEAK"
            elif rsi < 40:
                signal = "BEARISH"
                strength = "WEAK"
            else:
                signal = "NEUTRAL"
                strength = "NEUTRAL"
            
            # اعتماد بر اساس فاصله از مرزها
            if signal in ["OVERBOUGHT", "OVERSOLD"]:
                confidence = 0.8
            elif signal in ["BULLISH", "BEARISH"]:
                confidence = 0.6
            else:
                confidence = 0.5
            
            return {
                'rsi': rsi,
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'period': self.period,
                'overbought_level': self.overbought,
                'oversold_level': self.oversold,
                'timestamp': datetime.now().isoformat(),
                'interpretation': self._get_interpretation(signal, rsi)
            }
            
        except Exception as e:
            logger.error(f"خطا در تحلیل RSI: {e}")
            return self._get_default_analysis()
    
    def _get_interpretation(self, signal: str, rsi: float) -> str:
        """تفسیر وضعیت RSI"""
        interpretations = {
            "OVERBOUGHT": "اشباع خرید - احتمال اصلاح قیمت",
            "OVERSOLD": "اشباع فروش - احتمال رشد قیمت", 
            "BULLISH": "روند صعودی - قدرت خریداران",
            "BEARISH": "روند نزولی - قدرت فروشندگان",
            "NEUTRAL": "تعادل بین خریداران و فروشندگان"
        }
        return interpretations.get(signal, "وضعیت نامشخص")
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """تحلیل پیش‌فرض در صورت خطا"""
        return {
            'rsi': 50.0,
            'signal': 'NEUTRAL',
            'strength': 'NEUTRAL', 
            'confidence': 0.3,
            'period': self.period,
            'overbought_level': self.overbought,
            'oversold_level': self.oversold,
            'timestamp': datetime.now().isoformat(),
            'interpretation': 'داده ناکافی برای تحلیل',
            'error': True
        }

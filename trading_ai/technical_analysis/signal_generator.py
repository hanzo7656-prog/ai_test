# تولیدکننده سیگنال‌های معاملاتی
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from ..core.utils import ai_utils

logger = logging.getLogger(__name__)

class SignalGenerator:
    """تولیدکننده سیگنال‌های معاملاتی از تحلیل‌های مختلف"""
    
    def __init__(self, config=None):
        self.config = config or {}
        logger.info("✅ Signal Generator initialized")
    
    def generate_signal(self, analyses: List[Dict[str, Any]], 
                       market_data: Dict[str, Any]) -> Dict[str, Any]:
        """تولید سیگنال نهایی از تحلیل‌های مختلف"""
        try:
            if not analyses:
                return self._get_default_signal()
            
            # فیلتر تحلیل‌های معتبر
            valid_analyses = [a for a in analyses if not a.get('error', False)]
            
            if not valid_analyses:
                return self._get_default_signal()
            
            # ادغام تحلیل‌ها
            final_signal = ai_utils.merge_analyses(valid_analyses)
            
            # بهبود اعتماد بر اساس داده‌های بازار
            final_signal['confidence'] = self._enhance_confidence(
                final_signal['confidence'], 
                market_data
            )
            
            # اضافه کردن متادیتا
            final_signal.update({
                'symbol': market_data.get('symbol', 'UNKNOWN'),
                'analysis_count': len(valid_analyses),
                'timestamp': datetime.now().isoformat(),
                'signal_generator': 'advanced'
            })
            
            logger.info(f"🎯 سیگنال تولید شد: {final_signal['signal']} (اعتماد: {final_signal['confidence']:.2f})")
            return final_signal
            
        except Exception as e:
            logger.error(f"خطا در تولید سیگنال: {e}")
            return self._get_default_signal()
    
    def _enhance_confidence(self, base_confidence: float, market_data: Dict[str, Any]) -> float:
        """بهبود اعتماد بر اساس داده‌های بازار"""
        try:
            enhanced_confidence = base_confidence
            
            # بهبود بر اساس حجم معاملات
            volume = market_data.get('volume', 0)
            if volume > 1000000000:  # حجم بالا
                enhanced_confidence = min(enhanced_confidence + 0.1, 0.95)
            elif volume < 10000000:  # حجم پایین
                enhanced_confidence = max(enhanced_confidence - 0.1, 0.1)
            
            # بهبود بر اساس رتبه بازار
            rank = market_data.get('rank', 100)
            if rank <= 10:  # ارزهای برتر
                enhanced_confidence = min(enhanced_confidence + 0.05, 0.95)
            elif rank > 50:  # ارزهای کوچک
                enhanced_confidence = max(enhanced_confidence - 0.05, 0.1)
            
            # بهبود بر اساس نوسان
            price_change = abs(market_data.get('priceChange1d', 0))
            if price_change > 20:  # نوسان شدید
                enhanced_confidence = max(enhanced_confidence - 0.15, 0.1)
            elif price_change < 5:  # نوسان کم
                enhanced_confidence = min(enhanced_confidence + 0.05, 0.95)
            
            return round(enhanced_confidence, 2)
            
        except Exception as e:
            logger.error(f"خطا در بهبود اعتماد: {e}")
            return base_confidence
    
    def _get_default_signal(self) -> Dict[str, Any]:
        """سیگنال پیش‌فرض"""
        return {
            'signal': 'HOLD',
            'confidence': 0.3,
            'sources': ['default'],
            'timestamp': datetime.now().isoformat(),
            'analysis_count': 0,
            'signal_generator': 'basic',
            'note': 'سیگنال پیش‌فرض به دلیل عدم وجود تحلیل کافی'
        }
    
    def generate_stop_loss_take_profit(self, signal: Dict[str, Any], 
                                      current_price: float) -> Dict[str, float]:
        """تولید سطوح stop-loss و take-profit"""
        try:
            signal_type = signal.get('signal', 'HOLD')
            confidence = signal.get('confidence', 0.5)
            
            # محاسبه سطوح بر اساس نوع سیگنال و اعتماد
            if signal_type == 'STRONG_BUY':
                stop_loss = current_price * 0.92  # 8% کاهش
                take_profit = current_price * 1.15  # 15% افزایش
            elif signal_type == 'BUY':
                stop_loss = current_price * 0.94  # 6% کاهش
                take_profit = current_price * 1.10  # 10% افزایش
            elif signal_type == 'STRONG_SELL':
                stop_loss = current_price * 1.08  # 8% افزایش
                take_profit = current_price * 0.85  # 15% کاهش
            elif signal_type == 'SELL':
                stop_loss = current_price * 1.06  # 6% افزایش
                take_profit = current_price * 0.90  # 10% کاهش
            else:  # HOLD
                stop_loss = current_price * 0.97  # 3% کاهش
                take_profit = current_price * 1.03  # 3% افزایش
            
            # تنظیم بر اساس اعتماد
            confidence_factor = confidence * 0.5 + 0.5  # 0.5 تا 1.0
            stop_loss = current_price - (abs(current_price - stop_loss) * confidence_factor)
            take_profit = current_price + (abs(take_profit - current_price) * confidence_factor)
            
            return {
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'current_price': round(current_price, 4),
                'risk_reward_ratio': round(
                    abs(take_profit - current_price) / abs(current_price - stop_loss), 2
                )
            }
            
        except Exception as e:
            logger.error(f"خطا در تولید سطوح SL/TP: {e}")
            return {
                'stop_loss': round(current_price * 0.95, 4),
                'take_profit': round(current_price * 1.05, 4),
                'current_price': round(current_price, 4),
                'risk_reward_ratio': 1.0
            }
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """اعتبارسنجی سیگنال"""
        try:
            required_fields = ['signal', 'confidence', 'timestamp']
            
            for field in required_fields:
                if field not in signal:
                    return False
            
            # اعتبارسنجی مقادیر
            valid_signals = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
            if signal['signal'] not in valid_signals:
                return False
            
            if not 0 <= signal['confidence'] <= 1:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"خطا در اعتبارسنجی سیگنال: {e}")
            return False

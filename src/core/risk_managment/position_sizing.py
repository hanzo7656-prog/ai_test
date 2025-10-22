# 📁 src/core/risk_management/position_sizing.py

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from ...utils.memory_monitor import MemoryMonitor

@dataclass
class PositionSizingResult:
    symbol: str
    position_size: float
    risk_amount: float
    stop_loss: float
    take_profit: List[float]
    leverage: float
    max_position_value: float

class DynamicPositionSizing:
    """محاسبه سایز پویای پوزیشن با مدیریت ریسک پیشرفته"""
    
    def __init__(self, total_capital: float = 10000.0, max_risk_per_trade: float = 0.02):
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade  # 2% ریسک هر معامله
        self.memory_monitor = MemoryMonitor()
        
        # پارامترهای مدیریت ریسک
        self.volatility_adjustment = True
        self.correlation_penalty = True
        self.portfolio_heat_limit = 0.1  # حداکثر 10% سرمایه در ریسک
        
    def calculate_position_size(self, signal, market_data: Dict, 
                              current_portfolio: Dict = None) -> PositionSizingResult:
        """محاسبه سایز پوزیشن با درنظرگیری تمام پارامترها"""
        with self.memory_monitor.track("position_sizing"):
            # ریسک پایه
            base_risk = self.total_capital * self.max_risk_per_trade
            
            # تنظیم بر اساس نوسان
            volatility_factor = self._calculate_volatility_factor(signal.symbol, market_data)
            adjusted_risk = base_risk * volatility_factor
            
            # تنظیم بر اساس همبستگی
            correlation_factor = self._calculate_correlation_factor(signal.symbol, current_portfolio)
            final_risk = adjusted_risk * correlation_factor
            
            # محاسبه سایز پوزیشن
            price = signal.price
            stop_loss = signal.stop_loss
            risk_per_unit = abs(price - stop_loss)
            
            if risk_per_unit == 0:
                risk_per_unit = price * 0.02  # فرض 2% استاپ
            
            position_size = final_risk / risk_per_unit
            
            # محاسبه اهرم مجاز
            leverage = self._calculate_leverage(signal.symbol, volatility_factor)
            
            # محاسبه ارزش پوزیشن
            position_value = position_size * price
            
            return PositionSizingResult(
                symbol=signal.symbol,
                position_size=position_size,
                risk_amount=final_risk,
                stop_loss=stop_loss,
                take_profit=signal.targets,
                leverage=leverage,
                max_position_value=position_value
            )
    
    def _calculate_volatility_factor(self, symbol: str, market_data: Dict) -> float:
        """فاکتور تعدیل نوسان"""
        if not self.volatility_adjustment:
            return 1.0
        
        try:
            # محاسبه ATR یا نوسان اخیر
            data = market_data.get(symbol, {})
            if 'atr' in data and len(data['atr']) > 0:
                atr = data['atr'].iloc[-1]
                current_price = data['close'].iloc[-1]
                atr_percentage = (atr / current_price) * 100
                
                if atr_percentage > 5:
                    return 0.5  # کاهش 50% سایز برای نوسان بالا
                elif atr_percentage < 1:
                    return 1.2  # افزایش 20% برای نوسان پایین
            
            return 1.0
        except:
            return 1.0
    
    def _calculate_correlation_factor(self, symbol: str, current_portfolio: Optional[Dict]) -> float:
        """فاکتور تعدیل همبستگی"""
        if not self.correlation_penalty or not current_portfolio:
            return 1.0
        
        try:
            # شبیه‌سازی همبستگی - در نسخه کامل از داده‌های واقعی استفاده می‌شود
            correlated_positions = 0
            for pos_symbol, position in current_portfolio.items():
                # فرض همبستگی بالا برای جفت‌ارزهای مشابه
                if self._are_correlated(symbol, pos_symbol):
                    correlated_positions += position['risk_amount']
            
            total_risk = sum(pos['risk_amount'] for pos in current_portfolio.values())
            
            if total_risk > 0:
                correlation_ratio = correlated_positions / total_risk
                if correlation_ratio > 0.3:  # اگر بیش از 30% همبسته باشد
                    return 0.7  # کاهش 30% سایز
            
            return 1.0
        except:
            return 1.0
    
    def _are_correlated(self, symbol1: str, symbol2: str) -> bool:
        """بررسی همبستگی دو نماد"""
        # منطق ساده - در نسخه کامل از ماتریس همبستگی استفاده می‌شود
        base1 = symbol1.split('/')[0] if '/' in symbol1 else symbol1[:3]
        base2 = symbol2.split('/')[0] if '/' in symbol2 else symbol2[:3]
        
        return base1 == base2
    
    def _calculate_leverage(self, symbol: str, volatility_factor: float) -> float:
        """محاسبه اهرم مجاز"""
        # اهرم پایه بر اساس نوسان
        base_leverage = 3.0  # اهرم پیش‌فرض
        
        # کاهش اهرم برای نوسان بالا
        if volatility_factor < 0.7:
            leverage = base_leverage * 0.5
        elif volatility_factor > 1.2:
            leverage = base_leverage * 1.2
        else:
            leverage = base_leverage
        
        return min(5.0, max(1.0, leverage))  # محدودیت اهرم 1x تا 5x
    
    def validate_portfolio_risk(self, current_portfolio: Dict, new_position: PositionSizingResult) -> bool:
        """اعتبارسنجی ریسک کلی پورتفو"""
        total_risk = sum(pos.get('risk_amount', 0) for pos in current_portfolio.values())
        total_risk += new_position.risk_amount
        
        portfolio_risk_ratio = total_risk / self.total_capital
        
        return portfolio_risk_ratio <= self.portfolio_heat_limit
    
    def get_risk_metrics(self) -> Dict:
        """دریافت معیارهای ریسک"""
        return {
            "total_capital": self.total_capital,
            "max_risk_per_trade": self.max_risk_per_trade,
            "portfolio_heat_limit": self.portfolio_heat_limit,
            "volatility_adjustment": self.volatility_adjustment,
            "correlation_penalty": self.correlation_penalty,
            "memory_usage": self.memory_monitor.get_usage_stats()
        }

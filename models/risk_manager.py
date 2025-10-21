# risk_manager.py
from typing import Dict, List
import numpy as np

class RiskManager:
    """مدیریت ریسک و محاسبه سایز پوزیشن"""
    
    def __init__(self, max_position_size=0.3, stop_loss=0.15, risk_per_trade=0.02):
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_loss_price: float, risk_level: str = 'medium') -> Dict:
        """محاسبه سایز پوزیشن بر اساس ریسک"""
        
        # محاسبه فاصله تا استاپ لاس
        price_distance = abs(entry_price - stop_loss_price)
        risk_percentage = price_distance / entry_price
        
        if risk_percentage == 0:
            return {'size': 0, 'risk': 'نامشخص'}
        
        # محاسبه سایز بر اساس ریسک حساب
        risk_amount = account_balance * self.risk_per_trade
        position_size = risk_amount / risk_percentage
        
        # اعمال محدودیت‌های سطح ریسک
        if risk_level == 'low':
            position_size *= 0.5
        elif risk_level == 'high':
            position_size *= 1.5
        
        # محدودیت سایز ماکسیمم
        max_size = account_balance * self.max_position_size
        position_size = min(position_size, max_size)
        
        return {
            'size': position_size,
            'size_percentage': (position_size / account_balance) * 100,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage * 100
        }
    
    def assess_market_risk(self, volatility: float, correlation: float, 
                          liquidity: float, sentiment: str) -> str:
        """ارزیابی ریسک کلی بازار"""
        risk_score = 0
        
        # نوسان
        if volatility > 0.1:  # 10%+
            risk_score += 2
        elif volatility > 0.05:  # 5%+
            risk_score += 1
        
        # همبستگی
        if correlation > 0.8:
            risk_score += 1
        
        # نقدینگی
        if liquidity < 1000000:  # کمتر از ۱M
            risk_score += 2
        elif liquidity < 5000000:  # کمتر از ۵M
            risk_score += 1
        
        # احساسات
        if sentiment == 'EXTREME_FEAR':
            risk_score -= 1
        elif sentiment == 'EXTREME_GREED':
            risk_score += 1
        
        if risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def calculate_portfolio_risk(self, portfolio: Dict) -> float:
        """محاسبه ریسک کلی پورتفولیو"""
        total_risk = 0
        
        for coin, data in portfolio.items():
            position_size = data.get('size_percentage', 0) / 100
            coin_risk = data.get('risk_level', 1.0)
            total_risk += position_size * coin_risk
        
        return min(total_risk, 1.0)  # نرمالایز بین ۰ و ۱

# 📁 tests/unit/core/test_risk_management.py

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.core.risk_management.position_sizing import DynamicPositionSizing, PositionSizingResult
from src.core.technical_analysis.signal_engine import TradingSignal, SignalType
from datetime import datetime

class TestRiskManagement:
    """تست سیستم مدیریت ریسک"""
    
    def setup_method(self):
        self.risk_manager = DynamicPositionSizing(
            total_capital=10000,
            max_risk_per_trade=0.02
        )
        
        self.sample_signal = TradingSignal(
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=50000,
            timestamp=datetime.now(),
            reasons=["RSI oversold", "Trend support"],
            targets=[52000, 54000],
            stop_loss=48000,
            time_horizon="MEDIUM_TERM",
            risk_reward_ratio=2.5
        )
    
    def test_position_sizing_calculation(self):
        """تست محاسبه سایز پوزیشن"""
        market_data = {"BTC/USDT": {"close": [50000, 50100, 50200]}}
        
        position = self.risk_manager.calculate_position_size(
            self.sample_signal, market_data
        )
        
        assert isinstance(position, PositionSizingResult)
        assert position.symbol == "BTC/USDT"
        assert position.position_size > 0
        assert position.risk_amount > 0
        assert position.leverage >= 1.0
    
    def test_volatility_adjustment(self):
        """تست تعدیل نوسان"""
        # داده با نوسان بالا
        high_vol_data = {"BTC/USDT": {"atr": [1000, 1100, 1200]}}
        
        position = self.risk_manager.calculate_position_size(
            self.sample_signal, high_vol_data
        )
        
        # باید سایز پوزیشن کاهش یابد
        assert position.position_size < 0.1  # کمتر از 10% سرمایه
    
    def test_portfolio_risk_validation(self):
        """تست اعتبارسنجی ریسک پورتفو"""
        current_portfolio = {
            "ETH/USDT": {"risk_amount": 500},
            "ADA/USDT": {"risk_amount": 300}
        }
        
        new_position = PositionSizingResult(
            symbol="BTC/USDT",
            position_size=0.05,
            risk_amount=200,
            stop_loss=48000,
            take_profit=[52000],
            leverage=2.0,
            max_position_value=2500
        )
        
        is_valid = self.risk_manager.validate_portfolio_risk(
            current_portfolio, new_position
        )
        
        assert isinstance(is_valid, bool)
    
    def test_correlation_penalty(self):
        """تست جریمه همبستگی"""
        # پورتفوی با همبستگی بالا
        correlated_portfolio = {
            "BTC/USDT": {"risk_amount": 800},
            "WBTC/USDT": {"risk_amount": 400}  # همبسته با BTC
        }
        
        market_data = {"BTC/USDT": {"close": [50000]}}
        position = self.risk_manager.calculate_position_size(
            self.sample_signal, market_data, correlated_portfolio
        )
        
        # سایز باید به دلیل همبستگی کاهش یابد
        assert position.position_size < 0.08

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

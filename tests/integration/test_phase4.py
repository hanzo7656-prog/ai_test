# 📁 test_phase4.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtesting.walk_forward import WalkForwardAnalyzer, BacktestResult
from src.backtesting.monte_carlo import MonteCarloSimulator
from src.visualization.dashboard_builder import TradingDashboard

class SampleStrategy:
    """استراتژی نمونه برای تست"""
    
    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signal(self, data):
        """تولید سیگنال بر اساس RSI"""
        if len(data) < self.rsi_period + 1:
            return {'action': 'HOLD', 'confidence': 0}
        
        # محاسبه RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < self.rsi_oversold:
            return {'action': 'BUY', 'confidence': (self.rsi_oversold - current_rsi) / self.rsi_oversold}
        elif current_rsi > self.rsi_overbought:
            return {'action': 'SELL', 'confidence': (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)}
        else:
            return {'action': 'HOLD', 'confidence': 0}

def generate_sample_data():
    """تولید داده نمونه برای بک‌تست"""
    dates = pd.date_range('2023-01-01', '2024-01-15', freq='1h')
    
    # تولید داده قیمت با روند و نوسان
    trend = np.linspace(0, 0.2, len(dates))  # روند صعودی ملایم
    noise = np.random.randn(len(dates)) * 0.01
    prices = 50000 * np.exp(trend + noise.cumsum())
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(len(dates)) * 10,
        'high': prices + np.abs(np.random.randn(len(dates)) * 20),
        'low': prices - np.abs(np.random.randn(len(dates)) * 20),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    return data

def test_backtesting_system():
    """تست سیستم بک‌تستینگ"""
    print("📈 Testing Backtesting System...")
    
    # تولید داده
    data = generate_sample_data()
    
    # تست Walk-Forward Analysis
    wf_analyzer = WalkForwardAnalyzer(initial_capital=10000)
    
    optimization_params = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75]
    }
    
    wf_results = wf_analyzer.run_walk_forward_analysis(
        strategy=SampleStrategy,
        data=data,
        window_size=1000,  # 1000 ساعت آموزش
        step_size=200,     # 200 ساعت تست
        optimization_params=optimization_params
    )
    
    print(f"✅ Walk-Forward Analysis: {len(wf_results['window_results'])} windows")
    print(f"📊 Average Win Rate: {wf_results['summary']['average_win_rate']:.2f}%")
    
    # تست Monte Carlo Simulation
    mc_simulator = MonteCarloSimulator(num_simulations=5000)
    
    # استفاده از نتایج اولین پنجره برای شبیه‌سازی
    if wf_results['window_results']:
        first_result = wf_results['window_results'][0]['result']
        mc_result = mc_simulator.run_simulation(first_result)
        
        print(f"🎲 Monte Carlo Simulation:")
        print(f"   Expected Return: {mc_result.expected_return:.2f}%")
        print(f"   Value at Risk: ${mc_result.value_at_risk:.2f}")
        print(f"   Probability of Ruin: {mc_result.probability_of_ruin:.2%}")
    
    return {
        'walk_forward_results': wf_results,
        'monte_carlo_result': mc_result if 'mc_result' in locals() else None
    }

def test_visualization():
    """تست سیستم ویژوالایزیشن"""
    print("\n📊 Testing Visualization System...")
    
    # تولید نتایج نمونه
    data = generate_sample_data()
    strategy = SampleStrategy()
    wf_analyzer = WalkForwardAnalyzer()
    
    # بک‌تست ساده
    result = wf_analyzer._backtest_strategy(strategy, data, 10000)
    
    # شبیه‌سازی مونت کارلو
    mc_simulator = MonteCarloSimulator()
    mc_result = mc_simulator.run_simulation(result)
    
    # ایجاد داشبورد
    dashboard_builder = TradingDashboard()
    
    # داشبورد عملکرد
    performance_fig = dashboard_builder.create_performance_dashboard(result, mc_result)
    performance_fig.write_html("performance_dashboard.html")
    
    # داشبورد زنده (شبیه‌سازی)
    portfolio = {
        'BTC/USDT': {'allocation': 60, 'pnl': 1500},
        'ETH/USDT': {'allocation': 30, 'pnl': 800},
        'ADA/USDT': {'allocation': 10, 'pnl': -200}
    }
    
    market_data = {
        'BTC/USDT': data,
        'ETH/USDT': data * 0.07,  # شبیه‌سازی داده اتریوم
        'ADA/USDT': data * 0.02   # شبیه‌سازی داده کاردانو
    }
    
    from src.core.technical_analysis.signal_engine import TradingSignal, SignalType
    signals = [
        TradingSignal(
            symbol='BTC/USDT',
            signal_type=SignalType.BUY,
            confidence=0.75,
            price=52000,
            timestamp=datetime.now(),
            reasons=['RSI Oversold', 'Trend Support'],
            targets=[54000, 56000],
            stop_loss=51000,
            time_horizon='MEDIUM_TERM',
            risk_reward_ratio=2.5
        )
    ]
    
    live_fig = dashboard_builder.create_live_trading_dashboard(portfolio, market_data, signals)
    live_fig.write_html("live_dashboard.html")
    
    print("✅ Dashboards created: performance_dashboard.html, live_dashboard.html")
    
    return {
        'performance_dashboard': performance_fig,
        'live_dashboard': live_fig
    }

def integration_test_phase4():
    """تست یکپارچه فاز ۴"""
    print("🚀 Starting Phase 4 Integration Test...")
    
    # تست سیستم بک‌تستینگ
    backtesting_results = test_backtesting_system()
    
    # تست سیستم ویژوالایزیشن
    visualization_results = test_visualization()
    
    print("\n🎉 Phase 4 Integration Test Completed Successfully!")
    print("📋 Summary:")
    print("   - Walk-Forward Analysis: ✅ Parameter optimization and validation")
    print("   - Monte Carlo Simulation: ✅ Risk analysis and probability modeling") 
    print("   - Performance Dashboard: ✅ Interactive visualization")
    print("   - Live Trading Dashboard: ✅ Real-time monitoring")
    print("   - Strategy Testing: ✅ Robust backtesting framework")
    
    return {
        'backtesting': backtesting_results,
        'visualization': visualization_results
    }

if __name__ == "__main__":
    integration_test_phase4()

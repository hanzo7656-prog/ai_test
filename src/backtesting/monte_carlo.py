# 📁 src/backtesting/monte_carlo.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from .walk_forward import BacktestResult

@dataclass
class MonteCarloResult:
    confidence_level: float
    expected_return: float
    value_at_risk: float
    expected_shortfall: float
    probability_of_ruin: float
    best_case: float
    worst_case: float
    distribution: np.ndarray

class MonteCarloSimulator:
    """شبیه‌سازی مونت کارلو برای تحلیل ریسک استراتژی"""
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
    
    def run_simulation(self, backtest_result: BacktestResult, initial_capital: float = 10000,
                      time_horizon: int = 252, confidence_level: float = 0.95) -> MonteCarloResult:
        """اجرای شبیه‌سازی مونت کارلو"""
        print("🎲 Running Monte Carlo Simulation...")
        
        # استخراج بازدهی‌های تاریخی
        historical_returns = self._extract_returns_from_trades(backtest_result.trades, initial_capital)
        
        if len(historical_returns) < 10:
            print("❌ Insufficient historical data for simulation")
            return self._empty_result()
        
        # تولید توزیع بازدهی آینده
        simulated_returns = self._generate_simulated_returns(historical_returns, time_horizon)
        
        # محاسبه معیارهای ریسک
        final_values = initial_capital * (1 + simulated_returns)
        
        return MonteCarloResult(
            confidence_level=confidence_level,
            expected_return=np.mean(simulated_returns) * 100,
            value_at_risk=self._calculate_var(final_values, confidence_level),
            expected_shortfall=self._calculate_expected_shortfall(final_values, confidence_level),
            probability_of_ruin=np.mean(final_values < initial_capital * 0.5),  # از دست دادن 50% سرمایه
            best_case=np.percentile(final_values, 95),
            worst_case=np.percentile(final_values, 5),
            distribution=final_values
        )
    
    def _extract_returns_from_trades(self, trades: List, initial_capital: float) -> np.ndarray:
        """استخراج بازدهی‌های تاریخی از معاملات"""
        if not trades:
            return np.array([])
        
        returns = []
        capital = initial_capital
        
        for trade in trades:
            if trade.pnl is not None:
                trade_return = trade.pnl / capital
                returns.append(trade_return)
                capital += trade.pnl  # سرمایه شناور
        
        return np.array(returns)
    
    def _generate_simulated_returns(self, historical_returns: np.ndarray, time_horizon: int) -> np.ndarray:
        """تولید بازدهی‌های شبیه‌سازی شده"""
        # پارامترهای توزیع
        mean_return = np.mean(historical_returns)
        std_return = np.std(historical_returns)
        
        # شبیه‌سازی بازدهی‌ها (فرض توزیع نرمال)
        simulated_daily_returns = np.random.normal(
            mean_return, std_return, 
            (self.num_simulations, time_horizon)
        )
        
        # محاسبه بازدهی کل برای هر شبیه‌سازی
        total_returns = np.prod(1 + simulated_daily_returns, axis=1) - 1
        
        return total_returns
    
    def _calculate_var(self, final_values: np.ndarray, confidence_level: float) -> float:
        """محاسبه Value at Risk"""
        var = np.percentile(final_values, (1 - confidence_level) * 100)
        return var
    
    def _calculate_expected_shortfall(self, final_values: np.ndarray, confidence_level: float) -> float:
        """محاسبه Expected Shortfall (Conditional VaR)"""
        var = self._calculate_var(final_values, confidence_level)
        losses_below_var = final_values[final_values <= var]
        
        if len(losses_below_var) > 0:
            return np.mean(losses_below_var)
        else:
            return var
    
    def _empty_result(self) -> MonteCarloResult:
        """نتیجه خالی در صورت عدم داده کافی"""
        return MonteCarloResult(
            confidence_level=0.95,
            expected_return=0.0,
            value_at_risk=0.0,
            expected_shortfall=0.0,
            probability_of_ruin=0.0,
            best_case=0.0,
            worst_case=0.0,
            distribution=np.array([])
        )
    
    def analyze_strategy_robustness(self, backtest_results: List[BacktestResult], 
                                  initial_capital: float = 10000) -> Dict:
        """تحلیل استحکام استراتژی با شبیه‌سازی‌های مختلف"""
        print("🔍 Analyzing Strategy Robustness...")
        
        metrics = []
        
        for i, result in enumerate(backtest_results):
            mc_result = self.run_simulation(result, initial_capital)
            
            metrics.append({
                'simulation': i + 1,
                'expected_return': mc_result.expected_return,
                'var': mc_result.value_at_risk,
                'probability_of_ruin': mc_result.probability_of_ruin,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown
            })
        
        # محاسبه معیارهای کلی استحکام
        expected_returns = [m['expected_return'] for m in metrics]
        var_values = [m['var'] for m in metrics]
        ruin_probabilities = [m['probability_of_ruin'] for m in metrics]
        
        return {
            'average_expected_return': np.mean(expected_returns),
            'average_var': np.mean(var_values),
            'average_ruin_probability': np.mean(ruin_probabilities),
            'strategy_stability': self._calculate_stability_score(metrics),
            'risk_adjusted_score': self._calculate_risk_adjusted_score(metrics),
            'simulation_count': len(metrics),
            'detailed_metrics': metrics
        }
    
    def _calculate_stability_score(self, metrics: List[Dict]) -> float:
        """محاسبه نمره پایداری استراتژی"""
        returns = [m['expected_return'] for m in metrics]
        return 1.0 - (np.std(returns) / (np.mean(returns) + 1e-8))
    
    def _calculate_risk_adjusted_score(self, metrics: List[Dict]) -> float:
        """محاسبه نمره تعدیل‌شده بر اساس ریسک"""
        avg_return = np.mean([m['expected_return'] for m in metrics])
        avg_var = np.mean([m['var'] for m in metrics])
        avg_ruin_prob = np.mean([m['probability_of_ruin'] for m in metrics])
        
        if avg_var > 0:
            risk_score = avg_return / (avg_var + avg_ruin_prob * 1000)
        else:
            risk_score = avg_return
        
        return max(0.0, risk_score / 10.0)  # نرمال‌سازی

# 📁 src/backtesting/performance_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMetrics:
    calmar_ratio: float
    sortino_ratio: float
    omega_ratio: float
    var_95: float
    cvar_95: float
    ulcer_index: float
    tail_ratio: float
    common_sense_ratio: float

class PerformanceAnalyzer:
    """آنالایزر پیشرفته عملکرد استراتژی"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate / 252  # نرخ بدون ریسک روزانه
    
    def analyze_performance(self, returns: List[float], equity_curve: List[float]) -> Dict:
        """آنالیز پیشرفته عملکرد"""
        if not returns:
            return {}
        
        returns_series = pd.Series(returns)
        equity_series = pd.Series(equity_curve)
        
        # معیارهای پایه
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0] * 100
        volatility = returns_series.std() * np.sqrt(252) * 100  # نوسان سالانه
        
        # معیارهای پیشرفته
        advanced_metrics = self._calculate_advanced_metrics(returns_series, equity_series)
        
        # تحلیل آماری
        statistical_analysis = self._statistical_analysis(returns_series)
        
        # تحلیل ریسک
        risk_analysis = self._risk_analysis(returns_series)
        
        return {
            'basic_metrics': {
                'total_return': total_return,
                'annual_volatility': volatility,
                'sharpe_ratio': (returns_series.mean() - self.risk_free_rate) / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(equity_series)
            },
            'advanced_metrics': advanced_metrics,
            'statistical_analysis': statistical_analysis,
            'risk_analysis': risk_analysis
        }
    
    def _calculate_advanced_metrics(self, returns: pd.Series, equity: pd.Series) -> AdvancedMetrics:
        """محاسبه معیارهای پیشرفته"""
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown(equity)
        calmar_ratio = returns.mean() * 252 / abs(max_dd) if max_dd != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (returns.mean() - self.risk_free_rate) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Omega Ratio
        threshold = 0
        wins = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega_ratio = wins / losses if losses != 0 else float('inf')
        
        # Value at Risk و Conditional VaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(equity)
        
        # Tail Ratio
        tail_ratio = self._calculate_tail_ratio(returns)
        
        # Common Sense Ratio
        common_sense_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252) * max_dd) if max_dd != 0 else 0
        
        return AdvancedMetrics(
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            omega_ratio=omega_ratio,
            var_95=var_95,
            cvar_95=cvar_ratio,
            ulcer_index=ulcer_index,
            tail_ratio=tail_ratio,
            common_sense_ratio=common_sense_ratio
        )
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """محاسبه حداکثر drawdown"""
        rolling_max = equity.expanding().max()
        drawdowns = (rolling_max - equity) / rolling_max
        return drawdowns.max() * 100
    
    def _calculate_ulcer_index(self, equity: pd.Series) -> float:
        """محاسبه Ulcer Index"""
        rolling_max = equity.expanding().max()
        drawdowns = ((rolling_max - equity) / rolling_max) ** 2
        return np.sqrt(drawdowns.mean()) * 100
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """محاسبه Tail Ratio"""
        var_95 = np.percentile(returns, 5)
        var_5 = np.percentile(returns, 95)
        return abs(var_5 / var_95) if var_95 != 0 else 0
    
    def _statistical_analysis(self, returns: pd.Series) -> Dict:
        """تحلیل آماری بازدهی‌ها"""
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        jarque_bera = stats.jarque_bera(returns)
        normality_pvalue = stats.normaltest(returns).pvalue
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_statistic': jarque_bera[0],
            'jarque_bera_pvalue': jarque_bera[1],
            'normality_pvalue': normality_pvalue,
            'is_normal': normality_pvalue > 0.05,
            'positive_skew': skewness > 0,
            'fat_tails': kurtosis > 0
        }
    
    def _risk_analysis(self, returns: pd.Series) -> Dict:
        """تحلیل ریسک"""
        # نیم‌انحراف معیار
        downside_returns = returns[returns < 0]
        semi_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # بتا (شبیه‌سازی شده)
        beta = 1.2  # در نسخه واقعی نسبت به شاخص بازار محاسبه شود
        
        # نسبت اطلاعات
        tracking_error = returns.std() * 0.8  # شبیه‌سازی شده
        information_ratio = returns.mean() / tracking_error if tracking_error > 0 else 0
        
        return {
            'semi_deviation': semi_deviation,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'downside_risk': semi_deviation * np.sqrt(252) * 100,
            'value_at_risk_95': np.percentile(returns, 5) * 100,
            'expected_shortfall_95': returns[returns <= np.percentile(returns, 5)].mean() * 100
        }

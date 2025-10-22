# 📁 src/visualization/dashboard_builder.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json

class TradingDashboard:
    """سازنده داشبورد تریدینگ تعاملی"""
    
    def __init__(self):
        self.theme = {
            'background': '#0E1117',
            'text': '#FFFFFF',
            'grid': '#1E2130',
            'up': '#00C853',
            'down': '#FF1744',
            'neutral': '#FFC107'
        }
    
    def create_performance_dashboard(self, backtest_result, monte_carlo_result=None) -> go.Figure:
        """ایجاد داشبورد عملکرد استراتژی"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Equity Curve', 'Drawdown',
                'Trade Distribution', 'Monthly Returns',
                'Performance Metrics', 'Risk Analysis'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # منحنی equity
        if hasattr(backtest_result, 'equity_curve') and backtest_result.equity_curve is not None:
            fig.add_trace(
                go.Scatter(
                    x=backtest_result.equity_curve.index,
                    y=backtest_result.equity_curve.values,
                    line=dict(color=self.theme['up'], width=2),
                    name='Equity Curve'
                ),
                row=1, col=1
            )
        
        # Drawdown
        if hasattr(backtest_result, 'drawdown_curve') and backtest_result.drawdown_curve is not None:
            fig.add_trace(
                go.Scatter(
                    x=backtest_result.drawdown_curve.index,
                    y=backtest_result.drawdown_curve.values * 100,
                    fill='tozeroy',
                    fillcolor='rgba(255, 23, 68, 0.3)',
                    line=dict(color=self.theme['down'], width=1),
                    name='Drawdown %'
                ),
                row=1, col=2
            )
        
        # توزیع معاملات
        if hasattr(backtest_result, 'trades') and backtest_result.trades:
            pnl_values = [t.pnl for t in backtest_result.trades if t.pnl is not None]
            if pnl_values:
                fig.add_trace(
                    go.Histogram(
                        x=pnl_values,
                        nbinsx=20,
                        marker_color=self.theme['neutral'],
                        name='Trade PnL Distribution'
                    ),
                    row=2, col=1
                )
        
        # بازدهی ماهانه
        monthly_returns = self._calculate_monthly_returns(backtest_result)
        if monthly_returns is not None:
            colors = [self.theme['up'] if x >= 0 else self.theme['down'] for x in monthly_returns.values]
            
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index,
                    y=monthly_returns.values,
                    marker_color=colors,
                    name='Monthly Returns'
                ),
                row=2, col=2
            )
        
        # معیارهای عملکرد
        metrics_fig = self._create_metrics_gauge(backtest_result)
        fig.add_trace(metrics_fig.data[0], row=3, col=1)
        
        # تحلیل ریسک
        if monte_carlo_result:
            risk_fig = self._create_risk_analysis(monte_carlo_result)
            fig.add_trace(risk_fig.data[0], row=3, col=2)
        
        # به‌روزرسانی layout
        fig.update_layout(
            height=1200,
            title_text="Strategy Performance Dashboard",
            template="plotly_dark",
            showlegend=False,
            font=dict(color=self.theme['text']),
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['background']
        )
        
        return fig
    
    def _calculate_monthly_returns(self, backtest_result) -> Optional[pd.Series]:
        """محاسبه بازدهی ماهانه"""
        if not hasattr(backtest_result, 'equity_curve') or backtest_result.equity_curve is None:
            return None
        
        equity = backtest_result.equity_curve
        monthly_returns = equity.resample('M').last().pct_change().dropna() * 100
        return monthly_returns
    
    def _create_metrics_gauge(self, backtest_result) -> go.Figure:
        """ایجاد گیج معیارهای عملکرد"""
        fig = go.Figure()
        
        metrics = [
            ('Win Rate', backtest_result.win_rate, 0, 100, '%'),
            ('Sharpe Ratio', backtest_result.sharpe_ratio, 0, 3, ''),
            ('Profit Factor', min(backtest_result.profit_factor, 5), 0, 5, ''),
            ('Max Drawdown', backtest_result.max_drawdown * 100, 0, 50, '%')
        ]
        
        for i, (name, value, min_val, max_val, suffix) in enumerate(metrics):
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = value,
                title = {'text': name},
                domain = {'row': i, 'column': 0},
                gauge = {
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': self._get_metric_color(name, value)},
                    'steps': [
                        {'range': [min_val, max_val * 0.33], 'color': "lightgray"},
                        {'range': [max_val * 0.33, max_val * 0.66], 'color': "gray"}
                    ]
                }
            ))
        
        fig.update_layout(grid={'rows': 4, 'columns': 1, 'pattern': "independent"})
        return fig
    
    def _create_risk_analysis(self, monte_carlo_result) -> go.Figure:
        """ایجاد نمودار تحلیل ریسک"""
        fig = go.Figure()
        
        if monte_carlo_result.distribution.size > 0:
            fig.add_trace(go.Histogram(
                x=monte_carlo_result.distribution,
                nbinsx=50,
                name='Portfolio Value Distribution',
                marker_color=self.theme['neutral']
            ))
            
            # خطوط VaR و Expected Shortfall
            fig.add_vline(
                x=monte_carlo_result.value_at_risk,
                line_dash="dash",
                line_color=self.theme['down'],
                annotation_text=f"VaR {monte_carlo_result.confidence_level:.0%}"
            )
            
            fig.add_vline(
                x=monte_carlo_result.expected_shortfall,
                line_dash="dot",
                line_color=self.theme['down'],
                annotation_text="Expected Shortfall"
            )
        
        fig.update_layout(
            title="Monte Carlo Simulation Results",
            xaxis_title="Portfolio Value",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def _get_metric_color(self, metric_name: str, value: float) -> str:
        """رنگ‌بندی معیارها بر اساس کیفیت"""
        if metric_name == 'Max Drawdown':
            if value < 10: return self.theme['up']
            elif value < 20: return self.theme['neutral']
            else: return self.theme['down']
        elif metric_name == 'Win Rate':
            if value > 60: return self.theme['up']
            elif value > 40: return self.theme['neutral']
            else: return self.theme['down']
        elif metric_name == 'Sharpe Ratio':
            if value > 1.5: return self.theme['up']
            elif value > 0.5: return self.theme['neutral']
            else: return self.theme['down']
        else:  # Profit Factor
            if value > 2: return self.theme['up']
            elif value > 1: return self.theme['neutral']
            else: return self.theme['down']
    
    def create_live_trading_dashboard(self, portfolio: Dict, market_data: Dict, 
                                    signals: List) -> go.Figure:
        """ایجاد داشبورد تریدینگ زنده"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Portfolio Allocation', 'Market Overview',
                'Recent Signals', 'Performance Summary'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "indicator"}]
            ]
        )
        
        # تخصیص پورتفو
        if portfolio:
            symbols = list(portfolio.keys())
            allocations = [portfolio[s]['allocation'] for s in symbols]
            
            fig.add_trace(
                go.Pie(
                    labels=symbols,
                    values=allocations,
                    name="Portfolio Allocation"
                ),
                row=1, col=1
            )
        
        # نمای بازار
        if market_data:
            price_changes = []
            symbols = []
            
            for symbol, data in list(market_data.items())[:10]:  # 10 نماد اول
                if len(data) > 1:
                    change = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100
                    price_changes.append(change)
                    symbols.append(symbol)
            
            colors = [self.theme['up'] if x >= 0 else self.theme['down'] for x in price_changes]
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=price_changes,
                    marker_color=colors,
                    name="24h Price Change"
                ),
                row=1, col=2
            )
        
        # سیگنال‌های اخیر
        if signals:
            signal_data = []
            for signal in signals[:5]:  # 5 سیگنال اخیر
                signal_data.append([
                    signal.symbol,
                    signal.signal_type.value,
                    f"{signal.confidence:.1%}",
                    f"${signal.price:.2f}",
                    signal.time_horizon
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Symbol', 'Signal', 'Confidence', 'Price', 'Horizon'],
                        fill_color=self.theme['grid'],
                        font=dict(color=self.theme['text'])
                    ),
                    cells=dict(
                        values=list(zip(*signal_data)),
                        fill_color=[['rgba(0,0,0,0)'] * 5],
                        font=dict(color=self.theme['text'])
                    )
                ),
                row=2, col=1
            )
        
        # خلاصه عملکرد
        total_return = portfolio.get('total_return', 0) if portfolio else 0
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=total_return,
                number={'suffix': "%"},
                title={"text": "Total Return"},
                delta={'reference': 0}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Live Trading Dashboard",
            template="plotly_dark",
            showlegend=False
        )
        
        return fig

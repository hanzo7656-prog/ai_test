# 📁 src/visualization/chart_engine.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional

class ChartEngine:
    """موتور تولید نمودارهای حرفه‌ای"""
    
    def __init__(self):
        self.theme = {
            'background': '#0E1117',
            'text': '#FFFFFF',
            'grid': '#1E2130',
            'up': '#00C853',
            'down': '#FF1744',
            'neutral': '#FFC107'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, 
                               indicators: Dict = None,
                               signals: List[Dict] = None) -> go.Figure:
        """ایجاد نمودار کندل‌استیک با اندیکاتورها"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Price', 'Volume', 'RSI', 'MACD'],
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # کندل‌ها
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # حجم
        colors = [self.theme['up'] if data['close'].iloc[i] > data['open'].iloc[i] 
                 else self.theme['down'] for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # اضافه کردن اندیکاتورها
        if indicators:
            self._add_indicators(fig, indicators, data)
        
        # اضافه کردن سیگنال‌ها
        if signals:
            self._add_signals(fig, signals, data)
        
        # به‌روزرسانی layout
        fig.update_layout(
            title='Technical Analysis Chart',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _add_indicators(self, fig: go.Figure, indicators: Dict, data: pd.DataFrame):
        """اضافه کردن اندیکاتورها به نمودار"""
        # RSI
        if 'rsi' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['rsi'],
                    name='RSI',
                    line=dict(color=self.theme['neutral'])
                ),
                row=3, col=1
            )
            
            # اضافه کردن سطوح RSI
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd_signal'],
                    name='Signal',
                    line=dict(color='red')
                ),
                row=4, col=1
            )
            
            # هیستوگرام MACD
            histogram = indicators['macd'] - indicators['macd_signal']
            colors = [self.theme['up'] if val > 0 else self.theme['down'] for val in histogram]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=histogram,
                    name='MACD Histogram',
                    marker_color=colors
                ),
                row=4, col=1
            )
    
    def _add_signals(self, fig: go.Figure, signals: List[Dict], data: pd.DataFrame):
        """اضافه کردن سیگنال‌ها به نمودار"""
        buy_signals = [s for s in signals if s['signal_type'] in ['BUY', 'STRONG_BUY']]
        sell_signals = [s for s in signals if s['signal_type'] in ['SELL', 'STRONG_SELL']]
        
        # سیگنال‌های خرید
        if buy_signals:
            buy_dates = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=self.theme['up'],
                        line=dict(width=2, color='white')
                    )
                ),
                row=1, col=1
            )
        
        # سیگنال‌های فروش
        if sell_signals:
            sell_dates = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=self.theme['down'],
                        line=dict(width=2, color='white')
                    )
                ),
                row=1, col=1
            )
    
    def create_performance_chart(self, equity_curve: List[float], 
                               drawdown_curve: List[float]) -> go.Figure:
        """ایجاد نمودار عملکرد"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Equity Curve', 'Drawdown'],
            shared_xaxes=True
        )
        
        # منحنی سرمایه
        fig.add_trace(
            go.Scatter(
                y=equity_curve,
                mode='lines',
                name='Equity',
                line=dict(color=self.theme['up'], width=2)
            ),
            row=1, col=1
        )
        
        # درادداون
        fig.add_trace(
            go.Scatter(
                y=drawdown_curve,
                mode='lines',
                name='Drawdown',
                line=dict(color=self.theme['down'], width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Strategy Performance',
            template='plotly_dark',
            height=600
        )
        
        return fig

# 📁 src/backtesting/strategy_tester.py

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_winning_trade: float
    avg_losing_trade: float

class StrategyTester:
    """تستر استراتژی‌های معاملاتی"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1%
    
    def test_strategy(self, data: pd.DataFrame, strategy_func: Callable, 
                     **strategy_params) -> Dict:
        """تست یک استراتژی روی داده‌های تاریخی"""
        logger.info("🧪 Testing trading strategy...")
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(data)):
            current_data = data.iloc[:i]
            current_price = data['close'].iloc[i]
            
            # تولید سیگنال از استراتژی
            try:
                signal = strategy_func(current_data, **strategy_params)
            except Exception as e:
                logger.warning(f"Strategy error at step {i}: {e}")
                signal = {'action': 'HOLD', 'confidence': 0}
            
            # اجرای معامله
            if position is None and signal['action'] in ['BUY', 'SELL']:
                position = self._open_position(signal, current_price, i)
                
            elif position is not None:
                # بررسی شرایط خروج
                if self._should_exit_position(position, current_price, signal, i):
                    trade = self._close_position(position, current_price, i)
                    trades.append(trade)
                    capital += trade['pnl']
                    position = None
            
            # محاسبه equity
            if position is not None:
                current_equity = capital + self._calculate_unrealized_pnl(position, current_price)
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # بستن پوزیشن باز در پایان
        if position is not None:
            trade = self._close_position(position, data['close'].iloc[-1], len(data)-1)
            trades.append(trade)
            capital += trade['pnl']
        
        # محاسبه معیارهای عملکرد
        metrics = self._calculate_performance_metrics(trades, equity_curve)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital * 100
        }
    
    def _open_position(self, signal: Dict, price: float, step: int) -> Dict:
        """باز کردن پوزیشن جدید"""
        return {
            'entry_price': price,
            'entry_step': step,
            'entry_time': datetime.now(),
            'direction': 'LONG' if signal['action'] == 'BUY' else 'SHORT',
            'size': self.initial_capital * 0.1 / price,  # 10% سرمایه
            'signal_confidence': signal.get('confidence', 0.5),
            'stop_loss': price * 0.95 if signal['action'] == 'BUY' else price * 1.05,
            'take_profit': price * 1.05 if signal['action'] == 'BUY' else price * 0.95
        }
    
    def _should_exit_position(self, position: Dict, current_price: float, 
                            signal: Dict, step: int) -> bool:
        """بررسی شرایط خروج از پوزیشن"""
        # استاپ لاس
        if position['direction'] == 'LONG':
            if current_price <= position['stop_loss']:
                return True
            if current_price >= position['take_profit']:
                return True
        else:  # SHORT
            if current_price >= position['stop_loss']:
                return True
            if current_price <= position['take_profit']:
                return True
        
        # سیگنال معکوس
        if (position['direction'] == 'LONG' and signal['action'] == 'SELL') or \
           (position['direction'] == 'SHORT' and signal['action'] == 'BUY'):
            return True
        
        # تایم اوت (50 step)
        if step - position['entry_step'] > 50:
            return True
        
        return False
    
    def _close_position(self, position: Dict, exit_price: float, step: int) -> Dict:
        """بستن پوزیشن"""
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        # اعمال کمیسیون
        pnl -= (position['entry_price'] * position['size'] * self.commission)
        pnl -= (exit_price * position['size'] * self.commission)
        
        return {
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'direction': position['direction'],
            'pnl': pnl,
            'return_pct': (pnl / (position['entry_price'] * position['size'])) * 100,
            'holding_period': step - position['entry_step'],
            'entry_step': position['entry_step'],
            'exit_step': step
        }
    
    def _calculate_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """محاسده PnL تحقق نیافته"""
        if position['direction'] == 'LONG':
            return (current_price - position['entry_price']) * position['size']
        else:
            return (position['entry_price'] - current_price) * position['size']
    
    def _calculate_performance_metrics(self, trades: List[Dict], equity_curve: List[float]) -> StrategyMetrics:
        """محاسبه معیارهای عملکرد استراتژی"""
        if not trades:
            return StrategyMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # معاملات برنده و بازنده
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        # بازدهی کل
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe Ratio
        returns = [t['return_pct'] for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdowns.max() * 100
        
        # Win Rate
        win_rate = len(winning_trades) / len(trades) * 100
        
        # Profit Factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # میانگین معاملات
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        return StrategyMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_winning_trade=avg_win,
            avg_losing_trade=avg_loss
        )

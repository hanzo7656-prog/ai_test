# 📁 src/backtesting/walk_forward.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    holding_period: Optional[float]
    stop_loss: float
    take_profit: List[float]
    signal_confidence: float

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_trade: float
    best_trade: float
    worst_trade: float
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series

class WalkForwardAnalyzer:
    """آنالیزور Walk-Forward برای بهینه‌سازی و اعتبارسنجی استراتژی"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage
        
    def run_walk_forward_analysis(self, strategy: Callable, data: pd.DataFrame, 
                                window_size: int = 100, step_size: int = 20,
                                optimization_params: Dict = None) -> Dict:
        """اجرای تحلیل Walk-Forward"""
        print("🔄 Running Walk-Forward Analysis...")
        
        results = []
        optimized_params_history = []
        
        total_windows = (len(data) - window_size) // step_size
        
        for i in range(0, len(data) - window_size, step_size):
            # تقسیم داده به آموزش و تست
            train_data = data.iloc[i:i + window_size]
            test_data = data.iloc[i + window_size:i + window_size + step_size]
            
            if len(test_data) < 10:  # حداقل داده برای تست
                continue
            
            print(f"   Window {i//step_size + 1}/{total_windows}: "
                  f"Train {len(train_data)} bars, Test {len(test_data)} bars")
            
            # بهینه‌سازی پارامترها روی داده آموزش
            if optimization_params:
                optimized_params = self._optimize_parameters(strategy, train_data, optimization_params)
                optimized_params_history.append(optimized_params)
            else:
                optimized_params = {}
            
            # تست استراتژی روی داده تست
            strategy_instance = strategy(**optimized_params)
            result = self._backtest_strategy(strategy_instance, test_data, self.initial_capital)
            
            results.append({
                'window': i // step_size + 1,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'result': result,
                'optimized_params': optimized_params
            })
        
        # تحلیل کلی نتایج
        summary = self._analyze_walk_forward_results(results)
        
        print(f"✅ Walk-Forward Analysis Completed: {len(results)} windows analyzed")
        print(f"📊 Average Win Rate: {summary['average_win_rate']:.2f}%")
        print(f"📈 Average Return: {summary['average_return']:.2f}%")
        
        return {
            'window_results': results,
            'summary': summary,
            'optimized_params_history': optimized_params_history
        }
    
    def _optimize_parameters(self, strategy: Callable, data: pd.DataFrame, 
                           param_grid: Dict) -> Dict:
        """بهینه‌سازی پارامترهای استراتژی"""
        best_params = {}
        best_performance = -np.inf
        
        # تولید ترکیب‌های پارامتر
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations[:10]:  # تست 10 ترکیب اول برای سرعت
            try:
                strategy_instance = strategy(**params)
                result = self._backtest_strategy(strategy_instance, data, self.initial_capital)
                
                # معیار عملکرد: Sharpe ratio
                performance = result.sharpe_ratio
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = params
            except Exception as e:
                continue
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """تولید ترکیب‌های پارامتر"""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _backtest_strategy(self, strategy, data: pd.DataFrame, initial_capital: float) -> BacktestResult:
        """اجرای بک‌تست برای یک استراتژی"""
        capital = initial_capital
        position = None
        trades = []
        equity_curve = []
        drawdown_curve = []
        
        peak_capital = initial_capital
        
        for i in range(1, len(data)):
            current_data = data.iloc[:i]
            current_price = data['close'].iloc[i]
            current_time = data.index[i]
            
            # دریافت سیگنال از استراتژی
            try:
                signal = strategy.generate_signal(current_data)
            except:
                signal = {'action': 'HOLD', 'confidence': 0}
            
            # مدیریت پوزیشن
            if position is None and signal['action'] in ['BUY', 'SELL']:
                # باز کردن پوزیشن جدید
                position = self._open_position(signal, current_price, current_time, capital)
                
            elif position is not None:
                # بررسی خروج از پوزیشن
                exit_signal = self._check_exit_conditions(position, current_price, current_time, signal)
                
                if exit_signal:
                    # بستن پوزیشن
                    trade = self._close_position(position, current_price, current_time)
                    trades.append(trade)
                    
                    # به‌روزرسانی سرمایه
                    capital += trade.pnl
                    position = None
            
            # محاسبه equity و drawdown
            if position is not None:
                current_equity = capital + self._calculate_position_value(position, current_price)
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            peak_capital = max(peak_capital, current_equity)
            drawdown = (peak_capital - current_equity) / peak_capital
            drawdown_curve.append(drawdown)
        
        # بستن پوزیشن باز در پایان
        if position is not None:
            trade = self._close_position(position, data['close'].iloc[-1], data.index[-1])
            trades.append(trade)
            capital += trade.pnl
        
        # محاسبه معیارهای عملکرد
        metrics = self._calculate_performance_metrics(trades, equity_curve, initial_capital)
        
        return BacktestResult(
            total_trades=len(trades),
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            total_pnl=metrics['total_pnl'],
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            profit_factor=metrics['profit_factor'],
            avg_trade=metrics['avg_trade'],
            best_trade=metrics['best_trade'],
            worst_trade=metrics['worst_trade'],
            trades=trades,
            equity_curve=pd.Series(equity_curve, index=data.index[1:len(equity_curve)+1]),
            drawdown_curve=pd.Series(drawdown_curve, index=data.index[1:len(drawdown_curve)+1])
        )
    
    def _open_position(self, signal: Dict, price: float, timestamp: pd.Timestamp, capital: float) -> Trade:
        """باز کردن پوزیشن جدید"""
        # محاسبه سایز پوزیشن (2% ریسک)
        risk_amount = capital * 0.02
        stop_loss = price * (1 - 0.02) if signal['action'] == 'BUY' else price * (1 + 0.02)
        risk_per_share = abs(price - stop_loss)
        
        quantity = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # اعمال کمیسیون و slippage
        entry_price = price * (1 + self.slippage) if signal['action'] == 'BUY' else price * (1 - self.slippage)
        entry_price *= (1 + self.commission)
        
        direction = TradeDirection.LONG if signal['action'] == 'BUY' else TradeDirection.SHORT
        
        return Trade(
            entry_time=timestamp,
            exit_time=None,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            pnl=None,
            pnl_percentage=None,
            holding_period=None,
            stop_loss=stop_loss,
            take_profit=[price * (1 + 0.04) if direction == TradeDirection.LONG else price * (1 - 0.04)],
            signal_confidence=signal.get('confidence', 0.5)
        )
    
    def _close_position(self, position: Trade, price: float, timestamp: pd.Timestamp) -> Trade:
        """بستن پوزیشن"""
        # اعمال کمیسیون و slippage
        exit_price = price * (1 - self.slippage) if position.direction == TradeDirection.LONG else price * (1 + self.slippage)
        exit_price *= (1 - self.commission)
        
        # محاسبه PnL
        if position.direction == TradeDirection.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        pnl_percentage = (pnl / (position.entry_price * position.quantity)) * 100
        holding_period = (timestamp - position.entry_time).total_seconds() / 3600  # hours
        
        return Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            holding_period=holding_period,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            signal_confidence=position.signal_confidence
        )
    
    def _check_exit_conditions(self, position: Trade, current_price: float, 
                             current_time: pd.Timestamp, signal: Dict) -> bool:
        """بررسی شرایط خروج از پوزیشن"""
        # استاپ لاس
        if position.direction == TradeDirection.LONG:
            if current_price <= position.stop_loss:
                return True
        else:
            if current_price >= position.stop_loss:
                return True
        
        # تیک پروفیت
        for tp in position.take_profit:
            if position.direction == TradeDirection.LONG and current_price >= tp:
                return True
            elif position.direction == TradeDirection.SHORT and current_price <= tp:
                return True
        
        # سیگنال معکوس
        if signal['action'] == 'SELL' and position.direction == TradeDirection.LONG:
            return True
        elif signal['action'] == 'BUY' and position.direction == TradeDirection.SHORT:
            return True
        
        # تایم اوت (حداکثر 24 ساعت)
        holding_hours = (current_time - position.entry_time).total_seconds() / 3600
        if holding_hours > 24:
            return True
        
        return False
    
    def _calculate_position_value(self, position: Trade, current_price: float) -> float:
        """محاسبه ارزش فعلی پوزیشن"""
        if position.direction == TradeDirection.LONG:
            return (current_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - current_price) * position.quantity
    
    def _calculate_performance_metrics(self, trades: List[Trade], equity_curve: List[float], 
                                     initial_capital: float) -> Dict:
        """محاسبه معیارهای عملکرد"""
        if not trades:
            return {
                'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
                'total_pnl': 0, 'total_return': 0, 'sharpe_ratio': 0,
                'max_drawdown': 0, 'profit_factor': 0, 'avg_trade': 0,
                'best_trade': 0, 'worst_trade': 0
            }
        
        # معاملات برنده و بازنده
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
        win_rate = len(winning_trades) / len(trades) * 100
        
        # PnL
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        total_return = (total_pnl / initial_capital) * 100
        
        # Sharpe Ratio (ساده)
        returns = [t.pnl_percentage for t in trades if t.pnl_percentage]
        if returns and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdowns.max()
        
        # Profit Factor
        gross_profit = sum(t.pnl for t in winning_trades if t.pnl)
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # سایر معیارها
        avg_trade = total_pnl / len(trades)
        best_trade = max(t.pnl for t in trades if t.pnl) if trades else 0
        worst_trade = min(t.pnl for t in trades if t.pnl) if trades else 0
        
        return {
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }
    
    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict:
        """تحلیل کلی نتایج Walk-Forward"""
        if not results:
            return {}
        
        win_rates = [r['result'].win_rate for r in results]
        returns = [r['result'].total_return for r in results]
        sharpe_ratios = [r['result'].sharpe_ratio for r in results]
        max_drawdowns = [r['result'].max_drawdown for r in results]
        
        return {
            'average_win_rate': np.mean(win_rates),
            'average_return': np.mean(returns),
            'average_sharpe': np.mean(sharpe_ratios),
            'average_max_drawdown': np.mean(max_drawdowns),
            'win_rate_std': np.std(win_rates),
            'return_std': np.std(returns),
            'consistency_score': self._calculate_consistency(returns),
            'total_windows': len(results),
            'robustness_ratio': len([r for r in returns if r > 0]) / len(returns)
        }
    
    def _calculate_consistency(self, returns: List[float]) -> float:
        """محاسبه نمره سازگاری استراتژی"""
        if not returns:
            return 0.0
        
        positive_periods = len([r for r in returns if r > 0])
        total_periods = len(returns)
        
        return positive_periods / total_periods

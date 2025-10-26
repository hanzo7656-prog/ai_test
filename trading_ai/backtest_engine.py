# backtest_engine.py - بک‌تست واقعی روی داده‌های تاریخی
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

from database_manager import trading_db
from model_trainer import model_trainer

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """نتایج بک‌تست"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_profit: float
    avg_loss: float
    profit_factor: float
    best_trade: float
    worst_trade: float

class RealBacktestEngine:
    """موتور بک‌تست واقعی"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                    strategy_type: str = "ai_signals") -> BacktestResult:
        """اجرای بک‌تست روی دوره مشخص"""
        try:
            # دریافت داده‌های تاریخی
            df = trading_db.get_historical_data(symbol, 365)
            
            if df.empty:
                logger.error(f"❌ داده‌های تاریخی برای {symbol} یافت نشد")
                return None
            
            # فیلتر کردن براساس تاریخ
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            df_period = df[mask]
            
            if df_period.empty:
                logger.error(f"❌ داده‌ای در بازه {start_date} تا {end_date} یافت نشد")
                return None
            
            # اجرای استراتژی
            if strategy_type == "ai_signals":
                results = self._run_ai_strategy(df_period, symbol)
            elif strategy_type == "buy_hold":
                results = self._run_buy_hold_strategy(df_period)
            else:
                results = self._run_technical_strategy(df_period)
            
            logger.info(f"✅ بک‌تست {symbol} تکمیل شد")
            return results
            
        except Exception as e:
            logger.error(f"❌ خطا در بک‌تست {symbol}: {e}")
            return None
    
    def _run_ai_strategy(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """اجرای استراتژی مبتنی بر AI"""
        trades = []
        balance = self.initial_balance
        position = 0
        entry_price = 0
        
        equity_curve = [balance]
        
        for i in range(30, len(df)):  # شروع از روز 30ام برای داشتن اندیکاتورها
            current_data = df.iloc[i]
            current_date = df.index[i]
            
            # شبیه‌سازی داده‌های ورودی مدل
            feature_data = {
                'open': current_data['open'],
                'high': current_data['high'],
                'low': current_data['low'],
                'close': current_data['close'],
                'volume': current_data['volume'],
                'rsi': current_data.get('rsi', 50),
                'macd': current_data.get('macd', 0),
                'bollinger_upper': current_data.get('bollinger_upper', current_data['close'] * 1.1),
                'bollinger_middle': current_data.get('bollinger_middle', current_data['close']),
                'bollinger_lower': current_data.get('bollinger_lower', current_data['close'] * 0.9),
                'sma_20': current_data.get('sma_20', current_data['close']),
                'ema_12': current_data.get('ema_12', current_data['close']),
                'atr': current_data.get('atr', 0),
                'price_change': current_data['close'] / df.iloc[i-1]['close'] - 1,
                'volume_change': current_data['volume'] / df.iloc[i-1]['volume'] - 1,
                'high_low_ratio': current_data['high'] / current_data['low']
            }
            
            # دریافت سیگنال AI (شبیه‌سازی شده)
            signal = self._simulate_ai_signal(feature_data)
            
            # اجرای معامله
            if signal == 'BUY' and position <= 0:
                if position < 0:  # بستن پوزیشن فروش
                    pnl = (entry_price - current_data['close']) * abs(position)
                    balance += pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'type': 'SHORT_COVER',
                        'pnl': pnl
                    })
                
                # باز کردن پوزیشن خرید
                position = balance * 0.1 / current_data['close']  # 10% سرمایه
                entry_price = current_data['close']
                entry_date = current_date
                
            elif signal == 'SELL' and position >= 0:
                if position > 0:  # بستن پوزیشن خرید
                    pnl = (current_data['close'] - entry_price) * position
                    balance += pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'type': 'LONG_CLOSE',
                        'pnl': pnl
                    })
                
                # باز کردن پوزیشن فروش
                position = -balance * 0.1 / current_data['close']  # 10% سرمایه
                entry_price = current_data['close']
                entry_date = current_date
            
            # محاسبه ارزش سبد
            if position != 0:
                current_value = balance + position * (current_data['close'] - entry_price)
            else:
                current_value = balance
            
            equity_curve.append(current_value)
        
        # بستن پوزیشن آخر در پایان دوره
        if position != 0:
            last_price = df.iloc[-1]['close']
            if position > 0:
                pnl = (last_price - entry_price) * position
            else:
                pnl = (entry_price - last_price) * abs(position)
            balance += pnl
        
        return self._calculate_performance_metrics(equity_curve, trades)
    
    def _simulate_ai_signal(self, data: Dict) -> str:
        """شبیه‌سازی سیگنال AI (موقت - تا آموزش مدل واقعی)"""
        # منطق ساده براساس اندیکاتورها
        rsi = data.get('rsi', 50)
        macd = data.get('macd', 0)
        
        if rsi < 30 and macd > 0:
            return 'BUY'
        elif rsi > 70 and macd < 0:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _run_buy_hold_strategy(self, df: pd.DataFrame) -> BacktestResult:
        """استراتژی خرید و نگهداری"""
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        
        total_return = (end_price / start_price - 1) * 100
        equity_curve = [self.initial_balance * (1 + (df.iloc[i]['close'] / start_price - 1)) 
                       for i in range(len(df))]
        
        return self._calculate_performance_metrics(equity_curve, [])
    
    def _calculate_performance_metrics(self, equity_curve: List[float], 
                                    trades: List[Dict]) -> BacktestResult:
        """محاسبه معیارهای عملکرد"""
        if not equity_curve:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # محاسبات بازده
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # محاسبه drawdown
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # محاسبات معاملات
        profitable_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(profitable_trades) / len(trades) * 100 if trades else 0
        
        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
        
        best_trade = max([t['pnl'] for t in trades]) if trades else 0
        worst_trade = min([t['pnl'] for t in trades]) if trades else 0
        
        # محاسبه Sharpe Ratio (ساده‌شده)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(trades),
            profitable_trades=len(profitable_trades),
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            best_trade=best_trade,
            worst_trade=worst_trade
        )

# ایجاد نمونه گلوبال
backtest_engine = RealBacktestEngine()

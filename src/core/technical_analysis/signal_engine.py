# 📁 src/core/technical_analysis/signal_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from ...utils.performance_tracker import PerformanceTracker

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: pd.Timestamp
    reasons: List[str]
    targets: List[float]
    stop_loss: float
    time_horizon: str
    risk_reward_ratio: float

class IntelligentSignalEngine:
    """موتور تولید سیگنال هوشمند با ترکیب وزنی"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.signal_weights = {
            'momentum': 0.25,
            'trend': 0.20,
            'volume': 0.15,
            'volatility': 0.15,
            'pattern': 0.15,
            'risk': 0.10
        }
        
    def generate_signals(self, market_data: Dict, technical_indicators: Dict) -> List[TradingSignal]:
        """تولید سیگنال‌های معاملاتی"""
        signals = []
        
        with self.performance_tracker.track("signal_generation"):
            for symbol, data in market_data.items():
                signal = self._analyze_symbol(symbol, data, technical_indicators.get(symbol, {}))
                if signal:
                    signals.append(signal)
        
        return sorted(signals, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_symbol(self, symbol: str, data: pd.DataFrame, indicators: Dict) -> Optional[TradingSignal]:
        """آنالیز تکنیکال برای یک نماد"""
        if len(data) < 50:  # حداقل داده مورد نیاز
            return None
        
        current_price = data['close'].iloc[-1]
        
        # محاسبه امتیازات جزئی
        scores = {
            'momentum': self._calculate_momentum_score(data, indicators),
            'trend': self._calculate_trend_strength(data, indicators),
            'volume': self._analyze_volume_confirmation(data, indicators),
            'volatility': self._assess_volatility_conditions(data, indicators),
            'pattern': self._evaluate_patterns(indicators),
            'risk': self._calculate_risk_score(data, indicators)
        }
        
        # ترکیب وزنی امتیازات
        total_score = sum(score * self.signal_weights[category] 
                         for category, score in scores.items())
        
        # تبدیل به سیگنال
        signal_type, confidence = self._score_to_signal(total_score)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # محاسبه سطوح هدف و استاپ
        targets, stop_loss = self._calculate_levels(data, signal_type, indicators)
        
        # جمع‌آوری دلایل
        reasons = self._generate_reasons(scores, signal_type)
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            timestamp=data.index[-1],
            reasons=reasons,
            targets=targets,
            stop_loss=stop_loss,
            time_horizon=self._determine_time_horizon(scores['trend']),
            risk_reward_ratio=self._calculate_risk_reward(current_price, targets, stop_loss)
        )
    
    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        """محاسبه امتیاز مومنتوم"""
        score = 0.0
        
        # RSI Analysis
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if rsi < 30:
                score += 0.3  # Oversold
            elif rsi > 70:
                score -= 0.3  # Overbought
        
        # MACD Analysis
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            if macd > macd_signal and indicators['macd'].iloc[-2] <= indicators['macd_signal'].iloc[-2]:
                score += 0.4  # Bullish crossover
            elif macd < macd_signal and indicators['macd'].iloc[-2] >= indicators['macd_signal'].iloc[-2]:
                score -= 0.4  # Bearish crossover
        
        # Price Momentum
        returns_5 = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100
        returns_20 = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
        
        if returns_5 > 2 and returns_20 > 5:
            score += 0.3
        elif returns_5 < -2 and returns_20 < -5:
            score -= 0.3
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_trend_strength(self, data: pd.DataFrame, indicators: Dict) -> float:
        """محاسبه قدرت روند"""
        # Moving Average Analysis
        ma_20 = data['close'].rolling(20).mean()
        ma_50 = data['close'].rolling(50).mean()
        
        price_vs_ma20 = (data['close'].iloc[-1] / ma_20.iloc[-1] - 1) * 100
        ma20_vs_ma50 = (ma_20.iloc[-1] / ma_50.iloc[-1] - 1) * 100
        
        trend_score = 0.0
        
        if price_vs_ma20 > 2 and ma20_vs_ma50 > 1:
            trend_score = 0.8  # Strong uptrend
        elif price_vs_ma20 < -2 and ma20_vs_ma50 < -1:
            trend_score = -0.8  # Strong downtrend
        elif abs(price_vs_ma20) < 1 and abs(ma20_vs_ma50) < 0.5:
            trend_score = 0.0  # Ranging market
        
        return trend_score
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame, indicators: Dict) -> float:
        """تأیید حجم معاملات"""
        if 'volume' not in data:
            return 0.0
        
        volume_avg_20 = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / volume_avg_20.iloc[-1]
        
        price_change = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100
        
        if volume_ratio > 1.5 and price_change > 1:
            return 0.7  # Strong volume confirmation for uptrend
        elif volume_ratio > 1.5 and price_change < -1:
            return -0.7  # Strong volume confirmation for downtrend
        
        return 0.0
    
    def _assess_volatility_conditions(self, data: pd.DataFrame, indicators: Dict) -> float:
        """ارزیابی شرایط نوسان"""
        if 'atr' in indicators:
            atr = indicators['atr'].iloc[-1]
            atr_percentage = (atr / data['close'].iloc[-1]) * 100
            
            if atr_percentage > 3:
                return -0.5  # High volatility - risky
            elif atr_percentage < 1:
                return 0.3  # Low volatility - good for trend following
        
        return 0.0
    
    def _evaluate_patterns(self, indicators: Dict) -> float:
        """ارزیابی الگوهای تکنیکال"""
        pattern_score = 0.0
        
        if 'patterns' in indicators:
            patterns = indicators['patterns']
            latest_patterns = {k: v.iloc[-1] if hasattr(v, 'iloc') else v for k, v in patterns.items()}
            
            bullish_patterns = ['hammer', 'bullish_engulfing', 'morning_star']
            bearish_patterns = ['evening_star', 'bearish_engulfing']
            
            for pattern in bullish_patterns:
                if latest_patterns.get(pattern):
                    pattern_score += 0.2
            
            for pattern in bearish_patterns:
                if latest_patterns.get(pattern):
                    pattern_score -= 0.2
        
        return max(-0.6, min(0.6, pattern_score))
    
    def _calculate_risk_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        """محاسبه امتیاز ریسک"""
        risk_score = 0.0
        
        # Drawdown Analysis
        recent_high = data['high'].rolling(20).max().iloc[-1]
        current_drawdown = (data['close'].iloc[-1] / recent_high - 1) * 100
        
        if current_drawdown < -8:
            risk_score -= 0.4  # Significant drawdown
        
        # Volatility Risk
        if 'atr' in indicators:
            atr_percentage = (indicators['atr'].iloc[-1] / data['close'].iloc[-1]) * 100
            if atr_percentage > 4:
                risk_score -= 0.3
        
        return max(-1.0, min(0.0, risk_score))
    
    def _score_to_signal(self, total_score: float) -> Tuple[SignalType, float]:
        """تبدیل امتیاز به سیگنال"""
        confidence = min(1.0, abs(total_score))
        
        if total_score > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif total_score > 0.2:
            return SignalType.BUY, confidence
        elif total_score < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif total_score < -0.2:
            return SignalType.SELL, confidence
        else:
            return SignalType.HOLD, confidence
    
    def _calculate_levels(self, data: pd.DataFrame, signal_type: SignalType, indicators: Dict) -> Tuple[List[float], float]:
        """محاسبه سطوح هدف و استاپ لاس"""
        current_price = data['close'].iloc[-1]
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            # برای سیگنال خرید
            resistance_levels = self._find_resistance_levels(data)
            stop_loss = self._find_support_levels(data)[0] if self._find_support_levels(data) else current_price * 0.95
            
            targets = []
            for i, level in enumerate(resistance_levels[:3]):  # حداکثر 3 هدف
                if level > current_price * 1.02:  # حداقل 2% سود
                    targets.append(level)
            
            # اگر سطح مقاومت پیدا نشد، اهداف درصدی
            if not targets:
                targets = [
                    current_price * 1.03,
                    current_price * 1.06,
                    current_price * 1.10
                ]
        
        else:  # برای سیگنال فروش
            support_levels = self._find_support_levels(data)
            stop_loss = self._find_resistance_levels(data)[0] if self._find_resistance_levels(data) else current_price * 1.05
            
            targets = []
            for level in support_levels[:3]:
                if level < current_price * 0.98:  # حداقل 2% سود
                    targets.append(level)
            
            if not targets:
                targets = [
                    current_price * 0.97,
                    current_price * 0.94,
                    current_price * 0.90
                ]
        
        return targets, stop_loss
    
    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """پیدا کردن سطوح حمایت"""
        # پیاده‌سازی ساده - در نسخه کامل از الگوریتم‌های پیشرفته استفاده می‌شود
        low_20 = data['low'].rolling(20).min()
        support_levels = []
        
        for i in range(20, len(data)):
            if data['low'].iloc[i] == low_20.iloc[i]:
                support_levels.append(data['low'].iloc[i])
        
        return sorted(set(support_levels[-5:]))  # 5 سطح اخیر
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """پیدا کردن سطوح مقاومت"""
        high_20 = data['high'].rolling(20).max()
        resistance_levels = []
        
        for i in range(20, len(data)):
            if data['high'].iloc[i] == high_20.iloc[i]:
                resistance_levels.append(data['high'].iloc[i])
        
        return sorted(set(resistance_levels[-5:]), reverse=True)
    
    def _generate_reasons(self, scores: Dict, signal_type: SignalType) -> List[str]:
        """تولید دلایل سیگنال"""
        reasons = []
        
        if scores['momentum'] > 0.3:
            reasons.append("مومنتوم صعودی قوی")
        elif scores['momentum'] < -0.3:
            reasons.append("مومنتوم نزولی قوی")
        
        if scores['trend'] > 0.5:
            reasons.append("روند صعودی پایدار")
        elif scores['trend'] < -0.5:
            reasons.append("روند نزولی پایدار")
        
        if abs(scores['volume']) > 0.5:
            reasons.append("تأیید حجم معاملات")
        
        if scores['pattern'] > 0.3:
            reasons.append("الگوهای شمعی صعودی")
        elif scores['pattern'] < -0.3:
            reasons.append("الگوهای شمعی نزولی")
        
        return reasons
    
    def _determine_time_horizon(self, trend_score: float) -> str:
        """تعیین افق زمانی معامله"""
        if abs(trend_score) > 0.6:
            return "MEDIUM_TERM"  # 1-4 weeks
        elif abs(trend_score) > 0.3:
            return "SHORT_TERM"   # 1-7 days
        else:
            return "INTRADAY"     # 1-24 hours
    
    def _calculate_risk_reward(self, entry: float, targets: List[float], stop_loss: float) -> float:
        """محاسبه نسبت ریسک به ریوارد"""
        if not targets:
            return 1.0
        
        avg_target = sum(targets) / len(targets)
        potential_profit = avg_target - entry
        potential_loss = entry - stop_loss
        
        if potential_loss <= 0:
            return 1.0
        
        return round(potential_profit / potential_loss, 2)

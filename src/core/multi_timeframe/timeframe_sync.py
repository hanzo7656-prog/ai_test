# 📁 src/core/multi_timeframe/timeframe_sync.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from ...utils.performance_tracker import PerformanceTracker

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

@dataclass
class TimeFrameData:
    timeframe: TimeFrame
    data: pd.DataFrame
    indicators: Dict

class MultiTimeframeAnalyzer:
    """آنالیزور چندزمانه برای تحلیل سلسله مراتبی بازار"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.timeframe_hierarchy = [
            TimeFrame.W1,  # Primary trend
            TimeFrame.D1,  # Medium trend
            TimeFrame.H4,  # Short trend
            TimeFrame.H1,  # Intraday
            TimeFrame.M15  # Entry timing
        ]
    
    def analyze_symbol(self, symbol: str, multi_tf_data: Dict[TimeFrame, pd.DataFrame]) -> Dict:
        """آنالیز کامل نماد در تایم‌فریم‌های مختلف"""
        analysis_result = {}
        
        with self.performance_tracker.track("multi_timeframe_analysis"):
            # تحلیل هر تایم‌فریم
            for tf in self.timeframe_hierarchy:
                if tf in multi_tf_data:
                    tf_analysis = self._analyze_timeframe(symbol, tf, multi_tf_data[tf])
                    analysis_result[tf.value] = tf_analysis
            
            # تحلیل سلسله مراتبی
            analysis_result['hierarchical'] = self._hierarchical_analysis(analysis_result)
            analysis_result['consensus'] = self._calculate_consensus(analysis_result)
        
        return analysis_result
    
    def _analyze_timeframe(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame) -> Dict:
        """آنالیز یک تایم‌فریم خاص"""
        if len(data) < 20:
            return {}
        
        current_price = data['close'].iloc[-1]
        
        # محاسبه اندیکاتورهای پایه
        ma_20 = data['close'].rolling(20).mean()
        ma_50 = data['close'].rolling(50).mean()
        rsi = self._calculate_rsi(data['close'])
        
        # تشخیص روند
        trend_direction, trend_strength = self._detect_trend(data, ma_20, ma_50)
        
        # سطوح کلیدی
        support_levels = self._find_support_resistance(data, 'support')
        resistance_levels = self._find_support_resistance(data, 'resistance')
        
        return {
            'symbol': symbol,
            'timeframe': timeframe.value,
            'current_price': current_price,
            'trend': {
                'direction': trend_direction,
                'strength': trend_strength,
                'angle': self._calculate_trend_angle(data)
            },
            'moving_averages': {
                'ma_20': ma_20.iloc[-1],
                'ma_50': ma_50.iloc[-1],
                'price_vs_ma20': (current_price / ma_20.iloc[-1] - 1) * 100,
                'ma20_vs_ma50': (ma_20.iloc[-1] / ma_50.iloc[-1] - 1) * 100
            },
            'momentum': {
                'rsi': rsi.iloc[-1],
                'rsi_signal': self._get_rsi_signal(rsi.iloc[-1]),
                'price_change_1d': (current_price / data['close'].iloc[-24] - 1) * 100 if len(data) >= 24 else 0
            },
            'key_levels': {
                'support': support_levels[-3:],  # 3 سطح حمایت اخیر
                'resistance': resistance_levels[-3:],  # 3 سطح مقاومت اخیر
                'breakout_level': self._find_breakout_level(data, support_levels, resistance_levels)
            },
            'volume_analysis': {
                'volume_trend': self._analyze_volume_trend(data),
                'volume_vs_average': (data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]) if len(data) >= 20 else 1
            }
        }
    
    def _hierarchical_analysis(self, tf_analysis: Dict) -> Dict:
        """تحلیل سلسله مراتبی بین تایم‌فریم‌ها"""
        hierarchical = {
            'primary_trend': tf_analysis.get('1w', {}).get('trend', {}),
            'medium_trend': tf_analysis.get('1d', {}).get('trend', {}),
            'short_trend': tf_analysis.get('4h', {}).get('trend', {}),
            'alignment_score': 0.0,
            'conflicts': [],
            'trading_opportunity': 'NEUTRAL'
        }
        
        # محاسبه امتیاز همسویی
        alignment_scores = []
        trends = []
        
        for tf in ['1w', '1d', '4h']:
            if tf in tf_analysis:
                trend = tf_analysis[tf]['trend']
                trends.append(trend.get('direction', 'SIDEWAYS'))
                alignment_scores.append(trend.get('strength', 0))
        
        if trends:
            # بررسی همسویی روندها
            if all(t == 'UPTREND' for t in trends):
                hierarchical['alignment_score'] = sum(alignment_scores) / len(alignment_scores)
                hierarchical['trading_opportunity'] = 'STRONG_BUY'
            elif all(t == 'DOWNTREND' for t in trends):
                hierarchical['alignment_score'] = sum(alignment_scores) / len(alignment_scores)
                hierarchical['trading_opportunity'] = 'STRONG_SELL'
            else:
                # شناسایی تضادها
                for i, (tf1, trend1) in enumerate(zip(['1w', '1d', '4h'], trends)):
                    for j, (tf2, trend2) in enumerate(zip(['1w', '1d', '4h'], trends)):
                        if i < j and trend1 != trend2:
                            hierarchical['conflicts'].append(f"{tf1}({trend1}) vs {tf2}({trend2})")
                
                hierarchical['alignment_score'] = 0.3  # امتیاز پایین برای تضاد
        
        return hierarchical
    
    def _calculate_consensus(self, tf_analysis: Dict) -> Dict:
        """محاسبه اجماع بین تایم‌فریم‌ها"""
        consensus = {
            'overall_trend': 'SIDEWAYS',
            'confidence': 0.0,
            'key_support': None,
            'key_resistance': None,
            'momentum_bias': 'NEUTRAL'
        }
        
        # جمع‌آوری داده از تمام تایم‌فریم‌ها
        all_trends = []
        all_supports = []
        all_resistances = []
        all_momentums = []
        
        for tf, analysis in tf_analysis.items():
            if tf == 'hierarchical' or tf == 'consensus':
                continue
            
            # روندها
            trend = analysis.get('trend', {})
            if trend.get('direction') != 'SIDEWAYS':
                all_trends.append((trend['direction'], trend['strength']))
            
            # سطوح
            levels = analysis.get('key_levels', {})
            all_supports.extend(levels.get('support', []))
            all_resistances.extend(levels.get('resistance', []))
            
            # مومنتوم
            momentum = analysis.get('momentum', {})
            rsi_signal = momentum.get('rsi_signal', 'NEUTRAL')
            if rsi_signal != 'NEUTRAL':
                all_momentums.append(rsi_signal)
        
        # اجماع روند
        if all_trends:
            uptrend_weight = sum(weight for direction, weight in all_trends if direction == 'UPTREND')
            downtrend_weight = sum(weight for direction, weight in all_trends if direction == 'DOWNTREND')
            
            if uptrend_weight > downtrend_weight:
                consensus['overall_trend'] = 'UPTREND'
                consensus['confidence'] = uptrend_weight / (uptrend_weight + downtrend_weight)
            else:
                consensus['overall_trend'] = 'DOWNTREND'
                consensus['confidence'] = downtrend_weight / (uptrend_weight + downtrend_weight)
        
        # اجماع سطوح
        if all_supports:
            consensus['key_support'] = max(all_supports)  # قوی‌ترین حمایت
        if all_resistances:
            consensus['key_resistance'] = min(all_resistances)  # قوی‌ترین مقاومت
        
        # اجماع مومنتوم
        if all_momentums:
            bullish_count = all_momentums.count('BULLISH')
            bearish_count = all_momentums.count('BEARISH')
            if bullish_count > bearish_count:
                consensus['momentum_bias'] = 'BULLISH'
            elif bearish_count > bullish_count:
                consensus['momentum_bias'] = 'BEARISH'
        
        return consensus
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """محاسبه RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _detect_trend(self, data: pd.DataFrame, ma_20: pd.Series, ma_50: pd.Series) -> Tuple[str, float]:
        """تشخیص روند و قدرت آن"""
        if len(data) < 50:
            return "SIDEWAYS", 0.0
        
        current_price = data['close'].iloc[-1]
        price_ma20_ratio = (current_price / ma_20.iloc[-1] - 1) * 100
        ma20_ma50_ratio = (ma_20.iloc[-1] / ma_50.iloc[-1] - 1) * 100
        
        # تحلیل روند بر اساس موقعیت قیمت و میانگین‌های متحرک
        if price_ma20_ratio > 2 and ma20_ma50_ratio > 1:
            strength = min(1.0, (abs(price_ma20_ratio) + abs(ma20_ma50_ratio)) / 10)
            return "UPTREND", strength
        elif price_ma20_ratio < -2 and ma20_ma50_ratio < -1:
            strength = min(1.0, (abs(price_ma20_ratio) + abs(ma20_ma50_ratio)) / 10)
            return "DOWNTREND", strength
        else:
            # روند خنثی - محاسبه قدرت بر اساس نوسان
            volatility = data['close'].pct_change().std() * 100
            strength = min(0.3, volatility / 5)  # قدرت کم برای روند خنثی
            return "SIDEWAYS", strength
    
    def _calculate_trend_angle(self, data: pd.DataFrame) -> float:
        """محاسبه زاویه روند"""
        if len(data) < 20:
            return 0.0
        
        x = np.arange(len(data[-20:]))
        y = data['close'].values[-20:]
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            angle = np.degrees(np.arctan(slope / np.mean(y)))
            return angle
        except:
            return 0.0
    
    def _find_support_resistance(self, data: pd.DataFrame, level_type: str) -> List[float]:
        """پیدا کردن سطوح حمایت و مقاومت"""
        if level_type == 'support':
            price_series = data['low']
            window = 10
        else:  # resistance
            price_series = data['high']
            window = 10
        
        levels = []
        for i in range(window, len(data) - window):
            if level_type == 'support':
                if price_series.iloc[i] == price_series.iloc[i-window:i+window].min():
                    levels.append(price_series.iloc[i])
            else:
                if price_series.iloc[i] == price_series.iloc[i-window:i+window].max():
                    levels.append(price_series.iloc[i])
        
        return sorted(set(levels))
    
    def _find_breakout_level(self, data: pd.DataFrame, support: List[float], resistance: List[float]) -> float:
        """پیدا کردن سطح breakout بعدی"""
        current_price = data['close'].iloc[-1]
        
        if not resistance:
            return current_price * 1.05  # فرض 5% افزایش
        
        # نزدیک‌ترین سطح مقاومت بالاتر از قیمت فعلی
        above_resistance = [r for r in resistance if r > current_price]
        if above_resistance:
            return min(above_resistance)
        
        return current_price * 1.05
    
    def _analyze_volume_trend(self, data: pd.DataFrame) -> str:
        """تحلیل روند حجم"""
        if 'volume' not in data or len(data) < 20:
            return "NEUTRAL"
        
        volume_ma_20 = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma_20.iloc[-1]
        
        price_change = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100
        
        if volume_ratio > 1.2 and price_change > 1:
            return "BULLISH"
        elif volume_ratio > 1.2 and price_change < -1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """سیگنال RSI"""
        if rsi > 70:
            return "BEARISH"
        elif rsi < 30:
            return "BULLISH"
        else:
            return "NEUTRAL"

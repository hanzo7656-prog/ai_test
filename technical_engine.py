# technical_engine.py
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish" 
    RANGING = "ranging"
    VOLATILE = "volatile"
    TRANSITION = "transition"

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

@dataclass
class TechnicalSignal:
    symbol: str
    timeframe: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float  # 0-1
    confidence: float  # 0-1
    indicators: Dict[str, Any]
    entry_zones: List[float]
    stop_loss: float
    take_profit: List[float]
    risk_reward_ratio: float
    market_regime: MarketRegime
    timestamp: str

class SmartDataEngine:
    """لایه داده‌های هوشمند با پاک‌سازی خودکار"""
    
    def __init__(self):
        self.data_sources = ["internal", "external_api", "websocket"]
    
    def load_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """پاک‌سازی و نرمال‌سازی داده‌ها"""
        df_clean = df.copy()
        
        # حذف داده‌های نامعتبر
        df_clean = df_clean.dropna()
        
        # تشخیص و حذف outlierها با روش IQR
        Q1 = df_clean['close'].quantile(0.25)
        Q3 = df_clean['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean['close'] >= lower_bound) & 
                           (df_clean['close'] <= upper_bound)]
        
        # پر کردن شکاف‌های زمانی
        df_clean = self._fill_time_gaps(df_clean)
        
        # نرمال‌سازی حجم
        if 'volume' in df_clean.columns:
            df_clean['volume_sma'] = talib.SMA(df_clean['volume'], timeperiod=20)
            df_clean['volume_ratio'] = df_clean['volume'] / df_clean['volume_sma']
        
        return df_clean
    
    def _fill_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """پر کردن شکاف‌های زمانی در داده‌ها"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.resample('1min').last().ffill()
        return df.reset_index()

class MultiLayerComputationCore:
    """هسته محاسباتی چندلایه"""
    
    def __init__(self):
        self.classic_indicators = {}
        self.advanced_indicators = {}
        self.pattern_indicators = {}
        
    # لایه اول: اندیکاتورهای کلاسیک
    def calculate_classic_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای کلاسیک"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        
        indicators = {}
        
        # RSI تطبیقی
        rsi_periods = [6, 14, 21]
        for period in rsi_periods:
            indicators[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)
        
        # MACD چندزمانه
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['MACD'] = macd
        indicators['MACD_SIGNAL'] = macd_signal
        indicators['MACD_HIST'] = macd_hist
        
        # بولینگر باند تطبیقی
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        indicators['BB_UPPER'] = bb_upper
        indicators['BB_MIDDLE'] = bb_middle
        indicators['BB_LOWER'] = bb_lower
        
        # میانگین‌های متحرک
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
        indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # استوکاستیک
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        indicators['STOCH_K'] = slowk
        indicators['STOCH_D'] = slowd
        
        return indicators
    
    # لایه دوم: اندیکاتورهای پیشرفته
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای پیشرفته"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        
        advanced = {}
        
        # تشخیص رژیم بازار
        advanced['market_regime'] = self._detect_market_regime(df)
        
        # تحلیل پروفایل حجم
        advanced['volume_profile'] = self._analyze_volume_profile(df)
        
        # مومنتوم ترکیبی
        advanced['composite_momentum'] = self._calculate_composite_momentum(df)
        
        # نوسان ساز تطبیقی
        advanced['adaptive_oscillator'] = self._adaptive_oscillator(df)
        
        return advanced
    
    def _detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """تشخیص رژیم بازار"""
        close = df['close'].values
        
        # محاسبه ویژگی‌های مختلف
        sma_20 = talib.SMA(close, 20)
        sma_50 = talib.SMA(close, 50)
        adx = talib.ADX(df['high'].values, df['low'].values, close, timeperiod=14)
        atr = talib.ATR(df['high'].values, df['low'].values, close, timeperiod=14)
        
        # منطق تشخیص رژیم
        price_above_sma20 = close[-1] > sma_20[-1]
        price_above_sma50 = close[-1] > sma_50[-1]
        strong_trend = adx[-1] > 25
        high_volatility = (atr[-1] / close[-1]) > 0.02
        
        if strong_trend:
            if price_above_sma20 and price_above_sma50:
                return MarketRegime.BULLISH
            else:
                return MarketRegime.BEARISH
        elif high_volatility:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.RANGING
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """تحلیل پروفایل حجم"""
        if 'volume' not in df.columns:
            return {}
        
        volume = df['volume'].values
        close = df['close'].values
        
        # حجم نسبی
        volume_sma = talib.SMA(volume, 20)
        volume_ratio = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
        
        # تایید حجم در روند
        price_change = (close[-1] - close[-5]) / close[-5]
        volume_trend = volume_ratio > 1.2 and abs(price_change) > 0.01
        
        return {
            'volume_ratio': volume_ratio,
            'volume_trend_confirmation': volume_trend,
            'volume_surge': volume_ratio > 1.5
        }
    
    def _calculate_composite_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """محاسبه مومنتوم ترکیبی"""
        close = df['close'].values
        
        # ترکیب چندین اندیکاتور مومنتوم
        rsi = talib.RSI(close, 14)
        stoch_k, stoch_d = talib.STOCH(df['high'].values, df['low'].values, close)
        williams = talib.WILLR(df['high'].values, df['low'].values, close, timeperiod=14)
        cci = talib.CCI(df['high'].values, df['low'].values, close, timeperiod=20)
        
        # نرمال‌سازی و ترکیب
        momentum_score = (
            (70 - min(abs(rsi[-1] - 50), 20)) / 20 * 0.3 +
            (stoch_k[-1] / 100) * 0.25 +
            (100 - abs(williams[-1])) / 100 * 0.25 +
            (cci[-1] + 100) / 200 * 0.2
        )
        
        return {
            'composite_score': momentum_score,
            'rsi_momentum': (70 - abs(rsi[-1] - 50)) / 20,
            'stoch_momentum': stoch_k[-1] / 100,
            'trend_strength': momentum_score
        }
    
    def _adaptive_oscillator(self, df: pd.DataFrame) -> Dict[str, float]:
        """نوسان ساز تطبیقی"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # محاسبه نوسان
        volatility = talib.ATR(high, low, close, 14)[-1] / close[-1]
        
        # تنظیم دوره بر اساس نوسان
        adaptive_period = max(5, min(20, int(20 * (1 - volatility * 10))))
        
        # نوسان ساز با دوره تطبیقی
        oscillator = (close[-1] - talib.SMA(close, adaptive_period)[-1]) / close[-1]
        
        return {
            'adaptive_period': adaptive_period,
            'oscillator_value': oscillator,
            'volatility_adjusted': True
        }

class TrendAnalysisEngine:
    """سیستم تشخیص روند چندزمانه"""
    
    def __init__(self):
        self.timeframes = [tf.value for tf in TimeFrame]
    
    def analyze_trend_hierarchy(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """تحلیل سلسله مراتبی روند"""
        trend_analysis = {}
        
        for tf, df in multi_tf_data.items():
            if df.empty:
                continue
                
            trend_analysis[tf] = self._analyze_single_timeframe(df, tf)
        
        # ایجاد اجماع روند
        consensus = self._calculate_trend_consensus(trend_analysis)
        
        return {
            'timeframe_analysis': trend_analysis,
            'consensus': consensus,
            'primary_trend': consensus['primary_trend'],
            'trend_alignment': consensus['alignment_score']
        }
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """تحلیل روند در یک تایم‌فریم"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # اندیکاتورهای روند
        sma_20 = talib.SMA(close, 20)
        sma_50 = talib.SMA(close, 50)
        ema_12 = talib.EMA(close, 12)
        ema_26 = talib.EMA(close, 26)
        
        # قدرت روند
        adx = talib.ADX(high, low, close, 14)
        plus_di = talib.PLUS_DI(high, low, close, 14)
        minus_di = talib.MINUS_DI(high, low, close, 14)
        
        # تشخیص جهت روند
        price_above_sma20 = close[-1] > sma_20[-1] if not np.isnan(sma_20[-1]) else False
        price_above_sma50 = close[-1] > sma_50[-1] if not np.isnan(sma_50[-1]) else False
        ema_bullish = ema_12[-1] > ema_26[-1] if not np.isnan(ema_12[-1]) and not np.isnan(ema_26[-1]) else False
        di_bullish = plus_di[-1] > minus_di[-1] if not np.isnan(plus_di[-1]) and not np.isnan(minus_di[-1]) else False
        
        # محاسبه امتیاز روند
        trend_score = 0
        if price_above_sma20: trend_score += 0.25
        if price_above_sma50: trend_score += 0.25
        if ema_bullish: trend_score += 0.25
        if di_bullish: trend_score += 0.25
        
        # تعیین روند
        if trend_score >= 0.75:
            trend_direction = "BULLISH"
        elif trend_score <= 0.25:
            trend_direction = "BEARISH"
        else:
            trend_direction = "NEUTRAL"
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': adx[-1] if not np.isnan(adx[-1]) else 0,
            'trend_score': trend_score,
            'price_position': {
                'above_sma20': price_above_sma20,
                'above_sma50': price_above_sma50
            }
        }
    
    def _calculate_trend_consensus(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه اجماع روند بین تایم‌فریم‌ها"""
        if not trend_analysis:
            return {'primary_trend': 'NEUTRAL', 'alignment_score': 0}
        
        # وزن‌دهی تایم‌فریم‌ها
        timeframe_weights = {
            '1d': 0.35, '4h': 0.25, '1h': 0.20, 
            '15m': 0.15, '5m': 0.05
        }
        
        bullish_score = 0
        total_weight = 0
        
        for tf, analysis in trend_analysis.items():
            weight = timeframe_weights.get(tf, 0.1)
            
            if analysis['trend_direction'] == 'BULLISH':
                bullish_score += weight * analysis['trend_score']
            elif analysis['trend_direction'] == 'BEARISH':
                bullish_score += weight * (1 - analysis['trend_score'])
            else:  # NEUTRAL
                bullish_score += weight * 0.5
                
            total_weight += weight
        
        consensus_score = bullish_score / total_weight if total_weight > 0 else 0.5
        
        if consensus_score >= 0.6:
            primary_trend = "BULLISH"
        elif consensus_score <= 0.4:
            primary_trend = "BEARISH"
        else:
            primary_trend = "NEUTRAL"
        
        return {
            'primary_trend': primary_trend,
            'consensus_score': consensus_score,
            'alignment_score': self._calculate_alignment_score(trend_analysis),
            'timeframes_analyzed': len(trend_analysis)
        }
    
    def _calculate_alignment_score(self, trend_analysis: Dict[str, Any]) -> float:
        """محاسبه میزان همسویی تایم‌فریم‌ها"""
        directions = []
        for analysis in trend_analysis.values():
            if analysis['trend_direction'] == 'BULLISH':
                directions.append(1)
            elif analysis['trend_direction'] == 'BEARISH':
                directions.append(-1)
            else:
                directions.append(0)
        
        if not directions:
            return 0
        
        # هرچه واریانس کمتر باشد، همسویی بیشتر است
        variance = np.var(directions)
        alignment = 1 - min(variance, 1)  # نرمال‌سازی به 0-1
        
        return alignment

class IntelligentSignalEngine:
    """موتور سیگنال‌دهی هوشمند"""
    
    def __init__(self):
        self.signal_weights = {
            'momentum': 0.25,
            'trend': 0.30,
            'volume': 0.15,
            'volatility': 0.10,
            'pattern': 0.20
        }
    
    def generate_signals(self, symbol: str, df: pd.DataFrame, 
                        multi_tf_analysis: Dict[str, Any],
                        classic_indicators: Dict[str, Any],
                        advanced_indicators: Dict[str, Any]) -> TechnicalSignal:
        """تولید سیگنال‌های هوشمند"""
        
        # محاسبه امتیازهای جزئی
        momentum_score = self._calculate_momentum_score(classic_indicators, advanced_indicators)
        trend_score = self._calculate_trend_strength(multi_tf_analysis)
        volume_score = self._analyze_volume_confirmation(advanced_indicators)
        volatility_score = self._assess_volatility_conditions(df, classic_indicators)
        pattern_score = self._pattern_recognition_score(df, classic_indicators)
        
        # ترکیب وزنی سیگنال‌ها
        composite_score = (
            momentum_score * self.signal_weights['momentum'] +
            trend_score * self.signal_weights['trend'] +
            volume_score * self.signal_weights['volume'] +
            volatility_score * self.signal_weights['volatility'] +
            pattern_score * self.signal_weights['pattern']
        )
        
        # تولید سیگنال نهایی
        signal = self._generate_final_signal(
            symbol, df, composite_score, multi_tf_analysis, 
            classic_indicators, advanced_indicators
        )
        
        return signal
    
    def _calculate_momentum_score(self, classic_indicators: Dict[str, Any], 
                                advanced_indicators: Dict[str, Any]) -> float:
        """محاسبه امتیاز مومنتوم"""
        momentum_indicators = []
        
        # RSI momentum
        rsi_values = [v for k, v in classic_indicators.items() if k.startswith('RSI')]
        if rsi_values:
            latest_rsi = rsi_values[0][-1] if not np.isnan(rsi_values[0][-1]) else 50
            rsi_momentum = 1 - abs(latest_rsi - 50) / 50
            momentum_indicators.append(rsi_momentum)
        
        # MACD momentum
        if 'MACD_HIST' in classic_indicators:
            macd_hist = classic_indicators['MACD_HIST'][-1]
            if not np.isnan(macd_hist):
                macd_momentum = min(abs(macd_hist) * 10, 1)
                momentum_indicators.append(macd_momentum)
        
        # Stochastic momentum
        if 'STOCH_K' in classic_indicators:
            stoch_k = classic_indicators['STOCH_K'][-1]
            if not np.isnan(stoch_k):
                stoch_momentum = 1 - abs(stoch_k - 50) / 50
                momentum_indicators.append(stoch_momentum)
        
        # Composite momentum from advanced indicators
        if 'composite_momentum' in advanced_indicators:
            comp_momentum = advanced_indicators['composite_momentum'].get('composite_score', 0.5)
            momentum_indicators.append(comp_momentum)
        
        return np.mean(momentum_indicators) if momentum_indicators else 0.5
    
    def _calculate_trend_strength(self, multi_tf_analysis: Dict[str, Any]) -> float:
        """محاسبه قدرت روند"""
        consensus = multi_tf_analysis.get('consensus', {})
        alignment_score = consensus.get('alignment_score', 0)
        consensus_score = consensus.get('consensus_score', 0.5)
        
        # ترکیب همسویی و اجماع
        trend_strength = (alignment_score + abs(consensus_score - 0.5) * 2) / 2
        return trend_strength
    
    def _analyze_volume_confirmation(self, advanced_indicators: Dict[str, Any]) -> float:
        """تحلیل تایید حجم"""
        volume_profile = advanced_indicators.get('volume_profile', {})
        volume_ratio = volume_profile.get('volume_ratio', 1)
        volume_trend = volume_profile.get('volume_trend_confirmation', False)
        
        # امتیاز بر اساس نسبت حجم و تایید روند
        volume_score = min(volume_ratio / 2, 1)  # نرمال‌سازی
        if volume_trend:
            volume_score = min(volume_score + 0.2, 1)
        
        return volume_score
    
    def _assess_volatility_conditions(self, df: pd.DataFrame, 
                                    classic_indicators: Dict[str, Any]) -> float:
        """ارزیابی شرایط نوسان"""
        close = df['close'].values
        
        # محاسبه نوسان
        if 'BB_UPPER' in classic_indicators and 'BB_LOWER' in classic_indicators:
            bb_upper = classic_indicators['BB_UPPER'][-1]
            bb_lower = classic_indicators['BB_LOWER'][-1]
            bb_middle = classic_indicators['BB_MIDDLE'][-1]
            
            if not np.isnan(bb_upper) and not np.isnan(bb_lower) and not np.isnan(bb_middle):
                bb_width = (bb_upper - bb_lower) / bb_middle
                # نوسان مطلوب: نه خیلی کم، نه خیلی زیاد
                optimal_volatility = 1 - min(abs(bb_width - 0.04) / 0.04, 1)
                return optimal_volatility
        
        return 0.5
    
    def _pattern_recognition_score(self, df: pd.DataFrame, 
                                 classic_indicators: Dict[str, Any]) -> float:
        """امتیاز تشخیص الگو"""
        # در این نسخه ساده شده - در نسخه کامل از ML استفاده می‌شود
        close = df['close'].values
        
        # تشخیص ساده الگوهای قیمت
        price_trend = self._simple_price_pattern(close)
        support_resistance = self._detect_support_resistance(df)
        
        pattern_score = (price_trend + support_resistance) / 2
        return pattern_score
    
    def _simple_price_pattern(self, close: np.ndarray) -> float:
        """تشخیص الگوی ساده قیمت"""
        if len(close) < 10:
            return 0.5
        
        # بررسی روند کوتاه مدت
        short_trend = np.polyfit(range(5), close[-5:], 1)[0]
        # بررسی روند بلندمدت
        long_trend = np.polyfit(range(10), close[-10:], 1)[0]
        
        # همسویی روندها
        trend_alignment = 1 - abs(short_trend - long_trend) / max(abs(long_trend), 0.001)
        return max(0, min(1, trend_alignment))
    
    def _detect_support_resistance(self, df: pd.DataFrame) -> float:
        """تشخیص سطوح حمایت و مقاومت"""
        high = df['high'].values
        low = df['low'].values
        
        # استفاده از بولینگر باند به عنوان سطوح دینامیک
        if len(high) > 20:
            resistance_level = np.mean(high[-20:]) + np.std(high[-20:])
            support_level = np.mean(low[-20:]) - np.std(low[-20:])
            current_price = df['close'].iloc[-1]
            
            # فاصله از سطوح کلیدی
            distance_to_resistance = abs(current_price - resistance_level) / current_price
            distance_to_support = abs(current_price - support_level) / current_price
            
            # امتیاز بر اساس نزدیکی به سطوح
            level_score = 1 - min(distance_to_resistance, distance_to_support) / 0.1
            return max(0, min(1, level_score))
        
        return 0.5
    
    def _generate_final_signal(self, symbol: str, df: pd.DataFrame, 
                             composite_score: float,
                             multi_tf_analysis: Dict[str, Any],
                             classic_indicators: Dict[str, Any],
                             advanced_indicators: Dict[str, Any]) -> TechnicalSignal:
        """تولید سیگنال نهایی"""
        
        current_price = df['close'].iloc[-1]
        market_regime = advanced_indicators.get('market_regime', MarketRegime.RANGING)
        
        # تعیین نوع سیگنال بر اساس امتیاز ترکیبی
        if composite_score >= 0.7:
            signal_type = "BUY"
            strength = composite_score
        elif composite_score <= 0.3:
            signal_type = "SELL" 
            strength = 1 - composite_score
        else:
            signal_type = "HOLD"
            strength = 0.5
        
        # محاسبه سطوح ورود، استاپ و تیک پروفیت
        entry_zones, stop_loss, take_profit = self._calculate_trading_levels(
            df, signal_type, classic_indicators
        )
        
        # محاسبه نسبت ریسک به پاداش
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            current_price, entry_zones, stop_loss, take_profit
        )
        
        return TechnicalSignal(
            symbol=symbol,
            timeframe="1h",  # می‌تواند پویا باشد
            signal_type=signal_type,
            strength=strength,
            confidence=composite_score,
            indicators={
                'composite_score': composite_score,
                'market_regime': market_regime,
                'trend_alignment': multi_tf_analysis.get('consensus', {}).get('alignment_score', 0)
            },
            entry_zones=entry_zones,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            market_regime=market_regime,
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def _calculate_trading_levels(self, df: pd.DataFrame, signal_type: str,
                                classic_indicators: Dict[str, Any]) -> Tuple[List[float], float, List[float]]:
        """محاسبه سطوح معاملاتی"""
        current_price = df['close'].iloc[-1]
        
        if 'BB_LOWER' in classic_indicators and 'BB_UPPER' in classic_indicators:
            bb_lower = classic_indicators['BB_LOWER'][-1]
            bb_upper = classic_indicators['BB_UPPER'][-1]
            bb_middle = classic_indicators['BB_MIDDLE'][-1]
        else:
            # مقادیر پیش‌فرض اگر بولینگر موجود نباشد
            bb_middle = current_price
            bb_lower = current_price * 0.98
            bb_upper = current_price * 1.02
        
        if signal_type == "BUY":
            entry_zones = [bb_lower, bb_middle]
            stop_loss = bb_lower * 0.99  # 1% below lower band
            take_profit = [bb_upper, bb_upper * 1.02]
        elif signal_type == "SELL":
            entry_zones = [bb_upper, bb_middle] 
            stop_loss = bb_upper * 1.01  # 1% above upper band
            take_profit = [bb_lower, bb_lower * 0.98]
        else:  # HOLD
            entry_zones = [current_price]
            stop_loss = current_price * 0.99
            take_profit = [current_price * 1.01]
        
        return entry_zones, stop_loss, take_profit
    
    def _calculate_risk_reward_ratio(self, current_price: float, entry_zones: List[float],
                                   stop_loss: float, take_profit: List[float]) -> float:
        """محاسبه نسبت ریسک به پاداش"""
        if not entry_zones or not take_profit:
            return 1.0
        
        avg_entry = np.mean(entry_zones)
        avg_take_profit = np.mean(take_profit)
        
        risk = abs(avg_entry - stop_loss)
        reward = abs(avg_take_profit - avg_entry)
        
        if risk > 0:
            return reward / risk
        else:
            return 1.0

class AdvancedTechnicalEngine:
    """موتور تحلیل تکنیکال پیشرفته - کلاس اصلی"""
    
    def __init__(self):
        self.data_engine = SmartDataEngine()
        self.computation_core = MultiLayerComputationCore()
        self.trend_engine = TrendAnalysisEngine()
        self.signal_engine = IntelligentSignalEngine()
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame, 
                      multi_tf_data: Dict[str, pd.DataFrame] = None) -> TechnicalSignal:
        """آنالیز کامل یک نماد"""
        
        # پاک‌سازی داده‌ها
        df_clean = self.data_engine.load_and_clean_data(df)
        
        # محاسبه اندیکاتورهای کلاسیک
        classic_indicators = self.computation_core.calculate_classic_indicators(df_clean)
        
        # محاسبه اندیکاتورهای پیشرفته
        advanced_indicators = self.computation_core.calculate_advanced_indicators(df_clean)
        
        # تحلیل روند چندزمانه
        if multi_tf_data:
            multi_tf_analysis = self.trend_engine.analyze_trend_hierarchy(multi_tf_data)
        else:
            # استفاده از داده فعلی اگر داده چندزمانه موجود نباشد
            multi_tf_analysis = self.trend_engine.analyze_trend_hierarchy({'current': df_clean})
        
        # تولید سیگنال هوشمند
        signal = self.signal_engine.generate_signals(
            symbol, df_clean, multi_tf_analysis, 
            classic_indicators, advanced_indicators
        )
        
        return signal
    
    def batch_analyze(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, TechnicalSignal]:
        """آنالیز دسته‌ای چندین نماد"""
        results = {}
        
        for symbol, df in symbols_data.items():
            try:
                signal = self.analyze_symbol(symbol, df)
                results[symbol] = signal
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return results

# نمونه استفاده
if __name__ == "__main__":
    # ایجاد موتور
    engine = AdvancedTechnicalEngine()
    
    # نمونه داده تست (در عمل از API یا دیتابیس می‌آید)
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(50500, 1000, 100),
        'low': np.random.normal(49500, 1000, 100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    
    # آنالیز نمونه
    signal = engine.analyze_symbol("BTC/USDT", sample_data)
    
    print("🔍 نتایج تحلیل تکنیکال:")
    print(f"نماد: {signal.symbol}")
    print(f"سیگنال: {signal.signal_type}")
    print(f"قدرت سیگنال: {signal.strength:.2f}")
    print(f"اعتماد: {signal.confidence:.2f}")
    print(f"رژیم بازار: {signal.market_regime.value}")
    print(f"نسبت ریسک/پاداش: {signal.risk_reward_ratio:.2f}")
    print(f"مناطق ورود: {signal.entry_zones}")
    print(f"استاپ لاس: {signal.stop_loss:.2f}")
    print(f"تیک پروفیت: {signal.take_profit}")

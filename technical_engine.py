# technical_engine_complete.py
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish" 
    RANGING = "ranging"
    VOLATILE = "volatile"

class PatternType(Enum):
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"

@dataclass
class TechnicalSignal:
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float  # 0-1
    confidence: float  # 0-1
    indicators: Dict[str, Any]
    patterns: List[PatternType]
    market_regime: MarketRegime
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: str

class CompleteTechnicalEngine:
    """موتور تحلیل تکنیکال کامل با تمام اندیکاتورهای مهم"""
    
    def __init__(self):
        self.memory_limit = 120  # MB
        print("🚀 Complete Technical Engine Initialized")
    
    def calculate_all_indicators(self, ohlc_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """محاسبه تمام اندیکاتورهای مهم"""
        o = np.array(ohlc_data['open'], dtype=np.float32)
        h = np.array(ohlc_data['high'], dtype=np.float32)
        l = np.array(ohlc_data['low'], dtype=np.float32)
        c = np.array(ohlc_data['close'], dtype=np.float32)
        v = np.array(ohlc_data.get('volume', []), dtype=np.float32)
        
        if len(c) < 50:
            return self._get_default_indicators()
        
        indicators = {}
        
        # 🎯 اندیکاتورهای روند
        indicators.update(self._trend_indicators(c))
        
        # 📊 اندیکاتورهای مومنتوم
        indicators.update(self._momentum_indicators(h, l, c))
        
        # 🌊 اندیکاتورهای نوسان
        indicators.update(self._volatility_indicators(h, l, c))
        
        # 📦 اندیکاتورهای حجم
        if len(v) > 0:
            indicators.update(self._volume_indicators(c, v))
        
        # 🔮 ایچیموکو
        indicators.update(self._ichimoku_cloud(h, l, c))
        
        return indicators
    
    def _trend_indicators(self, close: np.ndarray) -> Dict[str, float]:
        """اندیکاتورهای روند"""
        trend = {}
        
        # میانگین‌های متحرک
        trend['sma_20'] = self._sma(close, 20)
        trend['sma_50'] = self._sma(close, 50)
        trend['sma_200'] = self._sma(close, 200)
        trend['ema_12'] = self._ema(close, 12)
        trend['ema_26'] = self._ema(close, 26)
        
        # MACD
        macd_line, signal_line, histogram = self._macd(close)
        trend['macd'] = macd_line
        trend['macd_signal'] = signal_line
        trend['macd_histogram'] = histogram
        
        # پارابولیک سار
        trend['sar'] = self._parabolic_sar(close)
        
        return trend
    
    def _momentum_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """اندیکاتورهای مومنتوم"""
        momentum = {}
        
        # RSI
        momentum['rsi'] = self._rsi(close)
        momentum['rsi_7'] = self._rsi(close, 7)
        
        # استوکاستیک
        stoch_k, stoch_d = self._stochastic(high, low, close)
        momentum['stoch_k'] = stoch_k
        momentum['stoch_d'] = stoch_d
        
        # CCI
        momentum['cci'] = self._cci(high, low, close)
        
        # Williams %R
        momentum['williams_r'] = self._williams_r(high, low, close)
        
        # مومنتوم
        momentum['momentum_10'] = self._momentum(close, 10)
        
        # نرخ تغییر (ROC)
        momentum['roc'] = self._roc(close)
        
        return momentum
    
    def _volatility_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """اندیکاتورهای نوسان"""
        volatility = {}
        
        # باندهای بولینگر
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close)
        volatility['bb_upper'] = bb_upper
        volatility['bb_middle'] = bb_middle
        volatility['bb_lower'] = bb_lower
        volatility['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR
        volatility['atr'] = self._atr(high, low, close)
        
        # انحراف معیار
        volatility['std_dev'] = np.std(close[-20:])
        
        return volatility
    
    def _volume_indicators(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """اندیکاتورهای حجم"""
        vol_indicators = {}
        
        # حجم متحرک
        vol_indicators['volume_sma'] = self._sma(volume, 20)
        vol_indicators['volume_ratio'] = volume[-1] / vol_indicators['volume_sma']
        
        # OBV
        vol_indicators['obv'] = self._obv(close, volume)
        
        # شاخص جریان پول (MFI)
        vol_indicators['mfi'] = self._mfi(high, low, close, volume)
        
        return vol_indicators
    
    def _ichimoku_cloud(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """ایچیموکو کلود"""
        ichimoku = {}
        
        # تنکان سن
        ichimoku['tenkan_sen'] = (np.max(high[-9:]) + np.min(low[-9:])) / 2
        
        # کیجون سن
        ichimoku['kijun_sen'] = (np.max(high[-26:]) + np.min(low[-26:])) / 2
        
        # سنکو اسپان A
        ichimoku['senkou_span_a'] = (ichimoku['tenkan_sen'] + ichimoku['kijun_sen']) / 2
        
        # سنکو اسپان B
        ichimoku['senkou_span_b'] = (np.max(high[-52:]) + np.min(low[-52:])) / 2
        
        # چیکو اسپان
        ichimoku['chikou_span'] = close[-26]  # 26 دوره قبل
        
        return ichimoku
    
    def detect_candlestick_patterns(self, ohlc_data: Dict[str, List[float]]) -> List[PatternType]:
        """تشخیص الگوهای کندل‌استیک"""
        patterns = []
        o = ohlc_data['open']
        h = ohlc_data['high']
        l = ohlc_data['low']
        c = ohlc_data['close']
        
        if len(c) < 3:
            return patterns
        
        # الگوهای تک‌کندلی
        if self._is_hammer(o, h, l, c):
            patterns.append(PatternType.HAMMER)
        if self._is_shooting_star(o, h, l, c):
            patterns.append(PatternType.SHOOTING_STAR)
        if self._is_doji(o, h, l, c):
            patterns.append(PatternType.DOJI)
        
        # الگوهای چندکندلی
        if self._is_engulfing(o, h, l, c):
            if c[-1] > o[-1] and c[-2] < o[-2]:  # صعودی
                patterns.append(PatternType.BULLISH_ENGULFING)
            elif c[-1] < o[-1] and c[-2] > o[-2]:  # نزولی
                patterns.append(PatternType.BEARISH_ENGULFING)
        
        # الگوهای نموداری
        if self._is_double_top(h, l, c):
            patterns.append(PatternType.DOUBLE_TOP)
        if self._is_double_bottom(h, l, c):
            patterns.append(PatternType.DOUBLE_BOTTOM)
        
        return patterns
    
    def calculate_support_resistance(self, ohlc_data: Dict[str, List[float]]) -> Tuple[List[float], List[float]]:
        """محاسبه سطوح حمایت و مقاومت"""
        h = ohlc_data['high']
        l = ohlc_data['low']
        c = ohlc_data['close']
        
        if len(c) < 20:
            return [], []
        
        # سطوح مقاومت (قله‌ها)
        resistance_levels = self._find_pivot_highs(h, 5)
        
        # سطوح حمایت (دره‌ها)
        support_levels = self._find_pivot_lows(l, 5)
        
        # فیلتر سطوح نزدیک
        current_price = c[-1]
        support_levels = [s for s in support_levels if s < current_price * 0.98]
        resistance_levels = [r for r in resistance_levels if r > current_price * 1.02]
        
        return support_levels[:3], resistance_levels[:3]  # 3 سطح اصلی
    
    # 🛠️ پیاده‌سازی توابع محاسباتی
    def _sma(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(np.mean(data)) if len(data) > 0 else 0.0
        return float(np.mean(data[-period:]))
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return float(np.dot(data[-period:], weights))
    
    def _macd(self, close: np.ndarray) -> Tuple[float, float, float]:
        ema_12 = self._ema(close, 12)
        ema_26 = self._ema(close, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._ema(np.array([macd_line]), 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _rsi(self, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = self._sma(gains[-period:], period)
        avg_loss = self._sma(losses[-period:], period)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float]:
        if len(close) < period:
            return 50.0, 50.0
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        if highest_high == lowest_low:
            return 50.0, 50.0
        k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        d = self._sma(np.array([k]), 3)  # SMA از %K
        return k, d
    
    def _bollinger_bands(self, close: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        if len(close) < period:
            current = close[-1] if len(close) > 0 else 0
            return current, current, current
        sma = self._sma(close, period)
        std = np.std(close[-period:])
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 0.0
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        return float(np.mean(tr[-period:]))
    
    # 🔍 تشخیص الگوهای کندل‌استیک
    def _is_hammer(self, o: List[float], h: List[float], l: List[float], c: List[float]) -> bool:
        if len(c) < 1: return False
        body = abs(c[-1] - o[-1])
        lower_wick = min(o[-1], c[-1]) - l[-1]
        upper_wick = h[-1] - max(o[-1], c[-1])
        return lower_wick >= 2 * body and upper_wick <= body * 0.5
    
    def _is_shooting_star(self, o: List[float], h: List[float], l: List[float], c: List[float]) -> bool:
        if len(c) < 1: return False
        body = abs(c[-1] - o[-1])
        upper_wick = h[-1] - max(o[-1], c[-1])
        lower_wick = min(o[-1], c[-1]) - l[-1]
        return upper_wick >= 2 * body and lower_wick <= body * 0.5
    
    def _is_doji(self, o: List[float], h: List[float], l: List[float], c: List[float]) -> bool:
        if len(c) < 1: return False
        body = abs(c[-1] - o[-1])
        total_range = h[-1] - l[-1]
        return body <= total_range * 0.1  # بدنه بسیار کوچک
    
    def _is_engulfing(self, o: List[float], h: List[float], l: List[float], c: List[float]) -> bool:
        if len(c) < 2: return False
        body_today = abs(c[-1] - o[-1])
        body_yesterday = abs(c[-2] - o[-2])
        return body_today > body_yesterday * 1.5
    
    def generate_complete_signal(self, symbol: str, ohlc_data: Dict[str, List[float]]) -> TechnicalSignal:
        """تولید سیگنال کامل با تمام آنالیزها"""
        try:
            # محاسبه تمام اندیکاتورها
            indicators = self.calculate_all_indicators(ohlc_data)
            
            # تشخیص الگوهای کندل‌استیک
            patterns = self.detect_candlestick_patterns(ohlc_data)
            
            # محاسبه سطوح حمایت و مقاومت
            support_levels, resistance_levels = self.calculate_support_resistance(ohlc_data)
            
            # تشخیص رژیم بازار
            market_regime = self.analyze_market_regime(ohlc_data)
            
            # تولید سیگنال
            signal_type, strength, confidence = self._calculate_advanced_signal_score(
                indicators, patterns, market_regime, ohlc_data
            )
            
            return TechnicalSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                indicators=indicators,
                patterns=patterns,
                market_regime=market_regime,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                timestamp=self._get_timestamp()
            )
            
        except Exception as e:
            print(f"Error in complete analysis: {e}")
            return self._get_default_signal(symbol)
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_default_signal(self, symbol: str) -> TechnicalSignal:
        return TechnicalSignal(
            symbol=symbol,
            signal_type="HOLD",
            strength=0.5,
            confidence=0.5,
            indicators={},
            patterns=[],
            market_regime=MarketRegime.RANGING,
            support_levels=[],
            resistance_levels=[],
            timestamp=self._get_timestamp()
        )

# تست عملکرد
if __name__ == "__main__":
    engine = CompleteTechnicalEngine()
    
    # داده‌های تست
    test_data = {
        'open': [50000 + i * 10 for i in range(100)],
        'high': [50200 + i * 10 for i in range(100)],
        'low': [49800 + i * 10 for i in range(100)],
        'close': [50100 + i * 10 for i in range(100)],
        'volume': [1000000 + i * 50000 for i in range(100)]
    }
    
    signal = engine.generate_complete_signal("BTC", test_data)
    
    print(f"🎯 Signal: {signal.signal_type}")
    print(f"💪 Strength: {signal.strength:.2f}")
    print(f"📊 RSI: {signal.indicators.get('rsi', 0):.1f}")
    print(f"📈 MACD: {signal.indicators.get('macd', 0):.3f}")
    print(f"🔍 Patterns: {[p.value for p in signal.patterns]}")
    print(f"🛡️ Support: {signal.support_levels}")
    print(f"🚧 Resistance: {signal.resistance_levels}")

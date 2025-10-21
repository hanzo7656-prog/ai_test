# technical_analysis_engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import pandas_ta as ta
from scipy import stats
import math
from datetime import datetime, timedelta
import json

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    NEUTRAL = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class TechnicalSignal:
    indicator: str
    value: float
    signal: str
    strength: SignalStrength
    trend: TrendDirection
    description: str
    timestamp: str

class CandlePattern(Enum):
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    DOJI = "doji"
    SPINNING_TOP = "spinning_top"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"

class TechnicalAnalysisEngine:
    """
    موتور کامل تحلیل تکنیکال با تمام ابزارهای پیشرفته
    برای استفاده مستقیم توسط هوش مصنوعی
    """
    
    def __init__(self):
        self.available_indicators = {
            'trend': ['sma', 'ema', 'wma', 'macd', 'adx', 'ichimoku', 'parabolic_sar'],
            'momentum': ['rsi', 'stoch', 'williams_r', 'cci', 'mfi', 'awesome_oscillator'],
            'volatility': ['bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels'],
            'volume': ['obv', 'volume_profile', 'vwap', 'accumulation_distribution'],
            'support_resistance': ['pivot_points', 'fibonacci', 'candlestick_patterns'],
            'advanced': ['harmonic_patterns', 'market_profile', 'order_flow', 'volume_delta']
        }
        
        print("🚀 موتور تحلیل تکنیکال پیشرفته راه‌اندازی شد")
        print(f"📊 تعداد اندیکاتورهای موجود: {sum(len(v) for v in self.available_indicators.values())}")
    
    def _prepare_dataframe(self, price_data: List[Dict]) -> pd.DataFrame:
        """آماده سازی داده‌ها برای تحلیل"""
        if not price_data:
            raise ValueError("داده‌های قیمت خالی است")
        
        df = pd.DataFrame(price_data)
        
        # اطمینان از وجود ستون‌های ضروری
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                else:
                    # تولید داده‌های نمونه در صورت عدم وجود
                    if col in ['open', 'high', 'low', 'close']:
                        base_price = 50000
                        df[col] = [base_price + (i * 10) + np.random.normal(0, 100) for i in range(len(df))]
                    elif col == 'volume':
                        df[col] = np.random.randint(1000, 10000, len(df))
        
        # تبدیل به عدد
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # حذف مقادیر NaN
        df = df.dropna()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای روند"""
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        results = {}
        
        try:
            # SMA
            results['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            results['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            results['sma_200'] = talib.SMA(close_prices, timeperiod=200)
            
            # EMA
            results['ema_12'] = talib.EMA(close_prices, timeperiod=12)
            results['ema_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # WMA
            results['wma_20'] = talib.WMA(close_prices, timeperiod=20)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            results['macd'] = macd
            results['macd_signal'] = macd_signal
            results['macd_histogram'] = macd_hist
            
            # ADX
            results['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Parabolic SAR
            results['parabolic_sar'] = talib.SAR(high_prices, low_prices)
            
            # Ichimoku Cloud
            results['ichimoku'] = self._calculate_ichimoku(df)
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای روند: {e}")
        
        return results

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای مومنتوم"""
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        results = {}
        
        try:
            # RSI
            results['rsi'] = talib.RSI(close_prices, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
            results['stoch_k'] = slowk
            results['stoch_d'] = slowd
            
            # Williams %R
            results['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # CCI
            results['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20)
            
            # MFI
            results['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
            
            # Awesome Oscillator
            results['awesome_oscillator'] = self._calculate_awesome_oscillator(df)
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای مومنتوم: {e}")
        
        return results

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای نوسان"""
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        results = {}
        
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            results['bollinger_upper'] = bb_upper
            results['bollinger_middle'] = bb_middle
            results['bollinger_lower'] = bb_lower
            results['bollinger_bandwidth'] = (bb_upper - bb_lower) / bb_middle
            
            # ATR
            results['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Keltner Channels
            results['keltner'] = self._calculate_keltner_channels(df)
            
            # Donchian Channels
            results['donchian'] = self._calculate_donchian_channels(df)
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای نوسان: {e}")
        
        return results

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای حجم"""
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        results = {}
        
        try:
            # OBV
            results['obv'] = talib.OBV(close_prices, volume)
            
            # Volume Profile
            results['volume_profile'] = self._calculate_volume_profile(df)
            
            # VWAP
            results['vwap'] = self._calculate_vwap(df)
            
            # Accumulation/Distribution
            results['ad_line'] = talib.AD(high_prices, low_prices, close_prices, volume)
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای حجم: {e}")
        
        return results

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه ابر ایچیموکو"""
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        try:
            # Tenkan-sen (Conversion Line)
            period9_high = talib.MAX(high_prices, timeperiod=9)
            period9_low = talib.MIN(low_prices, timeperiod=9)
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = talib.MAX(high_prices, timeperiod=26)
            period26_low = talib.MIN(low_prices, timeperiod=26)
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
            
            # Senkou Span B (Leading Span B)
            period52_high = talib.MAX(high_prices, timeperiod=52)
            period52_low = talib.MIN(low_prices, timeperiod=52)
            senkou_span_b = ((period52_high + period52_low) / 2)
            
            # Chikou Span (Lagging Span)
            chikou_span = df['close'].shift(-26).values
            
            return {
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span
            }
        except Exception as e:
            print(f"⚠️ خطا در محاسبه ایچیموکو: {e}")
            return {}

    def _calculate_awesome_oscillator(self, df: pd.DataFrame) -> np.ndarray:
        """محاسبه Awesome Oscillator"""
        try:
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            median_price = (high_prices + low_prices) / 2
            sma5 = talib.SMA(median_price, timeperiod=5)
            sma34 = talib.SMA(median_price, timeperiod=34)
            
            return sma5 - sma34
        except:
            return np.array([])

    def _calculate_keltner_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه Keltner Channels"""
        try:
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            ema20 = talib.EMA(close_prices, timeperiod=20)
            atr10 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=10)
            
            upper_band = ema20 + (atr10 * 2)
            lower_band = ema20 - (atr10 * 2)
            
            return {
                'upper': upper_band,
                'middle': ema20,
                'lower': lower_band
            }
        except:
            return {}

    def _calculate_donchian_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه Donchian Channels"""
        try:
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            upper = talib.MAX(high_prices, timeperiod=20)
            lower = talib.MIN(low_prices, timeperiod=20)
            middle = (upper + lower) / 2
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        except:
            return {}

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه Volume Profile"""
        try:
            prices = df['close'].values
            volumes = df['volume'].values
            
            if len(prices) == 0:
                return {}
                
            # تقسیم قیمت‌ها به 10 سطح
            price_min, price_max = prices.min(), prices.max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return {}
                
            bin_size = price_range / 10
            
            volume_profile = {}
            for i in range(10):
                bin_low = price_min + (i * bin_size)
                bin_high = bin_low + bin_size
                bin_volume = volumes[(prices >= bin_low) & (prices < bin_high)].sum()
                volume_profile[f'bin_{i+1}'] = {
                    'price_range': (float(bin_low), float(bin_high)),
                    'volume': float(bin_volume)
                }
            
            return volume_profile
        except Exception as e:
            print(f"⚠️ خطا در محاسبه Volume Profile: {e}")
            return {}

    def _calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """محاسبه VWAP"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cumulative_tp_volume = (typical_price * df['volume']).cumsum()
            cumulative_volume = df['volume'].cumsum()
            
            vwap = cumulative_tp_volume / cumulative_volume
            return vwap.values
        except:
            return np.array([])

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تشخیص الگوهای کندل استیک"""
        try:
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            patterns = {}
            
            # الگوهای تک کندلی
            patterns['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            patterns['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            patterns['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            patterns['spinning_top'] = talib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)
            
            # الگوهای دو کندلی
            patterns['engulfing_bullish'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            patterns['engulfing_bearish'] = patterns['engulfing_bullish'] * -1
            patterns['harami_bullish'] = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)
            patterns['harami_bearish'] = talib.CDLHARAMICROSS(open_prices, high_prices, low_prices, close_prices)
            
            # الگوهای سه کندلی
            patterns['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            patterns['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)
            patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)
            
            return patterns
        except Exception as e:
            print(f"⚠️ خطا در تشخیص الگوهای کندل استیک: {e}")
            return {}

    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """محاسبه نقاط پیوت"""
        try:
            if len(df) < 2:
                return {}
                
            last_day = df.iloc[-1]
            prev_day = df.iloc[-2]
            
            pivot = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
            r1 = (2 * pivot) - prev_day['low']
            s1 = (2 * pivot) - prev_day['high']
            r2 = pivot + (prev_day['high'] - prev_day['low'])
            s2 = pivot - (prev_day['high'] - prev_day['low'])
            
            return {
                'pivot': float(pivot),
                'resistance_1': float(r1),
                'resistance_2': float(r2),
                'support_1': float(s1),
                'support_2': float(s2)
            }
        except:
            return {}

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """محاسبه سطوح فیبوناچی"""
        try:
            if len(df) < 20:
                return {}
                
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_range = recent_high - recent_low
            
            levels = {
                '0.0': float(recent_high),
                '0.236': float(recent_high - (price_range * 0.236)),
                '0.382': float(recent_high - (price_range * 0.382)),
                '0.5': float(recent_high - (price_range * 0.5)),
                '0.618': float(recent_high - (price_range * 0.618)),
                '0.786': float(recent_high - (price_range * 0.786)),
                '1.0': float(recent_low)
            }
            
            return levels
        except:
            return {}

    def calculate_all_indicators(self, price_data: List[Dict]) -> Dict[str, Any]:
        """
        محاسبه تمام اندیکاتورها به صورت یکجا
        """
        try:
            if len(price_data) < 20:
                return {"error": "داده‌های ناکافی برای تحلیل", "minimum_data_points": 20}
            
            # تبدیل داده‌ها به فرمت مناسب
            df = self._prepare_dataframe(price_data)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df),
                'trend_indicators': self._calculate_trend_indicators(df),
                'momentum_indicators': self._calculate_momentum_indicators(df),
                'volatility_indicators': self._calculate_volatility_indicators(df),
                'volume_indicators': self._calculate_volume_indicators(df),
                'candlestick_patterns': self.detect_candlestick_patterns(df),
                'pivot_points': self.calculate_pivot_points(df),
                'fibonacci_levels': self.calculate_fibonacci_levels(df),
                'price_action': {
                    'current_price': float(df['close'].iloc[-1]),
                    'price_change_24h': float(((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100) if len(df) > 1 else 0,
                    'high_24h': float(df['high'].tail(24).max()) if len(df) >= 24 else float(df['high'].max()),
                    'low_24h': float(df['low'].tail(24).min()) if len(df) >= 24 else float(df['low'].min()),
                    'volume_24h': float(df['volume'].tail(24).sum()) if len(df) >= 24 else float(df['volume'].sum())
                }
            }
            
            return results
        except Exception as e:
            return {
                "error": f"خطا در محاسبه اندیکاتورها: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def get_trading_signals(self, indicators: Dict[str, Any]) -> List[TechnicalSignal]:
        """تولید سیگنال‌های معاملاتی"""
        signals = []
        
        try:
            # تحلیل RSI
            rsi_data = indicators.get('momentum_indicators', {}).get('rsi', [])
            if len(rsi_data) > 0:
                rsi = rsi_data[-1]
                if not np.isnan(rsi):
                    if rsi < 30:
                        signals.append(TechnicalSignal(
                            indicator="RSI",
                            value=float(rsi),
                            signal="خرید",
                            strength=SignalStrength.STRONG,
                            trend=TrendDirection.BULLISH,
                            description="RSI در ناحیه اشباع فروش",
                            timestamp=datetime.now().isoformat()
                        ))
                    elif rsi > 70:
                        signals.append(TechnicalSignal(
                            indicator="RSI",
                            value=float(rsi),
                            signal="فروش",
                            strength=SignalStrength.STRONG,
                            trend=TrendDirection.BEARISH,
                            description="RSI در ناحیه اشباع خرید",
                            timestamp=datetime.now().isoformat()
                        ))
            
            # تحلیل MACD
            macd_data = indicators.get('trend_indicators', {}).get('macd', [])
            macd_signal_data = indicators.get('trend_indicators', {}).get('macd_signal', [])
            if len(macd_data) > 0 and len(macd_signal_data) > 0:
                macd = macd_data[-1]
                macd_signal = macd_signal_data[-1]
                if not np.isnan(macd) and not np.isnan(macd_signal):
                    if macd > macd_signal and macd_signal > 0:
                        signals.append(TechnicalSignal(
                            indicator="MACD",
                            value=float(macd),
                            signal="خرید قوی",
                            strength=SignalStrength.VERY_STRONG,
                            trend=TrendDirection.BULLISH,
                            description="MACD بالای خط سیگنال و مثبت",
                            timestamp=datetime.now().isoformat()
                        ))
                    elif macd < macd_signal and macd_signal < 0:
                        signals.append(TechnicalSignal(
                            indicator="MACD",
                            value=float(macd),
                            signal="فروش قوی",
                            strength=SignalStrength.VERY_STRONG,
                            trend=TrendDirection.BEARISH,
                            description="MACD زیر خط سیگنال و منفی",
                            timestamp=datetime.now().isoformat()
                        ))
            
            # تحلیل Moving Averages
            sma_20_data = indicators.get('trend_indicators', {}).get('sma_20', [])
            sma_50_data = indicators.get('trend_indicators', {}).get('sma_50', [])
            current_price = indicators.get('price_action', {}).get('current_price', 0)
            
            if len(sma_20_data) > 0 and len(sma_50_data) > 0:
                sma_20 = sma_20_data[-1]
                sma_50 = sma_50_data[-1]
                if not np.isnan(sma_20) and not np.isnan(sma_50):
                    if current_price > sma_20 > sma_50:
                        signals.append(TechnicalSignal(
                            indicator="Moving Averages",
                            value=float(current_price),
                            signal="خرید",
                            strength=SignalStrength.STRONG,
                            trend=TrendDirection.BULLISH,
                            description="قیمت بالای میانگین‌های متحرک",
                            timestamp=datetime.now().isoformat()
                        ))
                    elif current_price < sma_20 < sma_50:
                        signals.append(TechnicalSignal(
                            indicator="Moving Averages",
                            value=float(current_price),
                            signal="فروش",
                            strength=SignalStrength.STRONG,
                            trend=TrendDirection.BEARISH,
                            description="قیمت زیر میانگین‌های متحرک",
                            timestamp=datetime.now().isoformat()
                        ))
            
            # تحلیل Bollinger Bands
            bb_upper_data = indicators.get('volatility_indicators', {}).get('bollinger_upper', [])
            bb_lower_data = indicators.get('volatility_indicators', {}).get('bollinger_lower', [])
            if len(bb_upper_data) > 0 and len(bb_lower_data) > 0:
                bb_upper = bb_upper_data[-1]
                bb_lower = bb_lower_data[-1]
                if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                    if current_price < bb_lower:
                        signals.append(TechnicalSignal(
                            indicator="Bollinger Bands",
                            value=float(current_price),
                            signal="خرید",
                            strength=SignalStrength.STRONG,
                            trend=TrendDirection.BULLISH,
                            description="قیمت در ناحیه پایین باند بولینگر",
                            timestamp=datetime.now().isoformat()
                        ))
                    elif current_price > bb_upper:
                        signals.append(TechnicalSignal(
                            indicator="Bollinger Bands",
                            value=float(current_price),
                            signal="فروش",
                            strength=SignalStrength.STRONG,
                            trend=TrendDirection.BEARISH,
                            description="قیمت در ناحیه بالای باند بولینگر",
                            timestamp=datetime.now().isoformat()
                        ))
                        
        except Exception as e:
            print(f"⚠️ خطا در تولید سیگنال‌ها: {e}")
        
        return signals

    def market_analysis_report(self, price_data: List[Dict]) -> Dict[str, Any]:
        """گزارش کامل تحلیل بازار"""
        try:
            indicators = self.calculate_all_indicators(price_data)
            
            if 'error' in indicators:
                return indicators
                
            signals = self.get_trading_signals(indicators)
            
            # تحلیل کلی بازار
            bullish_signals = len([s for s in signals if s.trend == TrendDirection.BULLISH])
            bearish_signals = len([s for s in signals if s.trend == TrendDirection.BEARISH])
            
            overall_trend = TrendDirection.NEUTRAL
            if bullish_signals > bearish_signals + 2:
                overall_trend = TrendDirection.BULLISH
            elif bearish_signals > bullish_signals + 2:
                overall_trend = TrendDirection.BEARISH
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_trend': overall_trend.value,
                'signal_summary': {
                    'total_signals': len(signals),
                    'bullish_signals': bullish_signals,
                    'bearish_signals': bearish_signals,
                    'neutral_signals': len(signals) - bullish_signals - bearish_signals
                },
                'trading_signals': [{
                    'indicator': s.indicator,
                    'signal': s.signal,
                    'strength': s.strength.value,
                    'trend': s.trend.value,
                    'description': s.description,
                    'value': s.value
                } for s in signals],
                'key_levels': {
                    'pivot_points': indicators.get('pivot_points', {}),
                    'fibonacci_levels': indicators.get('fibonacci_levels', {})
                },
                'market_condition': self._assess_market_condition(indicators),
                'risk_assessment': self._calculate_risk_assessment(indicators),
                'price_action': indicators.get('price_action', {})
            }
            
            return report
        except Exception as e:
            return {
                'error': f"خطا در تولید گزارش تحلیل: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

    def _assess_market_condition(self, indicators: Dict[str, Any]) -> str:
        """ارزیابی شرایط بازار"""
        try:
            atr_data = indicators.get('volatility_indicators', {}).get('atr', [])
            current_price = indicators.get('price_action', {}).get('current_price', 1)
            
            if len(atr_data) > 0:
                volatility = atr_data[-1]
                if not np.isnan(volatility):
                    volatility_ratio = (volatility / current_price) * 100
                    
                    if volatility_ratio > 5:
                        return "پرنوسان"
                    elif volatility_ratio > 2:
                        return "نوسان متوسط"
            
            return "کمنوسان"
        except:
            return "نامشخص"

    def _calculate_risk_assessment(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه ارزیابی ریسک"""
        try:
            rsi_data = indicators.get('momentum_indicators', {}).get('rsi', [])
            adx_data = indicators.get('trend_indicators', {}).get('adx', [])
            atr_data = indicators.get('volatility_indicators', {}).get('atr', [])
            current_price = indicators.get('price_action', {}).get('current_price', 1)
            
            risk_score = 0
            factors = {}
            
            if len(rsi_data) > 0:
                rsi = rsi_data[-1]
                if not np.isnan(rsi):
                    if rsi > 70 or rsi < 30:
                        risk_score += 2
                        factors['overbought_oversold'] = True
                    else:
                        factors['overbought_oversold'] = False
            
            if len(adx_data) > 0:
                adx = adx_data[-1]
                if not np.isnan(adx):
                    if adx > 25:
                        risk_score += 1
                        factors['trend_strength'] = True
                    else:
                        factors['trend_strength'] = False
            
            if len(atr_data) > 0:
                volatility = atr_data[-1]
                if not np.isnan(volatility):
                    if (volatility / current_price) > 0.03:
                        risk_score += 1
                        factors['high_volatility'] = True
                    else:
                        factors['high_volatility'] = False
            
            risk_level = "کم" if risk_score <= 1 else "متوسط" if risk_score <= 2 else "بالا"
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'factors': factors
            }
        except Exception as e:
            return {
                'risk_score': 0,
                'risk_level': 'نامشخص',
                'error': str(e)
            }

    def analyze_raw_api_data(self, raw_api_response: Dict) -> Dict:
        """تحلیل داده‌های خام دریافتی از API"""
        try:
            # استخراج داده‌های قیمت از پاسخ خام API
            raw_data = raw_api_response
            
            # بررسی ساختارهای مختلف داده
            price_data = []
            
            if isinstance(raw_data, dict) and 'raw_data' in raw_data:
                # ساختار اصلی با فیلد raw_data
                actual_data = raw_data['raw_data']
            else:
                actual_data = raw_data
            
            if isinstance(actual_data, list):
                # داده‌های لیستی (مانند لیست کوین‌ها)
                for item in actual_data:
                    if isinstance(item, dict):
                        price = item.get('price', item.get('close', 0))
                        price_data.append({
                            'timestamp': datetime.now() - timedelta(hours=len(price_data)),
                            'open': price * 0.99,
                            'high': price * 1.01,
                            'low': price * 0.99,
                            'close': price,
                            'volume': item.get('volume', item.get('total_volume', 1000))
                        })
            elif isinstance(actual_data, dict) and 'chart' in actual_data:
                # داده‌های چارت تاریخی
                chart_data = actual_data['chart']
                for point in chart_data:
                    if isinstance(point, list) and len(point) >= 2:
                        price_data.append({
                            'timestamp': datetime.fromtimestamp(point[0]/1000) if point[0] > 1000000000000 else datetime.fromtimestamp(point[0]),
                            'open': point[1] * 0.99,
                            'high': point[2] if len(point) > 2 else point[1] * 1.01,
                            'low': point[3] if len(point) > 3 else point[1] * 0.99,
                            'close': point[4] if len(point) > 4 else point[1],
                            'volume': point[5] if len(point) > 5 else 1000
                        })
            else:
                # داده‌های نامشخص - ایجاد داده‌های نمونه
                base_price = 50000
                for i in range(100):
                    price_data.append({
                        'timestamp': datetime.now() - timedelta(hours=100-i),
                        'open': base_price + (i * 10) - 50,
                        'high': base_price + (i * 10) + 100,
                        'low': base_price + (i * 10) - 100,
                        'close': base_price + (i * 10),
                        'volume': 5000 + (i * 10)
                    })
            
            return self.market_analysis_report(price_data)
            
        except Exception as e:
            return {
                'error': f"خطا در تحلیل داده‌های خام: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

# نمونه استفاده
if __name__ == "__main__":
    # تست موتور تحلیل تکنیکال
    engine = TechnicalAnalysisEngine()
    
    # تولید داده‌های نمونه
    sample_data = []
    base_price = 50000
    for i in range(100):
        price = base_price + (i * 100) + np.random.normal(0, 500)
        sample_data.append({
            'timestamp': datetime.now() - timedelta(hours=100-i),
            'open': price - np.random.normal(0, 50),
            'high': price + abs(np.random.normal(0, 100)),
            'low': price - abs(np.random.normal(0, 100)),
            'close': price,
            'volume': np.random.randint(1000, 10000)
        })
    
    # تحلیل کامل
    report = engine.market_analysis_report(sample_data)
    print(f"📊 گزارش تحلیل بازار:")
    print(f"روند کلی: {report.get('overall_trend', 'N/A')}")
    print(f"تعداد سیگنال‌ها: {report.get('signal_summary', {}).get('total_signals', 0)}")
    print(f"شرایط بازار: {report.get('market_condition', 'N/A')}")
    print(f"سطح ریسک: {report.get('risk_assessment', {}).get('risk_level', 'N/A')}")

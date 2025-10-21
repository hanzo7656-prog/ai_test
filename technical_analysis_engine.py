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

class TechnicalAnalysisEngine:
    """
    موتور کامل تحلیل تکنیکال با pandas-ta
    """
    
    def __init__(self):
        self.available_indicators = {
            'trend': ['sma', 'ema', 'wma', 'macd', 'adx', 'ichimoku', 'parabolic_sar'],
            'momentum': ['rsi', 'stoch', 'williams_r', 'cci', 'mfi', 'awesome_oscillator'],
            'volatility': ['bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels'],
            'volume': ['obv', 'volume_profile', 'vwap', 'accumulation_distribution'],
            'support_resistance': ['pivot_points', 'fibonacci', 'candlestick_patterns']
        }
        
        print("🚀 موتور تحلیل تکنیکال با pandas-ta راه‌اندازی شد")

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
                    if col in ['open', 'high', 'low', 'close']:
                        base_price = 50000
                        df[col] = [base_price + (i * 10) + np.random.normal(0, 100) for i in range(len(df))]
                    elif col == 'volume':
                        df[col] = np.random.randint(1000, 10000, len(df))
        
        # تبدیل به عدد
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna().sort_values('timestamp').reset_index(drop=True)
        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای روند"""
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        
        results = {}
        
        try:
            # Moving Averages
            results['sma_20'] = ta.sma(close_prices, length=20)
            results['sma_50'] = ta.sma(close_prices, length=50)
            results['sma_200'] = ta.sma(close_prices, length=200)
            
            results['ema_12'] = ta.ema(close_prices, length=12)
            results['ema_26'] = ta.ema(close_prices, length=26)
            
            results['wma_20'] = ta.wma(close_prices, length=20)
            
            # MACD
            macd_data = ta.macd(close_prices, fast=12, slow=26, signal=9)
            if macd_data is not None:
                results['macd'] = macd_data.get('MACD_12_26_9')
                results['macd_signal'] = macd_data.get('MACDs_12_26_9')
                results['macd_histogram'] = macd_data.get('MACDh_12_26_9')
            
            # ADX
            adx_data = ta.adx(high_prices, low_prices, close_prices, length=14)
            if adx_data is not None:
                results['adx'] = adx_data.get('ADX_14')
            
            # Parabolic SAR
            results['parabolic_sar'] = ta.psar(high_prices, low_prices)
            
            # Ichimoku Cloud
            ichimoku_data = ta.ichimoku(high_prices, low_prices, close_prices)
            if ichimoku_data is not None:
                results['ichimoku'] = {
                    'tenkan_sen': ichimoku_data.get('ITS_9'),
                    'kijun_sen': ichimoku_data.get('IKS_26'),
                    'senkou_span_a': ichimoku_data.get('ISA_9'),
                    'senkou_span_b': ichimoku_data.get('ISB_26'),
                    'chikou_span': ichimoku_data.get('ICS_26')
                }
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای روند: {e}")
        
        return results

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای مومنتوم"""
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        volume = df['volume']
        
        results = {}
        
        try:
            # RSI
            results['rsi'] = ta.rsi(close_prices, length=14)
            
            # Stochastic
            stoch_data = ta.stoch(high_prices, low_prices, close_prices)
            if stoch_data is not None:
                results['stoch_k'] = stoch_data.get('STOCHk_14_3_3')
                results['stoch_d'] = stoch_data.get('STOCHd_14_3_3')
            
            # Williams %R
            results['williams_r'] = ta.willr(high_prices, low_prices, close_prices, length=14)
            
            # CCI
            results['cci'] = ta.cci(high_prices, low_prices, close_prices, length=20)
            
            # MFI
            results['mfi'] = ta.mfi(high_prices, low_prices, close_prices, volume, length=14)
            
            # Awesome Oscillator
            results['awesome_oscillator'] = ta.ao(high_prices, low_prices)
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای مومنتوم: {e}")
        
        return results

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای نوسان"""
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        
        results = {}
        
        try:
            # Bollinger Bands
            bb_data = ta.bbands(close_prices, length=20, std=2)
            if bb_data is not None:
                results['bollinger_upper'] = bb_data.get('BBU_20_2.0')
                results['bollinger_middle'] = bb_data.get('BBM_20_2.0')
                results['bollinger_lower'] = bb_data.get('BBL_20_2.0')
                results['bollinger_bandwidth'] = bb_data.get('BBB_20_2.0')
            
            # ATR
            results['atr'] = ta.atr(high_prices, low_prices, close_prices, length=14)
            
            # Keltner Channels
            kc_data = ta.kc(high_prices, low_prices, close_prices)
            if kc_data is not None:
                results['keltner'] = {
                    'upper': kc_data.get('KCUe_20_2'),
                    'middle': kc_data.get('KCBe_20_2'),
                    'lower': kc_data.get('KCLe_20_2')
                }
            
            # Donchian Channels
            dc_data = ta.donchian(high_prices, low_prices, length=20)
            if dc_data is not None:
                results['donchian'] = {
                    'upper': dc_data.get('DCU_20'),
                    'middle': dc_data.get('DCM_20'),
                    'lower': dc_data.get('DCL_20')
                }
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای نوسان: {e}")
        
        return results

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه اندیکاتورهای حجم"""
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        volume = df['volume']
        
        results = {}
        
        try:
            # OBV
            results['obv'] = ta.obv(close_prices, volume)
            
            # Volume Profile
            results['volume_profile'] = self._calculate_volume_profile(df)
            
            # VWAP
            results['vwap'] = ta.vwap(high_prices, low_prices, close_prices, volume)
            
            # Accumulation/Distribution
            results['ad_line'] = ta.ad(high_prices, low_prices, close_prices, volume)
            
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اندیکاتورهای حجم: {e}")
        
        return results

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """محاسبه Volume Profile"""
        try:
            prices = df['close'].values
            volumes = df['volume'].values
            
            if len(prices) == 0:
                return {}
                
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

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تشخیص الگوهای کندل استیک"""
        try:
            open_prices = df['open']
            high_prices = df['high']
            low_prices = df['low']
            close_prices = df['close']
            
            patterns = {}
            
            # الگوهای کندل استیک با pandas-ta
            patterns['doji'] = ta.cdl_doji(open_prices, high_prices, low_prices, close_prices)
            patterns['hammer'] = ta.cdl_hammer(open_prices, high_prices, low_prices, close_prices)
            patterns['engulfing_bullish'] = ta.cdl_engulfing(open_prices, high_prices, low_prices, close_prices)
            patterns['engulfing_bearish'] = -ta.cdl_engulfing(open_prices, high_prices, low_prices, close_prices)
            patterns['morning_star'] = ta.cdl_morningstar(open_prices, high_prices, low_prices, close_prices)
            patterns['evening_star'] = ta.cdl_eveningstar(open_prices, high_prices, low_prices, close_prices)
            
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
            rsi_data = indicators.get('momentum_indicators', {}).get('rsi')
            if rsi_data is not None and len(rsi_data) > 0:
                rsi = rsi_data.iloc[-1] if hasattr(rsi_data, 'iloc') else rsi_data[-1]
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
            macd_data = indicators.get('trend_indicators', {}).get('macd')
            macd_signal_data = indicators.get('trend_indicators', {}).get('macd_signal')
            if macd_data is not None and macd_signal_data is not None:
                if len(macd_data) > 0 and len(macd_signal_data) > 0:
                    macd = macd_data.iloc[-1] if hasattr(macd_data, 'iloc') else macd_data[-1]
                    macd_signal = macd_signal_data.iloc[-1] if hasattr(macd_signal_data, 'iloc') else macd_signal_data[-1]
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
            
            # تحلیل Moving Averages
            sma_20_data = indicators.get('trend_indicators', {}).get('sma_20')
            sma_50_data = indicators.get('trend_indicators', {}).get('sma_50')
            current_price = indicators.get('price_action', {}).get('current_price', 0)
            
            if sma_20_data is not None and sma_50_data is not None:
                if len(sma_20_data) > 0 and len(sma_50_data) > 0:
                    sma_20 = sma_20_data.iloc[-1] if hasattr(sma_20_data, 'iloc') else sma_20_data[-1]
                    sma_50 = sma_50_data.iloc[-1] if hasattr(sma_50_data, 'iloc') else sma_50_data[-1]
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
            
            # تحلیل Bollinger Bands
            bb_upper_data = indicators.get('volatility_indicators', {}).get('bollinger_upper')
            bb_lower_data = indicators.get('volatility_indicators', {}).get('bollinger_lower')
            if bb_upper_data is not None and bb_lower_data is not None:
                if len(bb_upper_data) > 0 and len(bb_lower_data) > 0:
                    bb_upper = bb_upper_data.iloc[-1] if hasattr(bb_upper_data, 'iloc') else bb_upper_data[-1]
                    bb_lower = bb_lower_data.iloc[-1] if hasattr(bb_lower_data, 'iloc') else bb_lower_data[-1]
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
            atr_data = indicators.get('volatility_indicators', {}).get('atr')
            current_price = indicators.get('price_action', {}).get('current_price', 1)
            
            if atr_data is not None and len(atr_data) > 0:
                volatility = atr_data.iloc[-1] if hasattr(atr_data, 'iloc') else atr_data[-1]
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
            rsi_data = indicators.get('momentum_indicators', {}).get('rsi')
            adx_data = indicators.get('trend_indicators', {}).get('adx')
            atr_data = indicators.get('volatility_indicators', {}).get('atr')
            current_price = indicators.get('price_action', {}).get('current_price', 1)
            
            risk_score = 0
            factors = {}
            
            if rsi_data is not None and len(rsi_data) > 0:
                rsi = rsi_data.iloc[-1] if hasattr(rsi_data, 'iloc') else rsi_data[-1]
                if not np.isnan(rsi):
                    if rsi > 70 or rsi < 30:
                        risk_score += 2
                        factors['overbought_oversold'] = True
                    else:
                        factors['overbought_oversold'] = False
            
            if adx_data is not None and len(adx_data) > 0:
                adx = adx_data.iloc[-1] if hasattr(adx_data, 'iloc') else adx_data[-1]
                if not np.isnan(adx):
                    if adx > 25:
                        risk_score += 1
                        factors['trend_strength'] = True
                    else:
                        factors['trend_strength'] = False
            
            if atr_data is not None and len(atr_data) > 0:
                volatility = atr_data.iloc[-1] if hasattr(atr_data, 'iloc') else atr_data[-1]
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
            raw_data = raw_api_response
            
            price_data = []
            
            if isinstance(raw_data, dict) and 'raw_data' in raw_data:
                actual_data = raw_data['raw_data']
            else:
                actual_data = raw_data
            
            if isinstance(actual_data, list):
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

# تست موتور
if __name__ == "__main__":
    engine = TechnicalAnalysisEngine()
    
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
    
    report = engine.market_analysis_report(sample_data)
    print(f"📊 گزارش تحلیل بازار:")
    print(f"روند کلی: {report.get('overall_trend', 'N/A')}")
    print(f"تعداد سیگنال‌ها: {report.get('signal_summary', {}).get('total_signals', 0)}")
    print(f"شرایط بازار: {report.get('market_condition', 'N/A')}")
    print(f"سطح ریسک: {report.get('risk_assessment', {}).get('risk_level', 'N/A')}")

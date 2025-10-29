# main_trading_system.py - سیستم اصلی تحلیل تکنیکال هوشمند با داده‌های خام

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from trading_ai.database_manager import trading_db
from trading_ai.advanced_technical_engine import technical_engine
from config import trading_config
from complete_coinstats_manager import coin_stats_manager
from lbank_websocket import get_websocket_manager
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainTradingSystem:
    """ سیستم اصلی معاملاتی هوشمند با معماری اسپارس و داده‌های خام"""
    
    def __init__(self):
        self.is_initialized = False
        self.technical_engine = technical_engine
        self.analyzer = None
        self.ws_manager = get_websocket_manager()
        
        self.market_state = {
            "overall_trend": "neutral",
            "volatility_level": "medium", 
            "risk_appetite": "moderate",
            "active_signals": [],
            "raw_data_quality": {},
            "last_analysis_time": None
        }
        
        # کش برای داده‌های خام
        self.raw_data_cache = {}
        self.cache_expiry = 300  # 5 دقیقه

    def initialize_system(self) -> bool:
        """راه‌اندازی کامل سیستم معاملاتی با داده‌های خام"""
        try:
            logger.info("🚀 راه‌اندازی سیستم معاملاتی هوشمند با داده‌های خام...")

            # بارگذاری داده‌های تاریخی خام برای نمادهای اصلی
            successful_loads = 0
            raw_data_quality = {}
            
            for symbol in trading_config.SYMBOLS:
                try:
                    # دریافت داده‌های خام از CoinStats
                    raw_historical = self._get_raw_historical_data(symbol, trading_config.LOOKBACK_DAYS)
                    raw_current = self._get_raw_current_data(symbol)
                    
                    if raw_historical and raw_current:
                        successful_loads += 1
                        raw_data_quality[symbol] = {
                            "historical_data_points": len(raw_historical.get('result', [])),
                            "current_data_available": bool(raw_current),
                            "data_freshness": datetime.now().isoformat(),
                            "quality_score": self._calculate_data_quality(raw_historical, raw_current)
                        }
                        
                        logger.info(f"✅ داده‌های خام {symbol} بارگذاری شد: {raw_data_quality[symbol]['historical_data_points']} نقطه داده")
                    else:
                        logger.warning(f"⚠️ داده‌های خام ناکافی برای {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ خطا در بارگذاری داده‌های خام {symbol}: {e}")

            if successful_loads >= 2:  # حداقل دو نماد با داده کافی
                self.is_initialized = True
                self.market_state["raw_data_quality"] = raw_data_quality
                
                # تحلیل اولیه شرایط بازار با داده‌های خام
                self._analyze_market_conditions()
                
                logger.info(f"✅ سیستم معاملاتی راه‌اندازی شد: {successful_loads} نماد فعال - حالت داده خام")
                return True
            else:
                logger.error("❌ داده‌های خام کافی برای راه‌اندازی سیستم موجود نیست")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سیستم: {e}")
            return False

    def _get_raw_historical_data(self, symbol: str, days: int) -> Dict[str, Any]:
        """دریافت داده‌های تاریخی خام از CoinStats"""
        cache_key = f"historical_{symbol}_{days}"
        
        # بررسی کش
        if cache_key in self.raw_data_cache:
            cached_data, timestamp = self.raw_data_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            # دریافت داده‌های خام از CoinStats API
            period = self._days_to_period(days)
            raw_data = coin_stats_manager.get_coin_charts(symbol, period)
            
            # ذخیره در کش
            self.raw_data_cache[cache_key] = (raw_data, time.time())
            
            return raw_data
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های تاریخی خام {symbol}: {e}")
            return {}

    def _get_raw_current_data(self, symbol: str) -> Dict[str, Any]:
        """دریافت داده‌های جاری خام"""
        cache_key = f"current_{symbol}"
        
        # بررسی کش
        if cache_key in self.raw_data_cache:
            cached_data, timestamp = self.raw_data_cache[cache_key]
            if time.time() - timestamp < 60:  # 1 دقیقه برای داده‌های جاری
                return cached_data
        
        try:
            # دریافت از CoinStats
            raw_data = coin_stats_manager.get_coin_details(symbol, "USD")
            
            # دریافت از WebSocket (اگر موجود باشد)
            ws_data = self.ws_manager.get_realtime_data(symbol)
            
            combined_data = {
                "coinstats": raw_data,
                "websocket": ws_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # ذخیره در کش
            self.raw_data_cache[cache_key] = (combined_data, time.time())
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های جاری خام {symbol}: {e}")
            return {}

    def _days_to_period(self, days: int) -> str:
        """تبدیل روز به دوره معتبر CoinStats"""
        if days <= 1:
            return "24h"
        elif days <= 7:
            return "1w" 
        elif days <= 30:
            return "1m"
        elif days <= 90:
            return "3m"
        elif days <= 180:
            return "6m"
        elif days <= 365:
            return "1y"
        else:
            return "all"

    def _calculate_data_quality(self, historical_data: Dict, current_data: Dict) -> float:
        """محاسبه کیفیت داده‌های خام"""
        quality_score = 0.0
        
        try:
            # کیفیت داده‌های تاریخی
            if historical_data and 'result' in historical_data:
                historical_points = len(historical_data['result'])
                quality_score += min(historical_points / 1000, 1.0) * 0.6  # 60% وزن
            
            # کیفیت داده‌های جاری
            if current_data and current_data.get('coinstats'):
                quality_score += 0.3  # 30% وزن
                
            if current_data and current_data.get('websocket'):
                quality_score += 0.1  # 10% وزن
                
        except Exception as e:
            logger.error(f"خطا در محاسبه کیفیت داده: {e}")
            
        return round(quality_score, 3)

    def _analyze_market_conditions(self):
        """تحلیل شرایط کلی بازار با داده‌های خام"""
        try:
            trends = []
            volatilities = []
            raw_data_metrics = {}
            
            for symbol in trading_config.SYMBOLS[:4]:  # 4 نماد اول
                # دریافت داده‌های خام
                raw_data = self._get_raw_historical_data(symbol, 30)
                
                if raw_data and 'result' in raw_data:
                    # استخراج قیمت‌ها از داده خام
                    prices = []
                    for item in raw_data['result']:
                        if 'price' in item:
                            try:
                                prices.append(float(item['price']))
                            except (ValueError, TypeError):
                                continue
                    
                    if len(prices) >= 2:
                        # تحلیل روند ساده از داده خام
                        price_change = (prices[-1] / prices[0] - 1) * 100
                        trends.append(price_change)
                        
                        # محاسبه نوسان از داده خام
                        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                        volatility = np.std(returns) * 100 if returns else 0
                        volatilities.append(volatility)
                        
                        raw_data_metrics[symbol] = {
                            "data_points": len(prices),
                            "price_range": (min(prices), max(prices)),
                            "latest_price": prices[-1]
                        }
            
            if trends:
                avg_trend = np.mean(trends)
                avg_volatility = np.mean(volatilities)
                
                # تعیین وضعیت بازار از داده خام
                if avg_trend > 5:
                    self.market_state["overall_trend"] = "bullish"
                elif avg_trend < -5:
                    self.market_state["overall_trend"] = "bearish"
                else:
                    self.market_state["overall_trend"] = "neutral"

                if avg_volatility > 3:
                    self.market_state["volatility_level"] = "high"
                elif avg_volatility < 1:
                    self.market_state["volatility_level"] = "low"
                else:
                    self.market_state["volatility_level"] = "medium"
                
                self.market_state["raw_data_metrics"] = raw_data_metrics
                
                logger.info(f"📊 وضعیت بازار از داده خام: روند {self.market_state['overall_trend']} - نوسان {self.market_state['volatility_level']}")
                
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل شرایط بازار: {e}")

    def analyze_symbol(self, symbol: str, analysis_type: str = "comprehensive") -> Dict:
        """تحلیل کامل یک نماد با داده‌های خام"""
        if not self.is_initialized:
            self.initialize_system()

        try:
            logger.info(f"🔍 تحلیل نماد {symbol} با داده‌های خام...")

            # دریافت داده‌های خام
            raw_historical = self._get_raw_historical_data(symbol, 100)
            raw_current = self._get_raw_current_data(symbol)
            
            if not raw_historical or not raw_current:
                return {
                    'error': 'داده‌های خام کافی موجود نیست',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'raw_data_available': False
                }

            # تحلیل تکنیکال پیشرفته با داده خام
            technical_analysis = self._perform_technical_analysis(raw_historical, raw_current, symbol)
            
            # تحلیل شرایط بازار
            market_analysis = self._analyze_market_context(symbol)
            
            # تولید سیگنال معاملاتی از داده خام
            trading_signal = self._generate_trading_signal(technical_analysis, market_analysis)
            
            # جمع‌بندی نتایج
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis,
                'market_context': market_analysis,
                'trading_signal': trading_signal,
                'system_confidence': self._calculate_confidence(technical_analysis, market_analysis),
                'recommendations': self._generate_recommendations(trading_signal, technical_analysis),
                'raw_data_metrics': {
                    'historical_points': len(raw_historical.get('result', [])),
                    'current_data_sources': len([k for k, v in raw_current.items() if v]) if raw_current else 0,
                    'data_quality': technical_analysis.get('data_quality', 'unknown')
                }
            }

            logger.info(f"✅ تحلیل {symbol} تکمیل شد - سیگنال: {trading_signal['action']}")
            return result

        except Exception as e:
            logger.error(f"❌ خطا در تحلیل {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'raw_data_available': False
            }

    def _perform_technical_analysis(self, raw_historical: Dict, raw_current: Dict, symbol: str) -> Dict:
        """انجام تحلیل تکنیکال پیشرفته با داده‌های خام"""
        try:
            # استخراج قیمت‌ها از داده خام
            prices = []
            for item in raw_historical.get('result', []):
                if 'price' in item:
                    try:
                        prices.append(float(item['price']))
                    except (ValueError, TypeError):
                        continue
            
            if not prices:
                return {
                    'error': 'داده قیمتی در داده‌های خام یافت نشد',
                    'data_quality': 'poor'
                }

            # تبدیل به DataFrame برای پردازش
            df = pd.DataFrame({
                'close': prices,
                'timestamp': range(len(prices))  # جایگزین timestamp واقعی
            })
            
            # محاسبه اندیکاتورها از داده خام
            df_with_indicators = technical_engine.calculate_all_indicators(df)
            
            # استخراج ویژگی‌های فنی از داده خام
            technical_features = technical_engine.extract_technical_features(df_with_indicators)
            
            # تحلیل روند از داده خام
            trend_analysis = self._analyze_trend(df_with_indicators)
            
            # شناسایی سطوح کلیدی از داده خام
            key_levels = self._identify_key_levels(df_with_indicators)
            
            # تحلیل قدرت بازار از داده خام
            market_strength = self._analyze_market_strength(df_with_indicators)
            
            # داده‌های جاری از WebSocket
            current_ws_data = raw_current.get('websocket', {})
            current_price = current_ws_data.get('price', prices[-1] if prices else 0)
            
            return {
                'current_price': float(current_price),
                'price_change_24h': self._calculate_price_change(prices),
                'trend_analysis': trend_analysis,
                'key_levels': key_levels,
                'market_strength': market_strength,
                'technical_features': technical_features.tolist() if hasattr(technical_features, 'tolist') else [],
                'data_quality': 'high' if len(prices) > 50 else 'medium',
                'raw_data_used': True,
                'indicators': {
                    'rsi': float(df_with_indicators['rsi_14'].iloc[-1]) if 'rsi_14' in df_with_indicators.columns else 50,
                    'macd': float(df_with_indicators['macd'].iloc[-1]) if 'macd' in df_with_indicators.columns else 0,
                    'volume': float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0
                } if not df_with_indicators.empty else {}
            }

        except Exception as e:
            logger.error(f"❌ خطا در تحلیل تکنیکال {symbol}: {e}")
            return {
                'current_price': 0,
                'error': f'خطای تحلیل تکنیکال: {str(e)}',
                'data_quality': 'poor'
            }

    def _calculate_price_change(self, prices: List[float]) -> float:
        """محاسبه تغییرات قیمت"""
        if len(prices) < 2:
            return 0.0
        return float(((prices[-1] / prices[-2]) - 1) * 100)

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """تحلیل روند قیمت از داده خام"""
        if len(df) < 20:
            return {'direction': 'neutral', 'strength': 0.5, 'duration': 'short'}

        prices = df['close'].values
        
        # روند کوتاه مدت (5 روز)
        short_term = (prices[-1] / prices[-5] - 1) * 100 if len(prices) >= 5 else 0
        
        # روند میان مدت (20 روز)
        mid_term = (prices[-1] / prices[-20] - 1) * 100 if len(prices) >= 20 else 0

        # تعیین جهت روند
        if mid_term > 2 and short_term > 0:
            direction = "bullish"
            strength = min(abs(mid_term) / 10, 1.0)
        elif mid_term < -2 and short_term < 0:
            direction = "bearish" 
            strength = min(abs(mid_term) / 10, 1.0)
        else:
            direction = "neutral"
            strength = 0.3

        return {
            'direction': direction,
            'strength': strength,
            'short_term_change': short_term,
            'mid_term_change': mid_term,
            'calculated_from_raw': True
        }

    def _identify_key_levels(self, df: pd.DataFrame) -> Dict:
        """شناسایی سطوح کلیدی حمایت و مقاومت از داده خام"""
        if len(df) < 20:
            return {'support': 0, 'resistance': 0}

        prices = df['close'].values[-20:]  # 20 روز اخیر
        
        support = np.min(prices) * 0.98  # 2% زیر کمینه
        resistance = np.max(prices) * 1.02  # 2% بالای بیشینه
        
        current_price = prices[-1] if len(prices) > 0 else 0
        
        return {
            'support': float(support),
            'resistance': float(resistance),
            'current_to_support': float((current_price - support) / support * 100) if support > 0 else 0,
            'current_to_resistance': float((resistance - current_price) / current_price * 100) if current_price > 0 else 0,
            'calculated_from_raw': True
        }

    def _analyze_market_strength(self, df: pd.DataFrame) -> Dict:
        """تحلیل قدرت بازار از داده خام"""
        if len(df) < 10:
            return {'volume_trend': 'neutral', 'price_momentum': 0.5}

        # تحلیل حجم از داده خام
        if 'volume' in df.columns:
            volume_trend = "increasing" if df['volume'].iloc[-1] > df['volume'].iloc[-5] else "decreasing"
        else:
            volume_trend = "unknown"

        # تحلیل مومنتوم از داده خام
        price_changes = df['close'].pct_change().dropna()
        momentum = price_changes.tail(5).mean() if len(price_changes) >= 5 else 0

        return {
            'volume_trend': volume_trend,
            'price_momentum': float(momentum),
            'volatility': float(price_changes.std()) if len(price_changes) > 0 else 0,
            'calculated_from_raw': True
        }

    def _analyze_market_context(self, symbol: str) -> Dict:
        """تحلیل شرایط بازار برای نماد"""
        return {
            'overall_trend': self.market_state["overall_trend"],
            'volatility_level': self.market_state["volatility_level"],
            'market_phase': self._determine_market_phase(),
            'symbol_correlation': self._analyze_symbol_correlation(symbol),
            'raw_data_quality': self.market_state["raw_data_quality"].get(symbol, {})
        }

    def _determine_market_phase(self) -> str:
        """تعیین فاز بازار"""
        phases = {
            "bullish": ["accumulation", "uptrend", "distribution"],
            "bearish": ["distribution", "downtrend", "accumulation"], 
            "neutral": ["consolidation", "ranging", "accumulation"],
        }
        return phases.get(self.market_state["overall_trend"], ["neutral"])[0]

    def _analyze_symbol_correlation(self, symbol: str) -> str:
        """تحلیل همبستگی نماد با بازار"""
        if symbol.lower() in ['bitcoin', 'ethereum']:
            return "high"
        else:
            return "medium"

    def _generate_trading_signal(self, technical: Dict, market: Dict) -> Dict:
        """تولید سیگنال معاملاتی از داده خام"""
        trend = technical.get('trend_analysis', {})
        levels = technical.get('key_levels', {})
        current_price = technical.get('current_price', 0)
        support = levels.get('support', 0)
        resistance = levels.get('resistance', 0)
        
        # تصمیم‌گیری بر اساس موقعیت قیمت نسبت به سطوح
        if current_price <= support * 1.02:  # نزدیک حمایت
            action = "BUY"
            confidence = 0.7
            reasoning = "قیمت در ناحیه حمایتی - داده خام"
        elif current_price >= resistance * 0.98:  # نزدیک مقاومت
            action = "SELL" 
            confidence = 0.7
            reasoning = "قیمت در ناحیه مقاومتی - داده خام"
        elif trend.get('direction') == 'bullish' and trend.get('strength', 0) > 0.6:
            action = "BUY"
            confidence = 0.6
            reasoning = "روند صعودی قوی - داده خام"
        elif trend.get('direction') == 'bearish' and trend.get('strength', 0) > 0.6:
            action = "SELL"
            confidence = 0.6
            reasoning = "روند نزولی قوی - داده خام"
        else:
            action = "HOLD"
            confidence = 0.5
            reasoning = "بازار در حالت خنثی - داده خام"

        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_level': 'medium',
            'timeframe': 'short_term',
            'raw_data_based': True
        }

    def _calculate_confidence(self, technical: Dict, market: Dict) -> float:
        """محاسبه اطمینان کلی تحلیل"""
        confidence_factors = []
        
        # اعتماد به تحلیل تکنیکال
        if technical.get('current_price', 0) > 0:
            confidence_factors.append(0.6)
            
        # اعتماد به شرایط بازار
        if market.get('overall_trend') != 'unknown':
            confidence_factors.append(0.3)
            
        # اعتماد به کیفیت داده خام
        data_quality = technical.get('data_quality', 'unknown')
        if data_quality == 'high':
            confidence_factors.append(0.1)
        elif data_quality == 'medium':
            confidence_factors.append(0.05)
        else:
            confidence_factors.append(0.0)
            
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5

    def _generate_recommendations(self, signal: Dict, technical: Dict) -> List[str]:
        """تولید توصیه‌های عملی از داده خام"""
        recommendations = []
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0.5)
        
        if action == "BUY" and confidence > 0.6:
            recommendations.append("ورود تدریجی به پوزیشن خرید - مبتنی بر داده خام")
            recommendations.append("حد ضرر: 2% زیر سطح حمایت")
        elif action == "SELL" and confidence > 0.6:
            recommendations.append("خروج تدریجی از پوزیشن‌های خرید - مبتنی بر داده خام")
            recommendations.append("حد سود: ناحیه مقاومتی")
        else:
            recommendations.append("انتظار برای سیگنال واضح‌تر - تحلیل داده خام ادامه دارد")
            recommendations.append("مدیریت ریسک و حفظ نقدینگی")

        # توصیه‌های عمومی
        recommendations.append("استفاده از حجم معقول - داده‌های خام اعتبارسنجی شده")
        recommendations.append("رعایت مدیریت ریسک - تحلیل مبتنی بر داده‌های واقعی بازار")

        return recommendations

    def get_system_status(self) -> Dict:
        """دریافت وضعیت سیستم"""
        return {
            'initialized': self.is_initialized,
            'market_state': self.market_state,
            'active_symbols': trading_config.SYMBOLS,
            'supported_analysis': ['technical', 'trend', 'levels', 'signals'],
            'last_analysis_time': datetime.now().isoformat(),
            'raw_data_mode': True,
            'cache_size': len(self.raw_data_cache),
            'data_sources': ['CoinStats', 'WebSocket']
        }

    def clear_cache(self):
        """پاکسازی کش داده‌های خام"""
        self.raw_data_cache.clear()
        logger.info("✅ کش داده‌های خام پاکسازی شد")

# ایجاد نمونه گلوبال
main_trading_system = MainTradingSystem()

if __name__ == "__main__":
    # تست سیستم
    system = MainTradingSystem()
    
    if system.initialize_system():
        print("✅ سیستم معاملاتی راه‌اندازی شد")
        
        # تحلیل نمونه
        result = system.analyze_symbol('bitcoin')
        print("\n📊 نتایج تحلیل:")
        print(f"نماد: {result['symbol']}")
        print(f"سیگنال: {result['trading_signal']['action']}")
        print(f"اطمینان: {result['system_confidence']:.2f}")
        print(f"توصیه: {result['recommendations'][:2]}")
        
        print(f"\n📈 وضعیت سیستم: {system.get_system_status()}")

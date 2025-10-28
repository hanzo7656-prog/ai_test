# main_trading_system.py - سیستم اصلی تحلیل تکنیکال هوشمند
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from database_manager import trading_db
from advanced_technical_engine import technical_engine
from config import trading_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainTradingSystem:
    """سیستم اصلی معاملاتی هوشمند با معماری اسپارس"""
    
    def __init__(self):
        self.is_initialized = False
        self.technical_engine = technical_engine
        self.analyzer = None
        self.market_state = {
            "overall_trend": "neutral",
            "volatility_level": "medium", 
            "risk_appetite": "moderate",
            "active_signals": []
        }
        
    def initialize_system(self) -> bool:
        """راه‌اندازی کامل سیستم معاملاتی"""
        try:
            logger.info("🚀 راه‌اندازی سیستم معاملاتی هوشمند...")
            
            # بارگذاری داده‌های تاریخی برای نمادهای اصلی
            successful_loads = 0
            for symbol in trading_config.SYMBOLS:
                try:
                    df = trading_db.get_historical_data(symbol, trading_config.LOOKBACK_DAYS)
                    if not df.empty and len(df) > 100:
                        successful_loads += 1
                        logger.info(f"✅ داده‌های {symbol} بارگذاری شد: {len(df)} رکورد")
                    else:
                        logger.warning(f"⚠️ داده‌های ناکافی برای {symbol}")
                except Exception as e:
                    logger.error(f"❌ خطا در بارگذاری {symbol}: {e}")
            
            if successful_loads >= 2:  # حداقل دو نماد با داده کافی
                self.is_initialized = True
                logger.info(f"✅ سیستم معاملاتی راه‌اندازی شد: {successful_loads} نماد فعال")
                self._analyze_market_conditions()
                return True
            else:
                logger.error("❌ داده‌های کافی برای راه‌اندازی سیستم موجود نیست")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی سیستم: {e}")
            return False
    
    def _analyze_market_conditions(self):
        """تحلیل شرایط کلی بازار"""
        try:
            trends = []
            volatilities = []
            
            for symbol in trading_config.SYMBOLS[:4]:  # 4 نماد اول
                df = trading_db.get_historical_data(symbol, 30)  # 30 روز اخیر
                if not df.empty:
                    # تحلیل روند ساده
                    price_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                    trends.append(price_change)
                    
                    # محاسبه نوسان
                    volatility = df['close'].pct_change().std() * 100
                    volatilities.append(volatility)
            
            if trends:
                avg_trend = np.mean(trends)
                avg_volatility = np.mean(volatilities)
                
                # تعیین وضعیت بازار
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
                    
                logger.info(f"📊 وضعیت بازار: روند {self.market_state['overall_trend']}, نوسان {self.market_state['volatility_level']}")
                
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل شرایط بازار: {e}")
    
    def analyze_symbol(self, symbol: str, analysis_type: str = "comprehensive") -> Dict:
        """تحلیل کامل یک نماد"""
        if not self.is_initialized:
            self.initialize_system()
        
        try:
            logger.info(f"🔍 تحلیل نماد {symbol}...")
            
            # دریافت داده‌های تاریخی
            df = trading_db.get_historical_data(symbol, 100)
            if df.empty:
                return {
                    'error': 'داده تاریخی کافی موجود نیست',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat()
                }
            
            # تحلیل تکنیکال پیشرفته
            technical_analysis = self._perform_technical_analysis(df, symbol)
            
            # تحلیل شرایط بازار
            market_analysis = self._analyze_market_context(symbol)
            
            # تولید سیگنال معاملاتی
            trading_signal = self._generate_trading_signal(technical_analysis, market_analysis)
            
            # جمع‌بندی نتایج
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis,
                'market_context': market_analysis,
                'trading_signal': trading_signal,
                'system_confidence': self._calculate_confidence(technical_analysis, market_analysis),
                'recommendations': self._generate_recommendations(trading_signal, technical_analysis)
            }
            
            logger.info(f"✅ تحلیل {symbol} تکمیل شد - سیگنال: {trading_signal['action']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def _perform_technical_analysis(self, df: pd.DataFrame, symbol: str) -> Dict:
        """انجام تحلیل تکنیکال پیشرفته"""
        try:
            # محاسبه اندیکاتورها
            df_with_indicators = technical_engine.calculate_all_indicators(df)
            
            # استخراج ویژگی‌های فنی
            technical_features = technical_engine.extract_technical_features(df_with_indicators)
            
            # تحلیل روند
            trend_analysis = self._analyze_trend(df_with_indicators)
            
            # شناسایی سطوح کلیدی
            key_levels = self._identify_key_levels(df_with_indicators)
            
            # تحلیل قدرت بازار
            market_strength = self._analyze_market_strength(df_with_indicators)
            
            return {
                'current_price': float(df['close'].iloc[-1]),
                'price_change_24h': float(((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100) if len(df) > 1 else 0,
                'trend_analysis': trend_analysis,
                'key_levels': key_levels,
                'market_strength': market_strength,
                'technical_features': technical_features.tolist() if hasattr(technical_features, 'tolist') else [],
                'indicators': {
                    'rsi': float(df_with_indicators['rsi_14'].iloc[-1]) if 'rsi_14' in df_with_indicators.columns else 50,
                    'macd': float(df_with_indicators['macd'].iloc[-1]) if 'macd' in df_with_indicators.columns else 0,
                    'volume': float(df['volume'].iloc[-1])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل تکنیکال {symbol}: {e}")
            return {
                'current_price': float(df['close'].iloc[-1]) if not df.empty else 0,
                'error': f"خطای تحلیل تکنیکال: {str(e)}"
            }
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """تحلیل روند قیمت"""
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
            'mid_term_change': mid_term
        }
    
    def _identify_key_levels(self, df: pd.DataFrame) -> Dict:
        """شناسایی سطوح کلیدی حمایت و مقاومت"""
        if len(df) < 20:
            return {'support': 0, 'resistance': 0}
        
        prices = df['close'].values[-20:]  # 20 روز اخیر
        
        support = np.min(prices) * 0.98  # 2% زیر کمینه
        resistance = np.max(prices) * 1.02  # 2% بالای بیشینه
        
        return {
            'support': float(support),
            'resistance': float(resistance),
            'current_to_support': float((df['close'].iloc[-1] - support) / support * 100),
            'current_to_resistance': float((resistance - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100)
        }
    
    def _analyze_market_strength(self, df: pd.DataFrame) -> Dict:
        """تحلیل قدرت بازار"""
        if len(df) < 10:
            return {'volume_trend': 'neutral', 'price_momentum': 0.5}
        
        # تحلیل حجم
        volume_trend = "increasing" if df['volume'].iloc[-1] > df['volume'].iloc[-5] else "decreasing"
        
        # تحلیل مومنتوم
        price_changes = df['close'].pct_change().dropna()
        momentum = price_changes.tail(5).mean()
        
        return {
            'volume_trend': volume_trend,
            'price_momentum': float(momentum),
            'volatility': float(price_changes.std())
        }
    
    def _analyze_market_context(self, symbol: str) -> Dict:
        """تحلیل شرایط بازار برای نماد"""
        return {
            'overall_trend': self.market_state["overall_trend"],
            'volatility_level': self.market_state["volatility_level"],
            'market_phase': self._determine_market_phase(),
            'symbol_correlation': self._analyze_symbol_correlation(symbol)
        }
    
    def _determine_market_phase(self) -> str:
        """تعیین فاز بازار"""
        phases = {
            "bullish": ["accumulation", "uptrend", "distribution"],
            "bearish": ["distribution", "downtrend", "accumulation"], 
            "neutral": ["consolidation", "ranging", "accumulation"]
        }
        
        return phases.get(self.market_state["overall_trend"], ["neutral"])[0]
    
    def _analyze_symbol_correlation(self, symbol: str) -> str:
        """تحلیل همبستگی نماد با بازار"""
        # پیاده‌سازی ساده - می‌تواند گسترش یابد
        if symbol.lower() in ['bitcoin', 'ethereum']:
            return "high"
        else:
            return "medium"
    
    def _generate_trading_signal(self, technical: Dict, market: Dict) -> Dict:
        """تولید سیگنال معاملاتی"""
        
        # منطق ساده برای تولید سیگنال
        trend = technical.get('trend_analysis', {})
        levels = technical.get('key_levels', {})
        
        current_price = technical.get('current_price', 0)
        support = levels.get('support', 0)
        resistance = levels.get('resistance', 0)
        
        # تصمیم‌گیری بر اساس موقعیت قیمت نسبت به سطوح
        if current_price <= support * 1.02:  # نزدیک حمایت
            action = "BUY"
            confidence = 0.7
            reasoning = "قیمت در ناحیه حمایتی"
        elif current_price >= resistance * 0.98:  # نزدیک مقاومت
            action = "SELL" 
            confidence = 0.7
            reasoning = "قیمت در ناحیه مقاومتی"
        elif trend.get('direction') == 'bullish' and trend.get('strength', 0) > 0.6:
            action = "BUY"
            confidence = 0.6
            reasoning = "روند صعودی قوی"
        elif trend.get('direction') == 'bearish' and trend.get('strength', 0) > 0.6:
            action = "SELL"
            confidence = 0.6
            reasoning = "روند نزولی قوی"
        else:
            action = "HOLD"
            confidence = 0.5
            reasoning = "بازار در حالت خنثی"
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_level': 'medium',
            'timeframe': 'short_term'
        }
    
    def _calculate_confidence(self, technical: Dict, market: Dict) -> float:
        """محاسبه اطمینان کلی تحلیل"""
        confidence_factors = []
        
        # اعتماد به تحلیل تکنیکال
        if technical.get('current_price', 0) > 0:
            confidence_factors.append(0.7)
        
        # اعتماد به شرایط بازار
        if market.get('overall_trend') != 'unknown':
            confidence_factors.append(0.3)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5
    
    def _generate_recommendations(self, signal: Dict, technical: Dict) -> List[str]:
        """تولید توصیه‌های عملی"""
        recommendations = []
        
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0.5)
        
        if action == "BUY" and confidence > 0.6:
            recommendations.append("ورود پلکانی به پوزیشن خرید")
            recommendations.append("حد ضرر: 2% زیر سطح حمایت")
        elif action == "SELL" and confidence > 0.6:
            recommendations.append("خروج تدریجی از پوزیشن‌های خرید")
            recommendations.append("حد سود: ناحیه مقاومتی")
        else:
            recommendations.append("انتظار برای سیگنال واضح‌تر")
            recommendations.append("مدیریت ریسک و حفظ نقدینگی")
        
        # توصیه‌های عمومی
        recommendations.append("استفاده از حجم معقول")
        recommendations.append("رعایت مدیریت ریسک")
        
        return recommendations
    
    def get_system_status(self) -> Dict:
        """دریافت وضعیت سیستم"""
        return {
            'initialized': self.is_initialized,
            'market_state': self.market_state,
            'active_symbols': trading_config.SYMBOLS,
            'supported_analysis': ['technical', 'trend', 'levels', 'signals'],
            'last_analysis_time': datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال
main_trading_system = MainTradingSystem()

if __name__ == "__main__":
    # تست سیستم
    system = MainTradingSystem()
    
    if system.initialize_system():
        print("✅ سیستم معاملاتی راه‌اندازی شد")
        
        # تحلیل نمونه
        result = system.analyze_symbol('bitcoin')
        print(f"🎯 نتایج تحلیل:")
        print(f"نماد: {result['symbol']}")
        print(f"سیگنال: {result['trading_signal']['action']}")
        print(f"اطمینان: {result['system_confidence']:.2f}")
        print(f"توصیه‌ها: {result['recommendations'][:2]}")
        
        print(f"\n📊 وضعیت سیستم: {system.get_system_status()}")

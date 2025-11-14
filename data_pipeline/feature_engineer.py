# data_pipeline/feature_engineer.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import talib

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """سیستم مهندسی ویژگی‌های پیشرفته برای داده‌های مالی"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_importance = {}
        self.feature_stats = {}
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🔧 Feature Engineer initialized")

    def engineer_market_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """مهندسی ویژگی‌های پیشرفته بازار"""
        try:
            engineered_features = {
                'timestamp': datetime.now().isoformat(),
                'base_features': {},
                'technical_indicators': {},
                'statistical_features': {},
                'temporal_features': {},
                'feature_metadata': {}
            }
            
            # پردازش داده‌های هر منبع
            for source_name, source_data in raw_data.get('sources', {}).items():
                if source_data.get('status') == 'success':
                    source_features = self._process_source_features(source_name, source_data['data'])
                    engineered_features['base_features'][source_name] = source_features
            
            # ایجاد ویژگی‌های ترکیبی
            engineered_features['technical_indicators'] = self._create_technical_indicators(
                engineered_features['base_features']
            )
            
            # ویژگی‌های آماری
            engineered_features['statistical_features'] = self._create_statistical_features(
                engineered_features['base_features']
            )
            
            # ویژگی‌های زمانی
            engineered_features['temporal_features'] = self._create_temporal_features()
            
            # متادیتای ویژگی‌ها
            engineered_features['feature_metadata'] = self._generate_feature_metadata(engineered_features)
            
            # ذخیره در کش
            self.cache_manager.set_data(
                "utb", 
                "engineered_features:latest", 
                engineered_features, 
                expire=1800
            )
            
            logger.info(f"✅ Engineered {self._count_features(engineered_features)} features")
            return engineered_features
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'base_features': {},
                'technical_indicators': {},
                'statistical_features': {},
                'temporal_features': {}
            }

    def _process_source_features(self, source_name: str, data: Any) -> Dict[str, Any]:
        """پردازش ویژگی‌های یک منبع خاص"""
        features = {}
        
        try:
            if source_name == 'raw_coins':
                features = self._engineer_coin_features(data)
            elif source_name == 'raw_exchanges':
                features = self._engineer_exchange_features(data)
            elif source_name == 'raw_news':
                features = self._engineer_news_features(data)
            elif source_name == 'raw_insights':
                features = self._engineer_insight_features(data)
            else:
                features = {'raw_data': data, 'processed': False}
                
        except Exception as e:
            logger.error(f"❌ Error processing {source_name} features: {e}")
            features = {'error': str(e), 'processed': False}
        
        return features

    def _engineer_coin_features(self, coin_data: Any) -> Dict[str, Any]:
        """مهندسی ویژگی‌های داده‌های کوین"""
        features = {}
        
        try:
            if isinstance(coin_data, list) and len(coin_data) > 0:
                # استخراج داده‌های عددی
                prices = [item.get('price', 0) for item in coin_data if isinstance(item, dict)]
                volumes = [item.get('volume', 0) for item in coin_data if isinstance(item, dict)]
                market_caps = [item.get('market_cap', 0) for item in coin_data if isinstance(item, dict)]
                
                if prices:
                    price_array = np.array(prices)
                    features.update({
                        'price_mean': float(np.mean(price_array)),
                        'price_std': float(np.std(price_array)),
                        'price_trend': self._calculate_trend(price_array),
                        'price_volatility': float(np.std(price_array) / np.mean(price_array)) if np.mean(price_array) > 0 else 0,
                        'price_momentum': self._calculate_momentum(price_array),
                        'support_level': self._find_support_level(price_array),
                        'resistance_level': self._find_resistance_level(price_array)
                    })
                
                if volumes:
                    volume_array = np.array(volumes)
                    features.update({
                        'volume_mean': float(np.mean(volume_array)),
                        'volume_trend': self._calculate_trend(volume_array),
                        'volume_anomaly': self._detect_volume_anomaly(volume_array)
                    })
                    
                if market_caps:
                    market_cap_array = np.array(market_caps)
                    features.update({
                        'market_cap_mean': float(np.mean(market_cap_array)),
                        'market_cap_trend': self._calculate_trend(market_cap_array)
                    })
            
            features['processed'] = True
            features['sample_size'] = len(coin_data) if isinstance(coin_data, list) else 1
            
        except Exception as e:
            logger.error(f"❌ Error engineering coin features: {e}")
            features['error'] = str(e)
            features['processed'] = False
            
        return features

    def _engineer_exchange_features(self, exchange_data: Any) -> Dict[str, Any]:
        """مهندسی ویژگی‌های داده‌های صرافی"""
        features = {'processed': True}
        
        try:
            if isinstance(exchange_data, list):
                # ویژگی‌های حجم معاملات
                volumes = [item.get('volume', 0) for item in exchange_data if isinstance(item, dict)]
                if volumes:
                    volume_array = np.array(volumes)
                    features.update({
                        'total_volume': float(np.sum(volume_array)),
                        'volume_distribution': float(np.std(volume_array) / np.mean(volume_array)) if np.mean(volume_array) > 0 else 0,
                        'top_exchange_share': float(np.max(volume_array) / np.sum(volume_array)) if np.sum(volume_array) > 0 else 0
                    })
            
            features['exchange_count'] = len(exchange_data) if isinstance(exchange_data, list) else 1
            
        except Exception as e:
            logger.error(f"❌ Error engineering exchange features: {e}")
            features['error'] = str(e)
            features['processed'] = False
            
        return features

    def _engineer_news_features(self, news_data: Any) -> Dict[str, Any]:
        """مهندسی ویژگی‌های داده‌های خبری"""
        features = {'processed': True}
        
        try:
            if isinstance(news_data, list):
                # تحلیل ساده احساسات (می‌تواند پیچیده‌تر شود)
                sentiment_scores = []
                urgency_levels = []
                
                for item in news_data:
                    if isinstance(item, dict):
                        # استخراج احساسات از عنوان (ساده)
                        title = item.get('title', '')
                        sentiment = self._analyze_sentiment(title)
                        sentiment_scores.append(sentiment)
                        
                        # سطح فوریت
                        urgency = self._assess_urgency(item)
                        urgency_levels.append(urgency)
                
                if sentiment_scores:
                    features.update({
                        'avg_sentiment': float(np.mean(sentiment_scores)),
                        'sentiment_volatility': float(np.std(sentiment_scores)),
                        'positive_news_ratio': sum(1 for s in sentiment_scores if s > 0.1) / len(sentiment_scores),
                        'urgent_news_count': sum(1 for u in urgency_levels if u > 0.7)
                    })
            
            features['news_count'] = len(news_data) if isinstance(news_data, list) else 0
            
        except Exception as e:
            logger.error(f"❌ Error engineering news features: {e}")
            features['error'] = str(e)
            features['processed'] = False
            
        return features

    def _engineer_insight_features(self, insight_data: Any) -> Dict[str, Any]:
        """مهندسی ویژگی‌های داده‌های تحلیلی"""
        features = {'processed': True}
        
        try:
            if isinstance(insight_data, list):
                confidence_scores = []
                analysis_depths = []
                
                for item in insight_data:
                    if isinstance(item, dict):
                        confidence = item.get('confidence', 0.5)
                        depth = item.get('analysis_depth', 0.5)
                        
                        confidence_scores.append(confidence)
                        analysis_depths.append(depth)
                
                if confidence_scores:
                    features.update({
                        'avg_confidence': float(np.mean(confidence_scores)),
                        'avg_analysis_depth': float(np.mean(analysis_depths)),
                        'reliable_insights_ratio': sum(1 for c in confidence_scores if c > 0.7) / len(confidence_scores),
                        'deep_analysis_ratio': sum(1 for d in analysis_depths if d > 0.7) / len(analysis_depths)
                    })
            
            features['insight_count'] = len(insight_data) if isinstance(insight_data, list) else 0
            
        except Exception as e:
            logger.error(f"❌ Error engineering insight features: {e}")
            features['error'] = str(e)
            features['processed'] = False
            
        return features

    def _create_technical_indicators(self, base_features: Dict[str, Any]) -> Dict[str, float]:
        """ایجاد اندیکاتورهای تکنیکال پیشرفته"""
        indicators = {}
        
        try:
            # استفاده از داده‌های قیمت برای اندیکاتورها
            coin_features = base_features.get('raw_coins', {})
            
            if coin_features.get('processed'):
                # شبیه‌سازی اندیکاتورهای تکنیکال
                # در عمل از کتابخانه‌هایی مانند TA-Lib استفاده می‌شود
                price_trend = coin_features.get('price_trend', 0)
                price_volatility = coin_features.get('price_volatility', 0)
                
                indicators = {
                    'rsi_signal': self._simulate_rsi(price_trend),
                    'macd_signal': self._simulate_macd(price_trend),
                    'bollinger_band_position': self._simulate_bollinger(price_volatility),
                    'stochastic_oscillator': self._simulate_stochastic(price_trend),
                    'atr_volatility': price_volatility * 100,
                    'momentum_index': coin_features.get('price_momentum', 0) * 100
                }
                
        except Exception as e:
            logger.error(f"❌ Error creating technical indicators: {e}")
            indicators['error'] = str(e)
            
        return indicators

    def _create_statistical_features(self, base_features: Dict[str, Any]) -> Dict[str, float]:
        """ایجاد ویژگی‌های آماری"""
        stats = {}
        
        try:
            # جمع‌آوری تمام مقادیر عددی
            all_values = []
            
            for source_name, features in base_features.items():
                if features.get('processed'):
                    for key, value in features.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            all_values.append(value)
            
            if all_values:
                values_array = np.array(all_values)
                stats = {
                    'global_mean': float(np.mean(values_array)),
                    'global_std': float(np.std(values_array)),
                    'global_skewness': float(self._calculate_skewness(values_array)),
                    'global_kurtosis': float(self._calculate_kurtosis(values_array)),
                    'value_range': float(np.max(values_array) - np.min(values_array)),
                    'coefficient_of_variation': float(np.std(values_array) / np.mean(values_array)) if np.mean(values_array) > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error creating statistical features: {e}")
            stats['error'] = str(e)
            
        return stats

    def _create_temporal_features(self) -> Dict[str, Any]:
        """ایجاد ویژگی‌های زمانی"""
        now = datetime.now()
        
        return {
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': 1 if now.weekday() >= 5 else 0,
            'is_market_hours': 1 if 9 <= now.hour < 17 else 0,
            'month': now.month,
            'quarter': (now.month - 1) // 3 + 1
        }

    # متدهای کمکی برای محاسبات
    def _calculate_trend(self, data: np.ndarray) -> float:
        """محاسبه روند داده‌ها"""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return float(slope / np.mean(data) if np.mean(data) > 0 else slope)

    def _calculate_momentum(self, data: np.ndarray, period: int = 5) -> float:
        """محاسبه مومنتوم"""
        if len(data) < period:
            return 0
        return float((data[-1] - data[-period]) / data[-period] if data[-period] > 0 else 0)

    def _find_support_level(self, data: np.ndarray) -> float:
        """پیداکردن سطح حمایت"""
        if len(data) < 10:
            return float(np.min(data)) if len(data) > 0 else 0
        return float(np.percentile(data, 25))

    def _find_resistance_level(self, data: np.ndarray) -> float:
        """پیداکردن سطح مقاومت"""
        if len(data) < 10:
            return float(np.max(data)) if len(data) > 0 else 0
        return float(np.percentile(data, 75))

    def _detect_volume_anomaly(self, volume_data: np.ndarray) -> float:
        """تشخیص ناهنجاری حجم"""
        if len(volume_data) < 10:
            return 0
        z_scores = np.abs((volume_data - np.mean(volume_data)) / np.std(volume_data))
        return float(np.max(z_scores))

    def _analyze_sentiment(self, text: str) -> float:
        """تحلیل ساده احساسات متن"""
        positive_words = ['صعود', 'رشد', 'سود', 'مثبت', 'قوی', 'بهبود']
        negative_words = ['نزول', 'سقوط', 'ضرر', 'منفی', 'ضعیف', 'ریزش']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5  # خنثی
            
        return positive_count / total

    def _assess_urgency(self, news_item: Dict[str, Any]) -> float:
        """ارزیابی فوریت خبر"""
        # عوامل افزایش فوریت
        urgency_factors = 0
        
        title = news_item.get('title', '').lower()
        if any(word in title for word in ['فوری', 'اورژانسی', 'حادثه', 'شکست', 'سقوط']):
            urgency_factors += 1
            
        # زمان انتشار (خبرهای جدیدتر فوری‌تر)
        published_time = news_item.get('published_at')
        if published_time:
            try:
                news_time = datetime.fromisoformat(published_time.replace('Z', '+00:00'))
                time_diff = (datetime.now() - news_time).total_seconds() / 3600  # ساعت
                if time_diff < 1:
                    urgency_factors += 1
                elif time_diff < 6:
                    urgency_factors += 0.5
            except:
                pass
                
        return min(urgency_factors / 2, 1.0)  # نرمال‌سازی به 0-1

    # شبیه‌سازی اندیکاتورهای تکنیکال
    def _simulate_rsi(self, trend: float) -> float:
        """شبیه‌سازی RSI"""
        return 50 + (trend * 1000)  # ساده‌سازی

    def _simulate_macd(self, trend: float) -> float:
        """شبیه‌سازی MACD"""
        return trend * 100

    def _simulate_bollinger(self, volatility: float) -> float:
        """شبیه‌سازی Bollinger Bands"""
        return volatility * 1000

    def _simulate_stochastic(self, trend: float) -> float:
        """شبیه‌سازی Stochastic"""
        return 50 + (trend * 500)

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """محاسبه چولگی"""
        if len(data) < 3:
            return 0
        return float(((data - np.mean(data)) ** 3).mean() / (np.std(data) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """محاسبه کشیدگی"""
        if len(data) < 4:
            return 0
        return float(((data - np.mean(data)) ** 4).mean() / (np.std(data) ** 4)) - 3

    def _generate_feature_metadata(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """تولید متادیتای ویژگی‌ها"""
        feature_count = self._count_features(features)
        
        return {
            'total_features': feature_count,
            'feature_categories': list(features.keys()),
            'engineering_time': datetime.now().isoformat(),
            'feature_quality': 'high' if feature_count > 20 else 'medium'
        }

    def _count_features(self, features: Dict[str, Any]) -> int:
        """شمارش کل ویژگی‌های تولید شده"""
        count = 0
        for category, category_features in features.items():
            if isinstance(category_features, dict):
                count += len([k for k in category_features.keys() if not k.startswith('_')])
        return count

# نمونه global
feature_engineer = FeatureEngineer()

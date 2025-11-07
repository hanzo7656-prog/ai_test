# پردازش‌گر داده‌های شبکه عصبی
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataProcessor:
    """پردازش‌گر داده‌های مالی برای شبکه‌های عصبی"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_scalers = {}
        logger.info("✅ Data Processor initialized")
    
    def process_market_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """پردازش داده‌های خام بازار برای شبکه عصبی"""
        try:
            processed = {
                'symbol': raw_data.get('symbol', 'UNKNOWN'),
                'timestamp': datetime.now().isoformat(),
                'features': {},
                'metadata': {}
            }
            
            # استخراج داده‌های اصلی
            market_data = raw_data.get('market_data', {})
            
            # ویژگی‌های قیمت
            processed['features']['price'] = market_data.get('price', 0)
            processed['features']['price_change_24h'] = market_data.get('priceChange1d', 0)
            processed['features']['volume'] = market_data.get('volume', 0)
            processed['features']['market_cap'] = market_data.get('marketCap', 0)
            processed['features']['rank'] = market_data.get('rank', 100)
            
            # ویژگی‌های تکنیکال
            technical_features = self._extract_technical_features(raw_data)
            processed['features'].update(technical_features)
            
            # ویژگی‌های زمانی
            time_features = self._extract_time_features()
            processed['features'].update(time_features)
            
            # متادیتا
            processed['metadata']['data_quality'] = self._assess_data_quality(processed['features'])
            processed['metadata']['feature_count'] = len(processed['features'])
            processed['metadata']['processing_time'] = datetime.now().isoformat()
            
            return processed
            
        except Exception as e:
            logger.error(f"خطا در پردازش داده‌های بازار: {e}")
            return self._get_default_processed_data()
    
    def _extract_technical_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """استخراج ویژگی‌های تکنیکال"""
        try:
            features = {}
            market_data = raw_data.get('market_data', {})
            
            # داده‌های تاریخی
            price_charts = raw_data.get('price_charts', {})
            prices = price_charts.get('prices', [])
            
            if prices and len(prices) > 10:
                price_values = [p[1] for p in prices if len(p) > 1]  # مقدار قیمت
                
                if len(price_values) >= 20:
                    # نوسان
                    features['volatility'] = self._calculate_volatility(price_values[-20:])
                    
                    # روند
                    trend_info = self._calculate_trend(price_values[-50:])
                    features['trend_strength'] = trend_info['strength']
                    features['trend_direction'] = trend_info['direction']
                    
                    # میانگین‌های متحرک
                    features['sma_20'] = self._calculate_sma(price_values, 20)
                    features['sma_50'] = self._calculate_sma(price_values, 50)
                    
                    # RSI ساده
                    features['rsi'] = self._calculate_simple_rsi(price_values[-15:])
                else:
                    # مقادیر پیش‌فرض برای داده‌های ناکافی
                    features.update({
                        'volatility': 0.0,
                        'trend_strength': 0.0,
                        'trend_direction': 0.0,
                        'sma_20': market_data.get('price', 0),
                        'sma_50': market_data.get('price', 0),
                        'rsi': 50.0
                    })
            else:
                # مقادیر پیش‌فرض
                features.update({
                    'volatility': 0.0,
                    'trend_strength': 0.0,
                    'trend_direction': 0.0,
                    'sma_20': market_data.get('price', 0),
                    'sma_50': market_data.get('price', 0),
                    'rsi': 50.0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"خطا در استخراج ویژگی‌های تکنیکال: {e}")
            return {}
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """محاسبه نوسان"""
        try:
            if len(prices) < 2:
                return 0.0
            
            returns = np.diff(prices) / prices[:-1]
            return float(np.std(returns) * 100)  # درصد
            
        except:
            return 0.0
    
    def _calculate_trend(self, prices: List[float]) -> Dict[str, float]:
        """محاسبه روند"""
        try:
            if len(prices) < 10:
                return {'strength': 0.0, 'direction': 0.0}
            
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            strength = abs(slope) / (np.std(prices) + 1e-8)
            direction = 1.0 if slope > 0 else -1.0 if slope < 0 else 0.0
            
            return {
                'strength': float(min(strength, 1.0)),
                'direction': direction
            }
            
        except:
            return {'strength': 0.0, 'direction': 0.0}
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """محاسبه میانگین متحرک ساده"""
        try:
            if len(prices) < period:
                return float(prices[-1]) if prices else 0.0
            return float(np.mean(prices[-period:]))
        except:
            return 0.0
    
    def _calculate_simple_rsi(self, prices: List[float]) -> float:
        """محاسبه RSI ساده"""
        try:
            if len(prices) < 2:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except:
            return 50.0
    
    def _extract_time_features(self) -> Dict[str, float]:
        """استخراج ویژگی‌های زمانی"""
        try:
            now = datetime.now()
            
            return {
                'day_of_week': now.weekday() / 6.0,  # نرمال‌سازی
                'hour_of_day': now.hour / 23.0,      # نرمال‌سازی
                'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
                'market_hours': 1.0 if 9 <= now.hour <= 17 else 0.0
            }
            
        except:
            return {}
    
    def _assess_data_quality(self, features: Dict[str, float]) -> str:
        """ارزیابی کیفیت داده‌ها"""
        try:
            missing_count = sum(1 for v in features.values() if v == 0)
            total_count = len(features)
            
            quality_ratio = (total_count - missing_count) / total_count
            
            if quality_ratio >= 0.9:
                return 'EXCELLENT'
            elif quality_ratio >= 0.7:
                return 'GOOD'
            elif quality_ratio >= 0.5:
                return 'FAIR'
            else:
                return 'POOR'
                
        except:
            return 'UNKNOWN'
    
    def _get_default_processed_data(self) -> Dict[str, Any]:
        """داده‌های پردازش شده پیش‌فرض"""
        return {
            'symbol': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'price': 0.0,
                'price_change_24h': 0.0,
                'volume': 0.0,
                'market_cap': 0.0,
                'rank': 100.0,
                'volatility': 0.0,
                'trend_strength': 0.0,
                'trend_direction': 0.0,
                'sma_20': 0.0,
                'sma_50': 0.0,
                'rsi': 50.0,
                'day_of_week': 0.0,
                'hour_of_day': 0.0,
                'is_weekend': 0.0,
                'market_hours': 0.0
            },
            'metadata': {
                'data_quality': 'POOR',
                'feature_count': 15,
                'processing_time': datetime.now().isoformat(),
                'error': True
            }
        }
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """نرمال‌سازی ویژگی‌ها"""
        try:
            normalized = {}
            
            for key, value in features.items():
                if key in ['price', 'volume', 'market_cap']:
                    # نرمال‌سازی لگاریتمی برای مقادیر بزرگ
                    normalized[key] = np.log(value + 1) / 20
                elif key in ['price_change_24h', 'volatility']:
                    # نرمال‌سازی درصدی
                    normalized[key] = value / 100
                elif key in ['rsi']:
                    # نرمال‌سازی RSI
                    normalized[key] = value / 100
                elif key in ['rank']:
                    # نرمال‌سازی رتبه
                    normalized[key] = value / 100
                elif key in ['trend_strength']:
                    # نرمال‌سازی قدرت روند
                    normalized[key] = min(value, 1.0)
                else:
                    # نرمال‌سازی پیش‌فرض
                    normalized[key] = value
            
            return normalized
            
        except Exception as e:
            logger.error(f"خطا در نرمال‌سازی ویژگی‌ها: {e}")
            return features
    
    def create_feature_vector(self, processed_data: Dict[str, Any]) -> List[float]:
        """ایجاد بردار ویژگی برای شبکه عصبی"""
        try:
            features = processed_data.get('features', {})
            normalized_features = self.normalize_features(features)
            
            # تبدیل به لیست
            feature_vector = list(normalized_features.values())
            
            # اطمینان از طول ثابت
            expected_length = 20
            if len(feature_vector) < expected_length:
                # پر کردن با صفر
                feature_vector.extend([0.0] * (expected_length - len(feature_vector)))
            elif len(feature_vector) > expected_length:
                # قطع کردن
                feature_vector = feature_vector[:expected_length]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"خطا در ایجاد بردار ویژگی: {e}")
            return [0.0] * 20
    
    def save_processed_data(self, processed_data: Dict[str, Any], filepath: str):
        """ذخیره داده‌های پردازش شده"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 داده‌های پردازش شده در {filepath} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در ذخیره داده‌های پردازش شده: {e}")
    
    def load_processed_data(self, filepath: str) -> Dict[str, Any]:
        """بارگذاری داده‌های پردازش شده"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📂 داده‌های پردازش شده از {filepath} بارگذاری شد")
            return data
            
        except Exception as e:
            logger.error(f"خطا در بارگذاری داده‌های پردازش شده: {e}")
            return self._get_default_processed_data()

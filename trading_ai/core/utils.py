# ابزارهای کمکی Trading AI
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AIUtils:
    """کلاس ابزارهای کمکی هوش مصنوعی"""
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """نرمال‌سازی داده‌ها"""
        try:
            if method == 'zscore':
                # نرمال‌سازی Z-Score
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                return (data - mean) / (std + 1e-8)
            
            elif method == 'minmax':
                # نرمال‌سازی Min-Max
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                return (data - min_val) / (max_val - min_val + 1e-8)
            
            elif method == 'robust':
                # نرمال‌سازی Robust
                median = np.median(data, axis=0)
                iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
                return (data - median) / (iqr + 1e-8)
            
            else:
                return data
                
        except Exception as e:
            logger.error(f"خطا در نرمال‌سازی داده‌ها: {e}")
            return data
    
    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> float:
        """محاسبه نوسان قیمت"""
        try:
            if len(prices) < period:
                return 0.0
            
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # نوسان سالانه
            
            return float(volatility * 100)  # درصد
            
        except Exception as e:
            logger.error(f"خطا در محاسبه نوسان: {e}")
            return 0.0
    
    @staticmethod
    def detect_trend(prices: List[float], method: str = 'linear') -> Dict[str, Any]:
        """تشخیص روند قیمت"""
        try:
            if len(prices) < 10:
                return {'trend': 'SIDEWAYS', 'strength': 0.0, 'slope': 0.0}
            
            x = np.arange(len(prices))
            
            if method == 'linear':
                # رگرسیون خطی
                slope, intercept = np.polyfit(x, prices, 1)
                trend_strength = abs(slope) / (np.std(prices) + 1e-8)
                
            elif method == 'moving_avg':
                # میانگین متحرک
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-20:])
                slope = short_ma - long_ma
                trend_strength = abs(slope) / (np.std(prices) + 1e-8)
            
            else:
                slope = prices[-1] - prices[0]
                trend_strength = abs(slope) / (np.std(prices) + 1e-8)
            
            # تعیین روند
            if slope > 0 and trend_strength > 0.1:
                trend = 'UPTREND'
            elif slope < 0 and trend_strength > 0.1:
                trend = 'DOWNTREND'
            else:
                trend = 'SIDEWAYS'
            
            return {
                'trend': trend,
                'strength': float(min(trend_strength, 1.0)),
                'slope': float(slope),
                'method': method
            }
            
        except Exception as e:
            logger.error(f"خطا در تشخیص روند: {e}")
            return {'trend': 'UNKNOWN', 'strength': 0.0, 'slope': 0.0}
    
    @staticmethod
    def create_features(market_data: Dict[str, Any]) -> List[float]:
        """ایجاد ویژگی‌های ورودی برای شبکه عصبی"""
        try:
            features = []
            
            # ویژگی‌های قیمت
            price = market_data.get('price', 0)
            price_change = market_data.get('priceChange1d', 0)
            volume = market_data.get('volume', 0)
            market_cap = market_data.get('marketCap', 0)
            
            # نرمال‌سازی و اضافه کردن ویژگی‌ها
            features.extend([
                price / 100000,  # نرمال‌سازی قیمت
                price_change / 100,  # نرمال‌سازی تغییرات
                np.log(volume + 1) / 20,  # لگاریتم حجم
                np.log(market_cap + 1) / 25,  # لگاریتم مارکت کپ
                market_data.get('rank', 100) / 100,  # رتبه
            ])
            
            # ویژگی‌های تکنیکال ساده
            if 'price_history' in market_data:
                prices = market_data['price_history']
                if len(prices) > 10:
                    volatility = AIUtils.calculate_volatility(prices[-20:])
                    trend_info = AIUtils.detect_trend(prices[-50:])
                    
                    features.extend([
                        volatility / 100,
                        trend_info['strength'],
                        1.0 if trend_info['trend'] == 'UPTREND' else 0.0,
                        1.0 if trend_info['trend'] == 'DOWNTREND' else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # پر کردن تا ۲۰ ویژگی
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]  # قطع به ۲۰ ویژگی
            
        except Exception as e:
            logger.error(f"خطا در ایجاد ویژگی‌ها: {e}")
            return [0.0] * 20
    
    @staticmethod
    def generate_cache_key(symbol: str, analysis_type: str, params: Dict = None) -> str:
        """تولید کلید کش"""
        try:
            base_key = f"{symbol}_{analysis_type}"
            
            if params:
                param_str = json.dumps(params, sort_keys=True)
                param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
                base_key += f"_{param_hash}"
            
            return base_key
            
        except Exception as e:
            logger.error(f"خطا در تولید کلید کش: {e}")
            return f"{symbol}_{analysis_type}"
    
    @staticmethod
    def calculate_confidence(scores: List[float], method: str = 'weighted') -> float:
        """محاسبه اعتماد کلی"""
        try:
            if not scores:
                return 0.0
            
            if method == 'weighted':
                # میانگین وزنی - اهمیت بیشتر به مقادیر بالا
                weights = [score ** 2 for score in scores]
                return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            elif method == 'conservative':
                # محافظه‌کارانه - کمترین مقدار
                return min(scores)
            
            elif method == 'optimistic':
                # خوش‌بینانه - بیشترین مقدار
                return max(scores)
            
            else:
                # میانگین ساده
                return sum(scores) / len(scores)
                
        except Exception as e:
            logger.error(f"خطا در محاسبه اعتماد: {e}")
            return 0.5
    
    @staticmethod
    def format_timestamp(timestamp: str = None) -> str:
        """فرمت‌دهی زمان"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp
    
    @staticmethod
    def validate_market_data(data: Dict[str, Any]) -> bool:
        """اعتبارسنجی داده‌های بازار"""
        try:
            required_fields = ['price', 'volume', 'marketCap']
            
            for field in required_fields:
                if field not in data or data[field] is None:
                    return False
            
            # بررسی مقادیر معقول
            if data['price'] <= 0 or data['volume'] < 0 or data['marketCap'] < 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"خطا در اعتبارسنجی داده‌ها: {e}")
            return False
    
    @staticmethod
    def merge_analyses(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ادغام چندین تحلیل"""
        try:
            if not analyses:
                return {'signal': 'HOLD', 'confidence': 0.3, 'sources': []}
            
            signals = []
            confidences = []
            sources = []
            
            for analysis in analyses:
                if 'signal' in analysis and 'confidence' in analysis:
                    signals.append(analysis['signal'])
                    confidences.append(analysis['confidence'])
                    sources.append(analysis.get('source', 'unknown'))
            
            if not signals:
                return {'signal': 'HOLD', 'confidence': 0.3, 'sources': []}
            
            # محاسبه سیگنال نهایی (ساده)
            signal_weights = {
                'STRONG_BUY': 2, 'BUY': 1, 'HOLD': 0, 
                'SELL': -1, 'STRONG_SELL': -2
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for signal, confidence in zip(signals, confidences):
                weight = signal_weights.get(signal, 0)
                weighted_sum += weight * confidence
                total_weight += abs(weight) * confidence
            
            if total_weight == 0:
                final_signal = 'HOLD'
                final_confidence = 0.5
            else:
                score = weighted_sum / total_weight
                
                if score > 0.6:
                    final_signal = 'STRONG_BUY'
                elif score > 0.2:
                    final_signal = 'BUY'
                elif score < -0.6:
                    final_signal = 'STRONG_SELL'
                elif score < -0.2:
                    final_signal = 'SELL'
                else:
                    final_signal = 'HOLD'
                
                final_confidence = min(abs(score) * 1.5, 1.0)
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'sources': sources,
                'component_analyses': analyses,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"خطا در ادغام تحلیل‌ها: {e}")
            return {'signal': 'HOLD', 'confidence': 0.3, 'sources': []}

# نمونه جهانی
ai_utils = AIUtils()

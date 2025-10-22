# 📁 src/data/minimal_processors/normalizer.py

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from ...utils.memory_monitor import MemoryMonitor

@dataclass
class NormalizationConfig:
    """پیکربندی نرمال‌سازی برای انواع ویژگی‌های مالی"""
    # روش‌های نرمال‌سازی مختلف برای انواع ویژگی‌ها
    methods: Dict[str, str] = None
    # پارامترهای نرمال‌سازی
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = {
                'price_': 'log_normal',      # قیمت‌ها: نرمال‌سازی لگاریتمی
                'volume_': 'robust',         # حجم: نرمال‌سازی robust
                'change_': 'minmax',         # تغییرات: min-max
                'supply_': 'standard',       # عرضه: استاندارد
                'tech_': 'minmax'            # امتیازات: min-max
            }
        
        if self.params is None:
            self.params = {
                'log_normal': {'epsilon': 1e-8},
                'robust': {'quantile_range': (25, 75)},
                'minmax': {'feature_range': (0, 1)},
                'standard': {'with_std': True}
            }

class SmartNormalizer:
    """نرمال‌سازی هوشمند ویژگی‌های مالی"""
    
    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
        self.memory_monitor = MemoryMonitor()
        self.feature_stats = {}  # آمار ویژگی‌ها برای نرمال‌سازی
        
    def fit_transform(self, features_data: Dict[str, Any]) -> Dict[str, Any]:
        """آموزش و تبدیل داده‌ها"""
        if not features_data or 'result' not in features_data:
            return {}
        
        # استخراج تمام مقادیر برای محاسبه آمار
        self._compute_feature_stats(features_data['result'])
        
        # نرمال‌سازی داده‌ها
        normalized_coins = []
        for coin_features in features_data['result']:
            normalized_coin = self._normalize_coin(coin_features)
            normalized_coins.append(normalized_coin)
        
        result = {
            'meta': features_data.get('meta', {}),
            'result': normalized_coins,
            'normalization_info': {
                'features_normalized': len(self.feature_stats),
                'methods_used': list(set(self.config.methods.values()))
            }
        }
        
        self.memory_monitor.log_memory_usage("data_normalization")
        return result
    
    def _compute_feature_stats(self, coins_data: List[Dict]):
        """محاسبه آمار ویژگی‌ها برای نرمال‌سازی"""
        feature_values = {}
        
        # جمع‌آوری مقادیر تمام ویژگی‌ها
        for coin in coins_data:
            for feature, value in coin.items():
                if isinstance(value, (int, float)) and feature not in ['coin_id', 'symbol']:
                    if feature not in feature_values:
                        feature_values[feature] = []
                    feature_values[feature].append(value)
        
        # محاسبه آمار برای هر ویژگی
        for feature, values in feature_values.items():
            values_array = np.array(values)
            self.feature_stats[feature] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'q25': np.percentile(values_array, 25),
                'q75': np.percentile(values_array, 75)
            }
    
    def _normalize_coin(self, coin_features: Dict) -> Dict:
        """نرمال‌سازی ویژگی‌های یک کوین"""
        normalized = {'coin_id': coin_features.get('coin_id', ''),
                     'symbol': coin_features.get('symbol', '')}
        
        for feature, value in coin_features.items():
            if feature in ['coin_id', 'symbol']:
                continue
                
            if feature in self.feature_stats and isinstance(value, (int, float)):
                # یافتن روش نرمال‌سازی مناسب
                method = self._get_normalization_method(feature)
                normalized_value = self._apply_normalization(value, feature, method)
                normalized[feature] = normalized_value
            else:
                normalized[feature] = value
        
        return normalized
    
    def _get_normalization_method(self, feature_name: str) -> str:
        """تعیین روش نرمال‌سازی بر اساس نام ویژگی"""
        for prefix, method in self.config.methods.items():
            if feature_name.startswith(prefix):
                return method
        return 'minmax'  # روش پیش‌فرض
    
    def _apply_normalization(self, value: float, feature_name: str, method: str) -> float:
        """اعمال نرمال‌سازی بر اساس روش انتخاب‌شده"""
        stats = self.feature_stats[feature_name]
        
        if method == 'log_normal':
            # نرمال‌سازی لگاریتمی برای قیمت‌ها
            epsilon = self.config.params['log_normal']['epsilon']
            return np.log(value + epsilon)
            
        elif method == 'robust':
            # نرمال‌سازی robust برای outlierها
            q25, q75 = stats['q25'], stats['q75']
            iqr = q75 - q25
            if iqr == 0:
                return 0.0
            return (value - stats['mean']) / iqr
            
        elif method == 'minmax':
            # نرمال‌سازی min-max
            min_val, max_val = stats['min'], stats['max']
            if max_val - min_val == 0:
                return 0.0
            return (value - min_val) / (max_val - min_val)
            
        elif method == 'standard':
            # نرمال‌سازی استاندارد
            if stats['std'] == 0:
                return 0.0
            return (value - stats['mean']) / stats['std']
            
        else:
            return value

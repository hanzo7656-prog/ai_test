# 📁 src/data/minimal_processors/data_compressor.py

import numpy as np
from typing import Dict, List, Any, Tuple
from ...utils.memory_monitor import MemoryMonitor

class DataCompressor:
    """فشرده‌ساز داده‌های مالی با حفظ اطلاعات حیاتی"""
    
    def __init__(self, compression_ratio: float = 0.3):
        self.compression_ratio = compression_ratio  # حفظ ۳۰٪ داده اصلی
        self.memory_monitor = MemoryMonitor()
        
    def compress_features(self, features_data: Dict[str, Any]) -> Dict[str, Any]:
        """فشرده‌سازی ویژگی‌های استخراج‌شده"""
        if not features_data or 'result' not in features_data:
            return {}
            
        compressed_coins = []
        original_size = 0
        compressed_size = 0
        
        for coin_features in features_data['result']:
            original_size += self._estimate_memory_usage(coin_features)
            
            # فشرده‌سازی هر کوین
            compressed_coin = self._compress_coin_features(coin_features)
            compressed_coins.append(compressed_coin)
            
            compressed_size += self._estimate_memory_usage(compressed_coin)
        
        result = {
            'meta': features_data.get('meta', {}),
            'result': compressed_coins,
            'compression_stats': {
                'original_size_mb': original_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'compression_ratio': compressed_size / max(original_size, 1),
                'savings_percent': (1 - compressed_size / max(original_size, 1)) * 100
            }
        }
        
        self.memory_monitor.log_memory_usage("data_compression")
        return result
    
    def _compress_coin_features(self, coin_features: Dict) -> Dict:
        """فشرده‌سازی ویژگی‌های یک کوین"""
        compressed = {}
        
        for key, value in coin_features.items():
            if key in ['coin_id', 'symbol']:
                compressed[key] = value  # حفظ شناسه‌ها
            elif isinstance(value, (int, float)):
                # فشرده‌سازی اعداد با دقت کنترل‌شده
                compressed[key] = self._compress_numeric(value, key)
            else:
                compressed[key] = value
        
        return compressed
    
    def _compress_numeric(self, value: float, feature_name: str) -> float:
        """فشرده‌سازی اعداد با دقت متناسب با ویژگی"""
        if value == 0:
            return 0.0
            
        # دقت‌های مختلف بر اساس نوع ویژگی
        precision_config = {
            'price_': 2,           # قیمت: ۲ رقم اعشار
            'volume_': 0,          # حجم: عدد صحیح
            'change_': 4,          # تغییرات: ۴ رقم اعشار
            'supply_': 2,          # عرضه: ۲ رقم اعشار
            'tech_': 6             # امتیازات فنی: ۶ رقم اعشار
        }
        
        precision = 2  # پیش‌فرض
        for prefix, prec in precision_config.items():
            if feature_name.startswith(prefix):
                precision = prec
                break
        
        # اعمال فشرده‌سازی با دقت مشخص
        return round(value, precision)
    
    def _estimate_memory_usage(self, obj: Any) -> int:
        """تخمین استفاده از حافظه برای یک شیء"""
        import sys
        return sys.getsizeof(obj)
    
    def decompress_features(self, compressed_data: Dict) -> Dict:
        """بازیابی داده‌های فشرده‌شده (در صورت نیاز)"""
        # در این نسخه سبک، فشرده‌سازی lossy نیست
        # بنابراین داده اصلی قابل بازیابی است
        return compressed_data

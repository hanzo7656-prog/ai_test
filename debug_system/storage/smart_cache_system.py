import functools
import gzip
import json
import pickle
from datetime import datetime
from typing import Callable, Any, Dict, Optional
import asyncio
from .cache_debugger import cache_debugger

class SmartCache:
    """سیستم کش هوشمند با فشرده‌سازی و مانیتورینگ پیشرفته"""
    
    def __init__(self):
        # استراتژی‌های بهینه‌شده
        self.cache_strategies = {
            # داده‌های پردازش شده - اولویت بالا
            'processed': {
                'base_ttl': 600,      # 10 دقیقه
                'compress_threshold': 50000,
                'priority': 'high',
                'routes': ['coins', 'exchanges', 'news', 'insights']
            },
            # داده‌های خام - اولویت پایین
            'raw': {
                'base_ttl': 300,      # 5 دقیقه
                'compress_threshold': 50000,
                'priority': 'low', 
                'routes': ['raw_coins', 'raw_exchanges', 'raw_news', 'raw_insights']
            }
        }
        
        # آمار فشرده
        self.cache_stats = {
            'total_requests': 0,
            'hits': 0,
            'misses': 0,
            'compressions': 0,
            'errors': 0,
            'bytes_saved': 0,
            'performance': {
                'avg_response_time': 0,
                'last_cleanup': None,
                'health_score': 100
            }
        }
        
        # تنظیمات بهینه
        self.compression_enabled = True
        self.max_cache_size = 25 * 1024 * 1024  # 25MB
        self.cleanup_threshold = 0.7  # 70% پر شده

    def compress_data(self, data: Any) -> tuple[bytes, bool]:
        """فشرده‌سازی هوشمند داده"""
        try:
            if not self.compression_enabled:
                return pickle.dumps(data), False
            
            serialized = pickle.dumps(data)
            if len(serialized) < 2000:  # زیر 2KB فشرده نکن
                return serialized, False
            
            compressed = gzip.compress(serialized)
            if len(compressed) >= len(serialized) * 0.85:  # اگر کمتر از 15% بهبود
                return serialized, False
            
            self.cache_stats['compressions'] += 1
            self.cache_stats['bytes_saved'] += (len(serialized) - len(compressed))
            return compressed, True
            
        except Exception as e:
            self._log_error(f"خطای فشرده‌سازی: {e}")
            return pickle.dumps(data), False

    def decompress_data(self, data: bytes, was_compressed: bool) -> Any:
        """بازیابی داده فشرده"""
        try:
            if was_compressed:
                return pickle.loads(gzip.decompress(data))
            return pickle.loads(data)
        except Exception as e:
            self._log_error(f"خطای بازیابی: {e}")
            return None

    def get_ttl(self, strategy: str, data_size: int = 0) -> int:
        """TTL هوشمند بر اساس سایز"""
        base_ttl = self.cache_strategies[strategy]['base_ttl']
        
        if data_size > 5000000:    # بالای 5MB
            return max(60, base_ttl // 3)
        elif data_size > 1000000:  # بالای 1MB  
            return max(120, base_ttl // 2)
        
        return base_ttl

    async def cleanup_if_needed(self):
        """پاک‌سازی هوشمند"""
        try:
            # شبیه‌سازی بررسی سایز - در عمل باید از redis استفاده کنی
            current_size = 0  # await self.get_actual_cache_size()
            
            if current_size > self.max_cache_size * self.cleanup_threshold:
                self._log_info("🔥 پاک‌سازی خودکار کش")
                # پاک‌سازی داده‌های raw اول
                self.cache_stats['performance']['last_cleanup'] = datetime.now().isoformat()
                
        except Exception as e:
            self._log_error(f"خطای پاک‌سازی: {e}")

    def cache_strategy(self, strategy: str):
        """دکوراتور اصلی"""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = f"{strategy}:{func.__name__}"
                start_time = datetime.now()
                
                try:
                    # چک کش
                    cached_data = cache_debugger.get_data(cache_key)
                    
                    if cached_data is not None:
                        if isinstance(cached_data, tuple):
                            data, was_compressed = cached_data
                            result = self.decompress_data(data, was_compressed)
                        else:
                            result = cached_data
                            
                        if result is not None:
                            self._update_stats(True, start_time)
                            self._log_info(f"✅ HIT: {strategy}.{func.__name__}")
                            return result
                    
                    self._update_stats(False, start_time)
                    self._log_info(f"🔄 MISS: {strategy}.{func.__name__}")
                    
                    # اجرای تابع
                    result = await func(*args, **kwargs)
                    
                    if result is not None:
                        compressed_data, was_compressed = self.compress_data(result)
                        data_size = len(compressed_data)
                        
                        if data_size < self.max_cache_size * 0.15:  # فقط اگر کمتر از 15% فضاست
                            expire = self.get_ttl(strategy, data_size)
                            cache_value = (compressed_data, was_compressed) if was_compressed else result
                            cache_debugger.set_data(cache_key, cache_value, expire)
                            
                            self._log_info(f"💾 SET: {strategy}.{func.__name__} ({expire}s)")
                            await self.cleanup_if_needed()
                    
                    return result
                    
                except Exception as e:
                    self._log_error(f"❌ ERROR: {strategy}.{func.__name__} - {e}")
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator

    def _update_stats(self, hit: bool, start_time: datetime):
        """به‌روزرسانی آمار"""
        self.cache_stats['total_requests'] += 1
        
        if hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
        
        # محاسبه زمان پاسخ
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        current_avg = self.cache_stats['performance']['avg_response_time']
        requests = self.cache_stats['total_requests']
        
        # Moving average
        self.cache_stats['performance']['avg_response_time'] = (
            (current_avg * (requests - 1) + response_time) / requests
        )

    def _log_info(self, message: str):
        """لاگ اطلاعاتی"""
        print(f"ℹ️ [Cache] {datetime.now().strftime('%H:%M:%S')} - {message}")

    def _log_error(self, message: str):
        """لاگ خطا"""
        self.cache_stats['errors'] += 1
        print(f"❌ [Cache] {datetime.now().strftime('%H:%M:%S')} - {message}")

    def get_health_status(self) -> Dict[str, Any]:
        """وضعیت سلامت فشرده برای روت مادر"""
        total = self.cache_stats['total_requests']
        hit_rate = (self.cache_stats['hits'] / total * 100) if total > 0 else 0
        error_rate = (self.cache_stats['errors'] / max(total, 1) * 100)
        
        # محاسبه امتیاز سلامت
        health_score = max(0, 100 - (error_rate * 2) - ((100 - hit_rate) / 2))
        
        return {
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 60 else 'unhealthy',
            'health_score': round(health_score, 1),
            'summary': {
                'hit_rate': round(hit_rate, 1),
                'total_requests': total,
                'avg_response_time': round(self.cache_stats['performance']['avg_response_time'], 2),
                'compression_savings': self.cache_stats['bytes_saved'],
                'strategies_active': len(self.cache_strategies)
            },
            'timestamp': datetime.now().isoformat(),
            'cache_size': '25MB',  # می‌توانی داینامیک کنی
            'compression': self.compression_enabled
        }

# نمونه گلوبال
smart_cache = SmartCache()


coins_cache = smart_cache.cache_strategy("coins")
exchanges_cache = smart_cache.cache_strategy("exchanges")
news_cache = smart_cache.cache_strategy("news") 
insights_cache = smart_cache.cache_strategy("insights")

# برای routes داده خام
raw_coins_cache = smart_cache.cache_strategy("raw_coins")
raw_exchanges_cache = smart_cache.cache_strategy("raw_exchanges")
raw_news_cache = smart_cache.cache_strategy("raw_news")
raw_insights_cache = smart_cache.cache_strategy("raw_insights")

print("✅ Smart Cache System initialized with all decorators")

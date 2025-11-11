"""
سیستم کش هوشمند - یکپارچه با cache_debugger
Smart Cache System Integrated with Cache Debugger
"""

import functools
import gzip
import pickle
from datetime import datetime
from typing import Callable, Any, Dict, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

# ایمپورت سیستم کش واقعی
try:
    from .cache_debugger import cache_debugger
    CACHE_DEBUGGER_AVAILABLE = True
    logger.info("✅ Cache Debugger integrated with Smart Cache")
except ImportError as e:
    CACHE_DEBUGGER_AVAILABLE = False
    logger.error(f"❌ Cache Debugger not available: {e}")

class SmartCache:
    """سیستم کش هوشمند با یکپارچه‌سازی کامل"""
    
    def __init__(self):
        # استراتژی‌های هوشمند
        self.cache_strategies = {
            # پردازش شده - TTL بیشتر
            'coins': {
                'base_ttl': 300,
                'description': 'داده‌های پردازش شده کوین‌ها',
                'compress_threshold': 100000,
                'priority': 'high'
            },
            'exchanges': {
                'base_ttl': 600,
                'description': 'داده‌های پردازش شده صرافی‌ها',
                'compress_threshold': 100000,
                'priority': 'high'
            },
            'news': {
                'base_ttl': 600,
                'description': 'اخبار پردازش شده',
                'compress_threshold': 50000,
                'priority': 'medium'
            },
            'insights': {
                'base_ttl': 1800,
                'description': 'تحلیل‌های پردازش شده',
                'compress_threshold': 50000,
                'priority': 'high'
            },
            
            # داده خام - TTL کمتر
            'raw_coins': {
                'base_ttl': 180,
                'description': 'داده خام کوین‌ها',
                'compress_threshold': 50000,
                'priority': 'low'
            },
            'raw_exchanges': {
                'base_ttl': 300,
                'description': 'داده خام صرافی‌ها',
                'compress_threshold': 50000,
                'priority': 'low'
            },
            'raw_news': {
                'base_ttl': 300,
                'description': 'داده خام اخبار',
                'compress_threshold': 50000,
                'priority': 'low'
            },
            'raw_insights': {
                'base_ttl': 900,
                'description': 'داده خام تحلیل‌ها',
                'compress_threshold': 50000,
                'priority': 'medium'
            }
        }
        
        # آمار واقعی سیستم
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
            },
            'strategy_stats': {}
        }
        
        # تنظیمات پیشرفته
        self.compression_enabled = True
        self.max_cache_size = 25 * 1024 * 1024  # 25MB
        
        # مقداردهی اولیه آمار استراتژی‌ها
        for strategy in self.cache_strategies.keys():
            self.cache_stats['strategy_stats'][strategy] = {
                'hits': 0, 'misses': 0, 'size': 0, 'items': 0
            }

    def compress_data(self, data: Any) -> tuple[bytes, bool]:
        """فشرده‌سازی هوشمند داده"""
        try:
            if not self.compression_enabled:
                serialized = pickle.dumps(data)
                return serialized, False
            
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            # فقط برای داده‌های بزرگ فشرده‌سازی کن
            if original_size < 2000:  # کمتر از 2KB
                return serialized, False
            
            compressed = gzip.compress(serialized)
            compressed_size = len(compressed)
            
            # اگر فشرده‌سازی موثر نبود
            if compressed_size >= original_size * 0.9:
                return serialized, False
            
            self.cache_stats['compressions'] += 1
            self.cache_stats['bytes_saved'] += (original_size - compressed_size)
            return compressed, True
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"خطای فشرده‌سازی: {e}")
            return pickle.dumps(data), False

    def decompress_data(self, data: bytes, was_compressed: bool) -> Any:
        """بازیابی داده فشرده"""
        try:
            if was_compressed:
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            return pickle.loads(data)
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"خطای بازیابی: {e}")
            return None

    def get_ttl(self, strategy: str, data_size: int = 0) -> int:
        """TTL هوشمند بر اساس استراتژی و حجم داده"""
        strategy_config = self.cache_strategies.get(strategy, {'base_ttl': 300})
        base_ttl = strategy_config['base_ttl']
        
        # کاهش TTL برای داده‌های حجیم
        if data_size > 5000000:    # بیش از 5MB
            return max(60, base_ttl // 3)
        elif data_size > 1000000:  # بیش از 1MB
            return max(120, base_ttl // 2)
        
        return base_ttl

    def _update_stats(self, strategy: str, hit: bool, data_size: int = 0, response_time: float = 0):
        """به‌روزرسانی آمار واقعی"""
        self.cache_stats['total_requests'] += 1
        
        if hit:
            self.cache_stats['hits'] += 1
            self.cache_stats['strategy_stats'][strategy]['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
            self.cache_stats['strategy_stats'][strategy]['misses'] += 1
        
        # محاسبه میانگین زمان پاسخ
        current_avg = self.cache_stats['performance']['avg_response_time']
        total_requests = self.cache_stats['total_requests']
        
        if total_requests == 1:
            self.cache_stats['performance']['avg_response_time'] = response_time
        else:
            self.cache_stats['performance']['avg_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
        
        # به‌روزرسانی سایز داده
        if data_size > 0 and not hit:
            self.cache_stats['strategy_stats'][strategy]['size'] += data_size
            self.cache_stats['strategy_stats'][strategy]['items'] += 1

    def cache_strategy(self, strategy: str):
        """دکوراتور اصلی با یکپارچه‌سازی واقعی"""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not CACHE_DEBUGGER_AVAILABLE:
                    # Fallback: اجرای ساده بدون کش
                    return await func(*args, **kwargs)
                
                start_time = datetime.now()
                cache_key = f"{strategy}:{func.__module__}:{func.__name__}"
                
                try:
                    # چک کش در cache_debugger واقعی
                    cached_data = cache_debugger.get_data(cache_key)
                    
                    if cached_data is not None:
                        # داده ممکن است فشرده باشد
                        if isinstance(cached_data, tuple) and len(cached_data) == 2:
                            data, was_compressed = cached_data
                            result = self.decompress_data(data, was_compressed)
                        else:
                            result = cached_data
                            
                        if result is not None:
                            response_time = (datetime.now() - start_time).total_seconds() * 1000
                            self._update_stats(strategy, True, 0, response_time)
                            logger.info(f"✅ Cache HIT: {strategy}.{func.__name__}")
                            return result
                    
                    # Cache MISS
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_stats(strategy, False, 0, response_time)
                    logger.info(f"🔄 Cache MISS: {strategy}.{func.__name__}")
                    
                    # اجرای تابع اصلی
                    result = await func(*args, **kwargs)
                    
                    # ذخیره در کش
                    if result is not None:
                        # فشرده‌سازی اگر لازم باشد
                        compressed_data, was_compressed = self.compress_data(result)
                        data_size = len(compressed_data)
                        
                        # محاسبه TTL هوشمند
                        expire = self.get_ttl(strategy, data_size)
                        
                        # ذخیره در cache_debugger واقعی
                        cache_value = (compressed_data, was_compressed) if was_compressed else result
                        cache_debugger.set_data(cache_key, cache_value, expire)
                        
                        logger.info(f"💾 Cache SET: {strategy}.{func.__name__} ({expire}s, {data_size} bytes, compressed: {was_compressed})")
                    
                    return result
                    
                except Exception as e:
                    self.cache_stats['errors'] += 1
                    logger.error(f"❌ Cache ERROR in {strategy}.{func.__name__}: {e}")
                    # Fallback: اجرای تابع بدون کش
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator

    def get_health_status(self) -> Dict[str, Any]:
        """گزارش سلامت واقعی بر اساس آمار"""
        total_requests = self.cache_stats['total_requests']
        
        # محاسبه hit rate واقعی
        if total_requests > 0:
            hit_rate = (self.cache_stats['hits'] / total_requests) * 100
        else:
            hit_rate = 0
        
        # محاسبه امتیاز سلامت
        health_score = 100
        
        # کسر بر اساس خطاها
        error_rate = (self.cache_stats['errors'] / max(total_requests, 1)) * 100
        health_score -= min(30, error_rate * 3)
        
        # کسر بر اساس hit rate پایین
        if hit_rate < 50:
            health_score -= (50 - hit_rate) / 2
        
        health_score = max(0, min(100, health_score))
        
        # وضعیت کلی
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "health_score": round(health_score, 1),
            "summary": {
                "hit_rate": round(hit_rate, 1),
                "total_requests": total_requests,
                "avg_response_time": round(self.cache_stats['performance']['avg_response_time'], 2),
                "compression_savings": self.cache_stats['bytes_saved'],
                "strategies_active": len(self.cache_strategies)
            },
            "timestamp": datetime.now().isoformat(),
            "cache_size": "25MB",
            "compression": self.compression_enabled,
            "detailed_stats": {
                "hits": self.cache_stats['hits'],
                "misses": self.cache_stats['misses'],
                "compressions": self.cache_stats['compressions'],
                "errors": self.cache_stats['errors'],
                "strategy_breakdown": self.cache_stats['strategy_stats']
            }
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """آمار کامل سیستم کش"""
        return {
            "timestamp": datetime.now().isoformat(),
            "smart_cache_stats": self.cache_stats,
            "strategies": self.cache_strategies,
            "settings": {
                "compression_enabled": self.compression_enabled,
                "max_cache_size": f"{self.max_cache_size / 1024 / 1024}MB",
                "cache_debugger_available": CACHE_DEBUGGER_AVAILABLE
            }
        }

    def clear_cache(self):
        """پاک‌سازی کش (شبیه‌سازی)"""
        # در عمل، این باید با cache_debugger هماهنگ شود
        self.cache_stats = {
            'total_requests': 0,
            'hits': 0,
            'misses': 0,
            'compressions': 0,
            'errors': 0,
            'bytes_saved': 0,
            'performance': {'avg_response_time': 0, 'last_cleanup': datetime.now().isoformat(), 'health_score': 100},
            'strategy_stats': {s: {'hits': 0, 'misses': 0, 'size': 0, 'items': 0} for s in self.cache_strategies}
        }
        logger.info("🧹 Smart Cache statistics cleared")

# ایجاد نمونه اصلی
smart_cache = SmartCache()

# 🔽 دکوراتورهای از پیش تعریف شده برای ۸ فایل route

# برای routes پردازش شده
coins_cache = smart_cache.cache_strategy("coins")
exchanges_cache = smart_cache.cache_strategy("exchanges")
news_cache = smart_cache.cache_strategy("news")
insights_cache = smart_cache.cache_strategy("insights")

# برای routes داده خام
raw_coins_cache = smart_cache.cache_strategy("raw_coins")
raw_exchanges_cache = smart_cache.cache_strategy("raw_exchanges")
raw_news_cache = smart_cache.cache_strategy("raw_news")
raw_insights_cache = smart_cache.cache_strategy("raw_insights")

logger.info("🚀 Smart Cache System Initialized - Full Integration with Cache Debugger")

# 🔽 export برای استفاده در فایل‌های روت
__all__ = [
    "SmartCache", "smart_cache",
    "coins_cache", "exchanges_cache", "news_cache", "insights_cache",
    "raw_coins_cache", "raw_exchanges_cache", "raw_news_cache", "raw_insights_cache"
]

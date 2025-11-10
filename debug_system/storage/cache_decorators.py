import functools
import hashlib
import json
from typing import Any, Callable
from .cache_debugger import cache_debugger

def cache_response(expire: int = 300, key_prefix: str = ""):
    """
    دکوراتور برای کش کردن خودکار پاسخ endpointها
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # تولید کلید کش یکتا بر اساس تابع و پارامترها
            cache_key = generate_cache_key(func, key_prefix, *args, **kwargs)
            
            # چک کردن کش
            cached_result = cache_debugger.get_data(cache_key)
            if cached_result is not None:
                print(f"✅ Cache HIT: {func.__name__}")
                return cached_result
            
            # اجرای تابع اصلی
            result = await func(*args, **kwargs)
            
            # ذخیره در کش
            if result is not None:
                cache_debugger.set_data(cache_key, result, expire)
                print(f"💾 Cache SET: {func.__name__} ({expire}s)")
            
            return result
        return wrapper
    return decorator

def generate_cache_key(func: Callable, prefix: str, *args, **kwargs) -> str:
    """تولید کلید کش یکتا"""
    # استفاده از نام تابع و پارامترها
    key_data = {
        'func': func.__name__,
        'module': func.__module__,
        'args': str(args),
        'kwargs': str(sorted(kwargs.items()))
    }
    
    key_string = f"{prefix}:{json.dumps(key_data, sort_keys=True)}"
    return hashlib.md5(key_string.encode()).hexdigest()

# دکوراتورهای از پیش تنظیم شده برای انواع مختلف داده
def cache_coins(expire: int = 300):
    """دکوراتور مخصوص داده‌های کوین"""
    return cache_response(expire=expire, key_prefix="coins")

def cache_news(expire: int = 600):
    """دکوراتور مخصوص اخبار"""
    return cache_response(expire=expire, key_prefix="news")

def cache_insights(expire: int = 1800):
    """دکوراتور مخصوص تحلیل‌ها"""
    return cache_response(expire=expire, key_prefix="insights")

def cache_exchanges(expire: int = 600):
    """دکوراتور مخصوص صرافی‌ها"""
    return cache_response(expire=expire, key_prefix="exchanges")

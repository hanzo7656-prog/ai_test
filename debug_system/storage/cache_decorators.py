import functools
import hashlib
import json
from typing import Any, Callable
from .cache_debugger import cache_debugger

def cache_response(expire: int = 300, key_prefix: str = ""):
    """دکوراتور اصلی برای کش کردن"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(func, key_prefix, *args, **kwargs)
            
            cached_result = cache_debugger.get_data(cache_key)
            if cached_result is not None:
                print(f"✅ Cache HIT: {func.__name__}")
                return cached_result
            
            result = await func(*args, **kwargs)
            
            if result is not None:
                cache_debugger.set_data(cache_key, result, expire)
                print(f"💾 Cache SET: {func.__name__} ({expire}s)")
            
            return result
        return wrapper
    return decorator

def generate_cache_key(func: Callable, prefix: str, *args, **kwargs) -> str:
    key_data = {
        'func': func.__name__,
        'module': func.__module__,
        'args': str(args),
        'kwargs': str(sorted(kwargs.items()))
    }
    key_string = f"{prefix}:{json.dumps(key_data, sort_keys=True)}"
    return hashlib.md5(key_string.encode()).hexdigest()

# 🔽 دکوراتورهای مخصوص ۸ فایل route

# برای routes پردازش شده (۴ فایل)
def cache_coins(expire: int = 300):
    """دکوراتور مخصوص coins.py (پردازش شده)"""
    return cache_response(expire=expire, key_prefix="coins")

def cache_news(expire: int = 600):
    """دکوراتور مخصوص news.py (پردازش شده)"""
    return cache_response(expire=expire, key_prefix="news")

def cache_insights(expire: int = 1800):
    """دکوراتور مخصوص insights.py (پردازش شده)"""
    return cache_response(expire=expire, key_prefix="insights")

def cache_exchanges(expire: int = 600):
    """دکوراتور مخصوص exchanges.py (پردازش شده)"""
    return cache_response(expire=expire, key_prefix="exchanges")

# برای routes خام (۴ فایل) - TTL کمتر چون داده خام هست
def cache_raw_coins(expire: int = 180):  # ۳ دقیقه برای داده خام
    """دکوراتور مخصوص raw_coins.py (داده خام)"""
    return cache_response(expire=expire, key_prefix="raw_coins")

def cache_raw_news(expire: int = 300):   # ۵ دقیقه برای داده خام
    """دکوراتور مخصوص raw_news.py (داده خام)"""
    return cache_response(expire=expire, key_prefix="raw_news")

def cache_raw_insights(expire: int = 900):  # ۱۵ دقیقه برای داده خام
    """دکوراتور مخصوص raw_insights.py (داده خام)"""
    return cache_response(expire=expire, key_prefix="raw_insights")

def cache_raw_exchanges(expire: int = 300):  # ۵ دقیقه برای داده خام
    """دکوراتور مخصوص raw_exchanges.py (داده خام)"""
    return cache_response(expire=expire, key_prefix="raw_exchanges")

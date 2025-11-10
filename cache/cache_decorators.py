from functools import wraps
from .redis_manager import redis_manager
import hashlib
import json

def generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """تولید کلید کش یکتا بر اساس پارامترها"""
    key_parts = [func_name]
    
    # اضافه کردن args
    for arg in args:
        key_parts.append(str(arg))
    
    # اضافه کردن kwargs
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    full_key = ":".join(key_parts)
    return hashlib.md5(full_key.encode()).hexdigest()

def cache_response(expire: int = 300):
    """دکوریتور برای کش کردن پاسخ توابع"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # تولید کلید کش
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            
            # چک کردن کش
            cached_result = redis_manager.get(cache_key)
            if cached_result is not None:
                print(f"✅ Cache HIT for {func.__name__}")
                return cached_result
            
            # اجرای تابع اصلی
            result = await func(*args, **kwargs)
            
            # ذخیره در کش
            if result is not None:
                redis_manager.set(cache_key, result, expire)
                print(f"💾 Cache SET for {func.__name__} (expire: {expire}s)")
            
            return result
        return wrapper
    return decorator

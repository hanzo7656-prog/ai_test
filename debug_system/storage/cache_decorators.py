import functools
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

# ایمپورت صحیح از ماژول همسطح
try:
    # روش ۱: ایمپورت نسبی
    from .cache_debugger import cache_debugger
except ImportError:
    try:
        # روش ۲: ایمپورت مطلق  
        from debug_system.storage.cache_debugger import cache_debugger
    except ImportError:
        # روش ۳: Fallback برای توسعه
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from storage.cache_debugger import cache_debugger

# بقیه کد بدون تغییر...
# نقشه‌نگاری دیتابیس‌ها برای انواع مختلف داده
DATABASE_MAPPING = {
    # داده‌های پردازش شده AI - UTB
    'coins': 'utb',
    'news': 'utb', 
    'insights': 'utb',
    'exchanges': 'utb',
    
    # داده‌های خام - UTC
    'raw_coins': 'utc',
    'raw_news': 'utc',
    'raw_insights': 'utc',
    'raw_exchanges': 'utc',
    
    # داده‌های مدل AI - UTA
    'model_predictions': 'uta',
    'ai_analysis': 'uta',
    'technical_signals': 'uta',
    
    # داده‌های سیستم - MOTHER_A
    'user_data': 'mother_a',
    'system_config': 'mother_a',
    'transactions': 'mother_a',
    
    # کش عملیاتی - MOTHER_B
    'page_cache': 'mother_b',
    'session_data': 'mother_b',
    'temp_cache': 'mother_b',
    
    # آرشیو تاریخی - UTC
    'archive': 'utc',
    'historical': 'utc'
}

def cache_response(expire: int = 300, key_prefix: str = "", database: str = None):
    """دکوراتور اصلی برای کش کردن با پشتیبانی از ۵ دیتابیس"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # تعیین دیتابیس بر اساس prefix یا مقدار explicit
            target_db = database or DATABASE_MAPPING.get(key_prefix, 'utb')
            
            cache_key = generate_cache_key(func, key_prefix, *args, **kwargs)
            
            # تلاش برای دریافت از کش
            cached_result = cache_debugger.get_data(target_db, cache_key)
            if cached_result is not None:
                print(f"✅ Cache HIT: {func.__name__} [DB: {target_db.upper()}]")
                return cached_result
            
            # اجرای تابع اصلی
            result = await func(*args, **kwargs)
            
            # ذخیره نتیجه در کش
            if result is not None:
                cache_debugger.set_data(target_db, cache_key, result, expire)
                print(f"💾 Cache SET: {func.__name__} [DB: {target_db.upper()}, TTL: {expire}s]")
            
            return result
        return wrapper
    return decorator

def cache_with_archive(realtime_ttl: int = 300, archive_ttl: int = 365*24*3600, 
                      archive_strategy: str = "hourly", key_prefix: str = ""):
    """دکوراتور برای کش موقت + آرشیو تاریخی"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # تعیین دیتابیس
            target_db = DATABASE_MAPPING.get(key_prefix, 'utc')
            
            # کلید کش موقت
            realtime_key = generate_cache_key(func, f"realtime_{key_prefix}", *args, **kwargs)
            
            # ۱. بررسی کش موقت
            cached_realtime = cache_debugger.get_data(target_db, realtime_key)
            if cached_realtime is not None:
                print(f"✅ Realtime Cache HIT: {func.__name__} [DB: {target_db.upper()}]")
                return cached_realtime
            
            # ۲. اجرای تابع اصلی
            result = await func(*args, **kwargs)
            if result is None:
                return None
            
            # ۳. ذخیره در کش موقت
            cache_debugger.set_data(target_db, realtime_key, result, realtime_ttl)
            print(f"💾 Realtime Cache SET: {func.__name__} [DB: {target_db.upper()}, TTL: {realtime_ttl}s]")
            
            # ۴. ذخیره در آرشیو تاریخی (همیشه در UTC)
            archive_key = generate_archive_key(func, archive_strategy, key_prefix, *args, **kwargs)
            archive_data = {
                'timestamp': datetime.now().isoformat(),
                'data': result,
                'metadata': {
                    'function': func.__name__,
                    'prefix': key_prefix,
                    'strategy': archive_strategy,
                    'realtime_ttl': realtime_ttl,
                    'archive_ttl': archive_ttl
                }
            }
            
            cache_debugger.set_data("utc", archive_key, archive_data, archive_ttl)
            print(f"📦 Historical Archive SET: {func.__name__} [Strategy: {archive_strategy}, TTL: {archive_ttl}s]")
            
            return result
        return wrapper
    return decorator

def generate_cache_key(func: Callable, prefix: str, *args, **kwargs) -> str:
    """تولید کلید یکتا برای کش"""
    # فیلتر کردن آرگومان‌های غیرقابل سریال‌سازی
    filtered_kwargs = {}
    for k, v in kwargs.items():
        try:
            json.dumps(v)
            filtered_kwargs[k] = v
        except:
            filtered_kwargs[k] = str(v)
    
    key_data = {
        'func': func.__name__,
        'module': func.__module__,
        'args': str(args),
        'kwargs': str(sorted(filtered_kwargs.items()))
    }
    key_string = f"{prefix}:{json.dumps(key_data, sort_keys=True)}"
    return hashlib.md5(key_string.encode()).hexdigest()

def generate_archive_key(func: Callable, strategy: str, prefix: str, *args, **kwargs) -> str:
    """تولید کلید برای آرشیو تاریخی"""
    timestamp = datetime.now()
    
    if strategy == "minutely":
        time_part = timestamp.strftime("%Y%m%d_%H%M")
    elif strategy == "hourly":
        time_part = timestamp.strftime("%Y%m%d_%H")
    elif strategy == "daily":
        time_part = timestamp.strftime("%Y%m%d")
    elif strategy == "weekly":
        time_part = timestamp.strftime("%Y%W")
    else:  # monthly
        time_part = timestamp.strftime("%Y%m")
    
    base_key = generate_cache_key(func, prefix, *args, **kwargs)
    return f"archive:{strategy}:{prefix}:{time_part}:{base_key}"

# ==================== دکوراتورهای مخصوص ۸ فایل route با آرشیو ====================

# 🔽 برای routes داده‌های خام + آرشیو تاریخی (UTC)
def cache_raw_coins_with_archive():
    """دکوراتور مخصوص raw_coins.py - کش ۳ دقیقه + آرشیو ساعتی"""
    return cache_with_archive(
        realtime_ttl=180,           # ۳ دقیقه برای داده لحظه‌ای
        archive_ttl=30*24*3600,     # آرشیو ۳۰ روزه
        archive_strategy="hourly",  # ذخیره ساعتی
        key_prefix="raw_coins"
    )

def cache_raw_news_with_archive():
    """دکوراتور مخصوص raw_news.py - کش ۵ دقیقه + آرشیو روزانه"""
    return cache_with_archive(
        realtime_ttl=300,           # ۵ دقیقه
        archive_ttl=90*24*3600,     # آرشیو ۳ ماهه
        archive_strategy="daily",   # ذخیره روزانه
        key_prefix="raw_news"
    )

def cache_raw_insights_with_archive():
    """دکوراتور مخصوص raw_insights.py - کش ۱۵ دقیقه + آرشیو روزانه"""
    return cache_with_archive(
        realtime_ttl=900,           # ۱۵ دقیقه
        archive_ttl=180*24*3600,    # آرشیو ۶ ماهه
        archive_strategy="daily",   # ذخیره روزانه
        key_prefix="raw_insights"
    )

def cache_raw_exchanges_with_archive():
    """دکوراتور مخصوص raw_exchanges.py - کش ۵ دقیقه + آرشیو ساعتی"""
    return cache_with_archive(
        realtime_ttl=300,           # ۵ دقیقه
        archive_ttl=30*24*3600,     # آرشیو ۳۰ روزه
        archive_strategy="hourly",  # ذخیره ساعتی
        key_prefix="raw_exchanges"
    )

# 🔽 برای routes داده‌های پردازش شده + آرشیو (UTB)
def cache_coins_with_archive():
    """دکوراتور مخصوص coins.py - کش ۱۰ دقیقه + آرشیو روزانه"""
    return cache_with_archive(
        realtime_ttl=600,           # ۱۰ دقیقه
        archive_ttl=365*24*3600,    # آرشیو ۱ ساله
        archive_strategy="daily",   # ذخیره روزانه
        key_prefix="coins"
    )

def cache_news_with_archive():
    """دکوراتور مخصوص news.py - کش ۱۰ دقیقه + آرشیو هفتگی"""
    return cache_with_archive(
        realtime_ttl=600,           # ۱۰ دقیقه
        archive_ttl=180*24*3600,    # آرشیو ۶ ماهه
        archive_strategy="weekly",  # ذخیره هفتگی
        key_prefix="news"
    )

def cache_insights_with_archive():
    """دکوراتور مخصوص insights.py - کش ۱ ساعت + آرشیو هفتگی"""
    return cache_with_archive(
        realtime_ttl=3600,          # ۱ ساعت
        archive_ttl=365*24*3600,    # آرشیو ۱ ساله
        archive_strategy="weekly",  # ذخیره هفتگی
        key_prefix="insights"
    )

def cache_exchanges_with_archive():
    """دکوراتور مخصوص exchanges.py - کش ۱۰ دقیقه + آرشیو روزانه"""
    return cache_with_archive(
        realtime_ttl=600,           # ۱۰ دقیقه
        archive_ttl=180*24*3600,    # آرشیو ۶ ماهه
        archive_strategy="daily",   # ذخیره روزانه
        key_prefix="exchanges"
    )

# ==================== دکوراتورهای اصلی (بدون آرشیو) ====================

def cache_coins(expire: int = 600):
    """دکوراتور مخصوص coins.py (پردازش شده) - UTB"""
    return cache_response(expire=expire, key_prefix="coins", database="utb")

def cache_news(expire: int = 600):
    """دکوراتور مخصوص news.py (پردازش شده) - UTB"""
    return cache_response(expire=expire, key_prefix="news", database="utb")

def cache_insights(expire: int = 3600):
    """دکوراتور مخصوص insights.py (پردازش شده) - UTB"""
    return cache_response(expire=expire, key_prefix="insights", database="utb")

def cache_exchanges(expire: int = 600):
    """دکوراتور مخصوص exchanges.py (پردازش شده) - UTB"""
    return cache_response(expire=expire, key_prefix="exchanges", database="utb")

def cache_raw_coins(expire: int = 180):
    """دکوراتور مخصوص raw_coins.py (داده خام) - UTC"""
    return cache_response(expire=expire, key_prefix="raw_coins", database="utc")

def cache_raw_news(expire: int = 300):
    """دکوراتور مخصوص raw_news.py (داده خام) - UTC"""
    return cache_response(expire=expire, key_prefix="raw_news", database="utc")

def cache_raw_insights(expire: int = 900):
    """دکوراتور مخصوص raw_insights.py (داده خام) - UTC"""
    return cache_response(expire=expire, key_prefix="raw_insights", database="utc")

def cache_raw_exchanges(expire: int = 300):
    """دکوراتور مخصوص raw_exchanges.py (داده خام) - UTC"""
    return cache_response(expire=expire, key_prefix="raw_exchanges", database="utc")

# ==================== متدهای مدیریت آرشیو تاریخی ====================

def get_historical_data(function_name: str, prefix: str, start_date: str, end_date: str, 
                       strategy: str = "daily") -> List[Dict[str, Any]]:
    """دریافت داده‌های تاریخی از آرشیو"""
    historical_results = []
    
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        current = start
        while current <= end:
            if strategy == "hourly":
                # برای داده‌های ساعتی، تمام ساعات روز را بررسی کنید
                for hour in range(24):
                    time_part = current.strftime(f"%Y%m%d_{hour:02d}")
                    archive_pattern = f"archive:{strategy}:{prefix}:{time_part}:*"
                    keys = cache_debugger.get_keys("utc", archive_pattern)[0]
                    
                    for key in keys:
                        data = cache_debugger.get_data("utc", key)
                        if data and data.get('metadata', {}).get('function') == function_name:
                            historical_results.append(data)
            else:
                time_part = current.strftime("%Y%m%d")
                archive_pattern = f"archive:{strategy}:{prefix}:{time_part}:*"
                keys = cache_debugger.get_keys("utc", archive_pattern)[0]
                
                for key in keys:
                    data = cache_debugger.get_data("utc", key)
                    if data and data.get('metadata', {}).get('function') == function_name:
                        historical_results.append(data)
            
            current += timedelta(days=1)
        
        # مرتب‌سازی بر اساس timestamp
        historical_results.sort(key=lambda x: x.get('timestamp', ''))
        
    except Exception as e:
        print(f"❌ Error retrieving historical data: {e}")
    
    return historical_results

def get_archive_stats(prefix: str = None) -> Dict[str, Any]:
    """آمار داده‌های آرشیو شده"""
    archive_pattern = "archive:*" if not prefix else f"archive:*:{prefix}:*"
    archive_keys = cache_debugger.get_keys("utc", archive_pattern)[0]
    
    stats = {
        'total_archives': len(archive_keys),
        'by_strategy': defaultdict(int),
        'by_prefix': defaultdict(int),
        'by_function': defaultdict(int),
        'oldest_archive': None,
        'newest_archive': None,
        'total_size_mb': 0
    }
    
    for key in archive_keys:
        try:
            parts = key.split(':')
            if len(parts) >= 4:
                strategy = parts[1]
                archive_prefix = parts[2] if len(parts) > 2 else "unknown"
                time_part = parts[3] if len(parts) > 3 else "unknown"
                
                stats['by_strategy'][strategy] += 1
                stats['by_prefix'][archive_prefix] += 1
                
                # محاسبه اندازه تقریبی
                data = cache_debugger.get_data("utc", key)
                if data:
                    stats['total_size_mb'] += len(json.dumps(data)) / (1024 * 1024)
                    
                    function_name = data.get('metadata', {}).get('function', 'unknown')
                    stats['by_function'][function_name] += 1
                
                # به روزرسانی قدیمی‌ترین و جدیدترین
                if time_part != "unknown":
                    if not stats['oldest_archive'] or time_part < stats['oldest_archive']:
                        stats['oldest_archive'] = time_part
                    if not stats['newest_archive'] or time_part > stats['newest_archive']:
                        stats['newest_archive'] = time_part
        
        except Exception as e:
            print(f"❌ Error processing archive key {key}: {e}")
    
    stats['total_size_mb'] = round(stats['total_size_mb'], 2)
    return stats

def cleanup_old_archives(days_old: int = 365):
    """پاک کردن آرشیوهای قدیمی"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archive_keys = cache_debugger.get_keys("utc", "archive:*")[0]
        
        deleted_count = 0
        for key in archive_keys:
            try:
                parts = key.split(':')
                if len(parts) >= 4:
                    time_part = parts[3]
                    # تبدیل به datetime (فرمت: YYYYMMDD یا YYYYMMDD_HH)
                    if '_' in time_part:
                        archive_date = datetime.strptime(time_part.split('_')[0], "%Y%m%d")
                    else:
                        archive_date = datetime.strptime(time_part, "%Y%m%d")
                    
                    if archive_date < cutoff_date:
                        cache_debugger.delete_data("utc", key)
                        deleted_count += 1
            except:
                continue
        
        print(f"🧹 Cleaned up {deleted_count} archives older than {days_old} days")
        return deleted_count
        
    except Exception as e:
        print(f"❌ Error cleaning up old archives: {e}")
        return 0

# ==================== دکوراتورهای پیشرفته ====================

def cache_with_fallback(fallback_func: Callable = None, expire: int = 300, 
                       database: str = "utb", use_archive: bool = False):
    """دکوراتور با قابلیت fallback و آرشیو اختیاری"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(func, "fallback", *args, **kwargs)
            
            try:
                # تلاش برای دریافت از کش
                cached_result = cache_debugger.get_data(database, cache_key)
                if cached_result is not None:
                    print(f"✅ Cache HIT (Fallback): {func.__name__}")
                    return cached_result
                
                # اجرای تابع اصلی
                result = await func(*args, **kwargs)
                
                # ذخیره در کش
                if result is not None:
                    cache_debugger.set_data(database, cache_key, result, expire)
                    
                    # ذخیره در آرشیو اگر فعال باشد
                    if use_archive:
                        archive_key = generate_archive_key(func, "daily", "fallback", *args, **kwargs)
                        archive_data = {
                            'timestamp': datetime.now().isoformat(),
                            'data': result,
                            'metadata': {'function': func.__name__, 'type': 'fallback'}
                        }
                        cache_debugger.set_data("utc", archive_key, archive_data, 30*24*3600)
                
                return result
                
            except Exception as e:
                print(f"❌ Error in {func.__name__}: {e}")
                
                # استفاده از fallback
                if fallback_func:
                    print(f"🔄 Using fallback for {func.__name__}")
                    fallback_result = fallback_func(*args, **kwargs)
                    
                    # ذخیره نتیجه fallback در کش
                    if fallback_result is not None:
                        cache_debugger.set_data(database, cache_key, fallback_result, expire // 2)
                    
                    return fallback_result
                else:
                    # تلاش برای دریافت آخرین داده معتبر از کش
                    cached_result = cache_debugger.get_data(database, cache_key)
                    if cached_result is not None:
                        print(f"🔄 Using cached data as fallback for {func.__name__}")
                        return cached_result
                    raise e
        return wrapper
    return decorator

# utility function برای مدیریت دستی کش
def clear_cache_pattern(pattern: str, database: str = None):
    """پاک کردن کش بر اساس الگو"""
    from redis_manager import redis_manager
    
    if database:
        # پاک کردن از دیتابیس مشخص
        keys, _ = redis_manager.get_keys(database, pattern)
        for key in keys:
            redis_manager.delete(database, key)
        print(f"🧹 Cleared {len(keys)} keys from {database} matching pattern: {pattern}")
    else:
        # پاک کردن از تمام دیتابیس‌ها
        total_cleared = 0
        for db_name in ['uta', 'utb', 'utc', 'mother_a', 'mother_b']:
            keys, _ = redis_manager.get_keys(db_name, pattern)
            for key in keys:
                redis_manager.delete(db_name, key)
            total_cleared += len(keys)
            if keys:
                print(f"🧹 Cleared {len(keys)} keys from {db_name}")
        print(f"✅ Total cleared: {total_cleared} keys")

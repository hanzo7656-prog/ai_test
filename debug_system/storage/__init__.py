"""
Debug System Storage Package
مدیریت پیشرفته کش، تاریخچه، لاگ
"""

__version__ = "1.0.0"
__author__ = "Debug System Team"

# ==================== ایمپورت‌های پایه (بدون وابستگی) ====================
from .redis_manager import RedisCacheManager, redis_manager
from .log_manager import LogManager, log_manager
from .history_manager import HistoryManager, history_manager

# ==================== ایمپورت‌های سطح دوم (وابسته به پایه) ====================
from .cache_debugger import CacheDebugger, cache_debugger

# ==================== Lazy Import برای ماژول‌های وابسته ====================
_cache_decorators_module = None
_smart_cache_module = None

def _get_cache_decorators():
    """Lazy import برای cache_decorators"""
    global _cache_decorators_module
    if _cache_decorators_module is None:
        from . import cache_decorators as module
        _cache_decorators_module = module
    return _cache_decorators_module

def _get_smart_cache():
    """Lazy import برای smart_cache_system"""
    global _smart_cache_module
    if _smart_cache_module is None:
        try:
            from . import smart_cache_system as module
            _smart_cache_module = module
        except ImportError as e:
            print(f"⚠️ Smart cache system not available: {e}")
            _smart_cache_module = None
    return _smart_cache_module

# ==================== Lazy Attributes برای دکوراتورها ====================

# دکوراتورهای اصلی
@property
def cache_response():
    return _get_cache_decorators().cache_response

@property
def cache_with_archive():
    return _get_cache_decorators().cache_with_archive

@property
def cache_with_fallback():
    return _get_cache_decorators().cache_with_fallback

# توابع کمکی
@property
def generate_cache_key():
    return _get_cache_decorators().generate_cache_key

@property
def generate_archive_key():
    return _get_cache_decorators().generate_archive_key

@property
def get_historical_data():
    return _get_cache_decorators().get_historical_data

@property
def get_archive_stats():
    return _get_cache_decorators().get_archive_stats

@property
def cleanup_old_archives():
    return _get_cache_decorators().cleanup_old_archives

# دکوراتورهای مخصوص route ها (بدون آرشیو)
@property
def cache_coins():
    return _get_cache_decorators().cache_coins

@property
def cache_news():
    return _get_cache_decorators().cache_news

@property
def cache_insights():
    return _get_cache_decorators().cache_insights

@property
def cache_exchanges():
    return _get_cache_decorators().cache_exchanges

@property
def cache_raw_coins():
    return _get_cache_decorators().cache_raw_coins

@property
def cache_raw_news():
    return _get_cache_decorators().cache_raw_news

@property
def cache_raw_insights():
    return _get_cache_decorators().cache_raw_insights

@property
def cache_raw_exchanges():
    return _get_cache_decorators().cache_raw_exchanges

# دکوراتورهای با آرشیو
@property
def cache_coins_with_archive():
    return _get_cache_decorators().cache_coins_with_archive

@property
def cache_news_with_archive():
    return _get_cache_decorators().cache_news_with_archive

@property
def cache_insights_with_archive():
    return _get_cache_decorators().cache_insights_with_archive

@property
def cache_exchanges_with_archive():
    return _get_cache_decorators().cache_exchanges_with_archive

@property
def cache_raw_coins_with_archive():
    return _get_cache_decorators().cache_raw_coins_with_archive

@property
def cache_raw_news_with_archive():
    return _get_cache_decorators().cache_raw_news_with_archive

@property
def cache_raw_insights_with_archive():
    return _get_cache_decorators().cache_raw_insights_with_archive

@property
def cache_raw_exchanges_with_archive():
    return _get_cache_decorators().cache_raw_exchanges_with_archive

# نقشه‌نگاری دیتابیس
@property
def DATABASE_MAPPING():
    return _get_cache_decorators().DATABASE_MAPPING

# Smart Cache System
@property
def CacheOptimizationEngine():
    module = _get_smart_cache()
    if module:
        return module.CacheOptimizationEngine
    else:
        # Fallback class
        class FallbackOptimizationEngine:
            def get_health_status(self):
                return {"status": "not_available", "error": "Module not found"}
        return FallbackOptimizationEngine

@property
def cache_optimizer():
    module = _get_smart_cache()
    if module:
        return module.cache_optimizer
    else:
        return CacheOptimizationEngine()

# ==================== صادرات عمومی ====================

__all__ = [
    # کلاس‌های اصلی (ایمپورت مستقیم)
    'CacheDebugger',
    'HistoryManager', 
    'LogManager',
    'RedisCacheManager',
    
    # نمونه‌های گلوبال (ایمپورت مستقیم)
    'cache_debugger',
    'history_manager',
    'log_manager', 
    'redis_manager',
    
    # دکوراتورهای کش (Lazy)
    'cache_response',
    'cache_with_archive',
    'cache_with_fallback',
    
    # توابع کمکی (Lazy)
    'generate_cache_key',
    'generate_archive_key',
    'get_historical_data',
    'get_archive_stats',
    'cleanup_old_archives',
    
    # دکوراتورهای مخصوص (Lazy)
    'cache_coins', 'cache_news', 'cache_insights', 'cache_exchanges',
    'cache_raw_coins', 'cache_raw_news', 'cache_raw_insights', 'cache_raw_exchanges',
    'cache_coins_with_archive', 'cache_news_with_archive',
    'cache_insights_with_archive', 'cache_exchanges_with_archive', 
    'cache_raw_coins_with_archive', 'cache_raw_news_with_archive',
    'cache_raw_insights_with_archive', 'cache_raw_exchanges_with_archive',
    
    # نقشه‌نگاری (Lazy)
    'DATABASE_MAPPING',
    
    # Smart Cache (Lazy)
    'CacheOptimizationEngine',
    'cache_optimizer'
]

# ==================== تابع مقداردهی اولیه ====================

def initialize_storage_systems():
    """مقداردهی اولیه سیستم‌های ذخیره‌سازی"""
    print("🔄 Initializing storage systems...")
    
    # فعال کردن lazy imports برای اطمینان از لود شدن
    _ = cache_response
    _ = cache_optimizer
    
    status = {
        'cache_debugger': 'ready',
        'history_manager': 'ready',
        'log_manager': 'ready', 
        'redis_manager': 'ready',
        'cache_decorators': 'ready',
        'cache_optimizer': 'ready' if _get_smart_cache() else 'basic'
    }
    
    print("✅ Storage systems initialized successfully")
    return status

# ==================== پیام راه‌اندازی ====================

print(f"✅ Debug System Storage v{__version__} initialized")
print("📦 Available modules: cache_debugger, cache_decorators, history_manager, log_manager, redis_manager, smart_cache_system")
print("🔧 Storage system configured with lazy loading")

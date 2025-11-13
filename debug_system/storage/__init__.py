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

# ==================== Lazy Accessor Functions ====================

def get_cache_response():
    return _get_cache_decorators().cache_response

def get_cache_with_archive():
    return _get_cache_decorators().cache_with_archive

def get_cache_with_fallback():
    return _get_cache_decorators().cache_with_fallback

def get_generate_cache_key():
    return _get_cache_decorators().generate_cache_key

def get_generate_archive_key():
    return _get_cache_decorators().generate_archive_key

def get_historical_data():
    return _get_cache_decorators().get_historical_data

def get_archive_stats():
    return _get_cache_decorators().get_archive_stats

def get_cleanup_old_archives():
    return _get_cache_decorators().cleanup_old_archives

# دکوراتورهای مخصوص route ها
def get_cache_coins():
    return _get_cache_decorators().cache_coins

def get_cache_news():
    return _get_cache_decorators().cache_news

def get_cache_insights():
    return _get_cache_decorators().cache_insights

def get_cache_exchanges():
    return _get_cache_decorators().cache_exchanges

def get_cache_raw_coins():
    return _get_cache_decorators().cache_raw_coins

def get_cache_raw_news():
    return _get_cache_decorators().cache_raw_news

def get_cache_raw_insights():
    return _get_cache_decorators().cache_raw_insights

def get_cache_raw_exchanges():
    return _get_cache_decorators().cache_raw_exchanges

# دکوراتورهای با آرشیو
def get_cache_coins_with_archive():
    return _get_cache_decorators().cache_coins_with_archive

def get_cache_news_with_archive():
    return _get_cache_decorators().cache_news_with_archive

def get_cache_insights_with_archive():
    return _get_cache_decorators().cache_insights_with_archive

def get_cache_exchanges_with_archive():
    return _get_cache_decorators().cache_exchanges_with_archive

def get_cache_raw_coins_with_archive():
    return _get_cache_decorators().cache_raw_coins_with_archive

def get_cache_raw_news_with_archive():
    return _get_cache_decorators().cache_raw_news_with_archive

def get_cache_raw_insights_with_archive():
    return _get_cache_decorators().cache_raw_insights_with_archive

def get_cache_raw_exchanges_with_archive():
    return _get_cache_decorators().cache_raw_exchanges_with_archive

def get_database_mapping():
    return _get_cache_decorators().DATABASE_MAPPING

def get_cache_optimizer():
    module = _get_smart_cache()
    if module:
        return module.cache_optimizer
    else:
        # Fallback
        class FallbackOptimizer:
            def get_health_status(self):
                return {"status": "fallback", "message": "Smart cache not available"}
        return FallbackOptimizer()

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
    
    # توابع دسترسی (Lazy)
    'get_cache_response',
    'get_cache_with_archive', 
    'get_cache_with_fallback',
    'get_generate_cache_key',
    'get_generate_archive_key',
    'get_historical_data',
    'get_archive_stats',
    'get_cleanup_old_archives',
    'get_cache_coins', 'get_cache_news', 'get_cache_insights', 'get_cache_exchanges',
    'get_cache_raw_coins', 'get_cache_raw_news', 'get_cache_raw_insights', 'get_cache_raw_exchanges',
    'get_cache_coins_with_archive', 'get_cache_news_with_archive',
    'get_cache_insights_with_archive', 'get_cache_exchanges_with_archive',
    'get_cache_raw_coins_with_archive', 'get_cache_raw_news_with_archive',
    'get_cache_raw_insights_with_archive', 'get_cache_raw_exchanges_with_archive',
    'get_database_mapping',
    'get_cache_optimizer'
]

# ==================== تابع مقداردهی اولیه ====================

def initialize_storage_system():
    """مقداردهی اولیه سیستم‌های ذخیره‌سازی"""
    print("🔄 Initializing storage system...")
    
    # تست اتصال به Redis
    try:
        redis_status = redis_manager.health_check()
        print(f"✅ Redis connections: {len([k for k, v in redis_status.items() if v.get('status') == 'connected'])}/5")
    except Exception as e:
        print(f"❌ Redis initialization failed: {e}")
    
    # فعال کردن lazy imports
    _ = get_cache_response()
    _ = get_cache_optimizer()
    
    status = {
        'cache_debugger': 'ready',
        'history_manager': 'ready', 
        'log_manager': 'ready',
        'redis_manager': 'ready',
        'cache_decorators': 'ready',
        'cache_optimizer': 'ready' if _get_smart_cache() else 'basic'
    }
    
    print("✅ Storage system initialized successfully")
    return status

# ==================== پیام راه‌اندازی ====================

print(f"✅ Debug System Storage v{__version__} initialized")
print("📦 Available modules: cache_debugger, cache_decorators, history_manager, log_manager, redis_manager, smart_cache_system")
print("🔧 Storage system configured with lazy loading")

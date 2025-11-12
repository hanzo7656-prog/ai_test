"""
Debug System Storage Package
مدیریت پیشرفته کش، تاریخچه، لاگ
"""

__version__ = "1.0.0"
__author__ = "Debug System Team"

# ایمپورت ماژول‌های موجود
from .cache_debugger import CacheDebugger, cache_debugger
from .cache_decorators import (
    cache_response,
    cache_with_archive,
    cache_with_fallback,
    generate_cache_key,
    generate_archive_key,
    get_historical_data,
    get_archive_stats,
    cleanup_old_archives,
    
    # دکوراتورهای مخصوص route ها
    cache_coins, cache_news, cache_insights, cache_exchanges,
    cache_raw_coins, cache_raw_news, cache_raw_insights, cache_raw_exchanges,
    
    # دکوراتورهای با آرشیو
    cache_coins_with_archive, cache_news_with_archive,
    cache_insights_with_archive, cache_exchanges_with_archive,
    cache_raw_coins_with_archive, cache_raw_news_with_archive,
    cache_raw_insights_with_archive, cache_raw_exchanges_with_archive,
    
    # نقشه‌نگاری دیتابیس
    DATABASE_MAPPING
)
from .history_manager import HistoryManager, history_manager
from .log_manager import LogManager, log_manager
from .redis_manager import RedisCacheManager, redis_manager

# ایمپورت smart_cache_system (که قبلاً cache_optimizer بود)
try:
    from .smart_cache_system import CacheOptimizationEngine, cache_optimizer
    OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Smart cache system not available: {e}")
    # ایجاد stub
    class CacheOptimizationEngine:
        def get_health_status(self):
            return {"status": "not_available", "error": "Module not found"}
    cache_optimizer = CacheOptimizationEngine()
    OPTIMIZER_AVAILABLE = False

# صادرات عمومی
__all__ = [
    # کلاس‌های اصلی
    'CacheDebugger',
    'HistoryManager', 
    'LogManager',
    'RedisCacheManager',
    'CacheOptimizationEngine',
    
    # نمونه‌های گلوبال
    'cache_debugger',
    'history_manager',
    'log_manager', 
    'redis_manager',
    'cache_optimizer',
    
    # دکوراتورهای کش
    'cache_response',
    'cache_with_archive',
    'cache_with_fallback',
    
    # توابع کمکی
    'generate_cache_key',
    'generate_archive_key',
    'get_historical_data',
    'get_archive_stats',
    'cleanup_old_archives',
    
    # دکوراتورهای مخصوص
    'cache_coins', 'cache_news', 'cache_insights', 'cache_exchanges',
    'cache_raw_coins', 'cache_raw_news', 'cache_raw_insights', 'cache_raw_exchanges',
    'cache_coins_with_archive', 'cache_news_with_archive',
    'cache_insights_with_archive', 'cache_exchanges_with_archive', 
    'cache_raw_coins_with_archive', 'cache_raw_news_with_archive',
    'cache_raw_insights_with_archive', 'cache_raw_exchanges_with_archive',
    
    # نقشه‌نگاری
    'DATABASE_MAPPING'
]

def initialize_storage_systems():  # 🔥 نام تابع درست شده
    """مقداردهی اولیه سیستم‌های ذخیره‌سازی"""
    print("✅ Storage systems initialized")
    return {
        'cache_debugger': 'ready',
        'cache_decorators': 'ready', 
        'history_manager': 'ready',
        'log_manager': 'ready',
        'redis_manager': 'ready',
        'cache_optimizer': 'ready' if OPTIMIZER_AVAILABLE else 'basic'
    }

# پیام راه‌اندازی
print(f"✅ Debug System Storage v{__version__} initialized")
print("📦 Available modules: cache_debugger, cache_decorators, history_manager, log_manager, redis_manager, smart_cache_system")

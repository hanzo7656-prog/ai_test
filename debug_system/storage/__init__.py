"""
Debug System Storage Package
مدیریت پیشرفته کش، تاریخچه، لاگ و بهینه‌سازی

ماژول‌ها:
- cache_debugger: مانیتورینگ و دیباگ کش
- cache_decorators: دکوراتورهای هوشمند کش
- history_manager: مدیریت تاریخچه و آرشیو
- log_manager: سیستم لاگینگ پیشرفته  
- redis_manager: مدیریت اتصال به Redis
- smart_cache_system: آنالیز و بهینه‌سازی هوشمند
"""

__version__ = "1.0.0"
__author__ = "Debug System Team"

# ایمپورت ماژول‌های اصلی
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
from .smart_cache_system import CacheOptimizationEngine, cache_optimizer

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

# اطلاعات پکیج
PACKAGE_INFO = {
    "name": "debug-system-storage",
    "version": __version__,
    "description": "Advanced caching, monitoring and optimization system for debug infrastructure",
    "modules": [
        "cache_debugger - Real-time cache monitoring and analytics",
        "cache_decorators - Intelligent caching decorators with archive support", 
        "history_manager - Historical data and metrics storage",
        "log_manager - Advanced logging system with compression",
        "redis_manager - Multi-database Redis connection management",
        "cache_optimizer - AI-powered cache optimization engine"
    ],
    "databases": {
        "uta": "AI Model Core - Critical data",
        "utb": "AI Processing - Semi-critical data", 
        "utc": "Raw Data - Historical + Compressed",
        "mother_a": "System Processing - Critical data",
        "mother_b": "Operations & Cache - Temporary data"
    }
}

def get_package_info():
    """دریافت اطلاعات پکیج"""
    return PACKAGE_INFO.copy()

def initialize_storage_systems():
    """
    مقداردهی اولیه تمام سیستم‌های ذخیره‌سازی
    برای استفاده در startup برنامه
    """
    systems_status = {}
    
    try:
        # بررسی اتصال Redis
        redis_health = redis_manager.health_check()
        systems_status['redis'] = {
            'status': 'connected' if all(
                db.get('status') == 'connected' 
                for db in redis_health.values()
            ) else 'partial',
            'details': redis_health
        }
        
        # بررسی دیتابیس تاریخچه
        history_manager._init_database()
        systems_status['history_db'] = {'status': 'initialized'}
        
        # بررسی سیستم لاگ
        systems_status['log_system'] = {'status': 'active'}
        
        # بررسی سیستم بهینه‌سازی
        optimizer_health = cache_optimizer.get_health_status()
        systems_status['optimizer'] = optimizer_health
        
        # لاگ وضعیت
        log_manager.log_system_metrics({
            'component': 'storage_package',
            'action': 'initialization',
            'status': 'completed',
            'systems_status': systems_status,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
        
    except Exception as e:
        systems_status['error'] = str(e)
        # لاگ خطا
        log_manager.log_system_metrics({
            'component': 'storage_package',
            'action': 'initialization',
            'status': 'failed',
            'error': str(e),
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
    
    return systems_status

# پیام راه‌اندازی
print(f"✅ Debug System Storage v{__version__} initialized")
print("📦 Available modules: cache_debugger, cache_decorators, history_manager, log_manager, redis_manager, cache_optimizer")

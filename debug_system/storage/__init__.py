"""
VortexAI Storage System
Complete cache and storage management with 5 Redis databases
"""

__version__ = "1.0.0"
__author__ = "VortexAI Team"

# ایمپورت ماژول‌های اصلی storage
from .cache_debugger import CacheDebugger, cache_debugger
from .history_manager import HistoryManager, history_manager
from .log_manager import LogManager, log_manager
from .redis_manager import RedisCacheManager, redis_manager

# ایمپورت دکوراتورهای کش
from .cache_decorators import (
    # دکوراتورهای اصلی
    cache_response,
    cache_with_archive,
    
    # دکوراتورهای با آرشیو
    cache_coins_with_archive,
    cache_news_with_archive,
    cache_insights_with_archive,
    cache_exchanges_with_archive,
    cache_raw_coins_with_archive,
    cache_raw_news_with_archive,
    cache_raw_insights_with_archive,
    cache_raw_exchanges_with_archive,
    
    # دکوراتورهای ساده
    cache_coins,
    cache_news,
    cache_insights,
    cache_exchanges,
    cache_raw_coins,
    cache_raw_news,
    cache_raw_insights,
    cache_raw_exchanges,
    
    # متدهای مدیریت آرشیو
    get_historical_data,
    get_archive_stats,
    cleanup_old_archives,
    
    # دکوراتورهای پیشرفته
    cache_with_fallback,
    clear_cache_pattern
)

# ایمپورت Smart Cache System
try:
    from .smart_cache_system import CacheOptimizationEngine, cache_optimizer
    SMART_CACHE_AVAILABLE = True
except ImportError:
    SMART_CACHE_AVAILABLE = False
    cache_optimizer = None

# ایمپورت Unified Cache Manager
try:
    from .unified_cache_manager import UnifiedCacheManager, unified_cache_manager
    UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    UNIFIED_CACHE_AVAILABLE = False
    unified_cache_manager = None

def initialize_storage_system():
    """راه‌اندازی کامل سیستم storage"""
    try:
        print("🔄 Initializing Storage System...")
        
        # بررسی اتصال Redis
        redis_status = redis_manager.health_check()
        print(f"🎯 Redis Status: {redis_status.get('status', 'unknown')}")
        
        # راه‌اندازی ماژول‌ها
        storage_system = {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger,
            "redis_manager": redis_manager,
            "smart_cache": cache_optimizer if SMART_CACHE_AVAILABLE else "Not Available",
            "unified_cache_manager": unified_cache_manager if UNIFIED_CACHE_AVAILABLE else "Not Available"
        }
        
        # بررسی وضعیت کلی
        overall_status = "degraded"
        if redis_status.get("status") == "connected":
            if SMART_CACHE_AVAILABLE or UNIFIED_CACHE_AVAILABLE:
                overall_status = "advanced"
            else:
                overall_status = "basic"
        
        print(f"✅ Storage system initialized with Smart Cache integration")
        print(f"    - Log Manager: {type(log_manager).__name__}")
        print(f"    - History Manager: {type(history_manager).__name__}")
        print(f"    - Cache Debugger: {type(cache_debugger).__name__}")
        print(f"    - Redis Manager: {type(redis_manager).__name__}")
        print(f"    - Smart Cache: {'Available' if SMART_CACHE_AVAILABLE else 'Not Available'}")
        print(f"    - Unified Cache Manager: {'Available' if UNIFIED_CACHE_AVAILABLE else 'Not Available'}")
        print(f"    - Overall Cache Status: {overall_status}")
        
        # گزارش وضعیت جزئی
        status_details = {
            "smart_cache": "available" if SMART_CACHE_AVAILABLE else "not_available",
            "legacy_cache": "available",
            "redis": redis_status.get("status", "unknown")
        }
        
        for component, status in status_details.items():
            print(f"      - {component}: {status}")
        
        return storage_system
        
    except Exception as e:
        print(f"❌ Storage system initialization failed: {e}")
        # بازگشت حداقل سیستم
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger,
            "redis_manager": redis_manager,
            "smart_cache": "Not Available",
            "unified_cache_manager": "Not Available"
        }

# راه‌اندازی خودکار
storage_system = initialize_storage_system()

__all__ = [
    # ماژول‌های اصلی
    "CacheDebugger", "cache_debugger",
    "HistoryManager", "history_manager", 
    "LogManager", "log_manager",
    "RedisCacheManager", "redis_manager",
    
    # دکوراتورهای کش
    "cache_response",
    "cache_with_archive",
    
    # دکوراتورهای با آرشیو
    "cache_coins_with_archive",
    "cache_news_with_archive", 
    "cache_insights_with_archive",
    "cache_exchanges_with_archive",
    "cache_raw_coins_with_archive",
    "cache_raw_news_with_archive",
    "cache_raw_insights_with_archive", 
    "cache_raw_exchanges_with_archive",
    
    # دکوراتورهای ساده
    "cache_coins",
    "cache_news",
    "cache_insights",
    "cache_exchanges", 
    "cache_raw_coins",
    "cache_raw_news",
    "cache_raw_insights",
    "cache_raw_exchanges",
    
    # مدیریت آرشیو
    "get_historical_data",
    "get_archive_stats", 
    "cleanup_old_archives",
    
    # پیشرفته
    "cache_with_fallback",
    "clear_cache_pattern",
    
    # Smart Cache
    "CacheOptimizationEngine", "cache_optimizer",
    
    # Unified Cache
    "UnifiedCacheManager", "unified_cache_manager",
    
    # Initialization
    "initialize_storage_system", "storage_system"
]

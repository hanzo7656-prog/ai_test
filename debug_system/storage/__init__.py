"""
Debug System Storage Modules - Updated for Smart Cache Integration
Data persistence and history management for debug system
"""

import logging
from typing import Dict, Any, Optional
from ..core import debug_manager, metrics_collector
from .log_manager import LogManager
from .history_manager import HistoryManager
from .cache_debugger import CacheDebugger
from .redis_manager import RedisCacheManager

logger = logging.getLogger(__name__)

# 🔽 ایمپورت سیستم کش هوشمند جدید
try:
    from smart_cache_system import smart_cache, SmartCache
    SMART_CACHE_AVAILABLE = True
    logger.info("✅ Smart Cache System detected - integrating...")
except ImportError:
    SMART_CACHE_AVAILABLE = False
    smart_cache = None
    logger.warning("⚠️ Smart Cache System not available - using legacy cache")

# ایجاد نمونه‌های storage
log_manager = LogManager()
history_manager = HistoryManager()
cache_debugger = CacheDebugger()
redis_manager = RedisCacheManager()

class UnifiedCacheManager:
    """مدیریت یکپارچه کش - ترکیب سیستم قدیم و جدید"""
    
    def __init__(self):
        self.smart_cache_available = SMART_CACHE_AVAILABLE
        self.smart_cache = smart_cache
        self.legacy_cache = cache_debugger
        self.redis_manager = redis_manager
        
    def health_check(self) -> Dict[str, Any]:
        """بررسی سلامت تمام سیستم‌های کش"""
        health_report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "systems": {}
        }
        
        # وضعیت Smart Cache
        if self.smart_cache_available:
            try:
                smart_health = self.smart_cache.get_health_status()
                health_report["systems"]["smart_cache"] = {
                    "status": "available",
                    "health": smart_health,
                    "health_score": smart_health.get("health_score", 0)
                }
            except Exception as e:
                health_report["systems"]["smart_cache"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            health_report["systems"]["smart_cache"] = {
                "status": "not_available"
            }
        
        # وضعیت Legacy Cache
        try:
            legacy_stats = self.legacy_cache.get_cache_stats()
            health_report["systems"]["legacy_cache"] = {
                "status": "available",
                "stats": legacy_stats
            }
        except Exception as e:
            health_report["systems"]["legacy_cache"] = {
                "status": "error", 
                "error": str(e)
            }
        
        # وضعیت Redis
        try:
            redis_health = self.redis_manager.health_check()
            health_report["systems"]["redis"] = redis_health
        except Exception as e:
            health_report["systems"]["redis"] = {
                "status": "error",
                "error": str(e)
            }
        
        # محاسبه وضعیت کلی
        available_systems = [
            system for system in health_report["systems"].values() 
            if system.get("status") in ["available", "healthy"]
        ]
        
        health_report["overall_status"] = (
            "healthy" if len(available_systems) >= 2 else
            "degraded" if len(available_systems) >= 1 else "unhealthy"
        )
        
        return health_report
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """آمار ترکیبی از تمام سیستم‌های کش"""
        stats = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "cache_systems": {}
        }
        
        # آمار Smart Cache
        if self.smart_cache_available:
            try:
                smart_stats = self.smart_cache.get_cache_stats()
                stats["cache_systems"]["smart_cache"] = smart_stats
            except Exception as e:
                stats["cache_systems"]["smart_cache"] = {"error": str(e)}
        
        # آمار Legacy Cache
        try:
            legacy_stats = self.legacy_cache.get_cache_stats()
            stats["cache_systems"]["legacy_cache"] = legacy_stats
        except Exception as e:
            stats["cache_systems"]["legacy_cache"] = {"error": str(e)}
        
        # آمار Redis
        try:
            redis_stats = self.redis_manager.get_stats()
            stats["cache_systems"]["redis"] = redis_stats
        except Exception as e:
            stats["cache_systems"]["redis"] = {"error": str(e)}
        
        return stats
    
    def clear_all_caches(self) -> Dict[str, Any]:
        """پاک‌سازی تمام کش‌ها"""
        results = {}
        
        # پاک‌سازی Smart Cache
        if self.smart_cache_available:
            try:
                # اگر تابع پاک‌سازی داره
                if hasattr(self.smart_cache, 'clear_cache'):
                    self.smart_cache.clear_cache()
                    results["smart_cache"] = "cleared"
                else:
                    results["smart_cache"] = "no_clear_method"
            except Exception as e:
                results["smart_cache"] = f"error: {e}"
        
        # پاک‌سازی Legacy Cache
        try:
            self.legacy_cache.clear_old_operations(days=0)
            results["legacy_cache"] = "cleared"
        except Exception as e:
            results["legacy_cache"] = f"error: {e}"
        
        # پاک‌سازی Redis
        try:
            # اگر تابع پاک‌سازی کلی داره
            if hasattr(self.redis_manager, 'clear_all'):
                self.redis_manager.clear_all()
                results["redis"] = "cleared"
            else:
                results["redis"] = "no_clear_method"
        except Exception as e:
            results["redis"] = f"error: {e}"
        
        return {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "results": results
        }

# ایجاد مدیر یکپارچه
unified_cache_manager = UnifiedCacheManager()

def initialize_storage_system():
    """راه‌اندازی و ارتباط سیستم‌های ذخیره‌سازی - نسخه آپدیت شده"""
    try:
        # راه‌اندازی سیستم ذخیره‌سازی
        logger.info("✅ Storage system initialized with Smart Cache integration")
        logger.info(f"   - Log Manager: {type(log_manager).__name__}")
        logger.info(f"   - History Manager: {type(history_manager).__name__}")
        logger.info(f"   - Cache Debugger: {type(cache_debugger).__name__}")
        logger.info(f"   - Redis Manager: {type(redis_manager).__name__}")
        logger.info(f"   - Smart Cache: {'Available' if SMART_CACHE_AVAILABLE else 'Not Available'}")
        logger.info(f"   - Unified Cache Manager: {type(unified_cache_manager).__name__}")
        
        # تست سلامت سیستم‌های کش
        cache_health = unified_cache_manager.health_check()
        logger.info(f"   - Overall Cache Status: {cache_health['overall_status']}")
        
        # گزارش وضعیت هر سیستم
        for system_name, system_info in cache_health["systems"].items():
            status = system_info.get("status", "unknown")
            logger.info(f"     - {system_name}: {status}")
        
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger,
            "redis_manager": redis_manager,
            "smart_cache": smart_cache if SMART_CACHE_AVAILABLE else None,
            "unified_cache_manager": unified_cache_manager,
            "smart_cache_available": SMART_CACHE_AVAILABLE
        }
    except Exception as e:
        logger.error(f"❌ Storage system initialization failed: {e}")
        # بازگرداندن حداقل ماژول‌ها حتی در صورت خطا
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger,
            "redis_manager": redis_manager,
            "smart_cache": None,
            "unified_cache_manager": unified_cache_manager,
            "smart_cache_available": False
        }

# راه‌اندازی خودکار
storage_system = initialize_storage_system()

# 🔽 importهای دکوراتورها (اگر نیاز داری)
try:
    from .cache_decorators import (
        cache_response, 
        cache_coins, cache_news, cache_insights, cache_exchanges,
        cache_raw_coins, cache_raw_news, cache_raw_insights, cache_raw_exchanges,
        generate_cache_key
    )
    CACHE_DECORATORS_AVAILABLE = True
except ImportError:
    CACHE_DECORATORS_AVAILABLE = False
    logger.warning("⚠️ Cache decorators not available")

# 🔽 دکوراتورهای هوشمند جدید
try:
    if SMART_CACHE_AVAILABLE:
        from smart_cache_system import (
            coins_cache, exchanges_cache, news_cache, insights_cache,
            raw_coins_cache, raw_exchanges_cache, raw_news_cache, raw_insights_cache
        )
        SMART_DECORATORS_AVAILABLE = True
        logger.info("✅ Smart Cache decorators imported")
    else:
        SMART_DECORATORS_AVAILABLE = False
except ImportError:
    SMART_DECORATORS_AVAILABLE = False

__all__ = [
    # ماژول‌های اصلی
    "LogManager", "log_manager",
    "HistoryManager", "history_manager", 
    "CacheDebugger", "cache_debugger",
    "RedisCacheManager", "redis_manager",
    
    # سیستم کش هوشمند
    "SmartCache", "smart_cache",
    "UnifiedCacheManager", "unified_cache_manager",
    
    # فلگ‌های وضعیت
    "SMART_CACHE_AVAILABLE",
    "CACHE_DECORATORS_AVAILABLE", 
    "SMART_DECORATORS_AVAILABLE",
    
    # توابع
    "initialize_storage_system", "storage_system"
]

# 🔽 اضافه کردن دکوراتورهای قدیمی اگر موجود هستند
if CACHE_DECORATORS_AVAILABLE:
    __all__.extend([
        "cache_response", 
        "cache_coins", "cache_news", "cache_insights", "cache_exchanges",
        "cache_raw_coins", "cache_raw_news", "cache_raw_insights", "cache_raw_exchanges",
        "generate_cache_key"
    ])

# 🔽 اضافه کردن دکوراتورهای هوشمند اگر موجود هستند
if SMART_DECORATORS_AVAILABLE:
    __all__.extend([
        "coins_cache", "exchanges_cache", "news_cache", "insights_cache",
        "raw_coins_cache", "raw_exchanges_cache", "raw_news_cache", "raw_insights_cache"
    ])

logger.info("🎯 Storage system updated successfully with Smart Cache integration")

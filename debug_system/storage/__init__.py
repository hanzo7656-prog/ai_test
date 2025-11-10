"""
Debug System Storage Modules
Data persistence and history management for debug system
"""

import logging
from ..core import debug_manager, metrics_collector
from .log_manager import LogManager
from .history_manager import HistoryManager
from .cache_debugger import CacheDebugger
from .redis_manager import RedisCacheManager  # 🆕 اضافه کردن

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های storage با Signatureهای صحیح
log_manager = LogManager()  # ✅ بدون پارامتر - طبق تعریف اصلی
history_manager = HistoryManager()  # ✅ بدون پارامتر - طبق تعریف اصلی
cache_debugger = CacheDebugger()  # ✅ بدون پارامتر - طبق تعریف اصلی
redis_manager = RedisCacheManager()  # 🆕 جدید - مدیر مستقل Redis

def initialize_storage_system():
    """راه‌اندازی و ارتباط سیستم‌های ذخیره‌سازی"""
    try:
        # راه‌اندازی سیستم ذخیره‌سازی
        logger.info("✅ Storage system initialized")
        logger.info(f"   - Log Manager: {type(log_manager).__name__}")
        logger.info(f"   - History Manager: {type(history_manager).__name__}")
        logger.info(f"   - Cache Debugger: {type(cache_debugger).__name__}")
        logger.info(f"   - Redis Manager: {type(redis_manager).__name__}")
        
        # تست اتصال Redis
        redis_health = redis_manager.health_check()
        logger.info(f"   - Redis Status: {redis_health.get('status', 'unknown')}")
        
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger,
            "redis_manager": redis_manager  # 🆕 اضافه کردن
        }
    except Exception as e:
        logger.error(f"❌ Storage system initialization failed: {e}")
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger,
            "redis_manager": redis_manager  # 🆕 اضافه کردن
        }

# راه‌اندازی خودکار
storage_system = initialize_storage_system()

__all__ = [
    "LogManager", "log_manager",
    "HistoryManager", "history_manager", 
    "CacheDebugger", "cache_debugger",
    "RedisCacheManager", "redis_manager",  # 🆕 اضافه کردن
    "initialize_storage_system", "storage_system"
]

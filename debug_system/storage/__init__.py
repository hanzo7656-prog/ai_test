"""
Debug System Storage Modules
Data persistence and history management for debug system
"""

import logging
from ..core import debug_manager, metrics_collector
from .log_manager import LogManager
from .history_manager import HistoryManager
from .cache_debugger import CacheDebugger

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های storage با Signatureهای صحیح
log_manager = LogManager()  # ✅ بدون پارامتر - طبق تعریف اصلی
history_manager = HistoryManager()  # ✅ بدون پارامتر - طبق تعریف اصلی
cache_debugger = CacheDebugger()  # ✅ بدون پارامتر - طبق تعریف اصلی

def initialize_storage_system():
    """راه‌اندازی و ارتباط سیستم‌های ذخیره‌سازی"""
    try:
        # راه‌اندازی سیستم ذخیره‌سازی
        logger.info("✅ Storage system initialized")
        logger.info(f"   - Log Manager: {type(log_manager).__name__}")
        logger.info(f"   - History Manager: {type(history_manager).__name__}")
        logger.info(f"   - Cache Debugger: {type(cache_debugger).__name__}")
        
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger
        }
    except Exception as e:
        logger.error(f"❌ Storage system initialization failed: {e}")
        return {
            "log_manager": log_manager,
            "history_manager": history_manager,
            "cache_debugger": cache_debugger
        }

# راه‌اندازی خودکار
storage_system = initialize_storage_system()

__all__ = [
    "LogManager", "log_manager",
    "HistoryManager", "history_manager", 
    "CacheDebugger", "cache_debugger",
    "initialize_storage_system", "storage_system"
]

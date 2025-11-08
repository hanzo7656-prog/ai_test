"""
Debug System Core Modules
Central management for debugging and monitoring
"""

import logging
from .debug_manager import DebugManager
from .metrics_collector import RealTimeMetricsCollector
from .alert_manager import AlertManager, AlertLevel, AlertType

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های گلوبال با Dependency Injection
debug_manager = DebugManager()
metrics_collector = RealTimeMetricsCollector()
alert_manager = AlertManager()

# راه‌اندازی ارتباط بین ماژول‌های core
def initialize_core_system():
    """راه‌اندازی و ارتباط ماژول‌های هسته"""
    try:
        # تنظیم alert manager برای debug manager
        debug_manager.alert_manager = alert_manager
        
        # لاگ راه‌اندازی موفق
        logger.info("✅ Core debug system initialized with dependency injection")
        logger.info(f"   - Debug Manager: {type(debug_manager).__name__}")
        logger.info(f"   - Metrics Collector: {type(metrics_collector).__name__}")
        logger.info(f"   - Alert Manager: {type(alert_manager).__name__}")
        
        return {
            "debug_manager": debug_manager,
            "metrics_collector": metrics_collector,
            "alert_manager": alert_manager
        }
    except Exception as e:
        logger.error(f"❌ Core system initialization failed: {e}")
        # بازگشت نمونه‌های موجود حتی اگر خطا رخ دهد
        return {
            "debug_manager": debug_manager,
            "metrics_collector": metrics_collector,
            "alert_manager": alert_manager
        }

# راه‌اندازی خودکار
core_system = initialize_core_system()

__all__ = [
    "DebugManager", "debug_manager",
    "RealTimeMetricsCollector", "metrics_collector", 
    "AlertManager", "AlertLevel", "AlertType", "alert_manager",
    "initialize_core_system", "core_system"
]

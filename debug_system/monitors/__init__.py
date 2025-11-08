"""
Debug System Monitors
Specialized monitors for different aspects of the system
"""

import logging
from ..core import debug_manager, metrics_collector, alert_manager
from .endpoint_monitor import EndpointMonitor
from .system_monitor import SystemMonitor
from .performance_monitor import PerformanceMonitor
from .security_monitor import SecurityMonitor

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های مانیتور با Dependency Injection
endpoint_monitor = EndpointMonitor(debug_manager)
system_monitor = SystemMonitor(metrics_collector, alert_manager)
performance_monitor = PerformanceMonitor(debug_manager, alert_manager)
security_monitor = SecurityMonitor(alert_manager)

def initialize_monitors_system():
    """راه‌اندازی و ارتباط سیستم‌های مانیتورینگ"""
    try:
        # راه‌اندازی مانیتورهای سیستمی
        logger.info("✅ Monitoring system initialized with dependency injection")
        logger.info(f"   - Endpoint Monitor: {type(endpoint_monitor).__name__}")
        logger.info(f"   - System Monitor: {type(system_monitor).__name__}")
        logger.info(f"   - Performance Monitor: {type(performance_monitor).__name__}")
        logger.info(f"   - Security Monitor: {type(security_monitor).__name__}")
        
        return {
            "endpoint_monitor": endpoint_monitor,
            "system_monitor": system_monitor,
            "performance_monitor": performance_monitor,
            "security_monitor": security_monitor
        }
    except Exception as e:
        logger.error(f"❌ Monitors initialization failed: {e}")
        # بازگشت نمونه‌های موجود حتی اگر خطا رخ دهد
        return {
            "endpoint_monitor": endpoint_monitor,
            "system_monitor": system_monitor,
            "performance_monitor": performance_monitor,
            "security_monitor": security_monitor
        }

# راه‌اندازی خودکار
monitors_system = initialize_monitors_system()

__all__ = [
    "EndpointMonitor", "endpoint_monitor",
    "SystemMonitor", "system_monitor", 
    "PerformanceMonitor", "performance_monitor",
    "SecurityMonitor", "security_monitor",
    "initialize_monitors_system", "monitors_system"
]

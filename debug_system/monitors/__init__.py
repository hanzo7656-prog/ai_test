"""
Debug System Monitors - Optimized Version
سیستم مانیتورینگ بهینه‌شده با معماری متمرکز
"""

import logging
import time
import threading
from ..core import debug_manager, metrics_collector, alert_manager

logger = logging.getLogger(__name__)

# ایمپورت کلاس‌ها از فایل‌های جدید
from .system_monitor import SystemMetricsCollector, initialize_system_metrics_collector, get_metrics_collector
from .performance_analyzer import PerformanceAnalyzer, initialize_performance_analyzer
from .endpoint_analyzer import EndpointAnalyzer, initialize_endpoint_analyzer

# نمونه‌های گلوبال
system_monitor = None
performance_analyzer = None
endpoint_analyzer = None

def initialize_monitors_system(collection_interval: int = 30):
    """
    راه‌اندازی کامل سیستم مانیتورینگ
    
    Args:
        collection_interval: فاصله جمع‌آوری متریک به ثانیه
    
    Returns:
        Dict[str, Any]: وضعیت راه‌اندازی
    """
    global system_monitor, performance_analyzer, endpoint_analyzer
    
    try:
        logger.info("🚀 Starting optimized monitors system...")
        
        # 1. راه‌اندازی جمع‌آوری متریک (تنها نقطه جمع‌آوری)
        system_monitor = initialize_system_metrics_collector(collection_interval)
        
        # 2. راه‌اندازی تحلیلگر عملکرد
        performance_analyzer = initialize_performance_analyzer(system_monitor)
        
        # 3. راه‌اندازی تحلیلگر endpointها
        endpoint_analyzer = initialize_endpoint_analyzer(system_monitor)
        
        # 4. شروع جمع‌آوری خودکار
        system_monitor.start_collection()
        
        logger.info("✅ Optimized monitoring system initialized successfully")
        logger.info(f"   - Collection interval: {collection_interval}s")
        logger.info("   - Architecture: Centralized collector with pure analyzers")
        logger.info("   - No duplicate monitoring")
        
        return {
            "status": "success",
            "components": {
                "collector": "active",
                "performance_analyzer": "active",
                "endpoint_analyzer": "active"
            },
            "collection_interval": collection_interval
        }
        
    except Exception as e:
        logger.error(f"❌ Monitors initialization failed: {e}")
        
        # Fallback: ایجاد حداقلی
        if not system_monitor:
            system_monitor = SystemMetricsCollector(collection_interval)
        
        if not performance_analyzer:
            performance_analyzer = PerformanceAnalyzer(system_monitor)
        
        if not endpoint_analyzer:
            endpoint_analyzer = EndpointAnalyzer(system_monitor)
        
        return {
            "status": "partial",
            "error": str(e),
            "components": {
                "collector": "active" if system_monitor else "inactive",
                "performance_analyzer": "active" if performance_analyzer else "inactive",
                "endpoint_analyzer": "active" if endpoint_analyzer else "inactive"
            }
        }

# راه‌اندازی با تاخیر
def delayed_initialization():
    """راه‌اندازی با تاخیر برای جلوگیری از race conditions"""
    time.sleep(2)
    initialize_monitors_system()

# شروع initialization در background thread
init_thread = threading.Thread(target=delayed_initialization, daemon=True)
init_thread.start()

__all__ = [
    # کلاس‌ها
    "SystemMetricsCollector",
    "PerformanceAnalyzer",
    "EndpointAnalyzer",
    
    # نمونه‌ها
    "system_monitor",
    "performance_analyzer",
    "endpoint_analyzer",
    
    # توابع
    "initialize_monitors_system",
    "get_metrics_collector"
]

"""
Debug System Core Modules
Central management for debugging and monitoring
Optimized Version - Central Monitor Integration
"""

import logging
import time
from .debug_manager import DebugManager
from .metrics_collector import RealTimeMetricsCollector
from .alert_manager import AlertManager, AlertLevel, AlertType
from .system_monitor import central_monitor, initialize_central_monitoring

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های گلوبال
debug_manager = DebugManager()
metrics_collector = RealTimeMetricsCollector()
alert_manager = AlertManager()

def initialize_core_system():
    """راه‌اندازی و ارتباط ماژول‌های هسته با تاخیر هوشمند"""
    try:
        # مرحله ۱: راه‌اندازی اولیه سیستم‌ها
        logger.info("🚀 Starting core system initialization...")
        
        # تنظیم alert manager برای debug manager با تاخیر
        def delayed_alert_integration():
            time.sleep(2)  # صبر کن alert_manager کامل لود شود
            integration_success = debug_manager.set_alert_manager(alert_manager)
            if integration_success:
                logger.info("✅ Debug Manager ↔ Alert Manager integration established")
            else:
                logger.warning("⚠️ Alert Manager integration failed")
        
        integration_thread = threading.Thread(target=delayed_alert_integration, daemon=True)
        integration_thread.start()
        
        # مرحله ۲: راه‌اندازی central monitoring system
        logger.info("🔧 Initializing central monitoring system...")
        central_monitor_instance = initialize_central_monitoring(metrics_collector, alert_manager)
        
        # مرحله ۳: راه‌اندازی central monitor با تاخیر
        def start_central_monitor():
            time.sleep(3)  # صبر کن همه سیستم‌ها لود شوند
            if central_monitor_instance:
                central_monitor_instance.start_monitoring()
                logger.info("🎯 Central Monitoring System STARTED")
            else:
                logger.error("❌ Failed to initialize central monitor")
        
        monitor_thread = threading.Thread(target=start_central_monitor, daemon=True)
        monitor_thread.start()
        
        # مرحله ۴: منتظر بمان و وضعیت را چک کن
        def check_system_status():
            time.sleep(5)
            
            status_report = {
                'debug_manager': {
                    'active': debug_manager.is_active(),
                    'alert_integration': debug_manager.get_alert_integration_status().get('integration_status', 'unknown')
                },
                'metrics_collector': {
                    'active': True,  # همیشه active است
                    'mode': metrics_collector.get_connection_status().get('collection_mode', 'unknown')
                },
                'alert_manager': {
                    'active': True,
                    'notification_channels': list(alert_manager.notification_channels.keys())
                },
                'central_monitor': {
                    'active': central_monitor_instance.is_monitoring if central_monitor_instance else False,
                    'subscribers': len(central_monitor_instance.subscribers) if central_monitor_instance else 0
                }
            }
            
            logger.info("📊 Core System Status Report:")
            for system, info in status_report.items():
                status = "✅ ACTIVE" if info.get('active', False) else "❌ INACTIVE"
                details = " | ".join([f"{k}: {v}" for k, v in info.items() if k != 'active'])
                logger.info(f"   - {system}: {status} | {details}")
        
        status_thread = threading.Thread(target=check_system_status, daemon=True)
        status_thread.start()
        
        logger.info("✅ Core debug system initialized with CENTRAL MONITOR integration")
        logger.info("   - Debug Manager: Connected to Central Monitor")
        logger.info("   - Metrics Collector: Passive mode (Central Monitor source)")
        logger.info("   - Alert Manager: Bulk notifications enabled")
        logger.info("   - Central Monitor: Will start in 3 seconds")
        
        return {
            "debug_manager": debug_manager,
            "metrics_collector": metrics_collector,
            "alert_manager": alert_manager,
            "central_monitor": central_monitor_instance
        }
        
    except Exception as e:
        logger.error(f"❌ Core system initialization failed: {e}")
        logger.info("🔄 Continuing with basic functionality...")
        
        # Fallback: تنظیمات حداقلی
        debug_manager.set_alert_manager(alert_manager)
        
        return {
            "debug_manager": debug_manager,
            "metrics_collector": metrics_collector,
            "alert_manager": alert_manager,
            "central_monitor": None,
            "error": str(e)
        }

# Import threading برای delayed initialization
import threading

# راه‌اندازی خودکار با تاخیر
def delayed_initialization():
    """راه‌اندازی با تاخیر برای جلوگیری از race conditions"""
    time.sleep(1)  # صبر کن همه imports کامل شوند
    global core_system
    core_system = initialize_core_system()

# شروع initialization در background thread
init_thread = threading.Thread(target=delayed_initialization, daemon=True)
init_thread.start()

# ایجاد متغیر global
core_system = None

__all__ = [
    "DebugManager", "debug_manager",
    "RealTimeMetricsCollector", "metrics_collector", 
    "AlertManager", "AlertLevel", "AlertType", "alert_manager",
    "central_monitor", "initialize_central_monitoring",
    "initialize_core_system", "core_system"
]

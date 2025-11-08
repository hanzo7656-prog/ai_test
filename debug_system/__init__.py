"""
VortexAI Debug System
Complete monitoring and debugging system for VortexAI API
"""

__version__ = "1.0.0"
__author__ = "VortexAI Team"

# ایمپورت مستقیم از ماژول‌ها به جای ایمپورت کلی
from .core.debug_manager import DebugManager, debug_manager
from .core.metrics_collector import RealTimeMetricsCollector, metrics_collector
from .core.alert_manager import AlertManager, AlertLevel, AlertType, alert_manager

# راه‌اندازی کامل سیستم
def initialize_debug_system():
    """راه‌اندازی کامل سیستم دیباگ"""
    try:
        print("🔄 Initializing VortexAI Debug System...")
        
        # راه‌اندازی core system
        from .core import initialize_core_system
        core_system = initialize_core_system()
        
        # راه‌اندازی monitors
        from .monitors import initialize_monitors_system
        monitors_system = initialize_monitors_system()
        
        # راه‌اندازی storage
        from .storage import initialize_storage_system
        storage_system = initialize_storage_system()
        
        # راه‌اندازی realtime
        from .realtime import initialize_realtime_system
        realtime_system = initialize_realtime_system()
        
        # راه‌اندازی tools با dependencyهای لازم
        from .tools import initialize_tools_system
        tools_system = initialize_tools_system(
            debug_manager_instance=debug_manager,
            history_manager_instance=storage_system.get("history_manager")
        )
        
        print("✅ VortexAI Debug System fully initialized!")
        
        return {
            "core": core_system,
            "monitors": monitors_system,
            "storage": storage_system,
            "realtime": realtime_system,
            "tools": tools_system
        }
        
    except Exception as e:
        print(f"❌ Debug system initialization failed: {e}")
        # بازگشت حداقل سیستم حتی اگر خطا رخ دهد
        return {
            "core": {"debug_manager": debug_manager, "metrics_collector": metrics_collector, "alert_manager": alert_manager},
            "monitors": {},
            "storage": {},
            "realtime": {},
            "tools": {}
        }

# راه‌اندازی خودکار هنگام ایمپورت
debug_system = initialize_debug_system()

__all__ = [
    # Core
    "DebugManager", "debug_manager",
    "RealTimeMetricsCollector", "metrics_collector",
    "AlertManager", "AlertLevel", "AlertType", "alert_manager",
    
    # Initialization
    "initialize_debug_system", "debug_system"
]

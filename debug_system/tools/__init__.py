"""
VortexAI Debug System
Complete monitoring and debugging system for VortexAI API
"""

__version__ = "1.0.0"
__author__ = "VortexAI Team"

from .core import (
    DebugManager, debug_manager,
    RealTimeMetricsCollector, metrics_collector,
    AlertManager, AlertLevel, AlertType, alert_manager
)

from .monitors import (
    EndpointMonitor, endpoint_monitor,
    SystemMonitor, system_monitor,
    PerformanceMonitor, performance_monitor,
    SecurityMonitor, security_monitor
)

from .storage import (
    LogManager, log_manager,
    HistoryManager, history_manager,
    CacheDebugger, cache_debugger
)

from .realtime import (
    ConsoleStreamManager, console_stream,
    LiveDashboardManager, live_dashboard,
    WebSocketManager, websocket_manager
)

from .tools import (
    DevTools, dev_tools,
    TestingTools, testing_tools,
    ReportGenerator, report_generator
)

# راه‌اندازی کامل سیستم
def initialize_debug_system():
    """راه‌اندازی کامل سیستم دیباگ"""
    try:
        from .core import initialize_core_system
        from .monitors import initialize_monitors_system
        from .storage import initialize_storage_system
        from .realtime import initialize_realtime_system
        from .tools import initialize_tools_system
        
        # راه‌اندازی تمام زیرسیستم‌ها
        core_system = initialize_core_system()
        monitors_system = initialize_monitors_system()
        storage_system = initialize_storage_system()
        realtime_system = initialize_realtime_system()
        tools_system = initialize_tools_system()
        
        print("✅ VortexAI Debug System fully initialized!")
        print(f"   - Core System: {len(core_system)} modules")
        print(f"   - Monitors: {len(monitors_system)} monitors")
        print(f"   - Storage: {len(storage_system)} modules")
        print(f"   - Real-time: {len(realtime_system)} modules")
        print(f"   - Tools: {len(tools_system)} tools")
        
        return {
            "core": core_system,
            "monitors": monitors_system,
            "storage": storage_system,
            "realtime": realtime_system,
            "tools": tools_system
        }
        
    except Exception as e:
        print(f"❌ Debug system initialization failed: {e}")
        raise

# راه‌اندازی خودکار هنگام ایمپورت
debug_system = initialize_debug_system()

__all__ = [
    # Core
    "DebugManager", "debug_manager",
    "RealTimeMetricsCollector", "metrics_collector",
    "AlertManager", "AlertLevel", "AlertType", "alert_manager",
    
    # Monitors
    "EndpointMonitor", "endpoint_monitor",
    "SystemMonitor", "system_monitor",
    "PerformanceMonitor", "performance_monitor",
    "SecurityMonitor", "security_monitor",
    
    # Storage
    "LogManager", "log_manager",
    "HistoryManager", "history_manager",
    "CacheDebugger", "cache_debugger",
    
    # Real-time
    "ConsoleStreamManager", "console_stream",
    "LiveDashboardManager", "live_dashboard",
    "WebSocketManager", "websocket_manager",
    
    # Tools
    "DevTools", "dev_tools",
    "TestingTools", "testing_tools",
    "ReportGenerator", "report_generator",
    
    # Initialization
    "initialize_debug_system", "debug_system"
]

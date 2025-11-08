"""
Debug System Real-time Modules  
Live monitoring and real-time data streaming
"""

import logging

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های real-time - ابتدا None تعریف می‌کنیم
console_stream = None
live_dashboard = None
websocket_manager = None

def initialize_realtime_system(debug_manager=None, metrics_collector=None):
    """راه‌اندازی و ارتباط سیستم‌های real-time"""
    try:
        # Import داخل تابع برای جلوگیری از circular imports
        from .console_stream import ConsoleStreamManager
        from .live_dashboard import LiveDashboardManager
        from .websocket_manager import WebSocketManager
        
        global console_stream, live_dashboard, websocket_manager
        
        # ایجاد نمونه‌ها
        console_stream = ConsoleStreamManager()
        logger.info(f"✅ Console Stream Manager created: {type(console_stream).__name__}")
        
        # ایجاد live dashboard با dependencyهای لازم
        if debug_manager and metrics_collector:
            live_dashboard = LiveDashboardManager(debug_manager, metrics_collector)
        else:
            # اگر dependencyها ارائه نشدند، از core استفاده کن
            try:
                from ..core import debug_manager as core_debug_manager
                from ..core import metrics_collector as core_metrics_collector
                live_dashboard = LiveDashboardManager(core_debug_manager, core_metrics_collector)
            except ImportError as e:
                logger.warning(f"⚠️ Could not import core modules for live dashboard: {e}")
                live_dashboard = None
        
        websocket_manager = WebSocketManager()
        
        # راه‌اندازی سیستم real-time
        logger.info("✅ Real-time system initialized")
        logger.info(f"   - Console Stream: {type(console_stream).__name__}")
        logger.info(f"   - Live Dashboard: {type(live_dashboard).__name__ if live_dashboard else 'Not available'}")
        logger.info(f"   - WebSocket Manager: {type(websocket_manager).__name__}")
        
        return {
            "console_stream": console_stream,
            "live_dashboard": live_dashboard,
            "websocket_manager": websocket_manager
        }
        
    except Exception as e:
        logger.error(f"❌ Real-time system initialization failed: {e}")
        # ایجاد fallback console manager
        try:
            from .console_stream import ConsoleStreamManager
            console_stream = ConsoleStreamManager()
            logger.info("✅ Fallback Console Manager created")
        except Exception as fallback_error:
            logger.error(f"❌ Fallback console manager also failed: {fallback_error}")
            console_stream = None
        
        return {
            "console_stream": console_stream,
            "live_dashboard": live_dashboard,
            "websocket_manager": websocket_manager
        }

# راه‌اندازی اولیه
realtime_system = initialize_realtime_system()

__all__ = [
    "ConsoleStreamManager", "console_stream",
    "LiveDashboardManager", "live_dashboard", 
    "WebSocketManager", "websocket_manager",
    "initialize_realtime_system", "realtime_system"
]

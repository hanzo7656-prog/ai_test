"""
Debug System Real-time Modules  
Live monitoring and real-time data streaming
"""

import logging
from ..core import debug_manager, metrics_collector
from .console_stream import ConsoleStreamManager
from .live_dashboard import LiveDashboardManager
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های real-time با Dependency Injection صحیح
console_stream = ConsoleStreamManager()
live_dashboard = LiveDashboardManager(debug_manager, metrics_collector)  # ✅ اصلاح signature
websocket_manager = WebSocketManager()

def initialize_realtime_system():
    """راه‌اندازی و ارتباط سیستم‌های real-time"""
    try:
        # راه‌اندازی سیستم real-time
        logger.info("✅ Real-time system initialized")
        logger.info(f"   - Console Stream: {type(console_stream).__name__}")
        logger.info(f"   - Live Dashboard: {type(live_dashboard).__name__}")
        logger.info(f"   - WebSocket Manager: {type(websocket_manager).__name__}")
        
        return {
            "console_stream": console_stream,
            "live_dashboard": live_dashboard,
            "websocket_manager": websocket_manager
        }
    except Exception as e:
        logger.error(f"❌ Real-time system initialization failed: {e}")
        return {
            "console_stream": console_stream,
            "live_dashboard": live_dashboard,
            "websocket_manager": websocket_manager
        }

# راه‌اندازی خودکار
realtime_system = initialize_realtime_system()

__all__ = [
    "ConsoleStreamManager", "console_stream",
    "LiveDashboardManager", "live_dashboard", 
    "WebSocketManager", "websocket_manager",
    "initialize_realtime_system", "realtime_system"
]

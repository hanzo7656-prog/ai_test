"""
Debug System Real-time Modules  
Live monitoring and real-time data streaming
"""

from .console_stream import ConsoleStreamManager, console_stream
from .live_dashboard import LiveDashboardManager, live_dashboard
from .websocket_manager import WebSocketManager, websocket_manager

__all__ = [
    "ConsoleStreamManager", "console_stream",
    "LiveDashboardManager", "live_dashboard", 
    "WebSocketManager", "websocket_manager"
]

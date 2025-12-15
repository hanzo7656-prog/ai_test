"""
Debug System Real-time Modules  
Live monitoring and real-time data streaming
Optimized Version - Central Monitor Integration
"""

import logging
import asyncio
import threading
import time

logger = logging.getLogger(__name__)

# ایجاد نمونه‌های real-time - ابتدا None تعریف می‌کنیم
console_stream = None
live_dashboard = None
websocket_manager = None

def initialize_realtime_system(debug_manager=None, metrics_collector=None):
    """راه‌اندازی و ارتباط سیستم‌های real-time با تاخیر هوشمند"""
    try:
        # Import داخل تابع برای جلوگیری از circular imports
        from .console_stream import ConsoleStreamManager
        from .live_dashboard import LiveDashboardManager, initialize_live_dashboard
        from .websocket_manager import WebSocketManager
        
        global console_stream, live_dashboard, websocket_manager
        
        logger.info("🚀 Starting real-time system initialization...")
        
        # مرحله ۱: ایجاد console stream (سریع)
        console_stream = ConsoleStreamManager()
        logger.info(f"✅ Console Stream Manager created: {type(console_stream).__name__}")
        
        # مرحله ۲: ایجاد websocket manager
        websocket_manager = WebSocketManager()
        logger.info(f"✅ WebSocket Manager created: {type(websocket_manager).__name__}")
        
        # مرحله ۳: ایجاد live dashboard با تاخیر (نیاز به dependencies دارد)
        def initialize_dashboard():
            """مقداردهی اولیه dashboard با تاخیر"""
            time.sleep(3)  # صبر کن dependencies لود شوند
            
            try:
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
                
                if live_dashboard:
                    # شروع broadcast در background
                    asyncio.create_task(live_dashboard.start_dashboard_broadcast())
                    logger.info(f"✅ Live Dashboard created: {type(live_dashboard).__name__}")
                else:
                    logger.warning("⚠️ Live dashboard could not be initialized")
                    
            except Exception as e:
                logger.error(f"❌ Error initializing live dashboard: {e}")
        
        # اجرای dashboard initialization در background thread
        dashboard_thread = threading.Thread(target=initialize_dashboard, daemon=True)
        dashboard_thread.start()
        
        # مرحله ۴: اتصال سیستم‌ها به یکدیگر
        def connect_systems():
            """اتصال سیستم‌های real-time به یکدیگر"""
            time.sleep(5)  # صبر کن همه سیستم‌ها لود شوند
            
            status_report = {
                'Console Stream': {
                    'status': 'ACTIVE' if console_stream else 'INACTIVE',
                    'mode': 'Bulk messaging (3s interval)',
                    'central_monitor': 'Connected' if hasattr(console_stream, '_on_alert_received') else 'Not connected'
                },
                'WebSocket Manager': {
                    'status': 'ACTIVE' if websocket_manager else 'INACTIVE',
                    'connections': websocket_manager.get_connection_stats().get('total_connections', 0) if websocket_manager else 0,
                    'central_monitor': 'Connected' if hasattr(websocket_manager, '_on_broadcast_message') else 'Not connected'
                },
                'Live Dashboard': {
                    'status': 'ACTIVE' if live_dashboard else 'PENDING',
                    'mode': 'Delta updates (5s interval)' if live_dashboard else 'Not initialized',
                    'central_monitor': 'Connected' if live_dashboard and hasattr(live_dashboard, '_on_metrics_received') else 'Not connected'
                }
            }
            
            logger.info("📊 Real-time Systems Status Report:")
            for system, info in status_report.items():
                logger.info(f"   - {system}: {info['status']} | {info['mode']} | Central: {info.get('central_monitor', 'N/A')}")
        
        # اجرای status report با تاخیر
        status_thread = threading.Thread(target=connect_systems, daemon=True)
        status_thread.start()
        
        logger.info("✅ Real-time system initialized with OPTIMIZATIONS")
        logger.info("   - Console Stream: Bulk messaging (3s interval)")
        logger.info("   - WebSocket Manager: Connection grouping")
        logger.info("   - Live Dashboard: Delta updates (5s interval)")
        
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

# راه‌اندازی اولیه با تاخیر
def delayed_initialization():
    """راه‌اندازی real-time system با تاخیر"""
    time.sleep(4)  # صبر کن core و monitors سیستم‌ها لود شوند
    global realtime_system
    realtime_system = initialize_realtime_system()

# شروع initialization در background thread
init_thread = threading.Thread(target=delayed_initialization, daemon=True)
init_thread.start()

# ایجاد متغیر global
realtime_system = None

__all__ = [
    "ConsoleStreamManager", "console_stream",
    "LiveDashboardManager", "live_dashboard", 
    "WebSocketManager", "websocket_manager",
    "initialize_realtime_system", "realtime_system"
]

import time
import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import threading
import json
import traceback
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DebugLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class EndpointCall:
    endpoint: str
    method: str
    timestamp: datetime
    params: Dict[str, Any]
    response_time: float
    status_code: int
    cache_used: bool
    api_calls: int
    memory_used: float
    cpu_impact: float

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int

class DebugManager:
    def __init__(self):
        self.endpoint_calls = deque(maxlen=10000)
        self.system_metrics_history = deque(maxlen=1000)
        self.endpoint_stats = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': [],
            'last_call': None
        })
        
        self.alerts = []
        self.performance_thresholds = {
            'response_time_warning': 1.0,
            'response_time_critical': 3.0,
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0
        }
        
        self._start_background_monitoring()
        
    def log_endpoint_call(self, endpoint: str, method: str, params: Dict[str, Any], 
                         response_time: float, status_code: int, cache_used: bool, 
                         api_calls: int = 0):
        """ثبت فراخوانی اندپوینت"""
        try:
            memory_used = psutil.virtual_memory().percent
            cpu_impact = psutil.cpu_percent(interval=0.1)
            
            call = EndpointCall(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(),
                params=params,
                response_time=response_time,
                status_code=status_code,
                cache_used=cache_used,
                api_calls=api_calls,
                memory_used=memory_used,
                cpu_impact=cpu_impact
            )
            
            self.endpoint_calls.append(call)
            
            stats = self.endpoint_stats[endpoint]
            stats['total_calls'] += 1
            stats['total_response_time'] += response_time
            
            if 200 <= status_code < 300:
                stats['successful_calls'] += 1
            else:
                stats['failed_calls'] += 1
                stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'status_code': status_code,
                    'params': params
                })
                
            if cache_used:
                stats['cache_hits'] += 1
            else:
                stats['cache_misses'] += 1
                
            stats['api_calls'] += api_calls
            stats['last_call'] = datetime.now().isoformat()
            
            self._check_performance_alerts(endpoint, call)
            
            logger.debug(f"📊 Endpoint logged: {endpoint} - {response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error logging endpoint call: {e}")
    
    def log_error(self, endpoint: str, error: Exception, traceback_str: str, context: Dict[str, Any] = None):
        """ثبت خطا"""
        error_data = {
            'endpoint': endpoint,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback_str,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.endpoint_stats[endpoint]['errors'].append(error_data)
        
        if self._is_critical_error(error):
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical error in {endpoint}: {str(error)}",
                source=endpoint,
                data=error_data
            )
        
        logger.error(f"🚨 Error in {endpoint}: {error}")
    
    def get_endpoint_stats(self, endpoint: str = None) -> Dict[str, Any]:
        """دریافت آمار اندپوینت"""
        if endpoint:
            if endpoint not in self.endpoint_stats:
                return {'error': 'Endpoint not found'}
            
            stats = self.endpoint_stats[endpoint]
            avg_response_time = (stats['total_response_time'] / stats['total_calls']) if stats['total_calls'] > 0 else 0
            
            return {
                'endpoint': endpoint,
                'total_calls': stats['total_calls'],
                'successful_calls': stats['successful_calls'],
                'failed_calls': stats['failed_calls'],
                'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                'average_response_time': round(avg_response_time, 3),
                'cache_performance': {
                    'hits': stats['cache_hits'],
                    'misses': stats['cache_misses'],
                    'hit_rate': (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                },
                'api_calls': stats['api_calls'],
                'recent_errors': stats['errors'][-10:],
                'last_call': stats['last_call']
            }
        else:
            all_stats = {}
            total_calls = 0
            total_success = 0
            
            for endpoint, stats in self.endpoint_stats.items():
                all_stats[endpoint] = {
                    'total_calls': stats['total_calls'],
                    'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                    'average_response_time': round((stats['total_response_time'] / stats['total_calls']), 3) if stats['total_calls'] > 0 else 0,
                    'last_call': stats['last_call']
                }
                total_calls += stats['total_calls']
                total_success += stats['successful_calls']
            
            return {
                'overall': {
                    'total_endpoints': len(self.endpoint_stats),
                    'total_calls': total_calls,
                    'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                },
                'endpoints': all_stats
            }
    
    def get_recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """دریافت آخرین فراخوانی‌ها"""
        recent_calls = list(self.endpoint_calls)[-limit:]
        return [
            {
                'endpoint': call.endpoint,
                'method': call.method,
                'timestamp': call.timestamp.isoformat(),
                'response_time': call.response_time,
                'status_code': call.status_code,
                'cache_used': call.cache_used,
                'api_calls': call.api_calls,
                'memory_used': call.memory_used,
                'cpu_impact': call.cpu_impact
            }
            for call in recent_calls
        ]
    
    def get_system_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌های سیستم"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            {
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage': metrics.disk_usage,
                'network_io': metrics.network_io,
                'active_connections': metrics.active_connections
            }
            for metrics in self.system_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def _start_background_monitoring(self):
        """شروع مانیتورینگ پس‌زمینه سیستم"""
        def monitor_system():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"❌ System monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        logger.info("✅ Background system monitoring started")
    
    def _collect_system_metrics(self):
        """جمع‌آوری متریک‌های سیستم"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            active_connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
    
    def _check_performance_alerts(self, endpoint: str, call: EndpointCall):
        """بررسی هشدارهای performance"""
        if call.response_time > self.performance_thresholds['response_time_critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical response time in {endpoint}: {call.response_time:.2f}s",
                source=endpoint,
                data={
                    'response_time': call.response_time,
                    'threshold': self.performance_thresholds['response_time_critical']
                }
            )
        elif call.response_time > self.performance_thresholds['response_time_warning']:
            self._create_alert(
                level=DebugLevel.WARNING,
                message=f"High response time in {endpoint}: {call.response_time:.2f}s",
                source=endpoint,
                data={
                    'response_time': call.response_time,
                    'threshold': self.performance_thresholds['response_time_warning']
                }
            )
        
        if call.cpu_impact > self.performance_thresholds['cpu_critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical CPU usage in {endpoint}: {call.cpu_impact:.1f}%",
                source=endpoint,
                data={'cpu_usage': call.cpu_impact}
            )
    
    def _create_alert(self, level: DebugLevel, message: str, source: str, data: Dict[str, Any]):
        """ایجاد هشدار جدید"""
        alert = {
            'id': len(self.alerts) + 1,
            'level': level.value,
            'message': message,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        logger.warning(f"🚨 {level.value} Alert: {message}")
    
    def _is_critical_error(self, error: Exception) -> bool:
        """بررسی آیا خطا critical است"""
        critical_errors = [
            'Timeout',
            'ConnectionError',
            'MemoryError',
            'OSError'
        ]
        
        return any(critical_error in type(error).__name__ for critical_error in critical_errors)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id: int):
        """تأیید هشدار"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
    
    def clear_old_data(self, days: int = 7):
        """پاک کردن داده‌های قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        self.endpoint_calls = deque(
            [call for call in self.endpoint_calls if call.timestamp > cutoff_time],
            maxlen=10000
        )
        
        self.system_metrics_history = deque(
            [metrics for metrics in self.system_metrics_history if metrics.timestamp > cutoff_time],
            maxlen=1000
        )
        
        logger.info(f"🧹 Cleared data older than {days} days")

# ایجاد نمونه گلوبال
debug_manager = DebugManager()

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import logging
import time
import psutil
from pathlib import Path
import json
import asyncio
import logging
import sys

# ==================== DEBUG CODE ====================
print("=" * 60)
print("🛠️  VORTEXAI DEBUG - SYSTEM INITIALIZATION")
print("=" * 60)

# ایمپورت روت‌ها
try:
    from routes.health import health_router
    from routes.coins import coins_router
    from routes.exchanges import exchanges_router
    from routes.news import news_router
    from routes.insights import insights_router
    from routes.raw_coins import raw_coins_router
    from routes.raw_news import raw_news_router
    from routes.raw_insights import raw_insights_router
    from routes.raw_exchanges import raw_exchanges_router
    from routes.docs import docs_router
    print("✅ All routers imported successfully!")
except ImportError as e:
    print(f"❌ Router import error: {e}")

try:
    from complete_coinstats_manager import coin_stats_manager
    print("✅ coin_stats_manager imported successfully!")
    COINSTATS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CoinStats import error: {e}")
    COINSTATS_AVAILABLE = False

# ==================== DEBUG SYSTEM IMPORTS ====================
DEBUG_SYSTEM_AVAILABLE = False
live_dashboard_manager = None
console_stream_manager = None

try:
    from debug_system.core import core_system, debug_manager, metrics_collector, alert_manager
    from debug_system.monitors import monitors_system, endpoint_monitor, system_monitor, performance_monitor, security_monitor
    from debug_system.storage import history_manager, log_manager, cache_debugger
    from debug_system.realtime import websocket_manager, console_stream
    from debug_system.tools import tools_system, dev_tools, testing_tools, report_generator
    
    from debug_system.realtime.live_dashboard import LiveDashboardManager
    
    DEBUG_SYSTEM_AVAILABLE = True
    print("✅ Complete debug system imported successfully!")
except ImportError as e:
    print(f"❌ Debug system import error: {e}")
    DEBUG_SYSTEM_AVAILABLE = False

print("=" * 60)

# ==================== DEBUG SYSTEM INITIALIZATION ====================
if DEBUG_SYSTEM_AVAILABLE:
    try:
        print("🔄 Initializing debug system...")
        
        # مدیریت event loop
        print("   🔧 Setting up event loop...")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("   ✅ New event loop created")
            else:
                print("   ✅ Existing event loop used")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print("   ✅ New event loop created for runtime error")
        
        # راه‌اندازی سیستم‌های core
        print("   🔧 Setting up core systems...")
        if not core_system:
            from debug_system.core import initialize_core_system
            core_system = initialize_core_system()
            print("   ✅ Core systems initialized")
        
        # راه‌اندازی مانیتورها
        print("   📊 Setting up monitors...")
        if not monitors_system:
            from debug_system.monitors import initialize_monitors_system
            monitors_system = initialize_monitors_system()
            print("   ✅ Monitors system initialized")
        
        # راه‌اندازی ابزارها
        print("   🛠️ Setting up tools...")
        if not tools_system:
            from debug_system.tools import initialize_tools_system
            tools_system = initialize_tools_system(monitors_system["endpoint_monitor"])
            print("   ✅ Tools system initialized")
        
        # راه‌اندازی سیستم real-time
        print("   ⚡ Setting up real-time systems...")
        
        # راه‌اندازی Live Dashboard
        try:
            live_dashboard_manager = LiveDashboardManager(
                debug_manager, 
                metrics_collector
            )
            print("   ✅ Live Dashboard Manager created")
        except Exception as e:
            print(f"   ❌ Live Dashboard Manager error: {e}")
            live_dashboard_manager = None
        
        # راه‌اندازی Console Stream
        try:
            console_stream_manager = console_stream
            print("   ✅ Console Stream Manager created")
    
            
        except Exception as e:
            print(f"   ❌ Console Stream Manager error: {e}")
            
            # ایجاد fallback
            class SimpleConsoleManager:
                def __init__(self):
                    self.active_connections = []
                async def connect(self, websocket):
                    await websocket.accept()
                    self.active_connections.append(websocket)
                def disconnect(self, websocket):
                    if websocket in self.active_connections:
                        self.active_connections.remove(websocket)
                async def broadcast_message(self, message):
                    pass
    
            console_stream_manager = SimpleConsoleManager()
            print("   ✅ Fallback Console Manager created")
            
        # تابع برای شروع برودکست دشبورد
        async def start_dashboard_broadcast():
            if live_dashboard_manager:
                try:
                    await live_dashboard_manager.start_dashboard_broadcast()
                except Exception as e:
                    print(f"   ❌ Dashboard broadcast error: {e}")
            else:
                print("   ⚠️ Dashboard manager not available")
        
        # تابع برای پاک‌سازی دوره‌ای
        async def periodic_cleanup():
            while True:
                try:
                    debug_manager.clear_old_data(days=7)
                    if hasattr(alert_manager, 'cleanup_old_alerts'):
                        alert_manager.cleanup_old_alerts()
                    if hasattr(alert_manager, 'auto_resolve_alerts'):
                        alert_manager.auto_resolve_alerts()
                    
                    if hasattr(websocket_manager, 'cleanup_inactive_connections'):
                        websocket_manager.cleanup_inactive_connections()
                    
                    await asyncio.sleep(300)
                except Exception as e:
                    logger.error(f"   ❌ Cleanup error: {e}")
                    await asyncio.sleep(60)
        
        # راه‌اندازی WebSocket Manager
        try:
            async def handle_debug_message(client_id: str, message: Dict):
                try:
                    message_type = message.get('type')
                    if message_type == 'get_metrics':
                        current_metrics = metrics_collector.get_current_metrics()
                        await websocket_manager.send_message(client_id, {
                            'type': 'metrics_update',
                            'data': current_metrics,
                            'timestamp': datetime.now().isoformat()
                        })
                    elif message_type == 'get_alerts':
                        active_alerts = alert_manager.get_active_alerts()
                        await websocket_manager.send_message(client_id, {
                            'type': 'alerts_update',
                            'data': active_alerts,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    print(f"   ❌ WebSocket message handler error: {e}")
            
            websocket_manager.message_handlers['debug_message'] = handle_debug_message
            print("   ✅ WebSocket message handlers registered")
            
        except Exception as e:
            print(f"   ❌ WebSocket setup error: {e}")
        
        # راه‌اندازی سیستم لاگینگ real-time
        try:
            def log_to_console(level: str, message: str, data: Dict = None):
                if console_stream_manager:
                    console_stream_manager.broadcast_message({
                        'type': 'log_message',
                        'level': level,
                        'message': message,
                        'data': data or {},
                        'timestamp': datetime.now().isoformat()
                    })
            
            if hasattr(alert_manager, 'set_console_logger'):
                alert_manager.set_console_logger(log_to_console)
            
            if hasattr(debug_manager, 'set_console_logger'):
                debug_manager.set_console_logger(log_to_console)
                
            print("   ✅ Real-time logging configured")
            
        except Exception as e:
            print(f"   ❌ Real-time logging setup error: {e}")
        
        # تست اولیه سیستم‌ها
        print("   🧪 Running initial system tests...")
        try:
            current_metrics = metrics_collector.get_current_metrics()
            print(f"   ✅ Metrics collector: {len(current_metrics)} metrics collected")
            
            endpoint_stats = debug_manager.get_endpoint_stats()
            total_endpoints = len(endpoint_stats.get('endpoints', {}))
            print(f"   ✅ Debug manager: {total_endpoints} endpoints monitored")
            
            active_alerts = alert_manager.get_active_alerts()
            print(f"   ✅ Alert manager: {len(active_alerts)} active alerts")
            
            system_health = system_monitor.get_system_health()
            print(f"   ✅ System monitor: {system_health.get('overall_health', 'unknown')}")
            
            performance_report = performance_monitor.analyze_endpoint_performance()
            print(f"   ✅ Performance monitor: {len(performance_report.get('endpoint_performance', {}))} endpoints analyzed")
            
            security_report = security_monitor.get_security_report()
            print(f"   ✅ Security monitor: {security_report.get('total_suspicious_activities', 0)} security events")
            
        except Exception as e:
            print(f"   ⚠️ Initial tests had issues: {e}")
        
        print("✅ Complete debug system initialized and activated!")
        print(f"   📈 System Status:")
        print(f"   • Core Modules: {len(core_system) if core_system else 0} systems")
        print(f"   • Monitors: {len(monitors_system) if monitors_system else 0} monitors")
        print(f"   • Tools: {len(tools_system) if tools_system else 0} tools")
        print(f"   • Real-time: {'Active' if live_dashboard_manager else 'Inactive'}")
        print(f"   • WebSocket: {'Ready' if websocket_manager else 'Not ready'}")
        print(f"   • Console: {'Active' if console_stream_manager else 'Inactive'}")
        
    except Exception as e:
        print(f"❌ Debug system initialization error: {e}")
        import traceback
        traceback.print_exc()
        DEBUG_SYSTEM_AVAILABLE = False
        live_dashboard_manager = None
        console_stream_manager = None
else:
    print("❌ Debug system is not available")
    live_dashboard_manager = None
    console_stream_manager = None

# تنظیمات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VortexAI API", 
    version="4.0.0",
    description="Complete Crypto AI System with Advanced Debugging",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# بعد از ایجاد app (خط 400) این رو اضافه کن:

@app.on_event("startup")
async def startup_background_tasks():
    """شروع تسک‌های background بعد از راه‌اندازی سرور"""
    if DEBUG_SYSTEM_AVAILABLE and live_dashboard_manager:
        try:
            print("   🚀 Starting background tasks (on startup)...")
            
            # حالا event loop در حال اجراست
            asyncio.create_task(start_dashboard_broadcast())
            print("   ✅ Dashboard broadcast task started")
            
            asyncio.create_task(periodic_cleanup())
            print("   ✅ Periodic cleanup task started")
            
        except Exception as e:
            # 🔧 این خط رو هم اصلاح کن:
            logger.error(f"   ❌ Startup background tasks error: {e}")
    else:
        print("   ⚠️ Background tasks skipped - debug system not available")

# ثبت روت‌ها
app.include_router(health_router)
app.include_router(coins_router)
app.include_router(exchanges_router)
app.include_router(news_router)
app.include_router(insights_router)
app.include_router(raw_coins_router)
app.include_router(raw_news_router)
app.include_router(raw_insights_router)
app.include_router(raw_exchanges_router)
app.include_router(docs_router)

# ==================== DEBUG ROUTES ====================
@app.get("/api/debug/routes")
async def debug_all_routes():
    """لیست تمام مسیرهای ثبت شده"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, "name", "Unknown")
            })
    return {
        "total_routes": len(routes),
        "routes": routes
    }
    
if DEBUG_SYSTEM_AVAILABLE and live_dashboard_manager and console_stream_manager:
    @app.get("/debug/dashboard")
    async def debug_dashboard():
        """صفحه دشبورد دیباگ"""
        return FileResponse("debug_system/realtime/templates/dashboard.html")
    
    @app.get("/debug/console")
    async def debug_console():
        """صفحه کنسول دیباگ"""
        return FileResponse("debug_system/realtime/templates/console.html")
    
    @app.websocket("/debug/ws/dashboard")
    async def websocket_dashboard(websocket: WebSocket):
        """WebSocket برای دشبورد real-time"""
        await live_dashboard_manager.connect_dashboard(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            live_dashboard_manager.disconnect_dashboard(websocket)
    
    @app.websocket("/debug/ws/console")
    async def websocket_console(websocket: WebSocket):
        """WebSocket برای کنسول real-time"""
        await console_stream_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            console_stream_manager.disconnect(websocket)

# ==================== 🗺️ ROADMAP COMPLETE ====================

VORTEXAI_ROADMAP = {
    "project": "VortexAI API v4.0.0",
    "description": "Complete Crypto AI System with 9 Main Routes",
    "version": "4.0.0",
    "timestamp": datetime.now().isoformat(),
    
    "🚀 MAIN ROUTES": {
        "description": "۸ روت مادر اصلی سیستم",
        "routes": {
            "HEALTH": {
                "base_path": "/api/health",
                "description": "سلامت و مانیتورینگ سیستم",
                "endpoints": {
                    "status": "GET /api/health/status - وضعیت کلی سیستم",
                    "overview": "GET /api/health/overview - نمای کلی سیستم",
                    "ping": "GET /api/health/ping - تست حیات سیستم",
                    "version": "GET /api/health/version - نسخه‌های سیستم",
                    "debug_endpoints": "GET /api/health/debug/endpoints - دیباگ اندپوینت‌ها",
                    "debug_system": "GET /api/health/debug/system - دیباگ سیستم",
                    "debug_reports_daily": "GET /api/health/debug/reports/daily - گزارش روزانه",
                    "debug_reports_performance": "GET /api/health/debug/reports/performance - گزارش عملکرد",
                    "debug_reports_security": "GET /api/health/debug/reports/security - گزارش امنیتی",
                    "debug_metrics_live": "GET /api/health/debug/metrics/live - متریک‌های زنده"
                }
            },
            
            "COINS": {
                "base_path": "/api/coins",
                "description": "داده‌های پردازش شده نمادها",
                "endpoints": {
                    "list": "GET /api/coins/list - لیست نمادها",
                    "details": "GET /api/coins/details/{coin_id} - جزئیات نماد",
                    "charts": "GET /api/coins/charts/{coin_id} - چارت نماد", 
                    "multi_charts": "GET /api/coins/multi-charts - چارت چندنماد",
                    "price_avg": "GET /api/coins/price/avg - قیمت متوسط"
                }
            },
            
            "EXCHANGES": {
                "base_path": "/api/exchanges", 
                "description": "داده‌های پردازش شده صرافی‌ها",
                "endpoints": {
                    "list": "GET /api/exchanges/list - لیست صرافی‌ها",
                    "markets": "GET /api/exchanges/markets - مارکت‌ها",
                    "fiats": "GET /api/exchanges/fiats - ارزهای فیات",
                    "currencies": "GET /api/exchanges/currencies - ارزها",
                    "price": "GET /api/exchanges/price - قیمت صرافی"
                }
            },
            
            "NEWS": {
                "base_path": "/api/news",
                "description": "اخبار و تحلیل‌های پردازش شده", 
                "endpoints": {
                    "all": "GET /api/news/all - اخبار عمومی",
                    "by_type": "GET /api/news/type/{news_type} - اخبار بر اساس نوع",
                    "sources": "GET /api/news/sources - منابع خبری",
                    "detail": "GET /api/news/detail/{news_id} - جزئیات خبر"
                }
            },
            
            "INSIGHTS": {
                "base_path": "/api/insights",
                "description": "تحلیل‌های بازار و بینش‌ها",
                "endpoints": {
                    "btc_dominance": "GET /api/insights/btc-dominance - دامیننس بیت‌کوین",
                    "fear_greed": "GET /api/insights/fear-greed - شاخص ترس و طمع",
                    "fear_greed_chart": "GET /api/insights/fear-greed/chart - چارت ترس و طمع",
                    "rainbow_chart": "GET /api/insights/rainbow-chart/{coin_id} - چارت رنگین‌کمان"
                }
            },
            
            "RAW_COINS": {
                "base_path": "/api/raw/coins", 
                "description": "داده‌های خام نمادها - برای هوش مصنوعی",
                "endpoints": {
                    "list": "GET /api/raw/coins/list - لیست خام نمادها",
                    "details": "GET /api/raw/coins/details/{coin_id} - جزئیات خام نماد",
                    "charts": "GET /api/raw/coins/charts/{coin_id} - چارت خام نماد",
                    "multi_charts": "GET /api/raw/coins/multi-charts - چارت خام چندنماد",
                    "price_avg": "GET /api/raw/coins/price/avg - قیمت متوسط خام",
                    "exchange_price": "GET /api/raw/coins/price/exchange - قیمت صرافی خام",
                    "metadata": "GET /api/raw/coins/metadata - متادیتای نمادها",
                    "filters": "GET /api/raw/coins/filters - فیلترهای موجود"
                }
            },
            
            "RAW_NEWS": {
                "base_path": "/api/raw/news",
                "description": "داده‌های خام اخبار - برای هوش مصنوعی",
                "endpoints": {
                    "all": "GET /api/raw/news/all - اخبار عمومی خام", 
                    "by_type": "GET /api/raw/news/type/{news_type} - اخبار خام بر اساس نوع",
                    "sources": "GET /api/raw/news/sources - منابع خبری خام",
                    "detail": "GET /api/raw/news/detail/{news_id} - جزئیات خبر خام",
                    "sentiment_analysis": "GET /api/raw/news/sentiment-analysis - تحلیل احساسات",
                    "metadata": "GET /api/raw/news/metadata - متادیتای اخبار"
                }
            },
            
            "RAW_INSIGHTS": {
                "base_path": "/api/raw/insights",
                "description": "داده‌های خام بینش و تحلیل - برای هوش مصنوعی",
                "endpoints": {
                    "btc_dominance": "GET /api/raw/insights/btc-dominance - دامیننس بیت‌کوین خام",
                    "fear_greed": "GET /api/raw/insights/fear-greed - شاخص ترس و طمع خام", 
                    "fear_greed_chart": "GET /api/raw/insights/fear-greed/chart - چارت ترس و طمع خام",
                    "rainbow_chart": "GET /api/raw/insights/rainbow-chart/{coin_id} - چارت رنگین‌کمان خام",
                    "market_analysis": "GET /api/raw/insights/market-analysis - تحلیل جامع بازار",
                    "metadata": "GET /api/raw/insights/metadata - متادیتای بینش‌ها"
                }
            }
        }
    },
    
    "📚 DOCUMENTATION": {
        "description": "مستندات کامل و مثال‌های کاربردی",
        "routes": {
            "complete_docs": "GET /api/docs/complete - مستندات کامل API",
            "coins_docs": "GET /api/docs/coins - مستندات تخصصی نمادها", 
            "code_examples": "GET /api/docs/examples - مثال‌های کد",
            "interactive_docs": "GET /docs - مستندات تعاملی (Swagger UI)",
            "redoc_docs": "GET /redoc - مستندات زیبا (ReDoc)"
        }
    },
    
    "🔧 DEBUG & MONITORING": {
        "description": "سیستم دیباگ و مانیتورینگ پیشرفته",
        "routes": {
            "DEBUG_DASHBOARD": "GET /debug/dashboard - دشبورد دیباگ",
            "DEBUG_CONSOLE": "GET /debug/console - کنسول دیباگ",
            "DEBUG_WS_DASHBOARD": "WS /debug/ws/dashboard - WebSocket دشبورد",
            "DEBUG_WS_CONSOLE": "WS /debug/ws/console - WebSocket کنسول",
            "METRICS_ALL": "GET /api/health/metrics - تمام متریک‌ها",
            "ALERTS_ACTIVE": "GET /api/health/alerts - هشدارهای فعال",
            "REPORTS_DAILY": "GET /api/health/reports/daily - گزارش روزانه",
            "REALTIME_CONSOLE": "WS /api/health/debug/realtime/console - کنسول Real-Time",
            "REALTIME_DASHBOARD": "WS /api/health/debug/realtime/dashboard - دشبورد Real-Time"
        }
    }
}

@app.get("/")
async def root():
    """صفحه اصلی با راهنمای کامل روت‌ها"""
    return {
        "message": "🚀 VortexAI API Server v4.0.0 - Complete Crypto AI System",
        "version": "4.0.0", 
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc", 
            "roadmap": "/api/roadmap",
            "complete_docs": "/api/docs/complete",
            "code_examples": "/api/docs/examples"
        },
        "quick_start": {
            "health_check": "/api/health/status",
            "bitcoin_data": "/api/coins/details/bitcoin",
            "latest_news": "/api/news/all?limit=5",
            "market_sentiment": "/api/insights/fear-greed",
            "ai_data_samples": "/api/raw/coins/metadata",
            "debug_endpoints": "/api/health/debug/endpoints",
            "debug_system": "/api/health/debug/system"
        },
        "system_info": {
            "total_routes": len(app.routes),
            "debug_system": "active" if DEBUG_SYSTEM_AVAILABLE else "inactive",
            "coinstats_available": COINSTATS_AVAILABLE,
            "startup_time": datetime.now().isoformat(),
            "ai_ready": True
        }
    }

@app.get("/api/roadmap")
async def get_roadmap():
    """دریافت راهنمای کامل روت‌های سیستم"""
    return VORTEXAI_ROADMAP

@app.get("/api/quick-reference")
async def quick_reference():
    """مرجع سریع روت‌های مهم"""
    return {
        "title": "VortexAI API - Quick Reference",
        "description": "مرجع سریع برای دسترسی به اندپوینت‌های اصلی",
        "timestamp": datetime.now().isoformat(),
        
        "essential_endpoints": {
            "health": {
                "url": "/api/health/status",
                "description": "بررسی سلامت سیستم"
            },
            "coins_list": {
                "url": "/api/coins/list", 
                "description": "لیست نمادها"
            },
            "coin_details": {
                "url": "/api/coins/details/{coin_id}",
                "description": "جزئیات نماد خاص"
            },
            "coin_charts": {
                "url": "/api/coins/charts/{coin_id}",
                "description": "داده‌های چارت"
            },
            "news": {
                "url": "/api/news/all",
                "description": "اخبار بازار"
            },
            "fear_greed": {
                "url": "/api/insights/fear-greed",
                "description": "شاخص ترس و طمع"
            },
            "exchanges": {
                "url": "/api/exchanges/list",
                "description": "لیست صرافی‌ها"
            }
        }
    }

@app.get("/api/endpoints/count")
async def count_endpoints():
    """شمردن تعداد کل اندپوینت‌ها"""
    total_endpoints = 0
    routes_info = []
    
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            total_endpoints += len(route.methods)
            routes_info.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, "name", "Unknown")
            })
    
    return {
        "total_endpoints": total_endpoints,
        "total_routes": len(app.routes),
        "timestamp": datetime.now().isoformat(),
        "routes_by_category": {
            "health": len([r for r in routes_info if '/api/health' in r['path']]),
            "coins": len([r for r in routes_info if '/api/coins' in r['path']]),
            "raw_coins": len([r for r in routes_info if '/api/raw/coins' in r['path']]),
            "news": len([r for r in routes_info if '/api/news' in r['path']]),
            "raw_news": len([r for r in routes_info if '/api/raw/news' in r['path']]),
            "insights": len([r for r in routes_info if '/api/insights' in r['path']]),
            "raw_insights": len([r for r in routes_info if '/api/raw/insights' in r['path']]),
            "exchanges": len([r for r in routes_info if '/api/exchanges' in r['path']]),
            "documentation": len([r for r in routes_info if '/api/docs' in r['path']]),
            "debug": len([r for r in routes_info if '/debug' in r['path']])
        },
        "sample_routes": routes_info[:10]
    }

@app.get("/api/system/info")
async def system_info():
    """اطلاعات کامل سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "server_time": datetime.now().isoformat(),
            "uptime_seconds": int(time.time() - psutil.boot_time())
        },
        "resources": {
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2)
        },
        "api_status": {
            "total_endpoints": len(app.routes),
            "coinstats_available": COINSTATS_AVAILABLE,
            "debug_system_available": DEBUG_SYSTEM_AVAILABLE,
            "debug_system_status": "active" if DEBUG_SYSTEM_AVAILABLE else "inactive",
            "version": "4.0.0",
            "ai_ready": True
        },
        "timestamp": datetime.now().isoformat()
    }

# مدیریت خطای 404
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "timestamp": datetime.now().isoformat(),
            "suggestions": {
                "check_docs": "Visit /api/docs/complete for complete documentation",
                "check_roadmap": "Visit /api/roadmap for system overview", 
                "check_health": "Visit /api/health/status to check system health",
                "common_endpoints": {
                    "health": "/api/health/status",
                    "coins_list": "/api/coins/list", 
                    "news": "/api/news/all",
                    "insights": "/api/insights/fear-greed",
                    "ai_data": "/api/raw/coins/metadata",
                    "debug_endpoints": "/api/health/debug/endpoints"
                }
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    
    print("🚀" * 50)
    print("🎯 VORTEXAI API SERVER v4.0.0 - AI READY")
    print("🚀" * 50)
    print(f"📊 Total Routes: {len(app.routes)}")
    print(f"🌐 Server URL: http://localhost:{port}")
    print(f"📚 Documentation: http://localhost:{port}/docs")
    print(f"🗺️  Roadmap: http://localhost:{port}/api/roadmap")
    print(f"📖 Complete Docs: http://localhost:{port}/api/docs/complete")
    print("🎯 Quick Start:")
    print(f"   • Health Check: http://localhost:{port}/api/health/status")
    print(f"   • Bitcoin Details: http://localhost:{port}/api/coins/details/bitcoin") 
    print(f"   • Latest News: http://localhost:{port}/api/news/all?limit=5")
    print(f"   • Fear & Greed: http://localhost:{port}/api/insights/fear-greed")
    print(f"   • AI Data Samples: http://localhost:{port}/api/raw/coins/metadata")
    print(f"   • Debug Endpoints: http://localhost:{port}/api/health/debug/endpoints")
    print(f"   • Debug System: http://localhost:{port}/api/health/debug/system")
    print("🔧 Debug System: " + ("✅ FULLY ACTIVE" if DEBUG_SYSTEM_AVAILABLE else "❌ UNAVAILABLE"))
    if DEBUG_SYSTEM_AVAILABLE:
        print(f"   • Real-time Dashboard: http://localhost:{port}/debug/dashboard")
        print(f"   • Debug Console: http://localhost:{port}/debug/console")
        print(f"   • System Reports: http://localhost:{port}/api/health/debug/reports/daily")
    print("🤖 AI Ready: ✅ YES")
    print("📈 CoinStats API: " + ("✅ AVAILABLE" if COINSTATS_AVAILABLE else "❌ UNAVAILABLE"))
    print("🚀" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=True)

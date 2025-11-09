from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import psutil
import logging
import os

logger = logging.getLogger(__name__)

# ایجاد روت‌ر سلامت
health_router = APIRouter(prefix="/api/health", tags=["Health & Debug"])

# ==================== LAZY DEBUG SYSTEM IMPORTS ====================

class DebugSystemManager:
    """مدیریت lazy loading برای سیستم دیباگ"""
    
    _initialized = False
    _modules = {}
    
    @classmethod
    def initialize(cls):
        """مقداردهی اولیه lazy سیستم دیباگ - نسخه اصلاح شده"""
        if cls._initialized:
            return cls._modules
        
        try:
            logger.info("🔄 Initializing debug system (lazy loading)...")
            
            # ایمپورت core modules - اینها همیشه باید کار کنند
            from debug_system.core.debug_manager import debug_manager
            from debug_system.core.metrics_collector import metrics_collector
            from debug_system.core.alert_manager import alert_manager, AlertLevel, AlertType
            
            cls._modules.update({
                'debug_manager': debug_manager,
                'metrics_collector': metrics_collector,
                'alert_manager': alert_manager,
                'AlertLevel': AlertLevel,
                'AlertType': AlertType
            })
            
            # ایمپورت monitors - با dependency injection درست
            try:
                from debug_system.monitors.endpoint_monitor import EndpointMonitor
                from debug_system.monitors.system_monitor import SystemMonitor
                from debug_system.monitors.performance_monitor import PerformanceMonitor
                from debug_system.monitors.security_monitor import SecurityMonitor
                
                # ایجاد نمونه با dependencyهای لازم
                endpoint_monitor = EndpointMonitor(debug_manager)
                system_monitor = SystemMonitor(metrics_collector, alert_manager)
                performance_monitor = PerformanceMonitor(debug_manager, alert_manager)
                security_monitor = SecurityMonitor(alert_manager)
                
                cls._modules.update({
                    'endpoint_monitor': endpoint_monitor,
                    'system_monitor': system_monitor,
                    'performance_monitor': performance_monitor,
                    'security_monitor': security_monitor
                })
                
                logger.info("✅ Monitors initialized with dependency injection")
                
            except ImportError as e:
                logger.warning(f"⚠️ Could not load monitors: {e}")
            except Exception as e:
                logger.error(f"❌ Error initializing monitors: {e}")
            
            # ایمپورت storage
            try:
                from debug_system.storage.history_manager import history_manager
                from debug_system.storage.cache_debugger import cache_debugger
                
                cls._modules.update({
                    'history_manager': history_manager,
                    'cache_debugger': cache_debugger
                })
                
                logger.info("✅ Storage modules loaded")
                
            except ImportError as e:
                logger.warning(f"⚠️ Could not load storage: {e}")
            except Exception as e:
                logger.error(f"❌ Error loading storage: {e}")
            
            # ایمپورت realtime
            try:
                from debug_system.realtime.live_dashboard import LiveDashboardManager
                from debug_system.realtime.console_stream import ConsoleStreamManager
                
                # ایجاد live dashboard با dependency
                live_dashboard = LiveDashboardManager(debug_manager, metrics_collector)
                console_stream = ConsoleStreamManager()
                
                cls._modules.update({
                    'live_dashboard': live_dashboard,
                    'console_stream': console_stream
                })
                
                logger.info("✅ Realtime modules initialized")
                
            except ImportError as e:
                logger.warning(f"⚠️ Could not load realtime: {e}")
            except Exception as e:
                logger.error(f"❌ Error initializing realtime: {e}")
            
            # ایمپورت tools - این مشکل اصلی بود!
            try:
                from debug_system.tools.report_generator import ReportGenerator
                from debug_system.tools.dev_tools import DevTools
                from debug_system.tools.testing_tools import TestingTools
                
                # ایجاد tools با dependencyهای لازم
                history_manager_instance = cls._modules.get('history_manager')
                report_generator = ReportGenerator(debug_manager, history_manager_instance)
                dev_tools = DevTools(debug_manager)
                testing_tools = TestingTools(debug_manager)
                
                cls._modules.update({
                    'report_generator': report_generator,
                    'dev_tools': dev_tools,
                    'testing_tools': testing_tools
                })
                
                logger.info("✅ Tools initialized with dependencies")
                
            except ImportError as e:
                logger.error(f"❌ Could not load tools: {e}")
            except Exception as e:
                logger.error(f"❌ Error initializing tools: {e}")
            
            cls._initialized = True
            
            # لاگ ماژول‌های load شده
            loaded_modules = [name for name, module in cls._modules.items() if module is not None]
            failed_modules = [name for name, module in cls._modules.items() if module is None]
            
            logger.info(f"✅ Debug system initialization completed")
            logger.info(f"📦 Loaded modules ({len(loaded_modules)}): {loaded_modules}")
            
            if failed_modules:
                logger.warning(f"⚠️ Failed modules ({len(failed_modules)}): {failed_modules}")
            
        except Exception as e:
            logger.error(f"❌ Debug system initialization failed: {e}")
            # حتی اگر خطا داد، حداقل core modules را نگه دار
            cls._modules = cls._modules or {}
        
        return cls._modules
    
    @classmethod
    def get_module(cls, module_name: str, default=None):
        """دریافت یک ماژول از سیستم دیباگ - نسخه اصلاح شده"""
        if not cls._initialized:
            cls.initialize()
        
        module = cls._modules.get(module_name, default)
        
        # اگر ماژول None باشد، پیام خطای مفید
        if module is None and module_name in cls._modules:
            logger.warning(f"⚠️ Module '{module_name}' is None")
        
        return module
    
    @classmethod
    def is_available(cls):
        """بررسی آیا سیستم دیباگ در دسترس است"""
        if not cls._initialized:
            cls.initialize()
        return bool(cls._modules.get('debug_manager'))
    
    @classmethod
    def get_status_report(cls):
        """دریافت گزارش وضعیت سیستم دیباگ"""
        if not cls._initialized:
            cls.initialize()
        
        loaded_modules = [name for name, module in cls._modules.items() if module is not None]
        failed_modules = [name for name, module in cls._modules.items() if module is None]
        
        return {
            'initialized': cls._initialized,
            'total_modules': len(cls._modules),
            'loaded_modules': len(loaded_modules),
            'failed_modules': len(failed_modules),
            'available_modules': loaded_modules,
            'missing_modules': failed_modules,
            'core_available': bool(cls._modules.get('debug_manager'))
        }

# تابع کمکی برای دسترسی آسان به ماژول‌ها
def get_debug_module(module_name: str):
    """دریافت ماژول دیباگ با مدیریت خطا - نسخه نهایی"""
    module = DebugSystemManager.get_module(module_name)
    
    if module is None:
        status_report = DebugSystemManager.get_status_report()
        
        logger.error(f"❌ Debug module '{module_name}' is not available. Status: {status_report}")
        
        raise HTTPException(
            status_code=503, 
            detail={
                "error": f"Debug module '{module_name}' not properly initialized",
                "system_status": status_report,
                "hint": "Check server logs for initialization errors"
            }
        )
    
    return module

# ==================== BASIC HEALTH ENDPOINTS ====================

@health_router.get("/status")
async def health_status():
    """بررسی سلامت کلی سیستم"""
    debug_status = DebugSystemManager.get_status_report()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "services": {
            "api": "running",
            "database": "connected",
            "cache": "connected",
            "external_apis": "available",
            "debug_system": {
                "available": debug_status['core_available'],
                "loaded_modules": debug_status['loaded_modules'],
                "total_modules": debug_status['total_modules'],
                "status": "fully_initialized" if debug_status['loaded_modules'] == debug_status['total_modules'] else "partially_initialized"
            }
        }
    }

@health_router.get("/overview")
async def system_overview():
    """نمای کلی سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    debug_status = DebugSystemManager.get_status_report()
    
    return {
        "system": {
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "server_time": datetime.now().isoformat(),
            "platform": os.name
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
        "debug_system": debug_status
    }

@health_router.get("/ping")
async def health_ping():
    """تست ساده حیات سیستم"""
    return {
        "message": "pong",
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/version")
async def version_info():
    """اطلاعات نسخه‌های سیستم"""
    import sys
    return {
        "api_version": "4.0.0",
        "python_version": sys.version,
        "fastapi_version": "0.104.1",
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/metrics/system")
async def system_metrics():
    """متریک‌های سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "per_core": psutil.cpu_percent(percpu=True, interval=1),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        },
        "memory": {
            "percent": memory.percent,
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2)
        },
        "disk": {
            "usage_percent": disk.percent,
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "total_gb": round(disk.total / (1024**3), 2)
        },
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        },
        "timestamp": datetime.now().isoformat()
    }

# ==================== DEBUG ENDPOINTS ====================

@health_router.get("/debug/endpoints")
async def debug_endpoints():
    """دریافت وضعیت دیباگ اندپوینت‌ها"""
    endpoint_monitor = get_debug_module('endpoint_monitor')
    performance_monitor = get_debug_module('performance_monitor')
    
    return {
        "endpoint_health": endpoint_monitor.get_all_endpoints_health(),
        "performance_report": performance_monitor.get_performance_report(),
        "bottlenecks": performance_monitor.analyze_bottlenecks(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/system")
async def debug_system():
    """دریافت وضعیت کامل سیستم دیباگ"""
    system_monitor = get_debug_module('system_monitor')
    security_monitor = get_debug_module('security_monitor')
    alert_manager = get_debug_module('alert_manager')
    
    return {
        "system_health": system_monitor.get_system_health(),
        "security_report": security_monitor.get_security_report(),
        "active_alerts": alert_manager.get_active_alerts(),
        "resource_usage": system_monitor.get_resource_usage_trend(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/reports/daily")
async def debug_daily_report():
    """دریافت گزارش روزانه دیباگ"""
    report_generator = get_debug_module('report_generator')
    return report_generator.generate_daily_report()

@health_router.get("/debug/reports/performance")
async def debug_performance_report():
    """دریافت گزارش عملکرد دیباگ"""
    report_generator = get_debug_module('report_generator')
    return report_generator.generate_performance_report()

@health_router.get("/debug/reports/security")
async def debug_security_report():
    """دریافت گزارش امنیتی دیباگ"""
    report_generator = get_debug_module('report_generator')
    return report_generator.generate_security_report()

@health_router.get("/debug/metrics/live")
async def debug_live_metrics():
    """دریافت متریک‌های real-time"""
    metrics_collector = get_debug_module('metrics_collector')
    debug_manager = get_debug_module('debug_manager')
    performance_monitor = get_debug_module('performance_monitor')
    
    return {
        "system_metrics": metrics_collector.get_current_metrics(),
        "endpoint_metrics": debug_manager.get_endpoint_stats(),
        "performance_metrics": performance_monitor.get_performance_report(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/alerts")
async def debug_alerts():
    """دریافت هشدارهای فعال سیستم"""
    alert_manager = get_debug_module('alert_manager')
    
    return {
        "active_alerts": alert_manager.get_active_alerts(),
        "alert_stats": alert_manager.get_alert_stats(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.post("/debug/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, user: str = "system"):
    """تأیید هشدار"""
    alert_manager = get_debug_module('alert_manager')
    success = alert_manager.acknowledge_alert(alert_id, user)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {
        "message": f"Alert {alert_id} acknowledged by {user}",
        "alert_id": alert_id,
        "acknowledged_by": user,
        "timestamp": datetime.now().isoformat()
    }

@health_router.post("/debug/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int, resolved_by: str = "system", resolution_notes: str = ""):
    """حل هشدار"""
    alert_manager = get_debug_module('alert_manager')
    success = alert_manager.resolve_alert(alert_id, resolved_by, resolution_notes)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {
        "message": f"Alert {alert_id} resolved by {resolved_by}",
        "alert_id": alert_id,
        "resolved_by": resolved_by,
        "resolution_notes": resolution_notes,
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/performance/bottlenecks")
async def debug_performance_bottlenecks():
    """دریافت bottlenecks عملکرد"""
    performance_monitor = get_debug_module('performance_monitor')
    
    return {
        "bottlenecks": performance_monitor.analyze_bottlenecks(),
        "slowest_endpoints": performance_monitor.get_slowest_endpoints(),
        "most_called_endpoints": performance_monitor.get_most_called_endpoints(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/security/overview")
async def debug_security_overview():
    """نمای کلی امنیتی"""
    security_monitor = get_debug_module('security_monitor')
    
    return {
        "security_report": security_monitor.get_security_report(),
        "ip_reputation_sample": {
            "127.0.0.1": security_monitor.get_ip_reputation("127.0.0.1")
        },
        "timestamp": datetime.now().isoformat()
    }

# ==================== REAL-TIME ENDPOINTS ====================

@health_router.websocket("/debug/realtime/console")
async def websocket_console(websocket: WebSocket):
    """WebSocket برای کنسول Real-Time"""
    console_stream = get_debug_module('console_stream')
    
    await console_stream.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await console_stream.broadcast_message({
                "type": "client_message",
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        console_stream.disconnect(websocket)

@health_router.websocket("/debug/realtime/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket برای دشبورد Real-Time"""
    live_dashboard = get_debug_module('live_dashboard')
    
    await live_dashboard.connect_dashboard(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        live_dashboard.disconnect_dashboard(websocket)

# ==================== METRICS ENDPOINTS ====================

@health_router.get("/metrics")
async def get_all_metrics():
    """دریافت تمام متریک‌های سیستم"""
    metrics_collector = get_debug_module('metrics_collector')
    debug_manager = get_debug_module('debug_manager')
    cache_debugger = get_debug_module('cache_debugger')
    performance_monitor = get_debug_module('performance_monitor')
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_metrics": metrics_collector.get_current_metrics(),
        "endpoint_metrics": debug_manager.get_endpoint_stats(),
        "cache_metrics": cache_debugger.get_cache_stats(),
        "performance_metrics": performance_monitor.analyze_endpoint_performance()
    }

@health_router.get("/metrics/system")
async def get_system_metrics_detailed():
    """متریک‌های دقیق سیستم"""
    metrics_collector = get_debug_module('metrics_collector')
    return metrics_collector.get_detailed_metrics()

@health_router.get("/metrics/endpoints")
async def get_endpoints_metrics():
    """متریک‌های اندپوینت‌ها"""
    debug_manager = get_debug_module('debug_manager')
    return debug_manager.get_endpoint_stats()

@health_router.get("/metrics/cache")
async def get_cache_metrics():
    """متریک‌های عملکرد کش"""
    cache_debugger = get_debug_module('cache_debugger')
    
    return {
        "stats": cache_debugger.get_cache_stats(),
        "performance": cache_debugger.get_cache_performance(),
        "efficiency": cache_debugger.analyze_cache_efficiency()
    }

# ==================== ALERTS ENDPOINTS ====================

@health_router.get("/alerts")
async def get_active_alerts(
    level: str = Query(None, regex="^(INFO|WARNING|ERROR|CRITICAL)$"),
    alert_type: str = Query(None),
    source: str = Query(None)
):
    """دریافت هشدارهای فعال"""
    alert_manager = get_debug_module('alert_manager')
    AlertLevel = get_debug_module('AlertLevel')
    AlertType = get_debug_module('AlertType')
    
    return alert_manager.get_active_alerts(
        level=AlertLevel(level) if level else None,
        alert_type=AlertType(alert_type) if alert_type else None,
        source=source
    )

@health_router.get("/alerts/history")
async def get_alert_history(
    level: str = Query(None, regex="^(INFO|WARNING|ERROR|CRITICAL)$"),
    alert_type: str = Query(None),
    source: str = Query(None),
    hours: int = Query(24, ge=1, le=720),
    limit: int = Query(100, ge=1, le=1000)
):
    """تاریخچه هشدارها"""
    alert_manager = get_debug_module('alert_manager')
    AlertLevel = get_debug_module('AlertLevel')
    AlertType = get_debug_module('AlertType')
    
    start_date = datetime.now() - timedelta(hours=hours)
    
    return alert_manager.get_alert_history(
        level=AlertLevel(level) if level else None,
        alert_type=AlertType(alert_type) if alert_type else None,
        source=source,
        start_date=start_date,
        end_date=datetime.now(),
        limit=limit
    )

@health_router.get("/alerts/stats")
async def get_alert_stats(hours: int = Query(24, ge=1, le=720)):
    """آمار هشدارها"""
    alert_manager = get_debug_module('alert_manager')
    return alert_manager.get_alert_stats(hours)

# ==================== REPORTS ENDPOINTS ====================

@health_router.get("/reports/daily")
async def get_daily_report(date: str = None):
    """گزارش روزانه عملکرد سیستم"""
    report_generator = get_debug_module('report_generator')
    report_date = datetime.strptime(date, '%Y-%m-%d') if date else datetime.now()
    return report_generator.generate_daily_report(report_date)

@health_router.get("/reports/performance")
async def get_performance_report(days: int = Query(7, ge=1, le=30)):
    """گزارش عملکرد سیستم"""
    report_generator = get_debug_module('report_generator')
    return report_generator.generate_performance_report(days)

@health_router.get("/reports/security")
async def get_security_report(days: int = Query(30, ge=1, le=90)):
    """گزارش امنیتی سیستم"""
    report_generator = get_debug_module('report_generator')
    return report_generator.generate_security_report(days)

# ==================== TOOLS ENDPOINTS ====================

@health_router.post("/tools/test-traffic")
async def generate_test_traffic(
    background_tasks: BackgroundTasks,
    endpoint: str = None,
    duration_seconds: int = 60,
    requests_per_second: int = 10
):
    """تولید ترافیک تست برای شبیه‌سازی بار"""
    dev_tools = get_debug_module('dev_tools')
    
    background_tasks.add_task(
        dev_tools.generate_test_traffic,
        endpoint,
        duration_seconds,
        requests_per_second
    )
    
    return {
        "status": "test_traffic_started",
        "endpoint": endpoint,
        "duration_seconds": duration_seconds,
        "requests_per_second": requests_per_second,
        "started_at": datetime.now().isoformat()
    }

@health_router.post("/tools/load-test")
async def run_load_test(
    background_tasks: BackgroundTasks,
    endpoint: str,
    concurrent_users: int = 10,
    duration_seconds: int = 60
):
    """اجرای تست بار برای اندپوینت"""
    testing_tools = get_debug_module('testing_tools')
    
    background_tasks.add_task(
        testing_tools.run_load_test,
        endpoint,
        concurrent_users,
        duration_seconds
    )
    
    return {
        "status": "load_test_started",
        "endpoint": endpoint,
        "concurrent_users": concurrent_users,
        "duration_seconds": duration_seconds
    }

@health_router.get("/tools/dependencies")
async def check_dependencies():
    """بررسی وضعیت وابستگی‌های سیستم"""
    dev_tools = get_debug_module('dev_tools')
    return dev_tools.run_dependency_check()

@health_router.get("/tools/memory-analysis")
async def analyze_memory_usage():
    """آنالیز استفاده از حافظه"""
    dev_tools = get_debug_module('dev_tools')
    return dev_tools.analyze_memory_usage()

@health_router.get("/tools/cache-stats")
async def get_cache_stats():
    """آمار کامل کش سیستم"""
    cache_debugger = get_debug_module('cache_debugger')
    
    return {
        "cache_stats": cache_debugger.get_cache_stats(),
        "cache_performance": cache_debugger.get_cache_performance(),
        "cache_efficiency": cache_debugger.analyze_cache_efficiency(),
        "most_accessed_keys": cache_debugger.get_most_accessed_keys(),
        "timestamp": datetime.now().isoformat()
    }

# ==================== INITIALIZATION ====================

@health_router.on_event("startup")
async def startup_event():
    """رویداد startup برای مقداردهی اولیه سیستم دیباگ"""
    logger.info("🚀 Initializing debug system on startup...")
    DebugSystemManager.initialize()
    
    # گزارش وضعیت نهایی
    status = DebugSystemManager.get_status_report()
    logger.info(f"🎉 Debug system startup completed. Loaded {status['loaded_modules']}/{status['total_modules']} modules")

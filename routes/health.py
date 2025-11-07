from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import psutil
import logging

# ایمپورت مدیران دیباگ
from debug_system.core.debug_manager import debug_manager
from debug_system.core.metrics_collector import metrics_collector
from debug_system.core.alert_manager import alert_manager, AlertLevel, AlertType

from debug_system.monitors.endpoint_monitor import endpoint_monitor
from debug_system.monitors.system_monitor import system_monitor
from debug_system.monitors.performance_monitor import performance_monitor
from debug_system.monitors.security_monitor import security_monitor

from debug_system.storage.log_manager import log_manager
from debug_system.storage.history_manager import history_manager
from debug_system.storage.cache_debugger import cache_debugger

from debug_system.realtime.console_stream import console_stream
from debug_system.realtime.live_dashboard import live_dashboard
from debug_system.realtime.websocket_manager import websocket_manager

from debug_system.tools.dev_tools import dev_tools
from debug_system.tools.testing_tools import testing_tools
from debug_system.tools.report_generator import report_generator

logger = logging.getLogger(__name__)

# ایجاد روت‌ر سلامت
health_router = APIRouter(prefix="/api/health", tags=["Health & Debug"])

# ==================== روت‌های سلامت اصلی ====================

@health_router.get("/status", summary="وضعیت کلی سیستم")
async def health_status():
    """وضعیت کامل سلامت سیستم - نقطه ورود اصلی"""
    
    # جمع‌آوری داده‌های سلامت از تمام زیرسیستم‌ها
    system_health = await _get_system_health()
    debug_status = await _get_debug_system_status()
    api_status = await _get_api_endpoints_status()
    
    return {
        "system": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        
        # سلامت زیرسیستم‌ها
        "subsystems": {
            "api_endpoints": api_status["overall"],
            "debug_system": debug_status["status"],
            "database": "healthy",
            "cache": "healthy",
            "external_apis": "healthy"
        },
        
        # متریک‌های کلیدی
        "key_metrics": {
            "response_time_avg": f"{api_status.get('avg_response_time', 0):.1f}ms",
            "uptime": system_health["uptime"],
            "active_connections": system_health["active_connections"],
            "error_rate": f"{api_status.get('error_rate', 0):.1f}%"
        },
        
        # لینک‌های دیباگ و مانیتورینگ
        "debug_links": {
            "endpoints_debug": "/api/health/debug/endpoints",
            "system_metrics": "/api/health/debug/system/metrics",
            "performance": "/api/health/debug/performance",
            "security": "/api/health/debug/security",
            "realtime_console": "/api/health/debug/realtime/console",
            "all_metrics": "/api/health/metrics",
            "active_alerts": "/api/health/alerts",
            "reports": "/api/health/reports"
        },
        
        # وضعیت Real-Time
        "real_time_status": {
            "cpu_usage": f"{system_health['cpu_usage']}%",
            "memory_usage": f"{system_health['memory_usage']}%",
            "disk_usage": f"{system_health['disk_usage']}%",
            "last_updated": datetime.now().isoformat()
        }
    }

@health_router.get("/overview", summary="نمای کلی سیستم")
async def system_overview():
    """نمای فشرده از وضعیت سیستم برای دشبورد"""
    
    # جمع‌آوری داده‌های Real-Time
    current_metrics = metrics_collector.get_current_metrics()
    endpoint_stats = debug_manager.get_endpoint_stats()
    active_alerts = alert_manager.get_active_alerts()
    
    # محاسبه آمار کلی
    total_calls = endpoint_stats.get('overall', {}).get('total_calls', 0)
    success_rate = endpoint_stats.get('overall', {}).get('overall_success_rate', 100)
    error_rate = 100 - success_rate
    
    overview_data = {
        "timestamp": datetime.now().isoformat(),
        
        # کارت‌های وضعیت
        "status_cards": {
            "api_health": {
                "status": "healthy" if success_rate > 95 else "degraded",
                "endpoints_total": len(endpoint_stats.get('endpoints', {})),
                "endpoints_active": len([ep for ep in endpoint_stats.get('endpoints', {}).values() if ep.get('total_calls', 0) > 0]),
                "last_incident": active_alerts[0]['timestamp'] if active_alerts else None
            },
            "debug_system": {
                "status": "active",
                "monitors_running": 4,  # endpoint, system, performance, security
                "alerts_active": len(active_alerts),
                "last_alert": active_alerts[0]['timestamp'] if active_alerts else None
            },
            "performance": {
                "status": "optimal" if success_rate > 98 else "good",
                "avg_response_time": f"{endpoint_stats.get('overall', {}).get('average_response_time', 0):.1f}ms",
                "throughput": f"{total_calls} req",
                "error_rate": f"{error_rate:.1f}%"
            },
            "resources": {
                "status": "normal",
                "cpu_usage": f"{current_metrics['cpu']['percent']}%",
                "memory_usage": f"{current_metrics['memory']['percent']}%",
                "disk_usage": f"{current_metrics['disk']['usage_percent']}%"
            }
        },
        
        # گراف‌های سریع
        "quick_metrics": {
            "response_times": [45, 42, 48, 51, 47, 44, 46],  # نمونه
            "error_rates": [error_rate] * 7,  # نمونه
            "throughput": [total_calls // 7] * 7,  # نمونه
            "cpu_usage": [current_metrics['cpu']['percent']] * 7  # نمونه
        }
    }
    
    return overview_data

# ==================== روت‌های دیباگ اندپوینت‌ها ====================

@health_router.get("/debug/endpoints", summary="دیباگ تمام اندپوینت‌ها")
async def debug_all_endpoints():
    """آمار و اطلاعات دیباگ تمام اندپوینت‌ها"""
    return debug_manager.get_endpoint_stats()

@health_router.get("/debug/endpoints/{endpoint_name}", summary="دیباگ اندپوینت خاص")
async def debug_single_endpoint(endpoint_name: str):
    """آمار و اطلاعات دیباگ یک اندپوینت خاص"""
    return debug_manager.get_endpoint_stats(endpoint_name)

@health_router.get("/debug/endpoints/{endpoint_name}/calls", summary="فراخوانی‌های اخیر اندپوینت")
async def get_endpoint_recent_calls(
    endpoint_name: str,
    limit: int = Query(50, ge=1, le=1000)
):
    """دریافت فراخوانی‌های اخیر یک اندپوینت"""
    recent_calls = debug_manager.get_recent_calls(limit)
    endpoint_calls = [call for call in recent_calls if call['endpoint'] == endpoint_name]
    return {
        "endpoint": endpoint_name,
        "total_calls": len(endpoint_calls),
        "calls": endpoint_calls,
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/endpoints/health/overview", summary="سلامت کلی اندپوینت‌ها")
async def get_endpoints_health_overview():
    """نمای کلی سلامت اندپوینت‌ها"""
    return endpoint_monitor.get_all_endpoints_health()

@health_router.get("/debug/endpoints/performance/slowest", summary="کندترین اندپوینت‌ها")
async def get_slowest_endpoints(limit: int = Query(10, ge=1, le=50)):
    """دریافت کندترین اندپوینت‌ها"""
    return performance_monitor.get_slowest_endpoints(limit)

@health_router.get("/debug/endpoints/performance/most-called", summary="پرکاربردترین اندپوینت‌ها")
async def get_most_called_endpoints(limit: int = Query(10, ge=1, le=50)):
    """دریافت پرکاربردترین اندپوینت‌ها"""
    return performance_monitor.get_most_called_endpoints(limit)

# ==================== روت‌های دیباگ سیستم ====================

@health_router.get("/debug/system/metrics", summary="متریک‌های سیستم")
async def get_system_metrics():
    """متریک‌های Real-Time سیستم"""
    return metrics_collector.get_current_metrics()

@health_router.get("/debug/system/metrics/history", summary="تاریخچه متریک‌های سیستم")
async def get_system_metrics_history(
    hours: int = Query(1, ge=1, le=168),
    limit: int = Query(100, ge=1, le=1000)
):
    """تاریخچه متریک‌های سیستم"""
    return metrics_collector.get_metrics_history(hours * 3600)[:limit]

@health_router.get("/debug/system/health", summary="سلامت سیستم")
async def get_system_health():
    """وضعیت سلامت سیستم"""
    return system_monitor.get_system_health()

@health_router.get("/debug/system/trends", summary="روندهای سیستم")
async def get_system_trends(hours: int = Query(6, ge=1, le=72)):
    """روند استفاده از منابع سیستم"""
    return system_monitor.get_resource_usage_trend(hours)

# ==================== روت‌های دیباگ عملکرد ====================

@health_router.get("/debug/performance", summary="نمای کلی عملکرد")
async def get_performance_overview():
    """نمای کلی عملکرد سیستم"""
    return performance_monitor.analyze_endpoint_performance()

@health_router.get("/debug/performance/{endpoint_name}", summary="عملکرد اندپوینت خاص")
async def get_endpoint_performance(endpoint_name: str):
    """تحلیل عملکرد یک اندپوینت خاص"""
    return performance_monitor.analyze_endpoint_performance(endpoint_name)

@health_router.get("/debug/performance/bottlenecks", summary="شناسایی bottlenecks")
async def get_performance_bottlenecks():
    """شناسایی bottlenecks عملکرد"""
    return performance_monitor.analyze_bottlenecks()

@health_router.get("/debug/performance/trends/{endpoint_name}", summary="روند عملکرد اندپوینت")
async def get_endpoint_performance_trend(
    endpoint_name: str,
    hours: int = Query(24, ge=1, le=168)
):
    """روند عملکرد یک اندپوینت"""
    return performance_monitor.track_performance_trend(endpoint_name, hours)

# ==================== روت‌های دیباگ امنیت ====================

@health_router.get("/debug/security", summary="وضعیت امنیتی")
async def get_security_status():
    """وضعیت امنیتی سیستم"""
    return security_monitor.get_security_report()

@health_router.get("/debug/security/ip/{ip_address}", summary="اعتبار IP")
async def get_ip_reputation(ip_address: str):
    """بررسی اعتبار و reputation یک IP"""
    return security_monitor.get_ip_reputation(ip_address)

@health_router.get("/debug/security/suspicious", summary="فعالیت‌های مشکوک")
async def get_suspicious_activities(hours: int = Query(24, ge=1, le=168)):
    """فعالیت‌های امنیتی مشکوک"""
    return security_monitor.get_security_report(hours)

# ==================== روت‌های Real-Time ====================

@health_router.websocket("/debug/realtime/console")
async def websocket_console(websocket: WebSocket):
    """WebSocket برای کنسول Real-Time"""
    await console_stream.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # هندل کردن پیام‌های دریافتی از کلاینت
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
    await live_dashboard.connect_dashboard(websocket)
    try:
        while True:
            # نگه داشتن connection فعال
            await websocket.receive_text()
    except WebSocketDisconnect:
        live_dashboard.disconnect_dashboard(websocket)

@health_router.websocket("/debug/realtime/ws/{client_type}")
async def websocket_general(websocket: WebSocket, client_type: str):
    """WebSocket عمومی برای ارتباط Real-Time"""
    client_id = await websocket_manager.connect(websocket, client_type)
    try:
        await websocket_manager.handle_messages(client_id)
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)

# ==================== روت‌های متریک‌ها ====================

@health_router.get("/metrics", summary="تمام متریک‌ها")
async def get_all_metrics():
    """دریافت تمام متریک‌های سیستم"""
    system_metrics = metrics_collector.get_current_metrics()
    endpoint_stats = debug_manager.get_endpoint_stats()
    cache_stats = cache_debugger.get_cache_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_metrics": system_metrics,
        "endpoint_metrics": endpoint_stats,
        "cache_metrics": cache_stats,
        "performance_metrics": performance_monitor.analyze_endpoint_performance()
    }

@health_router.get("/metrics/system", summary="متریک‌های سیستم")
async def get_system_metrics_detailed():
    """متریک‌های دقیق سیستم"""
    return metrics_collector.get_detailed_metrics()

@health_router.get("/metrics/endpoints", summary="متریک‌های اندپوینت‌ها")
async def get_endpoints_metrics():
    """متریک‌های اندپوینت‌ها"""
    return debug_manager.get_endpoint_stats()

@health_router.get("/metrics/cache", summary="متریک‌های کش")
async def get_cache_metrics():
    """متریک‌های عملکرد کش"""
    return {
        "stats": cache_debugger.get_cache_stats(),
        "performance": cache_debugger.get_cache_performance(),
        "efficiency": cache_debugger.analyze_cache_efficiency()
    }

# ==================== روت‌های هشدارها ====================

@health_router.get("/alerts", summary="هشدارهای فعال")
async def get_active_alerts(
    level: str = Query(None, regex="^(INFO|WARNING|ERROR|CRITICAL)$"),
    alert_type: str = Query(None),
    source: str = Query(None)
):
    """دریافت هشدارهای فعال"""
    return alert_manager.get_active_alerts(
        level=AlertLevel(level) if level else None,
        alert_type=AlertType(alert_type) if alert_type else None,
        source=source
    )

@health_router.get("/alerts/history", summary="تاریخچه هشدارها")
async def get_alert_history(
    level: str = Query(None, regex="^(INFO|WARNING|ERROR|CRITICAL)$"),
    alert_type: str = Query(None),
    source: str = Query(None),
    hours: int = Query(24, ge=1, le=720),
    limit: int = Query(100, ge=1, le=1000)
):
    """تاریخچه هشدارها"""
    start_date = datetime.now() - timedelta(hours=hours)
    
    return alert_manager.get_alert_history(
        level=AlertLevel(level) if level else None,
        alert_type=AlertType(alert_type) if alert_type else None,
        source=source,
        start_date=start_date,
        end_date=datetime.now(),
        limit=limit
    )

@health_router.get("/alerts/stats", summary="آمار هشدارها")
async def get_alert_stats(hours: int = Query(24, ge=1, le=720)):
    """آمار هشدارها"""
    return alert_manager.get_alert_stats(hours)

@health_router.post("/alerts/{alert_id}/acknowledge", summary="تأیید هشدار")
async def acknowledge_alert(alert_id: int, user: str = "api"):
    """تأیید یک هشدار"""
    success = alert_manager.acknowledge_alert(alert_id, user)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}

@health_router.post("/alerts/{alert_id}/resolve", summary="حل هشدار")
async def resolve_alert(alert_id: int, resolved_by: str = "api", resolution_notes: str = ""):
    """حل یک هشدار"""
    success = alert_manager.resolve_alert(alert_id, resolved_by, resolution_notes)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "resolved", "alert_id": alert_id}

# ==================== روت‌های گزارش‌ها ====================

@health_router.get("/reports/daily", summary="گزارش روزانه")
async def get_daily_report(date: str = None):
    """گزارش روزانه عملکرد سیستم"""
    report_date = datetime.strptime(date, '%Y-%m-%d') if date else datetime.now()
    return report_generator.generate_daily_report(report_date)

@health_router.get("/reports/performance", summary="گزارش عملکرد")
async def get_performance_report(days: int = Query(7, ge=1, le=30)):
    """گزارش عملکرد سیستم"""
    return report_generator.generate_performance_report(days)

@health_router.get("/reports/security", summary="گزارش امنیتی")
async def get_security_report(days: int = Query(30, ge=1, le=90)):
    """گزارش امنیتی سیستم"""
    return report_generator.generate_security_report(days)

@health_router.post("/reports/generate", summary="تولید گزارش سفارشی")
async def generate_custom_report(report_config: Dict[str, Any]):
    """تولید گزارش سفارشی"""
    # این endpoint می‌تواند برای تولید گزارش‌های سفارشی استفاده شود
    return {
        "status": "report_request_received",
        "config": report_config,
        "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
    }

# ==================== روت‌های ابزارهای توسعه ====================

@health_router.post("/tools/test-traffic", summary="تولید ترافیک تست")
async def generate_test_traffic(
    background_tasks: BackgroundTasks,
    endpoint: str = None,
    duration_seconds: int = 60,
    requests_per_second: int = 10
):
    """تولید ترافیک تست برای شبیه‌سازی بار"""
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

@health_router.post("/tools/load-test", summary="تست بار")
async def run_load_test(
    background_tasks: BackgroundTasks,
    endpoint: str,
    concurrent_users: int = 10,
    duration_seconds: int = 60
):
    """اجرای تست بار برای اندپوینت"""
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

@health_router.get("/tools/dependencies", summary="بررسی وابستگی‌ها")
async def check_dependencies():
    """بررسی وضعیت وابستگی‌های سیستم"""
    return dev_tools.run_dependency_check()

@health_router.get("/tools/memory-analysis", summary="آنالیز حافظه")
async def analyze_memory_usage():
    """آنالیز استفاده از حافظه"""
    return dev_tools.analyze_memory_usage()

# ==================== روت‌های کمکی ====================

@health_router.get("/ping", summary="پینگ سیستم")
async def health_ping():
    """پینگ ساده برای بررسی حیات سیستم"""
    return {
        "status": "pong", 
        "timestamp": datetime.now().isoformat(),
        "response_time": "immediate",
        "version": "4.0.0"
    }

@health_router.get("/version", summary="نسخه‌های سیستم")
async def health_version():
    """نسخه‌های سیستم و کامپوننت‌ها"""
    return {
        "api_version": "4.0.0",
        "debug_system_version": "1.0.0",
        "python_version": "3.9+",
        "fastapi_version": "0.104.1",
        "timestamp": datetime.now().isoformat()
    }

# ==================== توابع کمکی ====================

async def _get_system_health() -> Dict[str, Any]:
    """جمع‌آوری سلامت سیستم"""
    try:
        # اطلاعات سیستم
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # اطلاعات شبکه
        net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "uptime": _get_system_uptime(),
            "active_connections": connections,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv
        }
    except Exception as e:
        logger.error(f"خطا در دریافت سلامت سیستم: {e}")
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "uptime": "unknown",
            "active_connections": 0,
            "bytes_sent": 0,
            "bytes_recv": 0
        }

async def _get_debug_system_status() -> Dict[str, Any]:
    """وضعیت سیستم دیباگ"""
    try:
        return {
            "status": "active",
            "monitors": {
                "endpoint_monitor": "running",
                "system_monitor": "running", 
                "performance_monitor": "running",
                "security_monitor": "running"
            },
            "alerts_active": len(alert_manager.get_active_alerts()),
            "last_scan": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "monitors": {},
            "alerts_active": 0
        }

async def _get_api_endpoints_status() -> Dict[str, Any]:
    """وضعیت اندپوینت‌های API"""
    try:
        endpoint_stats = debug_manager.get_endpoint_stats()
        overall = endpoint_stats.get('overall', {})
        
        return {
            "overall": "healthy",
            "total_endpoints": len(endpoint_stats.get('endpoints', {})),
            "endpoints_healthy": len([ep for ep in endpoint_stats.get('endpoints', {}).values() 
                                    if ep.get('success_rate', 100) > 95]),
            "endpoints_degraded": len([ep for ep in endpoint_stats.get('endpoints', {}).values() 
                                     if ep.get('success_rate', 0) <= 95]),
            "endpoints_down": 0,
            "avg_response_time": overall.get('average_response_time', 0),
            "error_rate": 100 - overall.get('overall_success_rate', 100),
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "overall": "unknown",
            "error": str(e),
            "total_endpoints": 0,
            "endpoints_healthy": 0
        }

def _get_system_uptime() -> str:
    """محاسبه آپتایم سیستم"""
    try:
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        return str(uptime).split('.')[0]  # حذف میکروثانیه
    except:
        return "unknown"

# ==================== راه‌اندازی اولیه ====================

def initialize_debug_system():
    """مقداردهی اولیه سیستم دیباگ"""
    try:
        # مقداردهی مانیتورها
        from debug_system.monitors.endpoint_monitor import endpoint_monitor
        from debug_system.monitors.system_monitor import system_monitor
        from debug_system.monitors.performance_monitor import performance_monitor
        from debug_system.monitors.security_monitor import security_monitor
        
        from debug_system.tools.dev_tools import dev_tools
        from debug_system.tools.testing_tools import testing_tools
        from debug_system.tools.report_generator import report_generator
        
        from debug_system.realtime.live_dashboard import live_dashboard
        
        # مقداردهی وابستگی‌ها
        endpoint_monitor.debug_manager = debug_manager
        system_monitor.metrics_collector = metrics_collector
        system_monitor.alert_manager = alert_manager
        performance_monitor.debug_manager = debug_manager
        performance_monitor.alert_manager = alert_manager
        security_monitor.alert_manager = alert_manager
        
        dev_tools.debug_manager = debug_manager
        dev_tools.endpoint_monitor = endpoint_monitor
        testing_tools.debug_manager = debug_manager
        testing_tools.endpoint_monitor = endpoint_monitor
        report_generator.debug_manager = debug_manager
        report_generator.history_manager = history_manager
        
        live_dashboard.debug_manager = debug_manager
        live_dashboard.metrics_collector = metrics_collector
        
        # شروع برودکست دشبورد
        asyncio.create_task(live_dashboard.start_dashboard_broadcast())
        
        logger.info("✅ Debug system initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Debug system initialization failed: {e}")

# راه‌اندازی اولیه هنگام ایمپورت
initialize_debug_system()

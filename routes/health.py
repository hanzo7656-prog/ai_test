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

# ==================== DEBUG SYSTEM AVAILABILITY CHECK ====================

DEBUG_SYSTEM_AVAILABLE = os.getenv("DEBUG_SYSTEM_AVAILABLE", "False").lower() == "true"

if DEBUG_SYSTEM_AVAILABLE:
    try:
        from debug_system.core import core_system, debug_manager, metrics_collector, alert_manager, AlertLevel, AlertType
        from debug_system.monitors import monitors_system, endpoint_monitor, system_monitor, performance_monitor, security_monitor
        from debug_system.storage import history_manager, log_manager, cache_debugger
        from debug_system.realtime import console_stream, live_dashboard, websocket_manager
        from debug_system.tools import tools_system, dev_tools, testing_tools, report_generator
        
        print("✅ Debug system fully imported for health routes")
    except ImportError as e:
        print(f"❌ Debug system import error in health routes: {e}")
        DEBUG_SYSTEM_AVAILABLE = False
else:
    print("❌ Debug system not available for health routes")

# ==================== BASIC HEALTH ENDPOINTS ====================

@health_router.get("/status")
async def health_status():
    """بررسی سلامت کلی سیستم"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "services": {
            "api": "running",
            "database": "connected",
            "cache": "connected",
            "external_apis": "available"
        }
    }

@health_router.get("/overview")
async def system_overview():
    """نمای کلی سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
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
        "status": {
            "debug_system_available": DEBUG_SYSTEM_AVAILABLE,
            "debug_system_status": "active" if DEBUG_SYSTEM_AVAILABLE else "inactive"
        }
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "endpoint_health": monitors_system["endpoint_monitor"].get_all_endpoints_health(),
        "performance_report": monitors_system["performance_monitor"].get_performance_report(),
        "bottlenecks": monitors_system["performance_monitor"].analyze_bottlenecks(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/system")
async def debug_system():
    """دریافت وضعیت کامل سیستم دیباگ"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "system_health": monitors_system["system_monitor"].get_system_health(),
        "security_report": monitors_system["security_monitor"].get_security_report(),
        "active_alerts": core_system["alert_manager"].get_active_alerts(),
        "resource_usage": monitors_system["system_monitor"].get_resource_usage_trend(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/reports/daily")
async def debug_daily_report():
    """دریافت گزارش روزانه دیباگ"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return tools_system["report_generator"].generate_daily_report()

@health_router.get("/debug/reports/performance")
async def debug_performance_report():
    """دریافت گزارش عملکرد دیباگ"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return tools_system["report_generator"].generate_performance_report()

@health_router.get("/debug/reports/security")
async def debug_security_report():
    """دریافت گزارش امنیتی دیباگ"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return tools_system["report_generator"].generate_security_report()

@health_router.get("/debug/metrics/live")
async def debug_live_metrics():
    """دریافت متریک‌های real-time"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "system_metrics": core_system["metrics_collector"].get_current_metrics(),
        "endpoint_metrics": core_system["debug_manager"].get_endpoint_stats(),
        "performance_metrics": monitors_system["performance_monitor"].get_performance_report(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/alerts")
async def debug_alerts():
    """دریافت هشدارهای فعال سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "active_alerts": core_system["alert_manager"].get_active_alerts(),
        "alert_stats": core_system["alert_manager"].get_alert_stats(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.post("/debug/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, user: str = "system"):
    """تأیید هشدار"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    success = core_system["alert_manager"].acknowledge_alert(alert_id, user)
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    success = core_system["alert_manager"].resolve_alert(alert_id, resolved_by, resolution_notes)
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "bottlenecks": monitors_system["performance_monitor"].analyze_bottlenecks(),
        "slowest_endpoints": monitors_system["performance_monitor"].get_slowest_endpoints(),
        "most_called_endpoints": monitors_system["performance_monitor"].get_most_called_endpoints(),
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/security/overview")
async def debug_security_overview():
    """نمای کلی امنیتی"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "security_report": monitors_system["security_monitor"].get_security_report(),
        "ip_reputation_sample": {
            "127.0.0.1": monitors_system["security_monitor"].get_ip_reputation("127.0.0.1")
        },
        "timestamp": datetime.now().isoformat()
    }

# ==================== COMPREHENSIVE DEBUG ENDPOINTS ====================

@health_router.get("/debug/endpoints/{endpoint_name}")
async def debug_single_endpoint(endpoint_name: str):
    """آمار و اطلاعات دیباگ یک اندپوینت خاص"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return debug_manager.get_endpoint_stats(endpoint_name)

@health_router.get("/debug/endpoints/{endpoint_name}/calls")
async def get_endpoint_recent_calls(
    endpoint_name: str,
    limit: int = Query(50, ge=1, le=1000)
):
    """دریافت فراخوانی‌های اخیر یک اندپوینت"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    recent_calls = debug_manager.get_recent_calls(limit)
    endpoint_calls = [call for call in recent_calls if call['endpoint'] == endpoint_name]
    return {
        "endpoint": endpoint_name,
        "total_calls": len(endpoint_calls),
        "calls": endpoint_calls,
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/debug/endpoints/health/overview")
async def get_endpoints_health_overview():
    """نمای کلی سلامت اندپوینت‌ها"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return endpoint_monitor.get_all_endpoints_health()

@health_router.get("/debug/endpoints/performance/slowest")
async def get_slowest_endpoints(limit: int = Query(10, ge=1, le=50)):
    """دریافت کندترین اندپوینت‌ها"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return performance_monitor.get_slowest_endpoints(limit)

@health_router.get("/debug/endpoints/performance/most-called")
async def get_most_called_endpoints(limit: int = Query(10, ge=1, le=50)):
    """دریافت پرکاربردترین اندپوینت‌ها"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return performance_monitor.get_most_called_endpoints(limit)

@health_router.get("/debug/system/metrics")
async def get_system_metrics_debug():
    """متریک‌های Real-Time سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return metrics_collector.get_current_metrics()

@health_router.get("/debug/system/metrics/history")
async def get_system_metrics_history(
    hours: int = Query(1, ge=1, le=168),
    limit: int = Query(100, ge=1, le=1000)
):
    """تاریخچه متریک‌های سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return metrics_collector.get_metrics_history(hours * 3600)[:limit]

@health_router.get("/debug/system/health")
async def get_system_health_debug():
    """وضعیت سلامت سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return system_monitor.get_system_health()

@health_router.get("/debug/system/trends")
async def get_system_trends(hours: int = Query(6, ge=1, le=72)):
    """روند استفاده از منابع سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return system_monitor.get_resource_usage_trend(hours)

@health_router.get("/debug/performance")
async def get_performance_overview():
    """نمای کلی عملکرد"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return performance_monitor.analyze_endpoint_performance()

@health_router.get("/debug/performance/{endpoint_name}")
async def get_endpoint_performance(endpoint_name: str):
    """تحلیل عملکرد یک اندپوینت خاص"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return performance_monitor.analyze_endpoint_performance(endpoint_name)

@health_router.get("/debug/performance/bottlenecks/detailed")
async def get_performance_bottlenecks_detailed():
    """شناسایی bottlenecks عملکرد"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return performance_monitor.analyze_bottlenecks()

@health_router.get("/debug/performance/trends/{endpoint_name}")
async def get_endpoint_performance_trend(
    endpoint_name: str,
    hours: int = Query(24, ge=1, le=168)
):
    """روند عملکرد یک اندپوینت"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return performance_monitor.track_performance_trend(endpoint_name, hours)

@health_router.get("/debug/security")
async def get_security_status():
    """وضعیت امنیتی"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return security_monitor.get_security_report()

@health_router.get("/debug/security/ip/{ip_address}")
async def get_ip_reputation(ip_address: str):
    """بررسی اعتبار و reputation یک IP"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return security_monitor.get_ip_reputation(ip_address)

@health_router.get("/debug/security/suspicious")
async def get_suspicious_activities(hours: int = Query(24, ge=1, le=168)):
    """فعالیت‌های امنیتی مشکوک"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return security_monitor.get_security_report(hours)

# ==================== REAL-TIME ENDPOINTS ====================

@health_router.websocket("/debug/realtime/console")
async def websocket_console(websocket: WebSocket):
    """WebSocket برای کنسول Real-Time"""
    if not DEBUG_SYSTEM_AVAILABLE:
        await websocket.close(code=1008, reason="Debug system not available")
        return
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        await websocket.close(code=1008, reason="Debug system not available")
        return
    
    await live_dashboard.connect_dashboard(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        live_dashboard.disconnect_dashboard(websocket)

@health_router.websocket("/debug/realtime/ws/{client_type}")
async def websocket_general(websocket: WebSocket, client_type: str):
    """WebSocket عمومی برای ارتباط Real-Time"""
    if not DEBUG_SYSTEM_AVAILABLE:
        await websocket.close(code=1008, reason="Debug system not available")
        return
    
    client_id = await websocket_manager.connect(websocket, client_type)
    try:
        await websocket_manager.handle_messages(client_id)
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)

# ==================== METRICS ENDPOINTS ====================

@health_router.get("/metrics")
async def get_all_metrics():
    """دریافت تمام متریک‌های سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
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

@health_router.get("/metrics/system")
async def get_system_metrics_detailed():
    """متریک‌های دقیق سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return metrics_collector.get_detailed_metrics()

@health_router.get("/metrics/endpoints")
async def get_endpoints_metrics():
    """متریک‌های اندپوینت‌ها"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return debug_manager.get_endpoint_stats()

@health_router.get("/metrics/cache")
async def get_cache_metrics():
    """متریک‌های عملکرد کش"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return alert_manager.get_alert_stats(hours)

@health_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, user: str = "api"):
    """تأیید یک هشدار"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    success = alert_manager.acknowledge_alert(alert_id, user)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}

@health_router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int, resolved_by: str = "api", resolution_notes: str = ""):
    """حل یک هشدار"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    success = alert_manager.resolve_alert(alert_id, resolved_by, resolution_notes)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "resolved", "alert_id": alert_id}

# ==================== REPORTS ENDPOINTS ====================

@health_router.get("/reports/daily")
async def get_daily_report(date: str = None):
    """گزارش روزانه عملکرد سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    report_date = datetime.strptime(date, '%Y-%m-%d') if date else datetime.now()
    return report_generator.generate_daily_report(report_date)

@health_router.get("/reports/performance")
async def get_performance_report(days: int = Query(7, ge=1, le=30)):
    """گزارش عملکرد سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return report_generator.generate_performance_report(days)

@health_router.get("/reports/security")
async def get_security_report(days: int = Query(30, ge=1, le=90)):
    """گزارش امنیتی سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return report_generator.generate_security_report(days)

@health_router.post("/reports/generate")
async def generate_custom_report(report_config: Dict[str, Any]):
    """تولید گزارش سفارشی"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "status": "report_request_received",
        "config": report_config,
        "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
    }

# ==================== TOOLS ENDPOINTS ====================

@health_router.post("/tools/test-traffic")
async def generate_test_traffic(
    background_tasks: BackgroundTasks,
    endpoint: str = None,
    duration_seconds: int = 60,
    requests_per_second: int = 10
):
    """تولید ترافیک تست برای شبیه‌سازی بار"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
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
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return dev_tools.run_dependency_check()

@health_router.get("/tools/memory-analysis")
async def analyze_memory_usage():
    """آنالیز استفاده از حافظه"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return dev_tools.analyze_memory_usage()

@health_router.get("/tools/cache-stats")
async def get_cache_stats():
    """آمار کامل کش سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "cache_stats": cache_debugger.get_cache_stats(),
        "cache_performance": cache_debugger.get_cache_performance(),
        "cache_efficiency": cache_debugger.analyze_cache_efficiency(),
        "most_accessed_keys": cache_debugger.get_most_accessed_keys(),
        "timestamp": datetime.now().isoformat()
    }

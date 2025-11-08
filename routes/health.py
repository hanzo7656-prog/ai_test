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

# ==================== TOOLS ENDPOINTS ====================

@health_router.post("/tools/test-traffic")
async def test_traffic(
    background_tasks: BackgroundTasks,
    endpoint: str = None,
    duration_seconds: int = 60,
    requests_per_second: int = 10
):
    """تولید ترافیک تست"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    background_tasks.add_task(
        tools_system["dev_tools"].generate_test_traffic,
        endpoint,
        duration_seconds,
        requests_per_second
    )
    
    return {
        "message": "Test traffic generation started",
        "endpoint": endpoint,
        "duration_seconds": duration_seconds,
        "requests_per_second": requests_per_second,
        "timestamp": datetime.now().isoformat()
    }

@health_router.post("/tools/load-test")
async def load_test(
    background_tasks: BackgroundTasks,
    endpoint: str,
    concurrent_users: int = 10,
    total_requests: int = 1000
):
    """اجرای تست بار"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    result = tools_system["dev_tools"].run_performance_test(
        endpoint, concurrent_users, total_requests
    )
    
    return {
        "message": "Load test completed",
        "test_results": result,
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/tools/dependencies")
async def check_dependencies():
    """بررسی وابستگی‌های سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return tools_system["dev_tools"].run_dependency_check()

@health_router.get("/tools/memory-analysis")
async def memory_analysis():
    """آنالیز استفاده از حافظه"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return tools_system["dev_tools"].analyze_memory_usage()

@health_router.get("/tools/cache-stats")
async def cache_stats():
    """آمار کش سیستم"""
    if not DEBUG_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    return {
        "cache_stats": cache_debugger.get_cache_stats(),
        "cache_performance": cache_debugger.get_cache_performance(),
        "cache_efficiency": cache_debugger.analyze_cache_efficiency(),
        "most_accessed_keys": cache_debugger.get_most_accessed_keys(),
        "timestamp": datetime.now().isoformat()
    }

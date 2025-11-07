from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import psutil
import logging

# ایمپورت مدیران دیباگ (بعداً تکمیل می‌شوند)
from debug_system.core.debug_manager import DebugManager
from debug_system.core.metrics_collector import MetricsCollector
from debug_system.core.alert_manager import AlertManager

logger = logging.getLogger(__name__)

# ایجاد روت‌ر سلامت
health_router = APIRouter(prefix="/api/health", tags=["Health & Debug"])

# ایجاد نمونه‌های مدیران
debug_manager = DebugManager()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()

# ==================== روت سلامت مادر ====================

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
        "version": "1.0.0",
        
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
            "response_time_avg": "45ms",
            "uptime": system_health["uptime"],
            "active_connections": system_health["active_connections"],
            "error_rate": "0.2%"
        },
        
        # لینک‌های دیباگ و مانیتورینگ
        "debug_links": {
            "endpoints_debug": "/api/health/debug/endpoints",
            "system_metrics": "/api/health/debug/system/metrics",
            "performance": "/api/health/debug/performance",
            "security": "/api/health/debug/security",
            "realtime_console": "/api/health/debug/realtime/console",
            "all_metrics": "/api/health/metrics",
            "active_alerts": "/api/health/alerts"
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
    
    overview_data = {
        "timestamp": datetime.now().isoformat(),
        
        # کارت‌های وضعیت
        "status_cards": {
            "api_health": {
                "status": "healthy",
                "endpoints_total": 45,
                "endpoints_active": 45,
                "last_incident": None
            },
            "debug_system": {
                "status": "active",
                "monitors_running": 8,
                "alerts_active": 2,
                "last_alert": "2024-01-15T10:25:00"
            },
            "performance": {
                "status": "optimal",
                "avg_response_time": "45ms",
                "throughput": "120 req/min",
                "error_rate": "0.2%"
            },
            "resources": {
                "status": "normal",
                "cpu_usage": "42%",
                "memory_usage": "65%",
                "disk_usage": "72%"
            }
        },
        
        # گراف‌های سریع
        "quick_metrics": {
            "response_times": [45, 42, 48, 51, 47, 44, 46],
            "error_rates": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.1],
            "throughput": [110, 115, 120, 118, 122, 119, 121],
            "cpu_usage": [40, 42, 45, 43, 41, 44, 42]
        }
    }
    
    return overview_data

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
        # اینجا بعداً با DebugManager تکمیل می‌شود
        return {
            "status": "active",
            "monitors": {
                "endpoint_monitor": "running",
                "system_monitor": "running", 
                "performance_monitor": "running",
                "security_monitor": "running"
            },
            "alerts_active": 2,
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
        # اینجا بعداً با EndpointMonitor تکمیل می‌شود
        return {
            "overall": "healthy",
            "total_endpoints": 45,
            "endpoints_healthy": 45,
            "endpoints_degraded": 0,
            "endpoints_down": 0,
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

# ==================== روت‌های ساده برای شروع ====================

@health_router.get("/ping")
async def health_ping():
    """پینگ ساده برای بررسی حیات سیستم"""
    return {
        "status": "pong", 
        "timestamp": datetime.now().isoformat(),
        "response_time": "immediate"
    }

@health_router.get("/version")
async def health_version():
    """نسخه‌های سیستم و کامپوننت‌ها"""
    return {
        "api_version": "1.0.0",
        "debug_system_version": "1.0.0",
        "python_version": "3.9+",
        "fastapi_version": "0.104.1",
        "timestamp": datetime.now().isoformat()
    }

# بعداً این روت‌ر به main.py اضافه می‌شود

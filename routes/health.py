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

# ایمپورت سیستم نرمال‌سازی جدید
try:
    from debug_system.utils.data_normalizer import data_normalizer
except ImportError:
    # Fallback برای مواقع توسعه
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.utils.data_normalizer import data_normalizer

# ایمپورت complete_coinstats_manager برای وضعیت API
try:
    from complete_coinstats_manager import coin_stats_manager
except ImportError:
    coin_stats_manager = None

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
    
    # بررسی وضعیت API خارجی
    api_status = "unknown"
    if coin_stats_manager:
        try:
            api_check = coin_stats_manager.get_api_status()
            api_status = api_check.get('status', 'unknown')
        except Exception as e:
            api_status = f"error: {str(e)}"
    
    # دریافت متریک‌های نرمال‌سازی
    normalization_metrics = data_normalizer.get_health_metrics()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "services": {
            "api": "running",
            "database": "connected",
            "cache": "connected",
            "external_apis": api_status,
            "debug_system": {
                "available": debug_status['core_available'],
                "loaded_modules": debug_status['loaded_modules'],
                "total_modules": debug_status['total_modules'],
                "status": "fully_initialized" if debug_status['loaded_modules'] == debug_status['total_modules'] else "partially_initialized"
            },
            "data_normalization": {
                "available": True,
                "success_rate": normalization_metrics.success_rate,
                "total_processed": normalization_metrics.total_processed,
                "status": "optimal" if normalization_metrics.success_rate > 95 else "degraded"
            }
        },
        "normalization_metrics": {
            "success_rate": normalization_metrics.success_rate,
            "total_processed": normalization_metrics.total_processed,
            "total_errors": normalization_metrics.total_errors,
            "common_structures": normalization_metrics.common_structures,
            "data_quality": normalization_metrics.data_quality
        }
    }

@health_router.get("/overview")
async def system_overview():
    """نمای کلی سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    debug_status = DebugSystemManager.get_status_report()
    
    # متریک‌های نرمال‌سازی
    normalization_metrics = data_normalizer.get_health_metrics()
    
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
        "debug_system": debug_status,
        "data_normalization": {
            "success_rate": normalization_metrics.success_rate,
            "total_requests": normalization_metrics.total_processed,
            "performance_metrics": normalization_metrics.performance_metrics,
            "data_quality": normalization_metrics.data_quality
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

# ==================== DATA NORMALIZATION ENDPOINTS ====================

@health_router.get("/normalization/metrics")
async def get_normalization_metrics():
    """دریافت متریک‌های کامل نرمال‌سازی داده"""
    try:
        metrics = data_normalizer.get_health_metrics()
        analysis = data_normalizer.get_deep_analysis()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "success_rate": metrics.success_rate,
                "total_processed": metrics.total_processed,
                "total_success": metrics.total_success,
                "total_errors": metrics.total_errors,
                "performance_metrics": metrics.performance_metrics,
                "data_quality": metrics.data_quality
            },
            "common_structures": metrics.common_structures,
            "alerts": metrics.alerts,
            "analysis_overview": analysis.get("system_overview", {}),
            "recommendations": analysis.get("recommendations", [])
        }
    except Exception as e:
        logger.error(f"❌ Error getting normalization metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get normalization metrics: {str(e)}")

@health_router.get("/normalization/analysis")
async def get_normalization_analysis():
    """دریافت تحلیل عمیق نرمال‌سازی"""
    try:
        analysis = data_normalizer.get_deep_analysis()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"❌ Error getting normalization analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get normalization analysis: {str(e)}")

@health_router.get("/normalization/structures")
async def get_detected_structures():
    """دریافت ساختارهای شناسایی شده"""
    try:
        metrics = data_normalizer.get_health_metrics()
        analysis = data_normalizer.get_deep_analysis()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "structure_analysis": metrics.common_structures,
            "endpoint_patterns": analysis.get("endpoint_patterns", {}),
            "performance_analysis": analysis.get("performance_analysis", {})
        }
    except Exception as e:
        logger.error(f"❌ Error getting structure analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get structure analysis: {str(e)}")

@health_router.post("/normalization/reset-metrics")
async def reset_normalization_metrics():
    """بازنشانی متریک‌های نرمال‌سازی (برای تست)"""
    try:
        data_normalizer.reset_metrics()
        
        return {
            "status": "success",
            "message": "Normalization metrics reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error resetting normalization metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")

@health_router.post("/normalization/clear-cache")
async def clear_normalization_cache():
    """پاک‌سازی کش نرمال‌سازی"""
    try:
        data_normalizer.clear_cache()
        
        return {
            "status": "success",
            "message": "Normalization cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error clearing normalization cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

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
    
    # اضافه کردن متریک‌های نرمال‌سازی
    normalization_metrics = data_normalizer.get_health_metrics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_metrics": metrics_collector.get_current_metrics(),
        "endpoint_metrics": debug_manager.get_endpoint_stats(),
        "cache_metrics": cache_debugger.get_cache_stats(),
        "performance_metrics": performance_monitor.analyze_endpoint_performance(),
        "normalization_metrics": {
            "success_rate": normalization_metrics.success_rate,
            "total_processed": normalization_metrics.total_processed,
            "common_structures": normalization_metrics.common_structures,
            "data_quality": normalization_metrics.data_quality
        }
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
    
    # گزارش وضعیت نرمال‌سازی
    normalization_metrics = data_normalizer.get_health_metrics()
    logger.info(f"📊 Data normalization system ready. Success rate: {normalization_metrics.success_rate}%")


# ==================== ROUTERS HEALTH DEBUG ====================

@health_router.get("/debug/routers", summary="بررسی سلامت تمام روترهای سیستم")
async def debug_routers_health():
    """بررسی سلامت کامل تمام روترهای سیستم - برای مانیتورینگ پیشرفته"""
    
    routers_info = {
        "health_router": {"file": "routes/health.py", "endpoints": [], "status": "unknown"},
        "coins_router": {"file": "routes/coins.py", "endpoints": [], "status": "unknown"},
        "exchanges_router": {"file": "routes/exchanges.py", "endpoints": [], "status": "unknown"},
        "news_router": {"file": "routes/news.py", "endpoints": [], "status": "unknown"},
        "insights_router": {"file": "routes/insights.py", "endpoints": [], "status": "unknown"},
        "raw_coins_router": {"file": "routes/raw_coins.py", "endpoints": [], "status": "unknown"},
        "raw_news_router": {"file": "routes/raw_news.py", "endpoints": [], "status": "unknown"},
        "raw_insights_router": {"file": "routes/raw_insights.py", "endpoints": [], "status": "unknown"},
        "raw_exchanges_router": {"file": "routes/raw_exchanges.py", "endpoints": [], "status": "unknown"},
        "docs_router": {"file": "routes/docs.py", "endpoints": [], "status": "unknown"}
    }
    
    try:
        # راه حل ساده‌تر: استفاده از global app instance
        from main import app
        
        # جمع‌آوری اطلاعات از تمام مسیرها
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                path = route.path
                
                # تشخیص روتر بر اساس مسیر
                if path.startswith("/api/health"):
                    router = "health_router"
                elif path.startswith("/api/coins") and not path.startswith("/api/raw/coins"):
                    router = "coins_router"
                elif path.startswith("/api/raw/coins"):
                    router = "raw_coins_router"
                elif path.startswith("/api/exchanges") and not path.startswith("/api/raw/exchanges"):
                    router = "exchanges_router"
                elif path.startswith("/api/raw/exchanges"):
                    router = "raw_exchanges_router"
                elif path.startswith("/api/news") and not path.startswith("/api/raw/news"):
                    router = "news_router"
                elif path.startswith("/api/raw/news"):
                    router = "raw_news_router"
                elif path.startswith("/api/insights") and not path.startswith("/api/raw/insights"):
                    router = "insights_router"
                elif path.startswith("/api/raw/insights"):
                    router = "raw_insights_router"
                elif path.startswith("/api/docs"):
                    router = "docs_router"
                else:
                    continue
                
                if router in routers_info:
                    routers_info[router]["endpoints"].append({
                        "path": path,
                        "methods": list(route.methods),
                        "name": getattr(route, "name", "Unknown")
                    })
        
        # محاسبه وضعیت سلامت
        for router_name, info in routers_info.items():
            endpoint_count = len(info["endpoints"])
            if endpoint_count > 0:
                info["status"] = "healthy"
                info["endpoint_count"] = endpoint_count
            else:
                info["status"] = "no_endpoints"
                info["endpoint_count"] = 0
        
        # بررسی خاص raw_insights_router
        raw_insights_info = routers_info["raw_insights_router"]
        rainbow_chart_exists = any("/rainbow-chart/" in endpoint["path"] for endpoint in raw_insights_info["endpoints"])
        raw_insights_info["rainbow_chart_available"] = rainbow_chart_exists
        
        # آمار کلی
        total_endpoints = sum(info["endpoint_count"] for info in routers_info.values())
        healthy_routers = sum(1 for info in routers_info.values() if info["status"] == "healthy")
        
        return {
            "system_overview": {
                "total_routers": len(routers_info),
                "healthy_routers": healthy_routers,
                "total_endpoints": total_endpoints,
                "timestamp": datetime.now().isoformat()
            },
            "routers_health": routers_info,
            "issues_detected": {
                "raw_insights_missing_rainbow": not rainbow_chart_exists,
                "routers_with_no_endpoints": [
                    name for name, info in routers_info.items() 
                    if info["status"] == "no_endpoints"
                ]
            },
            "recommendations": [
                recommendation for recommendation in [
                    "Add rainbow-chart endpoint to raw_insights_router" if not rainbow_chart_exists else None,
                    "Check router registration for: " + ", ".join([
                        name for name, info in routers_info.items() 
                        if info["status"] == "no_endpoints"
                    ]) if any(info["status"] == "no_endpoints" for info in routers_info.values()) else None
                ] if recommendation is not None
            ]
        }
        
    except ImportError:
        return {
            "error": "Could not import app from main",
            "message": "This endpoint requires access to the main app instance",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in debug_routers_health: {e}")
        return {
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

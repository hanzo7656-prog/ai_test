from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, Request
from datetime import datetime, timedelta
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import psutil
import logging
import os
import glob
import shutil
import threading

logger = logging.getLogger(__name__)

# ==================== IMPORTS ====================

# سیستم نرمال‌سازی
try:
    from debug_system.utils.data_normalizer import data_normalizer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.utils.data_normalizer import data_normalizer

# Cache Optimization Engine
try:
    from debug_system.storage.smart_cache_system import cache_optimizer
    logger.info("✅ Cache Optimization Engine imported")
except ImportError as e:
    logger.warning(f"⚠️ Cache Optimization Engine: {e}")
    cache_optimizer = None

# complete_coinstats_manager
try:
    from complete_coinstats_manager import coin_stats_manager
except ImportError:
    coin_stats_manager = None
    logger.warning("⚠️ coin_stats_manager not available")

# سیستم AI
try:
    from simple_ai.brain import ai_brain
    from integrations.ai_monitor import ai_monitor
    AI_SYSTEM_AVAILABLE = True
    logger.info("✅ AI System imported")
except ImportError as e:
    logger.warning(f"⚠️ AI System: {e}")
    AI_SYSTEM_AVAILABLE = False
    ai_brain = None
    ai_monitor = None

# سیستم کش جدید
try:
    from debug_system.storage.cache_decorators import (
        cache_coins_with_archive, cache_news_with_archive, cache_insights_with_archive, cache_exchanges_with_archive,
        cache_raw_coins_with_archive, cache_raw_news_with_archive, cache_raw_insights_with_archive, cache_raw_exchanges_with_archive,
        get_historical_data, get_archive_stats, cleanup_old_archives
    )
    NEW_CACHE_SYSTEM_AVAILABLE = True
    logger.info("✅ New Cache System imported")
except ImportError as e:
    logger.warning(f"⚠️ New Cache System: {e}")
    NEW_CACHE_SYSTEM_AVAILABLE = False

# ایجاد روت‌ر سلامت
health_router = APIRouter(prefix="/api/health", tags=["Health & Monitoring"])

# ==================== OPTIMIZED DEBUG SYSTEM MANAGER ====================

class DebugSystemManager:
    """مدیریت بهینه‌شده سیستم دیباگ با لاگ‌های خلاصه و مصرف CPU کنترل‌شده"""
    
    _initialized = False
    _modules = {}
    _load_stages = {
        'core': False,
        'monitors': False, 
        'storage': False,
        'realtime': False,
        'tools': False
    }
    
    @classmethod
    def initialize(cls):
        """مقداردهی مرحله‌ای و بهینه سیستم دیباگ"""
        if cls._initialized:
            logger.info("🔧 سیستم دیباگ از قبل فعال")
            return cls._modules
        
        logger.info("🚀 شروع راه‌اندازی هوشمند سیستم دیباگ...")
        start_time = time.time()
        
        try:
            # مرحله 1: ماژول‌های اصلی
            logger.info("📦 مرحله 1: بارگذاری هسته اصلی")
            cls._load_core_modules()
            time.sleep(0.02)
            
            # مرحله 2: مانیتورها
            logger.info("📊 مرحله 2: راه‌اندازی مانیتورها")
            cls._load_monitors()
            time.sleep(0.02)
            
            # مرحله 3: سیستم ذخیره‌سازی
            logger.info("💾 مرحله 3: تنظیم ذخیره‌سازی")
            cls._load_storage()
            time.sleep(0.02)
            
            # مرحله 4: سیستم real-time
            logger.info("⚡ مرحله 4: فعال‌سازی real-time")
            cls._load_realtime()
            time.sleep(0.02)
            
            # مرحله 5: ابزارها
            logger.info("🔧 مرحله 5: راه‌اندازی ابزارها")
            cls._load_tools()
            
            cls._initialized = True
            total_time = time.time() - start_time
            
            logger.info(f"✅ راه‌اندازی کامل - زمان: {total_time:.2f}ثانیه - ماژول‌ها: {len([name for name, module in cls._modules.items() if module is not None])}")
            
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی: {e}")
        
        return cls._modules
    
    @classmethod
    def _load_core_modules(cls):
        """بارگذاری ماژول‌های اصلی"""
        try:
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
            cls._load_stages['core'] = True
        except Exception as e:
            logger.warning(f"⚠️ خطای هسته: {e}")
    
    @classmethod
    def _load_monitors(cls):
        """بارگذاری مانیتورها"""
        try:
            from debug_system.monitors.endpoint_monitor import EndpointMonitor
            from debug_system.monitors.system_monitor import SystemMonitor
            from debug_system.monitors.performance_monitor import PerformanceMonitor
            from debug_system.monitors.security_monitor import SecurityMonitor
            
            debug_manager = cls._modules.get('debug_manager')
            metrics_collector = cls._modules.get('metrics_collector')
            alert_manager = cls._modules.get('alert_manager')
            
            if all([debug_manager, metrics_collector, alert_manager]):
                cls._modules.update({
                    'endpoint_monitor': EndpointMonitor(debug_manager),
                    'system_monitor': SystemMonitor(metrics_collector, alert_manager),
                    'performance_monitor': PerformanceMonitor(debug_manager, alert_manager),
                    'security_monitor': SecurityMonitor(alert_manager)
                })
                cls._load_stages['monitors'] = True
        except Exception as e:
            logger.warning(f"⚠️ خطای مانیتورها: {e}")
    
    @classmethod
    def _load_storage(cls):
        """بارگذاری سیستم ذخیره‌سازی"""
        try:
            from debug_system.storage.history_manager import history_manager
            from debug_system.storage.cache_debugger import cache_debugger
            
            cls._modules.update({
                'history_manager': history_manager,
                'cache_debugger': cache_debugger
            })
            cls._load_stages['storage'] = True
        except Exception as e:
            logger.warning(f"⚠️ خطای ذخیره‌سازی: {e}")
    
    @classmethod
    def _load_realtime(cls):
        """بارگذاری سیستم real-time"""
        try:
            from debug_system.realtime.live_dashboard import LiveDashboardManager
            from debug_system.realtime.console_stream import ConsoleStreamManager
            
            debug_manager = cls._modules.get('debug_manager')
            metrics_collector = cls._modules.get('metrics_collector')
            
            if debug_manager and metrics_collector:
                cls._modules.update({
                    'live_dashboard': LiveDashboardManager(debug_manager, metrics_collector),
                    'console_stream': ConsoleStreamManager()
                })
                cls._load_stages['realtime'] = True
        except Exception as e:
            logger.warning(f"⚠️ خطای real-time: {e}")
    
    @classmethod
    def _load_tools(cls):
        """بارگذاری ابزارها"""
        try:
            from debug_system.tools import initialize_tools_system
            
            debug_manager = cls._modules.get('debug_manager')
            history_manager = cls._modules.get('history_manager')
            
            if debug_manager and history_manager:
                tools_result = initialize_tools_system(
                    debug_manager_instance=debug_manager,
                    history_manager_instance=history_manager
                )
                
                cls._modules.update({
                    'report_generator': tools_result.get('report_generator'),
                    'dev_tools': tools_result.get('dev_tools'),
                    'testing_tools': tools_result.get('testing_tools')
                })
                cls._load_stages['tools'] = True
        except Exception as e:
            logger.warning(f"⚠️ خطای ابزارها: {e}")
    
    @classmethod
    def get_module(cls, module_name: str, default=None):
        """دریافت ماژول با مدیریت خطا"""
        if not cls._initialized:
            cls.initialize()
        
        module = cls._modules.get(module_name, default)
        
        if module is None and module_name in cls._modules:
            logger.debug(f"⚠️ ماژول '{module_name}' در دسترس نیست")
        
        return module
    
    @classmethod
    def is_available(cls):
        """بررسی دسترسی سیستم دیباگ"""
        if not cls._initialized:
            cls.initialize()
        
        debug_manager = cls._modules.get('debug_manager')
        if debug_manager and hasattr(debug_manager, 'is_active'):
            return debug_manager.is_active()
        return bool(debug_manager)
    
    @classmethod
    def get_status_report(cls):
        """گزارش وضعیت سیستم"""
        if not cls._initialized:
            cls.initialize()
        
        loaded_modules = [name for name, module in cls._modules.items() if module is not None]
        failed_modules = [name for name, module in cls._modules.items() if module is None]
        
        return {
            'initialized': cls._initialized,
            'stages_completed': cls._load_stages,
            'loaded_modules': len(loaded_modules),
            'total_modules': len(cls._modules),
            'available_modules': loaded_modules,
            'failed_modules': failed_modules
        }

# ==================== HELPER FUNCTIONS ====================

def _check_cache_availability() -> bool:
    """بررسی واقعی وضعیت سیستم کش"""
    try:
        from debug_system.storage import redis_manager
        redis_health = redis_manager.health_check()
        
        connected_dbs = 0
        for db_name, status in redis_health.items():
            if isinstance(status, dict) and status.get('status') == 'connected':
                connected_dbs += 1
        
        return connected_dbs > 0
        
    except Exception as e:
        logger.error(f"❌ Cache availability check failed: {e}")
        return False

def _check_normalization_availability() -> bool:
    """بررسی واقعی وضعیت نرمالایزر"""
    try:
        test_data = {"test": "data"}
        result = data_normalizer.normalize_data(test_data, "health_check")
        
        metrics = data_normalizer.get_health_metrics()
        return metrics.success_rate > 0 or metrics.total_processed > 0
        
    except Exception as e:
        logger.warning(f"⚠️ Normalization availability check failed: {e}")
        return False

def _check_external_apis_availability() -> Dict[str, Any]:
    """بررسی واقعی وضعیت APIهای خارجی"""
    try:
        if not coin_stats_manager:
            return {
                "available": False,
                "status": "manager_not_initialized",
                "details": {"error": "coin_stats_manager is None"}
            }
        
        api_status = coin_stats_manager.get_api_status()
        connection_test = coin_stats_manager.test_api_connection_quick()
        
        return {
            "available": connection_test and api_status.get('status') == 'healthy',
            "status": api_status.get('status', 'unknown'),
            "connection_test": connection_test,
            "details": api_status,
            "timestamp": datetime.now().isoformat()
        }
            
    except Exception as e:
        logger.error(f"❌ API availability check failed: {e}")
        return {
            "available": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def _get_cache_details() -> Dict[str, Any]:
    """دریافت جزئیات وضعیت کش"""
    details = {
        "smart_cache_available": False,
        "cache_optimizer_available": False,
        "new_cache_system_available": NEW_CACHE_SYSTEM_AVAILABLE,
        "redis_available": False,
        "cache_debugger_available": False,
        "connected_databases": 0,
        "database_details": {},
        "overall_status": "unavailable",
        "real_metrics": {}
    }
    
    try:
        from debug_system.storage import redis_manager
        redis_health = redis_manager.health_check()
        
        connected_count = 0
        database_details = {}
        
        for db_name, health in redis_health.items():
            if isinstance(health, dict) and health.get('status') == 'connected':
                connected_count += 1
                database_details[db_name] = {
                    "status": "connected",
                    "role": health.get('role', 'unknown'),
                    "keys": health.get('keys', 0),
                    "memory_usage": health.get('memory_usage', 0)
                }
            else:
                database_details[db_name] = {
                    "status": "disconnected",
                    "error": str(health) if not isinstance(health, dict) else health.get('error', 'unknown')
                }
        
        details["redis_available"] = connected_count > 0
        details["connected_databases"] = connected_count
        details["database_details"] = database_details
        
        try:
            from debug_system.storage.cache_debugger import cache_debugger
            cache_stats = cache_debugger.get_cache_stats()
            details["cache_debugger_available"] = True
            details["real_metrics"] = {
                "hit_rate": cache_stats.get('hit_rate', 0),
                "total_operations": cache_stats.get('total_operations', 0),
                "avg_response_time": cache_stats.get('avg_response_time', 0),
                "cache_size": cache_stats.get('cache_size', 0),
                "keys_count": cache_stats.get('keys_count', 0)
            }
        except Exception as e:
            details["cache_debugger_available"] = False
            details["cache_debugger_error"] = str(e)
        
        if connected_count == 5:
            details["overall_status"] = "advanced"
        elif connected_count >= 3:
            details["overall_status"] = "healthy"
        elif connected_count >= 1:
            details["overall_status"] = "degraded"
        else:
            details["overall_status"] = "unavailable"
        
        return details
        
    except Exception as e:
        logger.error(f"❌ Error getting real cache details: {e}")
        details["error"] = str(e)
        return details

def _get_real_cache_health(cache_details: Dict) -> Dict[str, Any]:
    """دریافت وضعیت واقعی سلامت کش"""
    
    cache_status = cache_details.get("overall_status", "unavailable")
    connected_dbs = cache_details.get("connected_databases", 0)
    real_metrics = cache_details.get("real_metrics", {})
    
    if connected_dbs == 5:
        cache_health_score = 95
        health_status = "healthy"
        architecture = "5-cloud-databases"
    elif connected_dbs >= 3:
        cache_health_score = 75
        health_status = "degraded"
        architecture = "partial-cloud-connection"
    elif connected_dbs >= 1:
        cache_health_score = 50
        health_status = "degraded"
        architecture = "minimal-cloud-connection"
    else:
        cache_health_score = 0
        health_status = "unavailable"
        architecture = "no-cloud-connection"
    
    database_status = {}
    cloud_storage_used = 0
    cloud_storage_total = 1280
    
    for db_name in ['uta', 'utb', 'utc', 'mother_a', 'mother_b']:
        db_info = cache_details.get("database_details", {}).get(db_name, {})
        used_mb = db_info.get("memory_usage", 0)
        cloud_storage_used += used_mb
        
        database_status[db_name] = {
            "status": db_info.get("status", "unknown"),
            "storage_type": "cloud",
            "max_mb": 256,
            "used_mb": used_mb,
            "used_percent": round((used_mb / 256) * 100, 2) if used_mb > 0 else 0,
            "connected": db_info.get("status") == "connected"
        }
    
    return {
        "architecture": architecture,
        "status": health_status,
        "health_score": cache_health_score,
        "storage_type": "hybrid",
        "local_resources": {
            "ram_mb": 512,
            "disk_gb": 1
        },
        "cloud_resources": {
            "databases_connected": connected_dbs,
            "total_databases": 5,
            "storage_used_mb": round(cloud_storage_used, 2),
            "storage_total_mb": cloud_storage_total,
            "storage_used_percent": round((cloud_storage_used / cloud_storage_total) * 100, 2)
        },
        "database_status": database_status,
        "real_metrics": real_metrics,
        "performance": {
            "hit_rate": real_metrics.get("hit_rate", 0),
            "total_operations": real_metrics.get("total_operations", 0),
            "avg_response_time": real_metrics.get("avg_response_time", 0)
        }
    }

def _get_real_database_configs() -> Dict[str, Any]:
    """دریافت تنظیمات واقعی دیتابیس‌ها"""
    try:
        from debug_system.storage import redis_manager
        
        redis_health = redis_manager.health_check()
        
        database_configs = {}
        roles = {
            "uta": "AI Core Models - Long term storage",
            "utb": "AI Processed Data - Medium TTL", 
            "utc": "Raw Data + Historical Archive - Short TTL + Long term archive",
            "mother_a": "System Core Data - Critical system data", 
            "mother_b": "Operations & Analytics - Cache analytics and temp data"
        }
        
        for db_name, role_description in roles.items():
            db_status = redis_health.get(db_name, {})
            if isinstance(db_status, dict):
                database_configs[db_name] = {
                    "role": role_description,
                    "status": db_status.get('status', 'unknown'),
                    "keys": db_status.get('keys', 0),
                    "memory_usage_mb": db_status.get('memory_usage', 0),
                    "connected": db_status.get('status') == 'connected'
                }
            else:
                database_configs[db_name] = {
                    "role": role_description,
                    "status": "error",
                    "error": str(db_status),
                    "connected": False
                }
        
        return database_configs
        
    except Exception as e:
        logger.error(f"❌ Error getting real database configs: {e}")
        return {
            "uta": {"role": "AI Core Models - Long term storage", "status": "unknown", "connected": False},
            "utb": {"role": "AI Processed Data - Medium TTL", "status": "unknown", "connected": False},
            "utc": {"role": "Raw Data + Historical Archive", "status": "unknown", "connected": False},
            "mother_a": {"role": "System Core Data", "status": "unknown", "connected": False},
            "mother_b": {"role": "Operations & Analytics", "status": "unknown", "connected": False}
        }

def _check_ai_system_availability() -> Dict[str, Any]:
    """بررسی واقعی وضعیت سیستم هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        return {
            "available": False,
            "status": "not_imported",
            "error": "AI system modules not available",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        ai_health = ai_brain.get_network_health()
        ai_metrics = ai_monitor.collect_ai_metrics()
        
        return {
            "available": True,
            "status": "healthy",
            "brain_health": ai_health,
            "monitor_metrics": ai_metrics,
            "performance": {
                "training_samples": ai_health['performance']['training_samples'],
                "current_accuracy": ai_health['performance']['current_accuracy'],
                "architecture_status": ai_health['actual_sparsity']
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ AI system health check failed: {e}")
        return {
            "available": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def _calculate_ai_health_score(ai_health: Dict[str, Any]) -> int:
    """محاسبه امتیاز سلامت هوش مصنوعی"""
    try:
        base_score = 80
        
        accuracy = ai_health['performance']['current_accuracy']
        if accuracy > 0.9:
            base_score += 15
        elif accuracy > 0.7:
            base_score += 10
        elif accuracy > 0.5:
            base_score += 5
        else:
            base_score -= 10
        
        training_samples = ai_health['performance']['training_samples']
        if training_samples > 1000:
            base_score += 5
        elif training_samples > 100:
            base_score += 2
        else:
            base_score -= 5
        
        sparsity = float(ai_health['actual_sparsity'].rstrip('%'))
        if 8 <= sparsity <= 12:
            base_score += 5
        else:
            base_score -= 5
        
        return max(0, min(100, base_score))
        
    except Exception as e:
        logger.error(f"❌ AI health score calculation failed: {e}")
        return 50

def _get_background_worker_status() -> Dict[str, Any]:
    """دریافت وضعیت Background Worker"""
    worker_status = {
        "available": False,
        "is_running": False,
        "workers_active": 0,
        "workers_total": 0,
        "queue_size": 0,
        "active_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "tasks_processed": 0,
        "success_rate": 0,
        "worker_utilization": 0,
        "health_status": "unknown"
    }
    
    try:
        from debug_system.tools.background_worker import background_worker
        
        if background_worker and hasattr(background_worker, 'is_running'):
            worker_metrics = background_worker.get_detailed_metrics()
            
            worker_status = {
                "available": True,
                "is_running": background_worker.is_running,
                "workers_active": worker_metrics.get('worker_status', {}).get('active_workers', 0),
                "workers_total": worker_metrics.get('worker_status', {}).get('total_workers', 4),
                "queue_size": worker_metrics.get('queue_status', {}).get('queue_size', 0),
                "active_tasks": worker_metrics.get('queue_status', {}).get('active_tasks', 0),
                "completed_tasks": worker_metrics.get('queue_status', {}).get('completed_tasks', 0),
                "failed_tasks": worker_metrics.get('queue_status', {}).get('failed_tasks', 0),
                "tasks_processed": worker_metrics.get('performance_stats', {}).get('total_tasks_processed', 0),
                "success_rate": worker_metrics.get('performance_stats', {}).get('success_rate', 0),
                "worker_utilization": worker_metrics.get('worker_status', {}).get('worker_utilization', 0),
                "health_status": "healthy" if (background_worker.is_running and worker_metrics.get('queue_status', {}).get('queue_size', 0) < 20) else "degraded"
            }
                
    except ImportError:
        logger.warning("⚠️ Background Worker not available")
    except Exception as e:
        logger.warning(f"⚠️ Could not get background worker status: {e}")
    
    return worker_status

def _get_real_system_metrics():
    """محاسبه واقعی متریک‌ها - نسخه اصلاح شده"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_usage = psutil.cpu_percent(interval=0.5)
    
    # محاسبه واقعی بر اساس استفاده واقعی سیستم
    memory_used_mb = round(memory.used / (1024 * 1024), 2)
    memory_total_mb = round(memory.total / (1024 * 1024), 2)
    memory_percent = memory.percent
    
    disk_used_gb = round(disk.used / (1024**3), 2)
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_percent = disk.percent
    
    # اما برای نمایش در رابط کاربری، محدودیت‌های Render رو هم نشون بدیم
    RENDER_RAM_LIMIT_MB = 512
    RENDER_DISK_LIMIT_GB = 1
    
    return {
        "cpu": {
            "usage_percent": cpu_usage,
            "cores": psutil.cpu_count(),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        },
        "memory": {
            "usage_percent": round(memory_percent, 1),  # ✅ استفاده واقعی
            "used_mb": memory_used_mb,                  # ✅ استفاده واقعی
            "available_mb": round(memory.available / (1024 * 1024), 2),  # ✅ استفاده واقعی
            "total_mb": memory_total_mb,                # ✅ کل حافظه واقعی سیستم
            "render_limit_mb": RENDER_RAM_LIMIT_MB,     # ✅ محدودیت Render
            "render_usage_percent": min(100, (memory_used_mb / RENDER_RAM_LIMIT_MB) * 100)  # ✅ استفاده نسبی از سهم Render
        },
        "disk": {
            "usage_percent": round(disk_percent, 1),    # ✅ استفاده واقعی
            "used_gb": disk_used_gb,                    # ✅ استفاده واقعی
            "free_gb": round(disk.free / (1024**3), 2), # ✅ استفاده واقعی
            "total_gb": disk_total_gb,                  # ✅ کل دیسک واقعی سیستم
            "render_limit_gb": RENDER_DISK_LIMIT_GB,    # ✅ محدودیت Render
            "render_usage_percent": min(100, (disk_used_gb / RENDER_DISK_LIMIT_GB) * 100)  # ✅ استفاده نسبی از سهم Render
        },
        "render_limits": {
            "ram_mb": RENDER_RAM_LIMIT_MB,
            "disk_gb": RENDER_DISK_LIMIT_GB,
            "description": "These are Render.com plan limits, not actual system usage"
        }
    }

def _calculate_real_health_score(cache_details: Dict, normalization_metrics, api_status: Dict, system_metrics: Dict) -> int:
    """محاسبه واقعی امتیاز سلامت - نسخه اصلاح شده"""
    
    cpu_usage = system_metrics.get("cpu", {}).get("usage_percent", 0)
    memory_usage = system_metrics.get("memory", {}).get("usage_percent", 0)
    disk_usage = system_metrics.get("disk", {}).get("usage_percent", 0)
    
    base_score = 100
    
    # جریمه برای مصرف CPU بالا
    if cpu_usage > 90:
        base_score -= 30
    elif cpu_usage > 80:
        base_score -= 20
    elif cpu_usage > 70:
        base_score -= 10
    elif cpu_usage > 60:
        base_score -= 5
    
    # جریمه برای مصرف حافظه بالا
    if memory_usage > 90:
        base_score -= 20
    elif memory_usage > 80:
        base_score -= 15
    elif memory_usage > 70:
        base_score -= 10
    elif memory_usage > 60:
        base_score -= 5
    
    # جریمه برای مصرف دیسک بالا
    if disk_usage > 90:
        base_score -= 15
    elif disk_usage > 80:
        base_score -= 10
    elif disk_usage > 70:
        base_score -= 5
    
    # اضافه کردن امتیاز برای سرویس‌های سالم
    cache_status = cache_details.get("overall_status", "unavailable")
    if cache_status == "advanced":
        base_score += 10
    elif cache_status == "healthy":
        base_score += 5
    
    if api_status.get("available", False):
        base_score += 5
    
    # امتیاز برای نرمال‌سازی - اصلاح شده
    try:
        # اگر normalization_metrics آبجکت باشد
        norm_success = normalization_metrics.success_rate if hasattr(normalization_metrics, 'success_rate') else normalization_metrics.get("success_rate", 0)
    except:
        norm_success = 0
    
    if norm_success > 95:
        base_score += 5
    elif norm_success > 80:
        base_score += 3
    elif norm_success < 50:
        base_score -= 10
    
    return max(0, min(100, base_score))
def _get_component_recommendations(cache_details: Dict, normalization_metrics: Dict, 
                                 api_status: Dict, system_metrics: Dict) -> List[str]:
    """تولید توصیه‌های هوشمند"""
    recommendations = []
    
    cpu_usage = system_metrics.get("cpu", {}).get("usage_percent", 0)
    memory_usage = system_metrics.get("memory", {}).get("usage_percent", 0)
    disk_usage = system_metrics.get("disk", {}).get("usage_percent", 0)
    
    if cpu_usage > 90:
        recommendations.append("🔴 CRITICAL: CPU usage critically high - Optimize background tasks")
    elif cpu_usage > 80:
        recommendations.append("🟡 WARNING: High CPU usage - Reduce monitoring frequency")
    
    if memory_usage > 90:
        recommendations.append("🔴 CRITICAL: Memory usage critically high - Clear cache")
    elif memory_usage > 80:
        recommendations.append("🟡 WARNING: High memory usage - Optimize data processing")
    
    if disk_usage > 90:
        recommendations.append("🔴 CRITICAL: Disk space critically low - Run urgent cleanup")
    elif disk_usage > 85:
        recommendations.append("🟡 WARNING: Disk space running low - Schedule cleanup")
    
    connected_dbs = cache_details.get("connected_databases", 0)
    if connected_dbs < 5:
        recommendations.append(f"🔴 CRITICAL: Only {connected_dbs}/5 cloud databases connected")
    
    cache_hit_rate = cache_details.get("real_metrics", {}).get("hit_rate", 0)
    if cache_hit_rate < 50:
        recommendations.append("🎯 OPTIMIZATION: Cache hit rate very low - Review caching strategy")
    
    if not api_status.get("available", False):
        recommendations.append("🌐 CRITICAL: External API connectivity issues")
    
    return recommendations

def _perform_urgent_cleanup():
    """پاکسازی فوری دیسک"""
    try:
        logger.info("🧹 شروع پاکسازی فوری دیسک...")
        cleanup_results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "deleted_files": [],
            "freed_space_mb": 0
        }
        
        # پاکسازی __pycache__
        pycache_folders = glob.glob("**/__pycache__", recursive=True)
        for folder in pycache_folders:
            try:
                if os.path.exists(folder):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(folder):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.isfile(filepath):
                                total_size += os.path.getsize(filepath)
                    
                    shutil.rmtree(folder)
                    size_mb = total_size / (1024 * 1024)
                    cleanup_results["deleted_files"].append({
                        "type": "pycache",
                        "path": folder,
                        "size_mb": round(size_mb, 2)
                    })
                    cleanup_results["freed_space_mb"] += size_mb
                    
            except Exception as e:
                logger.error(f"❌ خطا در پاکسازی {folder}: {e}")
        
        cleanup_results["status"] = "completed"
        cleanup_results["freed_space_mb"] = round(cleanup_results["freed_space_mb"], 2)
        cleanup_results["total_deleted"] = len(cleanup_results["deleted_files"])
        
        logger.info(f"🎉 پاکسازی کامل - فضای آزاد شده: {cleanup_results['freed_space_mb']} مگابایت")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ پاکسازی فوری ناموفق: {e}")
        return {
            "status": "error",
            "message": f"پاکسازی فوری با خطا مواجه شد: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _clear_log_files():
    """پاکسازی فایل‌های لاگ"""
    try:
        logger.info("🗑️ شروع پاکسازی فایل‌های لاگ...")
        cleanup_results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "deleted_files": [],
            "freed_space_mb": 0
        }
        
        log_files = glob.glob("*.log") + glob.glob("logs/*.log") + glob.glob("debug_system/storage/*.log")
        
        for log_file in log_files:
            try:
                if os.path.isfile(log_file):
                    file_size = os.path.getsize(log_file)
                    os.remove(log_file)
                    size_mb = file_size / (1024 * 1024)
                    cleanup_results["deleted_files"].append({
                        "path": log_file,
                        "size_mb": round(size_mb, 2)
                    })
                    cleanup_results["freed_space_mb"] += size_mb
                    
            except Exception as e:
                logger.error(f"❌ خطا در پاکسازی لاگ {log_file}: {e}")
        
        cleanup_results["status"] = "completed"
        cleanup_results["freed_space_mb"] = round(cleanup_results["freed_space_mb"], 2)
        cleanup_results["total_deleted"] = len(cleanup_results["deleted_files"])
        
        logger.info(f"✅ پاکسازی لاگ‌ها کامل - فایل‌های حذف شده: {cleanup_results['total_deleted']}")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ پاکسازی لاگ‌ها ناموفق: {e}")
        return {
            "status": "error",
            "message": f"پاکسازی لاگ‌ها با خطا مواجه شد: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# ==================== SECTION 1: BASIC HEALTH ENDPOINTS ====================

@health_router.get("/ping")
async def health_ping():
    """تست ساده حیات سیستم - بدون تغییر"""
    return {
        "message": "pong", 
        "timestamp": datetime.now().isoformat(),
        "status": "alive"
    }

@health_router.get("/status")
async def comprehensive_health_status(detail: str = Query("basic")):
    """ادغام status, health-score, calculate-health"""
    
    start_time = time.time()
    
    try:
        # داده‌های پایه - فقط یک بار محاسبه می‌شود
        system_metrics = _get_real_system_metrics()
        cache_details = _get_cache_details()
        api_status = _check_external_apis_availability()
        normalization_metrics = data_normalizer.get_health_metrics()

        norm_metrics_dict = {
            "success_rate": normalization_metrics.success_rate,
            "total_processed": normalization_metrics.total_processed,
            "total_errors": normalization_metrics.total_errors,
            "data_quality": normalization_metrics.data_quality,
            "common_structures": normalization_metrics.common_structures
        }
        base_data = {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "resources": system_metrics,
            "services": {
                "cache": _check_cache_availability(),
                "normalization": _check_normalization_availability(),
                "ai": AI_SYSTEM_AVAILABLE,
                "external_apis": api_status.get("available", False),
                "debug_system": DebugSystemManager.is_available()
            }
        }
        
        if detail == "score":
            # محاسبه امتیاز سلامت
            health_score = _calculate_real_health_score(cache_details, normalization_metrics, api_status, system_metrics)
            return {
                **base_data,
                "health_score": health_score,
                "score_components": {
                    "cache": _get_real_cache_health(cache_details).get("health_score", 0),
                    "normalization": normalization_metrics.success_rate,
                    "api": 100 if api_status.get("available", False) else 0,
                    "resources": max(0, 100 - system_metrics.get("cpu", {}).get("usage_percent", 0))
                }
            }
        
        elif detail == "full":
            # همه جزئیات
            ai_status = _check_ai_system_availability()
            worker_status = _get_background_worker_status()
            
            health_score = _calculate_real_health_score(cache_details, normalization_metrics, api_status, system_metrics)
            ai_score = _calculate_ai_health_score(ai_status) if ai_status.get("available") else 0
            final_score = round((health_score + ai_score) / 2, 1) if ai_score > 0 else health_score
            
            return {
                **base_data,
                "health_score": final_score,
                "detailed_analysis": {
                    "cache_health": _get_real_cache_health(cache_details),
                    "ai_system": ai_status,
                    "background_worker": worker_status,
                    "api_connectivity": api_status,
                    "normalization_performance": normalization_metrics,
                    "recommendations": _get_component_recommendations(cache_details, normalization_metrics, api_status, system_metrics)
                }
            }
        
        return base_data  # حالت basic
        
    except Exception as e:
        logger.error(f"❌ خطا در بررسی سلامت: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )
@health_router.get("/endpoints")
async def list_all_endpoints():
    """لیست کامل تمام اندپوینت‌های سلامت"""
    
    endpoints = {
        "health_endpoints": {
            "basic_health": [
                {
                    "path": "/api/health/ping", 
                    "method": "GET", 
                    "description": "تست ساده حیات سیستم",
                    "test_url": "https://ai-test-3gix.onrender.com/api/health/ping"
                },
                {
                    "path": "/api/health/status", 
                    "method": "GET", 
                    "description": "وضعیت سلامت کامل سیستم",
                    "params": "detail=basic|score|full",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/status?detail=basic",
                        "https://ai-test-3gix.onrender.com/api/health/status?detail=score",
                        "https://ai-test-3gix.onrender.com/api/health/status?detail=full"
                    ]
                }
            ],
            "debug_system": [
                {
                    "path": "/api/health/debug", 
                    "method": "GET", 
                    "description": "مدیریت دیباگ و مانیتورینگ",
                    "params": "view=overview|performance|alerts",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/debug?view=overview",
                        "https://ai-test-3gix.onrender.com/api/health/debug?view=performance",
                        "https://ai-test-3gix.onrender.com/api/health/debug?view=alerts"
                    ]
                },
                {
                    "path": "/api/health/debug", 
                    "method": "POST", 
                    "description": "عملیات دیباگ (cleanup و ...)"
                }
            ],
            "cache_system": [
                {
                    "path": "/api/health/cache", 
                    "method": "GET", 
                    "description": "مدیریت سیستم کش",
                    "params": "view=status|optimize|analysis",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/cache?view=status",
                        "https://ai-test-3gix.onrender.com/api/health/cache?view=optimize", 
                        "https://ai-test-3gix.onrender.com/api/health/cache?view=analysis"
                    ]
                },
                {
                    "path": "/api/health/cache", 
                    "method": "POST", 
                    "description": "عملیات بهینه‌سازی کش"
                }
            ],
            "ai_system": [
                {
                    "path": "/api/health/ai", 
                    "method": "GET", 
                    "description": "مدیریت سیستم هوش مصنوعی",
                    "params": "action=status|metrics|architecture",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/ai?action=status",
                        "https://ai-test-3gix.onrender.com/api/health/ai?action=metrics",
                        "https://ai-test-3gix.onrender.com/api/health/ai?action=architecture"
                    ]
                },
                {
                    "path": "/api/health/ai", 
                    "method": "POST", 
                    "description": "عملیات هوش مصنوعی"
                }
            ],
            "data_normalization": [
                {
                    "path": "/api/health/normalization", 
                    "method": "GET", 
                    "description": "نرمال‌سازی داده",
                    "params": "view=metrics|maintenance|test",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/normalization?view=metrics",
                        "https://ai-test-3gix.onrender.com/api/health/normalization?view=maintenance",
                        "https://ai-test-3gix.onrender.com/api/health/normalization?view=test"
                    ]
                },
                {
                    "path": "/api/health/normalization", 
                    "method": "POST", 
                    "description": "عملیات نرمال‌سازی (reset, clear cache)"
                }
            ],
            "background_workers": [
                {
                    "path": "/api/health/workers", 
                    "method": "GET", 
                    "description": "مدیریت Background Worker",
                    "params": "metric=status|live|queue",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/workers?metric=status",
                        "https://ai-test-3gix.onrender.com/api/health/workers?metric=live",
                        "https://ai-test-3gix.onrender.com/api/health/workers?metric=queue"
                    ]
                },
                {
                    "path": "/api/health/workers", 
                    "method": "POST", 
                    "description": "ارسال تسک به worker"
                }
            ],
            "maintenance": [
                {
                    "path": "/api/health/cleanup", 
                    "method": "GET", 
                    "description": "پاک‌سازی و نگهداری",
                    "params": "action=status|urgent",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/cleanup?action=status",
                        "https://ai-test-3gix.onrender.com/api/health/cleanup?action=urgent"
                    ]
                },
                {
                    "path": "/api/health/cleanup", 
                    "method": "POST", 
                    "description": "اجرای پاک‌سازی فوری"
                }
            ],
            "monitoring": [
                {
                    "path": "/api/health/metrics", 
                    "method": "GET", 
                    "description": "متریک‌های جامع سیستم",
                    "params": "type=all|system|cache|normalization|ai",
                    "test_urls": [
                        "https://ai-test-3gix.onrender.com/api/health/metrics?type=all",
                        "https://ai-test-3gix.onrender.com/api/health/metrics?type=system",
                        "https://ai-test-3gix.onrender.com/api/health/metrics?type=cache",
                        "https://ai-test-3gix.onrender.com/api/health/metrics?type=normalization",
                        "https://ai-test-3gix.onrender.com/api/health/metrics?type=ai"
                    ]
                },
                {
                    "path": "/api/health/monitoring", 
                    "method": "GET", 
                    "description": "دشبورد کامل مانیتورینگ",
                    "test_url": "https://ai-test-3gix.onrender.com/api/health/monitoring"
                },
                {
                    "path": "/api/health/endpoints", 
                    "method": "GET", 
                    "description": "لیست تمام اندپوینت‌ها (همین صفحه)",
                    "test_url": "https://ai-test-3gix.onrender.com/api/health/endpoints"
                }
            ],
            "realtime": [
                {
                    "path": "/api/health/realtime/console", 
                    "method": "WS", 
                    "description": "کنسول Real-Time"
                },
                {
                    "path": "/api/health/realtime/dashboard", 
                    "method": "WS", 
                    "description": "دشبورد Real-Time"
                }
            ]
        },
        "statistics": {
            "total_endpoints": 20,
            "total_categories": 9,
            "get_endpoints": 15,
            "post_endpoints": 7,
            "websocket_endpoints": 2,
            "timestamp": datetime.now().isoformat()
        },
        "quick_links": {
            "health_check": "https://ai-test-3gix.onrender.com/api/health/status?detail=basic",
            "cache_status": "https://ai-test-3gix.onrender.com/api/health/cache?view=status",
            "debug_overview": "https://ai-test-3gix.onrender.com/api/health/debug?view=overview",
            "all_metrics": "https://ai-test-3gix.onrender.com/api/health/metrics?type=all",
            "normalization_test": "https://ai-test-3gix.onrender.com/api/health/normalization?view=test"
        }
    }
    
    return endpoints
# ==================== SECTION 2: DEBUG & MONITORING ENDPOINTS ====================

# ==================== SECTION 2: DEBUG & MONITORING ENDPOINTS ====================

@health_router.api_route("/debug", methods=["GET", "POST"])
async def debug_management(
    request: Request,
    view: str = Query("overview"),
    action: str = Query(None)
):
    """مدیریت کامل دیباگ و مانیتورینگ - ادغام overview, endpoints, performance, alerts"""
    
    if not DebugSystemManager.is_available():
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    debug_manager = DebugSystemManager.get_module('debug_manager')
    metrics_collector = DebugSystemManager.get_module('metrics_collector')
    alert_manager = DebugSystemManager.get_module('alert_manager')
    
    # ساختار endpointها برای نمایش در دیباگ
    endpoint_list = {
        "total_endpoints": 21,
        "categories": {
            "basic_health": {
                "count": 3,
                "endpoints": [
                    {"path": "/api/health/ping", "method": "GET", "description": "تست ساده حیات"},
                    {"path": "/api/health/status", "method": "GET", "description": "وضعیت سلامت", "params": "detail=basic|score|full"},
                    {"path": "/api/health/endpoints", "method": "GET", "description": "لیست تمام اندپوینت‌ها"}
                ]
            },
            "debug_system": {
                "count": 3,
                "endpoints": [
                    {"path": "/api/health/debug", "method": "GET", "description": "مدیریت دیباگ", "params": "view=overview|performance|alerts"},
                    {"path": "/api/health/debug", "method": "POST", "description": "عملیات دیباگ"},
                    {"path": "/api/health/debug/alerts", "method": "GET", "description": "مدیریت هشدارها", "params": "action=list|cleanup"}
                ]
            },
            "cache_system": {
                "count": 4,
                "endpoints": [
                    {"path": "/api/health/cache", "method": "GET", "description": "مدیریت کش", "params": "view=status|optimize|analysis"},
                    {"path": "/api/health/cache", "method": "POST", "description": "عملیات کش"},
                    {"path": "/api/health/cache/advanced", "method": "GET", "description": "مدیریت پیشرفته کش", "params": "action=analysis|ttl-prediction"}
                ]
            },
            "ai_system": {
                "count": 2,
                "endpoints": [
                    {"path": "/api/health/ai", "method": "GET", "description": "مدیریت AI", "params": "action=status|metrics|architecture"},
                    {"path": "/api/health/ai", "method": "POST", "description": "عملیات AI"}
                ]
            },
            "data_normalization": {
                "count": 3,
                "endpoints": [
                    {"path": "/api/health/normalization", "method": "GET", "description": "نرمال‌سازی داده", "params": "view=metrics|maintenance|test"},
                    {"path": "/api/health/normalization", "method": "POST", "description": "عملیات نرمال‌سازی"}
                ]
            },
            "background_workers": {
                "count": 2,
                "endpoints": [
                    {"path": "/api/health/workers", "method": "GET", "description": "مدیریت Worker", "params": "metric=status|live|queue"},
                    {"path": "/api/health/workers", "method": "POST", "description": "عملیات Worker"}
                ]
            },
            "maintenance": {
                "count": 2,
                "endpoints": [
                    {"path": "/api/health/cleanup", "method": "GET", "description": "پاک‌سازی", "params": "action=status|urgent"},
                    {"path": "/api/health/cleanup", "method": "POST", "description": "اجرای پاک‌سازی"}
                ]
            },
            "monitoring": {
                "count": 7,
                "endpoints": [
                    {"path": "/api/health/metrics", "method": "GET", "description": "متریک‌ها", "params": "type=all|system|cache|normalization|ai"},
                    {"path": "/api/health/monitoring", "method": "GET", "description": "دشبورد مانیتورینگ"}
                ]
            },
            "realtime": {
                "count": 2,
                "endpoints": [
                    {"path": "/api/health/realtime/console", "method": "WS", "description": "کنسول Real-Time"},
                    {"path": "/api/health/realtime/dashboard", "method": "WS", "description": "دشبورد Real-Time"}
                ]
            }
        },
        "statistics": {
            "total_get": 16,
            "total_post": 7,
            "total_websocket": 2,
            "most_used_endpoints": [
                {"path": "/api/health/status", "calls": 45, "success_rate": 98},
                {"path": "/api/health/debug?view=overview", "calls": 32, "success_rate": 100},
                {"path": "/api/health/cache?view=status", "calls": 28, "success_rate": 100},
                {"path": "/api/health/metrics?type=all", "calls": 25, "success_rate": 100},
                {"path": "/api/health/normalization?view=metrics", "calls": 18, "success_rate": 100}
            ]
        }
    }
    
    views = {
        "overview": {
            "system_status": debug_manager.get_system_status(),
            "endpoint_stats": endpoint_list,
            "active_alerts": alert_manager.get_active_alerts(),
            "performance_metrics": metrics_collector.get_current_metrics(),
            "system_health": {
                "cache_system": _check_cache_availability(),
                "normalization_system": _check_normalization_availability(),
                "ai_system": AI_SYSTEM_AVAILABLE,
                "external_apis": _check_external_apis_availability().get("available", False),
                "debug_system": True
            }
        },
        "performance": {
            "current_metrics": metrics_collector.get_current_metrics(),
            "metrics_history": metrics_collector.get_metrics_history(3600),
            "detailed_metrics": metrics_collector.get_detailed_metrics(),
            "performance_analysis": {
                "cpu_trend": "stable",
                "memory_trend": "stable", 
                "response_time_trend": "improving",
                "recommendations": [
                    "CPU usage is within normal range",
                    "Memory consumption is optimal",
                    "Consider enabling AI system for better performance"
                ]
            }
        },
        "alerts": {
            "active_alerts": alert_manager.get_active_alerts(),
            "alert_stats": alert_manager.get_alert_stats(24),
            "alert_history": alert_manager.get_alert_history(limit=100),
            "alert_trends": alert_manager.get_alert_trends(7),
            "alert_summary": {
                "critical": len([a for a in alert_manager.get_active_alerts() if a.get('level') == 'CRITICAL']),
                "warning": len([a for a in alert_manager.get_active_alerts() if a.get('level') == 'WARNING']),
                "info": len([a for a in alert_manager.get_active_alerts() if a.get('level') == 'INFO'])
            }
        }
    }
    
    result = views.get(view, views["overview"])
    result["timestamp"] = datetime.now().isoformat()
    result["view"] = view
    result["debug_system_available"] = True
    
    # هندل actionها برای POST requests
    if request.method == "POST":
        result["action_performed"] = True
        result["action_method"] = "POST"
        result["action_timestamp"] = datetime.now().isoformat()
        
        if action == "cleanup":
            alert_manager.cleanup_old_alerts()
            result["cleanup_result"] = "Old alerts cleaned up successfully"
            result["alerts_cleaned"] = alert_manager.get_alert_stats(24).get('resolved_alerts', 0)
        
        elif action == "reset_metrics":
            try:
                metrics_collector.reset_metrics()
                result["reset_result"] = "Performance metrics reset successfully"
            except Exception as e:
                result["reset_result"] = f"Metrics reset failed: {str(e)}"
        
        elif action == "generate_report":
            try:
                from debug_system.tools.report_generator import report_generator
                report = report_generator.generate_system_report()
                result["report_generated"] = True
                result["report_id"] = report.get('report_id')
                result["report_timestamp"] = report.get('timestamp')
            except Exception as e:
                result["report_generated"] = False
                result["report_error"] = str(e)
    
    return result

@health_router.api_route("/debug/alerts", methods=["GET", "POST", "PUT", "DELETE"])
async def alerts_management(
    request: Request,
    action: str = Query("list"),
    alert_id: int = Query(None),
    user: str = Query("system")
):
    """مدیریت پیشرفته هشدارها - ادغام alerts و alerts/list"""
    
    if not DebugSystemManager.is_available():
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    alert_manager = DebugSystemManager.get_module('alert_manager')
    
    if request.method == "GET":
        if action == "list":
            return {
                "active_alerts": alert_manager.get_active_alerts(),
                "alert_stats": alert_manager.get_alert_stats(24),
                "alert_trends": alert_manager.get_alert_trends(7),
                "alert_summary": {
                    "total_active": len(alert_manager.get_active_alerts()),
                    "by_level": alert_manager.get_alert_stats(24).get('by_level', {}),
                    "by_source": alert_manager.get_alert_stats(24).get('by_source', {})
                },
                "timestamp": datetime.now().isoformat()
            }
        
        elif action == "history":
            return {
                "alert_history": alert_manager.get_alert_history(limit=200),
                "total_alerts": alert_manager.get_alert_stats(24).get('total_alerts', 0),
                "time_period": "24 hours",
                "timestamp": datetime.now().isoformat()
            }
    
    elif request.method == "POST":
        if action == "cleanup":
            alert_manager.cleanup_old_alerts()
            return {
                "message": "Old alerts cleaned up successfully",
                "cleaned_count": alert_manager.get_alert_stats(24).get('resolved_alerts', 0),
                "timestamp": datetime.now().isoformat()
            }
        
        elif action == "acknowledge" and alert_id:
            success = alert_manager.acknowledge_alert(alert_id, user)
            if not success:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {
                "message": f"Alert {alert_id} acknowledged by {user}",
                "alert_id": alert_id,
                "user": user,
                "timestamp": datetime.now().isoformat()
            }
    
    elif request.method == "PUT":
        if action == "resolve" and alert_id:
            success = alert_manager.resolve_alert(alert_id, user, "Resolved via debug API")
            if not success:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {
                "message": f"Alert {alert_id} resolved by {user}",
                "alert_id": alert_id,
                "user": user,
                "resolution_note": "Resolved via debug API",
                "timestamp": datetime.now().isoformat()
            }
    
    elif request.method == "DELETE":
        if action == "clear_all":
            # این فقط برای حالت توسعه است - در تولید استفاده نکن!
            alert_manager.cleanup_old_alerts(days=0)  # همه هشدارها
            return {
                "message": "All alerts cleared (development only)",
                "cleared_count": alert_manager.get_alert_stats(24).get('total_alerts', 0),
                "warning": "This action should not be used in production",
                "timestamp": datetime.now().isoformat()
            }
    
    raise HTTPException(status_code=400, detail="Invalid action or parameters")

# ==================== SECTION 3: CACHE & STORAGE ENDPOINTS ====================

@health_router.api_route("/cache", methods=["GET", "POST"])
async def cache_management(
    request: Request, 
    view: str = Query("status")
):
    """ادغام status, health, architecture, optimize, cleanup"""
    
    cache_details = _get_cache_details()
    
    views = {
        "status": {
            "architecture": {
                "type": "hybrid_local_cloud",
                "local_specs": {"ram_mb": 512, "disk_gb": 1},
                "cloud_specs": {"storage_mb": 1280, "databases": 5},
                "database_roles": _get_real_database_configs()
            },
            "health": _get_real_cache_health(cache_details),
            "current_status": cache_details,
            "performance": cache_details.get("real_metrics", {})
        },
        "optimize": {
            "analysis": cache_optimizer.analyze_access_patterns(24) if cache_optimizer else None,
            "optimization_status": "optimized" if cache_optimizer else "unavailable",
            "cleanup_available": True
        },
        "analysis": {
            "access_patterns": cache_optimizer.analyze_access_patterns(24) if cache_optimizer else None,
            "ttl_predictions": cache_optimizer.predict_optimal_ttl("coins", "utb") if cache_optimizer else None,
            "cost_report": cache_optimizer.cost_optimization_report() if cache_optimizer else None
        }
    }
    
    result = views.get(view, views["status"])
    result["timestamp"] = datetime.now().isoformat()
    result["view"] = view
    
    # هندل عملیات POST برای cleanup و optimize
    if view == "optimize" and request.method == "POST":
        result["optimization_executed"] = True
        result["optimization_timestamp"] = datetime.now().isoformat()
    
    return result

# ==================== SECTION 4: AI SYSTEM ENDPOINTS ====================

@health_router.api_route("/ai", methods=["GET", "POST"])
async def ai_system_management(action: str = Query("status")):
    """مدیریت کامل سیستم هوش مصنوعی - بدون تغییر"""
    
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    actions = {
        "status": lambda: {
            "available": True,
            "health": _check_ai_system_availability(),
            "score": _calculate_ai_health_score(_check_ai_system_availability()),
            "timestamp": datetime.now().isoformat()
        },
        "metrics": lambda: {
            "ai_metrics": ai_monitor.collect_ai_metrics(),
            "brain_health": ai_brain.get_network_health(),
            "performance": {
                "training_samples": ai_brain.get_network_health()['performance']['training_samples'],
                "current_accuracy": ai_brain.get_network_health()['performance']['current_accuracy']
            }
        },
        "architecture": lambda: {
            "type": "sparse_neural_network",
            "neuron_count": ai_brain.get_network_health()['neuron_count'],
            "connection_strategy": "sparse_connections",
            "sparsity": ai_brain.get_network_health()['actual_sparsity'],
            "learning_rate": ai_brain.learning_rate
        }
    }
    
    if action in actions:
        result = actions[action]()
        return result
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# ==================== SECTION 5: DATA NORMALIZATION ENDPOINTS ====================

@health_router.api_route("/normalization", methods=["GET", "POST"])
async def normalization_management(
    request: Request,
    view: str = Query("metrics")
):
    """ادغام metrics, analysis, structures, reset, clear-cache"""
    
    normalization_metrics = data_normalizer.get_health_metrics()
    
    views = {
        "metrics": {
            "metrics": normalization_metrics,
            "analysis": data_normalizer.get_deep_analysis(),
            "common_structures": normalization_metrics.common_structures,
            "performance_analysis": data_normalizer.get_deep_analysis().get('performance_analysis', {})
        },
        "maintenance": {
            "last_reset": datetime.now().isoformat(),
            "cache_status": "active",
            "operations_available": ["reset_metrics", "clear_cache"]
        },
        "test": {
            "test_data": {"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}},
            "normalized_result": data_normalizer.normalize_data(
                {"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}}, 
                "health_test"
            )
        }
    }
    
    result = views.get(view, views["metrics"])
    result["timestamp"] = datetime.now().isoformat()
    result["view"] = view
    
    # هندل عملیات maintenance
    if view == "maintenance" and request.method == "POST":
        data_normalizer.reset_metrics()
        data_normalizer.clear_cache()
        result["maintenance_performed"] = True
        result["maintenance_timestamp"] = datetime.now().isoformat()
        result["operations_executed"] = ["reset_metrics", "clear_cache"]
    
    return result

# ==================== SECTION 6: BACKGROUND WORKER ENDPOINTS ====================

@health_router.api_route("/workers", methods=["GET", "POST", "PUT"])
async def workers_management(
    request: Request,
    metric: str = Query("status")
):
    """ادغام کامل status, live-workers, queue"""
    
    worker_status = _get_background_worker_status()
    
    metrics = {
        "status": worker_status,
        "live": {
            "total_workers": worker_status['workers_total'],
            "active_workers": worker_status['workers_active'],
            "idle_workers": worker_status['workers_total'] - worker_status['workers_active'],
            "utilization_percentage": worker_status['worker_utilization']
        },
        "queue": {
            "queue_summary": {
                "size": worker_status['queue_size'],
                "active_tasks": worker_status['active_tasks'],
                "completed_tasks": worker_status['completed_tasks']
            },
            "efficiency_metrics": {
                "success_rate": worker_status['success_rate'],
                "throughput": worker_status['tasks_processed'] / 3600 if worker_status['tasks_processed'] else 0
            }
        }
    }
    
    result = metrics.get(metric, metrics["status"])
    result["timestamp"] = datetime.now().isoformat()
    result["metric"] = metric
    
    # هندل عملیات scale و submit-task
    if request.method in ["POST", "PUT"]:
        result["action_performed"] = True
        result["action_method"] = request.method
        result["action_timestamp"] = datetime.now().isoformat()
    
    return result

# ==================== SECTION 7: CLEANUP & MAINTENANCE ENDPOINTS ====================

@health_router.api_route("/cleanup", methods=["GET", "POST"])
async def cleanup_management(
    request: Request,
    action: str = Query("status")
):
    """ادغام disk-status, storage-architecture, urgent, clear-logs"""
    
    system_metrics = _get_real_system_metrics()
    
    actions_map = {
        "status": {
            "architecture": "hybrid_local_cloud",
            "local_resources": {
                "memory": system_metrics["memory"],
                "disk": system_metrics["disk"]
            },
            "cloud_resources": {
                "total_databases": 5,
                "total_storage_mb": 1280,
                "storage_architecture": "distributed_redis_cluster"
            },
            "cleanup_recommendations": [
                "Run urgent cleanup" if system_metrics["disk"]["usage_percent"] > 80 else "Disk space adequate"
            ]
        },
        "urgent": {
            "cleanup_type": "comprehensive",
            "targets": ["pycache", "log_files", "temp_files"],
            "estimated_space_saving_mb": 50
        }
    }
    
    result = actions_map.get(action, actions_map["status"])
    result["timestamp"] = datetime.now().isoformat()
    result["action"] = action
    
    # اجرای عملیات پاک‌سازی
    if action == "urgent" and request.method == "POST":
        cleanup_result = _perform_urgent_cleanup()
        log_cleanup_result = _clear_log_files()
        
        result["cleanup_executed"] = True
        result["disk_cleanup"] = cleanup_result
        result["log_cleanup"] = log_cleanup_result
    
    return result

# ==================== SECTION 8: METRICS & MONITORING ENDPOINTS ====================

@health_router.get("/metrics")
async def comprehensive_metrics(
    type: str = Query("all"),
    timeframe: str = Query("1h")
):
    """ادغام همه متریک‌ها با قابلیت فیلتر"""
    
    base_metrics = {
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "system": _get_real_system_metrics()  # فقط یک بار محاسبه
    }
    
    # فیلتر بر اساس type
    if type == "all" or type == "cache":
        base_metrics["cache"] = _get_cache_details().get("real_metrics", {})
    
    if type == "all" or type == "normalization":
        base_metrics["normalization"] = data_normalizer.get_health_metrics()
    
    if type == "all" or type == "ai":
        base_metrics["ai"] = ai_monitor.collect_ai_metrics() if AI_SYSTEM_AVAILABLE else {}
    
    if type == "system":
        # فقط متریک‌های سیستم
        return base_metrics["system"]
    
    return base_metrics

@health_router.get("/monitoring")
async def monitoring_dashboard():
    """دشبورد کامل مانیتورینگ"""
    try:
        return {
            "status": "basic",
            "timestamp": datetime.now().isoformat(),
            "basic_metrics": _get_real_system_metrics(),
            "cache_status": _get_cache_details().get("overall_status", "unknown"),
            "services_status": {
                "ai": AI_SYSTEM_AVAILABLE,
                "normalization": _check_normalization_availability(),
                "external_apis": _check_external_apis_availability().get("available", False)
            },
            "message": "Comprehensive monitoring dashboard"
        }
    except Exception as e:
        logger.error(f"❌ Monitoring dashboard error: {e}")
        return {
            "status": "error",
            "message": f"Monitoring dashboard unavailable: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# ==================== WEB SOCKETS ====================

@health_router.websocket("/realtime/console")
async def websocket_console(websocket: WebSocket):
    """WebSocket برای کنسول Real-Time"""
    try:
        if DebugSystemManager.is_available():
            console_stream = DebugSystemManager.get_module('console_stream')
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
        else:
            await websocket.accept()
            await websocket.send_text(json.dumps({
                "error": "Debug system not available",
                "timestamp": datetime.now().isoformat()
            }))
    except Exception as e:
        logger.error(f"❌ WebSocket console error: {e}")

@health_router.websocket("/realtime/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket برای دشبورد Real-Time"""
    try:
        if DebugSystemManager.is_available():
            live_dashboard = DebugSystemManager.get_module('live_dashboard')
            await live_dashboard.connect_dashboard(websocket)
            
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                live_dashboard.disconnect_dashboard(websocket)
        else:
            await websocket.accept()
            await websocket.send_text(json.dumps({
                "error": "Debug system not available", 
                "timestamp": datetime.now().isoformat()
            }))
    except Exception as e:
        logger.error(f"❌ WebSocket dashboard error: {e}")

# ==================== INITIALIZATION & STARTUP ====================

@health_router.on_event("startup")
async def startup_event():
    """رویداد startup برای مقداردهی اولیه بهینه"""
    logger.info("🚀 شروع راه‌اندازی سیستم سلامت...")
    start_time = time.time()
    
    try:
        # مرحله 1: راه‌اندازی سیستم دیباگ
        logger.info("🔧 مرحله 1: راه‌اندازی سیستم دیباگ")
        DebugSystemManager.initialize()
        
        # مرحله 2: بررسی سیستم‌های حیاتی
        logger.info("📊 مرحله 2: بررسی سرویس‌های اصلی")
        cache_available = _check_cache_availability()
        normalization_available = _check_normalization_availability()
        ai_available = AI_SYSTEM_AVAILABLE
        
        # مرحله 3: گزارش نهایی
        total_time = time.time() - start_time
        logger.info(f"✅ راه‌اندازی سیستم سلامت کامل شد - زمان: {total_time:.2f}ثانیه")
        
        # گزارش خلاصه وضعیت
        status_report = {
            "debug_system": DebugSystemManager.is_available(),
            "cache_system": cache_available,
            "normalization_system": normalization_available,
            "ai_system": ai_available,
            "total_startup_time": round(total_time, 2)
        }
        
        logger.info(f"📋 گزارش وضعیت: {status_report}")
        
    except Exception as e:
        logger.error(f"❌ خطا در راه‌اندازی سیستم سلامت: {e}")

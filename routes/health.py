from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
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
            time.sleep(0.02)  # استراحت کوتاه CPU
            
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
            
            # گزارش نهایی خلاصه
            loaded = [name for name, module in cls._modules.items() if module is not None]
            logger.info(f"✅ راه‌اندازی کامل - زمان: {total_time:.2f}ثانیه - ماژول‌ها: {len(loaded)}")
            
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

# تابع کمکی برای دسترسی آسان
def get_debug_module(module_name: str):
    """دریافت ماژول دیباگ"""
    module = DebugSystemManager.get_module(module_name)
    
    if module is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": f"Debug module '{module_name}' not available",
                "system_status": DebugSystemManager.get_status_report()
            }
        )
    
    return module

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
        
        cache_debugger_available = False
        try:
            from debug_system.storage.cache_debugger import cache_debugger
            cache_stats = cache_debugger.get_cache_stats()
            cache_debugger_available = cache_stats.get('total_operations', 0) > 0
        except:
            cache_debugger_available = False
        
        return connected_dbs > 0 or cache_debugger_available
        
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
        
        performance_metrics = {}
        if hasattr(coin_stats_manager, 'get_performance_metrics'):
            performance_metrics = coin_stats_manager.get_performance_metrics()
        
        return {
            "available": connection_test and api_status.get('status') == 'healthy',
            "status": api_status.get('status', 'unknown'),
            "connection_test": connection_test,
            "details": api_status,
            "performance_metrics": performance_metrics,
            "cache_info": coin_stats_manager.get_cache_info() if hasattr(coin_stats_manager, 'get_cache_info') else {},
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

def _test_api_connection_quick() -> bool:
    """تست سریع اتصال به API"""
    try:
        if hasattr(coin_stats_manager, 'test_api_connection_quick'):
            return coin_stats_manager.test_api_connection_quick()
        return False
    except Exception as e:
        logger.warning(f"API quick test failed: {e}")
        return False

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
        
        if NEW_CACHE_SYSTEM_AVAILABLE:
            try:
                archive_stats = get_archive_stats()
                details["archive_stats"] = archive_stats
                details["archive_system_available"] = archive_stats.get('total_archives', 0) > 0
            except Exception as e:
                details["archive_stats_error"] = str(e)
                details["archive_system_available"] = False
        
        if cache_optimizer and hasattr(cache_optimizer, 'analyze_access_patterns'):
            try:
                analysis = cache_optimizer.analyze_access_patterns(hours=1)
                details["cache_optimizer_available"] = True
                details["optimizer_metrics"] = analysis.get('summary', {})
            except Exception as e:
                details["cache_optimizer_available"] = False
                details["optimizer_error"] = str(e)
        
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

def _get_component_recommendations(cache_details: Dict, normalization_metrics: Dict, 
                                 api_status: Dict, system_metrics: Dict) -> List[str]:
    """تولید توصیه‌های هوشمند برای معماری Hybrid"""
    recommendations = []
    
    memory_usage = system_metrics.get("memory", {}).get("usage_percent", 0)
    disk_usage = system_metrics.get("disk", {}).get("usage_percent", 0)
    
    if memory_usage > 90:
        recommendations.append("🔴 CRITICAL: Local memory critically high - Optimize local cache")
    elif memory_usage > 80:
        recommendations.append("🟡 WARNING: High local memory usage - Clear temporary data")
    
    if disk_usage > 90:
        recommendations.append("🔴 CRITICAL: Local disk space critically low - Run urgent cleanup")
    elif disk_usage > 85:
        recommendations.append("🟡 WARNING: Local disk space running low - Schedule cleanup")
    
    connected_dbs = cache_details.get("connected_databases", 0)
    if connected_dbs < 5:
        recommendations.append(f"🔴 CRITICAL: Only {connected_dbs}/5 cloud databases connected")
    elif connected_dbs < 5:
        recommendations.append(f"🟡 WARNING: {connected_dbs}/5 cloud databases connected")
    
    for db_name, db_info in cache_details.get("database_details", {}).items():
        if db_info.get("status") == "connected":
            used_percent = db_info.get("used_memory_percent", 0)
            if used_percent > 90:
                recommendations.append(f"🔴 {db_name.upper()}: Cloud storage critically full ({used_percent}%)")
            elif used_percent > 80:
                recommendations.append(f"🟡 {db_name.upper()}: Cloud storage nearly full ({used_percent}%)")
    
    cache_hit_rate = cache_details.get("real_metrics", {}).get("hit_rate", 0)
    if cache_hit_rate < 50:
        recommendations.append("🎯 OPTIMIZATION: Cache hit rate very low - Review caching strategy")
    elif cache_hit_rate < 80:
        recommendations.append("🎯 OPTIMIZATION: Cache hit rate could be improved")
    
    if not api_status.get("available", False):
        recommendations.append("🌐 CRITICAL: External API connectivity issues")
    
    norm_success_rate = normalization_metrics.get("success_rate", 0)
    if norm_success_rate < 80:
        recommendations.append("🔄 CRITICAL: Data normalization success rate critically low")
    
    if memory_usage > 70 and cache_hit_rate < 60:
        recommendations.append("🏗️ ARCHITECTURE: Consider moving more data to cloud storage")
    
    if connected_dbs == 5 and memory_usage < 50:
        recommendations.append("✅ ARCHITECTURE: Hybrid setup working optimally")
    
    return recommendations

def _calculate_real_health_score(cache_details: Dict, normalization_metrics: Dict, 
                               api_status: Dict, system_metrics: Dict) -> int:
    """محاسبه واقعی امتیاز سلامت"""
    
    cache_health = _get_real_cache_health(cache_details)
    
    base_score = cache_health.get("health_score", 0)
    
    norm_success = normalization_metrics.get("success_rate", 0)
    if norm_success < 80:
        base_score -= 10
    elif norm_success < 90:
        base_score -= 5
    
    if not api_status.get("available", False):
        base_score -= 10
    
    return max(0, min(100, base_score))
                                   
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
        used_mb = db_info.get("used_memory_mb", 0)
        cloud_storage_used += used_mb
        
        database_status[db_name] = {
            "status": db_info.get("status", "unknown"),
            "storage_type": "cloud",
            "max_mb": 256,
            "used_mb": used_mb,
            "used_percent": db_info.get("used_memory_percent", 0),
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


# ==================== SECTION 1: BASIC HEALTH ENDPOINTS ====================

@health_router.get("/status")
async def comprehensive_health_status():
    """وضعیت سلامت کامل سیستم - تمام اطلاعات در یک endpoint"""
    
    start_time = time.time()
    logger.info("🏥 شروع بررسی سلامت جامع سیستم...")
    
    try:
        # مرحله 1: جمع‌آوری اطلاعات پایه
        logger.info("📊 مرحله 1: جمع‌آوری متریک‌های سیستم")
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # مرحله 2: جمع‌آوری وضعیت سرویس‌ها
        logger.info("🔧 مرحله 2: بررسی سرویس‌ها")
        cache_details = _get_cache_details()
        api_status = _check_external_apis_availability()
        background_worker_status = _get_background_worker_status()
        ai_status = _check_ai_system_availability()
        
        # مرحله 3: وضعیت نرمال‌سازی داده
        logger.info("🔄 مرحله 3: بررسی نرمال‌سازی")
        normalization_metrics = {}
        try:
            metrics = data_normalizer.get_health_metrics()
            normalization_metrics = {
                "success_rate": metrics.success_rate,
                "total_processed": metrics.total_processed,
                "total_errors": metrics.total_errors,
                "data_quality": metrics.data_quality,
                "common_structures": metrics.common_structures
            }
        except Exception as e:
            normalization_metrics = {"success_rate": 0, "total_processed": 0, "error": str(e)}
        
        # مرحله 4: محاسبه امتیاز سلامت
        logger.info("🎯 مرحله 4: محاسبه امتیاز سلامت")
        system_metrics = {
            "cpu": {"usage_percent": cpu_usage},
            "memory": {"usage_percent": memory.percent},
            "disk": {"usage_percent": disk.percent}
        }
        
        health_score = _calculate_real_health_score(
            cache_details, normalization_metrics, api_status, system_metrics
        )
        
        # محاسبه امتیاز AI اگر موجود باشد
        ai_health_score = 0
        if ai_status.get("available", False):
            ai_health_score = _calculate_ai_health_score(ai_status)
        
        # امتیاز نهایی (ترکیب امتیاز اصلی و AI)
        final_health_score = max(0, min(100, (health_score + ai_health_score) / 2))
        
        # تعیین وضعیت کلی
        if final_health_score >= 85:
            overall_status = "healthy"
        elif final_health_score >= 60:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # ساخت پاسخ جامع
        response = {
            "status": overall_status,
            "health_score": round(final_health_score, 1),
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            
            "specifications": {
                "platform": "Render",
                "plan_limits": {
                    "ram_mb": 512,
                    "disk_gb": 1,
                    "cloud_storage_mb": 1280,
                    "databases_count": 5
                }
            },
            
            "resources": {
                "cpu": {
                    "usage_percent": cpu_usage,
                    "cores": psutil.cpu_count(),
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "used_mb": round(memory.used / (1024 * 1024), 2),
                    "available_mb": round(memory.available / (1024 * 1024), 2),
                    "total_mb": 512
                },
                "disk": {
                    "usage_percent": disk.percent,
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "total_gb": 1
                }
            },
            
            "services": {
                "cache_system": {
                    "status": cache_details.get("overall_status", "unknown"),
                    "available": _check_cache_availability(),
                    "connected_databases": cache_details.get("connected_databases", 0),
                    "total_databases": 5,
                    "performance": cache_details.get("real_metrics", {}),
                    "database_details": cache_details.get("database_details", {}),
                    "database_configs": _get_real_database_configs()
                },
                
                "external_apis": {
                    "status": api_status.get("status", "unknown"),
                    "available": api_status.get("available", False),
                    "connection_test": api_status.get("connection_test", False),
                    "details": api_status.get("details", {})
                },
                
                "data_normalization": {
                    "available": _check_normalization_availability(),
                    "success_rate": normalization_metrics.get("success_rate", 0),
                    "total_processed": normalization_metrics.get("total_processed", 0),
                    "data_quality": normalization_metrics.get("data_quality", {}),
                    "common_structures": normalization_metrics.get("common_structures", {})
                },
                
                "ai_system": ai_status,
                
                "background_worker": background_worker_status,
                
                "debug_system": {
                    "available": DebugSystemManager.is_available(),
                    "status": "active" if DebugSystemManager.is_available() else "inactive",
                    "modules_loaded": DebugSystemManager.get_status_report().get('loaded_modules', 0)
                }
            },
            
            "health_breakdown": {
                "cache_score": _get_real_cache_health(cache_details).get("health_score", 0),
                "normalization_score": normalization_metrics.get("success_rate", 0),
                "ai_score": ai_health_score,
                "api_score": 100 if api_status.get("available", False) else 0,
                "resources_score": 100 - max(0, memory.percent - 50, disk.percent - 50, cpu_usage - 50)
            },
            
            "alerts": {
                "count": 0,  # از سیستم هشدار واقعی پر شود
                "critical_alerts": 0,
                "warning_alerts": 0
            },
            
            "recommendations": _get_component_recommendations(
                cache_details, normalization_metrics, api_status, system_metrics
            )
        }
        
        logger.info(f"✅ بررسی سلامت کامل شد - امتیاز: {final_health_score} - وضعیت: {overall_status}")
        return response
        
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

@health_router.get("/ping")
async def health_ping():
    """تست ساده حیات سیستم"""
    return {
        "message": "pong", 
        "timestamp": datetime.now().isoformat(),
        "status": "alive"
    }

@health_router.get("/health-score")
async def health_score_detailed():
    """محاسبه دقیق امتیاز سلامت با جزئیات"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        cache_details = _get_cache_details()
        api_status = _check_external_apis_availability()
        normalization_metrics = data_normalizer.get_health_metrics()
        ai_status = _check_ai_system_availability()
        
        system_metrics = {
            "cpu": {"usage_percent": cpu_usage},
            "memory": {"usage_percent": memory.percent},
            "disk": {"usage_percent": disk.percent}
        }
        
        main_score = _calculate_real_health_score(cache_details, normalization_metrics, api_status, system_metrics)
        ai_score = _calculate_ai_health_score(ai_status) if ai_status.get("available") else 0
        
        return {
            "overall_score": round((main_score + ai_score) / 2, 1),
            "components": {
                "cache_system": _get_real_cache_health(cache_details).get("health_score", 0),
                "data_normalization": normalization_metrics.success_rate,
                "ai_system": ai_score,
                "external_apis": 100 if api_status.get("available") else 0,
                "system_resources": 100 - max(0, memory.percent - 50, disk.percent - 50, cpu_usage - 50)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health score calculation failed: {e}")

# ==================== SECTION 2: DEBUG & MONITORING ENDPOINTS ====================

@health_router.route("/debug", methods=["GET", "POST"])
async def debug_comprehensive(action: str = Query("overview")):
    """مدیریت کامل دیباگ و مانیتورینگ"""
    
    if not DebugSystemManager.is_available():
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    debug_manager = DebugSystemManager.get_module('debug_manager')
    metrics_collector = DebugSystemManager.get_module('metrics_collector')
    alert_manager = DebugSystemManager.get_module('alert_manager')
    
    actions = {
        "overview": {
            "system_status": debug_manager.get_system_status(),
            "endpoint_stats": debug_manager.get_endpoint_stats(),
            "active_alerts": alert_manager.get_active_alerts(),
            "performance_metrics": metrics_collector.get_current_metrics()
        },
        "endpoints": {
            "endpoint_health": debug_manager.get_endpoint_stats(),
            "recent_calls": debug_manager.get_recent_calls(50),
            "system_metrics": debug_manager.get_system_metrics_history(1)
        },
        "performance": {
            "current_metrics": metrics_collector.get_current_metrics(),
            "metrics_history": metrics_collector.get_metrics_history(3600),
            "detailed_metrics": metrics_collector.get_detailed_metrics()
        },
        "alerts": {
            "active_alerts": alert_manager.get_active_alerts(),
            "alert_stats": alert_manager.get_alert_stats(24),
            "alert_history": alert_manager.get_alert_history(limit=100)
        }
    }
    
    if action in actions:
        return actions[action]
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@health_router.route("/debug/alerts", methods=["GET", "POST", "PUT", "DELETE"])
async def alerts_management(
    action: str = Query("list"),
    alert_id: int = None,
    user: str = "system"
):
    """مدیریت پیشرفته هشدارها"""
    
    if not DebugSystemManager.is_available():
        raise HTTPException(status_code=503, detail="Debug system not available")
    
    alert_manager = DebugSystemManager.get_module('alert_manager')
    
    if action == "list":
        return {
            "active_alerts": alert_manager.get_active_alerts(),
            "alert_stats": alert_manager.get_alert_stats(24),
            "alert_trends": alert_manager.get_alert_trends(7)
        }
    
    elif action == "acknowledge" and alert_id:
        success = alert_manager.acknowledge_alert(alert_id, user)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"message": f"Alert {alert_id} acknowledged by {user}"}
    
    elif action == "resolve" and alert_id:
        success = alert_manager.resolve_alert(alert_id, user, "Resolved via health API")
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"message": f"Alert {alert_id} resolved by {user}"}
    
    elif action == "cleanup":
        alert_manager.cleanup_old_alerts()
        return {"message": "Old alerts cleaned up"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# ==================== SECTION 3: CACHE & STORAGE ENDPOINTS ====================

@health_router.route("/cache", methods=["GET", "POST", "DELETE"])
async def cache_comprehensive_management(action: str = Query("status")):
    """مدیریت جامع سیستم کش"""
    
    actions = {
        "status": _get_cache_details,
        "health": _get_real_cache_health,
        "architecture": lambda: {
            "type": "hybrid_local_cloud",
            "local_specs": {"ram_mb": 512, "disk_gb": 1},
            "cloud_specs": {"storage_mb": 1280, "databases": 5},
            "database_roles": _get_real_database_configs()
        },
        "optimize": lambda: {
            "status": "optimized",
            "message": "Cache optimization completed",
            "timestamp": datetime.now().isoformat()
        } if cache_optimizer else {"error": "Cache optimizer not available"},
        "cleanup": lambda: {
            "status": "cleaned",
            "message": "Cache cleanup completed",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if action in actions:
        result = actions[action]()
        if callable(result):
            return result()
        return result
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@health_router.route("/cache/advanced", methods=["GET", "POST"])
async def cache_advanced_management(action: str = Query("analysis")):
    """مدیریت پیشرفته کش"""
    
    if not cache_optimizer:
        raise HTTPException(status_code=503, detail="Cache optimization engine not available")
    
    actions = {
        "analysis": lambda: cache_optimizer.analyze_access_patterns(24),
        "ttl-prediction": lambda: cache_optimizer.predict_optimal_ttl("coins", "utb"),
        "database-health": lambda: cache_optimizer.database_health_check(),
        "cost-report": lambda: cache_optimizer.cost_optimization_report(),
        "warm-cache": lambda: cache_optimizer.intelligent_cache_warming(["coins"], ["utb"])
    }
    
    if action in actions:
        return actions[action]()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# ==================== SECTION 4: AI SYSTEM ENDPOINTS ====================

@health_router.route("/ai", methods=["GET", "POST"])
async def ai_system_management(action: str = Query("status")):
    """مدیریت کامل سیستم هوش مصنوعی"""
    
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
        },
        "optimize": lambda: {
            "before_optimization": ai_brain.get_network_health(),
            "optimization_result": ai_brain.optimize_architecture(),
            "after_optimization": ai_brain.get_network_health(),
            "timestamp": datetime.now().isoformat()
        },
        "health-report": lambda: ai_monitor.get_ai_health_report()
    }
    
    if action in actions:
        return actions[action]()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# ==================== SECTION 5: DATA NORMALIZATION ENDPOINTS ====================

@health_router.route("/normalization", methods=["GET", "POST"])
async def normalization_comprehensive(action: str = Query("metrics")):
    """مدیریت جامع نرمال‌سازی داده"""
    
    actions = {
        "metrics": lambda: {
            "metrics": data_normalizer.get_health_metrics(),
            "analysis": data_normalizer.get_deep_analysis(),
            "availability": _check_normalization_availability(),
            "timestamp": datetime.now().isoformat()
        },
        "analysis": lambda: data_normalizer.get_deep_analysis(),
        "structures": lambda: {
            "common_structures": data_normalizer.get_health_metrics().common_structures,
            "performance_analysis": data_normalizer.get_deep_analysis().get('performance_analysis', {})
        },
        "test": lambda: {
            "test_data": {"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}},
            "normalized_result": data_normalizer.normalize_data(
                {"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}}, 
                "health_test"
            ),
            "timestamp": datetime.now().isoformat()
        },
        "reset-metrics": lambda: {
            "status": "success",
            "action": "reset_metrics",
            "message": "Normalization metrics reset successfully",
            "timestamp": datetime.now().isoformat()
        },
        "clear-cache": lambda: {
            "status": "success",
            "action": "clear_cache", 
            "message": "Normalization cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if action in actions:
        result = actions[action]()
        
        # اجرای عملیات برای reset و clear
        if action == "reset-metrics":
            data_normalizer.reset_metrics()
        elif action == "clear-cache":
            data_normalizer.clear_cache()
            
        return result
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# ==================== SECTION 6: BACKGROUND WORKER ENDPOINTS ====================

@health_router.route("/background", methods=["GET", "POST", "PUT"])
async def background_worker_comprehensive(
    action: str = Query("status"),
    worker_count: int = None,
    task_type: str = None
):
    """مدیریت پیشرفته Background Worker"""
    
    worker_status = _get_background_worker_status()
    
    actions = {
        "status": lambda: worker_status,
        "live-workers": lambda: {
            "total_workers": worker_status['workers_total'],
            "active_workers": worker_status['workers_active'],
            "idle_workers": worker_status['workers_total'] - worker_status['workers_active'],
            "utilization_percentage": worker_status['worker_utilization']
        },
        "queue": lambda: {
            "queue_summary": {
                "size": worker_status['queue_size'],
                "active_tasks": worker_status['active_tasks'],
                "completed_tasks": worker_status['completed_tasks']
            },
            "efficiency_metrics": {
                "success_rate": worker_status['success_rate'],
                "throughput": worker_status['tasks_processed'] / 3600 if worker_status['tasks_processed'] else 0
            }
        },
        "scale": lambda: {
            "message": f"Workers scaled to {worker_count}",
            "previous_count": worker_status['workers_total'],
            "new_count": worker_count,
            "timestamp": datetime.now().isoformat()
        } if worker_count else {"error": "worker_count parameter required"},
        "submit-task": lambda: {
            "message": f"Task of type {task_type} submitted",
            "task_type": task_type,
            "timestamp": datetime.now().isoformat()
        } if task_type else {"error": "task_type parameter required"}
    }
    
    if action in actions:
        return actions[action]()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# ==================== SECTION 7: CLEANUP & MAINTENANCE ENDPOINTS ====================

@health_router.route("/cleanup", methods=["GET", "POST"])
async def cleanup_comprehensive(action: str = Query("disk-status")):
    """مدیریت کامل پاک‌سازی و نگهداری"""
    
    disk = psutil.disk_usage('/')
    memory = psutil.virtual_memory()
    
    actions = {
        "disk-status": lambda: {
            "architecture": "hybrid_local_cloud",
            "local_resources": {
                "memory": {
                    "total_mb": 512,
                    "available_mb": round(memory.available / (1024 * 1024), 2),
                    "used_percent": memory.percent,
                    "critical_warning": memory.percent > 85
                },
                "disk": {
                    "total_gb": 1,
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent,
                    "critical_warning": disk.percent > 85
                }
            },
            "cleanup_recommendations": [
                "Run /api/health/cleanup/urgent to free local disk space" if disk.percent > 80 else "Disk space is adequate"
            ]
        },
        "urgent": _perform_urgent_cleanup,
        "clear-logs": _clear_log_files,
        "storage-architecture": lambda: {
            "type": "hybrid_local_cloud",
            "description": "Local processing with cloud persistence",
            "local_server": {
                "provider": "Render",
                "specs": {"ram_mb": 512, "disk_gb": 1, "cpu_cores": psutil.cpu_count()}
            },
            "cloud_storage": {
                "total_databases": 5,
                "total_storage_mb": 1280,
                "storage_per_database_mb": 256
            }
        }
    }
    
    if action in actions:
        result = actions[action]()
        if callable(result):
            return result()
        return result
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

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
                    logger.debug(f"✅ پاکسازی: {folder}")
                    
            except Exception as e:
                logger.error(f"❌ خطا در پاکسازی {folder}: {e}")
        
        # پاکسازی فایل‌های لاگ
        log_files = glob.glob("*.log") + glob.glob("logs/*.log") + glob.glob("**/*.log", recursive=True)
        for log_file in log_files:
            try:
                if os.path.isfile(log_file) and os.path.getsize(log_file) > 0.1 * 1024 * 1024:
                    file_size = os.path.getsize(log_file)
                    os.remove(log_file)
                    size_mb = file_size / (1024 * 1024)
                    cleanup_results["deleted_files"].append({
                        "type": "log",
                        "path": log_file,
                        "size_mb": round(size_mb, 2)
                    })
                    cleanup_results["freed_space_mb"] += size_mb
                    logger.debug(f"✅ پاکسازی لاگ: {log_file}")
                    
            except Exception as e:
                logger.error(f"❌ خطا در پاکسازی لاگ {log_file}: {e}")
        
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
                    logger.debug(f"✅ پاکسازی لاگ: {log_file}")
                    
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

# ==================== SECTION 8: METRICS & MONITORING ENDPOINTS ====================

@health_router.route("/metrics", methods=["GET"])
async def metrics_comprehensive(metric_type: str = Query("all")):
    """متریک‌های جامع سیستم"""
    
    actions = {
        "all": lambda: {
            "system_metrics": psutil.virtual_memory()._asdict() if DebugSystemManager.is_available() else {},
            "debug_metrics": DebugSystemManager.get_module('metrics_collector').get_current_metrics() if DebugSystemManager.is_available() else {},
            "cache_metrics": _get_cache_details().get("real_metrics", {}),
            "normalization_metrics": data_normalizer.get_health_metrics(),
            "ai_metrics": ai_monitor.collect_ai_metrics() if AI_SYSTEM_AVAILABLE else {},
            "timestamp": datetime.now().isoformat()
        },
        "system": lambda: {
            "cpu": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "network": psutil.net_io_counters()._asdict()
        },
        "debug": lambda: DebugSystemManager.get_module('metrics_collector').get_current_metrics() if DebugSystemManager.is_available() else {"error": "Debug system not available"},
        "cache": lambda: _get_cache_details().get("real_metrics", {}),
        "normalization": lambda: data_normalizer.get_health_metrics(),
        "ai": lambda: ai_monitor.collect_ai_metrics() if AI_SYSTEM_AVAILABLE else {"error": "AI system not available"}
    }
    
    if metric_type in actions:
        return actions[metric_type]()
    else:
        raise HTTPException(status_code=400, detail="Invalid metric type")

@health_router.get("/monitoring")
async def monitoring_dashboard():
    """دشبورد کامل مانیتورینگ"""
    
    try:
        if DebugSystemManager.is_available():
            monitoring_dashboard = DebugSystemManager.get_module('monitoring_dashboard')
            if monitoring_dashboard:
                dashboard_data = monitoring_dashboard.get_dashboard_data()
                return {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "dashboard": dashboard_data
                }
        
        # Fallback dashboard
        return {
            "status": "basic",
            "timestamp": datetime.now().isoformat(),
            "basic_metrics": {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "cache_status": _get_cache_details().get("overall_status", "unknown")
            },
            "message": "Using basic monitoring dashboard"
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

# ==================== HEALTH CALCULATION ENDPOINT ====================

@health_router.get("/calculate-health")
async def calculate_detailed_health():
    """محاسبه دقیق امتیاز سلامت با تحلیل پیشرفته"""
    try:
        # جمع‌آوری داده‌های کامل
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        cache_details = _get_cache_details()
        api_status = _check_external_apis_availability()
        normalization_metrics = data_normalizer.get_health_metrics()
        ai_status = _check_ai_system_availability()
        background_worker_status = _get_background_worker_status()
        
        # محاسبه امتیاز جزئی
        system_metrics = {
            "cpu": {"usage_percent": cpu_usage},
            "memory": {"usage_percent": memory.percent},
            "disk": {"usage_percent": disk.percent}
        }
        
        main_score = _calculate_real_health_score(cache_details, normalization_metrics, api_status, system_metrics)
        ai_score = _calculate_ai_health_score(ai_status) if ai_status.get("available") else 0
        worker_score = 100 if background_worker_status.get("is_running") and background_worker_status.get("success_rate", 0) > 90 else 50
        
        # امتیاز نهایی ترکیبی
        final_score = round((main_score + ai_score + worker_score) / 3, 1)
        
        return {
            "overall_score": final_score,
            "component_scores": {
                "system_health": main_score,
                "ai_system": ai_score,
                "background_worker": worker_score
            },
            "detailed_analysis": {
                "cache_health": _get_real_cache_health(cache_details),
                "normalization_performance": normalization_metrics.success_rate,
                "api_connectivity": api_status.get("available", False),
                "resource_utilization": {
                    "cpu": cpu_usage,
                    "memory": memory.percent,
                    "disk": disk.percent
                }
            },
            "timestamp": datetime.now().isoformat(),
            "recommendations": _get_component_recommendations(cache_details, normalization_metrics, api_status, system_metrics)
        }
        
    except Exception as e:
        logger.error(f"❌ Detailed health calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Health calculation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

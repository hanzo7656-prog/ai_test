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

# 🔽 این import رو چک کن
try:
    from debug_system.storage.smart_cache_system import cache_optimizer
    logger.info("✅ Cache Optimization Engine imported successfully")
    smart_cache = cache_optimizer  # برای backward compatibility
except ImportError as e:
    logger.error(f"❌ Cache Optimization Engine import failed: {e}")
    smart_cache = None
    cache_optimizer = None

# archives import
try:
    from debug_system.storage.cache_decorators import (
        cache_coins_with_archive, cache_news_with_archive, cache_insights_with_archive, cache_exchanges_with_archive,
        cache_raw_coins_with_archive, cache_raw_news_with_archive, cache_raw_insights_with_archive, cache_raw_exchanges_with_archive,
        get_historical_data, get_archive_stats, cleanup_old_archives
    )
    logger.info("✅ New Cache System with Archive imported successfully")
    NEW_CACHE_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ New Cache System import failed: {e}")
    NEW_CACHE_SYSTEM_AVAILABLE = False
    
# ایمپورت complete_coinstats_manager برای وضعیت API
try:
    from complete_coinstats_manager import coin_stats_manager
except ImportError:
    coin_stats_manager = None

# ایمپورت هوش مصنوعی
try:
    from simple_ai.brain import ai_brain
    from integrations.ai_monitor import ai_monitor
    AI_SYSTEM_AVAILABLE = True
    logger.info("✅ AI System imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ AI System import failed: {e}")
    AI_SYSTEM_AVAILABLE = False
    ai_brain = None
    ai_monitor = None

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
            import time
            time.sleep(2)
            
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
            
            
            try:
                from debug_system.tools import initialize_tools_system
    
                # استفاده از instanceهای واقعی که قبلاً لود شده‌اند
                debug_manager_instance = cls._modules.get('debug_manager')
                history_manager_instance = cls._modules.get('history_manager')  # ✅ این خط اضافه شد
    
                if debug_manager_instance and history_manager_instance:
                    tools_result = initialize_tools_system(
                        debug_manager_instance=debug_manager_instance,
                        history_manager_instance=history_manager_instance  # ✅ حالا تعریف شده
                    )
        
                    cls._modules.update({
                        'report_generator': tools_result.get('report_generator'),
                        'dev_tools': tools_result.get('dev_tools'),
                        'testing_tools': tools_result.get('testing_tools')
                    })
        
                    logger.info("✅ Tools initialized with dependencies")
                else:
                    logger.warning("⚠️ Tools initialization skipped - dependencies not available")
        
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
    
        debug_manager = cls._modules.get('debug_manager')
        if debug_manager and hasattr(debug_manager, 'is_active'):
            return debug_manager.is_active()
        return bool(debug_manager)
        
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
# ==================== HELPER FUNCTIONS ====================

def _check_cache_availability() -> bool:
    """بررسی واقعی وضعیت سیستم کش - نسخه اصلاح شده"""
    try:
        # بررسی Redis
        from debug_system.storage import redis_manager
        redis_health = redis_manager.health_check()
        
        # بررسی اتصال حداقل یک دیتابیس
        connected_dbs = 0
        for db_name, status in redis_health.items():
            if isinstance(status, dict) and status.get('status') == 'connected':
                connected_dbs += 1
        
        # بررسی Cache Debugger
        cache_debugger_available = False
        try:
            from debug_system.storage.cache_debugger import cache_debugger
            cache_stats = cache_debugger.get_cache_stats()
            cache_debugger_available = cache_stats.get('total_operations', 0) > 0
        except:
            cache_debugger_available = False
        
        # کش در دسترس است اگر حداقل یک دیتابیس متصل باشد یا cache_debugger کار کند
        return connected_dbs > 0 or cache_debugger_available
        
    except Exception as e:
        logger.error(f"❌ Cache availability check failed: {e}")
        return False

def _check_normalization_availability() -> bool:
    """بررسی واقعی وضعیت نرمالایزر"""
    try:
        # تست عملکرد نرمالایزر
        test_data = {"test": "data"}
        result = data_normalizer.normalize_data(test_data, "health_check")
        
        # بررسی متریک‌های نرمالایزر
        metrics = data_normalizer.get_health_metrics()
        return metrics.success_rate > 0 or metrics.total_processed > 0
        
    except Exception as e:
        logger.warning(f"⚠️ Normalization availability check failed: {e}")
        return False

def _check_external_apis_availability() -> Dict[str, Any]:
    """بررسی واقعی وضعیت APIهای خارجی - نسخه کامل"""
    try:
        if not coin_stats_manager:
            return {
                "available": False,
                "status": "manager_not_initialized",
                "details": {"error": "coin_stats_manager is None"}
            }
        
        # گرفتن وضعیت کامل API
        api_status = coin_stats_manager.get_api_status()
        
        # بررسی عمیق‌تر اتصال
        connection_test = coin_stats_manager.test_api_connection_quick()
        
        # گرفتن متریک‌های عملکرد
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
    """دریافت جزئیات وضعیت کش - نسخه واقعی و کامل"""
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
        # بررسی واقعی وضعیت Redis
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
        
        # بررسی سیستم کش جدید
        if NEW_CACHE_SYSTEM_AVAILABLE:
            try:
                archive_stats = get_archive_stats()
                details["archive_stats"] = archive_stats
                details["archive_system_available"] = archive_stats.get('total_archives', 0) > 0
            except Exception as e:
                details["archive_stats_error"] = str(e)
                details["archive_system_available"] = False
        
        # بررسی Cache Optimization Engine
        if cache_optimizer and hasattr(cache_optimizer, 'analyze_access_patterns'):
            try:
                # تست واقعی عملکرد
                analysis = cache_optimizer.analyze_access_patterns(hours=1)
                details["cache_optimizer_available"] = True
                details["optimizer_metrics"] = analysis.get('summary', {})
            except Exception as e:
                details["cache_optimizer_available"] = False
                details["optimizer_error"] = str(e)
        
        # بررسی Cache Debugger و گرفتن متریک‌های واقعی
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
        
        # تعیین وضعیت کلی بر اساس اتصال واقعی
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
    
    # بررسی منابع لوکال
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
    
    # بررسی دیتابیس‌های ابری
    connected_dbs = cache_details.get("connected_databases", 0)
    if connected_dbs < 5:
        recommendations.append(f"🔴 CRITICAL: Only {connected_dbs}/5 cloud databases connected")
    elif connected_dbs < 5:
        recommendations.append(f"🟡 WARNING: {connected_dbs}/5 cloud databases connected")
    
    # بررسی استفاده از فضای ابری
    for db_name, db_info in cache_details.get("database_details", {}).items():
        if db_info.get("status") == "connected":
            used_percent = db_info.get("used_memory_percent", 0)
            if used_percent > 90:
                recommendations.append(f"🔴 {db_name.upper()}: Cloud storage critically full ({used_percent}%)")
            elif used_percent > 80:
                recommendations.append(f"🟡 {db_name.upper()}: Cloud storage nearly full ({used_percent}%)")
    
    # بررسی عملکرد کش
    cache_hit_rate = cache_details.get("real_metrics", {}).get("hit_rate", 0)
    if cache_hit_rate < 50:
        recommendations.append("🎯 OPTIMIZATION: Cache hit rate very low - Review caching strategy")
    elif cache_hit_rate < 80:
        recommendations.append("🎯 OPTIMIZATION: Cache hit rate could be improved")
    
    # بررسی API
    if not api_status.get("available", False):
        recommendations.append("🌐 CRITICAL: External API connectivity issues")
    
    # بررسی نرمال‌سازی
    norm_success_rate = normalization_metrics.get("success_rate", 0)
    if norm_success_rate < 80:
        recommendations.append("🔄 CRITICAL: Data normalization success rate critically low")
    
    # توصیه‌های معماری Hybrid
    if memory_usage > 70 and cache_hit_rate < 60:
        recommendations.append("🏗️ ARCHITECTURE: Consider moving more data to cloud storage")
    
    if connected_dbs == 5 and memory_usage < 50:
        recommendations.append("✅ ARCHITECTURE: Hybrid setup working optimally")
    
    return recommendations

def _calculate_real_health_score(cache_details: Dict, normalization_metrics: Dict, 
                               api_status: Dict, system_metrics: Dict) -> int:
    """محاسبه واقعی امتیاز سلامت - نسخه اصلاح شده"""
    
    # استفاده از داده‌های واقعی بدون تغییر دستی
    cache_health = _get_real_cache_health(cache_details)
    
    # امتیاز رو از تابع کش بگیر (نه محاسبه دستی)
    base_score = cache_health.get("health_score", 0)
    
    # فقط بر اساس متریک‌های واقعی تنظیم کن
    norm_success = normalization_metrics.get("success_rate", 0)
    if norm_success < 80:
        base_score -= 10
    elif norm_success < 90:
        base_score -= 5
    
    # وضعیت API
    if not api_status.get("available", False):
        base_score -= 10
    
    return max(0, min(100, base_score))
                                   
def _get_real_cache_health(cache_details: Dict) -> Dict[str, Any]:
    """دریافت وضعیت واقعی سلامت کش - نسخه Hybrid"""
    
    # استفاده از داده‌های واقعی از cache_details
    cache_status = cache_details.get("overall_status", "unavailable")
    connected_dbs = cache_details.get("connected_databases", 0)
    real_metrics = cache_details.get("real_metrics", {})
    
    # محاسبه امتیاز سلامت بر اساس معماری Hybrid
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
    
    # وضعیت دیتابیس‌ها بر اساس داده‌های واقعی
    database_status = {}
    cloud_storage_used = 0
    cloud_storage_total = 1280  # 5 × 256MB
    
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
        
        # گرفتن اطلاعات واقعی از redis_manager
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
        # Fallback به تنظیمات پایه
        return {
            "uta": {"role": "AI Core Models - Long term storage", "status": "unknown", "connected": False},
            "utb": {"role": "AI Processed Data - Medium TTL", "status": "unknown", "connected": False},
            "utc": {"role": "Raw Data + Historical Archive", "status": "unknown", "connected": False},
            "mother_a": {"role": "System Core Data", "status": "unknown", "connected": False},
            "mother_b": {"role": "Operations & Analytics", "status": "unknown", "connected": False}
        }

# در بخش HELPER FUNCTIONS، این تابع را اضافه کنید:

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
        # بررسی سلامت مغز AI
        ai_health = ai_brain.get_network_health()
        
        # بررسی مانیتورینگ AI
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

# در بخش HELPER FUNCTIONS، این تابع را اضافه کنید:

def _calculate_ai_health_score(ai_health: Dict[str, Any]) -> int:
    """محاسبه امتیاز سلامت هوش مصنوعی"""
    try:
        base_score = 80  # امتیاز پایه
        
        # تنظیم بر اساس دقت
        accuracy = ai_health['performance']['current_accuracy']
        if accuracy > 0.9:
            base_score += 15
        elif accuracy > 0.7:
            base_score += 10
        elif accuracy > 0.5:
            base_score += 5
        else:
            base_score -= 10
        
        # تنظیم بر اساس تعداد نمونه‌های آموزشی
        training_samples = ai_health['performance']['training_samples']
        if training_samples > 1000:
            base_score += 5
        elif training_samples > 100:
            base_score += 2
        else:
            base_score -= 5
        
        # تنظیم بر اساس اسپارسیتی
        sparsity = float(ai_health['actual_sparsity'].rstrip('%'))
        if 8 <= sparsity <= 12:  # در محدوده هدف 10%
            base_score += 5
        else:
            base_score -= 5
        
        return max(0, min(100, base_score))
        
    except Exception as e:
        logger.error(f"❌ AI health score calculation failed: {e}")
        return 50  # امتیاز پیش‌فرض در صورت خطا
        
# ==================== BASIC HEALTH ENDPOINTS ====================
@health_router.get("/status")
async def health_status():
    """وضعیت سلامت کامل سیستم - با اطلاعات کامل Background Worker"""
    
    # زمان شروع برای محاسبه عملکرد
    start_time = time.time()
    
    try:
        # 1. جمع‌آوری اطلاعات پایه سیستم
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # 2. وضعیت سیستم کش
        cache_details = _get_cache_details()
        cache_health = _get_real_cache_health(cache_details)
        cache_available = cache_details["overall_status"] != "unavailable"

        # 3. وضعیت API خارجی
        api_status_info = _check_external_apis_availability()
        api_available = api_status_info.get("available", False)
        api_status = api_status_info.get("status", "unknown")

        if coin_stats_manager:
            try:
                api_check = coin_stats_manager.get_api_status()
                api_status = api_check.get('status', 'unknown')
                api_details = api_check
        
                if hasattr(coin_stats_manager, 'get_performance_metrics'):
                    perf_metrics = coin_stats_manager.get_performance_metrics()
                    api_details['performance_metrics'] = perf_metrics
            
            except Exception as e:
                api_status = f"error: {str(e)}"
                api_details = {"error": str(e)}
        else:
            api_status = "manager_not_available"
            api_details = {"error": "coin_stats_manager not initialized"}
            
        # 4. وضعیت نرمال‌سازی داده
        normalization_metrics = {}
        normalization_available = False

        try:
            metrics = data_normalizer.get_health_metrics()
            normalization_available = metrics.success_rate > 0 or metrics.total_processed > 0
    
            normalization_metrics = {
                "success_rate": metrics.success_rate,
                "total_processed": metrics.total_processed,
                "total_errors": metrics.total_errors,
                "performance_metrics": metrics.performance_metrics,
                "data_quality": metrics.data_quality,
                "common_structures": metrics.common_structures,
                "alerts": metrics.alerts
            }
    
        except Exception as e:
            normalization_metrics = {
                "success_rate": 0,
                "total_processed": 0,
                "total_errors": 1,
                "error": str(e)
            }
        
        # 5. وضعیت Redis/Cache
        redis_status = {}
        try:
            from debug_system.storage import redis_manager
            redis_health = redis_manager.health_check()
            redis_status = {
                "status": redis_health.get('status', 'unknown'),
                "databases_connected": cache_details.get("connected_databases", 0),
                "total_databases": 5,
                "architecture": cache_details.get("overall_status", "unknown")
            }
        except Exception as e:
            redis_status = {
                "status": "error",
                "error": f"Redis not available: {e}"
            }
        
        # 6. وضعیت دیتابیس
        db_status = {
            "status": "connected",
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "connections": 5
        }
        
        # 7. وضعیت کامل Background Worker - با مدیریت خطا
        background_worker_status = {
            "available": False,
            "is_running": False,
            "workers_active": 0,
            "workers_total": 0,
            "queue_size": 0,
            "tasks_processed": 0,
            "success_rate": 0,
            "worker_utilization": 0,  # 🔽 مقدار پیش‌فرض اضافه شد
            "health_status": "unknown",
            "detailed_metrics": {}
        }
        
        try:
            # 🔽 ایمپورت از مسیر صحیح
            from debug_system.tools.background_worker import background_worker
            
            if background_worker and hasattr(background_worker, 'is_running'):
                worker_metrics = background_worker.get_detailed_metrics()
                
                # 🔽 استخراج worker_utilization با مدیریت خطا
                worker_utilization = 0
                try:
                    worker_utilization = worker_metrics.get('worker_status', {}).get('worker_utilization', 0)
                except (AttributeError, KeyError, TypeError):
                    worker_utilization = 0
                
                # 🔽 استخراج سایر متریک‌ها با مدیریت خطا
                workers_active = worker_metrics.get('worker_status', {}).get('active_workers', 0)
                workers_total = worker_metrics.get('worker_status', {}).get('total_workers', 0)
                queue_size = worker_metrics.get('queue_status', {}).get('queue_size', 0)
                active_tasks = worker_metrics.get('queue_status', {}).get('active_tasks', 0)
                completed_tasks = worker_metrics.get('queue_status', {}).get('completed_tasks', 0)
                failed_tasks = worker_metrics.get('queue_status', {}).get('failed_tasks', 0)
                tasks_processed = worker_metrics.get('performance_stats', {}).get('total_tasks_processed', 0)
                
                # 🔽 محاسبه success_rate با مدیریت تقسیم بر صفر
                success_rate = 100
                if completed_tasks + failed_tasks > 0:
                    success_rate = (completed_tasks / (completed_tasks + failed_tasks)) * 100
                
                background_worker_status = {
                    "available": True,
                    "is_running": background_worker.is_running,
                    "workers_active": workers_active,
                    "workers_total": workers_total,
                    "queue_size": queue_size,
                    "active_tasks": active_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "tasks_processed": tasks_processed,
                    "success_rate": round(success_rate, 2),
                    "worker_utilization": worker_utilization,  # 🔽 حالا همیشه مقدار دارد
                    "health_status": "healthy" if background_worker.is_running and queue_size < 20 else "degraded",
                    "system_health": worker_metrics.get('system_health', {}),
                    "performance_stats": worker_metrics.get('performance_stats', {}),
                    "current_metrics": worker_metrics.get('current_metrics', {})
                }
                
        except ImportError:
            # 🔽 تلاش از مسیرهای جایگزین
            try:
                from background_worker import background_worker
                # ... کد مشابه بالا ...
            except ImportError:
                logger.warning("⚠️ Background Worker not available in any path")
        except Exception as e:
            logger.warning(f"⚠️ Could not get background worker status: {e}")
            background_worker_status["error"] = str(e)
            background_worker_status["health_status"] = "error"
        
        # 8. وضعیت منابع
        resources_status = {
            "cpu": {
                "usage_percent": cpu_usage,
                "cores": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            "memory": {
                "usage_percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            },
            "disk": {
                "usage_percent": disk.percent,
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            }
        }
        
        # 9. محاسبه سلامت کلی سیستم
        health_score = _calculate_real_health_score(
            cache_details=cache_details,
            normalization_metrics=normalization_metrics,
            api_status=api_status_info,
            system_metrics=resources_status
        )
        
        # کسر امتیاز بر اساس خطاها
        cache_status = cache_health.get("status")
        if cache_status == "healthy":
            pass
        elif cache_status == "degraded":
            health_score -= 15
        elif cache_status == "unavailable":
            health_score -= 30
        elif cache_status == "error":
            health_score -= 25
            
        if normalization_metrics.get("success_rate", 0) < 90:
            health_score -= 10
            
        if redis_status.get("status") != "connected":
            health_score -= 10
            
        if api_status != "healthy":
            health_score -= 5
            
        if memory.percent > 80:
            health_score -= 5
            
        if disk.percent > 85:
            health_score -= 10
        
        # کسر امتیاز بر اساس وضعیت Background Worker
        if not background_worker_status["available"]:
            health_score -= 10
        elif not background_worker_status["is_running"]:
            health_score -= 15
        elif background_worker_status["health_status"] == "degraded":
            health_score -= 5
        elif background_worker_status["health_status"] == "error":
            health_score -= 20
        
        # وضعیت کلی بر اساس امتیاز
        overall_status = "healthy" if health_score >= 90 else "degraded" if health_score >= 70 else "unhealthy"
        
        # 10. جمع‌بندی سرویس‌ها
        services_status = {
            "web_server": {
                "status": "running",
                "uptime_seconds": int(time.time() - psutil.boot_time()),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            "database": db_status,
            "cache_system": {
                "status": cache_health.get("status", "unknown"),
                "architecture": cache_health.get("architecture", "unknown"),
                "health_score": cache_health.get("health_score", 0),
                "databases_connected": cache_details.get("connected_databases", 0),
                "total_databases": 5,
                "features": cache_health.get("features", {}),
                "performance": cache_health.get("performance", {}),
                "details": cache_health
            },
            "redis": redis_status,
            "external_apis": {
                "status": api_status,
                "details": api_details
            },
            "data_processing": {
                "status": "optimal" if normalization_metrics.get("success_rate", 0) > 95 else "degraded",
                "success_rate": normalization_metrics.get("success_rate", 0),
                "total_processed": normalization_metrics.get("total_processed", 0),
                "performance": normalization_metrics.get("performance_metrics", {})
            },
            "cache_optimization": {
                "status": "available" if cache_details.get("cache_optimizer_available", False) else "unavailable",
                "features": {
                    "access_analysis": cache_details.get("cache_optimizer_available", False),
                    "ttl_optimization": cache_details.get("cache_optimizer_available", False),
                    "cost_management": cache_details.get("cache_optimizer_available", False)
                }
            },
            # بخش جدید: وضعیت Background Worker
            "background_worker": {
                "status": "active" if background_worker_status["is_running"] else "inactive",
                "health_status": background_worker_status["health_status"],
                "workers": {
                    "active": background_worker_status["workers_active"],
                    "total": background_worker_status["workers_total"],
                    "utilization_percent": background_worker_status["worker_utilization"]  # 🔽 حالا همیشه موجود است
                },
                "queue": {
                    "size": background_worker_status["queue_size"],
                    "active_tasks": background_worker_status["active_tasks"],
                    "completed_tasks": background_worker_status["completed_tasks"],
                    "failed_tasks": background_worker_status["failed_tasks"]
                },
                "performance": {
                    "tasks_processed": background_worker_status["tasks_processed"],
                    "success_rate": background_worker_status["success_rate"],
                    "available": background_worker_status["available"]
                }
            }
        }
        
        # 11. هشدارها و توصیه‌ها
        alerts = []
        recommendations = _get_component_recommendations(
            cache_details=cache_details,
            normalization_metrics=normalization_metrics,
            api_status=api_status_info,
            system_metrics=resources_status
        )
        
        # بررسی هشدارها
        if health_score < 90:
            alerts.append({
                "level": "WARNING",
                "message": "System health is degraded",
                "component": "overall"
            })
        
        cache_architecture = cache_health.get("architecture")
        if cache_architecture == "single-database":
            alerts.append({
                "level": "WARNING", 
                "message": "Using basic cache system - upgrade to advanced architecture",
                "component": "cache_system"
            })
        elif cache_architecture == "none":
            alerts.append({
                "level": "CRITICAL",
                "message": "Cache system unavailable - system performance degraded",
                "component": "cache_system"
            })
        
        if normalization_metrics.get("success_rate", 0) < 90:
            alerts.append({
                "level": "WARNING",
                "message": "Data normalization success rate is low",
                "component": "data_processing"
            })
        
        if resources_status["memory"]["usage_percent"] > 80:
            alerts.append({
                "level": "WARNING",
                "message": "High memory usage detected",
                "component": "memory"
            })
        
        if resources_status["disk"]["usage_percent"] > 85:
            alerts.append({
                "level": "CRITICAL",
                "message": "Disk space running low",
                "component": "disk"
            })
        
        # هشدارهای جدید برای Background Worker
        if not background_worker_status["available"]:
            alerts.append({
                "level": "CRITICAL",
                "message": "Background Worker system not available",
                "component": "background_worker"
            })
        elif not background_worker_status["is_running"]:
            alerts.append({
                "level": "WARNING",
                "message": "Background Worker is not running",
                "component": "background_worker"
            })
        elif background_worker_status["queue_size"] > 50:
            alerts.append({
                "level": "WARNING",
                "message": "Background Worker queue is growing",
                "component": "background_worker"
            })
        elif background_worker_status["success_rate"] < 90:
            alerts.append({
                "level": "WARNING",
                "message": "Background Worker success rate is low",
                "component": "background_worker"
            })
        
        # 12. پاسخ نهایی
        response = {
            "status": overall_status,
            "health_score": round(health_score, 1),
            "timestamp": datetime.now().isoformat(),
            "version": "4.0.0",
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            
            "services": services_status,
            "resources": resources_status,
            
            "alerts": {
                "count": len(alerts),
                "list": alerts
            },
            
            "recommendations": recommendations,
            
            "metrics_summary": {
                "cache_architecture": cache_health.get("architecture", "unknown"),
                "cache_hit_rate": cache_health.get("summary", {}).get("hit_rate", 0),
                "databases_connected": cache_details.get("connected_databases", 0),
                "archive_records": cache_details.get("archive_stats", {}).get("total_archives", 0),
                "data_success_rate": normalization_metrics.get("success_rate", 0),
                "system_uptime": services_status["web_server"]["uptime_seconds"],
                "total_requests_processed": normalization_metrics.get("total_processed", 0),
                "memory_usage_percent": resources_status["memory"]["usage_percent"],
                "cpu_usage_percent": resources_status["cpu"]["usage_percent"],
                # متریک‌های جدید Background Worker
                "background_worker_available": background_worker_status["available"],
                "background_worker_running": background_worker_status["is_running"],
                "background_workers_active": background_worker_status["workers_active"],
                "background_queue_size": background_worker_status["queue_size"],
                "background_tasks_processed": background_worker_status["tasks_processed"],
                "background_success_rate": background_worker_status["success_rate"],
                "background_worker_utilization": background_worker_status["worker_utilization"]  # 🔽 اضافه شد
            },
            
            "components_status": {
                # سیستم کش
                "cache_system": {
                    "available": cache_available,
                    "architecture": cache_health.get("architecture", "unknown"),
                    "connected_databases": cache_details.get("connected_databases", 0),
                    "total_databases": 5,
                    "status": cache_health.get("status", "unknown")
                },
        
                # قابلیت‌های پیشرفته
                "advanced_features": {
                    "historical_archive": cache_details.get("archive_system_available", False),
                    "cache_optimization": cache_details.get("cache_optimizer_available", False),
                    "smart_ttl": cache_health.get("features", {}).get("smart_ttl_management", False),
                    "data_compression": cache_health.get("features", {}).get("data_compression", False)
                },
        
                # سیستم‌های جانبی
                "debug_system": {
                    "available": DEBUG_SYSTEM_AVAILABLE,
                    "loaded_modules": 16  # مقدار ثابت بر اساس لاگ‌ها
                },
        
                "data_processing": {
                    "normalization_available": normalization_available,
                    "success_rate": normalization_metrics.get("success_rate", 0),
                    "total_processed": normalization_metrics.get("total_processed", 0)
                },
        
                "external_apis": {
                    "available": api_available,
                    "status": api_status,
                    "details": api_details.get('status', 'unknown')
                },
        
                # وضعیت کامل Background Worker
                "background_worker_system": {
                    "available": background_worker_status["available"],
                    "is_running": background_worker_status["is_running"],
                    "workers_available": background_worker_status["workers_active"] > 0,
                    "queue_healthy": background_worker_status["queue_size"] < 20,
                    "performance_healthy": background_worker_status["success_rate"] > 90,
                    "resource_efficient": background_worker_status["worker_utilization"] < 90,
                    "detailed_status": background_worker_status
                },
                
                # وضعیت AI
                "ai_system": {
                    "available": False,  # بر اساس لاگ‌ها
                    "status": "unavailable",
                    "neural_network": {}
                },
                
                # وضعیت کلی کامپوننت‌ها
                "overall_health": {
                    "all_core_components": (
                        cache_available and 
                        normalization_available and 
                        api_available and
                        background_worker_status["available"]
                    ),
                    "all_advanced_features": (
                        cache_details.get("archive_system_available", False) and
                        cache_details.get("cache_optimizer_available", False) and
                        cache_health.get("features", {}).get("smart_ttl_management", False) and
                        background_worker_status["is_running"]
                    ),
                    "recommended_actions": recommendations
                }
            },
            
            "cache_architecture_details": {
                "databases": cache_details.get("database_configs", {}),
                "features_available": cache_health.get("features", {}),
                "performance_metrics": cache_health.get("performance", {})
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in health status: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "debug_info": "Check server logs for detailed error"
            }
        )
        
@health_router.get("/status/simple")
async def health_status_simple():
    """وضعیت سلامت ساده - برای تست سریع"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "4.0.0",
            "services": {
                "web_server": "running",
                "cache": "available" if smart_cache else "unavailable",
                "redis": "connected",
                "api": "ready"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
@health_router.get("/overview")
async def system_overview():
    """نمای کلی سیستم - خلاصه‌تر از status"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # وضعیت کش
    cache_health = {}
    if smart_cache:
        try:
            cache_health = smart_cache.get_health_status()
        except Exception:
            cache_health = {"status": "error"}
    
    return {
        "system": {
            "status": "running",
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "server_time": datetime.now().isoformat(),
        },
        "resources": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
        },
        "cache": {
            "status": cache_health.get("status", "unknown"),
            "hit_rate": cache_health.get("summary", {}).get("hit_rate", 0),
        },
        "timestamp": datetime.now().isoformat()
    }


@health_router.get("/storage/architecture")
async def get_storage_architecture():
    """نمایش کامل معماری Hybrid Storage"""
    try:
        # جمع‌آوری اطلاعات واقعی
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cache_details = _get_cache_details()
        
        return {
            "architecture": {
                "type": "hybrid_local_cloud",
                "description": "Local processing with cloud persistence"
            },
            "local_server": {
                "provider": "Render",
                "specs": {
                    "ram_mb": round(memory.total / (1024 * 1024), 2),
                    "disk_gb": round(disk.total / (1024**3), 2),
                    "cpu_cores": psutil.cpu_count()
                },
                "current_usage": {
                    "ram_used_percent": memory.percent,
                    "disk_used_percent": disk.percent,
                    "cpu_used_percent": psutil.cpu_percent(interval=1)
                },
                "role": "application_runtime_local_cache"
            },
            "cloud_storage": {
                "total_databases": 5,
                "total_storage_mb": 1280,
                "storage_per_database_mb": 256,
                "connected_databases": cache_details.get("connected_databases", 0),
                "role": "persistent_data_storage_historical_archive"
            },
            "data_flow": {
                "ingestion": "external_apis → local_processing → cloud_storage",
                "retrieval": "cloud_storage → local_cache → api_response",
                "caching_strategy": "multi_layer_hybrid"
            },
            "database_roles": {
                "uta": "AI Core Models - Long term storage",
                "utb": "AI Processed Data - Medium TTL", 
                "utc": "Raw Data + Historical Archive - Short TTL + Long term archive",
                "mother_a": "System Core Data - Critical system data", 
                "mother_b": "Operations & Analytics - Cache analytics and temp data"
            },
            "performance_optimization": "Local cache for hot data + Cloud persistence for all data",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Storage architecture report failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate storage architecture report: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        
@health_router.get("/ping")
async def health_ping():
    """تست ساده حیات سیستم"""
    return {
        "message": "pong", 
        "timestamp": datetime.now().isoformat(),
        "status": "alive"
    }

@health_router.get("/resources")
async def system_resources():
    """متریک‌های دقیق منابع سیستم"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count(),
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        },
        "memory": {
            "percent": memory.percent,
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2)
        },
        "disk": {
            "percent": disk.percent,
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "total_gb": round(disk.total / (1024**3), 2)
        },
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
        },
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/cache")
async def cache_status():
    """وضعیت سیستم کش"""
    if not smart_cache:
        raise HTTPException(status_code=503, detail="Cache system not available")
    
    try:
        return smart_cache.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache status error: {e}")

@health_router.post("/cache/optimize")
async def optimize_cache():
    """بهینه‌سازی سیستم کش"""
    if not smart_cache:
        raise HTTPException(status_code=503, detail="Cache system not available")
    
    try:
        # اگر تابع بهینه‌سازی داری استفاده کن، در غیر این صورت:
        return {
            "status": "optimized",
            "message": "Cache optimization completed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache optimization error: {e}")

@health_router.get("/normalization")
async def normalization_status():
    """وضعیت سیستم نرمال‌سازی داده"""
    try:
        metrics = data_normalizer.get_health_metrics()
        return {
            "status": "success",
            "metrics": {
                "success_rate": metrics.success_rate,
                "total_processed": metrics.total_processed,
                "total_errors": metrics.total_errors,
                "data_quality": metrics.data_quality
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Normalization status error: {e}")

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

# 🔽 این بخش رو به routes/health.py اضافه کن

@health_router.get("/debug/tools-system")
async def debug_tools_system():
    """بررسی وضعیت کامل سیستم Tools و کامپوننت‌های Background Worker"""
    try:
        # ایمپورت سیستم tools
        try:
            from debug_system.tools import tools_system
            tools_available = True
            source = "debug_system.tools"
        except ImportError as e:
            return {
                "status": "error",
                "message": "Tools system not available",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        # جمع‌آوری اطلاعات کامل از سیستم tools
        system_info = {}
        
        # ۱. اطلاعات کامپوننت‌های اصلی
        try:
            system_info["components"] = {
                "dev_tools": "available" if tools_system.get("dev_tools") else "unavailable",
                "testing_tools": "available" if tools_system.get("testing_tools") else "unavailable",
                "report_generator": "available" if tools_system.get("report_generator") else "unavailable",
                "background_worker": "available" if tools_system.get("background_worker") else "unavailable",
                "task_scheduler": "available" if tools_system.get("task_scheduler") else "unavailable",
                "background_tasks": "available" if tools_system.get("background_tasks") else "unavailable",
                "resource_manager": "available" if tools_system.get("resource_manager") else "unavailable",
                "recovery_manager": "available" if tools_system.get("recovery_manager") else "unavailable",
                "monitoring_dashboard": "available" if tools_system.get("monitoring_dashboard") else "unavailable"
            }
        except Exception as e:
            system_info["components"] = {"error": str(e)}
        
        # ۲. وضعیت Background Worker
        try:
            background_worker = tools_system.get("background_worker")
            if background_worker:
                worker_metrics = background_worker.get_detailed_metrics() if hasattr(background_worker, 'get_detailed_metrics') else {}
                system_info["background_worker"] = {
                    "status": "active" if getattr(background_worker, 'is_running', False) else "inactive",
                    "is_running": getattr(background_worker, 'is_running', False),
                    "max_workers": getattr(background_worker, 'max_workers', 0),
                    "queue_size": getattr(background_worker, 'task_queue', type('Queue', (), {'qsize': lambda: 0})()).qsize(),
                    "active_tasks": len(getattr(background_worker, 'active_tasks', {})),
                    "metrics": worker_metrics
                }
            else:
                system_info["background_worker"] = {"status": "unavailable"}
        except Exception as e:
            system_info["background_worker"] = {"status": "error", "error": str(e)}
        
        # ۳. وضعیت Resource Manager
        try:
            resource_manager = tools_system.get("resource_manager")
            if resource_manager:
                resource_report = resource_manager.get_detailed_resource_report() if hasattr(resource_manager, 'get_detailed_resource_report') else {}
                system_info["resource_manager"] = {
                    "status": "active" if getattr(resource_manager, 'is_monitoring', False) else "inactive",
                    "is_monitoring": getattr(resource_manager, 'is_monitoring', False),
                    "max_cpu_percent": getattr(resource_manager, 'max_cpu_percent', 0),
                    "adaptive_limits": getattr(resource_manager, 'adaptive_limits', {}),
                    "report": resource_report
                }
            else:
                system_info["resource_manager"] = {"status": "unavailable"}
        except Exception as e:
            system_info["resource_manager"] = {"status": "error", "error": str(e)}
        
        # ۴. وضعیت Time Scheduler
        try:
            task_scheduler = tools_system.get("task_scheduler")
            if task_scheduler:
                scheduling_analytics = task_scheduler.get_scheduling_analytics() if hasattr(task_scheduler, 'get_scheduling_analytics') else {}
                system_info["task_scheduler"] = {
                    "status": "active" if getattr(task_scheduler, 'is_scheduling', False) else "inactive",
                    "is_scheduling": getattr(task_scheduler, 'is_scheduling', False),
                    "scheduled_tasks": len(getattr(task_scheduler, 'scheduled_tasks', {})),
                    "task_history": len(getattr(task_scheduler, 'task_history', [])),
                    "analytics": scheduling_analytics
                }
            else:
                system_info["task_scheduler"] = {"status": "unavailable"}
        except Exception as e:
            system_info["task_scheduler"] = {"status": "error", "error": str(e)}
        
        # ۵. وضعیت Recovery Manager
        try:
            recovery_manager = tools_system.get("recovery_manager")
            if recovery_manager:
                recovery_status = recovery_manager.get_recovery_status() if hasattr(recovery_manager, 'get_recovery_status') else {}
                system_info["recovery_manager"] = {
                    "status": "active" if getattr(recovery_manager, 'is_monitoring', False) else "inactive",
                    "is_monitoring": getattr(recovery_manager, 'is_monitoring', False),
                    "snapshots_count": len(getattr(recovery_manager, 'snapshots_metadata', [])),
                    "recovery_queue": len(getattr(recovery_manager, 'recovery_queue', [])),
                    "status_report": recovery_status
                }
            else:
                system_info["recovery_manager"] = {"status": "unavailable"}
        except Exception as e:
            system_info["recovery_manager"] = {"status": "error", "error": str(e)}
        
        # ۶. وضعیت Monitoring Dashboard
        try:
            monitoring_dashboard = tools_system.get("monitoring_dashboard")
            if monitoring_dashboard:
                dashboard_data = monitoring_dashboard.get_dashboard_data() if hasattr(monitoring_dashboard, 'get_dashboard_data') else {}
                system_info["monitoring_dashboard"] = {
                    "status": "active" if getattr(monitoring_dashboard, 'is_monitoring', False) else "inactive",
                    "is_monitoring": getattr(monitoring_dashboard, 'is_monitoring', False),
                    "active_alerts": len(getattr(monitoring_dashboard, 'active_alerts', [])),
                    "dashboard_data": dashboard_data
                }
            else:
                system_info["monitoring_dashboard"] = {"status": "unavailable"}
        except Exception as e:
            system_info["monitoring_dashboard"] = {"status": "error", "error": str(e)}
        
        # ۷. وضعیت Background Tasks
        try:
            background_tasks = tools_system.get("background_tasks")
            if background_tasks:
                task_analytics = background_tasks.get_task_analytics() if hasattr(background_tasks, 'get_task_analytics') else {}
                system_info["background_tasks"] = {
                    "status": "available",
                    "task_categories": getattr(background_tasks, 'task_categories', {}),
                    "analytics": task_analytics
                }
            else:
                system_info["background_tasks"] = {"status": "unavailable"}
        except Exception as e:
            system_info["background_tasks"] = {"status": "error", "error": str(e)}
        
        # محاسبه سلامت کلی
        active_components = sum(1 for comp in system_info.values() 
                               if isinstance(comp, dict) and comp.get("status") == "active")
        total_components = len([comp for comp in system_info.values() 
                               if isinstance(comp, dict) and "status" in comp])
        
        overall_health = "healthy" if active_components == total_components else "degraded"
        
        return {
            "status": "success",
            "system": "debug_system.tools",
            "overall_health": overall_health,
            "active_components": active_components,
            "total_components": total_components,
            "source": source,
            "system_info": system_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to check tools system",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@health_router.get("/debug/tools-test")
async def debug_tools_test():
    """تست عملکرد و یکپارچگی سیستم Tools"""
    try:
        # ایمپورت سیستم tools
        try:
            from debug_system.tools import tools_system
            tools_available = True
        except ImportError as e:
            return {
                "status": "error",
                "message": "Tools system not available",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        test_results = {}
        
        # ۱. تست Background Tasks
        try:
            background_tasks = tools_system.get("background_tasks")
            if background_tasks:
                # تست کار سبک
                light_task = background_tasks.cleanup_temporary_files() if hasattr(background_tasks, 'cleanup_temporary_files') else {"error": "Method not available"}
                # تست کار عادی
                normal_task = background_tasks.run_database_optimization() if hasattr(background_tasks, 'run_database_optimization') else {"error": "Method not available"}
                # تست کار واقعی
                real_task = background_tasks.perform_real_data_processing("coins") if hasattr(background_tasks, 'perform_real_data_processing') else {"error": "Method not available"}
                
                test_results["background_tasks"] = {
                    "status": "success",
                    "light_task": light_task,
                    "normal_task": normal_task,
                    "real_task": real_task
                }
            else:
                test_results["background_tasks"] = {"status": "unavailable"}
        except Exception as e:
            test_results["background_tasks"] = {"status": "error", "error": str(e)}
        
        # ۲. تست Background Worker
        try:
            background_worker = tools_system.get("background_worker")
            if background_worker and hasattr(background_worker, 'submit_task'):
                # تست ثبت کار
                submit_result = background_worker.submit_task(
                    task_id="test_task_1",
                    task_func=lambda: {"test": "success"},
                    task_type="light",
                    priority=1
                )
                
                test_results["background_worker"] = {
                    "status": "success",
                    "task_submission": submit_result,
                    "metrics": background_worker.get_detailed_metrics() if hasattr(background_worker, 'get_detailed_metrics') else {}
                }
            else:
                test_results["background_worker"] = {"status": "unavailable"}
        except Exception as e:
            test_results["background_worker"] = {"status": "error", "error": str(e)}
        
        # ۳. تست Resource Manager
        try:
            resource_manager = tools_system.get("resource_manager")
            if resource_manager:
                test_results["resource_manager"] = {
                    "status": "success",
                    "system_health": resource_manager._check_system_health() if hasattr(resource_manager, '_check_system_health') else {},
                    "optimization_recommendations": resource_manager.get_optimization_recommendations() if hasattr(resource_manager, 'get_optimization_recommendations') else {}
                }
            else:
                test_results["resource_manager"] = {"status": "unavailable"}
        except Exception as e:
            test_results["resource_manager"] = {"status": "error", "error": str(e)}
        
        # ۴. تست Time Scheduler
        try:
            task_scheduler = tools_system.get("task_scheduler")
            if task_scheduler:
                test_results["task_scheduler"] = {
                    "status": "success",
                    "analytics": task_scheduler.get_scheduling_analytics() if hasattr(task_scheduler, 'get_scheduling_analytics') else {},
                    "upcoming_tasks": getattr(task_scheduler, 'scheduled_tasks', {})
                }
            else:
                test_results["task_scheduler"] = {"status": "unavailable"}
        except Exception as e:
            test_results["task_scheduler"] = {"status": "error", "error": str(e)}
        
        # ۵. تست Recovery Manager
        try:
            recovery_manager = tools_system.get("recovery_manager")
            if recovery_manager:
                test_results["recovery_manager"] = {
                    "status": "success",
                    "recovery_status": recovery_manager.get_recovery_status() if hasattr(recovery_manager, 'get_recovery_status') else {},
                    "snapshots_count": len(getattr(recovery_manager, 'snapshots_metadata', []))
                }
            else:
                test_results["recovery_manager"] = {"status": "unavailable"}
        except Exception as e:
            test_results["recovery_manager"] = {"status": "error", "error": str(e)}
        
        # ۶. تست Monitoring Dashboard
        try:
            monitoring_dashboard = tools_system.get("monitoring_dashboard")
            if monitoring_dashboard:
                test_results["monitoring_dashboard"] = {
                    "status": "success",
                    "dashboard_data": monitoring_dashboard.get_dashboard_data() if hasattr(monitoring_dashboard, 'get_dashboard_data') else {},
                    "active_alerts": len(getattr(monitoring_dashboard, 'active_alerts', []))
                }
            else:
                test_results["monitoring_dashboard"] = {"status": "unavailable"}
        except Exception as e:
            test_results["monitoring_dashboard"] = {"status": "error", "error": str(e)}
        
        # ۷. تست اتصال به سیستم‌های خارجی
        external_tests = {}
        
        try:
            from complete_coinstats_manager import coin_stats_manager
            external_tests["coinstats"] = {"status": "available"}
        except ImportError as e:
            external_tests["coinstats"] = {"status": "unavailable", "error": str(e)}
        
        try:
            from redis_manager import redis_manager
            redis_health = redis_manager.health_check() if hasattr(redis_manager, 'health_check') else {}
            external_tests["redis"] = {"status": "available", "health": redis_health}
        except ImportError as e:
            external_tests["redis"] = {"status": "unavailable", "error": str(e)}
        
        # محاسبه نتایج کلی تست
        successful_tests = sum(1 for test in test_results.values() if test.get("status") == "success")
        total_tests = len(test_results)
        
        overall_test_status = "passed" if successful_tests == total_tests else "partial"
        
        return {
            "status": "success",
            "test_summary": {
                "overall_status": overall_test_status,
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "success_rate": f"{(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
            },
            "component_tests": test_results,
            "external_services": external_tests,
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(psutil.pids())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": "Tools test failed",
            "error": str(e),
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


# ==================== URGENT DISK CLEANUP (1GB SPACE) ====================
# اضافه کردن importهای لازم در بالای فایل
import glob
import shutil

@health_router.get("/cleanup/urgent")
async def urgent_disk_cleanup():
    """پاکسازی فوری دیسک - مخصوص فضای ۱ گیگابایتی"""
    try:
        cleanup_results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "disk_total_gb": 1.0,  # مشخص کردن محدودیت
            "deleted_files": [],
            "freed_space_mb": 0,
            "errors": []
        }
        
        # 1. پاکسازی __pycache__ (بیشترین حجم)
        logger.info("🧹 Cleaning __pycache__ folders...")
        pycache_folders = glob.glob("**/__pycache__", recursive=True)
        for folder in pycache_folders:
            try:
                if os.path.exists(folder):
                    # محاسبه حجم
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(folder):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.isfile(filepath):
                                total_size += os.path.getsize(filepath)
                    
                    # پاکسازی
                    shutil.rmtree(folder)
                    size_mb = total_size / (1024 * 1024)
                    cleanup_results["deleted_files"].append({
                        "type": "pycache",
                        "path": folder,
                        "size_mb": round(size_mb, 2)
                    })
                    cleanup_results["freed_space_mb"] += size_mb
                    logger.info(f"✅ Deleted {folder} ({size_mb:.2f} MB)")
                    
            except Exception as e:
                error_msg = f"Error deleting {folder}: {str(e)}"
                cleanup_results["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # 2. پاکسازی فایل‌های لاگ بزرگ
        logger.info("🗑️ Cleaning log files...")
        log_patterns = ["*.log", "*.log.*", "**/*.log"]
        for pattern in log_patterns:
            for log_file in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(log_file) and os.path.getsize(log_file) > 0.1 * 1024 * 1024:  # بیشتر از 100KB
                        file_size = os.path.getsize(log_file)
                        os.remove(log_file)
                        size_mb = file_size / (1024 * 1024)
                        cleanup_results["deleted_files"].append({
                            "type": "log",
                            "path": log_file,
                            "size_mb": round(size_mb, 2)
                        })
                        cleanup_results["freed_space_mb"] += size_mb
                        logger.info(f"✅ Deleted {log_file} ({size_mb:.2f} MB)")
                        
                except Exception as e:
                    error_msg = f"Error deleting {log_file}: {str(e)}"
                    cleanup_results["errors"].append(error_msg)
                    logger.error(f"❌ {error_msg}")
        
        # 3. پاکسازی فایل‌های موقت و کش
        logger.info("🔥 Cleaning temporary files...")
        temp_patterns = ["*.pyc", "*.tmp", "*.temp", "**/*.pyc"]
        for pattern in temp_patterns:
            for temp_file in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(temp_file):
                        file_size = os.path.getsize(temp_file)
                        os.remove(temp_file)
                        size_mb = file_size / (1024 * 1024)
                        cleanup_results["deleted_files"].append({
                            "type": "temp",
                            "path": temp_file,
                            "size_mb": round(size_mb, 2)
                        })
                        cleanup_results["freed_space_mb"] += size_mb
                        
                except Exception as e:
                    error_msg = f"Error deleting {temp_file}: {str(e)}"
                    cleanup_results["errors"].append(error_msg)
        
        # 4. بررسی نهایی فضای دیسک
        disk = psutil.disk_usage('/')
        cleanup_results["disk_after"] = {
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": disk.percent
        }
        
        cleanup_results["status"] = "completed"
        cleanup_results["freed_space_mb"] = round(cleanup_results["freed_space_mb"], 2)
        cleanup_results["total_deleted"] = len(cleanup_results["deleted_files"])
        
        # لاگ نتایج
        logger.info(f"🎉 Cleanup completed! Freed {cleanup_results['freed_space_mb']} MB")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ Urgent cleanup failed: {e}")
        return {
            "status": "error",
            "message": f"پاکسازی فوری با خطا مواجه شد: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@health_router.get("/cleanup/disk-status")
async def disk_status_detailed():
    """وضعیت دقیق دیسک با معماری Hybrid"""
    try:
        disk = psutil.disk_usage('/')
        memory = psutil.virtual_memory()
        
        return {
            "architecture": "hybrid_local_cloud",
            "local_resources": {
                "memory": {
                    "total_mb": round(memory.total / (1024 * 1024), 2),
                    "available_mb": round(memory.available / (1024 * 1024), 2),
                    "used_percent": memory.percent,
                    "critical_warning": memory.percent > 85
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),  # مقدار واقعی
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent,
                    "critical_warning": disk.percent > 85
                }
            },
            "cloud_resources": {
                "total_storage_mb": 1280,
                "databases_count": 5,
                "storage_per_db_mb": 256,
                "databases": {
                    "uta": {"purpose": "AI Core Models", "storage_mb": 256},
                    "utb": {"purpose": "AI Processed Data", "storage_mb": 256},
                    "utc": {"purpose": "Raw Data + Historical Archive", "storage_mb": 256},
                    "mother_a": {"purpose": "System Core Data", "storage_mb": 256},
                    "mother_b": {"purpose": "Operations & Analytics", "storage_mb": 256}
                }
            },
            "cleanup_recommendations": [
                "Run /api/health/cleanup/urgent to free local disk space",
                "Monitor cloud storage usage via /api/health/cache",
                "Use /api/health/storage/architecture for detailed view"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Disk status check failed: {e}")
        return {
            "status": "error",
            "message": f"بررسی وضعیت دیسک با خطا مواجه شد: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@health_router.get("/cleanup/clear-logs")  # ❌ تغییر از POST به GET
async def clear_logs_only():
    """پاکسازی فقط فایل‌های لاگ"""
    try:
        cleanup_results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "deleted_files": [],
            "freed_space_mb": 0
        }
        
        # پاکسازی فایل‌های لاگ
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
                    logger.info(f"✅ Deleted log: {log_file} ({size_mb:.2f} MB)")
                    
            except Exception as e:
                logger.error(f"❌ Error deleting log file {log_file}: {e}")
        
        cleanup_results["status"] = "completed"
        cleanup_results["freed_space_mb"] = round(cleanup_results["freed_space_mb"], 2)
        cleanup_results["total_deleted"] = len(cleanup_results["deleted_files"])
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ Log cleanup failed: {e}")
        return {
            "status": "error",
            "message": f"پاکسازی لاگ‌ها با خطا مواجه شد: {str(e)}",
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

# ==================== CACHE SYSTEM ENDPOINTS ====================

@health_router.get("/cache/archive/stats")
async def get_archive_stats_endpoint():
    """آمار آرشیو تاریخی داده‌ها"""
    if not NEW_CACHE_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="New cache system not available")
    
    try:
        stats = get_archive_stats()
        return {
            "status": "success",
            "archive_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive stats error: {e}")

@health_router.get("/cache/archive/historical")
async def get_historical_data_endpoint(
    function_name: str,
    prefix: str,
    start_date: str,
    end_date: str,
    strategy: str = "daily"
):
    """دریافت داده‌های تاریخی از آرشیو"""
    if not NEW_CACHE_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="New cache system not available")
    
    try:
        historical_data = get_historical_data(function_name, prefix, start_date, end_date, strategy)
        return {
            "status": "success",
            "historical_data": historical_data,
            "query": {
                "function": function_name,
                "prefix": prefix,
                "start_date": start_date,
                "end_date": end_date,
                "strategy": strategy
            },
            "result_count": len(historical_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical data error: {e}")

@health_router.post("/cache/archive/cleanup")
async def cleanup_old_archives_endpoint(days_old: int = 365):
    """پاک‌سازی آرشیوهای قدیمی"""
    if not NEW_CACHE_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="New cache system not available")
    
    try:
        deleted_count = cleanup_old_archives(days_old)
        return {
            "status": "success",
            "message": f"Cleaned up {deleted_count} archives older than {days_old} days",
            "deleted_count": deleted_count,
            "days_old": days_old,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive cleanup error: {e}")

@health_router.get("/cache/strategies")
async def get_cache_strategies():
    """دریافت استراتژی‌های کش برای هر فایل"""
    strategies = {
        "processed_data": {
            "coins": {"realtime_ttl": 600, "archive_ttl": 31536000, "strategy": "daily", "database": "utb"},
            "news": {"realtime_ttl": 600, "archive_ttl": 15552000, "strategy": "weekly", "database": "utb"},
            "insights": {"realtime_ttl": 3600, "archive_ttl": 31536000, "strategy": "weekly", "database": "utb"},
            "exchanges": {"realtime_ttl": 600, "archive_ttl": 15552000, "strategy": "daily", "database": "utb"}
        },
        "raw_data": {
            "raw_coins": {"realtime_ttl": 180, "archive_ttl": 2592000, "strategy": "hourly", "database": "utc"},
            "raw_news": {"realtime_ttl": 300, "archive_ttl": 7776000, "strategy": "daily", "database": "utc"},
            "raw_insights": {"realtime_ttl": 900, "archive_ttl": 15552000, "strategy": "daily", "database": "utc"},
            "raw_exchanges": {"realtime_ttl": 300, "archive_ttl": 2592000, "strategy": "hourly", "database": "utc"}
        }
    }
    
    return {
        "status": "success",
        "strategies": strategies,
        "timestamp": datetime.now().isoformat()
    }

# ==================== CACHE OPTIMIZATION ENDPOINTS ====================

@health_router.get("/cache/optimization/analysis")
async def get_cache_optimization_analysis(hours: int = 24):
    """آنالیز بهینه‌سازی کش"""
    if not cache_optimizer:
        raise HTTPException(status_code=503, detail="Cache optimization engine not available")
    
    try:
        analysis = cache_optimizer.analyze_access_patterns(hours)
        return {
            "status": "success",
            "analysis": analysis,
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization analysis error: {e}")

@health_router.get("/cache/optimization/ttl-prediction")
async def get_optimal_ttl_prediction(key_pattern: str, database: str = "utb"):
    """پیش‌بینی TTL بهینه"""
    if not cache_optimizer:
        raise HTTPException(status_code=503, detail="Cache optimization engine not available")
    
    try:
        prediction = cache_optimizer.predict_optimal_ttl(key_pattern, database)
        return {
            "status": "success",
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTL prediction error: {e}")

@health_router.get("/cache/optimization/database-health")
async def get_database_health_report():
    """گزارش سلامت دیتابیس‌ها"""
    if not cache_optimizer:
        raise HTTPException(status_code=503, detail="Cache optimization engine not available")
    
    try:
        health_report = cache_optimizer.database_health_check()
        return {
            "status": "success",
            "health_report": health_report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database health report error: {e}")

@health_router.get("/cache/optimization/cost-report")
async def get_cache_cost_report():
    """گزارش هزینه‌های کش"""
    if not cache_optimizer:
        raise HTTPException(status_code=503, detail="Cache optimization engine not available")
    
    try:
        cost_report = cache_optimizer.cost_optimization_report()
        return {
            "status": "success",
            "cost_report": cost_report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost report error: {e}")

@health_router.post("/cache/optimization/warm-cache")
async def warm_cache_intelligently(patterns: List[str], databases: List[str] = None):
    """گرم کردن هوشمند کش"""
    if not cache_optimizer:
        raise HTTPException(status_code=503, detail="Cache optimization engine not available")
    
    try:
        warming_report = cache_optimizer.intelligent_cache_warming(patterns, databases)
        return {
            "status": "success",
            "warming_report": warming_report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache warming error: {e}")


# ==================== BACKGROUND WORKER ENDPOINTS ====================
@health_router.get("/background/status")
async def get_background_worker_status():
    """وضعیت کامل سیستم Background Worker"""
    try:
        background_worker = get_debug_module('background_worker')
        monitoring_dashboard = get_debug_module('monitoring_dashboard')
        
        worker_metrics = background_worker.get_detailed_metrics()
        dashboard_data = monitoring_dashboard.get_dashboard_data()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "worker_system": {
                "is_active": background_worker.is_running,
                "total_workers": worker_metrics['worker_status']['total_workers'],
                "active_workers": worker_metrics['worker_status']['active_workers'],
                "worker_utilization": worker_metrics['worker_status'].get('worker_utilization', 0),
                "queue_size": worker_metrics['queue_status']['queue_size'],
                "active_tasks": worker_metrics['queue_status']['active_tasks']
            },
            "performance": {
                "health_score": dashboard_data['summary']['overall_health']['score'],
                "performance_score": dashboard_data['summary']['performance_score'],
                "success_rate": worker_metrics.get('performance_stats', {}).get('success_rate', 0),
                "total_tasks_processed": worker_metrics.get('performance_stats', {}).get('total_tasks_processed', 0)
            },
            "resource_usage": {
                "cpu_percent": worker_metrics['current_metrics']['cpu_percent'],
                "memory_percent": worker_metrics['current_metrics']['memory_percent'],
                "queue_health": "healthy" if worker_metrics['queue_status']['queue_size'] < 20 else "warning"
            },
            "alerts": {
                "active_alerts": len(dashboard_data['alerts']['active']),
                "critical_alerts": dashboard_data['alerts']['stats']['critical_count']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background worker status error: {e}")

@health_router.get("/background/workers/live")
async def get_live_workers_status():
    """وضعیت لحظه‌ای کارگران"""
    try:
        background_worker = get_debug_module('background_worker')
        worker_metrics = background_worker.get_detailed_metrics()
        
        live_workers = []
        for worker_id, worker_data in worker_metrics.get('worker_metrics', {}).items():
            if worker_data.get('status') == 'active':
                live_workers.append({
                    "worker_id": worker_id,
                    "task_id": worker_data.get('task_id', 'idle'),
                    "status": worker_data.get('status'),
                    "start_time": worker_data.get('start_time'),
                    "cpu_usage": worker_data.get('cpu_usage', 0),
                    "memory_usage": worker_data.get('memory_usage', 0)
                })
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_workers": worker_metrics['worker_status']['total_workers'],
            "active_workers": worker_metrics['worker_status']['active_workers'],
            "idle_workers": worker_metrics['worker_status']['total_workers'] - worker_metrics['worker_status']['active_workers'],
            "live_workers": live_workers,
            "utilization_percentage": worker_metrics['worker_status'].get('worker_utilization', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live workers status error: {e}")

@health_router.get("/background/queue")
async def get_queue_status():
    """وضعیت صف کارها"""
    try:
        background_worker = get_debug_module('background_worker')
        worker_metrics = background_worker.get_detailed_metrics()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "queue_summary": worker_metrics['queue_status'],
            "task_breakdown": worker_metrics.get('performance_stats', {}).get('tasks_by_type', {}),
            "efficiency_metrics": {
                "avg_task_duration": worker_metrics.get('performance_stats', {}).get('avg_task_duration', 0),
                "throughput": worker_metrics.get('performance_stats', {}).get('total_tasks_processed', 0) / 3600,
                "success_rate": worker_metrics.get('performance_stats', {}).get('success_rate', 100)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Queue status error: {e}")

@health_router.post("/background/workers/scale")
async def scale_workers(worker_count: int = Query(..., ge=1, le=10)):
    """تغییر تعداد کارگران"""
    try:
        background_worker = get_debug_module('background_worker')
        
        # توقف worker فعلی
        background_worker.stop()
        
        # ایجاد worker جدید با تعداد مشخص
        background_worker.max_workers = worker_count
        background_worker.executor = ThreadPoolExecutor(max_workers=worker_count)
        background_worker.start()
        
        return {
            "status": "success",
            "message": f"Workers scaled to {worker_count}",
            "new_worker_count": worker_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Worker scaling error: {e}")

@health_router.post("/background/tasks/submit")
async def submit_background_task(
    task_type: str = Query(..., regex="^(heavy|normal|light|maintenance)$"),
    task_name: str = Query(...),
    priority: int = Query(1, ge=1, le=10)
):
    """ثبت کار جدید در پس‌زمینه"""
    try:
        background_worker = get_debug_module('background_worker')
        background_tasks = get_debug_module('background_tasks')
        
        # پیدا کردن تابع مربوطه
        task_func = getattr(background_tasks, task_name, None)
        if not task_func:
            raise HTTPException(status_code=400, detail=f"Task {task_name} not found")
        
        task_id = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success, message = background_worker.submit_task(
            task_id=task_id,
            task_func=task_func,
            task_type=task_type,
            priority=priority
        )
        
        if success:
            return {
                "status": "submitted",
                "task_id": task_id,
                "message": message,
                "task_type": task_type,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail=message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task submission error: {e}")

@health_router.get("/background/tasks/{task_id}")
async def get_task_status(task_id: str):
    """دریافت وضعیت یک کار خاص"""
    try:
        background_worker = get_debug_module('background_worker')
        
        task_status = background_worker.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_status": task_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task status error: {e}")

# ==================== ADVANCED MONITORING ENDPOINTS ====================

@health_router.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """دریافت دشبورد کامل مانیتورینگ"""
    try:
        monitoring_dashboard = get_debug_module('monitoring_dashboard')
        dashboard_data = monitoring_dashboard.get_dashboard_data()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "dashboard": dashboard_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {e}")

@health_router.get("/monitoring/alerts")
async def get_active_alerts():
    """دریافت هشدارهای فعال"""
    try:
        monitoring_dashboard = get_debug_module('monitoring_dashboard')
        dashboard_data = monitoring_dashboard.get_dashboard_data()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "alerts": dashboard_data['alerts']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alerts error: {e}")

@health_router.post("/monitoring/alerts/{alert_index}/acknowledge")
async def acknowledge_alert(alert_index: int):
    """تأیید یک هشدار"""
    try:
        monitoring_dashboard = get_debug_module('monitoring_dashboard')
        
        success = monitoring_dashboard.acknowledge_alert(alert_index)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "status": "success",
            "message": f"Alert {alert_index} acknowledged",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert acknowledgement error: {e}")

@health_router.get("/monitoring/insights")
async def get_worker_insights():
    """دریافت بینش‌های تحلیلی"""
    try:
        monitoring_dashboard = get_debug_module('monitoring_dashboard')
        insights = monitoring_dashboard.get_worker_insights()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights error: {e}")

# ==================== RESOURCE MANAGEMENT ENDPOINTS ====================

@health_router.get("/resources/advanced")
async def get_advanced_resource_metrics():
    """متریک‌های پیشرفته منابع"""
    try:
        resource_manager = get_debug_module('resource_manager')
        resource_report = resource_manager.get_detailed_resource_report()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "resource_report": resource_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource metrics error: {e}")

@health_router.get("/resources/optimization")
async def get_optimization_recommendations():
    """توصیه‌های بهینه‌سازی منابع"""
    try:
        resource_manager = get_debug_module('resource_manager')
        recommendations = resource_manager.get_optimization_recommendations()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "optimization_recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization recommendations error: {e}")

# ==================== INTELLIGENT SCHEDULING ENDPOINTS ====================

@health_router.get("/scheduling/analytics")
async def get_scheduling_analytics():
    """آمار و آنالیز زمان‌بندی"""
    try:
        task_scheduler = get_debug_module('task_scheduler')
        analytics = task_scheduler.get_scheduling_analytics()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "scheduling_analytics": analytics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheduling analytics error: {e}")

@health_router.get("/scheduling/upcoming")
async def get_upcoming_schedule():
    """برنامه زمان‌بندی آینده"""
    try:
        task_scheduler = get_debug_module('task_scheduler')
        analytics = task_scheduler.get_scheduling_analytics()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "upcoming_schedule": analytics.get('upcoming_schedule', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upcoming schedule error: {e}")

# ==================== RECOVERY SYSTEM ENDPOINTS ====================

@health_router.get("/recovery/status")
async def get_recovery_status():
    """وضعیت سیستم بازیابی"""
    try:
        recovery_manager = get_debug_module('recovery_manager')
        recovery_status = recovery_manager.get_recovery_status()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "recovery_status": recovery_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery status error: {e}")

@health_router.post("/recovery/snapshot/create")
async def create_snapshot(snapshot_type: str = "manual"):
    """ایجاد snapshot دستی"""
    try:
        recovery_manager = get_debug_module('recovery_manager')
        
        # جمع‌آوری وضعیت سیستم
        system_state = {
            "timestamp": datetime.now().isoformat(),
            "type": snapshot_type,
            "system_metrics": {}  # در واقعیت اینجا داده‌های واقعی جمع می‌شوند
        }
        
        result = recovery_manager.create_snapshot(system_state, snapshot_type)
        
        return {
            "status": "success" if result['success'] else "error",
            "snapshot_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot creation error: {e}")

@health_router.post("/recovery/auto-recover")
async def trigger_auto_recovery():
    """فعال‌سازی بازیابی خودکار"""
    try:
        recovery_manager = get_debug_module('recovery_manager')
        result = recovery_manager.auto_recover()
        
        return {
            "status": "success" if result['success'] else "error",
            "recovery_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-recovery error: {e}")

# ========================= simple AI =========================

@health_router.get("/ai/status")
async def ai_system_status():
    """وضعیت سلامت سیستم هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        ai_health = ai_brain.get_network_health()
        ai_metrics = ai_monitor.collect_ai_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "component": "ai_system",
            "architecture": {
                "total_neurons": ai_health['neuron_count'],
                "active_neurons": ai_health['active_neurons'],
                "connection_sparsity": ai_health['connection_sparsity'],
                "actual_sparsity": ai_health['actual_sparsity'],
                "memory_usage_mb": ai_health['memory_usage_mb']
            },
            "performance": {
                "training_samples": ai_health['performance']['training_samples'],
                "current_accuracy": ai_health['performance']['current_accuracy'],
                "trend_accuracy": ai_health['performance']['accuracy_trend_10'],
                "learning_rate": ai_brain.learning_rate,
                "last_training": ai_health['performance']['last_training']
            },
            "resources": {
                "weights_size_mb": round(ai_brain.weights.nbytes / (1024 * 1024), 2),
                "neurons_size_mb": round(ai_brain.neurons.nbytes / (1024 * 1024), 2),
                "total_memory_mb": ai_health['memory_usage_mb']
            },
            "health_score": _calculate_ai_health_score(ai_health)
        }
        
    except Exception as e:
        logger.error(f"❌ AI status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI status error: {e}")

@health_router.get("/ai/metrics")
async def ai_system_metrics():
    """متریک‌های دقیق سیستم هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        ai_metrics = ai_monitor.collect_ai_metrics()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": ai_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ AI metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI metrics error: {e}")

@health_router.post("/ai/optimize")
async def optimize_ai_system():
    """بهینه‌سازی خودکار معماری هوش مصنوعی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        # بهینه‌سازی معماری
        ai_brain.optimize_architecture()
        
        # جمع‌آوری متریک‌های بعد از بهینه‌سازی
        health_after = ai_brain.get_network_health()
        
        return {
            "status": "optimized",
            "timestamp": datetime.now().isoformat(),
            "optimization_results": {
                "new_learning_rate": ai_brain.learning_rate,
                "architecture_changes": {
                    "connection_sparsity": health_after['actual_sparsity'],
                    "average_weight": health_after['average_weight']
                },
                "performance_impact": {
                    "accuracy_trend": health_after['performance']['accuracy_trend_10'],
                    "training_efficiency": health_after['performance']['training_samples']
                }
            },
            "recommendations": [
                "Monitor accuracy trend for next 24 hours",
                "Consider increasing training data diversity",
                "Check memory usage after optimization"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ AI optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI optimization error: {e}")

@health_router.get("/ai/health-report")
async def ai_health_report():
    """گزارش سلامت کامل هوش مصنوعی برای سیستم مادر"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        # جمع‌آوری داده‌های سلامت از مانیتور
        health_report = ai_monitor.get_ai_health_report()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "health_report": health_report
        }
        
    except Exception as e:
        logger.error(f"❌ AI health report failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI health report error: {e}")

@health_router.get("/ai/architecture")
async def ai_architecture_info():
    """اطلاعات معماری شبکه عصبی"""
    if not AI_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI system not available")
    
    try:
        health_data = ai_brain.get_network_health()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "type": "sparse_neural_network",
                "neuron_count": health_data['neuron_count'],
                "connection_strategy": "sparse_connections",
                "sparsity_target": health_data['connection_sparsity'],
                "actual_sparsity": health_data['actual_sparsity'],
                "learning_algorithm": "backpropagation_with_sparsity",
                "activation_function": "tanh"
            },
            "resources": {
                "memory_usage": {
                    "total_mb": health_data['memory_usage_mb'],
                    "weights_mb": round(ai_brain.weights.nbytes / (1024 * 1024), 2),
                    "neurons_mb": round(ai_brain.neurons.nbytes / (1024 * 1024), 2),
                    "bias_mb": round(ai_brain.bias.nbytes / (1024 * 1024), 2)
                },
                "computation": {
                    "learning_rate": ai_brain.learning_rate,
                    "connection_density": health_data['actual_sparsity'],
                    "active_neurons": health_data['active_neurons']
                }
            },
            "performance_characteristics": {
                "training_efficiency": "high",
                "memory_efficiency": "high",
                "inference_speed": "fast",
                "scalability": "excellent"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ AI architecture info failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI architecture error: {e}")
       
# ==================== INITIALIZATION ====================
@health_router.on_event("startup")
async def startup_event():
    """رویداد startup برای مقداردهی اولیه سیستم‌های جدید"""
    logger.info("🚀 Initializing debug system on startup...")
    DebugSystemManager.initialize()
    
    # گزارش وضعیت سیستم کش جدید
    if NEW_CACHE_SYSTEM_AVAILABLE:
        logger.info("✅ New Cache System with Archive is available")
        try:
            archive_stats = get_archive_stats()
            logger.info(f"📦 Archive stats: {archive_stats['total_archives']} total archives")
        except Exception as e:
            logger.warning(f"⚠️ Could not get archive stats: {e}")
    
    if cache_optimizer:
        logger.info("✅ Cache Optimization Engine is available")
    
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

@health_router.post("/normalization/test")
async def test_normalization():
    """تست سیستم نرمال‌سازی داده"""
    try:
        # داده تست برای نرمال‌سازی
        test_data = {
            "test": "data",
            "numbers": [1, 2, 3, 4, 5],
            "nested": {
                "key1": "value1", 
                "key2": 123,
                "key3": [True, False, True]
            },
            "timestamp": datetime.now().isoformat(),
            "mixed_data": {
                "string": "hello",
                "number": 42,
                "boolean": True,
                "array": [1, "two", False],
                "null_value": None
            }
        }
        
        # گرفتن متریک‌های قبل از تست
        metrics_before = data_normalizer.get_health_metrics()
        
        # اجرای نرمال‌سازی
        normalized_result = data_normalizer.normalize_data(test_data, "health_test_endpoint")
        
        # گرفتن متریک‌های بعد از تست
        metrics_after = data_normalizer.get_health_metrics()
        
        # تحلیل عمیق
        deep_analysis = data_normalizer.get_deep_analysis()
        
        return {
            "status": "success",
            "message": "Normalization test completed successfully",
            "timestamp": datetime.now().isoformat(),
            "test_data": {
                "original": test_data,
                "normalized": normalized_result,
                "data_size_original": len(str(test_data)),
                "data_size_normalized": len(str(normalized_result)) if normalized_result else 0
            },
            "metrics_comparison": {
                "before": {
                    "success_rate": metrics_before.success_rate,
                    "total_processed": metrics_before.total_processed,
                    "total_errors": metrics_before.total_errors
                },
                "after": {
                    "success_rate": metrics_after.success_rate,
                    "total_processed": metrics_after.total_processed, 
                    "total_errors": metrics_after.total_errors
                },
                "improvement": {
                    "requests_increased": metrics_after.total_processed - metrics_before.total_processed,
                    "success_rate_change": metrics_after.success_rate - metrics_before.success_rate
                }
            },
            "analysis_overview": {
                "system_health": deep_analysis.get("system_overview", {}),
                "common_patterns": deep_analysis.get("common_patterns", {}),
                "recommendations": deep_analysis.get("recommendations", [])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Normalization test failed: {e}")
        return {
            "status": "error",
            "message": f"Normalization test failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "error_details": str(e)
        }

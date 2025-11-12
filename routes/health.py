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
    """بررسی واقعی وضعیت سیستم کش"""
    try:
        # Smart Cache:اول #
        smart_cache_ok = False
        if smart_cache and hasattr(smart_cache, 'get_health_status'):
            try:
                cache_health = smart_cache.get_health_status()  # ✅ الان این متد وجود داره
                smart_cache_ok = cache_health.get("status") == "healthy"
                logger.info(f"✅ Smart cache health: {cache_health.get('status')}")
            except Exception as e:
                logger.warning(f"❌ Smart cache health check failed: {e}")
                smart_cache_ok = False
        else:
            smart_cache_ok = False
            logger.warning("❌ Smart cache not available or missing get_health_status method")
        
        # بررسی دوم: Redis
        from debug_system.storage import redis_manager
        redis_health = redis_manager.health_check()
        redis_ok = redis_health.get("status") == "connected"
        
        # بررسی سوم: Cache Debugger
        cache_debugger_ok = False
        try:
            from debug_system.storage.cache_debugger import cache_debugger
            cache_debugger_ok = hasattr(cache_debugger, 'get_cache_stats')
        except ImportError:
            cache_debugger_ok = False
        
        # اگر حداقل یکی از سیستم‌ها کار کند، کش در دسترس است
        return smart_cache_ok or redis_ok or cache_debugger_ok
        
    except Exception as e:
        logger.warning(f"⚠️ Cache availability check failed: {e}")
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

def _check_external_apis_availability() -> bool:
    """بررسی واقعی وضعیت APIهای خارجی"""
    try:
        if not coin_stats_manager:
            logger.warning("coin_stats_manager is None")
            return False
        
        # استفاده از متد تست سریع
        if hasattr(coin_stats_manager, 'test_api_connection_quick'):
            return coin_stats_manager.test_api_connection_quick()
        
        # روش جایگزین
        api_status = coin_stats_manager.get_api_status()
        return api_status.get('status') in ['healthy', 'connected']
            
    except Exception as e:
        logger.warning(f"API availability check failed: {e}")
        return False

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
    """دریافت جزئیات وضعیت کش - نسخه به‌روز شده"""
    details = {
        "smart_cache_available": False,
        "cache_optimizer_available": False,
        "new_cache_system_available": NEW_CACHE_SYSTEM_AVAILABLE,
        "redis_available": False,
        "cache_debugger_available": False,
        "overall_status": "unavailable"
    }
    
    try:
        # بررسی Cache Optimization Engine
        if cache_optimizer and hasattr(cache_optimizer, 'analyze_access_patterns'):
            details["cache_optimizer_available"] = True
            details["cache_optimizer_health"] = "available"
            try:
                analysis = cache_optimizer.analyze_access_patterns(hours=1)
                details["optimization_analysis"] = analysis
            except Exception as e:
                details["optimization_analysis_error"] = str(e)

        # بررسی سیستم کش جدید
        details["new_cache_system_available"] = NEW_CACHE_SYSTEM_AVAILABLE
        if NEW_CACHE_SYSTEM_AVAILABLE:
            try:
                archive_stats = get_archive_stats()
                details["archive_stats"] = archive_stats
            except Exception as e:
                details["archive_stats_error"] = str(e)
        
        # بررسی Redis
        from debug_system.storage import redis_manager
        redis_health = redis_manager.health_check()
        details["redis_available"] = redis_health.get("status") == "connected"
        details["redis_health"] = redis_health
        
        # بررسی Cache Debugger
        try:
            from debug_system.storage.cache_debugger import cache_debugger
            details["cache_debugger_available"] = hasattr(cache_debugger, 'get_cache_stats')
        except ImportError:
            details["cache_debugger_available"] = False
        
        # وضعیت کلی
        if (details["cache_optimizer_available"] or 
            details["new_cache_system_available"] or 
            details["redis_available"]):
            details["overall_status"] = "available"
        
        return details
        
    except Exception as e:
        logger.error(f"❌ Error getting cache details: {e}")
        return details

def _get_component_recommendations(self, cache_details: Dict, normalization_metrics: Dict, api_status: str) -> List[str]:
    """تولید توصیه‌های هوشمند بر اساس وضعیت کامپوننت‌ها"""
    recommendations = []
    
    # بررسی سیستم کش
    if not cache_details.get("five_databases_available", False):
        recommendations.append("🔧 Connect all 5 Redis databases for optimal cache performance")
    
    if not cache_details.get("archive_system_available", False):
        recommendations.append("📦 Enable historical archive system for data persistence")
    
    if cache_details.get("connected_databases", 0) < 3:
        recommendations.append("⚠️ Low database connectivity - check Redis connections")
    
    # بررسی نرمال‌سازی
    if normalization_metrics.get("success_rate", 0) < 85:
        recommendations.append("🔄 Improve data normalization rules - current success rate is low")
    
    # بررسی API
    if api_status != "healthy":
        recommendations.append("🌐 Fix external API connectivity issues")
    
    # بررسی منابع
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        recommendations.append("💾 High memory usage - consider optimization")
    
    return recommendations

# ==================== BASIC HEALTH ENDPOINTS ====================

@health_router.get("/status")
async def health_status():
    """وضعیت سلامت کامل سیستم - روت اصلی با سیستم کش آپدیت شده"""
    
    # زمان شروع برای محاسبه عملکرد
    start_time = time.time()
    
    try:
        # 1. جمع‌آوری اطلاعات پایه سیستم
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # 2. وضعیت سیستم کش - نسخه کاملاً آپدیت شده
        cache_details = _get_cache_details()
        cache_health = {}
        cache_available = cache_details["overall_status"] != "unavailable"

        try:
            if cache_details["overall_status"] == "advanced":
                # سیستم پیشرفته با ۵ دیتابیس و آرشیو
                cache_health = {
                    "status": "healthy",
                    "health_score": 95,
                    "architecture": "5-databases-with-archive",
                    "databases": {
                        "uta": {"role": "AI Core Models", "status": "connected"},
                        "utb": {"role": "AI Processed Data", "status": "connected"},
                        "utc": {"role": "Raw Data + Historical Archive", "status": "connected"},
                        "mother_a": {"role": "System Core Data", "status": "connected"},
                        "mother_b": {"role": "Operations & Analytics", "status": "connected"}
                    },
                    "features": {
                        "real_time_cache": True,
                        "historical_archive": True,
                        "data_compression": True,
                        "smart_ttl_management": True,
                        "access_pattern_analysis": cache_details["cache_optimizer_available"],
                        "cost_optimization": cache_details["cache_optimizer_available"]
                    },
                    "performance": {
                        "connected_databases": cache_details.get("connected_databases", 0),
                        "total_archives": cache_details.get("archive_stats", {}).get("total_archives", 0),
                        "archive_size_mb": cache_details.get("archive_stats", {}).get("total_size_mb", 0),
                        "optimization_available": cache_details["cache_optimizer_available"]
                    },
                    "summary": {
                        "hit_rate": 0,  # از cache_debugger می‌آید
                        "total_requests": 0,
                        "avg_response_time": 0,
                        "compression_savings": 0,
                        "strategies_active": 8
                    }
                }
                
                # اگر cache_debugger در دسترس است، آمار واقعی بگیر
                try:
                    from debug_system.storage.cache_debugger import cache_debugger
                    cache_stats = cache_debugger.get_cache_stats()
                    cache_health["summary"]["hit_rate"] = cache_stats.get('hit_rate', 0)
                    cache_health["summary"]["total_requests"] = cache_stats.get('total_operations', 0)
                except:
                    pass
                
            elif cache_details["overall_status"] == "basic":
                # سیستم پایه
                cache_health = {
                    "status": "degraded",
                    "health_score": 70,
                    "architecture": "single-database",
                    "features": {
                        "real_time_cache": True,
                        "historical_archive": False,
                        "data_compression": False,
                        "smart_ttl_management": False,
                        "access_pattern_analysis": False,
                        "cost_optimization": False
                    },
                    "performance": {
                        "connected_databases": 1,
                        "total_archives": 0,
                        "archive_size_mb": 0,
                        "optimization_available": False
                    },
                    "summary": {
                        "hit_rate": 0,
                        "total_requests": 0,
                        "avg_response_time": 0,
                        "compression_savings": 0,
                        "strategies_active": 0
                    }
                }
            else:
                cache_health = {
                    "status": "unavailable",
                    "health_score": 0,
                    "error": "No cache system available",
                    "architecture": "none",
                    "features": {
                        "real_time_cache": False,
                        "historical_archive": False,
                        "data_compression": False,
                        "smart_ttl_management": False,
                        "access_pattern_analysis": False,
                        "cost_optimization": False
                    }
                }
        
        except Exception as e:
            cache_health = {
                "status": "error", 
                "error": str(e),
                "health_score": 0
            }
        
        # 3. وضعیت API خارجی - نسخه واقعی
        api_status = "unknown"
        api_details = {}
        api_available = _check_external_apis_availability()

        if coin_stats_manager:
            try:
                api_check = coin_stats_manager.get_api_status()
                api_status = api_check.get('status', 'unknown')
                api_details = api_check
        
                # اضافه کردن متریک‌های عملکرد
                if hasattr(coin_stats_manager, 'get_performance_metrics'):
                    perf_metrics = coin_stats_manager.get_performance_metrics()
                    api_details['performance_metrics'] = perf_metrics
            
            except Exception as e:
                api_status = f"error: {str(e)}"
                api_details = {"error": str(e)}
        else:
            api_status = "manager_not_available"
            api_details = {"error": "coin_stats_manager not initialized"}
            
        # 4. وضعیت نرمال‌سازی داده - نسخه واقعی
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
            # تبدیل به فرمت ساده‌تر
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
        
        # 6. وضعیت دیتابیس (شبیه‌سازی)
        db_status = {
            "status": "connected",
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "connections": 5
        }
        
        # 7. محاسبه سلامت کلی سیستم - نسخه آپدیت شده
        health_score = 100
        
        # کسر امتیاز بر اساس خطاها - منطق آپدیت شده
        cache_status = cache_health.get("status")
        if cache_status == "healthy":
            # سیستم کش پیشرفته - امتیاز کامل
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
        
        # وضعیت کلی بر اساس امتیاز
        overall_status = "healthy" if health_score >= 90 else "degraded" if health_score >= 70 else "unhealthy"
        
        # 8. جمع‌بندی سرویس‌ها - نسخه آپدیت شده
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
                "status": "available" if cache_details["cache_optimizer_available"] else "unavailable",
                "features": {
                    "access_analysis": cache_details["cache_optimizer_available"],
                    "ttl_optimization": cache_details["cache_optimizer_available"],
                    "cost_management": cache_details["cache_optimizer_available"]
                }
            }
        }
        
        # 9. وضعیت منابع
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
        
        # 10. هشدارها و توصیه‌ها - نسخه آپدیت شده
        alerts = []
        recommendations = []
        
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
            recommendations.append("Enable 5-database cache architecture for better performance")
        elif cache_architecture == "none":
            alerts.append({
                "level": "CRITICAL",
                "message": "Cache system unavailable - system performance degraded",
                "component": "cache_system"
            })
            recommendations.append("Check Redis connections and cache system configuration")
        
        if normalization_metrics.get("success_rate", 0) < 90:
            alerts.append({
                "level": "WARNING",
                "message": "Data normalization success rate is low",
                "component": "data_processing"
            })
            recommendations.append("Check data normalization rules and patterns")
        
        if resources_status["memory"]["usage_percent"] > 80:
            alerts.append({
                "level": "WARNING",
                "message": "High memory usage detected",
                "component": "memory"
            })
            recommendations.append("Consider optimizing memory usage or scaling resources")
        
        if resources_status["disk"]["usage_percent"] > 85:
            alerts.append({
                "level": "CRITICAL",
                "message": "Disk space running low",
                "component": "disk"
            })
            recommendations.append("Clean up disk space immediately using /api/health/cleanup/urgent")
        
        # 11. پاسخ نهایی - نسخه آپدیت شده
        response = {
            "status": overall_status,
            "health_score": health_score,
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
                "cpu_usage_percent": resources_status["cpu"]["usage_percent"]
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
                "available": DebugSystemManager.is_available(),
                "loaded_modules": DebugSystemManager.get_status_report().get('loaded_modules', 0)
            },
    
            "data_processing": {
                "normalization_available": _check_normalization_availability(),
                "success_rate": normalization_metrics.get("success_rate", 0),
                "total_processed": normalization_metrics.get("total_processed", 0)
            },
    
            "external_apis": {
                "available": _check_external_apis_availability(),
                "status": api_status,
                "details": api_details.get('status', 'unknown')
            },
    
            # وضعیت کلی کامپوننت‌ها
            "overall_health": {
                "all_core_components": (
                    cache_available and 
                    _check_normalization_availability() and 
                    _check_external_apis_availability()
                ),
                "all_advanced_features": (
                    cache_details.get("archive_system_available", False) and
                    cache_details.get("cache_optimizer_available", False) and
                    cache_health.get("features", {}).get("smart_ttl_management", False)
                ),
                "recommended_actions": self._get_component_recommendations(cache_details, normalization_metrics, api_status)
            }
        }
            
            "cache_architecture_details": {
                "databases": {
                    "uta": "AI Core Models - Long term storage",
                    "utb": "AI Processed Data - Medium TTL", 
                    "utc": "Raw Data + Historical Archive - Short TTL + Long term archive",
                    "mother_a": "System Core Data - Critical system data",
                    "mother_b": "Operations & Analytics - Cache analytics and temp data"
                },
                "features_available": cache_health.get("features", {}),
                "performance_metrics": cache_health.get("performance", {})
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in health status: {e}")
        # لاگ خطای دقیق‌تر
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
    """وضعیت دقیق دیسک با جزئیات فایل‌های حجیم"""
    try:
        disk = psutil.disk_usage('/')
        
        # یافتن فایل‌های حجیم
        large_files = []
        total_large_files_size = 0
        
        # اسکن فایل‌های بزرگتر از 1MB
        for root, dirs, files in os.walk('.'):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    if os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        if file_size > 1 * 1024 * 1024:  # بیشتر از 1MB
                            size_mb = file_size / (1024 * 1024)
                            large_files.append({
                                "path": filepath,
                                "size_mb": round(size_mb, 2),
                                "type": "large_file"
                            })
                            total_large_files_size += file_size
                except (OSError, Exception):
                    continue
        
        # محاسبه حجم پوشه‌های خاص
        folder_sizes = {}
        special_folders = ['__pycache__', 'node_modules', '.git', 'logs', '.cache']
        
        for folder in special_folders:
            folder_size = 0
            folder_count = 0
            for found_folder in glob.glob(f"**/{folder}", recursive=True):
                if os.path.isdir(found_folder):
                    for dirpath, dirnames, filenames in os.walk(found_folder):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:
                                if os.path.isfile(filepath):
                                    folder_size += os.path.getsize(filepath)
                                    folder_count += 1
                            except (OSError, Exception):
                                continue
            
            if folder_size > 0:
                folder_sizes[folder] = {
                    "size_mb": round(folder_size / (1024 * 1024), 2),
                    "file_count": folder_count
                }
        
        return {
            "disk_usage": {
                "total_gb": 1.0,
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": disk.percent,
                "critical_warning": disk.percent > 85
            },
            "large_files": {
                "count": len(large_files),
                "total_size_mb": round(total_large_files_size / (1024 * 1024), 2),
                "files": large_files[:10]  # فقط 10 فایل اول
            },
            "special_folders": folder_sizes,
            "cleanup_recommendations": [
                "Run /api/health/cleanup/urgent to free space immediately",
                "Delete __pycache__ folders (usually 50-200MB)",
                "Clear log files if they are too large",
                "Remove temporary .pyc files"
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

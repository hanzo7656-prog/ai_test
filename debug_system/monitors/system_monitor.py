"""
SYSTEM METRICS COLLECTOR - نسخه نهایی و ساده‌شده
تنها فایل جمع‌آوری متریک در سیستم - استفاده از ماژول‌های واقعی
"""

import psutil
import time
import logging
import threading
import sys
import os
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """
    کلاس اصلی جمع‌آوری متریک‌ها - تنها نقطه جمع‌آوری متریک در سیستم
    """
    
    def __init__(self, collection_interval: int = 30):
        """
        مقداردهی اولیه
        
        Args:
            collection_interval: فاصله جمع‌آوری متریک‌ها به ثانیه (پیش‌فرض: 30)
        """
        # ایمپورت ماژول‌های واقعی سیستم
        try:
            from debug_system.storage.redis_manager import redis_manager
            from debug_system.storage.cache_debugger import cache_debugger
            from debug_system.storage.history_manager import history_manager
            from debug_system.storage.complete_coinstats_manager import coin_stats_manager
            
            self.redis = redis_manager
            self.cache = cache_debugger
            self.history = history_manager
            self.coin_stats = coin_stats_manager
            
            logger.info("✅ ماژول‌های واقعی سیستم ایمپورت شدند")
            
        except ImportError as e:
            logger.error(f"❌ خطا در ایمپورت ماژول‌ها: {e}")
            self.redis = None
            self.cache = None
            self.history = None
            self.coin_stats = None
        
        # متغیرهای حالت
        self.metrics_cache = {}
        self.last_collection_time = None
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.collection_thread = None
        
        # ساختارهای ساده ردیابی
        self.request_count = 0
        self.endpoint_counts = defaultdict(int)
        self.start_time = datetime.now()
        
        # وضعیت ماژول‌ها
        self.modules_available = all([
            self.redis is not None,
            self.cache is not None,
            self.history is not None,
            self.coin_stats is not None
        ])
        
        if self.modules_available:
            logger.info("✅ SystemMetricsCollector با ماژول‌های واقعی مقداردهی شد")
        else:
            logger.warning("⚠️ برخی ماژول‌ها در دسترس نیستند - متریک‌های محدودی جمع‌آوری می‌شود")
    
    def start_collection(self):
        """شروع جمع‌آوری دوره‌ای متریک‌ها"""
        if self.is_collecting:
            logger.warning("⚠️ Collection already running")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollectorThread"
        )
        self.collection_thread.start()
        logger.info(f"🔄 System metrics collection started ({self.collection_interval}s interval)")
    
    def stop_collection(self):
        """توقف جمع‌آوری"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("🛑 System metrics collection stopped")
    
    def _collection_loop(self):
        """حلقه اصلی جمع‌آوری"""
        while self.is_collecting:
            try:
                start_time = time.time()
                
                # جمع‌آوری متریک‌ها
                metrics = self.collect_all_metrics()
                self.metrics_cache = metrics
                self.last_collection_time = datetime.now()
                
                collection_time = time.time() - start_time
                if collection_time > 2.0:
                    logger.warning(f"⚠️ Collection took {collection_time:.2f}s")
                
                # خواب تا interval بعدی
                sleep_time = max(1, self.collection_interval - collection_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Collection error: {e}")
                time.sleep(60)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        جمع‌آوری تمام متریک‌های سیستم
        
        Returns:
            Dict[str, Any]: دیکشنری کامل متریک‌ها
        """
        timestamp = datetime.now()
        
        return {
            "timestamp": timestamp.isoformat(),
            
            # 1. متریک‌های سیستم‌عامل
            "system": self._get_system_metrics(),
            
            # 2. متریک‌های اپلیکیشن
            "application": self._get_app_metrics(),
            
            # 3. متریک‌های وابستگی‌های واقعی
            "dependencies": self._get_real_dependencies(),
            
            # 4. متریک‌های درخواست‌ها
            "requests": self._get_request_metrics(),
            
            # متادیتا
            "metadata": {
                "collection_interval": self.collection_interval,
                "modules_available": self.modules_available,
                "collection_version": "3.0_simplified",
                "collector_uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
        }
    
    # ==================== سیستم‌عامل ====================
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستم‌عامل"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory
            memory = psutil.virtual_memory()
            
            # Disk
            disk_usage = psutil.disk_usage('/')
            
            # Network
            net_io = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Uptime
            uptime_seconds = time.time() - psutil.boot_time()
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2)
                },
                "disk": {
                    "usage_percent": disk_usage.percent,
                    "free_gb": round(disk_usage.free / (1024**3), 2),
                    "total_gb": round(disk_usage.total / (1024**3), 2)
                },
                "network": {
                    "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                    "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2)
                },
                "process": {
                    "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                    "threads_count": threading.active_count()
                },
                "uptime_seconds": uptime_seconds,
                "uptime_hours": round(uptime_seconds / 3600, 1)
            }
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return {"error": str(e)}
    
    # ==================== اپلیکیشن ====================
    
    def _get_app_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های اپلیکیشن"""
        try:
            process = psutil.Process()
            
            return {
                "pid": process.pid,
                "start_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                "cpu_usage_percent": process.cpu_percent(),
                "memory": {
                    "rss_mb": round(process.memory_info().rss / (1024**2), 2),
                    "vms_mb": round(process.memory_info().vms / (1024**2), 2)
                },
                "open_files_count": len(process.open_files()) if hasattr(process, 'open_files') else 0,
                "python": {
                    "version": sys.version.split()[0],
                    "implementation": sys.implementation.name
                },
                "working_directory": os.getcwd()
            }
        except Exception as e:
            logger.error(f"❌ Error collecting application metrics: {e}")
            return {"error": str(e)}
    
    # ==================== وابستگی‌های واقعی ====================
    
    def _get_real_dependencies(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های وابستگی‌های واقعی"""
        try:
            return {
                "redis": self._check_redis_real(),
                "cache": self._check_cache_real(),
                "database": self._check_database_real(),
                "coin_api": self._check_coin_api_real()
            }
        except Exception as e:
            logger.error(f"❌ Error collecting dependency metrics: {e}")
            return {"error": str(e)}
    
    def _check_redis_real(self) -> Dict[str, Any]:
        """بررسی Redis واقعی"""
        if not self.redis:
            return {"status": "module_not_available"}
        
        try:
            health = self.redis.health_check()
            
            # شمارش دیتابیس‌های متصل
            connected_count = 0
            for db_info in health.values():
                if isinstance(db_info, dict) and db_info.get('status') == 'connected':
                    connected_count += 1
            
            return {
                "status": "healthy" if connected_count > 0 else "unhealthy",
                "databases_total": len(health),
                "databases_connected": connected_count,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error checking Redis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def _check_cache_real(self) -> Dict[str, Any]:
        """بررسی کش واقعی"""
        if not self.cache:
            return {"status": "module_not_available"}
        
        try:
            stats = self.cache.get_cache_stats()
            
            # محاسبه hit rate
            total_hits = stats.get('total_hits', 0)
            total_misses = stats.get('total_misses', 0)
            total_operations = total_hits + total_misses
            
            hit_rate = (total_hits / total_operations * 100) if total_operations > 0 else 0
            
            return {
                "status": "healthy",
                "hit_rate_percent": round(hit_rate, 2),
                "total_keys": stats.get('total_keys', 0),
                "total_databases": stats.get('total_databases', 0),
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error checking cache: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def _check_database_real(self) -> Dict[str, Any]:
        """بررسی دیتابیس واقعی"""
        if not self.history:
            return {"status": "module_not_available"}
        
        try:
            conn = self.history._get_connection()
            
            # تعداد رکوردهای هر جدول اصلی
            tables = {}
            try:
                endpoint_count = conn.execute("SELECT COUNT(*) as count FROM endpoint_history").fetchone()['count']
                tables['endpoint_history'] = endpoint_count
            except:
                tables['endpoint_history'] = 0
            
            try:
                metrics_count = conn.execute("SELECT COUNT(*) as count FROM system_metrics_history").fetchone()['count']
                tables['system_metrics_history'] = metrics_count
            except:
                tables['system_metrics_history'] = 0
            
            conn.close()
            
            total_records = sum(tables.values())
            
            return {
                "status": "healthy",
                "database_type": "SQLite",
                "total_records": total_records,
                "tables": tables,
                "database_path": str(self.history.db_path) if hasattr(self.history, 'db_path') else "unknown",
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error checking database: {e}")
            return {
                "status": "error",
                "error": str(e),
                "database_type": "SQLite",
                "last_checked": datetime.now().isoformat()
            }
    
    def _check_coin_api_real(self) -> Dict[str, Any]:
        """بررسی API ارز واقعی"""
        if not self.coin_stats:
            return {"status": "module_not_available"}
        
        try:
            # تست اتصال سریع
            is_connected = self.coin_stats.test_api_connection_quick()
            
            # وضعیت API
            api_status = self.coin_stats.get_api_status()
            
            return {
                "status": "connected" if is_connected else "disconnected",
                "connected": is_connected,
                "api_status": api_status,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error checking coin API: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    # ==================== درخواست‌ها ====================
    
    def _get_request_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های درخواست‌ها"""
        return {
            "total_requests": self.request_count,
            "endpoints": dict(self.endpoint_counts),
            "start_time": self.start_time.isoformat(),
            "requests_per_minute": self._calculate_requests_per_minute()
        }
    
    def _calculate_requests_per_minute(self) -> float:
        """محاسبه درخواست‌ها در دقیقه"""
        uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        if uptime_minutes > 0:
            return round(self.request_count / uptime_minutes, 2)
        return 0.0
    
    # ==================== API عمومی ====================
    
    def record_request(self, endpoint: str):
        """
        ثبت درخواست دریافتی
        
        Args:
            endpoint: آدرس endpoint فراخوانی شده
        """
        self.request_count += 1
        self.endpoint_counts[endpoint] += 1
        logger.debug(f"📊 Request recorded: {endpoint} (total: {self.request_count})")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        دریافت آخرین متریک‌های جمع‌آوری شده
        
        Returns:
            Dict[str, Any]: متریک‌های آخرین جمع‌آوری
        """
        if not self.metrics_cache:
            return self.collect_all_metrics()
        return self.metrics_cache
    
    def get_health_summary(self) -> Dict[str, bool]:
        """
        دریافت خلاصه سلامت سیستم
        
        Returns:
            Dict[str, bool]: وضعیت سلامت هر بخش
        """
        metrics = self.get_current_metrics()
        
        # بررسی وضعیت سیستم
        system_healthy = metrics.get('system', {}).get('cpu_percent', 100) < 90
        
        # بررسی وابستگی‌ها
        dependencies = metrics.get('dependencies', {})
        redis_healthy = dependencies.get('redis', {}).get('status') == 'healthy'
        cache_healthy = dependencies.get('cache', {}).get('status') == 'healthy'
        db_healthy = dependencies.get('database', {}).get('status') == 'healthy'
        api_healthy = dependencies.get('coin_api', {}).get('connected', False) is True
        
        return {
            "system": system_healthy,
            "redis": redis_healthy,
            "cache": cache_healthy,
            "database": db_healthy,
            "coin_api": api_healthy,
            "overall": all([system_healthy, redis_healthy, cache_healthy, db_healthy, api_healthy])
        }
    
    def reset_counters(self):
        """بازنشانی شمارنده‌ها"""
        self.request_count = 0
        self.endpoint_counts.clear()
        self.start_time = datetime.now()
        logger.info("🔁 Metrics counters reset")


# نمونه گلوبال
system_metrics_collector = None

def initialize_system_metrics_collector(collection_interval: int = 30):
    """
    مقداردهی اولیه جمع‌آوری متریک
    
    Args:
        collection_interval: فاصله جمع‌آوری به ثانیه
    
    Returns:
        SystemMetricsCollector: نمونه کلاس
    """
    global system_metrics_collector
    system_metrics_collector = SystemMetricsCollector(collection_interval)
    return system_metrics_collector

def get_metrics_collector():
    """
    دریافت نمونه جمع‌آوری متریک
    
    Returns:
        SystemMetricsCollector: نمونه موجود یا جدید
    """
    global system_metrics_collector
    if system_metrics_collector is None:
        system_metrics_collector = SystemMetricsCollector()
    return system_metrics_collector


# تست ساده
if __name__ == "__main__":
    import json
    
    # تنظیم لاگینگ
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing SystemMetricsCollector...")
    
    # ایجاد نمونه
    collector = SystemMetricsCollector(collection_interval=10)
    
    # ثبت چند درخواست نمونه
    collector.record_request("/api/health")
    collector.record_request("/api/metrics")
    collector.record_request("/api/health")
    
    # شروع جمع‌آوری
    collector.start_collection()
    
    # صبر برای جمع‌آوری اولیه
    time.sleep(12)
    
    # دریافت متریک‌ها
    print("\n📊 Current Metrics:")
    metrics = collector.get_current_metrics()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    # خلاصه سلامت
    print("\n🏥 Health Summary:")
    health = collector.get_health_summary()
    print(json.dumps(health, indent=2, ensure_ascii=False))
    
    # توقف
    collector.stop_collection()
    print("\n✅ Test completed successfully!")

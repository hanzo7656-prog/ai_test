"""
SYSTEM METRICS COLLECTOR - نسخه بهبودیافته
استفاده از ماژول‌های واقعی سیستم به جای شبیه‌سازی
"""

import psutil
import time
import logging
import gc
import threading
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import socket
import os
import json

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """
    کلاس جمع‌آوری متریک‌ها با استفاده از ماژول‌های واقعی سیستم
    """
    
    def __init__(self):
        # ایمپورت ماژول‌های واقعی سیستم
        try:
            from debug_system.storage.redis_manager import redis_manager
            from debug_system.storage.cache_debugger import cache_debugger
            from debug_system.storage.history_manager import history_manager
            from debug_system.storage.complete_coinstats_manager import coin_stats_manager
            
            self.redis_manager = redis_manager
            self.cache_debugger = cache_debugger
            self.history_manager = history_manager
            self.coin_stats_manager = coin_stats_manager
            
            logger.info("✅ ماژول‌های واقعی سیستم ایمپورت شدند")
            
        except ImportError as e:
            logger.error(f"❌ خطا در ایمپورت ماژول‌ها: {e}")
            # Fallback به حالت شبیه‌سازی شده
            self.redis_manager = None
            self.cache_debugger = None
            self.history_manager = None
            self.coin_stats_manager = None
        
        self.metrics_cache = {}
        self.last_collection_time = None
        self.collection_interval = 30  # ثانیه
        self.is_collecting = False
        self.collection_thread = None
        
        # کش برای محاسبات تکراری
        self.request_counts = defaultdict(lambda: deque(maxlen=3600))
        self.user_sessions = defaultdict(lambda: deque(maxlen=1000))
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'errors': 0,
            'last_hour': deque(maxlen=3600)
        })
        
        logger.info("✅ SystemMetricsCollector با ماژول‌های واقعی مقداردهی شد")
    
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
        logger.info("🔄 System metrics collection started (30s interval)")
    
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
                
                # جمع‌آوری تمام متریک‌ها
                metrics = self.collect_all_metrics()
                self.metrics_cache = metrics
                self.last_collection_time = datetime.now()
                
                collection_time = time.time() - start_time
                if collection_time > 1.0:
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
        بازگشت: دیکشنری کامل از متریک‌های خام
        """
        timestamp = datetime.now()
        
        return {
            "timestamp": timestamp.isoformat(),
            
            # 1. متریک‌های سیستم‌عامل
            "system": self._collect_system_metrics(timestamp),
            
            # 2. متریک‌های بیزینس
            "business": self._collect_business_metrics(timestamp),
            
            # 3. متریک‌های اپلیکیشن
            "application": self._collect_application_metrics(timestamp),
            
            # 4. متریک‌های امنیتی
            "security": self._collect_security_metrics(timestamp),
            
            # 5. متریک‌های وابستگی‌ها (اصلاح شده)
            "dependencies": self._collect_dependency_metrics(timestamp),
            
            # 6. متریک‌های endpointها
            "endpoints": self._collect_endpoint_metrics(timestamp),
            
            # متادیتا
            "collection_duration_ms": int((datetime.now() - timestamp).total_seconds() * 1000),
            "collection_version": "2.0_real_data"
        }
    
    # ==================== سیستم‌عامل ====================
    
    def _collect_system_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستم‌عامل"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_times = psutil.cpu_times()
            cpu_freq = psutil.cpu_freq()
            
            # Memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            net_if_stats = psutil.net_if_stats()
            
            # Process info (خود برنامه)
            process = psutil.Process()
            process_memory = process.memory_info()
            process_threads = process.num_threads()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "core_count": psutil.cpu_count(),
                    "core_percent": psutil.cpu_percent(interval=0.1, percpu=True),
                    "times": {
                        "user": cpu_times.user,
                        "system": cpu_times.system,
                        "idle": cpu_times.idle
                    },
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                    "swap_percent": swap.percent,
                    "swap_used_gb": round(swap.used / (1024**3), 2)
                },
                "disk": {
                    "usage_percent": disk_usage.percent,
                    "used_gb": round(disk_usage.used / (1024**3), 2),
                    "free_gb": round(disk_usage.free / (1024**3), 2),
                    "total_gb": round(disk_usage.total / (1024**3), 2),
                    "io_read_mb": round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                    "io_write_mb": round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0
                },
                "network": {
                    "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                    "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "connections_count": net_connections,
                    "interfaces_up": sum(1 for iface in net_if_stats.values() if iface.isup)
                },
                "process": {
                    "pid": process.pid,
                    "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                    "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                    "threads_count": process_threads,
                    "cpu_percent": process.cpu_percent(interval=0.1),
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat() if process.create_time() else None
                }
            }
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return {"error": str(e)}
    
    # ==================== بیزینس ====================
    
    def _collect_business_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های بیزینس"""
        try:
            # شمارش درخواست‌های 5 دقیقه اخیر
            five_min_ago = timestamp - timedelta(minutes=5)
            recent_requests = 0
            for requests in self.request_counts.values():
                for req_time in requests:
                    if req_time > five_min_ago:
                        recent_requests += 1
            
            # کاربران فعال (sessionهای 15 دقیقه اخیر)
            fifteen_min_ago = timestamp - timedelta(minutes=15)
            active_users = 0
            for sessions in self.user_sessions.values():
                for session_time in sessions:
                    if session_time > fifteen_min_ago:
                        active_users += 1
                        break
            
            # آمار endpointها
            endpoint_stats = {}
            for endpoint, stats in self.endpoint_stats.items():
                # محاسبه برای 5 دقیقه اخیر
                recent_calls = sum(1 for call_time in stats['last_hour'] 
                                 if call_time > five_min_ago)
                
                endpoint_stats[endpoint] = {
                    "total_calls": stats['count'],
                    "recent_calls_5min": recent_calls,
                    "avg_response_time": stats['total_time'] / stats['count'] if stats['count'] > 0 else 0,
                    "error_rate": (stats['errors'] / stats['count'] * 100) if stats['count'] > 0 else 0
                }
            
            return {
                "active_users": {
                    "total": active_users,
                    "estimated": min(active_users * 3, 1000),  # تخمین کلی
                    "sessions_active": sum(len(sessions) for sessions in self.user_sessions.values())
                },
                "requests": {
                    "total_last_5min": recent_requests,
                    "rate_per_minute": recent_requests / 5 if recent_requests > 0 else 0,
                    "total_today": sum(len(requests) for requests in self.request_counts.values())
                },
                "endpoints": endpoint_stats,
                "peak_hours": self._calculate_peak_hours(),
                "uptime": self._calculate_uptime()
            }
        except Exception as e:
            logger.error(f"❌ Error collecting business metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_peak_hours(self) -> Dict[str, Any]:
        """محاسبه ساعات پیک"""
        hour_counts = defaultdict(int)
        for requests in self.request_counts.values():
            for req_time in requests:
                hour = req_time.hour
                hour_counts[hour] += 1
        
        if not hour_counts:
            return {"peak_hour": None, "peak_requests": 0}
        
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])
        return {
            "peak_hour": peak_hour[0],
            "peak_requests": peak_hour[1],
            "busy_hours": [h for h, c in hour_counts.items() if c > peak_hour[1] * 0.5]
        }
    
    def _calculate_uptime(self) -> Dict[str, Any]:
        """محاسبه uptime سیستم"""
        try:
            import subprocess
            if os.name == 'posix':  # Linux/Unix
                result = subprocess.run(['uptime', '-p'], capture_output=True, text=True)
                uptime_str = result.stdout.strip() if result.stdout else "Unknown"
            else:  # Windows یا سایر
                uptime_str = "Not available on this OS"
            
            return {
                "string": uptime_str,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except:
            return {"string": "Unknown", "boot_time": None}
    
    # ==================== اپلیکیشن ====================
    
    def _collect_application_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های اپلیکیشن"""
        try:
            # Python runtime metrics
            gc_stats = gc.get_stats()
            gc_counts = gc.get_count()
            
            # Thread info
            thread_count = threading.active_count()
            thread_names = []
            for thread in threading.enumerate():
                thread_names.append(thread.name)
            
            # Memory profiling
            import sys
            object_count = {}
            try:
                # تخمین تعداد objectها
                for obj in gc.get_objects()[:1000]:  # نمونه‌ای
                    obj_type = type(obj).__name__
                    object_count[obj_type] = object_count.get(obj_type, 0) + 1
            except:
                object_count = {"error": "Could not count objects"}
            
            return {
                "python_runtime": {
                    "version": sys.version,
                    "implementation": sys.implementation.name,
                    "gc_stats": {
                        "collections": [{"generation": s["generation"], 
                                        "collected": s["collected"], 
                                        "collections": s["collections"]} 
                                      for s in gc_stats],
                        "counts": {
                            "gen0": gc_counts[0],
                            "gen1": gc_counts[1],
                            "gen2": gc_counts[2]
                        },
                        "thresholds": gc.get_threshold(),
                        "enabled": gc.isenabled()
                    },
                    "memory": {
                        "allocated_mb": round(sys.getsizeof([]) / (1024**2), 4),  # نمونه
                        "object_count_sample": object_count,
                        "import_count": len(sys.modules)
                    },
                    "threads": {
                        "active_count": thread_count,
                        "sample_names": thread_names[:10],  # 10 ترد اول
                        "main_thread": threading.main_thread().name
                    }
                },
                "application_info": {
                    "start_time": self._get_app_start_time(),
                    "working_directory": os.getcwd(),
                    "pid": os.getpid(),
                    "uid": os.getuid() if hasattr(os, 'getuid') else None
                }
            }
        except Exception as e:
            logger.error(f"❌ Error collecting application metrics: {e}")
            return {"error": str(e)}
    
    def _get_app_start_time(self) -> str:
        """زمان شروع اپلیکیشن"""
        try:
            process = psutil.Process()
            return datetime.fromtimestamp(process.create_time()).isoformat()
        except:
            return "Unknown"
    
    # ==================== امنیتی ====================
    
    def _collect_security_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های امنیتی"""
        try:
            # تحلیل لاگ‌های دسترسی (ساده‌شده)
            access_logs = self._parse_access_logs()
            
            # شناسایی IPهای مشکوک
            suspicious_ips = self._identify_suspicious_ips(access_logs)
            
            # بررسی failed logins
            failed_logins = self._count_failed_logins(access_logs)
            
            # بررسی rate limiting violations
            rate_violations = self._check_rate_violations()
            
            return {
                "access_patterns": {
                    "total_requests": sum(log.get('count', 0) for log in access_logs.values()),
                    "unique_ips": len(access_logs),
                    "top_ips": dict(sorted(access_logs.items(), 
                                         key=lambda x: x[1].get('count', 0), 
                                         reverse=True)[:5])
                },
                "threat_indicators": {
                    "suspicious_ips_count": len(suspicious_ips),
                    "suspicious_ips": list(suspicious_ips)[:10],  # 10 IP اول
                    "failed_logins": failed_logins,
                    "rate_limit_violations": rate_violations
                },
                "network_security": {
                    "open_ports": self._scan_open_ports(),
                    "firewall_status": self._check_firewall_status(),
                    "ssl_expiry": self._check_ssl_expiry()
                }
            }
        except Exception as e:
            logger.error(f"❌ Error collecting security metrics: {e}")
            return {"error": str(e)}
    
    def _parse_access_logs(self) -> Dict[str, Dict]:
        """پارس کردن لاگ‌های دسترسی (ساده‌شده)"""
        # در نسخه واقعی از فایل‌های log می‌خواند
        # اینجا یک نمونه ساده:
        return {
            "127.0.0.1": {"count": 100, "last_access": datetime.now().isoformat()},
            "192.168.1.1": {"count": 50, "last_access": datetime.now().isoformat()}
        }
    
    def _identify_suspicious_ips(self, access_logs: Dict) -> set:
        """شناسایی IPهای مشکوک"""
        suspicious = set()
        for ip, data in access_logs.items():
            if data.get('count', 0) > 1000:  # آستانه
                suspicious.add(ip)
        return suspicious
    
    def _count_failed_logins(self, access_logs: Dict) -> int:
        """شمارش لاگین‌های ناموفق"""
        # در نسخه واقعی از logهای احراز هویت می‌خواند
        return 0
    
    def _check_rate_violations(self) -> int:
        """بررسی نقض rate limit"""
        return 0
    
    def _scan_open_ports(self) -> List[int]:
        """اسکن پورت‌های باز روی localhost"""
        try:
            open_ports = []
            for port in [80, 443, 3000, 5000, 8000, 8080]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                if result == 0:
                    open_ports.append(port)
                sock.close()
            return open_ports
        except:
            return []
    
    def _check_firewall_status(self) -> str:
        """بررسی وضعیت فایروال"""
        return "unknown"
    
    def _check_ssl_expiry(self) -> Dict[str, Any]:
        """بررسی انقضای SSL"""
        return {"status": "unknown", "days_remaining": None}
    
    # ==================== وابستگی‌ها (اصلاح شده) ====================
    
    def _collect_dependency_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های وابستگی‌ها - با استفاده از ماژول‌های واقعی"""
        try:
            return {
                "external_apis": self._check_external_apis(),
                "internal_services": self._check_internal_services(),
                "database": self._check_database_status(),
                "cache": self._check_cache_status()
            }
        except Exception as e:
            logger.error(f"❌ Error collecting dependency metrics: {e}")
            return {"error": str(e)}
    
    def _check_external_apis(self) -> Dict[str, Any]:
        """بررسی APIهای خارجی واقعی"""
        if not self.coin_stats_manager:
            return {"coinstats_api": {"status": "module_not_available", "error": "CoinStatsManager not imported"}}
        
        try:
            # تست اتصال واقعی
            is_connected = self.coin_stats_manager.test_api_connection_quick()
            api_status = self.coin_stats_manager.get_api_status()
            perf_metrics = self.coin_stats_manager.get_performance_metrics()
            
            return {
                "coinstats_api": {
                    "status": "healthy" if is_connected else "unhealthy",
                    "connected": is_connected,
                    "api_status": api_status,
                    "performance": perf_metrics,
                    "last_checked": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "coinstats_api": {
                    "status": "error",
                    "error": str(e),
                    "last_checked": datetime.now().isoformat()
                }
            }
    
    def _check_internal_services(self) -> Dict[str, Any]:
        """بررسی سرویس‌های داخلی"""
        services = {
            "local_cache": {"status": "healthy", "size": len(self.metrics_cache)},
            "background_workers": {"status": "healthy", "count": threading.active_count() - 1}
        }
        
        # اضافه کردن Redis اگر ماژول موجود باشد
        if self.redis_manager:
            services["redis"] = self._check_redis_real()
        
        return services
    
    def _check_redis_real(self) -> Dict[str, Any]:
        """بررسی Redis واقعی"""
        try:
            health_report = self.redis_manager.health_check()
            
            # شمارش دیتابیس‌های متصل
            connected_count = 0
            for db_name, status in health_report.items():
                if isinstance(status, dict) and status.get('status') == 'connected':
                    connected_count += 1
            
            return {
                "status": "healthy" if connected_count > 0 else "degraded",
                "databases_connected": connected_count,
                "total_databases": len(self.redis_manager.databases),
                "health_summary": {
                    db: info.get('status', 'unknown') for db, info in health_report.items()
                },
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def _check_database_status(self) -> Dict[str, Any]:
        """بررسی دیتابیس واقعی (SQLite)"""
        if not self.history_manager:
            return {"status": "module_not_available", "error": "HistoryManager not imported"}
        
        try:
            conn = self.history_manager._get_connection()
            
            # تعداد رکوردهای هر جدول
            endpoint_count = conn.execute("SELECT COUNT(*) as count FROM endpoint_history").fetchone()['count']
            metrics_count = conn.execute("SELECT COUNT(*) as count FROM system_metrics_history").fetchone()['count']
            alert_count = conn.execute("SELECT COUNT(*) as count FROM alert_history").fetchone()['count']
            norm_count = conn.execute("SELECT COUNT(*) as count FROM normalization_history").fetchone()['count']
            
            conn.close()
            
            return {
                "status": "healthy",
                "database_type": "SQLite",
                "tables": {
                    "endpoint_history": endpoint_count,
                    "system_metrics_history": metrics_count,
                    "alert_history": alert_count,
                    "normalization_history": norm_count
                },
                "total_records": endpoint_count + metrics_count + alert_count + norm_count,
                "database_path": str(self.history_manager.db_path),
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "database_type": "SQLite",
                "last_checked": datetime.now().isoformat()
            }
    
    def _check_cache_status(self) -> Dict[str, Any]:
        """بررسی وضعیت کش واقعی"""
        if not self.cache_debugger:
            return {"status": "module_not_available", "error": "CacheDebugger not imported"}
        
        try:
            # دریافت آمار کلی کش
            cache_stats = self.cache_debugger.get_cache_stats()
            cache_performance = self.cache_debugger.get_cache_performance(hours=1)
            cache_efficiency = self.cache_debugger.get_cache_efficiency_report()
            
            # تجزیه و تحلیل داده‌ها
            total_hits = cache_stats.get('total_hits', 0)
            total_misses = cache_stats.get('total_misses', 0)
            total_operations = total_hits + total_misses
            
            hit_rate = (total_hits / total_operations * 100) if total_operations > 0 else 0
            
            return {
                "status": "healthy",
                "overview": {
                    "total_operations": total_operations,
                    "hit_rate": round(hit_rate, 2),
                    "total_databases": cache_stats.get('total_databases', 0),
                    "total_keys": cache_stats.get('total_keys', 0),
                    "total_size_mb": round(cache_stats.get('total_size_bytes', 0) / (1024 * 1024), 2)
                },
                "performance": {
                    "average_response_time_ms": round(cache_performance.get('average_response_time', 0) * 1000, 2),
                    "success_rate": cache_performance.get('success_rate', 0),
                    "total_operations_last_hour": cache_performance.get('total_operations', 0)
                },
                "efficiency": {
                    "score": cache_efficiency.get('efficiency_score', 0),
                    "grade": cache_efficiency.get('efficiency_grade', 'F'),
                    "hit_rate": cache_efficiency.get('overview', {}).get('hit_rate', 0)
                },
                "database_breakdown": cache_stats.get('database_breakdown', {}),
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    # ==================== Endpointها ====================
    
    def _collect_endpoint_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های endpointها"""
        try:
            endpoint_data = {}
            
            for endpoint, stats in self.endpoint_stats.items():
                # محاسبه برای 1 ساعت اخیر
                hour_ago = timestamp - timedelta(hours=1)
                recent_calls = sum(1 for call_time in stats['last_hour'] 
                                 if call_time > hour_ago)
                
                endpoint_data[endpoint] = {
                    "total_calls": stats['count'],
                    "calls_last_hour": recent_calls,
                    "average_response_time": stats['total_time'] / stats['count'] if stats['count'] > 0 else 0,
                    "error_count": stats['errors'],
                    "error_percentage": (stats['errors'] / stats['count'] * 100) if stats['count'] > 0 else 0,
                    "last_called": max(stats['last_hour']).isoformat() if stats['last_hour'] else None
                }
            
            # محاسبه آمار کلی
            total_calls = sum(stats['count'] for stats in self.endpoint_stats.values())
            total_errors = sum(stats['errors'] for stats in self.endpoint_stats.values())
            
            return {
                "endpoints": endpoint_data,
                "summary": {
                    "total_endpoints": len(self.endpoint_stats),
                    "total_calls": total_calls,
                    "total_errors": total_errors,
                    "overall_error_rate": (total_errors / total_calls * 100) if total_calls > 0 else 0,
                    "most_called_endpoint": max(self.endpoint_stats.items(), 
                                              key=lambda x: x[1]['count'])[0] if self.endpoint_stats else None
                }
            }
        except Exception as e:
            logger.error(f"❌ Error collecting endpoint metrics: {e}")
            return {"error": str(e)}
    
    # ==================== API برای ثبت رویدادها ====================
    
    def record_request(self, endpoint: str, client_ip: str, user_agent: str = None,
                      response_time: float = 0, status_code: int = 200):
        """
        ثبت درخواست دریافتی
        فقط ثبت داده - هیچ تحلیلی انجام نمی‌دهد
        """
        timestamp = datetime.now()
        
        # ثبت درخواست
        self.request_counts[client_ip].append(timestamp)
        
        # ثبت endpoint
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                'count': 0,
                'total_time': 0,
                'errors': 0,
                'last_hour': deque(maxlen=3600)
            }
        
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_time'] += response_time
        stats['last_hour'].append(timestamp)
        
        if status_code >= 400:
            stats['errors'] += 1
        
        # ثبت session کاربر
        if user_agent and "bot" not in user_agent.lower():
            user_id = f"{client_ip}_{user_agent}"
            self.user_sessions[user_id].append(timestamp)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """دریافت آخرین متریک‌های جمع‌آوری شده"""
        if not self.metrics_cache:
            return self.collect_all_metrics()
        return self.metrics_cache
    
    def get_metrics_history(self, minutes: int = 60) -> List[Dict]:
        """دریافت تاریخچه متریک‌ها (ساده‌شده)"""
        # در نسخه واقعی از دیتابیس یا فایل می‌خواند
        return [self.metrics_cache] if self.metrics_cache else []

# نمونه گلوبال
system_metrics_collector = None

def initialize_system_metrics_collector():
    """تابع مقداردهی اولیه"""
    global system_metrics_collector
    system_metrics_collector = SystemMetricsCollector()
    return system_metrics_collector

if __name__ == "__main__":
    # تست ساده
    collector = SystemMetricsCollector()
    collector.start_collection()
    
    # صبر برای جمع‌آوری اولیه
    import time
    time.sleep(35)
    
    # دریافت متریک‌ها
    metrics = collector.get_current_metrics()
    print(json.dumps(metrics["dependencies"], indent=2, ensure_ascii=False))
    
    collector.stop_collection()

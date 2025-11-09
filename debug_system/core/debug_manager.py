import time
import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import threading
import json
import traceback
from dataclasses import dataclass
from enum import Enum

# ایمپورت سیستم نرمال‌سازی جدید
try:
    from ..utils.data_normalizer import data_normalizer
except ImportError:
    # Fallback برای مواقع توسعه
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.utils.data_normalizer import data_normalizer

logger = logging.getLogger(__name__)

class DebugLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class EndpointCall:
    endpoint: str
    method: str
    timestamp: datetime
    params: Dict[str, Any]
    response_time: float
    status_code: int
    cache_used: bool
    api_calls: int
    memory_used: float
    cpu_impact: float
    normalization_info: Optional[Dict[str, Any]] = None  # ✅ اضافه شد

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    normalization_metrics: Optional[Dict[str, Any]] = None  # ✅ اضافه شد

class DebugManager:
    def __init__(self):
        self.endpoint_calls = deque(maxlen=10000)  # آخرین ۱۰۰۰۰ فراخوانی
        self.system_metrics_history = deque(maxlen=1000)  # آخرین ۱۰۰۰ متریک سیستم
        self.endpoint_stats = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'normalization_stats': {  # ✅ اضافه شد
                'total_normalized': 0,
                'normalization_errors': 0,
                'avg_quality_score': 0,
                'common_structures': {}
            },
            'errors': [],
            'last_call': None
        })
        
        self.alerts = []
        self.performance_thresholds = {
            'response_time_warning': 1.0,  # ثانیه
            'response_time_critical': 3.0,
            'cpu_warning': 80.0,  # درصد
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'normalization_error_threshold': 10  # ✅ اضافه شد
        }
        
        self.alert_manager = None  # ابتدا None، بعداً تنظیم می‌شود
        
        self._start_background_monitoring()
    
    def set_alert_manager(self, alert_manager):
        """تنظیم alert manager"""
        self.alert_manager = alert_manager
        logger.info("✅ Alert Manager set for Debug Manager")
        
    def log_endpoint_call(self, endpoint: str, method: str, params: Dict[str, Any], 
                         response_time: float, status_code: int, cache_used: bool, 
                         api_calls: int = 0, normalization_info: Dict[str, Any] = None):
        """ثبت فراخوانی اندپوینت"""
        try:
            # گرفتن متریک‌های سیستم در لحظه فراخوانی
            memory_used = psutil.virtual_memory().percent
            cpu_impact = psutil.cpu_percent(interval=0.1)
            
            call = EndpointCall(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(),
                params=params,
                response_time=response_time,
                status_code=status_code,
                cache_used=cache_used,
                api_calls=api_calls,
                memory_used=memory_used,
                cpu_impact=cpu_impact,
                normalization_info=normalization_info  # ✅ اضافه شد
            )
            
            self.endpoint_calls.append(call)
            
            # آپدیت آمار اندپوینت
            stats = self.endpoint_stats[endpoint]
            stats['total_calls'] += 1
            stats['total_response_time'] += response_time
            
            if 200 <= status_code < 300:
                stats['successful_calls'] += 1
            else:
                stats['failed_calls'] += 1
                stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'status_code': status_code,
                    'params': params
                })
                
            if cache_used:
                stats['cache_hits'] += 1
            else:
                stats['cache_misses'] += 1
                
            stats['api_calls'] += api_calls
            stats['last_call'] = datetime.now().isoformat()
            
            # ✅ آپدیت آمار نرمال‌سازی
            if normalization_info:
                norm_stats = stats['normalization_stats']
                norm_stats['total_normalized'] += 1
                
                if normalization_info.get('status') == 'error':
                    norm_stats['normalization_errors'] += 1
                
                # محاسبه میانگین کیفیت
                quality_score = normalization_info.get('quality_score', 0)
                current_avg = norm_stats['avg_quality_score']
                total_norm = norm_stats['total_normalized']
                norm_stats['avg_quality_score'] = (current_avg * (total_norm - 1) + quality_score) / total_norm
                
                # آپدیت ساختارهای رایج
                structure = normalization_info.get('detected_structure', 'unknown')
                norm_stats['common_structures'][structure] = norm_stats['common_structures'].get(structure, 0) + 1
            
            # بررسی هشدارهای performance
            self._check_performance_alerts(endpoint, call)
            
            logger.debug(f"📊 Endpoint logged: {endpoint} - {response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error logging endpoint call: {e}")
    
    def log_error(self, endpoint: str, error: Exception, traceback_str: str, context: Dict[str, Any] = None):
        """ثبت خطا"""
        error_data = {
            'endpoint': endpoint,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback_str,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # اضافه کردن به آمار اندپوینت
        self.endpoint_stats[endpoint]['errors'].append(error_data)
        
        # ایجاد هشدار برای خطاهای critical
        if self._is_critical_error(error):
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical error in {endpoint}: {str(error)}",
                source=endpoint,
                data=error_data
            )
        
        logger.error(f"🚨 Error in {endpoint}: {error}")
    
    def get_endpoint_stats(self, endpoint: str = None) -> Dict[str, Any]:
        """دریافت آمار اندپوینت"""
        if endpoint:
            if endpoint not in self.endpoint_stats:
                return {'error': 'Endpoint not found'}
            
            stats = self.endpoint_stats[endpoint]
            avg_response_time = (stats['total_response_time'] / stats['total_calls']) if stats['total_calls'] > 0 else 0
            
            # محاسبه آمار نرمال‌سازی
            norm_stats = stats['normalization_stats']
            normalization_success_rate = ((norm_stats['total_normalized'] - norm_stats['normalization_errors']) / norm_stats['total_normalized'] * 100) if norm_stats['total_normalized'] > 0 else 0
            
            return {
                'endpoint': endpoint,
                'total_calls': stats['total_calls'],
                'successful_calls': stats['successful_calls'],
                'failed_calls': stats['failed_calls'],
                'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                'average_response_time': round(avg_response_time, 3),
                'cache_performance': {
                    'hits': stats['cache_hits'],
                    'misses': stats['cache_misses'],
                    'hit_rate': (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                },
                'api_calls': stats['api_calls'],
                'normalization_performance': {  # ✅ اضافه شد
                    'total_normalized': norm_stats['total_normalized'],
                    'normalization_errors': norm_stats['normalization_errors'],
                    'success_rate': round(normalization_success_rate, 2),
                    'avg_quality_score': round(norm_stats['avg_quality_score'], 2),
                    'common_structures': norm_stats['common_structures']
                },
                'recent_errors': stats['errors'][-10:],  # آخرین ۱۰ خطا
                'last_call': stats['last_call']
            }
        else:
            # آمار تمام اندپوینت‌ها
            all_stats = {}
            total_calls = 0
            total_success = 0
            total_normalized = 0
            total_norm_errors = 0
            
            for endpoint, stats in self.endpoint_stats.items():
                norm_stats = stats['normalization_stats']
                total_normalized += norm_stats['total_normalized']
                total_norm_errors += norm_stats['normalization_errors']
                
                all_stats[endpoint] = {
                    'total_calls': stats['total_calls'],
                    'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                    'average_response_time': round((stats['total_response_time'] / stats['total_calls']), 3) if stats['total_calls'] > 0 else 0,
                    'normalization_success_rate': ((norm_stats['total_normalized'] - norm_stats['normalization_errors']) / norm_stats['total_normalized'] * 100) if norm_stats['total_normalized'] > 0 else 0,
                    'last_call': stats['last_call']
                }
                total_calls += stats['total_calls']
                total_success += stats['successful_calls']
            
            # دریافت متریک‌های کلی نرمال‌سازی
            overall_norm_metrics = data_normalizer.get_health_metrics()
            
            return {
                'overall': {
                    'total_endpoints': len(self.endpoint_stats),
                    'total_calls': total_calls,
                    'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0,
                    'normalization_overview': {  # ✅ اضافه شد
                        'total_normalized': total_normalized,
                        'normalization_errors': total_norm_errors,
                        'normalization_success_rate': ((total_normalized - total_norm_errors) / total_normalized * 100) if total_normalized > 0 else 0,
                        'system_success_rate': overall_norm_metrics.success_rate,
                        'common_structures': overall_norm_metrics.common_structures
                    },
                    'timestamp': datetime.now().isoformat()
                },
                'endpoints': all_stats
            }
    
    def get_recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """دریافت آخرین فراخوانی‌ها"""
        recent_calls = list(self.endpoint_calls)[-limit:]
        return [
            {
                'endpoint': call.endpoint,
                'method': call.method,
                'timestamp': call.timestamp.isoformat(),
                'response_time': call.response_time,
                'status_code': call.status_code,
                'cache_used': call.cache_used,
                'api_calls': call.api_calls,
                'memory_used': call.memory_used,
                'cpu_impact': call.cpu_impact,
                'normalization_info': call.normalization_info  # ✅ اضافه شد
            }
            for call in recent_calls
        ]
    
    def get_system_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌های سیستم"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # دریافت متریک‌های نرمال‌سازی فعلی
        current_norm_metrics = data_normalizer.get_health_metrics()
        
        return [
            {
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage': metrics.disk_usage,
                'network_io': metrics.network_io,
                'active_connections': metrics.active_connections,
                'normalization_metrics': {  # ✅ اضافه شد
                    'success_rate': current_norm_metrics.success_rate,
                    'total_processed': current_norm_metrics.total_processed,
                    'data_quality': current_norm_metrics.data_quality
                } if metrics.normalization_metrics is None else metrics.normalization_metrics
            }
            for metrics in self.system_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def _start_background_monitoring(self):
        """شروع مانیتورینگ پس‌زمینه سیستم"""
        def monitor_system():
            while True:
                try:
                    self._collect_system_metrics()
                    self._check_normalization_alerts()  # ✅ اضافه شد
                    time.sleep(5)  # هر ۵ ثانیه
                except Exception as e:
                    logger.error(f"❌ System monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        logger.info("✅ Background system monitoring started")
    
    def _collect_system_metrics(self):
        """جمع‌آوری متریک‌های سیستم"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            active_connections = len(psutil.net_connections())
            
            # دریافت متریک‌های نرمال‌سازی
            norm_metrics = data_normalizer.get_health_metrics()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections,
                normalization_metrics={  # ✅ اضافه شد
                    'success_rate': norm_metrics.success_rate,
                    'total_processed': norm_metrics.total_processed,
                    'data_quality': norm_metrics.data_quality
                }
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
    
    def _check_normalization_alerts(self):
        """بررسی هشدارهای نرمال‌سازی"""
        try:
            metrics = data_normalizer.get_health_metrics()
            
            # هشدار برای نرخ موفقیت پایین نرمال‌سازی
            if metrics.success_rate < 90:
                self._create_alert(
                    level=DebugLevel.WARNING,
                    message=f"Low normalization success rate: {metrics.success_rate}%",
                    source="data_normalizer",
                    data={
                        'success_rate': metrics.success_rate,
                        'total_processed': metrics.total_processed,
                        'total_errors': metrics.total_errors
                    }
                )
            
            # هشدار برای خطاهای متوالی نرمال‌سازی
            if metrics.total_errors > self.performance_thresholds['normalization_error_threshold']:
                self._create_alert(
                    level=DebugLevel.ERROR,
                    message=f"High normalization errors: {metrics.total_errors}",
                    source="data_normalizer",
                    data={
                        'total_errors': metrics.total_errors,
                        'threshold': self.performance_thresholds['normalization_error_threshold']
                    }
                )
                
        except Exception as e:
            logger.error(f"❌ Error checking normalization alerts: {e}")
    
    def _check_performance_alerts(self, endpoint: str, call: EndpointCall):
        """بررسی هشدارهای performance"""
        # هشدار زمان پاسخ‌گویی
        if call.response_time > self.performance_thresholds['response_time_critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical response time in {endpoint}: {call.response_time:.2f}s",
                source=endpoint,
                data={
                    'response_time': call.response_time,
                    'threshold': self.performance_thresholds['response_time_critical']
                }
            )
        elif call.response_time > self.performance_thresholds['response_time_warning']:
            self._create_alert(
                level=DebugLevel.WARNING,
                message=f"High response time in {endpoint}: {call.response_time:.2f}s",
                source=endpoint,
                data={
                    'response_time': call.response_time,
                    'threshold': self.performance_thresholds['response_time_warning']
                }
            )
        
        # هشدار مصرف CPU
        if call.cpu_impact > self.performance_thresholds['cpu_critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical CPU usage in {endpoint}: {call.cpu_impact:.1f}%",
                source=endpoint,
                data={'cpu_usage': call.cpu_impact}
            )
        
        # ✅ هشدار برای خطاهای نرمال‌سازی
        if call.normalization_info and call.normalization_info.get('status') == 'error':
            self._create_alert(
                level=DebugLevel.ERROR,
                message=f"Normalization error in {endpoint}: {call.normalization_info.get('error', 'Unknown error')}",
                source=endpoint,
                data=call.normalization_info
            )
    
    def _create_alert(self, level: DebugLevel, message: str, source: str, data: Dict[str, Any]):
        """ایجاد هشدار جدید"""
        alert = {
            'id': len(self.alerts) + 1,
            'level': level.value,
            'message': message,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        
        # اگر alert_manager تنظیم شده، از آن استفاده کن
        if self.alert_manager:
            try:
                # 🔧 اصلاح: استفاده از string-based comparison به جای import مستقیم
                # این از circular import جلوگیری می‌کند
                alert_level_map = {
                    DebugLevel.INFO.value: "INFO",
                    DebugLevel.WARNING.value: "WARNING", 
                    DebugLevel.ERROR.value: "ERROR",
                    DebugLevel.CRITICAL.value: "CRITICAL"
                }
                
                # استفاده از متد alert_manager بدون نیاز به import
                self.alert_manager.create_alert(
                    level=level.value
                    alert_type="PERFORMANCE",  # استفاده از string به جای enum
                    title=f"Performance Alert: {message}",
                    message=message,
                    source=source,
                    data=data
                )
            except Exception as e:
                logger.error(f"❌ Error creating alert in alert_manager: {e}")
        
        logger.warning(f"🚨 {level.value} Alert: {message}")
    
    def _is_critical_error(self, error: Exception) -> bool:
        """بررسی آیا خطا critical است"""
        critical_errors = [
            'Timeout',
            'ConnectionError', 
            'MemoryError',
            'OSError'
        ]
        
        return any(critical_error in type(error).__name__ for critical_error in critical_errors)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id: int):
        """تأیید هشدار"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
    
    def clear_old_data(self, days: int = 7):
        """پاک کردن داده‌های قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # پاک کردن فراخوانی‌های قدیمی
        self.endpoint_calls = deque(
            [call for call in self.endpoint_calls if call.timestamp > cutoff_time],
            maxlen=10000
        )
        
        # پاک کردن متریک‌های قدیمی
        self.system_metrics_history = deque(
            [metrics for metrics in self.system_metrics_history if metrics.timestamp > cutoff_time],
            maxlen=1000
        )
        
        logger.info(f"🧹 Cleared data older than {days} days")

# ایجاد نمونه گلوبال
debug_manager = DebugManager()

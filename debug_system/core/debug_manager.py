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

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int

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
            'memory_critical': 95.0
        }
        
        self.alert_manager = None  # ابتدا None، بعداً تنظیم می‌شود
        
        self._start_background_monitoring()
    
    def set_alert_manager(self, alert_manager):
        """تنظیم alert manager"""
        self.alert_manager = alert_manager
        logger.info("✅ Alert Manager set for Debug Manager")
        
    def log_endpoint_call(self, endpoint: str, method: str, params: Dict[str, Any], 
                         response_time: float, status_code: int, cache_used: bool, 
                         api_calls: int = 0):
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
                cpu_impact=cpu_impact
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
                'recent_errors': stats['errors'][-10:],  # آخرین ۱۰ خطا
                'last_call': stats['last_call']
            }
        else:
            # آمار تمام اندپوینت‌ها
            all_stats = {}
            total_calls = 0
            total_success = 0
            
            for endpoint, stats in self.endpoint_stats.items():
                all_stats[endpoint] = {
                    'total_calls': stats['total_calls'],
                    'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                    'average_response_time': round((stats['total_response_time'] / stats['total_calls']), 3) if stats['total_calls'] > 0 else 0,
                    'last_call': stats['last_call']
                }
                total_calls += stats['total_calls']
                total_success += stats['successful_calls']
            
            return {
                'overall': {
                    'total_endpoints': len(self.endpoint_stats),
                    'total_calls': total_calls,
                    'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0,
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
                'cpu_impact': call.cpu_impact
            }
            for call in recent_calls
        ]
    
    def get_system_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌های سیستم"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            {
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage': metrics.disk_usage,
                'network_io': metrics.network_io,
                'active_connections': metrics.active_connections
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
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
    
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
    
    def _create_alert(self, level: DebugLevel, message: str, source: str, data: Dict[str, Any]):
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
                # Import مستقیم برای جلوگیری از circular import
                from .alert_manager import AlertLevel, AlertType
            
                # تبدیل DebugLevel به AlertLevel
                alert_level_map = {
                    DebugLevel.INFO: AlertLevel.INFO,
                    DebugLevel.WARNING: AlertLevel.WARNING,
                    DebugLevel.ERROR: AlertLevel.ERROR,
                    DebugLevel.CRITICAL: AlertLevel.CRITICAL
                }
            
                alert_level = alert_level_map.get(level, AlertLevel.INFO)
            
                self.alert_manager.create_alert(
                    level=alert_level,
                    alert_type=AlertType.PERFORMANCE,
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

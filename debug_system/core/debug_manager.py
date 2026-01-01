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

# ایمپورت سیستم‌های مورد نیاز
try:
    from ..utils.data_normalizer import data_normalizer
    from .alert_manager import AlertLevel, AlertType, AlertManager
except ImportError:
    # Fallback برای مواقع توسعه
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.utils.data_normalizer import data_normalizer
    from debug_system.core.alert_manager import AlertLevel, AlertType, AlertManager

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
    normalization_info: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    normalization_metrics: Optional[Dict[str, Any]] = None

class DebugManager:
    """
    سیستم مدیریت دیباگ پیشرفته برای مانیتورینگ کامل عملکرد سیستم
    """
    
    def __init__(self):
        self.endpoint_calls = deque(maxlen=10000)
        self.system_metrics_history = deque(maxlen=1000)
        self.endpoint_stats = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'normalization_stats': {
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
            'response_time_warning': 1.0,
            'response_time_critical': 3.0,
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'normalization_error_threshold': 10,
            'normalization_success_threshold': 90.0
        }
        
        self.alert_manager = None
        self.alert_integration_enabled = False
        self._monitoring_active = True
        self._lock = threading.RLock()
        
        self._start_central_monitoring()
        logger.info("🚀 Debug Manager Initialized - Central Monitoring Active")
    
    def is_active(self) -> bool:
        """بررسی آیا دیباگ منیجر فعال است"""
        return self._monitoring_active and hasattr(self, 'endpoint_calls')
        
    def set_alert_manager(self, alert_manager: AlertManager) -> bool:
        """تنظیم alert manager با بررسی نوع و قابلیت"""
        try:
            if not hasattr(alert_manager, 'create_alert'):
                logger.error("❌ Invalid AlertManager instance - missing create_alert method")
                return False
            
            self.alert_manager = alert_manager
            self._alert_integration_enabled = True
            logger.info("✅ Alert Manager configured successfully - Integration Active")
            return True
          
        except Exception as e:
            logger.error(f"❌ Error setting Alert Manager: {e}")
            return False

    def _start_central_monitoring(self):
        """اتصال به سیستم مانیتورینگ مرکزی"""
        try:
            # منتظر می‌شویم central_monitor در system_monitor.py ایجاد شود
            # این در main.py بعد از راه‌اندازی انجام می‌شود
            logger.info("⏳ Waiting for central_monitor initialization...")
            
            # یک timer برای اتصال تاخیری تنظیم می‌کنیم
            def delayed_subscription():
                time.sleep(3)  # 3 ثانیه صبر کن
                self._subscribe_to_central_monitor()
            
            subscription_thread = threading.Thread(target=delayed_subscription, daemon=True)
            subscription_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Error starting central monitoring: {e}")
    
    def _subscribe_to_central_monitor(self):
        """عضویت در central_monitor"""
        try:
            from .system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت متریک‌های سیستم
                central_monitor.subscribe("debug_manager", self._on_central_metrics_update)
                logger.info("✅ DebugManager subscribed to central_monitor")
                
                # عضویت برای دریافت آلرت‌ها
                central_monitor.subscribe("debug_manager_alerts", self._on_central_alert)
                logger.info("✅ DebugManager subscribed to central_monitor alerts")
            else:
                logger.warning("⚠️ Central monitor not available, using minimal monitoring")
                self._start_minimal_monitoring()
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor, using fallback")
            self._start_minimal_monitoring()
        except Exception as e:
            logger.error(f"❌ Error subscribing to central_monitor: {e}")
            self._start_minimal_monitoring()
    
    def _on_central_metrics_update(self, metrics: Dict[str, Any]):
        """دریافت متریک‌ها از central_monitor"""
        try:
            # ذخیره در تاریخچه
            system_metrics = metrics.get('system', {})
            
            metric_obj = SystemMetrics(
                timestamp=datetime.fromisoformat(metrics['timestamp']),
                cpu_percent=system_metrics.get('cpu', {}).get('percent', 0),
                memory_percent=system_metrics.get('memory', {}).get('percent', 0),
                disk_usage=system_metrics.get('disk', {}).get('usage_percent', 0),
                network_io={
                    'bytes_sent': system_metrics.get('network', {}).get('bytes_sent', 0),
                    'bytes_recv': system_metrics.get('network', {}).get('bytes_recv', 0)
                },
                active_connections=system_metrics.get('network', {}).get('connections', 0),
                normalization_metrics=metrics.get('data_normalization', {})
            )
            
            with self._lock:
                self.system_metrics_history.append(metric_obj)
            
            # بررسی هشدارها با متریک‌های دریافتی
            self._check_system_health_with_metrics(system_metrics)
            self._check_normalization_alerts()
            
            logger.debug(f"📈 Received metrics from central_monitor - CPU: {system_metrics.get('cpu', {}).get('percent', 0)}%")
            
        except Exception as e:
            logger.error(f"❌ Error processing central metrics: {e}")
    
    def _on_central_alert(self, alert_data: Dict[str, Any]):
        """دریافت آلرت از central_monitor"""
        try:
            # فقط لاگ کنیم، آلرت تکراری ایجاد نکنیم
            logger.info(f"📨 Received alert from central_monitor: {alert_data.get('title', 'No title')}")
        except Exception as e:
            logger.error(f"❌ Error processing central alert: {e}")
    
    def _start_minimal_monitoring(self):
        """راه‌اندازی مانیتورینگ حداقلی (فقط برای fallback)"""
        def minimal_monitor():
            while self._monitoring_active:
                try:
                    # فقط چک‌های ضروری هر 30 ثانیه
                    self._collect_minimal_metrics()
                    time.sleep(30)  # 30 ثانیه interval برای fallback
                except Exception as e:
                    logger.error(f"❌ Minimal monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=minimal_monitor, daemon=True)
        monitor_thread.start()
        logger.info("🔄 Minimal fallback monitoring started (30s interval)")
    
    def _collect_minimal_metrics(self):
        """جمع‌آوری حداقلی متریک‌ها"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            metric_obj = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0},
                active_connections=0,
                normalization_metrics=None
            )
            
            with self._lock:
                self.system_metrics_history.append(metric_obj)
            
            # چک هشدارهای حیاتی
            if cpu_percent > self.performance_thresholds['cpu_critical']:
                self._create_alert(
                    DebugLevel.CRITICAL,
                    f"Critical CPU usage in fallback mode: {cpu_percent:.1f}%",
                    "debug_manager_fallback",
                    {'cpu_usage': cpu_percent}
                )
            
        except Exception as e:
            logger.error(f"❌ Minimal metrics collection error: {e}")

    def _check_system_health_with_metrics(self, metrics: Dict[str, Any]):
        """بررسی سلامت سیستم با متریک‌های دریافتی"""
        try:
            cpu_usage = metrics.get('cpu', {}).get('percent', 0)
            memory_usage = metrics.get('memory', {}).get('percent', 0)
            
            # هشدار CPU
            if cpu_usage > self.performance_thresholds['cpu_critical']:
                self._create_alert(
                    DebugLevel.CRITICAL,
                    f"Critical CPU usage: {cpu_usage:.1f}%",
                    "debug_manager",
                    {'cpu_usage': cpu_usage, 'source': 'central_monitor'}
                )
            elif cpu_usage > self.performance_thresholds['cpu_warning']:
                self._create_alert(
                    DebugLevel.WARNING,
                    f"High CPU usage: {cpu_usage:.1f}%",
                    "debug_manager",
                    {'cpu_usage': cpu_usage, 'source': 'central_monitor'}
                )
            
            # هشدار Memory
            if memory_usage > self.performance_thresholds['memory_critical']:
                self._create_alert(
                    DebugLevel.CRITICAL,
                    f"Critical memory usage: {memory_usage:.1f}%",
                    "debug_manager",
                    {'memory_usage': memory_usage, 'source': 'central_monitor'}
                )
            elif memory_usage > self.performance_thresholds['memory_warning']:
                self._create_alert(
                    DebugLevel.WARNING,
                    f"High memory usage: {memory_usage:.1f}%",
                    "debug_manager",
                    {'memory_usage': memory_usage, 'source': 'central_monitor'}
                )
                
        except Exception as e:
            logger.error(f"❌ Error checking system health: {e}")
    
    def log_endpoint_call(self, endpoint: str, method: str, params: Dict[str, Any], 
                         response_time: float, status_code: int, cache_used: bool, 
                         api_calls: int = 0, normalization_info: Dict[str, Any] = None):
        """ثبت فراخوانی اندپوینت با مدیریت خطای کامل"""
        try:
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
                normalization_info=normalization_info
            )
            
            with self._lock:
                self.endpoint_calls.append(call)
                
                # آپدیت آمار endpoint
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
                
                # آپدیت آمار نرمال‌سازی
                if normalization_info:
                    norm_stats = stats['normalization_stats']
                    norm_stats['total_normalized'] += 1
                    
                    if normalization_info.get('status') == 'error':
                        norm_stats['normalization_errors'] += 1
                    
                    quality_score = normalization_info.get('quality_score', 0)
                    current_avg = norm_stats['avg_quality_score']
                    total_norm = norm_stats['total_normalized']
                    if total_norm > 0:
                        norm_stats['avg_quality_score'] = (current_avg * (total_norm - 1) + quality_score) / total_norm
                    
                    structure = normalization_info.get('detected_structure', 'unknown')
                    norm_stats['common_structures'][structure] = norm_stats['common_structures'].get(structure, 0) + 1
            
            # بررسی هشدارهای performance
            self._check_performance_alerts(endpoint, call)
            
            logger.debug(f"📊 Endpoint logged: {endpoint} - {response_time:.3f}s - Status: {status_code}")
            
        except Exception as e:
            logger.error(f"❌ Error logging endpoint call for {endpoint}: {e}")
            self._create_internal_alert(
                DebugLevel.ERROR,
                f"Endpoint logging failed: {str(e)}",
                "debug_manager",
                {"endpoint": endpoint, "error": str(e)}
            )
    
    def _check_performance_alerts(self, endpoint: str, call: EndpointCall):
        """بررسی هشدارهای performance برای endpoint"""
        try:
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
            
            if call.cpu_impact > self.performance_thresholds['cpu_critical']:
                self._create_alert(
                    level=DebugLevel.CRITICAL,
                    message=f"Critical CPU impact in {endpoint}: {call.cpu_impact:.1f}%",
                    source=endpoint,
                    data={'cpu_impact': call.cpu_impact}
                )
        
            if call.normalization_info and call.normalization_info.get('status') == 'error':
                self._create_alert(
                    level=DebugLevel.ERROR,
                    message=f"Normalization error in {endpoint}: {call.normalization_info.get('error', 'Unknown error')}",
                    source=endpoint,
                    data=call.normalization_info
                )
                
        except Exception as e:
            logger.error(f"❌ Error checking performance alerts for {endpoint}: {e}")
    
    def _check_normalization_alerts(self):
        """بررسی هشدارهای نرمال‌سازی"""
        try:
            metrics = data_normalizer.get_health_metrics()
            
            # هشدار برای نرخ موفقیت پایین
            if metrics.success_rate < self.performance_thresholds['normalization_success_threshold']:
                self._create_alert(
                    level=DebugLevel.WARNING,
                    message=f"Low normalization success rate: {metrics.success_rate}%",
                    source="data_normalizer",
                    data={
                        'success_rate': metrics.success_rate,
                        'total_processed': metrics.total_processed,
                        'total_errors': metrics.total_errors,
                        'threshold': self.performance_thresholds['normalization_success_threshold']
                    }
                )
            
            # هشدار برای خطاهای زیاد
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
    
    def _create_alert(self, level: DebugLevel, message: str, source: str, data: Dict[str, Any]):
        """ایجاد هشدار جدید با مدیریت کامل خطا"""
        try:
            alert = {
                'id': len(self.alerts) + 1,
                'level': level.value,
                'message': message,
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'acknowledged': False,
                'sent_to_alert_manager': False
            }
            
            with self._lock:
                self.alerts.append(alert)
            
            # ارسال به alert_manager اگر تنظیم شده
            if self.alert_manager:
                self._send_to_alert_manager(level, message, source, data)
                alert['sent_to_alert_manager'] = True
            logger.warning(f"🚨 {level.value} Alert: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")
    
    def _send_to_alert_manager(self, level: DebugLevel, message: str, source: str, data: Dict[str, Any]):
        """ارسال هشدار به alert_manager با مدیریت خطای پیشرفته"""
        try:
            if not self._alert_integration_enabled or not self.alert_manager:
                return  # بدون خطا - فقط اگر انتگراسیون غیرفعال است
            
            # نگاشت DebugLevel به AlertLevel
            level_mapping = {
                DebugLevel.INFO: AlertLevel.INFO,
                DebugLevel.WARNING: AlertLevel.WARNING,
                DebugLevel.ERROR: AlertLevel.ERROR,
                DebugLevel.CRITICAL: AlertLevel.CRITICAL
            }
        
            # نگاشت منبع به نوع هشدار
            type_mapping = {
                "data_normalizer": AlertType.SYSTEM,
                "system_monitor": AlertType.SYSTEM,
                "debug_manager": AlertType.SYSTEM
            }
        
            alert_level = level_mapping.get(level)
            if not alert_level:
                logger.warning(f"⚠️ Unknown debug level for alert mapping: {level}")
                return
            
            alert_type = type_mapping.get(source, AlertType.PERFORMANCE)
        
            # ارسال هشدار
            self.alert_manager.create_alert(
                level=alert_level,
                alert_type=alert_type,
                title=f"{alert_level.value} Alert from {source}",
                message=message,
                source=source,
                data=data
            )
        
            logger.debug(f"📨 Alert sent to AlertManager: {message}")
        
        except Exception as e:
            logger.error(f"❌ Error sending to alert manager: {e}")
            # غیرفعال کردن انتگراسیون در صورت خطای مکرر
            self._alert_integration_enabled = False
        
    def _create_internal_alert(self, level: DebugLevel, message: str, source: str, data: Dict[str, Any]):
        """ایجاد هشدار داخلی بدون ارسال به alert_manager"""
        try:
            alert = {
                'id': len(self.alerts) + 1,
                'level': level.value,
                'message': message,
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'acknowledged': False
            }
            
            with self._lock:
                self.alerts.append(alert)
            
            logger.warning(f"🚨 {level.value} Alert: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create internal alert: {e}")
    
    def get_endpoint_stats(self, endpoint: str = None) -> Dict[str, Any]:
        """دریافت آمار اندپوینت با محاسبات ایمن"""
        try:
            with self._lock:
                if endpoint:
                    if endpoint not in self.endpoint_stats:
                        return {'error': 'Endpoint not found'}
                    
                    stats = self.endpoint_stats[endpoint]
                    avg_response_time = (stats['total_response_time'] / stats['total_calls']) if stats['total_calls'] > 0 else 0
                    
                    norm_stats = stats['normalization_stats']
                    normalization_success_rate = (
                        ((norm_stats['total_normalized'] - norm_stats['normalization_errors']) / norm_stats['total_normalized'] * 100) 
                        if norm_stats['total_normalized'] > 0 else 0
                    )
                    
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
                            'hit_rate': (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) 
                            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                        },
                        'api_calls': stats['api_calls'],
                        'normalization_performance': {
                            'total_normalized': norm_stats['total_normalized'],
                            'normalization_errors': norm_stats['normalization_errors'],
                            'success_rate': round(normalization_success_rate, 2),
                            'avg_quality_score': round(norm_stats['avg_quality_score'], 2),
                            'common_structures': norm_stats['common_structures']
                        },
                        'recent_errors': stats['errors'][-10:],
                        'last_call': stats['last_call']
                    }
                else:
                    return self._get_all_endpoints_stats()
                    
        except Exception as e:
            logger.error(f"❌ Error getting endpoint stats: {e}")
            return {'error': f'Failed to get stats: {str(e)}'}
    
    def _get_all_endpoints_stats(self) -> Dict[str, Any]:
        """محاسبه آمار کلی تمام endpointها"""
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
                'normalization_success_rate': (
                    ((norm_stats['total_normalized'] - norm_stats['normalization_errors']) / norm_stats['total_normalized'] * 100) 
                    if norm_stats['total_normalized'] > 0 else 0
                ),
                'last_call': stats['last_call']
            }
            total_calls += stats['total_calls']
            total_success += stats['successful_calls']
        
        try:
            overall_norm_metrics = data_normalizer.get_health_metrics()
        except Exception as e:
            logger.warning(f"⚠️ Could not get data normalizer metrics: {e}")
            overall_norm_metrics = None
        
        return {
            'overall': {
                'total_endpoints': len(self.endpoint_stats),
                'total_calls': total_calls,
                'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0,
                'normalization_overview': {
                    'total_normalized': total_normalized,
                    'normalization_errors': total_norm_errors,
                    'normalization_success_rate': (
                        ((total_normalized - total_norm_errors) / total_normalized * 100) 
                        if total_normalized > 0 else 0
                    ),
                    'system_success_rate': overall_norm_metrics.success_rate if overall_norm_metrics else 0,
                    'common_structures': overall_norm_metrics.common_structures if overall_norm_metrics else {}
                },
                'timestamp': datetime.now().isoformat()
            },
            'endpoints': all_stats
        }
    
    def get_recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """دریافت آخرین فراخوانی‌ها"""
        try:
            with self._lock:
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
                        'normalization_info': call.normalization_info
                    }
                    for call in recent_calls
                ]
        except Exception as e:
            logger.error(f"❌ Error getting recent calls: {e}")
            return []
    
    def get_system_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌های سیستم"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self._lock:
                metrics_history = [
                    metrics for metrics in self.system_metrics_history
                    if metrics.timestamp >= cutoff_time
                ]
            
            current_norm_metrics = data_normalizer.get_health_metrics()
            
            return [
                {
                    'timestamp': metrics.timestamp.isoformat(),
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'disk_usage': metrics.disk_usage,
                    'network_io': metrics.network_io,
                    'active_connections': metrics.active_connections,
                    'normalization_metrics': {
                        'success_rate': current_norm_metrics.success_rate,
                        'total_processed': current_norm_metrics.total_processed,
                        'data_quality': current_norm_metrics.data_quality
                    } if metrics.normalization_metrics is None else metrics.normalization_metrics
                }
                for metrics in metrics_history
            ]
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return []
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        with self._lock:
            return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id: int):
        """تأیید هشدار"""
        with self._lock:
            for alert in self.alerts:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    logger.info(f"✅ Alert {alert_id} acknowledged")
                    break
    
    def get_system_status(self) -> Dict[str, Any]:
        """دریافت وضعیت کامل سیستم"""
        try:
            endpoint_stats = self.get_endpoint_stats()
            recent_calls = self.get_recent_calls(10)
            system_metrics = self.get_system_metrics_history(1)
            active_alerts = self.get_active_alerts()
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'overview': {
                    'total_endpoints': endpoint_stats.get('overall', {}).get('total_endpoints', 0),
                    'total_calls': endpoint_stats.get('overall', {}).get('total_calls', 0),
                    'success_rate': endpoint_stats.get('overall', {}).get('overall_success_rate', 0),
                    'active_alerts': len(active_alerts),
                    'system_uptime': self._get_system_uptime()
                },
                'performance': {
                    'cpu_usage': system_metrics[-1]['cpu_percent'] if system_metrics else 0,
                    'memory_usage': system_metrics[-1]['memory_percent'] if system_metrics else 0,
                    'disk_usage': system_metrics[-1]['disk_usage'] if system_metrics else 0,
                    'normalization_success_rate': endpoint_stats.get('overall', {}).get('normalization_overview', {}).get('system_success_rate', 0)
                },
                'recent_activity': {
                    'calls': recent_calls,
                    'alerts': active_alerts[:5]  # 5 هشدار آخر
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_system_uptime(self) -> str:
        """محاسبه uptime سیستم"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        except:
            return "unknown"
    
    def clear_old_data(self, days: int = 7):
        """پاک کردن داده‌های قدیمی"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with self._lock:
                self.endpoint_calls = deque(
                    [call for call in self.endpoint_calls if call.timestamp > cutoff_time],
                    maxlen=10000
                )
                
                self.system_metrics_history = deque(
                    [metrics for metrics in self.system_metrics_history if metrics.timestamp > cutoff_time],
                    maxlen=1000
                )
                
                # پاک کردن هشدارهای قدیمی
                self.alerts = [alert for alert in self.alerts 
                             if datetime.fromisoformat(alert['timestamp']) > cutoff_time]
            
            logger.info(f"🧹 Cleared data older than {days} days")
            
        except Exception as e:
            logger.error(f"❌ Error clearing old data: {e}")
    
    def stop_monitoring(self):
        """توقف مانیتورینگ"""
        self._monitoring_active = False
        logger.info("🛑 Debug monitoring stopped")
    
    def __del__(self):
        """دمارکتور برای توقف تمیز"""
        self.stop_monitoring()

# ایجاد نمونه گلوبال
debug_manager = DebugManager()

"""
مانیتورینگ یکپارچه عملکرد سیستم و APIها
نسخه ادغام‌شده از Performance Monitor و API Performance Monitor
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import statistics
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class PerformanceGrade(Enum):
    """گریدهای عملکرد"""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"

@dataclass
class EndpointMetrics:
    """متریک‌های یکپارچه endpoint"""
    endpoint: str
    method: str = "GET"
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    response_times: List[float] = None
    success_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    normalization_success_rate: float = 100.0
    normalization_quality_score: float = 100.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class ExternalService:
    """اطلاعات سرویس خارجی"""
    name: str
    url: str
    status: str = "unknown"
    last_check: datetime = None
    latency_ms: float = None
    check_interval: int = 60
    uptime_percentage: float = 100.0

class PerformanceMonitor:
    """
    سیستم مانیتورینگ یکپارچه عملکرد:
    - مانیتورینگ سیستم و منابع
    - تحلیل عملکرد APIها
    - ردیابی سرویس‌های خارجی
    - مدیریت SLA و پیش‌بینی
    - تحلیل bottlenecks و بهینه‌سازی
    """
    
    def __init__(self, debug_manager, alert_manager, metrics_collector=None):
        self.debug_manager = debug_manager
        self.alert_manager = alert_manager
        self.metrics_collector = metrics_collector
        
        # ذخیره‌سازی متریک‌ها
        self.endpoint_metrics = {}  # endpoint -> EndpointMetrics
        self.response_times_history = defaultdict(lambda: deque(maxlen=1000))
        self.performance_history = deque(maxlen=1000)
        
        # مانیتورینگ سرویس‌های خارجی
        self.external_services = {}
        
        # الگوهای دسترسی
        self.access_patterns = defaultdict(lambda: defaultdict(int))
        
        # thresholds یکپارچه
        self.thresholds = {
            # زمان پاسخ (ثانیه)
            'response_time': {
                'excellent': 0.5,      # زیر 500ms
                'good': 1.0,           # زیر 1s
                'fair': 2.0,           # زیر 2s
                'poor': 3.0,           # زیر 3s
                'critical': 5.0        # بالای 5s
            },
            # زمان پاسخ API (میلی‌ثانیه)
            'api_response_time': {
                'warning': 500,        # 500ms
                'critical': 2000       # 2s
            },
            # نرخ خطا (درصد)
            'error_rate': {
                'warning': 5.0,        # 5%
                'critical': 10.0       # 10%
            },
            # throughput (درخواست در دقیقه)
            'throughput': {
                'warning': 1000,       # 1000 req/min
                'critical': 5000       # 5000 req/min
            },
            # منابع سیستم
            'system': {
                'cpu_warning': 80.0,
                'cpu_critical': 95.0,
                'memory_warning': 85.0,
                'memory_critical': 95.0,
                'disk_warning': 90.0,
                'disk_critical': 98.0
            },
            # نرمال‌سازی
            'normalization': {
                'success_warning': 90.0,    # 90%
                'success_critical': 80.0,   # 80%
                'quality_warning': 80.0,    # 80%
                'quality_critical': 60.0    # 60%
            }
        }
        
        # SLA targets
        self.sla_targets = {
            'uptime': 99.9,      # 99.9%
            'response_time': 200, # 200ms
            'error_rate': 0.1     # 0.1%
        }
        
        # اتصال به central_monitor
        self._connect_to_central_monitor()
        
        logger.info("✅ Performance Monitor Initialized - Unified Version")
    
    def _connect_to_central_monitor(self):
        """اتصال به central_monitor برای دریافت متریک‌های real-time"""
        try:
            from .system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت متریک‌های سیستم
                central_monitor.subscribe("performance_monitor", self._on_system_metrics_received)
                logger.info("✅ PerformanceMonitor subscribed to central_monitor")
            else:
                logger.warning("⚠️ Central monitor not available")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor")
        except Exception as e:
            logger.error(f"❌ Error connecting to central_monitor: {e}")
    
    def _on_system_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌های سیستم از central_monitor"""
        try:
            system_metrics = metrics.get('system', {})
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            
            # بررسی performance thresholds
            self._check_system_performance(cpu_usage, memory_usage)
            
            # ذخیره در تاریخچه
            self.performance_history.append({
                'timestamp': datetime.now(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'source': 'central_monitor'
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing system metrics: {e}")
    
    def _check_system_performance(self, cpu_usage: float, memory_usage: float):
        """بررسی performance سیستم"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            # بررسی CPU
            if cpu_usage > self.thresholds['system']['cpu_critical']:
                self._create_performance_alert(
                    AlertLevel.CRITICAL,
                    AlertType.PERFORMANCE,
                    "Critical CPU Performance",
                    f"CPU usage critically high: {cpu_usage:.1f}% - System performance degraded",
                    "performance_monitor",
                    {'cpu_usage': cpu_usage, 'threshold': self.thresholds['system']['cpu_critical']}
                )
            elif cpu_usage > self.thresholds['system']['cpu_warning']:
                self._create_performance_alert(
                    AlertLevel.WARNING,
                    AlertType.PERFORMANCE,
                    "High CPU Usage",
                    f"CPU usage high: {cpu_usage:.1f}% - Monitor system performance",
                    "performance_monitor",
                    {'cpu_usage': cpu_usage, 'threshold': self.thresholds['system']['cpu_warning']}
                )
            
            # بررسی Memory
            if memory_usage > self.thresholds['system']['memory_critical']:
                self._create_performance_alert(
                    AlertLevel.CRITICAL,
                    AlertType.PERFORMANCE,
                    "Critical Memory Performance",
                    f"Memory usage critically high: {memory_usage:.1f}% - System performance degraded",
                    "performance_monitor",
                    {'memory_usage': memory_usage, 'threshold': self.thresholds['system']['memory_critical']}
                )
            elif memory_usage > self.thresholds['system']['memory_warning']:
                self._create_performance_alert(
                    AlertLevel.WARNING,
                    AlertType.PERFORMANCE,
                    "High Memory Usage",
                    f"Memory usage high: {memory_usage:.1f}% - Monitor system performance",
                    "performance_monitor",
                    {'memory_usage': memory_usage, 'threshold': self.thresholds['system']['memory_warning']}
                )
                
        except Exception as e:
            logger.error(f"❌ Error checking system performance: {e}")
    
    def _create_performance_alert(self, level, alert_type, title, message, source, data):
        """ایجاد آلرت performance"""
        try:
            alert_result = self.alert_manager.create_alert(
                level=level,
                alert_type=alert_type,
                title=title,
                message=message,
                source=source,
                data=data
            )
            
            if alert_result:
                logger.info(f"⚡ Performance alert created: {title}")
            
        except Exception as e:
            logger.error(f"❌ Error creating performance alert: {e}")
    
    # ==================== API و Endpoint Management ====================
    
    def register_endpoint(self, endpoint: str, method: str = "GET", 
                         expected_response_time: int = 200,
                         monitoring_type: str = "api") -> bool:
        """ثبت endpoint برای مانیتورینگ"""
        if endpoint not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint] = EndpointMetrics(
                endpoint=endpoint,
                method=method,
                last_updated=datetime.now()
            )
            
            # ثبت در metrics_collector اگر موجود باشد
            if self.metrics_collector and hasattr(self.metrics_collector, 'register_api_endpoint'):
                try:
                    self.metrics_collector.register_api_endpoint(endpoint, expected_response_time)
                except Exception as e:
                    logger.warning(f"⚠️ Could not register endpoint in metrics collector: {e}")
            
            logger.info(f"✅ Endpoint registered: {method} {endpoint} ({monitoring_type})")
            return True
        return False
    
    def record_request(self, endpoint: str, response_time_ms: float, 
                      status_code: int, method: str = "GET",
                      user_agent: str = "", ip_address: str = "",
                      cache_used: bool = False, api_calls: int = 0,
                      normalization_info: Dict = None):
        """ثبت درخواست با تمام جزئیات"""
        # ثبت در endpoint metrics
        if endpoint not in self.endpoint_metrics:
            self.register_endpoint(endpoint, method)
        
        metrics = self.endpoint_metrics[endpoint]
        response_time_sec = response_time_ms / 1000.0
        
        # آپدیت متریک‌ها
        metrics.total_requests += 1
        metrics.total_response_time += response_time_sec
        metrics.response_times.append(response_time_sec)
        
        if 200 <= status_code < 400:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        metrics.min_response_time = min(metrics.min_response_time, response_time_sec)
        metrics.max_response_time = max(metrics.max_response_time, response_time_sec)
        
        if cache_used:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1
        
        metrics.api_calls += api_calls
        
        if normalization_info:
            metrics.normalization_success_rate = normalization_info.get('success_rate', 100)
            metrics.normalization_quality_score = normalization_info.get('quality_score', 100)
        
        metrics.success_rate = (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
        metrics.last_updated = datetime.now()
        
        # ذخیره در تاریخچه زمان پاسخ
        self.response_times_history[endpoint].append(response_time_ms)
        
        # ثبت در metrics_collector اگر موجود باشد
        if self.metrics_collector and hasattr(self.metrics_collector, 'record_api_request'):
            try:
                self.metrics_collector.record_api_request(
                    endpoint=endpoint,
                    latency_ms=response_time_ms,
                    status_code=status_code,
                    user_agent=user_agent
                )
            except Exception as e:
                logger.debug(f"⚠️ Could not record request in metrics collector: {e}")
        
        # ثبت الگوی دسترسی
        hour = datetime.now().hour
        self.access_patterns[endpoint][hour] += 1
        
        # بررسی performance
        self._check_endpoint_performance(endpoint, response_time_ms, status_code, metrics)
        
        return True
    
    def _check_endpoint_performance(self, endpoint: str, response_time_ms: float, 
                                   status_code: int, metrics: EndpointMetrics):
        """بررسی عملکرد endpoint"""
        # بررسی زمان پاسخ
        if response_time_ms > self.thresholds['api_response_time']['critical']:
            self._trigger_endpoint_alert(
                endpoint=endpoint,
                alert_type="response_time_critical",
                severity="critical",
                message=f"Critical response time: {response_time_ms}ms",
                metric_value=response_time_ms,
                threshold=self.thresholds['api_response_time']['critical']
            )
        elif response_time_ms > self.thresholds['api_response_time']['warning']:
            self._trigger_endpoint_alert(
                endpoint=endpoint,
                alert_type="response_time_warning",
                severity="warning",
                message=f"High response time: {response_time_ms}ms",
                metric_value=response_time_ms,
                threshold=self.thresholds['api_response_time']['warning']
            )
        
        # بررسی نرخ خطا
        error_rate = (metrics.failed_requests / metrics.total_requests) * 100 if metrics.total_requests > 0 else 0
        if error_rate > self.thresholds['error_rate']['critical']:
            self._trigger_endpoint_alert(
                endpoint=endpoint,
                alert_type="error_rate_critical",
                severity="critical",
                message=f"Critical error rate: {error_rate:.1f}%",
                metric_value=error_rate,
                threshold=self.thresholds['error_rate']['critical']
            )
        elif error_rate > self.thresholds['error_rate']['warning']:
            self._trigger_endpoint_alert(
                endpoint=endpoint,
                alert_type="error_rate_warning",
                severity="warning",
                message=f"High error rate: {error_rate:.1f}%",
                metric_value=error_rate,
                threshold=self.thresholds['error_rate']['warning']
            )
        
        # بررسی نرمال‌سازی
        if metrics.normalization_success_rate < self.thresholds['normalization']['success_critical']:
            self._trigger_endpoint_alert(
                endpoint=endpoint,
                alert_type="normalization_critical",
                severity="critical",
                message=f"Critical normalization success rate: {metrics.normalization_success_rate:.1f}%",
                metric_value=metrics.normalization_success_rate,
                threshold=self.thresholds['normalization']['success_critical']
            )
        elif metrics.normalization_success_rate < self.thresholds['normalization']['success_warning']:
            self._trigger_endpoint_alert(
                endpoint=endpoint,
                alert_type="normalization_warning",
                severity="warning",
                message=f"Low normalization success rate: {metrics.normalization_success_rate:.1f}%",
                metric_value=metrics.normalization_success_rate,
                threshold=self.thresholds['normalization']['success_warning']
            )
    
    def _trigger_endpoint_alert(self, endpoint: str, alert_type: str, severity: str, 
                               message: str, metric_value: float, threshold: float):
        """ایجاد هشدار برای endpoint"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            level = AlertLevel.CRITICAL if severity == "critical" else AlertLevel.WARNING
            
            self.alert_manager.create_alert(
                level=level,
                alert_type=AlertType.PERFORMANCE,
                title=f"API Performance Alert: {endpoint}",
                message=f"{message}. Threshold: {threshold}",
                source="performance_monitor",
                data={
                    'endpoint': endpoint,
                    'alert_type': alert_type,
                    'metric_value': metric_value,
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.warning(f"🚨 API Performance Alert: {endpoint} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Error triggering API performance alert: {e}")
    
    # ==================== تحلیل عملکرد ====================
    
    def analyze_endpoint_performance(self, endpoint: str = None, include_sla: bool = True) -> Dict[str, Any]:
        """آنالیز عملکرد اندپوینت"""
        try:
            if endpoint:
                return self._analyze_single_endpoint(endpoint, include_sla)
            else:
                return self._analyze_all_endpoints(include_sla)
        except Exception as e:
            logger.error(f"❌ Error in analyze_endpoint_performance: {e}")
            return self._get_empty_performance_response()
    
    def _analyze_single_endpoint(self, endpoint: str, include_sla: bool) -> Dict[str, Any]:
        """آنالیز عملکرد یک اندپوینت خاص"""
        try:
            if endpoint not in self.endpoint_metrics:
                # سعی کن از debug_manager بگیر
                debug_stats = self.debug_manager.get_endpoint_stats(endpoint)
                if 'error' in debug_stats:
                    return {'error': f'Endpoint {endpoint} not found'}
                metrics = self._convert_debug_stats_to_metrics(endpoint, debug_stats)
            else:
                metrics = self.endpoint_metrics[endpoint]
            
            performance_grade = self._calculate_performance_grade(metrics)
            bottlenecks = self._identify_bottlenecks(metrics)
            
            result = {
                'endpoint': endpoint,
                'performance_grade': performance_grade.value,
                'metrics': {
                    'average_response_time': metrics.total_response_time / metrics.total_requests if metrics.total_requests > 0 else 0,
                    'success_rate': metrics.success_rate,
                    'cache_hit_rate': (metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) * 100) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0,
                    'total_calls': metrics.total_requests,
                    'api_calls_per_request': metrics.api_calls / metrics.total_requests if metrics.total_requests > 0 else 0,
                    'normalization_success_rate': metrics.normalization_success_rate,
                    'normalization_quality_score': metrics.normalization_quality_score
                },
                'bottlenecks': bottlenecks,
                'recommendations': self._generate_recommendations(metrics, bottlenecks),
                'last_updated': metrics.last_updated.isoformat() if metrics.last_updated else datetime.now().isoformat()
            }
            
            if include_sla and metrics.total_requests > 0:
                result['sla_analysis'] = self._analyze_sla_compliance(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing single endpoint {endpoint}: {e}")
            return {
                'endpoint': endpoint,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def _analyze_all_endpoints(self, include_sla: bool) -> Dict[str, Any]:
        """آنالیز عملکرد تمام اندپوینت‌ها"""
        try:
            if not self.endpoint_metrics:
                # از debug_manager بگیر
                debug_stats = self.debug_manager.get_endpoint_stats()
                if 'endpoints' in debug_stats and debug_stats['endpoints']:
                    for ep, stats in debug_stats['endpoints'].items():
                        if ep not in self.endpoint_metrics:
                            self.endpoint_metrics[ep] = self._convert_debug_stats_to_metrics(ep, stats)
            
            if not self.endpoint_metrics:
                return self._get_empty_performance_response()
            
            performance_report = {}
            response_times = []
            success_rates = []
            
            for endpoint, metrics in self.endpoint_metrics.items():
                try:
                    avg_response_time = metrics.total_response_time / metrics.total_requests if metrics.total_requests > 0 else 0
                    performance_grade = self._calculate_performance_grade(metrics)
                    
                    performance_report[endpoint] = {
                        'performance_grade': performance_grade.value,
                        'average_response_time': avg_response_time,
                        'success_rate': metrics.success_rate,
                        'cache_hit_rate': (metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) * 100) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0,
                        'total_calls': metrics.total_requests,
                        'normalization_success_rate': metrics.normalization_success_rate
                    }
                    
                    response_times.append(avg_response_time)
                    success_rates.append(metrics.success_rate)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing endpoint {endpoint}: {e}")
                    continue
            
            # محاسبه SLA کلی
            overall_sla = None
            if include_sla and self.endpoint_metrics:
                overall_sla = self._calculate_overall_sla()
            
            return {
                'overall_performance': {
                    'average_response_time': statistics.mean(response_times) if response_times else 0,
                    'median_response_time': statistics.median(response_times) if response_times else 0,
                    'min_response_time': min(response_times) if response_times else 0,
                    'max_response_time': max(response_times) if response_times else 0,
                    'average_success_rate': statistics.mean(success_rates) if success_rates else 0,
                    'total_endpoints': len(self.endpoint_metrics),
                    'active_endpoints': len([ep for ep in performance_report.values() if ep['total_calls'] > 0])
                },
                'sla_compliance': overall_sla,
                'endpoint_performance': performance_report,
                'performance_distribution': self._calculate_performance_distribution(performance_report),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing all endpoints: {e}")
            return self._get_empty_performance_response()
    
    def _get_empty_performance_response(self) -> Dict[str, Any]:
        """پاسخ خالی وقتی داده‌ای وجود ندارد"""
        return {
            'overall_performance': {
                'average_response_time': 0,
                'median_response_time': 0,
                'min_response_time': 0,
                'max_response_time': 0,
                'average_success_rate': 0,
                'total_endpoints': 0,
                'active_endpoints': 0
            },
            'endpoint_performance': {},
            'performance_distribution': {'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
            'timestamp': datetime.now().isoformat(),
            'message': 'No endpoint data available for analysis'
        }
    
    def _calculate_performance_grade(self, metrics: EndpointMetrics) -> PerformanceGrade:
        """محاسبه گرید عملکرد"""
        try:
            if metrics.total_requests == 0:
                return PerformanceGrade.F
            
            avg_response_time = metrics.total_response_time / metrics.total_requests
            score = 0
            
            # امتیاز زمان پاسخ (35%)
            if avg_response_time <= self.thresholds['response_time']['excellent']:
                score += 35
            elif avg_response_time <= self.thresholds['response_time']['good']:
                score += 28
            elif avg_response_time <= self.thresholds['response_time']['fair']:
                score += 21
            elif avg_response_time <= self.thresholds['response_time']['poor']:
                score += 14
            else:
                score += 7
            
            # امتیاز نرخ موفقیت (25%)
            score += (metrics.success_rate / 100) * 25
            
            # امتیاز نرخ کش (15%)
            cache_total = metrics.cache_hits + metrics.cache_misses
            cache_hit_rate = (metrics.cache_hits / cache_total * 100) if cache_total > 0 else 0
            score += (cache_hit_rate / 100) * 15
            
            # امتیاز نرمال‌سازی (25%)
            norm_score = (metrics.normalization_success_rate / 100) * 12.5 + (metrics.normalization_quality_score / 100) * 12.5
            score += norm_score
            
            # تعیین گرید
            if score >= 90:
                return PerformanceGrade.A_PLUS
            elif score >= 80:
                return PerformanceGrade.A
            elif score >= 70:
                return PerformanceGrade.B
            elif score >= 60:
                return PerformanceGrade.C
            elif score >= 50:
                return PerformanceGrade.D
            else:
                return PerformanceGrade.F
                
        except Exception as e:
            logger.error(f"❌ Error calculating performance grade: {e}")
            return PerformanceGrade.F
    
    def _identify_bottlenecks(self, metrics: EndpointMetrics) -> List[Dict[str, Any]]:
        """شناسایی bottlenecks"""
        bottlenecks = []
        
        try:
            if metrics.total_requests == 0:
                return bottlenecks
            
            avg_response_time = metrics.total_response_time / metrics.total_requests
            
            # بررسی زمان پاسخ
            if avg_response_time > self.thresholds['response_time']['critical']:
                bottlenecks.append({
                    'type': 'response_time',
                    'severity': 'critical',
                    'message': f'Response time {avg_response_time:.2f}s is unacceptable',
                    'suggestion': 'Optimize database queries or implement caching'
                })
            elif avg_response_time > self.thresholds['response_time']['poor']:
                bottlenecks.append({
                    'type': 'response_time',
                    'severity': 'high',
                    'message': f'Response time {avg_response_time:.2f}s is poor',
                    'suggestion': 'Consider query optimization or adding indexes'
                })
            
            # بررسی نرخ موفقیت
            if metrics.success_rate < 95:
                bottlenecks.append({
                    'type': 'reliability',
                    'severity': 'high' if metrics.success_rate < 90 else 'medium',
                    'message': f'Success rate {metrics.success_rate:.1f}% is below target',
                    'suggestion': 'Investigate error patterns and improve error handling'
                })
            
            # بررسی کارایی کش
            cache_total = metrics.cache_hits + metrics.cache_misses
            if cache_total > 0:
                cache_hit_rate = (metrics.cache_hits / cache_total * 100)
                if cache_hit_rate < 50 and metrics.total_requests > 10:
                    bottlenecks.append({
                        'type': 'caching',
                        'severity': 'medium',
                        'message': f'Cache hit rate {cache_hit_rate:.1f}% is low',
                        'suggestion': 'Review cache strategy and TTL settings'
                    })
            
            # بررسی نرمال‌سازی
            if metrics.normalization_success_rate < 90:
                bottlenecks.append({
                    'type': 'normalization_reliability',
                    'severity': 'high' if metrics.normalization_success_rate < 80 else 'medium',
                    'message': f'Normalization success rate {metrics.normalization_success_rate:.1f}% is low',
                    'suggestion': 'Review data normalization rules and error handling'
                })
            
            # بررسی فراخوانی‌های API
            api_calls_ratio = metrics.api_calls / metrics.total_requests if metrics.total_requests > 0 else 0
            if api_calls_ratio > 3:
                bottlenecks.append({
                    'type': 'external_dependencies',
                    'severity': 'medium',
                    'message': f'High API calls per request: {api_calls_ratio:.1f}',
                    'suggestion': 'Implement request batching or caching for external APIs'
                })
            
        except Exception as e:
            logger.error(f"❌ Error identifying bottlenecks: {e}")
        
        return bottlenecks
    
    def _generate_recommendations(self, metrics: EndpointMetrics, bottlenecks: List[Dict]) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        
        try:
            if metrics.total_requests == 0:
                return recommendations
            
            avg_response_time = metrics.total_response_time / metrics.total_requests
            
            # توصیه‌های عمومی
            if avg_response_time > 1.0:
                recommendations.append("Implement response compression")
                recommendations.append("Consider using a CDN for static assets")
            
            cache_total = metrics.cache_hits + metrics.cache_misses
            if cache_total > 0:
                cache_hit_rate = (metrics.cache_hits / cache_total * 100)
                if cache_hit_rate < 60:
                    recommendations.append("Increase cache TTL for frequently accessed data")
                    recommendations.append("Implement cache warming for hot paths")
            
            if metrics.total_requests > 1000:
                recommendations.append("Consider implementing rate limiting")
                recommendations.append("Add database connection pooling")
            
            # توصیه‌های مبتنی بر bottlenecks
            for bottleneck in bottlenecks:
                if bottleneck['type'] in ['response_time', 'caching']:
                    recommendations.append(bottleneck['suggestion'])
            
            # حذف موارد تکراری
            return list(set(recommendations))
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return []
    
    def _calculate_performance_distribution(self, performance_report: Dict) -> Dict[str, int]:
        """محاسبه توزیع عملکرد"""
        distribution = {
            'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0
        }
        
        for endpoint_data in performance_report.values():
            grade = endpoint_data['performance_grade']
            distribution[grade] += 1
        
        return distribution
    
    # ==================== External Services Management ====================
    
    def register_external_service(self, service_name: str, check_url: str, 
                                 check_interval: int = 60) -> bool:
        """ثبت سرویس خارجی برای مانیتورینگ"""
        try:
            self.external_services[service_name] = ExternalService(
                name=service_name,
                url=check_url,
                check_interval=check_interval
            )
            
            # ثبت در metrics_collector اگر موجود باشد
            if self.metrics_collector and hasattr(self.metrics_collector, 'register_external_service'):
                try:
                    self.metrics_collector.register_external_service(service_name, check_url, check_interval)
                except Exception as e:
                    logger.warning(f"⚠️ Could not register external service in metrics collector: {e}")
            
            logger.info(f"✅ External service registered: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registering external service: {e}")
            return False
    
    async def check_external_service(self, service_name: str) -> Dict[str, Any]:
        """بررسی سلامت سرویس خارجی"""
        if service_name not in self.external_services:
            return {'error': f'Service {service_name} not registered'}
        
        service = self.external_services[service_name]
        
        try:
            import aiohttp
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(service.url) as response:
                    latency = (time.time() - start_time) * 1000
                    status = 'healthy' if response.status == 200 else 'unhealthy'
                    
                    service.status = status
                    service.last_check = datetime.now()
                    service.latency_ms = round(latency, 2)
                    
                    return {
                        'service': service_name,
                        'status': status,
                        'latency_ms': service.latency_ms,
                        'last_check': service.last_check.isoformat(),
                        'status_code': response.status
                    }
                    
        except Exception as e:
            service.status = 'unhealthy'
            service.last_check = datetime.now()
            service.latency_ms = None
            
            return {
                'service': service_name,
                'status': 'unhealthy',
                'last_check': service.last_check.isoformat(),
                'error': str(e)
            }
    
    def get_external_service_health(self) -> Dict[str, Any]:
        """سلامت سرویس‌های خارجی"""
        healthy_count = sum(1 for s in self.external_services.values() if s.status == 'healthy')
        total_count = len(self.external_services)
        
        return {
            'total_services': total_count,
            'healthy_services': healthy_count,
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 100,
            'services': {
                name: {
                    'status': service.status,
                    'last_check': service.last_check.isoformat() if service.last_check else None,
                    'latency_ms': service.latency_ms,
                    'url': service.url
                }
                for name, service in self.external_services.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    # ==================== گزارش‌های پیشرفته ====================
    
    def get_slowest_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت کندترین اندپوینت‌ها"""
        endpoints_with_times = []
        
        for endpoint, metrics in self.endpoint_metrics.items():
            if metrics.total_requests > 0:
                avg_response_time = metrics.total_response_time / metrics.total_requests
                endpoints_with_times.append({
                    'endpoint': endpoint,
                    'average_response_time': avg_response_time,
                    'total_calls': metrics.total_requests,
                    'performance_grade': self._calculate_performance_grade(metrics).value
                })
        
        # مرتب‌سازی بر اساس زمان پاسخ
        sorted_endpoints = sorted(
            endpoints_with_times,
            key=lambda x: x['average_response_time'],
            reverse=True
        )
        
        return sorted_endpoints[:limit]
    
    def get_most_called_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت پرفراخوانی‌ترین اندپوینت‌ها"""
        endpoints_with_calls = []
        
        for endpoint, metrics in self.endpoint_metrics.items():
            endpoints_with_calls.append({
                'endpoint': endpoint,
                'total_calls': metrics.total_requests,
                'average_response_time': metrics.total_response_time / metrics.total_requests if metrics.total_requests > 0 else 0,
                'success_rate': metrics.success_rate,
                'performance_grade': self._calculate_performance_grade(metrics).value
            })
        
        # مرتب‌سازی بر اساس تعداد فراخوانی
        sorted_endpoints = sorted(
            endpoints_with_calls,
            key=lambda x: x['total_calls'],
            reverse=True
        )
        
        return sorted_endpoints[:limit]
    
    def analyze_latency_trends(self, endpoint: str = None, hours: int = 24) -> Dict[str, Any]:
        """تحلیل روند latency"""
        if endpoint:
            if endpoint not in self.response_times_history or not self.response_times_history[endpoint]:
                return {'error': 'No data available'}
            
            response_times = list(self.response_times_history[endpoint])
            
            return {
                'endpoint': endpoint,
                'data_points': len(response_times),
                'average': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'min': min(response_times),
                'max': max(response_times),
                'percentiles': self._calculate_percentiles(response_times),
                'distribution': self._create_latency_distribution(response_times),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # تحلیل کلی
            all_response_times = []
            for endpoint_times in self.response_times_history.values():
                all_response_times.extend(endpoint_times)
            
            if not all_response_times:
                return {'error': 'No data available'}
            
            return {
                'total_endpoints': len(self.response_times_history),
                'total_data_points': len(all_response_times),
                'overall_average': statistics.mean(all_response_times),
                'overall_median': statistics.median(all_response_times),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_peak_usage_patterns(self) -> Dict[str, Any]:
        """الگوهای استفاده پیک"""
        patterns = {}
        
        for endpoint, hourly_counts in self.access_patterns.items():
            if hourly_counts:
                peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
                patterns[endpoint] = {
                    'peak_hour': peak_hour[0],
                    'peak_requests': peak_hour[1],
                    'total_daily_requests': sum(hourly_counts.values()),
                    'busy_hours': [hour for hour, count in hourly_counts.items() if count > peak_hour[1] * 0.5]
                }
        
        # شناسایی ساعت‌های پیک کلی
        all_hourly_counts = defaultdict(int)
        for hourly_counts in self.access_patterns.values():
            for hour, count in hourly_counts.items():
                all_hourly_counts[hour] += count
        
        overall_peak = None
        if all_hourly_counts:
            overall_peak = max(all_hourly_counts.items(), key=lambda x: x[1])
        
        return {
            'endpoint_patterns': patterns,
            'overall_peak_hour': overall_peak[0] if overall_peak else None,
            'overall_peak_requests': overall_peak[1] if overall_peak else 0,
            'recommendations': self._generate_peak_usage_recommendations(patterns, all_hourly_counts)
        }
    
    def get_performance_forecast(self, endpoint: str, forecast_hours: int = 6) -> Dict[str, Any]:
        """پیش‌بینی عملکرد"""
        hourly_counts = self.access_patterns.get(endpoint, {})
        if not hourly_counts:
            return {'error': 'Insufficient data for forecasting'}
        
        # میانگین‌گیری ساده
        recent_hours = list(hourly_counts.values())[-24:]  # 24 ساعت اخیر
        if len(recent_hours) < 6:
            return {'error': 'Need at least 6 hours of data'}
        
        avg_requests = sum(recent_hours) / len(recent_hours)
        
        # پیش‌بینی ساده (میانگین متحرک)
        forecast = []
        current_hour = datetime.now().hour
        
        for i in range(forecast_hours):
            forecast_hour = (current_hour + i) % 24
            historical_avg = hourly_counts.get(forecast_hour, avg_requests)
            forecast.append({
                'hour': forecast_hour,
                'estimated_requests': round(historical_avg * 0.8 + avg_requests * 0.2),
                'confidence': 'medium'
            })
        
        return {
            'endpoint': endpoint,
            'forecast_hours': forecast_hours,
            'current_hourly_avg': avg_requests,
            'forecast': forecast,
            'recommendations': self._generate_forecast_recommendations(forecast)
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """گزارش جامع عملکرد"""
        try:
            performance_overview = self.analyze_endpoint_performance()
            slowest_endpoints = self.get_slowest_endpoints(10)
            most_called_endpoints = self.get_most_called_endpoints(10)
            external_services = self.get_external_service_health()
            peak_patterns = self.get_peak_usage_patterns()
            
            # شناسایی اندپوینت‌های مشکل‌دار
            problematic_endpoints = []
            for endpoint in slowest_endpoints:
                if endpoint['performance_grade'] in ['D', 'F']:
                    problematic_endpoints.append(endpoint)
            
            # تحلیل SLA
            sla_analysis = self._calculate_overall_sla()
            
            return {
                'report_timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_endpoints_monitored': len(self.endpoint_metrics),
                    'active_endpoints': len([ep for ep in self.endpoint_metrics.values() if ep.total_requests > 0]),
                    'overall_performance_grade': self._calculate_overall_performance_grade(performance_overview),
                    'problematic_endpoints_count': len(problematic_endpoints),
                    'external_services_health': external_services['health_percentage'],
                    'total_requests_24h': sum(sum(hourly.values()) for hourly in self.access_patterns.values())
                },
                'performance_analysis': performance_overview,
                'slowest_endpoints': slowest_endpoints,
                'most_called_endpoints': most_called_endpoints,
                'problematic_endpoints': problematic_endpoints,
                'external_services': external_services,
                'peak_usage_patterns': peak_patterns,
                'sla_compliance': sla_analysis,
                'recommendations': self._generate_comprehensive_recommendations(
                    performance_overview, problematic_endpoints, external_services, sla_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive report: {e}")
            return {
                'report_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'message': 'Comprehensive report generation failed'
            }
    
    # ==================== Utility Methods ====================
    
    def _convert_debug_stats_to_metrics(self, endpoint: str, debug_stats: Dict) -> EndpointMetrics:
        """تبدیل آمار debug_manager به EndpointMetrics"""
        return EndpointMetrics(
            endpoint=endpoint,
            total_requests=debug_stats.get('total_calls', 0),
            successful_requests=debug_stats.get('successful_calls', 0),
            failed_requests=debug_stats.get('failed_calls', 0),
            total_response_time=debug_stats.get('average_response_time', 0) * debug_stats.get('total_calls', 0),
            success_rate=debug_stats.get('success_rate', 0),
            cache_hits=debug_stats.get('cache_performance', {}).get('hits', 0),
            cache_misses=debug_stats.get('cache_performance', {}).get('misses', 0),
            api_calls=debug_stats.get('api_calls', 0),
            normalization_success_rate=debug_stats.get('normalization_performance', {}).get('success_rate', 100),
            normalization_quality_score=debug_stats.get('normalization_performance', {}).get('avg_quality_score', 100),
            last_updated=datetime.now()
        )
    
    def _analyze_sla_compliance(self, metrics: EndpointMetrics) -> Dict[str, Any]:
        """تحلیل رعایت SLA"""
        if metrics.total_requests == 0:
            return {
                'compliance': 'no_data',
                'uptime_percentage': 100.0,
                'error_rate': 0.0,
                'average_response_time_ms': 0.0
            }
        
        uptime = 100 - (metrics.failed_requests / metrics.total_requests * 100)
        avg_response_time_ms = (metrics.total_response_time / metrics.total_requests) * 1000
        
        compliance = {
            'uptime': uptime >= self.sla_targets['uptime'],
            'response_time': avg_response_time_ms <= self.sla_targets['response_time'],
            'error_rate': (metrics.failed_requests / metrics.total_requests * 100) <= self.sla_targets['error_rate']
        }
        
        return {
            'compliance': 'fully_compliant' if all(compliance.values()) else 'partially_compliant' if any(compliance.values()) else 'non_compliant',
            'uptime_percentage': uptime,
            'error_rate': (metrics.failed_requests / metrics.total_requests * 100),
            'average_response_time_ms': avg_response_time_ms,
            'targets': self.sla_targets,
            'violations': self._identify_sla_violations(uptime, avg_response_time_ms, metrics)
        }
    
    def _calculate_overall_sla(self) -> Dict[str, Any]:
        """محاسبه SLA کلی"""
        if not self.endpoint_metrics:
            return {'error': 'No endpoints available'}
        
        total_requests = sum(ep.total_requests for ep in self.endpoint_metrics.values())
        total_failed = sum(ep.failed_requests for ep in self.endpoint_metrics.values())
        total_response_time = sum(ep.total_response_time for ep in self.endpoint_metrics.values())
        
        if total_requests == 0:
            return {'error': 'No requests recorded'}
        
        uptime = 100 - (total_failed / total_requests * 100)
        avg_response_time_ms = (total_response_time / total_requests) * 1000
        error_rate = (total_failed / total_requests * 100)
        
        compliance = {
            'uptime': uptime >= self.sla_targets['uptime'],
            'response_time': avg_response_time_ms <= self.sla_targets['response_time'],
            'error_rate': error_rate <= self.sla_targets['error_rate']
        }
        
        return {
            'compliance_summary': {
                'fully_compliant': all(compliance.values()),
                'partially_compliant': any(compliance.values()),
                'non_compliant': not any(compliance.values())
            },
            'metrics': {
                'uptime_percentage': uptime,
                'average_response_time_ms': avg_response_time_ms,
                'error_rate': error_rate
            },
            'targets': self.sla_targets
        }
    
    def _identify_sla_violations(self, uptime: float, response_time_ms: float, metrics: EndpointMetrics) -> List[Dict[str, Any]]:
        """شناسایی نقض SLAها"""
        violations = []
        
        if uptime < self.sla_targets['uptime']:
            violations.append({
                'metric': 'uptime',
                'current': uptime,
                'target': self.sla_targets['uptime'],
                'violation': f"Uptime {uptime:.1f}% < {self.sla_targets['uptime']}%"
            })
        
        if response_time_ms > self.sla_targets['response_time']:
            violations.append({
                'metric': 'response_time',
                'current': response_time_ms,
                'target': self.sla_targets['response_time'],
                'violation': f"Response time {response_time_ms:.1f}ms > {self.sla_targets['response_time']}ms"
            })
        
        error_rate = (metrics.failed_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
        if error_rate > self.sla_targets['error_rate']:
            violations.append({
                'metric': 'error_rate',
                'current': error_rate,
                'target': self.sla_targets['error_rate'],
                'violation': f"Error rate {error_rate:.2f}% > {self.sla_targets['error_rate']}%"
            })
        
        return violations
    
    def _calculate_percentiles(self, data: List[float], percentiles: List[int] = None) -> Dict[str, float]:
        """محاسبه percentiles"""
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        
        if not data:
            return {f'p{p}': 0.0 for p in percentiles}
        
        sorted_data = sorted(data)
        result = {}
        
        for p in percentiles:
            index = (p / 100) * (len(sorted_data) - 1)
            if index.is_integer():
                result[f'p{p}'] = sorted_data[int(index)]
            else:
                lower = sorted_data[int(index)]
                upper = sorted_data[int(index) + 1]
                result[f'p{p}'] = lower + (upper - lower) * (index % 1)
        
        return result
    
    def _create_latency_distribution(self, response_times: List[float]) -> Dict[str, int]:
        """ایجاد توزیع frequency"""
        if not response_times:
            return {}
        
        bins = [0, 100, 200, 500, 1000, 2000, 5000, float('inf')]
        distribution = {f"<={bins[i]}": 0 for i in range(1, len(bins))}
        
        for rt in response_times:
            for i in range(1, len(bins)):
                if rt <= bins[i]:
                    distribution[f"<={bins[i]}"] += 1
                    break
        
        return distribution
    
    def _calculate_overall_performance_grade(self, performance_overview: Dict) -> str:
        """محاسبه گرید عملکرد کلی سیستم"""
        grades = {
            'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0
        }
        
        total_score = 0
        count = 0
        
        for endpoint_data in performance_overview.get('endpoint_performance', {}).values():
            grade = endpoint_data.get('performance_grade', 'F')
            total_score += grades.get(grade, 0)
            count += 1
        
        if count == 0:
            return 'N/A'
        
        average_score = total_score / count
        
        if average_score >= 4.5:
            return 'A+'
        elif average_score >= 3.5:
            return 'A'
        elif average_score >= 2.5:
            return 'B'
        elif average_score >= 1.5:
            return 'C'
        elif average_score >= 0.5:
            return 'D'
        else:
            return 'F'
    
    def _generate_peak_usage_recommendations(self, patterns: Dict, all_hourly_counts: Dict) -> List[str]:
        """تولید توصیه‌های استفاده پیک"""
        recommendations = []
        
        # شناسایی endpoints با پیک‌های شدید
        for endpoint, pattern in patterns.items():
            if pattern['peak_requests'] > 1000:
                recommendations.append(
                    f"Endpoint {endpoint} has high peak usage ({pattern['peak_requests']} reqs). Consider load balancing."
                )
        
        # شناسایی ساعت‌های شلوغ
        busy_hours = [hour for hour, count in all_hourly_counts.items() if count > 1000]
        if busy_hours:
            rec = f"System experiences high load during hours: {sorted(busy_hours)}. Consider auto-scaling."
            recommendations.append(rec)
        
        if not recommendations:
            recommendations.append("Usage patterns are within normal ranges.")
        
        return recommendations
    
    def _generate_forecast_recommendations(self, forecast: List[Dict]) -> List[str]:
        """تولید توصیه‌های مبتنی بر پیش‌بینی"""
        recommendations = []
        
        peak_forecast = max(forecast, key=lambda x: x['estimated_requests'])
        
        if peak_forecast['estimated_requests'] > 1000:
            recommendations.append(
                f"High traffic forecasted for hour {peak_forecast['hour']}: "
                f"{peak_forecast['estimated_requests']} requests expected. Prepare resources."
            )
        
        # شناسایی ساعت‌های کم‌مصرف برای maintenance
        low_traffic_hours = [f for f in forecast if f['estimated_requests'] < 100]
        if low_traffic_hours:
            hours = [f['hour'] for f in low_traffic_hours]
            recommendations.append(
                f"Consider maintenance during low traffic hours: {hours}"
            )
        
        if not recommendations:
            recommendations.append("Traffic forecast is within normal ranges.")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(self, performance_overview: Dict, 
                                              problematic_endpoints: List, 
                                              external_services: Dict,
                                              sla_analysis: Dict) -> List[str]:
        """تولید توصیه‌های جامع"""
        recommendations = []
        
        # بررسی SLA violations
        if sla_analysis.get('compliance_summary', {}).get('non_compliant', False):
            recommendations.append("SLA compliance issues detected. Review and optimize critical endpoints.")
        
        # بررسی problematic endpoints
        if problematic_endpoints:
            problematic_names = [ep['endpoint'] for ep in problematic_endpoints[:3]]
            recommendations.append(f"Focus optimization on: {', '.join(problematic_names)}")
        
        # بررسی external services
        if external_services.get('health_percentage', 100) < 90:
            recommendations.append("External services health is degraded. Check connectivity and dependencies.")
        
        # بررسی overall performance
        overall_grade = self._calculate_overall_performance_grade(performance_overview)
        if overall_grade in ['D', 'F']:
            recommendations.append("Overall system performance is poor. Consider architectural review.")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters. Continue monitoring.")
        
        return recommendations
    
    def clear_old_data(self, days: int = 30) -> int:
        """پاک کردن داده‌های قدیمی"""
        cutoff = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        # پاک کردن endpointهای قدیمی
        endpoints_to_remove = []
        for endpoint, metrics in self.endpoint_metrics.items():
            if metrics.last_updated and metrics.last_updated < cutoff:
                endpoints_to_remove.append(endpoint)
        
        for endpoint in endpoints_to_remove:
            del self.endpoint_metrics[endpoint]
            if endpoint in self.response_times_history:
                del self.response_times_history[endpoint]
            if endpoint in self.access_patterns:
                del self.access_patterns[endpoint]
            removed_count += 1
        
        # پاک کردن performance history قدیمی
        self.performance_history = deque(
            [p for p in self.performance_history if p['timestamp'] > cutoff],
            maxlen=1000
        )
        
        logger.info(f"🧹 Cleared old data for {removed_count} endpoints")
        return removed_count

# ایجاد نمونه گلوبال
performance_monitor = None

def initialize_performance_monitor(debug_manager, alert_manager, metrics_collector=None):
    """راه‌اندازی Performance Monitor"""
    global performance_monitor
    performance_monitor = PerformanceMonitor(debug_manager, alert_manager, metrics_collector)
    return performance_monitor

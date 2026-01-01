import psutil
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
import json
from dataclasses import dataclass

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

@dataclass
class APIPerformanceMetrics:
    """متریک‌های عملکرد API"""
    endpoint: str
    latency_ms: float
    status_code: int
    timestamp: datetime
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    user_agent: str = ""

class AdvancedMetricsCollector:
    """
    نسخه پیشرفته MetricsCollector با قابلیت‌های جدید:
    - مانیتورینگ عملکرد API
    - تحلیل SLA/SLO
    - مانیتورینگ سرویس‌های خارجی
    - تحلیل رفتار کاربران
    """
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=3600)
        self.process = psutil.Process()
        
        # کش متریک‌ها
        self.current_metrics_cache = {
            'cpu': {'percent': 0, 'per_core': [], 'load_avg': []},
            'memory': {'percent': 0, 'used_gb': 0, 'available_gb': 0},
            'disk': {'usage_percent': 0, 'io_read': 0, 'io_write': 0},
            'network': {'bytes_sent': 0, 'bytes_recv': 0, 'connections': 0},
            'process': {'memory_mb': 0, 'cpu_percent': 0, 'threads': 0},
            'data_normalization': {
                'success_rate': 0,
                'total_processed': 0,
                'total_errors': 0,
                'common_structures': {},
                'data_quality': {'avg_quality_score': 0}
            },
            # متریک‌های جدید
            'api_performance': defaultdict(lambda: deque(maxlen=100)),
            'external_services': {},
            'user_behavior': {
                'active_sessions': 0,
                'requests_per_minute': 0,
                'peak_hours': []
            },
            'sla_metrics': {
                'uptime_percentage': 100.0,
                'error_rate': 0.0,
                'average_response_time': 0.0
            }
        }
        
        self.cache_last_updated = None
        self.cache_ttl = 10  # افزایش TTL برای کاهش بار
        
        # مانیتورینگ API
        self.api_endpoints = {}
        self.sla_targets = {
            'uptime': 99.9,  # 99.9%
            'response_time': 200,  # 200ms
            'error_rate': 0.1  # 0.1%
        }
        
        # مانیتورینگ سرویس‌های خارجی
        self.external_services = {}
        
        # اتصال به central_monitor
        self._connect_to_central_monitor()
        
        logger.info("✅ AdvancedMetricsCollector Initialized")
    
    def _connect_to_central_monitor(self):
        """اتصال به سیستم مانیتورینگ مرکزی"""
        try:
            def delayed_connection():
                time.sleep(3)
                self._subscribe_to_monitor()
            
            connect_thread = threading.Thread(target=delayed_connection, daemon=True)
            connect_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Error connecting to central monitor: {e}")
            # راه‌اندازی جمع‌آوری پیشرفته مستقل
            self._start_advanced_collection()
    
    def _subscribe_to_monitor(self):
        """عضویت در central_monitor"""
        try:
            from .system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت متریک‌های سیستم
                central_monitor.subscribe("metrics_collector", self._on_system_metrics_received)
                logger.info("✅ MetricsCollector subscribed to central_monitor")
                
                # عضویت برای دریافت متریک‌های نرمال‌سازی
                central_monitor.subscribe("metrics_collector_norm", self._on_normalization_metrics_received)
                logger.info("✅ MetricsCollector subscribed to normalization metrics")
                
                # عضویت برای دریافت هشدارها
                central_monitor.subscribe("metrics_collector_alerts", self._on_alerts_received)
                logger.info("✅ MetricsCollector subscribed to alerts")
            else:
                logger.warning("⚠️ Central monitor not available, starting advanced collection")
                self._start_advanced_collection()
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor, starting advanced collection")
            self._start_advanced_collection()
        except Exception as e:
            logger.error(f"❌ Error subscribing to monitor: {e}")
            self._start_advanced_collection()
    
    def _on_system_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌های سیستم از central_monitor"""
        try:
            system_metrics = metrics.get('system', {})
            
            self.current_metrics_cache.update({
                'cpu': {
                    'percent': system_metrics.get('cpu', {}).get('percent', 0),
                    'per_core': system_metrics.get('cpu', {}).get('per_core', []),
                    'load_average': system_metrics.get('cpu', {}).get('load_average', [])
                },
                'memory': {
                    'percent': system_metrics.get('memory', {}).get('percent', 0),
                    'used_gb': system_metrics.get('memory', {}).get('used_gb', 0),
                    'available_gb': system_metrics.get('memory', {}).get('available_gb', 0)
                },
                'disk': {
                    'usage_percent': system_metrics.get('disk', {}).get('usage_percent', 0),
                    'io_read': 0,
                    'io_write': 0
                },
                'network': {
                    'bytes_sent': system_metrics.get('network', {}).get('bytes_sent', 0),
                    'bytes_recv': system_metrics.get('network', {}).get('bytes_recv', 0),
                    'connections': system_metrics.get('network', {}).get('connections', 0)
                },
                'process': {
                    'memory_mb': system_metrics.get('process', {}).get('memory_rss_mb', 0),
                    'cpu_percent': system_metrics.get('process', {}).get('cpu_percent', 0),
                    'threads': system_metrics.get('process', {}).get('threads_count', 0)
                }
            })
            
            self.cache_last_updated = datetime.now()
            
            # اضافه کردن به بافر تاریخچه
            self._add_to_history_buffer(system_metrics)
            
            # به‌روزرسانی SLA metrics
            self._update_sla_metrics()
            
            logger.debug(f"📈 System metrics updated from central_monitor")
            
        except Exception as e:
            logger.error(f"❌ Error processing system metrics: {e}")
    
    def _on_normalization_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌های نرمال‌سازی از central_monitor"""
        try:
            norm_metrics = metrics.get('data_normalization', {})
            
            self.current_metrics_cache['data_normalization'] = {
                'success_rate': norm_metrics.get('success_rate', 0),
                'total_processed': norm_metrics.get('total_processed', 0),
                'total_errors': norm_metrics.get('total_errors', 0),
                'common_structures': norm_metrics.get('common_structures', {}),
                'data_quality': norm_metrics.get('data_quality', {'avg_quality_score': 0})
            }
            
            logger.debug(f"📊 Normalization metrics updated")
            
        except Exception as e:
            logger.error(f"❌ Error processing normalization metrics: {e}")
    
    def _on_alerts_received(self, metrics: Dict[str, Any]):
        """دریافت هشدارها از central_monitor"""
        try:
            alerts = metrics.get('alerts', [])
            for alert in alerts[:5]:  # فقط 5 هشدار اخیر
                logger.warning(f"🔔 Alert received: {alert.get('title', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing alerts: {e}")
    
    def _start_advanced_collection(self):
        """راه‌اندازی جمع‌آوری پیشرفته مستقل"""
        def advanced_collection_loop():
            """حلقه جمع‌آوری پیشرفته"""
            while True:
                try:
                    # جمع‌آوری متریک‌های سیستم
                    system_metrics = self._collect_advanced_metrics()
                    
                    # جمع‌آوری متریک‌های API
                    self._collect_api_metrics()
                    
                    # جمع‌آوری متریک‌های سرویس‌های خارجی
                    self._collect_external_service_metrics()
                    
                    # تحلیل رفتار کاربران
                    self._analyze_user_behavior()
                    
                    # به‌روزرسانی SLA
                    self._update_sla_metrics()
                    
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"❌ Advanced collection error: {e}")
                    time.sleep(60)
        
        collection_thread = threading.Thread(target=advanced_collection_loop, daemon=True)
        collection_thread.start()
        logger.info("🔄 Advanced metrics collection started")
    
    def _collect_advanced_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های پیشرفته سیستم"""
        timestamp = datetime.now()
        
        # CPU با جزئیات بیشتر
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Memory
        memory = psutil.virtual_memory()
        
        # Disk با I/O
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network با جزئیات
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        # Process info
        process_memory = self.process.memory_info()
        
        return {
            'timestamp': timestamp,
            'cpu': {
                'percent': sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0,
                'per_core': cpu_percent,
                'load_average': self._get_load_average()
            },
            'memory': {
                'percent': memory.percent,
                'used_gb': round(memory.used / (1024**3), 3),
                'available_gb': round(memory.available / (1024**3), 3)
            },
            'disk': {
                'usage_percent': disk_usage.percent,
                'io_read': disk_io.read_bytes if disk_io else 0,
                'io_write': disk_io.write_bytes if disk_io else 0
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'connections': net_connections
            },
            'process': {
                'memory_mb': round(process_memory.rss / (1024**2), 2),
                'cpu_percent': self.process.cpu_percent(interval=0.1),
                'threads': self.process.num_threads()
            }
        }
    
    # 🔧 قابلیت‌های جدید
    
    def register_api_endpoint(self, endpoint: str, sla_response_time_ms: int = 200):
        """ثبت endpoint API برای مانیتورینگ"""
        self.api_endpoints[endpoint] = {
            'sla_response_time': sla_response_time_ms,
            'metrics': deque(maxlen=100),
            'total_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0
        }
        logger.info(f"✅ API endpoint registered for monitoring: {endpoint}")
    
    def record_api_request(self, endpoint: str, latency_ms: float, status_code: int, 
                          request_size: int = 0, response_size: int = 0, user_agent: str = ""):
        """ثبت درخواست API برای تحلیل عملکرد"""
        if endpoint not in self.api_endpoints:
            self.register_api_endpoint(endpoint)
        
        endpoint_data = self.api_endpoints[endpoint]
        
        metric = APIPerformanceMetrics(
            endpoint=endpoint,
            latency_ms=latency_ms,
            status_code=status_code,
            timestamp=datetime.now(),
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            user_agent=user_agent
        )
        
        endpoint_data['metrics'].append(metric)
        endpoint_data['total_requests'] += 1
        endpoint_data['total_response_time'] += latency_ms
        
        if status_code >= 400:
            endpoint_data['failed_requests'] += 1
        
        # ذخیره در کش عمومی
        if endpoint not in self.current_metrics_cache['api_performance']:
            self.current_metrics_cache['api_performance'][endpoint] = deque(maxlen=100)
        
        self.current_metrics_cache['api_performance'][endpoint].append({
            'latency_ms': latency_ms,
            'status_code': status_code,
            'timestamp': datetime.now().isoformat()
        })
    
    def _collect_api_metrics(self):
        """جمع‌آوری متریک‌های API"""
        for endpoint, data in self.api_endpoints.items():
            if data['metrics']:
                recent_metrics = list(data['metrics'])[-10:]  # 10 درخواست اخیر
                
                avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
                success_rate = (sum(1 for m in recent_metrics if m.status_code < 400) / len(recent_metrics)) * 100
                
                # بررسی SLA
                sla_breach = avg_latency > data['sla_response_time']
                
                if sla_breach:
                    logger.warning(f"⚠️ SLA breach for {endpoint}: {avg_latency:.1f}ms > {data['sla_response_time']}ms")
    
    def register_external_service(self, service_name: str, check_url: str, check_interval: int = 60):
        """ثبت سرویس خارجی برای مانیتورینگ"""
        self.external_services[service_name] = {
            'url': check_url,
            'check_interval': check_interval,
            'last_check': None,
            'status': 'unknown',
            'latency_ms': None,
            'uptime_percentage': 100.0
        }
        
        self.current_metrics_cache['external_services'][service_name] = {
            'status': 'unknown',
            'last_check': None
        }
        
        logger.info(f"✅ External service registered: {service_name}")
    
    def _collect_external_service_metrics(self):
        """جمع‌آوری متریک‌های سرویس‌های خارجی"""
        for service_name, service_info in self.external_services.items():
            try:
                import requests
                
                start_time = time.time()
                response = requests.get(service_info['url'], timeout=10)
                latency = (time.time() - start_time) * 1000
                
                service_info.update({
                    'last_check': datetime.now(),
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'latency_ms': round(latency, 2),
                    'last_status_code': response.status_code
                })
                
            except Exception as e:
                service_info.update({
                    'last_check': datetime.now(),
                    'status': 'unhealthy',
                    'latency_ms': None,
                    'error': str(e)
                })
            
            # به‌روزرسانی کش
            self.current_metrics_cache['external_services'][service_name] = {
                'status': service_info['status'],
                'last_check': service_info['last_check'].isoformat() if service_info['last_check'] else None,
                'latency_ms': service_info['latency_ms']
            }
    
    def _analyze_user_behavior(self):
        """تحلیل رفتار کاربران"""
        # این تابع می‌تواند از لاگ‌ها یا دیتابیس اطلاعات بگیرد
        # در این نسخه شبیه‌سازی می‌کنیم
        
        import random
        
        self.current_metrics_cache['user_behavior'].update({
            'active_sessions': random.randint(50, 200),
            'requests_per_minute': random.randint(100, 500),
            'peak_hours': self._calculate_peak_hours()
        })
    
    def _calculate_peak_hours(self) -> List[Dict[str, Any]]:
        """محاسبه ساعت‌های پیک استفاده"""
        # شبیه‌سازی - در نسخه واقعی از داده‌های تاریخی استفاده می‌شود
        peak_hours = []
        
        for hour in range(24):
            if 9 <= hour <= 17:  # ساعات کاری
                traffic_level = random.randint(70, 100)
            elif 18 <= hour <= 22:  # عصر
                traffic_level = random.randint(40, 70)
            else:  # شب
                traffic_level = random.randint(10, 40)
            
            peak_hours.append({
                'hour': hour,
                'traffic_level': traffic_level,
                'period': 'peak' if traffic_level > 70 else 'normal' if traffic_level > 30 else 'low'
            })
        
        return sorted(peak_hours, key=lambda x: x['traffic_level'], reverse=True)[:5]
    
    def _update_sla_metrics(self):
        """به‌روزرسانی متریک‌های SLA"""
        try:
            # محاسبات SLA (شبیه‌سازی شده)
            total_requests = sum(data['total_requests'] for data in self.api_endpoints.values())
            failed_requests = sum(data['failed_requests'] for data in self.api_endpoints.values())
            
            if total_requests > 0:
                error_rate = (failed_requests / total_requests) * 100
                uptime = 100 - min(error_rate, 100)
            else:
                error_rate = 0.0
                uptime = 100.0
            
            avg_response_time = 0
            if self.api_endpoints:
                total_time = sum(data['total_response_time'] for data in self.api_endpoints.values())
                total_reqs = sum(len(data['metrics']) for data in self.api_endpoints.values())
                avg_response_time = total_time / total_reqs if total_reqs > 0 else 0
            
            self.current_metrics_cache['sla_metrics'].update({
                'uptime_percentage': round(uptime, 2),
                'error_rate': round(error_rate, 3),
                'average_response_time': round(avg_response_time, 1),
                'sla_compliance': {
                    'uptime': uptime >= self.sla_targets['uptime'],
                    'response_time': avg_response_time <= self.sla_targets['response_time'],
                    'error_rate': error_rate <= self.sla_targets['error_rate']
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error updating SLA metrics: {e}")
    
    def _add_to_history_buffer(self, system_metrics: Dict[str, Any]):
        """اضافه کردن متریک‌ها به بافر تاریخچه"""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'cpu_percent': system_metrics.get('cpu', {}).get('percent', 0),
                'memory_percent': system_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': system_metrics.get('disk', {}).get('usage_percent', 0),
                'network_sent_mb_sec': 0,
                'network_recv_mb_sec': 0,
                'process_memory_mb': system_metrics.get('process', {}).get('memory_rss_mb', 0),
                'normalization_success_rate': self.current_metrics_cache['data_normalization']['success_rate'],
                'normalization_total_processed': self.current_metrics_cache['data_normalization']['total_processed'],
                'api_performance': dict(self.current_metrics_cache['api_performance']),
                'sla_metrics': self.current_metrics_cache['sla_metrics']
            }
            
            self.metrics_buffer.append(history_entry)
            
        except Exception as e:
            logger.error(f"❌ Error adding to history buffer: {e}")
    
    def _refresh_normalization_metrics(self):
        """رفرش متریک‌های نرمال‌سازی"""
        try:
            metrics = data_normalizer.get_health_metrics()
            
            self.current_metrics_cache['data_normalization'] = {
                'success_rate': metrics.success_rate,
                'total_processed': metrics.total_processed,
                'total_errors': metrics.total_errors,
                'common_structures': metrics.common_structures,
                'data_quality': metrics.data_quality
            }
            
        except Exception as e:
            logger.error(f"❌ Error refreshing normalization metrics: {e}")
    
    def _get_load_average(self) -> List[float]:
        """دریافت load average"""
        try:
            return list(psutil.getloadavg())
        except:
            return [0, 0, 0]
    
    # 🎯 APIهای جدید
    
    def get_api_performance_report(self, endpoint: str = None) -> Dict[str, Any]:
        """گزارش عملکرد API"""
        if endpoint:
            if endpoint not in self.api_endpoints:
                return {'error': f'Endpoint {endpoint} not found'}
            
            data = self.api_endpoints[endpoint]
            metrics = list(data['metrics'])
            
            if not metrics:
                return {'endpoint': endpoint, 'message': 'No data available'}
            
            latencies = [m.latency_ms for m in metrics]
            status_codes = [m.status_code for m in metrics]
            
            return {
                'endpoint': endpoint,
                'total_requests': data['total_requests'],
                'failed_requests': data['failed_requests'],
                'success_rate': ((data['total_requests'] - data['failed_requests']) / data['total_requests'] * 100) if data['total_requests'] > 0 else 0,
                'average_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                'sla_compliance': data.get('sla_response_time', 0) >= (sum(latencies) / len(latencies) if latencies else 0),
                'status_code_distribution': {code: status_codes.count(code) for code in set(status_codes)},
                'recent_requests': [
                    {
                        'latency_ms': m.latency_ms,
                        'status_code': m.status_code,
                        'timestamp': m.timestamp.isoformat()
                    }
                    for m in metrics[-5:]
                ]
            }
        else:
            # گزارش کلی همه endpoints
            report = {}
            for ep in self.api_endpoints:
                report[ep] = self.get_api_performance_report(ep)
            
            return {
                'total_endpoints': len(self.api_endpoints),
                'endpoints': report,
                'overall_success_rate': self._calculate_overall_api_success_rate(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_overall_api_success_rate(self) -> float:
        """محاسبه نرخ موفقیت کلی APIها"""
        total_requests = 0
        successful_requests = 0
        
        for data in self.api_endpoints.values():
            total_requests += data['total_requests']
            successful_requests += (data['total_requests'] - data['failed_requests'])
        
        return (successful_requests / total_requests * 100) if total_requests > 0 else 100.0
    
    def get_external_service_health(self) -> Dict[str, Any]:
        """سلامت سرویس‌های خارجی"""
        healthy_count = sum(1 for s in self.external_services.values() if s['status'] == 'healthy')
        total_count = len(self.external_services)
        
        return {
            'total_services': total_count,
            'healthy_services': healthy_count,
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 100,
            'services': {
                name: {
                    'status': info['status'],
                    'last_check': info['last_check'].isoformat() if info['last_check'] else None,
                    'latency_ms': info['latency_ms']
                }
                for name, info in self.external_services.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_user_behavior_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """تحلیل رفتار کاربران"""
        # در نسخه واقعی، این داده‌ها از دیتابیس خوانده می‌شود
        return {
            'time_period_hours': hours,
            'current_metrics': self.current_metrics_cache['user_behavior'],
            'peak_usage_times': self._calculate_peak_hours(),
            'recommendations': self._generate_user_behavior_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_user_behavior_recommendations(self) -> List[str]:
        """تولید توصیه‌های مبتنی بر رفتار کاربران"""
        recommendations = []
        
        behavior = self.current_metrics_cache['user_behavior']
        
        if behavior['active_sessions'] > 150:
            recommendations.append("High active sessions detected. Consider scaling up.")
        
        if behavior['requests_per_minute'] > 400:
            recommendations.append("High request rate. Optimize API endpoints and consider caching.")
        
        peak_hours = behavior.get('peak_hours', [])
        if any(hour['traffic_level'] > 80 for hour in peak_hours):
            recommendations.append("Peak traffic hours detected. Implement auto-scaling during these hours.")
        
        if not recommendations:
            recommendations.append("User behavior patterns are normal. No immediate actions required.")
        
        return recommendations
    
    def get_sla_compliance_report(self) -> Dict[str, Any]:
        """گزارش رعایت SLA"""
        sla_metrics = self.current_metrics_cache['sla_metrics']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': sla_metrics,
            'targets': self.sla_targets,
            'compliance_summary': {
                'fully_compliant': all(sla_metrics.get('sla_compliance', {}).values()),
                'partially_compliant': any(sla_metrics.get('sla_compliance', {}).values()),
                'non_compliant': not any(sla_metrics.get('sla_compliance', {}).values())
            },
            'violations': self._identify_sla_violations(),
            'recommendations': self._generate_sla_recommendations()
        }
    
    def _identify_sla_violations(self) -> List[Dict[str, Any]]:
        """شناسایی نقض SLAها"""
        violations = []
        sla_metrics = self.current_metrics_cache['sla_metrics']
        compliance = sla_metrics.get('sla_compliance', {})
        
        if not compliance.get('uptime', True):
            violations.append({
                'metric': 'uptime',
                'current': sla_metrics['uptime_percentage'],
                'target': self.sla_targets['uptime'],
                'violation': f"Uptime {sla_metrics['uptime_percentage']}% < {self.sla_targets['uptime']}%"
            })
        
        if not compliance.get('response_time', True):
            violations.append({
                'metric': 'response_time',
                'current': sla_metrics['average_response_time'],
                'target': self.sla_targets['response_time'],
                'violation': f"Response time {sla_metrics['average_response_time']}ms > {self.sla_targets['response_time']}ms"
            })
        
        if not compliance.get('error_rate', True):
            violations.append({
                'metric': 'error_rate',
                'current': sla_metrics['error_rate'],
                'target': self.sla_targets['error_rate'],
                'violation': f"Error rate {sla_metrics['error_rate']}% > {self.sla_targets['error_rate']}%"
            })
        
        return violations
    
    def _generate_sla_recommendations(self) -> List[str]:
        """تولید توصیه‌های SLA"""
        recommendations = []
        violations = self._identify_sla_violations()
        
        for violation in violations:
            if violation['metric'] == 'response_time':
                recommendations.append("Optimize database queries and implement caching for slow endpoints")
            elif violation['metric'] == 'error_rate':
                recommendations.append("Improve error handling and implement retry logic for failed requests")
            elif violation['metric'] == 'uptime':
                recommendations.append("Implement better monitoring and automatic recovery for service failures")
        
        if not recommendations:
            recommendations.append("All SLA targets are being met. Continue current practices.")
        
        return recommendations
    
    def get_comprehensive_advanced_report(self) -> Dict[str, Any]:
        """گزارش جامع پیشرفته"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_current_metrics(),
            'api_performance': self.get_api_performance_report(),
            'external_services': self.get_external_service_health(),
            'user_behavior': self.get_user_behavior_analysis(),
            'sla_compliance': self.get_sla_compliance_report(),
            'data_normalization': self.get_normalization_metrics(),
            'recommendations_summary': self._generate_comprehensive_recommendations()
        }
    
    def _generate_comprehensive_recommendations(self) -> Dict[str, Any]:
        """تولید توصیه‌های جامع"""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # بررسی SLA violations
        sla_report = self.get_sla_compliance_report()
        violations = sla_report.get('violations', [])
        
        if violations:
            for violation in violations:
                recommendations['high_priority'].append(f"Fix {violation['metric']} SLA violation: {violation['violation']}")
        
        # بررسی API performance
        api_report = self.get_api_performance_report()
        if isinstance(api_report, dict) and 'overall_success_rate' in api_report:
            if api_report['overall_success_rate'] < 95:
                recommendations['medium_priority'].append(f"Improve API success rate: {api_report['overall_success_rate']:.1f}%")
        
        # بررسی منابع سیستم
        system_metrics = self.get_current_metrics()
        cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
        memory_usage = system_metrics.get('memory', {}).get('percent', 0)
        
        if cpu_usage > 80:
            recommendations['medium_priority'].append(f"High CPU usage: {cpu_usage}%. Consider optimizing or scaling.")
        
        if memory_usage > 85:
            recommendations['medium_priority'].append(f"High memory usage: {memory_usage}%. Check for memory leaks.")
        
        # اگر هیچ توصیه‌ای نبود
        if not any(recommendations.values()):
            recommendations['low_priority'].append("System is operating within optimal parameters. Continue monitoring.")
        
        return recommendations
    
    # 📡 APIهای موجود (برای backward compatibility)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های فعلی - API بدون تغییر"""
        if (self.cache_last_updated and 
            (datetime.now() - self.cache_last_updated).total_seconds() > self.cache_ttl):
            logger.debug("⚠️ Metrics cache expired, returning cached data")
        
        return self.current_metrics_cache
    
    def get_metrics_history(self, seconds: int = 300) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌ها - API بدون تغییر"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        return [
            {
                'timestamp': metrics['timestamp'].isoformat(),
                'cpu_percent': metrics['cpu_percent'],
                'memory_percent': metrics['memory_percent'],
                'disk_usage': metrics['disk_usage'],
                'network_sent_mb_sec': metrics['network_sent_mb_sec'],
                'network_recv_mb_sec': metrics['network_recv_mb_sec'],
                'process_memory_mb': metrics['process_memory_mb'],
                'normalization_success_rate': metrics['normalization_success_rate'],
                'normalization_total_processed': metrics['normalization_total_processed'],
                'api_performance_summary': len(metrics.get('api_performance', {})),
                'sla_uptime': metrics.get('sla_metrics', {}).get('uptime_percentage', 100)
            }
            for metrics in self.metrics_buffer
            if metrics['timestamp'] >= cutoff_time
        ]
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های دقیق - API بدون تغییر"""
        return self.get_current_metrics()
    
    def get_normalization_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های نرمال‌سازی - API بدون تغییر"""
        return self.current_metrics_cache['data_normalization']
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه متریک‌ها - API بدون تغییر"""
        metrics = self.get_current_metrics()
        normalization = metrics['data_normalization']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_usage': f"{metrics['cpu']['percent']}%",
                'memory_usage': f"{metrics['memory']['percent']}%",
                'disk_usage': f"{metrics['disk']['usage_percent']}%",
                'network_activity': "Advanced Monitor Active"
            },
            'process_health': {
                'memory_usage': f"{metrics['process']['memory_mb']}MB",
                'cpu_usage': f"{metrics['process']['cpu_percent']}%",
                'threads': metrics['process']['threads']
            },
            'data_normalization_health': {
                'success_rate': f"{normalization.get('success_rate', 0)}%",
                'total_processed': normalization.get('total_processed', 0),
                'data_quality': f"{normalization.get('data_quality', {}).get('avg_quality_score', 0)}%",
                'common_structures': len(normalization.get('common_structures', {}))
            },
            'api_performance': {
                'monitored_endpoints': len(self.api_endpoints),
                'external_services': len(self.external_services)
            }
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """دریافت گزارش جامع - API بدون تغییر"""
        current_metrics = self.get_current_metrics()
        metrics_history = self.get_metrics_history(seconds=3600)
        
        cpu_trend = self._analyze_trend([m['cpu_percent'] for m in metrics_history])
        memory_trend = self._analyze_trend([m['memory_percent'] for m in metrics_history])
        normalization_trend = self._analyze_trend([m['normalization_success_rate'] for m in metrics_history])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'trend_analysis': {
                'cpu': cpu_trend,
                'memory': memory_trend,
                'normalization': normalization_trend
            },
            'normalization_insights': self.get_normalization_metrics(),
            'performance_indicators': {
                'system_stability': 'high' if cpu_trend['stability'] > 0.8 and memory_trend['stability'] > 0.8 else 'medium',
                'normalization_reliability': 'high' if normalization_trend['stability'] > 0.9 else 'medium',
                'resource_utilization': 'optimal' if current_metrics['cpu']['percent'] < 70 and current_metrics['memory']['percent'] < 80 else 'high',
                'api_performance': 'good' if len(self.api_endpoints) > 0 else 'unknown'
            },
            'advanced_features': {
                'api_monitoring': True,
                'external_service_monitoring': True,
                'user_behavior_analysis': True,
                'sla_tracking': True
            }
        }
    
    def _analyze_trend(self, data: List[float]) -> Dict[str, Any]:
        """تحلیل روند داده‌ها"""
        if len(data) < 2:
            return {'trend': 'stable', 'stability': 1.0, 'volatility': 0.0}
        
        changes = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
        avg_change = sum(changes) / len(changes) if changes else 0
        max_value = max(data) if data else 0
        volatility = avg_change / max_value if max_value > 0 else 0
        
        if len(data) >= 3:
            recent_avg = sum(data[-3:]) / 3
            older_avg = sum(data[-6:-3]) / 3 if len(data) >= 6 else data[0]
            trend = 'improving' if recent_avg > older_avg else 'declining' if recent_avg < older_avg else 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'stability': 1.0 - min(volatility, 1.0),
            'volatility': round(volatility, 3),
            'data_points': len(data)
        }
    
    def get_connection_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال"""
        return {
            'cache_age_seconds': (datetime.now() - self.cache_last_updated).total_seconds() if self.cache_last_updated else None,
            'metrics_buffer_size': len(self.metrics_buffer),
            'cache_ttl': self.cache_ttl,
            'collection_mode': 'central_monitor' if self.cache_last_updated else 'advanced_standalone',
            'monitored_endpoints': len(self.api_endpoints),
            'external_services': len(self.external_services),
            'advanced_features_active': True,
            'timestamp': datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال با همان نام دقیق
metrics_collector = AdvancedMetricsCollector()

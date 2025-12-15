import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import inspect
import functools

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

class EndpointMonitor:
    def __init__(self, debug_manager):
        self.debug_manager = debug_manager
        self.endpoint_registry = {}
        self.dependency_graph = defaultdict(list)
        self.performance_baselines = {}
        
        # آمار کیفیت داده نرمال‌سازی
        self.normalization_quality = defaultdict(lambda: {
            'total_calls': 0,
            'failed_normalizations': 0,
            'avg_quality_score': 0,
            'structure_patterns': defaultdict(int),
            'last_check': None
        })
        
        # اتصال به central_monitor برای دریافت metrics
        self._connect_to_central_monitor()
        
        logger.info("✅ Endpoint Monitor Initialized - Central Monitor Connected")
        
    def _connect_to_central_monitor(self):
        """اتصال به central_monitor برای دریافت متریک‌های endpoint"""
        try:
            from .system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت متریک‌های سیستم (برای context)
                central_monitor.subscribe("endpoint_monitor", self._on_system_metrics_received)
                logger.info("✅ EndpointMonitor subscribed to central_monitor")
            else:
                logger.warning("⚠️ Central monitor not available - endpoint monitor will use debug_manager only")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - endpoint monitor will use debug_manager only")
        except Exception as e:
            logger.error(f"❌ Error connecting to central_monitor: {e}")
    
    def _on_system_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌های سیستم از central_monitor"""
        try:
            # می‌توانیم از system metrics برای context استفاده کنیم
            system_metrics = metrics.get('system', {})
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            
            # اگر سیستم تحت فشار است، endpoint performance را بررسی کن
            if cpu_usage > 80 or memory_usage > 85:
                self._check_endpoint_performance_under_load()
                
        except Exception as e:
            logger.error(f"❌ Error processing system metrics: {e}")
    
    def _check_endpoint_performance_under_load(self):
        """بررسی performance endpointها در زمان load بالا"""
        try:
            # فقط sample checking - برای جلوگیری از overload
            endpoint_stats = self.debug_manager.get_endpoint_stats()
            
            slow_endpoints = []
            for endpoint, stats in endpoint_stats.get('endpoints', {}).items():
                if stats.get('average_response_time', 0) > 2.0:  # بیش از 2 ثانیه
                    slow_endpoints.append({
                        'endpoint': endpoint,
                        'response_time': stats['average_response_time']
                    })
            
            if slow_endpoints:
                logger.warning(f"⚠️ {len(slow_endpoints)} endpoints are slow under high system load")
                
        except Exception as e:
            logger.error(f"❌ Error checking endpoint performance: {e}")

    # بقیه متدها بدون تغییر (مثل قبل)
    def monitor_endpoint(self, endpoint_name: str, method: str = "GET"):
        """دکوراتور برای مانیتورینگ اندپوینت"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                cache_used = False
                api_calls = 0
                status_code = 200
                normalization_info = None
                
                try:
                    # اجرای اندپوینت
                    result = await func(*args, **kwargs)
                    
                    # بررسی اگر نتیجه شامل status_code باشد
                    if isinstance(result, dict) and 'status_code' in result:
                        status_code = result['status_code']
                    
                    # استخراج اطلاعات نرمال‌سازی از نتیجه
                    if isinstance(result, dict):
                        normalization_info = result.get('normalization_info')
                    
                    return result
                    
                except Exception as e:
                    status_code = 500
                    raise e
                    
                finally:
                    # محاسبه زمان پاسخ
                    response_time = time.time() - start_time
                    
                    # جمع‌آوری پارامترها (بدون اطلاعات حساس)
                    safe_params = self._sanitize_parameters(kwargs)
                    
                    # ثبت در دیباگ منیجر با اطلاعات نرمال‌سازی
                    self.debug_manager.log_endpoint_call(
                        endpoint=endpoint_name,
                        method=method,
                        params=safe_params,
                        response_time=response_time,
                        status_code=status_code,
                        cache_used=cache_used,
                        api_calls=api_calls,
                        normalization_info=normalization_info
                    )
                    
                    # آپدیت آمار کیفیت نرمال‌سازی
                    if normalization_info:
                        self._update_normalization_quality(endpoint_name, normalization_info)
            
            # ثبت اندپوینت در رجیستری
            self.endpoint_registry[endpoint_name] = {
                'function': wrapper,
                'method': method,
                'registered_at': datetime.now().isoformat()
            }
            
            return wrapper
        return decorator

    def _update_normalization_quality(self, endpoint: str, normalization_info: Dict[str, Any]):
        """آپدیت آمار کیفیت نرمال‌سازی"""
        quality_metrics = self.normalization_quality[endpoint]
        quality_metrics['total_calls'] += 1
        quality_metrics['last_check'] = datetime.now().isoformat()
        
        # ثبت وضعیت نرمال‌سازی
        if normalization_info.get('status') == 'error':
            quality_metrics['failed_normalizations'] += 1
        
        # آپدیت امتیاز کیفیت
        quality_score = normalization_info.get('quality_score', 0)
        current_avg = quality_metrics['avg_quality_score']
        total_calls = quality_metrics['total_calls']
        
        quality_metrics['avg_quality_score'] = (
            (current_avg * (total_calls - 1) + quality_score) / total_calls
        )
        
        # ثبت الگوی ساختاری
        structure = normalization_info.get('detected_structure', 'unknown')
        quality_metrics['structure_patterns'][structure] += 1

    def track_api_call(self, endpoint_name: str, api_name: str):
        """ردیابی فراخوانی API خارجی"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    api_duration = time.time() - start_time
                    # ثبت فراخوانی API در دیباگ منیجر
                    self._log_api_call(endpoint_name, api_name, api_duration, kwargs)
            return wrapper
        return decorator

    def _log_api_call(self, endpoint: str, api_name: str, duration: float, params: Dict):
        """ثبت فراخوانی API خارجی"""
        logger.debug(f"🔗 API Call: {endpoint} -> {api_name} ({duration:.3f}s)")

    def _sanitize_parameters(self, params: Dict) -> Dict:
        """پاکسازی پارامترها از اطلاعات حساس"""
        safe_params = {}
        sensitive_keys = ['password', 'token', 'secret', 'key', 'authorization']
        
        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                safe_params[key] = '***HIDDEN***'
            else:
                safe_params[key] = str(value)[:100]  # محدود کردن طول
                
        return safe_params

    def add_dependency(self, endpoint: str, depends_on: str):
        """اضافه کردن وابستگی بین اندپوینت‌ها"""
        self.dependency_graph[endpoint].append(depends_on)

    def get_endpoint_health(self, endpoint: str) -> Dict[str, Any]:
        """دریافت سلامت یک اندپوینت"""
        stats = self.debug_manager.get_endpoint_stats(endpoint)
        
        if 'error' in stats:
            return {'status': 'unknown', 'error': stats['error']}
        
        # ارزیابی سلامت بر اساس آمار
        health_score = self._calculate_health_score(stats)
        
        # اطلاعات کیفیت نرمال‌سازی
        quality_info = self.normalization_quality.get(endpoint, {})
        
        return {
            'endpoint': endpoint,
            'status': self._get_health_status(health_score),
            'health_score': health_score,
            'performance': {
                'average_response_time': stats['average_response_time'],
                'success_rate': stats['success_rate'],
                'cache_hit_rate': stats['cache_performance']['hit_rate']
            },
            'data_quality': {
                'normalization_success_rate': stats.get('normalization_performance', {}).get('success_rate', 0),
                'avg_quality_score': quality_info.get('avg_quality_score', 0),
                'structure_patterns': dict(quality_info.get('structure_patterns', {})),
                'failed_normalizations': quality_info.get('failed_normalizations', 0)
            },
            'last_updated': datetime.now().isoformat()
        }

    def get_all_endpoints_health(self) -> Dict[str, Any]:
        """دریافت سلامت تمام اندپوینت‌ها"""
        all_stats = self.debug_manager.get_endpoint_stats()
        health_report = {}
        
        for endpoint, stats in all_stats['endpoints'].items():
            health_score = self._calculate_health_score(stats)
            quality_info = self.normalization_quality.get(endpoint, {})
            
            health_report[endpoint] = {
                'status': self._get_health_status(health_score),
                'health_score': health_score,
                'performance': {
                    'average_response_time': stats['average_response_time'],
                    'success_rate': stats['success_rate']
                },
                'data_quality': {
                    'normalization_success_rate': stats.get('normalization_performance', {}).get('success_rate', 0),
                    'avg_quality_score': quality_info.get('avg_quality_score', 0),
                    'quality_level': self._assess_data_quality_level(quality_info)
                }
            }
        
        # متریک‌های کلی نرمال‌سازی
        normalization_metrics = data_normalizer.get_health_metrics()
        
        return {
            'overall_health': self._calculate_overall_health(health_report),
            'endpoints': health_report,
            'total_endpoints': len(health_report),
            'data_normalization_overview': {
                'system_success_rate': normalization_metrics.success_rate,
                'total_processed': normalization_metrics.total_processed,
                'common_structures': normalization_metrics.common_structures,
                'data_quality': normalization_metrics.data_quality
            },
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_health_score(self, stats: Dict) -> float:
        """محاسبه امتیاز سلامت"""
        score = 0
        
        # امتیاز بر اساس نرخ موفقیت (40%)
        success_rate = stats.get('success_rate', 0)
        score += (success_rate / 100) * 40
        
        # امتیاز بر اساس زمان پاسخ (25%)
        avg_response = stats.get('average_response_time', 0)
        if avg_response < 0.5:
            score += 25
        elif avg_response < 1.0:
            score += 20
        elif avg_response < 2.0:
            score += 15
        elif avg_response < 3.0:
            score += 10
        else:
            score += 5
            
        # امتیاز بر اساس کش (15%)
        cache_hit_rate = stats.get('cache_performance', {}).get('hit_rate', 0)
        score += (cache_hit_rate / 100) * 15
        
        # امتیاز بر اساس کیفیت داده نرمال‌سازی (20%)
        norm_performance = stats.get('normalization_performance', {})
        norm_success_rate = norm_performance.get('success_rate', 100)
        score += (norm_success_rate / 100) * 20
        
        return min(score, 100)

    def _assess_data_quality_level(self, quality_info: Dict) -> str:
        """ارزیابی سطح کیفیت داده"""
        avg_quality = quality_info.get('avg_quality_score', 0)
        failed_count = quality_info.get('failed_normalizations', 0)
        total_calls = quality_info.get('total_calls', 1)
        failure_rate = (failed_count / total_calls) * 100
        
        if failure_rate > 20 or avg_quality < 60:
            return "POOR"
        elif failure_rate > 10 or avg_quality < 75:
            return "FAIR"
        elif failure_rate > 5 or avg_quality < 85:
            return "GOOD"
        else:
            return "EXCELLENT"

    def _get_health_status(self, health_score: float) -> str:
        """تعیین وضعیت سلامت بر اساس امتیاز"""
        if health_score >= 90:
            return 'excellent'
        elif health_score >= 75:
            return 'good'
        elif health_score >= 60:
            return 'fair'
        elif health_score >= 40:
            return 'degraded'
        else:
            return 'poor'

    def _calculate_overall_health(self, health_report: Dict) -> str:
        """محاسبه سلامت کلی سیستم"""
        if not health_report:
            return 'unknown'
            
        status_counts = defaultdict(int)
        for endpoint_health in health_report.values():
            status_counts[endpoint_health['status']] += 1
            
        total_endpoints = len(health_report)
        
        if status_counts.get('poor', 0) > total_endpoints * 0.2:  # بیش از 20% poor
            return 'poor'
        elif status_counts.get('degraded', 0) > total_endpoints * 0.3:  # بیش از 30% degraded
            return 'degraded'
        elif status_counts.get('excellent', 0) >= total_endpoints * 0.8:  # حداقل 80% excellent
            return 'excellent'
        else:
            return 'good'

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """گزارش عملکرد اندپوینت‌ها"""
        recent_calls = self.debug_manager.get_recent_calls(limit=1000)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_calls = [
            call for call in recent_calls 
            if datetime.fromisoformat(call['timestamp']) >= cutoff_time
        ]
        
        endpoint_performance = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0,
            'errors': 0,
            'cache_hits': 0,
            'normalization_errors': 0,
            'total_quality_score': 0
        })
        
        for call in filtered_calls:
            ep = endpoint_performance[call['endpoint']]
            ep['calls'] += 1
            ep['total_time'] += call['response_time']
            
            if call['status_code'] >= 400:
                ep['errors'] += 1
                
            if call['cache_used']:
                ep['cache_hits'] += 1
            
            # آمار نرمال‌سازی
            norm_info = call.get('normalization_info', {})
            if norm_info.get('status') == 'error':
                ep['normalization_errors'] += 1
            ep['total_quality_score'] += norm_info.get('quality_score', 0)
        
        # محاسبه میانگین‌ها
        report = {}
        for endpoint, data in endpoint_performance.items():
            total_calls = data['calls']
            report[endpoint] = {
                'total_calls': total_calls,
                'average_response_time': data['total_time'] / total_calls if total_calls > 0 else 0,
                'error_rate': (data['errors'] / total_calls * 100) if total_calls > 0 else 0,
                'cache_hit_rate': (data['cache_hits'] / total_calls * 100) if total_calls > 0 else 0,
                'normalization_performance': {
                    'error_rate': (data['normalization_errors'] / total_calls * 100) if total_calls > 0 else 0,
                    'avg_quality_score': (data['total_quality_score'] / total_calls) if total_calls > 0 else 0
                }
            }
        
        # متریک‌های کلی نرمال‌سازی
        system_norm_metrics = data_normalizer.get_health_metrics()
        
        return {
            'time_period_hours': hours,
            'total_calls': len(filtered_calls),
            'endpoint_performance': report,
            'system_normalization_metrics': {
                'success_rate': system_norm_metrics.success_rate,
                'total_processed': system_norm_metrics.total_processed,
                'data_quality': system_norm_metrics.data_quality
            },
            'timestamp': datetime.now().isoformat()
        }

    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """آنالیز bottlenecks در اندپوینت‌ها"""
        bottlenecks = []
        all_stats = self.debug_manager.get_endpoint_stats()
        
        for endpoint, stats in all_stats['endpoints'].items():
            issues = []
            
            # بررسی زمان پاسخ بالا
            if stats.get('average_response_time', 0) > 2.0:
                issues.append({
                    'type': 'slow_response',
                    'severity': 'high',
                    'message': f'Average response time {stats["average_response_time"]}s exceeds threshold'
                })
            
            # بررسی نرخ خطای بالا
            if stats.get('success_rate', 100) < 95:
                issues.append({
                    'type': 'high_error_rate',
                    'severity': 'medium',
                    'message': f'Success rate {stats["success_rate"]}% below threshold'
                })
            
            # بررسی نرخ کش پایین
            cache_hit_rate = stats.get('cache_performance', {}).get('hit_rate', 0)
            if cache_hit_rate < 50 and stats.get('total_calls', 0) > 10:
                issues.append({
                    'type': 'low_cache_efficiency',
                    'severity': 'low',
                    'message': f'Cache hit rate {cache_hit_rate}% is low'
                })
            
            # بررسی مشکلات نرمال‌سازی
            norm_perf = stats.get('normalization_performance', {})
            if norm_perf.get('success_rate', 100) < 90:
                issues.append({
                    'type': 'normalization_issues',
                    'severity': 'medium',
                    'message': f'Normalization success rate {norm_perf["success_rate"]}% is low'
                })
            
            if norm_perf.get('avg_quality_score', 100) < 70:
                issues.append({
                    'type': 'data_quality_issues',
                    'severity': 'high',
                    'message': f'Data quality score {norm_perf["avg_quality_score"]}% is low'
                })
            
            if issues:
                bottlenecks.append({
                    'endpoint': endpoint,
                    'issues': issues,
                    'total_calls': stats.get('total_calls', 0)
                })
        
        return sorted(bottlenecks, key=lambda x: len(x['issues']), reverse=True)

# ایجاد نمونه گلوبال
endpoint_monitor = None

def initialize_endpoint_monitor(debug_manager):
    """تابع مقداردهی اولیه برای endpoint monitor"""
    global endpoint_monitor
    endpoint_monitor = EndpointMonitor(debug_manager)
    logger.info("✅ Endpoint Monitor Global Instance Initialized")
    return endpoint_monitor

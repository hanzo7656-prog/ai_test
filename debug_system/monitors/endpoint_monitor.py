import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import inspect
import functools

logger = logging.getLogger(__name__)

class EndpointMonitor:
    def __init__(self, debug_manager):
        self.debug_manager = debug_manager
        self.endpoint_registry = {}
        self.dependency_graph = defaultdict(list)
        self.performance_baselines = {}
        
    def monitor_endpoint(self, endpoint_name: str, method: str = "GET"):
        """دکوراتور برای مانیتورینگ اندپوینت"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                cache_used = False
                api_calls = 0
                status_code = 200
                
                try:
                    # اجرای اندپوینت
                    result = await func(*args, **kwargs)
                    
                    # بررسی اگر نتیجه شامل status_code باشد
                    if isinstance(result, dict) and 'status_code' in result:
                        status_code = result['status_code']
                    
                    return result
                    
                except Exception as e:
                    status_code = 500
                    raise e
                    
                finally:
                    # محاسبه زمان پاسخ
                    response_time = time.time() - start_time
                    
                    # جمع‌آوری پارامترها (بدون اطلاعات حساس)
                    safe_params = self._sanitize_parameters(kwargs)
                    
                    # ثبت در دیباگ منیجر
                    self.debug_manager.log_endpoint_call(
                        endpoint=endpoint_name,
                        method=method,
                        params=safe_params,
                        response_time=response_time,
                        status_code=status_code,
                        cache_used=cache_used,
                        api_calls=api_calls
                    )
            
            # ثبت اندپوینت در رجیستری
            self.endpoint_registry[endpoint_name] = {
                'function': wrapper,
                'method': method,
                'registered_at': datetime.now().isoformat()
            }
            
            return wrapper
        return decorator

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
        # این متد می‌تواند برای ثبت جزئیات فراخوانی‌های خارجی استفاده شود
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
        
        return {
            'endpoint': endpoint,
            'status': self._get_health_status(health_score),
            'health_score': health_score,
            'performance': {
                'average_response_time': stats['average_response_time'],
                'success_rate': stats['success_rate'],
                'cache_hit_rate': stats['cache_performance']['hit_rate']
            },
            'last_updated': datetime.now().isoformat()
        }

    def get_all_endpoints_health(self) -> Dict[str, Any]:
        """دریافت سلامت تمام اندپوینت‌ها"""
        all_stats = self.debug_manager.get_endpoint_stats()
        health_report = {}
        
        for endpoint, stats in all_stats['endpoints'].items():
            health_score = self._calculate_health_score(stats)
            health_report[endpoint] = {
                'status': self._get_health_status(health_score),
                'health_score': health_score,
                'performance': {
                    'average_response_time': stats['average_response_time'],
                    'success_rate': stats['success_rate']
                }
            }
        
        return {
            'overall_health': self._calculate_overall_health(health_report),
            'endpoints': health_report,
            'total_endpoints': len(health_report),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_health_score(self, stats: Dict) -> float:
        """محاسبه امتیاز سلامت"""
        score = 0
        
        # امتیاز بر اساس نرخ موفقیت (50%)
        success_rate = stats.get('success_rate', 0)
        score += (success_rate / 100) * 50
        
        # امتیاز بر اساس زمان پاسخ (30%)
        avg_response = stats.get('average_response_time', 0)
        if avg_response < 0.5:
            score += 30
        elif avg_response < 1.0:
            score += 25
        elif avg_response < 2.0:
            score += 20
        elif avg_response < 3.0:
            score += 10
        else:
            score += 5
            
        # امتیاز بر اساس کش (20%)
        cache_hit_rate = stats.get('cache_performance', {}).get('hit_rate', 0)
        score += (cache_hit_rate / 100) * 20
        
        return min(score, 100)

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
            'cache_hits': 0
        })
        
        for call in filtered_calls:
            ep = endpoint_performance[call['endpoint']]
            ep['calls'] += 1
            ep['total_time'] += call['response_time']
            if call['status_code'] >= 400:
                ep['errors'] += 1
            if call['cache_used']:
                ep['cache_hits'] += 1
        
        # محاسبه میانگین‌ها
        report = {}
        for endpoint, data in endpoint_performance.items():
            report[endpoint] = {
                'total_calls': data['calls'],
                'average_response_time': data['total_time'] / data['calls'] if data['calls'] > 0 else 0,
                'error_rate': (data['errors'] / data['calls'] * 100) if data['calls'] > 0 else 0,
                'cache_hit_rate': (data['cache_hits'] / data['calls'] * 100) if data['calls'] > 0 else 0
            }
        
        return {
            'time_period_hours': hours,
            'total_calls': len(filtered_calls),
            'endpoint_performance': report,
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
            
            if issues:
                bottlenecks.append({
                    'endpoint': endpoint,
                    'issues': issues,
                    'total_calls': stats.get('total_calls', 0)
                })
        
        return sorted(bottlenecks, key=lambda x: len(x['issues']), reverse=True)

# ایجاد نمونه گلوبال (بعداً در main.py مقداردهی می‌شود)
endpoint_monitor = None

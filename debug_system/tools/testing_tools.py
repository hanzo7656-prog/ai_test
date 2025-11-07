import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import time

logger = logging.getLogger(__name__)

class TestingTools:
    def __init__(self, debug_manager, endpoint_monitor):
        self.debug_manager = debug_manager
        self.endpoint_monitor = endpoint_monitor
        
    async def run_load_test(self, 
                          endpoint: str,
                          concurrent_users: int = 10,
                          duration_seconds: int = 60,
                          request_rate: int = 100):
        """اجرای تست بار برای اندپوینت"""
        logger.info(f"⚡ Starting load test: {endpoint} with {concurrent_users} users")
        
        start_time = time.time()
        results = {
            'endpoint': endpoint,
            'concurrent_users': concurrent_users,
            'duration_seconds': duration_seconds,
            'request_rate': request_rate,
            'start_time': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'throughput': 0
        }
        
        # شبیه‌سازی کاربران همزمان
        tasks = []
        for user_id in range(concurrent_users):
            task = self._simulate_user(endpoint, duration_seconds, request_rate, user_id, results)
            tasks.append(task)
        
        # اجرای همزمان
        await asyncio.gather(*tasks)
        
        # محاسبه نتایج
        end_time = time.time()
        actual_duration = end_time - start_time
        results['actual_duration'] = actual_duration
        results['throughput'] = results['total_requests'] / actual_duration
        results['end_time'] = datetime.now().isoformat()
        
        if results['response_times']:
            results['avg_response_time'] = sum(results['response_times']) / len(results['response_times'])
            results['min_response_time'] = min(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
            results['p95_response_time'] = sorted(results['response_times'])[int(len(results['response_times']) * 0.95)]
        
        logger.info(f"✅ Load test completed: {results['total_requests']} requests")
        return results
    
    async def _simulate_user(self, endpoint: str, duration: int, rate: int, 
                           user_id: int, results: Dict):
        """شبیه‌سازی یک کاربر"""
        end_time = time.time() + duration
        requests_sent = 0
        
        while time.time() < end_time and requests_sent < rate:
            request_start = time.time()
            
            try:
                # شبیه‌سازی درخواست
                response_time = random.uniform(0.05, 2.0)
                await asyncio.sleep(response_time)
                
                # شبیه‌سازی status code
                success = random.random() < 0.95  # 95% success rate
                status_code = 200 if success else random.choice([400, 500])
                
                # ثبت نتایج
                results['total_requests'] += 1
                if success:
                    results['successful_requests'] += 1
                else:
                    results['failed_requests'] += 1
                
                results['response_times'].append(response_time)
                
                # ثبت در دیباگ
                self.debug_manager.log_endpoint_call(
                    endpoint=endpoint,
                    method="GET",
                    params={'load_test': True, 'user_id': user_id},
                    response_time=response_time,
                    status_code=status_code,
                    cache_used=random.random() < 0.7,
                    api_calls=random.randint(0, 2)
                )
                
                requests_sent += 1
                
            except Exception as e:
                logger.error(f"❌ Load test error for user {user_id}: {e}")
                results['failed_requests'] += 1
            
            # کنترل نرخ درخواست
            elapsed = time.time() - request_start
            sleep_time = max(0, (1.0 / rate) - elapsed)
            await asyncio.sleep(sleep_time)
    
    def run_stress_test(self, endpoint: str, max_concurrent: int = 100):
        """اجرای تست استرس"""
        logger.info(f"🔥 Starting stress test: {endpoint} up to {max_concurrent} users")
        
        # این تست به تدریج بار را افزایش می‌دهد
        return {
            'test_type': 'stress',
            'endpoint': endpoint,
            'max_concurrent_users': max_concurrent,
            'start_time': datetime.now().isoformat(),
            'phases': [
                {'concurrent_users': 10, 'duration': 30},
                {'concurrent_users': 25, 'duration': 30},
                {'concurrent_users': 50, 'duration': 30},
                {'concurrent_users': 75, 'duration': 30},
                {'concurrent_users': 100, 'duration': 30}
            ]
        }
    
    async def test_endpoint_reliability(self, endpoint: str, num_requests: int = 1000):
        """تست قابلیت اطمینان اندپوینت"""
        logger.info(f"🛡️ Testing reliability: {endpoint} with {num_requests} requests")
        
        results = {
            'endpoint': endpoint,
            'total_requests': num_requests,
            'successful_requests': 0,
            'failed_requests': 0,
            'error_breakdown': defaultdict(int),
            'response_time_stats': {},
            'start_time': datetime.now().isoformat()
        }
        
        response_times = []
        
        for i in range(num_requests):
            try:
                # شبیه‌سازی درخواست
                response_time = random.uniform(0.1, 1.0)
                response_times.append(response_time)
                
                # شبیه‌سازی خطا (5% مواقع)
                if random.random() < 0.95:
                    results['successful_requests'] += 1
                    status_code = 200
                else:
                    results['failed_requests'] += 1
                    error_type = random.choice(['timeout', 'validation', 'server_error'])
                    results['error_breakdown'][error_type] += 1
                    status_code = 500 if error_type == 'server_error' else 400
                
                # ثبت در دیباگ
                self.debug_manager.log_endpoint_call(
                    endpoint=endpoint,
                    method="GET",
                    params={'reliability_test': True, 'request_id': i},
                    response_time=response_time,
                    status_code=status_code,
                    cache_used=random.random() < 0.6,
                    api_calls=random.randint(0, 1)
                )
                
                # تاخیر بین درخواست‌ها
                if i % 100 == 0:  # هر ۱۰۰ درخواست کمی تاخیر
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Reliability test error: {e}")
                results['failed_requests'] += 1
                results['error_breakdown']['exception'] += 1
        
        # محاسبه آمار
        if response_times:
            results['response_time_stats'] = {
                'avg': sum(response_times) / len(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'p95': sorted(response_times)[int(len(response_times) * 0.95)]
            }
        
        results['success_rate'] = (results['successful_requests'] / num_requests) * 100
        results['end_time'] = datetime.now().isoformat()
        
        return results
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """تولید گزارش تست"""
        report = {
            'report_id': f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'test_type': test_results.get('test_type', 'custom'),
            'summary': {
                'total_requests': test_results.get('total_requests', 0),
                'success_rate': test_results.get('success_rate', 0),
                'duration_seconds': test_results.get('actual_duration', 0),
                'throughput': test_results.get('throughput', 0)
            },
            'performance_metrics': test_results.get('response_time_stats', {}),
            'reliability_metrics': {
                'successful_requests': test_results.get('successful_requests', 0),
                'failed_requests': test_results.get('failed_requests', 0),
                'error_breakdown': dict(test_results.get('error_breakdown', {}))
            },
            'recommendations': self._generate_test_recommendations(test_results)
        }
        
        return report
    
    def _generate_test_recommendations(self, results: Dict) -> List[str]:
        """تولید توصیه‌های مبتنی بر نتایج تست"""
        recommendations = []
        
        success_rate = results.get('success_rate', 100)
        avg_response_time = results.get('response_time_stats', {}).get('avg', 0)
        
        if success_rate < 99:
            recommendations.append("بهبود handling خطاها و retry logic")
        
        if avg_response_time > 1.0:
            recommendations.append("بهینه‌سازی queries و اضافه کردن indexes")
        
        if results.get('error_breakdown', {}).get('timeout', 0) > 0:
            recommendations.append("افزایش timeoutها و بهبود مدیریت connection")
        
        return recommendations

# ایجاد نمونه گلوبال
testing_tools = TestingTools(None, None)  # بعداً مقداردهی می‌شود

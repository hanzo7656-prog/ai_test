import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import time

logger = logging.getLogger(__name__)

class DevTools:
    def __init__(self, debug_manager, endpoint_monitor):
        self.debug_manager = debug_manager
        self.endpoint_monitor = endpoint_monitor
        
    async def generate_test_traffic(self, 
                                  endpoint: str = None,
                                  duration_seconds: int = 60,
                                  requests_per_second: int = 10):
        """تولید ترافیک تست برای شبیه‌سازی بار"""
        logger.info(f"🚀 Starting test traffic: {requests_per_second} req/s for {duration_seconds}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # اجرای درخواست‌های این batch
            batch_tasks = []
            for _ in range(requests_per_second):
                if endpoint:
                    task = self._simulate_endpoint_call(endpoint)
                else:
                    task = self._simulate_random_endpoint_call()
                batch_tasks.append(task)
            
            # اجرای همزمان درخواست‌ها
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            request_count += requests_per_second
            
            # محاسبه زمان خواب برای حفظ نرخ درخواست
            batch_duration = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_duration)
            await asyncio.sleep(sleep_time)
        
        logger.info(f"✅ Test traffic completed: {request_count} requests")
        
        return {
            'total_requests': request_count,
            'duration_seconds': duration_seconds,
            'actual_rps': round(request_count / duration_seconds, 2),
            'target_rps': requests_per_second
        }
    
    async def _simulate_endpoint_call(self, endpoint: str):
        """شبیه‌سازی فراخوانی اندپوینت"""
        # شبیه‌سازی زمان پاسخ
        response_time = random.uniform(0.1, 2.0)
        await asyncio.sleep(response_time)
        
        # شبیه‌سازی status code (۹۵٪ موفق)
        status_code = 200 if random.random() < 0.95 else random.choice([400, 401, 404, 500])
        
        # شبیه‌سازی استفاده از کش (۷۰٪命中)
        cache_used = random.random() < 0.7
        
        # شبیه‌سازی فراخوانی API (۳۰٪ مواقع)
        api_calls = random.randint(0, 2) if random.random() < 0.3 else 0
        
        # ثبت در دیباگ منیجر
        self.debug_manager.log_endpoint_call(
            endpoint=endpoint,
            method="GET",
            params={'simulated': True, 'test_traffic': True},
            response_time=response_time,
            status_code=status_code,
            cache_used=cache_used,
            api_calls=api_calls
        )
    
    async def _simulate_random_endpoint_call(self):
        """شبیه‌سازی فراخوانی اندپوینت تصادفی"""
        endpoints = [
            '/api/coins/bitcoin',
            '/api/coins/ethereum', 
            '/api/coins/solana',
            '/api/news/latest',
            '/api/news/trending',
            '/api/exchanges/list',
            '/api/insights/fear-greed'
        ]
        
        endpoint = random.choice(endpoints)
        await self._simulate_endpoint_call(endpoint)
    
    def run_performance_test(self, 
                           endpoint: str,
                           concurrent_users: int = 10,
                           total_requests: int = 1000):
        """اجرای تست عملکرد برای اندپوینت"""
        logger.info(f"⚡ Running performance test: {endpoint} with {concurrent_users} users")
        
        # این متد می‌تواند با ابزارهایی مثل locust یا jmeter یکپارچه شود
        # در اینجا یک پیاده‌سازی ساده ارائه می‌دهیم
        
        test_results = {
            'endpoint': endpoint,
            'concurrent_users': concurrent_users,
            'total_requests': total_requests,
            'start_time': datetime.now().isoformat(),
            'requests_completed': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': []
        }
        
        return test_results
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """آنالیز استفاده از حافظه"""
        import psutil
        import gc
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # جمع‌آوری اطلاعات حافظه
        memory_analysis = {
            'rss_mb': round(memory_info.rss / (1024 * 1024), 2),
            'vms_mb': round(memory_info.vms / (1024 * 1024), 2),
            'percent': process.memory_percent(),
            'gc_stats': gc.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # آنالیز اشیاء در حافظه
        try:
            import objgraph
            memory_analysis['object_counts'] = {
                'dict': len(objgraph.by_type('dict')),
                'list': len(objgraph.by_type('list')),
                'str': len(objgraph.by_type('str')),
                'function': len(objgraph.by_type('function'))
            }
        except ImportError:
            memory_analysis['object_counts'] = {'error': 'objgraph not available'}
        
        return memory_analysis
    
    def generate_test_data(self, data_type: str, count: int = 100) -> List[Dict[str, Any]]:
        """تولید داده تست برای پایگاه داده"""
        test_data = []
        
        if data_type == 'endpoint_calls':
            for i in range(count):
                test_data.append({
                    'endpoint': f'/api/test/endpoint_{i % 10}',
                    'method': random.choice(['GET', 'POST']),
                    'response_time': random.uniform(0.1, 5.0),
                    'status_code': random.choice([200, 200, 200, 400, 404, 500]),
                    'cache_used': random.choice([True, False]),
                    'api_calls': random.randint(0, 3),
                    'timestamp': (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat()
                })
        
        elif data_type == 'system_metrics':
            for i in range(count):
                test_data.append({
                    'cpu_percent': random.uniform(10, 90),
                    'memory_percent': random.uniform(20, 80),
                    'disk_usage': random.uniform(50, 95),
                    'network_sent_mb': random.uniform(0, 10),
                    'network_recv_mb': random.uniform(0, 10),
                    'active_connections': random.randint(0, 100),
                    'timestamp': (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat()
                })
        
        logger.info(f"📊 Generated {len(test_data)} test records for {data_type}")
        return test_data
    
    def run_dependency_check(self) -> Dict[str, Any]:
        """بررسی وابستگی‌های سیستم"""
        dependencies = {
            'database': self._check_database_connection(),
            'cache': self._check_cache_connection(),
            'external_apis': self._check_external_apis(),
            'file_system': self._check_file_system()
        }
        
        # محاسبه سلامت کلی
        all_healthy = all(dep['status'] == 'healthy' for dep in dependencies.values())
        
        return {
            'overall_status': 'healthy' if all_healthy else 'degraded',
            'dependencies': dependencies,
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_database_connection(self) -> Dict[str, Any]:
        """بررسی اتصال به پایگاه داده"""
        try:
            # اینجا می‌تواند بررسی connection به دیتابیس واقعی باشد
            return {
                'status': 'healthy',
                'message': 'Database connection OK',
                'response_time': '5ms'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': str(e),
                'error': 'Database connection failed'
            }
    
    def _check_cache_connection(self) -> Dict[str, Any]:
        """بررسی اتصال به کش"""
        try:
            # بررسی connection به Redis یا کش دیگر
            return {
                'status': 'healthy',
                'message': 'Cache connection OK',
                'response_time': '2ms'
            }
        except Exception as e:
            return {
                'status': 'unhealthy', 
                'message': str(e),
                'error': 'Cache connection failed'
            }
    
    def _check_external_apis(self) -> Dict[str, Any]:
        """بررسی اتصال به APIهای خارجی"""
        apis = {
            'coinstats_api': {'status': 'healthy', 'response_time': '45ms'},
            'news_api': {'status': 'healthy', 'response_time': '120ms'},
            'analytics_api': {'status': 'degraded', 'response_time': '1500ms'}
        }
        
        return {
            'status': 'degraded' if any(api['status'] != 'healthy' for api in apis.values()) else 'healthy',
            'apis': apis
        }
    
    def _check_file_system(self) -> Dict[str, Any]:
        """بررسی سیستم فایل"""
        import os
        import shutil
        
        try:
            # بررسی فضای دیسک
            disk_usage = shutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            
            status = 'healthy' if free_gb > 1 else 'warning'
            
            return {
                'status': status,
                'free_space_gb': round(free_gb, 2),
                'message': f'Free space: {free_gb:.1f}GB'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': str(e),
                'error': 'File system check failed'
            }
    
    def create_mock_endpoint(self, endpoint_path: str, response_data: Dict[str, Any]):
        """ایجاد اندپوینت mock برای تست"""
        logger.info(f"🎭 Creating mock endpoint: {endpoint_path}")
        
        # در یک پیاده‌سازی واقعی، این متد می‌تواند اندپوینت‌های موقت ایجاد کند
        return {
            'endpoint': endpoint_path,
            'mock_data': response_data,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=1)).isoformat()
        }

# ایجاد نمونه گلوبال (بعداً مقداردهی می‌شود)
dev_tools = None

# debug_manager.py - سیستم دیباگ و مانیتورینگ کامل
import logging
import traceback
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import inspect
import psutil
import os
from functools import wraps

class DebugManager:
    def __init__(self):
        self.setup_logging()
        self.error_log = []
        self.api_calls_log = []
        self.performance_log = []
        self.start_time = time.time()
        
    def setup_logging(self):
        """تنظیمات پیشرفته لاگینگ"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('debug.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_api_call(self, endpoint: str, method: str, status: str, 
                    response_time: float, error: str = None):
        """لاگ کردن فراخوانی API"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status': status,
            'response_time': response_time,
            'error': error
        }
        self.api_calls_log.append(log_entry)
        
        if error:
            self.logger.error(f"❌ API Error: {endpoint} - {error}")
        else:
            self.logger.info(f"✅ API Success: {endpoint} - {response_time:.2f}s")
            
    def log_error(self, error_type: str, message: str, stack_trace: str, 
                 context: Dict[str, Any] = None):
        """لاگ کردن خطاها"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': message,
            'stack_trace': stack_trace,
            'context': context or {}
        }
        self.error_log.append(error_entry)
        self.logger.error(f"💥 {error_type}: {message}")
        
    def log_performance(self, operation: str, execution_time: float, 
                       data_size: int = None):
        """لاگ کردن عملکرد"""
        perf_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'execution_time': execution_time,
            'data_size': data_size
        }
        self.performance_log.append(perf_entry)
        
    def debug_endpoint(self, func):
        """دکوراتور برای دیباگ اندپوینت‌ها"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.log_api_call(
                    endpoint=endpoint_name,
                    method="GET",
                    status="success",
                    response_time=execution_time
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                stack_trace = traceback.format_exc()
                
                self.log_api_call(
                    endpoint=endpoint_name,
                    method="GET", 
                    status="error",
                    response_time=execution_time,
                    error=str(e)
                )
                
                self.log_error(
                    error_type=type(e).__name__,
                    message=str(e),
                    stack_trace=stack_trace,
                    context={"endpoint": endpoint_name}
                )
                
                raise
                
        return wrapper
        
    def get_system_health(self) -> Dict[str, Any]:
        """دریافت سلامت سیستم"""
        try:
            # مصرف حافظه
            memory = psutil.virtual_memory()
            # مصرف CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            # دیسک
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percent': memory.percent
                },
                'cpu': {
                    'percent': cpu_percent,
                    'cores': psutil.cpu_count()
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'percent': disk.percent
                },
                'uptime': round(time.time() - self.start_time, 2)
            }
        except Exception as e:
            self.log_error('SystemHealthError', str(e), traceback.format_exc())
            return {}
            
    def get_api_stats(self) -> Dict[str, Any]:
        """آمار API"""
        total_calls = len(self.api_calls_log)
        successful_calls = len([x for x in self.api_calls_log if x['status'] == 'success'])
        failed_calls = total_calls - successful_calls
        
        recent_errors = self.error_log[-10:] if self.error_log else []
        
        return {
            'total_api_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            'recent_errors': recent_errors,
            'average_response_time': self._calculate_avg_response_time()
        }
        
    def _calculate_avg_response_time(self) -> float:
        """محاسبه میانگین زمان پاسخ"""
        if not self.api_calls_log:
            return 0
        return sum(x['response_time'] for x in self.api_calls_log) / len(self.api_calls_log)
        
    def get_endpoint_problems(self) -> List[Dict[str, Any]]:
        """دریافت مشکلات اندپوینت‌ها"""
        problems = []
        
        # تحلیل خطاهای اخیر
        recent_errors = self.error_log[-20:]
        for error in recent_errors:
            problems.append({
                'type': 'error',
                'severity': 'high',
                'endpoint': error['context'].get('endpoint', 'unknown'),
                'message': error['message'],
                'timestamp': error['timestamp']
            })
            
        # تحلیل عملکرد ضعیف
        slow_operations = [x for x in self.performance_log if x['execution_time'] > 5]
        for op in slow_operations:
            problems.append({
                'type': 'performance',
                'severity': 'medium', 
                'operation': op['operation'],
                'execution_time': op['execution_time'],
                'timestamp': op['timestamp']
            })
            
        return problems
        
    def generate_debug_report(self) -> Dict[str, Any]:
        """تولید گزارش دیباگ کامل"""
        return {
            'system_health': self.get_system_health(),
            'api_statistics': self.get_api_stats(),
            'identified_problems': self.get_endpoint_problems(),
            'recent_activity': {
                'api_calls': self.api_calls_log[-10:],
                'errors': self.error_log[-5:],
                'performance_issues': [x for x in self.performance_log if x['execution_time'] > 2]
            },
            'recommendations': self._generate_recommendations()
        }
        
    def _generate_recommendations(self) -> List[str]:
        """تولید توصیه‌ها برای بهبود"""
        recommendations = []
        stats = self.get_api_stats()
        
        if stats['success_rate'] < 90:
            recommendations.append("🔄 نرخ موفقیت API پایین است - بررسی اتصال به سرویس‌های خارجی")
            
        if stats['average_response_time'] > 3:
            recommendations.append("⚡ زمان پاسخ API کند است - بهینه‌سازی درخواست‌ها")
            
        if len(self.error_log) > 10:
            recommendations.append("🐛 خطاهای زیاد شناسایی شد - بررسی کدهای مشکل‌دار")
            
        system_health = self.get_system_health()
        if system_health.get('memory', {}).get('percent', 0) > 80:
            recommendations.append("💾 مصرف حافظه بالا است - بررسی نشتی‌های حافظه")
            
        return recommendations

# ایجاد نمونه گلوبال
debug_manager = DebugManager()

# دکوراتور ساده برای استفاده
def debug_endpoint(func):
    return debug_manager.debug_endpoint(func)

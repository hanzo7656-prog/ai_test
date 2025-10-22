# 📁 src/monitoring/health_check.py

import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    name: str
    status: str  # healthy, degraded, failed
    last_check: datetime
    response_time: float
    details: Dict

class SystemHealthChecker:
    """بررسی سلامت سیستم و کامپوننت‌ها"""
    
    def __init__(self):
        self.components: Dict[str, ComponentStatus] = {}
        self.start_time = datetime.now()
        self.successful_cycles = 0
        self.failed_cycles = 0
    
    def update_component_status(self, component_name: str, status: str, details: Dict = None):
        """به‌روزرسانی وضعیت یک کامپوننت"""
        self.components[component_name] = ComponentStatus(
            name=component_name,
            status=status,
            last_check=datetime.now(),
            response_time=details.get('response_time', 0) if details else 0,
            details=details or {}
        )
    
    def record_successful_cycle(self):
        """ثبت سیکل موفق"""
        self.successful_cycles += 1
    
    def record_failed_cycle(self):
        """ثبت سیکل ناموفق"""
        self.failed_cycles += 1
    
    def get_system_status(self) -> Dict:
        """دریافت وضعیت کلی سیستم"""
        # بررسی منابع سیستم
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # وضعیت کامپوننت‌ها
        healthy_components = sum(1 for comp in self.components.values() if comp.status == 'healthy')
        total_components = len(self.components)
        
        # محاسبه uptime
        uptime = datetime.now() - self.start_time
        
        return {
            'overall_status': 'healthy' if healthy_components == total_components else 'degraded',
            'system_metrics': {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'uptime_seconds': uptime.total_seconds()
            },
            'component_health': {
                'total_components': total_components,
                'healthy_components': healthy_components,
                'degraded_components': sum(1 for comp in self.components.values() if comp.status == 'degraded'),
                'failed_components': sum(1 for comp in self.components.values() if comp.status == 'failed')
            },
            'performance_metrics': {
                'successful_cycles': self.successful_cycles,
                'failed_cycles': self.failed_cycles,
                'success_rate': self.successful_cycles / max(1, self.successful_cycles + self.failed_cycles) * 100
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def check_database_health(self) -> Dict:
        """بررسی سلامت دیتابیس"""
        try:
            # شبیه‌سازی بررسی دیتابیس
            return {
                'status': 'healthy',
                'response_time': 0.1,
                'connections': 5,
                'details': {'message': 'Database connection successful'}
            }
        except Exception as e:
            return {
                'status': 'failed', 
                'response_time': 0,
                'details': {'error': str(e)}
            }
    
    def check_api_health(self) -> Dict:
        """بررسی سلامت API"""
        try:
            # شبیه‌سازی بررسی API
            return {
                'status': 'healthy',
                'response_time': 0.05,
                'details': {'message': 'API endpoints responding correctly'}
            }
        except Exception as e:
            return {
                'status': 'failed',
                'response_time': 0,
                'details': {'error': str(e)}
            }

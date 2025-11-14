# ml_core/health_monitor.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import psutil
import torch

logger = logging.getLogger(__name__)

class MLHealthMonitor:
    """مانیتورینگ سلامت سیستم هوش مصنوعی"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.health_history = []
        self.start_time = datetime.now()
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🏥 ML Health Monitor initialized")

    def get_system_health(self) -> Dict[str, Any]:
        """دریافت سلامت کلی سیستم AI"""
        try:
            # سلامت مدل‌ها
            models_health = self._get_models_health()
            
            # سلامت منابع سیستم
            system_health = self._get_system_resources()
            
            # سلامت دیتابیس‌های AI
            database_health = self._get_database_health()
            
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'components': {
                    'models': models_health,
                    'system': system_health,
                    'databases': database_health
                },
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'version': '1.0.0'
            }
            
            # تعیین وضعیت کلی
            if any(comp['status'] == 'degraded' for comp in [models_health, system_health, database_health]):
                health_report['overall_status'] = 'degraded'
            elif any(comp['status'] == 'unhealthy' for comp in [models_health, system_health, database_health]):
                health_report['overall_status'] = 'unhealthy'
            
            # ذخیره در کش برای گزارش به روت سلامت مادر
            self.cache_manager.set_data("mother_a", "ml_health_report", health_report, expire=300)
            
            return health_report
            
        except Exception as e:
            logger.error(f"❌ Error in health monitoring: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }

    def _get_models_health(self) -> Dict[str, Any]:
        """سلامت مدل‌های ثبت شده"""
        models_health = {
            'status': 'healthy',
            'total_models': len(self.model_manager.active_models),
            'models': {}
        }
        
        for model_name in self.model_manager.active_models.keys():
            try:
                model_health = self.model_manager.get_model_health(model_name)
                models_health['models'][model_name] = model_health
                
                if model_health.get('health') == 'degraded':
                    models_health['status'] = 'degraded'
                elif model_health.get('health') == 'unhealthy':
                    models_health['status'] = 'unhealthy'
                    
            except Exception as e:
                models_health['models'][model_name] = {'status': 'error', 'error': str(e)}
                models_health['status'] = 'degraded'
        
        return models_health

    def _get_system_resources(self) -> Dict[str, Any]:
        """سلامت منابع سیستم"""
        try:
            # استفاده از CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # استفاده از حافظه
            memory = psutil.virtual_memory()
            
            # استفاده از GPU (اگر موجود باشد)
            gpu_info = self._get_gpu_info()
            
            system_health = {
                'status': 'healthy',
                'cpu': {
                    'usage_percent': cpu_percent,
                    'status': 'healthy' if cpu_percent < 80 else 'degraded'
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'available_gb': round(memory.available / (1024**3), 2),
                    'total_gb': round(memory.total / (1024**3), 2),
                    'status': 'healthy' if memory.percent < 85 else 'degraded'
                },
                'gpu': gpu_info
            }
            
            # تعیین وضعیت کلی سیستم
            if (cpu_percent > 90 or memory.percent > 95):
                system_health['status'] = 'unhealthy'
            elif (cpu_percent > 80 or memory.percent > 85):
                system_health['status'] = 'degraded'
                
            return system_health
            
        except Exception as e:
            logger.error(f"❌ Error getting system resources: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """اطلاعات GPU"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpus = []
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    total = props.total_memory / (1024**3)
                    
                    gpu_usage = (allocated / total) * 100 if total > 0 else 0
                    
                    gpus.append({
                        'name': props.name,
                        'memory_allocated_gb': round(allocated, 2),
                        'memory_cached_gb': round(cached, 2),
                        'memory_total_gb': round(total, 2),
                        'usage_percent': round(gpu_usage, 2),
                        'status': 'healthy' if gpu_usage < 90 else 'degraded'
                    })
                
                return {
                    'available': True,
                    'count': gpu_count,
                    'devices': gpus
                }
            else:
                return {'available': False}
                
        except Exception as e:
            logger.error(f"❌ Error getting GPU info: {e}")
            return {'available': False, 'error': str(e)}

    def _get_database_health(self) -> Dict[str, Any]:
        """سلامت دیتابیس‌های AI"""
        try:
            from debug_system.storage.redis_manager import redis_manager
            
            database_health = {
                'status': 'healthy',
                'databases': {}
            }
            
            # بررسی سلامت ۳ دیتابیس اصلی AI
            ai_databases = ['uta', 'utb', 'utc']
            
            for db_name in ai_databases:
                try:
                    health_report = redis_manager.health_check(db_name)
                    database_health['databases'][db_name] = health_report
                    
                    if health_report.get('status') != 'connected':
                        database_health['status'] = 'degraded'
                        
                except Exception as e:
                    database_health['databases'][db_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    database_health['status'] = 'degraded'
            
            return database_health
            
        except Exception as e:
            logger.error(f"❌ Error checking database health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های عملکرد"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': self.model_manager.performance_metrics,
            'system_health': self._get_system_resources(),
            'request_stats': self._get_request_statistics()
        }
        
        return metrics

    def _get_request_statistics(self) -> Dict[str, Any]:
        """آمار درخواست‌های پردازش شده"""
        try:
            # دریافت از کش
            stats = self.cache_manager.get_data("uta", "request_statistics") or {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_response_time': 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting request statistics: {e}")
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_response_time': 0
            }

# نمونه global
ml_health_monitor = None

def initialize_health_monitor(model_manager):
    """مقداردهی اولیه مانیتور سلامت"""
    global ml_health_monitor
    ml_health_monitor = MLHealthMonitor(model_manager)
    return ml_health_monitor

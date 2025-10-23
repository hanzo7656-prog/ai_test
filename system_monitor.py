# system_monitor.py - نسخه اصلاح شده
import psutil
import os
import time
from datetime import datetime

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_system_usage(self):
        """دریافت مصرف واقعی منابع سیستم"""
        try:
            # مصرف RAM
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # مصرف CPU
            cpu_percent = self.process.cpu_percent()
            
            # مصرف دیسک
            disk_usage = psutil.disk_usage('.')
            disk_used_gb = disk_usage.used / 1024 / 1024 / 1024
            disk_total_gb = disk_usage.total / 1024 / 1024 / 1024
            
            return {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'used_mb': round(memory_mb, 2),
                    'percent': round(self.process.memory_percent(), 2),
                    'available_mb': round(psutil.virtual_memory().available / 1024 / 1024, 2),
                    'total_mb': round(psutil.virtual_memory().total / 1024 / 1024, 2)
                },
                'cpu': {
                    'process_percent': round(cpu_percent, 2),
                    'system_percent': round(psutil.cpu_percent(), 2)
                },
                'disk': {
                    'used_gb': round(disk_used_gb, 2),
                    'total_gb': round(disk_total_gb, 2),
                    'percent': round(disk_usage.percent, 2)
                }
            }
        except Exception as e:
            # اگر خطا در محاسبات پیش آمد
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'memory': {'used_mb': 0, 'percent': 0},
                'cpu': {'process_percent': 0, 'system_percent': 0},
                'disk': {'used_gb': 0, 'total_gb': 0, 'percent': 0}
            }

def get_project_size():
    """محاسبه حجم واقعی پروژه"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if any(ignore in dirpath for ignore in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                    continue
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    continue
        return round(total_size / 1024 / 1024, 2)
    except Exception as e:
        return 0

def get_library_sizes():
    """محاسبه حجم واقعی کتابخانه‌ها"""
    try:
        lib_sizes = {}
        import site
        libs_to_check = ['fastapi', 'uvicorn', 'torch', 'numpy', 'websocket', 'requests', 'pydantic', 'psutil']
        
        for sitepack in site.getsitepackages():
            for lib in libs_to_check:
                lib_path = os.path.join(sitepack, lib)
                if os.path.exists(lib_path):
                    size = 0
                    for dirpath, dirnames, filenames in os.walk(lib_path):
                        for filename in filenames:
                            try:
                                filepath = os.path.join(dirpath, filename)
                                size += os.path.getsize(filepath)
                            except OSError:
                                continue
                    lib_sizes[lib] = round(size / 1024 / 1024, 2)
        return lib_sizes
    except Exception as e:
        return {}

def get_cache_size():
    """محاسبه حجم کش داده‌ها"""
    try:
        import glob
        cache_size = 0
        cache_patterns = [
            "coinstats_collected_data/*.json",
            "raw_data/**/*.json",
            "raw_data/**/*.csv",
            "*.log",
            "logs/*.log"
        ]
        
        for pattern in cache_patterns:
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(filepath):
                        cache_size += os.path.getsize(filepath)
                except OSError:
                    continue
        
        return round(cache_size / 1024 / 1024, 2)
    except Exception as e:
        return 0

def get_log_size():
    """محاسبه حجم فایل‌های لاگ"""
    try:
        import glob
        log_size = 0
        log_patterns = ["*.log", "logs/*.log", "*.txt"]
        
        for pattern in log_patterns:
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(filepath):
                        log_size += os.path.getsize(filepath)
                except OSError:
                    continue
        
        return round(log_size / 1024 / 1024, 2)
    except Exception as e:
        return 0

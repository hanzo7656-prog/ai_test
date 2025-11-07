import psutil
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
import json

logger = logging.getLogger(__name__)

class RealTimeMetricsCollector:
    def __init__(self):
        self.metrics_buffer = deque(maxlen=3600)  # 1 hour of metrics (1 per second)
        self.process = psutil.Process()
        
        # متریک‌های Real-Time
        self.current_metrics = {
            'cpu': {'percent': 0, 'per_core': [], 'load_avg': []},
            'memory': {'percent': 0, 'used_gb': 0, 'available_gb': 0},
            'disk': {'usage_percent': 0, 'io_read': 0, 'io_write': 0},
            'network': {'bytes_sent': 0, 'bytes_recv': 0, 'connections': 0},
            'process': {'memory_mb': 0, 'cpu_percent': 0, 'threads': 0}
        }
        
        self._start_real_time_collection()
    
    def _start_real_time_collection(self):
        """شروع جمع‌آوری Real-Time متریک‌ها"""
        def collect_metrics():
            last_disk_io = psutil.disk_io_counters()
            last_net_io = psutil.net_io_counters()
            
            while True:
                try:
                    # جمع‌آوری متریک‌ها
                    metrics = self._collect_all_metrics(last_disk_io, last_net_io)
                    self.metrics_buffer.append(metrics)
                    self.current_metrics = metrics
                    
                    # آپدیت آخرین مقادیر برای محاسبه تفاضلی
                    last_disk_io = psutil.disk_io_counters()
                    last_net_io = psutil.net_io_counters()
                    
                    time.sleep(1)  # هر ثانیه
                    
                except Exception as e:
                    logger.error(f"❌ Real-time metrics collection error: {e}")
                    time.sleep(5)
        
        collection_thread = threading.Thread(target=collect_metrics, daemon=True)
        collection_thread.start()
        logger.info("✅ Real-time metrics collection started")
    
    def _collect_all_metrics(self, last_disk_io, last_net_io) -> Dict[str, Any]:
        """جمع‌آوری تمام متریک‌ها"""
        timestamp = datetime.now()
        
        # CPU متریک‌ها
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_core_percent = psutil.cpu_percent(percpu=True, interval=0.1)
        
        # حافظه
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # دیسک
        disk_usage = psutil.disk_usage('/')
        current_disk_io = psutil.disk_io_counters()
        
        # شبکه
        current_net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())
        
        # پردازش
        process_memory = self.process.memory_info()
        process_cpu = self.process.cpu_percent()
        process_threads = self.process.num_threads()
        
        # محاسبه تفاضلی برای IO
        disk_io_read = current_disk_io.read_bytes - last_disk_io.read_bytes if last_disk_io else 0
        disk_io_write = current_disk_io.write_bytes - last_disk_io.write_bytes if last_disk_io else 0
        net_io_sent = current_net_io.bytes_sent - last_net_io.bytes_sent if last_net_io else 0
        net_io_recv = current_net_io.bytes_recv - last_net_io.bytes_recv if last_net_io else 0
        
        return {
            'timestamp': timestamp,
            'cpu': {
                'percent': cpu_percent,
                'per_core': per_core_percent,
                'load_average': self._get_load_average(),
                'frequency': self._get_cpu_frequency()
            },
            'memory': {
                'percent': memory.percent,
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'total_gb': round(memory.total / (1024**3), 2),
                'swap_percent': swap.percent,
                'swap_used_gb': round(swap.used / (1024**3), 2)
            },
            'disk': {
                'usage_percent': disk_usage.percent,
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'io_read_bytes_per_sec': disk_io_read,
                'io_write_bytes_per_sec': disk_io_write,
                'io_read_mb_per_sec': round(disk_io_read / (1024**2), 3),
                'io_write_mb_per_sec': round(disk_io_write / (1024**2), 3)
            },
            'network': {
                'bytes_sent_per_sec': net_io_sent,
                'bytes_recv_per_sec': net_io_recv,
                'mb_sent_per_sec': round(net_io_sent / (1024**2), 3),
                'mb_recv_per_sec': round(net_io_recv / (1024**2), 3),
                'connections': connections,
                'packets_sent': current_net_io.packets_sent,
                'packets_recv': current_net_io.packets_recv
            },
            'process': {
                'memory_mb': round(process_memory.rss / (1024**2), 2),
                'cpu_percent': process_cpu,
                'threads': process_threads,
                'open_files': len(self.process.open_files()),
                'connections': len(self.process.connections())
            },
            'system': {
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'users': len(psutil.users()),
                'temperature': self._get_temperature()
            }
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های فعلی"""
        return self.current_metrics
    
    def get_metrics_history(self, seconds: int = 300) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌ها"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        return [
            {
                'timestamp': metrics['timestamp'].isoformat(),
                'cpu_percent': metrics['cpu']['percent'],
                'memory_percent': metrics['memory']['percent'],
                'disk_usage': metrics['disk']['usage_percent'],
                'network_sent_mb_sec': metrics['network']['mb_sent_per_sec'],
                'network_recv_mb_sec': metrics['network']['mb_recv_per_sec'],
                'process_memory_mb': metrics['process']['memory_mb']
            }
            for metrics in self.metrics_buffer
            if metrics['timestamp'] >= cutoff_time
        ]
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های دقیق"""
        return self.current_metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه متریک‌ها"""
        metrics = self.current_metrics
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_usage': f"{metrics['cpu']['percent']}%",
                'memory_usage': f"{metrics['memory']['percent']}%",
                'disk_usage': f"{metrics['disk']['usage_percent']}%",
                'network_activity': f"↑{metrics['network']['mb_sent_per_sec']}MB/s ↓{metrics['network']['mb_recv_per_sec']}MB/s"
            },
            'process_health': {
                'memory_usage': f"{metrics['process']['memory_mb']}MB",
                'cpu_usage': f"{metrics['process']['cpu_percent']}%",
                'threads': metrics['process']['threads']
            }
        }
    
    def _get_load_average(self) -> List[float]:
        """دریافت load average"""
        try:
            return list(psutil.getloadavg())
        except:
            return [0, 0, 0]
    
    def _get_cpu_frequency(self) -> Dict[str, float]:
        """دریافت فرکانس CPU"""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return {
                    'current': freq.current,
                    'min': freq.min,
                    'max': freq.max
                }
        except:
            pass
        return {'current': 0, 'min': 0, 'max': 0}
    
    def _get_temperature(self) -> Optional[Dict[str, float]]:
        """دریافت دمای سیستم"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # بازگشت اولین سنسور دمایی که پیدا شود
                for name, entries in temps.items():
                    if entries:
                        return {
                            'sensor': name,
                            'current': entries[0].current,
                            'high': entries[0].high,
                            'critical': entries[0].critical
                        }
        except:
            pass
        return None

# ایجاد نمونه گلوبال
metrics_collector = RealTimeMetricsCollector()

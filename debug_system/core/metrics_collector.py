import psutil
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
import json

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

class RealTimeMetricsCollector:
    """
    نسخه بهینه‌شده RealTimeMetricsCollector
    - اتصال به central_monitor برای کاهش مصرف منابع
    - حفظ backward compatibility کامل
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
            }
        }
        
        self.cache_last_updated = None
        self.cache_ttl = 5  # 5 seconds
        
        # اتصال به central_monitor
        self._connect_to_central_monitor()
        
        logger.info("✅ RealTimeMetricsCollector Initialized - Central Monitor Connected")
    
    def _connect_to_central_monitor(self):
        """اتصال به سیستم مانیتورینگ مرکزی"""
        try:
            # تاخیر برای اطمینان از لود شدن central_monitor
            def delayed_connection():
                time.sleep(3)
                self._subscribe_to_monitor()
            
            connect_thread = threading.Thread(target=delayed_connection, daemon=True)
            connect_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Error connecting to central monitor: {e}")
            # Fallback: راه‌اندازی جمع‌آوری حداقلی
            self._start_minimal_collection()
    
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
            else:
                logger.warning("⚠️ Central monitor not available, starting fallback collection")
                self._start_minimal_collection()
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor, starting fallback collection")
            self._start_minimal_collection()
        except Exception as e:
            logger.error(f"❌ Error subscribing to monitor: {e}")
            self._start_minimal_collection()
    
    def _on_system_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌های سیستم از central_monitor"""
        try:
            system_metrics = metrics.get('system', {})
            
            # آپدیت کش
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
                    'io_read': 0,  # اینها فقط در central_monitor محاسبه می‌شوند
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
            
            logger.debug(f"📈 System metrics updated from central_monitor - CPU: {system_metrics.get('cpu', {}).get('percent', 0)}%")
            
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
            
            logger.debug(f"📊 Normalization metrics updated from central_monitor")
            
        except Exception as e:
            logger.error(f"❌ Error processing normalization metrics: {e}")
    
    def _add_to_history_buffer(self, system_metrics: Dict[str, Any]):
        """اضافه کردن متریک‌ها به بافر تاریخچه"""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'cpu_percent': system_metrics.get('cpu', {}).get('percent', 0),
                'memory_percent': system_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': system_metrics.get('disk', {}).get('usage_percent', 0),
                'network_sent_mb_sec': 0,  # از central_monitor می‌آید
                'network_recv_mb_sec': 0,  # از central_monitor می‌آید
                'process_memory_mb': system_metrics.get('process', {}).get('memory_rss_mb', 0),
                'normalization_success_rate': self.current_metrics_cache['data_normalization']['success_rate'],
                'normalization_total_processed': self.current_metrics_cache['data_normalization']['total_processed']
            }
            
            self.metrics_buffer.append(history_entry)
            
        except Exception as e:
            logger.error(f"❌ Error adding to history buffer: {e}")
    
    def _start_minimal_collection(self):
        """راه‌اندازی جمع‌آوری حداقلی (fallback)"""
        def minimal_collection_loop():
            """حلقه جمع‌آوری حداقلی - هر 30 ثانیه"""
            last_disk_io = psutil.disk_io_counters()
            last_net_io = psutil.net_io_counters()
            
            while True:
                try:
                    # فقط متریک‌های ضروری هر 30 ثانیه
                    metrics = self._collect_minimal_metrics(last_disk_io, last_net_io)
                    
                    # آپدیت کش
                    self.current_metrics_cache.update({
                        'cpu': {'percent': metrics['cpu']['percent'], 'per_core': [], 'load_average': []},
                        'memory': {'percent': metrics['memory']['percent'], 'used_gb': 0, 'available_gb': 0},
                        'disk': {'usage_percent': metrics['disk']['usage_percent'], 'io_read': 0, 'io_write': 0},
                        'network': {'bytes_sent': 0, 'bytes_recv': 0, 'connections': 0},
                        'process': {'memory_mb': metrics['process']['memory_mb'], 'cpu_percent': 0, 'threads': 0}
                    })
                    
                    self.cache_last_updated = datetime.now()
                    
                    # اضافه به تاریخچه
                    self.metrics_buffer.append({
                        'timestamp': datetime.now(),
                        'cpu_percent': metrics['cpu']['percent'],
                        'memory_percent': metrics['memory']['percent'],
                        'disk_usage': metrics['disk']['usage_percent'],
                        'network_sent_mb_sec': 0,
                        'network_recv_mb_sec': 0,
                        'process_memory_mb': metrics['process']['memory_mb'],
                        'normalization_success_rate': self.current_metrics_cache['data_normalization']['success_rate'],
                        'normalization_total_processed': self.current_metrics_cache['data_normalization']['total_processed']
                    })
                    
                    # آپدیت normalization هر 60 ثانیه
                    if int(time.time()) % 60 == 0:
                        self._refresh_normalization_metrics()
                    
                    time.sleep(30)  # هر 30 ثانیه
                    
                except Exception as e:
                    logger.error(f"❌ Minimal collection error: {e}")
                    time.sleep(60)
        
        collection_thread = threading.Thread(target=minimal_collection_loop, daemon=True)
        collection_thread.start()
        logger.info("🔄 Minimal metrics collection started (30s interval)")
    
    def _collect_minimal_metrics(self, last_disk_io, last_net_io) -> Dict[str, Any]:
        """جمع‌آوری حداقلی متریک‌ها"""
        timestamp = datetime.now()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        
        # Disk
        disk_usage = psutil.disk_usage('/')
        
        # Process
        process_memory = self.process.memory_info()
        
        return {
            'timestamp': timestamp,
            'cpu': {
                'percent': cpu_percent,
                'load_average': self._get_load_average()
            },
            'memory': {
                'percent': memory.percent
            },
            'disk': {
                'usage_percent': disk_usage.percent
            },
            'process': {
                'memory_mb': round(process_memory.rss / (1024**2), 2)
            }
        }
    
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
            
            logger.debug(f"🔄 Normalization metrics refreshed")
            
        except Exception as e:
            logger.error(f"❌ Error refreshing normalization metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های فعلی - API بدون تغییر"""
        # بررسی منقضی شدن کش
        if (self.cache_last_updated and 
            (datetime.now() - self.cache_last_updated).total_seconds() > self.cache_ttl):
            logger.debug("⚠️ Metrics cache expired, returning cached data")
        
        # برگرداندن ساختار دقیقاً مشابه قبل
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
                'normalization_total_processed': metrics['normalization_total_processed']
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
                'network_activity': "Central Monitor Active"
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
            }
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """دریافت گزارش جامع - API بدون تغییر"""
        current_metrics = self.get_current_metrics()
        metrics_history = self.get_metrics_history(seconds=3600)
        
        # تحلیل روندها
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
                'resource_utilization': 'optimal' if current_metrics['cpu']['percent'] < 70 and current_metrics['memory']['percent'] < 80 else 'high'
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
    
    def _get_load_average(self) -> List[float]:
        """دریافت load average"""
        try:
            return list(psutil.getloadavg())
        except:
            return [0, 0, 0]
    
    def get_connection_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال به central_monitor"""
        return {
            'cache_age_seconds': (datetime.now() - self.cache_last_updated).total_seconds() if self.cache_last_updated else None,
            'metrics_buffer_size': len(self.metrics_buffer),
            'cache_ttl': self.cache_ttl,
            'collection_mode': 'central_monitor' if self.cache_last_updated else 'fallback',
            'timestamp': datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال با همان نام دقیق
metrics_collector = RealTimeMetricsCollector()

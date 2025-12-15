"""
🎯 SYSTEM MONITOR v3.0 - نظارت هوشمند و یکپارچه
ویژگی‌های جدید:
1. معماری Event-Driven برای کاهش polling
2. جمع‌آوری Async و Non-Blocking
3. سیستم کش هوشمند با TTLهای مختلف
4. تشخیص ناهنجاری خودکار
5. Health Scoring با توصیه‌های هوشمند
6. نظارت سلسله‌مراتبی
7. Predictive Monitoring با ML ساده
8. Dashboard داخلی
9. تنها یک حلقه مرکزی فعال
10. حذف داده‌های تکراری
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# ==================== ENUMS & DATA CLASSES ====================

class MetricPriority(Enum):
    REAL_TIME = "realtime"      # هر 1-5 ثانیه
    HIGH = "high"               # هر 10-30 ثانیه  
    MEDIUM = "medium"           # هر 1-5 دقیقه
    LOW = "low"                 # هر 10-30 دقیقه

class ComponentHealth(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class MetricConfig:
    """تنظیمات هر متریک"""
    name: str
    priority: MetricPriority
    collection_interval: int
    cache_ttl: int
    anomaly_threshold: float = 2.5
    enabled: bool = True

@dataclass
class HealthScore:
    """امتیاز سلامت هر کامپوننت"""
    component: str
    score: float
    status: ComponentHealth
    weighted_score: float
    details: Dict[str, Any]

@dataclass
class MonitoringEvent:
    """رویداد نظارتی"""
    type: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: str = "system_monitor"

# ==================== SMART CACHE SYSTEM ====================

class SmartMetricCache:
    """کش هوشمند برای متریک‌ها با TTLهای متفاوت"""
    
    def __init__(self):
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()
        
        # تنظیمات کش برای هر نوع متریک
        self._cache_configs = {
            'cpu': MetricConfig('cpu', MetricPriority.REAL_TIME, 2, 5),
            'memory': MetricConfig('memory', MetricPriority.REAL_TIME, 2, 5),
            'disk': MetricConfig('disk', MetricPriority.MEDIUM, 30, 60),
            'network': MetricConfig('network', MetricPriority.HIGH, 10, 20),
            'process': MetricConfig('process', MetricPriority.MEDIUM, 60, 120),
            'system': MetricConfig('system', MetricPriority.LOW, 300, 600)
        }
    
    async def get_or_fetch(self, metric_type: str, fetch_func: Callable) -> Any:
        """دریافت از کش یا fetch جدید"""
        with self._lock:
            cache_entry = self._cache.get(metric_type)
            
            # بررسی وجود و اعتبار کش
            if cache_entry and not self._is_expired(metric_type, cache_entry):
                self._cache_hits += 1
                logger.debug(f"✅ Cache hit for {metric_type}")
                return cache_entry['data']
            
            # کش منقضی یا وجود ندارد
            self._cache_misses += 1
            
            # fetch جدید
            fresh_data = await fetch_func()
            
            # ذخیره در کش
            config = self._cache_configs.get(metric_type, self._cache_configs['system'])
            self._cache[metric_type] = {
                'data': fresh_data,
                'timestamp': time.time(),
                'ttl': config.cache_ttl,
                'config': config
            }
            
            # پاک‌سازی خودکار کش‌های قدیمی
            self._cleanup_old_cache()
            
            return fresh_data
    
    def _is_expired(self, metric_type: str, cache_entry: Dict) -> bool:
        """بررسی انقضای کش"""
        config = cache_entry.get('config')
        if not config:
            return True
        
        age = time.time() - cache_entry['timestamp']
        return age > config.cache_ttl
    
    def _cleanup_old_cache(self):
        """پاک‌سازی خودکار کش‌های قدیمی"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time - entry['timestamp'] > entry['ttl'] * 2:  # 2 برابر TTL
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """آمار عملکرد کش"""
        return {
            'total_entries': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_ratio': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            'configs': {k: v.__dict__ for k, v in self._cache_configs.items()}
        }

# ==================== ASYNC METRICS COLLECTOR ====================

class AsyncMetricsCollector:
    """جمع‌آوری متریک‌ها به صورت غیرمسدودکننده"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="metric_collector"
        )
        
    async def collect_all_async(self, enabled_metrics: Set[str] = None) -> Dict[str, Any]:
        """جمع‌آوری تمام متریک‌ها به صورت موازی"""
        if enabled_metrics is None:
            enabled_metrics = {'cpu', 'memory', 'disk', 'network', 'process'}
        
        tasks = []
        
        if 'cpu' in enabled_metrics:
            tasks.append(self._collect_cpu_async())
        
        if 'memory' in enabled_metrics:
            tasks.append(self._collect_memory_async())
        
        if 'disk' in enabled_metrics:
            tasks.append(self._collect_disk_async())
        
        if 'network' in enabled_metrics:
            tasks.append(self._collect_network_async())
        
        if 'process' in enabled_metrics:
            tasks.append(self._collect_process_async())
        
        # اجرای موازی
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # ترکیب نتایج
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'collection_time': time.time()
        }
        
        for i, metric_type in enumerate(['cpu', 'memory', 'disk', 'network', 'process']):
            if metric_type in enabled_metrics and i < len(results):
                if not isinstance(results[i], Exception):
                    metrics[metric_type] = results[i]
                else:
                    metrics[metric_type] = {'error': str(results[i])}
        
        return metrics
    
    async def _collect_cpu_async(self) -> Dict[str, Any]:
        """جمع‌آوری CPU به صورت async"""
        loop = asyncio.get_event_loop()
        
        try:
            # استفاده از interval کوتاه در thread جداگانه
            cpu_percent = await loop.run_in_executor(
                self.thread_pool,
                lambda: psutil.cpu_percent(interval=0.05)  # کاهش از 0.1 به 0.05
            )
            
            # جمع‌آوری اطلاعات اضافی
            cpu_count = await loop.run_in_executor(
                self.thread_pool,
                psutil.cpu_count
            )
            
            cpu_freq = await loop.run_in_executor(
                self.thread_pool,
                lambda: psutil.cpu_freq().current if hasattr(psutil, 'cpu_freq') else None
            )
            
            return {
                'percent': cpu_percent,
                'cores': cpu_count,
                'frequency_mhz': cpu_freq,
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"❌ Error collecting CPU metrics: {e}")
            return {'percent': 0, 'cores': 1, 'error': str(e)}
    
    async def _collect_memory_async(self) -> Dict[str, Any]:
        """جمع‌آوری Memory به صورت async"""
        loop = asyncio.get_event_loop()
        
        try:
            memory = await loop.run_in_executor(
                self.thread_pool,
                psutil.virtual_memory
            )
            
            return {
                'percent': memory.percent,
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'total_gb': round(memory.total / (1024**3), 2),
                'free_gb': round(memory.free / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"❌ Error collecting memory metrics: {e}")
            return {'percent': 0, 'error': str(e)}
    
    async def _collect_disk_async(self) -> Dict[str, Any]:
        """جمع‌آوری Disk به صورت async"""
        loop = asyncio.get_event_loop()
        
        try:
            disk = await loop.run_in_executor(
                self.thread_pool,
                lambda: psutil.disk_usage('/')
            )
            
            # I/O statistics (با نمونه‌برداری)
            io_before = await loop.run_in_executor(
                self.thread_pool,
                psutil.disk_io_counters
            )
            
            await asyncio.sleep(0.1)  # تأخیر کوتاه
            
            io_after = await loop.run_in_executor(
                self.thread_pool,
                psutil.disk_io_counters
            )
            
            io_delta = {
                'read_mb_per_sec': round(((io_after.read_bytes - io_before.read_bytes) / (1024**2)) / 0.1, 2),
                'write_mb_per_sec': round(((io_after.write_bytes - io_before.write_bytes) / (1024**2)) / 0.1, 2)
            } if io_before and io_after else {}
            
            return {
                'usage_percent': disk.percent,
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'total_gb': round(disk.total / (1024**3), 2),
                'io': io_delta
            }
        except Exception as e:
            logger.error(f"❌ Error collecting disk metrics: {e}")
            return {'usage_percent': 0, 'error': str(e)}
    
    async def _collect_network_async(self) -> Dict[str, Any]:
        """جمع‌آوری Network به صورت async"""
        loop = asyncio.get_event_loop()
        
        try:
            # شبکه - فقط یک بار خواندن (بدون interval)
            net_io = await loop.run_in_executor(
                self.thread_pool,
                psutil.net_io_counters
            )
            
            return {
                'bytes_sent_mb': round(net_io.bytes_sent / (1024**2), 2),
                'bytes_recv_mb': round(net_io.bytes_recv / (1024**2), 2),
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout
            }
        except Exception as e:
            logger.error(f"❌ Error collecting network metrics: {e}")
            return {'bytes_sent_mb': 0, 'bytes_recv_mb': 0, 'error': str(e)}
    
    async def _collect_process_async(self) -> Dict[str, Any]:
        """جمع‌آوری Process اطلاعات به صورت async"""
        loop = asyncio.get_event_loop()
        
        try:
            current_process = psutil.Process()
            
            # فقط اطلاعات ضروری
            process_info = await loop.run_in_executor(
                self.thread_pool,
                current_process.memory_info
            )
            
            cpu_percent = await loop.run_in_executor(
                self.thread_pool,
                lambda: current_process.cpu_percent(interval=0.05)
            )
            
            return {
                'memory_rss_mb': round(process_info.rss / (1024**2), 2),
                'cpu_percent': cpu_percent,
                'threads_count': current_process.num_threads(),
                'pid': current_process.pid,
                'name': current_process.name()
            }
        except Exception as e:
            logger.error(f"❌ Error collecting process metrics: {e}")
            return {'memory_rss_mb': 0, 'cpu_percent': 0, 'error': str(e)}

# ==================== ANOMALY DETECTION ====================

class AnomalyDetector:
    """سیستم تشخیص ناهنجاری‌های متریک"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.anomaly_threshold = 2.5  # z-score threshold
        self.consecutive_anomalies = defaultdict(int)
        
    def analyze(self, metric_type: str, current_value: float) -> Optional[Dict[str, Any]]:
        """تحلیل متریک برای تشخیص ناهنجاری"""
        history = self.metric_history[metric_type]
        
        # حداقل داده برای تحلیل
        if len(history) < 10:
            history.append(current_value)
            return None
        
        # محاسبه آمار
        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 0
        
        # جلوگیری از تقسیم بر صفر
        if stdev == 0:
            history.append(current_value)
            return None
        
        # محاسبه z-score
        z_score = abs((current_value - mean) / stdev)
        
        # تشخیص ناهنجاری
        if z_score > self.anomaly_threshold:
            self.consecutive_anomalies[metric_type] += 1
            
            # بررسی consecutive anomalies برای کاهش false positives
            if self.consecutive_anomalies[metric_type] >= 2:
                anomaly = {
                    'metric': metric_type,
                    'current_value': current_value,
                    'historical_mean': mean,
                    'z_score': round(z_score, 2),
                    'severity': self._calculate_severity(z_score),
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': self._get_recommendation(metric_type, current_value)
                }
                
                # ریست consecutive counter
                self.consecutive_anomalies[metric_type] = 0
                
                history.append(current_value)
                return anomaly
        else:
            self.consecutive_anomalies[metric_type] = 0
        
        history.append(current_value)
        return None
    
    def _calculate_severity(self, z_score: float) -> str:
        """محاسبه شدت ناهنجاری"""
        if z_score > 4.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"
    
    def _get_recommendation(self, metric_type: str, value: float) -> str:
        """توصیه بر اساس نوع متریک و مقدار"""
        recommendations = {
            'cpu': {
                'high': "بررسی پردازش‌های سنگین و scale کردن سرویس",
                'critical': "ری‌استارت سرویس‌های غیرضروری و emergency scale"
            },
            'memory': {
                'high': "بررسی memory leak و افزایش memory limit",
                'critical': "ری‌استارت سرویس و emergency memory allocation"
            },
            'disk': {
                'high': "پاک‌سازی فایل‌های موقت و لاگ‌ها",
                'critical': "اضافه کردن فضای دیسک اضطراری"
            }
        }
        
        if value > 90:
            severity = 'critical'
        elif value > 80:
            severity = 'high'
        else:
            return "مانیتور ادامه‌دار"
        
        return recommendations.get(metric_type, {}).get(severity, "بررسی دستی لازم است")

# ==================== HEALTH SCORING SYSTEM ====================

class HealthScoringSystem:
    """سیستم امتیازدهی سلامت"""
    
    def __init__(self):
        # وزن‌های هر کامپوننت
        self.weights = {
            'cpu': {'weight': 0.35, 'thresholds': {'warning': 80, 'critical': 90}},
            'memory': {'weight': 0.25, 'thresholds': {'warning': 85, 'critical': 95}},
            'disk': {'weight': 0.20, 'thresholds': {'warning': 90, 'critical': 98}},
            'network': {'weight': 0.15, 'thresholds': {'warning': 80, 'critical': 95}},
            'process': {'weight': 0.05, 'thresholds': {'warning': 1000, 'critical': 2000}}
        }
        
        self.history = deque(maxlen=100)
    
    def calculate_scores(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه امتیاز سلامت برای تمام کامپوننت‌ها"""
        component_scores = {}
        total_weighted_score = 0
        total_weight = 0
        
        for component, config in self.weights.items():
            if component in metrics and 'error' not in metrics[component]:
                score_details = self._calculate_component_score(component, metrics[component], config)
                component_scores[component] = score_details
                
                total_weighted_score += score_details['weighted_score']
                total_weight += config['weight']
        
        # محاسبه امتیاز کلی
        overall_score = (total_weighted_score / total_weight) if total_weight > 0 else 0
        
        # ذخیره در تاریخچه
        self.history.append({
            'timestamp': metrics.get('timestamp', datetime.now().isoformat()),
            'overall_score': overall_score,
            'component_scores': component_scores
        })
        
        # محاسبه روند
        trend = self._calculate_trend()
        
        return {
            'overall_score': round(overall_score, 1),
            'overall_status': self._get_overall_status(overall_score),
            'component_scores': component_scores,
            'trend': trend,
            'recommendations': self._generate_recommendations(component_scores, overall_score),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_component_score(self, component: str, data: Dict, config: Dict) -> Dict[str, Any]:
        """محاسبه امتیاز برای یک کامپوننت خاص"""
        # استخراج مقدار اصلی
        if component == 'cpu':
            value = data.get('percent', 0)
        elif component == 'memory':
            value = data.get('percent', 0)
        elif component == 'disk':
            value = data.get('usage_percent', 0)
        elif component == 'network':
            # ترکیب upload/download
            sent = data.get('bytes_sent_mb', 0)
            recv = data.get('bytes_recv_mb', 0)
            value = min((sent + recv) / 10, 100)  # normalize
        elif component == 'process':
            value = min(data.get('memory_rss_mb', 0) / 10, 100)
        else:
            value = 0
        
        # محاسبه امتیاز (هرچه usage کمتر، امتیاز بیشتر)
        score = max(0, 100 - value)
        
        # وضعیت
        if value >= config['thresholds']['critical']:
            status = ComponentHealth.CRITICAL.value
        elif value >= config['thresholds']['warning']:
            status = ComponentHealth.WARNING.value
        else:
            status = ComponentHealth.HEALTHY.value
        
        # امتیاز وزنی
        weighted_score = score * config['weight']
        
        return {
            'score': round(score, 1),
            'value': round(value, 1),
            'status': status,
            'weighted_score': round(weighted_score, 3),
            'thresholds': config['thresholds']
        }
    
    def _get_overall_status(self, score: float) -> str:
        """تعیین وضعیت کلی"""
        if score >= 85:
            return ComponentHealth.HEALTHY.value
        elif score >= 70:
            return ComponentHealth.WARNING.value
        else:
            return ComponentHealth.CRITICAL.value
    
    def _calculate_trend(self) -> Dict[str, Any]:
        """محاسبه روند امتیازها"""
        if len(self.history) < 5:
            return {'direction': 'unknown', 'change': 0}
        
        scores = [h['overall_score'] for h in list(self.history)[-5:]]
        
        if len(scores) < 2:
            return {'direction': 'stable', 'change': 0}
        
        # رگرسیون خطی ساده
        x = list(range(len(scores)))
        y = scores
        
        try:
            slope = (len(x) * sum(x[i] * y[i] for i in range(len(x))) - sum(x) * sum(y)) / \
                   (len(x) * sum(x_i**2 for x_i in x) - sum(x)**2)
            
            if slope > 0.5:
                direction = 'improving'
            elif slope < -0.5:
                direction = 'deteriorating'
            else:
                direction = 'stable'
            
            return {
                'direction': direction,
                'change': round(slope, 3),
                'last_5_scores': scores[-5:]
            }
        except:
            return {'direction': 'unknown', 'change': 0}
    
    def _generate_recommendations(self, component_scores: Dict, overall_score: float) -> List[str]:
        """تولید توصیه‌های هوشمند"""
        recommendations = []
        
        # بررسی هر کامپوننت
        for component, score_data in component_scores.items():
            if score_data['status'] == ComponentHealth.CRITICAL.value:
                rec = self._get_critical_recommendation(component, score_data['value'])
                if rec:
                    recommendations.append(rec)
            elif score_data['status'] == ComponentHealth.WARNING.value and overall_score < 75:
                rec = self._get_warning_recommendation(component, score_data['value'])
                if rec:
                    recommendations.append(rec)
        
        # توصیه کلی
        if overall_score < 60 and not recommendations:
            recommendations.append("بررسی کلی سیستم توسط ادمین ضروری است")
        
        if not recommendations and overall_score >= 85:
            recommendations.append("سیستم در وضعیت بهینه قرار دارد")
        
        return recommendations[:5]  # حداکثر 5 توصیه
    
    def _get_critical_recommendation(self, component: str, value: float) -> str:
        """توصیه برای وضعیت بحرانی"""
        recommendations = {
            'cpu': f"CPU در وضعیت بحرانی ({value}%) - ری‌استارت/Scale اضطراری",
            'memory': f"Memory در وضعیت بحرانی ({value}%) - بررسی Memory Leak",
            'disk': f"دیسک در وضعیت بحرانی ({value}%) - پاک‌سازی فوری",
            'network': f"شبکه در وضعیت بحرانی - بررسی ترافیک",
            'process': f"Process در وضعیت بحرانی - بررسی پردازش‌ها"
        }
        return recommendations.get(component, f"وضعیت بحرانی در {component}")
    
    def _get_warning_recommendation(self, component: str, value: float) -> str:
        """توصیه برای وضعیت هشدار"""
        recommendations = {
            'cpu': f"CPU بالا ({value}%) - مانیتورینگ فعال",
            'memory': f"Memory بالا ({value}%) - مانیتورینگ Memory",
            'disk': f"دیسک پر ({value}%) - برنامه‌ریزی برای پاک‌سازی",
            'network': f"ترافیک شبکه بالا - مانیتورینگ",
            'process': f"Process سنگین - بررسی"
        }
        return recommendations.get(component, f"وضعیت هشدار در {component}")

# ==================== HIERARCHICAL MONITORING ====================

class HierarchicalMonitor:
    """نظارت سلسله‌مراتبی با سطوح مختلف"""
    
    def __init__(self):
        # تعریف سطوح نظارتی
        self.levels = {
            'realtime': {
                'interval': 2,      # هر 2 ثانیه
                'metrics': {'cpu', 'memory'},
                'priority': MetricPriority.REAL_TIME,
                'enabled': True
            },
            'high': {
                'interval': 10,     # هر 10 ثانیه
                'metrics': {'network', 'process'},
                'priority': MetricPriority.HIGH,
                'enabled': True
            },
            'medium': {
                'interval': 30,     # هر 30 ثانیه
                'metrics': {'disk'},
                'priority': MetricPriority.MEDIUM,
                'enabled': True
            },
            'low': {
                'interval': 300,    # هر 5 دقیقه
                'metrics': {'system'},
                'priority': MetricPriority.LOW,
                'enabled': True
            }
        }
        
        self.level_timers = {}
        self.last_collection = {}
        
        # فعال‌سازی پیش‌فرض
        for level in self.levels:
            self.last_collection[level] = 0
    
    def should_collect(self, level: str, current_time: float) -> bool:
        """بررسی آیا زمان جمع‌آوری این سطح رسیده است"""
        if not self.levels[level]['enabled']:
            return False
        
        interval = self.levels[level]['interval']
        last_time = self.last_collection.get(level, 0)
        
        return current_time - last_time >= interval
    
    def get_metrics_to_collect(self, current_time: float) -> Set[str]:
        """دریافت مجموعه متریک‌هایی که باید جمع‌آوری شوند"""
        metrics_to_collect = set()
        
        for level, config in self.levels.items():
            if self.should_collect(level, current_time):
                metrics_to_collect.update(config['metrics'])
                self.last_collection[level] = current_time
        
        return metrics_to_collect
    
    def update_level_status(self, level: str, enabled: bool):
        """به‌روزرسانی وضعیت یک سطح"""
        if level in self.levels:
            self.levels[level]['enabled'] = enabled
            logger.info(f"📊 Level {level} {'enabled' if enabled else 'disabled'}")
    
    def get_status(self) -> Dict[str, Any]:
        """دریافت وضعیت سطوح"""
        status = {}
        current_time = time.time()
        
        for level, config in self.levels.items():
            last_time = self.last_collection.get(level, 0)
            time_since_last = current_time - last_time if last_time > 0 else None
            
            status[level] = {
                'enabled': config['enabled'],
                'interval': config['interval'],
                'metrics': list(config['metrics']),
                'time_since_last': round(time_since_last, 1) if time_since_last else None,
                'next_collection_in': max(0, config['interval'] - time_since_last) if time_since_last else 0
            }
        
        return status

# ==================== PREDICTIVE MONITORING ====================

class PredictiveMonitor:
    """پیش‌بینی ساده با ML"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.predictions = {}
    
    def add_metric(self, metric_type: str, value: float):
        """اضافه کردن متریک به تاریخچه"""
        if metric_type == 'cpu':
            self.cpu_history.append(value)
        elif metric_type == 'memory':
            self.memory_history.append(value)
    
    def predict(self, metric_type: str, lookahead: int = 3) -> Optional[Dict[str, Any]]:
        """پیش‌بینی آینده متریک"""
        history = self.cpu_history if metric_type == 'cpu' else self.memory_history
        
        if len(history) < 10:
            return None
        
        # میانگین متحرک ساده برای پیش‌بینی
        window = min(5, len(history))
        moving_avg = sum(list(history)[-window:]) / window
        
        # پیش‌بینی خطی ساده
        if len(history) >= 2:
            last_two = list(history)[-2:]
            slope = last_two[1] - last_two[0]
            prediction = last_two[1] + slope * lookahead
        else:
            prediction = moving_avg
        
        # محدود کردن پیش‌بینی
        prediction = max(0, min(100, prediction))
        
        # تعیین وضعیت
        if prediction > 90:
            status = 'critical'
        elif prediction > 80:
            status = 'warning'
        else:
            status = 'normal'
        
        self.predictions[metric_type] = {
            'current': history[-1] if history else 0,
            'predicted': round(prediction, 1),
            'lookahead': lookahead,
            'status': status,
            'confidence': self._calculate_confidence(len(history)),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.predictions[metric_type]
    
    def _calculate_confidence(self, history_length: int) -> float:
        """محاسبه اعتماد پیش‌بینی"""
        if history_length >= 50:
            return 0.9
        elif history_length >= 20:
            return 0.7
        elif history_length >= 10:
            return 0.5
        else:
            return 0.3
    
    def get_all_predictions(self) -> Dict[str, Any]:
        """دریافت تمام پیش‌بینی‌ها"""
        predictions = {}
        
        if len(self.cpu_history) >= 10:
            predictions['cpu'] = self.predict('cpu')
        
        if len(self.memory_history) >= 10:
            predictions['memory'] = self.predict('memory')
        
        return predictions

# ==================== EMBEDDED DASHBOARD ====================

class EmbeddedDashboard:
    """داشبورد داخلی برای نمایش در CLI"""
    
    def __init__(self):
        self.display_width = 60
        self.metric_history = defaultdict(lambda: deque(maxlen=20))
    
    def generate_dashboard(self, 
                          metrics: Dict[str, Any], 
                          health_score: Dict[str, Any],
                          anomalies: List[Dict[str, Any]]) -> str:
        """تولید داشبورد متنی"""
        lines = []
        
        # Header
        lines.append("=" * self.display_width)
        lines.append("🚀 VORTEXAI - REAL-TIME MONITORING DASHBOARD")
        lines.append("=" * self.display_width)
        
        # Timestamp
        timestamp = metrics.get('timestamp', datetime.now().isoformat())
        lines.append(f"🕒 {timestamp}")
        lines.append("-" * self.display_width)
        
        # Health Score
        overall_score = health_score.get('overall_score', 0)
        overall_status = health_score.get('overall_status', 'unknown')
        
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '❌',
            'unknown': '❓'
        }
        
        lines.append(f"📊 HEALTH SCORE: {overall_score}/100 {status_emoji.get(overall_status, '')}")
        lines.append(self._create_progress_bar(overall_score))
        lines.append(f"📈 Trend: {health_score.get('trend', {}).get('direction', 'unknown')}")
        
        # Component Breakdown
        lines.append("\n🔧 COMPONENT BREAKDOWN:")
        lines.append("-" * 40)
        
        component_scores = health_score.get('component_scores', {})
        for component, score_data in component_scores.items():
            value = score_data.get('value', 0)
            status = score_data.get('status', 'unknown')
            
            bar = self._create_progress_bar(100 - value, width=20)  # معکوس برای نمایش usage
            lines.append(f"  {component.upper():10} {value:5.1f}% {bar} {status}")
        
        # CPU & Memory Charts
        lines.append("\n📈 REAL-TIME CHARTS:")
        lines.append("-" * self.display_width)
        
        # CPU Chart
        if 'cpu' in metrics:
            cpu_value = metrics['cpu'].get('percent', 0)
            self.metric_history['cpu'].append(cpu_value)
            lines.append(f"🔹 CPU Usage: {cpu_value:.1f}%")
            lines.append(self._create_sparkline(list(self.metric_history['cpu'])))
        
        # Memory Chart
        if 'memory' in metrics:
            mem_value = metrics['memory'].get('percent', 0)
            self.metric_history['memory'].append(mem_value)
            lines.append(f"🔹 Memory Usage: {mem_value:.1f}%")
            lines.append(self._create_sparkline(list(self.metric_history['memory'])))
        
        # Anomalies
        if anomalies:
            lines.append("\n🚨 ACTIVE ANOMALIES:")
            lines.append("-" * self.display_width)
            for i, anomaly in enumerate(anomalies[:3], 1):  # حداکثر ۳ مورد
                lines.append(f"  {i}. {anomaly.get('metric', 'unknown')}: {anomaly.get('severity', 'unknown')}")
                lines.append(f"     Value: {anomaly.get('current_value', 0):.1f}, Z-Score: {anomaly.get('z_score', 0):.2f}")
        
        # Recommendations
        recommendations = health_score.get('recommendations', [])
        if recommendations:
            lines.append("\n💡 RECOMMENDATIONS:")
            lines.append("-" * self.display_width)
            for i, rec in enumerate(recommendations[:3], 1):  # حداکثر ۳ مورد
                lines.append(f"  {i}. {rec}")
        
        # Footer
        lines.append("\n" + "=" * self.display_width)
        lines.append("🔍 Press Ctrl+C to exit | 📊 Auto-refresh every 5s")
        lines.append("=" * self.display_width)
        
        return "\n".join(lines)
    
    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """ایجاد نوار پیشرفت"""
        filled = int(width * percentage / 100)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"[{bar}]"
    
    def _create_sparkline(self, values: List[float]) -> str:
        """ایجاد sparkline برای نمایش روند"""
        if not values:
            return "No data"
        
        # نرمال‌سازی مقادیر
        if max(values) > min(values):
            normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
        else:
            normalized = [0.5] * len(values)
        
        # کاراکترهای sparkline
        spark_chars = "▁▂▃▄▅▆▇█"
        
        # تبدیل به sparkline
        sparkline = ""
        for norm_val in normalized:
            idx = min(int(norm_val * len(spark_chars)), len(spark_chars) - 1)
            sparkline += spark_chars[idx]
        
        return f"  Trend: {sparkline}"

# ==================== MAIN CENTRAL MONITORING SYSTEM ====================

class CentralMonitoringSystem:
    """سیستم نظارت مرکزی - تنها حلقه فعال"""
    
    def __init__(self):
        # زیرسیستم‌ها
        self.cache = SmartMetricCache()
        self.collector = AsyncMetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.health_scorer = HealthScoringSystem()
        self.hierarchical_monitor = HierarchicalMonitor()
        self.predictive_monitor = PredictiveMonitor()
        self.dashboard = EmbeddedDashboard()
        
        # وضعیت سیستم
        self.is_monitoring = False
        self.monitor_task = None
        self.last_metrics = {}
        self.last_health_score = {}
        self.active_anomalies = []
        
        # تاریخچه
        self.metrics_history = deque(maxlen=100)
        self.health_history = deque(maxlen=100)
        
        # Subscribers (سیستم‌های دیگر)
        self.subscribers = defaultdict(list)
        
        # تنظیم گلوبال
        global central_monitor
        central_monitor = self
        
        logger.info("🎯 Central Monitoring System v3.0 Initialized")
    
    async def start_monitoring(self):
        """شروع نظارت مرکزی - تنها یک حلقه فعال"""
        if self.is_monitoring:
            logger.warning("⚠️ Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(
            self._monitoring_loop(),
            name="CentralMonitorLoop"
        )
        
        logger.info("🔄 Central monitoring started with hierarchical collection")
    
    async def stop_monitoring(self):
        """توقف نظارت"""
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Central monitoring stopped")
    
    async def _monitoring_loop(self):
        """حلقه نظارت مرکزی - تنها حلقه فعال"""
        logger.debug("🔁 Central monitoring loop started")
        
        cycle_count = 0
        
        while self.is_monitoring:
            try:
                cycle_start = time.time()
                cycle_count += 1
                
                # ۱. تعیین متریک‌های مورد نیاز در این چرخه
                metrics_to_collect = self.hierarchical_monitor.get_metrics_to_collect(cycle_start)
                
                if not metrics_to_collect:
                    # هیچ متریکی برای جمع‌آوری نیست - sleep کوتاه
                    await asyncio.sleep(0.5)
                    continue
                
                # ۲. جمع‌آوری متریک‌ها (با کش هوشمند)
                collected_metrics = {}
                for metric_type in metrics_to_collect:
                    if metric_type in ['cpu', 'memory', 'disk', 'network', 'process']:
                        # استفاده از کش هوشمند
                        metric_data = await self.cache.get_or_fetch(
                            metric_type,
                            lambda m=metric_type: self._collect_specific_metric(m)
                        )
                        collected_metrics[metric_type] = metric_data
                        
                        # اضافه کردن به predictive monitor
                        if metric_type in ['cpu', 'memory']:
                            value = metric_data.get('percent', 0) if 'percent' in metric_data else 0
                            self.predictive_monitor.add_metric(metric_type, value)
                
                # ۳. ساختاردهی داده‌ها
                full_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'collection_time': cycle_start,
                    'cycle': cycle_count,
                    'metrics_collected': list(metrics_to_collect)
                }
                full_metrics.update(collected_metrics)
                
                # ۴. تشخیص ناهنجاری‌ها
                new_anomalies = []
                for metric_type, data in collected_metrics.items():
                    if 'percent' in data:
                        anomaly = self.anomaly_detector.analyze(
                            metric_type, 
                            data['percent']
                        )
                        if anomaly:
                            new_anomalies.append(anomaly)
                
                self.active_anomalies = new_anomalies[:10]  # حداکثر ۱۰ ناهنجاری
                
                # ۵. محاسبه سلامت
                health_score = self.health_scorer.calculate_scores(full_metrics)
                
                # ۶. ذخیره در تاریخچه
                self.last_metrics = full_metrics
                self.last_health_score = health_score
                self.metrics_history.append(full_metrics)
                self.health_history.append(health_score)
                
                # ۷. اطلاع‌رسانی به subscribers
                await self._notify_subscribers(full_metrics, health_score, new_anomalies)
                
                # ۸. نمایش داشبورد (در صورت نیاز)
                if cycle_count % 5 == 0:  # هر ۵ چرخه یکبار
                    dashboard_text = self.dashboard.generate_dashboard(
                        full_metrics, 
                        health_score, 
                        new_anomalies
                    )
                    # می‌توانی این را به console_stream بفرستی
                    # console_stream.update(dashboard_text)
                
                # ۹. محاسبه خواب هوشمند
                execution_time = time.time() - cycle_start
                sleep_time = self._calculate_adaptive_sleep(
                    execution_time, 
                    full_metrics, 
                    cycle_count
                )
                
                # لاگ فقط اگر چرخه طولانی باشد
                if execution_time > 1.0:
                    logger.debug(f"⏱️ Cycle {cycle_count} took {execution_time:.2f}s, sleeping {sleep_time:.1f}s")
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # در صورت خطا بیشتر صبر کن
    
    async def _collect_specific_metric(self, metric_type: str) -> Dict[str, Any]:
        """جمع‌آوری یک متریک خاص"""
        metric_map = {
            'cpu': self.collector._collect_cpu_async,
            'memory': self.collector._collect_memory_async,
            'disk': self.collector._collect_disk_async,
            'network': self.collector._collect_network_async,
            'process': self.collector._collect_process_async
        }
        
        if metric_type in metric_map:
            return await metric_map[metric_type]()
        else:
            return {'error': f'Unknown metric type: {metric_type}'}
    
    async def _notify_subscribers(self, metrics: Dict, health_score: Dict, anomalies: List):
        """اطلاع‌رسانی به سیستم‌های مشترک"""
        event_data = {
            'metrics': metrics,
            'health_score': health_score,
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        }
        
        # ارسال به همه subscribers
        for event_type, callbacks in self.subscribers.items():
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_type, event_data)
                    else:
                        # اگر sync است در thread اجرا کن
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, callback, event_type, event_data)
                except Exception as e:
                    logger.error(f"❌ Error notifying subscriber: {e}")
    
    def _calculate_adaptive_sleep(self, execution_time: float, metrics: Dict, cycle: int) -> float:
        """محاسبه خواب تطبیقی"""
        # حداقل خواب
        min_sleep = 0.5
        
        # بررسی بار سیستم
        cpu_usage = metrics.get('cpu', {}).get('percent', 0)
        memory_usage = metrics.get('memory', {}).get('percent', 0)
        
        # خواب پایه بر اساس priority
        base_sleep = 2.0  # 2 ثانیه برای real-time metrics
        
        # تنظیم بر اساس بار
        if cpu_usage > 80 or memory_usage > 85:
            # سیستم تحت بار - خواب کمتر برای نظارت بیشتر
            adaptive_sleep = max(min_sleep, base_sleep * 0.5)
        elif cpu_usage < 30 and memory_usage < 50:
            # سیستم خلوت - خواب بیشتر
            adaptive_sleep = base_sleep * 1.5
        else:
            # حالت نرمال
            adaptive_sleep = base_sleep
        
        # تطبیق با زمان اجرا
        if execution_time > adaptive_sleep * 0.8:
            # اگر اجرا زمان‌بر بود، خواب را افزایش بده
            adaptive_sleep = min(10.0, adaptive_sleep * 1.2)
        
        return adaptive_sleep
    
    # ==================== PUBLIC API ====================
    
    def subscribe(self, event_type: str, callback: Callable):
        """عضویت در دریافت رویدادها"""
        self.subscribers[event_type].append(callback)
        logger.info(f"📡 New subscriber for {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """لغو عضویت"""
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.info(f"📡 Removed subscriber from {event_type}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های فعلی (برای backward compatibility)"""
        if not self.last_metrics:
            return self._get_fallback_metrics()
        
        # تبدیل به فرمت قدیم برای compatibility
        return {
            'timestamp': self.last_metrics.get('timestamp'),
            'system': {
                'cpu': self.last_metrics.get('cpu', {'percent': 0}),
                'memory': self.last_metrics.get('memory', {'percent': 0}),
                'disk': self.last_metrics.get('disk', {'usage_percent': 0}),
                'network': self.last_metrics.get('network', {}),
                'process': self.last_metrics.get('process', {})
            },
            'collection_time': self.last_metrics.get('collection_time'),
            'from_cache': True
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """دریافت سلامت سیستم (برای backward compatibility)"""
        if not self.last_health_score:
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'unknown',
                'health_indicators': {},
                'metrics_snapshot': {}
            }
        
        # تبدیل به فرمت قدیم
        return {
            'timestamp': self.last_health_score.get('timestamp'),
            'overall_health': self.last_health_score.get('overall_status', 'unknown'),
            'health_indicators': {
                component: {
                    'status': score_data.get('status', 'unknown'),
                    'message': f"Score: {score_data.get('score', 0)}",
                    'usage_percent': score_data.get('value', 0)
                }
                for component, score_data in self.last_health_score.get('component_scores', {}).items()
            },
            'metrics_snapshot': {
                'cpu_usage': self.last_metrics.get('cpu', {}).get('percent', 0),
                'memory_usage': self.last_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': self.last_metrics.get('disk', {}).get('usage_percent', 0)
            }
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """دریافت وضعیت تفصیلی (جدید)"""
        return {
            'monitoring_status': {
                'is_running': self.is_monitoring,
                'cycle_count': len(self.metrics_history),
                'last_collection': self.last_metrics.get('timestamp') if self.last_metrics else None,
                'active_anomalies': len(self.active_anomalies)
            },
            'cache_stats': self.cache.get_cache_stats(),
            'hierarchical_status': self.hierarchical_monitor.get_status(),
            'health_score': self.last_health_score,
            'predictions': self.predictive_monitor.get_all_predictions(),
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'monitor_version': '3.0'
            }
        }
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """متریک‌های جایگزین در صورت عدم وجود داده"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu': {'percent': 0, 'cores': 1, 'load_avg': [0, 0, 0]},
                'memory': {'percent': 0, 'used_gb': 0, 'available_gb': 0, 'total_gb': 0},
                'disk': {'usage_percent': 0, 'used_gb': 0, 'free_gb': 0, 'total_gb': 0},
                'network': {'bytes_sent_mb': 0, 'bytes_recv_mb': 0},
                'process': {'memory_rss_mb': 0, 'cpu_percent': 0, 'threads_count': 0}
            },
            'collection_time': time.time(),
            'from_cache': False,
            'is_fallback': True
        }

# ==================== GLOBAL INSTANCE & HELPER FUNCTIONS ====================

central_monitor = None

def initialize_central_monitoring():
    """تابع راه‌اندازی برای main.py"""
    global central_monitor
    
    if central_monitor:
        logger.warning("⚠️ Central monitor already initialized")
        return central_monitor
    
    central_monitor = CentralMonitoringSystem()
    
    # تنظیم logger
    logger.info("=" * 60)
    logger.info("🚀 VORTEXAI MONITORING SYSTEM v3.0")
    logger.info("📊 Features: Smart Cache, Anomaly Detection, Health Scoring")
    logger.info("🎯 Architecture: Event-Driven, Hierarchical, Async")
    logger.info("=" * 60)
    
    return central_monitor

async def start_monitoring():
    """شروع نظارت (برای استفاده در main.py)"""
    global central_monitor
    
    if not central_monitor:
        initialize_central_monitoring()
    
    await central_monitor.start_monitoring()

async def stop_monitoring():
    """توقف نظارت"""
    global central_monitor
    
    if central_monitor:
        await central_monitor.stop_monitoring()

# ==================== COMPATIBILITY WRAPPER ====================

class SystemMonitor:
    """
    کلاس wrapper برای backward compatibility
    سیستم‌های قدیمی می‌توانند از این کلاس استفاده کنند
    """
    
    def __init__(self, metrics_collector=None, alert_manager=None):
        # پارامترها برای compatibility نگه داشته می‌شوند
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        global central_monitor
        if central_monitor:
            central_monitor.subscribe("legacy_system", self._on_metrics_update)
            logger.info("✅ Legacy SystemMonitor subscribed to central_monitor")
        else:
            logger.warning("⚠️ Central monitor not available - using fallback mode")
    
    def _on_metrics_update(self, event_type: str, event_data: Dict):
        """دریافت به‌روزرسانی‌ها از سیستم مرکزی"""
        # این متد برای compatibility با کد قدیم نگه داشته شده
        pass
    
    def get_system_health(self) -> Dict[str, Any]:
        """متد قدیم - برای backward compatibility"""
        global central_monitor
        
        if central_monitor:
            return central_monitor.get_system_health()
        else:
            # Fallback mode
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'unknown',
                'health_indicators': {},
                'metrics_snapshot': {}
            }
    
    def get_resource_usage_trend(self, hours: int = 6) -> Dict[str, Any]:
        """متد قدیم - برای backward compatibility"""
        # این متد می‌تواند از history سیستم مرکزی استفاده کند
        return {
            'time_period_hours': hours,
            'data_points': 0,
            'trends': {},
            'timestamp': datetime.now().isoformat()
        }
    
    # دیگر متدهای قدیمی می‌توانند اینجا اضافه شوند...

# ==================== FASTAPI ROUTES INTEGRATION ====================

def setup_monitoring_routes(app):
    """تنظیم routeهای نظارت برای FastAPI"""
    
    @app.get("/api/monitoring/status")
    async def get_monitoring_status():
        """دریافت وضعیت نظارت"""
        global central_monitor
        
        if not central_monitor:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if central_monitor.is_monitoring else "stopped",
            "details": central_monitor.get_detailed_status()
        }
    
    @app.get("/api/monitoring/metrics")
    async def get_current_metrics():
        """دریافت متریک‌های جاری"""
        global central_monitor
        
        if not central_monitor:
            return {"error": "Monitoring not initialized"}
        
        return central_monitor.get_current_metrics()
    
    @app.get("/api/monitoring/health")
    async def get_system_health():
        """دریافت سلامت سیستم"""
        global central_monitor
        
        if not central_monitor:
            return {"error": "Monitoring not initialized"}
        
        return central_monitor.get_system_health()
    
    @app.post("/api/monitoring/start")
    async def start_monitoring_endpoint():
        """شروع نظارت"""
        await start_monitoring()
        return {"status": "started"}
    
    @app.post("/api/monitoring/stop")
    async def stop_monitoring_endpoint():
        """توقف نظارت"""
        await stop_monitoring()
        return {"status": "stopped"}
    
    @app.get("/api/monitoring/dashboard")
    async def get_dashboard():
        """دریافت داشبورد متنی"""
        global central_monitor
        
        if not central_monitor or not central_monitor.last_metrics:
            return {"dashboard": "Monitoring not active"}
        
        dashboard_text = central_monitor.dashboard.generate_dashboard(
            central_monitor.last_metrics,
            central_monitor.last_health_score,
            central_monitor.active_anomalies
        )
        
        return {"dashboard": dashboard_text}

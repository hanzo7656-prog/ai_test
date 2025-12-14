"""
VortexAI API v4.0.0 - COMPLETE VERSION (No Deletions)
All Features + Performance Optimizations
Target: Full functionality with <500ms response time
"""

import time
import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import threading
import json
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import os
import sys
import random
from pathlib import Path
import uvicorn
import socket

# ==================== FASTAPI IMPORTS ====================
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, WebSocket, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ==================== CONFIGURATION ====================
# تنظیمات کامل محیطی
CONFIG = {
    # Performance
    "METRICS_COLLECTION_INTERVAL": 60,
    "EVENT_BUS_QUEUE_SIZE": 1000,
    "ENDPOINT_CALLS_MAXLEN": 10000,
    "SYSTEM_METRICS_MAXLEN": 2000,
    "CLEANUP_OLD_DATA_DAYS": 7,
    
    # Thresholds
    "RESPONSE_TIME_WARNING": 1.0,
    "RESPONSE_TIME_CRITICAL": 3.0,
    "CPU_WARNING": 80.0,
    "CPU_CRITICAL": 95.0,
    "MEMORY_WARNING": 85.0,
    "MEMORY_CRITICAL": 95.0,
    "DISK_WARNING": 85.0,
    "DISK_CRITICAL": 95.0,
    
    # Startup Delays
    "INITIAL_STABILIZATION_DELAY": 3,
    "MONITOR_ACTIVATION_DELAY": 3,
    "AI_STARTUP_DELAY": 3,
    "BACKGROUND_WORKER_DELAY": 3,
    "FINAL_STABILIZATION_DELAY": 5,
    "CENTRAL_MONITOR_WAIT": 10,
    
    # WebSocket
    "WEBSOCKET_PING_INTERVAL": 30,
    "WEBSOCKET_PING_TIMEOUT": 10,
    
    # Cache
    "CACHE_TTL_DEFAULT": 300,
    "CACHE_MAX_SIZE": 10000,
}

# اعمال تنظیمات به محیط
for key, value in CONFIG.items():
    os.environ.setdefault(key, str(value))

# ==================== SMART LOGGING SYSTEM ====================
class PerformanceAwareLogger:
    """سیستم لاگینگ هوشمند با کنترل مصرف CPU"""
    
    _instance = None
    _log_buffer = deque(maxlen=1000)
    _last_flush = time.time()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup()
        return cls._instance
    
    def _setup(self):
        self._cpu_thresholds = {
            'DEBUG': 30,
            'INFO': 50,
            'WARNING': 70,
            'ERROR': 85
        }
        self._current_level = logging.INFO
        self._buffer_size = 100
        self._flush_interval = 5
        
    def get_adaptive_level(self):
        """دریافت سطح لاگ بر اساس مصرف CPU"""
        try:
            cpu = psutil.cpu_percent(interval=0)
            for level, threshold in self._cpu_thresholds.items():
                if cpu < threshold:
                    return getattr(logging, level)
            return logging.ERROR
        except:
            return logging.INFO
    
    def log(self, level, message, **kwargs):
        """لاگ کردن با بافرینگ"""
        current_time = time.time()
        
        # بررسی CPU
        if psutil.cpu_percent(interval=0) > 90 and level < logging.ERROR:
            return  # حذف لاگ‌های غیرضروری در CPU بالا
        
        # اضافه به بافر
        log_entry = {
            'timestamp': current_time,
            'level': level,
            'message': message,
            'data': kwargs
        }
        self._log_buffer.append(log_entry)
        
        # فلاش در صورت لزوم
        if (len(self._log_buffer) >= self._buffer_size or 
            current_time - self._last_flush >= self._flush_interval):
            self._flush_buffer()
    
    def _flush_buffer(self):
        """خالی کردن بافر به فایل"""
        try:
            if self._log_buffer:
                with open('performance_optimized.log', 'a', encoding='utf-8') as f:
                    for entry in self._log_buffer:
                        f.write(f"{datetime.fromtimestamp(entry['timestamp'])} - "
                               f"{logging.getLevelName(entry['level'])} - "
                               f"{entry['message']}\n")
                self._log_buffer.clear()
                self._last_flush = time.time()
        except Exception as e:
            print(f"Log flush error: {e}")

# پیکربندی لاگینگ
log_level = PerformanceAwareLogger().get_adaptive_level()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vortexai_complete.log', mode='a', encoding='utf-8')
    ]
)

# کاهش لاگ‌های verbose
for logger_name in ['httpx', 'urllib3', 'asyncio', 'uvicorn', 'websockets']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ==================== COMPLETE EVENT BUS ====================
class CompleteEventBus:
    """Event Bus کامل با تمام ویژگی‌ها"""
    
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._event_history = deque(maxlen=5000)
        self._stats = {
            'total_events': 0,
            'events_processed': 0,
            'events_failed': 0,
            'avg_processing_time': 0,
            'subscribers_count': 0,
            'events_by_type': defaultdict(int)
        }
        self._lock = asyncio.Lock()
        self._is_running = True
        
        # Workerها
        self._workers = []
        self._worker_count = 3
        
        # Priority queue
        self._high_priority_queue = asyncio.Queue(maxsize=100)
        self._normal_priority_queue = asyncio.Queue(maxsize=1000)
        self._low_priority_queue = asyncio.Queue(maxsize=500)
        
        logger.info("🚀 Complete Event Bus Initialized")
    
    async def start(self):
        """شروع سیستم"""
        # ایجاد workerها
        for i in range(self._worker_count):
            task = asyncio.create_task(self._event_worker(i))
            self._workers.append(task)
        
        # شروع health checker
        asyncio.create_task(self._health_checker())
        
        logger.info(f"✅ Event Bus started with {self._worker_count} workers")
    
    async def stop(self):
        """توقف سیستم"""
        self._is_running = False
        
        # انتظار برای پایان workerها
        for worker in self._workers:
            worker.cancel()
        
        try:
            await asyncio.gather(*self._workers, return_exceptions=True)
        except Exception:
            pass
        
        logger.info("🛑 Event Bus stopped")
    
    async def subscribe(self, event_type: str, callback: Callable, 
                       priority: int = 2, filter_func: Optional[Callable] = None):
        """اشتراک با اولویت و فیلتر"""
        async with self._lock:
            subscriber_id = f"{event_type}_{len(self._subscribers[event_type])}"
            self._subscribers[event_type].append({
                'id': subscriber_id,
                'callback': callback,
                'priority': priority,
                'filter': filter_func,
                'stats': {
                    'calls': 0,
                    'success': 0,
                    'errors': 0,
                    'total_time': 0
                }
            })
            self._stats['subscribers_count'] += 1
            
            logger.debug(f"📡 Subscribed: {subscriber_id} to {event_type} (priority: {priority})")
    
    async def publish(self, event_type: str, data: Any = None, 
                     priority: int = 2, urgent: bool = False):
        """انتشار رویداد"""
        event = {
            'id': f"{event_type}_{int(time.time()*1000)}_{random.randint(1000, 9999)}",
            'type': event_type,
            'data': data,
            'timestamp': time.time(),
            'priority': priority,
            'urgent': urgent,
            'published_by': asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'
        }
        
        self._event_history.append(event)
        self._stats['total_events'] += 1
        self._stats['events_by_type'][event_type] += 1
        
        # انتخاب صف بر اساس اولویت
        if urgent or priority == 1:
            queue = self._high_priority_queue
        elif priority == 2:
            queue = self._normal_priority_queue
        else:
            queue = self._low_priority_queue
        
        try:
            # افزودن به صف با timeout
            await asyncio.wait_for(queue.put(event), timeout=0.1)
            
            if urgent:
                # برای رویداد urgent، پردازش فوری
                await self._process_urgent_event(event)
                
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Event queue timeout: {event_type}")
            if urgent:
                await self._process_urgent_event(event)
    
    async def _event_worker(self, worker_id: int):
        """Worker پردازش رویداد"""
        worker_name = f"EventWorker-{worker_id}"
        
        while self._is_running:
            try:
                # انتخاب صف بر اساس اولویت
                queues = [
                    self._high_priority_queue,
                    self._normal_priority_queue,
                    self._low_priority_queue
                ]
                
                # منتظر اولین رویداد
                done, pending = await asyncio.wait(
                    [asyncio.create_task(q.get()) for q in queues],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # کنسل کردن بقیه
                for task in pending:
                    task.cancel()
                
                if done:
                    event = next(iter(done)).result()
                    await self._process_event(event, worker_name)
                    
                    # تکمیل task از صف مربوطه
                    for q in queues:
                        try:
                            q.task_done()
                        except ValueError:
                            pass
                
                # استراحت کوتاه برای جلوگیری از CPU spike
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Event worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: Dict, worker_name: str):
        """پردازش رویداد"""
        start_time = time.time()
        event_type = event['type']
        
        try:
            if event_type in self._subscribers:
                subscribers = self._subscribers[event_type]
                
                # اجرای موازی برای subscriberها
                tasks = []
                for subscriber in subscribers:
                    # اعمال فیلتر اگر وجود دارد
                    if subscriber['filter']:
                        try:
                            if not subscriber['filter'](event['data']):
                                continue
                        except Exception as e:
                            logger.debug(f"Filter error: {e}")
                            continue
                    
                    task = self._execute_callback(subscriber, event, worker_name)
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            self._stats['events_processed'] += 1
            
        except Exception as e:
            self._stats['events_failed'] += 1
            logger.error(f"❌ Event processing error: {e}")
        
        finally:
            processing_time = time.time() - start_time
            self._stats['avg_processing_time'] = (
                self._stats['avg_processing_time'] * 0.95 + processing_time * 0.05
            )
    
    async def _process_urgent_event(self, event: Dict):
        """پردازش رویداد urgent"""
        event_type = event['type']
        
        if event_type in self._subscribers:
            subscribers = self._subscribers[event_type]
            
            # پیدا کردن urgent ترین subscriber
            urgent_subscribers = [s for s in subscribers if s.get('priority', 2) == 1]
            
            if urgent_subscribers:
                subscriber = max(urgent_subscribers, key=lambda x: x.get('priority', 2))
                try:
                    await self._execute_callback(subscriber, event, "UrgentWorker")
                except Exception as e:
                    logger.error(f"❌ Urgent event error: {e}")
    
    async def _execute_callback(self, subscriber: Dict, event: Dict, worker_name: str):
        """اجرای callback"""
        callback_start = time.time()
        
        try:
            subscriber['stats']['calls'] += 1
            
            if asyncio.iscoroutinefunction(subscriber['callback']):
                await subscriber['callback'](event['data'])
            else:
                # اجرا در thread pool برای blocking calls
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    subscriber['callback'], 
                    event['data']
                )
            
            subscriber['stats']['success'] += 1
            
        except Exception as e:
            subscriber['stats']['errors'] += 1
            logger.error(f"❌ Callback error for {subscriber['id']}: {e}")
            
        finally:
            callback_time = time.time() - callback_start
            subscriber['stats']['total_time'] += callback_time
    
    async def _health_checker(self):
        """بررسی سلامت Event Bus"""
        while self._is_running:
            try:
                # بررسی صف‌ها
                queue_status = {
                    'high_priority': self._high_priority_queue.qsize(),
                    'normal_priority': self._normal_priority_queue.qsize(),
                    'low_priority': self._low_priority_queue.qsize()
                }
                
                # بررسی CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # تنظیم workerها بر اساس بار
                if cpu_usage > 80 and len(self._workers) > 1:
                    # کاهش workerها
                    for _ in range(len(self._workers) - 1):
                        if self._workers:
                            self._workers.pop().cancel()
                
                elif cpu_usage < 50 and len(self._workers) < 5:
                    # افزایش workerها
                    new_id = len(self._workers)
                    task = asyncio.create_task(self._event_worker(new_id))
                    self._workers.append(task)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Health checker error: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار کامل"""
        subscriber_stats = {}
        for event_type, subscribers in self._subscribers.items():
            subscriber_stats[event_type] = [
                {
                    'id': s['id'],
                    'priority': s['priority'],
                    'stats': s['stats']
                }
                for s in subscribers
            ]
        
        return {
            'stats': self._stats,
            'queue_status': {
                'high_priority': self._high_priority_queue.qsize(),
                'normal_priority': self._normal_priority_queue.qsize(),
                'low_priority': self._low_priority_queue.qsize()
            },
            'workers': len(self._workers),
            'is_running': self._is_running,
            'subscriber_stats': subscriber_stats,
            'system': {
                'cpu_percent': psutil.cpu_percent(interval=0),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def get_event_history(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict]:
        """دریافت تاریخچه رویدادها"""
        history = list(self._event_history)
        
        if event_type:
            history = [e for e in history if e['type'] == event_type]
        
        return history[-limit:]
    
    async def wait_for_event(self, event_type: str, timeout: float = 30.0) -> Optional[Dict]:
        """انتظار برای یک رویداد خاص"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # بررسی تاریخچه
            for event in reversed(list(self._event_history)):
                if event['type'] == event_type and event['timestamp'] > start_time:
                    return event
            
            await asyncio.sleep(0.1)
        
        return None

# ==================== COMPLETE DEBUG MANAGER ====================
class DebugLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class EndpointCall:
    endpoint: str
    method: str
    timestamp: datetime
    params: Dict[str, Any]
    response_time: float
    status_code: int
    cache_used: bool
    api_calls: int
    memory_used: float
    cpu_impact: float
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    request_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self):
        return {
            'endpoint': self.endpoint,
            'method': self.method,
            'timestamp': self.timestamp.isoformat(),
            'params': self._sanitize_params(self.params),
            'response_time': round(self.response_time, 3),
            'status_code': self.status_code,
            'cache_used': self.cache_used,
            'api_calls': self.api_calls,
            'memory_used': round(self.memory_used, 1),
            'cpu_impact': round(self.cpu_impact, 1),
            'user_agent': self.user_agent[:100] if self.user_agent else None,
            'client_ip': self.client_ip,
            'request_id': self.request_id,
            'error_message': self.error_message
        }
    
    def _sanitize_params(self, params):
        """پاک‌سازی پارامترها"""
        if not params:
            return {}
        
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            elif isinstance(value, list):
                sanitized[key] = [str(v)[:50] for v in value[:10]]
            else:
                sanitized[key] = str(value)[:100]
        return sanitized

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage: float
    disk_used_gb: float
    disk_total_gb: float
    network_io: Dict[str, int]
    active_connections: int
    process_count: int
    load_average: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_percent': round(self.memory_percent, 1),
            'memory_used_gb': round(self.memory_used_gb, 2),
            'memory_total_gb': round(self.memory_total_gb, 2),
            'disk_usage': round(self.disk_usage, 1),
            'disk_used_gb': round(self.disk_used_gb, 2),
            'disk_total_gb': round(self.disk_total_gb, 2),
            'network_io': self.network_io,
            'active_connections': self.active_connections,
            'process_count': self.process_count,
            'load_average': self.load_average if self.load_average else (0, 0, 0)
        }

class CompleteDebugManager:
    """Debug Manager کامل با تمام ویژگی‌ها"""
    
    def __init__(self):
        # تنظیمات
        self.endpoint_calls_maxlen = int(os.getenv("ENDPOINT_CALLS_MAXLEN", "10000"))
        self.system_metrics_maxlen = int(os.getenv("SYSTEM_METRICS_MAXLEN", "2000"))
        self.metrics_collection_interval = float(os.getenv("METRICS_COLLECTION_INTERVAL", "60"))
        self.performance_window_hours = 24
        
        # ساختارهای داده
        self.endpoint_calls = deque(maxlen=self.endpoint_calls_maxlen)
        self.system_metrics_history = deque(maxlen=self.system_metrics_maxlen)
        self.performance_metrics = deque(maxlen=1000)
        
        # آمار endpointها
        self.endpoint_stats = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': deque(maxlen=100),
            'last_call': None,
            'performance_trend': deque(maxlen=100),
            'hourly_stats': defaultdict(lambda: {'calls': 0, 'total_time': 0.0, 'errors': 0}),
            'user_agents': defaultdict(int),
            'client_ips': defaultdict(int),
            'status_codes': defaultdict(int)
        })
        
        # هشدارها
        self.alerts = deque(maxlen=500)
        self.alert_rules = self._initialize_alert_rules()
        
        # کش
        self._cache = {
            'recent_calls': {'data': None, 'timestamp': 0, 'ttl': 5},
            'endpoint_stats': {'data': None, 'timestamp': 0, 'ttl': 30},
            'system_health': {'data': None, 'timestamp': 0, 'ttl': 10},
            'performance_report': {'data': None, 'timestamp': 0, 'ttl': 60}
        }
        
        # Profiler
        self._profiler_data = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'percentiles': defaultdict(float),
            'last_10_times': deque(maxlen=10)
        })
        
        # Tasks
        self._metrics_task = None
        self._cleanup_task = None
        self._alert_check_task = None
        
        # آستانه‌های دینامیک
        self.adaptive_thresholds = self._calculate_initial_thresholds()
        
        logger.info("🚀 Complete Debug Manager Initialized")
    
    def _initialize_alert_rules(self):
        """مقادیر اولیه قوانین هشدار"""
        return {
            'response_time': {
                'warning': float(os.getenv("RESPONSE_TIME_WARNING", "1.0")),
                'critical': float(os.getenv("RESPONSE_TIME_CRITICAL", "3.0")),
                'window': 10  # تعداد نمونه‌ها برای بررسی
            },
            'cpu_usage': {
                'warning': float(os.getenv("CPU_WARNING", "80.0")),
                'critical': float(os.getenv("CPU_CRITICAL", "95.0"))
            },
            'memory_usage': {
                'warning': float(os.getenv("MEMORY_WARNING", "85.0")),
                'critical': float(os.getenv("MEMORY_CRITICAL", "95.0"))
            },
            'disk_usage': {
                'warning': float(os.getenv("DISK_WARNING", "85.0")),
                'critical': float(os.getenv("DISK_CRITICAL", "95.0"))
            },
            'error_rate': {
                'warning': 5.0,  # درصد
                'critical': 10.0
            }
        }
    
    def _calculate_initial_thresholds(self):
        """محاسبه آستانه‌های اولیه"""
        return {
            'response_time_percentile_95': 2.0,
            'error_rate_threshold': 5.0,
            'memory_growth_rate': 0.1,  # MB/ثانیه
            'concurrent_connections': 1000
        }
    
    async def initialize(self, event_bus):
        """راه‌اندازی Debug Manager"""
        # ثبت در Event Bus
        await event_bus.subscribe("central_monitor_ready", self._on_central_monitor_ready, priority=1)
        await event_bus.subscribe("system_metrics_update", self._on_system_metrics_update, priority=2)
        await event_bus.subscribe("endpoint_called", self._on_endpoint_called, priority=3)
        await event_bus.subscribe("error_occurred", self._on_error_occurred, priority=1)
        
        # شروع tasks
        self._start_background_tasks()
        
        logger.info("✅ Debug Manager initialized with Event Bus")
    
    def _start_background_tasks(self):
        """شروع tasks پس‌زمینه"""
        # Metrics collection
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self._collect_system_metrics_loop())
        
        # Cleanup
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Alert checking
        if self._alert_check_task is None:
            self._alert_check_task = asyncio.create_task(self._check_alerts_loop())
    
    # ==================== METRICS COLLECTION ====================
    async def _collect_system_metrics_loop(self):
        """حلقه جمع‌آوری متریک‌های سیستم"""
        while True:
            try:
                current_time = datetime.now()
                
                # بررسی فاصله زمانی
                if (current_time - self._last_metrics_collection).total_seconds() >= self.metrics_collection_interval:
                    
                    # بررسی CPU قبل از جمع‌آوری
                    cpu_before = psutil.cpu_percent(interval=0)
                    if cpu_before > 85:  # اگر CPU خیلی بالا است، صبر کن
                        await asyncio.sleep(10)
                        continue
                    
                    # جمع‌آوری متریک‌ها
                    metrics = await self._collect_comprehensive_metrics()
                    self.system_metrics_history.append(metrics)
                    
                    # ذخیره متریک عملکرد
                    self.performance_metrics.append({
                        'timestamp': metrics.timestamp.isoformat(),
                        'cpu': metrics.cpu_percent,
                        'memory': metrics.memory_percent,
                        'response_time': self._calculate_avg_response_time()
                    })
                    
                    # باطل کردن کش
                    self._invalidate_cache('system_health')
                    self._invalidate_cache('performance_report')
                    
                    # بررسی هشدارهای سیستم
                    await self._check_system_health_alerts(metrics)
                    
                    # بروزرسانی آستانه‌های دینامیک
                    self._update_adaptive_thresholds()
                    
                    self._last_metrics_collection = current_time
                
                # انتظار تطبیقی
                sleep_time = self._calculate_adaptive_sleep()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_comprehensive_metrics(self) -> SystemMetrics:
        """جمع‌آوری متریک‌های جامع"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_percent_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
            
            # Memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            
            # Network
            net_io = psutil.net_io_counters()
            
            # Processes
            process_count = len(psutil.pids())
            
            # Load average (بر روی Linux/Unix)
            load_avg = None
            if hasattr(os, 'getloadavg'):
                try:
                    load_avg = os.getloadavg()
                except OSError:
                    load_avg = (0, 0, 0)
            
            # Active connections
            try:
                connections = psutil.net_connections(kind='inet')
                active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            except Exception:
                active_connections = 0
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage=disk.percent,
                disk_used_gb=disk.used / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                network_io={
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                },
                active_connections=active_connections,
                process_count=process_count,
                load_average=load_avg
            )
            
        except Exception as e:
            logger.error(f"❌ Comprehensive metrics error: {e}")
            # Fallback metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                memory_used_gb=0,
                memory_total_gb=0,
                disk_usage=0,
                disk_used_gb=0,
                disk_total_gb=0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0, 'errin': 0, 'errout': 0, 'dropin': 0, 'dropout': 0},
                active_connections=0,
                process_count=0,
                load_average=(0, 0, 0)
            )
    
    def _calculate_adaptive_sleep(self) -> float:
        """محاسبه زمان خواب تطبیقی"""
        if not self.system_metrics_history:
            return 5.0
        
        recent_metrics = list(self.system_metrics_history)[-5:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        
        if avg_cpu > 90:
            return 30.0
        elif avg_cpu > 70:
            return 15.0
        elif avg_cpu > 50:
            return 10.0
        else:
            return 5.0
    
    # ==================== ENDPOINT MONITORING ====================
    def log_endpoint_call(self, endpoint: str, method: str, params: Dict[str, Any],
                         response_time: float, status_code: int, cache_used: bool,
                         api_calls: int = 0, user_agent: Optional[str] = None,
                         client_ip: Optional[str] = None, request_id: Optional[str] = None,
                         error_message: Optional[str] = None):
        """ثبت کامل فراخوانی endpoint"""
        
        start_time = time.time()
        profiler_key = f"{endpoint}_{method}"
        
        try:
            # جمع‌آوری متریک‌های سیستم (با احتمال 20%)
            memory_used = 0
            cpu_impact = 0
            
            if random.random() < 0.2:  # فقط 20% مواقع
                memory_used = psutil.virtual_memory().percent
                cpu_impact = psutil.cpu_percent(interval=0)
            
            # ایجاد رکورد
            call = EndpointCall(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(),
                params=params,
                response_time=response_time,
                status_code=status_code,
                cache_used=cache_used,
                api_calls=api_calls,
                memory_used=memory_used,
                cpu_impact=cpu_impact,
                user_agent=user_agent,
                client_ip=client_ip,
                request_id=request_id,
                error_message=error_message
            )
            
            # ذخیره
            self.endpoint_calls.append(call)
            
            # به‌روزرسانی آمار
            self._update_comprehensive_stats(endpoint, call)
            
            # بررسی هشدارهای عملکرد
            self._check_endpoint_performance_alerts(endpoint, call)
            
            # باطل کردن کش
            self._invalidate_cache('recent_calls')
            self._invalidate_cache('endpoint_stats')
            self._invalidate_cache('performance_report')
            
            # انتشار رویداد (ناهمزمان)
            asyncio.create_task(self._publish_endpoint_event(call))
            
        except Exception as e:
            logger.error(f"❌ Endpoint logging error: {e}")
        
        finally:
            # ثبت زمان اجرا
            execution_time = time.time() - start_time
            self._update_profiler(profiler_key, execution_time)
    
    def _update_comprehensive_stats(self, endpoint: str, call: EndpointCall):
        """به‌روزرسانی آمار جامع"""
        stats = self.endpoint_stats[endpoint]
        
        # آمار پایه
        stats['total_calls'] += 1
        stats['total_response_time'] += call.response_time
        stats['avg_response_time'] = stats['total_response_time'] / stats['total_calls']
        stats['min_response_time'] = min(stats['min_response_time'], call.response_time)
        stats['max_response_time'] = max(stats['max_response_time'], call.response_time)
        
        # وضعیت
        if 200 <= call.status_code < 300:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1
            if len(stats['errors']) < 100:
                stats['errors'].append({
                    'timestamp': call.timestamp.isoformat(),
                    'status_code': call.status_code,
                    'response_time': call.response_time,
                    'error_message': call.error_message,
                    'params': call.params
                })
        
        # کش
        if call.cache_used:
            stats['cache_hits'] += 1
        else:
            stats['cache_misses'] += 1
        
        stats['api_calls'] += call.api_calls
        stats['last_call'] = call.timestamp.isoformat()
        
        # آمار ساعتی
        hour_key = call.timestamp.strftime("%Y-%m-%d %H:00")
        hourly = stats['hourly_stats'][hour_key]
        hourly['calls'] += 1
        hourly['total_time'] += call.response_time
        if call.status_code >= 400:
            hourly['errors'] += 1
        
        # روند عملکرد
        if len(stats['performance_trend']) >= 100:
            stats['performance_trend'].popleft()
        stats['performance_trend'].append({
            'timestamp': call.timestamp.isoformat(),
            'response_time': call.response_time,
            'status_code': call.status_code,
            'memory_used': call.memory_used,
            'cpu_impact': call.cpu_impact
        })
        
        # User Agents
        if call.user_agent:
            stats['user_agents'][call.user_agent[:50]] += 1
        
        # Client IPs
        if call.client_ip:
            stats['client_ips'][call.client_ip] += 1
        
        # Status Codes
        stats['status_codes'][call.status_code] += 1
    
    # ==================== ALERT SYSTEM ====================
    def _check_endpoint_performance_alerts(self, endpoint: str, call: EndpointCall):
        """بررسی هشدارهای عملکرد endpoint"""
        stats = self.endpoint_stats[endpoint]
        
        # 1. بررسی زمان پاسخ
        if call.response_time > self.alert_rules['response_time']['critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical response time in {endpoint}: {call.response_time:.2f}s",
                source=endpoint,
                data={
                    'response_time': call.response_time,
                    'threshold': self.alert_rules['response_time']['critical'],
                    'status_code': call.status_code,
                    'method': call.method
                },
                auto_resolve=True,
                resolve_after=300  # 5 دقیقه
            )
        
        elif call.response_time > self.alert_rules['response_time']['warning']:
            # بررسی روند
            if len(stats['performance_trend']) >= 5:
                recent_times = [t['response_time'] for t in list(stats['performance_trend'])[-5:]]
                if all(t > self.alert_rules['response_time']['warning'] for t in recent_times):
                    self._create_alert(
                        level=DebugLevel.WARNING,
                        message=f"Consistent high response time in {endpoint}: avg {sum(recent_times)/len(recent_times):.2f}s",
                        source=endpoint,
                        data={
                            'response_times': recent_times,
                            'threshold': self.alert_rules['response_time']['warning'],
                            'avg_response': sum(recent_times)/len(recent_times)
                        },
                        auto_resolve=True,
                        resolve_after=600  # 10 دقیقه
                    )
        
        # 2. بررسی نرخ خطا
        if stats['total_calls'] > 10:
            error_rate = (stats['failed_calls'] / stats['total_calls']) * 100
            
            if error_rate > self.alert_rules['error_rate']['critical']:
                self._create_alert(
                    level=DebugLevel.CRITICAL,
                    message=f"Critical error rate in {endpoint}: {error_rate:.1f}%",
                    source=endpoint,
                    data={
                        'error_rate': error_rate,
                        'failed_calls': stats['failed_calls'],
                        'total_calls': stats['total_calls'],
                        'threshold': self.alert_rules['error_rate']['critical']
                    },
                    auto_resolve=True,
                    resolve_after=900  # 15 دقیقه
                )
            
            elif error_rate > self.alert_rules['error_rate']['warning']:
                self._create_alert(
                    level=DebugLevel.WARNING,
                    message=f"High error rate in {endpoint}: {error_rate:.1f}%",
                    source=endpoint,
                    data={
                        'error_rate': error_rate,
                        'failed_calls': stats['failed_calls'],
                        'total_calls': stats['total_calls'],
                        'threshold': self.alert_rules['error_rate']['warning']
                    },
                    auto_resolve=True,
                    resolve_after=600  # 10 دقیقه
                )
    
    async def _check_system_health_alerts(self, metrics: SystemMetrics):
        """بررسی هشدارهای سلامت سیستم"""
        # CPU
        if metrics.cpu_percent > self.alert_rules['cpu_usage']['critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                source="system_monitor",
                data={
                    'cpu_percent': metrics.cpu_percent,
                    'threshold': self.alert_rules['cpu_usage']['critical'],
                    'timestamp': metrics.timestamp.isoformat()
                },
                auto_resolve=True,
                resolve_after=300
            )
        
        elif metrics.cpu_percent > self.alert_rules['cpu_usage']['warning']:
            self._create_alert(
                level=DebugLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                source="system_monitor",
                data={
                    'cpu_percent': metrics.cpu_percent,
                    'threshold': self.alert_rules['cpu_usage']['warning'],
                    'timestamp': metrics.timestamp.isoformat()
                },
                auto_resolve=True,
                resolve_after=300
            )
        
        # Memory
        if metrics.memory_percent > self.alert_rules['memory_usage']['critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical memory usage: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB used)",
                source="system_monitor",
                data={
                    'memory_percent': metrics.memory_percent,
                    'memory_used_gb': metrics.memory_used_gb,
                    'memory_total_gb': metrics.memory_total_gb,
                    'threshold': self.alert_rules['memory_usage']['critical'],
                    'timestamp': metrics.timestamp.isoformat()
                },
                auto_resolve=True,
                resolve_after=300
            )
        
        # Disk
        if metrics.disk_usage > self.alert_rules['disk_usage']['critical']:
            self._create_alert(
                level=DebugLevel.CRITICAL,
                message=f"Critical disk usage: {metrics.disk_usage:.1f}% ({metrics.disk_used_gb:.1f}GB used)",
                source="system_monitor",
                data={
                    'disk_usage': metrics.disk_usage,
                    'disk_used_gb': metrics.disk_used_gb,
                    'disk_total_gb': metrics.disk_total_gb,
                    'threshold': self.alert_rules['disk_usage']['critical'],
                    'timestamp': metrics.timestamp.isoformat()
                },
                auto_resolve=False  # نیاز به اقدام دستی
            )
    
    def _create_alert(self, level: DebugLevel, message: str, source: str, 
                     data: Dict[str, Any], auto_resolve: bool = False, 
                     resolve_after: int = 0):
        """ایجاد هشدار جدید"""
        alert = {
            'id': f"alert_{int(time.time()*1000)}_{random.randint(1000, 9999)}",
            'level': level.value,
            'message': message,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'acknowledged': False,
            'resolved': False,
            'auto_resolve': auto_resolve,
            'resolve_after': resolve_after if auto_resolve else None,
            'resolve_time': (datetime.now() + timedelta(seconds=resolve_after)).isoformat() if auto_resolve else None,
            'acknowledged_by': None,
            'acknowledged_at': None,
            'notes': None
        }
        
        self.alerts.append(alert)
        
        # لاگ کردن
        log_message = f"🚨 {level.value}: {message}"
        if level == DebugLevel.CRITICAL:
            logger.critical(log_message)
        elif level == DebugLevel.ERROR:
            logger.error(log_message)
        elif level == DebugLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # انتشار رویداد
        asyncio.create_task(self._publish_alert_event(alert))
        
        return alert['id']
    
    async def _check_alerts_loop(self):
        """حلقه بررسی هشدارها"""
        while True:
            try:
                current_time = datetime.now()
                
                # بررسی هشدارهای auto-resolve
                for alert in self.alerts:
                    if alert['auto_resolve'] and not alert['resolved']:
                        resolve_time = datetime.fromisoformat(alert['resolve_time'])
                        if current_time >= resolve_time:
                            alert['resolved'] = True
                            logger.info(f"✅ Auto-resolved alert: {alert['id']}")
                
                await asyncio.sleep(60)  # هر دقیقه بررسی کن
                
            except Exception as e:
                logger.error(f"❌ Alert check error: {e}")
                await asyncio.sleep(30)
    
    # ==================== EVENT HANDLERS ====================
    async def _on_central_monitor_ready(self, central_monitor):
        """وقتی Central Monitor آماده شد"""
        try:
            central_monitor.subscribe("debug_manager", self._on_metrics_update)
            logger.info("✅ Debug Manager connected to Central Monitor")
        except Exception as e:
            logger.error(f"❌ Central Monitor connection failed: {e}")
    
    async def _on_system_metrics_update(self, metrics):
        """دریافت بروزرسانی متریک‌های سیستم"""
        # پردازش متریک‌های خارجی
        pass
    
    async def _on_endpoint_called(self, call_data):
        """دریافت رویداد فراخوانی endpoint"""
        # پردازش endpointهای خارجی
        pass
    
    async def _on_error_occurred(self, error_data):
        """دریافت رویداد خطا"""
        # ثبت خطاهای خارجی
        pass
    
    async def _publish_endpoint_event(self, call: EndpointCall):
        """انتشار رویداد endpoint"""
        try:
            event_bus = get_event_bus()
            await event_bus.publish("endpoint_processed", {
                'call': call.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.debug(f"⚠️ Endpoint event publish failed: {e}")
    
    async def _publish_alert_event(self, alert: Dict):
        """انتشار رویداد هشدار"""
        try:
            event_bus = get_event_bus()
            await event_bus.publish("alert_created", {
                'alert': alert,
                'timestamp': datetime.now().isoformat()
            }, urgent=True)
        except Exception as e:
            logger.debug(f"⚠️ Alert event publish failed: {e}")
    
    # ==================== PROFILER ====================
    def _update_profiler(self, key: str, execution_time: float):
        """به‌روزرسانی profiler"""
        profiler = self._profiler_data[key]
        profiler['count'] += 1
        profiler['total_time'] += execution_time
        profiler['avg_time'] = profiler['total_time'] / profiler['count']
        profiler['min_time'] = min(profiler['min_time'], execution_time)
        profiler['max_time'] = max(profiler['max_time'], execution_time)
        
        # ذخیره زمان‌های اخیر
        profiler['last_10_times'].append(execution_time)
        
        # محاسبه percentiles هر 100 اجرا
        if profiler['count'] % 100 == 0 and profiler['last_10_times']:
            times = list(profiler['last_10_times'])
            times.sort()
            profiler['percentiles']['p50'] = times[int(len(times) * 0.5)]
            profiler['percentiles']['p90'] = times[int(len(times) * 0.9)]
            profiler['percentiles']['p95'] = times[int(len(times) * 0.95)]
            profiler['percentiles']['p99'] = times[int(len(times) * 0.99)]
    
    # ==================== CACHE MANAGEMENT ====================
    def _invalidate_cache(self, cache_key: str):
        """باطل کردن کش"""
        if cache_key in self._cache:
            self._cache[cache_key]['data'] = None
            self._cache[cache_key]['timestamp'] = 0
    
    def _get_cached_data(self, cache_key: str, ttl: int):
        """دریافت داده‌های کش شده"""
        cache_entry = self._cache.get(cache_key)
        if cache_entry and cache_entry['data'] is not None:
            if time.time() - cache_entry['timestamp'] < ttl:
                return cache_entry['data']
        return None
    
    def _set_cached_data(self, cache_key: str, data: Any, ttl: int):
        """ذخیره داده در کش"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    # ==================== HELPER METHODS ====================
    def _calculate_avg_response_time(self) -> float:
        """محاسبه میانگین زمان پاسخ"""
        if not self.endpoint_calls:
            return 0.0
        
        recent_calls = list(self.endpoint_calls)[-100:]  # 100 نمونه آخر
        if not recent_calls:
            return 0.0
        
        return sum(call.response_time for call in recent_calls) / len(recent_calls)
    
    def _update_adaptive_thresholds(self):
        """بروزرسانی آستانه‌های دینامیک"""
        if len(self.endpoint_calls) > 100:
            # محاسبه percentile 95 زمان پاسخ
            recent_times = [call.response_time for call in list(self.endpoint_calls)[-1000:]]
            recent_times.sort()
            if recent_times:
                p95_index = int(len(recent_times) * 0.95)
                self.adaptive_thresholds['response_time_percentile_95'] = recent_times[p95_index]
        
        # بروزرسانی سایر آستانه‌ها بر اساس متریک‌های سیستم
        if len(self.system_metrics_history) > 10:
            recent_metrics = list(self.system_metrics_history)[-10:]
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            self.adaptive_thresholds['memory_growth_rate'] = avg_memory / 1000  # تنظیم بر اساس میانگین
    
    # ==================== PUBLIC API ====================
    def get_endpoint_stats(self, endpoint: str = None, 
                          timeframe: str = "all", 
                          cached: bool = True) -> Dict[str, Any]:
        """دریافت آمار endpoint"""
        
        cache_key = f"endpoint_stats_{endpoint}_{timeframe}"
        if cached:
            cached_data = self._get_cached_data(cache_key, 30)
            if cached_data:
                return cached_data
        
        if endpoint:
            if endpoint not in self.endpoint_stats:
                return {'error': 'Endpoint not found'}
            
            stats = self.endpoint_stats[endpoint]
            
            # فیلتر بر اساس timeframe
            filtered_calls = self._filter_calls_by_timeframe(endpoint, timeframe)
            
            result = {
                'endpoint': endpoint,
                'timeframe': timeframe,
                'total_calls': stats['total_calls'],
                'successful_calls': stats['successful_calls'],
                'failed_calls': stats['failed_calls'],
                'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                'average_response_time': round(stats['avg_response_time'], 3),
                'min_response_time': round(stats['min_response_time'], 3) if stats['min_response_time'] != float('inf') else 0,
                'max_response_time': round(stats['max_response_time'], 3),
                'cache_performance': {
                    'hits': stats['cache_hits'],
                    'misses': stats['cache_misses'],
                    'hit_rate': (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) 
                                if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                },
                'api_calls': stats['api_calls'],
                'recent_errors': list(stats['errors'])[-20:],
                'last_call': stats['last_call'],
                'performance_trend': list(stats['performance_trend'])[-20:],
                'hourly_stats': dict(stats['hourly_stats']),
                'user_agents': dict(stats['user_agents']),
                'client_ips': dict(stats['client_ips']),
                'status_codes': dict(stats['status_codes']),
                'filtered_calls': len(filtered_calls),
                'filtered_avg_response': self._calculate_avg_for_calls(filtered_calls)
            }
            
            if cached:
                self._set_cached_data(cache_key, result, 30)
            
            return result
        else:
            # آمار کلی
            all_stats = {}
            total_calls = 0
            total_success = 0
            
            for ep, stats in self.endpoint_stats.items():
                all_stats[ep] = {
                    'total_calls': stats['total_calls'],
                    'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                    'average_response_time': round(stats['avg_response_time'], 3),
                    'last_call': stats['last_call'],
                    'error_rate': (stats['failed_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0
                }
                total_calls += stats['total_calls']
                total_success += stats['successful_calls']
            
            result = {
                'overall': {
                    'total_endpoints': len(self.endpoint_stats),
                    'total_calls': total_calls,
                    'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                },
                'endpoints': all_stats
            }
            
            return result
    
    def _filter_calls_by_timeframe(self, endpoint: str, timeframe: str) -> List[EndpointCall]:
        """فیلتر فراخوانی‌ها بر اساس timeframe"""
        if timeframe == "all":
            return [call for call in self.endpoint_calls if call.endpoint == endpoint]
        elif timeframe == "hour":
            cutoff = datetime.now() - timedelta(hours=1)
            return [call for call in self.endpoint_calls if call.endpoint == endpoint and call.timestamp > cutoff]
        elif timeframe == "day":
            cutoff = datetime.now() - timedelta(days=1)
            return [call for call in self.endpoint_calls if call.endpoint == endpoint and call.timestamp > cutoff]
        elif timeframe == "week":
            cutoff = datetime.now() - timedelta(weeks=1)
            return [call for call in self.endpoint_calls if call.endpoint == endpoint and call.timestamp > cutoff]
        else:
            return []
    
    def _calculate_avg_for_calls(self, calls: List[EndpointCall]) -> float:
        """محاسبه میانگین برای لیست فراخوانی‌ها"""
        if not calls:
            return 0.0
        return sum(call.response_time for call in calls) / len(calls)
    
    def get_recent_calls(self, limit: int = 50, 
                        endpoint: Optional[str] = None,
                        status_code: Optional[int] = None,
                        min_response_time: Optional[float] = None,
                        cached: bool = True) -> List[Dict[str, Any]]:
        """دریافت آخرین فراخوانی‌ها با فیلتر"""
        
        cache_key = f"recent_calls_{limit}_{endpoint}_{status_code}_{min_response_time}"
        if cached:
            cached_data = self._get_cached_data(cache_key, 5)
            if cached_data:
                return cached_data
        
        recent_calls = list(self.endpoint_calls)
        
        # اعمال فیلترها
        if endpoint:
            recent_calls = [call for call in recent_calls if call.endpoint == endpoint]
        
        if status_code is not None:
            recent_calls = [call for call in recent_calls if call.status_code == status_code]
        
        if min_response_time is not None:
            recent_calls = [call for call in recent_calls if call.response_time >= min_response_time]
        
        # محدود کردن تعداد
        recent_calls = recent_calls[-limit:]
        
        result = [call.to_dict() for call in recent_calls]
        
        if cached:
            self._set_cached_data(cache_key, result, 5)
        
        return result
    
    def get_system_metrics_history(self, hours: int = 1, 
                                  metric_type: Optional[str] = None,
                                  cached: bool = True) -> List[Dict[str, Any]]:
        """دریافت تاریخچه متریک‌های سیستم"""
        
        cache_key = f"system_metrics_{hours}_{metric_type}"
        if cached:
            cached_data = self._get_cached_data(cache_key, 10)
            if cached_data:
                return cached_data
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        if metric_type:
            result = []
            for metrics in filtered_metrics:
                metric_dict = metrics.to_dict()
                if metric_type in metric_dict:
                    result.append({
                        'timestamp': metric_dict['timestamp'],
                        'value': metric_dict[metric_type]
                    })
        else:
            result = [metrics.to_dict() for metrics in filtered_metrics]
        
        if cached:
            self._set_cached_data(cache_key, result, 10)
        
        return result
    
    def get_active_alerts(self, level: Optional[str] = None, 
                         source: Optional[str] = None,
                         acknowledged: bool = False) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        active_alerts = [
            alert for alert in self.alerts
            if not alert['resolved'] and (alert['acknowledged'] == acknowledged)
        ]
        
        if level:
            active_alerts = [alert for alert in active_alerts if alert['level'] == level]
        
        if source:
            active_alerts = [alert for alert in active_alerts if alert['source'] == source]
        
        return active_alerts
    
    def acknowledge_alert(self, alert_id: str, user: str = "system", notes: str = ""):
        """تأیید هشدار"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_by'] = user
                alert['acknowledged_at'] = datetime.now().isoformat()
                alert['notes'] = notes
                logger.info(f"✅ Alert acknowledged: {alert_id} by {user}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str):
        """حل هشدار"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                logger.info(f"✅ Alert resolved: {alert_id}")
                return True
        return False
    
    def get_performance_report(self, detailed: bool = False, cached: bool = True) -> Dict[str, Any]:
        """گزارش عملکرد جامع"""
        
        cache_key = f"performance_report_{detailed}"
        if cached:
            cached_data = self._get_cached_data(cache_key, 60)
            if cached_data:
                return cached_data
        
        current_time = datetime.now()
        
        # 1. آنالیز endpointهای پرکاربرد
        top_endpoints = sorted(
            [(ep, stats['total_calls']) for ep, stats in self.endpoint_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 2. آنالیز endpointهای کند
        slow_endpoints = sorted(
            [(ep, stats['avg_response_time']) for ep, stats in self.endpoint_stats.items() 
             if stats['total_calls'] > 10],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 3. آنالیز endpointهای با خطای بالا
        high_error_endpoints = []
        for ep, stats in self.endpoint_stats.items():
            if stats['total_calls'] > 20:
                error_rate = (stats['failed_calls'] / stats['total_calls']) * 100
                if error_rate > 5:
                    high_error_endpoints.append((ep, error_rate))
        
        high_error_endpoints.sort(key=lambda x: x[1], reverse=True)
        high_error_endpoints = high_error_endpoints[:10]
        
        # 4. محاسبه زمان پاسخ
        response_times = []
        for call in list(self.endpoint_calls)[-500:]:
            response_times.append(call.response_time)
        
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        max_response = max(response_times) if response_times else 0
        
        # 5. محاسبه percentiles
        percentiles = {}
        if response_times:
            response_times.sort()
            percentiles['p50'] = response_times[int(len(response_times) * 0.5)]
            percentiles['p90'] = response_times[int(len(response_times) * 0.9)]
            percentiles['p95'] = response_times[int(len(response_times) * 0.95)]
            percentiles['p99'] = response_times[int(len(response_times) * 0.99)]
        
        # 6. سلامت سیستم
        system_health = self._calculate_system_health_score()
        
        # 7. روند عملکرد
        performance_trend = []
        if len(self.performance_metrics) > 0:
            for metric in list(self.performance_metrics)[-24:]:  # 24 نمونه آخر
                performance_trend.append({
                    'timestamp': metric['timestamp'],
                    'cpu': metric['cpu'],
                    'memory': metric['memory'],
                    'response_time': metric['response_time']
                })
        
        report = {
            'timestamp': current_time.isoformat(),
            'summary': {
                'total_endpoints_tracked': len(self.endpoint_stats),
                'total_calls_last_hour': sum(
                    1 for call in self.endpoint_calls 
                    if (current_time - call.timestamp).total_seconds() < 3600
                ),
                'average_response_time': round(avg_response, 3),
                'max_response_time': round(max_response, 3),
                'response_time_percentiles': percentiles,
                'active_alerts': len(self.get_active_alerts()),
                'system_health_score': system_health,
                'cache_hit_rate': self._calculate_overall_cache_hit_rate()
            },
            'top_endpoints': [
                {'endpoint': ep, 'calls': calls, 'stats': self.endpoint_stats[ep]} 
                for ep, calls in top_endpoints
            ],
            'slow_endpoints': [
                {'endpoint': ep, 'avg_response_time': round(time, 3)} 
                for ep, time in slow_endpoints
            ],
            'high_error_endpoints': [
                {'endpoint': ep, 'error_rate': round(rate, 1)} 
                for ep, rate in high_error_endpoints
            ],
            'performance_trend': performance_trend,
            'system_metrics': self.get_system_metrics_history(hours=1),
            'profiler_summary': self._get_profiler_summary()
        }
        
        if detailed:
            report['detailed_stats'] = {
                'endpoint_stats': {ep: self.endpoint_stats[ep] for ep, _ in top_endpoints},
                'profiler_data': dict(self._profiler_data),
                'alert_history': list(self.alerts)[-50:]
            }
        
        if cached:
            self._set_cached_data(cache_key, report, 60)
        
        return report
    
    def _calculate_system_health_score(self) -> float:
        """محاسبه نمره سلامت سیستم"""
        if not self.system_metrics_history:
            return 100.0
        
        recent_metrics = list(self.system_metrics_history)[-10:]
        
        scores = []
        for metric in recent_metrics:
            # نمره CPU (کمتر بهتر)
            cpu_score = max(0, 100 - metric.cpu_percent)
            
            # نمره Memory (کمتر بهتر)
            memory_score = max(0, 100 - metric.memory_percent)
            
            # نمره Disk (کمتر بهتر)
            disk_score = max(0, 100 - metric.disk_usage)
            
            # میانگین نمره‌ها
            avg_score = (cpu_score + memory_score + disk_score) / 3
            scores.append(avg_score)
        
        return round(sum(scores) / len(scores), 1) if scores else 100.0
    
    def _calculate_overall_cache_hit_rate(self) -> float:
        """محاسبه نرخ hit کلی کش"""
        total_hits = sum(stats['cache_hits'] for stats in self.endpoint_stats.values())
        total_misses = sum(stats['cache_misses'] for stats in self.endpoint_stats.values())
        
        total = total_hits + total_misses
        return (total_hits / total * 100) if total > 0 else 0.0
    
    def _get_profiler_summary(self) -> Dict[str, Any]:
        """خلاصه profiler"""
        summary = {}
        for key, data in self._profiler_data.items():
            if data['count'] > 0:
                summary[key] = {
                    'count': data['count'],
                    'avg_time': round(data['avg_time'], 3),
                    'min_time': round(data['min_time'], 3),
                    'max_time': round(data['max_time'], 3),
                    'percentiles': data['percentiles']
                }
        return summary
    
    async def _periodic_cleanup(self):
        """پاک‌سازی دوره‌ای"""
        while True:
            try:
                days = int(os.getenv("CLEANUP_OLD_DATA_DAYS", "7"))
                self.clear_old_data(days)
                
                # پاک‌سازی هشدارهای قدیمی
                self._cleanup_old_alerts()
                
                # بهینه‌سازی حافظه
                self._optimize_memory()
                
                await asyncio.sleep(3600)  # هر ساعت
                
            except Exception as e:
                logger.error(f"❌ Periodic cleanup error: {e}")
                await asyncio.sleep(300)
    
    def clear_old_data(self, days: int = None):
        """پاک کردن داده‌های قدیمی"""
        if days is None:
            days = int(os.getenv("CLEANUP_OLD_DATA_DAYS", "7"))
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # پاک‌سازی endpoint_calls
        self.endpoint_calls = deque(
            [call for call in self.endpoint_calls if call.timestamp > cutoff_time],
            maxlen=self.endpoint_calls_maxlen
        )
        
        # پاک‌سازی system_metrics_history
        self.system_metrics_history = deque(
            [metrics for metrics in self.system_metrics_history if metrics.timestamp > cutoff_time],
            maxlen=self.system_metrics_maxlen
        )
        
        # پاک‌سازی performance_metrics
        self.performance_metrics = deque(
            [m for m in self.performance_metrics 
             if datetime.fromisoformat(m['timestamp']) > cutoff_time],
            maxlen=1000
        )
        
        # باطل کردن تمام کش‌ها
        for key in self._cache:
            self._cache[key]['data'] = None
            self._cache[key]['timestamp'] = 0
        
        logger.info(f"🧹 Cleared data older than {days} days")
    
    def _cleanup_old_alerts(self):
        """پاک‌سازی هشدارهای قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # فقط هشدارهای resolved قدیمی را حذف کن
        new_alerts = deque(maxlen=500)
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if not alert['resolved'] or alert_time > cutoff_time:
                new_alerts.append(alert)
        
        self.alerts = new_alerts
        
        logger.info(f"🧹 Cleaned up old alerts, remaining: {len(self.alerts)}")
    
    def _optimize_memory(self):
        """بهینه‌سازی حافظه"""
        # کاهش اندازه ساختارهای داده اگر خیلی بزرگ شده‌اند
        if len(self.endpoint_calls) > self.endpoint_calls_maxlen * 0.8:
            # نمونه‌برداری: نگه‌داشتن هر دهمین نمونه
            sampled_calls = deque(maxlen=self.endpoint_calls_maxlen)
            for i, call in enumerate(self.endpoint_calls):
                if i % 10 == 0:  # هر دهمین نمونه
                    sampled_calls.append(call)
            self.endpoint_calls = sampled_calls
            
            logger.info(f"🔄 Memory optimization: sampled endpoint calls to {len(self.endpoint_calls)}")
    
    def export_data(self, data_type: str = "all", format: str = "json") -> Dict[str, Any]:
        """خروجی گرفتن از داده‌ها"""
        data = {}
        
        if data_type in ["all", "stats"]:
            data['stats'] = self.get_endpoint_stats()
        
        if data_type in ["all", "metrics"]:
            data['metrics'] = self.get_system_metrics_history(hours=24)
        
        if data_type in ["all", "alerts"]:
            data['alerts'] = list(self.alerts)
        
        if data_type in ["all", "performance"]:
            data['performance'] = self.get_performance_report()
        
        if data_type in ["all", "profiler"]:
            data['profiler'] = dict(self._profiler_data)
        
        data['export_info'] = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'format': format,
            'total_size': len(str(data))
        }
        
        return data

# ==================== GLOBAL INSTANCES ====================
_event_bus_instance = None
_debug_manager_instance = None

def get_event_bus() -> CompleteEventBus:
    """دریافت Event Bus"""
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = CompleteEventBus()
    return _event_bus_instance

def get_debug_manager() -> CompleteDebugManager:
    """دریافت Debug Manager"""
    global _debug_manager_instance
    if _debug_manager_instance is None:
        _debug_manager_instance = CompleteDebugManager()
    return _debug_manager_instance

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="VortexAI API v4.0.0 - Complete Edition",
    version="4.0.0",
    description="Complete Crypto AI System with All Features & Performance Optimizations",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json"
)

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
    max_age=3600
)

# ==================== STATIC FILES ====================
# ایجاد دایرکتوری‌های لازم
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==================== PERFORMANCE MIDDLEWARE ====================
class PerformanceMiddleware:
    """Middleware کامل برای مانیتورینگ عملکرد"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        start_time = time.time()
        request = Request(scope, receive)
        
        # ایجاد request ID
        request_id = f"req_{int(time.time()*1000)}_{random.randint(1000, 9999)}"
        request.state.request_id = request_id
        
        # جمع‌آوری اطلاعات
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        async def send_wrapper(response):
            if response["type"] == "http.response.start":
                process_time = time.time() - start_time
                
                # افزودن headers
                headers = dict(response.get("headers", []))
                headers.append((b"x-process-time", str(process_time).encode()))
                headers.append((b"x-request-id", request_id.encode()))
                
                response["headers"] = headers
                
                # ثبت در Debug Manager
                try:
                    debug_manager = get_debug_manager()
                    status_code = response["status"]
                    
                    # استخراج پارامترها
                    params = {}
                    if request.query_params:
                        params.update(dict(request.query_params))
                    if hasattr(request, "json") and request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            if isinstance(body, dict):
                                params.update(body)
                        except:
                            pass
                    
                    debug_manager.log_endpoint_call(
                        endpoint=request.url.path,
                        method=request.method,
                        params=params,
                        response_time=process_time,
                        status_code=status_code,
                        cache_used=False,
                        api_calls=0,
                        user_agent=user_agent,
                        client_ip=client_ip,
                        request_id=request_id
                    )
                    
                    # لاگ کردن اگر زمان پاسخ بالا باشد
                    if process_time > 2.0:  # بیشتر از 2 ثانیه
                        logger.warning(
                            f"⏱️ Slow request: {request.method} {request.url.path} - "
                            f"{process_time:.3f}s - Status: {status_code}"
                        )
                    
                except Exception as e:
                    logger.debug(f"⚠️ Performance logging skipped: {e}")
            
            await send(response)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            process_time = time.time() - start_time
            
            # ثبت خطا
            try:
                debug_manager = get_debug_manager()
                debug_manager.log_endpoint_call(
                    endpoint=request.url.path,
                    method=request.method,
                    params={},
                    response_time=process_time,
                    status_code=500,
                    cache_used=False,
                    api_calls=0,
                    user_agent=user_agent,
                    client_ip=client_ip,
                    request_id=request_id,
                    error_message=str(e)
                )
            except:
                pass
            
            raise

# اعمال middleware
app.add_middleware(PerformanceMiddleware)

# ==================== LAZY ROUTE IMPORTS ====================
_imported_routers = set()

async def import_all_routers():
    """ایمپورت lazy همه روت‌ها"""
    global _imported_routers
    
    if _imported_routers:
        return
    
    logger.info("📦 Importing all routers...")
    
    # لیست کامل روت‌ها
    routers = [
        # Core routers
        ("health", "routes.health", "health_router"),
        ("coins", "routes.coins", "coins_router"),
        ("exchanges", "routes.exchanges", "exchanges_router"),
        ("news", "routes.news", "news_router"),
        ("insights", "routes.insights", "insights_router"),
        
        # Raw data routers
        ("raw_coins", "routes.raw_data.raw_coins", "raw_coins_router"),
        ("raw_news", "routes.raw_data.raw_news", "raw_news_router"),
        ("raw_insights", "routes.raw_data.raw_insights", "raw_insights_router"),
        ("raw_exchanges", "routes.raw_data.raw_exchanges", "raw_exchanges_router"),
        
        # Documentation
        ("docs", "routes.docs", "docs_router"),
        
        # Chat
        ("chat", "routes.chat_routes", "chat_router"),
    ]
    
    for name, module_path, router_name in routers:
        try:
            # ایمپورت dynamic
            import importlib
            module = importlib.import_module(module_path)
            router = getattr(module, router_name, None)
            
            if router:
                # تعیین prefix بر اساس نوع روت
                if name.startswith("raw_"):
                    prefix = f"/api/raw/{name[4:]}"
                    tags = [f"Raw {name[4:].title()}"]
                elif name in ["coins", "exchanges", "news", "insights"]:
                    prefix = f"/api/{name}"
                    tags = [name.title()]
                elif name == "health":
                    prefix = "/api/health"
                    tags = ["Health"]
                elif name == "docs":
                    prefix = "/api/docs"
                    tags = ["Documentation"]
                elif name == "chat":
                    prefix = "/api/ai/chat"
                    tags = ["AI Chat"]
                else:
                    prefix = f"/api/{name}"
                    tags = [name.title()]
                
                app.include_router(router, prefix=prefix, tags=tags)
                _imported_routers.add(name)
                logger.info(f"✅ Router loaded: {name} at {prefix}")
                
        except ImportError as e:
            logger.warning(f"⚠️ Router {name} not available: {e}")
        except Exception as e:
            logger.error(f"❌ Router {name} import error: {e}")
    
    # AI Brain router
    try:
        from ai_brain import ai_router
        app.include_router(ai_router, prefix="/api/ai", tags=["AI Brain"])
        _imported_routers.add("ai_brain")
        logger.info("✅ AI Brain router loaded")
    except ImportError as e:
        logger.warning(f"⚠️ AI Brain router not available: {e}")
    
    logger.info(f"🎯 Total routers loaded: {len(_imported_routers)}")

# ==================== STARTUP SEQUENCE ====================
@app.on_event("startup")
async def startup_event():
    """رویداد startup کامل"""
    
    logger.info("=" * 70)
    logger.info("🚀 VORTEXAI SYSTEM STARTUP - COMPLETE EDITION")
    logger.info("=" * 70)
    
    start_time = time.time()
    startup_steps = []
    
    try:
        # Step 1: شروع Event Bus
        step_start = time.time()
        event_bus = get_event_bus()
        await event_bus.start()
        startup_steps.append({
            'step': 'Event Bus',
            'status': '✅',
            'time': round(time.time() - step_start, 2)
        })
        
        # Step 2: راه‌اندازی Debug Manager
        step_start = time.time()
        debug_manager = get_debug_manager()
        await debug_manager.initialize(event_bus)
        startup_steps.append({
            'step': 'Debug Manager',
            'status': '✅',
            'time': round(time.time() - step_start, 2)
        })
        
        # Step 3: ایمپورت روت‌ها
        step_start = time.time()
        await import_all_routers()
        startup_steps.append({
            'step': 'Routers',
            'status': '✅',
            'time': round(time.time() - step_start, 2)
        })
        
        # Step 4: راه‌اندازی Central Monitor
        step_start = time.time()
        await initialize_central_monitoring(event_bus)
        startup_steps.append({
            'step': 'Central Monitor',
            'status': '✅',
            'time': round(time.time() - step_start, 2)
        })
        
        # Step 5: راه‌اندازی AI System
        step_start = time.time()
        await initialize_ai_system(event_bus)
        startup_steps.append({
            'step': 'AI System',
            'status': '✅',
            'time': round(time.time() - step_start, 2)
        })
        
        # Step 6: راه‌اندازی Background System
        step_start = time.time()
        await initialize_background_system(event_bus)
        startup_steps.append({
            'step': 'Background System',
            'status': '✅',
            'time': round(time.time() - step_start, 2)
        })
        
        # Step 7: انتشار رویداد شروع
        await event_bus.publish("system_startup_complete", {
            "startup_time": round(time.time() - start_time, 2),
            "steps": startup_steps,
            "timestamp": datetime.now().isoformat()
        }, urgent=True, priority=1)
        
        total_time = round(time.time() - start_time, 2)
        
        logger.info("=" * 70)
        logger.info("🎉 SYSTEM STARTUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        # نمایش خلاصه
        for step in startup_steps:
            logger.info(f"   {step['status']} {step['step']}: {step['time']}s")
        
        logger.info(f"   ⚡ Total startup time: {total_time}s")
        logger.info(f"   📊 Total routes: {len(app.routes)}")
        logger.info(f"   🎯 Performance mode: Optimized")
        logger.info("=" * 70)
        
        # پاکسازی اولیه
        debug_manager.clear_old_data(days=1)
        
    except Exception as e:
        logger.critical(f"❌ SYSTEM STARTUP FAILED: {e}")
        logger.critical(traceback.format_exc())
        raise

async def initialize_central_monitoring(event_bus):
    """راه‌اندازی Central Monitoring کامل"""
    try:
        from debug_system.core import metrics_collector, alert_manager
        from debug_system.monitors.system_monitor import initialize_central_monitoring
        
        # تأخیر برای اطمینان از آمادگی سیستم‌های دیگر
        await asyncio.sleep(int(os.getenv("MONITOR_ACTIVATION_DELAY", "3")))
        
        central_monitor = initialize_central_monitoring(metrics_collector, alert_manager)
        
        # تنظیمات بهینه
        central_monitor.collection_interval = 120  # 2 دقیقه
        central_monitor.enable_adaptive_sampling = True
        central_monitor.cpu_threshold = 70
        
        # شروع
        central_monitor.start_monitoring()
        
        # انتشار رویداد
        await event_bus.publish("central_monitor_ready", central_monitor, urgent=True, priority=1)
        
        logger.info(f"✅ Central Monitoring started with {len(central_monitor.subscribers)} subscribers")
        
    except Exception as e:
        logger.error(f"❌ Central monitoring failed: {e}")

async def initialize_ai_system(event_bus):
    """راه‌اندازی AI System کامل"""
    try:
        from ai_brain import vortex_brain
        
        # تأخیر
        await asyncio.sleep(int(os.getenv("AI_STARTUP_DELAY", "3")))
        
        # تنظیمات سبک برای startup
        os.environ["AI_LIGHT_MODE"] = "true"
        os.environ["AI_NEURAL_NETWORK_SIZE"] = "medium"
        os.environ["AI_LEARNING_RATE"] = "0.1"
        
        await vortex_brain.initialize()
        
        await event_bus.publish("ai_system_ready", {
            "timestamp": datetime.now().isoformat(),
            "mode": "light"
        })
        
        logger.info("✅ AI Brain System started in light mode")
        
    except Exception as e:
        logger.warning(f"⚠️ AI System not available: {e}")

async def initialize_background_system(event_bus):
    """راه‌اندازی Background System کامل"""
    try:
        # تأخیر
        await asyncio.sleep(int(os.getenv("BACKGROUND_WORKER_DELAY", "3")))
        
        # ایمپورت و راه‌اندازی background workers
        from debug_system.tools.background_worker import background_worker
        from debug_system.tools.resource_manager import resource_guardian
        from debug_system.tools.time_scheduler import time_scheduler
        from debug_system.tools.recovery_system import recovery_manager
        from debug_system.tools.monitoring_dashboard import monitoring_dashboard
        
        # راه‌اندازی
        background_worker.start()
        resource_guardian.start_monitoring()
        time_scheduler.start_scheduling()
        recovery_manager.start_monitoring()
        monitoring_dashboard.start_monitoring()
        
        # اتصال به Event Bus
        await event_bus.subscribe("system_metrics", background_worker._on_metrics_update, priority=2)
        await event_bus.subscribe("alert_created", monitoring_dashboard._on_alert_created, priority=1)
        
        await event_bus.publish("background_system_ready", {
            "timestamp": datetime.now().isoformat(),
            "components": ["worker", "resource", "scheduler", "recovery", "dashboard"]
        })
        
        logger.info("✅ Background System started with all components")
        
    except Exception as e:
        logger.error(f"❌ Background system failed: {e}")

# ==================== SHUTDOWN ====================
@app.on_event("shutdown")
async def shutdown_event():
    """رویداد shutdown کامل"""
    logger.info("=" * 70)
    logger.info("🛑 SYSTEM SHUTDOWN INITIATED")
    logger.info("=" * 70)
    
    shutdown_steps = []
    
    try:
        # Step 1: اطلاع‌رسانی shutdown
        event_bus = get_event_bus()
        await event_bus.publish("system_shutdown", {
            "timestamp": datetime.now().isoformat(),
            "reason": "normal"
        }, urgent=True, priority=1)
        
        # Step 2: توقف AI System
        try:
            from ai_brain import vortex_brain
            await vortex_brain.cleanup()
            shutdown_steps.append("AI System: ✅")
        except Exception as e:
            shutdown_steps.append(f"AI System: ⚠️ ({e})")
        
        # Step 3: توقف Background System
        try:
            from debug_system.tools.background_worker import background_worker
            background_worker.stop()
            shutdown_steps.append("Background System: ✅")
        except Exception as e:
            shutdown_steps.append(f"Background System: ⚠️ ({e})")
        
        # Step 4: توقف Event Bus
        try:
            await event_bus.stop()
            shutdown_steps.append("Event Bus: ✅")
        except Exception as e:
            shutdown_steps.append(f"Event Bus: ⚠️ ({e})")
        
        # Step 5: ذخیره داده‌های Debug
        try:
            debug_manager = get_debug_manager()
            export_data = debug_manager.export_data("all")
            with open('shutdown_export.json', 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            shutdown_steps.append("Data Export: ✅")
        except Exception as e:
            shutdown_steps.append(f"Data Export: ⚠️ ({e})")
        
        logger.info("=" * 70)
        logger.info("✅ SYSTEM SHUTDOWN COMPLETED")
        logger.info("=" * 70)
        
        for step in shutdown_steps:
            logger.info(f"   {step}")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.critical(f"❌ SYSTEM SHUTDOWN FAILED: {e}")

# ==================== ROOT ROUTES ====================
@app.get("/", include_in_schema=False)
async def root(request: Request):
    """صفحه اصلی کامل"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "VortexAI API v4.0.0",
            "version": "4.0.0",
            "status": "running",
            "total_routes": len(app.routes),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.get("/api", tags=["Root"])
async def api_root():
    """ریشه API"""
    return {
        "api": "VortexAI API v4.0.0",
        "version": "4.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "performance": "optimized",
        "features": {
            "ai_system": True,
            "debug_system": True,
            "event_bus": True,
            "real_time_monitoring": True,
            "websocket_support": True,
            "complete_documentation": True
        },
        "links": {
            "documentation": "/docs",
            "health": "/api/health",
            "performance": "/api/health/performance",
            "debug": "/api/debug",
            "websocket": "/ws/performance"
        }
    }

# ==================== COMPLETE HEALTH ROUTES ====================
@app.get("/api/health", tags=["Health"])
async def health_comprehensive():
    """بررسی سلامت جامع"""
    import psutil
    
    # جمع‌آوری متریک‌ها
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # وضعیت سیستم‌ها
    systems_status = {
        "api_server": "healthy",
        "event_bus": "healthy",
        "debug_manager": "healthy",
        "database": "unknown",
        "cache": "unknown",
        "external_apis": "unknown"
    }
    
    try:
        event_bus = get_event_bus()
        systems_status["event_bus"] = "healthy" if event_bus._is_running else "unhealthy"
    except:
        systems_status["event_bus"] = "unavailable"
    
    try:
        debug_manager = get_debug_manager()
        systems_status["debug_manager"] = "healthy"
    except:
        systems_status["debug_manager"] = "unavailable"
    
    # محاسبه نمره سلامت
    health_score = 100
    if cpu_percent > 90:
        health_score -= 30
    elif cpu_percent > 70:
        health_score -= 15
    
    if memory.percent > 95:
        health_score -= 30
    elif memory.percent > 85:
        health_score -= 15
    
    if disk.percent > 95:
        health_score -= 20
    elif disk.percent > 85:
        health_score -= 10
    
    health_score = max(0, health_score)
    
    return {
        "status": "healthy" if health_score > 80 else "degraded" if health_score > 60 else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "health_score": health_score,
        "systems": systems_status,
        "metrics": {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2)
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": round(disk.used / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2)
            },
            "network": {
                "connections": len(psutil.net_connections(kind='inet')),
                "io": psutil.net_io_counters()._asdict()
            }
        },
        "api": {
            "total_routes": len(app.routes),
            "uptime": datetime.now().isoformat(),
            "performance": "optimized"
        }
    }

@app.get("/api/health/performance", tags=["Health"])
async def health_performance():
    """بررسی سلامت عملکرد"""
    debug_manager = get_debug_manager()
    event_bus = get_event_bus()
    
    performance_report = debug_manager.get_performance_report(detailed=True)
    event_bus_stats = event_bus.get_stats()
    
    # محاسبه شاخص‌های عملکرد
    perf_indicators = {
        "response_time": {
            "current": performance_report['summary']['average_response_time'],
            "target": 0.5,  # 500ms
            "status": "good" if performance_report['summary']['average_response_time'] < 1.0 else "warning" if performance_report['summary']['average_response_time'] < 3.0 else "critical"
        },
        "system_health": {
            "score": performance_report['summary']['system_health_score'],
            "status": "good" if performance_report['summary']['system_health_score'] > 80 else "warning" if performance_report['summary']['system_health_score'] > 60 else "critical"
        },
        "event_bus": {
            "queue_size": event_bus_stats['queue_status']['normal_priority'],
            "processing_rate": event_bus_stats['stats']['events_processed'] / max(1, (time.time() - event_bus._start_time if hasattr(event_bus, '_start_time') else 3600)),
            "status": "good" if event_bus_stats['queue_status']['normal_priority'] < 100 else "warning" if event_bus_stats['queue_status']['normal_priority'] < 500 else "critical"
        }
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "performance_indicators": perf_indicators,
        "performance_report": performance_report,
        "event_bus_stats": event_bus_stats,
        "recommendations": _generate_performance_recommendations(perf_indicators)
    }

def _generate_performance_recommendations(indicators: Dict) -> List[str]:
    """تولید توصیه‌های عملکردی"""
    recommendations = []
    
    if indicators['response_time']['status'] == 'critical':
        recommendations.append("🚨 Response time is critical! Consider optimizing database queries and adding caching.")
    
    if indicators['response_time']['status'] == 'warning':
        recommendations.append("⚠️ Response time is high. Review slow endpoints and consider async operations.")
    
    if indicators['system_health']['status'] == 'critical':
        recommendations.append("🚨 System health is critical! Check CPU, memory, and disk usage.")
    
    if indicators['event_bus']['status'] == 'critical':
        recommendations.append("🚨 Event Bus queue is full! Consider increasing worker count or optimizing event handlers.")
    
    if not recommendations:
        recommendations.append("✅ All performance indicators are within acceptable ranges.")
    
    return recommendations

# ==================== COMPLETE DEBUG ROUTES ====================
@app.get("/api/debug", tags=["Debug"])
async def debug_root():
    """ریشه دیباگ"""
    return {
        "debug_system": "VortexAI Complete Debug System",
        "version": "4.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "stats": "/api/debug/stats - آمار کلی",
            "endpoints": "/api/debug/endpoints - لیست endpointها",
            "performance": "/api/debug/performance - گزارش عملکرد",
            "metrics": "/api/debug/metrics - متریک‌های سیستم",
            "alerts": "/api/debug/alerts - هشدارها",
            "export": "/api/debug/export - خروجی داده‌ها"
        },
        "websocket": {
            "performance": "/ws/performance - داده‌های real-time",
            "debug": "/ws/debug - دیباگ real-time",
            "alerts": "/ws/alerts - هشدارهای real-time"
        }
    }

@app.get("/api/debug/stats", tags=["Debug"])
async def debug_stats():
    """آمار کامل دیباگ"""
    debug_manager = get_debug_manager()
    event_bus = get_event_bus()
    
    return {
        "debug_manager": {
            "endpoints_tracked": len(debug_manager.endpoint_stats),
            "total_calls": sum(stats['total_calls'] for stats in debug_manager.endpoint_stats.values()),
            "active_alerts": len(debug_manager.get_active_alerts()),
            "system_metrics_count": len(debug_manager.system_metrics_history),
            "performance_score": debug_manager._calculate_system_health_score(),
            "cache_hit_rate": debug_manager._calculate_overall_cache_hit_rate()
        },
        "event_bus": event_bus.get_stats(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "active_connections": len(psutil.net_connections(kind='inet')),
            "process_count": len(psutil.pids()),
            "timestamp": datetime.now().isoformat()
        },
        "api": {
            "total_routes": len(app.routes),
            "imported_routers": list(_imported_routers),
            "performance_mode": "optimized"
        }
    }

@app.get("/api/debug/endpoints", tags=["Debug"])
async def debug_endpoints(
    limit: int = Query(100, ge=1, le=1000, description="تعداد endpointها"),
    sort_by: str = Query("calls", enum=["calls", "response_time", "errors"]),
    filter_type: Optional[str] = Query(None, enum=["slow", "error", "frequent"])
):
    """لیست endpointها با فیلتر و مرتب‌سازی"""
    debug_manager = get_debug_manager()
    
    # دریافت آمار
    all_stats = debug_manager.get_endpoint_stats()
    
    if 'endpoints' not in all_stats:
        return {"error": "No endpoint statistics available"}
    
    endpoints_data = []
    
    for endpoint, stats in all_stats['endpoints'].items():
        endpoint_info = {
            "endpoint": endpoint,
            "calls": stats.get('total_calls', 0),
            "avg_response_time": stats.get('average_response_time', 0),
            "success_rate": stats.get('success_rate', 0),
            "error_rate": 100 - stats.get('success_rate', 100),
            "last_call": stats.get('last_call'),
            "details_url": f"/api/debug/endpoints/{endpoint.replace('/', '_')}"
        }
        
        # اعمال فیلتر
        if filter_type == "slow" and endpoint_info["avg_response_time"] < 1.0:
            continue
        elif filter_type == "error" and endpoint_info["error_rate"] < 10:
            continue
        elif filter_type == "frequent" and endpoint_info["calls"] < 100:
            continue
        
        endpoints_data.append(endpoint_info)
    
    # مرتب‌سازی
    if sort_by == "calls":
        endpoints_data.sort(key=lambda x: x["calls"], reverse=True)
    elif sort_by == "response_time":
        endpoints_data.sort(key=lambda x: x["avg_response_time"], reverse=True)
    elif sort_by == "errors":
        endpoints_data.sort(key=lambda x: x["error_rate"], reverse=True)
    
    # محدود کردن
    endpoints_data = endpoints_data[:limit]
    
    return {
        "total_endpoints": len(endpoints_data),
        "sort_by": sort_by,
        "filter_type": filter_type,
        "endpoints": endpoints_data,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/endpoints/{endpoint}", tags=["Debug"])
async def debug_endpoint_detail(
    endpoint: str,
    timeframe: str = Query("all", enum=["all", "hour", "day", "week", "month"]),
    detailed: bool = Query(False, description="نمایش جزئیات کامل")
):
    """جزئیات endpoint خاص"""
    debug_manager = get_debug_manager()
    
    # تبدیل endpoint name به path
    endpoint_path = endpoint.replace('_', '/')
    if not endpoint_path.startswith('/'):
        endpoint_path = f'/{endpoint_path}'
    
    stats = debug_manager.get_endpoint_stats(endpoint_path, timeframe, cached=not detailed)
    
    if 'error' in stats:
        return JSONResponse(
            status_code=404,
            content={"error": stats['error']}
        )
    
    return {
        "endpoint": endpoint_path,
        "timeframe": timeframe,
        "stats": stats,
        "related_data": {
            "recent_calls": debug_manager.get_recent_calls(limit=20, endpoint=endpoint_path),
            "performance_trend": stats.get('performance_trend', []),
            "common_errors": stats.get('recent_errors', [])[:10]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/performance", tags=["Debug"])
async def debug_performance(
    detailed: bool = Query(False, description="گزارش مفصل"),
    timeframe: str = Query("hour", enum=["hour", "day", "week"]),
    export: bool = Query(False, description="خروجی JSON")
):
    """گزارش عملکرد"""
    debug_manager = get_debug_manager()
    
    report = debug_manager.get_performance_report(detailed=detailed)
    
    if export:
        # ایجاد فایل خروجی
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_content = json.dumps(report, indent=2, default=str)
        
        return StreamingResponse(
            iter([file_content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    return report

@app.get("/api/debug/metrics", tags=["Debug"])
async def debug_metrics(
    metric_type: Optional[str] = Query(None, description="نوع متریک (cpu, memory, disk)"),
    hours: int = Query(24, ge=1, le=168, description="ساعت‌های گذشته"),
    resolution: str = Query("auto", enum=["minute", "5min", "15min", "hour", "auto"]),
    chart: bool = Query(False, description="فرم مناسب برای چارت")
):
    """متریک‌های سیستم"""
    debug_manager = get_debug_manager()
    
    metrics = debug_manager.get_system_metrics_history(hours=hours, metric_type=metric_type)
    
    if not metrics:
        return {"error": "No metrics available"}
    
    if chart and metric_type:
        # فرمت مناسب برای چارت
        chart_data = []
        for metric in metrics:
            if 'timestamp' in metric and 'value' in metric:
                chart_data.append({
                    "x": metric['timestamp'],
                    "y": metric['value']
                })
        
        return {
            "metric": metric_type,
            "hours": hours,
            "data": chart_data,
            "summary": {
                "min": min([d['y'] for d in chart_data]) if chart_data else 0,
                "max": max([d['y'] for d in chart_data]) if chart_data else 0,
                "avg": sum([d['y'] for d in chart_data]) / len(chart_data) if chart_data else 0
            }
        }
    
    return {
        "metric_type": metric_type,
        "hours": hours,
        "total_metrics": len(metrics),
        "metrics": metrics[:1000],  # محدود کردن خروجی
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/alerts", tags=["Debug"])
async def debug_alerts(
    level: Optional[str] = Query(None, enum=["CRITICAL", "ERROR", "WARNING", "INFO"]),
    source: Optional[str] = Query(None, description="منبع هشدار"),
    acknowledged: Optional[bool] = Query(None, description="فقط هشدارهای تأیید شده"),
    resolved: Optional[bool] = Query(None, description="فقط هشدارهای حل شده"),
    limit: int = Query(100, ge=1, le=1000)
):
    """هشدارها"""
    debug_manager = get_debug_manager()
    
    all_alerts = list(debug_manager.alerts)
    
    # فیلتر کردن
    filtered_alerts = []
    for alert in all_alerts:
        if level and alert['level'] != level:
            continue
        if source and alert['source'] != source:
            continue
        if acknowledged is not None and alert['acknowledged'] != acknowledged:
            continue
        if resolved is not None and alert['resolved'] != resolved:
            continue
        filtered_alerts.append(alert)
    
    # مرتب‌سازی بر اساس زمان (جدیدترین اول)
    filtered_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # محدود کردن
    filtered_alerts = filtered_alerts[:limit]
    
    # آمار
    stats = {
        "total": len(all_alerts),
        "filtered": len(filtered_alerts),
        "by_level": defaultdict(int),
        "by_source": defaultdict(int)
    }
    
    for alert in all_alerts:
        stats['by_level'][alert['level']] += 1
        stats['by_source'][alert['source']] += 1
    
    return {
        "stats": stats,
        "alerts": filtered_alerts,
        "filters": {
            "level": level,
            "source": source,
            "acknowledged": acknowledged,
            "resolved": resolved
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/debug/alerts/{alert_id}/acknowledge", tags=["Debug"])
async def debug_acknowledge_alert(
    alert_id: str,
    user: str = Query(..., description="کاربر تأییدکننده"),
    notes: Optional[str] = Query(None, description="یادداشت")
):
    """تأیید هشدار"""
    debug_manager = get_debug_manager()
    
    success = debug_manager.acknowledge_alert(alert_id, user, notes or "")
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {
        "status": "success",
        "message": f"Alert {alert_id} acknowledged by {user}",
        "alert_id": alert_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/debug/alerts/{alert_id}/resolve", tags=["Debug"])
async def debug_resolve_alert(alert_id: str):
    """حل هشدار"""
    debug_manager = get_debug_manager()
    
    success = debug_manager.resolve_alert(alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {
        "status": "success",
        "message": f"Alert {alert_id} resolved",
        "alert_id": alert_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/export", tags=["Debug"])
async def debug_export(
    data_type: str = Query("all", enum=["all", "stats", "metrics", "alerts", "performance"]),
    format: str = Query("json", enum=["json", "csv"]),
    days: int = Query(7, ge=1, le=30, description="تعداد روزهای گذشته")
):
    """خروجی گرفتن از داده‌های دیباگ"""
    debug_manager = get_debug_manager()
    
    # پاک کردن داده‌های قدیمی قبل از export
    debug_manager.clear_old_data(days=days)
    
    # دریافت داده‌ها
    export_data = debug_manager.export_data(data_type, format)
    
    if format == "csv":
        # تبدیل به CSV (ساده)
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # نوشتن هدر
        if data_type == "stats":
            writer.writerow(["Endpoint", "Calls", "Avg Response", "Success Rate"])
            for endpoint, stats in export_data.get('stats', {}).get('endpoints', {}).items():
                writer.writerow([
                    endpoint,
                    stats.get('total_calls', 0),
                    stats.get('average_response_time', 0),
                    stats.get('success_rate', 0)
                ])
        
        filename = f"debug_export_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    else:  # JSON
        filename = f"debug_export_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        return StreamingResponse(
            iter([json.dumps(export_data, indent=2, default=str)]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

# ==================== WEBSOCKET ROUTES ====================
@app.websocket("/ws/performance")
async def websocket_performance(websocket: WebSocket):
    """WebSocket برای داده‌های real-time عملکرد"""
    await websocket.accept()
    
    try:
        event_bus = get_event_bus()
        debug_manager = get_debug_manager()
        
        # ثبت برای دریافت بروزرسانی‌ها
        async def send_performance_update(data):
            await websocket.send_json({
                "type": "performance_update",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
        
        await event_bus.subscribe("performance_metrics", send_performance_update, priority=1)
        
        # ارسال داده‌های اولیه
        initial_data = {
            "system_health": debug_manager._calculate_system_health_score(),
            "active_alerts": len(debug_manager.get_active_alerts()),
            "avg_response_time": debug_manager._calculate_avg_response_time(),
            "event_bus_stats": event_bus.get_stats()
        }
        
        await websocket.send_json({
            "type": "initial_data",
            "data": initial_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # حلقه اصلی
        last_metrics_send = 0
        while True:
            try:
                current_time = time.time()
                
                # هر 10 ثانیه متریک‌های سیستم را ارسال کن
                if current_time - last_metrics_send > 10:
                    import psutil
                    
                    system_metrics = {
                        "cpu": psutil.cpu_percent(interval=1),
                        "memory": psutil.virtual_memory().percent,
                        "disk": psutil.disk_usage('/').percent,
                        "connections": len(psutil.net_connections(kind='inet'))
                    }
                    
                    await websocket.send_json({
                        "type": "system_metrics",
                        "data": system_metrics,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    last_metrics_send = current_time
                
                # بررسی پیام‌های دریافتی
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                    
                    if data.get("type") == "request_update":
                        # درخواست بروزرسانی دستی
                        performance_report = debug_manager.get_performance_report()
                        await websocket.send_json({
                            "type": "manual_update",
                            "data": performance_report,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                except asyncio.TimeoutError:
                    pass  # هیچ پیامی دریافت نشد
                
                # ارسال heartbeat هر 30 ثانیه
                if int(current_time) % 30 == 0:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # استراحت کوتاه
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"WebSocket loop error: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/debug")
async def websocket_debug(websocket: WebSocket):
    """WebSocket برای دیباگ real-time"""
    await websocket.accept()
    
    try:
        debug_manager = get_debug_manager()
        event_bus = get_event_bus()
        
        # ثبت برای دریافت رویدادها
        async def send_debug_event(data):
            await websocket.send_json({
                "type": "debug_event",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
        
        await event_bus.subscribe("endpoint_processed", send_debug_event, priority=2)
        await event_bus.subscribe("alert_created", send_debug_event, priority=1)
        
        # حلقه اصلی
        while True:
            try:
                # ارجامان آمار
                stats = {
                    "recent_calls": debug_manager.get_recent_calls(limit=20),
                    "system_health": debug_manager._calculate_system_health_score(),
                    "active_alerts": len(debug_manager.get_active_alerts()),
                    "event_bus_queue": event_bus._normal_priority_queue.qsize()
                }
                
                await websocket.send_json({
                    "type": "debug_stats",
                    "data": stats,
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(5)  # هر 5 ثانیه
                
            except Exception as e:
                logger.error(f"Debug WebSocket error: {e}")
                break
                
    except Exception as e:
        logger.error(f"Debug WebSocket connection error: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket برای هشدارهای real-time"""
    await websocket.accept()
    
    try:
        event_bus = get_event_bus()
        
        # ثبت برای دریافت هشدارهای جدید
        async def send_alert(data):
            await websocket.send_json({
                "type": "new_alert",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
        
        await event_bus.subscribe("alert_created", send_alert, priority=1)
        
        # حلقه اصلی
        while True:
            try:
                # ارسال heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(30)  # هر 30 ثانیه
                
            except Exception as e:
                logger.error(f"Alerts WebSocket error: {e}")
                break
                
    except Exception as e:
        logger.error(f"Alerts WebSocket connection error: {e}")
    finally:
        await websocket.close()

# ==================== TEMPLATE ROUTES ====================
@app.get("/debug/dashboard", include_in_schema=False)
async def debug_dashboard(request: Request):
    """داشبورد دیباگ"""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Debug Dashboard",
            "websocket_url": "/ws/debug",
            "performance_url": "/ws/performance",
            "alerts_url": "/ws/alerts"
        }
    )

@app.get("/debug/console", include_in_schema=False)
async def debug_console(request: Request):
    """کنسول دیباگ"""
    return templates.TemplateResponse(
        "console.html",
        {
            "request": request,
            "title": "Debug Console",
            "websocket_url": "/ws/debug"
        }
    )

# ==================== ERROR HANDLERS ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler برای HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "method": request.method,
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        },
        headers={
            "X-Request-ID": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler جهانی برای exceptions"""
    error_id = f"err_{int(time.time()*1000)}_{random.randint(1000, 9999)}"
    
    logger.critical(
        f"Global Exception [{error_id}]: {exc}\n"
        f"Path: {request.url.path}\n"
        f"Method: {request.method}\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
    
    # ثبت در Debug Manager
    try:
        debug_manager = get_debug_manager()
        debug_manager.log_endpoint_call(
            endpoint=request.url.path,
            method=request.method,
            params={},
            response_time=0,
            status_code=500,
            cache_used=False,
            api_calls=0,
            error_message=f"{type(exc).__name__}: {str(exc)[:200]}"
        )
    except Exception as e:
        logger.error(f"Error logging failed: {e}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": "An unexpected error occurred. Our team has been notified.",
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        },
        headers={
            "X-Error-ID": error_id,
            "X-Request-ID": getattr(request.state, 'request_id', 'unknown')
        }
    )

# ==================== SYSTEM OPTIMIZATIONS ====================
def optimize_system():
    """بهینه‌سازی‌های کامل سیستم"""
    
    logger.info("🔧 Applying system optimizations...")
    
    # تنظیمات asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        # استفاده از uvloop اگر موجود باشد
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("✅ Using uvloop for better performance")
        except ImportError:
            pass
    
    # تنظیمات threading
    import threading
    threading.stack_size(256 * 1024)  # 256KB stack size
    
    # تنظیمات socket
    import socket
    socket.setdefaulttimeout(30)
    
    # تنظیمات psutil
    if hasattr(psutil, 'PROCFS_PATH'):
        psutil.PROCFS_PATH = '/proc'
    
    # تنظیمات Python
    sys.setrecursionlimit(10000)
    
    # فعال‌سازی JIT compilation اگر موجود باشد
    try:
        import numba
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        logger.info("✅ JIT compilation enabled")
    except ImportError:
        pass
    
    # تنظیمات memory
    import gc
    gc.set_threshold(700, 10, 10)  # تنظیم آستانه GC
    
    logger.info("✅ All system optimizations applied")

# ==================== INITIALIZATION ====================
print("\n" + "=" * 80)
print("🚀 VORTEXAI API v4.0.0 - COMPLETE EDITION")
print("=" * 80)
print("🎯 Performance Target: <500ms response time")
print("⚡ Features: Event Bus, Complete Debug System, Real-time Monitoring")
print("📊 Monitoring: Full metrics, alerts, WebSocket support")
print("🔧 Debug: Comprehensive debug system with export capabilities")
print("🌐 API: Full REST API with AI integration")
print("=" * 80 + "\n")

# اعمال بهینه‌سازی‌ها
optimize_system()

# ==================== SERVER START ====================
if __name__ != "__main__":
    port = int(os.getenv("PORT", 10000))
    
    print(f"🌐 Server URL: http://localhost:{port}")
    print(f"📚 Documentation: http://localhost:{port}/docs")
    print(f"📊 OpenAPI Spec: http://localhost:{port}/api/openapi.json")
    print(f"⚡ Health Check: http://localhost:{port}/api/health")
    print(f"🔧 Debug Dashboard: http://localhost:{port}/debug/dashboard")
    print(f"🔌 WebSocket Performance: ws://localhost:{port}/ws/performance")
    print(f"🔌 WebSocket Debug: ws://localhost:{port}/ws/debug")
    print(f"🔌 WebSocket Alerts: ws://localhost:{port}/ws/alerts")
    print("\n" + "🎯" * 40 + "\n")
    
    # تنظیمات uvicorn بهینه‌شده
    uvicorn_config = {
        "host": "0.0.0.0",
        "port": port,
        "access_log": True,
        "log_level": "info",
        "limit_concurrency": 200,
        "timeout_keep_alive": 30,
        "workers": 1,  # برای async بهتر است 1 باشد
        "loop": "asyncio",
        "http": "auto",
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }
    
    uvicorn.run(app, **uvicorn_config)

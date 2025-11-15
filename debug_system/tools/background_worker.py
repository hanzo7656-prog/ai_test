import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from queue import Queue, Empty
import psutil
import os
import json

logger = logging.getLogger(__name__)

class IntelligentBackgroundWorker:
    """سیستم هوشمند مدیریت کارهای پس‌زمینه با مانیتورینگ پیشرفته"""
    
    def __init__(self, max_workers: int = 3, max_cpu_percent: float = 70.0):
        self.max_workers = max_workers
        self.max_cpu_percent = max_cpu_percent
        self.task_queue = Queue()
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: List[Dict] = []
        self.failed_tasks: List[Dict] = []
        self.worker_metrics: Dict[int, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.monitor_thread = None
        self.alert_handlers = []
        
        # آمار عملکرد
        self.performance_stats = {
            'total_tasks_processed': 0,
            'total_execution_time': 0,
            'avg_task_duration': 0,
            'peak_worker_usage': 0,
            'tasks_by_type': {},
            'hourly_pattern': {}
        }
        
        # سیستم هشدار
        self.alert_thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 90.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'queue_warning': 20,
            'queue_critical': 50,
            'task_timeout': 300  # 5 دقیقه
        }
        
        logger.info("🎯 Intelligent Background Worker initialized")
        
    def start(self):
        """راه‌اندازی کارگر پس‌زمینه"""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # شروع مانیتورینگ کارگران
        self._start_worker_monitoring()
        
        logger.info("🎬 Background Worker started with advanced monitoring")

    # 🔽 این متد رو به کلاس IntelligentBackgroundWorker اضافه کن (قبل از متد stop):

    def submit_real_tasks(self):
        """ثبت کارهای واقعی در سیستم"""
        try:
            from background_tasks import background_tasks
        
            # ۱. کار پردازش داده‌های کوین‌ها
            self.submit_task(
                task_id="process_coins_data",
                task_func=background_tasks.perform_real_data_processing,
                task_type="normal",
                priority=1,
                data_type="coins"
            )
         
            # ۲. کار پردازش اخبار
            self.submit_task(
                task_id="process_news_data", 
                task_func=background_tasks.perform_real_data_processing,
                task_type="normal",
                priority=2,
                data_type="news"
            )
        
            # ۳. کار گزارش عملکرد
            self.submit_task(
                task_id="generate_performance_report",
                task_func=background_tasks.generate_real_performance_report,
                task_type="heavy",
                priority=3,
                days=1,
                detail_level="basic"
            )
        
            logger.info("📥 Real tasks submitted to background worker")
        
        except Exception as e:
            logger.error(f"❌ Error submitting real tasks: {e}")
            
    def stop(self):
        """توقف کارگر پس‌زمینه"""
        self.is_running = False
        self.executor.shutdown(wait=False)
        logger.info("🛑 Background Worker stopped")
        
    def submit_task(self, task_id: str, task_func: Callable, task_type: str = "normal",
                   priority: int = 1, *args, **kwargs) -> Tuple[bool, str]:
        """ثبت یک کار جدید در صف با اولویت‌بندی"""
        if not self.is_running:
            return False, "Worker is not running"
            
        # بررسی سلامت سیستم قبل از ثبت کار
        system_health = self._check_system_health()
        if not system_health["healthy"]:
            return False, f"System health check failed: {system_health['message']}"
        
        # بررسی محدودیت‌های کارهای سنگین
        if task_type == "heavy" and not self._can_run_heavy_task():
            return False, "Heavy tasks can only run on weekends or during night hours (1-7 AM)"
            
        task_data = {
            'task_id': task_id,
            'function': task_func,
            'args': args,
            'kwargs': kwargs,
            'task_type': task_type,
            'priority': priority,
            'submitted_at': datetime.now(),
            'status': 'queued',
            'retry_count': 0,
            'max_retries': 3
        }
        
        self.task_queue.put(task_data)
        self.active_tasks[task_id] = task_data
        
        # به‌روزرسانی آمار
        self._update_task_stats(task_type, "submitted")
        
        logger.info(f"📥 Task {task_id} submitted (Type: {task_type}, Priority: {priority})")
        return True, "Task submitted successfully"
        
    def _monitor_loop(self):
        """حلقه نظارت پیشرفته بر اجرای کارها"""
        while self.is_running:
            try:
                # جمع‌آوری متریک‌های سیستم
                system_metrics = self._collect_system_metrics()
                
                # بررسی هشدارها
                self._check_alerts(system_metrics)
                
                # اجرای کارها اگر شرایط مناسب باشد
                if (system_metrics['cpu_percent'] < self.max_cpu_percent and 
                    system_metrics['memory_percent'] < 85 and
                    not self.task_queue.empty()):
                    
                    task_data = self.task_queue.get(timeout=1)
                    self._execute_task_with_monitoring(task_data)
                else:
                    # بهینه‌سازی مصرف منابع در زمان شلوغی
                    time.sleep(self._calculate_optimal_sleep_time(system_metrics))
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Monitor loop error: {e}")
                time.sleep(5)
                
    def _execute_task_with_monitoring(self, task_data: Dict):
        """اجرای کار با مانیتورینگ کامل"""
        task_id = task_data['task_id']
        worker_id = threading.get_ident()
        
        try:
            # ثبت شروع کار
            task_data['status'] = 'running'
            task_data['started_at'] = datetime.now()
            task_data['worker_id'] = worker_id
            
            logger.info(f"⚡ Executing task: {task_id} on worker {worker_id}")
            
            # مانیتورینگ کارگر
            self._start_worker_monitoring_task(worker_id, task_id)
            
            # اجرای کار
            start_time = time.time()
            future = self.executor.submit(task_data['function'], *task_data['args'], **task_data['kwargs'])
            result = future.result(timeout=self.alert_thresholds['task_timeout'])
            execution_time = time.time() - start_time
            
            # ثبت موفقیت
            task_data['status'] = 'completed'
            task_data['completed_at'] = datetime.now()
            task_data['execution_time'] = execution_time
            task_data['result'] = result
            
            # به‌روزرسانی آمار
            self._update_performance_stats(task_data, execution_time)
            self._update_worker_metrics(worker_id, 'completed', execution_time)
            
            logger.info(f"✅ Task {task_id} completed in {execution_time:.2f}s")
            
            # انتقال به لیست کارهای انجام شده
            self.completed_tasks.append(task_data.copy())
            
        except Exception as e:
            # مدیریت خطا و تلاش مجدد
            self._handle_task_failure(task_data, str(e), worker_id)
            
        finally:
            # پاک‌سازی
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            self._stop_worker_monitoring_task(worker_id)
            
    def _handle_task_failure(self, task_data: Dict, error: str, worker_id: int):
        """مدیریت خطاهای کار و تلاش مجدد"""
        task_id = task_data['task_id']
        task_data['status'] = 'failed'
        task_data['failed_at'] = datetime.now()
        task_data['error'] = error
        task_data['retry_count'] += 1
        
        self._update_worker_metrics(worker_id, 'failed', 0)
        
        # بررسی امکان تلاش مجدد
        if task_data['retry_count'] <= task_data['max_retries']:
            logger.warning(f"🔄 Retrying task {task_id} ({task_data['retry_count']}/{task_data['max_retries']})")
            task_data['status'] = 'queued'
            self.task_queue.put(task_data)
        else:
            logger.error(f"❌ Task {task_id} failed after {task_data['max_retries']} retries: {error}")
            self.failed_tasks.append(task_data.copy())
            self._trigger_alert('task_failed', f"Task {task_id} failed permanently", task_data)
            
    def _check_system_health(self) -> Dict[str, Any]:
        """بررسی سلامت سیستم"""
        metrics = self._collect_system_metrics()
        
        health_issues = []
        
        if metrics['cpu_percent'] > self.alert_thresholds['cpu_critical']:
            health_issues.append("CPU usage critically high")
        elif metrics['cpu_percent'] > self.alert_thresholds['cpu_warning']:
            health_issues.append("CPU usage high")
            
        if metrics['memory_percent'] > self.alert_thresholds['memory_critical']:
            health_issues.append("Memory usage critically high")
        elif metrics['memory_percent'] > self.alert_thresholds['memory_warning']:
            health_issues.append("Memory usage high")
            
        if self.task_queue.qsize() > self.alert_thresholds['queue_critical']:
            health_issues.append("Task queue critically long")
        elif self.task_queue.qsize() > self.alert_thresholds['queue_warning']:
            health_issues.append("Task queue long")
            
        return {
            'healthy': len(health_issues) == 0,
            'message': "; ".join(health_issues) if health_issues else "System healthy",
            'metrics': metrics
        }
        
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستم"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'queue_size': self.task_queue.qsize(),
            'active_tasks_count': len(self.active_tasks),
            'active_workers': len([w for w in self.worker_metrics.values() if w.get('status') == 'active'])
        }
        
    def _check_alerts(self, metrics: Dict):
        """بررسی و فعال کردن هشدارها"""
        alerts = []
        
        if metrics['cpu_percent'] > self.alert_thresholds['cpu_critical']:
            alerts.append(('critical', 'cpu', f"CPU critical: {metrics['cpu_percent']}%"))
        elif metrics['cpu_percent'] > self.alert_thresholds['cpu_warning']:
            alerts.append(('warning', 'cpu', f"CPU warning: {metrics['cpu_percent']}%"))
            
        if metrics['memory_percent'] > self.alert_thresholds['memory_critical']:
            alerts.append(('critical', 'memory', f"Memory critical: {metrics['memory_percent']}%"))
        elif metrics['memory_percent'] > self.alert_thresholds['memory_warning']:
            alerts.append(('warning', 'memory', f"Memory warning: {metrics['memory_percent']}%"))
            
        # فعال کردن هشدارها
        for level, category, message in alerts:
            self._trigger_alert(level, category, message, metrics)
            
    def _trigger_alert(self, level: str, category: str, message: str, data: Dict = None):
        """فعال کردن هشدار"""
        alert = {
            'level': level,
            'category': category,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        logger.warning(f"🚨 ALERT {level.upper()}: {message}")
        
        # ارسال به هندلرهای هشدار
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"❌ Alert handler error: {e}")
                
    def _can_run_heavy_task(self) -> bool:
        """بررسی امکان اجرای کارهای سنگین"""
        now = datetime.now()
        
        # آخر هفته (جمعه و شنبه)
        if now.weekday() in [4, 5]:  # Friday, Saturday
            return True
            
        # شب‌ها از ۱ تا ۷ صبح
        if 1 <= now.hour <= 7:
            return True
            
        return False
        
    def _calculate_optimal_sleep_time(self, metrics: Dict) -> float:
        """محاسبه زمان خواب بهینه بر اساس بار سیستم"""
        base_sleep = 2.0
        
        if metrics['cpu_percent'] > 80:
            return base_sleep * 3  # خواب بیشتر هنگام شلوغی
        elif metrics['cpu_percent'] < 30:
            return base_sleep * 0.5  # خواب کمتر هنگام خلوت
            
        return base_sleep
        
    def _start_worker_monitoring(self):
        """شروع مانیتورینگ کارگران"""
        # پیاده‌سازی در فایل‌های بعدی تکمیل می‌شود
        pass
        
    def _start_worker_monitoring_task(self, worker_id: int, task_id: str):
        """شروع مانیتورینگ یک کارگر خاص"""
        self.worker_metrics[worker_id] = {
            'worker_id': worker_id,
            'task_id': task_id,
            'status': 'active',
            'start_time': datetime.now(),
            'cpu_usage': 0,
            'memory_usage': 0,
            'task_start_time': datetime.now()
        }
        
    def _stop_worker_monitoring_task(self, worker_id: int):
        """توقف مانیتورینگ یک کارگر"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id]['status'] = 'idle'
            self.worker_metrics[worker_id]['end_time'] = datetime.now()
            
    def _update_worker_metrics(self, worker_id: int, status: str, execution_time: float):
        """به‌روزرسانی متریک‌های کارگر"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].update({
                'last_status': status,
                'last_execution_time': execution_time,
                'last_update': datetime.now()
            })
            
    def _update_task_stats(self, task_type: str, action: str):
        """به‌روزرسانی آمار کارها"""
        if task_type not in self.performance_stats['tasks_by_type']:
            self.performance_stats['tasks_by_type'][task_type] = {
                'submitted': 0,
                'completed': 0,
                'failed': 0
            }
            
        self.performance_stats['tasks_by_type'][task_type][action] += 1
        
    def _update_performance_stats(self, task_data: Dict, execution_time: float):
        """به‌روزرسانی آمار عملکرد"""
        self.performance_stats['total_tasks_processed'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        self.performance_stats['avg_task_duration'] = (
            self.performance_stats['total_execution_time'] / 
            self.performance_stats['total_tasks_processed']
        )
        
        # الگوی ساعتی
        hour = datetime.now().hour
        if hour not in self.performance_stats['hourly_pattern']:
            self.performance_stats['hourly_pattern'][hour] = 0
        self.performance_stats['hourly_pattern'][hour] += 1
        
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های دقیق سیستم"""
        system_metrics = self._collect_system_metrics()
        health_status = self._check_system_health()
        
        return {
            'system_health': health_status,
            'performance_stats': self.performance_stats,
            'current_metrics': system_metrics,
            'worker_status': {
                'total_workers': self.max_workers,
                'active_workers': len([w for w in self.worker_metrics.values() if w.get('status') == 'active']),
                'idle_workers': len([w for w in self.worker_metrics.values() if w.get('status') == 'idle']),
                'worker_details': list(self.worker_metrics.values())
            },
            'queue_status': {
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            },
            'task_breakdown': self.performance_stats['tasks_by_type'],
            'timestamp': datetime.now().isoformat()
        }
        
    def add_alert_handler(self, handler: Callable):
        """اضافه کردن هندلر هشدار"""
        self.alert_handlers.append(handler)
        
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """دریافت وضعیت یک کار"""
        return self.active_tasks.get(task_id)

# نمونه گلوبال
background_worker = IntelligentBackgroundWorker(max_workers=4, max_cpu_percent=65.0)

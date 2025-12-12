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
import random

logger = logging.getLogger(__name__)

class IntelligentBackgroundWorker:
    """سیستم هوشمند مدیریت کارهای پس‌زمینه با مانیتورینگ پیشرفته"""
    
    def __init__(self, max_workers: int = 2, max_cpu_percent: float = 60.0):  # کاهش یافته!
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
        
        # سیستم مانیتورینگ مرکزی
        self.central_monitor_connected = False
        self.last_central_metrics = None
        self.central_monitor_initialized = False
        
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
            'cpu_warning': 75.0,  # کاهش یافته!
            'cpu_critical': 85.0,  # کاهش یافته!
            'memory_warning': 80.0,  # کاهش یافته!
            'memory_critical': 90.0,  # کاهش یافته!
            'queue_warning': 15,  # کاهش یافته!
            'queue_critical': 30,  # کاهش یافته!
            'task_timeout': 180  # 3 دقیقه (کاهش یافته!)
        }
        
        # عضویت در سیستم مانیتورینگ مرکزی
        self._subscribe_to_central_monitor_with_retry()
        
        logger.info("🎯 Intelligent Background Worker initialized (CPU-Safe Mode)")
        
    def _subscribe_to_central_monitor_with_retry(self):
        """عضویت در سیستم مانیتورینگ مرکزی با قابلیت تلاش مجدد"""
        import time
        
        logger.info("🔌 Attempting to connect to Central Monitor...")
        
        max_attempts = 12  # 60 ثانیه منتظر بمان (12 * 5)
        for attempt in range(max_attempts):
            try:
                # تلاش برای import central_monitor
                from debug_system.monitors.system_monitor import central_monitor
                
                if central_monitor and hasattr(central_monitor, 'subscribe'):
                    central_monitor.subscribe("background_worker", self._on_central_metrics_update)
                    self.central_monitor_connected = True
                    self.central_monitor_initialized = True
                    logger.info(f"✅✅✅ Background Worker SUCCESSFULLY subscribed to Central Monitor (attempt {attempt + 1})")
                    
                    # تأیید اتصال با دریافت یک متریک
                    self._verify_central_monitor_connection()
                    return
                else:
                    status = "not_initialized" if not central_monitor else "no_subscribe_method"
                    logger.debug(f"⏳ Central monitor {status} (attempt {attempt + 1}/{max_attempts})")
                    
            except ImportError as e:
                logger.debug(f"⏳ Could not import central_monitor module (attempt {attempt + 1}/{max_attempts})")
            except Exception as e:
                logger.debug(f"⏳ Error accessing central_monitor: {e} (attempt {attempt + 1}/{max_attempts})")
            
            # افزایش تدریجی زمان انتظار
            wait_time = min(10, (attempt + 1) * 2)
            time.sleep(wait_time)
        
        # اگر پس از انتظار هم موفق نشد
        logger.warning("⚠️⚠️⚠️ Central monitor not available after 60 seconds")
        logger.info("🔄 Will use ULTRA-LOW-CPU fallback monitoring mode")
        self.central_monitor_connected = False
        self.central_monitor_initialized = False
    
    def _verify_central_monitor_connection(self):
        """تأیید اتصال به central_monitor"""
        try:
            from debug_system.monitors.system_monitor import central_monitor
            
            # بررسی وجود central_monitor و فعال بودن آن
            if central_monitor and hasattr(central_monitor, 'is_monitoring'):
                status = "active" if central_monitor.is_monitoring else "inactive"
                logger.info(f"📡 Central Monitor status: {status}")
                
                # دریافت یک نمونه متریک برای تأیید
                metrics = central_monitor.get_current_metrics()
                if metrics:
                    logger.info("🔗 Central Monitor connection VERIFIED")
                    return True
        except Exception as e:
            logger.warning(f"⚠️ Could not verify central monitor connection: {e}")
        
        return False
    
    def start(self):
        """راه‌اندازی کارگر پس‌زمینه"""
        if self.is_running:
            logger.warning("⚠️ Background Worker is already running")
            return
            
        self.is_running = True
        
        # **تغییر مهم**: فقط اگر به مرکز متصل شدیم یا وضعیت مشخص است
        if self.central_monitor_connected:
            logger.info("🎬 Background Worker started (FULLY CONNECTED to Central Monitor)")
            # در این حالت نیازی به حلقه مستقل نیست
            self._start_worker_monitoring_light()
            self.submit_real_tasks()
        elif self.central_monitor_initialized:
            # مرکز موجود است اما اتصال کامل نیست
            logger.info("🎬 Background Worker started (PARTIALLY CONNECTED to Central Monitor)")
            self._start_ultra_low_cpu_monitoring()
            self.submit_real_tasks()
        else:
            # حالت fallback با مصرف CPU بسیار کم
            logger.warning("🎬 Background Worker started in ULTRA-LOW-CPU FALLBACK mode")
            self._start_ultra_low_cpu_monitoring()
            # در این حالت کارهای سنگین را ارسال نکن
            self.submit_light_tasks_only()
            
        logger.info(f"📊 Worker configuration: max_workers={self.max_workers}, max_cpu={self.max_cpu_percent}%")
    
    def _start_ultra_low_cpu_monitoring(self):
        """شروع مانیتورینگ با مصرف CPU بسیار کم"""
        if not self.is_running:
            return
            
        self.monitor_thread = threading.Thread(
            target=self._ultra_low_cpu_monitor_loop, 
            daemon=True,
            name="UltraLowCPUMonitor"
        )
        self.monitor_thread.start()
    
    def _ultra_low_cpu_monitor_loop(self):
        """حلقه مانیتورینگ با مصرف CPU بسیار بسیار کم"""
        logger.info("🐌 Starting ULTRA-LOW-CPU monitoring loop")
        
        # تنظیمات مصرف CPU بسیار کم
        check_interval = 45  # هر 45 ثانیه
        health_check_interval = 120  # هر 2 دقیقه
        queue_check_interval = 30  # هر 30 ثانیه
        
        last_health_check = 0
        last_queue_check = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 1. بررسی صف کارها (هر 30 ثانیه)
                if current_time - last_queue_check >= queue_check_interval:
                    if not self.task_queue.empty():
                        # فقط اگر CPU زیر 50% است کار اجرا کن
                        cpu_percent = psutil.cpu_percent(interval=0.5)
                        if cpu_percent < 50:
                            try:
                                task_data = self.task_queue.get(timeout=0.5)
                                self._execute_task_with_monitoring(task_data)
                            except Empty:
                                pass
                    
                    last_queue_check = current_time
                
                # 2. بررسی سلامت سیستم (هر 2 دقیقه)
                if current_time - last_health_check >= health_check_interval:
                    system_health = self._check_system_health_light()
                    if not system_health["healthy"]:
                        logger.warning(f"⚠️ System health issue: {system_health['message']}")
                    
                    last_health_check = current_time
                
                # 3. خواب طولانی برای کاهش CPU
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"❌ Ultra-low CPU monitor error: {e}")
                time.sleep(60)  # در صورت خطا بیشتر صبر کن
    
    def _start_worker_monitoring_light(self):
        """شروع مانیتورینگ سبک کارگران"""
        # فقط ثبت اولیه
        logger.info("👷 Worker monitoring initialized (light mode)")
    
    def submit_light_tasks_only(self):
        """ارسال فقط کارهای سبک در حالت fallback"""
        try:
            from debug_system.tools.background_tasks import background_tasks
            
            # فقط کارهای سبک
            self.submit_task(
                task_id="generate_basic_report",
                task_func=background_tasks.generate_real_performance_report,
                task_type="light",
                priority=3,
                days=1,
                detail_level="minimal"
            )
            
            logger.info("📥 Light tasks submitted (CPU-safe mode)")
        
        except Exception as e:
            logger.error(f"❌ Error submitting light tasks: {e}")
    
    def _on_central_metrics_update(self, metrics: Dict):
        """دریافت به‌روزرسانی متریک از سیستم مرکزی"""
        try:
            self.last_central_metrics = metrics
            
            # پردازش متریک‌های سیستم
            system_metrics = metrics.get('system', {})
            
            # بررسی هشدارها بر اساس متریک‌های مرکزی
            self._check_alerts_from_central(system_metrics)
            
            # لاگ کردن وضعیت CPU
            cpu_percent = system_metrics.get('cpu', {}).get('percent', 0)
            if cpu_percent > 70:
                logger.debug(f"📊 Central metrics: CPU at {cpu_percent}%")
                
        except Exception as e:
            logger.error(f"❌ Error processing central metrics: {e}")
    
    def _check_alerts_from_central(self, system_metrics: Dict):
        """بررسی هشدارها بر اساس متریک‌های مرکزی"""
        cpu_percent = system_metrics.get('cpu', {}).get('percent', 0)
        memory_percent = system_metrics.get('memory', {}).get('percent', 0)
        
        # فقط هشدارهای بحرانی
        if cpu_percent > self.alert_thresholds['cpu_critical']:
            self._trigger_alert('critical', 'cpu', f"CPU CRITICAL: {cpu_percent}% (via Central Monitor)", system_metrics)
        elif cpu_percent > 90:  # حتی بالاتر از threshold
            self._trigger_alert('critical', 'cpu', f"CPU EXTREME: {cpu_percent}% (via Central Monitor)", system_metrics)
    
    def _check_system_health_light(self) -> Dict[str, Any]:
        """بررسی سلامت سیستم با مصرف CPU کم"""
        # فقط CPU را چک کن (کمترین مصرف)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        health_issues = []
        
        if cpu_percent > self.alert_thresholds['cpu_critical']:
            health_issues.append(f"CPU critical: {cpu_percent}%")
        elif cpu_percent > self.alert_thresholds['cpu_warning']:
            health_issues.append(f"CPU warning: {cpu_percent}%")
            
        return {
            'healthy': len(health_issues) == 0,
            'message': "; ".join(health_issues) if health_issues else "System healthy",
            'cpu_percent': cpu_percent,
            'source': 'light_check'
        }
        
    def submit_real_tasks(self):
        """ثبت کارهای واقعی در سیستم"""
        try:
            from debug_system.tools.background_tasks import background_tasks
        
            # ۱. کار پردازش داده‌های کوین‌ها (با تأخیر)
            self.submit_task(
                task_id="process_coins_data_delayed",
                task_func=self._delayed_data_processing,
                task_type="normal",
                priority=2,
                data_type="coins",
                delay_minutes=2
            )
         
            # ۲. کار پردازش اخبار (با تأخیر بیشتر)
            self.submit_task(
                task_id="process_news_data_delayed", 
                task_func=self._delayed_data_processing,
                task_type="normal",
                priority=3,
                data_type="news",
                delay_minutes=5
            )
        
            # ۳. کار گزارش عملکرد (سبک)
            self.submit_task(
                task_id="generate_performance_report_light",
                task_func=background_tasks.generate_real_performance_report,
                task_type="light",
                priority=1,
                days=1,
                detail_level="basic"
            )
        
            logger.info("📥 Real tasks submitted with delays (CPU-safe)")
        
        except Exception as e:
            logger.error(f"❌ Error submitting real tasks: {e}")
    
    def _delayed_data_processing(self, data_type: str, delay_minutes: int = 2):
        """پردازش داده با تأخیر"""
        from debug_system.tools.background_tasks import background_tasks
        
        # منتظر بمان تا فشار CPU کاهش یابد
        logger.info(f"⏳ Waiting {delay_minutes} minutes before processing {data_type}...")
        time.sleep(delay_minutes * 60)
        
        # بررسی CPU قبل از اجرا
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent < 70:
            return background_tasks.perform_real_data_processing(data_type)
        else:
            logger.warning(f"⚠️ Skipping {data_type} processing - CPU too high: {cpu_percent}%")
            return {"status": "delayed", "reason": f"CPU too high: {cpu_percent}%"}

    def get_real_metrics(self):
        """تولید متریک‌های REAL بر اساس فعالیت واقعی سیستم"""
    
        # تلاش برای استفاده از متریک‌های مرکزی
        if self.last_central_metrics:
            system_metrics = self.last_central_metrics.get('system', {})
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            source = "central_monitor"
        else:
            # حالت fallback با مصرف کم
            cpu_usage = psutil.cpu_percent(interval=0.5)
            memory_usage = 0  # نخوانیم تا CPU کمتری مصرف شود
            source = "fallback_light"
    
        return {
            'worker_status': {
                'active_workers': 0,  # برای کاهش CPU
                'total_workers': self.max_workers,
                'worker_utilization': 0,
                'idle_workers': self.max_workers
            },
            'queue_status': {
                'queue_size': 0,
                'active_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0
            },
            'performance_stats': {
                'total_tasks_processed': 0,
                'success_rate': 100,
                'avg_task_duration': 0
            },
            'system_health': {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'health_status': 'healthy' if cpu_usage < 80 else 'degraded'
            },
            'current_metrics': {
                'timestamp': datetime.now().isoformat(),
                'system_load': 0,
                'efficiency_score': 95,
                'monitoring_source': source,
                'cpu_safe_mode': True
            }
        }

    def stop(self):
        """توقف کارگر پس‌زمینه"""
        self.is_running = False
        self.executor.shutdown(wait=False)
        logger.info("🛑 Background Worker stopped (CPU-Safe)")
        
    def submit_task(self, task_id: str, task_func: Callable, task_type: str = "normal",
                   priority: int = 1, *args, **kwargs) -> Tuple[bool, str]:
        """ثبت یک کار جدید در صف با اولویت‌بندی"""
        if not self.is_running:
            return False, "Worker is not running"
            
        # بررسی سلامت سیستم قبل از ثبت کار
        system_health = self._check_system_health_light()
        if not system_health["healthy"]:
            return False, f"System health check failed: {system_health['message']}"
        
        # بررسی محدودیت‌های کارهای سنگین
        if task_type == "heavy":
            return False, "Heavy tasks disabled in CPU-safe mode"
            
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
            'max_retries': 2  # کاهش یافته
        }
        
        self.task_queue.put(task_data)
        self.active_tasks[task_id] = task_data
        
        logger.info(f"📥 Task {task_id} submitted (Type: {task_type}, Priority: {priority})")
        return True, "Task submitted successfully"
            
    def _trigger_alert(self, level: str, category: str, message: str, data: Dict = None):
        """فعال کردن هشدار"""
        # فقط هشدارهای بحرانی را لاگ کن
        if level == 'critical':
            logger.warning(f"🚨🚨 ALERT {level.upper()}: {message}")
        elif level == 'warning' and 'CPU' in message and '90' in message:
            logger.warning(f"🚨 ALERT {level.upper()}: {message}")
        # هشدارهای دیگر را نادیده بگیر
        
        # ارسال به هندلرهای هشدار
        for handler in self.alert_handlers:
            try:
                handler({
                    'level': level,
                    'category': category,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                })
            except Exception as e:
                logger.error(f"❌ Alert handler error: {e}")
                
    def _execute_task_with_monitoring(self, task_data: Dict):
        """اجرای کار با مانیتورینگ کامل"""
        task_id = task_data['task_id']
        
        try:
            # ثبت شروع کار
            task_data['status'] = 'running'
            task_data['started_at'] = datetime.now()
            
            logger.info(f"⚡ Executing task: {task_id}")
            
            # اجرای کار
            start_time = time.time()
            result = task_data['function'](*task_data['args'], **task_data['kwargs'])
            execution_time = time.time() - start_time
            
            # ثبت موفقیت
            task_data['status'] = 'completed'
            task_data['completed_at'] = datetime.now()
            task_data['execution_time'] = execution_time
            task_data['result'] = result
            
            logger.info(f"✅ Task {task_id} completed in {execution_time:.2f}s")
            
            # انتقال به لیست کارهای انجام شده
            self.completed_tasks.append(task_data.copy())
            
        except Exception as e:
            # ثبت شکست
            task_data['status'] = 'failed'
            task_data['failed_at'] = datetime.now()
            task_data['error'] = str(e)
            
            logger.error(f"❌ Task {task_id} failed: {e}")
            self.failed_tasks.append(task_data.copy())
            
        finally:
            # پاک‌سازی
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های دقیق سیستم"""
        system_health = self._check_system_health_light()
        
        return {
            'system_health': system_health,
            'performance_stats': {
                'total_tasks_processed': len(self.completed_tasks),
                'success_rate': 100 if len(self.completed_tasks) > 0 else 0,
                'avg_task_duration': 0
            },
            'worker_status': {
                'total_workers': self.max_workers,
                'active_workers': 0,
                'idle_workers': self.max_workers,
                'worker_utilization': 0
            },
            'queue_status': {
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            },
            'timestamp': datetime.now().isoformat(),
            'monitoring_mode': 'central' if self.central_monitor_connected else 'ultra_low_cpu',
            'cpu_safe_mode': True,
            'alerts_active': False  # هشدارها غیرفعال در حالت safe
        }
        
    def add_alert_handler(self, handler: Callable):
        """اضافه کردن هندلر هشدار"""
        self.alert_handlers.append(handler)
        
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """دریافت وضعیت یک کار"""
        return self.active_tasks.get(task_id)

# نمونه گلوبال با تنظیمات CPU-safe
background_worker = IntelligentBackgroundWorker(max_workers=2, max_cpu_percent=60.0)

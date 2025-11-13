import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
import json
import random

logger = logging.getLogger(__name__)

class TimeAwareScheduler:
    """سیستم زمان‌بندی هوشمند با قابلیت یادگیری الگوهای مصرف"""
    
    def __init__(self, resource_manager=None):
        self.resource_manager = resource_manager
        self.scheduled_tasks: Dict[str, Dict] = {}
        self.task_history: List[Dict] = []
        self.learning_data: Dict[str, Any] = {}
        self.optimization_rules = {}
        self.is_scheduling = False
        self.scheduler_thread = None
        
        # الگوهای زمانی
        self.time_patterns = {
            'weekend_hours': [4, 5],  # جمعه و شنبه
            'night_hours': list(range(1, 7)),  # ۱ تا ۷ صبح
            'peak_hours': [10, 11, 14, 15, 19, 20],  # ساعات اوج
            'quiet_hours': [2, 3, 4, 13, 23]  # ساعات خلوت
        }
        
        # قوانین بهینه‌سازی
        self._initialize_optimization_rules()
        self._load_learning_data()
        
        logger.info("⏰ Time-Aware Scheduler initialized")
    
    def start_scheduling(self):
        """شروع سیستم زمان‌بندی"""
        if self.is_scheduling:
            return
            
        self.is_scheduling = True
        self.scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("🔄 Intelligent scheduling started")
    
    def stop_scheduling(self):
        """توقف سیستم زمان‌بندی"""
        self.is_scheduling = False
        logger.info("🛑 Intelligent scheduling stopped")
    
    def schedule_task(self, task_id: str, task_func: Callable, task_type: str,
                     interval_seconds: int = 3600, 
                     preferred_times: List[str] = None,
                     priority: int = 1,
                     *args, **kwargs) -> Tuple[bool, str]:
        """زمان‌بندی یک کار با در نظر گرفتن الگوهای بهینه"""
        
        if task_id in self.scheduled_tasks:
            return False, f"Task {task_id} already scheduled"
        
        # محاسبه زمان بهینه برای اجرا
        optimal_time = self._calculate_optimal_time(task_type, interval_seconds, preferred_times)
        
        task_data = {
            'task_id': task_id,
            'function': task_func,
            'task_type': task_type,
            'interval_seconds': interval_seconds,
            'preferred_times': preferred_times or [],
            'priority': priority,
            'args': args,
            'kwargs': kwargs,
            'scheduled_at': datetime.now(),
            'next_execution': optimal_time['next_execution'],
            'execution_window': optimal_time['execution_window'],
            'estimated_duration': optimal_time['estimated_duration'],
            'success_probability': optimal_time['success_probability'],
            'last_execution': None,
            'execution_count': 0,
            'success_count': 0,
            'total_execution_time': 0
        }
        
        self.scheduled_tasks[task_id] = task_data
        
        # یادگیری الگو
        self._learn_scheduling_pattern(task_data, optimal_time)
        
        logger.info(f"📅 Task {task_id} scheduled optimally for {optimal_time['next_execution']}")
        return True, f"Task scheduled optimally (Success probability: {optimal_time['success_probability']}%)"
    
    def _scheduling_loop(self):
        """حلقه اصلی زمان‌بندی"""
        while self.is_scheduling:
            try:
                now = datetime.now()
                tasks_to_execute = []
                
                # بررسی کارهای زمان‌رسیده
                for task_id, task_data in self.scheduled_tasks.items():
                    if task_data['next_execution'] <= now:
                        if self._should_execute_now(task_data, now):
                            tasks_to_execute.append(task_data)
                
                # اجرای کارهای واجد شرایط
                for task_data in tasks_to_execute:
                    self._execute_scheduled_task(task_data)
                
                # بهینه‌سازی زمان‌بندی‌های آینده
                if now.minute % 30 == 0:  # هر 30 دقیقه
                    self._optimize_future_schedules()
                
                time.sleep(10)  # بررسی هر 10 ثانیه
                
            except Exception as e:
                logger.error(f"❌ Scheduling loop error: {e}")
                time.sleep(30)
    
    def _calculate_optimal_time(self, task_type: str, interval: int, preferred_times: List[str]) -> Dict[str, Any]:
        """محاسبه زمان بهینه برای اجرای کار"""
        now = datetime.now()
        
        # تشخیص نوع کار و اعمال قوانین مربوطه
        scheduling_rules = self.optimization_rules.get(task_type, self.optimization_rules['default'])
        
        # محاسبه زمان اجرای بعدی
        if scheduling_rules['constraint'] == 'weekend_night_only':
            next_execution = self._find_next_weekend_night_slot(now, interval)
        elif scheduling_rules['constraint'] == 'night_only':
            next_execution = self._find_next_night_slot(now, interval)
        elif scheduling_rules['constraint'] == 'quiet_hours':
            next_execution = self._find_next_quiet_hour(now, interval)
        else:
            next_execution = now + timedelta(seconds=interval)
        
        # محاسبه احتمال موفقیت
        success_probability = self._calculate_success_probability(task_type, next_execution)
        
        # محاسبه پنجره اجرا
        execution_window = self._calculate_execution_window(next_execution, task_type)
        
        return {
            'next_execution': next_execution,
            'execution_window': execution_window,
            'estimated_duration': scheduling_rules['estimated_duration'],
            'success_probability': success_probability,
            'scheduling_strategy': scheduling_rules['constraint'],
            'resource_requirements': scheduling_rules['resource_requirements']
        }
    
    def _find_next_weekend_night_slot(self, from_time: datetime, interval: int) -> datetime:
        """پیدا کردن زمان بعدی در آخر هفته یا شب"""
        current_time = from_time
        
        for _ in range(168):  # جستجو برای 1 هفته
            # بررسی آخر هفته
            if current_time.weekday() in self.time_patterns['weekend_hours']:
                return current_time
            
            # بررسی ساعات شب
            if current_time.hour in self.time_patterns['night_hours']:
                return current_time
            
            current_time += timedelta(hours=1)
        
        # اگر زمان مناسب پیدا نشد، برگردان به حالت عادی
        return from_time + timedelta(seconds=interval)
    
    def _find_next_night_slot(self, from_time: datetime, interval: int) -> datetime:
        """پیدا کردن زمان بعدی در ساعات شب"""
        current_time = from_time
        
        for _ in range(24):  # جستجو برای 24 ساعت
            if current_time.hour in self.time_patterns['night_hours']:
                return current_time
            
            current_time += timedelta(hours=1)
        
        return from_time + timedelta(seconds=interval)
    
    def _find_next_quiet_hour(self, from_time: datetime, interval: int) -> datetime:
        """پیدا کردن زمان بعدی در ساعات خلوت"""
        current_time = from_time
        
        for _ in range(24):
            if current_time.hour in self.time_patterns['quiet_hours']:
                return current_time
            
            current_time += timedelta(hours=1)
        
        return from_time + timedelta(seconds=interval)
    
    def _calculate_success_probability(self, task_type: str, execution_time: datetime) -> float:
        """محاسبه احتمال موفقیت اجرای کار"""
        base_probability = 85.0  # احتمال پایه
        
        # تنظیم بر اساس نوع کار
        type_modifiers = {
            'heavy': -15.0,
            'normal': 0.0,
            'light': 10.0,
            'maintenance': -5.0
        }
        
        # تنظیم بر اساس ساعت روز
        hour = execution_time.hour
        if hour in self.time_patterns['peak_hours']:
            base_probability -= 20.0
        elif hour in self.time_patterns['quiet_hours']:
            base_probability += 15.0
        elif hour in self.time_patterns['night_hours']:
            base_probability += 10.0
        
        # تنظیم بر اساس روز هفته
        if execution_time.weekday() in self.time_patterns['weekend_hours']:
            base_probability += 25.0
        
        # اعمال اصلاحات
        probability = base_probability + type_modifiers.get(task_type, 0.0)
        
        return max(10.0, min(99.0, probability))  # محدود کردن بین 10% تا 99%
    
    def _calculate_execution_window(self, execution_time: datetime, task_type: str) -> Dict[str, datetime]:
        """محاسبه پنجره زمانی برای اجرای کار"""
        if task_type == 'heavy':
            window_hours = 6  # پنجره بزرگ برای کارهای سنگین
        elif task_type == 'normal':
            window_hours = 3
        else:
            window_hours = 1
        
        start_window = execution_time - timedelta(hours=window_hours/2)
        end_window = execution_time + timedelta(hours=window_hours/2)
        
        return {
            'start': start_window,
            'end': end_window,
            'duration_hours': window_hours
        }
    
    def _should_execute_now(self, task_data: Dict, current_time: datetime) -> bool:
        """بررسی آیا کار باید همین حالا اجرا شود"""
        # بررسی پنجره اجرا
        window = task_data['execution_window']
        if not (window['start'] <= current_time <= window['end']):
            return False
        
        # بررسی منابع سیستم
        if self.resource_manager:
            system_health = self.resource_manager._check_system_health()
            if not system_health['healthy']:
                logger.warning(f"⏳ Delaying {task_data['task_id']} due to system health: {system_health['message']}")
                return False
        
        # بررسی احتمال موفقیت
        current_probability = self._calculate_success_probability(
            task_data['task_type'], current_time
        )
        
        if current_probability < 50:  # حداقل 50% احتمال موفقیت
            logger.warning(f"⏳ Delaying {task_data['task_id']} due to low success probability: {current_probability}%")
            return False
        
        return True
    
    def _execute_scheduled_task(self, task_data: Dict):
        """اجرای یک کار زمان‌بندی شده"""
        task_id = task_data['task_id']
        
        try:
            logger.info(f"⚡ Executing scheduled task: {task_id}")
            
            # ثبت شروع اجرا
            task_data['last_execution'] = datetime.now()
            task_data['execution_count'] += 1
            
            start_time = time.time()
            
            # اجرای کار
            result = task_data['function'](*task_data['args'], **task_data['kwargs'])
            execution_time = time.time() - start_time
            
            # ثبت موفقیت
            task_data['success_count'] += 1
            task_data['total_execution_time'] += execution_time
            task_data['last_execution_time'] = execution_time
            task_data['last_result'] = result
            
            # محاسبه زمان اجرای بعدی
            task_data['next_execution'] = self._calculate_next_execution_time(task_data)
            
            # ذخیره در تاریخچه
            self._record_task_execution(task_data, execution_time, True, result)
            
            logger.info(f"✅ Scheduled task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            # ثبت شکست
            execution_time = time.time() - start_time
            self._record_task_execution(task_data, execution_time, False, str(e))
            
            logger.error(f"❌ Scheduled task {task_id} failed: {e}")
            
            # محاسبه زمان اجرای بعدی با در نظر گرفتن شکست
            task_data['next_execution'] = self._calculate_retry_time(task_data)
    
    def _calculate_next_execution_time(self, task_data: Dict) -> datetime:
        """محاسبه زمان اجرای بعدی پس از موفقیت"""
        base_time = task_data['last_execution'] or datetime.now()
        
        # تنظیم فاصله بر اساس الگوی یادگیری
        learned_interval = self._get_learned_interval(task_data['task_id'])
        interval = learned_interval or task_data['interval_seconds']
        
        return base_time + timedelta(seconds=interval)
    
    def _calculate_retry_time(self, task_data: Dict) -> datetime:
        """محاسبه زمان تلاش مجدد پس از شکست"""
        base_time = datetime.now()
        
        # استراتژی افزایش تدریجی فاصله (Exponential Backoff)
        retry_count = task_data['execution_count'] - task_data['success_count']
        backoff_seconds = min(3600 * 24, 300 * (2 ** retry_count))  # حداکثر 24 ساعت
        
        return base_time + timedelta(seconds=backoff_seconds)
    
    def _record_task_execution(self, task_data: Dict, execution_time: float, success: bool, result: Any):
        """ثبت اجرای کار در تاریخچه"""
        execution_record = {
            'task_id': task_data['task_id'],
            'task_type': task_data['task_type'],
            'execution_time': datetime.now().isoformat(),
            'duration_seconds': execution_time,
            'success': success,
            'result': result if success else str(result),
            'scheduled_time': task_data['next_execution'].isoformat(),
            'actual_time': datetime.now().isoformat(),
            'delay_seconds': (datetime.now() - task_data['next_execution']).total_seconds(),
            'resource_usage': self._get_current_resource_usage()
        }
        
        self.task_history.append(execution_record)
        
        # حفظ اندازه تاریخچه
        if len(self.task_history) > 10000:
            self.task_history = self.task_history[-10000:]
    
    def _get_current_resource_usage(self) -> Dict[str, float]:
        """دریافت استفاده فعلی منابع"""
        try:
            if self.resource_manager:
                metrics = self.resource_manager._collect_comprehensive_metrics()
                return {
                    'cpu_percent': metrics['cpu']['percent'],
                    'memory_percent': metrics['memory']['percent'],
                    'disk_percent': metrics['disk']['usage_percent']
                }
        except:
            pass
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def _learn_scheduling_pattern(self, task_data: Dict, optimal_time: Dict):
        """یادگیری الگوهای زمان‌بندی"""
        task_id = task_data['task_id']
        
        if task_id not in self.learning_data:
            self.learning_data[task_id] = {
                'execution_patterns': [],
                'success_rates': {},
                'optimal_hours': [],
                'avoid_hours': [],
                'performance_metrics': {
                    'avg_duration': 0,
                    'success_rate': 0,
                    'total_executions': 0
                }
            }
    
    def _get_learned_interval(self, task_id: str) -> Optional[int]:
        """دریافت فاصله یادگیری شده برای کار"""
        if task_id in self.learning_data:
            patterns = self.learning_data[task_id]['execution_patterns']
            if patterns:
                # محاسبه فاصله بهینه بر اساس تاریخچه
                intervals = [p.get('interval', 3600) for p in patterns[-10:]]  # 10 اجرای اخیر
                return sum(intervals) // len(intervals)
        
        return None
    
    def _optimize_future_schedules(self):
        """بهینه‌سازی زمان‌بندی‌های آینده"""
        for task_id, task_data in self.scheduled_tasks.items():
            # تحلیل الگوی اجرا
            analysis = self._analyze_task_pattern(task_id)
            
            if analysis['needs_optimization']:
                new_optimal_time = self._calculate_optimal_time(
                    task_data['task_type'],
                    task_data['interval_seconds'],
                    task_data['preferred_times']
                )
                
                # به‌روزرسانی زمان‌بندی اگر بهبود یافته باشد
                if new_optimal_time['success_probability'] > task_data['success_probability'] + 10:
                    task_data['next_execution'] = new_optimal_time['next_execution']
                    task_data['success_probability'] = new_optimal_time['success_probability']
                    
                    logger.info(f"🔄 Optimized schedule for {task_id}: {new_optimal_time['success_probability']}% success probability")
    
    def _analyze_task_pattern(self, task_id: str) -> Dict[str, Any]:
        """آنالیز الگوی اجرای یک کار"""
        task_history = [h for h in self.task_history if h['task_id'] == task_id]
        
        if len(task_history) < 5:
            return {'needs_optimization': False, 'reason': 'insufficient_data'}
        
        # محاسبه نرخ موفقیت
        success_count = sum(1 for h in task_history if h['success'])
        success_rate = success_count / len(task_history)
        
        # محاسبه میانگین تاخیر
        avg_delay = sum(abs(h['delay_seconds']) for h in task_history) / len(task_history)
        
        # شناسایی ساعات مشکل‌دار
        failure_hours = [
            datetime.fromisoformat(h['execution_time']).hour 
            for h in task_history if not h['success']
        ]
        
        needs_optimization = (
            success_rate < 0.7 or 
            avg_delay > 300 or  # تاخیر بیش از 5 دقیقه
            len(set(failure_hours)) > 0  # شکست در ساعات خاص
        )
        
        return {
            'needs_optimization': needs_optimization,
            'success_rate': success_rate,
            'avg_delay_seconds': avg_delay,
            'problem_hours': list(set(failure_hours)),
            'total_executions': len(task_history),
            'recommendation': 'adjust_schedule' if needs_optimization else 'maintain_current'
        }
    
    def _initialize_optimization_rules(self):
        """مقداردهی اولیه قوانین بهینه‌سازی"""
        self.optimization_rules = {
            'heavy': {
                'constraint': 'weekend_night_only',
                'estimated_duration': 1800,  # 30 دقیقه
                'resource_requirements': 'high',
                'retry_strategy': 'exponential_backoff',
                'max_retries': 3
            },
            'normal': {
                'constraint': 'night_only',
                'estimated_duration': 600,  # 10 دقیقه
                'resource_requirements': 'medium',
                'retry_strategy': 'linear_backoff',
                'max_retries': 5
            },
            'light': {
                'constraint': 'quiet_hours',
                'estimated_duration': 300,  # 5 دقیقه
                'resource_requirements': 'low',
                'retry_strategy': 'immediate',
                'max_retries': 10
            },
            'maintenance': {
                'constraint': 'weekend_night_only',
                'estimated_duration': 1200,  # 20 دقیقه
                'resource_requirements': 'medium',
                'retry_strategy': 'linear_backoff',
                'max_retries': 3
            },
            'default': {
                'constraint': 'none',
                'estimated_duration': 300,
                'resource_requirements': 'low',
                'retry_strategy': 'linear_backoff',
                'max_retries': 5
            }
        }
    
    def _load_learning_data(self):
        """بارگذاری داده‌های یادگیری (در آینده از فایل)"""
        try:
            # اینجا می‌تواند از فایل JSON بارگذاری شود
            self.learning_data = {}
        except:
            self.learning_data = {}
    
    def get_scheduling_analytics(self) -> Dict[str, Any]:
        """دریافت آمار و آنالیز زمان‌بندی"""
        now = datetime.now()
        
        # کارهای آینده
        upcoming_tasks = []
        for task_id, task_data in self.scheduled_tasks.items():
            if task_data['next_execution'] > now:
                upcoming_tasks.append({
                    'task_id': task_id,
                    'task_type': task_data['task_type'],
                    'scheduled_time': task_data['next_execution'].isoformat(),
                    'success_probability': task_data['success_probability'],
                    'priority': task_data['priority']
                })
        
        # آنالیز عملکرد
        performance_analysis = self._analyze_scheduling_performance()
        
        # پیش‌بینی‌ها
        predictions = self._generate_scheduling_predictions()
        
        return {
            'scheduling_status': {
                'active_tasks': len(self.scheduled_tasks),
                'upcoming_tasks': len(upcoming_tasks),
                'total_executions': len(self.task_history),
                'is_scheduling_active': self.is_scheduling
            },
            'upcoming_schedule': sorted(upcoming_tasks, key=lambda x: x['scheduled_time'])[:10],  # 10 کار بعدی
            'performance_analysis': performance_analysis,
            'predictions': predictions,
            'learning_insights': {
                'tasks_with_learned_patterns': len(self.learning_data),
                'total_learning_records': sum(len(data.get('execution_patterns', [])) for data in self.learning_data.values()),
                'optimization_opportunities': self._find_optimization_opportunities()
            },
            'timestamp': now.isoformat()
        }
    
    def _analyze_scheduling_performance(self) -> Dict[str, Any]:
        """آنالیز عملکرد کلی زمان‌بندی"""
        if not self.task_history:
            return {'status': 'no_data'}
        
        recent_history = self.task_history[-100:]  # 100 اجرای اخیر
        
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        avg_duration = sum(h['duration_seconds'] for h in recent_history) / len(recent_history)
        avg_delay = sum(abs(h['delay_seconds']) for h in recent_history) / len(recent_history)
        
        # تحلیل بر اساس نوع کار
        task_type_analysis = {}
        for task_type in ['heavy', 'normal', 'light', 'maintenance']:
            type_history = [h for h in recent_history if h.get('task_type') == task_type]
            if type_history:
                task_type_analysis[task_type] = {
                    'count': len(type_history),
                    'success_rate': sum(1 for h in type_history if h['success']) / len(type_history),
                    'avg_duration': sum(h['duration_seconds'] for h in type_history) / len(type_history)
                }
        
        return {
            'overall_success_rate': round(success_rate * 100, 2),
            'avg_execution_duration': round(avg_duration, 2),
            'avg_scheduling_delay': round(avg_delay, 2),
            'task_type_breakdown': task_type_analysis,
            'efficiency_score': self._calculate_scheduling_efficiency(recent_history),
            'recommendations': self._generate_performance_recommendations(recent_history)
        }
    
    def _calculate_scheduling_efficiency(self, history: List[Dict]) -> float:
        """محاسبه امتیاز کارایی زمان‌بندی"""
        if not history:
            return 0.0
        
        success_rate = sum(1 for h in history if h['success']) / len(history)
        avg_delay = sum(abs(h['delay_seconds']) for h in history) / len(history)
        
        # نرمال‌سازی تاخیر (تاخیر کمتر از 60 ثانیه ایده‌آل است)
        delay_efficiency = max(0, 1 - (avg_delay / 300))  # حداکثر 5 دقیقه تاخیر قابل قبول
        
        return round((success_rate * 0.7 + delay_efficiency * 0.3) * 100, 2)
    
    def _generate_performance_recommendations(self, history: List[Dict]) -> List[str]:
        """تولید توصیه‌های بهبود عملکرد"""
        recommendations = []
        
        if not history:
            return recommendations
        
        success_rate = sum(1 for h in history if h['success']) / len(history)
        avg_delay = sum(abs(h['delay_seconds']) for h in history) / len(history)
        
        if success_rate < 0.8:
            recommendations.append("Improve task success rate by adjusting execution times")
        
        if avg_delay > 300:
            recommendations.append("Reduce scheduling delays by optimizing time calculations")
        
        # تحلیل ساعات مشکل‌دار
        failure_hours = {}
        for h in history:
            if not h['success']:
                hour = datetime.fromisoformat(h['execution_time']).hour
                failure_hours[hour] = failure_hours.get(hour, 0) + 1
        
        if failure_hours:
            worst_hour = max(failure_hours, key=failure_hours.get)
            recommendations.append(f"Avoid scheduling during hour {worst_hour} due to high failure rate")
        
        return recommendations
    
    def _generate_scheduling_predictions(self) -> Dict[str, Any]:
        """تولید پیش‌بینی‌های زمان‌بندی"""
        now = datetime.now()
        
        # پیش‌بینی بار هفتگی
        weekly_load = {}
        for i in range(7):
            day = (now + timedelta(days=i)).weekday()
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]
            
            if day in self.time_patterns['weekend_hours']:
                load_level = 'low'
            elif day in [0, 1, 2, 3]:  # روزهای کاری
                load_level = 'high'
            else:
                load_level = 'medium'
            
            weekly_load[day_name] = {
                'load_level': load_level,
                'recommended_strategy': 'aggressive' if load_level == 'low' else 'conservative'
            }
        
        return {
            'weekly_load_prediction': weekly_load,
            'optimal_scheduling_windows': self._find_optimal_windows_next_week(),
            'resource_availability_forecast': self._forecast_resource_availability()
        }
    
    def _find_optimal_windows_next_week(self) -> List[Dict]:
        """پیدا کردن پنجره‌های بهینه در هفته آینده"""
        optimal_windows = []
        now = datetime.now()
        
        for i in range(7):
            day = now + timedelta(days=i)
            
            if day.weekday() in self.time_patterns['weekend_hours']:
                # آخر هفته - کل روز بهینه است
                optimal_windows.append({
                    'date': day.strftime('%Y-%m-%d'),
                    'window_type': 'full_day',
                    'reason': 'weekend_optimal'
                })
            else:
                # روزهای کاری - ساعات شب بهینه هستند
                optimal_windows.append({
                    'date': day.strftime('%Y-%m-%d'),
                    'window_type': 'night_hours',
                    'hours': self.time_patterns['night_hours'],
                    'reason': 'low_system_load'
                })
        
        return optimal_windows
    
    def _forecast_resource_availability(self) -> Dict[str, Any]:
        """پیش‌بینی دسترسی به منابع"""
        return {
            'next_24_hours': {
                'cpu_availability': 'high' if datetime.now().hour in self.time_patterns['night_hours'] else 'medium',
                'memory_availability': 'high',
                'recommended_task_types': ['heavy', 'normal', 'light'] if datetime.now().hour in self.time_patterns['night_hours'] else ['light', 'normal']
            },
            'next_week': {
                'optimal_days': ['Friday', 'Saturday'],
                'avoid_days': ['Monday', 'Tuesday'],
                'strategy_recommendation': 'Schedule heavy tasks on weekends, light tasks on weekdays'
            }
        }
    
    def _find_optimization_opportunities(self) -> List[str]:
        """یافتن فرصت‌های بهینه‌سازی"""
        opportunities = []
        
        for task_id, task_data in self.scheduled_tasks.items():
            analysis = self._analyze_task_pattern(task_id)
            
            if analysis['needs_optimization']:
                opportunities.append(f"Optimize scheduling for {task_id} (success rate: {analysis['success_rate']*100:.1f}%)")
        
        # فرصت‌های کلی
        if len(self.task_history) > 1000:
            opportunities.append("Consider archiving old task history to improve performance")
        
        if len(self.scheduled_tasks) > 50:
            opportunities.append("Evaluate task priorities and consider deprecating low-priority tasks")
        
        return opportunities

# نمونه گلوبال
time_scheduler = TimeAwareScheduler()

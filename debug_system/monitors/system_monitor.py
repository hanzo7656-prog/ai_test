import psutil
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, metrics_collector, alert_manager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.system_thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 90.0,
            'disk_critical': 98.0,
            'temperature_warning': 80.0,
            'temperature_critical': 90.0
        }
        
        self.health_check_running = False
        self._start_system_health_check()

    def _start_system_health_check(self):
        """شروع چک سلامت سیستم - نسخه اصلاح شده بدون async"""
        def health_check_loop():
            self.health_check_running = True
            while self.health_check_running:
                try:
                    self._perform_health_check()
                    time.sleep(30)  # هر ۳۰ ثانیه
                except Exception as e:
                    logger.error(f"❌ System health check error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=health_check_loop, daemon=True)
        monitor_thread.start()
        logger.info("✅ System health monitoring started")

    def stop_health_check(self):
        """توقف چک سلامت سیستم"""
        self.health_check_running = False
        logger.info("🛑 System health monitoring stopped")

    def _perform_health_check(self):
        """انجام چک سلامت سیستم - کاملاً synchronous"""
        try:
            metrics = self.metrics_collector.get_current_metrics()
            
            # Import مستقیم Enumها برای جلوگیری از circular import
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            # بررسی CPU
            cpu_usage = metrics['cpu']['percent']
            if cpu_usage > self.system_thresholds['cpu_critical']:
                self._create_alert_sync(
                    level=AlertLevel.CRITICAL,
                    alert_type=AlertType.SYSTEM,
                    title="High CPU Usage",
                    message=f"CPU usage is critically high: {cpu_usage}%",
                    source="system_monitor",
                    data={'cpu_usage': cpu_usage, 'threshold': self.system_thresholds['cpu_critical']}
                )
            elif cpu_usage > self.system_thresholds['cpu_warning']:
                self._create_alert_sync(
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.SYSTEM,
                    title="High CPU Usage",
                    message=f"CPU usage is high: {cpu_usage}%",
                    source="system_monitor",
                    data={'cpu_usage': cpu_usage, 'threshold': self.system_thresholds['cpu_warning']}
                )

            # بررسی حافظه
            memory_usage = metrics['memory']['percent']
            if memory_usage > self.system_thresholds['memory_critical']:
                self._create_alert_sync(
                    level=AlertLevel.CRITICAL,
                    alert_type=AlertType.SYSTEM,
                    title="High Memory Usage",
                    message=f"Memory usage is critically high: {memory_usage}%",
                    source="system_monitor",
                    data={'memory_usage': memory_usage, 'threshold': self.system_thresholds['memory_critical']}
                )
            elif memory_usage > self.system_thresholds['memory_warning']:
                self._create_alert_sync(
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.SYSTEM,
                    title="High Memory Usage", 
                    message=f"Memory usage is high: {memory_usage}%",
                    source="system_monitor",
                    data={'memory_usage': memory_usage, 'threshold': self.system_thresholds['memory_warning']}
                )

            # بررسی دیسک
            disk_usage = metrics['disk']['usage_percent']
            if disk_usage > self.system_thresholds['disk_critical']:
                self._create_alert_sync(
                    level=AlertLevel.CRITICAL,
                    alert_type=AlertType.SYSTEM,
                    title="High Disk Usage",
                    message=f"Disk usage is critically high: {disk_usage}%",
                    source="system_monitor", 
                    data={'disk_usage': disk_usage, 'threshold': self.system_thresholds['disk_critical']}
                )
            elif disk_usage > self.system_thresholds['disk_warning']:
                self._create_alert_sync(
                    level=AlertLevel.WARNING,
                    alert_type=AlertType.SYSTEM,
                    title="High Disk Usage",
                    message=f"Disk usage is high: {disk_usage}%",
                    source="system_monitor",
                    data={'disk_usage': disk_usage, 'threshold': self.system_thresholds['disk_warning']}
                )

        except Exception as e:
            logger.error(f"❌ Error in system health check: {e}")

    def _create_alert_sync(self, level, alert_type, title, message, source, data):
        """ایجاد هشدار به صورت کاملاً synchronous"""
        try:
            # ایجاد هشدار به صورت مستقیم - بدون async
            alert_result = self.alert_manager.create_alert(
                level=level,
                alert_type=alert_type,
                title=title,
                message=message,
                source=source,
                data=data
            )
            
            if alert_result:
                logger.info(f"🚨 Alert created: {title}")
            else:
                logger.warning(f"⚠️ Alert was not created (might be in cooldown): {title}")
                
        except Exception as e:
            logger.error(f"❌ Error creating alert: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """دریافت سلامت کلی سیستم"""
        metrics = self.metrics_collector.get_current_metrics()
        
        health_indicators = {
            'cpu': self._evaluate_cpu_health(metrics['cpu']),
            'memory': self._evaluate_memory_health(metrics['memory']),
            'disk': self._evaluate_disk_health(metrics['disk']),
            'network': self._evaluate_network_health(metrics['network']),
            'process': self._evaluate_process_health(metrics['process'])
        }
        
        overall_health = self._calculate_overall_system_health(health_indicators)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'health_indicators': health_indicators,
            'metrics_snapshot': {
                'cpu_usage': metrics['cpu']['percent'],
                'memory_usage': metrics['memory']['percent'],
                'disk_usage': metrics['disk']['usage_percent'],
                'network_activity': f"↑{metrics['network']['mb_sent_per_sec']}MB/s ↓{metrics['network']['mb_recv_per_sec']}MB/s"
            }
        }

    def _evaluate_cpu_health(self, cpu_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت CPU"""
        usage = cpu_metrics['percent']
        
        if usage > self.system_thresholds['cpu_critical']:
            status = 'critical'
            message = f'CPU usage critically high: {usage}%'
        elif usage > self.system_thresholds['cpu_warning']:
            status = 'warning'
            message = f'CPU usage high: {usage}%'
        else:
            status = 'healthy'
            message = f'CPU usage normal: {usage}%'
        
        return {
            'status': status,
            'message': message,
            'usage_percent': usage,
            'load_average': cpu_metrics.get('load_average', []),
            'per_core_usage': cpu_metrics.get('per_core', [])
        }

    def _evaluate_memory_health(self, memory_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت حافظه"""
        usage = memory_metrics['percent']
        
        if usage > self.system_thresholds['memory_critical']:
            status = 'critical'
            message = f'Memory usage critically high: {usage}%'
        elif usage > self.system_thresholds['memory_warning']:
            status = 'warning' 
            message = f'Memory usage high: {usage}%'
        else:
            status = 'healthy'
            message = f'Memory usage normal: {usage}%'
        
        return {
            'status': status,
            'message': message,
            'usage_percent': usage,
            'used_gb': memory_metrics['used_gb'],
            'available_gb': memory_metrics['available_gb'],
            'total_gb': memory_metrics['total_gb']
        }

    def _evaluate_disk_health(self, disk_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت دیسک"""
        usage = disk_metrics['usage_percent']
        
        if usage > self.system_thresholds['disk_critical']:
            status = 'critical'
            message = f'Disk usage critically high: {usage}%'
        elif usage > self.system_thresholds['disk_warning']:
            status = 'warning'
            message = f'Disk usage high: {usage}%'
        else:
            status = 'healthy'
            message = f'Disk usage normal: {usage}%'
        
        return {
            'status': status,
            'message': message,
            'usage_percent': usage,
            'used_gb': disk_metrics['used_gb'],
            'free_gb': disk_metrics['free_gb'],
            'total_gb': disk_metrics['total_gb'],
            'io_activity': {
                'read_mb_sec': disk_metrics['io_read_mb_per_sec'],
                'write_mb_sec': disk_metrics['io_write_mb_per_sec']
            }
        }

    def _evaluate_network_health(self, network_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت شبکه"""
        sent_speed = network_metrics['mb_sent_per_sec']
        recv_speed = network_metrics['mb_recv_per_sec']
        connections = network_metrics['connections']
        
        # منطق ساده برای ارزیابی شبکه
        if sent_speed > 100 or recv_speed > 100:  # 100MB/s threshold
            status = 'warning'
            message = f'High network activity: ↑{sent_speed}MB/s ↓{recv_speed}MB/s'
        elif connections > 1000:
            status = 'warning'
            message = f'High number of connections: {connections}'
        else:
            status = 'healthy'
            message = f'Network activity normal: ↑{sent_speed}MB/s ↓{recv_speed}MB/s'
        
        return {
            'status': status,
            'message': message,
            'upload_speed_mb_sec': sent_speed,
            'download_speed_mb_sec': recv_speed,
            'active_connections': connections
        }

    def _evaluate_process_health(self, process_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت پردازش"""
        memory_mb = process_metrics['memory_mb']
        cpu_percent = process_metrics['cpu_percent']
        threads = process_metrics['threads']
        
        issues = []
        
        if memory_mb > 1000:  # 1GB threshold
            issues.append(f'High memory usage: {memory_mb}MB')
        
        if cpu_percent > 50:
            issues.append(f'High CPU usage: {cpu_percent}%')
        
        if threads > 100:
            issues.append(f'High thread count: {threads}')
        
        if issues:
            status = 'warning'
            message = 'Process health issues: ' + ', '.join(issues)
        else:
            status = 'healthy'
            message = 'Process health normal'
        
        return {
            'status': status,
            'message': message,
            'memory_usage_mb': memory_mb,
            'cpu_usage_percent': cpu_percent,
            'thread_count': threads,
            'open_files': process_metrics.get('open_files', 0),
            'connections': process_metrics.get('connections', 0)
        }

    def _calculate_overall_system_health(self, health_indicators: Dict) -> str:
        """محاسبه سلامت کلی سیستم"""
        status_weights = {
            'critical': 3,
            'warning': 2, 
            'healthy': 1
        }
        
        total_weight = 0
        for indicator in health_indicators.values():
            total_weight += status_weights.get(indicator['status'], 1)
        
        average_weight = total_weight / len(health_indicators)
        
        if average_weight >= 2.5:
            return 'critical'
        elif average_weight >= 1.8:
            return 'warning'
        else:
            return 'healthy'

    def get_resource_usage_trend(self, hours: int = 6) -> Dict[str, Any]:
        """دریافت روند استفاده از منابع"""
        metrics_history = self.metrics_collector.get_metrics_history(seconds=hours*3600)
        
        trends = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network_sent': [],
            'network_recv': []
        }
        
        for metric in metrics_history:
            trends['cpu'].append(metric['cpu_percent'])
            trends['memory'].append(metric['memory_percent'])
            trends['disk'].append(metric['disk_usage'])
            trends['network_sent'].append(metric['network_sent_mb_sec'])
            trends['network_recv'].append(metric['network_recv_mb_sec'])
        
        return {
            'time_period_hours': hours,
            'data_points': len(metrics_history),
            'trends': trends,
            'timestamp': datetime.now().isoformat()
        }

class CentralMonitoringSystem:
    """سیستم نظارت متمرکز - مرجع اصلی همه متریک‌ها"""
    
    def __init__(self, metrics_collector, alert_manager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # تنظیمات متمرکز
        self.collection_interval = 30  # ثانیه - از ۵ به ۳۰ افزایش دادیم
        self.metrics_cache = {}
        self.cache_ttl = 30  # ثانیه
        self.last_collection_time = None
        self.subscribers = {}  # سیستم‌های مشترک
        self.is_monitoring = False
        self.monitor_thread = None
        
        # تاریخچه هشدارها برای جلوگیری از تکرار
        self.alert_cooldown = {}
        self.cooldown_period = 60  # حداقل ۱ دقیقه بین هشدارهای مشابه
        
        logger.info("🎯 Central Monitoring System initialized")
    
    def start_monitoring(self):
        """شروع نظارت متمرکز - فقط یک حلقه در کل سیستم"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._central_monitoring_loop, 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("🔄 Central monitoring started (interval: 30s)")
    
    def stop_monitoring(self):
        """توقف نظارت متمرکز"""
        self.is_monitoring = False
        logger.info("🛑 Central monitoring stopped")
    
    def _central_monitoring_loop(self):
        """حلقه نظارت متمرکز - تنها حلقه فعال"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # ۱. جمع‌آوری متریک‌ها (فقط یک بار)
                metrics = self._collect_all_metrics_once()
                
                # ۲. ذخیره در کش
                self.metrics_cache = metrics
                self.last_collection_time = datetime.now()
                
                # ۳. بررسی هشدارها (فقط یک سیستم)
                self._check_and_trigger_alerts(metrics)
                
                # ۴. اطلاع‌رسانی به مشترکین
                self._notify_subscribers(metrics)
                
                execution_time = time.time() - start_time
                
                # لاگ‌گیری با جزئیات
                if execution_time > 2:  # اگر جمع‌آوری بیش از ۲ ثانیه طول کشید
                    logger.warning(f"⚠️ Metrics collection took {execution_time:.2f}s")
                
                # خواب هوشمند - اگر سیستم شلوغ است بیشتر صبر کن
                sleep_time = self._calculate_smart_sleep(metrics, execution_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Central monitoring error: {e}")
                time.sleep(60)  # در صورت خطا بیشتر صبر کن
    
    def _collect_all_metrics_once(self) -> Dict[str, Any]:
        """جمع‌آوری یک‌باره تمام متریک‌های مورد نیاز"""
        timestamp = datetime.now()
        
        # جمع‌آوری متریک‌های اصلی
        system_metrics = self._collect_system_metrics()
        
        # اضافه کردن متریک‌های تخصصی (اگر سیستم‌ها در دسترس باشند)
        specialized_metrics = self._collect_specialized_metrics()
        
        return {
            'timestamp': timestamp.isoformat(),
            'system': system_metrics,
            'specialized': specialized_metrics,
            'collection_time': time.time(),
            'collection_duration': 0  # بعداً محاسبه می‌شود
        }
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستم"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent': psutil.cpu_percent(interval=0.5),  # interval را افزایش دادیم
                    'cores': psutil.cpu_count(),
                    'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                'memory': {
                    'percent': memory.percent,
                    'used_gb': round(memory.used / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'total_gb': round(memory.total / (1024**3), 2)
                },
                'disk': {
                    'usage_percent': disk.percent,
                    'used_gb': round(disk.used / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'total_gb': round(disk.total / (1024**3), 2)
                },
                'network': {
                    'bytes_sent_mb': round(net_io.bytes_sent / (1024**2), 2),
                    'bytes_recv_mb': round(net_io.bytes_recv / (1024**2), 2)
                }
            }
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return self._get_fallback_metrics()
    
    def _collect_specialized_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های تخصصی از دیگر سیستم‌ها"""
        specialized = {
            'worker': {},
            'scheduler': {},
            'recovery': {},
            'dashboard': {}
        }
        
        # این بخش بعداً با اتصال سیستم‌ها پر می‌شود
        return specialized
    
    def _check_and_trigger_alerts(self, metrics: Dict):
        """بررسی و ایجاد هشدارهای متمرکز"""
        cpu_usage = metrics['system']['cpu']['percent']
        memory_usage = metrics['system']['memory']['percent']
        
        # بررسی CPU
        self._check_cpu_alerts(cpu_usage, metrics)
        
        # بررسی Memory
        self._check_memory_alerts(memory_usage, metrics)
        
        # بررسی Disk
        disk_usage = metrics['system']['disk']['usage_percent']
        if disk_usage > 90:
            self._trigger_alert('critical', 'disk', f"Disk usage critically high: {disk_usage}%", metrics)
    
    def _check_cpu_alerts(self, cpu_usage: float, metrics: Dict):
        """بررسی هشدارهای CPU با cooldown"""
        alert_key = f"cpu_{int(cpu_usage // 10)}"  # گروه‌بندی ۱۰٪ی
        
        # بررسی cooldown
        if self._is_in_cooldown(alert_key):
            return
        
        # بررسی سطوح
        if cpu_usage > 90:
            self._trigger_alert('critical', 'cpu', f"CPU usage critically high: {cpu_usage}%", metrics)
            self._set_cooldown(alert_key, 30)  # 30 ثانیه cooldown برای critical
        elif cpu_usage > 80:
            self._trigger_alert('warning', 'cpu', f"CPU usage high: {cpu_usage}%", metrics)
            self._set_cooldown(alert_key, 60)  # 60 ثانیه cooldown برای warning
        elif cpu_usage > 70:
            # فقط لاگ، هشدار نمی‌دهیم
            logger.info(f"📊 CPU usage elevated: {cpu_usage}%")
    
    def _check_memory_alerts(self, memory_usage: float, metrics: Dict):
        """بررسی هشدارهای Memory"""
        if memory_usage > 90:
            self._trigger_alert('critical', 'memory', f"Memory usage critically high: {memory_usage}%", metrics)
        elif memory_usage > 85:
            self._trigger_alert('warning', 'memory', f"Memory usage high: {memory_usage}%", metrics)
    
    def _trigger_alert(self, level: str, category: str, message: str, metrics: Dict):
        """ایجاد هشدار متمرکز"""
        try:
            # Import مستقیم Enumها
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            level_enum = AlertLevel.CRITICAL if level == 'critical' else AlertLevel.WARNING
            
            self.alert_manager.create_alert(
                level=level_enum,
                alert_type=AlertType.SYSTEM,
                title=f"High {category.title()} Usage",
                message=message,
                source="central_monitor",
                data={
                    'usage_percent': metrics['system'][category]['percent'] if category in metrics['system'] else 0,
                    'threshold': 90 if level == 'critical' else 80,
                    'timestamp': metrics['timestamp']
                }
            )
            logger.warning(f"🚨 {level.upper()} ALERT ({category}): {message}")
            
        except Exception as e:
            logger.error(f"❌ Error triggering alert: {e}")
    
    def _is_in_cooldown(self, alert_key: str) -> bool:
        """بررسی آیا هشدار در cooldown است"""
        last_alert = self.alert_cooldown.get(alert_key)
        if not last_alert:
            return False
        
        time_since_last = (datetime.now() - last_alert).total_seconds()
        return time_since_last < self.cooldown_period
    
    def _set_cooldown(self, alert_key: str, seconds: int = 60):
        """تنظیم cooldown برای هشدار"""
        self.alert_cooldown[alert_key] = datetime.now()
    
    def _calculate_smart_sleep(self, metrics: Dict, execution_time: float) -> int:
        """محاسبه خواب هوشمند بر اساس بار سیستم"""
        base_interval = self.collection_interval
        
        cpu_usage = metrics['system']['cpu']['percent']
        
        # اگر CPU بالا است، interval را افزایش بده
        if cpu_usage > 80:
            return min(base_interval * 2, 120)  # حداکثر ۲ دقیقه
        elif cpu_usage > 60:
            return min(base_interval * 1.5, 90)  # حداکثر ۱.۵ دقیقه
        
        # اگر جمع‌آوری طول کشید، کمی بیشتر صبر کن
        if execution_time > 5:
            return base_interval + 10
        
        return base_interval
    
    def _notify_subscribers(self, metrics: Dict):
        """اطلاع‌رسانی به سیستم‌های مشترک"""
        for sub_name, callback in self.subscribers.items():
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"❌ Error notifying subscriber {sub_name}: {e}")
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """متریک‌های جایگزین در صورت خطا"""
        return {
            'cpu': {'percent': 0, 'cores': 1, 'load_avg': [0, 0, 0]},
            'memory': {'percent': 0, 'used_gb': 0, 'available_gb': 0, 'total_gb': 0},
            'disk': {'usage_percent': 0, 'used_gb': 0, 'free_gb': 0, 'total_gb': 0},
            'network': {'bytes_sent_mb': 0, 'bytes_recv_mb': 0}
        }
    
    # 📡 API برای دیگر سیستم‌ها
    
    def subscribe(self, name: str, callback: Callable):
        """عضویت سیستم در دریافت به‌روزرسانی‌ها"""
        self.subscribers[name] = callback
        logger.info(f"📡 {name} subscribed to central monitor")
    
    def unsubscribe(self, name: str):
        """لغو عضویت"""
        if name in self.subscribers:
            del self.subscribers[name]
            logger.info(f"📡 {name} unsubscribed from central monitor")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های فعلی (برای سیستم‌های دیگر)"""
        if not self.metrics_cache:
            return self._get_fallback_metrics()
        
        # اگر داده قدیمی است، یک جمع‌آوری سریع انجام بده
        if (self.last_collection_time and 
            (datetime.now() - self.last_collection_time).total_seconds() > self.cache_ttl):
            logger.debug("📊 Cache expired, collecting fresh metrics")
            return self._collect_all_metrics_once()
        
        return self.metrics_cache
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """دریافت snapshot فعلی"""
        return {
            'cache_age_seconds': (
                (datetime.now() - self.last_collection_time).total_seconds() 
                if self.last_collection_time else None
            ),
            'subscribers_count': len(self.subscribers),
            'is_monitoring': self.is_monitoring,
            'last_alert_cooldowns': self.alert_cooldown,
            'metrics': self.get_current_metrics()
        }


# ایجاد نمونه گلوبال ارتقا یافته
system_monitor = None
central_monitor = None  # نمونه جدید متمرکز

def initialize_central_monitoring(metrics_collector, alert_manager):
    """تابع راه‌اندازی برای main.py"""
    global central_monitor
    central_monitor = CentralMonitoringSystem(metrics_collector, alert_manager)
    return central_monitor

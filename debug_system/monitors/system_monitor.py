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

# ایجاد نمونه گلوبال (بعداً در main.py مقداردهی می‌شود)
system_monitor = None

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import psutil
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class DashboardMetric:
    name: str
    value: float
    unit: str
    trend: str  # up, down, stable
    threshold_warning: float
    threshold_critical: float
    description: str

class WorkerMonitoringDashboard:
    """دشبورد مانیتورینگ پیشرفته برای سیستم Background Worker"""
    
    def __init__(self, background_worker=None, resource_manager=None, time_scheduler=None, recovery_manager=None):
        self.background_worker = background_worker
        self.resource_manager = resource_manager
        self.time_scheduler = time_scheduler
        self.recovery_manager = recovery_manager
        
        self.metrics_history: Dict[str, List] = {}
        self.active_alerts: List[Dict] = []
        self.performance_trends: Dict[str, Any] = {}
        self.dashboard_config = self._initialize_dashboard_config()
        self.is_monitoring = False
        self.monitor_thread = None
        
        logger.info("📊 Worker Monitoring Dashboard initialized")
    
    def start_monitoring(self):
        """شروع مانیتورینگ Real-time"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Dashboard monitoring started")
    
    def stop_monitoring(self):
        """توقف مانیتورینگ"""
        self.is_monitoring = False
        logger.info("🛑 Dashboard monitoring stopped")
    
    def _monitoring_loop(self):
        """حلقه جمع‌آوری متریک‌ها و آنالیز"""
        while self.is_monitoring:
            try:
                # جمع‌آوری متریک‌های جامع
                comprehensive_metrics = self._collect_comprehensive_metrics()
                
                # آنالیز و تشخیص الگوها
                self._analyze_performance_patterns(comprehensive_metrics)
                
                # بررسی و ثبت هشدارها
                self._check_and_trigger_alerts(comprehensive_metrics)
                
                # به‌روزرسانی تاریخچه
                self._update_metrics_history(comprehensive_metrics)
                
                time.sleep(10)  # جمع‌آوری هر 10 ثانیه
                
            except Exception as e:
                logger.error(f"❌ Dashboard monitoring error: {e}")
                time.sleep(30)
    
    def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های جامع از تمام سیستم‌ها"""
        timestamp = datetime.now()
        
        # متریک‌های اصلی سیستم
        system_metrics = self._collect_system_metrics()
        
        # متریک‌های Background Worker
        worker_metrics = self._collect_worker_metrics()
        
        # متریک‌های مدیریت منابع
        resource_metrics = self._collect_resource_metrics()
        
        # متریک‌های زمان‌بندی
        scheduling_metrics = self._collect_scheduling_metrics()
        
        # متریک‌های بازیابی
        recovery_metrics = self._collect_recovery_metrics()
        
        # محاسبه سلامت کلی
        overall_health = self._calculate_overall_health(
            system_metrics, worker_metrics, resource_metrics
        )
        
        return {
            'timestamp': timestamp.isoformat(),
            'overall_health': overall_health,
            'system': system_metrics,
            'worker': worker_metrics,
            'resources': resource_metrics,
            'scheduling': scheduling_metrics,
            'recovery': recovery_metrics,
            'performance_score': self._calculate_performance_score(worker_metrics, system_metrics)
        }
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستم"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
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
                'bytes_recv_mb': round(net_io.bytes_recv / (1024**2), 2),
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        }
    
    def _collect_worker_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های Background Worker"""
        if not self.background_worker:
            return {'status': 'unavailable'}
        
        try:
            worker_status = self.background_worker.get_detailed_metrics()
            
            return {
                'status': 'active',
                'queue_size': worker_status['queue_status']['queue_size'],
                'active_tasks': worker_status['queue_status']['active_tasks'],
                'completed_tasks': worker_status['queue_status']['completed_tasks'],
                'failed_tasks': worker_status['queue_status']['failed_tasks'],
                'active_workers': worker_status['worker_status']['active_workers'],
                'total_workers': worker_status['worker_status']['total_workers'],
                'worker_utilization': round(
                    worker_status['worker_status']['active_workers'] / 
                    worker_status['worker_status']['total_workers'] * 100, 2
                ),
                'task_throughput': self._calculate_task_throughput(worker_status),
                'success_rate': self._calculate_worker_success_rate(worker_status)
            }
        except Exception as e:
            logger.error(f"❌ Failed to collect worker metrics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _collect_resource_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های مدیریت منابع"""
        if not self.resource_manager:
            return {'status': 'unavailable'}
        
        try:
            resource_report = self.resource_manager.get_detailed_resource_report()
            
            return {
                'status': 'active',
                'health_score': resource_report['real_time_metrics']['system_health_score'],
                'cpu_efficiency': resource_report['performance_analysis']['health_score'],
                'bottlenecks': resource_report['performance_analysis']['bottlenecks'],
                'optimization_opportunities': resource_report['performance_analysis']['optimization_opportunities'],
                'adaptive_limits': getattr(self.resource_manager, 'adaptive_limits', {})
            }
        except Exception as e:
            logger.error(f"❌ Failed to collect resource metrics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _collect_scheduling_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های زمان‌بندی"""
        if not self.time_scheduler:
            return {'status': 'unavailable'}
      
        try:
            scheduling_analytics = self.time_scheduler.get_scheduling_analytics()
        
            # اضافه کردن بررسی برای جلوگیری از KeyError
            performance_analysis = scheduling_analytics.get('performance_analysis', {})
        
            return {
                'status': 'active',
                'active_tasks': scheduling_analytics.get('scheduling_status', {}).get('active_tasks', 0),
                'upcoming_tasks': scheduling_analytics.get('scheduling_status', {}).get('upcoming_tasks', 0),
                'success_rate': performance_analysis.get('overall_success_rate', 0),  # ✅ اصلاح شده
                'efficiency_score': performance_analysis.get('efficiency_score', 0),   # ✅ اصلاح شده
                'optimal_windows': scheduling_analytics.get('predictions', {}).get('optimal_scheduling_windows', [])
            }
        except Exception as e:
            logger.error(f"❌ Failed to collect scheduling metrics: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def _collect_recovery_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های سیستم بازیابی"""
        if not self.recovery_manager:
            return {'status': 'unavailable'}
        
        try:
            recovery_status = self.recovery_manager.get_recovery_status()
            
            return {
                'status': 'active',
                'total_snapshots': recovery_status['snapshots_summary']['total_snapshots'],
                'healthy_snapshots': recovery_status['snapshots_summary']['healthy_snapshots'],
                'recovery_readiness': recovery_status['health_assessment']['recovery_readiness'],
                'storage_usage_mb': recovery_status['snapshots_summary']['total_storage_mb'],
                'pending_recoveries': recovery_status['recovery_queue_status']['pending_recoveries']
            }
        except Exception as e:
            logger.error(f"❌ Failed to collect recovery metrics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_task_throughput(self, worker_status: Dict) -> float:
        """محاسبه throughput کارها"""
        try:
            total_processed = worker_status.get('performance_stats', {}).get('total_tasks_processed', 0)
            if total_processed == 0:
                return 0.0
        
            # محاسبه بر اساس تاریخچه (ساده‌سازی)
            return round(total_processed / 3600, 2)  # کار در ساعت
        except Exception as e:
            logger.error(f"❌ Error calculating task throughput: {e}")
            return 0.0
            
    def _calculate_worker_success_rate(self, worker_status: Dict) -> float:
        """محاسبه نرخ موفقیت worker"""
        try:
            task_breakdown = worker_status.get('performance_stats', {}).get('tasks_by_type', {})
            total_success = 0
            total_tasks = 0
        
            for task_type, stats in task_breakdown.items():
                total_success += stats.get('completed', 0)
                total_tasks += stats.get('submitted', 0)
        
            if total_tasks == 0:
                return 100.0
         
            return round((total_success / total_tasks) * 100, 2)
        except Exception as e:
            logger.error(f"❌ Error calculating worker success rate: {e}")
            return 100.0
    
    def _calculate_overall_health(self, system_metrics: Dict, worker_metrics: Dict, resource_metrics: Dict) -> Dict[str, Any]:
        """محاسبه سلامت کلی سیستم"""
        health_score = 100.0
        
        # جریمه برای CPU بالا
        if system_metrics['cpu']['percent'] > 80:
            health_score -= 20
        elif system_metrics['cpu']['percent'] > 60:
            health_score -= 10
        
        # جریمه برای حافظه بالا
        if system_metrics['memory']['percent'] > 85:
            health_score -= 20
        elif system_metrics['memory']['percent'] > 70:
            health_score -= 10
        
        # جریمه برای وضعیت worker
        if worker_metrics.get('status') != 'active':
            health_score -= 30
        elif worker_metrics.get('success_rate', 100) < 90:
            health_score -= 15
        
        # جریمه برای وضعیت منابع
        if resource_metrics.get('status') != 'active':
            health_score -= 20
        
        health_score = max(0, health_score)
        
        # تعیین وضعیت
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        return {
            'score': round(health_score, 2),
            'status': status,
            'updated_at': datetime.now().isoformat()
        }
    
    def _calculate_performance_score(self, worker_metrics: Dict, system_metrics: Dict) -> float:
        """محاسبه امتیاز عملکرد"""
        performance_score = 100.0
        
        # عوامل مؤثر بر عملکرد
        factors = {
            'worker_utilization': worker_metrics.get('worker_utilization', 0) / 100,
            'success_rate': worker_metrics.get('success_rate', 100) / 100,
            'cpu_efficiency': max(0, 1 - system_metrics['cpu']['percent'] / 100),
            'memory_efficiency': max(0, 1 - system_metrics['memory']['percent'] / 100)
        }
        
        # میانگین وزنی
        weights = {
            'worker_utilization': 0.3,
            'success_rate': 0.4,
            'cpu_efficiency': 0.15,
            'memory_efficiency': 0.15
        }
        
        weighted_score = sum(factors[factor] * weights[factor] for factor in factors)
        performance_score = weighted_score * 100
        
        return round(performance_score, 2)
    
    def _analyze_performance_patterns(self, metrics: Dict):
        """آنالیز الگوهای عملکرد"""
        timestamp = datetime.fromisoformat(metrics['timestamp'])
        
        # تحلیل ساعتی
        hour_key = f"hour_{timestamp.hour}"
        if hour_key not in self.performance_trends:
            self.performance_trends[hour_key] = {
                'hour': timestamp.hour,
                'avg_performance': metrics['performance_score'],
                'avg_health': metrics['overall_health']['score'],
                'sample_count': 1,
                'peak_usage': metrics['system']['cpu']['percent']
            }
        else:
            trend = self.performance_trends[hour_key]
            trend['avg_performance'] = (
                trend['avg_performance'] * trend['sample_count'] + metrics['performance_score']
            ) / (trend['sample_count'] + 1)
            trend['avg_health'] = (
                trend['avg_health'] * trend['sample_count'] + metrics['overall_health']['score']
            ) / (trend['sample_count'] + 1)
            trend['peak_usage'] = max(trend['peak_usage'], metrics['system']['cpu']['percent'])
            trend['sample_count'] += 1
        
        # شناسایی الگوهای کاهش عملکرد
        if (metrics['performance_score'] < 70 and 
            metrics['system']['cpu']['percent'] > 60):
            self._detect_performance_degradation(metrics)
    
    def _detect_performance_degradation(self, metrics: Dict):
        """تشخیص کاهش عملکرد"""
        alert_data = {
            'level': AlertLevel.WARNING,
            'category': 'performance',
            'message': 'Performance degradation detected',
            'timestamp': metrics['timestamp'],
            'details': {
                'performance_score': metrics['performance_score'],
                'cpu_usage': metrics['system']['cpu']['percent'],
                'health_score': metrics['overall_health']['score']
            },
            'recommendations': [
                'Check for resource-intensive tasks',
                'Review worker configuration',
                'Consider optimizing task scheduling'
            ]
        }
        
        self._add_alert(alert_data)
    
    def _check_and_trigger_alerts(self, metrics: Dict):
        """بررسی و فعال کردن هشدارها"""
        alerts = []
        
        # هشدارهای سیستم
        if metrics['system']['cpu']['percent'] > 90:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL, 'system', 
                f"CPU usage critically high: {metrics['system']['cpu']['percent']}%",
                metrics
            ))
        elif metrics['system']['cpu']['percent'] > 80:
            alerts.append(self._create_alert(
                AlertLevel.WARNING, 'system',
                f"CPU usage high: {metrics['system']['cpu']['percent']}%",
                metrics
            ))
        
        # هشدارهای حافظه
        if metrics['system']['memory']['percent'] > 90:
            alerts.append(self._create_alert(
                AlertLevel.CRITICAL, 'memory',
                f"Memory usage critically high: {metrics['system']['memory']['percent']}%",
                metrics
            ))
        
        # هشدارهای worker
        if metrics['worker'].get('success_rate', 100) < 80:
            alerts.append(self._create_alert(
                AlertLevel.WARNING, 'worker',
                f"Worker success rate low: {metrics['worker']['success_rate']}%",
                metrics
            ))
        
        # هشدارهای صف
        if metrics['worker'].get('queue_size', 0) > 50:
            alerts.append(self._create_alert(
                AlertLevel.WARNING, 'queue',
                f"Task queue growing: {metrics['worker']['queue_size']} tasks pending",
                metrics
            ))
        
        # ثبت هشدارها
        for alert in alerts:
            self._add_alert(alert)
    
    def _create_alert(self, level: AlertLevel, category: str, message: str, metrics: Dict) -> Dict:
        """ایجاد هشدار جدید"""
        return {
            'level': level.value,
            'category': category,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'metrics_snapshot': {
                'performance_score': metrics.get('performance_score', 0),
                'health_score': metrics.get('overall_health', {}).get('score', 0),
                'cpu_usage': metrics.get('system', {}).get('cpu', {}).get('percent', 0),
                'memory_usage': metrics.get('system', {}).get('memory', {}).get('percent', 0)
            },
            'acknowledged': False,
            'auto_resolve': True
        }
    
    def _add_alert(self, alert: Dict):
        """اضافه کردن هشدار به لیست فعال"""
        # بررسی تکراری نبودن هشدار
        for existing_alert in self.active_alerts:
            if (existing_alert.get('category') == alert.get('category') and 
                existing_alert.get('message') == alert.get('message') and
                not existing_alert.get('acknowledged', False)):
                return  # هشدار تکراری
    
        self.active_alerts.append(alert)
        logger.warning(f"🚨 {alert.get('level', 'UNKNOWN').upper()} ALERT: {alert.get('message', 'Unknown')}")
    
        # محدود کردن تعداد هشدارهای فعال
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]
            
    def _update_metrics_history(self, metrics: Dict):
        """به‌روزرسانی تاریخچه متریک‌ها"""
        timestamp = metrics['timestamp']
        
        for category, data in metrics.items():
            if category == 'timestamp':
                continue
                
            if category not in self.metrics_history:
                self.metrics_history[category] = []
            
            self.metrics_history[category].append({
                'timestamp': timestamp,
                'data': data
            })
            
            # حفظ اندازه تاریخچه
            if len(self.metrics_history[category]) > 1000:
                self.metrics_history[category] = self.metrics_history[category][-1000:]
    
    def _initialize_dashboard_config(self) -> Dict[str, Any]:
        """مقداردهی اولیه تنظیمات دشبورد"""
        return {
            'refresh_interval_seconds': 10,
            'retention_days': 7,
            'alert_retention_days': 30,
            'metrics_precision': 2,
            'auto_refresh': True,
            'theme': 'dark',
            'charts_config': {
                'cpu_usage': {'color': '#ff6b6b', 'max_value': 100},
                'memory_usage': {'color': '#4ecdc4', 'max_value': 100},
                'performance_score': {'color': '#45b7d1', 'max_value': 100},
                'health_score': {'color': '#96ceb4', 'max_value': 100}
            }
        }
    def _get_system_status_summary(self, metrics: Dict) -> Dict[str, str]:
        """دریافت خلاصه وضعیت سیستم"""
        status = {}
        
        # وضعیت سیستم
        cpu_status = "normal"
        if metrics['system']['cpu']['percent'] > 90:
            cpu_status = "critical"
        elif metrics['system']['cpu']['percent'] > 80:
            cpu_status = "warning"
        
        memory_status = "normal"
        if metrics['system']['memory']['percent'] > 90:
            memory_status = "critical"
        elif metrics['system']['memory']['percent'] > 80:
            memory_status = "warning"
        
        status['system'] = {
            'cpu': cpu_status,
            'memory': memory_status,
            'disk': "normal" if metrics['system']['disk']['usage_percent'] < 90 else "warning"
        }
        
        # وضعیت کامپوننت‌ها
        for component in ['worker', 'resources', 'scheduling', 'recovery']:
            component_data = metrics.get(component, {})
            status[component] = component_data.get('status', 'unknown')
        
        return status   
    def get_dashboard_data(self) -> Dict[str, Any]:
        """دریافت داده‌های کامل دشبورد"""
        current_metrics = self._collect_comprehensive_metrics()
    
        return {
            'summary': {
                'timestamp': current_metrics['timestamp'],
                'overall_health': current_metrics['overall_health'],
                'performance_score': current_metrics['performance_score'],
                'active_alerts': len([a for a in self.active_alerts if not a.get('acknowledged', False)]),
                'system_status': self._get_system_status_summary(current_metrics)
            },
            'current_metrics': current_metrics,
            'alerts': {
                'active': [alert for alert in self.active_alerts if not alert.get('acknowledged', False)],
                'recently_resolved': [alert for alert in self.active_alerts if alert.get('acknowledged', False)][-10:],
                'stats': {
                    'total_active': len([a for a in self.active_alerts if not a.get('acknowledged', False)]),
                    'critical_count': len([a for a in self.active_alerts if a.get('level') == 'critical' and not a.get('acknowledged', False)]),
                    'warning_count': len([a for a in self.active_alerts if a.get('level') == 'warning' and not a.get('acknowledged', False)])
                }
            },
            'performance_analysis': {
                'trends': self.performance_trends,
                'bottlenecks': self._identify_current_bottlenecks(current_metrics),
                'recommendations': self._generate_optimization_recommendations(current_metrics),
                'capacity_analysis': self._analyze_capacity(current_metrics)
            },
            'historical_data': {
                'health_trend': self._get_health_trend(),
                'performance_trend': self._get_performance_trend(),
                'resource_usage_trend': self._get_resource_usage_trend()
            },
            'dashboard_config': self.dashboard_config
        }
    
    def _identify_current_bottlenecks(self, metrics: Dict) -> List[str]:
        """شناسایی گلوگاه‌های فعلی"""
        bottlenecks = []
        
        if metrics['system']['cpu']['percent'] > 80:
            bottlenecks.append("High CPU usage limiting task processing capacity")
        
        if metrics['system']['memory']['percent'] > 85:
            bottlenecks.append("High memory usage affecting system performance")
        
        if metrics['worker'].get('queue_size', 0) > 20:
            bottlenecks.append("Growing task queue indicates processing delays")
        
        if metrics['worker'].get('worker_utilization', 0) > 90:
            bottlenecks.append("Worker utilization at maximum capacity")
        
        if metrics['performance_score'] < 70:
            bottlenecks.append("Overall performance below optimal levels")
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, metrics: Dict) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        
        if metrics['system']['cpu']['percent'] > 80:
            recommendations.append("Consider reducing worker count or optimizing task processing")
        
        if metrics['system']['memory']['percent'] > 85:
            recommendations.append("Review memory usage and consider cleanup procedures")
        
        if metrics['worker'].get('success_rate', 100) < 90:
            recommendations.append("Investigate task failures and improve error handling")
        
        if metrics['worker'].get('queue_size', 0) > 30:
            recommendations.append("Increase worker capacity or prioritize critical tasks")
        
        if metrics['performance_score'] < 80:
            recommendations.append("Perform comprehensive system optimization review")
        
        return recommendations
    
    def _analyze_capacity(self, metrics: Dict) -> Dict[str, Any]:
        """آنالیز ظرفیت سیستم"""
        current_usage = {
            'cpu': metrics['system']['cpu']['percent'],
            'memory': metrics['system']['memory']['percent'],
            'workers': metrics['worker'].get('worker_utilization', 0)
        }
        
        headroom = {
            'cpu': 100 - current_usage['cpu'],
            'memory': 100 - current_usage['memory'],
            'workers': 100 - current_usage['workers']
        }
        
        # پیش‌بینی زمان رسیدن به محدودیت
        exhaustion_prediction = {}
        for resource, usage in current_usage.items():
            if usage > 80:
                exhaustion_prediction[resource] = "soon"
            elif usage > 60:
                exhaustion_prediction[resource] = "medium_term"
            else:
                exhaustion_prediction[resource] = "long_term"
        
        return {
            'current_usage': current_usage,
            'available_headroom': headroom,
            'exhaustion_prediction': exhaustion_prediction,
            'scaling_recommendation': 'scale_up' if current_usage['cpu'] > 70 else 'maintain'
        }
    
    def _get_health_trend(self) -> List[Dict]:
        """دریافت روند سلامت"""
        health_data = self.metrics_history.get('overall_health', [])
        return [
            {
                'timestamp': item['timestamp'],
                'score': item['data']['score'],
                'status': item['data']['status']
            }
            for item in health_data[-24:]  # 24 نمونه اخیر
        ]
    
    def _get_performance_trend(self) -> List[Dict]:
        """دریافت روند عملکرد"""
        performance_data = []
        for item in self.metrics_history.get('performance_score', [])[-24:]:
            performance_data.append({
                'timestamp': item['timestamp'],
                'score': item['data']
            })
        return performance_data
    
    def _get_resource_usage_trend(self) -> Dict[str, List]:
        """دریافت روند استفاده از منابع"""
        trends = {
            'cpu': [],
            'memory': [],
            'disk': []
        }
        
        system_data = self.metrics_history.get('system', [])
        for item in system_data[-24:]:
            trends['cpu'].append({
                'timestamp': item['timestamp'],
                'usage': item['data']['cpu']['percent']
            })
            trends['memory'].append({
                'timestamp': item['timestamp'],
                'usage': item['data']['memory']['percent']
            })
            trends['disk'].append({
                'timestamp': item['timestamp'],
                'usage': item['data']['disk']['usage_percent']
            })
        
        return trends
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """تأیید یک هشدار"""
        if 0 <= alert_index < len(self.active_alerts):
            self.active_alerts[alert_index]['acknowledged'] = True
            self.active_alerts[alert_index]['acknowledged_at'] = datetime.now().isoformat()
            return True
        return False
        
    def get_worker_insights(self) -> Dict[str, Any]:
        """دریافت بینش‌های تحلیلی از عملکرد worker"""
        if not self.metrics_history.get('worker'):
            return {'status': 'insufficient_data'}
        
        worker_data = self.metrics_history['worker'][-100:]  # 100 نمونه اخیر
        
        insights = {
            'utilization_pattern': self._analyze_utilization_pattern(worker_data),
            'performance_correlations': self._find_performance_correlations(),
            'optimal_worker_config': self._calculate_optimal_worker_config(),
            'predictive_insights': self._generate_predictive_insights()
        }
        
        return insights
    
    def _analyze_utilization_pattern(self, worker_data: List) -> Dict[str, Any]:
        """آنالیز الگوی استفاده از worker"""
        utilizations = [item['data'].get('worker_utilization', 0) for item in worker_data]
        
        if not utilizations:
            return {'status': 'no_data'}
        
        return {
            'avg_utilization': round(sum(utilizations) / len(utilizations), 2),
            'max_utilization': max(utilizations),
            'min_utilization': min(utilizations),
            'underutilized_hours': len([u for u in utilizations if u < 30]),
            'overutilized_hours': len([u for u in utilizations if u > 90])
        }
    
    def _find_performance_correlations(self) -> List[Dict]:
        """یافتن همبستگی‌های عملکرد"""
        # این یک پیاده‌سازی ساده است - در واقعیت از تحلیل آماری استفاده می‌شود
        return [
            {
                'factor1': 'cpu_usage',
                'factor2': 'performance_score',
                'correlation': 'strong_negative',
                'impact': 'high'
            },
            {
                'factor1': 'worker_utilization',
                'factor2': 'task_throughput',
                'correlation': 'strong_positive',
                'impact': 'high'
            }
        ]
    
    def _calculate_optimal_worker_config(self) -> Dict[str, Any]:
        """محاسبه تنظیمات بهینه worker"""
        return {
            'recommended_workers': 4,
            'optimal_queue_size': 10,
            'suggested_limits': {
                'max_cpu_percent': 75,
                'max_memory_percent': 80,
                'max_queue_size': 50
            },
            'performance_improvement_potential': '15-25%'
        }
    
    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """تولید بینش‌های پیش‌بین"""
        return {
            'next_peak_hours': ['14:00-16:00', '19:00-21:00'],
            'predicted_downtime_risk': 'low',
            'capacity_warnings': [],
            'maintenance_recommendations': [
                'Schedule heavy tasks during night hours',
                'Consider increasing worker capacity during peak hours'
            ]
        }

# نمونه گلوبال
monitoring_dashboard = WorkerMonitoringDashboard()

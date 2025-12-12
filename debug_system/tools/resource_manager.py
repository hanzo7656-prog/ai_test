import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import psutil
import os
import json

logger = logging.getLogger(__name__)

class ResourceGuardian:
    """نگهبان هوشمند منابع - مدیریت و بهینه‌سازی مصرف منابع"""
    
    def __init__(self, max_cpu_percent: float = 70.0, max_memory_percent: float = 80.0):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.resource_history: List[Dict] = []
        self.optimization_strategies = {}
        self.performance_baseline = {}
        self.adaptive_limits = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        # متریک‌های تاریخی
        self.historical_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'peak_times': [],
            'quiet_times': []
        }
        
        # استراتژی‌های بهینه‌سازی
        self._initialize_optimization_strategies()
        self._establish_performance_baseline()
        
        # عضویت در سیستم مانیتورینگ مرکزی
        self._subscribe_to_central_monitor()
        
        logger.info("🛡️ Resource Guardian initialized (Connected to Central Monitor)")
    
    def _subscribe_to_central_monitor(self):
        """عضویت در سیستم مانیتورینگ مرکزی"""
        import time
    
        logger.info("🔌 Resource Guardian connecting to Central Monitor...")
    
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                from debug_system.monitors.system_monitor import central_monitor
                if central_monitor and hasattr(central_monitor, 'subscribe'):
                    central_monitor.subscribe("resource_guardian", self._on_central_metrics_update)
                    logger.info(f"✅✅ Resource Guardian SUCCESSFULLY subscribed to Central Monitor (attempt {attempt + 1})")
                    return
                else:
                    logger.debug(f"⏳ Central monitor not ready for Resource Guardian (attempt {attempt + 1}/{max_attempts})")
            except ImportError:
                logger.debug(f"⏳ Waiting for central_monitor module (attempt {attempt + 1}/{max_attempts})")
        
            time.sleep(3)  # افزایش زمان انتظار
    
        logger.warning("⚠️ Resource Guardian could not connect to Central Monitor after waiting")
        logger.info("🔄 Resource Guardian will work in independent mode")
        
    def _on_central_metrics_update(self, metrics: Dict):
        """دریافت به‌روزرسانی متریک از سیستم مرکزی"""
        try:
            system_metrics = metrics.get('system', {})
            
            # ذخیره در تاریخچه
            self._update_history_from_central(system_metrics)
            
            # آنالیز الگوهای مصرف
            self._analyze_resource_patterns_central(system_metrics)
            
            # به‌روزرسانی محدودیت‌های تطبیقی
            self._update_adaptive_limits_central(system_metrics)
            
            # بررسی ناهنجاری‌ها
            self._check_resource_anomalies_central(system_metrics)
            
        except Exception as e:
            logger.error(f"❌ Error processing central metrics: {e}")
    
    def _update_history_from_central(self, metrics: Dict):
        """به‌روزرسانی تاریخچه از داده‌های مرکزی"""
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': metrics.get('cpu', {}).get('percent', 0)
            },
            'memory': {
                'percent': metrics.get('memory', {}).get('percent', 0)
            },
            'disk': {
                'usage_percent': metrics.get('disk', {}).get('usage_percent', 0)
            },
            'network': {
                'bytes_sent_mb': metrics.get('network', {}).get('bytes_sent_mb', 0),
                'bytes_recv_mb': metrics.get('network', {}).get('bytes_recv_mb', 0)
            }
        }
        
        self.resource_history.append(current_metrics)
        
        # حفظ اندازه تاریخچه
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
    
    def _analyze_resource_patterns_central(self, metrics: Dict):
        """آنالیز الگوهای مصرف از داده‌های مرکزی"""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        
        # الگوی ساعتی
        hour_key = f"hour_{current_hour}"
        hour_pattern = next((p for p in self.historical_metrics['peak_times'] 
                           if p.get('hour') == current_hour), None)
        
        if not hour_pattern:
            self.historical_metrics['peak_times'].append({
                'hour': current_hour,
                'avg_cpu': cpu_percent,
                'avg_memory': memory_percent,
                'sample_count': 1
            })
        else:
            hour_pattern['avg_cpu'] = (
                hour_pattern['avg_cpu'] * hour_pattern['sample_count'] + cpu_percent
            ) / (hour_pattern['sample_count'] + 1)
            hour_pattern['avg_memory'] = (
                hour_pattern['avg_memory'] * hour_pattern['sample_count'] + memory_percent
            ) / (hour_pattern['sample_count'] + 1)
            hour_pattern['sample_count'] += 1
        
        # شناسایی زمان‌های خلوت
        if cpu_percent < 30 and memory_percent < 50:
            self.historical_metrics['quiet_times'].append({
                'timestamp': datetime.now().isoformat(),
                'cpu': cpu_percent,
                'memory': memory_percent,
                'hour': current_hour,
                'day': current_day
            })
    
    def _update_adaptive_limits_central(self, metrics: Dict):
        """به‌روزرسانی محدودیت‌های تطبیقی از داده‌های مرکزی"""
        current_hour = datetime.now().hour
        
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        
        # تنظیم محدودیت‌ها بر اساس زمان روز و وضعیت فعلی
        if cpu_percent > 80 or memory_percent > 85:
            # حالت بحرانی
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent * 0.5,
                'max_memory': self.max_memory_percent * 0.5,
                'max_workers': 1,
                'quality_mode': 'conservative',
                'status': 'critical'
            }
        elif 1 <= current_hour <= 7:  # شب
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent * 0.9,
                'max_memory': self.max_memory_percent * 0.8,
                'max_workers': 2,
                'quality_mode': 'balanced',
                'status': 'night_mode'
            }
        elif current_hour in [10, 11, 14, 15, 19, 20]:  # ساعات اوج
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent * 0.7,
                'max_memory': self.max_memory_percent * 0.6,
                'max_workers': 1,
                'quality_mode': 'conservative',
                'status': 'peak_hours'
            }
        else:  # زمان عادی
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent,
                'max_memory': self.max_memory_percent,
                'max_workers': 3,
                'quality_mode': 'standard',
                'status': 'normal'
            }
    
    def _check_resource_anomalies_central(self, metrics: Dict):
        """بررسی ناهنجاری‌های منابع از داده‌های مرکزی"""
        anomalies = []
        
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        disk_percent = metrics.get('disk', {}).get('usage_percent', 0)
        
        # بررسی افزایش ناگهانی CPU
        if len(self.resource_history) >= 2:
            prev_cpu = self.resource_history[-2]['cpu']['percent']
            if cpu_percent > prev_cpu * 2 and cpu_percent > 50:  # افزایش 100% و بالای 50%
                anomalies.append(f"CPU spike detected: {prev_cpu}% -> {cpu_percent}%")
        
        # بررسی نشت حافظه
        if memory_percent > 85:
            anomalies.append(f"High memory usage: {memory_percent}%")
        
        # بررسی دیسک پر
        if disk_percent > 90:
            anomalies.append(f"Disk almost full: {disk_percent}%")
        
        # ثبت هشدار برای ناهنجاری‌ها
        for anomaly in anomalies:
            logger.warning(f"🚨 Resource anomaly: {anomaly}")
    
    def _calculate_system_health_score(self, cpu_percent: float, memory_percent: float) -> float:
        """محاسبه امتیاز سلامت سیستم"""
        # نرمال‌سازی مقادیر
        cpu_score = max(0, 100 - cpu_percent)
        memory_score = max(0, 100 - memory_percent)
        
        # وزن‌دهی (CPU مهم‌تر است)
        health_score = (cpu_score * 0.6 + memory_score * 0.4)
        
        # جریمه برای مقادیر بحرانی
        if cpu_percent > 90:
            health_score *= 0.7
        if memory_percent > 90:
            health_score *= 0.8
        
        return round(health_score, 2)
    
    def _initialize_optimization_strategies(self):
        """مقداردهی اولیه استراتژی‌های بهینه‌سازی"""
        self.optimization_strategies = {
            'cpu_optimization': {
                'low_usage': self._optimize_for_low_cpu,
                'high_usage': self._optimize_for_high_cpu,
                'critical_usage': self._survival_mode_cpu
            },
            'memory_optimization': {
                'low_usage': self._optimize_for_low_memory,
                'high_usage': self._optimize_for_high_memory,
                'critical_usage': self._survival_mode_memory
            },
            'disk_optimization': {
                'cleanup_threshold': 85,
                'compression_threshold': 90
            },
            'network_optimization': {
                'compression_enabled': True,
                'batch_size_optimization': True
            }
        }
    
    def _establish_performance_baseline(self):
        """ایجاد خط پایه عملکرد"""
        # خط پایه بر اساس مقادیر پیش‌فرض
        self.performance_baseline = {
            'avg_cpu': 20.0,
            'avg_memory': 40.0,
            'established_at': datetime.now().isoformat(),
            'sample_count': 0
        }
        
        logger.info("📈 Performance baseline established (will update with real data)")
    
    def _optimize_for_low_cpu(self, current_cpu: float) -> Dict[str, Any]:
        """بهینه‌سازی برای CPU پایین"""
        return {
            'action': 'increase_throughput',
            'max_workers': 4,
            'task_priority': 'all',
            'quality_level': 'high',
            'compression': 'enabled'
        }
    
    def _optimize_for_high_cpu(self, current_cpu: float) -> Dict[str, Any]:
        """بهینه‌سازی برای CPU بالا"""
        return {
            'action': 'reduce_load',
            'max_workers': 2,
            'task_priority': 'normal_and_high',
            'quality_level': 'medium',
            'compression': 'enabled'
        }
    
    def _survival_mode_cpu(self, current_cpu: float) -> Dict[str, Any]:
        """حالت بقا برای CPU بحرانی"""
        return {
            'action': 'critical_reduction',
            'max_workers': 1,
            'task_priority': 'high_only',
            'quality_level': 'low',
            'compression': 'aggressive',
            'delay_non_essential': True
        }
    
    def _optimize_for_low_memory(self, current_memory: float) -> Dict[str, Any]:
        """بهینه‌سازی برای حافظه پایین"""
        return {
            'action': 'normal_operations',
            'cache_size': 'large',
            'buffer_pool': 'enabled',
            'garbage_collection': 'standard'
        }
    
    def _optimize_for_high_memory(self, current_memory: float) -> Dict[str, Any]:
        """بهینه‌سازی برای حافظه بالا"""
        return {
            'action': 'reduce_memory_footprint',
            'cache_size': 'medium',
            'buffer_pool': 'reduced',
            'garbage_collection': 'aggressive'
        }
    
    def _survival_mode_memory(self, current_memory: float) -> Dict[str, Any]:
        """حالت بقا برای حافظه بحرانی"""
        return {
            'action': 'emergency_cleanup',
            'cache_size': 'minimal',
            'buffer_pool': 'disabled',
            'garbage_collection': 'forced',
            'clear_temporary_data': True
        }
    
    def get_current_metrics_from_history(self) -> Dict[str, Any]:
        """دریافت آخرین متریک‌های موجود در تاریخچه"""
        if not self.resource_history:
            # برگرداندن مقادیر پیش‌فرض اگر تاریخچه خالی است
            return {
                'cpu': {'percent': 0},
                'memory': {'percent': 0},
                'disk': {'usage_percent': 0},
                'timestamp': datetime.now().isoformat()
            }
        
        return self.resource_history[-1]
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """دریافت توصیه‌های بهینه‌سازی"""
        current_metrics = self.get_current_metrics_from_history()
        cpu_percent = current_metrics['cpu']['percent']
        memory_percent = current_metrics['memory']['percent']
        
        # انتخاب استراتژی بر اساس شرایط فعلی
        if cpu_percent > 90:
            cpu_strategy = self.optimization_strategies['cpu_optimization']['critical_usage'](cpu_percent)
        elif cpu_percent > 70:
            cpu_strategy = self.optimization_strategies['cpu_optimization']['high_usage'](cpu_percent)
        else:
            cpu_strategy = self.optimization_strategies['cpu_optimization']['low_usage'](cpu_percent)
        
        if memory_percent > 90:
            memory_strategy = self.optimization_strategies['memory_optimization']['critical_usage'](memory_percent)
        elif memory_percent > 75:
            memory_strategy = self.optimization_strategies['memory_optimization']['high_usage'](memory_percent)
        else:
            memory_strategy = self.optimization_strategies['memory_optimization']['low_usage'](memory_percent)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'health_score': self._calculate_system_health_score(cpu_percent, memory_percent)
            },
            'recommendations': {
                'cpu_optimization': cpu_strategy,
                'memory_optimization': memory_strategy,
                'adaptive_limits': self.adaptive_limits,
                'predicted_peak_times': self._predict_peak_times()
            },
            'resource_patterns': {
                'peak_hours': self.historical_metrics['peak_times'][-6:],  # 6 ساعت اخیر
                'quiet_periods': self.historical_metrics['quiet_times'][-10:]  # 10 دوره اخیر
            }
        }
    
    def _predict_peak_times(self) -> List[Dict]:
        """پیش‌بینی زمان‌های اوج مصرف"""
        predictions = []
        current_hour = datetime.now().hour
        
        for hour_data in self.historical_metrics['peak_times'][-24:]:  # 24 ساعت اخیر
            if hour_data['avg_cpu'] > 60 or hour_data['avg_memory'] > 70:
                predictions.append({
                    'hour': hour_data['hour'],
                    'probability': 'high' if hour_data['avg_cpu'] > 75 else 'medium',
                    'expected_cpu': hour_data['avg_cpu'],
                    'expected_memory': hour_data['avg_memory']
                })
        
        return predictions
    
    def get_detailed_resource_report(self) -> Dict[str, Any]:
        """دریافت گزارش دقیق منابع"""
        current_metrics = self.get_current_metrics_from_history()
        
        return {
            'real_time_metrics': current_metrics,
            'historical_trends': {
                'cpu_trend': self._calculate_trend('cpu', 'percent'),
                'memory_trend': self._calculate_trend('memory', 'percent'),
                'disk_trend': self._calculate_trend('disk', 'usage_percent')
            },
            'performance_analysis': {
                'health_score': self._calculate_system_health_score(
                    current_metrics['cpu']['percent'], 
                    current_metrics['memory']['percent']
                ),
                'bottlenecks': self._identify_bottlenecks(current_metrics),
                'optimization_opportunities': self._find_optimization_opportunities(current_metrics),
                'capacity_planning': self._capacity_planning_analysis()
            },
            'resource_guardian_status': {
                'is_monitoring': self.is_monitoring,
                'adaptive_limits_active': bool(self.adaptive_limits),
                'optimization_strategies_loaded': len(self.optimization_strategies) > 0,
                'historical_data_points': len(self.resource_history),
                'connected_to_central_monitor': hasattr(self, '_on_central_metrics_update')
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_trend(self, resource: str, metric: str) -> str:
        """محاسبه روند تغییرات منابع"""
        if len(self.resource_history) < 2:
            return "insufficient_data"
        
        recent = self.resource_history[-1][resource][metric]
        previous = self.resource_history[-2][resource][metric]
        
        if recent > previous * 1.1:
            return "increasing"
        elif recent < previous * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_bottlenecks(self, metrics: Dict) -> List[str]:
        """شناسایی گلوگاه‌های منابع"""
        bottlenecks = []
        
        if metrics['cpu']['percent'] > 80:
            bottlenecks.append("High CPU usage - consider optimizing algorithms or scaling")
        
        if metrics['memory']['percent'] > 85:
            bottlenecks.append("High memory usage - review caching strategy and data structures")
        
        if metrics.get('disk', {}).get('usage_percent', 0) > 90:
            bottlenecks.append("Disk space critical - implement cleanup procedures")
        
        return bottlenecks
    
    def _find_optimization_opportunities(self, metrics: Dict) -> List[str]:
        """یافتن فرصت‌های بهینه‌سازی"""
        opportunities = []
        
        if metrics['cpu']['percent'] < 30 and metrics['memory']['percent'] < 50:
            opportunities.append("Resources underutilized - can increase task throughput")
        
        if metrics.get('disk', {}).get('usage_percent', 0) < 50:
            opportunities.append("Disk space available - can enable more caching")
        
        return opportunities
    
    def _capacity_planning_analysis(self) -> Dict[str, Any]:
        """آنالیز برنامه‌ریزی ظرفیت"""
        if len(self.resource_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_cpu = [m['cpu']['percent'] for m in self.resource_history[-10:]]
        recent_memory = [m['memory']['percent'] for m in self.resource_history[-10:]]
        
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        return {
            'current_utilization': {
                'cpu': avg_cpu,
                'memory': avg_memory
            },
            'headroom': {
                'cpu': 100 - avg_cpu,
                'memory': 100 - avg_memory
            },
            'scaling_recommendation': 'scale_up' if avg_cpu > 70 else 'scale_down' if avg_cpu < 30 else 'maintain',
            'projected_exhaustion_days': self._calculate_exhaustion_timeline(avg_cpu, avg_memory)
        }
    
    def _calculate_exhaustion_timeline(self, avg_cpu: float, avg_memory: float) -> Dict[str, Any]:
        """محاسبه زمان تخمینی اتمام منابع"""
        if avg_cpu > 90:
            cpu_timeline = "imminent"
        elif avg_cpu > 70:
            cpu_timeline = "30_days"
        else:
            cpu_timeline = "90_plus_days"
        
        if avg_memory > 90:
            memory_timeline = "imminent"
        elif avg_memory > 70:
            memory_timeline = "45_days"
        else:
            memory_timeline = "90_plus_days"
        
        return {
            'cpu': cpu_timeline,
            'memory': memory_timeline,
            'overall': memory_timeline if memory_timeline == "imminent" else cpu_timeline
        }
        
    def start_monitoring(self):
        """شروع مانیتورینگ منابع (پیاده‌سازی ساده)"""
        if not self.is_monitoring:
            self.is_monitoring = True
            logger.info("🛡️ Resource Guardian monitoring started")
        return self.is_monitoring

    def stop_monitoring(self):
        """توقف مانیتورینگ"""
        self.is_monitoring = False
        logger.info("🛡️ Resource Guardian monitoring stopped")
        return True

# نمونه گلوبال
resource_guardian = ResourceGuardian(max_cpu_percent=70.0, max_memory_percent=80.0)

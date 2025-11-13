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
        
        logger.info("🛡️ Resource Guardian initialized")
    
    def start_monitoring(self):
        """شروع مانیتورینگ Real-time منابع"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Resource monitoring started")
    
    def stop_monitoring(self):
        """توقف مانیتورینگ منابع"""
        self.is_monitoring = False
        logger.info("🛑 Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """حلقه مانیتورینگ منابع"""
        while self.is_monitoring:
            try:
                current_metrics = self._collect_comprehensive_metrics()
                self._analyze_resource_patterns(current_metrics)
                self._update_adaptive_limits(current_metrics)
                self._check_resource_anomalies(current_metrics)
                
                # ذخیره در تاریخچه
                self.resource_history.append(current_metrics)
                
                # حفظ اندازه تاریخچه
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]
                
                time.sleep(5)  # جمع‌آوری هر 5 ثانیه
                
            except Exception as e:
                logger.error(f"❌ Resource monitoring error: {e}")
                time.sleep(10)
    
    def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های جامع منابع"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_times = psutil.cpu_times()
        cpu_freq = psutil.cpu_freq()
        
        # Memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network
        net_io = psutil.net_io_counters()
        
        # Process-specific
        current_process = psutil.Process()
        process_memory = current_process.memory_info()
        process_cpu = current_process.cpu_percent()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'cores': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                'user_time': cpu_times.user,
                'system_time': cpu_times.system,
                'frequency_mhz': cpu_freq.current if cpu_freq else 0
            },
            'memory': {
                'percent': memory.percent,
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'total_gb': round(memory.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2)
            },
            'disk': {
                'usage_percent': disk_usage.percent,
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
            },
            'network': {
                'bytes_sent_mb': net_io.bytes_sent / (1024**2),
                'bytes_recv_mb': net_io.bytes_recv / (1024**2),
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'process': {
                'memory_rss_mb': process_memory.rss / (1024**2),
                'memory_vms_mb': process_memory.vms / (1024**2),
                'cpu_percent': process_cpu,
                'threads_count': current_process.num_threads(),
                'open_files': len(current_process.open_files())
            },
            'system_health_score': self._calculate_system_health_score(cpu_percent, memory.percent)
        }
    
    def _analyze_resource_patterns(self, metrics: Dict):
        """آنالیز الگوهای مصرف منابع"""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # الگوی ساعتی
        hour_key = f"hour_{current_hour}"
        if hour_key not in self.historical_metrics['peak_times']:
            self.historical_metrics['peak_times'].append({
                'hour': current_hour,
                'avg_cpu': metrics['cpu']['percent'],
                'avg_memory': metrics['memory']['percent'],
                'sample_count': 1
            })
        else:
            pattern = next(p for p in self.historical_metrics['peak_times'] if f"hour_{p['hour']}" == hour_key)
            pattern['avg_cpu'] = (pattern['avg_cpu'] * pattern['sample_count'] + metrics['cpu']['percent']) / (pattern['sample_count'] + 1)
            pattern['avg_memory'] = (pattern['avg_memory'] * pattern['sample_count'] + metrics['memory']['percent']) / (pattern['sample_count'] + 1)
            pattern['sample_count'] += 1
        
        # شناسایی زمان‌های خلوت
        if metrics['cpu']['percent'] < 30 and metrics['memory']['percent'] < 50:
            self.historical_metrics['quiet_times'].append({
                'timestamp': metrics['timestamp'],
                'cpu': metrics['cpu']['percent'],
                'memory': metrics['memory']['percent'],
                'hour': current_hour,
                'day': current_day
            })
    
    def _update_adaptive_limits(self, metrics: Dict):
        """به‌روزرسانی محدودیت‌های تطبیقی"""
        current_hour = datetime.now().hour
        
        # تنظیم محدودیت‌ها بر اساس زمان روز
        if 1 <= current_hour <= 7:  # شب
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent * 0.9,  # محدودتر در شب
                'max_memory': self.max_memory_percent * 0.8,
                'max_workers': 2,
                'quality_mode': 'balanced'
            }
        elif current_hour in [10, 11, 14, 15, 19, 20]:  # ساعات اوج
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent * 0.7,  # بسیار محدود
                'max_memory': self.max_memory_percent * 0.6,
                'max_workers': 1,
                'quality_mode': 'conservative'
            }
        else:  # زمان عادی
            self.adaptive_limits = {
                'max_cpu': self.max_cpu_percent,
                'max_memory': self.max_memory_percent,
                'max_workers': 3,
                'quality_mode': 'standard'
            }
    
    def _check_resource_anomalies(self, metrics: Dict):
        """بررسی ناهنجاری‌های منابع"""
        anomalies = []
        
        # بررسی افزایش ناگهانی CPU
        if len(self.resource_history) >= 2:
            prev_cpu = self.resource_history[-2]['cpu']['percent']
            current_cpu = metrics['cpu']['percent']
            if current_cpu > prev_cpu * 2 and current_cpu > 50:  # افزایش 100% و بالای 50%
                anomalies.append(f"CPU spike detected: {prev_cpu}% -> {current_cpu}%")
        
        # بررسی نشت حافظه
        if metrics['memory']['percent'] > 85:
            anomalies.append(f"High memory usage: {metrics['memory']['percent']}%")
        
        # بررسی دیسک پر
        if metrics['disk']['usage_percent'] > 90:
            anomalies.append(f"Disk almost full: {metrics['disk']['usage_percent']}%")
        
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
        # جمع‌آوری داده‌های اولیه برای 30 ثانیه
        baseline_metrics = []
        for _ in range(6):
            baseline_metrics.append(self._collect_comprehensive_metrics())
            time.sleep(5)
        
        # محاسبه میانگین‌ها
        avg_cpu = sum(m['cpu']['percent'] for m in baseline_metrics) / len(baseline_metrics)
        avg_memory = sum(m['memory']['percent'] for m in baseline_metrics) / len(baseline_metrics)
        
        self.performance_baseline = {
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'established_at': datetime.now().isoformat(),
            'sample_count': len(baseline_metrics)
        }
        
        logger.info(f"📈 Performance baseline established: CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%")
    
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
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """دریافت توصیه‌های بهینه‌سازی"""
        current_metrics = self._collect_comprehensive_metrics()
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
                'health_score': current_metrics['system_health_score']
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
        current_metrics = self._collect_comprehensive_metrics()
        
        return {
            'real_time_metrics': current_metrics,
            'historical_trends': {
                'cpu_trend': self._calculate_trend('cpu', 'percent'),
                'memory_trend': self._calculate_trend('memory', 'percent'),
                'disk_trend': self._calculate_trend('disk', 'usage_percent')
            },
            'performance_analysis': {
                'health_score': current_metrics['system_health_score'],
                'bottlenecks': self._identify_bottlenecks(current_metrics),
                'optimization_opportunities': self._find_optimization_opportunities(current_metrics),
                'capacity_planning': self._capacity_planning_analysis()
            },
            'resource_guardian_status': {
                'is_monitoring': self.is_monitoring,
                'adaptive_limits_active': bool(self.adaptive_limits),
                'optimization_strategies_loaded': len(self.optimization_strategies) > 0,
                'historical_data_points': len(self.resource_history)
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
        
        if metrics['disk']['usage_percent'] > 90:
            bottlenecks.append("Disk space critical - implement cleanup procedures")
        
        if metrics['process']['memory_rss_mb'] > 500:  # 500MB threshold
            bottlenecks.append("Process memory high - check for memory leaks")
        
        return bottlenecks
    
    def _find_optimization_opportunities(self, metrics: Dict) -> List[str]:
        """یافتن فرصت‌های بهینه‌سازی"""
        opportunities = []
        
        if metrics['cpu']['percent'] < 30 and metrics['memory']['percent'] < 50:
            opportunities.append("Resources underutilized - can increase task throughput")
        
        if metrics['disk']['usage_percent'] < 50:
            opportunities.append("Disk space available - can enable more caching")
        
        if metrics['network']['bytes_sent_mb'] > 100:  # 100MB sent
            opportunities.append("High network usage - consider compression or batching")
        
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
        # این یک محاسبه ساده است - در واقعیت باید بر اساس نرخ رشد باشد
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

# نمونه گلوبال
resource_guardian = ResourceGuardian(max_cpu_percent=70.0, max_memory_percent=80.0)

# ml_core/performance_tracker.py
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """ردیابی و آنالیز عملکرد مدل‌های هوش مصنوعی"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))  # ذخیره 1000 نمونه اخیر
        self.alert_thresholds = {
            'inference_time': 5.0,  # ثانیه
            'memory_usage': 0.9,    # 90%
            'error_rate': 0.05,     # 5%
            'confidence_drop': 0.2   # 20% کاهش
        }
        self.alerts = []
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("📊 Performance Tracker initialized")

    def track_inference(self, model_name: str, inference_time: float, 
                       confidence: float, success: bool, input_size: tuple):
        """ردیابی یک inference جدید"""
        timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp.isoformat(),
            'inference_time': inference_time,
            'confidence': confidence,
            'success': success,
            'input_size': input_size,
            'throughput': 1 / inference_time if inference_time > 0 else 0
        }
        
        # ذخیره در تاریخچه
        self.performance_history[model_name].append(metrics)
        
        # بررسی هشدارها
        self._check_alerts(model_name, metrics)
        
        # به‌روزرسانی متریک‌های تجمعی
        self._update_aggregate_metrics(model_name)
        
        logger.debug(f"📈 Tracked inference for {model_name}: {inference_time:.3f}s")

    def _check_alerts(self, model_name: str, metrics: Dict[str, Any]):
        """بررسی و ایجاد هشدار در صورت نیاز"""
        alerts_triggered = []
        
        # بررسی زمان inference
        if metrics['inference_time'] > self.alert_thresholds['inference_time']:
            alerts_triggered.append({
                'type': 'slow_inference',
                'model': model_name,
                'value': metrics['inference_time'],
                'threshold': self.alert_thresholds['inference_time'],
                'timestamp': metrics['timestamp']
            })
        
        # بررسی کاهش confidence
        recent_confidence = self._get_recent_confidence(model_name)
        if recent_confidence and metrics['confidence'] < recent_confidence - self.alert_thresholds['confidence_drop']:
            alerts_triggered.append({
                'type': 'confidence_drop',
                'model': model_name,
                'current': metrics['confidence'],
                'previous_avg': recent_confidence,
                'drop_amount': recent_confidence - metrics['confidence'],
                'timestamp': metrics['timestamp']
            })
        
        # ذخیره هشدارها
        for alert in alerts_triggered:
            self.alerts.append(alert)
            logger.warning(f"🚨 Performance alert: {alert['type']} for {model_name}")
            
            # ذخیره در کش برای سیستم هشدار
            self.cache_manager.set_data(
                "uta", 
                f"alert:{model_name}:{datetime.now().timestamp()}", 
                alert, 
                expire=3600
            )

    def _get_recent_confidence(self, model_name: str, window: int = 50) -> Optional[float]:
        """میانگین confidence در نمونه‌های اخیر"""
        history = list(self.performance_history[model_name])
        if len(history) < window:
            return None
            
        recent = history[-window:]
        confidences = [m['confidence'] for m in recent if m['success']]
        return np.mean(confidences) if confidences else None

    def _update_aggregate_metrics(self, model_name: str):
        """به‌روزرسانی متریک‌های تجمعی"""
        history = list(self.performance_history[model_name])
        if not history:
            return
        
        # محاسبه متریک‌های تجمعی
        successful_inferences = [m for m in history if m['success']]
        failed_inferences = [m for m in history if not m['success']]
        
        aggregate_metrics = {
            'total_inferences': len(history),
            'successful_inferences': len(successful_inferences),
            'failed_inferences': len(failed_inferences),
            'success_rate': len(successful_inferences) / len(history) if history else 0,
            'avg_inference_time': np.mean([m['inference_time'] for m in successful_inferences]) if successful_inferences else 0,
            'avg_confidence': np.mean([m['confidence'] for m in successful_inferences]) if successful_inferences else 0,
            'throughput_1min': self._calculate_throughput(model_name, window=60),
            'throughput_5min': self._calculate_throughput(model_name, window=300),
            'last_updated': datetime.now().isoformat()
        }
        
        # ذخیره در کش
        self.cache_manager.set_data(
            "uta", 
            f"aggregate_metrics:{model_name}", 
            aggregate_metrics, 
            expire=600
        )

    def _calculate_throughput(self, model_name: str, window: int) -> float:
        """محاسبه throughput در بازه زمانی مشخص (ثانیه)"""
        cutoff_time = datetime.now() - timedelta(seconds=window)
        recent_inferences = [
            m for m in self.performance_history[model_name]
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        return len(recent_inferences) / window if recent_inferences else 0

    def get_model_performance(self, model_name: str, time_window: str = "1h") -> Dict[str, Any]:
        """دریافت عملکرد مدل در بازه زمانی مشخص"""
        try:
            # تبدیل بازه زمانی به ثانیه
            window_seconds = {
                "1h": 3600,
                "6h": 21600,
                "24h": 86400,
                "7d": 604800
            }.get(time_window, 3600)
            
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            
            # فیلتر تاریخچه بر اساس بازه زمانی
            history = [
                m for m in self.performance_history[model_name]
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if not history:
                return {
                    'model': model_name,
                    'time_window': time_window,
                    'total_inferences': 0,
                    'message': 'No data available for the specified time window'
                }
            
            # محاسبه متریک‌های دقیق
            successful = [m for m in history if m['success']]
            failed = [m for m in history if not m['success']]
            
            inference_times = [m['inference_time'] for m in successful]
            confidences = [m['confidence'] for m in successful]
            
            performance_report = {
                'model': model_name,
                'time_window': time_window,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_inferences': len(history),
                    'successful_inferences': len(successful),
                    'failed_inferences': len(failed),
                    'success_rate': len(successful) / len(history),
                    'error_rate': len(failed) / len(history)
                },
                'timing_metrics': {
                    'avg_inference_time': np.mean(inference_times) if inference_times else 0,
                    'std_inference_time': np.std(inference_times) if inference_times else 0,
                    'p95_inference_time': np.percentile(inference_times, 95) if inference_times else 0,
                    'min_inference_time': min(inference_times) if inference_times else 0,
                    'max_inference_time': max(inference_times) if inference_times else 0
                },
                'quality_metrics': {
                    'avg_confidence': np.mean(confidences) if confidences else 0,
                    'std_confidence': np.std(confidences) if confidences else 0,
                    'min_confidence': min(confidences) if confidences else 0,
                    'max_confidence': max(confidences) if confidences else 0
                },
                'throughput_metrics': {
                    'current_throughput': self._calculate_throughput(model_name, window=60),
                    'avg_throughput': len(history) / window_seconds,
                    'peak_throughput': self._find_peak_throughput(model_name, window_seconds)
                }
            }
            
            return performance_report
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report for {model_name}: {e}")
            return {
                'model': model_name,
                'time_window': time_window,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _find_peak_throughput(self, model_name: str, window_seconds: int) -> float:
        """پیداکردن peak throughput در بازه زمانی"""
        # پیاده‌سازی ساده - می‌تواند پیچیده‌تر شود
        history = list(self.performance_history[model_name])
        if not history:
            return 0
        
        # گروه‌بندی بر اساس دقیقه
        minute_groups = defaultdict(int)
        for metric in history:
            dt = datetime.fromisoformat(metric['timestamp'])
            minute_key = dt.strftime("%Y%m%d%H%M")
            minute_groups[minute_key] += 1
        
        return max(minute_groups.values()) / 60 if minute_groups else 0  # درخواست بر ثانیه

    def get_comparative_analysis(self) -> Dict[str, Any]:
        """آنالیز مقایسه‌ای بین مدل‌ها"""
        comparative_report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'rankings': {}
        }
        
        for model_name in self.model_manager.active_models.keys():
            try:
                performance = self.get_model_performance(model_name, "24h")
                comparative_report['models'][model_name] = performance
            except Exception as e:
                logger.error(f"❌ Error analyzing {model_name}: {e}")
                comparative_report['models'][model_name] = {'error': str(e)}
        
        # رتبه‌بندی مدل‌ها
        if comparative_report['models']:
            # رتبه‌بندی بر اساس success rate
            success_rates = {
                name: data.get('summary', {}).get('success_rate', 0)
                for name, data in comparative_report['models'].items()
                if 'error' not in data
            }
            comparative_report['rankings']['by_success_rate'] = dict(
                sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
            )
            
            # رتبه‌بندی بر اساس سرعت
            speeds = {
                name: data.get('timing_metrics', {}).get('avg_inference_time', float('inf'))
                for name, data in comparative_report['models'].items()
                if 'error' not in data
            }
            comparative_report['rankings']['by_speed'] = dict(
                sorted(speeds.items(), key=lambda x: x[1])
            )
        
        return comparative_report

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """دریافت هشدارهای فعال"""
        # فیلتر هشدارهای اخیر (۱ ساعت گذشته)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        return recent_alerts

    def clear_old_data(self, days_old: int = 7):
        """پاک‌کردن داده‌های قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        for model_name in self.performance_history.keys():
            self.performance_history[model_name] = deque([
                m for m in self.performance_history[model_name]
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ], maxlen=1000)
        
        logger.info(f"🧹 Cleared performance data older than {days_old} days")

# نمونه global
performance_tracker = None

def initialize_performance_tracker(model_manager):
    """مقداردهی اولیه performance tracker"""
    global performance_tracker
    performance_tracker = PerformanceTracker(model_manager)
    return performance_tracker

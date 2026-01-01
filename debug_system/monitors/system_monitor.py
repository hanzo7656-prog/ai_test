import psutil
import time
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import deque, defaultdict
import threading
import statistics
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# نمونه گلوبال برای دسترسی از فایل‌های دیگر
central_monitor = None

class AnomalyType(Enum):
    """انواع ناهنجاری‌ها"""
    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    DISK_IO_SURGE = "disk_io_surge"
    NETWORK_FLASH = "network_flash"
    PROCESS_CRASH = "process_crash"
    SERVICE_DEGRADATION = "service_degradation"
    API_LATENCY_SPIKE = "api_latency_spike"
    DATA_QUALITY_DROP = "data_quality_drop"

@dataclass
class AnomalyDetection:
    """شناسایی ناهنجاری"""
    anomaly_type: AnomalyType
    severity: float  # 0-1
    detected_at: datetime
    metric_value: float
    baseline_value: float
    confidence: float

@dataclass
class CorrelationEvent:
    """رویداد همبستگی"""
    event_id: str
    events: List[str]
    correlation_score: float
    detected_at: datetime
    root_cause: Optional[str]

class PredictiveAlerting:
    """سیستم هشدار پیش‌بینانه"""
    
    def __init__(self, forecast_horizon_minutes: int = 30):
        self.forecast_horizon = forecast_horizon_minutes
        self.trend_data = defaultdict(lambda: deque(maxlen=100))
        self.prediction_models = {}
        
    def add_metric_point(self, metric_name: str, value: float, timestamp: datetime):
        """اضافه کردن نقطه متریک برای تحلیل روند"""
        self.trend_data[metric_name].append((timestamp, value))
        
    def predict_threshold_breach(self, metric_name: str, threshold: float) -> Optional[datetime]:
        """پیش‌بینی زمان عبور از آستانه"""
        data = list(self.trend_data[metric_name])
        if len(data) < 10:
            return None
            
        timestamps, values = zip(*data)
        
        # تحلیل روند ساده (میانگین متحرک)
        if len(values) >= 5:
            recent_trend = np.polyfit(range(len(values[-5:])), values[-5:], 1)[0]
            
            if recent_trend > 0:  # روند صعودی
                current_value = values[-1]
                if current_value < threshold:
                    time_to_breach = (threshold - current_value) / recent_trend
                    if time_to_breach > 0:
                        estimated_breach = datetime.now() + timedelta(minutes=time_to_breach * 5)  # scale factor
                        return estimated_breach
        return None

class DependencyMonitor:
    """مانیتورینگ وضعیت وابستگی‌ها"""
    
    def __init__(self):
        self.dependencies = {
            'redis': {'status': 'unknown', 'last_check': None, 'latency_ms': None},
            'database': {'status': 'unknown', 'last_check': None, 'latency_ms': None},
            'external_apis': defaultdict(lambda: {'status': 'unknown', 'last_check': None, 'latency_ms': None}),
            'internal_services': defaultdict(lambda: {'status': 'unknown', 'last_check': None, 'latency_ms': None})
        }
        
    async def check_redis(self, redis_client=None) -> Dict[str, Any]:
        """بررسی سلامت Redis"""
        try:
            start_time = time.time()
            
            # اگر کلاینت Redis داریم، ping می‌کنیم
            if redis_client and hasattr(redis_client, 'ping'):
                redis_client.ping()
                latency = (time.time() - start_time) * 1000
                self.dependencies['redis'] = {
                    'status': 'healthy',
                    'last_check': datetime.now(),
                    'latency_ms': round(latency, 2),
                    'memory_usage': redis_client.info().get('used_memory_human', 'N/A')
                }
            else:
                # شبیه‌سازی برای تست
                await asyncio.sleep(0.01)
                self.dependencies['redis'] = {
                    'status': 'simulated_healthy',
                    'last_check': datetime.now(),
                    'latency_ms': 10.5,
                    'memory_usage': '256MB'
                }
                
            return self.dependencies['redis']
            
        except Exception as e:
            self.dependencies['redis'] = {
                'status': 'unhealthy',
                'last_check': datetime.now(),
                'latency_ms': None,
                'error': str(e)
            }
            return self.dependencies['redis']
    
    def check_database(self, db_connection=None) -> Dict[str, Any]:
        """بررسی سلامت دیتابیس"""
        try:
            start_time = time.time()
            
            if db_connection:
                # اجرای کوئری تست ساده
                with db_connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    
                latency = (time.time() - start_time) * 1000
                self.dependencies['database'] = {
                    'status': 'healthy',
                    'last_check': datetime.now(),
                    'latency_ms': round(latency, 2)
                }
            else:
                # شبیه‌سازی
                self.dependencies['database'] = {
                    'status': 'simulated_healthy',
                    'last_check': datetime.now(),
                    'latency_ms': 15.2
                }
                
            return self.dependencies['database']
            
        except Exception as e:
            self.dependencies['database'] = {
                'status': 'unhealthy',
                'last_check': datetime.now(),
                'latency_ms': None,
                'error': str(e)
            }
            return self.dependencies['database']
    
    def add_external_api(self, name: str, url: str, check_interval: int = 60):
        """اضافه کردن API خارجی برای مانیتورینگ"""
        self.dependencies['external_apis'][name] = {
            'url': url,
            'status': 'pending',
            'last_check': None,
            'check_interval': check_interval,
            'latency_ms': None
        }
    
    async def check_external_api(self, name: str) -> Dict[str, Any]:
        """بررسی API خارجی"""
        if name not in self.dependencies['external_apis']:
            return {'error': f'API {name} not registered'}
            
        api_info = self.dependencies['external_apis'][name]
        
        try:
            import aiohttp
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(api_info['url']) as response:
                    latency = (time.time() - start_time) * 1000
                    status = 'healthy' if response.status == 200 else 'unhealthy'
                    
                    api_info.update({
                        'status': status,
                        'last_check': datetime.now(),
                        'latency_ms': round(latency, 2),
                        'status_code': response.status
                    })
                    
            return api_info
            
        except Exception as e:
            api_info.update({
                'status': 'unhealthy',
                'last_check': datetime.now(),
                'latency_ms': None,
                'error': str(e)
            })
            return api_info
    
    def get_dependency_health_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه سلامت وابستگی‌ها"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'unhealthy_count': 0,
            'total_count': 0,
            'details': {}
        }
        
        # بررسی Redis
        redis = self.dependencies['redis']
        summary['details']['redis'] = redis
        summary['total_count'] += 1
        if redis['status'] != 'healthy':
            summary['unhealthy_count'] += 1
            
        # بررسی Database
        db = self.dependencies['database']
        summary['details']['database'] = db
        summary['total_count'] += 1
        if db['status'] != 'healthy':
            summary['unhealthy_count'] += 1
            
        # بررسی APIهای خارجی
        for name, api in self.dependencies['external_apis'].items():
            summary['details'][f'external_api_{name}'] = api
            summary['total_count'] += 1
            if api['status'] != 'healthy':
                summary['unhealthy_count'] += 1
        
        # محاسبه سلامت کلی
        if summary['unhealthy_count'] > 0:
            summary['overall_health'] = 'degraded' if summary['unhealthy_count'] < summary['total_count'] / 2 else 'unhealthy'
        
        return summary

class AnomalyDetector:
    """تشخیص‌دهنده ناهنجاری‌ها"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        self.anomalies_detected = deque(maxlen=100)
        
    def add_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """اضافه کردن متریک برای تحلیل"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.metric_history[metric_name].append((timestamp, value))
        
        # به‌روزرسانی baseline
        if len(self.metric_history[metric_name]) >= 10:
            values = [v for _, v in self.metric_history[metric_name]]
            self.baselines[metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.1,
                'min': min(values),
                'max': max(values)
            }
    
    def detect_anomalies(self) -> List[AnomalyDetection]:
        """تشخیص ناهنجاری‌ها"""
        anomalies = []
        current_time = datetime.now()
        
        for metric_name, history in self.metric_history.items():
            if len(history) < 5 or metric_name not in self.baselines:
                continue
                
            latest_value = history[-1][1]
            baseline = self.baselines[metric_name]
            
            # تشخیص spike (فراتر از 3 انحراف معیار)
            if baseline['std'] > 0:
                z_score = abs(latest_value - baseline['mean']) / baseline['std']
                
                if z_score > 3.0:
                    anomaly_type = self._determine_anomaly_type(metric_name, latest_value, baseline)
                    
                    anomaly = AnomalyDetection(
                        anomaly_type=anomaly_type,
                        severity=min(z_score / 5.0, 1.0),  # نرمال‌سازی به 0-1
                        detected_at=current_time,
                        metric_value=latest_value,
                        baseline_value=baseline['mean'],
                        confidence=min(z_score / 4.0, 0.95)
                    )
                    anomalies.append(anomaly)
                    self.anomalies_detected.append(anomaly)
        
        return anomalies
    
    def _determine_anomaly_type(self, metric_name: str, value: float, baseline: Dict) -> AnomalyType:
        """تعیین نوع ناهنجاری"""
        metric_lower = metric_name.lower()
        
        if 'cpu' in metric_lower:
            return AnomalyType.CPU_SPIKE
        elif 'memory' in metric_lower:
            return AnomalyType.MEMORY_LEAK if value > baseline['mean'] else AnomalyType.PROCESS_CRASH
        elif 'disk' in metric_lower and ('io' in metric_lower or 'read' in metric_lower or 'write' in metric_lower):
            return AnomalyType.DISK_IO_SURGE
        elif 'network' in metric_lower or 'bytes' in metric_lower:
            return AnomalyType.NETWORK_FLASH
        elif 'api' in metric_lower or 'latency' in metric_lower:
            return AnomalyType.API_LATENCY_SPIKE
        elif 'quality' in metric_lower or 'success' in metric_lower:
            return AnomalyType.DATA_QUALITY_DROP
        else:
            return AnomalyType.SERVICE_DEGRADATION
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """خلاصه ناهنجاری‌های اخیر"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_anomalies = [a for a in self.anomalies_detected if a.detected_at > cutoff]
        
        anomaly_by_type = defaultdict(list)
        for anomaly in recent_anomalies:
            anomaly_by_type[anomaly.anomaly_type.value].append(anomaly)
        
        return {
            'total_anomalies': len(recent_anomalies),
            'anomalies_by_type': {k: len(v) for k, v in anomaly_by_type.items()},
            'most_common_anomaly': max(anomaly_by_type.items(), key=lambda x: len(x[1]))[0] if anomaly_by_type else None,
            'highest_severity': max((a.severity for a in recent_anomalies), default=0),
            'time_period_hours': hours
        }

class CorrelationEngine:
    """موتور تحلیل همبستگی رویدادها"""
    
    def __init__(self):
        self.events = deque(maxlen=1000)
        self.correlations = deque(maxlen=100)
        self.event_types = set()
        
    def add_event(self, event_type: str, event_data: Dict[str, Any], timestamp: datetime = None):
        """اضافه کردن رویداد"""
        if timestamp is None:
            timestamp = datetime.now()
            
        event_id = f"{event_type}_{timestamp.timestamp()}_{hash(str(event_data)) % 10000}"
        
        event = {
            'id': event_id,
            'type': event_type,
            'data': event_data,
            'timestamp': timestamp,
            'metrics': event_data.get('metrics', {})
        }
        
        self.events.append(event)
        self.event_types.add(event_type)
        
        # بررسی همبستگی با رویدادهای اخیر
        self._check_correlations(event)
        
        return event_id
    
    def _check_correlations(self, new_event: Dict[str, Any]):
        """بررسی همبستگی با رویدادهای اخیر"""
        recent_events = [e for e in self.events 
                        if e['timestamp'] > datetime.now() - timedelta(minutes=5)
                        and e['id'] != new_event['id']]
        
        if not recent_events:
            return
            
        for event in recent_events:
            correlation_score = self._calculate_correlation_score(new_event, event)
            
            if correlation_score > 0.7:  # آستانه همبستگی بالا
                correlation = CorrelationEvent(
                    event_id=f"corr_{new_event['id']}_{event['id']}",
                    events=[new_event['id'], event['id']],
                    correlation_score=correlation_score,
                    detected_at=datetime.now(),
                    root_cause=self._identify_root_cause(new_event, event)
                )
                self.correlations.append(correlation)
    
    def _calculate_correlation_score(self, event1: Dict, event2: Dict) -> float:
        """محاسبه امتیاز همبستگی"""
        score = 0.0
        
        # همبستگی زمانی (وقتی نزدیک هم اتفاق بیفتند)
        time_diff = abs((event1['timestamp'] - event2['timestamp']).total_seconds())
        if time_diff < 60:  # در عرض 1 دقیقه
            time_score = 1.0 - (time_diff / 60)
            score += time_score * 0.4
        
        # همبستگی در متریک‌ها
        metrics1 = event1.get('metrics', {})
        metrics2 = event2.get('metrics', {})
        
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        if common_metrics:
            metric_correlations = []
            for metric in common_metrics:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # همبستگی ساده: اگر هر دو بالا یا پایین باشند
                    if (val1 > 50 and val2 > 50) or (val1 < 30 and val2 < 30):
                        metric_correlations.append(1.0)
                    else:
                        metric_correlations.append(0.0)
            
            if metric_correlations:
                metric_score = sum(metric_correlations) / len(metric_correlations)
                score += metric_score * 0.6
        
        return min(score, 1.0)
    
    def _identify_root_cause(self, event1: Dict, event2: Dict) -> Optional[str]:
        """شناسایی علت اصلی"""
        # منطق ساده برای شناسایی root cause
        event_types = {event1['type'], event2['type']}
        
        if AnomalyType.CPU_SPIKE.value in event_types and AnomalyType.API_LATENCY_SPIKE.value in event_types:
            return "High CPU usage causing API latency"
        elif AnomalyType.MEMORY_LEAK.value in event_types and AnomalyType.SERVICE_DEGRADATION.value in event_types:
            return "Memory leak leading to service degradation"
        elif AnomalyType.DISK_IO_SURGE.value in event_types and AnomalyType.DATA_QUALITY_DROP.value in event_types:
            return "Disk I/O issues affecting data processing"
        
        return None
    
    def get_correlations_summary(self, hours: int = 6) -> Dict[str, Any]:
        """خلاصه همبستگی‌های اخیر"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_correlations = [c for c in self.correlations if c.detected_at > cutoff]
        
        return {
            'total_correlations': len(recent_correlations),
            'high_confidence_correlations': len([c for c in recent_correlations if c.correlation_score > 0.8]),
            'common_root_causes': self._extract_common_root_causes(recent_correlations),
            'most_correlated_event_types': self._find_most_correlated_types(recent_correlations),
            'time_period_hours': hours
        }
    
    def _extract_common_root_causes(self, correlations: List[CorrelationEvent]) -> List[str]:
        """استخراج علل اصلی رایج"""
        causes = [c.root_cause for c in correlations if c.root_cause]
        cause_counts = defaultdict(int)
        for cause in causes:
            cause_counts[cause] += 1
        
        return [cause for cause, count in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _find_most_correlated_types(self, correlations: List[CorrelationEvent]) -> List[Tuple[str, str, int]]:
        """یافتن پرتکرارترین جفت رویدادهای مرتبط"""
        type_pairs = defaultdict(int)
        
        for corr in correlations:
            # استخراج رویدادها و انواعشان
            events = self._get_events_by_ids(corr.events)
            if len(events) >= 2:
                types = tuple(sorted([e['type'] for e in events[:2]]))
                type_pairs[types] += 1
        
        return [(t1, t2, count) for (t1, t2), count in sorted(type_pairs.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _get_events_by_ids(self, event_ids: List[str]) -> List[Dict]:
        """دریافت رویدادها بر اساس ID"""
        return [e for e in self.events if e['id'] in event_ids]

class TrendAnalyzer:
    """تحلیل‌گر روند و پیش‌بینی"""
    
    def __init__(self, forecast_window: int = 7):
        self.forecast_window = forecast_window
        self.trend_data = defaultdict(lambda: deque(maxlen=500))
        
    def add_data_point(self, metric_name: str, value: float, timestamp: datetime = None):
        """اضافه کردن نقطه داده"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.trend_data[metric_name].append((timestamp, value))
    
    def analyze_trend(self, metric_name: str, window_hours: int = 24) -> Dict[str, Any]:
        """تحلیل روند یک متریک"""
        data = list(self.trend_data[metric_name])
        if not data:
            return {'error': 'No data available'}
        
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_data = [(ts, val) for ts, val in data if ts > cutoff]
        
        if len(recent_data) < 3:
            return {'error': 'Insufficient data for analysis'}
        
        timestamps, values = zip(*recent_data)
        
        # محاسبه روند (linear regression)
        x = np.arange(len(values))
        try:
            slope, intercept = np.polyfit(x, values, 1)
        except:
            slope, intercept = 0, values[0]
        
        # محاسبه پیش‌بینی
        forecast = []
        for i in range(self.forecast_window):
            predicted = slope * (len(values) + i) + intercept
            forecast.append(max(predicted, 0))  # منفی نشود
        
        # تحلیل نوسانات
        volatility = np.std(values) if len(values) > 1 else 0
        
        return {
            'metric': metric_name,
            'data_points': len(recent_data),
            'current_value': values[-1],
            'trend_direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
            'trend_strength': abs(slope),
            'volatility': volatility,
            'forecast': forecast,
            'time_to_threshold': self._estimate_time_to_threshold(values, slope),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _estimate_time_to_threshold(self, values: List[float], slope: float) -> Optional[Dict[str, Any]]:
        """تخمین زمان رسیدن به آستانه‌های مهم"""
        if abs(slope) < 0.001:
            return None
        
        current_value = values[-1]
        thresholds = {
            'warning_80': 80,
            'critical_90': 90,
            'warning_low_20': 20
        }
        
        estimates = {}
        for name, threshold in thresholds.items():
            if (slope > 0 and current_value < threshold) or (slope < 0 and current_value > threshold):
                time_to_reach = abs((threshold - current_value) / slope)
                # تبدیل به ساعت (با فرض اینکه هر نقطه داده 5 دقیقه فاصله دارد)
                hours_to_reach = (time_to_reach * 5) / 60
                
                if hours_to_reach > 0:
                    estimates[name] = {
                        'threshold': threshold,
                        'estimated_hours': round(hours_to_reach, 1),
                        'estimated_time': (datetime.now() + timedelta(hours=hours_to_reach)).isoformat()
                    }
        
        return estimates if estimates else None
    
    def get_metric_forecast(self, metric_name: str) -> Dict[str, Any]:
        """دریافت پیش‌بینی برای یک متریک"""
        return self.analyze_trend(metric_name)
    
    def get_all_trends_summary(self) -> Dict[str, Any]:
        """خلاصه روندهای تمام متریک‌ها"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics_analyzed': [],
            'critical_trends': [],
            'improving_trends': []
        }
        
        for metric_name in self.trend_data.keys():
            trend = self.analyze_trend(metric_name, window_hours=6)
            
            if 'error' not in trend:
                summary['metrics_analyzed'].append({
                    'name': metric_name,
                    'trend': trend['trend_direction'],
                    'current': trend['current_value']
                })
                
                if trend['trend_direction'] == 'increasing' and trend['current_value'] > 50:
                    summary['critical_trends'].append(metric_name)
                elif trend['trend_direction'] == 'decreasing' and trend['current_value'] < 50:
                    summary['improving_trends'].append(metric_name)
        
        return summary

class SelfHealingAdvisor:
    """مشاور خود-ترمیمی"""
    
    def __init__(self):
        self.recommendations_db = {
            'high_cpu': [
                "Restart the service to clear stuck processes",
                "Check for infinite loops in recent deployments",
                "Increase CPU allocation if running in container",
                "Optimize database queries that might be causing load"
            ],
            'high_memory': [
                "Restart application to free up memory leaks",
                "Increase Java heap size if applicable",
                "Check for memory leaks in recent code changes",
                "Reduce cache size or implement cache eviction"
            ],
            'high_disk': [
                "Clean up temporary files and logs",
                "Implement log rotation if not already done",
                "Archive old data to cold storage",
                "Increase disk space allocation"
            ],
            'network_issues': [
                "Check firewall rules and network configuration",
                "Restart network services",
                "Verify DNS resolution",
                "Check for DDoS attacks or unusual traffic"
            ],
            'api_latency': [
                "Optimize database indexes",
                "Implement response caching",
                "Check external API dependencies",
                "Scale up application instances"
            ],
            'service_degradation': [
                "Restart affected services",
                "Check dependency health (database, redis, etc.)",
                "Verify recent deployment changes",
                "Rollback to last stable version if possible"
            ]
        }
        
        self.action_history = deque(maxlen=100)
    
    def get_recommendations(self, issue_type: str, severity: float = 0.5) -> List[Dict[str, Any]]:
        """دریافت توصیه‌های ترمیمی"""
        recommendations = []
        
        # پیدا کردن توصیه‌های مرتبط
        for key, suggestions in self.recommendations_db.items():
            if key in issue_type.lower() or issue_type.lower() in key:
                for i, suggestion in enumerate(suggestions):
                    recommendations.append({
                        'id': f"{key}_{i}_{int(time.time())}",
                        'issue_type': key,
                        'recommendation': suggestion,
                        'priority': 'high' if severity > 0.7 else 'medium' if severity > 0.4 else 'low',
                        'estimated_impact': 'high' if i == 0 else 'medium' if i == 1 else 'low',
                        'implementation_time': 'minutes' if i < 2 else 'hours'
                    })
        
        # اگر هیچ توصیه مستقیمی پیدا نشد
        if not recommendations:
            recommendations.append({
                'id': f"generic_{int(time.time())}",
                'issue_type': 'generic',
                'recommendation': "Check system logs and metrics for root cause analysis",
                'priority': 'medium',
                'estimated_impact': 'medium',
                'implementation_time': 'hours'
            })
        
        return recommendations
    
    def record_action(self, action: Dict[str, Any], success: bool = True, notes: str = ""):
        """ثبت اقدام انجام شده"""
        action_record = {
            'action_id': action.get('id', f"action_{int(time.time())}"),
            'action_type': action.get('issue_type', 'unknown'),
            'recommendation': action.get('recommendation', ''),
            'executed_at': datetime.now(),
            'success': success,
            'notes': notes,
            'resolved_issue': action.get('resolved_issue', False)
        }
        
        self.action_history.append(action_record)
        return action_record
    
    def get_action_history_summary(self, days: int = 7) -> Dict[str, Any]:
        """خلاصه تاریخچه اقدامات"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_actions = [a for a in self.action_history if a['executed_at'] > cutoff]
        
        if not recent_actions:
            return {'message': 'No recent actions found'}
        
        success_rate = sum(1 for a in recent_actions if a['success']) / len(recent_actions)
        resolved_rate = sum(1 for a in recent_actions if a.get('resolved_issue', False)) / len(recent_actions) if recent_actions else 0
        
        return {
            'total_actions': len(recent_actions),
            'success_rate': f"{success_rate * 100:.1f}%",
            'issue_resolution_rate': f"{resolved_rate * 100:.1f}%",
            'most_common_action': self._get_most_common_action_type(recent_actions),
            'recent_actions': recent_actions[-10:],  # 10 اقدام اخیر
            'time_period_days': days
        }
    
    def _get_most_common_action_type(self, actions: List[Dict]) -> str:
        """یافتن پرتکرارترین نوع اقدام"""
        action_types = [a['action_type'] for a in actions]
        if not action_types:
            return 'none'
        
        from collections import Counter
        return Counter(action_types).most_common(1)[0][0]

class SystemMonitor:
    """کلاس اصلی مانیتورینگ سیستم با قابلیت‌های پیشرفته"""
    
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
        
        # سیستم‌های پیشرفته
        self.dependency_monitor = DependencyMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.correlation_engine = CorrelationEngine()
        self.trend_analyzer = TrendAnalyzer()
        self.predictive_alerter = PredictiveAlerting()
        self.self_healing_advisor = SelfHealingAdvisor()
        
        # عضویت در central_monitor
        global central_monitor
        if central_monitor:
            central_monitor.subscribe("system_monitor", self._on_metrics_update)
            logger.info("✅ SystemMonitor subscribed to central_monitor with advanced features")
        else:
            logger.warning("⚠️ Central monitor not available - system monitor will be passive")
        
        # کش برای جلوگیری از duplicate alerts
        self.alert_cache = {}
        self.cache_ttl = 60  # 60 seconds
        
        # شروع مانیتورینگ وابستگی‌ها
        self._start_dependency_monitoring()
    
    async def _start_dependency_monitoring(self):
        """شروع مانیتورینگ وابستگی‌ها"""
        async def monitor_loop():
            while True:
                try:
                    # بررسی Redis هر 30 ثانیه
                    await self.dependency_monitor.check_redis()
                    
                    # بررسی Database هر 60 ثانیه
                    if int(time.time()) % 60 == 0:
                        self.dependency_monitor.check_database()
                    
                    # بررسی APIهای خارجی هر 2 دقیقه
                    if int(time.time()) % 120 == 0:
                        for api_name in list(self.dependency_monitor.dependencies['external_apis'].keys()):
                            await self.dependency_monitor.check_external_api(api_name)
                    
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    logger.error(f"❌ Dependency monitoring error: {e}")
                    await asyncio.sleep(60)
        
        # اجرای در background
        asyncio.create_task(monitor_loop())
        logger.info("🔄 Advanced dependency monitoring started")
    
    def _on_metrics_update(self, metrics: Dict[str, Any]):
        """دریافت متریک‌ها از سیستم مرکزی"""
        try:
            system_metrics = metrics.get('system', {})
            
            # ذخیره برای تحلیل روند
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            
            timestamp = datetime.now()
            self.anomaly_detector.add_metric('cpu_usage', cpu_usage, timestamp)
            self.anomaly_detector.add_metric('memory_usage', memory_usage, timestamp)
            
            self.trend_analyzer.add_data_point('cpu_usage', cpu_usage, timestamp)
            self.trend_analyzer.add_data_point('memory_usage', memory_usage, timestamp)
            
            self.predictive_alerter.add_metric_point('cpu_usage', cpu_usage, timestamp)
            self.predictive_alerter.add_metric_point('memory_usage', memory_usage, timestamp)
            
            # تشخیص ناهنجاری
            anomalies = self.anomaly_detector.detect_anomalies()
            for anomaly in anomalies:
                self._handle_anomaly(anomaly, system_metrics)
            
            # پیش‌بینی هشدارها
            self._check_predictive_alerts(system_metrics)
            
            # انجام چک سلامت
            self._perform_health_check_with_metrics(system_metrics)
            
        except Exception as e:
            logger.error(f"❌ Error processing metrics update: {e}")
    
    def _handle_anomaly(self, anomaly: AnomalyDetection, metrics: Dict[str, Any]):
        """مدیریت ناهنجاری شناسایی شده"""
        try:
            # ایجاد رویداد در موتور همبستگی
            event_id = self.correlation_engine.add_event(
                anomaly.anomaly_type.value,
                {
                    'severity': anomaly.severity,
                    'metric_value': anomaly.metric_value,
                    'baseline_value': anomaly.baseline_value,
                    'metrics': {
                        'cpu': metrics.get('cpu', {}).get('percent', 0),
                        'memory': metrics.get('memory', {}).get('percent', 0),
                        'disk': metrics.get('disk', {}).get('usage_percent', 0)
                    }
                }
            )
            
            # ایجاد هشدار برای ناهنجاری‌های شدید
            if anomaly.severity > 0.7:
                from debug_system.core.alert_manager import AlertLevel, AlertType
                
                self.alert_manager.create_alert(
                    level=AlertLevel.WARNING if anomaly.severity < 0.9 else AlertLevel.CRITICAL,
                    alert_type=AlertType.SYSTEM,
                    title=f"Anomaly Detected: {anomaly.anomaly_type.value.replace('_', ' ').title()}",
                    message=f"Detected {anomaly.anomaly_type.value} with severity {anomaly.severity:.2f}. "
                           f"Value: {anomaly.metric_value:.1f}, Baseline: {anomaly.baseline_value:.1f}",
                    source="anomaly_detector",
                    data={
                        'anomaly_type': anomaly.anomaly_type.value,
                        'severity': anomaly.severity,
                        'confidence': anomaly.confidence,
                        'event_id': event_id
                    }
                )
                
                logger.warning(f"🚨 Anomaly detected: {anomaly.anomaly_type.value} (severity: {anomaly.severity:.2f})")
            
            # دریافت توصیه‌های خود-ترمیمی
            if anomaly.severity > 0.6:
                recommendations = self.self_healing_advisor.get_recommendations(
                    anomaly.anomaly_type.value,
                    anomaly.severity
                )
                
                logger.info(f"💡 Self-healing recommendations for {anomaly.anomaly_type.value}: {len(recommendations)} found")
                
        except Exception as e:
            logger.error(f"❌ Error handling anomaly: {e}")
    
    def _check_predictive_alerts(self, metrics: Dict[str, Any]):
        """بررسی هشدارهای پیش‌بینانه"""
        try:
            cpu_usage = metrics.get('cpu', {}).get('percent', 0)
            memory_usage = metrics.get('memory', {}).get('percent', 0)
            
            # پیش‌بینی زمان رسیدن به آستانه
            cpu_breach_time = self.predictive_alerter.predict_threshold_breach(
                'cpu_usage', 
                self.system_thresholds['cpu_warning']
            )
            
            memory_breach_time = self.predictive_alerter.predict_threshold_breach(
                'memory_usage',
                self.system_thresholds['memory_warning']
            )
            
            # ایجاد هشدار پیش‌بینانه
            if cpu_breach_time and cpu_usage < self.system_thresholds['cpu_warning']:
                time_to_breach = (cpu_breach_time - datetime.now()).total_seconds() / 60  # به دقیقه
                if time_to_breach < 30:  # اگر کمتر از 30 دقیقه باقی مانده
                    self._create_predictive_alert(
                        'CPU usage',
                        time_to_breach,
                        self.system_thresholds['cpu_warning'],
                        cpu_usage
                    )
            
            if memory_breach_time and memory_usage < self.system_thresholds['memory_warning']:
                time_to_breach = (memory_breach_time - datetime.now()).total_seconds() / 60
                if time_to_breach < 30:
                    self._create_predictive_alert(
                        'Memory usage',
                        time_to_breach,
                        self.system_thresholds['memory_warning'],
                        memory_usage
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error checking predictive alerts: {e}")
    
    def _create_predictive_alert(self, metric_name: str, minutes_to_breach: float, threshold: float, current_value: float):
        """ایجاد هشدار پیش‌بینانه"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            self.alert_manager.create_alert(
                level=AlertLevel.INFO,
                alert_type=AlertType.SYSTEM,
                title=f"Predictive Alert: {metric_name}",
                message=f"{metric_name} expected to reach {threshold}% in {minutes_to_breach:.1f} minutes. "
                       f"Current: {current_value:.1f}%",
                source="predictive_monitor",
                data={
                    'metric': metric_name,
                    'minutes_to_breach': minutes_to_breach,
                    'threshold': threshold,
                    'current_value': current_value
                }
            )
            
            logger.info(f"🔮 Predictive alert: {metric_name} will breach in {minutes_to_breach:.1f} minutes")
            
        except Exception as e:
            logger.error(f"❌ Error creating predictive alert: {e}")
    
    def _perform_health_check_with_metrics(self, metrics: Dict[str, Any]):
        """انجام چک سلامت با متریک‌های داده شده"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            cpu_usage = metrics.get('cpu', {}).get('percent', 0)
            self._check_cpu_health(cpu_usage, metrics)
            
            memory_usage = metrics.get('memory', {}).get('percent', 0)
            self._check_memory_health(memory_usage, metrics)
            
            disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
            self._check_disk_health(disk_usage, metrics)
            
        except Exception as e:
            logger.error(f"❌ Error in system health check: {e}")
    
    def _check_cpu_health(self, cpu_usage: float, metrics: Dict):
        """بررسی سلامت CPU با cache"""
        alert_key = f"cpu_{int(cpu_usage // 10)}"
        
        if self._is_cached_alert(alert_key):
            return
        
        if cpu_usage > self.system_thresholds['cpu_critical']:
            self._create_cached_alert(
                alert_key,
                AlertLevel.CRITICAL,
                AlertType.SYSTEM,
                "Critical CPU Usage",
                f"CPU usage is critically high: {cpu_usage:.1f}%",
                "system_monitor",
                {'cpu_usage': cpu_usage, 'threshold': self.system_thresholds['cpu_critical']}
            )
        elif cpu_usage > self.system_thresholds['cpu_warning']:
            self._create_cached_alert(
                alert_key,
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                "High CPU Usage",
                f"CPU usage is high: {cpu_usage:.1f}%",
                "system_monitor",
                {'cpu_usage': cpu_usage, 'threshold': self.system_thresholds['cpu_warning']}
            )
    
    def _check_memory_health(self, memory_usage: float, metrics: Dict):
        """بررسی سلامت Memory با cache"""
        alert_key = f"memory_{int(memory_usage // 10)}"
        
        if self._is_cached_alert(alert_key):
            return
            
        if memory_usage > self.system_thresholds['memory_critical']:
            self._create_cached_alert(
                alert_key,
                AlertLevel.CRITICAL,
                AlertType.SYSTEM,
                "Critical Memory Usage",
                f"Memory usage is critically high: {memory_usage:.1f}%",
                "system_monitor",
                {'memory_usage': memory_usage, 'threshold': self.system_thresholds['memory_critical']}
            )
        elif memory_usage > self.system_thresholds['memory_warning']:
            self._create_cached_alert(
                alert_key,
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                "High Memory Usage", 
                f"Memory usage is high: {memory_usage:.1f}%",
                "system_monitor",
                {'memory_usage': memory_usage, 'threshold': self.system_thresholds['memory_warning']}
            )
    
    def _check_disk_health(self, disk_usage: float, metrics: Dict):
        """بررسی سلامت Disk با cache"""
        alert_key = f"disk_{int(disk_usage // 10)}"
        
        if self._is_cached_alert(alert_key):
            return
        
        if disk_usage > self.system_thresholds['disk_critical']:
            self._create_cached_alert(
                alert_key,
                AlertLevel.CRITICAL,
                AlertType.SYSTEM,
                "Critical Disk Usage",
                f"Disk usage is critically high: {disk_usage:.1f}%",
                "system_monitor", 
                {'disk_usage': disk_usage, 'threshold': self.system_thresholds['disk_critical']}
            )
        elif disk_usage > self.system_thresholds['disk_warning']:
            self._create_cached_alert(
                alert_key,
                AlertLevel.WARNING,
                AlertType.SYSTEM,
                "High Disk Usage",
                f"Disk usage is high: {disk_usage:.1f}%",
                "system_monitor",
                {'disk_usage': disk_usage, 'threshold': self.system_thresholds['disk_warning']}
            )
    
    def _is_cached_alert(self, alert_key: str) -> bool:
        """بررسی آیا alert در cache است"""
        if alert_key in self.alert_cache:
            cache_time = self.alert_cache[alert_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return True
        return False
    
    def _create_cached_alert(self, alert_key: str, level, alert_type, title, message, source, data):
        """ایجاد alert با cache"""
        self.alert_cache[alert_key] = datetime.now()
        self._create_alert_sync(level, alert_type, title, message, source, data)
    
    def _create_alert_sync(self, level, alert_type, title, message, source, data):
        """ایجاد هشدار به صورت synchronous"""
        try:
            alert_result = self.alert_manager.create_alert(
                level=level,
                alert_type=alert_type,
                title=title,
                message=message,
                source=source,
                data=data
            )
            
            if alert_result:
                logger.info(f"🚨 System alert created: {title}")
            else:
                logger.debug(f"⚠️ System alert was not created (might be in cooldown): {title}")
                
        except Exception as e:
            logger.error(f"❌ Error creating system alert: {e}")
    
    # APIهای پیشرفته جدید
    
    def get_advanced_health_report(self) -> Dict[str, Any]:
        """گزارش سلامت پیشرفته"""
        try:
            # متریک‌های فعلی
            metrics = self.get_system_health()
            
            # تحلیل ناهنجاری
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
            
            # تحلیل همبستگی
            correlation_summary = self.correlation_engine.get_correlations_summary()
            
            # تحلیل روند
            trend_summary = self.trend_analyzer.get_all_trends_summary()
            
            # سلامت وابستگی‌ها
            dependency_summary = self.dependency_monitor.get_dependency_health_summary()
            
            # تاریخچه اقدامات خود-ترمیمی
            healing_summary = self.self_healing_advisor.get_action_history_summary(days=1)
            
            # پیش‌بینی‌ها
            cpu_forecast = self.trend_analyzer.get_metric_forecast('cpu_usage')
            memory_forecast = self.trend_analyzer.get_metric_forecast('memory_usage')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'basic_health': metrics,
                'anomaly_analysis': anomaly_summary,
                'correlation_analysis': correlation_summary,
                'trend_analysis': trend_summary,
                'dependency_health': dependency_summary,
                'self_healing_history': healing_summary,
                'predictions': {
                    'cpu_forecast': cpu_forecast,
                    'memory_forecast': memory_forecast
                },
                'recommendations': self._generate_recommendations(metrics, anomaly_summary, dependency_summary)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating advanced health report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_recommendations(self, health_metrics: Dict, anomaly_summary: Dict, dependency_summary: Dict) -> List[Dict[str, Any]]:
        """تولید توصیه‌های هوشمند"""
        recommendations = []
        
        # بررسی سلامت عمومی
        if health_metrics.get('overall_health') == 'critical':
            recommendations.append({
                'type': 'urgent',
                'message': 'System health is critical. Immediate attention required.',
                'action': 'Check critical alerts and restart services if necessary'
            })
        
        # بررسی ناهنجاری‌ها
        if anomaly_summary.get('total_anomalies', 0) > 5:
            recommendations.append({
                'type': 'warning',
                'message': f"High number of anomalies detected: {anomaly_summary.get('total_anomalies')}",
                'action': 'Review anomaly logs and investigate root causes'
            })
        
        # بررسی وابستگی‌ها
        if dependency_summary.get('overall_health') != 'healthy':
            recommendations.append({
                'type': 'dependency',
                'message': f"Dependencies health: {dependency_summary.get('overall_health')}",
                'action': 'Check external services and network connectivity'
            })
        
        # بررسی روندهای بحرانی
        trends = self.trend_analyzer.get_all_trends_summary()
        for critical in trends.get('critical_trends', []):
            recommendations.append({
                'type': 'trend',
                'message': f"Critical increasing trend detected: {critical}",
                'action': f'Monitor {critical} closely and consider scaling resources'
            })
        
        # اگر هیچ توصیه‌ای نبود
        if not recommendations:
            recommendations.append({
                'type': 'info',
                'message': 'System health is stable. No immediate actions required.',
                'action': 'Continue regular monitoring'
            })
        
        return recommendations
    
    def register_external_api(self, name: str, url: str, check_interval: int = 60) -> bool:
        """ثبت API خارجی برای مانیتورینگ"""
        try:
            self.dependency_monitor.add_external_api(name, url, check_interval)
            logger.info(f"✅ External API registered for monitoring: {name} ({url})")
            return True
        except Exception as e:
            logger.error(f"❌ Error registering external API: {e}")
            return False
    
    def get_root_cause_analysis(self, event_ids: List[str] = None) -> Dict[str, Any]:
        """تحلیل علت اصلی مشکلات"""
        try:
            if not event_ids:
                # استفاده از رویدادهای اخیر
                recent_events = list(self.correlation_engine.events)[-10:]
                event_ids = [e['id'] for e in recent_events]
            
            events = []
            for event_id in event_ids[:5]:  # حداکثر 5 رویداد
                for event in self.correlation_engine.events:
                    if event['id'] == event_id:
                        events.append(event)
                        break
            
            # یافتن همبستگی‌های مرتبط
            related_correlations = []
            for corr in self.correlation_engine.correlations:
                if any(event_id in corr.events for event_id in event_ids):
                    related_correlations.append(corr)
            
            # تحلیل زنجیره رویدادها
            event_chain = self._build_event_chain(event_ids, events, related_correlations)
            
            # شناسایی root cause احتمالی
            root_cause = self._identify_root_cause_from_chain(event_chain)
            
            return {
                'analyzed_events': len(events),
                'related_correlations': len(related_correlations),
                'event_chain': event_chain,
                'probable_root_cause': root_cause,
                'recommended_actions': self.self_healing_advisor.get_recommendations(
                    root_cause.get('type', 'generic') if root_cause else 'generic',
                    0.7
                )[:3]  # 3 اقدام برتر
            }
            
        except Exception as e:
            logger.error(f"❌ Error in root cause analysis: {e}")
            return {'error': str(e)}
    
    def _build_event_chain(self, event_ids: List[str], events: List[Dict], correlations: List[CorrelationEvent]) -> List[Dict]:
        """ساخت زنجیره رویدادها"""
        chain = []
        
        # مرتب‌سازی بر اساس زمان
        events.sort(key=lambda x: x['timestamp'])
        
        for event in events:
            chain.append({
                'id': event['id'],
                'type': event['type'],
                'timestamp': event['timestamp'].isoformat(),
                'related_to': [c.events for c in correlations if event['id'] in c.events]
            })
        
        return chain
    
    def _identify_root_cause_from_chain(self, event_chain: List[Dict]) -> Optional[Dict[str, Any]]:
        """شناسایی root cause از زنجیره رویدادها"""
        if not event_chain:
            return None
        
        # اولین رویداد در زنجیره معمولاً root cause است
        first_event = event_chain[0]
        
        # یافتن رویداد با بیشترین ارتباطات
        most_connected = max(event_chain, key=lambda x: len(x['related_to']))
        
        return {
            'type': first_event['type'],
            'event_id': first_event['id'],
            'timestamp': first_event['timestamp'],
            'confidence': 'high' if len(most_connected['related_to']) > 2 else 'medium',
            'reasoning': 'First event in chain with multiple correlations' 
                        if most_connected == first_event else 
                        'Most connected event in correlation chain'
        }
    
    def trigger_self_healing_action(self, issue_type: str, severity: float = 0.7) -> Dict[str, Any]:
        """راه‌اندازی اقدام خود-ترمیمی"""
        try:
            # دریافت توصیه‌ها
            recommendations = self.self_healing_advisor.get_recommendations(issue_type, severity)
            
            if not recommendations:
                return {'success': False, 'message': 'No recommendations found for this issue'}
            
            # انتخاب بهترین توصیه (اولین مورد با بالاترین اولویت)
            best_recommendation = recommendations[0]
            
            # در این نسخه، فقط لاگ می‌کنیم. در نسخه واقعی، اقدام اجرا می‌شود
            logger.info(f"🔧 Executing self-healing action: {best_recommendation['recommendation']}")
            
            # ثبت اقدام
            action_record = self.self_healing_advisor.record_action(
                best_recommendation,
                success=True,
                notes="Action logged (simulation mode)"
            )
            
            return {
                'success': True,
                'action_id': action_record['action_id'],
                'recommendation': best_recommendation['recommendation'],
                'priority': best_recommendation['priority'],
                'notes': 'Action logged in simulation mode. Real execution would require manual implementation.'
            }
            
        except Exception as e:
            logger.error(f"❌ Error triggering self-healing action: {e}")
            return {'success': False, 'error': str(e)}
    
    # APIهای موجود (برای backward compatibility)
    
    def get_system_health(self) -> Dict[str, Any]:
        """دریافت سلامت کلی سیستم (موجود)"""
        if central_monitor:
            metrics = central_monitor.get_current_metrics()
            system_metrics = metrics.get('system', {})
        else:
            metrics = self.metrics_collector.get_current_metrics()
            system_metrics = metrics
        
        health_indicators = {
            'cpu': self._evaluate_cpu_health(system_metrics.get('cpu', {})),
            'memory': self._evaluate_memory_health(system_metrics.get('memory', {})),
            'disk': self._evaluate_disk_health(system_metrics.get('disk', {})),
            'network': self._evaluate_network_health(system_metrics.get('network', {})),
            'process': self._evaluate_process_health(system_metrics.get('process', {}))
        }
        
        overall_health = self._calculate_overall_system_health(health_indicators)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'health_indicators': health_indicators,
            'metrics_snapshot': {
                'cpu_usage': system_metrics.get('cpu', {}).get('percent', 0),
                'memory_usage': system_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': system_metrics.get('disk', {}).get('usage_percent', 0),
                'network_activity': f"↑{system_metrics.get('network', {}).get('mb_sent_per_sec', 0)}MB/s ↓{system_metrics.get('network', {}).get('mb_recv_per_sec', 0)}MB/s"
            }
        }
    
    def _evaluate_cpu_health(self, cpu_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت CPU"""
        usage = cpu_metrics.get('percent', 0)
        
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
        usage = memory_metrics.get('percent', 0)
        
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
            'used_gb': memory_metrics.get('used_gb', 0),
            'available_gb': memory_metrics.get('available_gb', 0),
            'total_gb': memory_metrics.get('total_gb', 0)
        }
    
    def _evaluate_disk_health(self, disk_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت دیسک"""
        usage = disk_metrics.get('usage_percent', 0)
        
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
            'used_gb': disk_metrics.get('used_gb', 0),
            'free_gb': disk_metrics.get('free_gb', 0),
            'total_gb': disk_metrics.get('total_gb', 0),
            'io_activity': {
                'read_mb_sec': disk_metrics.get('io_read_mb_per_sec', 0),
                'write_mb_sec': disk_metrics.get('io_write_mb_per_sec', 0)
            }
        }
    
    def _evaluate_network_health(self, network_metrics: Dict) -> Dict[str, Any]:
        """ارزیابی سلامت شبکه"""
        sent_speed = network_metrics.get('bytes_sent_mb', 0)
        recv_speed = network_metrics.get('bytes_recv_mb', 0)
        connections = network_metrics.get('connections', 0)
        
        if sent_speed > 100 or recv_speed > 100:
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
        memory_mb = process_metrics.get('memory_rss_mb', 0)
        cpu_percent = process_metrics.get('cpu_percent', 0)
        threads = process_metrics.get('threads_count', 0)
        
        issues = []
        
        if memory_mb > 1000:
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
        if central_monitor:
            metrics_history = central_monitor.get_metrics_history(seconds=hours*3600)
        else:
            metrics_history = self.metrics_collector.get_metrics_history(seconds=hours*3600)
        
        trends = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network_sent': [],
            'network_recv': []
        }
        
        for metric in metrics_history:
            system_metric = metric.get('system', metric)
            trends['cpu'].append(system_metric.get('cpu', {}).get('percent', 0))
            trends['memory'].append(system_metric.get('memory', {}).get('percent', 0))
            trends['disk'].append(system_metric.get('disk', {}).get('usage_percent', 0))
            trends['network_sent'].append(system_metric.get('network', {}).get('bytes_sent_mb', 0))
            trends['network_recv'].append(system_metric.get('network', {}).get('bytes_recv_mb', 0))
        
        return {
            'time_period_hours': hours,
            'data_points': len(metrics_history),
            'trends': trends,
            'timestamp': datetime.now().isoformat()
        }

class CentralMonitoringSystem:
    """سیستم نظارت متمرکز با قابلیت‌های پیشرفته"""
    
    def __init__(self, metrics_collector, alert_manager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # سیستم‌های پیشرفته
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.correlation_engine = CorrelationEngine()
        
        # تنظیمات متمرکز
        self.collection_interval = 30
        self.metrics_cache = {}
        self.cache_ttl = 30
        self.last_collection_time = None
        self.subscribers = {}
        self.is_monitoring = False
        self.monitor_task = None
        
        # تاریخچه هشدارها
        self.alert_cooldown = {}
        self.cooldown_period = 60
        
        # تاریخچه متریک‌ها
        self.metrics_history = deque(maxlen=500)
        
        # تنظیم global instance
        global central_monitor
        central_monitor = self
        
        logger.info("🎯 Central Monitoring System initialized with advanced features")
    
    async def start_monitoring(self):
        """شروع نظارت متمرکز - async version"""
        if self.is_monitoring:
            logger.warning("⚠️ Central monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._async_monitoring_loop())
        logger.info("🔄 Central monitoring started (async, interval: 30s)")
    
    async def stop_monitoring(self):
        """توقف نظارت متمرکز"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Central monitoring stopped")
    
    async def _async_monitoring_loop(self):
        """حلقه نظارت async"""
        logger.debug("🔁 Async central monitoring loop started")
        
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # جمع‌آوری متریک‌ها
                metrics = await self._collect_essential_metrics_async()
                
                # ذخیره در کش و تاریخچه
                self.metrics_cache = metrics
                self.last_collection_time = datetime.now()
                self.metrics_history.append(metrics)
                
                # تحلیل روند و ناهنجاری
                await self._analyze_metrics_async(metrics)
                
                # اطلاع‌رسانی به مشترکین
                self._notify_subscribers(metrics)
                
                # بررسی هشدارها
                self._check_and_trigger_alerts(metrics)
                
                execution_time = time.time() - start_time
                sleep_time = self._calculate_smart_sleep(metrics, execution_time)
                
                if execution_time > 1.5:
                    logger.warning(f"⚠️ Metrics collection took {execution_time:.2f}s, sleeping {sleep_time}s")
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Central monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_essential_metrics_async(self) -> Dict[str, Any]:
        """جمع‌آوری async متریک‌های ضروری"""
        timestamp = datetime.now()
        
        try:
            # اجرای جمع‌آوری در thread pool برای جلوگیری از block
            loop = asyncio.get_event_loop()
            
            # جمع‌آوری همزمان چند متریک
            cpu_future = loop.run_in_executor(None, psutil.cpu_percent, 0.05)
            memory_future = loop.run_in_executor(None, psutil.virtual_memory)
            
            cpu_percent = await cpu_future
            memory = await memory_future
            
            # متریک‌های غیرضروری فقط گاهی اوقات
            collect_disk = timestamp.second % 30 == 0
            collect_network = timestamp.second % 20 == 0
            
            disk_metrics = {}
            if collect_disk:
                disk_future = loop.run_in_executor(None, psutil.disk_usage, '/')
                disk = await disk_future
                disk_metrics = {
                    'usage_percent': disk.percent,
                    'used_gb': round(disk.used / (1024**3), 3),
                    'free_gb': round(disk.free / (1024**3), 3)
                }
            
            network_metrics = {}
            if collect_network:
                net_future = loop.run_in_executor(None, psutil.net_io_counters)
                net_io = await net_future
                network_metrics = {
                    'bytes_sent_mb': round(net_io.bytes_sent / (1024**2), 3),
                    'bytes_recv_mb': round(net_io.bytes_recv / (1024**2), 3)
                }
            
            # اطلاعات process
            process = psutil.Process()
            process_info = process.memory_info()
            
            return {
                'timestamp': timestamp.isoformat(),
                'system': {
                    'cpu': {'percent': cpu_percent},
                    'memory': {'percent': memory.percent},
                    'disk': disk_metrics,
                    'network': network_metrics,
                    'process': {
                        'memory_rss_mb': round(process_info.rss / (1024**2), 3),
                        'cpu_percent': process.cpu_percent(interval=0.05)
                    }
                },
                'collection_time': time.time(),
                'collection_duration': round(time.time() - timestamp.timestamp(), 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Error collecting metrics async: {e}")
            return self._get_fallback_metrics(timestamp)
    
    async def _analyze_metrics_async(self, metrics: Dict[str, Any]):
        """تحلیل async متریک‌ها"""
        try:
            system_metrics = metrics.get('system', {})
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            
            timestamp = datetime.now()
            
            # تحلیل روند
            self.trend_analyzer.add_data_point('cpu_usage', cpu_usage, timestamp)
            self.trend_analyzer.add_data_point('memory_usage', memory_usage, timestamp)
            
            # تشخیص ناهنجاری
            self.anomaly_detector.add_metric('cpu_usage', cpu_usage, timestamp)
            self.anomaly_detector.add_metric('memory_usage', memory_usage, timestamp)
            
            anomalies = self.anomaly_detector.detect_anomalies()
            for anomaly in anomalies:
                if anomaly.severity > 0.7:
                    # ثبت رویداد همبستگی
                    self.correlation_engine.add_event(
                        anomaly.anomaly_type.value,
                        {
                            'severity': anomaly.severity,
                            'metric_value': anomaly.metric_value,
                            'metrics': system_metrics
                        },
                        timestamp
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error analyzing metrics: {e}")
    
    def _check_and_trigger_alerts(self, metrics: Dict):
        """بررسی و ایجاد هشدارهای متمرکز"""
        try:
            cpu_usage = metrics['system']['cpu']['percent']
            memory_usage = metrics['system']['memory']['percent']
            
            # بررسی CPU
            self._check_cpu_alerts(cpu_usage, metrics)
            
            # بررسی Memory
            self._check_memory_alerts(memory_usage, metrics)
            
            # بررسی تحلیل روند
            self._check_trend_alerts(metrics)
                
        except Exception as e:
            logger.error(f"❌ Error checking alerts: {e}")
    
    def _check_cpu_alerts(self, cpu_usage: float, metrics: Dict):
        """بررسی هشدارهای CPU"""
        alert_key = f"cpu_{int(cpu_usage // 10)}"
        
        if self._is_in_cooldown(alert_key):
            return
        
        if cpu_usage > 90:
            self._trigger_alert('critical', 'cpu', f"CPU usage critically high: {cpu_usage}%", metrics)
            self._set_cooldown(alert_key, 30)
        elif cpu_usage > 80:
            self._trigger_alert('warning', 'cpu', f"CPU usage high: {cpu_usage}%", metrics)
            self._set_cooldown(alert_key, 60)
    
    def _check_memory_alerts(self, memory_usage: float, metrics: Dict):
        """بررسی هشدارهای Memory"""
        alert_key = f"memory_{int(memory_usage // 10)}"
        
        if self._is_in_cooldown(alert_key):
            return
            
        if memory_usage > 90:
            self._trigger_alert('critical', 'memory', f"Memory usage critically high: {memory_usage}%", metrics)
            self._set_cooldown(alert_key, 30)
        elif memory_usage > 85:
            self._trigger_alert('warning', 'memory', f"Memory usage high: {memory_usage}%", metrics)
            self._set_cooldown(alert_key, 60)
    
    def _check_trend_alerts(self, metrics: Dict):
        """بررسی هشدارهای مبتنی بر روند"""
        try:
            cpu_trend = self.trend_analyzer.analyze_trend('cpu_usage', 1)  # 1 ساعت اخیر
            memory_trend = self.trend_analyzer.analyze_trend('memory_usage', 1)
            
            # هشدار برای روند صعودی سریع
            if cpu_trend.get('trend_direction') == 'increasing' and cpu_trend.get('trend_strength', 0) > 0.5:
                if not self._is_in_cooldown('cpu_trend_alert'):
                    self._trigger_alert('warning', 'trend', 
                                      f"CPU usage increasing rapidly: +{cpu_trend['trend_strength']:.2f}/hr", 
                                      metrics)
                    self._set_cooldown('cpu_trend_alert', 300)  # 5 دقیقه
            
            if memory_trend.get('trend_direction') == 'increasing' and memory_trend.get('trend_strength', 0) > 0.3:
                if not self._is_in_cooldown('memory_trend_alert'):
                    self._trigger_alert('warning', 'trend',
                                      f"Memory usage increasing: +{memory_trend['trend_strength']:.2f}/hr",
                                      metrics)
                    self._set_cooldown('memory_trend_alert', 300)
                    
        except Exception as e:
            logger.error(f"❌ Error checking trend alerts: {e}")
    
    def _trigger_alert(self, level: str, category: str, message: str, metrics: Dict):
        """ایجاد هشدار متمرکز"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            level_enum = AlertLevel.CRITICAL if level == 'critical' else AlertLevel.WARNING
            
            self.alert_manager.create_alert(
                level=level_enum,
                alert_type=AlertType.SYSTEM,
                title=f"High {category.title()} Usage",
                message=message,
                source="central_monitor",
                data={
                    'usage_percent': metrics['system'].get(category, {}).get('percent', 0),
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
        """محاسبه خواب هوشمند"""
        base_interval = self.collection_interval
        
        cpu_usage = metrics['system']['cpu']['percent']
        
        if cpu_usage > 85:
            return 60
        elif cpu_usage > 75:
            return 45
        elif cpu_usage < 30:
            return 20
        
        if execution_time > 2:
            return base_interval + 10
        
        return base_interval
    
    def _notify_subscribers(self, metrics: Dict):
        """اطلاع‌رسانی به سیستم‌های مشترک"""
        for sub_name, callback in self.subscribers.items():
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"❌ Error notifying subscriber {sub_name}: {e}")
    
    def _get_fallback_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """متریک‌های جایگزین در صورت خطا"""
        return {
            'timestamp': timestamp.isoformat(),
            'system': {
                'cpu': {'percent': 0},
                'memory': {'percent': 0},
                'disk': {},
                'network': {},
                'process': {'memory_rss_mb': 0, 'cpu_percent': 0}
            },
            'collection_time': time.time(),
            'collection_duration': 0
        }
    
    # APIهای پیشرفته جدید
    
    def get_trend_analysis(self, metric_name: str = None) -> Dict[str, Any]:
        """دریافت تحلیل روند"""
        if metric_name:
            return self.trend_analyzer.get_metric_forecast(metric_name)
        else:
            return self.trend_analyzer.get_all_trends_summary()
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """دریافت گزارش ناهنجاری‌ها"""
        return self.anomaly_detector.get_anomaly_summary()
    
    def get_correlation_insights(self) -> Dict[str, Any]:
        """دریافت بینش‌های همبستگی"""
        return self.correlation_engine.get_correlations_summary()
    
    def get_advanced_metrics_report(self) -> Dict[str, Any]:
        """گزارش متریک‌های پیشرفته"""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': self.metrics_cache,
            'trend_analysis': self.get_trend_analysis(),
            'anomaly_report': self.get_anomaly_report(),
            'correlation_insights': self.get_correlation_insights(),
            'metrics_history_size': len(self.metrics_history),
            'subscribers_count': len(self.subscribers),
            'is_monitoring': self.is_monitoring
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
        """دریافت متریک‌های فعلی"""
        if not self.metrics_cache:
            return self._get_fallback_metrics(datetime.now())
        
        if (self.last_collection_time and 
            (datetime.now() - self.last_collection_time).total_seconds() > self.cache_ttl):
            logger.debug("📊 Cache expired, returning fallback")
            return self._get_fallback_metrics(datetime.now())
        
        return self.metrics_cache
    
    def get_metrics_history(self, seconds: int = 3600) -> List[Dict]:
        """دریافت تاریخچه متریک‌ها"""
        cutoff_time = time.time() - seconds
        return [
            m for m in self.metrics_history 
            if m.get('collection_time', 0) > cutoff_time
        ]
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """دریافت snapshot فعلی"""
        return {
            'cache_age_seconds': (
                (datetime.now() - self.last_collection_time).total_seconds() 
                if self.last_collection_time else None
            ),
            'subscribers_count': len(self.subscribers),
            'is_monitoring': self.is_monitoring,
            'last_alert_cooldowns': len(self.alert_cooldown),
            'metrics_history_size': len(self.metrics_history),
            'last_collection_time': self.last_collection_time.isoformat() if self.last_collection_time else None,
            'advanced_features': {
                'trend_analysis': True,
                'anomaly_detection': True,
                'correlation_engine': True
            }
        }


def initialize_central_monitoring(metrics_collector, alert_manager):
    """تابع راه‌اندازی برای main.py"""
    global central_monitor
    
    if central_monitor:
        logger.warning("⚠️ Central monitor already initialized")
        return central_monitor
    
    central_monitor = CentralMonitoringSystem(metrics_collector, alert_manager)
    return central_monitor

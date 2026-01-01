"""
مانیتورینگ پیشرفته عملکرد API
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class EndpointMetrics:
    """متریک‌های یک endpoint"""
    endpoint: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    last_updated: datetime = None

@dataclass
class PerformanceAlert:
    """هشدار عملکرد"""
    endpoint: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float

class APIPerformanceMonitor:
    """مانیتورینگ پیشرفته عملکرد API"""
    
    def __init__(self, metrics_collector, alert_manager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # ذخیره‌سازی متریک‌ها
        self.endpoint_metrics = {}
        self.response_times = defaultdict(lambda: deque(maxlen=1000))
        
        # thresholds
        self.performance_thresholds = {
            'response_time_warning': 500,   # 500ms
            'response_time_critical': 2000, # 2s
            'error_rate_warning': 5.0,      # 5%
            'error_rate_critical': 10.0,    # 10%
            'throughput_warning': 1000,     # 1000 req/min
            'throughput_critical': 5000     # 5000 req/min
        }
        
        # الگوهای دسترسی
        self.access_patterns = defaultdict(lambda: defaultdict(int))
        
        logger.info("✅ API Performance Monitor initialized")
    
    def register_endpoint(self, endpoint: str, method: str = "GET", 
                         expected_response_time: int = 200):
        """ثبت endpoint جدید برای مانیتورینگ"""
        if endpoint not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint] = EndpointMetrics(
                endpoint=endpoint,
                last_updated=datetime.now()
            )
            
            # ثبت در metrics_collector
            try:
                self.metrics_collector.register_api_endpoint(
                    endpoint, 
                    expected_response_time
                )
            except AttributeError:
                logger.warning(f"Metrics collector doesn't support API endpoint registration")
            
            logger.info(f"✅ Endpoint registered: {method} {endpoint}")
            return True
        return False
    
    def record_request(self, endpoint: str, response_time_ms: float, 
                      status_code: int, method: str = "GET",
                      user_agent: str = "", ip_address: str = ""):
        """ثبت درخواست API"""
        # ثبت در endpoint metrics
        if endpoint not in self.endpoint_metrics:
            self.register_endpoint(endpoint, method)
        
        metrics = self.endpoint_metrics[endpoint]
        metrics.total_requests += 1
        metrics.total_response_time += response_time_ms
        
        if 200 <= status_code < 400:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        metrics.min_response_time = min(metrics.min_response_time, response_time_ms)
        metrics.max_response_time = max(metrics.max_response_time, response_time_ms)
        metrics.last_updated = datetime.now()
        
        # ذخیره زمان پاسخ برای تحلیل
        self.response_times[endpoint].append(response_time_ms)
        
        # ثبت در metrics_collector
        try:
            self.metrics_collector.record_api_request(
                endpoint=endpoint,
                latency_ms=response_time_ms,
                status_code=status_code,
                user_agent=user_agent
            )
        except AttributeError:
            pass
        
        # ثبت الگوی دسترسی
        hour = datetime.now().hour
        self.access_patterns[endpoint][hour] += 1
        
        # بررسی performance
        self._check_performance(endpoint, response_time_ms, status_code)
        
        return True
    
    def _check_performance(self, endpoint: str, response_time_ms: float, status_code: int):
        """بررسی عملکرد endpoint"""
        metrics = self.endpoint_metrics.get(endpoint)
        if not metrics or metrics.total_requests < 10:
            return
        
        # بررسی زمان پاسخ
        if response_time_ms > self.performance_thresholds['response_time_critical']:
            self._trigger_alert(
                endpoint=endpoint,
                alert_type="response_time_critical",
                severity="critical",
                message=f"Critical response time: {response_time_ms}ms",
                metric_value=response_time_ms,
                threshold=self.performance_thresholds['response_time_critical']
            )
        elif response_time_ms > self.performance_thresholds['response_time_warning']:
            self._trigger_alert(
                endpoint=endpoint,
                alert_type="response_time_warning",
                severity="warning",
                message=f"High response time: {response_time_ms}ms",
                metric_value=response_time_ms,
                threshold=self.performance_thresholds['response_time_warning']
            )
        
        # بررسی نرخ خطا
        error_rate = (metrics.failed_requests / metrics.total_requests) * 100
        if error_rate > self.performance_thresholds['error_rate_critical']:
            self._trigger_alert(
                endpoint=endpoint,
                alert_type="error_rate_critical",
                severity="critical",
                message=f"Critical error rate: {error_rate:.1f}%",
                metric_value=error_rate,
                threshold=self.performance_thresholds['error_rate_critical']
            )
        elif error_rate > self.performance_thresholds['error_rate_warning']:
            self._trigger_alert(
                endpoint=endpoint,
                alert_type="error_rate_warning",
                severity="warning",
                message=f"High error rate: {error_rate:.1f}%",
                metric_value=error_rate,
                threshold=self.performance_thresholds['error_rate_warning']
            )
        
        # بررسی throughput
        requests_last_minute = self._get_requests_last_minute(endpoint)
        if requests_last_minute > self.performance_thresholds['throughput_critical']:
            self._trigger_alert(
                endpoint=endpoint,
                alert_type="throughput_critical",
                severity="critical",
                message=f"Critical throughput: {requests_last_minute} req/min",
                metric_value=requests_last_minute,
                threshold=self.performance_thresholds['throughput_critical']
            )
        elif requests_last_minute > self.performance_thresholds['throughput_warning']:
            self._trigger_alert(
                endpoint=endpoint,
                alert_type="throughput_warning",
                severity="warning",
                message=f"High throughput: {requests_last_minute} req/min",
                metric_value=requests_last_minute,
                threshold=self.performance_thresholds['throughput_warning']
            )
    
    def _get_requests_last_minute(self, endpoint: str) -> int:
        """دریافت تعداد درخواست‌های دقیقه اخیر"""
        # در این نسخه ساده، از الگوهای دسترسی استفاده می‌کنیم
        current_hour = datetime.now().hour
        return self.access_patterns[endpoint].get(current_hour, 0)
    
    def _trigger_alert(self, endpoint: str, alert_type: str, severity: str, 
                      message: str, metric_value: float, threshold: float):
        """ایجاد هشدار عملکرد"""
        try:
            from debug_system.core.alert_manager import AlertLevel, AlertType
            
            level = AlertLevel.CRITICAL if severity == "critical" else AlertLevel.WARNING
            
            self.alert_manager.create_alert(
                level=level,
                alert_type=AlertType.PERFORMANCE,
                title=f"API Performance Alert: {endpoint}",
                message=f"{message}. Threshold: {threshold}",
                source="api_performance_monitor",
                data={
                    'endpoint': endpoint,
                    'alert_type': alert_type,
                    'metric_value': metric_value,
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.warning(f"🚨 API Performance Alert: {endpoint} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Error triggering API performance alert: {e}")
    
    def get_endpoint_performance(self, endpoint: str) -> Dict[str, Any]:
        """دریافت عملکرد یک endpoint"""
        metrics = self.endpoint_metrics.get(endpoint)
        if not metrics:
            return {'error': f'Endpoint {endpoint} not found'}
        
        response_times = list(self.response_times[endpoint])
        
        return {
            'endpoint': endpoint,
            'total_requests': metrics.total_requests,
            'successful_requests': metrics.successful_requests,
            'failed_requests': metrics.failed_requests,
            'success_rate': (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0,
            'error_rate': (metrics.failed_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0,
            'average_response_time': metrics.total_response_time / metrics.total_requests if metrics.total_requests > 0 else 0,
            'min_response_time': metrics.min_response_time if metrics.min_response_time != float('inf') else 0,
            'max_response_time': metrics.max_response_time,
            'p95_response_time': self._calculate_percentile(response_times, 95),
            'p99_response_time': self._calculate_percentile(response_times, 99),
            'throughput_last_hour': sum(self.access_patterns[endpoint].values()),
            'access_pattern': dict(self.access_patterns[endpoint]),
            'last_updated': metrics.last_updated.isoformat() if metrics.last_updated else None
        }
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """محاسبه percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index % 1)
    
    def get_all_endpoints_performance(self) -> Dict[str, Any]:
        """دریافت عملکرد تمام endpoints"""
        performance = {}
        
        for endpoint in self.endpoint_metrics:
            performance[endpoint] = self.get_endpoint_performance(endpoint)
        
        # محاسبات کلی
        total_requests = sum(m.total_requests for m in self.endpoint_metrics.values())
        total_successful = sum(m.successful_requests for m in self.endpoint_metrics.values())
        total_response_time = sum(m.total_response_time for m in self.endpoint_metrics.values())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_endpoints': len(self.endpoint_metrics),
            'total_requests': total_requests,
            'overall_success_rate': (total_successful / total_requests * 100) if total_requests > 0 else 100,
            'average_response_time': total_response_time / total_requests if total_requests > 0 else 0,
            'endpoints': performance,
            'performance_summary': self._generate_performance_summary(performance)
        }
    
    def _generate_performance_summary(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """تولید خلاصه عملکرد"""
        slow_endpoints = []
        high_error_endpoints = []
        high_traffic_endpoints = []
        
        for endpoint, data in performance.items():
            if data.get('average_response_time', 0) > self.performance_thresholds['response_time_warning']:
                slow_endpoints.append({
                    'endpoint': endpoint,
                    'response_time': data['average_response_time']
                })
            
            if data.get('error_rate', 0) > self.performance_thresholds['error_rate_warning']:
                high_error_endpoints.append({
                    'endpoint': endpoint,
                    'error_rate': data['error_rate']
                })
            
            if data.get('total_requests', 0) > 1000:
                high_traffic_endpoints.append({
                    'endpoint': endpoint,
                    'requests': data['total_requests']
                })
        
        return {
            'slow_endpoints': sorted(slow_endpoints, key=lambda x: x['response_time'], reverse=True)[:5],
            'high_error_endpoints': sorted(high_error_endpoints, key=lambda x: x['error_rate'], reverse=True)[:5],
            'high_traffic_endpoints': sorted(high_traffic_endpoints, key=lambda x: x['requests'], reverse=True)[:5],
            'recommendations': self._generate_performance_recommendations(
                slow_endpoints, high_error_endpoints, high_traffic_endpoints
            )
        }
    
    def _generate_performance_recommendations(self, slow_endpoints: List, 
                                            high_error_endpoints: List, 
                                            high_traffic_endpoints: List) -> List[str]:
        """تولید توصیه‌های عملکرد"""
        recommendations = []
        
        if slow_endpoints:
            rec = "Optimize slow endpoints: " + ", ".join([e['endpoint'] for e in slow_endpoints[:3]])
            recommendations.append(rec)
        
        if high_error_endpoints:
            rec = "Fix high error rate endpoints: " + ", ".join([e['endpoint'] for e in high_error_endpoints[:3]])
            recommendations.append(rec)
        
        if high_traffic_endpoints:
            rec = "Consider scaling for high traffic endpoints: " + ", ".join([e['endpoint'] for e in high_traffic_endpoints[:3]])
            recommendations.append(rec)
        
        if not recommendations:
            recommendations.append("All endpoints performing within acceptable thresholds.")
        
        return recommendations
    
    def get_latency_analysis(self, endpoint: str = None, hours: int = 24) -> Dict[str, Any]:
        """تحلیل latency"""
        if endpoint:
            response_times = list(self.response_times.get(endpoint, []))
            
            if not response_times:
                return {'error': 'No data available'}
            
            return {
                'endpoint': endpoint,
                'data_points': len(response_times),
                'average': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'min': min(response_times),
                'max': max(response_times),
                'percentiles': {
                    'p50': self._calculate_percentile(response_times, 50),
                    'p90': self._calculate_percentile(response_times, 90),
                    'p95': self._calculate_percentile(response_times, 95),
                    'p99': self._calculate_percentile(response_times, 99)
                },
                'distribution': self._create_distribution(response_times),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # تحلیل کلی
            all_response_times = []
            for endpoint_times in self.response_times.values():
                all_response_times.extend(endpoint_times)
            
            if not all_response_times:
                return {'error': 'No data available'}
            
            return {
                'total_endpoints': len(self.response_times),
                'total_data_points': len(all_response_times),
                'overall_average': statistics.mean(all_response_times),
                'overall_median': statistics.median(all_response_times),
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_distribution(self, response_times: List[float]) -> Dict[str, int]:
        """ایجاد توزیع frequency"""
        if not response_times:
            return {}
        
        max_time = max(response_times)
        bins = [0, 100, 200, 500, 1000, 2000, 5000, float('inf')]
        distribution = {f"<={bins[i]}": 0 for i in range(1, len(bins))}
        
        for rt in response_times:
            for i in range(1, len(bins)):
                if rt <= bins[i]:
                    distribution[f"<={bins[i]}"] += 1
                    break
        
        return distribution
    
    def get_peak_usage_patterns(self) -> Dict[str, Any]:
        """الگوهای استفاده پیک"""
        patterns = {}
        
        for endpoint, hourly_counts in self.access_patterns.items():
            if hourly_counts:
                peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
                patterns[endpoint] = {
                    'peak_hour': peak_hour[0],
                    'peak_requests': peak_hour[1],
                    'total_daily_requests': sum(hourly_counts.values()),
                    'busy_hours': [hour for hour, count in hourly_counts.items() if count > peak_hour[1] * 0.5]
                }
        
        # شناسایی ساعت‌های پیک کلی
        all_hourly_counts = defaultdict(int)
        for hourly_counts in self.access_patterns.values():
            for hour, count in hourly_counts.items():
                all_hourly_counts[hour] += count
        
        if all_hourly_counts:
            overall_peak = max(all_hourly_counts.items(), key=lambda x: x[1])
        
        return {
            'endpoint_patterns': patterns,
            'overall_peak_hour': overall_peak[0] if all_hourly_counts else None,
            'overall_peak_requests': overall_peak[1] if all_hourly_counts else 0,
            'recommendations': self._generate_peak_usage_recommendations(patterns, all_hourly_counts)
        }
    
    def _generate_peak_usage_recommendations(self, patterns: Dict, all_hourly_counts: Dict) -> List[str]:
        """تولید توصیه‌های استفاده پیک"""
        recommendations = []
        
        # شناسایی endpoints با پیک‌های شدید
        for endpoint, pattern in patterns.items():
            if pattern['peak_requests'] > 1000:
                recommendations.append(
                    f"Endpoint {endpoint} has high peak usage ({pattern['peak_requests']} reqs). Consider load balancing."
                )
        
        # شناسایی ساعت‌های شلوغ
        busy_hours = [hour for hour, count in all_hourly_counts.items() if count > 1000]
        if busy_hours:
            rec = f"System experiences high load during hours: {sorted(busy_hours)}. Consider auto-scaling."
            recommendations.append(rec)
        
        if not recommendations:
            recommendations.append("Usage patterns are within normal ranges.")
        
        return recommendations
    
    def get_performance_forecast(self, endpoint: str, forecast_hours: int = 6) -> Dict[str, Any]:
        """پیش‌بینی عملکرد"""
        # این یک پیاده‌سازی ساده است. در نسخه واقعی از مدل‌های پیشرفته‌تر استفاده می‌شود.
        
        hourly_counts = self.access_patterns.get(endpoint, {})
        if not hourly_counts:
            return {'error': 'Insufficient data for forecasting'}
        
        # میانگین‌گیری ساده
        recent_hours = list(hourly_counts.values())[-24:]  # 24 ساعت اخیر
        if len(recent_hours) < 6:
            return {'error': 'Need at least 6 hours of data'}
        
        avg_requests = sum(recent_hours) / len(recent_hours)
        
        # پیش‌بینی ساده (میانگین متحرک)
        forecast = []
        current_hour = datetime.now().hour
        
        for i in range(forecast_hours):
            forecast_hour = (current_hour + i) % 24
            # استفاده از الگوی تاریخی اگر موجود باشد
            historical_avg = hourly_counts.get(forecast_hour, avg_requests)
            forecast.append({
                'hour': forecast_hour,
                'estimated_requests': round(historical_avg * 0.8 + avg_requests * 0.2),  # weighted average
                'confidence': 'medium'
            })
        
        return {
            'endpoint': endpoint,
            'forecast_hours': forecast_hours,
            'current_hourly_avg': avg_requests,
            'forecast': forecast,
            'recommendations': self._generate_forecast_recommendations(forecast)
        }
    
    def _generate_forecast_recommendations(self, forecast: List[Dict]) -> List[str]:
        """تولید توصیه‌های مبتنی بر پیش‌بینی"""
        recommendations = []
        
        peak_forecast = max(forecast, key=lambda x: x['estimated_requests'])
        
        if peak_forecast['estimated_requests'] > 1000:
            recommendations.append(
                f"High traffic forecasted for hour {peak_forecast['hour']}: "
                f"{peak_forecast['estimated_requests']} requests expected. Prepare resources."
            )
        
        # شناسایی ساعت‌های کم‌مصرف برای maintenance
        low_traffic_hours = [f for f in forecast if f['estimated_requests'] < 100]
        if low_traffic_hours:
            hours = [f['hour'] for f in low_traffic_hours]
            recommendations.append(
                f"Consider maintenance during low traffic hours: {hours}"
            )
        
        if not recommendations:
            recommendations.append("Traffic forecast is within normal ranges.")
        
        return recommendations
    
    def clear_old_data(self, days: int = 30):
        """پاک کردن داده‌های قدیمی"""
        cutoff = datetime.now() - timedelta(days=days)
        
        endpoints_to_remove = []
        for endpoint, metrics in self.endpoint_metrics.items():
            if metrics.last_updated and metrics.last_updated < cutoff:
                endpoints_to_remove.append(endpoint)
        
        for endpoint in endpoints_to_remove:
            del self.endpoint_metrics[endpoint]
            if endpoint in self.response_times:
                del self.response_times[endpoint]
            if endpoint in self.access_patterns:
                del self.access_patterns[endpoint]
        
        logger.info(f"🧹 Cleared old data for {len(endpoints_to_remove)} endpoints")
        return len(endpoints_to_remove)

# ایجاد نمونه گلوبال
api_performance_monitor = None

def initialize_api_performance_monitor(metrics_collector, alert_manager):
    """راه‌اندازی مانیتورینگ عملکرد API"""
    global api_performance_monitor
    api_performance_monitor = APIPerformanceMonitor(metrics_collector, alert_manager)
    return api_performance_monitor

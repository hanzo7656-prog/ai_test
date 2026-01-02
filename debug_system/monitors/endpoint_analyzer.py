"""
ENDPOINT ANALYZER
تحلیل‌گر تخصصی endpointها و درخواست‌ها
فقط تحلیل می‌کند - داده‌ها را از SystemMetricsCollector می‌گیرد
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class EndpointHealth:
    """سلامت endpoint"""
    endpoint: str
    health_score: float  # 0-100
    health_status: str  # healthy, warning, critical
    total_calls: int
    avg_response_time_ms: float
    error_rate_percent: float
    calls_per_minute: float
    last_called: str

@dataclass
class EndpointRecommendation:
    """توصیه بهینه‌سازی endpoint"""
    endpoint: str
    issue_type: str
    severity: str
    description: str
    recommendation: str

class EndpointAnalyzer:
    """
    تحلیل‌گر endpointها - فقط تحلیل می‌کند
    تمام داده‌ها را از SystemMetricsCollector دریافت می‌کند
    """
    
    def __init__(self, metrics_collector):
        """
        پارامترها:
            metrics_collector: نمونه SystemMetricsCollector
        """
        self.collector = metrics_collector
        self.endpoint_history = defaultdict(list)  # برای ذخیره تاریخچه response times
        self.endpoint_errors = defaultdict(list)   # برای ذخیره تاریخچه خطاها
        
        # آستانه‌های endpoint
        self.thresholds = {
            'response_time': {
                'excellent': 100,    # زیر 100ms
                'good': 300,         # زیر 300ms
                'fair': 1000,        # زیر 1s
                'poor': 3000,        # زیر 3s
                'critical': 5000     # بالای 5s
            },
            'error_rate': {
                'excellent': 0.1,    # 0.1%
                'good': 0.5,         # 0.5%
                'fair': 1.0,         # 1.0%
                'poor': 2.0,         # 2.0%
                'critical': 5.0      # 5.0%
            },
            'calls_per_minute': {
                'low': 1,
                'medium': 10,
                'high': 100,
                'very_high': 1000
            }
        }
        
        logger.info("✅ EndpointAnalyzer initialized - Pure analyzer")
    
    def analyze_endpoints(self) -> Dict[str, Any]:
        """
        تحلیل جامع endpointها
        
        Returns:
            Dict[str, Any]: گزارش تحلیل کامل
        """
        try:
            # دریافت متریک‌های فعلی
            metrics = self.collector.get_current_metrics()
            requests_metrics = metrics.get("requests", {})
            
            # تحلیل سلامت endpointها
            endpoint_health = self._analyze_endpoint_health(requests_metrics)
            
            # شناسایی مشکلات
            issues = self._identify_endpoint_issues(endpoint_health)
            
            # تحلیل روند
            trends = self._analyze_endpoint_trends()
            
            # تولید توصیه‌ها
            recommendations = self._generate_endpoint_recommendations(issues)
            
            # شناسایی endpointهای حیاتی
            critical_endpoints = self._identify_critical_endpoints(endpoint_health)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_endpoints": len(endpoint_health),
                "total_calls": requests_metrics.get("total_requests", 0),
                "calls_per_minute": requests_metrics.get("requests_per_minute", 0),
                "endpoint_health": [h.__dict__ for h in endpoint_health],
                "issues_count": len(issues),
                "issues": [i.__dict__ for i in issues],
                "recommendations": [r.__dict__ for r in recommendations],
                "critical_endpoints": critical_endpoints,
                "trends": trends,
                "top_endpoints": self._get_top_endpoints(requests_metrics)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in endpoint analysis: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "endpoint_health": [],
                "issues": [],
                "recommendations": []
            }
    
    def _analyze_endpoint_health(self, requests_metrics: Dict) -> List[EndpointHealth]:
        """تحلیل سلامت endpointها"""
        endpoint_health = []
        
        try:
            endpoint_counts = requests_metrics.get("endpoints", {})
            start_time_str = requests_metrics.get("start_time")
            
            if not endpoint_counts or not start_time_str:
                return endpoint_health
            
            start_time = datetime.fromisoformat(start_time_str)
            uptime_minutes = max(1, (datetime.now() - start_time).total_seconds() / 60)
            
            for endpoint, call_count in endpoint_counts.items():
                # محاسبه سلامت (ساده‌شده - در نسخه واقعی از تاریخچه استفاده می‌شود)
                health_score = self._calculate_endpoint_health_score(endpoint, call_count)
                
                # محاسبه میانگین زمان پاسخ (ساده‌شده)
                avg_response_time = self._estimate_response_time(endpoint)
                
                # محاسبه نرخ خطا (ساده‌شده)
                error_rate = self._calculate_error_rate(endpoint)
                
                # محاسبه calls per minute
                calls_per_minute = call_count / uptime_minutes
                
                endpoint_health.append(EndpointHealth(
                    endpoint=endpoint,
                    health_score=health_score,
                    health_status=self._score_to_health_status(health_score),
                    total_calls=call_count,
                    avg_response_time_ms=avg_response_time,
                    error_rate_percent=error_rate,
                    calls_per_minute=round(calls_per_minute, 2),
                    last_called=datetime.now().isoformat()  # در نسخه واقعی زمان واقعی
                ))
            
        except Exception as e:
            logger.error(f"❌ Error analyzing endpoint health: {e}")
        
        return endpoint_health
    
    def _calculate_endpoint_health_score(self, endpoint: str, call_count: int) -> float:
        """محاسبه امتیاز سلامت endpoint"""
        # منطق ساده - در نسخه واقعی پیچیده‌تر است
        base_score = 100
        
        # کاهش بر اساس تعداد خطاها
        error_count = len(self.endpoint_errors.get(endpoint, []))
        if error_count > 0:
            error_penalty = min(error_count * 5, 50)
            base_score -= error_penalty
        
        # افزایش بر اساس تعداد فراخوانی‌ها (endpointهای فعال سالم‌تر در نظر گرفته می‌شوند)
        if call_count > 100:
            base_score += 5  # bonus برای endpointهای پراستفاده
        elif call_count == 0:
            base_score = 50  # endpoint غیرفعال
        
        return max(0, min(100, base_score))
    
    def _estimate_response_time(self, endpoint: str) -> float:
        """تخمین زمان پاسخ endpoint"""
        # در نسخه واقعی از تاریخچه واقعی استفاده می‌شود
        # اینجا یک تخمین ساده:
        
        if not self.endpoint_history.get(endpoint):
            # مقدار پیش‌فرض بر اساس نوع endpoint
            if "/api/" in endpoint:
                return 150.0  # 150ms برای APIها
            elif "/health" in endpoint:
                return 50.0   # 50ms برای health check
            else:
                return 200.0  # 200ms پیش‌فرض
        
        # اگر تاریخچه وجود دارد، میانگین بگیر
        try:
            return statistics.mean(self.endpoint_history[endpoint][-10:])  # 10 مورد آخر
        except:
            return 200.0
    
    def _calculate_error_rate(self, endpoint: str) -> float:
        """محاسبه نرخ خطای endpoint"""
        error_count = len(self.endpoint_errors.get(endpoint, []))
        total_calls = sum(1 for _ in self.endpoint_history.get(endpoint, [])) + error_count
        
        if total_calls == 0:
            return 0.0
        
        return (error_count / total_calls) * 100
    
    def _score_to_health_status(self, score: float) -> str:
        """تبدیل امتیاز به وضعیت سلامت"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"
    
    def _identify_endpoint_issues(self, endpoint_health: List[EndpointHealth]) -> List[EndpointRecommendation]:
        """شناسایی مشکلات endpointها"""
        issues = []
        
        try:
            for health in endpoint_health:
                # بررسی زمان پاسخ بالا
                if health.avg_response_time_ms > self.thresholds['response_time']['critical']:
                    issues.append(EndpointRecommendation(
                        endpoint=health.endpoint,
                        issue_type="high_response_time",
                        severity="critical",
                        description=f"Response time too high: {health.avg_response_time_ms:.0f}ms",
                        recommendation="Optimize database queries or implement caching"
                    ))
                elif health.avg_response_time_ms > self.thresholds['response_time']['poor']:
                    issues.append(EndpointRecommendation(
                        endpoint=health.endpoint,
                        issue_type="high_response_time",
                        severity="high",
                        description=f"Response time high: {health.avg_response_time_ms:.0f}ms",
                        recommendation="Review query performance and add indexes"
                    ))
                
                # بررسی نرخ خطای بالا
                if health.error_rate_percent > self.thresholds['error_rate']['critical']:
                    issues.append(EndpointRecommendation(
                        endpoint=health.endpoint,
                        issue_type="high_error_rate",
                        severity="critical",
                        description=f"Error rate too high: {health.error_rate_percent:.1f}%",
                        recommendation="Investigate error patterns and improve error handling"
                    ))
                elif health.error_rate_percent > self.thresholds['error_rate']['poor']:
                    issues.append(EndpointRecommendation(
                        endpoint=health.endpoint,
                        issue_type="high_error_rate",
                        severity="high",
                        description=f"Error rate high: {health.error_rate_percent:.1f}%",
                        recommendation="Monitor error trends and fix common issues"
                    ))
                
                # بررسی endpointهای کم‌استفاده
                if health.total_calls < 10 and health.health_score < 60:
                    issues.append(EndpointRecommendation(
                        endpoint=health.endpoint,
                        issue_type="low_usage",
                        severity="low",
                        description=f"Low usage endpoint with poor health: {health.total_calls} calls",
                        recommendation="Consider removing or consolidating this endpoint"
                    ))
                
                # بررسی endpointهای بسیار پراستفاده
                if health.calls_per_minute > self.thresholds['calls_per_minute']['very_high']:
                    issues.append(EndpointRecommendation(
                        endpoint=health.endpoint,
                        issue_type="high_traffic",
                        severity="medium",
                        description=f"Very high traffic: {health.calls_per_minute:.1f} calls/minute",
                        recommendation="Consider load balancing and caching strategies"
                    ))
            
        except Exception as e:
            logger.error(f"❌ Error identifying endpoint issues: {e}")
        
        return issues
    
    def _analyze_endpoint_trends(self) -> Dict[str, Any]:
        """تحلیل روند endpointها"""
        # در نسخه ساده، تحلیل حداقلی ارائه می‌دهیم
        # در نسخه واقعی از تاریخچه استفاده می‌شود
        
        return {
            "analysis": "Endpoint trend analysis requires historical data",
            "recommendation": "Implement endpoint history tracking for detailed trend analysis",
            "available_data_points": sum(len(times) for times in self.endpoint_history.values())
        }
    
    def _generate_endpoint_recommendations(self, issues: List[EndpointRecommendation]) -> List[EndpointRecommendation]:
        """تولید توصیه‌های بهینه‌سازی endpoint"""
        recommendations = []
        
        try:
            # اضافه کردن issues به recommendations
            recommendations.extend(issues)
            
            # توصیه‌های عمومی
            if not recommendations:
                recommendations.append(EndpointRecommendation(
                    endpoint="all",
                    issue_type="general",
                    severity="info",
                    description="All endpoints are performing well",
                    recommendation="Continue monitoring and regular maintenance"
                ))
            else:
                # اضافه کردن توصیه کلی
                critical_issues = len([i for i in issues if i.severity == "critical"])
                if critical_issues > 0:
                    recommendations.append(EndpointRecommendation(
                        endpoint="system",
                        issue_type="multiple_critical_issues",
                        severity="critical",
                        description=f"{critical_issues} critical endpoint issues detected",
                        recommendation="Prioritize fixing critical endpoint issues immediately"
                    ))
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
        
        return recommendations
    
    def _identify_critical_endpoints(self, endpoint_health: List[EndpointHealth]) -> List[str]:
        """شناسایی endpointهای حیاتی"""
        critical_endpoints = []
        
        try:
            for health in endpoint_health:
                # endpointهای با سلامت بحرانی
                if health.health_status == "critical":
                    critical_endpoints.append(health.endpoint)
                
                # endpointهای حیاتی از نظر business
                elif self._is_business_critical(health.endpoint):
                    critical_endpoints.append(health.endpoint)
        
        except Exception as e:
            logger.error(f"❌ Error identifying critical endpoints: {e}")
        
        return list(set(critical_endpoints))  # حذف موارد تکراری
    
    def _is_business_critical(self, endpoint: str) -> bool:
        """بررسی حیاتی بودن endpoint از نظر business"""
        critical_patterns = [
            "/api/health",
            "/api/metrics",
            "/api/coins",
            "/api/exchanges",
            "/auth/",
            "/login"
        ]
        
        return any(pattern in endpoint for pattern in critical_patterns)
    
    def _get_top_endpoints(self, requests_metrics: Dict) -> List[Dict[str, Any]]:
        """دریافت endpointهای برتر"""
        endpoint_counts = requests_metrics.get("endpoints", {})
        
        if not endpoint_counts:
            return []
        
        # مرتب‌سازی بر اساس تعداد فراخوانی
        sorted_endpoints = sorted(
            endpoint_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_endpoints = []
        for endpoint, count in sorted_endpoints[:10]:  # 10 مورد اول
            top_endpoints.append({
                "endpoint": endpoint,
                "call_count": count,
                "percentage": round((count / requests_metrics.get("total_requests", 1)) * 100, 1)
            })
        
        return top_endpoints
    
    def record_endpoint_call(self, endpoint: str, response_time_ms: float, 
                           status_code: int):
        """
        ثبت فراخوانی endpoint برای تحلیل
        این داده‌ها برای تحلیل روند استفاده می‌شوند
        
        Args:
            endpoint: آدرس endpoint
            response_time_ms: زمان پاسخ به میلی‌ثانیه
            status_code: کد وضعیت HTTP
        """
        try:
            # ثبت زمان پاسخ
            self.endpoint_history[endpoint].append(response_time_ms)
            
            # حفظ فقط 1000 مورد آخر برای هر endpoint
            if len(self.endpoint_history[endpoint]) > 1000:
                self.endpoint_history[endpoint] = self.endpoint_history[endpoint][-1000:]
            
            # ثبت خطاها
            if status_code >= 400:
                self.endpoint_errors[endpoint].append({
                    "timestamp": datetime.now().isoformat(),
                    "status_code": status_code,
                    "response_time_ms": response_time_ms
                })
                
                # حفظ فقط 100 خطای آخر
                if len(self.endpoint_errors[endpoint]) > 100:
                    self.endpoint_errors[endpoint] = self.endpoint_errors[endpoint][-100:]
            
            # ثبت در collector (برای متریک‌های کلی)
            self.collector.record_request(endpoint)
            
            logger.debug(f"📊 Endpoint call recorded: {endpoint} - {response_time_ms}ms - {status_code}")
            
        except Exception as e:
            logger.error(f"❌ Error recording endpoint call: {e}")
    
    def get_endpoint_report(self, endpoint: str = None) -> Dict[str, Any]:
        """
        دریافت گزارش endpoint
        
        Args:
            endpoint: آدرس endpoint خاص (اگر None باشد، گزارش کلی)
        
        Returns:
            Dict[str, Any]: گزارش endpoint
        """
        if endpoint:
            return self._get_single_endpoint_report(endpoint)
        else:
            return self.analyze_endpoints()
    
    def _get_single_endpoint_report(self, endpoint: str) -> Dict[str, Any]:
        """گزارش endpoint خاص"""
        try:
            # تحلیل کلی
            analysis = self.analyze_endpoints()
            
            # پیدا کردن سلامت endpoint خاص
            endpoint_health = None
            for health in analysis.get("endpoint_health", []):
                if isinstance(health, dict) and health.get("endpoint") == endpoint:
                    endpoint_health = health
                    break
                elif isinstance(health, EndpointHealth) and health.endpoint == endpoint:
                    endpoint_health = health.__dict__
                    break
            
            if not endpoint_health:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": endpoint,
                    "error": "Endpoint not found in analysis"
                }
            
            # تاریخچه زمان پاسخ
            response_times = self.endpoint_history.get(endpoint, [])
            
            # تاریخچه خطاها
            errors = self.endpoint_errors.get(endpoint, [])
            
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "health": endpoint_health,
                "statistics": {
                    "response_time": {
                        "current": endpoint_health.get("avg_response_time_ms", 0),
                        "average": statistics.mean(response_times) if response_times else 0,
                        "min": min(response_times) if response_times else 0,
                        "max": max(response_times) if response_times else 0,
                        "data_points": len(response_times)
                    },
                    "errors": {
                        "total": len(errors),
                        "rate_percent": endpoint_health.get("error_rate_percent", 0),
                        "recent_errors": errors[-5:] if errors else []  # 5 خطای آخر
                    }
                },
                "performance_grade": self._calculate_performance_grade(endpoint_health),
                "recommendations": [
                    r.__dict__ for r in self._generate_endpoint_recommendations(
                        self._identify_endpoint_issues([EndpointHealth(**endpoint_health)])
                    )
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting endpoint report: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "error": str(e)
            }
    
    def _calculate_performance_grade(self, endpoint_health: Dict) -> str:
        """محاسبه گرید عملکرد endpoint"""
        score = endpoint_health.get("health_score", 0)
        
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def get_slow_endpoints(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        دریافت کندترین endpointها
        
        Args:
            limit: تعداد endpointها
        
        Returns:
            List[Dict[str, Any]]: لیست endpointهای کند
        """
        analysis = self.analyze_endpoints()
        endpoint_health = analysis.get("endpoint_health", [])
        
        # مرتب‌سازی بر اساس زمان پاسخ
        sorted_endpoints = sorted(
            endpoint_health,
            key=lambda x: x.get("avg_response_time_ms", 0) 
                         if isinstance(x, dict) else x.avg_response_time_ms,
            reverse=True
        )
        
        slow_endpoints = []
        for health in sorted_endpoints[:limit]:
            if isinstance(health, dict):
                slow_endpoints.append({
                    "endpoint": health.get("endpoint"),
                    "response_time_ms": health.get("avg_response_time_ms", 0),
                    "health_score": health.get("health_score", 0)
                })
            else:
                slow_endpoints.append({
                    "endpoint": health.endpoint,
                    "response_time_ms": health.avg_response_time_ms,
                    "health_score": health.health_score
                })
        
        return slow_endpoints

# نمونه گلوبال
endpoint_analyzer = None

def initialize_endpoint_analyzer(metrics_collector):
    """
    مقداردهی اولیه تحلیلگر endpointها
    
    Args:
        metrics_collector: نمونه SystemMetricsCollector
    
    Returns:
        EndpointAnalyzer: نمونه تحلیلگر
    """
    global endpoint_analyzer
    endpoint_analyzer = EndpointAnalyzer(metrics_collector)
    return endpoint_analyzer

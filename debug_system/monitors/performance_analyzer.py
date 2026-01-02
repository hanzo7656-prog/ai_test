"""
PERFORMANCE ANALYZER
تحلیل‌گر تخصصی عملکرد سیستم
فقط تحلیل می‌کند - داده‌ها را از SystemMetricsCollector می‌گیرد
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PerformanceGrade(Enum):
    """گریدهای عملکرد"""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"

@dataclass
class Bottleneck:
    """شناسایی bottleneck عملکرد"""
    type: str
    severity: str  # critical, high, medium, low
    metric: str
    current_value: float
    threshold: float
    description: str
    recommendation: str

class PerformanceAnalyzer:
    """
    تحلیل‌گر عملکرد - فقط تحلیل می‌کند
    تمام داده‌ها را از SystemMetricsCollector دریافت می‌کند
    """
    
    def __init__(self, metrics_collector):
        """
        پارامترها:
            metrics_collector: نمونه SystemMetricsCollector
        """
        self.collector = metrics_collector
        self.analysis_history = []
        
        # آستانه‌های عملکرد
        self.thresholds = {
            'cpu': {'warning': 70, 'critical': 85},
            'memory': {'warning': 75, 'critical': 85},
            'disk': {'warning': 85, 'critical': 95},
            'response_time': {'warning': 1000, 'critical': 3000},  # ms
            'error_rate': {'warning': 2.0, 'critical': 5.0}  # درصد
        }
        
        logger.info("✅ PerformanceAnalyzer initialized - Pure analyzer")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        تحلیل جامع عملکرد بر اساس آخرین متریک‌ها
        
        Returns:
            Dict[str, Any]: گزارش تحلیل کامل
        """
        try:
            # دریافت متریک‌های فعلی
            metrics = self.collector.get_current_metrics()
            
            # اجرای تحلیل‌ها
            bottlenecks = self._identify_bottlenecks(metrics)
            performance_grade = self._calculate_performance_grade(metrics)
            health_summary = self._generate_health_summary(metrics)
            recommendations = self._generate_recommendations(bottlenecks)
            
            # ذخیره تحلیل
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "performance_grade": performance_grade.value,
                "bottlenecks_count": len(bottlenecks),
                "bottlenecks": [b.__dict__ for b in bottlenecks],
                "health_summary": health_summary,
                "recommendations": recommendations,
                "metrics_snapshot": self._create_metrics_snapshot(metrics)
            }
            
            self.analysis_history.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error in performance analysis: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "performance_grade": "F",
                "bottlenecks": [],
                "recommendations": ["Check system metrics collection"]
            }
    
    def _identify_bottlenecks(self, metrics: Dict) -> List[Bottleneck]:
        """شناسایی bottlenecks عملکرد"""
        bottlenecks = []
        
        try:
            # 1. بررسی CPU bottleneck
            cpu_percent = metrics.get("system", {}).get("cpu_percent", 0)
            if cpu_percent > self.thresholds['cpu']['critical']:
                bottlenecks.append(Bottleneck(
                    type="cpu",
                    severity="critical",
                    metric="cpu_percent",
                    current_value=cpu_percent,
                    threshold=self.thresholds['cpu']['critical'],
                    description=f"CPU usage critically high: {cpu_percent}%",
                    recommendation="Scale up CPU resources or optimize CPU-intensive operations"
                ))
            elif cpu_percent > self.thresholds['cpu']['warning']:
                bottlenecks.append(Bottleneck(
                    type="cpu",
                    severity="high",
                    metric="cpu_percent",
                    current_value=cpu_percent,
                    threshold=self.thresholds['cpu']['warning'],
                    description=f"CPU usage high: {cpu_percent}%",
                    recommendation="Monitor CPU usage and consider optimization"
                ))
            
            # 2. بررسی Memory bottleneck
            memory_percent = metrics.get("system", {}).get("memory", {}).get("percent", 0)
            if memory_percent > self.thresholds['memory']['critical']:
                bottlenecks.append(Bottleneck(
                    type="memory",
                    severity="critical",
                    metric="memory_percent",
                    current_value=memory_percent,
                    threshold=self.thresholds['memory']['critical'],
                    description=f"Memory usage critically high: {memory_percent}%",
                    recommendation="Increase memory allocation or fix memory leaks"
                ))
            elif memory_percent > self.thresholds['memory']['warning']:
                bottlenecks.append(Bottleneck(
                    type="memory",
                    severity="high",
                    metric="memory_percent",
                    current_value=memory_percent,
                    threshold=self.thresholds['memory']['warning'],
                    description=f"Memory usage high: {memory_percent}%",
                    recommendation="Monitor memory usage and optimize memory allocation"
                ))
            
            # 3. بررسی Disk bottleneck
            disk_percent = metrics.get("system", {}).get("disk", {}).get("usage_percent", 0)
            if disk_percent > self.thresholds['disk']['critical']:
                bottlenecks.append(Bottleneck(
                    type="disk",
                    severity="critical",
                    metric="disk_usage_percent",
                    current_value=disk_percent,
                    threshold=self.thresholds['disk']['critical'],
                    description=f"Disk usage critically high: {disk_percent}%",
                    recommendation="Clean up disk space or increase storage capacity"
                ))
            elif disk_percent > self.thresholds['disk']['warning']:
                bottlenecks.append(Bottleneck(
                    type="disk",
                    severity="medium",
                    metric="disk_usage_percent",
                    current_value=disk_percent,
                    threshold=self.thresholds['disk']['warning'],
                    description=f"Disk usage high: {disk_percent}%",
                    recommendation="Monitor disk space and plan for cleanup"
                ))
            
            # 4. بررسی وابستگی‌ها
            dependencies = metrics.get("dependencies", {})
            
            # بررسی Redis
            redis_status = dependencies.get("redis", {}).get("status")
            if redis_status == "error" or redis_status == "unhealthy":
                bottlenecks.append(Bottleneck(
                    type="redis",
                    severity="critical" if redis_status == "error" else "high",
                    metric="redis_status",
                    current_value=0,
                    threshold=1,
                    description=f"Redis is {redis_status}",
                    recommendation="Check Redis connection and restart if necessary"
                ))
            
            # بررسی Cache
            cache_hit_rate = dependencies.get("cache", {}).get("hit_rate_percent", 100)
            if cache_hit_rate < 50:
                bottlenecks.append(Bottleneck(
                    type="cache",
                    severity="medium",
                    metric="cache_hit_rate",
                    current_value=cache_hit_rate,
                    threshold=50,
                    description=f"Cache hit rate low: {cache_hit_rate}%",
                    recommendation="Review cache strategy and TTL settings"
                ))
            
            # بررسی Database
            db_status = dependencies.get("database", {}).get("status")
            if db_status == "error":
                bottlenecks.append(Bottleneck(
                    type="database",
                    severity="critical",
                    metric="database_status",
                    current_value=0,
                    threshold=1,
                    description=f"Database error: {dependencies.get('database', {}).get('error', 'unknown')}",
                    recommendation="Check database connection and logs"
                ))
            
            # بررسی Coin API
            api_connected = dependencies.get("coin_api", {}).get("connected", False)
            if not api_connected:
                bottlenecks.append(Bottleneck(
                    type="external_api",
                    severity="high",
                    metric="api_connection",
                    current_value=0,
                    threshold=1,
                    description="Coin API disconnected",
                    recommendation="Check API connectivity and implement fallback mechanism"
                ))
            
        except Exception as e:
            logger.error(f"❌ Error identifying bottlenecks: {e}")
        
        return bottlenecks
    
    def _calculate_performance_grade(self, metrics: Dict) -> PerformanceGrade:
        """محاسبه گرید عملکرد کلی"""
        try:
            score = 100  # شروع از 100
            
            # کسر بر اساس CPU
            cpu_percent = metrics.get("system", {}).get("cpu_percent", 0)
            if cpu_percent > 90:
                score -= 30
            elif cpu_percent > 70:
                score -= 15
            elif cpu_percent > 50:
                score -= 5
            
            # کسر بر اساس Memory
            memory_percent = metrics.get("system", {}).get("memory", {}).get("percent", 0)
            if memory_percent > 90:
                score -= 25
            elif memory_percent > 75:
                score -= 12
            elif memory_percent > 60:
                score -= 5
            
            # کسر بر اساس Disk
            disk_percent = metrics.get("system", {}).get("disk", {}).get("usage_percent", 0)
            if disk_percent > 95:
                score -= 20
            elif disk_percent > 85:
                score -= 10
            
            # کسر بر اساس وابستگی‌ها
            dependencies = metrics.get("dependencies", {})
            unhealthy_deps = 0
            
            if dependencies.get("redis", {}).get("status") != "healthy":
                unhealthy_deps += 1
            
            if not dependencies.get("coin_api", {}).get("connected", False):
                unhealthy_deps += 1
            
            if dependencies.get("database", {}).get("status") == "error":
                unhealthy_deps += 1
            
            score -= unhealthy_deps * 15
            
            # تعیین گرید نهایی
            if score >= 90:
                return PerformanceGrade.A_PLUS
            elif score >= 80:
                return PerformanceGrade.A
            elif score >= 70:
                return PerformanceGrade.B
            elif score >= 60:
                return PerformanceGrade.C
            elif score >= 50:
                return PerformanceGrade.D
            else:
                return PerformanceGrade.F
                
        except Exception as e:
            logger.error(f"❌ Error calculating performance grade: {e}")
            return PerformanceGrade.F
    
    def _generate_health_summary(self, metrics: Dict) -> Dict[str, Any]:
        """تولید خلاصه سلامت"""
        health = self.collector.get_health_summary()
        
        # اضافه کردن جزئیات بیشتر
        system_metrics = metrics.get("system", {})
        dependencies = metrics.get("dependencies", {})
        
        return {
            "overall_health": "healthy" if health.get("overall", False) else "unhealthy",
            "components": {
                "system": {
                    "status": "healthy" if health.get("system", False) else "unhealthy",
                    "cpu_percent": system_metrics.get("cpu_percent", 0),
                    "memory_percent": system_metrics.get("memory", {}).get("percent", 0)
                },
                "redis": {
                    "status": "healthy" if health.get("redis", False) else "unhealthy",
                    "databases_connected": dependencies.get("redis", {}).get("databases_connected", 0)
                },
                "cache": {
                    "status": "healthy" if health.get("cache", False) else "unhealthy",
                    "hit_rate_percent": dependencies.get("cache", {}).get("hit_rate_percent", 0)
                },
                "database": {
                    "status": "healthy" if health.get("database", False) else "unhealthy",
                    "total_records": dependencies.get("database", {}).get("total_records", 0)
                },
                "coin_api": {
                    "status": "connected" if health.get("coin_api", False) else "disconnected",
                    "connected": dependencies.get("coin_api", {}).get("connected", False)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, bottlenecks: List[Bottleneck]) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        
        try:
            # تبدیل bottlenecks به توصیه‌ها
            for bottleneck in bottlenecks:
                if bottleneck.severity in ["critical", "high"]:
                    recommendations.append(bottleneck.recommendation)
            
            # توصیه‌های عمومی
            if not recommendations:
                recommendations.append("System performance is within acceptable ranges. Continue regular monitoring.")
            
            # اضافه کردن توصیه‌های پیشگیرانه
            if len([b for b in bottlenecks if b.type == "cpu"]) == 0:
                recommendations.append("Consider implementing auto-scaling for future load increases.")
            
            if len([b for b in bottlenecks if b.type == "cache"]) == 0:
                recommendations.append("Review cache strategy periodically for optimization opportunities.")
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            recommendations = ["Error generating recommendations"]
        
        return list(set(recommendations))  # حذف موارد تکراری
    
    def _create_metrics_snapshot(self, metrics: Dict) -> Dict[str, Any]:
        """ایجاد snapshot از متریک‌های کلیدی"""
        try:
            return {
                "system": {
                    "cpu_percent": round(metrics.get("system", {}).get("cpu_percent", 0), 1),
                    "memory_percent": round(metrics.get("system", {}).get("memory", {}).get("percent", 0), 1),
                    "disk_usage_percent": round(metrics.get("system", {}).get("disk", {}).get("usage_percent", 0), 1)
                },
                "dependencies": {
                    "redis_status": metrics.get("dependencies", {}).get("redis", {}).get("status", "unknown"),
                    "cache_hit_rate": round(metrics.get("dependencies", {}).get("cache", {}).get("hit_rate_percent", 0), 1),
                    "coin_api_connected": metrics.get("dependencies", {}).get("coin_api", {}).get("connected", False)
                },
                "requests": {
                    "total": metrics.get("requests", {}).get("total_requests", 0),
                    "per_minute": round(metrics.get("requests", {}).get("requests_per_minute", 0), 2)
                }
            }
        except Exception as e:
            logger.error(f"❌ Error creating metrics snapshot: {e}")
            return {"error": str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        دریافت گزارش عملکرد کامل
        
        Returns:
            Dict[str, Any]: گزارش شامل تحلیل فعلی و تاریخچه
        """
        current_analysis = self.analyze_performance()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "current_analysis": current_analysis,
            "history_summary": {
                "total_analyses": len(self.analysis_history),
                "recent_analyses": len([a for a in self.analysis_history 
                                      if datetime.fromisoformat(a["timestamp"]) > 
                                      datetime.now() - timedelta(hours=1)]),
                "average_grade": self._calculate_average_grade()
            },
            "collector_info": {
                "last_collection": self.collector.last_collection_time.isoformat() 
                                  if self.collector.last_collection_time else "never",
                "collection_interval": self.collector.collection_interval
            }
        }
    
    def _calculate_average_grade(self) -> str:
        """محاسبه میانگین گریدهای اخیر"""
        if not self.analysis_history:
            return "N/A"
        
        # فقط تحلیل‌های 1 ساعت اخیر
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_analyses = [
            a for a in self.analysis_history 
            if datetime.fromisoformat(a["timestamp"]) > one_hour_ago
        ]
        
        if not recent_analyses:
            recent_analyses = self.analysis_history[-10:]  # 10 مورد آخر
        
        grade_scores = {
            "A+": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0
        }
        
        total_score = 0
        for analysis in recent_analyses:
            grade = analysis.get("performance_grade", "F")
            total_score += grade_scores.get(grade, 0)
        
        avg_score = total_score / len(recent_analyses)
        
        if avg_score >= 4.5:
            return "A+"
        elif avg_score >= 3.5:
            return "A"
        elif avg_score >= 2.5:
            return "B"
        elif avg_score >= 1.5:
            return "C"
        elif avg_score >= 0.5:
            return "D"
        else:
            return "F"
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت سلامت سیستم به صورت سریع
        
        Returns:
            Dict[str, Any]: وضعیت سلامت
        """
        metrics = self.collector.get_current_metrics()
        health = self.collector.get_health_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy" if health.get("overall", False) else "unhealthy",
            "components": health,
            "critical_issues": len([b for b in self._identify_bottlenecks(metrics) 
                                  if b.severity == "critical"]),
            "performance_grade": self._calculate_performance_grade(metrics).value,
            "recommended_action": self._get_recommended_action(health)
        }
    
    def _get_recommended_action(self, health: Dict[str, bool]) -> str:
        """دریافت اقدام توصیه شده بر اساس سلامت"""
        if not health.get("overall", False):
            # پیدا کردن اولین کامپوننت ناسالم
            for component, status in health.items():
                if component != "overall" and not status:
                    return f"Fix {component} issues immediately"
            return "Investigate system issues"
        
        return "Continue monitoring - System is healthy"
    
    def track_endpoint_performance(self, endpoint: str, response_time_ms: float, 
                                 status_code: int):
        """
        ردیابی عملکرد endpoint خاص
        این داده در collector ثبت می‌شود و تحلیلگر فقط تحلیل می‌کند
        
        Args:
            endpoint: آدرس endpoint
            response_time_ms: زمان پاسخ به میلی‌ثانیه
            status_code: کد وضعیت HTTP
        """
        # فقط لاگ می‌کنیم - داده در collector ثبت می‌شود
        logger.debug(f"📊 Endpoint performance tracked: {endpoint} - {response_time_ms}ms - {status_code}")

# نمونه گلوبال
performance_analyzer = None

def initialize_performance_analyzer(metrics_collector):
    """
    مقداردهی اولیه تحلیلگر عملکرد
    
    Args:
        metrics_collector: نمونه SystemMetricsCollector
    
    Returns:
        PerformanceAnalyzer: نمونه تحلیلگر
    """
    global performance_analyzer
    performance_analyzer = PerformanceAnalyzer(metrics_collector)
    return performance_analyzer

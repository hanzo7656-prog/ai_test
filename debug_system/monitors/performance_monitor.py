import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, debug_manager, alert_manager):
        self.debug_manager = debug_manager
        self.alert_manager = alert_manager
        self.performance_baselines = {}
        self.response_time_thresholds = {
            'excellent': 0.5,    # زیر 500ms
            'good': 1.0,         # زیر 1s
            'fair': 2.0,         # زیر 2s
            'poor': 3.0,         # زیر 3s
            'unacceptable': 5.0  # بالای 5s
        }
        
        self.performance_history = deque(maxlen=1000)

    def analyze_endpoint_performance(self, endpoint: str = None) -> Dict[str, Any]:
        """آنالیز عملکرد اندپوینت"""
        if endpoint:
            return self._analyze_single_endpoint(endpoint)
        else:
            return self._analyze_all_endpoints()

    def _analyze_single_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """آنالیز عملکرد یک اندپوینت خاص"""
        stats = self.debug_manager.get_endpoint_stats(endpoint)
        
        if 'error' in stats:
            return {'error': stats['error']}

        performance_grade = self._calculate_performance_grade(stats)
        bottlenecks = self._identify_bottlenecks(stats, endpoint)
        
        return {
            'endpoint': endpoint,
            'performance_grade': performance_grade,
            'metrics': {
                'average_response_time': stats['average_response_time'],
                'success_rate': stats['success_rate'],
                'cache_hit_rate': stats['cache_performance']['hit_rate'],
                'total_calls': stats['total_calls'],
                'api_calls_per_request': stats['api_calls'] / stats['total_calls'] if stats['total_calls'] > 0 else 0
            },
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_recommendations(stats, bottlenecks),
            'last_updated': datetime.now().isoformat()
        }

    def _analyze_all_endpoints(self) -> Dict[str, Any]:
        """آنالیز عملکرد تمام اندپوینت‌ها"""
        all_stats = self.debug_manager.get_endpoint_stats()
        performance_report = {}
        
        for endpoint, stats in all_stats['endpoints'].items():
            performance_grade = self._calculate_performance_grade(stats)
            performance_report[endpoint] = {
                'performance_grade': performance_grade,
                'average_response_time': stats['average_response_time'],
                'success_rate': stats['success_rate'],
                'cache_hit_rate': stats['cache_performance']['hit_rate'],
                'total_calls': stats['total_calls']
            }

        # محاسبه آمار کلی
        response_times = [ep['average_response_time'] for ep in performance_report.values()]
        success_rates = [ep['success_rate'] for ep in performance_report.values()]
        
        return {
            'overall_performance': {
                'average_response_time': statistics.mean(response_times) if response_times else 0,
                'median_response_time': statistics.median(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'average_success_rate': statistics.mean(success_rates) if success_rates else 0
            },
            'endpoint_performance': performance_report,
            'performance_distribution': self._calculate_performance_distribution(performance_report),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_performance_grade(self, stats: Dict) -> str:
        """محاسبه گرید عملکرد با درنظرگیری نرمال‌سازی"""
        response_time = stats['average_response_time']
        success_rate = stats['success_rate']
        cache_hit_rate = stats['cache_performance']['hit_rate']
    
        # ✅ استفاده از داده‌های نرمال‌سازی از normalization_info
        norm_performance = stats.get('normalization_performance', {})
        norm_success_rate = norm_performance.get('success_rate', 100)
        norm_quality_score = norm_performance.get('avg_quality_score', 100)
    
        # محاسبه امتیاز
        score = 0
    
        # امتیاز زمان پاسخ (35%)
        if response_time <= self.response_time_thresholds['excellent']:
            score += 35
        elif response_time <= self.response_time_thresholds['good']:
            score += 28
        elif response_time <= self.response_time_thresholds['fair']:
            score += 21
        elif response_time <= self.response_time_thresholds['poor']:
            score += 14
        else:
            score += 7
          
        # امتیاز نرخ موفقیت (25%)
        score += (success_rate / 100) * 25
    
        # امتیاز نرخ کش (15%)
        score += (cache_hit_rate / 100) * 15
    
        # ✅ امتیاز نرمال‌سازی (25%)
        norm_score = (norm_success_rate / 100) * 12.5 + (norm_quality_score / 100) * 12.5
        score += norm_score
    
        # تعیین گرید
        if score >= 90: return 'A+'
        elif score >= 80: return 'A'
        elif score >= 70: return 'B'
        elif score >= 60: return 'C'
        elif score >= 50: return 'D'
        else: return 'F'

    def _identify_bottlenecks(self, stats: Dict, endpoint: str) -> List[Dict[str, Any]]:
        """شناسایی bottlenecks"""
        bottlenecks = []
        response_time = stats['average_response_time']
        success_rate = stats['success_rate']
        cache_hit_rate = stats['cache_performance']['hit_rate']
        api_calls_ratio = stats['api_calls'] / stats['total_calls'] if stats['total_calls'] > 0 else 0

        norm_performance = stats.get('normalization_performance', {})
        norm_success_rate = norm_performance.get('success_rate', 100)
        norm_quality_score = norm_performance.get('avg_quality_score', 100)

        # بررسی زمان پاسخ
        if response_time > self.response_time_thresholds['unacceptable']:
            bottlenecks.append({
                'type': 'response_time',
                'severity': 'critical',
                'message': f'Response time {response_time:.2f}s is unacceptable',
                'suggestion': 'Optimize database queries or implement caching'
            })
        elif response_time > self.response_time_thresholds['poor']:
            bottlenecks.append({
                'type': 'response_time', 
                'severity': 'high',
                'message': f'Response time {response_time:.2f}s is poor',
                'suggestion': 'Consider query optimization or adding indexes'
            })

        # بررسی نرخ موفقیت
        if success_rate < 95:
            bottlenecks.append({
                'type': 'reliability',
                'severity': 'high' if success_rate < 90 else 'medium',
                'message': f'Success rate {success_rate:.1f}% is below target',
                'suggestion': 'Investigate error patterns and improve error handling'
            })

        # بررسی کارایی کش
        if cache_hit_rate < 50 and stats['total_calls'] > 10:
            bottlenecks.append({
                'type': 'caching',
                'severity': 'medium',
                'message': f'Cache hit rate {cache_hit_rate:.1f}% is low',
                'suggestion': 'Review cache strategy and TTL settings'
            })

        if norm_success_rate < 90:
            bottlenecks.append({
                'type': 'normalization_reliability',
                'severity': 'high' if norm_success_rate < 80 else 'medium',
                'message': f'Normalization success rate {norm_success_rate:.1f}% is low',
                'suggestion': 'Review data normalization rules and error handling'
            })
            
        # بررسی فراخوانی‌های API
        if api_calls_ratio > 3:
            bottlenecks.append({
                'type': 'external_dependencies',
                'severity': 'medium', 
                'message': f'High API calls per request: {api_calls_ratio:.1f}',
                'suggestion': 'Implement request batching or caching for external APIs'
            })

        return bottlenecks

    def _generate_recommendations(self, stats: Dict, bottlenecks: List[Dict]) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        response_time = stats['average_response_time']
        cache_hit_rate = stats['cache_performance']['hit_rate']

        # توصیه‌های عمومی
        if response_time > 1.0:
            recommendations.append("Implement response compression")
            recommendations.append("Consider using a CDN for static assets")

        if cache_hit_rate < 60:
            recommendations.append("Increase cache TTL for frequently accessed data")
            recommendations.append("Implement cache warming for hot paths")

        if stats['total_calls'] > 1000:
            recommendations.append("Consider implementing rate limiting")
            recommendations.append("Add database connection pooling")

        # توصیه‌های مبتنی بر bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'response_time':
                recommendations.append(bottleneck['suggestion'])
            elif bottleneck['type'] == 'caching':
                recommendations.append(bottleneck['suggestion'])

        return list(set(recommendations))  # حذف موارد تکراری

    def _calculate_performance_distribution(self, performance_report: Dict) -> Dict[str, int]:
        """محاسبه توزیع عملکرد"""
        distribution = {
            'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0
        }
        
        for endpoint_data in performance_report.values():
            grade = endpoint_data['performance_grade']
            distribution[grade] += 1
            
        return distribution

    def get_slowest_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت کندترین اندپوینت‌ها"""
        all_stats = self.debug_manager.get_endpoint_stats()
        endpoints_with_times = []
        
        for endpoint, stats in all_stats['endpoints'].items():
            if stats['total_calls'] > 0:  # فقط اندپوینت‌های فعال
                endpoints_with_times.append({
                    'endpoint': endpoint,
                    'average_response_time': stats['average_response_time'],
                    'total_calls': stats['total_calls'],
                    'performance_grade': self._calculate_performance_grade(stats)
                })
        
        # مرتب‌سازی بر اساس زمان پاسخ
        sorted_endpoints = sorted(
            endpoints_with_times, 
            key=lambda x: x['average_response_time'], 
            reverse=True
        )
        
        return sorted_endpoints[:limit]

    def get_most_called_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت پرفراخوانی‌ترین اندپوینت‌ها"""
        all_stats = self.debug_manager.get_endpoint_stats()
        endpoints_with_calls = []
        
        for endpoint, stats in all_stats['endpoints'].items():
            endpoints_with_calls.append({
                'endpoint': endpoint,
                'total_calls': stats['total_calls'],
                'average_response_time': stats['average_response_time'],
                'success_rate': stats['success_rate']
            })
        
        # مرتب‌سازی بر اساس تعداد فراخوانی
        sorted_endpoints = sorted(
            endpoints_with_calls, 
            key=lambda x: x['total_calls'], 
            reverse=True
        )
        
        return sorted_endpoints[:limit]

    def track_performance_trend(self, endpoint: str, hours: int = 24) -> Dict[str, Any]:
        """ردیابی روند عملکرد یک اندپوینت"""
        recent_calls = self.debug_manager.get_recent_calls(limit=5000)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        endpoint_calls = [
            call for call in recent_calls 
            if call['endpoint'] == endpoint and 
            datetime.fromisoformat(call['timestamp']) >= cutoff_time
        ]
        
        if not endpoint_calls:
            return {'error': 'No data available for the specified period'}
        
        # گروه‌بندی بر اساس ساعت
        hourly_performance = defaultdict(list)
        for call in endpoint_calls:
            call_time = datetime.fromisoformat(call['timestamp'])
            hour_key = call_time.replace(minute=0, second=0, microsecond=0)
            hourly_performance[hour_key].append(call['response_time'])
        
        # محاسبه میانگین هر ساعت
        trend_data = []
        for hour, response_times in sorted(hourly_performance.items()):
            trend_data.append({
                'hour': hour.isoformat(),
                'average_response_time': statistics.mean(response_times),
                'call_count': len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            })
        
        return {
            'endpoint': endpoint,
            'time_period_hours': hours,
            'total_calls': len(endpoint_calls),
            'overall_average': statistics.mean([call['response_time'] for call in endpoint_calls]),
            'trend_data': trend_data,
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """تولید گزارش کامل عملکرد با اطلاعات نرمال‌سازی"""
        performance_overview = self.analyze_endpoint_performance()
        slowest_endpoints = self.get_slowest_endpoints(10)
        most_called_endpoints = self.get_most_called_endpoints(10)
    
        # ✅ جمع‌آوری متریک‌های نرمال‌سازی از endpointها
        normalization_metrics = []
        for endpoint, stats in performance_overview['endpoint_performance'].items():
            norm_perf = stats.get('normalization_performance', {})
            if norm_perf:
                normalization_metrics.append({
                    'endpoint': endpoint,
                    'success_rate': norm_perf.get('success_rate', 0),
                    'quality_score': norm_perf.get('avg_quality_score', 0),
                    'total_normalized': norm_perf.get('total_normalized', 0),
                    'normalization_errors': norm_perf.get('normalization_errors', 0)
                })
    
        # شناسایی اندپوینت‌های مشکل‌دار
        problematic_endpoints = []
        for endpoint in slowest_endpoints:
            if endpoint['performance_grade'] in ['D', 'F']:
                problematic_endpoints.append(endpoint)
    
        # ✅ شناسایی مشکلات نرمال‌سازی
        normalization_problems = [
            ep for ep in normalization_metrics 
            if ep['success_rate'] < 85 or ep['quality_score'] < 70
        ]
    
        return {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_endpoints_monitored': len(performance_overview['endpoint_performance']),
                'overall_performance_grade': self._calculate_overall_performance_grade(performance_overview),
                'problematic_endpoints_count': len(problematic_endpoints),
                # ✅ اضافه شد
                'normalization_issues_count': len(normalization_problems),
                'avg_normalization_success': statistics.mean([nm['success_rate'] for nm in normalization_metrics]) if normalization_metrics else 0,
                'avg_data_quality': statistics.mean([nm['quality_score'] for nm in normalization_metrics]) if normalization_metrics else 0
            },
            'performance_overview': performance_overview,
            'slowest_endpoints': slowest_endpoints,
            'most_called_endpoints': most_called_endpoints,
            'problematic_endpoints': problematic_endpoints,
            # ✅ اضافه شد
            'normalization_analysis': {
                'metrics': normalization_metrics,
                'problematic_endpoints': normalization_problems,
                'recommendations': self._generate_normalization_recommendations(normalization_problems)
            },
            'recommendations': self._generate_system_recommendations(performance_overview, problematic_endpoints)
        }

    def _calculate_overall_performance_grade(self, performance_overview: Dict) -> str:
        """محاسبه گرید عملکرد کلی سیستم"""
        grades = {
            'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0
        }
        
        total_score = 0
        count = 0
        
        for endpoint_data in performance_overview['endpoint_performance'].values():
            grade = endpoint_data['performance_grade']
            total_score += grades.get(grade, 0)
            count += 1
        
        if count == 0:
            return 'N/A'
            
        average_score = total_score / count
        
        if average_score >= 4.5:
            return 'A+'
        elif average_score >= 3.5:
            return 'A'
        elif average_score >= 2.5:
            return 'B'
        elif average_score >= 1.5:
            return 'C'
        elif average_score >= 0.5:
            return 'D'
        else:
            return 'F'

    def _generate_system_recommendations(self, performance_overview: Dict, problematic_endpoints: List) -> List[str]:
        """تولید توصیه‌های سیستمی"""
        recommendations = []
        overall_performance = performance_overview['overall_performance']
        
        # توصیه‌های مبتنی بر آمار کلی
        if overall_performance['average_response_time'] > 1.0:
            recommendations.append("Consider implementing a global caching layer")
            recommendations.append("Review database indexing strategy system-wide")
        
        if overall_performance['average_success_rate'] < 98:
            recommendations.append("Implement comprehensive error handling and retry mechanisms")
            recommendations.append("Add circuit breaker pattern for external dependencies")
        
        # توصیه‌های مبتنی بر اندپوینت‌های مشکل‌دار
        if problematic_endpoints:
            recommendations.append(f"Focus optimization efforts on {len(problematic_endpoints)} problematic endpoints")
            recommendations.append("Consider implementing auto-scaling for high-traffic endpoints")
        
        # توصیه‌های مبتنی بر توزیع عملکرد
        distribution = performance_overview['performance_distribution']
        if distribution['F'] + distribution['D'] > len(performance_overview['endpoint_performance']) * 0.2:
            recommendations.append("Conduct system-wide performance audit")
            recommendations.append("Consider architectural improvements for low-performing endpoints")
        
        return recommendations


    def _generate_normalization_recommendations(self, normalization_metrics: List[Dict]) -> List[str]:
        """تولید توصیه‌های مربوط به نرمال‌سازی"""
        recommendations = []
    
        if not normalization_metrics:
            return ["📊 No normalization data available for analysis"]
    
        # تحلیل متریک‌های نرمال‌سازی
        success_rates = [nm.get('success_rate', 0) for nm in normalization_metrics]
        quality_scores = [nm.get('quality_score', 0) for nm in normalization_metrics]
    
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
        # تولید توصیه‌ها بر اساس آمار
        if avg_success < 85:
            recommendations.append("🔄 Normalization success rate is low - Review data patterns")
    
        if avg_quality < 70:
            recommendations.append("📊 Data quality needs improvement - Check validation rules")
    
        # شناسایی endpointهای مشکل‌دار
        problematic_endpoints = [
            nm for nm in normalization_metrics 
            if nm.get('success_rate', 100) < 80 or nm.get('quality_score', 100) < 60
        ]
    
        if problematic_endpoints:
            ep_names = [ep['endpoint'] for ep in problematic_endpoints[:3]]
            recommendations.append(f"🎯 Focus on endpoints: {', '.join(ep_names)}")
    
        if not recommendations:
            recommendations.append("✅ Normalization system is performing well")
    
        return recommendations

    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """آنالیز bottlenecks در اندپوینت‌ها"""
        all_stats = self.debug_manager.get_endpoint_stats()
        bottlenecks = []
        
        for endpoint, stats in all_stats['endpoints'].items():
            issues = []
            
            # بررسی زمان پاسخ بالا
            if stats.get('average_response_time', 0) > 2.0:
                issues.append({
                    'type': 'slow_response',
                    'severity': 'high',
                    'message': f'Average response time {stats["average_response_time"]}s exceeds threshold'
                })
            
            # بررسی نرخ خطای بالا
            if stats.get('success_rate', 100) < 95:
                issues.append({
                    'type': 'high_error_rate',
                    'severity': 'medium',
                    'message': f'Success rate {stats["success_rate"]}% below threshold'
                })
            
            # بررسی نرخ کش پایین
            cache_hit_rate = stats.get('cache_performance', {}).get('hit_rate', 0)
            if cache_hit_rate < 50 and stats.get('total_calls', 0) > 10:
                issues.append({
                    'type': 'low_cache_efficiency',
                    'severity': 'low',
                    'message': f'Cache hit rate {cache_hit_rate}% is low'
                })
            
            if issues:
                bottlenecks.append({
                    'endpoint': endpoint,
                    'issues': issues,
                    'total_calls': stats.get('total_calls', 0)
                })
        
        return sorted(bottlenecks, key=lambda x: len(x['issues']), reverse=True)

# ایجاد نمونه گلوبال (بعداً در main.py مقداردهی می‌شود)
performance_monitor = None

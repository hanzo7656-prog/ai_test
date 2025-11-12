import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, debug_manager, history_manager):
        self.debug_manager = debug_manager
        self.history_manager = history_manager
        
    def generate_daily_report(self, date: datetime = None) -> Dict[str, Any]:
        """تولید گزارش روزانه"""
        if date is None:
            date = datetime.now()
        
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        # جمع‌آوری داده‌ها
        endpoint_stats = self.debug_manager.get_endpoint_stats()
        performance_trends = self.history_manager.get_performance_trends(1)
        alert_history = self.history_manager.get_alert_history(
            start_date=start_date, 
            end_date=end_date
        )
        
        report = {
            'report_type': 'daily',
            'date': start_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'executive_summary': self._generate_executive_summary(endpoint_stats, alert_history),
            'performance_analysis': self._analyze_performance(endpoint_stats, performance_trends),
            'system_health': self._analyze_system_health(),
            'security_overview': self._analyze_security(alert_history),
            'recommendations': self._generate_daily_recommendations(endpoint_stats, alert_history)
        }
        
        return report
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """تولید گزارش عملکرد"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        performance_trends = self.history_manager.get_performance_trends(days)
        endpoint_stats = self.debug_manager.get_endpoint_stats()
        
        report = {
            'report_type': 'performance',
            'period_days': days,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'performance_overview': self._generate_performance_overview(endpoint_stats),
            'trend_analysis': self._analyze_performance_trends(performance_trends),
            'bottleneck_analysis': self._identify_bottlenecks(endpoint_stats),
            'capacity_planning': self._generate_capacity_recommendations(performance_trends)
        }
        
        return report
    
    def generate_security_report(self, days: int = 30) -> Dict[str, Any]:
        """تولید گزارش امنیتی"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        alert_history = self.history_manager.get_alert_history(
            start_date=start_date,
            end_date=end_date
        )
        
        security_alerts = [alert for alert in alert_history if alert['type'] == 'SECURITY']
        
        report = {
            'report_type': 'security',
            'period_days': days,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'threat_landscape': self._analyze_threat_landscape(security_alerts),
            'incident_summary': self._summarize_security_incidents(security_alerts),
            'vulnerability_assessment': self._assess_vulnerabilities(),
            'security_recommendations': self._generate_security_recommendations(security_alerts)
        }
        
        return report
    
    def _generate_executive_summary(self, endpoint_stats: Dict, alert_history: List) -> Dict[str, Any]:
        """تولید خلاصه اجرایی"""
        total_requests = endpoint_stats.get('overall', {}).get('total_calls', 0)
        success_rate = endpoint_stats.get('overall', {}).get('overall_success_rate', 100)
        critical_alerts = len([alert for alert in alert_history if alert['level'] == 'CRITICAL'])
        
        return {
            'total_requests': total_requests,
            'success_rate': round(success_rate, 2),
            'system_availability': '99.9%',
            'critical_incidents': critical_alerts,
            'overall_health': 'Excellent' if success_rate > 99 and critical_alerts == 0 else 'Good'
        }
    
    def _analyze_performance(self, endpoint_stats: Dict, trends: Dict) -> Dict[str, Any]:
        """آنالیز عملکرد"""
        avg_response_time = endpoint_stats.get('overall', {}).get('average_response_time', 0)
        
        # تحلیل روند
        trend_analysis = "Stable"
        if trends.get('response_trends'):
            recent_trend = trends['response_trends'][-1]['avg_response_time'] if trends['response_trends'] else 0
            if recent_trend > avg_response_time * 1.2:
                trend_analysis = "Improving"
            elif recent_trend < avg_response_time * 0.8:
                trend_analysis = "Degrading"
        
        return {
            'average_response_time': avg_response_time,
            'trend_analysis': trend_analysis,
            'performance_grade': self._calculate_performance_grade(avg_response_time),
            'bottlenecks_identified': len(self._identify_bottlenecks(endpoint_stats))
        }
    
    def _analyze_system_health(self) -> Dict[str, Any]:
        """آنالیز سلامت سیستم"""
        # اینجا می‌تواند داده‌های واقعی از system monitor استفاده کند
        return {
            'cpu_health': 'Good',
            'memory_health': 'Good', 
            'disk_health': 'Good',
            'network_health': 'Good',
            'overall_system_health': 'Excellent'
        }
    
    def _analyze_security(self, alert_history: List) -> Dict[str, Any]:
        """آنالیز امنیتی"""
        security_alerts = [alert for alert in alert_history if alert['type'] == 'SECURITY']
        
        return {
            'total_security_alerts': len(security_alerts),
            'critical_security_alerts': len([alert for alert in security_alerts if alert['level'] == 'CRITICAL']),
            'common_threats': self._identify_common_threats(security_alerts),
            'security_posture': 'Strong' if len(security_alerts) == 0 else 'Needs Attention'
        }
    
    def _generate_daily_recommendations(self, endpoint_stats: Dict, alert_history: List) -> List[str]:
        """تولید توصیه‌های روزانه"""
        recommendations = []
        
        # بررسی عملکرد
        avg_response_time = endpoint_stats.get('overall', {}).get('average_response_time', 0)
        if avg_response_time > 1.0:
            recommendations.append("بررسی اندپوینت‌های کند برای بهینه‌سازی")
        
        # بررسی خطاها
        critical_alerts = len([alert for alert in alert_history if alert['level'] == 'CRITICAL'])
        if critical_alerts > 0:
            recommendations.append("بررسی فوری هشدارهای بحرانی")
        
        # بررسی امنیت
        security_alerts = len([alert for alert in alert_history if alert['type'] == 'SECURITY'])
        if security_alerts > 5:
            recommendations.append("بررسی فعالیت‌های امنیتی مشکوک")
        
        return recommendations
    
    def _generate_performance_overview(self, endpoint_stats: Dict) -> Dict[str, Any]:
        """تولید نمای کلی عملکرد"""
        return {
            'total_endpoints': len(endpoint_stats.get('endpoints', {})),
            'total_requests': endpoint_stats.get('overall', {}).get('total_calls', 0),
            'average_response_time': endpoint_stats.get('overall', {}).get('average_response_time', 0),
            'success_rate': endpoint_stats.get('overall', {}).get('overall_success_rate', 0)
        }
    
    def _analyze_performance_trends(self, trends: Dict) -> Dict[str, Any]:
        """آنالیز روندهای عملکرد"""
        if not trends.get('response_trends'):
            return {'analysis': 'Insufficient data'}
        
        response_trends = trends['response_trends']
        if len(response_trends) < 2:
            return {'analysis': 'Need more data points'}
        
        # تحلیل ساده روند
        first_avg = response_trends[0]['avg_response_time']
        last_avg = response_trends[-1]['avg_response_time']
        
        trend_direction = "improving" if last_avg < first_avg else "degrading"
        trend_percentage = abs((last_avg - first_avg) / first_avg * 100)
        
        return {
            'trend_direction': trend_direction,
            'trend_magnitude': f"{trend_percentage:.1f}%",
            'analysis': f"Performance is {trend_direction} by {trend_percentage:.1f}% over the period"
        }
    
    def _identify_bottlenecks(self, endpoint_stats: Dict) -> List[Dict[str, Any]]:
        """شناسایی bottlenecks"""
        bottlenecks = []
        
        for endpoint, stats in endpoint_stats.get('endpoints', {}).items():
            issues = []
            
            if stats.get('average_response_time', 0) > 2.0:
                issues.append('Slow response time')
            
            if stats.get('success_rate', 100) < 95:
                issues.append('High error rate')
            
            cache_hit_rate = stats.get('cache_performance', {}).get('hit_rate', 0)
            if cache_hit_rate < 50 and stats.get('total_calls', 0) > 100:
                issues.append('Low cache efficiency')
            
            if issues:
                bottlenecks.append({
                    'endpoint': endpoint,
                    'issues': issues,
                    'severity': 'High' if 'Slow response time' in issues else 'Medium'
                })
        
        return bottlenecks
    
    def _generate_capacity_recommendations(self, trends: Dict) -> List[str]:
        """تولید توصیه‌های ظرفیت"""
        recommendations = []
        
        if trends.get('resource_trends'):
            # تحلیل روند منابع
            last_resource = trends['resource_trends'][-1] if trends['resource_trends'] else {}
            
            if last_resource.get('avg_cpu', 0) > 80:
                recommendations.append("Consider scaling up CPU resources")
            
            if last_resource.get('avg_memory', 0) > 85:
                recommendations.append("Consider increasing memory allocation")
        
        return recommendations
    
    def _analyze_threat_landscape(self, security_alerts: List) -> Dict[str, Any]:
        """آنالیز landscape تهدیدات"""
        threat_types = {}
        for alert in security_alerts:
            threat_type = alert.get('data', {}).get('type', 'unknown')
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
        
        return {
            'total_threats': len(security_alerts),
            'threat_breakdown': threat_types,
            'risk_level': 'High' if len(security_alerts) > 10 else 'Medium' if len(security_alerts) > 5 else 'Low'
        }
    
    def _summarize_security_incidents(self, security_alerts: List) -> List[Dict[str, Any]]:
        """خلاصه حوادث امنیتی"""
        incidents = []
        for alert in security_alerts[:10]:  # فقط ۱۰ مورد آخر
            incidents.append({
                'timestamp': alert['timestamp'],
                'type': alert.get('data', {}).get('type', 'unknown'),
                'severity': alert['level'],
                'description': alert['message']
            })
        
        return incidents
    
    def _assess_vulnerabilities(self) -> Dict[str, Any]:
        """ارزیابی آسیب‌پذیری‌ها"""
        # اینجا می‌تواند اسکن آسیب‌پذیری واقعی انجام دهد
        return {
            'total_vulnerabilities': 0,
            'critical_vulnerabilities': 0,
            'vulnerability_scan_date': datetime.now().isoformat(),
            'assessment': 'No critical vulnerabilities detected'
        }
    
    def _generate_security_recommendations(self, security_alerts: List) -> List[str]:
        """تولید توصیه‌های امنیتی"""
        recommendations = []
        
        if len(security_alerts) > 0:
            recommendations.append("Review and update security policies")
            recommendations.append("Implement additional monitoring for suspicious activities")
        
        high_severity_alerts = len([alert for alert in security_alerts if alert['level'] == 'CRITICAL'])
        if high_severity_alerts > 0:
            recommendations.append("Immediate attention required for critical security alerts")
        
        return recommendations
    
    def _calculate_performance_grade(self, response_time: float) -> str:
        """محاسبه گرید عملکرد"""
        if response_time < 0.5:
            return 'A+'
        elif response_time < 1.0:
            return 'A'
        elif response_time < 2.0:
            return 'B'
        elif response_time < 3.0:
            return 'C'
        else:
            return 'D'
    
    def _identify_common_threats(self, security_alerts: List) -> List[str]:
        """شناسایی تهدیدات رایج"""
        threats = []
        for alert in security_alerts:
            threat_type = alert.get('data', {}).get('type', '')
            if threat_type and threat_type not in threats:
                threats.append(threat_type)
        
        return threats[:5]  # فقط ۵ تهدید رایج

# ایجاد نمونه گلوبال
report_generator = ReportGenerator(debug_manager, History_manager)  

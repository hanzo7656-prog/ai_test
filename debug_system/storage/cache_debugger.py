import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class CacheDebugger:
    def __init__(self):
        self.cache_operations = deque(maxlen=10000)
        self.cache_stats = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'total_size': 0,
            'last_operation': None
        })
        
    def log_cache_operation(self, operation: str, key: str, success: bool, 
                          response_time: float, size: int = 0):
        """ثبت عملیات کش"""
        operation_data = {
            'operation': operation,
            'key': key,
            'success': success,
            'response_time': response_time,
            'size': size,
            'timestamp': datetime.now().isoformat()
        }
        
        self.cache_operations.append(operation_data)
        
        # آپدیت آمار
        stats = self.cache_stats[key]
        stats['last_operation'] = datetime.now().isoformat()
        
        if operation == 'GET':
            if success:
                stats['hits'] += 1
            else:
                stats['misses'] += 1
        elif operation == 'SET':
            stats['sets'] += 1
            stats['total_size'] += size
        elif operation == 'DELETE':
            stats['deletes'] += 1
            stats['total_size'] = max(0, stats['total_size'] - size)
    
    def get_cache_stats(self, key: str = None) -> Dict[str, Any]:
        """دریافت آمار کش"""
        if key:
            if key not in self.cache_stats:
                return {'error': 'Key not found'}
            return self.cache_stats[key]
        
        # آمار کلی
        total_stats = {
            'total_keys': len(self.cache_stats),
            'total_hits': sum(stats['hits'] for stats in self.cache_stats.values()),
            'total_misses': sum(stats['misses'] for stats in self.cache_stats.values()),
            'total_sets': sum(stats['sets'] for stats in self.cache_stats.values()),
            'total_size': sum(stats['total_size'] for stats in self.cache_stats.values()),
            'hit_rate': 0
        }
        
        total_operations = total_stats['total_hits'] + total_stats['total_misses']
        if total_operations > 0:
            total_stats['hit_rate'] = (total_stats['total_hits'] / total_operations) * 100
        
        return total_stats
    
    def get_cache_performance(self, hours: int = 24) -> Dict[str, Any]:
        """دریافت عملکرد کش"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_operations = [
            op for op in self.cache_operations
            if datetime.fromisoformat(op['timestamp']) >= cutoff_time
        ]
        
        performance_data = {
            'total_operations': len(recent_operations),
            'operations_by_type': defaultdict(int),
            'average_response_time': 0,
            'success_rate': 0
        }
        
        total_response_time = 0
        successful_operations = 0
        
        for op in recent_operations:
            performance_data['operations_by_type'][op['operation']] += 1
            total_response_time += op['response_time']
            if op['success']:
                successful_operations += 1
        
        if recent_operations:
            performance_data['average_response_time'] = total_response_time / len(recent_operations)
            performance_data['success_rate'] = (successful_operations / len(recent_operations)) * 100
        
        return performance_data
    
    def get_most_accessed_keys(self, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت کلیدهای پر دسترس"""
        keys_with_access = []
        
        for key, stats in self.cache_stats.items():
            total_access = stats['hits'] + stats['misses']
            if total_access > 0:
                keys_with_access.append({
                    'key': key,
                    'total_access': total_access,
                    'hits': stats['hits'],
                    'misses': stats['misses'],
                    'hit_rate': (stats['hits'] / total_access * 100) if total_access > 0 else 0,
                    'last_accessed': stats['last_operation']
                })
        
        return sorted(keys_with_access, key=lambda x: x['total_access'], reverse=True)[:limit]
    
    def analyze_cache_efficiency(self) -> Dict[str, Any]:
        """آنالیز کارایی کش"""
        stats = self.get_cache_stats()
        performance = self.get_cache_performance(24)
        
        efficiency_score = 0
        
        # امتیاز بر اساس hit rate (50%)
        hit_rate = stats.get('hit_rate', 0)
        efficiency_score += min(hit_rate * 0.5, 50)
        
        # امتیاز بر اساس سرعت پاسخ (30%)
        avg_response_time = performance.get('average_response_time', 0)
        if avg_response_time < 0.001:  # 1ms
            efficiency_score += 30
        elif avg_response_time < 0.01:  # 10ms
            efficiency_score += 25
        elif avg_response_time < 0.1:   # 100ms
            efficiency_score += 20
        else:
            efficiency_score += 10
        
        # امتیاز بر اساس نرخ موفقیت (20%)
        success_rate = performance.get('success_rate', 0)
        efficiency_score += (success_rate / 100) * 20
        
        return {
            'efficiency_score': round(efficiency_score, 2),
            'efficiency_grade': self._get_efficiency_grade(efficiency_score),
            'hit_rate': hit_rate,
            'average_response_time': avg_response_time,
            'success_rate': success_rate,
            'recommendations': self._generate_cache_recommendations(hit_rate, avg_response_time)
        }
    
    def _get_efficiency_grade(self, score: float) -> str:
        """دریافت گرید کارایی"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def _generate_cache_recommendations(self, hit_rate: float, response_time: float) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی کش"""
        recommendations = []
        
        if hit_rate < 60:
            recommendations.append("افزایش TTL برای داده‌های پرتکرار")
            recommendations.append("پیاده‌سازی cache warming برای داده‌های مهم")
        
        if response_time > 0.01:  # بیش از 10ms
            recommendations.append("بررسی شبکه و سخت‌افزار کش")
            recommendations.append("بهینه‌سازی سریالایز داده‌ها")
        
        if hit_rate > 90 and response_time < 0.001:
            recommendations.append("کارایی کش عالی است - حفظ وضعیت فعلی")
        
        return recommendations

# ایجاد نمونه گلوبال
cache_debugger = CacheDebugger()

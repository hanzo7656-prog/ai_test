import logging
import time
import redis
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
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
            'errors': 0,
            'total_size': 0,
            'total_response_time': 0,
            'last_operation': None,
            'database': None  # اضافه شدن فیلد دیتابیس
        })
        
        # ایمپورت مدیر Redis اصلی از فایل redis_manager
        from redis_manager import redis_manager
        self.redis_manager = redis_manager
        
    def log_cache_operation(self, operation: str, key: str, success: bool, 
                          response_time: float, size: int = 0, error: str = None, 
                          database: str = None):
        """ثبت عملیات کش با جزئیات کامل"""
        operation_data = {
            'operation': operation,
            'key': key,
            'database': database,
            'success': success,
            'response_time': response_time,
            'size': size,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.cache_operations.append(operation_data)
        
        # آپدیت آمار
        stats_key = f"{database}:{key}" if database else key
        stats = self.cache_stats[stats_key]
        stats['last_operation'] = datetime.now().isoformat()
        stats['total_response_time'] += response_time
        stats['database'] = database
        
        if operation == 'GET':
            if success:
                stats['hits'] += 1
            else:
                stats['misses'] += 1
        elif operation == 'SET':
            if success:
                stats['sets'] += 1
                stats['total_size'] += size
            else:
                stats['errors'] += 1
        elif operation == 'DELETE':
            if success:
                stats['deletes'] += 1
                stats['total_size'] = max(0, stats['total_size'] - size)
            else:
                stats['errors'] += 1
        
        if error:
            stats['errors'] += 1
    
    # ==================== API های کاربردی برای routes ====================
    
    def set_data(self, database: str, key: str, value: Any, expire: int = 300) -> bool:
        """ذخیره داده در کش (برای استفاده در routes)"""
        success, response_time = self.redis_manager.set(database, key, value, expire)
        size = len(json.dumps(value, ensure_ascii=False)) if success else 0
        self.log_cache_operation('SET', key, success, response_time, size, database=database)
        return success
    
    def get_data(self, database: str, key: str) -> Optional[Any]:
        """دریافت داده از کش (برای استفاده در routes)"""
        data, response_time = self.redis_manager.get(database, key)
        success = data is not None
        size = len(json.dumps(data, ensure_ascii=False)) if success else 0
        self.log_cache_operation('GET', key, success, response_time, size, database=database)
        return data
    
    def delete_data(self, database: str, key: str) -> bool:
        """حذف داده از کش (برای استفاده در routes)"""
        success, response_time = self.redis_manager.delete(database, key)
        # تخمین اندازه برای حذف
        stats_key = f"{database}:{key}"
        estimated_size = self.cache_stats[stats_key].get('total_size', 0) / max(self.cache_stats[stats_key].get('sets', 1), 1)
        self.log_cache_operation('DELETE', key, success, response_time, int(estimated_size), database=database)
        return success
    
    def exists_data(self, database: str, key: str) -> bool:
        """بررسی وجود داده در کش"""
        exists, response_time = self.redis_manager.exists(database, key)
        self.log_cache_operation('EXISTS', key, exists, response_time, database=database)
        return exists
    
    # ==================== متدهای مانیتورینگ و آنالیز ====================
    
    def get_cache_stats(self, database: str = None, key: str = None) -> Dict[str, Any]:
        """دریافت آمار کش"""
        if key and database:
            stats_key = f"{database}:{key}"
            if stats_key not in self.cache_stats:
                return {'error': 'Key not found in specified database'}
            
            stats = self.cache_stats[stats_key]
            total_operations = stats['hits'] + stats['misses'] + stats['sets'] + stats['deletes']
            avg_response_time = (stats['total_response_time'] / total_operations) if total_operations > 0 else 0
            
            return {
                'database': database,
                'key': key,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'sets': stats['sets'],
                'deletes': stats['deletes'],
                'errors': stats['errors'],
                'total_size_bytes': stats['total_size'],
                'average_response_time': round(avg_response_time, 4),
                'hit_rate': (stats['hits'] / (stats['hits'] + stats['misses']) * 100) if (stats['hits'] + stats['misses']) > 0 else 0,
                'last_operation': stats['last_operation']
            }
        
        # آمار بر اساس دیتابیس
        if database:
            db_stats = {
                'database': database,
                'total_keys': 0,
                'total_hits': 0,
                'total_misses': 0,
                'total_sets': 0,
                'total_deletes': 0,
                'total_errors': 0,
                'total_size_bytes': 0,
                'total_response_time': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            for stats_key, stats in self.cache_stats.items():
                if stats['database'] == database:
                    db_stats['total_keys'] += 1
                    db_stats['total_hits'] += stats['hits']
                    db_stats['total_misses'] += stats['misses']
                    db_stats['total_sets'] += stats['sets']
                    db_stats['total_deletes'] += stats['deletes']
                    db_stats['total_errors'] += stats['errors']
                    db_stats['total_size_bytes'] += stats['total_size']
                    db_stats['total_response_time'] += stats['total_response_time']
            
            total_operations = db_stats['total_hits'] + db_stats['total_misses'] + db_stats['total_sets'] + db_stats['total_deletes']
            db_stats['total_operations'] = total_operations
            
            if total_operations > 0:
                db_stats['average_response_time'] = round(db_stats['total_response_time'] / total_operations, 4)
            
            read_operations = db_stats['total_hits'] + db_stats['total_misses']
            if read_operations > 0:
                db_stats['hit_rate'] = round((db_stats['total_hits'] / read_operations) * 100, 2)
            
            db_stats['redis_health'] = self.redis_manager.health_check(database)
            return db_stats
        
        # آمار کلی تمام دیتابیس‌ها
        total_stats = {
            'total_databases': 5,
            'total_keys': len(self.cache_stats),
            'total_hits': sum(stats['hits'] for stats in self.cache_stats.values()),
            'total_misses': sum(stats['misses'] for stats in self.cache_stats.values()),
            'total_sets': sum(stats['sets'] for stats in self.cache_stats.values()),
            'total_deletes': sum(stats['deletes'] for stats in self.cache_stats.values()),
            'total_errors': sum(stats['errors'] for stats in self.cache_stats.values()),
            'total_size_bytes': sum(stats['total_size'] for stats in self.cache_stats.values()),
            'total_operations': 0,
            'hit_rate': 0,
            'average_response_time': 0,
            'database_breakdown': {},
            'redis_health': self.redis_manager.health_check(),
            'timestamp': datetime.now().isoformat()
        }
        
        # آمار تفکیک شده بر اساس دیتابیس
        for db_name in ['uta', 'utb', 'utc', 'mother_a', 'mother_b']:
            db_stats = self.get_cache_stats(database=db_name)
            total_stats['database_breakdown'][db_name] = db_stats
        
        total_operations = total_stats['total_hits'] + total_stats['total_misses'] + total_stats['total_sets'] + total_stats['total_deletes']
        total_stats['total_operations'] = total_operations
        
        if total_operations > 0:
            total_response_time = sum(stats['total_response_time'] for stats in self.cache_stats.values())
            total_stats['average_response_time'] = round(total_response_time / total_operations, 4)
        
        read_operations = total_stats['total_hits'] + total_stats['total_misses']
        if read_operations > 0:
            total_stats['hit_rate'] = round((total_stats['total_hits'] / read_operations) * 100, 2)
        
        return total_stats
    
    def get_cache_performance(self, database: str = None, hours: int = 24) -> Dict[str, Any]:
        """دریافت عملکرد کش در بازه زمانی"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_operations = [
            op for op in self.cache_operations
            if datetime.fromisoformat(op['timestamp']) >= cutoff_time
        ]
        
        if database:
            recent_operations = [op for op in recent_operations if op['database'] == database]
        
        performance_data = {
            'database': database or 'all',
            'period_hours': hours,
            'total_operations': len(recent_operations),
            'operations_by_type': defaultdict(int),
            'operations_by_database': defaultdict(int),
            'successful_operations': 0,
            'failed_operations': 0,
            'total_response_time': 0,
            'total_data_size': 0,
            'average_response_time': 0,
            'success_rate': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        for op in recent_operations:
            performance_data['operations_by_type'][op['operation']] += 1
            performance_data['operations_by_database'][op['database']] += 1
            performance_data['total_response_time'] += op['response_time']
            performance_data['total_data_size'] += op['size']
            
            if op['success']:
                performance_data['successful_operations'] += 1
            else:
                performance_data['failed_operations'] += 1
        
        if recent_operations:
            performance_data['average_response_time'] = round(
                performance_data['total_response_time'] / len(recent_operations), 4
            )
            performance_data['success_rate'] = round(
                (performance_data['successful_operations'] / len(recent_operations)) * 100, 2
            )
        
        return performance_data
    
    def get_most_accessed_keys(self, database: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """دریافت کلیدهای پر دسترس"""
        keys_with_access = []
        
        for stats_key, stats in self.cache_stats.items():
            if database and stats['database'] != database:
                continue
                
            total_access = stats['hits'] + stats['misses']
            if total_access > 0:
                # استخراج نام کلید از stats_key
                key_name = stats_key.split(':', 1)[1] if ':' in stats_key else stats_key
                
                keys_with_access.append({
                    'database': stats['database'],
                    'key': key_name,
                    'total_access': total_access,
                    'hits': stats['hits'],
                    'misses': stats['misses'],
                    'hit_rate': round((stats['hits'] / total_access * 100), 2) if total_access > 0 else 0,
                    'average_response_time': round((stats['total_response_time'] / total_access), 4) if total_access > 0 else 0,
                    'last_accessed': stats['last_operation'],
                    'total_size_bytes': stats['total_size']
                })
        
        return sorted(keys_with_access, key=lambda x: x['total_access'], reverse=True)[:limit]
    
    def get_database_usage_report(self) -> Dict[str, Any]:
        """گزارش استفاده از دیتابیس‌ها"""
        usage_report = {}
        
        for db_name in ['uta', 'utb', 'utc', 'mother_a', 'mother_b']:
            db_stats = self.get_cache_stats(database=db_name)
            db_performance = self.get_cache_performance(database=db_name, hours=24)
            db_health = self.redis_manager.health_check(db_name)
            
            usage_report[db_name] = {
                'stats': db_stats,
                'performance': db_performance,
                'health': db_health,
                'efficiency': self._calculate_efficiency_score(db_stats, db_performance)
            }
        
        return {
            'database_usage': usage_report,
            'overall_efficiency': self._calculate_overall_efficiency(usage_report),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cache_efficiency_report(self, database: str = None) -> Dict[str, Any]:
        """گزارش کامل کارایی کش"""
        stats = self.get_cache_stats(database)
        performance = self.get_cache_performance(database, 24)
        top_keys = self.get_most_accessed_keys(database, 5)
        
        efficiency_score = self._calculate_efficiency_score(stats, performance)
        
        return {
            'database': database or 'all',
            'efficiency_score': efficiency_score,
            'efficiency_grade': self._get_efficiency_grade(efficiency_score),
            'overview': {
                'hit_rate': stats.get('hit_rate', 0),
                'average_response_time_ms': round(performance.get('average_response_time', 0) * 1000, 2),
                'success_rate': performance.get('success_rate', 0),
                'total_keys': stats.get('total_keys', 0),
                'total_size_mb': round(stats.get('total_size_bytes', 0) / (1024 * 1024), 2)
            },
            'performance': performance,
            'top_accessed_keys': top_keys,
            'recommendations': self._generate_recommendations(stats, performance),
            'redis_health': stats.get('redis_health', {}),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_efficiency_score(self, stats: Dict, performance: Dict) -> float:
        """محاسبه امتیاز کارایی کش"""
        efficiency_score = 0
        
        # Hit Rate (40%)
        hit_rate = stats.get('hit_rate', 0)
        efficiency_score += min(hit_rate * 0.4, 40)
        
        # Response Time (30%)
        avg_response_time = performance.get('average_response_time', 0)
        if avg_response_time < 0.001:  # < 1ms
            efficiency_score += 30
        elif avg_response_time < 0.005:  # < 5ms
            efficiency_score += 25
        elif avg_response_time < 0.01:   # < 10ms
            efficiency_score += 20
        elif avg_response_time < 0.1:    # < 100ms
            efficiency_score += 15
        else:
            efficiency_score += 5
        
        # Success Rate (20%)
        success_rate = performance.get('success_rate', 0)
        efficiency_score += (success_rate / 100) * 20
        
        # Memory Efficiency (10%)
        total_size_mb = stats.get('total_size_bytes', 0) / (1024 * 1024)
        if total_size_mb < 10:  # < 10MB
            efficiency_score += 10
        elif total_size_mb < 50:  # < 50MB
            efficiency_score += 8
        elif total_size_mb < 100:  # < 100MB
            efficiency_score += 5
        else:
            efficiency_score += 2
        
        return round(efficiency_score, 2)
    
    def _calculate_overall_efficiency(self, usage_report: Dict) -> float:
        """محاسبه کارایی کلی"""
        total_score = 0
        count = 0
        
        for db_name, report in usage_report.items():
            if 'efficiency' in report:
                total_score += report['efficiency']
                count += 1
        
        return round(total_score / count, 2) if count > 0 else 0
    
    def _get_efficiency_grade(self, score: float) -> str:
        """دریافت گرید کارایی"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, stats: Dict, performance: Dict) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        hit_rate = stats.get('hit_rate', 0)
        response_time = performance.get('average_response_time', 0)
        success_rate = performance.get('success_rate', 0)
        
        if hit_rate < 50:
            recommendations.append("🔴 افزایش TTL برای داده‌های پرتکرار")
            recommendations.append("🟡 پیاده‌سازی cache warming برای اندپوینت‌های مهم")
            recommendations.append("🔵 بررسی الگوی دسترسی و بهینه‌سازی کلیدها")
        
        if response_time > 0.01:  # بیش از 10ms
            recommendations.append("🔴 بررسی شبکه و اتصال Redis")
            recommendations.append("🟡 بهینه‌سازی اندازه داده‌های ذخیره شده")
        
        if success_rate < 95:
            recommendations.append("🟡 بررسی خطاهای اتصال Redis")
            recommendations.append("🔵 افزودن retry mechanism برای عملیات کش")
        
        if hit_rate > 80 and response_time < 0.005 and success_rate > 98:
            recommendations.append("✅ کارایی کش عالی - حفظ وضعیت فعلی")
        
        if not recommendations:
            recommendations.append("ℹ️  هیچ اقدام فوری مورد نیاز نیست")
        
        return recommendations
    
    def clear_old_operations(self, days: int = 7):
        """پاک کردن عملیات قدیمی"""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.cache_operations = deque(
            [op for op in self.cache_operations if datetime.fromisoformat(op['timestamp']) > cutoff_time],
            maxlen=10000
        )
        
        # پاک کردن آمار کلیدهای بدون استفاده
        current_time = datetime.now()
        for key in list(self.cache_stats.keys()):
            last_op = self.cache_stats[key]['last_operation']
            if last_op and (current_time - datetime.fromisoformat(last_op)).days > days:
                del self.cache_stats[key]
        
        logger.info(f"🧹 Cleared cache operations older than {days} days")

# ایجاد نمونه گلوبال
cache_debugger = CacheDebugger()

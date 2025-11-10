import logging
import time
import redis
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RedisCacheManager:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.client = None
        self._connect()
        
    def _connect(self):
        """اتصال به Redis"""
        try:
            self.client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=10
            )
            self.client.ping()
            logger.info("✅ Redis Cache connected to Debug System!")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.client = None
    
    def set(self, key: str, value: Any, expire: int = 300) -> Tuple[bool, float]:
        """ذخیره داده در کش - بازگشت (موفقیت, زمان پاسخ)"""
        if not self.client:
            return False, 0
        
        try:
            start_time = time.time()
            serialized_value = json.dumps(value, ensure_ascii=False)
            success = bool(self.client.setex(key, expire, serialized_value))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False, 0
    
    def get(self, key: str) -> Tuple[Optional[Any], float]:
        """دریافت داده از کش - بازگشت (داده, زمان پاسخ)"""
        if not self.client:
            return None, 0
        
        try:
            start_time = time.time()
            value = self.client.get(key)
            response_time = time.time() - start_time
            
            if value:
                data = json.loads(value)
                return data, response_time
            else:
                return None, response_time
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None, 0
    
    def delete(self, key: str) -> Tuple[bool, float]:
        """حذف داده از کش - بازگشت (موفقیت, زمان پاسخ)"""
        if not self.client:
            return False, 0
        
        try:
            start_time = time.time()
            success = bool(self.client.delete(key))
            response_time = time.time() - start_time
            return success, response_time
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False, 0
    
    def exists(self, key: str) -> Tuple[bool, float]:
        """بررسی وجود کلید - بازگشت (وجود دارد, زمان پاسخ)"""
        if not self.client:
            return False, 0
        
        try:
            start_time = time.time()
            exists = bool(self.client.exists(key))
            response_time = time.time() - start_time
            return exists, response_time
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False, 0
    
    def get_keys(self, pattern: str = "*") -> Tuple[List[str], float]:
        """دریافت کلیدها - بازگشت (لیست کلیدها, زمان پاسخ)"""
        if not self.client:
            return [], 0
        
        try:
            start_time = time.time()
            keys = self.client.keys(pattern)
            response_time = time.time() - start_time
            return keys, response_time
        except Exception as e:
            logger.error(f"Redis keys error for pattern {pattern}: {e}")
            return [], 0
    
    def get_memory_usage(self, key: str) -> Tuple[Optional[int], float]:
        """دریافت مصرف حافظه یک کلید - بازگشت (بایت, زمان پاسخ)"""
        if not self.client:
            return None, 0
        
        try:
            start_time = time.time()
            # استفاده از Redis MEMORY USAGE (اگر موجود باشد)
            memory = self.client.memory_usage(key) if hasattr(self.client, 'memory_usage') else None
            response_time = time.time() - start_time
            return memory, response_time
        except Exception as e:
            logger.error(f"Redis memory usage error for key {key}: {e}")
            return None, 0
    
    def health_check(self) -> Dict[str, Any]:
        """بررسی سلامت کامل Redis"""
        if not self.client:
            return {
                "status": "disconnected", 
                "error": "No Redis client available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            start_time = time.time()
            self.client.ping()
            ping_time = time.time() - start_time
            
            info = self.client.info()
            return {
                "status": "connected",
                "type": "redis_cloud",
                "ping_time_ms": round(ping_time * 1000, 2),
                "used_memory": info.get('used_memory_human', 'N/A'),
                "used_memory_bytes": info.get('used_memory', 0),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "uptime_in_seconds": info.get('uptime_in_seconds', 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

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
            'last_operation': None
        })
        
        # مدیر Redis
        self.redis_manager = RedisCacheManager()
        
    def log_cache_operation(self, operation: str, key: str, success: bool, 
                          response_time: float, size: int = 0, error: str = None):
        """ثبت عملیات کش با جزئیات کامل"""
        operation_data = {
            'operation': operation,
            'key': key,
            'success': success,
            'response_time': response_time,
            'size': size,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.cache_operations.append(operation_data)
        
        # آپدیت آمار
        stats = self.cache_stats[key]
        stats['last_operation'] = datetime.now().isoformat()
        stats['total_response_time'] += response_time
        
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
    
    def set_data(self, key: str, value: Any, expire: int = 300) -> bool:
        """ذخیره داده در کش (برای استفاده در routes)"""
        success, response_time = self.redis_manager.set(key, value, expire)
        size = len(json.dumps(value, ensure_ascii=False)) if success else 0
        self.log_cache_operation('SET', key, success, response_time, size)
        return success
    
    def get_data(self, key: str) -> Optional[Any]:
        """دریافت داده از کش (برای استفاده در routes)"""
        data, response_time = self.redis_manager.get(key)
        success = data is not None
        size = len(json.dumps(data, ensure_ascii=False)) if success else 0
        self.log_cache_operation('GET', key, success, response_time, size)
        return data
    
    def delete_data(self, key: str) -> bool:
        """حذف داده از کش (برای استفاده در routes)"""
        success, response_time = self.redis_manager.delete(key)
        # تخمین اندازه برای حذف - استفاده از میانگین اگر موجود نباشد
        estimated_size = self.cache_stats[key].get('total_size', 0) / max(self.cache_stats[key].get('sets', 1), 1)
        self.log_cache_operation('DELETE', key, success, response_time, int(estimated_size))
        return success
    
    def exists_data(self, key: str) -> bool:
        """بررسی وجود داده در کش"""
        exists, response_time = self.redis_manager.exists(key)
        self.log_cache_operation('EXISTS', key, exists, response_time)
        return exists
    
    # ==================== متدهای مانیتورینگ و آنالیز ====================
    
    def get_cache_stats(self, key: str = None) -> Dict[str, Any]:
        """دریافت آمار کش"""
        if key:
            if key not in self.cache_stats:
                return {'error': 'Key not found'}
            
            stats = self.cache_stats[key]
            total_operations = stats['hits'] + stats['misses'] + stats['sets'] + stats['deletes']
            avg_response_time = (stats['total_response_time'] / total_operations) if total_operations > 0 else 0
            
            return {
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
        
        # آمار کلی
        total_stats = {
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
            'redis_health': self.redis_manager.health_check(),
            'timestamp': datetime.now().isoformat()
        }
        
        total_operations = total_stats['total_hits'] + total_stats['total_misses'] + total_stats['total_sets'] + total_stats['total_deletes']
        total_stats['total_operations'] = total_operations
        
        if total_operations > 0:
            total_response_time = sum(stats['total_response_time'] for stats in self.cache_stats.values())
            total_stats['average_response_time'] = round(total_response_time / total_operations, 4)
        
        read_operations = total_stats['total_hits'] + total_stats['total_misses']
        if read_operations > 0:
            total_stats['hit_rate'] = round((total_stats['total_hits'] / read_operations) * 100, 2)
        
        return total_stats
    
    def get_cache_performance(self, hours: int = 24) -> Dict[str, Any]:
        """دریافت عملکرد کش در بازه زمانی"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_operations = [
            op for op in self.cache_operations
            if datetime.fromisoformat(op['timestamp']) >= cutoff_time
        ]
        
        performance_data = {
            'period_hours': hours,
            'total_operations': len(recent_operations),
            'operations_by_type': defaultdict(int),
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
                    'hit_rate': round((stats['hits'] / total_access * 100), 2) if total_access > 0 else 0,
                    'average_response_time': round((stats['total_response_time'] / total_access), 4) if total_access > 0 else 0,
                    'last_accessed': stats['last_operation'],
                    'total_size_bytes': stats['total_size']
                })
        
        return sorted(keys_with_access, key=lambda x: x['total_access'], reverse=True)[:limit]
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """گزارش کامل کارایی کش"""
        stats = self.get_cache_stats()
        performance = self.get_cache_performance(24)
        top_keys = self.get_most_accessed_keys(5)
        
        efficiency_score = self._calculate_efficiency_score(stats, performance)
        
        return {
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

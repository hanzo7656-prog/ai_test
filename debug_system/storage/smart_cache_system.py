"""
Cache Analytics & Optimization Engine
آنالیز و بهینه‌سازی هوشمند عملکرد کش
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class CacheOptimizationEngine:
    """موتور بهینه‌سازی و آنالیز عملکرد کش"""
    
    def __init__(self):
        # ایمپورت سیستم‌های اصلی
        from .cache_debugger import cache_debugger
        from .redis_manager import redis_manager
        
        self.debugger = cache_debugger
        self.redis_manager = redis_manager
        
        # دیتابیس برای آنالیتیکس (MOTHER_B)
        self.analytics_db = "mother_b"
        
        # الگوهای دسترسی
        self.access_patterns = defaultdict(lambda: {
            'access_count': 0,
            'last_access': None,
            'access_times': deque(maxlen=100),
            'size_history': deque(maxlen=50),
            'hit_miss_ratio': 0
        })
        
        # پیشنهادات بهینه‌سازی
        self.optimization_suggestions = deque(maxlen=100)
        
        # آمار پیشرفته
        self.advanced_stats = {
            'peak_usage_times': defaultdict(int),
            'key_lifespan_analysis': defaultdict(list),
            'database_load_distribution': defaultdict(int),
            'compression_efficiency': 0,
            'cost_savings_estimate': 0
        }

    def analyze_access_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """آنالیز الگوهای دسترسی به کش"""
        try:
            # جمع‌آوری داده‌های دسترسی از cache_debugger
            recent_operations = [
                op for op in self.debugger.cache_operations
                if datetime.fromisoformat(op['timestamp']) >= datetime.now() - timedelta(hours=hours)
            ]
            
            analysis = {
                'period_hours': hours,
                'total_operations': len(recent_operations),
                'operations_by_hour': defaultdict(int),
                'hot_keys': [],
                'cold_keys': [],
                'access_trends': {},
                'recommendations': []
            }
            
            # تحلیل ساعتی
            for op in recent_operations:
                hour = datetime.fromisoformat(op['timestamp']).hour
                analysis['operations_by_hour'][hour] += 1
            
            # شناسایی کلیدهای داغ و سرد
            key_access_count = defaultdict(int)
            for op in recent_operations:
                key_access_count[op['key']] += 1
            
            sorted_keys = sorted(key_access_count.items(), key=lambda x: x[1], reverse=True)
            if sorted_keys:
                analysis['hot_keys'] = [{'key': k, 'access_count': v} for k, v in sorted_keys[:10]]
                analysis['cold_keys'] = [{'key': k, 'access_count': v} for k, v in sorted_keys[-10:]]
            
            # تولید پیشنهادات
            self._generate_access_recommendations(analysis, recent_operations)
            
            # ذخیره آنالیز
            self._store_analytics('access_patterns', analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing access patterns: {e}")
            return {'error': str(e)}

    def predict_optimal_ttl(self, key_pattern: str, database: str = "utb") -> Dict[str, Any]:
        """پیش‌بینی TTL بهینه بر اساس الگوی دسترسی"""
        try:
            # جمع‌آوری داده‌های تاریخی
            keys = self.redis_manager.get_keys(database, key_pattern)[0]
            
            ttl_analysis = {
                'pattern': key_pattern,
                'database': database,
                'sample_size': len(keys),
                'current_avg_ttl': 0,
                'recommended_ttl': 300,
                'confidence_score': 0,
                'key_analysis': []
            }
            
            total_ttl = 0
            analyzed_keys = 0
            
            for key in keys[:50]:  # نمونه‌گیری از 50 کلید اول
                try:
                    # بررسی TTL فعلی
                    ttl = self.redis_manager.get_client(database).ttl(key)
                    if ttl > 0:
                        total_ttl += ttl
                        analyzed_keys += 1
                        
                        # تحلیل الگوی دسترسی این کلید
                        access_stats = self._get_key_access_stats(key)
                        ttl_analysis['key_analysis'].append({
                            'key': key,
                            'current_ttl': ttl,
                            'access_count': access_stats.get('access_count', 0),
                            'last_access': access_stats.get('last_access')
                        })
                except:
                    continue
            
            if analyzed_keys > 0:
                current_avg = total_ttl / analyzed_keys
                ttl_analysis['current_avg_ttl'] = current_avg
                
                # محاسبه TTL بهینه
                recommended_ttl = self._calculate_optimal_ttl(ttl_analysis['key_analysis'])
                ttl_analysis['recommended_ttl'] = recommended_ttl
                ttl_analysis['confidence_score'] = min(100, analyzed_keys * 2)
            
            return ttl_analysis
            
        except Exception as e:
            logger.error(f"❌ Error predicting optimal TTL: {e}")
            return {'error': str(e)}

    def database_health_check(self) -> Dict[str, Any]:
        """بررسی سلامت و تعادل دیتابیس‌ها"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'database_health': {},
            'load_balancing': {},
            'recommendations': [],
            'alerts': []
        }
        
        databases = ['uta', 'utb', 'utc', 'mother_a', 'mother_b']
        
        for db in databases:
            try:
                # سلامت اتصال
                health = self.redis_manager.health_check(db)
                
                # استفاده از حافظه
                usage = self.redis_manager.get_database_usage().get(db, {})
                
                health_report['database_health'][db] = {
                    'status': health.get('status', 'unknown'),
                    'memory_usage_percentage': usage.get('used_memory_percentage', 0),
                    'memory_used': usage.get('used_memory_human', 'N/A'),
                    'keys_count': usage.get('keys_count', 0),
                    'connected_clients': health.get('connected_clients', 0),
                    'ping_time_ms': health.get('ping_time_ms', 0)
                }
                
                # بررسی هشدارها
                if usage.get('used_memory_percentage', 0) > 80:
                    health_report['alerts'].append(f"🔴 {db}: حافظه نزدیک به ظرفیت")
                
                if health.get('status') != 'connected':
                    health_report['alerts'].append(f"🔴 {db}: مشکل اتصال")
                    
            except Exception as e:
                health_report['database_health'][db] = {'error': str(e)}
                health_report['alerts'].append(f"🔴 {db}: خطای بررسی سلامت")
        
        # تحلیل تعادل بار
        self._analyze_load_balancing(health_report)
        
        return health_report

    def cost_optimization_report(self) -> Dict[str, Any]:
        """گزارش بهینه‌سازی هزینه‌ها"""
        try:
            # محاسبه هزینه‌های تخمینی (بر اساس استفاده از Upstash)
            report = {
                'timestamp': datetime.now().isoformat(),
                'cost_estimation': {},
                'optimization_opportunities': [],
                'monthly_savings_estimate': 0
            }
            
            usage_data = self.redis_manager.get_database_usage()
            
            for db_name, usage in usage_data.items():
                # محاسبه هزینه تخمینی (فرمول ساده شده)
                memory_usage_mb = usage.get('used_memory_bytes', 0) / (1024 * 1024)
                estimated_cost = max(0.50, memory_usage_mb * 0.01)  # مدل هزینه ساده
                
                report['cost_estimation'][db_name] = {
                    'memory_usage_mb': round(memory_usage_mb, 2),
                    'estimated_monthly_cost': round(estimated_cost, 2),
                    'keys_count': usage.get('keys_count', 0),
                    'efficiency_score': self._calculate_efficiency_score(usage)
                }
            
            # شناسایی فرصت‌های بهینه‌سازی
            self._identify_cost_savings(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating cost report: {e}")
            return {'error': str(e)}

    def intelligent_cache_warming(self, key_patterns: List[str], databases: List[str] = None):
        """گرم کردن هوشمند کش بر اساس الگوهای پیش‌بینی شده"""
        if databases is None:
            databases = ['utb', 'utc']  # دیتابیس‌های اصلی
        
        warming_report = {
            'timestamp': datetime.now().isoformat(),
            'warmed_keys': 0,
            'success_rate': 0,
            'performance_impact': 'low',
            'details': []
        }
        
        successful_warms = 0
        total_attempts = 0
        
        for db in databases:
            for pattern in key_patterns:
                try:
                    keys = self.redis_manager.get_keys(db, pattern)[0]
                    total_attempts += len(keys)
                    
                    for key in keys[:20]:  # محدود کردن برای جلوگیری از overload
                        # بررسی وجود کلید (شبیه‌سازی دسترسی)
                        exists, _ = self.redis_manager.exists(db, key)
                        if exists:
                            successful_warms += 1
                            warming_report['details'].append({
                                'database': db,
                                'key': key,
                                'status': 'warmed'
                            })
                    
                except Exception as e:
                    warming_report['details'].append({
                        'database': db,
                        'pattern': pattern,
                        'status': 'error',
                        'error': str(e)
                    })
        
        if total_attempts > 0:
            warming_report['warmed_keys'] = successful_warms
            warming_report['success_rate'] = round((successful_warms / total_attempts) * 100, 2)
        
        return warming_report

    def _generate_access_recommendations(self, analysis: Dict, operations: List):
        """تولید پیشنهادات بر اساس الگوی دسترسی"""
        recommendations = []
        
        # تحلیل ساعات پیک
        peak_hours = sorted(analysis['operations_by_hour'].items(), key=lambda x: x[1], reverse=True)[:3]
        if peak_hours:
            recommendations.append(f"🕒 ساعات پیک دسترسی: {[h[0] for h in peak_hours]}")
        
        # تحلیل کلیدهای داغ
        if analysis['hot_keys']:
            hot_key = analysis['hot_keys'][0]
            recommendations.append(f"🔥 کلید داغ: {hot_key['key']} ({hot_key['access_count']} دسترسی)")
        
        # تحلیل کلیدهای سرد
        if analysis['cold_keys']:
            cold_key_count = len([k for k in analysis['cold_keys'] if k['access_count'] == 1])
            if cold_key_count > 10:
                recommendations.append(f"🧊 {cold_key_count} کلید با دسترسی تک‌باره - امکان حذف")
        
        analysis['recommendations'] = recommendations

    def _calculate_optimal_ttl(self, key_analysis: List[Dict]) -> int:
        """محاسبه TTL بهینه"""
        if not key_analysis:
            return 300  # پیش‌فرض
        
        # میانگین TTL فعلی
        current_ttls = [k['current_ttl'] for k in key_analysis if k['current_ttl'] > 0]
        if not current_ttls:
            return 300
        
        avg_ttl = sum(current_ttls) / len(current_ttls)
        
        # تنظیم بر اساس الگوی دسترسی
        access_counts = [k['access_count'] for k in key_analysis]
        avg_access = sum(access_counts) / len(access_counts) if access_counts else 1
        
        if avg_access > 50:  # دسترسی زیاد
            return min(3600, int(avg_ttl * 1.5))
        elif avg_access < 5:  # دسترسی کم
            return max(60, int(avg_ttl * 0.7))
        else:
            return int(avg_ttl)

    def _analyze_load_balancing(self, health_report: Dict):
        """تحلیل تعادل بار بین دیتابیس‌ها"""
        memory_usage = []
        for db, health in health_report['database_health'].items():
            if 'memory_usage_percentage' in health:
                memory_usage.append(health['memory_usage_percentage'])
        
        if memory_usage:
            avg_usage = sum(memory_usage) / len(memory_usage)
            max_usage = max(memory_usage)
            min_usage = min(memory_usage)
            
            imbalance = max_usage - min_usage
            if imbalance > 30:  # عدم تعادل قابل توجه
                health_report['recommendations'].append(
                    f"⚖️ عدم تعادل حافظه: {imbalance:.1f}% - بازتوزیع داده‌ها پیشنهاد می‌شود"
                )

    def _identify_cost_savings(self, report: Dict):
        """شناسایی فرصت‌های صرفه‌جویی در هزینه"""
        total_cost = sum([db['estimated_monthly_cost'] for db in report['cost_estimation'].values()])
        
        # تحلیل کارایی
        for db_name, data in report['cost_estimation'].items():
            efficiency = data['efficiency_score']
            if efficiency < 60:
                report['optimization_opportunities'].append(
                    f"🔧 {db_name}: کارایی پایین ({efficiency}%) - امکان بهینه‌سازی"
                )
        
        # پیشنهاد consolidating اگر هزینه بالا باشد
        if total_cost > 10:  # اگر هزینه کل بیش از 10 دلار باشد
            report['optimization_opportunities'].append(
                "💰 هزینه ماهانه بالا - امکان ادغام دیتابیس‌ها"
            )

    def _calculate_efficiency_score(self, usage: Dict) -> float:
        """محاسبه امتیاز کارایی"""
        memory_usage = usage.get('used_memory_percentage', 0)
        keys_count = usage.get('keys_count', 0)
        
        # هرچه حافظه کمتر استفاده شده و کلیدهای بیشتری داشته باشد، کارایی بالاتر
        if keys_count == 0:
            return 0
        
        efficiency = (100 - memory_usage) * (min(keys_count, 1000) / 1000)
        return round(min(efficiency, 100), 1)

    def _get_key_access_stats(self, key: str) -> Dict[str, Any]:
        """دریافت آمار دسترسی یک کلید"""
        # این متد نیاز به پیاده‌سازی دقیق‌تر دارد
        # در حال حاضر یک پیاده‌سازی ساده
        return {
            'access_count': 0,
            'last_access': None
        }

    def _store_analytics(self, analytics_type: str, data: Dict):
        """ذخیره نتایج آنالیتیکس"""
        try:
            key = f"analytics:{analytics_type}:{datetime.now().strftime('%Y%m%d_%H')}"
            self.redis_manager.set(
                self.analytics_db, 
                key, 
                data, 
                expire=7*24*3600  # نگهداری 7 روز
            )
        except Exception as e:
            logger.error(f"❌ Error storing analytics: {e}")

# ایجاد نمونه اصلی
cache_optimizer = CacheOptimizationEngine()

logger.info("🚀 Cache Optimization Engine Initialized - Advanced Analytics & Optimization")

__all__ = ["CacheOptimizationEngine", "cache_optimizer"]

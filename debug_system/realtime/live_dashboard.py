import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import WebSocket
from collections import defaultdict, deque
import psutil

# ایمپورت سیستم نرمال‌سازی جدید
try:
    from ..utils.data_normalizer import data_normalizer
except ImportError:
    # Fallback برای مواقع توسعه
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.utils.data_normalizer import data_normalizer

logger = logging.getLogger(__name__)

class LiveDashboardManager:
    def __init__(self, debug_manager, metrics_collector):
        self.debug_manager = debug_manager
        self.metrics_collector = metrics_collector
        self.dashboard_connections: List[WebSocket] = []
        self.dashboard_data_buffer = deque(maxlen=100)
        
        # داده‌های ویژه نرمال‌سازی
        self.normalization_insights = {
            'structure_evolution': deque(maxlen=50),
            'quality_trends': deque(maxlen=100),
            'performance_metrics': deque(maxlen=200)
        }
        
    async def connect_dashboard(self, websocket: WebSocket):
        """اتصال دشبورد جدید"""
        await websocket.accept()
        self.dashboard_connections.append(websocket)
        logger.info(f"📊 Dashboard client connected: {id(websocket)}")
        
        # ارسال داده اولیه
        initial_data = await self.get_dashboard_data()
        await websocket.send_text(json.dumps(initial_data))
    
    def disconnect_dashboard(self, websocket: WebSocket):
        """قطع ارتباط دشبورد"""
        if websocket in self.dashboard_connections:
            self.dashboard_connections.remove(websocket)
            logger.info(f"📊 Dashboard client disconnected: {id(websocket)}")
    
    async def broadcast_dashboard_update(self):
        """ارسال بروزرسانی به تمام دشبوردها"""
        dashboard_data = await self.get_dashboard_data()
        self.dashboard_data_buffer.append(dashboard_data)
        
        # آپدیت بینش‌های نرمال‌سازی
        self._update_normalization_insights(dashboard_data)
        
        disconnected_connections = []
        
        for connection in self.dashboard_connections:
            try:
                await connection.send_text(json.dumps(dashboard_data))
            except Exception as e:
                logger.error(f"❌ Error sending to dashboard: {e}")
                disconnected_connections.append(connection)
        
        # حذف connectionهای قطع شده
        for connection in disconnected_connections:
            self.disconnect_dashboard(connection)
    
    def _update_normalization_insights(self, dashboard_data: Dict[str, Any]):
        """آپدیت بینش‌های نرمال‌سازی"""
        try:
            # ثبت متریک‌های نرمال‌سازی
            norm_data = dashboard_data.get('data_normalization', {})
            if norm_data:
                self.normalization_insights['performance_metrics'].append({
                    'timestamp': datetime.now().isoformat(),
                    'success_rate': norm_data.get('success_rate', 0),
                    'total_processed': norm_data.get('total_processed', 0),
                    'avg_quality': norm_data.get('data_quality', {}).get('avg_quality_score', 0)
                })
            
            # ثبت الگوهای ساختاری
            common_structures = norm_data.get('common_structures', {})
            if common_structures:
                main_structure = max(common_structures.items(), key=lambda x: x[1], default=(None, 0))[0]
                if main_structure:
                    self.normalization_insights['structure_evolution'].append({
                        'timestamp': datetime.now().isoformat(),
                        'main_structure': main_structure,
                        'distribution': common_structures
                    })
                    
        except Exception as e:
            logger.error(f"❌ Error updating normalization insights: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """دریافت داده‌های دشبورد"""
        # داده‌های Real-Time
        current_metrics = self.metrics_collector.get_current_metrics()
        endpoint_stats = self.debug_manager.get_endpoint_stats()
        recent_calls = self.debug_manager.get_recent_calls(limit=20)
        
        # متریک‌های نرمال‌سازی
        normalization_metrics = data_normalizer.get_health_metrics()
        
        # محاسبه آمار کلی
        total_calls = endpoint_stats['overall']['total_calls']
        success_rate = endpoint_stats['overall']['overall_success_rate']
        
        # آمار نرمال‌سازی از endpointها
        total_normalized = endpoint_stats['overall'].get('normalization_overview', {}).get('total_normalized', 0)
        norm_success_rate = endpoint_stats['overall'].get('normalization_overview', {}).get('normalization_success_rate', 0)
        
        # اندپوینت‌های پرکاربرد
        popular_endpoints = sorted(
            [(ep, stats['total_calls']) for ep, stats in endpoint_stats['endpoints'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # کندترین اندپوینت‌ها
        slow_endpoints = sorted(
            [(ep, stats['average_response_time']) for ep, stats in endpoint_stats['endpoints'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # اندپوینت‌های با بهترین کیفیت داده
        quality_endpoints = sorted(
            [(ep, stats.get('normalization_success_rate', 0)) for ep, stats in endpoint_stats['endpoints'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_requests': total_calls,
                'success_rate': round(success_rate, 2),
                'active_connections': len(self.dashboard_connections),
                'system_uptime': self._get_system_uptime(),
                'data_normalization': {  # ✅ اضافه شد
                    'total_normalized': total_normalized,
                    'normalization_success_rate': round(norm_success_rate, 2),
                    'system_success_rate': normalization_metrics.success_rate
                }
            },
            'system_metrics': {
                'cpu': {
                    'usage': current_metrics['cpu']['percent'],
                    'cores': len(current_metrics['cpu']['per_core']),
                    'load_average': current_metrics['cpu']['load_average']
                },
                'memory': {
                    'usage': current_metrics['memory']['percent'],
                    'used_gb': current_metrics['memory']['used_gb'],
                    'total_gb': current_metrics['memory']['total_gb']
                },
                'disk': {
                    'usage': current_metrics['disk']['usage_percent'],
                    'used_gb': current_metrics['disk']['used_gb'],
                    'total_gb': current_metrics['disk']['total_gb']
                },
                'network': {
                    'upload_mbps': current_metrics['network']['mb_sent_per_sec'],
                    'download_mbps': current_metrics['network']['mb_recv_per_sec']
                }
            },
            'data_normalization': {  # ✅ اضافه شد
                'success_rate': normalization_metrics.success_rate,
                'total_processed': normalization_metrics.total_processed,
                'total_errors': normalization_metrics.total_errors,
                'common_structures': normalization_metrics.common_structures,
                'performance_metrics': normalization_metrics.performance_metrics,
                'data_quality': normalization_metrics.data_quality,
                'alerts': normalization_metrics.alerts[-3:]  # آخرین ۳ هشدار
            },
            'endpoints': {
                'popular': [
                    {'endpoint': ep, 'calls': calls} 
                    for ep, calls in popular_endpoints
                ],
                'slowest': [
                    {'endpoint': ep, 'response_time': round(rt, 3)} 
                    for ep, rt in slow_endpoints
                ],
                'best_quality': [  # ✅ اضافه شد
                    {'endpoint': ep, 'quality_score': round(score, 2)} 
                    for ep, score in quality_endpoints
                ]
            },
            'recent_activity': {
                'calls': recent_calls,
                'alerts': self.debug_manager.get_active_alerts()[:10],
                'normalization_insights': {  # ✅ اضافه شد
                    'recent_structures': list(self.normalization_insights['structure_evolution'])[-5:],
                    'quality_trend': list(self.normalization_insights['quality_trends'])[-10:],
                    'performance_history': list(self.normalization_insights['performance_metrics'])[-15:]
                }
            },
            'performance_indicators': {
                'avg_response_time': endpoint_stats['overall'].get('average_response_time', 0),
                'cache_hit_rate': self._calculate_overall_cache_hit_rate(endpoint_stats),
                'error_rate': 100 - success_rate,
                'data_quality_score': normalization_metrics.data_quality.get('avg_quality_score', 0),  # ✅ اضافه شد
                'normalization_efficiency': self._calculate_normalization_efficiency(endpoint_stats)  # ✅ اضافه شد
            }
        }
    
    def _calculate_normalization_efficiency(self, endpoint_stats: Dict) -> float:
        """محاسبه کارایی نرمال‌سازی"""
        try:
            norm_overview = endpoint_stats['overall'].get('normalization_overview', {})
            total_normalized = norm_overview.get('total_normalized', 0)
            total_calls = endpoint_stats['overall']['total_calls']
            
            if total_calls > 0:
                efficiency = (total_normalized / total_calls) * 100
                return round(efficiency, 2)
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error calculating normalization efficiency: {e}")
            return 0.0
    
    def _get_system_uptime(self) -> str:
        """دریافت آپتایم سیستم"""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            return str(uptime).split('.')[0]
        except:
            return "Unknown"
    
    def _calculate_overall_cache_hit_rate(self, endpoint_stats: Dict) -> float:
        """محاسبه نرخ کلی hit کش"""
        total_hits = 0
        total_misses = 0
        
        for stats in endpoint_stats['endpoints'].values():
            cache_perf = stats.get('cache_performance', {})
            total_hits += cache_perf.get('hits', 0)
            total_misses += cache_perf.get('misses', 0)
        
        total = total_hits + total_misses
        return (total_hits / total * 100) if total > 0 else 0
    
    async def start_dashboard_broadcast(self):
        """شروع برودکست دوره‌ای دشبورد"""
        while True:
            try:
                await self.broadcast_dashboard_update()
                await asyncio.sleep(2)  # بروزرسانی هر ۲ ثانیه
            except Exception as e:
                logger.error(f"❌ Dashboard broadcast error: {e}")
                await asyncio.sleep(5)
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """دریافت آمار دشبورد"""
        return {
            'active_dashboards': len(self.dashboard_connections),
            'data_buffer_size': len(self.dashboard_data_buffer),
            'last_broadcast': datetime.now().isoformat(),
            'total_broadcasts': len(self.dashboard_data_buffer),
            'normalization_insights': {  # ✅ اضافه شد
                'structure_records': len(self.normalization_insights['structure_evolution']),
                'quality_records': len(self.normalization_insights['quality_trends']),
                'performance_records': len(self.normalization_insights['performance_metrics'])
            }
        }
    
    def get_normalization_widget_data(self) -> Dict[str, Any]:
        """دریافت داده‌های ویژه ویجت نرمال‌سازی"""
        try:
            metrics = data_normalizer.get_health_metrics()
            analysis = data_normalizer.get_deep_analysis()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': {
                    'success_rate': metrics.success_rate,
                    'total_processed': metrics.total_processed,
                    'avg_processing_time': metrics.performance_metrics.get('avg_processing_time_ms', 0),
                    'data_quality': metrics.data_quality
                },
                'structure_analysis': metrics.common_structures,
                'performance_trends': analysis.get('performance_analysis', {}),
                'recommendations': analysis.get('recommendations', [])[:3]  # ۳ توصیه برتر
            }
        except Exception as e:
            logger.error(f"❌ Error getting normalization widget data: {e}")
            return {'error': str(e)}

# ایجاد نمونه گلوبال
live_dashboard = None

def initialize_live_dashboard(debug_manager, metrics_collector):
    """تابع مقداردهی اولیه برای live dashboard"""
    global live_dashboard
    live_dashboard = LiveDashboardManager(debug_manager, metrics_collector)
    logger.info("✅ Live Dashboard Global Instance Initialized")
    
    # شروع برودکست در background
    asyncio.create_task(live_dashboard.start_dashboard_broadcast())
    
    return live_dashboard

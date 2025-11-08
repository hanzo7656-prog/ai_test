import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import WebSocket
from collections import defaultdict, deque
import psutil

logger = logging.getLogger(__name__)

class LiveDashboardManager:
    def __init__(self, debug_manager, metrics_collector):  # ✅ اصلاح signature
        self.debug_manager = debug_manager
        self.metrics_collector = metrics_collector
        self.dashboard_connections: List[WebSocket] = []
        self.dashboard_data_buffer = deque(maxlen=100)
        
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
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """دریافت داده‌های دشبورد"""
        # داده‌های Real-Time
        current_metrics = self.metrics_collector.get_current_metrics()
        endpoint_stats = self.debug_manager.get_endpoint_stats()
        recent_calls = self.debug_manager.get_recent_calls(limit=20)
        
        # محاسبه آمار کلی
        total_calls = endpoint_stats['overall']['total_calls']
        success_rate = endpoint_stats['overall']['overall_success_rate']
        
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
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_requests': total_calls,
                'success_rate': round(success_rate, 2),
                'active_connections': len(self.dashboard_connections),
                'system_uptime': self._get_system_uptime()
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
            'endpoints': {
                'popular': [
                    {'endpoint': ep, 'calls': calls} 
                    for ep, calls in popular_endpoints
                ],
                'slowest': [
                    {'endpoint': ep, 'response_time': round(rt, 3)} 
                    for ep, rt in slow_endpoints
                ]
            },
            'recent_activity': {
                'calls': recent_calls,
                'alerts': self.debug_manager.get_active_alerts()[:10]  # ✅ از debug_manager استفاده می‌کند
            },
            'performance_indicators': {
                'avg_response_time': endpoint_stats['overall'].get('average_response_time', 0),
                'cache_hit_rate': self._calculate_overall_cache_hit_rate(endpoint_stats),
                'error_rate': 100 - success_rate
            }
        }
    
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
            'total_broadcasts': len(self.dashboard_data_buffer)
        }

# ایجاد نمونه گلوبال (بعداً مقداردهی می‌شود)
live_dashboard = None

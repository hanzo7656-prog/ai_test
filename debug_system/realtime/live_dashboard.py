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
        
        # بهینه‌سازی: delta updates
        self.last_dashboard_data = None
        self.update_interval = 5  # 5 seconds (به جای 2)
        
        # اتصال به central_monitor
        self._connect_to_central_monitor()
        
        logger.info("✅ Live Dashboard Manager Initialized - Delta Updates")
        
    def _connect_to_central_monitor(self):
        """اتصال به central_monitor برای دریافت real-time metrics"""
        try:
            from ..core.system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت متریک‌ها
                central_monitor.subscribe("live_dashboard", self._on_metrics_received)
                logger.info("✅ LiveDashboard subscribed to central_monitor")
            else:
                logger.warning("⚠️ Central monitor not available - using metrics_collector")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - using metrics_collector")
        except Exception as e:
            logger.error(f"❌ Error connecting to central_monitor: {e}")
    
    def _on_metrics_received(self, metrics: Dict[str, Any]):
        """دریافت متریک‌ها از central_monitor"""
        try:
            # ذخیره متریک‌های سیستم
            system_metrics = metrics.get('system', {})
            
            # آپدیت normalization insights اگر data موجود باشد
            norm_data = metrics.get('data_normalization', {})
            if norm_data:
                self._update_normalization_insights_from_central(norm_data, metrics['timestamp'])
                
            logger.debug(f"📈 LiveDashboard received metrics from central_monitor")
            
        except Exception as e:
            logger.error(f"❌ Error processing metrics from central_monitor: {e}")
    
    def _update_normalization_insights_from_central(self, norm_data: Dict[str, Any], timestamp: str):
        """آپدیت بینش‌های نرمال‌سازی از central_monitor"""
        try:
            self.normalization_insights['performance_metrics'].append({
                'timestamp': timestamp,
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
                        'timestamp': timestamp,
                        'main_structure': main_structure,
                        'distribution': common_structures
                    })
                    
        except Exception as e:
            logger.error(f"❌ Error updating normalization insights from central: {e}")
    
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
        """ارسال بروزرسانی به تمام دشبوردها با delta updates"""
        try:
            current_data = await self.get_dashboard_data()
            
            # محاسبه delta (فقط تغییرات)
            delta_data = self._calculate_delta_update(current_data)
            
            # فقط اگر تغییری وجود دارد broadcast کن
            if delta_data:
                self.dashboard_data_buffer.append(current_data)
                self.last_dashboard_data = current_data
                
                await self._send_delta_updates(delta_data)
            else:
                # فقط timestamp آپدیت کن
                await self._send_heartbeat()
                
        except Exception as e:
            logger.error(f"❌ Dashboard broadcast error: {e}")
    
    def _calculate_delta_update(self, current_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """محاسبه delta update"""
        if not self.last_dashboard_data:
            return current_data  # اولین بروزرسانی کامل است
        
        delta = {}
        changed = False
        
        # بررسی تغییرات در system metrics
        current_system = current_data.get('system_metrics', {})
        last_system = self.last_dashboard_data.get('system_metrics', {})
        
        system_delta = self._calculate_metrics_delta(current_system, last_system)
        if system_delta:
            delta['system_metrics'] = system_delta
            changed = True
        
        # بررسی تغییرات در endpoints
        current_endpoints = current_data.get('endpoints', {})
        last_endpoints = self.last_dashboard_data.get('endpoints', {})
        
        endpoints_delta = self._calculate_endpoints_delta(current_endpoints, last_endpoints)
        if endpoints_delta:
            delta['endpoints'] = endpoints_delta
            changed = True
        
        # بررسی تغییرات در normalization
        current_norm = current_data.get('data_normalization', {})
        last_norm = self.last_dashboard_data.get('data_normalization', {})
        
        norm_delta = self._calculate_normalization_delta(current_norm, last_norm)
        if norm_delta:
            delta['data_normalization'] = norm_delta
            changed = True
        
        if changed:
            delta['timestamp'] = current_data['timestamp']
            delta['type'] = 'delta_update'
            return delta
        
        return None
    
    def _calculate_metrics_delta(self, current: Dict, last: Dict) -> Dict:
        """محاسبه delta برای metrics"""
        delta = {}
        threshold = 1.0  # 1% threshold for changes
        
        # CPU
        current_cpu = current.get('cpu', {}).get('percent', 0)
        last_cpu = last.get('cpu', {}).get('percent', 0)
        if abs(current_cpu - last_cpu) > threshold:
            delta['cpu'] = current.get('cpu', {})
        
        # Memory
        current_mem = current.get('memory', {}).get('percent', 0)
        last_mem = last.get('memory', {}).get('percent', 0)
        if abs(current_mem - last_mem) > threshold:
            delta['memory'] = current.get('memory', {})
        
        # Disk
        current_disk = current.get('disk', {}).get('usage_percent', 0)
        last_disk = last.get('disk', {}).get('usage_percent', 0)
        if abs(current_disk - last_disk) > threshold:
            delta['disk'] = current.get('disk', {})
        
        return delta
    
    def _calculate_endpoints_delta(self, current: Dict, last: Dict) -> Dict:
        """محاسبه delta برای endpoints"""
        delta = {}
        
        # فقط endpointهای با تغییرات significant
        for category in ['popular', 'slowest', 'best_quality']:
            current_list = current.get(category, [])
            last_list = last.get(category, [])
            
            # اگر لیست تغییر کرده
            if current_list != last_list:
                delta[category] = current_list[:5]  # فقط ۵ آیتم اول
        
        return delta
    
    def _calculate_normalization_delta(self, current: Dict, last: Dict) -> Dict:
        """محاسبه delta برای normalization"""
        delta = {}
        threshold = 2.0  # 2% threshold
        
        # Success rate
        current_success = current.get('success_rate', 0)
        last_success = last.get('success_rate', 0)
        if abs(current_success - last_success) > threshold:
            delta['success_rate'] = current_success
        
        # Data quality
        current_quality = current.get('data_quality', {}).get('avg_quality_score', 0)
        last_quality = last.get('data_quality', {}).get('avg_quality_score', 0)
        if abs(current_quality - last_quality) > threshold:
            delta['data_quality'] = current.get('data_quality', {})
        
        return delta
    
    async def _send_delta_updates(self, delta_data: Dict[str, Any]):
        """ارسال delta updates"""
        disconnected_connections = []
        delta_json = json.dumps(delta_data)
        
        for connection in self.dashboard_connections:
            try:
                await connection.send_text(delta_json)
            except Exception as e:
                logger.error(f"❌ Error sending delta update to dashboard: {e}")
                disconnected_connections.append(connection)
        
        # حذف connectionهای قطع شده
        for connection in disconnected_connections:
            self.disconnect_dashboard(connection)
    
    async def _send_heartbeat(self):
        """ارسال heartbeat (فقط timestamp)"""
        heartbeat = {
            'type': 'heartbeat',
            'timestamp': datetime.now().isoformat()
        }
        
        disconnected_connections = []
        heartbeat_json = json.dumps(heartbeat)
        
        for connection in self.dashboard_connections:
            try:
                await connection.send_text(heartbeat_json)
            except Exception:
                disconnected_connections.append(connection)
        
        # حذف connectionهای قطع شده
        for connection in disconnected_connections:
            self.disconnect_dashboard(connection)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """دریافت داده‌های دشبورد"""
        # استفاده از central_monitor اگر available باشد
        try:
            from ..core.system_monitor import central_monitor
            if central_monitor:
                metrics = central_monitor.get_current_metrics()
                system_metrics = metrics.get('system', {})
                norm_metrics = metrics.get('data_normalization', {})
            else:
                raise ImportError("Central monitor not available")
        except:
            # Fallback به metrics_collector
            current_metrics = self.metrics_collector.get_current_metrics()
            system_metrics = current_metrics
            norm_metrics = current_metrics.get('data_normalization', {})
        
        # داده‌های endpoint از debug_manager
        endpoint_stats = self.debug_manager.get_endpoint_stats()
        recent_calls = self.debug_manager.get_recent_calls(limit=20)
        
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
                'data_normalization': {
                    'total_normalized': total_normalized,
                    'normalization_success_rate': round(norm_success_rate, 2),
                    'system_success_rate': norm_metrics.get('success_rate', 0)
                }
            },
            'system_metrics': {
                'cpu': {
                    'usage': system_metrics.get('cpu', {}).get('percent', 0),
                    'cores': len(system_metrics.get('cpu', {}).get('per_core', [])),
                    'load_average': system_metrics.get('cpu', {}).get('load_average', [])
                },
                'memory': {
                    'usage': system_metrics.get('memory', {}).get('percent', 0),
                    'used_gb': system_metrics.get('memory', {}).get('used_gb', 0),
                    'total_gb': system_metrics.get('memory', {}).get('total_gb', 0)
                },
                'disk': {
                    'usage': system_metrics.get('disk', {}).get('usage_percent', 0),
                    'used_gb': system_metrics.get('disk', {}).get('used_gb', 0),
                    'total_gb': system_metrics.get('disk', {}).get('total_gb', 0)
                },
                'network': {
                    'upload_mbps': system_metrics.get('network', {}).get('mb_sent_per_sec', 0),
                    'download_mbps': system_metrics.get('network', {}).get('mb_recv_per_sec', 0)
                }
            },
            'data_normalization': {
                'success_rate': norm_metrics.get('success_rate', 0),
                'total_processed': norm_metrics.get('total_processed', 0),
                'total_errors': norm_metrics.get('total_errors', 0),
                'common_structures': norm_metrics.get('common_structures', {}),
                'performance_metrics': norm_metrics.get('performance_metrics', {}),
                'data_quality': norm_metrics.get('data_quality', {}),
                'alerts': norm_metrics.get('alerts', [])[-3:]  # آخرین ۳ هشدار
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
                'best_quality': [
                    {'endpoint': ep, 'quality_score': round(score, 2)} 
                    for ep, score in quality_endpoints
                ]
            },
            'recent_activity': {
                'calls': recent_calls,
                'alerts': self.debug_manager.get_active_alerts()[:10],
                'normalization_insights': {
                    'recent_structures': list(self.normalization_insights['structure_evolution'])[-5:],
                    'quality_trend': list(self.normalization_insights['quality_trends'])[-10:],
                    'performance_history': list(self.normalization_insights['performance_metrics'])[-15:]
                }
            },
            'performance_indicators': {
                'avg_response_time': endpoint_stats['overall'].get('average_response_time', 0),
                'cache_hit_rate': self._calculate_overall_cache_hit_rate(endpoint_stats),
                'error_rate': 100 - success_rate,
                'data_quality_score': norm_metrics.get('data_quality', {}).get('avg_quality_score', 0),
                'normalization_efficiency': self._calculate_normalization_efficiency(endpoint_stats)
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
                await asyncio.sleep(self.update_interval)  # 5 seconds
            except Exception as e:
                logger.error(f"❌ Dashboard broadcast error: {e}")
                await asyncio.sleep(self.update_interval * 2)
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """دریافت آمار دشبورد"""
        return {
            'active_dashboards': len(self.dashboard_connections),
            'data_buffer_size': len(self.dashboard_data_buffer),
            'last_broadcast': datetime.now().isoformat(),
            'total_broadcasts': len(self.dashboard_data_buffer),
            'update_interval': self.update_interval,
            'delta_updates_enabled': True,
            'normalization_insights': {
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

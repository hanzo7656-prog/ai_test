"""
Debug Manager - Pure Consumer Version
Only processes metrics from central_monitor, no independent monitoring
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EndpointCall:
    endpoint: str
    method: str
    timestamp: datetime
    params: Dict[str, Any]
    response_time: float
    status_code: int
    cache_used: bool
    api_calls: int

class DebugManager:
    """
    Pure consumer debug manager
    Only processes metrics received from central_monitor
    """
    
    def __init__(self):
        self.endpoint_calls = deque(maxlen=10000)
        self.system_metrics_history = deque(maxlen=1000)
        self.endpoint_stats = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'last_call': None,
            'errors': []
        })
        
        self.alerts = []
        self.performance_thresholds = {
            'response_time_warning': 1.0,
            'response_time_critical': 3.0,
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0
        }
        
        self.alert_manager = None
        self._lock = threading.RLock()
        
        # Subscribe to central_monitor
        self._subscribe_to_central_monitor()
        
        logger.info("✅ PureDebugManager Initialized (Central Monitor Only)")
    
    def _subscribe_to_central_monitor(self):
        """Subscribe to central monitor for metrics updates"""
        try:
            from .system_monitor import central_monitor
            
            if central_monitor:
                central_monitor.subscribe("pure_debug_manager", self._on_metrics_received)
                logger.info("✅ PureDebugManager subscribed to central_monitor")
            else:
                logger.warning("⚠️ Central monitor not available - debug manager will remain idle")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - debug manager idle")
        except Exception as e:
            logger.error(f"❌ Error subscribing to central_monitor: {e}")
    
    def _on_metrics_received(self, metrics: Dict[str, Any]):
        """Process metrics received from central_monitor"""
        try:
            system_metrics = metrics.get('system', {})
            
            # Store in history
            history_entry = {
                'timestamp': datetime.fromisoformat(metrics['timestamp']),
                'cpu_percent': system_metrics.get('cpu', {}).get('percent', 0),
                'memory_percent': system_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': system_metrics.get('disk', {}).get('usage_percent', 0),
                'network_io': {
                    'bytes_sent': system_metrics.get('network', {}).get('bytes_sent', 0),
                    'bytes_recv': system_metrics.get('network', {}).get('bytes_recv', 0)
                },
                'active_connections': system_metrics.get('network', {}).get('connections', 0)
            }
            
            with self._lock:
                self.system_metrics_history.append(history_entry)
            
            # Check for system health alerts
            self._check_system_health_alerts(system_metrics)
            
            logger.debug(f"📊 Debug metrics updated from central_monitor")
            
        except Exception as e:
            logger.error(f"❌ Error processing debug metrics: {e}")
    
    def _check_system_health_alerts(self, metrics: Dict[str, Any]):
        """Check system health and create alerts if needed"""
        try:
            cpu_usage = metrics.get('cpu', {}).get('percent', 0)
            memory_usage = metrics.get('memory', {}).get('percent', 0)
            
            if cpu_usage > self.performance_thresholds['cpu_critical']:
                self._create_alert(
                    'CRITICAL',
                    f"Critical CPU usage: {cpu_usage:.1f}%",
                    "system_monitor",
                    {'cpu_usage': cpu_usage}
                )
            elif cpu_usage > self.performance_thresholds['cpu_warning']:
                self._create_alert(
                    'WARNING', 
                    f"High CPU usage: {cpu_usage:.1f}%",
                    "system_monitor",
                    {'cpu_usage': cpu_usage}
                )
            
            if memory_usage > self.performance_thresholds['memory_critical']:
                self._create_alert(
                    'CRITICAL',
                    f"Critical memory usage: {memory_usage:.1f}%",
                    "system_monitor",
                    {'memory_usage': memory_usage}
                )
            elif memory_usage > self.performance_thresholds['memory_warning']:
                self._create_alert(
                    'WARNING',
                    f"High memory usage: {memory_usage:.1f}%",
                    "system_monitor",
                    {'memory_usage': memory_usage}
                )
                
        except Exception as e:
            logger.error(f"❌ Error checking system health: {e}")
    
    def _create_alert(self, level: str, message: str, source: str, data: Dict[str, Any]):
        """Create an alert"""
        try:
            alert = {
                'id': len(self.alerts) + 1,
                'level': level,
                'message': message,
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'acknowledged': False
            }
            
            with self._lock:
                self.alerts.append(alert)
            
            logger.warning(f"🚨 {level} Alert: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create alert: {e}")
    
    def log_endpoint_call(self, endpoint: str, method: str, params: Dict[str, Any], 
                         response_time: float, status_code: int, cache_used: bool, 
                         api_calls: int = 0):
        """Log an endpoint call"""
        try:
            call = EndpointCall(
                endpoint=endpoint,
                method=method,
                timestamp=datetime.now(),
                params=params,
                response_time=response_time,
                status_code=status_code,
                cache_used=cache_used,
                api_calls=api_calls
            )
            
            with self._lock:
                self.endpoint_calls.append(call)
                
                # Update endpoint stats
                stats = self.endpoint_stats[endpoint]
                stats['total_calls'] += 1
                stats['total_response_time'] += response_time
                
                if 200 <= status_code < 300:
                    stats['successful_calls'] += 1
                else:
                    stats['failed_calls'] += 1
                    stats['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'status_code': status_code,
                        'params': params
                    })
                
                if cache_used:
                    stats['cache_hits'] += 1
                else:
                    stats['cache_misses'] += 1
                
                stats['api_calls'] += api_calls
                stats['last_call'] = datetime.now().isoformat()
            
            # Check performance alerts
            if response_time > self.performance_thresholds['response_time_critical']:
                self._create_alert(
                    'CRITICAL',
                    f"Critical response time in {endpoint}: {response_time:.2f}s",
                    endpoint,
                    {'response_time': response_time}
                )
            elif response_time > self.performance_thresholds['response_time_warning']:
                self._create_alert(
                    'WARNING',
                    f"High response time in {endpoint}: {response_time:.2f}s", 
                    endpoint,
                    {'response_time': response_time}
                )
            
            logger.debug(f"📝 Endpoint logged: {endpoint} - {response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error logging endpoint call: {e}")
    
    def set_alert_manager(self, alert_manager):
        """Set external alert manager"""
        self.alert_manager = alert_manager
        logger.info("✅ Alert manager configured")
        return True
    
    def get_endpoint_stats(self, endpoint: str = None) -> Dict[str, Any]:
        """Get endpoint statistics"""
        try:
            with self._lock:
                if endpoint:
                    if endpoint not in self.endpoint_stats:
                        return {'error': 'Endpoint not found'}
                    
                    stats = self.endpoint_stats[endpoint]
                    avg_response_time = (stats['total_response_time'] / stats['total_calls']) if stats['total_calls'] > 0 else 0
                    
                    return {
                        'endpoint': endpoint,
                        'total_calls': stats['total_calls'],
                        'successful_calls': stats['successful_calls'],
                        'failed_calls': stats['failed_calls'],
                        'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                        'average_response_time': round(avg_response_time, 3),
                        'cache_performance': {
                            'hits': stats['cache_hits'],
                            'misses': stats['cache_misses'],
                            'hit_rate': (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) 
                            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
                        },
                        'api_calls': stats['api_calls'],
                        'recent_errors': stats['errors'][-10:],
                        'last_call': stats['last_call']
                    }
                else:
                    # All endpoints
                    all_stats = {}
                    total_calls = 0
                    total_success = 0
                    
                    for ep, stats in self.endpoint_stats.items():
                        all_stats[ep] = {
                            'total_calls': stats['total_calls'],
                            'success_rate': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                            'average_response_time': round((stats['total_response_time'] / stats['total_calls']), 3) if stats['total_calls'] > 0 else 0,
                            'last_call': stats['last_call']
                        }
                        total_calls += stats['total_calls']
                        total_success += stats['successful_calls']
                    
                    return {
                        'overall': {
                            'total_endpoints': len(self.endpoint_stats),
                            'total_calls': total_calls,
                            'overall_success_rate': (total_success / total_calls * 100) if total_calls > 0 else 0,
                            'timestamp': datetime.now().isoformat()
                        },
                        'endpoints': all_stats
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error getting endpoint stats: {e}")
            return {'error': f'Failed to get stats: {str(e)}'}
    
    def get_recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent endpoint calls"""
        try:
            with self._lock:
                recent_calls = list(self.endpoint_calls)[-limit:]
                return [
                    {
                        'endpoint': call.endpoint,
                        'method': call.method,
                        'timestamp': call.timestamp.isoformat(),
                        'response_time': call.response_time,
                        'status_code': call.status_code,
                        'cache_used': call.cache_used,
                        'api_calls': call.api_calls
                    }
                    for call in recent_calls
                ]
        except Exception as e:
            logger.error(f"❌ Error getting recent calls: {e}")
            return []
    
    def get_system_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get system metrics history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self._lock:
                metrics_history = [
                    metrics for metrics in self.system_metrics_history
                    if metrics['timestamp'] >= cutoff_time
                ]
            
            return [
                {
                    'timestamp': metrics['timestamp'].isoformat(),
                    'cpu_percent': metrics['cpu_percent'],
                    'memory_percent': metrics['memory_percent'],
                    'disk_usage': metrics['disk_usage'],
                    'network_io': metrics['network_io'],
                    'active_connections': metrics['active_connections']
                }
                for metrics in metrics_history
            ]
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return []
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        with self._lock:
            return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_id: int):
        """Acknowledge an alert"""
        with self._lock:
            for alert in self.alerts:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    logger.info(f"✅ Alert {alert_id} acknowledged")
                    break
    
    def get_debug_status(self) -> Dict[str, Any]:
        """Get debug manager status"""
        try:
            endpoint_stats = self.get_endpoint_stats()
            recent_calls = self.get_recent_calls(10)
            system_metrics = self.get_system_metrics_history(1)
            active_alerts = self.get_active_alerts()
            
            return {
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'overview': {
                    'total_endpoints': endpoint_stats.get('overall', {}).get('total_endpoints', 0),
                    'total_calls': endpoint_stats.get('overall', {}).get('total_calls', 0),
                    'success_rate': endpoint_stats.get('overall', {}).get('overall_success_rate', 0),
                    'active_alerts': len(active_alerts)
                },
                'system_health': {
                    'cpu_usage': system_metrics[-1]['cpu_percent'] if system_metrics else 0,
                    'memory_usage': system_metrics[-1]['memory_percent'] if system_metrics else 0,
                    'data_points': len(system_metrics)
                },
                'recent_activity': {
                    'calls_count': len(recent_calls),
                    'alerts_count': len(active_alerts[:5])
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting debug status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def clear_old_data(self, days: int = 7):
        """Clear old data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with self._lock:
                self.endpoint_calls = deque(
                    [call for call in self.endpoint_calls if call.timestamp > cutoff_time],
                    maxlen=10000
                )
                
                self.system_metrics_history = deque(
                    [metrics for metrics in self.system_metrics_history if metrics['timestamp'] > cutoff_time],
                    maxlen=1000
                )
                
                self.alerts = [alert for alert in self.alerts 
                             if datetime.fromisoformat(alert['timestamp']) > cutoff_time]
            
            logger.info(f"🧹 Cleared data older than {days} days")
            
        except Exception as e:
            logger.error(f"❌ Error clearing old data: {e}")

# Global instance
debug_manager = DebugManager()

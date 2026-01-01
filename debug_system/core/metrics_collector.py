"""
Metrics Collector - Pure Consumer Version
Only receives metrics from central_monitor, no independent monitoring
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
class APIPerformanceMetrics:
    endpoint: str
    latency_ms: float
    status_code: int
    timestamp: datetime
    request_size_bytes: int = 0
    response_size_bytes: int = 0

class AdvancedMetricsCollector:
    """
    Pure consumer metrics collector
    Only processes metrics received from central_monitor
    """
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=3600)
        self.current_metrics_cache = {
            'cpu': {'percent': 0},
            'memory': {'percent': 0},
            'disk': {'usage_percent': 0},
            'network': {'bytes_sent': 0, 'bytes_recv': 0},
            'process': {'memory_mb': 0, 'cpu_percent': 0}
        }
        
        self.cache_last_updated = None
        self.api_endpoints = {}
        self.external_services = {}
        
        # Subscribe to central_monitor
        self._subscribe_to_central_monitor()
        
        logger.info("✅ PureMetricsCollector Initialized (Central Monitor Only)")
    
    def _subscribe_to_central_monitor(self):
        """Subscribe to central monitor for metrics updates"""
        try:
            # Wait for central_monitor to be available
            from .system_monitor import central_monitor
            
            if central_monitor:
                central_monitor.subscribe("pure_metrics_collector", self._on_metrics_received)
                logger.info("✅ PureMetricsCollector subscribed to central_monitor")
            else:
                logger.warning("⚠️ Central monitor not available - collector will remain idle")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - collector idle")
        except Exception as e:
            logger.error(f"❌ Error subscribing to central_monitor: {e}")
    
    def _on_metrics_received(self, metrics: Dict[str, Any]):
        """Process metrics received from central_monitor"""
        try:
            system_metrics = metrics.get('system', {})
            
            # Update cache with received metrics
            self.current_metrics_cache.update({
                'cpu': {
                    'percent': system_metrics.get('cpu', {}).get('percent', 0),
                    'per_core': system_metrics.get('cpu', {}).get('per_core', []),
                    'load_average': system_metrics.get('cpu', {}).get('load_average', [])
                },
                'memory': {
                    'percent': system_metrics.get('memory', {}).get('percent', 0),
                    'used_gb': system_metrics.get('memory', {}).get('used_gb', 0),
                    'available_gb': system_metrics.get('memory', {}).get('available_gb', 0)
                },
                'disk': {
                    'usage_percent': system_metrics.get('disk', {}).get('usage_percent', 0),
                    'used_gb': system_metrics.get('disk', {}).get('used_gb', 0),
                    'free_gb': system_metrics.get('disk', {}).get('free_gb', 0)
                },
                'network': {
                    'bytes_sent': system_metrics.get('network', {}).get('bytes_sent', 0),
                    'bytes_recv': system_metrics.get('network', {}).get('bytes_recv', 0),
                    'connections': system_metrics.get('network', {}).get('connections', 0)
                },
                'process': {
                    'memory_mb': system_metrics.get('process', {}).get('memory_rss_mb', 0),
                    'cpu_percent': system_metrics.get('process', {}).get('cpu_percent', 0),
                    'threads': system_metrics.get('process', {}).get('threads_count', 0)
                }
            })
            
            self.cache_last_updated = datetime.now()
            
            # Add to history buffer
            self._add_to_history_buffer(system_metrics)
            
            logger.debug(f"📈 Metrics updated from central_monitor")
            
        except Exception as e:
            logger.error(f"❌ Error processing metrics: {e}")
    
    def _add_to_history_buffer(self, system_metrics: Dict[str, Any]):
        """Add metrics to history buffer"""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'cpu_percent': system_metrics.get('cpu', {}).get('percent', 0),
                'memory_percent': system_metrics.get('memory', {}).get('percent', 0),
                'disk_usage': system_metrics.get('disk', {}).get('usage_percent', 0),
                'network_sent': system_metrics.get('network', {}).get('bytes_sent', 0),
                'network_recv': system_metrics.get('network', {}).get('bytes_recv', 0),
                'process_memory': system_metrics.get('process', {}).get('memory_rss_mb', 0)
            }
            
            self.metrics_buffer.append(history_entry)
            
        except Exception as e:
            logger.error(f"❌ Error adding to history buffer: {e}")
    
    # API methods remain the same (just use cached data)
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics from cache"""
        if not self.cache_last_updated:
            return {
                'status': 'waiting_for_central_monitor',
                'message': 'Metrics collector is waiting for central_monitor updates',
                'timestamp': datetime.now().isoformat()
            }
        
        cache_age = (datetime.now() - self.cache_last_updated).total_seconds()
        
        return {
            **self.current_metrics_cache,
            'cache_age_seconds': cache_age,
            'cache_last_updated': self.cache_last_updated.isoformat(),
            'source': 'central_monitor'
        }
    
    def get_metrics_history(self, seconds: int = 300) -> List[Dict[str, Any]]:
        """Get metrics history"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        return [
            {
                'timestamp': metrics['timestamp'].isoformat(),
                'cpu_percent': metrics['cpu_percent'],
                'memory_percent': metrics['memory_percent'],
                'disk_usage': metrics['disk_usage'],
                'network_sent': metrics['network_sent'],
                'network_recv': metrics['network_recv'],
                'process_memory': metrics['process_memory']
            }
            for metrics in self.metrics_buffer
            if metrics['timestamp'] >= cutoff_time
        ]
    
    def register_api_endpoint(self, endpoint: str, sla_response_time_ms: int = 200):
        """Register API endpoint for tracking"""
        self.api_endpoints[endpoint] = {
            'sla_response_time': sla_response_time_ms,
            'metrics': deque(maxlen=100),
            'total_requests': 0,
            'failed_requests': 0
        }
        logger.info(f"✅ API endpoint registered: {endpoint}")
    
    def record_api_request(self, endpoint: str, latency_ms: float, status_code: int):
        """Record API request"""
        if endpoint not in self.api_endpoints:
            self.register_api_endpoint(endpoint)
        
        endpoint_data = self.api_endpoints[endpoint]
        endpoint_data['total_requests'] += 1
        
        if status_code >= 400:
            endpoint_data['failed_requests'] += 1
        
        metric = APIPerformanceMetrics(
            endpoint=endpoint,
            latency_ms=latency_ms,
            status_code=status_code,
            timestamp=datetime.now()
        )
        
        endpoint_data['metrics'].append(metric)
    
    def get_api_performance(self, endpoint: str = None) -> Dict[str, Any]:
        """Get API performance data"""
        if not self.api_endpoints:
            return {'message': 'No API endpoints registered'}
        
        if endpoint:
            if endpoint not in self.api_endpoints:
                return {'error': f'Endpoint {endpoint} not found'}
            
            data = self.api_endpoints[endpoint]
            metrics = list(data['metrics'])
            
            if not metrics:
                return {'endpoint': endpoint, 'message': 'No metrics available'}
            
            latencies = [m.latency_ms for m in metrics]
            
            return {
                'endpoint': endpoint,
                'total_requests': data['total_requests'],
                'failed_requests': data['failed_requests'],
                'success_rate': ((data['total_requests'] - data['failed_requests']) / 
                                data['total_requests'] * 100) if data['total_requests'] > 0 else 0,
                'average_latency': sum(latencies) / len(latencies) if latencies else 0,
                'recent_requests': len(metrics)
            }
        else:
            # All endpoints
            result = {}
            for ep in self.api_endpoints:
                result[ep] = self.get_api_performance(ep)
            
            return {
                'total_endpoints': len(self.api_endpoints),
                'endpoints': result,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_collector_status(self) -> Dict[str, Any]:
        """Get collector status"""
        return {
            'cache_age_seconds': (datetime.now() - self.cache_last_updated).total_seconds() 
            if self.cache_last_updated else None,
            'metrics_buffer_size': len(self.metrics_buffer),
            'api_endpoints_count': len(self.api_endpoints),
            'external_services_count': len(self.external_services),
            'status': 'active' if self.cache_last_updated else 'waiting',
            'timestamp': datetime.now().isoformat()
        }

# Global instance
metrics_collector = AdvancedMetricsCollector()

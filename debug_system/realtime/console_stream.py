import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

class ConsoleStreamManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.message_buffer = deque(maxlen=1000)
        self.connection_stats = {}
        
    async def connect(self, websocket: WebSocket):
        """اتصال کلاینت جدید به کنسول"""
        await websocket.accept()
        self.active_connections.append(websocket)
        client_id = id(websocket)
        self.connection_stats[client_id] = {
            'connected_at': datetime.now().isoformat(),
            'message_count': 0,
            'last_activity': datetime.now().isoformat()
        }
        
        logger.info(f"🔌 Console client connected: {client_id}")
        
        # ارسال آخرین پیام‌های بافر به کلاینت جدید
        for message in list(self.message_buffer)[-50:]:  # آخرین ۵۰ پیام
            try:
                await websocket.send_text(json.dumps(message))
            except:
                break
    
    def disconnect(self, websocket: WebSocket):
        """قطع ارتباط کلاینت"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = id(websocket)
            self.connection_stats.pop(client_id, None)
            logger.info(f"🔌 Console client disconnected: {client_id}")
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """ارسال پیام به تمام کلاینت‌های متصل"""
        message['timestamp'] = datetime.now().isoformat()
        self.message_buffer.append(message)
        
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
                client_id = id(connection)
                if client_id in self.connection_stats:
                    self.connection_stats[client_id]['message_count'] += 1
                    self.connection_stats[client_id]['last_activity'] = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"❌ Error sending to console client: {e}")
                disconnected_connections.append(connection)
        
        # حذف connectionهای قطع شده
        for connection in disconnected_connections:
            self.disconnect(connection)
    
    def log_endpoint_call(self, endpoint_data: Dict[str, Any]):
        """ثبت لاگ فراخوانی اندپوینت برای کنسول"""
        asyncio.create_task(self.broadcast_message({
            'type': 'endpoint_call',
            'level': 'info',
            'message': f"🔗 {endpoint_data['method']} {endpoint_data['endpoint']} - {endpoint_data['response_time']:.3f}s",
            'data': {
                'endpoint': endpoint_data['endpoint'],
                'method': endpoint_data['method'],
                'response_time': endpoint_data['response_time'],
                'status_code': endpoint_data['status_code'],
                'cache_used': endpoint_data['cache_used']
            }
        }))
    
    def log_system_metrics(self, metrics_data: Dict[str, Any]):
        """ثبت لاگ متریک‌های سیستم برای کنسول"""
        asyncio.create_task(self.broadcast_message({
            'type': 'system_metrics',
            'level': 'info', 
            'message': f"📊 System - CPU: {metrics_data['cpu_percent']}% | Memory: {metrics_data['memory_percent']}% | Disk: {metrics_data['disk_usage']}%",
            'data': metrics_data
        }))
    
    def log_security_alert(self, alert_data: Dict[str, Any]):
        """ثبت هشدار امنیتی برای کنسول"""
        asyncio.create_task(self.broadcast_message({
            'type': 'security_alert',
            'level': 'warning',
            'message': f"🚨 SECURITY: {alert_data.get('message', 'Suspicious activity detected')}",
            'data': alert_data
        }))
    
    def log_performance_alert(self, alert_data: Dict[str, Any]):
        """ثبت هشدار عملکرد برای کنسول"""
        asyncio.create_task(self.broadcast_message({
            'type': 'performance_alert', 
            'level': 'warning',
            'message': f"⚡ PERFORMANCE: {alert_data.get('message', 'Performance issue detected')}",
            'data': alert_data
        }))
    
    def log_error(self, error_data: Dict[str, Any]):
        """ثبت خطا برای کنسول"""
        asyncio.create_task(self.broadcast_message({
            'type': 'error',
            'level': 'error',
            'message': f"🔴 ERROR: {error_data.get('message', 'An error occurred')}",
            'data': error_data
        }))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """دریافت آمار connectionهای کنسول"""
        return {
            'active_connections': len(self.active_connections),
            'total_messages_sent': sum(
                stats['message_count'] for stats in self.connection_stats.values()
            ),
            'connection_details': self.connection_stats,
            'message_buffer_size': len(self.message_buffer),
            'timestamp': datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال
console_stream = ConsoleStreamManager()

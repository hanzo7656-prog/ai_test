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
        
        # بهینه‌سازی: bulk messaging
        self.message_queue = deque(maxlen=100)
        self._start_bulk_processor()
        
        # اتصال به central_monitor برای دریافت alerts
        self._connect_to_central_monitor()
        
        logger.info("✅ Console Stream Manager Initialized - Bulk Mode")
    
    def _connect_to_central_monitor(self):
        """اتصال به central_monitor برای دریافت alerts"""
        try:
            from ..core.system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت alerts
                central_monitor.subscribe("console_stream", self._on_alert_received)
                logger.info("✅ ConsoleStream subscribed to central_monitor alerts")
            else:
                logger.warning("⚠️ Central monitor not available - console will show local alerts only")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - console will show local alerts only")
        except Exception as e:
            logger.error(f"❌ Error connecting to central_monitor: {e}")
    
    def _on_alert_received(self, alert_data: Dict[str, Any]):
        """دریافت alert از central_monitor"""
        try:
            # تبدیل به format console
            console_alert = {
                'type': 'central_alert',
                'level': alert_data.get('level', 'INFO').lower(),
                'message': f"[CENTRAL] {alert_data.get('title', 'Alert')}: {alert_data.get('message', '')}",
                'data': alert_data.get('data', {}),
                'timestamp': alert_data.get('timestamp', datetime.now().isoformat())
            }
            
            # اضافه به queue برای bulk processing
            self.message_queue.append(console_alert)
            
        except Exception as e:
            logger.error(f"❌ Error processing central alert: {e}")
    
    def _start_bulk_processor(self):
        """راه‌اندازی پردازشگر bulk messages"""
        async def bulk_processor():
            """پردازش bulk messages هر 3 ثانیه"""
            while True:
                try:
                    if self.message_queue:
                        await self._process_bulk_messages()
                    await asyncio.sleep(3)  # هر 3 ثانیه
                except Exception as e:
                    logger.error(f"❌ Bulk processor error: {e}")
                    await asyncio.sleep(10)
        
        # اجرا در event loop موجود یا ایجاد جدید
        try:
            loop = asyncio.get_event_loop()
            asyncio.create_task(bulk_processor())
        except RuntimeError:
            # اگر event loop وجود ندارد، ایجاد کن
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.create_task(bulk_processor())
        
        logger.info("🔄 Console bulk message processor started")
    
    async def _process_bulk_messages(self):
        """پردازش bulk messages"""
        try:
            messages = list(self.message_queue)
            self.message_queue.clear()
            
            if not messages:
                return
            
            # اگر فقط یک message است، معمولی ارسال کن
            if len(messages) == 1:
                await self.broadcast_message(messages[0])
                return
            
            # برای چند message، bulk ارسال کن
            bulk_message = {
                'type': 'bulk_update',
                'level': 'info',
                'message': f'Bulk update: {len(messages)} messages',
                'data': {
                    'messages': messages,
                    'count': len(messages),
                    'timestamp': datetime.now().isoformat()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            await self.broadcast_message(bulk_message)
            
        except Exception as e:
            logger.error(f"❌ Error processing bulk messages: {e}")
    
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
        
        # ارسال آخرین پیام‌های بافر به کلاینت جدید (حداکثر 20 تا)
        recent_messages = list(self.message_buffer)[-20:]
        if recent_messages:
            try:
                bulk_welcome = {
                    'type': 'welcome_bulk',
                    'level': 'info',
                    'message': f'Welcome! Sending {len(recent_messages)} recent messages',
                    'data': {'messages': recent_messages},
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(bulk_welcome))
            except:
                pass
    
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
        
        if not self.active_connections:
            return
        
        disconnected_connections = []
        message_json = json.dumps(message)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
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
        # فقط endpointهای کند یا مشکل‌دار را نشان بده
        response_time = endpoint_data.get('response_time', 0)
        status_code = endpoint_data.get('status_code', 200)
        
        if response_time > 2.0 or status_code >= 400:
            message = {
                'type': 'endpoint_call',
                'level': 'warning' if response_time > 2.0 else 'error',
                'message': f"🔗 {endpoint_data['method']} {endpoint_data['endpoint']} - {response_time:.3f}s (Status: {status_code})",
                'data': {
                    'endpoint': endpoint_data['endpoint'],
                    'method': endpoint_data['method'],
                    'response_time': response_time,
                    'status_code': status_code,
                    'cache_used': endpoint_data.get('cache_used', False)
                }
            }
            self.message_queue.append(message)
    
    def log_system_metrics(self, metrics_data: Dict[str, Any]):
        """ثبت لاگ متریک‌های سیستم برای کنسول"""
        # فقط اگر متریک‌ها critical باشند
        cpu_percent = metrics_data.get('cpu_percent', 0)
        memory_percent = metrics_data.get('memory_percent', 0)
        
        if cpu_percent > 80 or memory_percent > 85:
            message = {
                'type': 'system_metrics',
                'level': 'warning', 
                'message': f"📊 System Alert - CPU: {cpu_percent}% | Memory: {memory_percent}%",
                'data': metrics_data
            }
            self.message_queue.append(message)
    
    def log_security_alert(self, alert_data: Dict[str, Any]):
        """ثبت هشدار امنیتی برای کنسول"""
        message = {
            'type': 'security_alert',
            'level': 'warning',
            'message': f"🚨 SECURITY: {alert_data.get('message', 'Suspicious activity detected')}",
            'data': alert_data
        }
        self.message_queue.append(message)
    
    def log_performance_alert(self, alert_data: Dict[str, Any]):
        """ثبت هشدار عملکرد برای کنسول"""
        message = {
            'type': 'performance_alert', 
            'level': 'warning',
            'message': f"⚡ PERFORMANCE: {alert_data.get('message', 'Performance issue detected')}",
            'data': alert_data
        }
        self.message_queue.append(message)
    
    def log_error(self, error_data: Dict[str, Any]):
        """ثبت خطا برای کنسول"""
        message = {
            'type': 'error',
            'level': 'error',
            'message': f"🔴 ERROR: {error_data.get('message', 'An error occurred')}",
            'data': error_data
        }
        self.message_queue.append(message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """دریافت آمار connectionهای کنسول"""
        return {
            'active_connections': len(self.active_connections),
            'total_messages_sent': sum(
                stats['message_count'] for stats in self.connection_stats.values()
            ),
            'connection_details': self.connection_stats,
            'message_buffer_size': len(self.message_buffer),
            'message_queue_size': len(self.message_queue),
            'bulk_mode': True,
            'timestamp': datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال
console_stream = ConsoleStreamManager()

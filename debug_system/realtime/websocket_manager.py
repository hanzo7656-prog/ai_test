import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class WebSocketMessageType(Enum):
    DEBUG_LOG = "debug_log"
    SYSTEM_METRICS = "system_metrics" 
    ENDPOINT_STATS = "endpoint_stats"
    ALERT = "alert"
    COMMAND = "command"
    HEALTH_CHECK = "health_check"

class WebSocketManager:
    def __init__(self):
        self.connection_pool = {}
        self.message_handlers = {}
        self._initialize_handlers()
        
        # بهینه‌سازی: connection groups
        self.connection_groups = defaultdict(list)
        
        # اتصال به central_monitor
        self._connect_to_central_monitor()
        
        logger.info("✅ WebSocket Manager Initialized - Optimized")
    
    def _connect_to_central_monitor(self):
        """اتصال به central_monitor برای broadcast messages"""
        try:
            from ..core.system_monitor import central_monitor
            
            if central_monitor:
                # عضویت برای دریافت broadcast messages
                central_monitor.subscribe("websocket_manager", self._on_broadcast_message)
                logger.info("✅ WebSocketManager subscribed to central_monitor")
            else:
                logger.warning("⚠️ Central monitor not available - WebSocket will work independently")
                
        except ImportError:
            logger.warning("⚠️ Could not import central_monitor - WebSocket will work independently")
        except Exception as e:
            logger.error(f"❌ Error connecting to central_monitor: {e}")
    
    def _on_broadcast_message(self, message_data: Dict[str, Any]):
        """دریافت broadcast message از central_monitor"""
        try:
            message_type = message_data.get('type', 'broadcast')
            
            # ارسال به group مناسب
            if message_type == 'system_metrics':
                await self.broadcast_message(message_data, client_type='dashboard')
            elif message_type == 'alert':
                await self.broadcast_message(message_data)
            elif message_type == 'debug_log':
                await self.broadcast_message(message_data, client_type='debug_console')
                
        except Exception as e:
            logger.error(f"❌ Error processing broadcast message: {e}")
    
    def _initialize_handlers(self):
        """مقداردهی اولیه هندلرهای پیام"""
        self.message_handlers = {
            WebSocketMessageType.HEALTH_CHECK.value: self._handle_health_check,
            WebSocketMessageType.COMMAND.value: self._handle_command
        }
    
    async def connect(self, websocket: WebSocket, client_type: str = "unknown"):
        """اتصال کلاینت جدید"""
        await websocket.accept()
        client_id = str(uuid.uuid4())
        
        self.connection_pool[client_id] = {
            'websocket': websocket,
            'client_type': client_type,
            'connected_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        
        # اضافه به group
        self.connection_groups[client_type].append(client_id)
        
        logger.info(f"🔌 WebSocket client connected: {client_id} ({client_type})")
        
        # ارسال پیام خوش‌آمدگویی
        await self.send_message(client_id, {
            'type': 'connection_established',
            'message': f'Connected as {client_type}',
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
        
        return client_id
    
    def disconnect(self, client_id: str):
        """قطع ارتباط کلاینت"""
        if client_id in self.connection_pool:
            client_info = self.connection_pool.pop(client_id)
            client_type = client_info['client_type']
            
            # حذف از group
            if client_id in self.connection_groups[client_type]:
                self.connection_groups[client_type].remove(client_id)
            
            logger.info(f"🔌 WebSocket client disconnected: {client_id} ({client_type})")
    
    async def handle_messages(self, client_id: str):
        """مدیریت پیام‌های دریافتی از کلاینت"""
        if client_id not in self.connection_pool:
            return
        
        websocket = self.connection_pool[client_id]['websocket']
        
        try:
            while True:
                # دریافت پیام
                message_data = await websocket.receive_text()
                self.connection_pool[client_id]['last_activity'] = datetime.now().isoformat()
                
                try:
                    message = json.loads(message_data)
                    await self._process_message(client_id, message)
                    
                except json.JSONDecodeError:
                    await self.send_error(client_id, "Invalid JSON format")
                except Exception as e:
                    await self.send_error(client_id, f"Message processing error: {str(e)}")
                    
        except WebSocketDisconnect:
            self.disconnect(client_id)
        except Exception as e:
            logger.error(f"❌ WebSocket error for {client_id}: {e}")
            self.disconnect(client_id)
    
    async def _process_message(self, client_id: str, message: Dict[str, Any]):
        """پردازش پیام دریافتی"""
        message_type = message.get('type')
        
        if not message_type:
            await self.send_error(client_id, "Message type is required")
            return
        
        # پیدا کردن هندلر مناسب
        handler = self.message_handlers.get(message_type)
        if handler:
            await handler(client_id, message)
        else:
            await self.send_error(client_id, f"Unknown message type: {message_type}")
    
    async def _handle_health_check(self, client_id: str, message: Dict[str, Any]):
        """هندلر پیام سلامت"""
        await self.send_message(client_id, {
            'type': 'health_response',
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'server_time': datetime.now().isoformat()
        })
    
    async def _handle_command(self, client_id: str, message: Dict[str, Any]):
        """هندلر پیام دستور"""
        command = message.get('command')
        data = message.get('data', {})
        
        response = {
            'type': 'command_response',
            'command': command,
            'timestamp': datetime.now().isoformat()
        }
        
        if command == 'get_stats':
            response['data'] = self.get_connection_stats()
        elif command == 'ping':
            response['data'] = {'message': 'pong'}
        else:
            response['error'] = f"Unknown command: {command}"
        
        await self.send_message(client_id, response)
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """ارسال پیام به کلاینت خاص"""
        if client_id not in self.connection_pool:
            return
        
        try:
            websocket = self.connection_pool[client_id]['websocket']
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error sending message to {client_id}: {e}")
            self.disconnect(client_id)
    
    async def broadcast_message(self, message: Dict[str, Any], client_type: str = None):
        """ارسال پیام به تمام کلاینت‌ها یا نوع خاصی از کلاینت‌ها"""
        message['timestamp'] = datetime.now().isoformat()
        
        # تعیین target clients
        target_clients = []
        
        if client_type:
            # فقط clients از type خاص
            target_clients = self.connection_groups.get(client_type, [])
        else:
            # تمام clients
            target_clients = list(self.connection_pool.keys())
        
        if not target_clients:
            return
        
        # ارسال به groups
        await self._send_to_clients(target_clients, message)
    
    async def _send_to_clients(self, client_ids: List[str], message: Dict[str, Any]):
        """ارسال پیام به لیستی از clients"""
        message_json = json.dumps(message)
        disconnected_clients = []
        
        for client_id in client_ids:
            if client_id in self.connection_pool:
                try:
                    websocket = self.connection_pool[client_id]['websocket']
                    await websocket.send_text(message_json)
                except Exception as e:
                    logger.error(f"❌ Broadcast error for {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # حذف clients قطع شده
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_debug_log(self, log_data: Dict[str, Any]):
        """ارسال لاگ دیباگ به تمام کلاینت‌ها"""
        await self.broadcast_message({
            'type': WebSocketMessageType.DEBUG_LOG.value,
            'data': log_data
        }, client_type='debug_console')
    
    async def broadcast_system_metrics(self, metrics_data: Dict[str, Any]):
        """ارسال متریک‌های سیستم به تمام کلاینت‌ها"""
        await self.broadcast_message({
            'type': WebSocketMessageType.SYSTEM_METRICS.value,
            'data': metrics_data
        }, client_type='dashboard')
    
    async def broadcast_endpoint_stats(self, stats_data: Dict[str, Any]):
        """ارسال آمار اندپوینت به تمام کلاینت‌ها"""
        await self.broadcast_message({
            'type': WebSocketMessageType.ENDPOINT_STATS.value,
            'data': stats_data
        }, client_type='monitor')
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """ارسال هشدار به تمام کلاینت‌ها"""
        await self.broadcast_message({
            'type': WebSocketMessageType.ALERT.value,
            'data': alert_data
        })
    
    async def send_error(self, client_id: str, error_message: str):
        """ارسال خطا به کلاینت"""
        await self.send_message(client_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """دریافت آمار connectionها"""
        client_types = defaultdict(int)
        for client_info in self.connection_pool.values():
            client_types[client_info['client_type']] += 1
        
        return {
            'total_connections': len(self.connection_pool),
            'connections_by_type': dict(client_types),
            'connection_groups': {k: len(v) for k, v in self.connection_groups.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_inactive_connections(self, max_inactive_minutes: int = 30):
        """پاک‌سازی connectionهای غیرفعال"""
        cutoff_time = datetime.now() - timedelta(minutes=max_inactive_minutes)
        inactive_clients = []
        
        for client_id, client_info in self.connection_pool.items():
            last_activity = datetime.fromisoformat(client_info['last_activity'])
            if last_activity < cutoff_time:
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            logger.info(f"🧹 Cleaning up inactive connection: {client_id}")
            self.disconnect(client_id)
        
        return len(inactive_clients)

# ایجاد نمونه گلوبال
websocket_manager = WebSocketManager()

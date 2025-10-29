# lbank_websocket.py - نسخه کامل با روت FastAPI و داده‌های خام

import websocket
import json
import threading
import time
from typing import Dict, Any, Callable, List
import logging
from fastapi import APIRouter, HTTPException
import pandas as pd
from datetime import datetime

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ایجاد روت
router = APIRouter(prefix="/websocket", tags=["WebSocket"])

class LBankWebSocketManager:
    def __init__(self, gist_manager=None):
        # آدرس درست بر اساس کد Node.js
        self.ws_url = "wss://www.lbkex.net/ws/V2/"
        self.ws = None
        self.connected = False
        self.realtime_data = {}
        self.subscribed_pairs = set()
        self.callbacks: List[Callable] = []
        self.gist_manager = gist_manager
        
        # تنظیمات
        self.ping_interval = 30
        self.ping_timeout = 10
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # آمار عملکرد
        self.performance_metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'last_message_time': None,
            'connection_uptime': 0
        }
        
        # شروع اتصال
        self.connect()
        
        logger.info("🚀 LBank WebSocket Manager Initialized - Raw Data Mode")

    def connect(self):
        """اتصال WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            def run_ws():
                self.ws.run_forever(
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout
                )
            
            thread = threading.Thread(target=run_ws)
            thread.daemon = True
            thread.start()
            
            logger.info("🔄 Starting LBank WebSocket Connection...")
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {e}")
            self._schedule_reconnect()

    def _on_open(self, ws):
        """اتصال باز شد"""
        logger.info("✅ LBank WebSocket Connected Successfully")
        self.connected = True
        self.reconnect_attempts = 0
        self.performance_metrics['connection_uptime'] = time.time()
        self.subscribe_to_all_pairs()

    def _on_message(self, ws, message):
        """دریافت پیام - پردازش داده‌های خام"""
        try:
            self.performance_metrics['messages_received'] += 1
            
            # پردازش داده خام
            raw_data = json.loads(message)
            
            if raw_data.get('type') == 'tick' and raw_data.get('tick'):
                symbol = raw_data.get('pair', '').upper()
                tick_data = raw_data.get('tick', {})
                
                # استخراج داده‌های خام
                raw_tick_data = {
                    'symbol': symbol,
                    'price': float(tick_data.get('latest', 0)),
                    'high_24h': float(tick_data.get('high', 0)),
                    'low_24h': float(tick_data.get('low', 0)),
                    'volume': float(tick_data.get('vol', 0)),
                    'change': float(tick_data.get('change', 0)),
                    'timestamp': raw_data.get('TS', ''),
                    'raw_message': raw_data,  # ذخیره کل پیام خام
                    'processed_at': datetime.now().isoformat()
                }
                
                # ذخیره در Gist اگر موجود باشد
                if self.gist_manager:
                    try:
                        self.gist_manager.add_price(symbol, raw_tick_data['price'])
                    except Exception as e:
                        logger.debug(f"⚠️ Gist save skipped: {e}")
                
                # ذخیره داده خام
                self.realtime_data[symbol] = raw_tick_data
                self.performance_metrics['messages_processed'] += 1
                self.performance_metrics['last_message_time'] = datetime.now().isoformat()
                
                # فراخوانی callback ها با داده خام
                for callback in self.callbacks:
                    try:
                        callback(symbol, raw_tick_data)
                    except Exception as e:
                        logger.error(f"❌ Callback error: {e}")
                
                logger.debug(f"📊 {symbol}: ${raw_tick_data['price']} (Raw Data)")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")

    def _on_error(self, ws, error):
        """مدیریت خطاها"""
        logger.error(f"❌ WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """اتصال بسته شد"""
        logger.info(f"🔌 WebSocket disconnected - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        self._schedule_reconnect()

    def _schedule_reconnect(self):
        """برنامه‌ریزی برای اتصال مجدد"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(5 * self.reconnect_attempts, 30)  # افزایش تدریجی تاخیر
            logger.info(f"🔄 Attempting WebSocket reconnection in {delay} seconds... (Attempt {self.reconnect_attempts})")
            time.sleep(delay)
            self.connect()
        else:
            logger.error("❌ Maximum reconnection attempts reached")

    def subscribe_to_all_pairs(self):
        """اشتراک در تمام جفت ارزها"""
        if self.connected and self.ws:
            from config import MAJOR_TRADING_PAIRS
            pairs = MAJOR_TRADING_PAIRS
            
            logger.info(f"📡 Subscribing to {len(pairs)} trading pairs")
            batch_size = 10
            
            for i in range(0, len(pairs), batch_size):
                time.sleep(0.1)  # تاخیر بین batchها
                batch = pairs[i:i + batch_size]
                self.subscribe_batch(batch)

    def subscribe_batch(self, pairs):
        """اشتراک دسته‌ای جفت ارزها"""
        if not self.ws or not self.connected:
            return
            
        successful_subscriptions = 0
        
        for pair in pairs:
            subscription_msg = {
                "action": "subscribe",
                "subscribe": "tick",
                "pair": pair
            }
            
            try:
                self.ws.send(json.dumps(subscription_msg))
                self.subscribed_pairs.add(pair)
                successful_subscriptions += 1
            except Exception as e:
                logger.error(f"❌ Subscription error for {pair}: {e}")
        
        logger.info(f"✅ Subscribed to {successful_subscriptions} pairs")

    # ========================= سازگاری با کد قدیمی =========================

    def is_connected(self):
        """بررسی وضعیت اتصال - سازگاری با کد قدیمی"""
        return self.connected

    def start(self):
        """متد start - سازگاری با کد قدیمی"""
        logger.info("🔄 WebSocket already auto-started in constructor")
        
        # اگر قطع شده، reconnect کن
        if not self.connected:
            logger.info("🔄 WebSocket disconnected, attempting reconnect...")
            self.connect()
            
        return self.connected

    @property
    def connected(self):
        """Property برای وضعیت اتصال"""
        return self._connected

    @connected.setter
    def connected(self, value):
        """Setter برای وضعیت اتصال"""
        self._connected = value

    # ========================= متدهای اصلی =========================

    def subscribe_to_major_pairs(self):
        """اشتراک در جفت ارزهای اصلی - برای سازگاری با کد قدیمی"""
        self.subscribe_to_all_pairs()

    def subscribe_pair(self, pair: str):
        """اشتراک در یک جفت ارز خاص"""
        if not self.connected or not self.ws:
            return

        subscription_msg = {
            "action": "subscribe",
            "subscribe": "tick",
            "pair": pair
        }

        try:
            self.ws.send(json.dumps(subscription_msg))
            self.subscribed_pairs.add(pair)
            logger.info(f"✅ Subscribed to {pair}")
        except Exception as e:
            logger.error(f"❌ Subscription error for {pair}: {e}")

    def get_realtime_data(self, symbol: str = None) -> Dict[str, Any]:
        """دریافت داده لحظه‌ای خام"""
        if symbol:
            return self.realtime_data.get(symbol.upper(), {})
        return self.realtime_data

    def get_connection_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال"""
        current_uptime = time.time() - self.performance_metrics['connection_uptime'] if self.performance_metrics['connection_uptime'] else 0
        
        return {
            'connected': self.connected,
            'active_pairs': list(self.realtime_data.keys()),
            'total_subscribed': len(self.subscribed_pairs),
            'subscribed_pairs': list(self.subscribed_pairs),
            'data_count': len(self.realtime_data),
            'performance_metrics': {
                'messages_received': self.performance_metrics['messages_received'],
                'messages_processed': self.performance_metrics['messages_processed'],
                'last_message_time': self.performance_metrics['last_message_time'],
                'connection_uptime_seconds': round(current_uptime, 2)
            },
            'reconnect_attempts': self.reconnect_attempts
        }

    def add_callback(self, callback: Callable):
        """اضافه کردن callback برای داده‌های جدید"""
        self.callbacks.append(callback)
        logger.info(f"✅ Callback added - Total: {len(self.callbacks)}")

    def disconnect(self):
        """قطع اتصال WebSocket"""
        if self.ws:
            self.ws.close()
            self.connected = False
            logger.info("🔌 WebSocket disconnected manually")

    def get_active_pairs(self):
        """دریافت لیست جفت ارزهای فعال"""
        return list(self.realtime_data.keys())

    def get_raw_data_quality(self) -> Dict[str, Any]:
        """بررسی کیفیت داده‌های خام"""
        total_pairs = len(self.realtime_data)
        if total_pairs == 0:
            return {'status': 'no_data', 'quality_score': 0}
        
        # محاسبه کیفیت بر اساس تازگی داده‌ها
        current_time = time.time()
        freshness_scores = []
        
        for symbol, data in self.realtime_data.items():
            if 'processed_at' in data:
                processed_time = datetime.fromisoformat(data['processed_at']).timestamp()
                freshness = max(0, 1 - (current_time - processed_time) / 60)  # 1 دقیقه
                freshness_scores.append(freshness)
        
        avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0
        quality_score = (avg_freshness * 0.7) + (total_pairs / 50 * 0.3)  # ترکیب تازگی و تعداد
        
        return {
            'status': 'healthy' if quality_score > 0.5 else 'degraded',
            'quality_score': round(quality_score, 3),
            'total_pairs': total_pairs,
            'avg_freshness': round(avg_freshness, 3),
            'data_points_per_second': self.performance_metrics['messages_processed'] / max(1, current_time - self.performance_metrics['connection_uptime'])
        }

# ایجاد WebSocket Manager
ws_manager = LBankWebSocketManager()

# تابع برای دسترسی از فایل‌های دیگر
def get_websocket_manager():
    return ws_manager

# ========================= روت‌های WebSocket =========================

@router.get("/status")
async def websocket_status():
    """وضعیت WebSocket"""
    return ws_manager.get_connection_status()

@router.get("/data")
async def get_websocket_data(symbol: str = None):
    """دریافت داده‌های لحظه‌ای خام"""
    data = ws_manager.get_realtime_data(symbol)
    
    if symbol and not data:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
    return {
        'data': data,
        'raw_data_mode': True,
        'timestamp': datetime.now().isoformat()
    }

@router.get("/pairs/active")
async def get_active_pairs():
    """لیست جفت ارزهای فعال"""
    return {
        "active_pairs": ws_manager.get_active_pairs(),
        "total": len(ws_manager.get_active_pairs()),
        "raw_data_quality": ws_manager.get_raw_data_quality()
    }

@router.post("/pairs/subscribe/{pair}")
async def subscribe_pair(pair: str):
    """اشتراک در جفت ارز جدید"""
    try:
        ws_manager.subscribe_pair(pair.upper())
        return {"status": "success", "message": f"Subscribed to {pair}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pairs/subscribed")
async def get_subscribed_pairs():
    """لیست جفت ارزهای مشترک شده"""
    return {
        "subscribed_pairs": list(ws_manager.subscribed_pairs),
        "total": len(ws_manager.subscribed_pairs)
    }

@router.get("/health")
async def websocket_health():
    """سلامت WebSocket"""
    status = ws_manager.get_connection_status()
    
    return {
        "status": "connected" if status['connected'] else "disconnected",
        "active_connections": len(status['active_pairs']),
        "performance": status['performance_metrics'],
        "raw_data_quality": ws_manager.get_raw_data_quality(),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/performance")
async def websocket_performance():
    """آمار عملکرد WebSocket"""
    status = ws_manager.get_connection_status()
    
    return {
        "performance_metrics": status['performance_metrics'],
        "connection_status": "stable" if status['reconnect_attempts'] == 0 else "unstable",
        "data_throughput": f"{status['performance_metrics']['messages_processed']} messages",
        "raw_data_available": len(status['active_pairs']) > 0
    }

# تست مستقل
if __name__ == "__main__":
    def test_callback(symbol, data):
        print(f"📨 {symbol}: ${data['price']}")

    ws_manager = LBankWebSocketManager()
    ws_manager.add_callback(test_callback)

    # تست متدهای سازگاری
    print("🔌 Connected:", ws_manager.is_connected())
    print("🔌 connected property:", ws_manager.connected)

    # منتظر داده‌ها
    time.sleep(10)

    print("📊 Real-time data:", ws_manager.get_realtime_data())
    print("📈 Connection status:", ws_manager.get_connection_status())
    print("🎯 Raw data quality:", ws_manager.get_raw_data_quality())

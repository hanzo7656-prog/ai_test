# lbank_websocket.py - نسخه اصلاح شده با URL درست
import websocket
import json
import threading
import time
from typing import Dict, Any, Callable, List
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LBankWebSocketManager:
    def __init__(self, gist_manager=None):
        # آدرس درست بر اساس کد Node.js شما
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
        
        # شروع اتصال
        self.connect()

    def connect(self):
        """اتصال به WebSocket"""
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
            logger.info("🚀 Starting LBank WebSocket...")
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {e}")
            # تلاش مجدد پس از 10 ثانیه
            time.sleep(10)
            self.connect()

    def _on_open(self, ws):
        """اتصال باز شد"""
        logger.info("✅ LBank WebSocket Connected Successfully")
        self.connected = True
        self.subscribe_to_all_pairs()

    def _on_message(self, ws, message):
        """دریافت پیام"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'tick' and data.get('tick'):
                symbol = data.get('pair', '')
                tick_data = data.get('tick', {})
                current_price = float(tick_data.get('latest', 0))

                # ذخیره در Gist (اگر موجود باشد)
                if self.gist_manager:
                    try:
                        self.gist_manager.add_price(symbol, current_price)
                    except Exception as e:
                        logger.debug(f"⚠️ Gist save skipped: {e}")

                # ذخیره در داده real-time
                self.realtime_data[symbol] = {
                    'symbol': symbol,
                    'price': current_price,
                    'high_24h': float(tick_data.get('high', 0)),
                    'low_24h': float(tick_data.get('low', 0)),
                    'volume': float(tick_data.get('vol', 0)),
                    'change': float(tick_data.get('change', 0)),
                    'timestamp': data.get('TS', ''),
                    'last_updated': time.time()
                }

                # فراخوانی callbackها
                for callback in self.callbacks:
                    try:
                        callback(symbol, self.realtime_data[symbol])
                    except Exception as e:
                        logger.error(f"❌ Callback error: {e}")

                logger.debug(f"📊 {symbol}: ${current_price}")
                
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
        logger.info(f"🔴 WebSocket disconnected - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        
        # تلاش برای اتصال مجدد پس از 5 ثانیه
        logger.info("🔄 Attempting WebSocket reconnection in 5 seconds...")
        time.sleep(5)
        self.connect()

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
        
        for pair in pairs:
            subscription_msg = {
                "action": "subscribe",
                "subscribe": "tick",
                "pair": pair
            }
            try:
                self.ws.send(json.dumps(subscription_msg))
                self.subscribed_pairs.add(pair)
            except Exception as e:
                logger.error(f"❌ Subscription error for {pair}: {e}")
        
        logger.info(f"✅ Subscribed to {len(pairs)} pairs")

    def subscribe_to_major_pairs(self):
        """اشتراک در جفت ارزهای اصلی (برای سازگاری با کد قدیمی)"""
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
        """دریافت داده لحظه‌ای"""
        if symbol:
            return self.realtime_data.get(symbol.upper(), {})
        return self.realtime_data

    def get_connection_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال"""
        return {
            'connected': self.connected,
            'active_pairs': list(self.realtime_data.keys()),
            'total_subscribed': len(self.subscribed_pairs),
            'subscribed_pairs': list(self.subscribed_pairs),
            'data_count': len(self.realtime_data)
        }

    def add_callback(self, callback: Callable):
        """اضافه کردن callback برای داده‌های جدید"""
        self.callbacks.append(callback)

    def test_gist_connection(self):
        """تست اتصال Gist"""
        if self.gist_manager:
            try:
                status = self.gist_manager.get_status()
                logger.info(f"🧪 Gist Manager Test: {status}")
                return status
            except Exception as e:
                return {'error': f'Gist Manager error: {e}'}
        return {'error': 'Gist Manager not available'}

    # متدهای اضافی برای مدیریت
    def disconnect(self):
        """قطع اتصال WebSocket"""
        if self.ws:
            self.ws.close()
        self.connected = False

    def is_connected(self):
        """بررسی وضعیت اتصال"""
        return self.connected

    def get_active_pairs(self):
        """دریافت لیست جفت ارزهای فعال"""
        return list(self.realtime_data.keys())

# تست مستقل
if __name__ == "__main__":
    def test_callback(symbol, data):
        print(f"📊 {symbol}: ${data['price']}")
    
    ws_manager = LBankWebSocketManager()
    ws_manager.add_callback(test_callback)
    
    # منتظر داده‌ها
    time.sleep(10)
    
    print("📈 Real-time data:", ws_manager.get_realtime_data())
    print("🔗 Connection status:", ws_manager.get_connection_status())

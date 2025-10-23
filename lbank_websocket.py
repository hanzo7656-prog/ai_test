# lbank_websocket.py - نسخه اصلاح شده
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
    def __init__(self):
        # آدرس جدید بر اساس مستندات
        self.ws_url = "wss://www.lbank.com/ws/V2/"
        self.ws = None
        self.connected = False
        self.realtime_data = {}
        self.callbacks: List[Callable] = []
        
        # تنظیمات جدید
        self.ping_interval = 30  # ارسال ping هر 30 ثانیه
        self.ping_timeout = 10   # timeout برای ping

    def start(self):
        """اتصال WebSocket با تنظیمات جدید"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )
            
            def run_ws():
                # اضافه کردن headerها برای جلوگیری از redirect
                self.ws.run_forever(
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    origin="https://www.lbank.com"
                )
            
            thread = threading.Thread(target=run_ws)
            thread.daemon = True
            thread.start()
            logger.info("🚀 Starting LBank WebSocket v2...")
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {e}")

    def _on_open(self, ws):
        """اتصال باز شد"""
        logger.info("✅ LBank WebSocket Connected Successfully")
        self.connected = True
        self._subscribe_major_pairs()

    def _on_message(self, ws, message):
        """دریافت پیام"""
        try:
            data = json.loads(message)
            logger.debug(f"📨 Received message: {data}")
            
            # پردازش انواع مختلف پیام
            if 'action' in data and data['action'] == 'ping':
                self._handle_ping(data)
            elif 'ping' in data:
                self._handle_ping_response(data)
            elif 'tick' in data:
                self._handle_tick_data(data)
            elif 'depth' in data:
                self._handle_depth_data(data)
            elif 'trade' in data:
                self._handle_trade_data(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")

    def _handle_ping(self, data):
        """مدیریت ping از سرور"""
        try:
            pong_msg = {
                "action": "pong",
                "ping": data.get("ping", "")
            }
            self.ws.send(json.dumps(pong_msg))
            logger.debug("🏓 Sent pong response")
        except Exception as e:
            logger.error(f"❌ Error sending pong: {e}")

    def _handle_ping_response(self, data):
        """مدیریت پاسخ ping"""
        logger.debug("🏓 Received ping response")

    def _handle_tick_data(self, data):
        """پردازش داده‌های تیک"""
        try:
            symbol = data.get('pair', '').upper()
            tick_data = data.get('tick', {})
            
            self.realtime_data[symbol] = {
                'symbol': symbol,
                'price': float(tick_data.get('latest', 0)),
                'high_24h': float(tick_data.get('high', 0)),
                'low_24h': float(tick_data.get('low', 0)),
                'volume': float(tick_data.get('vol', 0)),
                'change': float(tick_data.get('change', 0)),
                'timestamp': data.get('TS', ''),
                'last_updated': time.time(),
                'source': 'lbank_websocket'
            }
            
            # فراخوانی callbackها
            for callback in self.callbacks:
                try:
                    callback(symbol, self.realtime_data[symbol])
                except Exception as e:
                    logger.error(f"❌ Callback error: {e}")
                    
            logger.debug(f"📊 Updated {symbol}: ${self.realtime_data[symbol]['price']}")
            
        except Exception as e:
            logger.error(f"❌ Error processing tick data: {e}")

    def _handle_depth_data(self, data):
        """پردازش داده‌های عمق بازار"""
        # برای future use
        pass

    def _handle_trade_data(self, data):
        """پردازش داده‌های معاملات"""
        # برای future use
        pass

    def _on_error(self, ws, error):
        """مدیریت خطاها"""
        logger.error(f"❌ WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """اتصال بسته شد"""
        logger.info(f"🔴 WebSocket disconnected - Code: {close_status_code}, Message: {close_msg}")
        self.connected = False
        
        # تلاش برای اتصال مجدد پس از 10 ثانیه
        logger.info("🔄 Reconnecting in 10 seconds...")
        time.sleep(10)
        self.start()

    def _on_ping(self, ws, data):
        """مدیریت ping"""
        logger.debug("🏓 WebSocket ping")

    def _on_pong(self, ws, data):
        """مدیریت pong"""
        logger.debug("🏓 WebSocket pong")

    def _subscribe_major_pairs(self):
        """اشتراک در جفت ارزهای اصلی"""
        major_pairs = [
            "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt",
            "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
            "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
        ]

        for pair in major_pairs:
            self._subscribe_pair(pair)

    def _subscribe_pair(self, pair: str):
        """اشتراک در یک جفت ارز"""
        if not self.connected or not self.ws:
            return

        # ساختار پیام اشتراک بر اساس مستندات
        subscription_msg = {
            "action": "subscribe",
            "subscribe": "tick",
            "pair": pair
        }

        try:
            self.ws.send(json.dumps(subscription_msg))
            logger.info(f"✅ Subscribed to {pair}")
        except Exception as e:
            logger.error(f"❌ Subscription error for {pair}: {e}")

    def get_realtime_data(self, symbol: str = None) -> Dict[str, Any]:
        """دریافت داده لحظه‌ای"""
        if symbol:
            return self.realtime_data.get(symbol.upper(), {})
        return self.realtime_data

    def add_callback(self, callback: Callable):
        """اضافه کردن callback برای داده‌های جدید"""
        self.callbacks.append(callback)

    def get_connection_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال"""
        return {
            'connected': self.connected,
            'active_pairs': list(self.realtime_data.keys()),
            'data_count': len(self.realtime_data),
            'last_update': max([data.get('last_updated', 0) for data in self.realtime_data.values()]) if self.realtime_data else 0
        }

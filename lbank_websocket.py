# lbank_websocket.py
import websocket
import json
import threading
import time
from typing import Dict, Any, Callable, List

class LBankWebSocketManager:
    def __init__(self):
        self.ws_url = "wss://www.lbank.net/ws/V2/"
        self.ws = None
        self.connected = False
        self.realtime_data = {}
        self.callbacks: List[Callable] = []
        
    def start(self):
        """شروع اتصال WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            def run_ws():
                self.ws.run_forever()
            
            thread = threading.Thread(target=run_ws)
            thread.daemon = True
            thread.start()
            
            print("🔄 Starting LBank WebSocket...")
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            
    def _on_open(self, ws):
        """اتصال باز شد"""
        print("✅ LBank WebSocket Connected")
        self.connected = True
        self._subscribe_major_pairs()
        
    def _on_message(self, ws, message):
        """دریافت پیام"""
        try:
            data = json.loads(message)
            if data.get('type') == 'tick' and 'tick' in data:
                symbol = data.get('pair', '')
                tick_data = data['tick']
                
                self.realtime_data[symbol] = {
                    'symbol': symbol,
                    'price': float(tick_data.get('latest', 0)),
                    'high_24h': float(tick_data.get('high', 0)),
                    'low_24h': float(tick_data.get('low', 0)),
                    'volume': float(tick_data.get('vol', 0)),
                    'change': float(tick_data.get('change', 0)),
                    'timestamp': data.get('TS', ''),
                    'last_updated': time.time()
                }
                
                # اطلاع به callback ها
                for callback in self.callbacks:
                    try:
                        callback(symbol, self.realtime_data[symbol])
                    except Exception as e:
                        print(f"Callback error: {e}")
                        
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            
    def _on_error(self, ws, error):
        """خطا"""
        print(f"❌ WebSocket error: {error}")
        self.connected = False
        
    def _on_close(self, ws, close_status_code, close_msg):
        """اتصال بسته شد"""
        print(f"🔴 WebSocket disconnected")
        self.connected = False
        time.sleep(5)
        self.start()  # reconnect
        
    def _subscribe_major_pairs(self):
        """اشتراک در جفت ارزهای اصلی"""
        major_pairs = [
            "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", 
            "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt"
        ]
        
        for pair in major_pairs:
            self._subscribe_pair(pair)
            
    def _subscribe_pair(self, pair: str):
        """اشتراک در یک جفت ارز"""
        if not self.connected or not self.ws:
            return
            
        subscription_msg = {
            "action": "subscribe",
            "subscribe": "tick",
            "pair": pair
        }
        
        try:
            self.ws.send(json.dumps(subscription_msg))
            print(f"✅ Subscribed to {pair}")
        except Exception as e:
            print(f"❌ Subscription error for {pair}: {e}")
            
    def get_realtime_data(self, symbol: str = None) -> Dict[str, Any]:
        """دریافت داده لحظه‌ای"""
        if symbol:
            return self.realtime_data.get(symbol, {})
        return self.realtime_data
        
    def add_callback(self, callback: Callable):
        """اضافه کردن callback برای داده‌های جدید"""
        self.callbacks.append(callback)

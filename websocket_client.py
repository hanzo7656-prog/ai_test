# websocket_client.py
import websocket
import json
import threading
import time
from typing import Dict, Set, Any

class LBankWebSocketClient:
    def __init__(self, data_manager=None):
        self.ws_url = "wss://www.lbank.net/ws/V2/"
        self.ws = None
        self.connected = False
        self.realtime_data = {}
        self.subscribed_pairs = set()
        self.data_manager = data_manager
        self.connect()
    
    def connect(self):
        """اتصال به WebSocket LBank"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # اجرای WebSocket در thread جداگانه
            def run_ws():
                self.ws.run_forever()
            
            thread = threading.Thread(target=run_ws)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.schedule_reconnect()
    
    def on_open(self, ws):
        """هنگام باز شدن اتصال"""
        print("✅ WebSocket connected to LBank")
        self.connected = True
        self.subscribe_to_major_pairs()
    
    def on_message(self, ws, message):
        """پردازش پیام‌های دریافتی"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'tick' and 'tick' in data:
                symbol = data.get('pair', '')
                tick_data = data['tick']
                
                # ذخیره داده لحظه‌ای
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
                
                # نمایش لاگ
                if symbol in ['btc_usdt', 'eth_usdt']:
                    print(f"📊 {symbol}: ${self.realtime_data[symbol]['price']:.2f}")
                    
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
    
    def on_error(self, ws, error):
        """مدیریت خطاها"""
        print(f"❌ WebSocket error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """هنگام بسته شدن اتصال"""
        print(f"🔴 WebSocket disconnected - Code: {close_status_code}, Reason: {close_msg}")
        self.connected = False
        self.schedule_reconnect()
    
    def schedule_reconnect(self):
        """برنامه‌ریزی برای اتصال مجدد"""
        print("🔄 Attempting reconnection in 5 seconds...")
        time.sleep(5)
        self.connect()
    
    def subscribe_to_major_pairs(self):
        """اشتراک در جفت‌ارزهای اصلی"""
        major_pairs = [
            "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", 
            "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt",
            "ltc_usdt", "link_usdt", "matic_usdt", "avax_usdt"
        ]
        
        for pair in major_pairs:
            self.subscribe_to_pair(pair)
    
    def subscribe_to_pair(self, pair: str):
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
            print(f"✅ Subscribed to {pair}")
        except Exception as e:
            print(f"❌ Subscription error for {pair}: {e}")
    
    def get_realtime_data(self, symbol: str = None) -> Dict:
        """دریافت داده‌های لحظه‌ای"""
        if symbol:
            return self.realtime_data.get(symbol, {})
        return self.realtime_data
    
    def get_connection_status(self) -> Dict[str, Any]:
        """دریافت وضعیت اتصال"""
        return {
            'connected': self.connected,
            'subscribed_pairs': list(self.subscribed_pairs),
            'active_data_count': len(self.realtime_data),
            'major_pairs_data': {
                pair: data for pair, data in self.realtime_data.items()
                if pair in ['btc_usdt', 'eth_usdt', 'sol_usdt']
            }
        }

# نمونه استفاده
if __name__ == "__main__":
    # ایجاد کلاینت WebSocket
    ws_client = LBankWebSocketClient()
    
    # منتظر ماندن برای جمع‌آوری داده
    time.sleep(10)
    
    # نمایش وضعیت
    status = ws_client.get_connection_status()
    print("\n" + "="*50)
    print("WebSocket Status:")
    print(f"Connected: {status['connected']}")
    print(f"Subscribed pairs: {len(status['subscribed_pairs'])}")
    print(f"Active data: {status['active_data_count']}")
    
    # نمایش داده‌های اصلی
    print("\nMajor Pairs Data:")
    for pair, data in status['major_pairs_data'].items():
        print(f"  {pair}: ${data.get('price', 0):.2f}")

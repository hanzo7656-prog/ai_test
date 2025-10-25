# coinstats_manager_fixed.py
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from config import API_CONFIG, RAW_DATA_CONFIG, MAJOR_COINS, SUPPORTED_TIMEFRAMES

class CoinStatsManager:
    def __init__(self):
        self.base_url = API_CONFIG['base_url']
        self.api_key = API_CONFIG['api_key']  # ✅ مستقیم از کانفیگ
        self.headers = {"X-API-KEY": self.api_key}
        self.timeout = API_CONFIG['timeout']
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # تنظیمات داده خام
        self.save_raw = RAW_DATA_CONFIG['save_raw_responses']
        self.raw_folder = RAW_DATA_CONFIG['raw_data_folder']
        
        print(f"🔑 کلید API: {self.api_key[:20]}...")  # دیباگ
        
        # ایجاد پوشه داده خام
        if self.save_raw and not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)
    
    def _save_raw_response(self, endpoint: str, params: Dict, response_data: Any):
        """ذخیره پاسخ خام API"""
        if not self.save_raw:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{endpoint.replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(self.raw_folder, filename)
        
        raw_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'params': params,
            'data': response_data
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
            print(f"💾 داده خام ذخیره شد: {filename}")
        except Exception as e:
            print(f"❌ خطا در ذخیره داده خام: {e}")
    
    def _make_raw_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """درخواست API و بازگشت داده خام"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            print(f"📡 درخواست خام: {endpoint}")
            if params:
                print(f"📋 پارامترها: {params}")
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                raw_data = response.json()
                self._save_raw_response(endpoint, params, raw_data)
                print(f"✅ داده خام دریافت شد: {len(str(raw_data))} بایت")
                return raw_data
            else:
                print(f"❌ خطای HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"💥 خطا در درخواست: {e}")
            return None

    # ========================= 12 اندپوینت =========================
    
    def get_coins_list_raw(self, **filters) -> Optional[Dict]:
        """1. دریافت لیست کوین‌ها - داده خام"""
        return self._make_raw_request("coins", filters)
    
    def get_coins_charts_raw(self, coin_ids: str, period: str = "all") -> Optional[Dict]:
        """2. دریافت چارت چندکوینه - داده خام"""
        params = {"coinIds": coin_ids, "period": period}
        return self._make_raw_request("coins/charts", params)
    
    def get_coin_details_raw(self, coin_id: str, currency: str = "USD") -> Optional[Dict]:
        """3. دریافت جزئیات کوین خاص - داده خام"""
        params = {"currency": currency}
        return self._make_raw_request(f"coins/{coin_id}", params)
    
    def get_coin_charts_raw(self, coin_id: str, period: str = "all") -> Optional[Dict]:
        """4. دریافت چارت کوین خاص - داده خام"""
        params = {"period": period}
        return self._make_raw_request(f"coins/{coin_id}/charts", params)
    
    def get_coin_price_avg_raw(self, coin_id: str, timestamp: str) -> Optional[Dict]:
        """5. دریافت قیمت متوسط - داده خام"""
        params = {"coinId": coin_id, "timestamp": timestamp}
        return self._make_raw_request("coins/price/avg", params)
    
    def get_exchange_price_raw(self, exchange: str, from_coin: str, to_coin: str, timestamp: str) -> Optional[Dict]:
        """6. دریافت قیمت مبادله - داده خام"""
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        return self._make_raw_request("coins/price/exchange", params)
    
    def get_tickers_exchanges_raw(self) -> Optional[Dict]:
        """7. دریافت لیست صرافی‌ها - داده خام"""
        return self._make_raw_request("tickers/exchanges")
    
    def get_tickers_markets_raw(self) -> Optional[Dict]:
        """8. دریافت لیست بازارها - داده خام"""
        return self._make_raw_request("tickers/markets")
    
    def get_fiats_raw(self) -> Optional[Dict]:
        """9. دریافت لیست ارزهای فیات - داده خام"""
        return self._make_raw_request("fiats")
    
    def get_markets_raw(self) -> Optional[Dict]:
        """10. دریافت داده‌های بازار - داده خام"""
        return self._make_raw_request("markets")
    
    def get_currencies_raw(self) -> Optional[Dict]:
        """11. دریافت لیست ارزها - داده خام"""
        return self._make_raw_request("currencies")
    
    def get_news_sources_raw(self) -> Optional[Dict]:
        """12. دریافت منابع خبری - داده خام"""
        return self._make_raw_request("news/sources")

# ========================= تست =========================

if __name__ == "__main__":
    manager = CoinStatsManager()
    
    # تست اتصال
    print("🧪 تست اتصال...")
    test_data = manager.get_coins_list_raw(limit=1)
    
    if test_data and 'result' in test_data and test_data['result']:
        coin = test_data['result'][0]
        print(f"✅ متصل! نمونه: {coin.get('name')} - ${coin.get('price', 0):.2f}")
    else:
        print("❌ خطا در اتصال به API")

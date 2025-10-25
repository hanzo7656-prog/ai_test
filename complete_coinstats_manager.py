# raw_coinstats_manager.py
import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from config import API_CONFIG, RAW_DATA_CONFIG, ENDPOINTS_CONFIG

class RawCoinStatsManager:
    def __init__(self):
        self.base_url = API_CONFIG['base_url']
        self.api_key = API_CONFIG['api_key']
        self.headers = {"X-API-KEY": self.api_key}
        self.timeout = API_CONFIG['timeout']
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # تنظیمات داده خام
        self.save_raw = RAW_DATA_CONFIG['save_raw_responses']
        self.raw_folder = RAW_DATA_CONFIG['raw_data_folder']
        
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

    # ========================= اندپوینت‌های اصلی (6 تای اول) =========================
    
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

    # ========================= اندپوینت‌های جدید (6 تای دوم) =========================
    
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

    # ========================= متدهای کمکی =========================
    
    def get_all_raw_data(self) -> Dict[str, Any]:
        """دریافت تمام داده‌های خام از 12 اندپوینت"""
        print("🚀 شروع دریافت تمام داده‌های خام از 12 اندپوینت...")
        
        all_data = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'coinstats_raw_api',
            'endpoints_called': []
        }
        
        # تست اندپوینت‌های اصلی
        endpoints_to_test = [
            ('coins_list', lambda: self.get_coins_list_raw(limit=5)),
            ('coins_charts', lambda: self.get_coins_charts_raw("bitcoin,ethereum", "1d")),
            ('bitcoin_details', lambda: self.get_coin_details_raw("bitcoin")),
            ('bitcoin_charts', lambda: self.get_coin_charts_raw("bitcoin", "7d")),
            ('price_avg', lambda: self.get_coin_price_avg_raw("bitcoin", "1636315200")),
            ('exchange_price', lambda: self.get_exchange_price_raw("Binance", "BTC", "ETH", "1636315200")),
            ('tickers_exchanges', self.get_tickers_exchanges_raw),
            ('tickers_markets', self.get_tickers_markets_raw),
            ('fiats', self.get_fiats_raw),
            ('markets', self.get_markets_raw),
            ('currencies', self.get_currencies_raw),
            ('news_sources', self.get_news_sources_raw),
        ]
        
        for name, endpoint_func in endpoints_to_test:
            print(f"\n🔍 تست {name}...")
            data = endpoint_func()
            all_data[name] = data
            all_data['endpoints_called'].append(name)
            
            if data:
                print(f"✅ {name}: موفق")
            else:
                print(f"❌ {name}: شکست")
        
        print(f"\n📊 جمع‌بندی: {len([x for x in all_data['endpoints_called'] if all_data[x]])}/12 اندپوینت موفق")
        return all_data
    
    def test_connection(self):
        """تست اتصال سریع"""
        print("🧪 تست اتصال API...")
        test_data = self.get_coins_list_raw(limit=1)
        
        if test_data and 'result' in test_data and test_data['result']:
            coin = test_data['result'][0]
            return f"✅ متصل! نمونه: {coin.get('name')} - ${coin.get('price', 0):.2f}"
        else:
            return "❌ خطا در اتصال به API"

# ========================= تست =========================

if __name__ == "__main__":
    manager = RawCoinStatsManager()
    
    print("🔑 کلید API:", manager.api_key[:20] + "..." if manager.api_key else "None")
    
    # تست اتصال
    connection_result = manager.test_connection()
    print(connection_result)
    
    if "✅" in connection_result:
        # دریافت تمام داده‌های خام
        print("\n" + "🎯" * 40)
        all_raw_data = manager.get_all_raw_data()
        
        # نمایش خلاصه
        print("\n📋 خلاصه داده‌های خام:")
        for endpoint in all_raw_data['endpoints_called']:
            status = "✅ دارد" if all_raw_data.get(endpoint) else "❌ ندارد"
            print(f"  {endpoint}: {status}")

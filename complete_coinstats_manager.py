# coinstats_complete_endpoints.py
import requests
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

class CoinStatsAPI:
    def __init__(self, api_key: str = None):
        self.base_url = "https://openapiv1.coinstats.app"
        self.api_key = api_key or "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="
        self.headers = {"X-API-KEY": self.api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # تنظیمات لاگینگ
        self.setup_logging()
    
    def setup_logging(self):
        """تنظیمات لاگینگ"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def debug_request(self, endpoint: str, params: Dict = None, response = None):
        """لاگ کامل درخواست و پاسخ"""
        print("\n" + "="*60)
        print(f"🔍 دیباگ اندپوینت: {endpoint}")
        print("="*60)
        print(f"📤 درخواست: GET {self.base_url}/{endpoint}")
        if params:
            print(f"📋 پارامترها: {params}")
        
        if response:
            print(f"📥 وضعیت: {response.status_code}")
            print(f"⏱ زمان پاسخ: {response.elapsed.total_seconds():.2f} ثانیه")
            
            # نمایش هدرهای مهم
            important_headers = ['X-RateLimit-Remaining', 'X-RateLimit-Limit', 'Content-Type']
            for header in important_headers:
                if header in response.headers:
                    print(f"📊 {header}: {response.headers[header]}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'result' in data and isinstance(data['result'], list):
                        print(f"✅ داده‌ها: {len(data['result'])} آیتم دریافت شد")
                        if data['result']:
                            sample = data['result'][0]
                            print(f"📝 نمونه: {sample.get('name', 'N/A')} - ${sample.get('price', 0):.2f}")
                    else:
                        print(f"✅ پاسخ: {json.dumps(data, ensure_ascii=False)[:200]}...")
                except:
                    print(f"📝 پاسخ متنی: {response.text[:200]}...")
            else:
                print(f"❌ خطا: {response.text}")
        print("="*60)

    def make_request(self, endpoint: str, params: Dict = None, max_retries: int = 2) -> Optional[Dict]:
        """ساخت درخواست به API با دیباگ و retry"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=15)
                self.debug_request(endpoint, params, response)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"⏳ Rate limit! صبر {wait_time} ثانیه...")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"❌ خطای {response.status_code} در {endpoint}")
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"⏰ Timeout در {endpoint} (تلاش {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
            except requests.exceptions.ConnectionError:
                self.logger.error(f"🌐 خطای اتصال در {endpoint}")
                return None
            except Exception as e:
                self.logger.error(f"💥 خطای ناشناخته در {endpoint}: {e}")
                return None
        
        return None

    # ========================= اندپوینت 1: دریافت لیست کوین‌ها =========================
    
    def get_coins_list(self, 
                      page: int = 1,
                      limit: int = 20,
                      coin_ids: str = None,
                      currency: str = "USD",
                      name: str = None,
                      symbol: str = None,
                      blockchains: str = None,
                      include_risk_score: bool = False,
                      categories: str = None,
                      sort_by: str = "rank",
                      sort_dir: str = "asc",
                      **filters) -> Optional[Dict]:
        """
        دریافت لیست کامل کوین‌ها با تمام فیلترهای موجود
        
        Parameters based on PDF:
        - page: شماره صفحه (پیش‌فرض: 1)
        - limit: تعداد در هر صفحه (پیش‌فرض: 20)
        - coin_ids: فیلتر با ID کوین‌ها (مثلاً bitcoin,ethereum)
        - currency: ارز نمایش قیمت (پیش‌فرض: USD)
        - name: جستجو بر اساس نام
        - symbol: فیلتر بر اساس نماد (مثلاً BTC)
        - blockchains: فیلتر بر اساس بلاکچین
        - include_risk_score: شامل شدن امتیاز ریسک
        - categories: فیلتر بر اساس دسته‌بندی
        - sort_by: فیلد مرتب‌سازی
        - sort_dir: جهت مرتب‌سازی (asc/desc)
        """
        
        params = {
            "page": page,
            "limit": limit,
            "currency": currency,
            "sortBy": sort_by,
            "sortDir": sort_dir,
        }
        
        # اضافه کردن پارامترهای اختیاری
        if coin_ids:
            params["coinIds"] = coin_ids
        if name:
            params["name"] = name
        if symbol:
            params["symbol"] = symbol
        if blockchains:
            params["blockchains"] = blockchains
        if include_risk_score:
            params["includeRiskScore"] = "true"
        if categories:
            params["categories"] = categories
            
        # اضافه کردن فیلترهای پیشرفته از PDF
        filter_mappings = {
            'market_cap_greater_than': 'marketCap~greaterThan',
            'market_cap_less_than': 'marketCap~lessThan',
            'volume_greater_than': 'volume~greaterThan', 
            'price_change_1h_greater_than': 'priceChange1h~greaterThan',
            'price_change_1d_greater_than': 'priceChange1d~greaterThan',
            'price_change_7d_greater_than': 'priceChange7d~greaterThan',
            'rank_greater_than': 'rank~greaterThan',
            'rank_less_than': 'rank~lessThan',
            'price_greater_than': 'price~greaterThan',
            'price_less_than': 'price~lessThan'
        }
        
        for filter_key, api_key in filter_mappings.items():
            if filter_key in filters and filters[filter_key] is not None:
                params[api_key] = str(filters[filter_key])
        
        return self.make_request("coins", params)

    # ========================= اندپوینت 2: دریافت چارت چندکوینه =========================
    
    def get_coins_charts(self, 
                        coin_ids: str, 
                        period: str = "all") -> Optional[Dict]:
        """
        دریافت چارت برای چند کوین
        
        Parameters:
        - coin_ids: لیست ID کوین‌ها (مثلاً bitcoin,ethereum,solana)
        - period: بازه زمانی (1h, 4h, 8h, 1d, 7d, 1m, 3m, 1y, all)
        """
        
        if period not in ["1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"]:
            self.logger.warning("⚠️ تایم‌فریم نامعتبر، استفاده از all")
            period = "all"
            
        params = {
            "coinIds": coin_ids,
            "period": period
        }
        
        return self.make_request("coins/charts", params)

    # ========================= اندپوینت 3: دریافت جزئیات کوین خاص =========================
    
    def get_coin_details(self, 
                        coin_id: str, 
                        currency: str = "USD") -> Optional[Dict]:
        """
        دریافت جزئیات کامل یک کوین خاص
        
        Parameters:
        - coin_id: شناسه کوین (مثلاً bitcoin, ethereum)
        - currency: ارز نمایش قیمت
        """
        
        params = {"currency": currency}
        return self.make_request(f"coins/{coin_id}", params)

    # ========================= اندپوینت 4: دریافت چارت کوین خاص =========================
    
    def get_coin_charts(self, 
                       coin_id: str, 
                       period: str = "all") -> Optional[Dict]:
        """
        دریافت چارت تاریخی برای یک کوین خاص
        
        Parameters:
        - coin_id: شناسه کوین
        - period: بازه زمانی
        """
        
        if period not in ["1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"]:
            period = "all"
            
        params = {"period": period}
        return self.make_request(f"coins/{coin_id}/charts", params)

    # ========================= اندپوینت 5: دریافت قیمت متوسط =========================
    
    def get_coin_price_avg(self, 
                          coin_id: str, 
                          timestamp: str) -> Optional[Dict]:
        """
        دریافت قیمت متوسط کوین در زمان مشخص
        
        Parameters:
        - coin_id: شناسه کوین
        - timestamp: timestamp زمانی
        """
        
        params = {
            "coinId": coin_id,
            "timestamp": timestamp
        }
        
        return self.make_request("coins/price/avg", params)

    # ========================= اندپوینت 6: دریافت قیمت مبادله =========================
    
    def get_exchange_price(self, 
                          exchange: str, 
                          from_coin: str, 
                          to_coin: str, 
                          timestamp: str) -> Optional[Dict]:
        """
        دریافت قیمت مبادله در صرافی خاص
        
        Parameters:
        - exchange: نام صرافی (مثلاً Binance)
        - from_coin: ارز مبدأ (مثلاً BTC)
        - to_coin: ارز مقصد (مثلاً ETH)
        - timestamp: timestamp زمانی
        """
        
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        
        return self.make_request("coins/price/exchange", params)

    # ========================= تست تمام اندپوینت‌ها =========================
    
    def test_all_endpoints(self):
        """تست کامل تمام 6 اندپوینت"""
        print("\n" + "🎯" * 30)
        print("🚀 شروع تست کامل 6 اندپوینت اول")
        print("🎯" * 30)
        
        test_results = {}
        
        # تست 1: لیست کوین‌ها
        print("\n1️⃣ تست دریافت لیست کوین‌ها")
        test_results['coins_list'] = self.get_coins_list(limit=5)
        
        # تست 2: چارت چندکوینه
        print("\n2️⃣ تست چارت چندکوینه")
        test_results['multi_chart'] = self.get_coins_charts("bitcoin,ethereum", "1d")
        
        # تست 3: جزئیات کوین خاص
        print("\n3️⃣ تست جزئیات بیت‌کوین")
        test_results['bitcoin_details'] = self.get_coin_details("bitcoin")
        
        # تست 4: چارت کوین خاص
        print("\n4️⃣ تست چارت بیت‌کوین")
        test_results['bitcoin_chart'] = self.get_coin_charts("bitcoin", "7d")
        
        # تست 5: قیمت متوسط
        print("\n5️⃣ تست قیمت متوسط")
        test_results['price_avg'] = self.get_coin_price_avg("bitcoin", "1636315200")
        
        # تست 6: قیمت مبادله
        print("\n6️⃣ تست قیمت مبادله")
        test_results['exchange_price'] = self.get_exchange_price("Binance", "BTC", "ETH", "1636315200")
        
        # خلاصه نتایج
        print("\n" + "📊" * 30)
        print("نتایج تست:")
        print("📊" * 30)
        
        for endpoint, result in test_results.items():
            status = "✅ موفق" if result else "❌ شکست"
            print(f"{endpoint}: {status}")
            
        return test_results

    # ========================= مثال‌های کاربردی =========================
    
    def get_top_coins(self, limit: int = 10) -> List[Dict]:
        """دریافت کوین‌های برتر"""
        data = self.get_coins_list(limit=limit, sort_by="marketCap", sort_dir="desc")
        if data and 'result' in data:
            return data['result']
        return []
    
    def search_coins_by_name(self, name: str, limit: int = 20) -> List[Dict]:
        """جستجوی کوین بر اساس نام"""
        data = self.get_coins_list(name=name, limit=limit)
        if data and 'result' in data:
            return data['result']
        return []
    
    def get_coin_historical_data(self, coin_id: str, days: int = 30) -> Dict:
        """دریافت داده‌های تاریخی کوین"""
        period_map = {
            1: "1d", 7: "7d", 30: "1m", 90: "3m", 365: "1y"
        }
        period = period_map.get(days, "all")
        return self.get_coin_charts(coin_id, period)
    
    def get_coins_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """دریافت کوین‌ها بر اساس دسته‌بندی"""
        data = self.get_coins_list(categories=category, limit=limit)
        if data and 'result' in data:
            return data['result']
        return []

# ========================= اجرای تست =========================

if __name__ == "__main__":
    # ایجاد نمونه API
    api = CoinStatsAPI()
    
    print("🔑 کلید API:", api.api_key[:20] + "..." if api.api_key else "None")
    
    # تست اتصال اولیه
    print("\n🧪 تست اتصال اولیه...")
    test_data = api.get_coins_list(limit=1)
    
    if test_data:
        print("✅ اتصال API موفقیت‌آمیز بود!")
        
        # تست کامل تمام اندپوینت‌ها
        results = api.test_all_endpoints()
        
        # نمایش نمونه‌ای از داده‌ها
        if results.get('coins_list'):
            coins = results['coins_list'].get('result', [])
            if coins:
                print(f"\n🎉 نمونه داده: {coins[0].get('name')} - ${coins[0].get('price', 0):.2f}")
    else:
        print("❌ خطا در اتصال به API")
        print("💡 بررسی کنید:")
        print("   - کلید API معتبر است")
        print("   - اتصال اینترنت برقرار است")
        print("   - سرور API در دسترس است")

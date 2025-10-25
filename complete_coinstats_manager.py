# coinstats_complete_endpoints.py - نسخه کامل با 14 اندپوینت
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
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'result' in data and isinstance(data['result'], list):
                        print(f"✅ داده‌ها: {len(data['result'])} آیتم دریافت شد")
                    else:
                        print(f"✅ پاسخ دریافت شد")
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
                    wait_time = 2 ** attempt
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
            except Exception as e:
                self.logger.error(f"💥 خطای ناشناخته در {endpoint}: {e}")
                return None
        
        return None

    # ========================= اندپوینت‌های اصلی (6 تای اول) =========================
    
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
            
        # اضافه کردن فیلترهای پیشرفته
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

    def get_coins_charts(self, 
                        coin_ids: str, 
                        period: str = "1w") -> Optional[Dict]:
        """
        دریافت چارت برای چند کوین
        """
        valid_periods = ["24h", "1w", "1m", "3m", "6m", "1y", "all"]
        if period not in valid_periods:
            self.logger.warning("⚠️ تایم‌فریم نامعتبر، استفاده از 1w")
            period = "1w"
            
        params = {
            "coinIds": coin_ids,
            "period": period
        }
        
        return self.make_request("coins/charts", params)

    def get_coin_details(self, 
                        coin_id: str, 
                        currency: str = "USD") -> Optional[Dict]:
        """
        دریافت جزئیات کامل یک کوین خاص
        """
        params = {"currency": currency}
        return self.make_request(f"coins/{coin_id}", params)

    def get_coin_charts(self, 
                       coin_id: str, 
                       period: str = "1w") -> Optional[Dict]:
        """
        دریافت چارت تاریخی برای یک کوین خاص
        """
        valid_periods = ["24h", "1w", "1m", "3m", "6m", "1y", "all"]
        if period not in valid_periods:
            period = "1w"
            
        params = {"period": period}
        return self.make_request(f"coins/{coin_id}/charts", params)

    def get_coin_price_avg(self, 
                          coin_id: str, 
                          timestamp: str) -> Optional[Dict]:
        """
        دریافت قیمت متوسط کوین در زمان مشخص
        """
        params = {
            "coinId": coin_id,
            "timestamp": timestamp
        }
        
        return self.make_request("coins/price/avg", params)

    def get_exchange_price(self, 
                          exchange: str, 
                          from_coin: str, 
                          to_coin: str, 
                          timestamp: str) -> Optional[Dict]:
        """
        دریافت قیمت مبادله در صرافی خاص
        """
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        
        return self.make_request("coins/price/exchange", params)

    # ========================= اندپوینت‌های جدید (8 تای جدید) =========================

    def get_tickers_exchanges(self) -> Optional[Dict]:
        """
        دریافت لیست صرافی‌ها
        از مستندات: /tickers/exchanges
        """
        return self.make_request("tickers/exchanges")

    def get_tickers_markets(self) -> Optional[Dict]:
        """
        دریافت لیست مارکت‌ها
        از مستندات: /tickers/markets
        """
        return self.make_request("tickers/markets")

    def get_fiats(self) -> Optional[Dict]:
        """
        دریافت لیست ارزهای فیات
        از مستندات: /fiats
        """
        return self.make_request("fiats")

    def get_markets(self) -> Optional[Dict]:
        """
        دریافت لیست مارکت‌ها (نسخه دیگر)
        از مستندات: /markets
        """
        return self.make_request("markets")

    def get_currencies(self) -> Optional[Dict]:
        """
        دریافت لیست ارزها
        از مستندات: /currencies
        """
        return self.make_request("currencies")

    def get_news_sources(self) -> Optional[Dict]:
        """
        دریافت منابع خبری
        از مستندات: /news/sources
        """
        return self.make_request("news/sources")

    def get_news(self, 
                limit: int = 20,
                type: str = None) -> Optional[Dict]:
        """
        دریافت اخبار
        از مستندات: /news
        انواع: handpicked, trending, latest, bullish, bearish
        """
        params = {"limit": limit}
        if type:
            params["type"] = type
        
        return self.make_request("news", params)

    def get_news_detail(self, news_id: str) -> Optional[Dict]:
        """
        دریافت جزئیات خبر خاص
        از مستندات: /news/{newsId}
        """
        return self.make_request(f"news/{news_id}")

    def get_news_by_type(self, news_type: str, limit: int = 20) -> Optional[Dict]:
        """
        دریافت اخبار بر اساس نوع
        از مستندات: /news/type/{type}
        انواع: handpicked, trending, latest, bullish, bearish
        """
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            self.logger.warning(f"⚠️ نوع خبر نامعتبر، استفاده از latest")
            news_type = "latest"
            
        return self.make_request(f"news/type/{news_type}", {"limit": limit})

    def get_btc_dominance(self, type: str = "all") -> Optional[Dict]:
        """
        دریافت دامیننس بیت‌کوین
        از مستندات: /insights/btc-dominance
        انواع: all, 24h, etc.
        """
        params = {"type": type}
        return self.make_request("insights/btc-dominance", params)

    def get_fear_greed(self) -> Optional[Dict]:
        """
        دریافت شاخص ترس و طمع
        از مستندات: /insights/fear-and-greed
        """
        return self.make_request("insights/fear-and-greed")

    def get_fear_greed_chart(self) -> Optional[Dict]:
        """
        دریافت چارت ترس و طمع
        از مستندات: /insights/fear-and-greed/chart
        """
        return self.make_request("insights/fear-and-greed/chart")

    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        """
        دریافت چارت رنگین‌کمان
        از مستندات: /insights/rainbow-chart/{coinId}
        """
        return self.make_request(f"insights/rainbow-chart/{coin_id}")

    # ========================= تست تمام اندپوینت‌ها =========================
    
    def test_all_endpoints(self):
        """تست کامل تمام 14 اندپوینت"""
        print("\n" + "🎯" * 30)
        print("🚀 شروع تست کامل 14 اندپوینت")
        print("🎯" * 30)
        
        test_results = {}
        
        # تست اندپوینت‌های اصلی
        print("\n1️⃣ تست دریافت لیست کوین‌ها")
        test_results['coins_list'] = self.get_coins_list(limit=5)
        
        print("\n2️⃣ تست چارت چندکوینه")
        test_results['multi_chart'] = self.get_coins_charts("bitcoin,ethereum", "1w")
        
        print("\n3️⃣ تست جزئیات بیت‌کوین")
        test_results['bitcoin_details'] = self.get_coin_details("bitcoin")
        
        print("\n4️⃣ تست چارت بیت‌کوین")
        test_results['bitcoin_chart'] = self.get_coin_charts("bitcoin", "1w")
        
        print("\n5️⃣ تست قیمت متوسط")
        test_results['price_avg'] = self.get_coin_price_avg("bitcoin", "1636315200")
        
        print("\n6️⃣ تست قیمت مبادله")
        test_results['exchange_price'] = self.get_exchange_price("Binance", "BTC", "ETH", "1636315200")
        
        # تست اندپوینت‌های جدید
        print("\n7️⃣ تست لیست صرافی‌ها")
        test_results['exchanges'] = self.get_tickers_exchanges()
        
        print("\n8️⃣ تست لیست مارکت‌ها")
        test_results['markets'] = self.get_tickers_markets()
        
        print("\n9️⃣ تست ارزهای فیات")
        test_results['fiats'] = self.get_fiats()
        
        print("\n🔟 تست مارکت‌ها (نسخه دیگر)")
        test_results['markets_v2'] = self.get_markets()
        
        print("\n1️⃣1️⃣ تست ارزها")
        test_results['currencies'] = self.get_currencies()
        
        print("\n1️⃣2️⃣ تست منابع خبری")
        test_results['news_sources'] = self.get_news_sources()
        
        print("\n1️⃣3️⃣ تست اخبار")
        test_results['news'] = self.get_news(limit=5)
        
        print("\n1️⃣4️⃣ تست دامیننس بیت‌کوین")
        test_results['btc_dominance'] = self.get_btc_dominance()
        
        print("\n1️⃣5️⃣ تست ترس و طمع")
        test_results['fear_greed'] = self.get_fear_greed()
        
        print("\n1️⃣6️⃣ تست چارت رنگین‌کمان")
        test_results['rainbow_chart'] = self.get_rainbow_chart("bitcoin")
        
        # خلاصه نتایج
        print("\n" + "📊" * 30)
        print("نتایج تست 14 اندپوینت:")
        print("📊" * 30)
        
        successful = 0
        for endpoint, result in test_results.items():
            status = "✅ موفق" if result else "❌ شکست"
            print(f"{endpoint}: {status}")
            if result:
                successful += 1
        
        print(f"\n🎉 موفق: {successful}/14 اندپوینت")
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
            1: "24h", 7: "1w", 30: "1m", 90: "3m", 365: "1y"
        }
        period = period_map.get(days, "all")
        return self.get_coin_charts(coin_id, period)
    
    def get_coins_by_category(self, category: str, limit: int = 20) -> List[Dict]:
        """دریافت کوین‌ها بر اساس دسته‌بندی"""
        data = self.get_coins_list(categories=category, limit=limit)
        if data and 'result' in data:
            return data['result']
        return []
    
    def get_latest_news(self, limit: int = 10, news_type: str = "latest") -> List[Dict]:
        """دریافت آخرین اخبار"""
        data = self.get_news(limit=limit, type=news_type)
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

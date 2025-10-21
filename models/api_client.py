# api_client.py
import requests
import json
import time
import os
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

class CoinStatsAPIClient:
    """
    کلاینت کامل برای API کوین‌استتس
    شامل تمام اندپوینت‌های مستند و داده‌های خام
    """
    
    def __init__(self, api_key: str = "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=", 
                 base_url: str = "https://openapiv1.coinstats.app"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-API-KEY": api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # تنظیمات لاگ
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # کش داخلی برای داده‌های تکراری
        self._cache = {}
        
    def _make_request(self, endpoint: str, params: Dict = None, method: str = "GET") -> Optional[Union[Dict, List]]:
        """ساخت درخواست عمومی با مدیریت خطا"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=30)
            else:
                response = self.session.post(url, json=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"خطای {response.status_code} برای {url}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"خطا در درخواست {url}: {e}")
            return None
    
    def _get_from_cache_or_api(self, cache_key: str, endpoint: str, params: Dict = None):
        """اول از کش می‌خواند، اگر نبود از API می‌گیرد"""
        if cache_key in self._cache:
            self.logger.info(f"داده از کش بازیابی شد: {cache_key}")
            return self._cache[cache_key]
        
        data = self._make_request(endpoint, params)
        if data:
            self._cache[cache_key] = data
        return data

    # ===== COINS ENDPOINTS =====
    
    def get_coins_list(self, limit: int = 100, page: int = 1, currency: str = "USD", 
                      sort_by: str = "marketCap", sort_dir: str = "desc",
                      coin_ids: str = None, search: str = None, symbol: str = None,
                      blockchains: str = None, categories: str = None,
                      include_risk_score: bool = False) -> Optional[Dict]:
        """
        دریافت لیست کوین‌ها
        مستندات: صفحه ۱-۶
        """
        params = {
            "limit": limit,
            "page": page,
            "currency": currency,
            "sortBy": sort_by,
            "sortDir": sort_dir
        }
        
        # پارامترهای اختیاری
        if coin_ids:
            params["coinIds"] = coin_ids
        if search:
            params["name"] = search
        if symbol:
            params["symbol"] = symbol
        if blockchains:
            params["blockchains"] = blockchains
        if categories:
            params["categories"] = categories
        if include_risk_score:
            params["includeRiskScore"] = "true"
            
        return self._make_request("coins", params)
    
    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Optional[Dict]:
        """
        دریافت جزئیات یک کوین خاص
        مستندات: صفحه ۳۵-۳۶
        """
        return self._make_request(f"coins/{coin_id}", {"currency": currency})
    
    def get_coin_chart(self, coin_id: str, period: str = "all") -> Optional[List]:
        """
        دریافت داده‌های چارت یک کوین
        مستندات: صفحه ۳۴-۳۵, ۳۷
        
        دوره‌های زمانی:
        - all: تمام تاریخ
        - 1y: یک سال
        - 1m: یک ماه
        - 1w: یک هفته
        - 1d: یک روز
        - 1h: یک ساعت
        """
        valid_periods = ["all", "1y", "1m", "1w", "1d", "1h"]
        if period not in valid_periods:
            self.logger.warning(f"دوره زمانی نامعتبر: {period}")
            return None
            
        return self._make_request(f"coins/{coin_id}/charts", {"period": period})
    
    def get_multiple_coins_charts(self, coin_ids: List[str], period: str = "all") -> Dict[str, Optional[List]]:
        """
        دریافت چارت چندین کوین به صورت همزمان
        """
        results = {}
        for coin_id in coin_ids:
            results[coin_id] = self.get_coin_chart(coin_id, period)
            time.sleep(0.5)  # مکث برای جلوگیری از rate limit
        return results

    # ===== PRICE ENDPOINTS =====
    
    def get_average_price(self, coin_id: str, timestamp: int) -> Optional[Dict]:
        """
        دریافت قیمت متوسط در زمان مشخص
        مستندات: صفحه ۳۸
        """
        return self._make_request("coins/price/avg", {
            "coinId": coin_id,
            "timestamp": timestamp
        })
    
    def get_exchange_price(self, exchange: str, from_coin: str, to_coin: str, timestamp: int) -> Optional[Dict]:
        """
        دریافت قیمت مبادله در صرافی خاص
        مستندات: صفحه ۳۹-۴۰
        """
        return self._make_request("coins/price/exchange", {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        })

    # ===== MARKET DATA ENDPOINTS =====
    
    def get_exchanges(self) -> Optional[List]:
        """
        دریافت لیست صرافی‌ها
        مستندات: صفحه ۴۰-۴۱
        """
        return self._make_request("tickers/exchanges")
    
    def get_markets(self) -> Optional[List]:
        """
        دریافت لیست بازارهای معاملاتی
        مستندات: صفحه ۴۱-۴۲
        """
        return self._make_request("tickers/markets")
    
    def get_fiats(self) -> Optional[List]:
        """
        دریافت لیست ارزهای فیات
        مستندات: صفحه ۴۲-۴۳
        """
        return self._make_request("fiats")
    
    def get_currencies(self) -> Optional[List]:
        """
        دریافت لیست ارزها
        مستندات: صفحه ۴۴-۴۵
        """
        return self._make_request("currencies")

    # ===== NEWS ENDPOINTS =====
    
    def get_news_sources(self) -> Optional[List]:
        """
        دریافت منابع خبری
        مستندات: صفحه ۴۵-۴۶
        """
        return self._make_request("news/sources")
    
    def get_news(self, limit: int = 50) -> Optional[List]:
        """
        دریافت اخبار
        مستندات: صفحه ۴۶
        """
        return self._make_request("news", {"limit": limit})
    
    def get_news_by_type(self, news_type: str) -> Optional[List]:
        """
        دریافت اخبار بر اساس نوع
        مستندات: صفحه ۴۷
        
        انواع:
        - handpicked: انتخاب شده
        - trending: ترندینگ
        - latest: آخرین
        - bullish: صعودی
        - bearish: نزولی
        """
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            self.logger.warning(f"نوع خبر نامعتبر: {news_type}")
            return None
            
        return self._make_request(f"news/type/{news_type}")
    
    def get_news_detail(self, news_id: str) -> Optional[Dict]:
        """
        دریافت جزئیات یک خبر خاص
        مستندات: صفحه ۴۸-۴۹
        """
        return self._make_request(f"news/{news_id}")

    # ===== INSIGHTS ENDPOINTS =====
    
    def get_btc_dominance(self, timeframe: str = "all") -> Optional[Dict]:
        """
        دریافت تسلط بیت‌کوین
        مستندات: صفحه ۴۹-۵۰
        
        تایم‌فریم‌ها:
        - all, 1y, 1m, 1w, 1d, 1h
        """
        return self._make_request("insights/btc-dominance", {"type": timeframe})
    
    def get_fear_greed_index(self) -> Optional[Dict]:
        """
        دریافت شاخص ترس و طمع
        مستندات: صفحه ۵۰-۵۱
        """
        return self._make_request("insights/fear-and-greed")
    
    def get_fear_greed_chart(self) -> Optional[List]:
        """
        دریافت نمودار شاخص ترس و طمع
        مستندات: صفحه ۵۱-۵۲
        """
        return self._make_request("insights/fear-and-greed/chart")
    
    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Optional[List]:
        """
        دریافت چارت رنگین کمان
        مستندات: صفحه ۵۲-۵۳
        """
        return self._make_request(f"insights/rainbow-chart/{coin_id}")

    # ===== DATA MANAGEMENT =====
    
    def load_raw_data(self, file_path: str) -> Optional[Dict]:
        """بارگذاری داده‌های خام از فایل"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری فایل {file_path}: {e}")
            return None
    
    def save_raw_data(self, data: Dict, file_path: str):
        """ذخیره داده‌های خام در فایل"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"داده در {file_path} ذخیره شد")
        except Exception as e:
            self.logger.error(f"خطا در ذخیره فایل {file_path}: {e}")
    
    def get_data_with_fallback(self, cache_key: str, endpoint: str, params: Dict = None, 
                             file_path: str = None) -> Optional[Union[Dict, List]]:
        """
        دریافت داده با fallback: اول فایل، سپس API
        """
        # اول از فایل بخوان
        if file_path and os.path.exists(file_path):
            file_data = self.load_raw_data(file_path)
            if file_data:
                self.logger.info(f"داده از فایل بازیابی شد: {file_path}")
                return file_data
        
        # اگر فایل نبود، از API بگیر
        api_data = self._get_from_cache_or_api(cache_key, endpoint, params)
        if api_data and file_path:
            self.save_raw_data(api_data, file_path)
            
        return api_data

    # ===== BATCH OPERATIONS =====
    
    def get_complete_market_data(self, output_dir: str = "crypto_data") -> Dict[str, any]:
        """
        دریافت کامل داده‌های بازار برای هوش مصنوعی
        """
        market_data = {}
        
        # 1. داده‌های اصلی
        market_data["coins"] = self.get_coins_list(limit=200)
        market_data["exchanges"] = self.get_exchanges()
        market_data["markets"] = self.get_markets()
        
        # 2. داده‌های تحلیلی
        market_data["analytics"] = {
            "fear_greed": self.get_fear_greed_index(),
            "fear_greed_chart": self.get_fear_greed_chart(),
            "btc_dominance": self.get_btc_dominance("all"),
            "rainbow_btc": self.get_rainbow_chart("bitcoin"),
            "rainbow_eth": self.get_rainbow_chart("ethereum")
        }
        
        # 3. اخبار
        market_data["news"] = {
            "trending": self.get_news_by_type("trending"),
            "latest": self.get_news_by_type("latest"),
            "bullish": self.get_news_by_type("bullish"),
            "bearish": self.get_news_by_type("bearish")
        }
        
        # ذخیره در فایل
        if output_dir:
            self.save_raw_data(market_data, f"{output_dir}/complete_market_data.json")
        
        return market_data

    # ===== REAL-TIME DATA INTEGRATION =====
    
    def get_realtime_data(self, file_path: str = "shared/realtime_prices.json") -> Dict:
        """
        خواندن داده‌های real-time از فایل مشترک با Node.js WebSocket
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('realtime_data', {})
        except FileNotFoundError:
            self.logger.warning("فایل realtime data یافت نشد")
            return {}
        except Exception as e:
            self.logger.error(f"خطا در خواندن realtime data: {e}")
            return {}
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """
        دریافت قیمت لحظه‌ای یک symbol خاص
        """
        realtime_data = self.get_realtime_data()
        coin_data = realtime_data.get(symbol, {})
        return coin_data.get('price')
    
    def get_market_overview(self) -> Dict:
        """
        نمای کلی بازار از ترکیب داده‌های API و real-time
        """
        api_data = self.get_coins_list(limit=50)
        realtime_data = self.get_realtime_data()
        
        return {
            'api_data': api_data,
            'realtime_data': realtime_data,
            'combined_coins': self._combine_data(api_data, realtime_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _combine_data(self, api_data: Dict, realtime_data: Dict) -> List[Dict]:
        """
        ترکیب داده‌های API و real-time
        """
        combined = []
        
        if api_data and 'result' in api_data:
            for coin in api_data['result']:
                symbol = coin.get('symbol')
                if symbol and symbol in realtime_data:
                    combined.append({
                        **coin,
                        'live_price': realtime_data[symbol].get('price'),
                        'live_volume': realtime_data[symbol].get('volume'),
                        'live_change': realtime_data[symbol].get('change'),
                        'last_updated': realtime_data[symbol].get('last_updated')
                    })
        
        return combined

    # ===== UTILITY METHODS =====
    
    def clear_cache(self):
        """پاک کردن کش داخلی"""
        self._cache.clear()
        self.logger.info("کش پاک شد")
    
    def get_api_status(self) -> Dict:
        """بررسی وضعیت API"""
        test_data = self.get_coins_list(limit=1)
        
        return {
            "api_connected": test_data is not None,
            "cache_size": len(self._cache),
            "last_checked": datetime.now().isoformat()
        }


# نمونه استفاده سریع
if __name__ == "__main__":
    # ایجاد کلاینت
    client = CoinStatsAPIClient()
    
    # تست اتصال
    status = client.get_api_status()
    print(f"✅ وضعیت API: {status}")
    
    # تست دریافت داده
    coins = client.get_coins_list(limit=5)
    if coins:
        print(f"📊 تست موفق - تعداد کوین‌ها: {len(coins.get('result', []))}")
        
        # تست داده‌های real-time
        realtime = client.get_realtime_data()
        print(f"📡 داده‌های real-time: {len(realtime)} نماد")
    else:
        print("❌ خطا در اتصال به API")

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
    کلاینت بهینه‌شده برای استفاده از داده‌های remote + fallback به API
    """
    
    def __init__(self, api_key: str = "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA=", 
                 base_url: str = "https://openapiv1.coinstats.app",
                 data_repo_url: str = "https://raw.githubusercontent.com/hanzo7656-prog/crypto-ai-dataset/main/raw_data"):
        
        self.api_key = api_key
        self.base_url = base_url
        self.data_repo_url = data_repo_url
        self.headers = {"X-API-KEY": api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # تنظیمات لاگ
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # کش داخلی
        self._cache = {}

    def _make_request(self, endpoint: str, params: Dict = None, method: str = "GET") -> Optional[Union[Dict, List]]:
        """ساخت درخواست به API با مدیریت خطا"""
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

    def _load_remote_data(self, filename: str) -> Optional[Union[Dict, List]]:
        """بارگذاری داده از ریپوی GitHub با پشتیبانی از پوشه‌ها"""
        try:
            # اول سعی کن مستقیم بارگذاری کنی
            direct_url = f"{self.data_repo_url}/{filename}"
            response = requests.get(direct_url, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"🌐 داده مستقیم بارگذاری شد: {filename}")
                return response.json()
            
            # اگر فایل مستقیم پیدا نشد، در پوشه‌ها جستجو کن (برای فایل‌های کوین)
            if filename.endswith('.json') and not filename.startswith(('coins_list', 'analytical', 'market_news')):
                coin_id = filename.replace('.json', '')
                return self._find_coin_in_folders(coin_id)
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری {filename}: {e}")
            return None

    def _find_coin_in_folders(self, coin_id: str) -> Optional[Dict]:
        """جستجوی کوین در پوشه‌های مختلف"""
        folders = ['A', 'B', 'C', 'D']
        
        for folder in folders:
            try:
                url = f"{self.data_repo_url}/{folder}/{coin_id}.json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    self.logger.info(f"🌐 داده از پوشه {folder} بارگذاری شد: {coin_id}")
                    return response.json()
            except Exception as e:
                self.logger.debug(f"خطا در بارگذاری از پوشه {folder}: {e}")
                continue
        
        self.logger.warning(f"فایل {coin_id}.json در هیچ پوشه‌ای یافت نشد")
        return None

    # ===== COINS ENDPOINTS =====
    
    def get_coins_list(self, limit: int = 100, use_local: bool = True, **kwargs) -> Optional[Dict]:
        """دریافت لیست کوین‌ها - اول از ریپو سپس API"""
        if use_local:
            coins_data = self._load_remote_data("coins_list.json")
            if coins_data and 'result' in coins_data:
                # فیلتر کردن بر اساس limit
                filtered_data = coins_data.copy()
                if limit < len(filtered_data['result']):
                    filtered_data['result'] = filtered_data['result'][:limit]
                return filtered_data
        
        # Fallback به API
        params = {"limit": limit, **kwargs}
        return self._make_request("coins", params)
    
    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Optional[Dict]:
        """دریافت جزئیات یک کوین - فقط API (داده dynamic)"""
        return self._make_request(f"coins/{coin_id}", {"currency": currency})
    
    def get_coin_chart(self, coin_id: str, period: str = "all", use_local: bool = True) -> Optional[List]:
        """دریافت داده‌های چارت - اول از ریپو سپس API"""
        if use_local:
            # اول از ریپوی داده‌ها بارگذاری کن
            chart_data = self._find_coin_in_folders(coin_id)
            if chart_data and period in chart_data:
                self.logger.info(f"📊 چارت {coin_id} ({period}) از ریپو بارگذاری شد")
                return chart_data[period]
        
        # Fallback به API
        valid_periods = ["all", "1y", "1m", "1w", "1d", "1h"]
        if period not in valid_periods:
            self.logger.warning(f"دوره زمانی نامعتبر: {period}")
            return None
            
        return self._make_request(f"coins/{coin_id}/charts", {"period": period})
    
    def get_multiple_coins_charts(self, coin_ids: List[str], period: str = "all") -> Dict[str, Optional[List]]:
        """دریافت چارت چندین کوین به صورت همزمان"""
        results = {}
        for coin_id in coin_ids:
            results[coin_id] = self.get_coin_chart(coin_id, period)
            time.sleep(0.5)  # مکث برای جلوگیری از rate limit
        return results

    # ===== PRICE ENDPOINTS =====
    
    def get_average_price(self, coin_id: str, timestamp: int) -> Optional[Dict]:
        """دریافت قیمت متوسط در زمان مشخص"""
        return self._make_request("coins/price/avg", {
            "coinId": coin_id,
            "timestamp": timestamp
        })
    
    def get_exchange_price(self, exchange: str, from_coin: str, to_coin: str, timestamp: int) -> Optional[Dict]:
        """دریافت قیمت مبادله در صرافی خاص"""
        return self._make_request("coins/price/exchange", {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        })

    # ===== ANALYTICAL DATA =====
    
    def get_analytical_data(self, use_local: bool = True) -> Optional[Dict]:
        """دریافت داده‌های تحلیلی - اول از ریپو سپس API"""
        if use_local:
            analytical_data = self._load_remote_data("analytical_indicators.json")
            if analytical_data:
                return analytical_data
        
        # Fallback به API - جمع‌آوری تدریجی
        analytical_data = {}
        
        analytical_data['rainbow_btc'] = self.get_rainbow_chart("bitcoin")
        analytical_data['rainbow_eth'] = self.get_rainbow_chart("ethereum")
        analytical_data['fear_greed'] = self.get_fear_greed_index()
        analytical_data['fear_greed_chart'] = self.get_fear_greed_chart()
        analytical_data['btc_dominance'] = self.get_btc_dominance("all")
        
        return analytical_data

    # ===== MARKET DATA ENDPOINTS =====
    
    def get_market_news_data(self, use_local: bool = True) -> Optional[Dict]:
        """دریافت داده‌های بازار و اخبار - اول از ریپو سپس API"""
        if use_local:
            market_data = self._load_remote_data("market_news_data.json")
            if market_data:
                return market_data
        
        # Fallback به API
        market_data = {}
        
        market_data['exchanges'] = self.get_exchanges()
        market_data['markets'] = self.get_markets()
        market_data['fiats'] = self.get_fiats()
        
        # اخبار
        news_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        market_data['news'] = {}
        
        for news_type in news_types:
            market_data['news'][news_type] = self.get_news_by_type(news_type)
            time.sleep(0.3)
        
        return market_data

    def get_exchanges(self) -> Optional[List]:
        """دریافت لیست صرافی‌ها"""
        return self._make_request("tickers/exchanges")
    
    def get_markets(self) -> Optional[List]:
        """دریافت لیست بازارهای معاملاتی"""
        return self._make_request("tickers/markets")
    
    def get_fiats(self) -> Optional[List]:
        """دریافت لیست ارزهای فیات"""
        return self._make_request("fiats")

    # ===== NEWS ENDPOINTS =====
    
    def get_news_sources(self) -> Optional[List]:
        """دریافت منابع خبری"""
        return self._make_request("news/sources")
    
    def get_news(self, limit: int = 50) -> Optional[List]:
        """دریافت اخبار"""
        return self._make_request("news", {"limit": limit})
    
    def get_news_by_type(self, news_type: str) -> Optional[List]:
        """دریافت اخبار بر اساس نوع"""
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            self.logger.warning(f"نوع خبر نامعتبر: {news_type}")
            return None
            
        return self._make_request(f"news/type/{news_type}")
    
    def get_news_detail(self, news_id: str) -> Optional[Dict]:
        """دریافت جزئیات یک خبر خاص"""
        return self._make_request(f"news/{news_id}")

    # ===== INSIGHTS ENDPOINTS =====
    
    def get_btc_dominance(self, timeframe: str = "all") -> Optional[Dict]:
        """دریافت تسلط بیت‌کوین"""
        return self._make_request("insights/btc-dominance", {"type": timeframe})
    
    def get_fear_greed_index(self) -> Optional[Dict]:
        """دریافت شاخص ترس و طمع"""
        return self._make_request("insights/fear-and-greed")
    
    def get_fear_greed_chart(self) -> Optional[List]:
        """دریافت نمودار شاخص ترس و طمع"""
        return self._make_request("insights/fear-and-greed/chart")
    
    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Optional[List]:
        """دریافت چارت رنگین کمان"""
        return self._make_request(f"insights/rainbow-chart/{coin_id}")

    # ===== REAL-TIME DATA INTEGRATION =====
    
    def get_realtime_data(self, file_path: str = "shared/realtime_prices.json") -> Dict:
        """خواندن داده‌های real-time از فایل مشترک"""
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
        """دریافت قیمت لحظه‌ای"""
        realtime_data = self.get_realtime_data()
        coin_data = realtime_data.get(symbol, {})
        return coin_data.get('price')

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
        api_data = self._make_request(endpoint, params)
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
        market_data["coins"] = self.get_coins_list(limit=200, use_local=True)
        market_data["exchanges"] = self.get_exchanges()
        market_data["markets"] = self.get_markets()
        
        # 2. داده‌های تحلیلی
        market_data["analytics"] = self.get_analytical_data(use_local=True)
        
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

    # ===== UTILITY METHODS =====
    
    def clear_cache(self):
        """پاک کردن کش داخلی"""
        self._cache.clear()
        self.logger.info("کش پاک شد")
    
    def get_api_status(self) -> Dict:
        """بررسی وضعیت API"""
        test_data = self.get_coins_list(limit=1, use_local=False)
        
        return {
            "api_connected": test_data is not None,
            "cache_size": len(self._cache),
            "last_checked": datetime.now().isoformat()
        }
    
    def get_data_status(self) -> Dict:
        """وضعیت داده‌های available"""
        # تست اتصال به ریپوی داده‌ها
        test_files = ["coins_list.json", "analytical_indicators.json"]
        remote_available = False
        
        for test_file in test_files:
            if self._load_remote_data(test_file) is not None:
                remote_available = True
                break
        
        return {
            "remote_data_available": remote_available,
            "api_connected": self.get_coins_list(limit=1, use_local=False) is not None,
            "realtime_available": len(self.get_realtime_data()) > 0,
            "data_repo_url": self.data_repo_url
        }


# نمونه استفاده سریع
if __name__ == "__main__":
    # ایجاد کلاینت
    client = CoinStatsAPIClient()
    
    # تست اتصال
    status = client.get_data_status()
    print(f"✅ وضعیت داده‌ها: {status}")
    
    # تست دریافت داده
    coins = client.get_coins_list(limit=5, use_local=True)
    if coins:
        print(f"📊 تست موفق - تعداد کوین‌ها: {len(coins.get('result', []))}")
        
        # تست داده‌های تحلیلی
        analytics = client.get_analytical_data(use_local=True)
        print(f"📈 داده‌های تحلیلی: {'✅' if analytics else '❌'}")
        
        # تست چارت کوین
        btc_chart = client.get_coin_chart("bitcoin", "1m", use_local=True)
        print(f"📊 چارت بیت‌کوین: {'✅' if btc_chart else '❌'}")
        
        # تست داده‌های real-time
        realtime = client.get_realtime_data()
        print(f"📡 داده‌های real-time: {len(realtime)} نماد")
    else:
        print("❌ خطا در اتصال به داده‌ها")

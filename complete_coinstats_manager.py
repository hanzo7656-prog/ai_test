# complete_coinstats_manager.py - با سیستم کش محلی و داده‌های خام

import requests
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import glob
from pathlib import Path

logger = logging.getLogger(__name__)

class CompleteCoinStatsManager:
    def __init__(self, api_key: str = None):
        self.base_url = "https://openapiv1.coinstats.app"
        self.api_key = api_key or "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="
                                   
        self.headers = {"X-API-KEY": self.api_key}
        
        # تنظیمات کش
        self.cache_dir = "./coinstats_cache"
        self.cache_duration = 300  # 5 دقیقه
        
        # ایجاد پوشه کش
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # WebSocket compatibility
        self.ws_connected = False
        self.realtime_data = {}
        
        logger.info("✔️ CoinStats Manager Initialized with Local Cache - Raw Data Mode")

    def _get_cache_path(self, endpoint: str, params: Dict = None) -> str:
        """ایجاد مسیر فایل کش"""
        cache_key = endpoint.replace('/', '_')
        if params:
            params_str = '__'.join(f"{k}_{v}" for k, v in sorted(params.items()))
            cache_key += f"_{params_str}"
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """بررسی اعتبار کش"""
        if not os.path.exists(cache_path):
            return False
        file_time = os.path.getmtime(cache_path)
        return (time.time() - file_time) < self.cache_duration

    def _save_to_cache(self, cache_path: str, data: Dict):
        """ذخیره در کش"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'data': data,
                    'cached_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(seconds=self.cache_duration)).isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"✗ Cache save error: {e}")

    def _load_from_cache(self, cache_path: str) -> Optional[Dict]:
        """بارگذاری از کش"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            return cached_data.get('data')
        except Exception as e:
            logger.error(f"✗ Cache load error: {e}")
            return None

    def _make_api_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Dict:
        """ساخت درخواست به API با کش - بازگشت داده خام"""
        cache_path = self._get_cache_path(endpoint, params)
        
        # بررسی کش
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"🔍 Using cache for: {endpoint}")
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                return cached_data

        # درخواست به API
        url = f"{self.base_url}/{endpoint}"
        try:
            logger.info(f"🔍 API Raw Data Request: {endpoint}")
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # ذخیره در کش
                if use_cache:
                    self._save_to_cache(cache_path, data)
                
                logger.info(f"✅ Raw data received from {endpoint}")
                return data
            else:
                logger.error(f"✗ API Error {response.status_code}: {response.text}")
                # اگر API خطا داد، از کش استفاده کن (اگر موجود باشد)
                if use_cache and os.path.exists(cache_path):
                    logger.info("🔍 Using expired cache due to API error")
                    return self._load_from_cache(cache_path) or {}
                
                return {}
                
        except Exception as e:
            logger.error(f"🔍 API Request error: {e}")
            # استفاده از کش در صورت خطا
            if use_cache and os.path.exists(cache_path):
                logger.info("🔍 Using cache due to connection error")
                return self._load_from_cache(cache_path) or {}
            
            return {}

    def clear_cache(self, endpoint: str = None):
        """پاک کردن کش"""
        try:
            if endpoint:
                # پاک کردن کش خاص
                pattern = self._get_cache_path(endpoint, {}).replace('.json', '*.json')
                for file_path in glob.glob(pattern):
                    os.remove(file_path)
                    logger.info(f"🧹 Cleared cache: {os.path.basename(file_path)}")
            else:
                # پاک کردن تمام کش
                for file_path in glob.glob(os.path.join(self.cache_dir, "*.json")):
                    os.remove(file_path)
                logger.info("🧹 Cleared all cache")
        except Exception as e:
            logger.error(f"✗ Cache clear error: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """اطلاعات کش"""
        cache_files = list(Path(self.cache_dir).glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            'total_files': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': self.cache_dir,
            'cache_duration_seconds': self.cache_duration
        }

    # =============================== اندپوینت‌های اصلی =============================

    def get_coins_list(self, limit: int = 20, page: int = 1, currency: str = "USD",
                      sort_by: str = "rank", sort_dir: str = "asc", **filters) -> Dict:
        """دریافت لیست کوین‌ها - داده خام"""
        params = {
            "limit": limit,
            "page": page,
            "currency": currency,
            "sortBy": sort_by,
            "sortDir": sort_dir
        }
        
        # اضافه کردن فیلترها
        params.update(filters)
        
        return self._make_api_request("coins", params)

    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین - داده خام"""
        params = {"currency": currency}
        return self._make_api_request(f"coins/{coin_id}", params)

    def get_coin_charts(self, coin_id: str, period: str = "1w") -> Dict:
        """دریافت چارت کوین - داده خام"""
        valid_periods = ["24h", "1w", "1m", "3m", "6m", "1y", "all"]
        if period not in valid_periods:
            period = "1w"
        params = {"period": period}
        return self._make_api_request(f"coins/{coin_id}/charts", params)

    def get_coins_charts(self, coin_ids: str, period: str = "1w") -> Dict:
        """دریافت چارت چندکوینه - داده خام"""
        valid_periods = ["24h", "1w", "1m", "3m", "6m", "1y", "all"]
        if period not in valid_periods:
            period = "1w"
        params = {
            "coinIds": coin_ids,
            "period": period
        }
        return self._make_api_request("coins/charts", params)

    def get_coin_price_avg(self, coin_id: str, timestamp: str) -> Dict:
        """دریافت قیمت متوسط - داده خام"""
        params = {
            "coinId": coin_id,
            "timestamp": timestamp
        }
        return self._make_api_request("coins/price/avg", params)

    def get_exchange_price(self, exchange: str, from_coin: str, to_coin: str, timestamp: str) -> Dict:
        """دریافت قیمت مبادله - داده خام"""
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        return self._make_api_request("coins/price/exchange", params)

    # ============================= اندپوینت‌های جدید ============================

    def get_tickers_exchanges(self) -> Dict:
        """دریافت لیست صرافی‌ها - داده خام"""
        return self._make_api_request("tickers/exchanges")

    def get_tickers_markets(self) -> Dict:
        """دریافت لیست مارکت‌ها - داده خام"""
        return self._make_api_request("tickers/markets")

    def get_markets(self) -> Dict:
        """دریافت مارکت‌ها - داده خام"""
        return self._make_api_request("markets")

    def get_fiats(self) -> Dict:
        """دریافت ارزهای فیات - داده خام"""
        return self._make_api_request("fiats")

    def get_currencies(self) -> Dict:
        """دریافت ارزها - داده خام"""
        return self._make_api_request("currencies")

    # ============================= اندپوینت‌های اخبار =========================

    def get_news_sources(self) -> Dict:
        """دریافت منابع خبری - داده خام"""
        return self._make_api_request("news/sources")

    def get_news(self, limit: int = 50) -> Dict:
        """دریافت اخبار عمومی - داده خام"""
        params = {"limit": limit}
        return self._make_api_request("news", params)

    def get_news_by_type(self, news_type: str, limit: int = 50) -> Dict:
        """دریافت اخبار بر اساس نوع - داده خام"""
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            news_type = "latest"
        return self._make_api_request(f"news/type/{news_type}", {"limit": limit})

    def get_news_detail(self, news_id: str) -> Dict:
        """دریافت جزئیات خبر - داده خام"""
        return self._make_api_request(f"news/{news_id}")

    # ============================= اندپوینت‌های پیش‌بازار =========================

    def get_btc_dominance(self, period_type: str = "all") -> Dict:
        """دریافت دامیننس بیت کوین - داده خام"""
        valid_periods = ["all", "24h", "1w", "1m", "3m", "1y"]
        if period_type not in valid_periods:
            period_type = "all"
        params = {"type": period_type}
        return self._make_api_request("insights/btc-dominance", params)

    def get_fear_greed(self) -> Dict:
        """دریافت شاخص ترس و طمع - داده خام"""
        return self._make_api_request("insights/fear-and-greed")

    def get_fear_greed_chart(self) -> Dict:
        """دریافت چارت ترس و طمع - داده خام"""
        return self._make_api_request("insights/fear-and-greed/chart")

    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Dict:
        """دریافت چارت رنگین‌کمان - داده خام"""
        return self._make_api_request(f"insights/rainbow-chart/{coin_id}")

    # ============================= متدهای سازگاری =============================

    def get_realtime_price(self, symbol: str) -> Dict:
        """متد سازگاری برای WebSocket - داده خام"""
        # این متد برای سازگاری با کدهای قدیمی
        coin_data = self.get_coin_details(symbol.lower())
        if coin_data and 'result' in coin_data:
            result = coin_data['result']
            return {
                'price': result.get('price', 0),
                'volume': result.get('volume', 0),
                'change': result.get('priceChange1d', 0),
                'high_24h': result.get('high', 0),
                'low_24h': result.get('low', 0),
                'timestamp': datetime.now().isoformat()
            }
        return {}

    def _load_raw_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های خام از کش - سازگاری با AI"""
        # این متد داده‌های کش شده رو به AI می‌دهد
        cache_files = list(Path(self.cache_dir).glob("*.json"))
        raw_data = {}
        
        for file_path in cache_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cache_content = json.load(f)
                filename = file_path.name
                raw_data[filename] = cache_content.get('data', {})
            except Exception as e:
                logger.error(f"Error loading cache file {file_path}: {e}")
        
        return raw_data

    def get_all_coins(self, limit: int = 100) -> List[Dict]:
        """دریافت تمام کوین‌ها - سازگاری با AI - داده خام"""
        data = self.get_coins_list(limit=limit)
        return data.get('result', [])

# ایجاد نمونه گلوبال
coin_stats_manager = CompleteCoinStatsManager()

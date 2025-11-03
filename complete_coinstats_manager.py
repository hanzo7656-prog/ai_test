import requests
import json
import os
import time
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import glob
from pathlib import Path

logger = logging.getLogger(__name__)

class CompleteCoinStatsManager:
    def __init__(self, api_key: str = None):
        self.base_url = "https://openapiv1.coinstats.app"
        self.api_key = api_key or "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="

        self.session = requests.Session()
        self.headers = {"X-API-KEY": self.api_key}
        self.session.headers.update(self.headers)
        
        # تنظیمات کش
        self.cache_dir = "./coinstats_cache"
        self.cache_duration = 300  # 5 دقیقه
        
        # ایجاد پوشه کش
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # ریت لیمیتینگ
        self.last_request_time = 0
        self.min_interval = 0.2  # 200ms بین درخواست‌ها
        
        logger.info("✅ CoinStats Manager Initialized - Hybrid Mode Ready")

    def _rate_limit(self):
        """مدیریت ریت لیمیت"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()

    def _get_cache_path(self, endpoint: str, params: Dict = None) -> str:
        """ایجاد مسیر فایل کش"""
        import hashlib
        cache_key = endpoint.replace('/', '_')
        if params:
            params_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            cache_key += f"_{params_hash}"
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
            logger.error(f"❌ Cache save error: {e}")

    def _load_from_cache(self, cache_path: str) -> Optional[Dict]:
        """بارگذاری از کش"""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            return cached_data.get('data')
        except Exception:
            return None

    def _make_api_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Union[Dict, List]:
        """ساخت درخواست به API - نسخه ساده‌شده"""
        self._rate_limit()
        cache_path = self._get_cache_path(endpoint, params)

        # بررسی کش
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"🔍 Using cache for: {endpoint}")
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data

        # درخواست به API
        url = f"{self.base_url}/{endpoint}"
        try:
            logger.info(f"🔍 API Request: {endpoint}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=20
            )
            
            logger.info(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # ذخیره در کش
                if use_cache:
                    self._save_to_cache(cache_path, data)
                
                logger.info(f"✅ Data received from {endpoint}")
                return data
                
            else:
                logger.error(f"❌ API Error {response.status_code} for {endpoint}")
                return {"error": f"HTTP {response.status_code}", "status": "error"}
                
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout for {endpoint}")
            return {"error": "Timeout", "status": "error"}
            
        except Exception as e:
            logger.error(f"🚨 Error in {endpoint}: {e}")
            return {"error": str(e), "status": "error"}

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

    def get_coin_price_avg(self, coin_id: str = "bitcoin", timestamp: str = "2024-01-01") -> Dict:
        """دریافت قیمت متوسط"""
        timestamp_fixed = self._date_to_timestamp(timestamp)
        params = {
            "coinId": coin_id,
            "timestamp": timestamp_fixed
        }
        return self._make_api_request("coins/price/avg", params)

    def get_exchange_price(self, exchange: str = "Binance", from_coin: str = "BTC", 
                          to_coin: str = "ETH", timestamp: str = "1636315200") -> Dict:
        """دریافت قیمت exchange"""
        timestamp_fixed = self._date_to_timestamp(timestamp)
        params = {
            "exchange": "Binance",
            "from": "BTC",
            "to": "ETH",
            "timestamp": timestamp_fixed
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

    def get_news_by_type(self, news_type: str = "trending", limit: int = 10) -> Dict:
        """دریافت اخبار بر اساس نوع"""
        valid_types = ["trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            news_type = "trending"
        params = {"limit": limit}
        return self._make_api_request(f"news/type/{news_type}", params)

    def get_news_detail(self, news_id: str = "sample") -> Dict:
        """دریافت جزئیات خبر"""
        try:
            return self._make_api_request(f"news/{news_id}")
        except:
            return {"error": "News not available", "id": news_id}

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

    # ============================= متدهای کمکی =============================

    def _date_to_timestamp(self, date_str: str) -> str:
        """تبدیل تاریخ به تایم‌استمپ"""
        try:
            if not date_str:
                return str(int(datetime.now().timestamp()))
            
            if isinstance(date_str, (int, float)):
                return str(int(date_str))
            
            if isinstance(date_str, str):
                date_str = date_str.strip()
                
                if date_str.isdigit():
                    timestamp = int(date_str)
                    if len(date_str) >= 13:
                        timestamp = timestamp // 1000
                    return str(timestamp)
                
                # فرمت‌های تاریخ
                date_formats = [
                    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%d/%m/%Y", "%d/%m/%Y %H:%M:%S",
                    "%m/%d/%Y", "%m/%d/%Y %H:%M:%S", "%d-%m-%Y", "%d-%m-%Y %H:%M:%S"
                ]
                
                for date_format in date_formats:
                    try:
                        dt = datetime.strptime(date_str, date_format)
                        return str(int(dt.timestamp()))
                    except ValueError:
                        continue
            
            return str(int(datetime.now().timestamp()))
            
        except Exception as e:
            logger.error(f"❌ Error converting date '{date_str}': {e}")
            return str(int(datetime.now().timestamp()))

    def clear_cache(self, endpoint: str = None):
        """پاک کردن کش"""
        try:
            if endpoint:
                pattern = self._get_cache_path(endpoint, {}).replace('.json', '*.json')
                for file_path in glob.glob(pattern):
                    os.remove(file_path)
                    logger.info(f"🧹 Cleared cache: {os.path.basename(file_path)}")
            else:
                for file_path in glob.glob(os.path.join(self.cache_dir, "*.json")):
                    os.remove(file_path)
                logger.info("🧹 Cleared all cache")
        except Exception as e:
            logger.error(f"❌ Cache clear error: {e}")

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

    def get_all_coins(self, limit: int = 100) -> List[Dict]:
        """دریافت تمام کوین‌ها - سازگاری با AI"""
        data = self.get_coins_list(limit=limit)
        return data.get('result', [])

    def get_api_status(self) -> Dict[str, Any]:
        """وضعیت API"""
        try:
            test_data = self.get_coins_list(limit=1)
            return {
                'status': 'connected' if test_data and 'result' in test_data else 'disconnected',
                'timestamp': datetime.now().isoformat(),
                'cache_info': self.get_cache_info()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # ============================= متدهای جدید برای سیستم وضعیت =============================

    def test_all_endpoints(self) -> Dict[str, Any]:
        """تست سلامت تمام اندپوینت‌های API"""
        endpoints = {
            "coins_list": lambda: self.get_coins_list(limit=1),
            "coin_details_btc": lambda: self.get_coin_details("bitcoin"),
            "coin_details_eth": lambda: self.get_coin_details("ethereum"),
            "coin_charts": lambda: self.get_coin_charts("bitcoin", "1w"),
            "news": lambda: self.get_news(limit=5),
            "btc_dominance": lambda: self.get_btc_dominance(),
            "fear_greed": lambda: self.get_fear_greed(),
            "tickers_exchanges": lambda: self.get_tickers_exchanges(),
            "markets": lambda: self.get_markets(),
            "fiats": lambda: self.get_fiats()
        }
        
        results = {}
        for name, endpoint_func in endpoints.items():
            try:
                start_time = time.time()
                result = endpoint_func()
                response_time = round((time.time() - start_time) * 1000, 2)
                
                if isinstance(result, dict) and "error" in result:
                    results[name] = {"status": "error", "error": result["error"], "response_time": response_time}
                else:
                    results[name] = {"status": "success", "response_time": response_time}
                    
            except Exception as e:
                results[name] = {"status": "error", "error": str(e), "response_time": 0}
        
        return results

    def get_system_metrics(self) -> Dict[str, Any]:
        """دریافت متریک‌های سیستم"""
        try:
            # مصرف RAM
            memory = psutil.virtual_memory()
            # مصرف CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            # مصرف دیسک
            disk = psutil.disk_usage('/')
            
            return {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent
                },
                "cpu": {
                    "percent": cpu_percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": disk.percent
                }
            }
        except Exception as e:
            return {"error": f"System metrics error: {e}"}

# ایجاد نمونه گلوبال
coin_stats_manager = CompleteCoinStatsManager()

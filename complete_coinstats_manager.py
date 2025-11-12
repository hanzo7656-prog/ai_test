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

# ایمپورت سیستم نرمال‌سازی جدید
try:
    from debug_system.utils.data_normalizer import DataNormalizer, data_normalizer
except ImportError:
    # Fallback برای مواقع توسعه
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.utils.data_normalizer import DataNormalizer, data_normalizer

logger = logging.getLogger(__name__)

class CompleteCoinStatsManager:
    """
    مدیر کامل CoinStats API - نسخه جامع
    پشتیبانی از تمام endpointهای مستندات رسمی
    """
    
    def __init__(self, api_key: str = None):
        self.base_url = "https://openapiv1.coinstats.app"
        self.api_key = api_key or "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="

        # تنظیمات session
        self.session = requests.Session()
        self.headers = {"X-API-KEY": self.api_key}
        self.session.headers.update(self.headers)
        
        # سیستم نرمال‌سازی
        self.normalizer = data_normalizer
        
        # تنظیمات کش
        self.cache_dir = "./coinstats_cache"
        self.cache_duration = 300  # 5 دقیقه
        
        # ایجاد پوشه کش
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # ریت لیمیتینگ
        self.last_request_time = 0
        self.min_interval = 0.2  # 200ms بین درخواست‌ها
        
        # متریک‌های عملکرد
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("🚀 Complete CoinStats Manager Initialized - Full API Support")

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
            params_str = json.dumps(params, sort_keys=True)
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

    def test_api_connection_quick(self) -> bool:
        """تست سریع اتصال API - برای سیستم سلامت"""
        try:
            result = self._make_api_request('coins', {'limit': 1}, use_cache=False, simple_test=True)
            # بررسی اینکه پاسخ معتبر است و خطا ندارد
            return (result is not None and 
                    'error' not in result and 
                    isinstance(result, dict) and
                    'result' in result)  # بررسی ساختار مورد انتظار
        except Exception:
            return False
                logger.error(f"🔌 Connection error for {endpoint}")
                return {"error": "Connection error", "status": "error"}
            
            
    def _make_api_request(self, endpoint: str, params: Dict = None, use_cache: bool = True, 
                         simple_test: bool = False) -> Dict:
        """ساخت درخواست به API با مدیریت کامل خطا"""
    
        # برای تست سلامت، کش و ریت لیمیت را غیرفعال می‌کنیم
        if simple_test:
            use_cache = False
            # ریت لیمیت برای تست سریع
            current_time = time.time()
            if current_time - self.last_request_time < 0.1:  # 100ms
                time.sleep(0.1)
            self.last_request_time = current_time
        else:
            self._rate_limit()
    
        self.metrics['total_requests'] += 1
    
        if not simple_test:
            cache_path = self._get_cache_path(endpoint, params)
            # بررسی کش (فقط در حالت عادی)
            if use_cache and self._is_cache_valid(cache_path):
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    self.metrics['cache_hits'] += 1
                    logger.debug(f"🔍 Cache hit for: {endpoint}")
                    return cached_data
        
            self.metrics['cache_misses'] += 1

        url = f"{self.base_url}/{endpoint}"
        try:
            if not simple_test:
                logger.info(f"🌐 API Request: {endpoint} - Params: {params}")
        
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10 if simple_test else 20  # تایم‌اوت کوتاه‌تر برای تست
            )
        
            if not simple_test:
                logger.info(f"📡 API Response Status: {response.status_code}")
        
            if response.status_code == 200:
                data = response.json()
              
                # ذخیره در کش (فقط در حالت عادی)
                if not simple_test and use_cache:
                    self._save_to_cache(cache_path, data)
            
                self.metrics['successful_requests'] += 1
                if not simple_test:
                    logger.info(f"✅ Success: {endpoint}")
                return data
            else:
                self.metrics['failed_requests'] += 1
                if not simple_test:
                    logger.error(f"❌ API Error {response.status_code} for {endpoint}: {response.text}")
                return {
                    "error": f"HTTP {response.status_code}",
                    "message": response.text[:100] if simple_test else response.text,  # کوتاه برای تست
                    "status": "error"
                }
            
        except requests.exceptions.Timeout:
            self.metrics['failed_requests'] += 1
            if not simple_test:
                logger.error(f"⏰ Timeout for {endpoint}")
            return {"error": "Timeout", "status": "error"}
        
        except requests.exceptions.ConnectionError:
            self.metrics['failed_requests'] += 1
            if not simple_test:
                logger.error(f"🔌 Connection error for {endpoint}")
            return {"error": "Connection error", "status": "error"}
        
        except Exception as e:
            self.metrics['failed_requests'] += 1
            if not simple_test:
                logger.error(f"🚨 Unexpected error in {endpoint}: {e}")
            return {"error": str(e), "status": "error"}

        except Exception as e:
                self.metrics['failed_requests'] += 1
                logger.error(f"🚨 Unexpected error in {endpoint}: {e}")
            return {"error": str(e), "status": "error"}
    # =============================== COINS ENDPOINTS =============================

    def get_coins_list(self, limit: int = 20, page: int = 1, currency: str = "USD",
                      sort_by: str = "rank", sort_dir: str = "asc", **filters) -> Dict:
        """
        دریافت لیست کوین‌ها - مطابق مستندات صفحه 1-6
        پشتیبانی از تمام فیلترها و پارامترها
        """
        params = {
            "limit": limit,
            "page": page,
            "currency": currency,
            "sortBy": sort_by,
            "sortDir": sort_dir
        }
        
        # اضافه کردن فیلترهای اختیاری
        valid_filters = [
            'coinIds', 'name', 'symbol', 'blockchains', 'includeRiskScore',
            'categories', 'marketCap~greaterThan', 'marketCap~equals', 'marketCap~lessThan',
            'fullyDilutedValuation~greaterThan', 'fullyDilutedValuation~equals', 'fullyDilutedValuation~lessThan',
            'volume~greaterThan', 'volume~equals', 'volume~lessThan',
            'priceChange1h~greaterThan', 'priceChange1h~equals', 'priceChange1h~lessThan',
            'priceChange1d~greaterThan', 'priceChange1d~equals', 'priceChange1d~lessThan',
            'priceChange7d~greaterThan', 'priceChange7d~equals', 'priceChange7d~lessThan',
            'availableSupply~greaterThan', 'availableSupply~equals', 'availableSupply~lessThan',
            'totalSupply~greaterThan', 'totalSupply~equals', 'totalSupply~lessThan',
            'rank~greaterThan', 'rank~equals', 'rank~lessThan',
            'price~greaterThan', 'price~equals', 'price~lessThan',
            'riskScore~greaterThan', 'riskScore~equals', 'riskScore~lessThan'
        ]
        
        for filter_key, filter_value in filters.items():
            if filter_key in valid_filters and filter_value is not None:
                params[filter_key] = filter_value
        
        raw_data = self._make_api_request("coins", params)
        
        if "error" in raw_data:
            return raw_data
        
        return {
            "status": "success",
            "data": raw_data.get("result", []),
            "meta": raw_data.get("meta", {}),
            "pagination": {
                "page": raw_data.get("meta", {}).get("page", page),
                "limit": raw_data.get("meta", {}).get("limit", limit),
                "total": raw_data.get("meta", {}).get("itemCount", 0),
                "pages": raw_data.get("meta", {}).get("pageCount", 0)
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین - مطابق مستندات صفحه 35-36"""
        params = {"currency": currency}
        raw_data = self._make_api_request(f"coins/{coin_id}", params)
        
        if "error" in raw_data:
            return raw_data
        
        return {
            "status": "success",
            "data": raw_data,
            "timestamp": datetime.now().isoformat()
        }

    def get_coin_charts(self, coin_id: str, period: str = "1w") -> Dict:
        """دریافت چارت کوین - مطابق مستندات صفحه 37"""
        params = {"period": period, "coinIds": coin_id}
        raw_data = self._make_api_request("coins/charts", params)
        
        if "error" in raw_data:
            return raw_data
        
        return {
            "status": "success",
            "data": raw_data.get("result", []),
            "coin_id": coin_id,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }

    def get_coins_charts(self, coin_ids: str, period: str = "all") -> Dict:
        """دریافت چارت چندکوینه - مطابق مستندات صفحه 34-35"""
        params = {"coinIds": coin_ids, "period": period}
        raw_data = self._make_api_request("coins/charts", params)
        
        if "error" in raw_data:
            return raw_data
        
        return {
            "status": "success",
            "data": raw_data.get("result", []),
            "coin_ids": coin_ids,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }

    def get_coin_price_avg(self, coin_id: str = "bitcoin", timestamp: str = None) -> Dict:
        """دریافت قیمت متوسط - مطابق مستندات صفحه 38"""
        if not timestamp:
            timestamp = str(int(datetime.now().timestamp()))
            
        params = {"coinId": coin_id, "timestamp": timestamp}
        raw_data = self._make_api_request("coins/price/avg", params)
        
        if "error" in raw_data:
            return raw_data
        
        return {
            "status": "success",
            "data": raw_data,
            "coin_id": coin_id,
            "timestamp_query": timestamp,
            "timestamp": datetime.now().isoformat()
        }

    def get_exchange_price(self, exchange: str = "Binance", from_coin: str = "BTC", 
                          to_coin: str = "ETH", timestamp: str = None) -> Dict:
        """دریافت قیمت exchange - ساختار جدید"""
        if not timestamp:
            timestamp = str(int(datetime.now().timestamp()))
        
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        raw_data = self._make_api_request("coins/price/exchange", params)
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید - قیمت در data.price قرار دارد
        price_data = raw_data.get("data", {})
    
        return {
            "status": "success",
            "data": price_data,
            "exchange": exchange,
            "from_coin": from_coin,
            "to_coin": to_coin,
            "timestamp_query": timestamp,
            "timestamp": datetime.now().isoformat()
        }

    # ============================= NEWS ENDPOINTS =========================
    def get_news(self, limit: int = 50) -> Dict:
        """دریافت اخبار عمومی - مطابق مستندات صفحه 46"""
        raw_data = self._make_api_request("news")
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش داده‌ها بر اساس ساختار مستندات
        if isinstance(raw_data, list):
            news_list = raw_data
        elif isinstance(raw_data, dict):
            # از مستندات: داده در کلید 'result' قرار دارد
            news_list = raw_data.get("result", [])
        else:
            news_list = []
    
        limited_data = news_list[:limit]
    
        return {
            "status": "success",
            "data": limited_data,
            "total": len(limited_data),
            "timestamp": datetime.now().isoformat()
        }

    def get_news_by_type(self, news_type: str = "latest", limit: int = 10) -> Dict:
        """دریافت اخبار بر اساس نوع - مطابق مستندات صفحه 47"""
    
        # انواع معتبر از مستندات
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
    
        if news_type not in valid_types:
            return {
                "error": f"Invalid news type: {news_type}",
                "valid_types": valid_types,
                "status": "error"
            }
    
        # استفاده از endpoint مستند
        endpoint = f"news/type/{news_type}"
    
        raw_data = self._make_api_request(endpoint)
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش داده‌ها
        if isinstance(raw_data, list):
            news_list = raw_data
        elif isinstance(raw_data, dict):
            news_list = raw_data.get("result", [])
        else:
            news_list = []
    
        # اعمال محدودیت
        limited_data = news_list[:limit] if limit else news_list
    
        return {
            "status": "success",
            "data": limited_data,
            "news_type": news_type,
            "total": len(limited_data),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }

    def get_news_sources(self) -> Dict:
        """دریافت منابع خبری - مطابق مستندات صفحه 45"""
        raw_data = self._make_api_request("news/sources")
    
        if "error" in raw_data:
            return raw_data
      
        # پردازش داده‌ها
        if isinstance(raw_data, list):
            sources_list = raw_data
        elif isinstance(raw_data, dict):
            sources_list = raw_data.get("result", [])
        else:
            sources_list = []
    
        return {
            "status": "success",
            "data": sources_list,
            "timestamp": datetime.now().isoformat()
        }

    def get_news_detail(self, news_id: str) -> Dict:
        """دریافت جزئیات خبر - مطابق مستندات صفحه 48"""
        raw_data = self._make_api_request(f"news/{news_id}")
    
        if "error" in raw_data:
            return raw_data
    
        return {
            "status": "success",
            "data": raw_data,
            "timestamp": datetime.now().isoformat()
        }
    # ============================= EXCHANGES & MARKETS =========================
    def get_exchanges(self) -> Dict:
        """دریافت لیست صرافی‌ها - ساختار جدید"""
        raw_data = self._make_api_request("tickers/exchanges")
      
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید - استفاده از data به جای result
        exchanges_data = raw_data.get("data", raw_data.get("result", []))
    
        return {
            "status": "success",
            "data": exchanges_data,
            "timestamp": datetime.now().isoformat()
        }

    def get_exchanges_processed(self) -> Dict:
        """دریافت لیست صرافی‌های پردازش شده"""
        raw_data = self.get_exchanges()
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش داده‌های صرافی‌ها
        processed_exchanges = []
        for exchange in raw_data.get('data', []):
            processed_exchanges.append({
                'id': exchange.get('id'),
                'name': exchange.get('name'),
                'rank': exchange.get('rank'),
                'percentTotalVolume': exchange.get('percentTotalVolume'),
                'volumeUsd': exchange.get('volumeUsd'),
                'tradingPairs': exchange.get('tradingPairs'),
                'socket': exchange.get('socket'),
                'exchangeUrl': exchange.get('exchangeUrl'),
                'last_updated': datetime.now().isoformat()
            })
    
        return {
            'status': 'success',
            'data': processed_exchanges,
            'total': len(processed_exchanges),
            'timestamp': datetime.now().isoformat()
        }

    def get_markets(self) -> Dict:
        """دریافت مارکت‌ها - ساختار جدید"""
        raw_data = self._make_api_request("tickers/markets")
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید - استفاده از data به جای result
        markets_data = raw_data.get("data", raw_data.get("result", []))
    
        return {
            "status": "success",
            "data": markets_data,
            "timestamp": datetime.now().isoformat()
        }

    def get_fiats(self) -> Dict:
        """دریافت ارزهای فیات - ساختار جدید"""
        raw_data = self._make_api_request("fiats")
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید - استفاده از data به جای result
        fiats_data = raw_data.get("data", raw_data.get("result", []))
    
        return {
            "status": "success",
            "data": fiats_data,
            "timestamp": datetime.now().isoformat()
        }

    def get_currencies(self) -> Dict:
        """دریافت ارزها - ساختار جدید"""
        raw_data = self._make_api_request("currencies")
       
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید - استفاده از data به جای result
        currencies_data = raw_data.get("data", raw_data.get("result", []))
    
        return {
            "status": "success",
            "data": currencies_data,
            "timestamp": datetime.now().isoformat()
        }

    # ============================= INSIGHTS ENDPOINTS =========================

    def get_btc_dominance(self, period_type: str = "all") -> Dict:
        """دریافت دامیننس بیت کوین - مطابق مستندات صفحه 49-50"""
        params = {"type": period_type}
        raw_data = self._make_api_request("insights/btc-dominance", params)
        
        if "error" in raw_data:
            return raw_data
        
        return {
            "status": "success",
            "data": raw_data,
            "period_type": period_type,
            "timestamp": datetime.now().isoformat()
        }

    def get_fear_greed(self) -> Dict:
        """دریافت شاخص ترس و طمع - ساختار جدید"""
        raw_data = self._make_api_request("insights/fear-and-greed")
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید API
        if "now" in raw_data:
            return {
                "status": "success",
                "data": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Fallback برای ساختار قدیمی
            return {
                "status": "success", 
                "data": raw_data,
                "timestamp": datetime.now().isoformat()
            }

    def get_fear_greed_processed(self) -> Dict:
        """دریافت شاخص ترس و طمع پردازش شده - ساختار جدید"""
        raw_data = self.get_fear_greed()
    
        if "error" in raw_data:
            return raw_data
    
        fear_greed_data = raw_data.get('data', {})
    
        # پردازش ساختار جدید
        if "now" in fear_greed_data:
            current_data = fear_greed_data["now"]
            value = current_data.get('value', 50)
            value_classification = current_data.get('value_classification', 'Neutral')
        else:
            # Fallback برای ساختار قدیمی
            value = fear_greed_data.get('value', 50)
            value_classification = fear_greed_data.get('value_classification', 'Neutral')
    
        # تحلیل و پردازش
        if value >= 75:
            sentiment = "extreme_greed"
            recommendation = "CAUTION: Consider taking profits"
        elif value >= 55:
            sentiment = "greed" 
            recommendation = "OPTIMISTIC: Good for holding"
        elif value >= 45:
            sentiment = "neutral"
            recommendation = "NEUTRAL: Good for accumulation"
        elif value >= 25:
            sentiment = "fear"
            recommendation = "CAUTIOUS: Look for opportunities"
        else:
            sentiment = "extreme_fear"
            recommendation = "OPPORTUNITY: Potential for rebounds"
    
        processed_data = {
            'value': value,
            'value_classification': value_classification,
            'timestamp': datetime.now().isoformat(),
            'analysis': {
                'sentiment': sentiment,
                'risk_level': 'high' if value >= 75 or value <= 25 else 'medium',
                'market_condition': sentiment.replace('_', ' ').title()
            },
            'recommendation': recommendation,
            'last_updated': datetime.now().isoformat()
        }
    
        return {
            'status': 'success',
            'data': processed_data,
            'timestamp': datetime.now().isoformat()
        }

    def get_fear_greed_chart(self) -> Dict:
        """دریافت چارت ترس و طمع - ساختار جدید"""
        raw_data = self._make_api_request("insights/fear-and-greed/chart")
      
        if "error" in raw_data:
            return raw_data
     
        # پردازش ساختار جدید
        chart_data = raw_data.get('data', [])
    
        return {
            "status": "success",
            "data": chart_data,
            "timestamp": datetime.now().isoformat()
        }

    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Dict:
        """دریافت چارت رنگین‌کمان - ساختار جدید"""
        raw_data = self._make_api_request(f"insights/rainbow-chart/{coin_id}")
    
        if "error" in raw_data:
            return raw_data
    
        # پردازش ساختار جدید (لیست مستقیم)
        if isinstance(raw_data, list):
            return {
                "status": "success",
                "data": raw_data,
                "coin_id": coin_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Fallback برای ساختار قدیمی
            return {
                "status": "success",
                "data": raw_data.get('result', []),
                "coin_id": coin_id,
                "timestamp": datetime.now().isoformat()
            }

    # ============================= ADVANCED METHODS =========================

    def get_coins_list_processed(self, limit: int = 20, page: int = 1, currency: str = "USD",
                               sort_by: str = "rank", sort_dir: str = "asc", **filters) -> Dict:
        """دریافت لیست کوین‌ها به صورت پردازش شده"""
        raw_data = self.get_coins_list(limit, page, currency, sort_by, sort_dir, **filters)
        
        if "error" in raw_data:
            return raw_data
        
        # پردازش اضافی روی داده‌ها
        processed_coins = []
        for coin in raw_data.get('data', []):
            processed_coins.append({
                'id': coin.get('id'),
                'name': coin.get('name'),
                'symbol': coin.get('symbol'),
                'price': coin.get('price'),
                'price_change_24h': coin.get('priceChange1d'),
                'price_change_1h': coin.get('priceChange1h'),
                'price_change_1w': coin.get('priceChange1w'),
                'volume_24h': coin.get('volume'),
                'market_cap': coin.get('marketCap'),
                'rank': coin.get('rank'),
                'website': coin.get('websiteUrl'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_coins,
            'pagination': raw_data.get('pagination', {}),
            'timestamp': datetime.now().isoformat()
        }

    def get_coin_details_processed(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین به صورت پردازش شده"""
        raw_data = self.get_coin_details(coin_id, currency)
        
        if "error" in raw_data:
            return raw_data
        
        coin_data = raw_data.get('data', {})
        
        processed_data = {
            'id': coin_data.get('id'),
            'name': coin_data.get('name'),
            'symbol': coin_data.get('symbol'),
            'price': coin_data.get('price'),
            'price_change_24h': coin_data.get('priceChange1d'),
            'price_change_1h': coin_data.get('priceChange1h'),
            'price_change_1w': coin_data.get('priceChange1w'),
            'volume_24h': coin_data.get('volume'),
            'market_cap': coin_data.get('marketCap'),
            'rank': coin_data.get('rank'),
            'website': coin_data.get('websiteUrl'),
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'timestamp': datetime.now().isoformat()
        }

    def get_fear_greed_processed(self) -> Dict:
        """دریافت شاخص ترس و طمع به صورت پردازش شده"""
        raw_data = self.get_fear_greed()
        
        if "error" in raw_data:
            return raw_data
        
        fear_greed_data = raw_data.get('data', {})
        
        value = fear_greed_data.get('value', 50)
        if value >= 75:
            sentiment = "extreme_greed"
            recommendation = "CAUTION: Consider taking profits"
        elif value >= 55:
            sentiment = "greed" 
            recommendation = "OPTIMISTIC: Good for holding"
        elif value >= 45:
            sentiment = "neutral"
            recommendation = "NEUTRAL: Good for accumulation"
        elif value >= 25:
            sentiment = "fear"
            recommendation = "CAUTIOUS: Look for opportunities"
        else:
            sentiment = "extreme_fear"
            recommendation = "OPPORTUNITY: Potential for rebounds"
        
        processed_data = {
            'value': fear_greed_data.get('value'),
            'value_classification': fear_greed_data.get('value_classification'),
            'timestamp': fear_greed_data.get('timestamp'),
            'time_until_update': fear_greed_data.get('time_until_update'),
            'analysis': {
                'sentiment': sentiment,
                'risk_level': 'high' if value >= 75 or value <= 25 else 'medium',
                'market_condition': sentiment.replace('_', ' ').title()
            },
            'recommendation': recommendation,
            'last_updated': datetime.now().isoformat()
        }
        
        return {
            'status': 'success',
            'data': processed_data,
            'timestamp': datetime.now().isoformat()
        }

    # ============================= SYSTEM METHODS =============================

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

    def get_performance_metrics(self) -> Dict[str, Any]:
        """متریک‌های عملکرد"""
        total_requests = self.metrics['total_requests']
        success_rate = (self.metrics['successful_requests'] / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': round(success_rate, 2),
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'timestamp': datetime.now().isoformat()
        }

    def get_api_status(self) -> Dict[str, Any]:
        """وضعیت API - نسخه سازگار با health system"""
        try:
            # تست اتصال
            is_connected = self.test_api_connection_quick()
        
            return {
                'status': 'healthy' if is_connected else 'degraded',
                'connected': is_connected,
                'timestamp': datetime.now().isoformat(),
                'cache_info': self.get_cache_info(),
                'performance_metrics': self.get_performance_metrics()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def debug_endpoint(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """ابزار دیباگ برای تست endpointها"""
        raw_data = self._make_api_request(endpoint, params, use_cache=False)
        
        return {
            "endpoint": endpoint,
            "params": params,
            "response_status": "success" if "error" not in raw_data else "error",
            "response_data": raw_data,
            "response_type": str(type(raw_data)),
            "timestamp": datetime.now().isoformat()
        }

# ایجاد نمونه گلوبال
coin_stats_manager = CompleteCoinStatsManager()

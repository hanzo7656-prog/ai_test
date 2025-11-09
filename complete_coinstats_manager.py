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
    def __init__(self, api_key: str = None):
        self.base_url = "https://openapiv1.coinstats.app"
        self.api_key = api_key or "oYGlUrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="

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
        
        logger.info("✅ CoinStats Manager Initialized - With Smart Data Normalization")

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

    def _make_api_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Dict:
        """ساخت درخواست به API"""
        self._rate_limit()
        cache_path = self._get_cache_path(endpoint, params)

        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"🔍 Using cache for: {endpoint}")
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data

        url = f"{self.base_url}/{endpoint}"
        try:
            logger.info(f"🔍 API Request: {endpoint} - Params: {params}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=20
            )
            
            logger.info(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
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

    # =============================== COINS ENDPOINTS =============================

    def get_coins_list(self, limit: int = 20, page: int = 1, currency: str = "USD",
                      sort_by: str = "rank", sort_dir: str = "asc", **filters) -> Dict:
        """دریافت لیست کوین‌ها - مطابق مستندات صفحه 1-6"""
        params = {
            "limit": limit,
            "page": page,
            "currency": currency,
            "sortBy": sort_by,
            "sortDir": sort_dir
        }
        
        # اضافه کردن فیلترهای اختیاری
        params.update(filters)
        
        raw_data = self._make_api_request("coins", params)
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "coins/list")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "meta": normalized_result.metadata,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین - مطابق مستندات صفحه 35-36"""
        params = {"currency": currency}
        raw_data = self._make_api_request(f"coins/{coin_id}", params)
        
        # نرمال‌سازی داده‌ها - انتظار دیکشنری برای جزئیات کوین
        if isinstance(raw_data, dict) and "error" not in raw_data:
            # برای جزئیات کوین، داده را مستقیماً برمی‌گردانیم (لیست نیست)
            return {
                "status": "success",
                "result": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            normalized_result = self.normalizer.normalize(raw_data, f"coins/{coin_id}")
            
            if normalized_result.status == "error":
                return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
            
            # برای جزئیات کوین، اولین آیتم را برمی‌گردانیم
            result_data = normalized_result.data[0] if normalized_result.data else {}
            
            return {
                "status": "success",
                "result": result_data,
                "normalization_info": normalized_result.normalization_info,
                "timestamp": datetime.now().isoformat()
            }

    def get_coin_charts(self, coin_id: str, period: str = "1w") -> Dict:
        """دریافت چارت کوین - مطابق مستندات صفحه 37"""
        # اصلاح پارامترها بر اساس مستندات - باید coinIds باشد
        params = {
            "period": period,
            "coinIds": coin_id  # ✅ اصلاح بر اساس مستندات
        }
        raw_data = self._make_api_request("coins/charts", params)
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, f"coins/charts/{coin_id}")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "coin_id": coin_id,
            "period": period,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_coins_charts(self, coin_ids: str, period: str = "all") -> Dict:
        """دریافت چارت چندکوینه - مطابق مستندات صفحه 34-35"""
        params = {
            "coinIds": coin_ids,
            "period": period
        }
        raw_data = self._make_api_request("coins/charts", params)
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "coins/charts/multiple")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "coin_ids": coin_ids,
            "period": period,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_coin_price_avg(self, coin_id: str = "bitcoin", timestamp: str = "1636315200") -> Dict:
        """دریافت قیمت متوسط - مطابق مستندات صفحه 38"""
        timestamp_fixed = self._date_to_timestamp(timestamp)
        params = {
            "coinId": coin_id,
            "timestamp": timestamp_fixed
        }
        raw_data = self._make_api_request("coins/price/avg", params)
        
        # برای قیمت متوسط، داده را مستقیماً برمی‌گردانیم
        if isinstance(raw_data, dict) and "error" not in raw_data:
            return {
                "status": "success",
                "result": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "result": {"price": raw_data} if not isinstance(raw_data, dict) else raw_data,
                "timestamp": datetime.now().isoformat()
            }

    def get_exchange_price(self, exchange: str = "Binance", from_coin: str = "BTC", 
                          to_coin: str = "ETH", timestamp: str = "1636315200") -> Dict:
        """دریافت قیمت exchange - مطابق مستندات صفحه 39-40"""
        timestamp_fixed = self._date_to_timestamp(timestamp)
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp_fixed
        }
        raw_data = self._make_api_request("coins/price/exchange", params)
        
        # برای قیمت صرافی، داده را مستقیماً برمی‌گردانیم
        if isinstance(raw_data, dict) and "error" not in raw_data:
            return {
                "status": "success",
                "result": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "result": {"price": raw_data} if not isinstance(raw_data, dict) else raw_data,
                "timestamp": datetime.now().isoformat()
            }

    # ============================= EXCHANGES ENDPOINTS ===========================

    def get_exchanges(self) -> Dict:
        """دریافت لیست صرافی‌ها - مطابق مستندات صفحه 40-41"""
        raw_data = self._make_api_request("tickers/exchanges")
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "exchanges/list")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_markets(self) -> Dict:
        """دریافت مارکت‌ها - مطابق مستندات صفحه 43"""
        # اصلاح endpoint بر اساس مستندات - باید tickers/markets باشد
        raw_data = self._make_api_request("tickers/markets")  # ✅ اصلاح شده
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "markets")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_fiats(self) -> Dict:
        """دریافت ارزهای فیات - مطابق مستندات صفحه 42"""
        raw_data = self._make_api_request("fiats")
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "fiats")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_currencies(self) -> Dict:
        """دریافت ارزها - مطابق مستندات صفحه 44"""
        raw_data = self._make_api_request("currencies")
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "currencies")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    # ============================= NEWS ENDPOINTS =========================

    def get_news_sources(self) -> Dict:
        """دریافت منابع خبری - مطابق مستندات صفحه 45"""
        raw_data = self._make_api_request("news/sources")
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "news/sources")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_news(self, limit: int = 50) -> Dict:
        """دریافت اخبار عمومی - مطابق مستندات صفحه 46"""
        raw_data = self._make_api_request("news")
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, "news")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        # اعمال limit دستی (چون API پارامتر limit ندارد)
        limited_data = normalized_result.data[:limit] if normalized_result.data else []
        
        return {
            "status": "success",
            "result": limited_data,
            "total": len(limited_data),
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_news_by_type(self, news_type: str = "handpicked", limit: int = 10) -> Dict:
        """دریافت اخبار بر اساس نوع - مطابق مستندات صفحه 47"""
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            news_type = "handpicked"
            
        raw_data = self._make_api_request(f"news/type/{news_type}")
        
        # نرمال‌سازی داده‌ها
        normalized_result = self.normalizer.normalize(raw_data, f"news/type/{news_type}")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        # اعمال limit
        limited_data = normalized_result.data[:limit] if normalized_result.data else []
        
        return {
            "status": "success",
            "result": limited_data,
            "type": news_type,
            "total": len(limited_data),
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_news_detail(self, news_id: str) -> Dict:
        """دریافت جزئیات خبر - مطابق مستندات صفحه 48-49"""
        raw_data = self._make_api_request(f"news/{news_id}")
        
        # برای جزئیات خبر، داده را مستقیماً برمی‌گردانیم
        if isinstance(raw_data, dict) and "error" not in raw_data:
            return {
                "status": "success",
                "result": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            normalized_result = self.normalizer.normalize(raw_data, f"news/{news_id}")
            
            if normalized_result.status == "error":
                return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
            
            # اولین آیتم را برمی‌گردانیم
            result_data = normalized_result.data[0] if normalized_result.data else {}
            
            return {
                "status": "success",
                "result": result_data,
                "normalization_info": normalized_result.normalization_info,
                "timestamp": datetime.now().isoformat()
            }

    # ============================= INSIGHTS ENDPOINTS =========================

    def get_btc_dominance(self, period_type: str = "all") -> Dict:
        """دریافت دامیننس بیت کوین - مطابق مستندات صفحه 49-50"""
        params = {"type": period_type}
        raw_data = self._make_api_request("insights/btc-dominance", params)
        
        # برای دامیننس، داده را مستقیماً برمی‌گردانیم
        if isinstance(raw_data, dict) and "error" not in raw_data:
            return {
                "status": "success",
                "result": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "result": {"dominance": raw_data} if not isinstance(raw_data, dict) else raw_data,
                "timestamp": datetime.now().isoformat()
            }

    def get_fear_greed(self) -> Dict:
        """دریافت شاخص ترس و طمع - مطابق مستندات صفحه 50-51"""
        raw_data = self._make_api_request("insights/fear-and-greed")
        
        # برای شاخص ترس و طمع، داده را مستقیماً برمی‌گردانیم
        if isinstance(raw_data, dict) and "error" not in raw_data:
            return {
                "status": "success",
                "result": raw_data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "success",
                "result": {"value": raw_data} if not isinstance(raw_data, dict) else raw_data,
                "timestamp": datetime.now().isoformat()
            }

    def get_fear_greed_chart(self) -> Dict:
        """دریافت چارت ترس و طمع - مطابق مستندات صفحه 51-52"""
        raw_data = self._make_api_request("insights/fear-and-greed/chart")
        
        # نرمال‌سازی داده‌های چارت
        normalized_result = self.normalizer.normalize(raw_data, "fear-greed/chart")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    def get_rainbow_chart(self, coin_id: str = "bitcoin") -> Dict:
        """دریافت چارت رنگین‌کمان - مطابق مستندات صفحه 52-53"""
        raw_data = self._make_api_request(f"insights/rainbow-chart/{coin_id}")
        
        # نرمال‌سازی داده‌های چارت
        normalized_result = self.normalizer.normalize(raw_data, f"rainbow-chart/{coin_id}")
        
        if normalized_result.status == "error":
            return {"error": normalized_result.normalization_info.get("error", "Normalization failed"), "status": "error"}
        
        return {
            "status": "success",
            "result": normalized_result.data,
            "coin_id": coin_id,
            "normalization_info": normalized_result.normalization_info,
            "timestamp": datetime.now().isoformat()
        }

    # ============================= HYBRID DATA METHODS =========================

    def get_coins_list_processed(self, limit: int = 20, page: int = 1, currency: str = "USD",
                               sort_by: str = "rank", sort_dir: str = "asc", **filters) -> Dict:
        """دریافت لیست کوین‌ها به صورت پردازش شده"""
        raw_data = self.get_coins_list(limit, page, currency, sort_by, sort_dir, **filters)
        
        if "error" in raw_data:
            return raw_data
        
        # پردازش اضافی روی داده‌های نرمال‌شده
        processed_coins = []
        for coin in raw_data.get('result', []):
            processed_coins.append({
                'id': coin.get('id'),
                'name': coin.get('name'),
                'symbol': coin.get('symbol'),
                'price': coin.get('price'),
                'price_change_24h': coin.get('priceChange1d'),
                'volume_24h': coin.get('volume'),
                'market_cap': coin.get('marketCap'),
                'rank': coin.get('rank'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_coins,
            'pagination': raw_data.get('meta', {}),
            'normalization_info': raw_data.get('normalization_info', {}),
            'timestamp': datetime.now().isoformat()
        }

    def get_exchanges_processed(self) -> Dict:
        """دریافت لیست صرافی‌ها به صورت پردازش شده"""
        raw_data = self.get_exchanges()
        
        if "error" in raw_data:
            return raw_data
        
        # پردازش اضافی
        processed_exchanges = []
        for exchange in raw_data.get('result', []):
            processed_exchanges.append({
                'id': exchange.get('id'),
                'name': exchange.get('name'),
                'year_established': exchange.get('year_established'),
                'country': exchange.get('country'),
                'trust_score': exchange.get('trust_score'),
                'trade_volume_24h_btc': exchange.get('trade_volume_24h_btc'),
                'url': exchange.get('url'),
                'image': exchange.get('image'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_exchanges,
            'total': len(processed_exchanges),
            'normalization_info': raw_data.get('normalization_info', {}),
            'timestamp': datetime.now().isoformat()
        }

    def get_fear_greed_processed(self) -> Dict:
        """دریافت شاخص ترس و طمع به صورت پردازش شده"""
        raw_data = self.get_fear_greed()
        
        if "error" in raw_data:
            return raw_data
        
        # پردازش و تحلیل پیشرفته
        fear_greed_data = raw_data.get('result', {})
        
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
            'raw_data': raw_data.get('result'),
            'normalization_info': raw_data.get('normalization_info', {}),
            'timestamp': datetime.now().isoformat()
        }

    # ============================= HELPER METHODS =============================

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

    def get_api_status(self) -> Dict[str, Any]:
        """وضعیت API"""
        try:
            test_data = self.get_coins_list(limit=1)
            return {
                'status': 'connected' if test_data and 'result' in test_data else 'disconnected',
                'timestamp': datetime.now().isoformat(),
                'cache_info': self.get_cache_info(),
                'normalization_stats': self.normalizer.get_health_metrics()  # ✅ اضافه شد
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# ایجاد نمونه گلوبال
coin_stats_manager = CompleteCoinStatsManager()

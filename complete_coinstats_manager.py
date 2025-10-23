# complete_coinstats_manager.py - نسخه کامل با تمام اندپوینت‌ها
import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import glob
import time
import logging

# تنظیم لاگینگ
logger = logging.getLogger(__name__)

class CompleteCoinStatsManager:
    def __init__(self, raw_data_path: str = "raw_data", repo_url: str = None):
        self.raw_data_path = raw_data_path
        self.repo_url = repo_url or "https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data"
        self.api_base_url = "https://openapiv1.coinstats.app"
        self.api_key = "oYGllJrdvcdApdgxLTNs9jUnvR/RUGAMhZjt123YtbpA="
        self.headers = {"X-API-KEY": self.api_key}

        # WebSocket configuration
        self.ws_manager = None
        self._initialize_websocket()

        # تابع‌فریم‌های پشتیبانی شده
        self.supported_timeframes = ["1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"]

        # انواع خبر
        self.news_types = ["handpicked", "trending", "latest", "bullish", "bearish"]

        # پوشه‌های موجود در رپو
        self.repo_folders = ["A", "B", "C", "D"]
        
        # آدرس‌های مستقیم GitHub
        self.github_raw_urls = [
            "https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data/A",
            "https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data/B", 
            "https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data/C",
            "https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data/D"
        ]

    def _initialize_websocket(self):
        """راه‌اندازی WebSocket برای داده‌های لحظه‌ای"""
        try:
            from lbank_websocket import LBankWebSocketManager
            self.ws_manager = LBankWebSocketManager()
            self.ws_manager.add_callback(self._on_websocket_data)
            logger.info("✅ WebSocket Manager Initialized")
        except Exception as e:
            logger.error(f"❌ خطا در راه اندازی WebSocket: {e}")

    def _on_websocket_data(self, symbol, data):
        """Callback برای داده‌های WebSocket"""
        try:
            logger.debug(f"📊 WebSocket data received for {symbol}: ${data.get('price', 0)}")
        except Exception as e:
            logger.error(f"❌ خطا در پردازش داده WebSocket: {e}")

    # ========================= متدهای مدیریت داده خام =========================

    def _download_from_github(self, folder: str, filename: str) -> Optional[Dict]:
        """دانلود فایل از GitHub"""
        try:
            raw_url = f"https://raw.githubusercontent.com/hanzo7656-prog/my-dataset/main/raw_data/{folder}/{filename}"
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ فایل {filename} در {folder} یافت نشد")
                return None
        except Exception as e:
            logger.error(f"❌ خطا در دانلود از GitHub: {e}")
            return None

    def _get_github_file_list(self, folder: str) -> List[str]:
        """دریافت لیست فایل‌های موجود در GitHub"""
        common_files = [
            "coins.json", "market_data.json", "news.json", "charts.json",
            "bitcoin_data.json", "ethereum_data.json", "crypto_news.json",
            "price_data.json", "historical_data.json", "technical_indicators.json",
            "coin_details.json", "exchange_data.json", "fiat_data.json",
            "market_insights.json", "fear_greed.json", "rainbow_chart.json"
        ]
        return common_files

    def _load_raw_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های خام از GitHub"""
        raw_data = {}
        total_files_found = 0

        logger.info("🌐 در حال دریافت داده‌ها از GitHub...")

        for folder in self.repo_folders:
            logger.info(f"📁 بررسی پوشه {folder}...")
            file_list = self._get_github_file_list(folder)
            
            for filename in file_list:
                file_data = self._download_from_github(folder, filename)
                if file_data:
                    key = f"{folder}/{filename}"
                    raw_data[key] = {
                        'data': file_data,
                        'source': f'github/{key}',
                        'folder': folder
                    }
                    total_files_found += 1
                    logger.info(f"✅ فایل {filename} از {folder} بارگذاری شد")

        if total_files_found == 0:
            logger.warning("⚠️ هیچ فایلی در GitHub یافت نشد - استفاده از API")
            return self._load_fallback_data()
        
        logger.info(f"✅ تعداد {total_files_found} فایل از GitHub بارگذاری شد")
        return raw_data

    def _load_fallback_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های جایگزین از API"""
        logger.info("🔄 استفاده از داده‌های جایگزین از API...")
        fallback_data = {}
        
        try:
            # دریافت داده‌های اصلی از API
            coins_data = self._make_api_request("coins", {"limit": 100})
            if coins_data:
                fallback_data["api/coins"] = {
                    'data': coins_data,
                    'source': 'coinstats_api',
                    'folder': 'api_fallback'
                }
            
            news_data = self._make_api_request("news", {"limit": 20})
            if news_data:
                fallback_data["api/news"] = {
                    'data': news_data,
                    'source': 'coinstats_api', 
                    'folder': 'api_fallback'
                }
                
            logger.info("✅ داده‌های جایگزین از API بارگذاری شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در دریافت داده‌های جایگزین: {e}")
            
        return fallback_data

    def _find_in_raw_data(self, raw_data: Dict, keywords: List[str]) -> Optional[Any]:
        """جستجوی هوشمند در داده‌های خام"""
        for file_path, file_info in raw_data.items():
            file_data = file_info['data']
            filename = os.path.basename(file_path).lower()

            for keyword in keywords:
                if keyword.lower() in filename:
                    return file_data

        return None

    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """ساخت درخواست به API"""
        url = f"{self.api_base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ خطا در درخواست API به {endpoint}: {e}")
            return {}

    # ========================= اندپوینت‌های اصلی =========================

    def get_coins_list(self, **filters) -> Dict:
        """دریافت لیست کوین‌ها با تمام فیلترهای موجود"""
        raw_data = self._load_raw_data()
        keywords = ["coins", "tokens", "cryptocurrencies", "market_data", "coinlist"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            logger.info("✅ استفاده از داده‌های کوین از منبع محلی")
            return found_data

        logger.info("🔄 دریافت لیست کوین‌ها از API...")
        filters['limit'] = filters.get('limit', 100)
        return self._make_api_request("coins", params=filters)

    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین خاص"""
        raw_data = self._load_raw_data()
        keywords = [f"{coin_id}", "coin_details", "coin_info", "crypto_details"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request(f"coins/{coin_id}", params={"currency": currency})

    # ========================= اندپوینت‌های چارت تاریخی =========================

    def get_coins_charts(self, coin_ids: str, period: str = "all") -> Dict:
        """دریافت چارت برای چند کوین"""
        if period not in self.supported_timeframes:
            period = "all"

        raw_data = self._load_raw_data()
        keywords = [f"charts_{coin_ids}", "multi_coin_charts", "coins_charts", "historical"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("coins/charts",
                                    params={"coinIds": coin_ids, "period": period})

    def get_coin_charts(self, coin_id: str, period: str = "all") -> Dict:
        """دریافت چارت برای یک کوین خاص"""
        if period not in self.supported_timeframes:
            period = "all"

        raw_data = self._load_raw_data()
        keywords = [
            f"{coin_id}_chart",
            f"{coin_id}_historical",
            f"chart_{period}",
            f"{coin_id}_{period}",
            'price_history'
        ]

        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request(f"coins/{coin_id}/charts", params={"period": period})

    def get_all_timeframes_charts(self, coin_id: str) -> Dict:
        """دریافت تمام تایم‌فریم‌های چارت برای یک کوین"""
        all_timeframes_data = {}

        for timeframe in self.supported_timeframes:
            all_timeframes_data[timeframe] = self.get_coin_charts(coin_id, timeframe)

        return all_timeframes_data

    # ========================= اندپوینت‌های قیمت =========================

    def get_coin_price_avg(self, coin_id: str, timestamp: str) -> Dict:
        """دریافت قیمت متوسط کوین در زمان مشخص"""
        raw_data = self._load_raw_data()
        keywords = [f"{coin_id}_price_avg", "historical_price", "price_average", "timestamp_price"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("coins/price/avg",
                                    params={"coinid": coin_id, "timestamp": timestamp})

    def get_exchange_price(self, exchange: str, from_coin: str, to_coin: str, timestamp: str) -> Dict:
        """دریافت قیمت مبادله در صرافی خاص"""
        raw_data = self._load_raw_data()
        keywords = [f"exchange_{exchange}", f"{from_coin}_{to_coin}_price", "trading_pair"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("coins/price/exchange",
                                    params={
                                        "exchange": exchange,
                                        "from": from_coin,
                                        "to": to_coin,
                                        "timestamp": timestamp
                                    })

    # ========================= اندپوینت‌های بازار =========================

    def get_tickers_exchanges(self) -> Dict:
        """دریافت لیست صرافی‌ها"""
        raw_data = self._load_raw_data()
        keywords = ["exchanges", "tickers_exchanges", "crypto_exchanges", "exchange_list"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("tickers/exchanges")

    def get_tickers_markets(self) -> Dict:
        """دریافت لیست بازارها"""
        raw_data = self._load_raw_data()
        keywords = ["markets", "tickers_markets", "trading_markets", "market_list"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("tickers/markets")

    def get_fiats(self) -> Dict:
        """دریافت لیست ارزهای فیات"""
        raw_data = self._load_raw_data()
        keywords = ["fiats", "fiat_currencies", "fiat_list", "currencies_fiat"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("fiats")

    def get_markets(self) -> Dict:
        """دریافت داده‌های بازار"""
        raw_data = self._load_raw_data()
        keywords = ["markets_data", "all_markets", "market_info", "trading_data"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("markets")

    def get_currencies(self) -> Dict:
        """دریافت لیست ارزها"""
        raw_data = self._load_raw_data()
        keywords = ["currencies", "all_currencies", "currency_list", "crypto_currencies"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("currencies")

    # ========================= اندپوینت‌های اخبار =========================

    def get_news_sources(self) -> Dict:
        """دریافت منابع خبری"""
        raw_data = self._load_raw_data()
        keywords = ["news_sources", "news_providers", "content_sources", "sources_list"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("news/sources")

    def get_news(self, limit: int = 20) -> Dict:
        """دریافت اخبار عمومی"""
        raw_data = self._load_raw_data()
        keywords = ["general_news", "crypto_news", "news_feed", "latest_news"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("news", params={"limit": limit})

    def get_news_by_type(self, news_type: str) -> Dict:
        """دریافت اخبار بر اساس نوع - 5 حالت مختلف"""
        if news_type not in self.news_types:
            news_type = "latest"

        raw_data = self._load_raw_data()
        keywords = [f"news_{news_type}", f"{news_type}_news", "filtered_news", "crypto_news"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request(f"news/type/{news_type}")

    def get_news_by_id(self, news_id: str) -> Dict:
        """دریافت خبر خاص بر اساس ID"""
        raw_data = self._load_raw_data()
        keywords = [f"news_{news_id}", "specific_news", "news_detail", "article"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request(f"news/{news_id}")

    def get_all_news_types(self) -> Dict:
        """دریافت تمام نوع اخبار - 5 نوع"""
        all_news_data = {}

        for news_type in self.news_types:
            all_news_data[news_type] = self.get_news_by_type(news_type)

        return all_news_data

    # ========================= اندپوینت‌های بینش بازار =========================

    def get_btc_dominance(self, insight_type: str = "all") -> Dict:
        """دریافت دامیننس بیت‌کوین"""
        raw_data = self._load_raw_data()
        keywords = ["btc_dominance", "dominance", "market_dominance", "bitcoin_dominance"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("insights/btc-dominance", params={"type": insight_type})

    def get_fear_greed_index(self) -> Dict:
        """دریافت شاخص ترس و طمع"""
        raw_data = self._load_raw_data()
        keywords = ["fear_greed", "fear-and-greed", "market_sentiment", "sentiment_index"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("insights/fear-and-greed")

    def get_fear_greed_chart(self) -> Dict:
        """دریافت چارت ترس و طمع"""
        raw_data = self._load_raw_data()
        keywords = ["fear_greed_chart", "sentiment_chart", "market_sentiment_chart"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request("insights/fear-and-greed/chart")

    def get_rainbow_chart(self, coin: str = "bitcoin") -> Dict:
        """دریافت چارت رنگین کمان برای بیت‌کوین و اتریوم"""
        if coin not in ["bitcoin", "ethereum"]:
            coin = "bitcoin"

        raw_data = self._load_raw_data()
        keywords = [f"{coin}_rainbow", "rainbow_chart", "rainbow_data", "technical_analysis"]
        found_data = self._find_in_raw_data(raw_data, keywords)

        if found_data:
            return found_data

        return self._make_api_request(f"insights/rainbow-chart/{coin}")

    def get_all_rainbow_charts(self) -> Dict:
        """دریافت تمام چارت‌های رنگین کمان"""
        return {
            "bitcoin": self.get_rainbow_chart("bitcoin"),
            "ethereum": self.get_rainbow_chart("ethereum")
        }

    # ========================= متدهای داده لحظه‌ای =========================

    def get_realtime_price(self, symbol: str = None) -> Dict:
        """دریافت قیمت لحظه‌ای از WebSocket"""
        if self.ws_manager:
            return self.ws_manager.get_realtime_data(symbol)
        return {}

    def get_websocket_status(self) -> Dict[str, Any]:
        """دریافت وضعیت WebSocket"""
        if self.ws_manager:
            status = self.ws_manager.get_connection_status()
            return {
                'websocket_connected': status['connected'],
                'active_realtime_pairs': status['active_pairs'],
                'total_subscribed': status['total_subscribed'],
                'data_count': status['data_count']
            }
        return {
            'websocket_connected': False,
            'active_realtime_pairs': [],
            'total_subscribed': 0,
            'data_count': 0
        }

    @property
    def ws_connected(self):
        """Property برای وضعیت اتصال"""
        if self.ws_manager:
            return self.ws_manager.connected
        return False

    @property 
    def realtime_data(self):
        """Property برای داده‌های لحظه‌ای"""
        if self.ws_manager:
            return self.ws_manager.realtime_data
        return {}

    # ========================= متدهای کمکی و جمع‌آوری داده =========================

    def get_all_coins(self, limit: int = 150) -> List[Dict]:
        """دریافت لیست کامل کوین‌ها"""
        coins_data = self.get_coins_list(limit=limit)
        if coins_data and 'result' in coins_data:
            return coins_data['result']
        return []

    def get_top_coins(self, count: int = 10) -> List[Dict]:
        """دریافت برترین کوین‌ها"""
        all_coins = self.get_all_coins(count)
        return all_coins[:count] if all_coins else []

    def collect_comprehensive_data(self) -> Dict[str, Any]:
        """جمع‌آوری جامع تمام داده ها از تمام منابع"""
        comprehensive_data = {
            "timestamp": datetime.now().isoformat(),
            "data_source": "complete_hybrid_system",
            "repo_url": self.repo_url,
            "raw_data_available": False,
            "websocket_status": self.get_websocket_status(),
            "collected_data": {}
        }
        
        # بارگذاری داده‌های خام
        raw_data = self._load_raw_data()
        if raw_data:
            comprehensive_data["raw_data_available"] = True
            comprehensive_data["raw_files_count"] = len(raw_data)

        # 1. داده‌های لحظه‌ای WebSocket
        ws_data = self.get_realtime_price()
        comprehensive_data["collected_data"]['realtime'] = {
            "websocket_data": ws_data,
            "major_prices": {
                'BTC': self.get_realtime_price('btc_usdt'),
                'ETH': self.get_realtime_price('eth_usdt'),
                'SOL': self.get_realtime_price('sol_usdt')
            }
        }

        # 2. داده‌های اصلی کوین‌ها
        comprehensive_data["collected_data"]['coins'] = {
            "list": self.get_coins_list(limit=100),
            "major_coins": {
                "bitcoin": self.get_coin_details("bitcoin"),
                "ethereum": self.get_coin_details("ethereum"),
                "solana": self.get_coin_details("solana")
            }
        }

        # 3. چارت‌های تاریخی برای تمام تایم‌فریم‌ها
        comprehensive_data["collected_data"]["historical_charts"] = {}
        major_coins = ["bitcoin", "ethereum"]
        for coin in major_coins:
            comprehensive_data["collected_data"]["historical_charts"][coin] = self.get_all_timeframes_charts(coin)

        # 4. چارت‌های چندکوینه
        comprehensive_data["collected_data"]["multi_coin_charts"] = self.get_coins_charts("bitcoin,ethereum,solana", "all")

        # 5. داده‌های قیمت
        comprehensive_data["collected_data"]["price_data"] = {
            "bitcoin_avg": self.get_coin_price_avg("bitcoin", "1636315200"),
            "exchange_rate": self.get_exchange_price("Binance", "BTC", "ETH", "1636315200")
        }

        # 6. داده‌های بازار
        comprehensive_data["collected_data"]["market_data"] = {
            "exchanges": self.get_tickers_exchanges(),
            "markets": self.get_tickers_markets(),
            "fiats": self.get_fiats(),
            "all_markets": self.get_markets(),
            "currencies": self.get_currencies()
        }

        # 7. اخبار - 5 نوع
        comprehensive_data["collected_data"]["news"] = {
            "sources": self.get_news_sources(),
            "general": self.get_news(limit=10),
            "by_type": self.get_all_news_types(),
            "sample_news": self.get_news_by_id("376f390df50a1d44cb5593c9bff6faafabed18ee90e0d4d737d3b6d3eea50c80")
        }

        # 8. بینش بازار
        comprehensive_data["collected_data"]["market_insights"] = {
            "btc_dominance": self.get_btc_dominance("all"),
            "fear_greed": {
                "index": self.get_fear_greed_index(),
                "chart": self.get_fear_greed_chart()
            },
            "rainbow_charts": self.get_all_rainbow_charts()
        }

        return comprehensive_data

    def test_connections(self) -> Dict[str, Any]:
        """تست تمام اتصالات"""
        results = {
            'github_access': False,
            'api_access': False,
            'websocket_connected': False,
            'total_coins': 0,
            'total_news': 0,
            'all_endpoints_tested': {}
        }

        # تست GitHub
        try:
            test_data = self._load_raw_data()
            results['github_access'] = len(test_data) > 0
            results['github_files'] = len(test_data)
        except Exception as e:
            logger.error(f"❌ تست GitHub failed: {e}")

        # تست API Endpoints
        endpoints_to_test = [
            ('coins', self.get_coins_list(limit=5)),
            ('news', self.get_news(limit=2)),
            ('exchanges', self.get_tickers_exchanges()),
            ('markets', self.get_markets()),
            ('fiats', self.get_fiats()),
            ('fear_greed', self.get_fear_greed_index()),
            ('btc_dominance', self.get_btc_dominance())
        ]

        for endpoint_name, endpoint_result in endpoints_to_test:
            try:
                results['all_endpoints_tested'][endpoint_name] = bool(endpoint_result)
            except Exception as e:
                results['all_endpoints_tested'][endpoint_name] = False

        # تست WebSocket
        results['websocket_connected'] = self.ws_connected

        # تست تعداد کوین‌ها
        try:
            results['total_coins'] = len(self.get_all_coins(150))
        except Exception as e:
            logger.error(f"❌ تست تعداد کوین‌ها failed: {e}")

        return results

# نمونه استفاده
if __name__ == "__main__":
    manager = CompleteCoinStatsManager()
    
    print("🧪 تست تمام اندپوینت‌ها...")
    results = manager.test_connections()
    
    print("\n📊 نتایج تست:")
    for key, value in results.items():
        if key != 'all_endpoints_tested':
            print(f"  {key}: {value}")
    
    print("\n🔧 وضعیت اندپوینت‌ها:")
    for endpoint, status in results['all_endpoints_tested'].items():
        print(f"  {endpoint}: {'✅' if status else '❌'}")
    
    print(f"\n💰 تعداد کل کوین‌ها: {results['total_coins']}")
    print(f"📡 وضعیت WebSocket: {'✅ متصل' if results['websocket_connected'] else '❌ قطع'}")

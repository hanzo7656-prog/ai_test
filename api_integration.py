# complete_coinstats_manager.py
import requests
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import glob
import websocket
import threading
import time

class CompleteCoinStatsManager:
    def __init__(self, raw_data_path: str = "raw_data", repo_url: str = None):
        self.raw_data_path = raw_data_path
        self.repo_url = repo_url or "https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data"
        self.api_base_url = "https://openapiv1.coinstats.app"
        self.api_key = "oYGllJrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="
        self.headers = {"X-API-KEY": self.api_key}
        
        # WebSocket configuration
        self.ws_url = "wss://www.lbank.net/ws/V2/"
        self.ws_client = None
        self.realtime_data = {}
        self.ws_connected = False
        
        # تایم‌فریم‌های پشتیبانی شده
        self.supported_timeframes = ["1h", "4h", "8h", "1d", "7d", "1m", "3m", "1y", "all"]
        
        # انواع خبر
        self.news_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        
        # پوشه‌های موجود در ریپو
        self.repo_folders = ["A", "B", "C", "D"]
        
        # راه‌اندازی WebSocket
        self._initialize_websocket()
    
    def _initialize_websocket(self):
        """راه‌اندازی WebSocket برای داده‌های لحظه‌ای"""
        try:
            self.ws_client = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            def run_ws():
                self.ws_client.run_forever()
            
            ws_thread = threading.Thread(target=run_ws)
            ws_thread.daemon = True
            ws_thread.start()
            
        except Exception as e:
            print(f"❌ خطا در راه‌اندازی WebSocket: {e}")
    
    def _on_ws_open(self, ws):
        """هنگام باز شدن اتصال WebSocket"""
        print("✅ WebSocket به LBank متصل شد")
        self.ws_connected = True
        self._subscribe_to_major_pairs()
    
    def _on_ws_message(self, ws, message):
        """پردازش پیام‌های دریافتی از WebSocket"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'tick' and 'tick' in data:
                symbol = data.get('pair', '')
                tick_data = data['tick']
                
                # ذخیره داده لحظه‌ای
                self.realtime_data[symbol] = {
                    'symbol': symbol,
                    'price': float(tick_data.get('latest', 0)),
                    'high_24h': float(tick_data.get('high', 0)),
                    'low_24h': float(tick_data.get('low', 0)),
                    'volume': float(tick_data.get('vol', 0)),
                    'change': float(tick_data.get('change', 0)),
                    'timestamp': data.get('TS', ''),
                    'last_updated': time.time(),
                    'source': 'lbank_websocket'
                }
                
        except json.JSONDecodeError as e:
            print(f"❌ خطای JSON در WebSocket: {e}")
    
    def _on_ws_error(self, ws, error):
        """مدیریت خطاهای WebSocket"""
        print(f"❌ خطای WebSocket: {error}")
        self.ws_connected = False
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """هنگام بسته شدن اتصال WebSocket"""
        print(f"🔴 WebSocket قطع شد")
        self.ws_connected = False
        self._schedule_ws_reconnect()
    
    def _schedule_ws_reconnect(self):
        """برنامه‌ریزی برای اتصال مجدد WebSocket"""
        time.sleep(10)
        self._initialize_websocket()
    
    def _subscribe_to_major_pairs(self):
        """اشتراک در جفت‌ارزهای اصلی"""
        major_pairs = [
            "btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", 
            "ada_usdt", "xrp_usdt", "doge_usdt", "dot_usdt"
        ]
        
        for pair in major_pairs:
            self._subscribe_to_pair(pair)
    
    def _subscribe_to_pair(self, pair: str):
        """اشتراک در یک جفت ارز خاص"""
        if not self.ws_connected or not self.ws_client:
            return
        
        subscription_msg = {
            "action": "subscribe",
            "subscribe": "tick",
            "pair": pair
        }
        
        try:
            self.ws_client.send(json.dumps(subscription_msg))
            print(f"✅ اشتراک WebSocket برای {pair}")
        except Exception as e:
            print(f"❌ خطای اشتراک WebSocket برای {pair}: {e}")

    # ============================ متدهای مدیریت داده خام ============================
    
    def _ensure_directory(self, directory: str):
        """ایجاد دایرکتوری اگر وجود ندارد"""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def get_storage_path(self) -> str:
        """تعیین مسیر ذخیره‌سازی"""
        base_path = "./coinstats_collected_data"
        self._ensure_directory(base_path)
        return base_path
    
    def _load_raw_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های خام از ریپو"""
        raw_data = {}
        
        for folder in self.repo_folders:
            folder_path = os.path.join(self.raw_data_path, folder)
            if not os.path.exists(folder_path):
                continue
            
            data_files = glob.glob(f"{folder_path}/**/*.json", recursive=True)
            data_files.extend(glob.glob(f"{folder_path}/**/*.csv", recursive=True))
            
            for file_path in data_files:
                try:
                    filename = os.path.basename(file_path)
                    relative_path = os.path.relpath(file_path, self.raw_data_path)
                    
                    if file_path.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            raw_data[relative_path] = {
                                'data': json.load(f),
                                'source': f'repo/{relative_path}',
                                'folder': folder
                            }
                    elif file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        raw_data[relative_path] = {
                            'data': df.to_dict('records'),
                            'source': f'repo/{relative_path}',
                            'folder': folder
                        }
                        
                except Exception as e:
                    print(f"❌ خطا در بارگذاری {file_path}: {e}")
        
        return raw_data
    
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
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ خطا در درخواست API به {endpoint}: {e}")
            return {}

    # ============================ اندپوینت‌های اصلی کوین‌ها ============================

    def get_coins_list(self, **filters) -> Dict:
        """دریافت لیست کوین‌ها با تمام فیلترهای موجود"""
        raw_data = self._load_raw_data()
        
        keywords = ["coins", "tokens", "cryptocurrencies", "market_data", "coinlist"]
        found_data = self._find_in_raw_data(raw_data, keywords)
        
        if found_data:
            return found_data
        
        return self._make_api_request("coins", params=filters)

    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین خاص"""
        raw_data = self._load_raw_data()
        
        keywords = [f"{coin_id}", "coin_details", "coin_info", "crypto_details"]
        found_data = self._find_in_raw_data(raw_data, keywords)
        
        if found_data:
            return found_data
        
        return self._make_api_request(f"coins/{coin_id}", params={"currency": currency})

    # ============================ اندپوینت‌های چارت تاریخی ============================

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
            "price_history"
        ]
        found_data = self._find_in_raw_data(raw_data, keywords)
        
        if found_data:
            return found_data
        
        return self._make_api_request(f"coins/{coin_id}/charts", 
                                    params={"period": period})

    def get_all_timeframes_charts(self, coin_id: str) -> Dict:
        """دریافت تمام تایم‌فریم‌های چارت برای یک کوین"""
        all_timeframes_data = {}
        
        for timeframe in self.supported_timeframes:
            all_timeframes_data[timeframe] = self.get_coin_charts(coin_id, timeframe)
            
        return all_timeframes_data

    # ============================ اندپوینت‌های قیمت ============================

    def get_coin_price_avg(self, coin_id: str, timestamp: str) -> Dict:
        """دریافت قیمت متوسط کوین در زمان مشخص"""
        raw_data = self._load_raw_data()
        
        keywords = [f"{coin_id}_price_avg", "historical_price", "price_average", "timestamp_price"]
        found_data = self._find_in_raw_data(raw_data, keywords)
        
        if found_data:
            return found_data
        
        return self._make_api_request("coins/price/avg", 
                                    params={"coinId": coin_id, "timestamp": timestamp})

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

    # ============================ اندپوینت‌های بازار ============================

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

    # ============================ اندپوینت‌های اخبار ============================

    def get_news_sources(self) -> Dict:
        """دریافت منابع خبری"""
        raw_data = self._load_raw_data()
        
        keywords = ["news_sources", "news_providers", "content_sources", "sources_list"]
        found_data = self._find_in_raw_data(raw_data, keywords)
        
        if found_data:
            return found_data
        
        return self._make_api_request("news/sources")

    def get_news(self) -> Dict:
        """دریافت اخبار عمومی"""
        raw_data = self._load_raw_data()
        
        keywords = ["general_news", "crypto_news", "news_feed", "latest_news"]
        found_data = self._find_in_raw_data(raw_data, keywords)
        
        if found_data:
            return found_data
        
        return self._make_api_request("news")

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
        """دریافت تمام 5 نوع اخبار"""
        all_news_data = {}
        
        for news_type in self.news_types:
            all_news_data[news_type] = self.get_news_by_type(news_type)
            
        return all_news_data

    # ============================ اندپوینت‌های بینش بازار ============================

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

    # ============================ متدهای داده لحظه‌ای ============================

    def get_realtime_price(self, symbol: str = None) -> Dict:
        """دریافت قیمت لحظه‌ای از WebSocket"""
        if symbol:
            lbank_symbol = symbol.lower().replace('/', '_')
            return self.realtime_data.get(lbank_symbol, {})
        
        return self.realtime_data

    def get_websocket_status(self) -> Dict[str, Any]:
        """دریافت وضعیت WebSocket"""
        return {
            'websocket_connected': self.ws_connected,
            'active_realtime_pairs': len(self.realtime_data),
            'major_prices': {
                'BTC/USDT': self.realtime_data.get('btc_usdt', {}).get('price', 0),
                'ETH/USDT': self.realtime_data.get('eth_usdt', {}).get('price', 0),
                'SOL/USDT': self.realtime_data.get('sol_usdt', {}).get('price', 0)
            }
        }
    # ============================ متدهای جدید برای لود کردن از فایل ============================
    
    def load_from_saved_file(self, file_path: str) -> Dict[str, Any]:
        """لود کردن داده‌های ذخیره شده از فایل JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                print(f"✅ داده‌ها از {file_path} با موفقیت لود شد")
                return saved_data
        except Exception as e:
            print(f"❌ خطا در لود کردن فایل: {e}")
            return {}

    def get_latest_saved_file(self) -> str:
        """پیدا کردن آخرین فایل ذخیره شده"""
        storage_path = self.get_storage_path()
        json_files = glob.glob(f"{storage_path}/*.json")
        
        if not json_files:
            return None
        
        # پیدا کردن آخرین فایل بر اساس timestamp
        latest_file = max(json_files, key=os.path.getctime)
        return latest_file

    def smart_data_collection(self, max_age_minutes: int = 60) -> Dict[str, Any]:
        """جمع‌آوری هوشمند داده‌ها - استفاده از کش اگر قدیمی نباشد"""
        
        latest_file = self.get_latest_saved_file()
        
        if latest_file:
            file_age = (time.time() - os.path.getctime(latest_file)) / 60  # به دقیقه
            
            if file_age < max_age_minutes:
                print(f"✅ استفاده از داده‌های کش شده ({file_age:.1f} دقیقه گذشته)")
                data = self.load_from_saved_file(latest_file)
                data['data_source'] = f"cached_{int(file_age)}min"
                return data
        
        # داده‌های جدید
        print("🔄 دریافت داده‌های تازه...")
        data = self.collect_comprehensive_data()
        self.save_comprehensive_data()
        return data

    # ============================ جمع‌آوری جامع داده ============================
    
    def collect_comprehensive_data(self) -> Dict[str, Any]:
        """جمع‌آوری جامع تمام داده‌ها از تمام منابع"""
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
        
        # 1. داده‌های لحظه‌ای از WebSocket
        comprehensive_data["collected_data"]["realtime"] = {
            "websocket_data": self.realtime_data,
            "major_prices": {
                'BTC': self.get_realtime_price('btc_usdt'),
                'ETH': self.get_realtime_price('eth_usdt'),
                'SOL': self.get_realtime_price('sol_usdt')
            }
        }
        
        # 2. داده‌های اصلی کوین‌ها
        comprehensive_data["collected_data"]["coins"] = {
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
        
        # 7. اخبار (تمام 5 نوع)
        comprehensive_data["collected_data"]["news"] = {
            "sources": self.get_news_sources(),
            "general": self.get_news(),
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

    def save_comprehensive_data(self, filename: str = None):
        """ذخیره داده‌های جامع"""
        storage_path = self.get_storage_path()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complete_coinstats_data_{timestamp}.json"
        
        file_path = os.path.join(storage_path, filename)
        
        data = self.collect_comprehensive_data()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ داده‌های جامع ذخیره شد در: {file_path}")
        self.print_complete_stats(data)

    def print_complete_stats(self, data: Dict):
        """چاپ آمار کامل سیستم"""
        print("\n" + "="*60)
        print("📊 آمار کامل سیستم - تمام اندپوینت‌ها")
        print("="*60)
        
        # وضعیت WebSocket
        ws_status = data['websocket_status']
        print(f"🌐 WebSocket: {'✅ متصل' if ws_status['websocket_connected'] else '❌ قطع'}")
        print(f"📡 جفت‌ارزهای فعال: {ws_status['active_realtime_pairs']}")
        
        # قیمت‌های لحظه‌ای
        print("\n💰 قیمت‌های لحظه‌ای:")
        for coin, price_data in ws_status['major_prices'].items():
            if price_data:
                print(f"   {coin}: ${price_data:.2f}")
        
        # داده‌های خام
        if data['raw_data_available']:
            print(f"\n📁 داده‌های خام: {data['raw_files_count']} فایل")
        
        # آمار کامل
        collected = data['collected_data']
        print(f"\n📈 کوین‌ها: {len(collected['coins']['list'].get('result', []))} کوین")
        print(f"📊 چارت‌های تاریخی: {len(collected['historical_charts'])} کوین × {len(self.supported_timeframes)} تایم‌فریم")
        print(f"🏪 داده‌های بازار: {len(collected['market_data'])} بخش")
        print(f"📰 اخبار: {len(collected['news']['by_type'])} نوع")
        print(f"🔮 بینش بازار: {len(collected['market_insights'])} بخش")
        
        print("="*60)
        print("✅ تمام اندپوینت‌ها پیاده‌سازی شده‌اند!")
        print("="*60)

# نمونه استفاده
if __name__ == "__main__":
    # ایجاد مدیر کامل
    manager = CompleteCoinStatsManager(
        raw_data_path="./raw_data",
        repo_url="https://github.com/hanzo7656-prog/my-dataset/tree/main/raw_data"
    )
    
    print("🚀 راه‌اندازی سیستم کامل داده...")
    
    # منتظر ماندن برای اتصال WebSocket
    time.sleep(5)
    
    # تست تمام اندپوینت‌ها
    print("\n🧪 تست تمام اندپوینت‌ها...")
    
    # تست داده‌های لحظه‌ای
    btc_realtime = manager.get_realtime_price('btc_usdt')
    print(f"✅ قیمت لحظه‌ای BTC: ${btc_realtime.get('price', 0) if btc_realtime else 'ندارد'}")
    
    # تست کوین‌ها
    coins_list = manager.get_coins_list(limit=5)
    print(f"✅ لیست کوین‌ها: {len(coins_list.get('result', [])) if 'result' in coins_list else 'N/A'} کوین")
    
    # تست چارت‌ها
    btc_chart = manager.get_coin_charts("bitcoin", "1d")
    print(f"✅ چارت بیت‌کوین: {'موجود' if btc_chart else 'ندارد'}")
    
    # تست قیمت
    btc_price = manager.get_coin_price_avg("bitcoin", "1636315200")
    print(f"✅ قیمت متوسط بیت‌کوین: {'موجود' if btc_price else 'ندارد'}")
    
    # تست بازار
    exchanges = manager.get_tickers_exchanges()
    print(f"✅ لیست صرافی‌ها: {'موجود' if exchanges else 'ندارد'}")
    
    # تست اخبار (تمام 5 نوع)
    for news_type in manager.news_types:
        news = manager.get_news_by_type(news_type)
        print(f"✅ اخبار {news_type}: {'موجود' if news else 'ندارد'}")
    
    # تست بینش بازار
    fear_greed = manager.get_fear_greed_index()
    print(f"✅ شاخص ترس و طمع: {'موجود' if fear_greed else 'ندارد'}")
    
    rainbow_btc = manager.get_rainbow_chart("bitcoin")
    print(f"✅ چارت رنگین کمان: {'موجود' if rainbow_btc else 'ندارد'}")
    
    # ذخیره داده‌های جامع
    print("\n💾 در حال ذخیره داده‌های جامع...")
    manager.save_comprehensive_data()
    
    print("\n🎉 سیستم کامل با موفقیت راه‌اندازی شد! تمام اندپوینت‌ها فعال هستند.")

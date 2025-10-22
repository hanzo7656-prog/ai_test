# api_integration.py
import requests
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import glob

class DataManager:
    def __init__(self, raw_data_path: str = "raw_data"):
        self.raw_data_path = raw_data_path
        self.api_base_url = "https://openapiv1.coinstats.app"
        self.api_key = "oYGllJrdvcdApdgxLTNs9jUnvR/RUGAMhZjt1Z3YtbpA="
        self.headers = {"X-API-KEY": self.api_key}
    
    def _load_raw_data(self) -> Dict[str, Any]:
        """بارگذاری داده‌های خام از ریپو"""
        raw_data = {}
        
        # جستجو برای فایل‌های داده خام
        data_files = glob.glob(f"{self.raw_data_path}/**/*.json", recursive=True)
        data_files.extend(glob.glob(f"{self.raw_data_path}/**/*.csv", recursive=True))
        
        for file_path in data_files:
            try:
                filename = os.path.basename(file_path)
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_data[filename] = json.load(f)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    raw_data[filename] = df.to_dict('records')
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return raw_data
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """ساخت درخواست به API در صورت عدم وجود داده خام"""
        url = f"{self.api_base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in API request to {endpoint}: {e}")
            return {}
    
    def get_historical_data(self, coin_id: str, timeframe: str = "all") -> Dict:
        """
        دریافت داده‌های تاریخی با پوشش کامل تایم‌فریم‌ها
        
        timeframe های پشتیبانی شده:
        - 1h, 4h, 8h, 1d, 7d, 1m, 3m, 1y, all
        """
        # اول از داده‌های خام استفاده می‌کنیم
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام برای اطلاعات تاریخی
        historical_keywords = [f"{coin_id}_historical", f"{coin_id}_chart", "historical", "price_history"]
        
        for keyword in historical_keywords:
            for filename, data in raw_data.items():
                if keyword in filename.lower():
                    print(f"Found historical data in raw files: {filename}")
                    return data
        
        # اگر در داده‌های خام پیدا نشد، از API استفاده می‌کنیم
        print(f"Using API for {coin_id} historical data (timeframe: {timeframe})")
        
        # برای چارت کلی کوین‌ها
        if "coinIds" in coin_id or "," in coin_id:
            return self._make_api_request("coins/charts", 
                                        params={"coinIds": coin_id, "period": timeframe})
        # برای چارت کوین خاص
        else:
            return self._make_api_request(f"coins/{coin_id}/charts", 
                                        params={"period": timeframe})
    
    def get_coins_list(self, **filters) -> Dict:
        """دریافت لیست کوین‌ها با فیلترهای مختلف"""
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام برای لیست کوین‌ها
        coin_list_keywords = ["coins", "tokens", "cryptocurrencies", "market_data"]
        
        for keyword in coin_list_keywords:
            for filename, data in raw_data.items():
                if keyword in filename.lower():
                    print(f"Found coins data in raw files: {filename}")
                    # اعمال فیلترها روی داده‌های خام
                    return self._apply_filters_to_data(data, filters)
        
        # استفاده از API
        print("Using API for coins list")
        return self._make_api_request("coins", params=filters)
    
    def _apply_filters_to_data(self, data: Any, filters: Dict) -> Any:
        """اعمال فیلترها روی داده‌های خام"""
        if not filters or not isinstance(data, list):
            return data
        
        filtered_data = data
        
        # فیلتر بر اساس نام
        if 'name' in filters:
            filtered_data = [item for item in filtered_data 
                           if filters['name'].lower() in item.get('name', '').lower()]
        
        # فیلتر بر اساس سیمبول
        if 'symbol' in filters:
            filtered_data = [item for item in filtered_data 
                           if item.get('symbol', '').upper() == filters['symbol'].upper()]
        
        # فیلتر بر اساس رنج قیمت
        if 'price~greaterThan' in filters:
            filtered_data = [item for item in filtered_data 
                           if item.get('price', 0) > float(filters['price~greaterThan'])]
        
        if 'price~lessThan' in filters:
            filtered_data = [item for item in filtered_data 
                           if item.get('price', 0) < float(filters['price~lessThan'])]
        
        return filtered_data
    
    def get_coin_details(self, coin_id: str, currency: str = "USD") -> Dict:
        """دریافت جزئیات کوین خاص"""
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام
        for filename, data in raw_data.items():
            if coin_id.lower() in filename.lower():
                print(f"Found {coin_id} details in raw files: {filename}")
                return data
        
        # استفاده از API
        print(f"Using API for {coin_id} details")
        return self._make_api_request(f"coins/{coin_id}", params={"currency": currency})
    
    def get_rainbow_chart(self, coin: str = "bitcoin") -> Dict:
        """دریافت چارت رنگین کمان برای بیت‌کوین و اتریوم"""
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام
        rainbow_keywords = [f"{coin}_rainbow", "rainbow_chart", "rainbow_data"]
        
        for keyword in rainbow_keywords:
            for filename, data in raw_data.items():
                if keyword in filename.lower():
                    print(f"Found rainbow chart data in raw files: {filename}")
                    return data
        
        # استفاده از API
        print(f"Using API for {coin} rainbow chart")
        return self._make_api_request(f"insights/rainbow-chart/{coin}")
    
    def get_fear_greed_data(self) -> Dict:
        """دریافت داده‌های شاخص ترس و طمع"""
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام
        fear_greed_keywords = ["fear_greed", "fear-and-greed", "market_sentiment"]
        
        for keyword in fear_greed_keywords:
            for filename, data in raw_data.items():
                if keyword in filename.lower():
                    print(f"Found fear-greed data in raw files: {filename}")
                    return data
        
        # استفاده از API
        print("Using API for fear-greed index")
        return self._make_api_request("insights/fear-and-greed")
    
    def get_btc_dominance(self, insight_type: str = "all") -> Dict:
        """دریافت دامیننس بیت‌کوین"""
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام
        dominance_keywords = ["btc_dominance", "dominance", "market_dominance"]
        
        for keyword in dominance_keywords:
            for filename, data in raw_data.items():
                if keyword in filename.lower():
                    print(f"Found BTC dominance data in raw files: {filename}")
                    return data
        
        # استفاده از API
        print("Using API for BTC dominance")
        return self._make_api_request("insights/btc-dominance", params={"type": insight_type})
    
    def get_news(self, news_type: str = "latest") -> Dict:
        """دریافت اخبار بر اساس نوع"""
        valid_types = ["handpicked", "trending", "latest", "bullish", "bearish"]
        if news_type not in valid_types:
            news_type = "latest"
        
        raw_data = self._load_raw_data()
        
        # جستجو در داده‌های خام
        news_keywords = [f"news_{news_type}", "crypto_news", "market_news"]
        
        for keyword in news_keywords:
            for filename, data in raw_data.items():
                if keyword in filename.lower():
                    print(f"Found {news_type} news in raw files: {filename}")
                    return data
        
        # استفاده از API
        print(f"Using API for {news_type} news")
        return self._make_api_request(f"news/type/{news_type}")
    
    def collect_comprehensive_data(self) -> Dict[str, Any]:
        """جمع‌آوری جامع تمام داده‌های موجود"""
        comprehensive_data = {
            "timestamp": datetime.now().isoformat(),
            "data_source": "hybrid",  # ترکیبی از داده خام و API
            "raw_data_available": False,
            "collected_data": {}
        }
        
        # بارگذاری داده‌های خام
        raw_data = self._load_raw_data()
        if raw_data:
            comprehensive_data["raw_data_available"] = True
            comprehensive_data["raw_files"] = list(raw_data.keys())
            comprehensive_data["collected_data"]["raw_data"] = raw_data
        
        # کوین‌های اصلی برای جمع‌آوری داده
        major_coins = ["bitcoin", "ethereum", "solana", "binance-coin", "cardano", "ripple"]
        
        # جمع‌آوری داده‌های تاریخی برای تمام تایم‌فریم‌ها
        timeframes = ["1h", "4h", "1d", "7d", "1m", "3m", "1y", "all"]
        
        comprehensive_data["collected_data"]["historical"] = {}
        for coin in major_coins:
            comprehensive_data["collected_data"]["historical"][coin] = {}
            for timeframe in timeframes:
                comprehensive_data["collected_data"]["historical"][coin][timeframe] = self.get_historical_data(coin, timeframe)
        
        # داده‌های چارت رنگین کمان برای بیت‌کوین و اتریوم
        comprehensive_data["collected_data"]["rainbow_charts"] = {
            "bitcoin": self.get_rainbow_chart("bitcoin"),
            "ethereum": self.get_rainbow_chart("ethereum")
        }
        
        # داده‌های بازار و احساسات
        comprehensive_data["collected_data"]["market_insights"] = {
            "fear_greed": self.get_fear_greed_data(),
            "btc_dominance": self.get_btc_dominance(),
            "fear_greed_chart": self._make_api_request("insights/fear-and-greed/chart")
        }
        
        # اخبار
        comprehensive_data["collected_data"]["news"] = {}
        for news_type in ["handpicked", "trending", "latest", "bullish", "bearish"]:
            comprehensive_data["collected_data"]["news"][news_type] = self.get_news(news_type)
        
        # لیست کوین‌ها
        comprehensive_data["collected_data"]["coins_list"] = self.get_coins_list(limit=50)
        
        return comprehensive_data
    
    def save_consolidated_data(self, filename: str = "consolidated_crypto_data.json"):
        """ذخیره داده‌های تلفیقی شده"""
        data = self.collect_comprehensive_data()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Consolidated data saved to {filename}")
        print(f"📊 Data includes:")
        print(f"   - Historical data for {len(data['collected_data']['historical'])} coins")
        print(f"   - {len(data['collected_data']['historical']['bitcoin'])} timeframes per coin")
        print(f"   - Rainbow charts for BTC & ETH")
        print(f"   - Market insights and news")
        if data['raw_data_available']:
            print(f"   - Raw data from {len(data['raw_files'])} files")

# نمونه استفاده
if __name__ == "__main__":
    # استفاده با مسیر داده‌های خام
    data_manager = DataManager(raw_data_path="./raw_data")
    
    # تست عملکرد
    print("🧪 Testing data collection...")
    
    # تست داده‌های تاریخی با تایم‌فریم‌های مختلف
    btc_daily = data_manager.get_historical_data("bitcoin", "1d")
    print(f"BTC Daily data samples: {len(btc_daily) if isinstance(btc_daily, list) else 'N/A'}")
    
    # تست چارت رنگین کمان
    btc_rainbow = data_manager.get_rainbow_chart("bitcoin")
    eth_rainbow = data_manager.get_rainbow_chart("ethereum")
    print(f"BTC Rainbow data: {bool(btc_rainbow)}")
    print(f"ETH Rainbow data: {bool(eth_rainbow)}")
    
    # ذخیره داده‌های تلفیقی
    data_manager.save_consolidated_data()

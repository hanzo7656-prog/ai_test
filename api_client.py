import requests
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

class VortexAPIClient:
    """
    کلاس جامع برای ارتباط با سرور میانی VortexAI
    مدیریت تمام اندپوینت‌های مورد نیاز هوش مصنوعی
    """
    
    def __init__(self, base_url: str, timeout: int = 15):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # هدرهای پیش‌فرض
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'VortexAI-Client/2.0',
            'Accept': 'application/json'
        })
        
        # لاگ آخرین درخواست
        self.last_request_time = None
        self.request_count = 0
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """ساختار پایه برای درخواست‌های API"""
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            print(f"🌐 درحال دریافت از: {endpoint}")
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ دریافت موفق از {endpoint}")
                return data
            else:
                print(f"❌ خطا در {endpoint}: کد {response.status_code} - {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ اتصال به {endpoint} timeout خورد")
            return None
        except requests.exceptions.ConnectionError:
            print(f"🔌 خطای اتصال به {endpoint}")
            return None
        except Exception as e:
            print(f"🚨 خطای ناشناخته در {endpoint}: {str(e)}")
            return None
    
    # ========== 📊 داده‌های بازار و ارزها ==========
    
    def get_currencies(self) -> Optional[Dict]:
        """دریافت لیست ارزهای فیات و رمزارزها"""
        return self._make_request("/currencies")
    
    def get_market_cap(self) -> Optional[Dict]:
        """دریافت اطلاعات مارکت کپ کلی بازار"""
        return self._make_request("/markets/cap")
    
    # ========== 📈 داده‌های تاریخی و تایم‌فریم ==========
    
    def get_historical_data(self, symbol: str, timeframe: str = "24h") -> Optional[Dict]:
        """دریافت داده‌های تاریخی یک ارز"""
        return self._make_request(f"/coin/{symbol}/history/{timeframe}")
    
    def get_timeframes(self) -> Optional[Dict]:
        """دریافت تایم‌فریم‌های موجود"""
        return self._make_request("/timeframes-api")
    
    # ========== 🧠 insights و تحلیل‌های احساسی ==========
    
    def get_insights_dashboard(self) -> Optional[Dict]:
        """دریافت کل داشبورد insights (رینبو، دامیننس، ترس و طمع)"""
        return self._make_request("/insights/dashboard")
    
    def get_btc_dominance(self, dominance_type: str = "all") -> Optional[Dict]:
        """دریافت دامیننس بیت‌کوین"""
        params = {"type": dominance_type} if dominance_type != "all" else {}
        return self._make_request("/insights/btc-dominance", params)
    
    def get_fear_greed(self) -> Optional[Dict]:
        """دریافت شاخص ترس و طمع"""
        return self._make_request("/insights/fear-greed")
    
    def get_fear_greed_chart(self) -> Optional[Dict]:
        """دریافت نمودار تاریخی ترس و طمع"""
        return self._make_request("/insights/fear-greed/chart")
    
    def get_rainbow_chart(self, symbol: str = "bitcoin") -> Optional[Dict]:
        """دریافت نمودار رینبو (از طریق insights)"""
        insights = self.get_insights_dashboard()
        if insights and 'data' in insights:
            return insights['data'].get('rainbow_chart')
        return None
    
    # ========== 📰 اخبار و منابع ==========
    
    def get_news(self, page: int = 1, limit: int = 20) -> Optional[Dict]:
        """دریافت اخبار ارزهای دیجیتال"""
        params = {"page": page, "limit": limit}
        return self._make_request("/news", params)
    
    def get_news_sources(self) -> Optional[Dict]:
        """دریافت منابع خبری"""
        return self._make_request("/news/sources")
    
    # ========== 🤖 داده‌های خام برای AI ==========
    
    def get_ai_raw_single(self, symbol: str, timeframe: str = "24h", limit: int = 500) -> Optional[Dict]:
        """داده‌های خام یک ارز برای AI"""
        params = {"timeframe": timeframe, "limit": limit}
        return self._make_request(f"/ai/raw/single/{symbol}", params)
    
    def get_ai_raw_multi(self, symbols: str = "btc,eth,sol", timeframe: str = "24h", limit: int = 100) -> Optional[Dict]:
        """داده‌های خام چند ارز برای AI"""
        params = {"symbols": symbols, "timeframe": timeframe, "limit": limit}
        return self._make_request("/ai/raw/multi", params)
    
    def get_ai_raw_market(self) -> Optional[Dict]:
        """داده‌های خام overview بازار برای AI"""
        return self._make_request("/ai/raw/market")
    
    # ========== 🩺 سلامت سیستم و مانیتورینگ ==========
    
    def get_health_combined(self) -> Optional[Dict]:
        """دریافت سلامت کلی سیستم"""
        return self._make_request("/health-combined")
    
    def get_api_data(self) -> Optional[Dict]:
        """دریافت اجزای سیستم API"""
        return self._make_request("/api-data")
    
    def get_websocket_status(self) -> Optional[Dict]:
        """دریافت وضعیت وب‌سوکت"""
        health_data = self.get_health_combined()
        if health_data and 'websocket_status' in health_data:
            return health_data['websocket_status']
        
        api_data = self.get_api_data()
        if api_data and 'api_status' in api_data:
            return api_data['api_status'].get('websocket', {})
            
        return None
    
    # ========== 🔄 متدهای کمکی و ترکیبی ==========
    
    def test_connection(self) -> bool:
        """تست اتصال به سرور میانی"""
        try:
            health_data = self.get_health_combined()
            if health_data and health_data.get('status') == 'healthy':
                print("✅ اتصال به سرور میانی برقرار است")
                return True
            else:
                print("❌ وضعیت سرور میانی نامشخص")
                return False
        except:
            print("🔌 خطا در تست اتصال")
            return False
    
    def get_all_market_data(self) -> Dict[str, Any]:
        """دریافت کلیه داده‌های بازار در یک متد"""
        print("🚀 شروع دریافت کلیه داده‌های بازار...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "currencies": self.get_currencies(),
            "market_cap": self.get_market_cap(),
            "insights_dashboard": self.get_insights_dashboard(),
            "btc_dominance": self.get_btc_dominance(),
            "fear_greed": self.get_fear_greed(),
            "fear_greed_chart": self.get_fear_greed_chart(),
            "news": self.get_news(limit=10),
            "news_sources": self.get_news_sources(),
            "timeframes": self.get_timeframes(),
            "system_health": self.get_health_combined(),
            "api_components": self.get_api_data(),
            "websocket_status": self.get_websocket_status()
        }
        
        print(f"🎉 دریافت کلیه داده‌ها کامل شد. {len(results)} بخش دریافت شد")
        return results
    
    def get_ai_training_data(self) -> Dict[str, Any]:
        """دریافت داده‌های آموزشی برای AI"""
        print("🧠 شروع دریافت داده‌های آموزشی AI...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "raw_single_btc": self.get_ai_raw_single("btc"),
            "raw_single_eth": self.get_ai_raw_single("eth"),
            "raw_multi": self.get_ai_raw_multi("btc,eth,sol,ada,dot"),
            "raw_market": self.get_ai_raw_market(),
            "market_overview": {
                "currencies": self.get_currencies(),
                "market_cap": self.get_market_cap(),
                "insights": self.get_insights_dashboard()
            }
        }
        
        print("✅ دریافت داده‌های آموزشی AI کامل شد")
        return results
    
    def get_status_report(self) -> Dict[str, Any]:
        """گزارش وضعیت کلی سیستم"""
        health = self.get_health_combined()
        api_data = self.get_api_data()
        websocket = self.get_websocket_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "connection_status": "connected" if health else "disconnected",
            "server_health": health.get('status') if health else "unknown",
            "websocket_connected": websocket.get('connected') if websocket else False,
            "active_coins": websocket.get('active_coins') if websocket else 0,
            "total_requests": self.request_count,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else "never"
        }


# نمونه سریع برای تست
if __name__ == "__main__":
    print("🔧 شروع تست VortexAPIClient...")
    
    # ایجاد کلاینت
    client = VortexAPIClient("https://server-test-ovta.onrender.com/api")
    
    # تست اتصال
    if client.test_connection():
        print("\n📊 تست دریافت داده‌های اصلی...")
        
        # تست دریافت داده‌های بازار
        market_data = client.get_all_market_data()
        print(f"✅ داده‌های بازار دریافت شد: {len(market_data)} بخش")
        
        # تست دریافت داده‌های آموزشی AI
        ai_data = client.get_ai_training_data()
        print(f"✅ داده‌های آموزشی AI دریافت شد: {len(ai_data)} بخش")
        
        # تست وضعیت سیستم
        status = client.get_status_report()
        print(f"📈 گزارش وضعیت: {status}")
        
    else:
        print("❌ تست اتصال ناموفق بود")

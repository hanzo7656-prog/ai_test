import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class AIDataConnector:
    """اتصال به داده‌های خام سیستم"""
    
    def __init__(self, base_url: str = "https://ai-test-3gix.onrender.com"):
        self.base_url = base_url
        self.connection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_successful_call': None
        }
    
    def get_raw_coins_data(self, limit: int = 50) -> List[Dict]:
        """دریافت داده‌های خام ارزها"""
        try:
            # در این نسخه، از endpoint سلامت برای تست استفاده می‌کنیم
            # در نسخه واقعی به endpointهای raw_data متصل می‌شود
            response = requests.get(f"{self.base_url}/api/health/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self._update_stats(success=True)
                
                # شبیه‌سازی داده‌های ارزی
                simulated_data = self._simulate_coins_data(data, limit)
                return simulated_data
            else:
                self._update_stats(success=False)
                return []
                
        except Exception as e:
            print(f"❌ Error fetching coins data: {e}")
            self._update_stats(success=False)
            return []
    
    def get_raw_news_data(self, limit: int = 20) -> List[Dict]:
        """دریافت اخبار و تحلیل‌ها"""
        try:
            # شبیه‌سازی داده‌های خبری
            news_data = []
            
            sample_news = [
                "بیت‌کوین به سطح ۴۵۰۰۰ دلار رسید",
                "اتریوم با رشد ۵ درصدی مواجه شد",
                "کاردانو قراردادهای هوشمند را فعال کرد",
                "سولانا با مشکل شبکه مواجه شد",
                "دوج کوین توسط ایلان ماسک توییت شد",
                "بازار ارزهای دیجیتال امروز صعودی بود",
                "نفتان در صرافی بایننس لیست شد",
                "پولکادات پاراچین‌های جدید راه‌اندازی کرد",
                "آوالانچ سرعت تراکنش‌ها را افزایش داد",
                "لایت‌کوین تراکنش‌های سریع‌تری ارائه داد"
            ]
            
            for i in range(min(limit, len(sample_news))):
                news_data.append({
                    'id': f"news_{i+1}",
                    'title': sample_news[i],
                    'content': f"این یک نمونه خبر درباره {sample_news[i]} است. این متن برای آموزش هوش مصنوعی استفاده می‌شود.",
                    'category': 'crypto_news',
                    'timestamp': datetime.now().isoformat(),
                    'sentiment': 'positive' if i % 3 == 0 else 'neutral'
                })
            
            self._update_stats(success=True)
            return news_data
            
        except Exception as e:
            print(f"❌ Error generating news data: {e}")
            self._update_stats(success=False)
            return []
    
    def get_raw_insights_data(self, limit: int = 15) -> List[Dict]:
        """دریافت بینش‌های بازار"""
        try:
            insights_data = []
            
            sample_insights = [
                "تحلیل تکنیکال بیت‌کوین نشان‌دهنده روند صعودی است",
                "شاخص ترس و طمع در منطقه طمع قرار دارد",
                "حجم معاملات اتریوم افزایش یافته است",
                "پول‌های هوشمند در حال خرید کاردانو هستند",
                "سطح حمایت بیت‌کوین ۴۰۰۰۰ دلار است",
                "مقاومت بعدی اتریوم ۳۲۰۰ دلار خواهد بود",
                "نوسانات بازار در هفته آینده افزایش می‌یابد",
                "انتظار می‌رود آلت‌کوین‌ها outperformance داشته باشند",
                "شبکه لایت‌کوین شلوغ شده است",
                "توسعه‌دهندگان پولکادات فعال هستند"
            ]
            
            for i in range(min(limit, len(sample_insights))):
                insights_data.append({
                    'id': f"insight_{i+1}",
                    'analysis': sample_insights[i],
                    'type': 'technical_analysis',
                    'confidence': 0.7 + (i * 0.02),
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': 'buy' if i % 2 == 0 else 'hold'
                })
            
            self._update_stats(success=True)
            return insights_data
            
        except Exception as e:
            print(f"❌ Error generating insights data: {e}")
            self._update_stats(success=False)
            return []
    
    def get_raw_exchanges_data(self, limit: int = 10) -> List[Dict]:
        """دریافت داده‌های صرافی‌ها"""
        try:
            exchanges_data = []
            
            sample_exchanges = [
                "بایننس - حجم معاملات: ۲۵ میلیارد دلار",
                "کوینبیس - کاربران فعال: ۷۳ میلیون",
                "کراکن - امنیت بالا و نقدینگی خوب",
                "FTX - derivatives و محصولات پیشرفته",
                "بایبیت - صرافی آسیایی در حال رشد",
                "کوکوین - altcoins متنوع",
                "گیت‌آیو - Launchpad و محصولات جدید",
                "هوبی - صرافی چینی پیشرو",
                "اوکی‌ایکس - محصولات DeFi",
                "بیت‌استامپ - صرافی اروپایی"
            ]
            
            for i in range(min(limit, len(sample_exchanges))):
                exchanges_data.append({
                    'id': f"exchange_{i+1}",
                    'name': sample_exchanges[i].split(' - ')[0],
                    'details': sample_exchanges[i].split(' - ')[1],
                    'volume_usd': 1000000000 * (i + 1),  # شبیه‌سازی حجم
                    'active_users': 5000000 * (i + 1),   # شبیه‌سازی کاربران
                    'timestamp': datetime.now().isoformat()
                })
            
            self._update_stats(success=True)
            return exchanges_data
            
        except Exception as e:
            print(f"❌ Error generating exchanges data: {e}")
            self._update_stats(success=False)
            return []
    
    def _simulate_coins_data(self, health_data: Dict, limit: int) -> List[Dict]:
        """شبیه‌سازی داده‌های ارزی بر اساس داده‌های سلامت"""
        coins_data = []
        
        sample_coins = [
            "بیت‌کوین", "اتریوم", "بایننس کوین", "ریپل", "کاردانو",
            "سولانا", "دوج کوین", "پولکادات", "لایت‌کوین", "اتریوم کلاسیک"
        ]
        
        for i in range(min(limit, len(sample_coins))):
            # استفاده از داده‌های واقعی سلامت برای شبیه‌سازی
            system_health = health_data.get('health_score', 50)
            
            coins_data.append({
                'id': f"coin_{i+1}",
                'name': sample_coins[i],
                'symbol': sample_coins[i][:3].upper(),
                'price_usd': 1000 * (i + 1) + (system_health * 10),
                'change_24h': (i - 5) * 2.5,  # تغییرات متنوع
                'volume_24h': 500000000 * (i + 1),
                'market_cap': 10000000000 * (i + 1),
                'timestamp': datetime.now().isoformat()
            })
        
        return coins_data
    
    def _update_stats(self, success: bool):
        """به‌روزرسانی آمار اتصال"""
        self.connection_stats['total_requests'] += 1
        
        if success:
            self.connection_stats['successful_requests'] += 1
            self.connection_stats['last_successful_call'] = datetime.now().isoformat()
        else:
            self.connection_stats['failed_requests'] += 1
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """دریافت آمار اتصال"""
        success_rate = (self.connection_stats['successful_requests'] / 
                       max(1, self.connection_stats['total_requests'])) * 100
        
        return {
            'connection_stats': self.connection_stats,
            'success_rate_percent': round(success_rate, 2),
            'base_url': self.base_url,
            'status': 'connected' if success_rate > 80 else 'degraded',
            'last_update': datetime.now().isoformat()
        }

# نمونه گلوبال
ai_data_connector = AIDataConnector()

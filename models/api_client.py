# api_client.py
import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

class VortexAPIClient:
    def __init__(self, base_url: str = "https://server-test-ovta.onrender.com/api"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Vortex-AI-Client/1.0',
            'Accept': 'application/json'
        })
        self.timeout = 30

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """درخواست عمومی به API"""
        try:
            url = f"{self.base_url}{endpoint}"
            print(f"🤖 AI → {url}")
            
            # اضافه کردن پارامتر raw برای تمام درخواست‌ها
            if params is None:
                params = {}
            params['raw'] = 'true'
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            return {
                'success': True,
                'data': response.json(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ API Error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # ==================== RAW COINS ENDPOINTS ====================

    def get_raw_coins(self, limit: int = 500, currency: str = 'USD') -> Dict:
        """دریافت داده‌های خام تمام کوین‌ها"""
        return self._make_request('/coins', {'limit': limit, 'currency': currency})

    def get_raw_coin_details(self, coin_id: str, currency: str = 'USD') -> Dict:
        """دریافت داده‌های خام یک کوین خاص"""
        return self._make_request(f'/coins/{coin_id}/details', {'currency': currency})

    def get_raw_coin_charts(self, coin_id: str, period: str = '24h') -> Dict:
        """دریافت داده‌های خام چارت کوین"""
        return self._make_request(f'/coins/{coin_id}/charts', {'period': period})

    # ==================== RAW MARKET DATA ENDPOINTS ====================

    def get_raw_market_cap(self) -> Dict:
        """دریافت داده‌های خام مارکت کپ"""
        return self._make_request('/markets/cap')

    def get_raw_market_summary(self) -> Dict:
        """دریافت داده‌های خام خلاصه بازار"""
        return self._make_request('/markets/summary')

    def get_raw_market_exchanges(self) -> Dict:
        """دریافت داده‌های خام صرافی‌ها"""
        return self._make_request('/markets/exchanges')

    def get_raw_global_data(self) -> Dict:
        """دریافت داده‌های خام جهانی"""
        return self._make_request('/markets/cap')

    # ==================== RAW NEWS ENDPOINTS ====================

    def get_raw_all_news(self, limit: int = 100, page: int = 1) -> Dict:
        """دریافت داده‌های خام تمام اخبار"""
        return self._make_request('/news', {'limit': limit, 'page': page})

    def get_raw_latest_news(self, limit: int = 50) -> Dict:
        """دریافت داده‌های خام آخرین اخبار"""
        return self._make_request('/news/latest', {'limit': limit})

    def get_raw_trending_news(self, limit: int = 50) -> Dict:
        """دریافت داده‌های خام اخبار ترند"""
        return self._make_request('/news/trending', {'limit': limit})

    def get_raw_handpicked_news(self, limit: int = 50) -> Dict:
        """دریافت داده‌های خام اخبار منتخب"""
        return self._make_request('/news/handpicked', {'limit': limit})

    def get_raw_bullish_news(self, limit: int = 50) -> Dict:
        """دریافت داده‌های خام اخبار صعودی"""
        return self._make_request('/news/bullish', {'limit': limit})

    def get_raw_bearish_news(self, limit: int = 50) -> Dict:
        """دریافت داده‌های خام اخبار نزولی"""
        return self._make_request('/news/bearish', {'limit': limit})

    def get_raw_news_sources(self) -> Dict:
        """دریافت داده‌های خام منابع خبری"""
        return self._make_request('/news/sources')

    # ==================== RAW INSIGHTS ENDPOINTS ====================

    def get_raw_fear_greed_index(self) -> Dict:
        """دریافت داده‌های خام Fear & Greed"""
        return self._make_request('/insights/fear-greed')

    def get_raw_fear_greed_chart(self) -> Dict:
        """دریافت داده‌های خام چارت Fear & Greed"""
        return self._make_request('/insights/fear-greed-chart')

    def get_raw_btc_dominance(self, dominance_type: str = 'all') -> Dict:
        """دریافت داده‌های خام BTC Dominance"""
        return self._make_request('/insights/btc-dominance', {'type': dominance_type})

    def get_raw_rainbow_chart(self, coin: str = 'bitcoin') -> Dict:
        """دریافت داده‌های خام چارت رنگین کمان"""
        return self._make_request('/insights/rainbow-chart', {'coin': coin})

    def get_raw_market_insights(self) -> Dict:
        """دریافت داده‌های خام بینش بازار"""
        return self._make_request('/insights/dashboard')

    # ==================== RAW HISTORICAL DATA ENDPOINTS ====================

    def get_raw_historical_1h(self, coin_id: str = 'bitcoin') -> Dict:
        """داده‌های تاریخی 1 ساعته (خام)"""
        return self._make_request(f'/coins/{coin_id}/charts', {'period': '1h'})

    def get_raw_historical_24h(self, coin_id: str = 'bitcoin') -> Dict:
        """داده‌های تاریخی 24 ساعته (خام)"""
        return self._make_request(f'/coins/{coin_id}/charts', {'period': '24h'})

    def get_raw_historical_7d(self, coin_id: str = 'bitcoin') -> Dict:
        """داده‌های تاریخی 7 روزه (خام)"""
        return self._make_request(f'/coins/{coin_id}/charts', {'period': '7d'})

    def get_raw_historical_30d(self, coin_id: str = 'bitcoin') -> Dict:
        """داده‌های تاریخی 30 روزه (خام)"""
        return self._make_request(f'/coins/{coin_id}/charts', {'period': '30d'})

    # ==================== RAW ANALYSIS ENDPOINTS ====================

    def get_raw_technical_analysis(self, symbol: str, timeframe: str = '24h') -> Dict:
        """دریافت داده‌های خام تحلیل تکنیکال"""
        return self._make_request('/analysis/technical', {'symbol': symbol, 'timeframe': timeframe})

    def get_raw_market_scan(self, limit: int = 100, filter_type: str = 'volume', timeframe: str = '24h') -> Dict:
        """دریافت داده‌های خام اسکن بازار"""
        return self._make_request('/scan', {'limit': limit, 'filter': filter_type, 'timeframe': timeframe})

    # ==================== RAW SYSTEM ENDPOINTS ====================

    def get_raw_system_stats(self) -> Dict:
        """دریافت داده‌های خام آمار سیستم"""
        return self._make_request('/system/stats')

    def get_raw_websocket_status(self) -> Dict:
        """دریافت داده‌های خام وضعیت وب‌سوکت"""
        return self._make_request('/websocket/status')

    def get_raw_performance_stats(self) -> Dict:
        """دریافت داده‌های خام آمار عملکرد"""
        return self._make_request('/system/stats')

    # ==================== BATCH DATA FOR AI ====================

    def get_ai_training_data(self) -> Dict:
        """دریافت تمام داده‌های مورد نیاز برای آموزش هوش مصنوعی"""
        start_time = time.time()
        
        # دریافت داده‌های اصلی
        data_sources = {
            'market_data': self.get_raw_coins(200),
            'historical_1h': self.get_raw_historical_1h(),
            'historical_24h': self.get_raw_historical_24h(),
            'historical_7d': self.get_raw_historical_7d(),
            'fear_greed': self.get_raw_fear_greed_index(),
            'fear_greed_chart': self.get_raw_fear_greed_chart(),
            'btc_dominance': self.get_raw_btc_dominance(),
            'market_insights': self.get_raw_market_insights(),
            'rainbow_chart': self.get_raw_rainbow_chart(),
            'news': self.get_raw_all_news(50),
            'market_scan': self.get_raw_market_scan(50),
            'system_stats': self.get_raw_system_stats()
        }
        
        processing_time = time.time() - start_time
        
        # تحلیل موفقیت‌ها
        successful_sources = [k for k, v in data_sources.items() if v['success']]
        
        return {
            'success': len(successful_sources) > 0,
            'data_sources': data_sources,
            'successful_sources': successful_sources,
            'total_sources': len(data_sources),
            'success_rate': f"{len(successful_sources)}/{len(data_sources)}",
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().isoformat(),
            'data_type': 'raw_training_data'
        }

    def get_ai_prediction_data(self) -> Dict:
        """داده‌های بهینه‌شده برای پیش‌بینی هوش مصنوعی"""
        start_time = time.time()
        
        # دریافت داده‌های ضروری برای پیش‌بینی
        essential_data = {
            'current_market': self.get_raw_coins(100),
            'bitcoin_metrics': self.get_raw_coin_details('bitcoin'),
            'market_sentiment': self.get_raw_fear_greed_index(),
            'recent_trends': self.get_raw_historical_24h(),
            'btc_dominance': self.get_raw_btc_dominance(),
            'market_scan': self.get_raw_market_scan(30)
        }
        
        processing_time = time.time() - start_time
        
        successful_sources = [k for k, v in essential_data.items() if v['success']]
        
        return {
            'success': len(successful_sources) > 0,
            'prediction_data': essential_data,
            'successful_sources': successful_sources,
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().isoformat(),
            'data_type': 'prediction_optimized'
        }

    # ==================== COMPATIBILITY METHODS ====================

    def get_all_market_data(self) -> Dict:
        """متد سازگاری برای هوش مصنوعی فعلی"""
        raw_data = self.get_ai_training_data()
        
        if raw_data['success']:
            return {
                'insights_dashboard': raw_data['data_sources']['market_insights'],
                'fear_greed': raw_data['data_sources']['fear_greed'],
                'fear_greed_chart': raw_data['data_sources']['fear_greed_chart'],
                'btc_dominance': raw_data['data_sources']['btc_dominance'],
                'market_cap': raw_data['data_sources']['market_data'],
                'news': raw_data['data_sources']['news'],
                'historical_data': raw_data['data_sources']['historical_24h'],
                'rainbow_chart': raw_data['data_sources']['rainbow_chart'],
                'market_scan': raw_data['data_sources']['market_scan']
            }
        return {}

    def get_ai_raw_single(self, symbol: str) -> Dict:
        """داده‌های خام برای یک ارز خاص"""
        return {
            'coin_data': self.get_raw_coin_details(symbol),
            'historical_1h': self.get_raw_historical_1h(symbol),
            'historical_24h': self.get_raw_historical_24h(symbol),
            'technical_analysis': self.get_raw_technical_analysis(symbol)
        }

    def get_historical_data(self, symbol: str) -> Dict:
        """داده‌های تاریخی برای سازگاری"""
        return self.get_raw_historical_24h(symbol)

    def get_market_cap(self) -> Dict:
        """داده‌های مارکت کپ"""
        return self.get_raw_market_cap()

    def get_fear_greed(self) -> Dict:
        """داده‌های Fear & Greed"""
        return self.get_raw_fear_greed_index()

    def get_btc_dominance(self) -> Dict:
        """داده‌های BTC Dominance"""
        return self.get_raw_btc_dominance()

    def get_insights_dashboard(self) -> Dict:
        """داده‌های دشبورد"""
        return self.get_raw_market_insights()

    def get_health_combined(self) -> Dict:
        """داده‌های سلامت سیستم"""
        return self._make_request('/health/combined')

    # ==================== TEST METHODS ====================

    def test_connection(self) -> bool:
        """تست اتصال به سرور"""
        try:
            result = self._make_request('/health')
            return result['success']
        except:
            return False

    def get_status_report(self) -> Dict:
        """گزارش وضعیت اتصال"""
        tests = [
            ('Health Check', self.test_connection),
            ('Market Data', lambda: self.get_raw_coins(1)['success']),
            ('Historical Data', lambda: self.get_raw_historical_1h()['success']),
            ('Fear Greed', lambda: self.get_raw_fear_greed_index()['success']),
            ('BTC Dominance', lambda: self.get_raw_btc_dominance()['success']),
            ('News', lambda: self.get_raw_all_news(1)['success']),
            ('System Stats', lambda: self.get_raw_system_stats()['success'])
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except:
                results[test_name] = False
        
        return {
            'connection_tests': results,
            'success_rate': f"{sum(results.values())}/{len(results)}",
            'timestamp': datetime.now().isoformat()
        }

    def comprehensive_test(self) -> Dict:
        """تست جامع تمام اندپوینت‌ها"""
        print("🧪 شروع تست جامع API...")
        start_time = time.time()
        
        # تست تمام دسته‌های داده
        test_categories = {
            'coins': [
                ('لیست کوین‌ها', lambda: self.get_raw_coins(5)),
                ('جزییات بیت‌کوین', lambda: self.get_raw_coin_details('bitcoin'))
            ],
            'market': [
                ('مارکت کپ', self.get_raw_market_cap),
                ('خلاصه بازار', self.get_raw_market_summary),
                ('صرافی‌ها', self.get_raw_market_exchanges)
            ],
            'news': [
                ('اخبار', lambda: self.get_raw_all_news(3)),
                ('اخبار ترند', lambda: self.get_raw_trending_news(3)),
                ('منابع خبری', self.get_raw_news_sources)
            ],
            'insights': [
                ('Fear Greed', self.get_raw_fear_greed_index),
                ('BTC Dominance', self.get_raw_btc_dominance),
                ('چارت رنگین کمان', self.get_raw_rainbow_chart)
            ],
            'historical': [
                ('تاریخی 1H', self.get_raw_historical_1h),
                ('تاریخی 24H', self.get_raw_historical_24h),
                ('تاریخی 7D', self.get_raw_historical_7d)
            ],
            'analysis': [
                ('اسکن بازار', lambda: self.get_raw_market_scan(10)),
                ('تحلیل تکنیکال', lambda: self.get_raw_technical_analysis('bitcoin'))
            ]
        }
        
        results = {}
        for category, tests in test_categories.items():
            results[category] = {}
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    results[category][test_name] = {
                        'success': result['success'],
                        'response_time': 'N/A',
                        'data_size': len(str(result)) if result['success'] else 0
                    }
                except Exception as e:
                    results[category][test_name] = {
                        'success': False,
                        'error': str(e)
                    }
                time.sleep(0.5)  # تاخیر بین درخواست‌ها
        
        total_time = time.time() - start_time
        
        # محاسبه آمار
        total_tests = sum(len(tests) for tests in test_categories.values())
        successful_tests = sum(
            1 for category in results.values() 
            for test in category.values() 
            if test.get('success', False)
        )
        
        return {
            'test_results': results,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': f"{(successful_tests/total_tests)*100:.1f}%",
                'total_time': f"{total_time:.2f}ثانیه"
            },
            'timestamp': datetime.now().isoformat()
        }

# AIClient.py
import requests
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

class VortexAIClient:
    def __init__(self):
        self.server_base_url = "https://server-test-ovta.onrender.com/api"
        self.ai_base_url = "https://ai-test-2nxq.onrender.com"
        
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Vortex-AI-Client/1.0',
            'Accept': 'application/json'
        })
        
        self.health_status = {
            'last_check': None,
            'is_healthy': False,
            'endpoints': {}
        }

    # ==================== UTILITY METHODS ====================
    
    def _make_request(self, endpoint: str, params: Dict = None, is_raw: bool = True) -> Dict:
        start_time = time.time()
        url = f"{self.server_base_url}{endpoint}"
        
        if params is None:
            params = {}
            
        try:
            print(f"🤖 AI Client → {url}")
            
            # اضافه کردن پارامتر raw برای درخواست‌های خام
            request_params = params.copy()
            if is_raw:
                request_params['raw'] = 'true'
            
            response = self.session.get(url, params=request_params, timeout=40)
            response.raise_for_status()
            
            duration = (time.time() - start_time) * 1000
            data = response.json()
            
            print(f"✅ AI Client ← {endpoint} ({duration:.0f}ms)")
            
            # آپدیت وضعیت سلامت
            self.health_status['endpoints'][endpoint] = {
                'status': 'healthy',
                'last_call': datetime.now().isoformat(),
                'response_time': duration
            }
            
            return {
                'success': True,
                'data': data,
                'metadata': {
                    'endpoint': endpoint,
                    'response_time': duration,
                    'timestamp': datetime.now().isoformat(),
                    'is_raw': is_raw
                }
            }
            
        except Exception as error:
            duration = (time.time() - start_time) * 1000
            print(f"❌ AI Client Error ← {endpoint} ({duration:.0f}ms): {str(error)}")
            
            self.health_status['endpoints'][endpoint] = {
                'status': 'error',
                'last_call': datetime.now().isoformat(),
                'response_time': duration,
                'error': str(error)
            }
            
            return {
                'success': False,
                'error': str(error),
                'metadata': {
                    'endpoint': endpoint,
                    'response_time': duration,
                    'timestamp': datetime.now().isoformat(),
                    'is_raw': is_raw
                }
            }

    # ==================== RAW COINS ENDPOINTS ====================

    def get_raw_coins(self, limit: int = 500, currency: str = 'USD') -> Dict:
        return self._make_request('/raw/coins', {'limit': limit, 'currency': currency})

    def get_raw_coin_details(self, coin_id: str, currency: str = 'USD') -> Dict:
        return self._make_request(f'/raw/coins/{coin_id}', {'currency': currency})

    # ==================== RAW MARKET DATA ENDPOINTS ====================

    def get_raw_market_cap(self) -> Dict:
        return self._make_request('/raw/markets/cap')

    def get_raw_market_summary(self) -> Dict:
        return self._make_request('/raw/markets/summary')

    def get_raw_market_exchanges(self) -> Dict:
        return self._make_request('/raw/markets/exchanges')

    def get_raw_global_data(self) -> Dict:
        return self._make_request('/raw/markets/global')

    # ==================== RAW NEWS ENDPOINTS ====================

    def get_raw_all_news(self, limit: int = 100, page: int = 1) -> Dict:
        return self._make_request('/raw/news', {'limit': limit, 'page': page})

    def get_raw_latest_news(self, limit: int = 50) -> Dict:
        return self._make_request('/raw/news/latest', {'limit': limit})

    def get_raw_trending_news(self, limit: int = 50) -> Dict:
        return self._make_request('/raw/news/trending', {'limit': limit})

    def get_raw_handpicked_news(self, limit: int = 50) -> Dict:
        return self._make_request('/raw/news/handpicked', {'limit': limit})

    def get_raw_bullish_news(self, limit: int = 50) -> Dict:
        return self._make_request('/raw/news/bullish', {'limit': limit})

    def get_raw_bearish_news(self, limit: int = 50) -> Dict:
        return self._make_request('/raw/news/bearish', {'limit': limit})

    def get_raw_news_sources(self) -> Dict:
        return self._make_request('/raw/news/sources')

    # ==================== RAW INSIGHTS ENDPOINTS ====================

    def get_raw_fear_greed_index(self) -> Dict:
        return self._make_request('/raw/insights/fear-greed')

    def get_raw_fear_greed_chart(self) -> Dict:
        return self._make_request('/raw/insights/fear-greed-chart')

    def get_raw_btc_dominance(self, dominance_type: str = 'all') -> Dict:
        return self._make_request('/raw/insights/btc-dominance', {'type': dominance_type})

    def get_raw_rainbow_chart(self, coin: str = 'bitcoin') -> Dict:
        return self._make_request('/raw/insights/rainbow-chart', {'coin': coin})

    def get_raw_market_insights(self) -> Dict:
        return self._make_request('/raw/insights/dashboard')

    # ==================== RAW HISTORICAL DATA ENDPOINTS ====================

    def get_raw_historical_1h(self, coin_id: str = 'bitcoin') -> Dict:
        return self._make_request(f'/raw/historical/{coin_id}', {'period': '1h'})

    def get_raw_historical_24h(self, coin_id: str = 'bitcoin') -> Dict:
        return self._make_request(f'/raw/historical/{coin_id}', {'period': '24h'})

    def get_raw_historical_7d(self, coin_id: str = 'bitcoin') -> Dict:
        return self._make_request(f'/raw/historical/{coin_id}', {'period': '7d'})

    def get_raw_historical_30d(self, coin_id: str = 'bitcoin') -> Dict:
        return self._make_request(f'/raw/historical/{coin_id}', {'period': '30d'})

    def get_raw_multiple_historical(self, coin_ids: List[str] = None, period: str = '24h') -> Dict:
        if coin_ids is None:
            coin_ids = ['bitcoin', 'ethereum']
        coins = ','.join(coin_ids)
        return self._make_request('/raw/historical/multiple', {'coins': coins, 'period': period})

    # ==================== RAW ANALYSIS ENDPOINTS ====================

    def get_raw_technical_analysis(self, symbol: str, timeframe: str = '24h') -> Dict:
        return self._make_request('/raw/analysis/technical', {'symbol': symbol, 'timeframe': timeframe})

    def get_raw_market_scan(self, limit: int = 100, filter_type: str = 'volume', timeframe: str = '24h') -> Dict:
        return self._make_request('/raw/scan', {'limit': limit, 'filter': filter_type, 'timeframe': timeframe})

    def get_raw_top_gainers(self, limit: int = 20) -> Dict:
        return self._make_request('/raw/analysis/top-gainers', {'limit': limit})

    def get_raw_top_losers(self, limit: int = 20) -> Dict:
        return self._make_request('/raw/analysis/top-losers', {'limit': limit})

    # ==================== RAW SYSTEM ENDPOINTS ====================

    def get_raw_system_stats(self) -> Dict:
        return self._make_request('/raw/system/stats')

    def get_raw_websocket_status(self) -> Dict:
        return self._make_request('/raw/websocket/status')

    def get_raw_gist_status(self) -> Dict:
        return self._make_request('/raw/gist/status')

    def get_raw_performance_stats(self) -> Dict:
        return self._make_request('/raw/system/performance')

    # ==================== BATCH DATA ENDPOINTS ====================

    def get_all_raw_market_data(self) -> Dict:
        import asyncio
        import concurrent.futures
        
        def run_in_parallel():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.get_raw_coins, 200),
                    executor.submit(self.get_raw_market_cap),
                    executor.submit(self.get_raw_fear_greed_index),
                    executor.submit(self.get_raw_btc_dominance),
                    executor.submit(self.get_raw_all_news, 50),
                    executor.submit(self.get_raw_historical_24h, 'bitcoin')
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                return results
        
        results = run_in_parallel()
        
        return {
            'success': True,
            'batch_timestamp': datetime.now().isoformat(),
            'data': {
                'coins': results[0],
                'market_cap': results[1],
                'fear_greed': results[2],
                'btc_dominance': results[3],
                'news': results[4],
                'historical': results[5]
            }
        }

    def get_all_raw_news_data(self) -> Dict:
        import concurrent.futures
        
        def run_in_parallel():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.get_raw_all_news, 50),
                    executor.submit(self.get_raw_latest_news, 30),
                    executor.submit(self.get_raw_trending_news, 30),
                    executor.submit(self.get_raw_handpicked_news, 30),
                    executor.submit(self.get_raw_bullish_news, 30),
                    executor.submit(self.get_raw_bearish_news, 30),
                    executor.submit(self.get_raw_news_sources)
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                return results
        
        results = run_in_parallel()
        
        return {
            'success': True,
            'batch_timestamp': datetime.now().isoformat(),
            'data': {
                'all_news': results[0],
                'latest': results[1],
                'trending': results[2],
                'handpicked': results[3],
                'bullish': results[4],
                'bearish': results[5],
                'sources': results[6]
            }
        }

    def get_all_raw_historical_data(self, coin_id: str = 'bitcoin') -> Dict:
        import concurrent.futures
        
        def run_in_parallel():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.get_raw_historical_1h, coin_id),
                    executor.submit(self.get_raw_historical_24h, coin_id),
                    executor.submit(self.get_raw_historical_7d, coin_id),
                    executor.submit(self.get_raw_historical_30d, coin_id)
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                return results
        
        results = run_in_parallel()
        
        return {
            'success': True,
            'batch_timestamp': datetime.now().isoformat(),
            'coin': coin_id,
            'data': {
                '1h': results[0],
                '24h': results[1],
                '7d': results[2],
                '30d': results[3]
            }
        }

    # ==================== TEST ENDPOINTS ====================

    def test_server_health(self) -> Dict:
        return self._make_request('/health', is_raw=False)

    def test_all_endpoints_comprehensive(self) -> Dict:
        start_time = time.time()
        print("🧪 Starting Comprehensive AI Client Tests...")
        
        test_categories = {
            'coins': [
                {'name': 'Raw Coins', 'test': lambda: self.get_raw_coins(10)},
                {'name': 'Raw Coin Details', 'test': lambda: self.get_raw_coin_details('bitcoin')}
            ],
            'market': [
                {'name': 'Raw Market Cap', 'test': self.get_raw_market_cap},
                {'name': 'Raw Market Summary', 'test': self.get_raw_market_summary},
                {'name': 'Raw Market Exchanges', 'test': self.get_raw_market_exchanges}
            ],
            'news': [
                {'name': 'Raw All News', 'test': lambda: self.get_raw_all_news(10)},
                {'name': 'Raw Latest News', 'test': lambda: self.get_raw_latest_news(10)},
                {'name': 'Raw Trending News', 'test': lambda: self.get_raw_trending_news(10)},
                {'name': 'Raw Bullish News', 'test': lambda: self.get_raw_bullish_news(10)},
                {'name': 'Raw Bearish News', 'test': lambda: self.get_raw_bearish_news(10)},
                {'name': 'Raw News Sources', 'test': self.get_raw_news_sources}
            ],
            'insights': [
                {'name': 'Raw Fear Greed', 'test': self.get_raw_fear_greed_index},
                {'name': 'Raw BTC Dominance', 'test': self.get_raw_btc_dominance},
                {'name': 'Raw Rainbow Chart', 'test': self.get_raw_rainbow_chart}
            ],
            'historical': [
                {'name': 'Raw Historical 1H', 'test': self.get_raw_historical_1h},
                {'name': 'Raw Historical 24H', 'test': self.get_raw_historical_24h},
                {'name': 'Raw Historical 7D', 'test': self.get_raw_historical_7d}
            ],
            'system': [
                {'name': 'Raw System Stats', 'test': self.get_raw_system_stats},
                {'name': 'Raw WebSocket Status', 'test': self.get_raw_websocket_status}
            ]
        }

        results = {
            'categories': {},
            'summary': {}
        }

        for category, tests in test_categories.items():
            print(f"\n📊 Testing {category.upper()}...")
            results['categories'][category] = []

            for test in tests:
                print(f"   🧪 {test['name']}")
                result = test['test']()
                
                results['categories'][category].append({
                    'test': test['name'],
                    'success': result['success'],
                    'response_time': result['metadata']['response_time'] if 'metadata' in result else None,
                    'error': result['error'] if 'error' in result else None
                })

                # تاخیر بین تست‌ها
                time.sleep(0.5)

        total_duration = (time.time() - start_time) * 1000
        
        # محاسبه خلاصه
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results['categories'].items():
            total_tests += len(category_results)
            passed_tests += sum(1 for r in category_results if r['success'])
        
        results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_duration_ms': total_duration
        }

        print(f"\n📊 Test Summary: {results['summary']['passed_tests']}/{results['summary']['total_tests']} passed")
        print(f"🎯 Success Rate: {results['summary']['success_rate']:.1f}%")
        
        return results

    def test_raw_data_integrity(self) -> Dict:
        print("🔍 Testing Raw Data Integrity...")
        
        tests = [
            {
                'name': 'Coins Data Structure',
                'test': lambda: self.get_raw_coins(3)
            },
            {
                'name': 'Historical Data Structure', 
                'test': lambda: self.get_raw_historical_24h('bitcoin')
            },
            {
                'name': 'Market Data Structure',
                'test': self.get_raw_market_cap
            },
            {
                'name': 'News Data Structure',
                'test': lambda: self.get_raw_all_news(5)
            },
            {
                'name': 'Fear Greed Data Structure',
                'test': self.get_raw_fear_greed_index
            }
        ]

        results = []
        for test in tests:
            test_result = test['test']()
            if test_result['success']:
                data = test_result['data']['raw_data'] if 'raw_data' in test_result['data'] else test_result['data']
                results.append({
                    'test': test['name'],
                    'valid': True,
                    'data_type': type(data).__name__,
                    'data_size': len(data) if hasattr(data, '__len__') else 'N/A'
                })
            else:
                results.append({
                    'test': test['name'],
                    'valid': False,
                    'error': test_result['error']
                })

        return {
            'success': all(r['valid'] for r in results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    # ==================== HEALTH MONITORING ====================

    def get_health_status(self) -> Dict:
        health_check = self.test_server_health()
        self.health_status['is_healthy'] = health_check['success']
        self.health_status['last_check'] = datetime.now().isoformat()
        
        return {
            'server': self.health_status,
            'ai_service': {
                'base_url': self.ai_base_url,
                'status': 'active'
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_current_status(self) -> Dict:
        healthy_endpoints = sum(1 for e in self.health_status['endpoints'].values() if e['status'] == 'healthy')
        
        return {
            **self.health_status,
            'endpoints_count': len(self.health_status['endpoints']),
            'healthy_endpoints': healthy_endpoints
        }

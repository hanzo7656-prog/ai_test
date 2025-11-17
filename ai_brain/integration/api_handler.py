import httpx
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class APIHandler:
    """مدیریت ارتباط با APIهای داخلی سیستم"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://ai-test-3gix.onrender.com')
        self.timeout = config.get('timeout_seconds', 30.0)
        
        # نگاشت intent به endpoint
        self.intent_endpoints = {
            'health_check': '/api/health/status',
            'system_status': '/api/health/status',
            'cache_status': '/api/health/cache',
            'alerts_status': '/api/health/debug',
            'metrics_status': '/api/health/metrics',
            
            'price_request': '/api/coins/details/{coin_id}',
            'list_request': '/api/coins/list',
            'coin_details': '/api/coins/details/{coin_id}',
            'coin_charts': '/api/coins/charts/{coin_id}',
            
            'news_request': '/api/news/all',
            'news_by_type': '/api/news/type/{news_type}',
            'news_sources': '/api/news/sources',
            
            'fear_greed': '/api/insights/fear-greed',
            'fear_greed_chart': '/api/insights/fear-greed/chart',
            'btc_dominance': '/api/insights/btc-dominance',
            'rainbow_chart': '/api/insights/rainbow-chart/{coin_id}',
            
            'exchanges_list': '/api/exchanges/list',
            'markets_list': '/api/exchanges/markets',
            'fiats_list': '/api/exchanges/fiats'
        }
        
        # کلاینت HTTP
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info("🚀 مدیریت APIهای داخلی راه‌اندازی شد")
    
    async def call_api(self, intent: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """فراخوانی API مربوط به intent"""
        if intent not in self.intent_endpoints:
            return self._create_error_response(f"Intent ناشناخته: {intent}")
        
        endpoint_template = self.intent_endpoints[intent]
        endpoint = self._build_endpoint(endpoint_template, params or {})
        query_params = self._build_query_params(intent, params or {})
        
        try:
            start_time = time.time()
            
            # ساخت URL کامل
            url = f"{self.base_url}{endpoint}"
            if query_params:
                from urllib.parse import urlencode
                url = f"{url}?{urlencode(query_params)}"
            
            logger.info(f"🌐 درخواست API: {url}")
            
            # ارسال درخواست
            response = await self.client.get(url)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            response_data = response.json()
            
            logger.info(f"✅ پاسخ API دریافت شد: {intent} ({response_time:.2f}ثانیه)")
            
            return {
                'success': True,
                'data': response_data,
                'response_time': response_time,
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat()
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ خطای HTTP {e.response.status_code} برای {intent}: {e}")
            return self._create_error_response(f"خطای سرور: {e.response.status_code}")
            
        except httpx.RequestError as e:
            logger.error(f"❌ خطای اتصال برای {intent}: {e}")
            return self._create_error_response(f"خطای اتصال به سرور: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ خطای غیرمنتظره برای {intent}: {e}")
            return self._create_error_response(f"خطای پردازش: {str(e)}")
    
    def _build_endpoint(self, endpoint_template: str, params: Dict[str, Any]) -> str:
        """ساخت endpoint نهایی با جایگزینی پارامترها"""
        endpoint = endpoint_template
        
        # جایگزینی پارامترهای مسیر
        if '{coin_id}' in endpoint and 'coin_id' in params:
            endpoint = endpoint.replace('{coin_id}', params['coin_id'])
        elif '{news_type}' in endpoint and 'news_type' in params:
            endpoint = endpoint.replace('{news_type}', params['news_type'])
        
        # مقدار پیش‌فرض برای coin_id
        if '{coin_id}' in endpoint and 'coin_id' not in params:
            endpoint = endpoint.replace('{coin_id}', 'bitcoin')
        
        return endpoint
    
    def _build_query_params(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت پارامترهای query بر اساس intent"""
        query_params = {}
        
        if intent == 'list_request':
            query_params.update({
                'limit': params.get('limit', 10),
                'page': params.get('page', 1),
                'sort_by': params.get('sort_by', 'rank'),
                'sort_dir': params.get('sort_dir', 'asc')
            })
        
        elif intent == 'news_request':
            query_params.update({
                'limit': params.get('limit', 5)
            })
        
        elif intent in ['health_check', 'system_status']:
            query_params.update({
                'detail': params.get('detail', 'basic')
            })
        
        elif intent == 'cache_status':
            query_params.update({
                'view': params.get('view', 'status')
            })
        
        elif intent == 'alerts_status':
            query_params.update({
                'view': params.get('view', 'alerts')
            })
        
        elif intent == 'metrics_status':
            query_params.update({
                'type': params.get('type', 'system')
            })
        
        # حذف پارامترهای None
        return {k: v for k, v in query_params.items() if v is not None}
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """ساخت پاسخ خطا"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def map_intent_to_api(self, intent: str, user_input: str, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """نگاشت intent و پارامترها به درخواست API"""
        
        # تشخیص خودکار coin_id از ورودی کاربر
        if 'coin_id' not in extracted_params:
            coin_id = self._detect_coin_id(user_input)
            if coin_id:
                extracted_params['coin_id'] = coin_id
        
        # تنظیم پارامترهای پیش‌فرض بر اساس intent
        default_params = self._get_default_params(intent)
        final_params = {**default_params, **extracted_params}
        
        # اعتبارسنجی پارامترها
        validated_params = self._validate_params(intent, final_params)
        
        logger.debug(f"🎯 نگاشت intent: {intent} → پارامترها: {validated_params}")
        
        return {
            'intent': intent,
            'params': validated_params,
            'endpoint': self.intent_endpoints.get(intent, 'unknown')
        }
    
    def _detect_coin_id(self, user_input: str) -> Optional[str]:
        """تشخیص خودکار coin_id از متن کاربر"""
        input_lower = user_input.lower()
        
        coin_mappings = {
            'bitcoin': ['بیتکوین', 'bitcoin', 'btc', 'بیت کوین'],
            'ethereum': ['اتریوم', 'ethereum', 'eth', 'اتریوم'],
            'solana': ['سولانا', 'solana', 'sol'],
            'cardano': ['کاردانو', 'cardano', 'ada'],
            'ripple': ['ریپل', 'ripple', 'xrp'],
            'polkadot': ['پولکادات', 'polkadot', 'dot'],
            'dogecoin': ['دوج کوین', 'dogecoin', 'doge']
        }
        
        for coin_id, keywords in coin_mappings.items():
            if any(keyword in input_lower for keyword in keywords):
                return coin_id
        
        return None
    
    def _get_default_params(self, intent: str) -> Dict[str, Any]:
        """پارامترهای پیش‌فرض برای هر intent"""
        defaults = {
            'list_request': {'limit': 10, 'sort_by': 'rank'},
            'news_request': {'limit': 5},
            'health_check': {'detail': 'basic'},
            'price_request': {'coin_id': 'bitcoin'},
            'coin_details': {'coin_id': 'bitcoin'}
        }
        
        return defaults.get(intent, {})
    
    def _validate_params(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """اعتبارسنجی پارامترها"""
        validated = params.copy()
        
        # اعتبارسنجی limit
        if 'limit' in validated:
            validated['limit'] = min(max(1, int(validated['limit'])), 100)
        
        # اعتبارسنجی page
        if 'page' in validated:
            validated['page'] = max(1, int(validated['page']))
        
        # اعتبارسنجی sort_dir
        if 'sort_dir' in validated and validated['sort_dir'] not in ['asc', 'desc']:
            validated['sort_dir'] = 'asc'
        
        return validated
    
    async def test_api_connections(self) -> Dict[str, Any]:
        """تست اتصال به APIهای اصلی"""
        test_endpoints = {
            'health': '/api/health/ping',
            'coins': '/api/coins/list?limit=1',
            'news': '/api/news/all?limit=1'
        }
        
        results = {}
        
        for name, endpoint in test_endpoints.items():
            try:
                url = f"{self.base_url}{endpoint}"
                response = await self.client.get(url)
                
                results[name] = {
                    'status': 'connected' if response.status_code == 200 else 'error',
                    'status_code': response.status_code,
                    'response_time': None  # می‌توان زمان پاسخ را اضافه کرد
                }
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def get_supported_intents(self) -> List[str]:
        """لیست intentهای پشتیبانی شده"""
        return list(self.intent_endpoints.keys())
    
    async def close(self):
        """بستن کلاینت HTTP"""
        await self.client.aclose()
        logger.info("🔌 کلاینت HTTP بسته شد")

# ml_core/data_integration.py
import logging
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DataIntegration:
    """یکپارچه‌سازی داده‌های خام از routes مختلف"""
    
    def __init__(self):
        self.base_url = os.getenv("SERVICE_URL")  # آدرس سرور اصلی
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🔗 Data Integration initialized")

    async def collect_raw_data(self) -> Dict[str, Any]:
        """جمع‌آوری داده از ۴ روت خام"""
        raw_data = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'metadata': {
                'total_sources': 4,
                'successful_sources': 0,
                'failed_sources': 0
            }
        }
        
        # تعریف منابع داده خام
        data_sources = {
            'raw_coins': '/api/raw_data/coins',
            'raw_exchanges': '/api/raw_data/exchanges',
            'raw_news': '/api/raw_data/news', 
            'raw_insights': '/api/raw_data/insights'
        }
        
        # جمع‌آوری موازی داده‌ها
        tasks = []
        for source_name, endpoint in data_sources.items():
            task = self._fetch_from_source(source_name, endpoint)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # پردازش نتایج
        for i, (source_name, endpoint) in enumerate(data_sources.items()):
            result = results[i]
            
            if isinstance(result, Exception):
                raw_data['sources'][source_name] = {
                    'status': 'error',
                    'error': str(result),
                    'endpoint': endpoint
                }
                raw_data['metadata']['failed_sources'] += 1
                logger.error(f"❌ Failed to fetch {source_name}: {result}")
            else:
                raw_data['sources'][source_name] = {
                    'status': 'success',
                    'data': result,
                    'endpoint': endpoint,
                    'data_size': len(str(result))
                }
                raw_data['metadata']['successful_sources'] += 1
        
        # ذخیره داده‌های جمع‌آوری شده در کش UTC
        if raw_data['metadata']['successful_sources'] > 0:
            cache_key = f"raw_data_batch:{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.cache_manager.set_data("utc", cache_key, raw_data, expire=1800)  # 30 دقیقه
            
            logger.info(f"✅ Collected data from {raw_data['metadata']['successful_sources']}/4 sources")
        else:
            logger.warning("⚠️ No data collected from any source")
        
        return raw_data

    async def _fetch_from_source(self, source_name: str, endpoint: str) -> Any:
        """دریافت داده از یک منبع خاص"""
        try:
            # بررسی کش اول
            cache_key = f"source_cache:{source_name}"
            cached_data = self.cache_manager.get_data("utc", cache_key)
            
            if cached_data is not None:
                logger.info(f"✅ Cache HIT for {source_name}")
                return cached_data
            
            # دریافت داده از API
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # ذخیره در کش
                        self.cache_manager.set_data("utc", cache_key, data, expire=300)  # 5 دقیقه
                        
                        logger.info(f"✅ Fetched fresh data for {source_name}")
                        return data
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
                        
        except asyncio.TimeoutError:
            raise Exception(f"Timeout while fetching {source_name}")
        except Exception as e:
            raise Exception(f"Error fetching {source_name}: {str(e)}")

    async def get_structured_training_data(self) -> Dict[str, Any]:
        """تهیه داده‌های ساختاریافته برای آموزش مدل"""
        try:
            # جمع‌آوری داده‌های خام
            raw_data = await self.collect_raw_data()
            
            # ساختاردهی داده‌ها برای آموزش
            structured_data = {
                'timestamp': datetime.now().isoformat(),
                'training_ready': False,
                'datasets': {},
                'statistics': {
                    'total_samples': 0,
                    'feature_count': 0,
                    'data_quality': 'unknown'
                }
            }
            
            # پردازش هر منبع داده
            for source_name, source_data in raw_data['sources'].items():
                if source_data['status'] == 'success':
                    processed = self._process_data_source(source_name, source_data['data'])
                    structured_data['datasets'][source_name] = processed
                    structured_data['statistics']['total_samples'] += processed.get('sample_count', 0)
            
            # ارزیابی کیفیت داده
            if structured_data['statistics']['total_samples'] > 0:
                structured_data['training_ready'] = True
                structured_data['statistics']['data_quality'] = self._assess_data_quality(structured_data)
            
            # ذخیره در کش UTB برای استفاده مدل‌ها
            if structured_data['training_ready']:
                cache_key = "training_data:latest"
                self.cache_manager.set_data("utb", cache_key, structured_data, expire=3600)  # 1 ساعت
                
                logger.info(f"✅ Prepared training data with {structured_data['statistics']['total_samples']} samples")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'training_ready': False,
                'error': str(e),
                'datasets': {},
                'statistics': {'total_samples': 0, 'data_quality': 'poor'}
            }

    def _process_data_source(self, source_name: str, raw_data: Any) -> Dict[str, Any]:
        """پردازش داده‌های یک منبع خاص"""
        try:
            if source_name == 'raw_coins':
                return self._process_coins_data(raw_data)
            elif source_name == 'raw_exchanges':
                return self._process_exchanges_data(raw_data)
            elif source_name == 'raw_news':
                return self._process_news_data(raw_data)
            elif source_name == 'raw_insights':
                return self._process_insights_data(raw_data)
            else:
                return {'sample_count': 0, 'features': [], 'error': 'Unknown source'}
                
        except Exception as e:
            logger.error(f"❌ Error processing {source_name}: {e}")
            return {'sample_count': 0, 'features': [], 'error': str(e)}

    def _process_coins_data(self, data: Any) -> Dict[str, Any]:
        """پردازش داده‌های کوین‌ها"""
        # پیاده‌سازی بر اساس ساختار داده‌های شما
        return {
            'sample_count': len(data) if isinstance(data, list) else 1,
            'features': ['price', 'volume', 'market_cap', 'change_24h'],
            'data_type': 'numeric',
            'processing_time': datetime.now().isoformat()
        }

    def _process_exchanges_data(self, data: Any) -> Dict[str, Any]:
        """پردازش داده‌های صرافی‌ها"""
        return {
            'sample_count': len(data) if isinstance(data, list) else 1,
            'features': ['volume', 'pairs', 'liquidity', 'fees'],
            'data_type': 'numeric',
            'processing_time': datetime.now().isoformat()
        }

    def _process_news_data(self, data: Any) -> Dict[str, Any]:
        """پردازش داده‌های خبری"""
        return {
            'sample_count': len(data) if isinstance(data, list) else 1,
            'features': ['sentiment', 'topics', 'urgency', 'relevance'],
            'data_type': 'textual',
            'processing_time': datetime.now().isoformat()
        }

    def _process_insights_data(self, data: Any) -> Dict[str, Any]:
        """پردازش داده‌های تحلیلی"""
        return {
            'sample_count': len(data) if isinstance(data, list) else 1,
            'features': ['analysis_depth', 'confidence', 'trends', 'patterns'],
            'data_type': 'analytical',
            'processing_time': datetime.now().isoformat()
        }

    def _assess_data_quality(self, structured_data: Dict[str, Any]) -> str:
        """ارزیابی کیفیت داده‌های جمع‌آوری شده"""
        total_samples = structured_data['statistics']['total_samples']
        source_count = len([d for d in structured_data['datasets'].values() if d.get('sample_count', 0) > 0])
        
        if total_samples > 1000 and source_count >= 3:
            return 'excellent'
        elif total_samples > 500 and source_count >= 2:
            return 'good'
        elif total_samples > 100:
            return 'fair'
        else:
            return 'poor'

    async def validate_data_sources(self) -> Dict[str, Any]:
        """اعتبارسنجی تمام منابع داده"""
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_status': 'healthy'
        }
        
        data_sources = {
            'raw_coins': '/api/raw_data/coins',
            'raw_exchanges': '/api/raw_data/exchanges',
            'raw_news': '/api/raw_data/news',
            'raw_insights': '/api/raw_data/insights'
        }
        
        for source_name, endpoint in data_sources.items():
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        status = 'available' if response.status == 200 else 'unavailable'
                        validation_report['sources'][source_name] = {
                            'status': status,
                            'response_time': 'N/A',  # می‌توانید زمان پاسخ را اندازه بگیرید
                            'endpoint': endpoint
                        }
                        
                        if status == 'unavailable':
                            validation_report['overall_status'] = 'degraded'
                            
            except Exception as e:
                validation_report['sources'][source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'endpoint': endpoint
                }
                validation_report['overall_status'] = 'degraded'
        
        # ذخیره گزارش اعتبارسنجی
        self.cache_manager.set_data("mother_a", "data_validation_report", validation_report, expire=600)
        
        return validation_report

# نمونه global
data_integrator = DataIntegration()

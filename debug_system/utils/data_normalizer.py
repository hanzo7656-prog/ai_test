"""
🤖 Data Normalizer v2.1 - با سیستم ذخیره‌سازی فایل و مدیریت عمر داده
ویژگی‌های جدید:
- ذخیره‌سازی تمام داده‌ها در پوشه کش
- مدیریت خودکار عمر داده (10 روز)
- قابلیت بازیابی از فایل‌ها
- سیستم پاک‌سازی دوره‌ای
"""

import json
import time
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class StructureType(Enum):
    """انواع ساختارهای شناسایی شده - نسخه پیشرفته"""
    # ساختارهای پایه
    DIRECT_LIST = "direct_list"
    SINGLE_ITEM_LIST = "single_item_list"
    
    # ساختارهای دیکشنری با لیست
    DICT_WITH_DATA = "dict_with_data"
    DICT_WITH_RESULT = "dict_with_result" 
    DICT_WITH_ITEMS = "dict_with_items"
    DICT_WITH_COINS = "dict_with_coins"
    DICT_WITH_NEWS = "dict_with_news"
    DICT_WITH_RESULTS = "dict_with_results"
    
    # ساختارهای CoinStats API خاص
    COIN_STATS_PAGINATED = "coin_stats_paginated"
    COIN_STATS_SINGLE_COIN = "coin_stats_single_coin"
    COIN_STATS_NEWS = "coin_stats_news"
    COIN_STATS_DETAILED = "coin_stats_detailed"
    COIN_STATS_CHART = "coin_stats_chart"
    COIN_STATS_MARKETS = "coin_stats_markets"
    COIN_STATS_INSIGHTS = "coin_stats_insights"
    COIN_STATS_ERROR = "coin_stats_error"
    COIN_STATS_SENTIMENT = "coin_stats_sentiment"
    
    # ساختارهای پیچیده
    NESTED_STRUCTURE = "nested_structure"
    PAGINATED_RESPONSE = "paginated_response"
    
    # fallback
    CUSTOM_STRUCTURE = "custom_structure"
    UNKNOWN = "unknown"

class NormalizationStrategy(Enum):
    """استراتژی‌های نرمال‌سازی"""
    SMART = "smart"
    STRICT = "strict"
    LENIENT = "lenient"
    COIN_STATS_OPTIMIZED = "coin_stats_optimized"

@dataclass
class NormalizationResult:
    """نتیجه نرمال‌سازی"""
    status: str  # success | error
    data: List[Any]
    metadata: Dict[str, Any]
    raw_data: Any
    normalization_info: Dict[str, Any]
    quality_score: float

@dataclass  
class HealthMetrics:
    """متریک‌های سلامت سیستم"""
    success_rate: float
    total_processed: int
    total_success: int
    total_errors: int
    common_structures: Dict[str, int]
    performance_metrics: Dict[str, Any]
    alerts: List[str]
    data_quality: Dict[str, float]
    endpoint_intelligence: Dict[str, Any]

class DataNormalizer:
    """
    سیستم هوشمند نرمال‌سازی داده‌های API - نسخه با ذخیره‌سازی فایل
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._setup_logging()
        self._initialize_cache_system()
        self._reset_metrics()
        
        # استراتژی پیش‌فرض
        self.default_strategy = NormalizationStrategy.COIN_STATS_OPTIMIZED
        
        # ساختارهای پشتیبانی شده
        self.supported_structures = {
            StructureType.DIRECT_LIST: self._normalize_direct_list,
            StructureType.SINGLE_ITEM_LIST: self._normalize_single_item_list,
            StructureType.DICT_WITH_DATA: self._normalize_dict_with_data,
            StructureType.DICT_WITH_RESULT: self._normalize_dict_with_result,
            StructureType.DICT_WITH_ITEMS: self._normalize_dict_with_items,
            StructureType.DICT_WITH_COINS: self._normalize_dict_with_coins,
            StructureType.DICT_WITH_NEWS: self._normalize_dict_with_news,
            StructureType.DICT_WITH_RESULTS: self._normalize_dict_with_results,
            StructureType.COIN_STATS_PAGINATED: self._normalize_coin_stats_paginated,
            StructureType.COIN_STATS_SINGLE_COIN: self._normalize_coin_stats_single_coin,
            StructureType.COIN_STATS_NEWS: self._normalize_coin_stats_news,
            StructureType.COIN_STATS_DETAILED: self._normalize_coin_stats_detailed,
            StructureType.COIN_STATS_CHART: self._normalize_coin_stats_chart,
            StructureType.COIN_STATS_MARKETS: self._normalize_coin_stats_markets,
            StructureType.COIN_STATS_INSIGHTS: self._normalize_coin_stats_insights,
            StructureType.COIN_STATS_ERROR: self._normalize_coin_stats_error,
            StructureType.COIN_STATS_SENTIMENT: self._normalize_coin_stats_sentiment,
            StructureType.PAGINATED_RESPONSE: self._normalize_paginated_response,
            StructureType.NESTED_STRUCTURE: self._normalize_nested_structure,
        }
        
        # الگوهای شناخته شده
        self.known_patterns = {
            "coins/list": StructureType.COIN_STATS_PAGINATED,
            "coins/bitcoin": StructureType.COIN_STATS_SINGLE_COIN,
            "coins/ethereum": StructureType.COIN_STATS_SINGLE_COIN,
            "news/type/handpicked": StructureType.COIN_STATS_NEWS,
            "news/type/trending": StructureType.COIN_STATS_NEWS,
            "news/type/latest": StructureType.COIN_STATS_NEWS,
            "news/type/bullish": StructureType.COIN_STATS_NEWS,
            "news/type/bearish": StructureType.COIN_STATS_NEWS,
            "exchanges/list": StructureType.COIN_STATS_MARKETS,
            "markets": StructureType.COIN_STATS_MARKETS,
            "insights/fear-and-greed": StructureType.COIN_STATS_INSIGHTS,
            "insights/btc-dominance": StructureType.COIN_STATS_INSIGHTS,
            "coins/charts": StructureType.COIN_STATS_CHART,
        }
        
        # بارگذاری داده‌های ذخیره شده
        self._load_persisted_data()
        
        # راه‌اندازی پاک‌ساز خودکار
        self._start_auto_cleanup()
        
        logger.info("🚀 Data Normalizer v2.1 Initialized - File Storage Enabled")

    def _setup_logging(self):
        """تنظیمات لاگ‌گیری"""
        self.logger = logging.getLogger(__name__)

    def _reset_metrics(self):
        """بازنشانی متریک‌ها"""
        self.metrics = {
            'total_processed': 0,
            'total_success': 0, 
            'total_errors': 0,
            'structure_counts': {stype.value: 0 for stype in StructureType},
            'processing_times': [],
            'endpoint_patterns': {},
            'quality_scores': [],
            'alerts': [],
            'confidence_scores': [],
            'pattern_matches': 0,
        }

    def _initialize_cache_system(self):
        """راه‌اندازی سیستم ذخیره‌سازی فایل"""
        # مسیر پوشه کش
        self.cache_base_path = Path(__file__).parent / "data_normalizer_cache"
        
        # ایجاد پوشه‌های لازم
        self.cache_dirs = {
            'structure': self.cache_base_path / "structure",
            'patterns': self.cache_base_path / "patterns", 
            'metrics': self.cache_base_path / "metrics",
            'samples': self.cache_base_path / "samples",
            'logs': self.cache_base_path / "logs",
        }
        
        for dir_path in self.cache_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"📁 Cache system initialized at: {self.cache_base_path}")

    def _load_persisted_data(self):
        """بارگذاری داده‌های ذخیره شده از فایل‌ها"""
        try:
            # بارگذاری الگوها
            patterns_file = self.cache_dirs['patterns'] / "known_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    saved_patterns = json.load(f)
                    # تبدیل به StructureType
                    for endpoint, struct_type in saved_patterns.items():
                        if struct_type in [st.value for st in StructureType]:
                            self.known_patterns[endpoint] = StructureType(struct_type)
            
            # بارگذاری متریک‌ها
            metrics_file = self.cache_dirs['metrics'] / "health_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    saved_metrics = json.load(f)
                    self.metrics.update(saved_metrics)
                    
            logger.info("✅ Persisted data loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load persisted data: {e}")

    def _save_to_cache(self, data_type: str, key: str, data: Any, timestamp: str = None):
        """ذخیره داده در فایل کش"""
        try:
            if data_type not in self.cache_dirs:
                logger.error(f"❌ Unknown cache type: {data_type}")
                return
                
            timestamp = timestamp or datetime.now().isoformat()
            filename = f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.cache_dirs[data_type] / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'data': data,
                    'timestamp': timestamp,
                    'expires_at': (datetime.now() + timedelta(days=10)).isoformat()
                }, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"❌ Failed to save cache {data_type}/{key}: {e}")

    def _load_from_cache(self, data_type: str, key: str) -> Optional[Any]:
        """بارگذاری داده از فایل کش"""
        try:
            if data_type not in self.cache_dirs:
                return None
                
            # پیدا کردن جدیدترین فایل برای این key
            pattern = f"{key}_*.json"
            files = list(self.cache_dirs[data_type].glob(pattern))
            if not files:
                return None
                
            # گرفتن جدیدترین فایل
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                
            # بررسی انقضا
            expires_at = datetime.fromisoformat(cached_data.get('expires_at', '2000-01-01'))
            if datetime.now() > expires_at:
                os.remove(latest_file)
                return None
                
            return cached_data.get('data')
            
        except Exception as e:
            logger.error(f"❌ Failed to load cache {data_type}/{key}: {e}")
            return None

    def _save_sample_data(self, endpoint: str, raw_data: Any, structure_type: StructureType):
        """ذخیره نمونه داده برای آنالیز"""
        try:
            # محدود کردن تعداد نمونه‌ها (حداکثر 10 نمونه در روز)
            endpoint_dir = self.cache_dirs['samples'] / endpoint.replace('/', '_')
            endpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # شمارش نمونه‌های امروز
            today_pattern = f"sample_{datetime.now().strftime('%Y%m%d')}_*.json"
            today_samples = list(endpoint_dir.glob(today_pattern))
            
            if len(today_samples) < 10:
                filename = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                file_path = endpoint_dir / filename
                
                sample_data = {
                    'endpoint': endpoint,
                    'structure_type': structure_type.value,
                    'timestamp': datetime.now().isoformat(),
                    'raw_data_preview': str(raw_data)[:500] + "..." if len(str(raw_data)) > 500 else str(raw_data),
                    'data_size': len(str(raw_data)) if hasattr(raw_data, '__len__') else 'unknown'
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            logger.error(f"❌ Failed to save sample data for {endpoint}: {e}")

    def _start_auto_cleanup(self):
        """راه‌اندازی پاک‌ساز خودکار"""
        try:
            self._cleanup_old_files()
            # پاک‌سازی هر 24 ساعت
            import threading
            def cleanup_scheduler():
                while True:
                    time.sleep(24 * 60 * 60)  # 24 ساعت
                    self._cleanup_old_files()
                    
            thread = threading.Thread(target=cleanup_scheduler, daemon=True)
            thread.start()
            logger.info("🧹 Auto-cleanup scheduler started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start auto-cleanup: {e}")

    def _cleanup_old_files(self):
        """پاک‌سازی فایل‌های قدیمی‌تر از 10 روز"""
        try:
            cutoff_time = datetime.now() - timedelta(days=10)
            deleted_count = 0
            
            for dir_type, dir_path in self.cache_dirs.items():
                if dir_path.exists():
                    for file_path in dir_path.rglob("*.json"):
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_time:
                                file_path.unlink()
                                deleted_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete {file_path}: {e}")
            
            # پاک‌سازی پوشه‌های خالی
            for dir_path in self.cache_dirs.values():
                if dir_path.exists() and dir_path.is_dir():
                    try:
                        # حذف پوشه‌های خالی در samples
                        if dir_path.name == "samples":
                            for subdir in dir_path.iterdir():
                                if subdir.is_dir() and not any(subdir.iterdir()):
                                    subdir.rmdir()
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clean empty directories: {e}")
            
            logger.info(f"🧹 Cleanup completed: {deleted_count} old files deleted")
            
            # ذخیره لاگ پاک‌سازی
            cleanup_log = {
                'timestamp': datetime.now().isoformat(),
                'deleted_files': deleted_count,
                'cutoff_time': cutoff_time.isoformat()
            }
            
            log_file = self.cache_dirs['logs'] / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(cleanup_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

    def _persist_metrics(self):
        """ذخیره متریک‌ها در فایل"""
        try:
            metrics_file = self.cache_dirs['metrics'] / "health_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, default=str)
                
            # ذخیره الگوها
            patterns_file = self.cache_dirs['patterns'] / "known_patterns.json"
            patterns_to_save = {k: v.value for k, v in self.known_patterns.items()}
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_to_save, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to persist metrics: {e}")

    # ========================== متدهای اصلی ==========================

    def normalize(self, raw_data: Any, endpoint: str = "unknown", 
                 strategy: NormalizationStrategy = None) -> NormalizationResult:
        """نرمال‌سازی با ذخیره‌سازی فایل"""
        start_time = time.time()
        self.metrics['total_processed'] += 1
        
        try:
            # تشخیص ساختار
            structure_type, confidence, pattern_used = self._detect_structure_advanced(raw_data, endpoint)
            self.metrics['structure_counts'][structure_type.value] += 1
            self.metrics['confidence_scores'].append(confidence)
            
            if pattern_used:
                self.metrics['pattern_matches'] += 1
            
            # نرمال‌سازی
            normalization_func = self.supported_structures.get(
                structure_type, 
                self._normalize_fallback_advanced
            )
            
            normalized_data = normalization_func(raw_data)
            
            # محاسبه کیفیت
            quality_score = self._calculate_quality_score_advanced(normalized_data, structure_type, confidence)
            self.metrics['quality_scores'].append(quality_score)
            
            # ذخیره‌سازی داده‌ها
            self._save_sample_data(endpoint, raw_data, structure_type)
            self._update_endpoint_intelligence(endpoint, structure_type, confidence, raw_data)
            self._persist_metrics()
            
            processing_time = time.time() - start_time
            self.metrics['processing_times'].append(processing_time)
            self.metrics['total_success'] += 1
            
            result = NormalizationResult(
                status="success",
                data=normalized_data,
                metadata=self._extract_metadata_advanced(raw_data, structure_type),
                raw_data=raw_data,
                normalization_info={
                    "detected_structure": structure_type.value,
                    "confidence": confidence,
                    "pattern_used": pattern_used,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "endpoint": endpoint,
                    "strategy": (strategy or self.default_strategy).value,
                    "data_quality": quality_score,
                    "timestamp": datetime.now().isoformat(),
                    "cache_used": False
                },
                quality_score=quality_score
            )
            
            logger.info(f"✅ Normalized {endpoint} - {structure_type.value} (Conf: {confidence}%) - Quality: {quality_score}%")
            return result
            
        except Exception as e:
            self.metrics['total_errors'] += 1
            error_msg = f"Normalization failed for {endpoint}: {str(e)}"
            logger.error(error_msg)
            self.metrics['alerts'].append(error_msg)
            
            # ذخیره خطا
            self._save_to_cache('logs', f"error_{endpoint}", {
                'error': str(e),
                'endpoint': endpoint,
                'timestamp': datetime.now().isoformat(),
                'raw_data_preview': str(raw_data)[:200] if raw_data else 'None'
            })
            
            return NormalizationResult(
                status="error",
                data=[],
                metadata={},
                raw_data=raw_data,
                normalization_info={
                    "error": str(e),
                    "endpoint": endpoint,
                    "timestamp": datetime.now().isoformat(),
                    "fallback_used": True
                },
                quality_score=0.0
            )

    def _detect_structure_advanced(self, raw_data: Any, endpoint: str = "unknown") -> Tuple[StructureType, float, bool]:
        """تشخیص پیشرفته ساختار با الگوی endpoint"""
        # استفاده از الگوهای شناخته شده
        if endpoint in self.known_patterns:
            known_structure = self.known_patterns[endpoint]
            logger.debug(f"🎯 Using known pattern for {endpoint}: {known_structure.value}")
            return known_structure, 0.95, True
        
        # تشخیص بر اساس محتوا برای CoinStats
        if isinstance(raw_data, dict):
            # تشخیص خطاهای CoinStats
            if 'error' in raw_data or 'statusCode' in raw_data:
                return StructureType.COIN_STATS_ERROR, 0.90, False
            
            # تشخیص ساختارهای خاص CoinStats
            if 'now' in raw_data and 'value' in raw_data.get('now', {}):
                return StructureType.COIN_STATS_INSIGHTS, 0.88, False
                
            if 'meta' in raw_data and 'result' in raw_data:
                return StructureType.COIN_STATS_PAGINATED, 0.94, False
                
        # تشخیص پایه
        return self._detect_structure_basic(raw_data)

    def _detect_structure_basic(self, raw_data: Any) -> Tuple[StructureType, float, bool]:
        """تشخیص ساختار پایه"""
        if isinstance(raw_data, list):
            if len(raw_data) == 1 and isinstance(raw_data[0], dict):
                return StructureType.SINGLE_ITEM_LIST, 0.92, False
            elif len(raw_data) > 0:
                return StructureType.DIRECT_LIST, 0.90, False
            else:
                return StructureType.DIRECT_LIST, 0.70, False
        
        elif isinstance(raw_data, dict):
            if 'result' in raw_data and isinstance(raw_data['result'], list):
                if 'meta' in raw_data:
                    return StructureType.COIN_STATS_PAGINATED, 0.94, False
                else:
                    return StructureType.DICT_WITH_RESULT, 0.88, False
            
            key_structures = {
                'data': StructureType.DICT_WITH_DATA,
                'items': StructureType.DICT_WITH_ITEMS,
                'coins': StructureType.DICT_WITH_COINS,
                'news': StructureType.DICT_WITH_NEWS,
                'results': StructureType.DICT_WITH_RESULTS,
            }
            
            for key, structure in key_structures.items():
                if key in raw_data and isinstance(raw_data[key], list):
                    return structure, 0.85, False
            
            nested_list = self._find_nested_list(raw_data)
            if nested_list:
                return StructureType.NESTED_STRUCTURE, 0.80, False
        
        return StructureType.UNKNOWN, 0.5, False

    def _find_nested_list(self, data: Dict, max_depth: int = 3) -> Optional[List]:
        """پیدا کردن لیست در ساختارهای تودرتو"""
        def find_recursive(obj, depth=0):
            if depth >= max_depth:
                return None
            
            if isinstance(obj, list) and len(obj) > 0:
                return obj
            elif isinstance(obj, dict):
                for value in obj.values():
                    result = find_recursive(value, depth + 1)
                    if result:
                        return result
            return None
        
        return find_recursive(data)

    # ========================== نرمال‌سازهای اصلی ==========================

    def _normalize_direct_list(self, raw_data: List) -> List:
        return raw_data

    def _normalize_single_item_list(self, raw_data: List) -> List:
        return raw_data

    def _normalize_dict_with_data(self, raw_data: Dict) -> List:
        return raw_data.get('data', [])

    def _normalize_dict_with_result(self, raw_data: Dict) -> List:
        return raw_data.get('result', [])

    def _normalize_dict_with_items(self, raw_data: Dict) -> List:
        return raw_data.get('items', [])

    def _normalize_dict_with_coins(self, raw_data: Dict) -> List:
        return raw_data.get('coins', [])

    def _normalize_dict_with_news(self, raw_data: Dict) -> List:
        return raw_data.get('news', [])

    def _normalize_dict_with_results(self, raw_data: Dict) -> List:
        return raw_data.get('results', [])

    def _normalize_coin_stats_paginated(self, raw_data: Dict) -> List:
        return raw_data.get('result', [])

    def _normalize_coin_stats_single_coin(self, raw_data: List) -> List:
        return raw_data

    def _normalize_coin_stats_news(self, raw_data: Dict) -> List:
        if 'result' in raw_data:
            return raw_data['result']
        elif 'news' in raw_data:
            return raw_data['news']
        else:
            return self._extract_data_from_complex_structure(raw_data)

    def _normalize_coin_stats_detailed(self, raw_data: Dict) -> List:
        if 'data' in raw_data and isinstance(raw_data['data'], dict):
            return [raw_data['data']]
        return [raw_data]

    def _normalize_coin_stats_chart(self, raw_data: Any) -> List:
        if isinstance(raw_data, list):
            return raw_data
        elif isinstance(raw_data, dict) and 'result' in raw_data:
            return raw_data['result']
        return []

    def _normalize_coin_stats_markets(self, raw_data: Dict) -> List:
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_coin_stats_insights(self, raw_data: Dict) -> List:
        if 'now' in raw_data:
            return [raw_data]
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_coin_stats_error(self, raw_data: Dict) -> List:
        return [raw_data]

    def _normalize_coin_stats_sentiment(self, raw_data: Dict) -> List:
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_paginated_response(self, raw_data: Dict) -> List:
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_nested_structure(self, raw_data: Dict) -> List:
        nested_list = self._find_nested_list(raw_data)
        return nested_list or []

    def _normalize_fallback_advanced(self, raw_data: Any) -> List:
        """Fallback پیشرفته"""
        if isinstance(raw_data, dict):
            lists = [v for v in raw_data.values() if isinstance(v, list)]
            if lists:
                return max(lists, key=len)
            
            for value in raw_data.values():
                if isinstance(value, list):
                    return value
            
            return [raw_data]
        
        elif isinstance(raw_data, list):
            return raw_data
        
        else:
            return [raw_data] if raw_data is not None else []

    def _extract_data_from_complex_structure(self, raw_data: Any) -> List:
        if isinstance(raw_data, dict):
            lists_in_dict = [v for v in raw_data.values() if isinstance(v, list)]
            if lists_in_dict:
                return max(lists_in_dict, key=len)
        return []

    def _extract_metadata_advanced(self, raw_data: Any, structure_type: StructureType) -> Dict[str, Any]:
        """استخراج متادیتا"""
        metadata = {
            "structure_type": structure_type.value,
            "extracted_at": datetime.now().isoformat(),
            "data_source": "coinstats_api",
            "structure_complexity": self._calculate_structure_complexity(raw_data)
        }
        
        if isinstance(raw_data, dict):
            common_meta_keys = ['meta', 'metadata', 'pagination', 'info', 'total', 'count', 'page', 'limit']
            for key in common_meta_keys:
                if key in raw_data:
                    metadata[key] = raw_data[key]
                    
        return metadata

    def _calculate_structure_complexity(self, data: Any) -> str:
        """محاسبه پیچیدگی ساختار"""
        if isinstance(data, list):
            return "low" if len(data) < 10 else "medium"
        elif isinstance(data, dict):
            key_count = len(data)
            if key_count < 5:
                return "low"
            elif key_count < 15:
                return "medium"
            else:
                return "high"
        else:
            return "unknown"

    def _calculate_quality_score_advanced(self, normalized_data: List, structure_type: StructureType, confidence: float) -> float:
        """محاسبه امتیاز کیفیت"""
        if not normalized_data:
            return 0.0
            
        score = 0.0
        
        # امتیاز بر اساس حجم داده
        data_count = len(normalized_data)
        if data_count > 0:
            score += min(data_count / 50, 0.3)
            
        # امتیاز بر اساس ساختار
        structure_scores = {
            StructureType.COIN_STATS_PAGINATED: 0.3,
            StructureType.COIN_STATS_SINGLE_COIN: 0.25,
            StructureType.COIN_STATS_NEWS: 0.25,
            StructureType.DICT_WITH_RESULT: 0.25,
            StructureType.DICT_WITH_DATA: 0.25,
            StructureType.SINGLE_ITEM_LIST: 0.2,
            StructureType.DIRECT_LIST: 0.2,
            StructureType.DICT_WITH_ITEMS: 0.2,
            StructureType.DICT_WITH_COINS: 0.2,
            StructureType.PAGINATED_RESPONSE: 0.25,
            StructureType.NESTED_STRUCTURE: 0.15,
            StructureType.CUSTOM_STRUCTURE: 0.1,
            StructureType.UNKNOWN: 0.05
        }
        score += structure_scores.get(structure_type, 0.1)
        
        # امتیاز بر اساس confidence
        score += confidence * 0.3
        
        # امتیاز بر اساس یکنواختی
        if data_count > 1:
            uniformity_score = self._calculate_uniformity_score(normalized_data)
            score += uniformity_score * 0.2
            
        return min(score * 100, 100.0)

    def _calculate_uniformity_score(self, data: List) -> float:
        if not data or len(data) < 2:
            return 0.5
            
        try:
            if all(isinstance(item, dict) for item in data):
                first_keys = set(data[0].keys())
                common_keys = first_keys.intersection(*(set(item.keys()) for item in data[1:]))
                return len(common_keys) / len(first_keys) if first_keys else 0.5
        except:
            pass
            
        return 0.5

    def _update_endpoint_intelligence(self, endpoint: str, structure_type: StructureType, confidence: float, raw_data: Any):
        """سیستم یادگیری هوشمند endpointها"""
        if endpoint not in self.metrics['endpoint_patterns']:
            self.metrics['endpoint_patterns'][endpoint] = {
                'total_requests': 0,
                'structure_counts': {},
                'confidence_history': [],
                'last_detected': None,
                'pattern_stability': 0.0
            }
            
        pattern = self.metrics['endpoint_patterns'][endpoint]
        pattern['total_requests'] += 1
        pattern['structure_counts'][structure_type.value] = pattern['structure_counts'].get(structure_type.value, 0) + 1
        pattern['confidence_history'].append(confidence)
        pattern['last_detected'] = datetime.now().isoformat()
        
        # محاسبه پایداری الگو
        if pattern['total_requests'] > 1:
            main_structure_count = max(pattern['structure_counts'].values())
            pattern['pattern_stability'] = main_structure_count / pattern['total_requests']
            
        # یادگیری الگوی جدید اگر پایداری بالا باشد
        if (pattern['pattern_stability'] > 0.8 and 
            pattern['total_requests'] > 5 and 
            endpoint not in self.known_patterns):
            self.known_patterns[endpoint] = structure_type
            logger.info(f"🎓 Learned new pattern: {endpoint} -> {structure_type.value}")

    # ========================== متدهای عمومی ==========================

    def get_endpoint_intelligence(self, endpoint: str = None) -> Dict[str, Any]:
        """دریافت هوش جمع‌آوری شده برای endpointها"""
        if endpoint:
            return self.metrics['endpoint_patterns'].get(endpoint, {})
        else:
            return {
                'total_endpoints': len(self.metrics['endpoint_patterns']),
                'endpoints': self.metrics['endpoint_patterns'],
                'pattern_efficiency': f"{(self.metrics['pattern_matches'] / self.metrics['total_processed'] * 100) if self.metrics['total_processed'] > 0 else 0:.1f}%",
                'timestamp': datetime.now().isoformat()
            }

    def add_known_pattern(self, endpoint: str, structure_type: StructureType):
        """اضافه کردن الگوی شناخته شده"""
        self.known_patterns[endpoint] = structure_type
        logger.info(f"🎯 Added known pattern: {endpoint} -> {structure_type.value}")

    def get_health_metrics(self) -> HealthMetrics:
        total_processed = self.metrics['total_processed']
        success_rate = (self.metrics['total_success'] / total_processed * 100) if total_processed > 0 else 0
        
        processing_times = self.metrics['processing_times']
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        quality_scores = self.metrics['quality_scores']
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        confidence_scores = self.metrics['confidence_scores']
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return HealthMetrics(
            success_rate=round(success_rate, 2),
            total_processed=total_processed,
            total_success=self.metrics['total_success'],
            total_errors=self.metrics['total_errors'],
            common_structures=self.metrics['structure_counts'],
            performance_metrics={
                'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
                'total_processing_time_ms': round(sum(processing_times) * 1000, 2),
                'requests_per_second': round(total_processed / (sum(processing_times) or 1), 2),
                'avg_confidence': round(avg_confidence, 2),
                'pattern_efficiency': f"{(self.metrics['pattern_matches'] / total_processed * 100) if total_processed > 0 else 0:.1f}%"
            },
            alerts=self.metrics['alerts'][-10:],
            data_quality={
                'avg_quality_score': round(avg_quality, 2),
                'completeness_score': round(success_rate, 2),
                'consistency_score': round(self._calculate_consistency_score(), 2)
            },
            endpoint_intelligence=self.get_endpoint_intelligence()
        )

    def _calculate_consistency_score(self) -> float:
        endpoint_patterns = self.metrics['endpoint_patterns']
        if not endpoint_patterns:
            return 0.0
            
        consistency_scores = []
        for endpoint, pattern in endpoint_patterns.items():
            if pattern['total_requests'] > 1:
                main_structure = max(pattern['structure_counts'].items(), key=lambda x: x[1])
                consistency = main_structure[1] / pattern['total_requests']
                consistency_scores.append(consistency)
                
        return sum(consistency_scores) / len(consistency_scores) * 100 if consistency_scores else 0.0

    def get_cache_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات کش"""
        cache_info = {
            'cache_path': str(self.cache_base_path),
            'total_size_mb': 0,
            'file_counts': {},
            'oldest_file': None,
            'newest_file': None
        }
        
        try:
            total_size = 0
            for dir_type, dir_path in self.cache_dirs.items():
                if dir_path.exists():
                    file_count = len(list(dir_path.rglob("*.json")))
                    cache_info['file_counts'][dir_type] = file_count
                    
                    for file_path in dir_path.rglob("*.json"):
                        total_size += file_path.stat().st_size
                        
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if (cache_info['oldest_file'] is None or 
                            file_time < datetime.fromisoformat(cache_info['oldest_file'])):
                            cache_info['oldest_file'] = file_time.isoformat()
                            
                        if (cache_info['newest_file'] is None or 
                            file_time > datetime.fromisoformat(cache_info['newest_file'])):
                            cache_info['newest_file'] = file_time.isoformat()
            
            cache_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache info: {e}")
            
        return cache_info

    def clear_cache(self, cache_type: str = None):
        """پاک‌سازی کش"""
        try:
            if cache_type and cache_type in self.cache_dirs:
                shutil.rmtree(self.cache_dirs[cache_type])
                self.cache_dirs[cache_type].mkdir()
                logger.info(f"🧹 Cleared {cache_type} cache")
            else:
                shutil.rmtree(self.cache_base_path)
                self._initialize_cache_system()
                logger.info("🧹 Cleared all cache")
                
        except Exception as e:
            logger.error(f"❌ Cache clear failed: {e}")

    def reset_metrics(self):
        self._reset_metrics()
        logger.info("🔄 Data Normalizer metrics reset")

# نمونه گلوبال
data_normalizer = DataNormalizer()

"""
🤖 Data Normalizer v2.1 - با سیستم ذخیره‌سازی فایل و مدیریت عمر داده
ویژگی‌های جدید:
- تشخیص 20 ساختار مختلف
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

# بقیه Enumها و dataclassها مانند قبل...
class NormalizationStrategy(Enum):
    SMART = "smart"
    STRICT = "strict"
    LENIENT = "lenient"
    COIN_STATS_OPTIMIZED = "coin_stats_optimized"

@dataclass
class NormalizationResult:
    status: str
    data: List[Any]
    metadata: Dict[str, Any]
    raw_data: Any
    normalization_info: Dict[str, Any]
    quality_score: float

@dataclass  
class HealthMetrics:
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
        self._initialize_cache_system()  # تغییر به سیستم فایل
        self._reset_metrics()
        
        # استراتژی پیش‌فرض
        self.default_strategy = NormalizationStrategy.COIN_STATS_OPTIMIZED
        
        # ساختارهای پشتیبانی شده (مانند قبل)
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

    # ========================== متدهای اصلی (با ذخیره‌سازی) ==========================

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
            self._persist_metrics()  # ذخیره متریک‌ها
            
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
                    "cache_used": False  # در این نسخه از کش فایل استفاده می‌شود
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

    # ========================== نرمال‌سازهای جدید برای CoinStats ==========================

    def _normalize_coin_stats_detailed(self, raw_data: Dict) -> List:
        """نرمال‌سازی داده‌های تک کوین با جزئیات"""
        if 'data' in raw_data and isinstance(raw_data['data'], dict):
            return [raw_data['data']]
        return [raw_data]

    def _normalize_coin_stats_chart(self, raw_data: Any) -> List:
        """نرمال‌سازی داده‌های چارتی"""
        if isinstance(raw_data, list):
            return raw_data
        elif isinstance(raw_data, dict) and 'result' in raw_data:
            return raw_data['result']
        return []

    def _normalize_coin_stats_markets(self, raw_data: Dict) -> List:
        """نرمال‌سازی داده‌های بازار"""
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_coin_stats_insights(self, raw_data: Dict) -> List:
        """نرمال‌سازی داده‌های تحلیلی"""
        if 'now' in raw_data:
            return [raw_data]  # ساختار Fear & Greed
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_coin_stats_error(self, raw_data: Dict) -> List:
        """نرمال‌سازی ساختار خطا"""
        return [raw_data]  # خطاها را به صورت لیست تک‌عنصری برمی‌گردانیم

    def _normalize_coin_stats_sentiment(self, raw_data: Dict) -> List:
        """نرمال‌سازی داده‌های احساسات"""
        return raw_data.get('data', raw_data.get('result', []))

    # بقیه متدها مانند قبل با بهبودهای جزئی...

    def _detect_structure_advanced(self, raw_data: Any, endpoint: str = "unknown") -> Tuple[StructureType, float, bool]:
        """تشخیص پیشرفته ساختار - نسخه بهبود یافته"""
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
                
        # بقیه منطق تشخیص مانند قبل...
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

    # متدهای کمکی موجود (بدون تغییر)
    def _find_nested_list(self, data: Dict, max_depth: int = 3) -> Optional[List]:
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

    def _update_endpoint_intelligence(self, endpoint: str, structure_type: StructureType, confidence: float, raw_data: Any):
        """به‌روزرسانی هوش endpoint با ذخیره‌سازی فایل"""
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

# نمونه گلوبال
data_normalizer = DataNormalizer()

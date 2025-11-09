"""
🤖 Data Normalizer v2 - سیستم هوشمند استانداردسازی داده‌های API
ویژگی‌های جدید:
- پشتیبانی از 15+ ساختار مختلف
- تشخیص الگوهای خاص CoinStats API
- آنالیز عمیق‌تر داده‌های تودرتو
- سیستم یادگیری خودکار endpointها
- fallbackهای هوشمندتر
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StructureType(Enum):
    """انواع ساختارهای شناسایی شده - نسخه پیشرفته"""
    # ساختارهای پایه
    DIRECT_LIST = "direct_list"
    SINGLE_ITEM_LIST = "single_item_list"  # جدید: لیست تک‌آیتم [{}]
    
    # ساختارهای دیکشنری با لیست
    DICT_WITH_DATA = "dict_with_data"
    DICT_WITH_RESULT = "dict_with_result" 
    DICT_WITH_ITEMS = "dict_with_items"
    DICT_WITH_COINS = "dict_with_coins"
    DICT_WITH_NEWS = "dict_with_news"  # جدید
    DICT_WITH_RESULTS = "dict_with_results"  # جدید
    
    # ساختارهای CoinStats API خاص
    COIN_STATS_PAGINATED = "coin_stats_paginated"  # {"result": [], "meta": {}}
    COIN_STATS_SINGLE_COIN = "coin_stats_single_coin"  # [{}] برای coins/bitcoin
    COIN_STATS_NEWS = "coin_stats_news"  # ساختار خاص اخبار
    
    # ساختارهای پیچیده
    NESTED_STRUCTURE = "nested_structure"  # داده‌های تودرتو
    PAGINATED_RESPONSE = "paginated_response"  # پاسخ صفحه‌بندی شده
    
    # fallback
    CUSTOM_STRUCTURE = "custom_structure"
    UNKNOWN = "unknown"

class NormalizationStrategy(Enum):
    """استراتژی‌های نرمال‌سازی"""
    SMART = "smart"
    STRICT = "strict"
    LENIENT = "lenient"
    COIN_STATS_OPTIMIZED = "coin_stats_optimized"  # جدید

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
    endpoint_intelligence: Dict[str, Any]  # جدید

class DataNormalizer:
    """
    سیستم هوشمند نرمال‌سازی داده‌های API - نسخه پیشرفته
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._setup_logging()
        self._initialize_cache()
        self._reset_metrics()
        
        # استراتژی پیش‌فرض
        self.default_strategy = NormalizationStrategy.COIN_STATS_OPTIMIZED
        
        # ساختارهای پشتیبانی شده
        self.supported_structures = {
            # ساختارهای پایه
            StructureType.DIRECT_LIST: self._normalize_direct_list,
            StructureType.SINGLE_ITEM_LIST: self._normalize_single_item_list,
            
            # ساختارهای دیکشنری
            StructureType.DICT_WITH_DATA: self._normalize_dict_with_data,
            StructureType.DICT_WITH_RESULT: self._normalize_dict_with_result,
            StructureType.DICT_WITH_ITEMS: self._normalize_dict_with_items,
            StructureType.DICT_WITH_COINS: self._normalize_dict_with_coins,
            StructureType.DICT_WITH_NEWS: self._normalize_dict_with_news,
            StructureType.DICT_WITH_RESULTS: self._normalize_dict_with_results,
            
            # ساختارهای CoinStats API
            StructureType.COIN_STATS_PAGINATED: self._normalize_coin_stats_paginated,
            StructureType.COIN_STATS_SINGLE_COIN: self._normalize_coin_stats_single_coin,
            StructureType.COIN_STATS_NEWS: self._normalize_coin_stats_news,
            
            # ساختارهای پیچیده
            StructureType.PAGINATED_RESPONSE: self._normalize_paginated_response,
            StructureType.NESTED_STRUCTURE: self._normalize_nested_structure,
        }
        
        # الگوهای شناخته شده برای endpointها
        self.known_patterns = {
            "coins/list": StructureType.COIN_STATS_PAGINATED,
            "coins/bitcoin": StructureType.SINGLE_ITEM_LIST,
            "coins/ethereum": StructureType.SINGLE_ITEM_LIST,
            "news/type/handpicked": StructureType.COIN_STATS_NEWS,
            "news/type/trending": StructureType.COIN_STATS_NEWS,
            "exchanges/list": StructureType.DICT_WITH_RESULT,
        }
        
        logger.info("🚀 Data Normalizer v2 Initialized - CoinStats Optimized")

    def _setup_logging(self):
        """تنظیمات لاگ‌گیری"""
        self.logger = logging.getLogger(__name__)

    def _initialize_cache(self):
        """راه‌اندازی کش"""
        self.structure_cache = {}
        self.health_cache = {}
        self.analysis_cache = {}
        self.pattern_cache = {}  # کش الگوها
        
        self.cache_ttl = {
            'structure': timedelta(days=7),
            'health': timedelta(hours=1),
            'analysis': timedelta(minutes=30),
            'patterns': timedelta(days=1),  # کش الگوها
        }

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
            'confidence_scores': [],  # جدید
            'pattern_matches': 0,  # جدید
        }

    def normalize(self, raw_data: Any, endpoint: str = "unknown", 
                 strategy: NormalizationStrategy = None) -> NormalizationResult:
        """
        نرمال‌سازی هوشمند داده‌های ورودی - نسخه پیشرفته
        """
        start_time = time.time()
        self.metrics['total_processed'] += 1
        
        try:
            # تشخیص ساختار با الگوی endpoint
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
            
            # محاسبه کیفیت پیشرفته
            quality_score = self._calculate_quality_score_advanced(normalized_data, structure_type, confidence)
            self.metrics['quality_scores'].append(quality_score)
            
            # یادگیری الگو
            self._update_endpoint_intelligence(endpoint, structure_type, confidence, raw_data)
            
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
                    "timestamp": datetime.now().isoformat()
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
        """
        تشخیص پیشرفته ساختار با الگوی endpoint
        """
        # اول بررسی الگوهای شناخته شده
        if endpoint in self.known_patterns:
            known_structure = self.known_patterns[endpoint]
            logger.debug(f"🎯 Using known pattern for {endpoint}: {known_structure.value}")
            return known_structure, 0.95, True
        
        # تشخیص بر اساس داده
        if isinstance(raw_data, list):
            if len(raw_data) == 1 and isinstance(raw_data[0], dict):
                # لیست تک‌آیتم - معمولاً برای coins/bitcoin
                return StructureType.SINGLE_ITEM_LIST, 0.92, False
            elif len(raw_data) > 0:
                return StructureType.DIRECT_LIST, 0.90, False
            else:
                return StructureType.DIRECT_LIST, 0.70, False
        
        elif isinstance(raw_data, dict):
            # ساختارهای CoinStats API
            if 'result' in raw_data and isinstance(raw_data['result'], list):
                if 'meta' in raw_data:
                    return StructureType.COIN_STATS_PAGINATED, 0.94, False
                else:
                    return StructureType.DICT_WITH_RESULT, 0.88, False
            
            # ساختارهای عمومی
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
            
            # تشخیص ساختارهای تودرتو
            nested_list = self._find_nested_list(raw_data)
            if nested_list:
                return StructureType.NESTED_STRUCTURE, 0.80, False
        
        # fallback به آنالیز پیشرفته
        return self._advanced_structure_analysis(raw_data), 0.5, False

    def _advanced_structure_analysis(self, raw_data: Any) -> StructureType:
        """آنالیز پیشرفته ساختار"""
        if isinstance(raw_data, dict):
            # شمارش لیست‌ها در سطوح مختلف
            list_count = self._count_lists_in_dict(raw_data)
            if list_count == 1:
                return StructureType.NESTED_STRUCTURE
            elif list_count > 1:
                return StructureType.CUSTOM_STRUCTURE
        
        return StructureType.UNKNOWN

    def _count_lists_in_dict(self, data: Dict, max_depth: int = 3) -> int:
        """شمارش لیست‌ها در دیکشنری"""
        def count_recursive(obj, depth=0):
            if depth >= max_depth:
                return 0
            
            count = 0
            if isinstance(obj, list):
                return 1
            elif isinstance(obj, dict):
                for value in obj.values():
                    count += count_recursive(value, depth + 1)
            return count
        
        return count_recursive(data)

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

    # ========================== نرمال‌سازهای جدید ==========================

    def _normalize_single_item_list(self, raw_data: List) -> List:
        """نرمال‌سازی لیست تک‌آیتم"""
        return raw_data  # لیست را مستقیماً برمی‌گردانیم

    def _normalize_dict_with_news(self, raw_data: Dict) -> List:
        """نرمال‌سازی دیکشنری با کلید news"""
        return raw_data.get('news', [])

    def _normalize_dict_with_results(self, raw_data: Dict) -> List:
        """نرمال‌سازی دیکشنری با کلید results"""
        return raw_data.get('results', [])

    def _normalize_coin_stats_paginated(self, raw_data: Dict) -> List:
        """نرمال‌سازی ساختار صفحه‌بندی شده CoinStats"""
        return raw_data.get('result', [])

    def _normalize_coin_stats_single_coin(self, raw_data: List) -> List:
        """نرمال‌سازی ساختار تک کوین CoinStats"""
        return raw_data  # [{}] را مستقیماً برمی‌گردانیم

    def _normalize_coin_stats_news(self, raw_data: Dict) -> List:
        """نرمال‌سازی ساختار اخبار CoinStats"""
        if 'result' in raw_data:
            return raw_data['result']
        elif 'news' in raw_data:
            return raw_data['news']
        else:
            return self._extract_data_from_complex_structure(raw_data)

    def _normalize_paginated_response(self, raw_data: Dict) -> List:
        """نرمال‌سازی پاسخ صفحه‌بندی شده"""
        return raw_data.get('data', raw_data.get('result', []))

    def _normalize_nested_structure(self, raw_data: Dict) -> List:
        """نرمال‌سازی ساختارهای تودرتو"""
        nested_list = self._find_nested_list(raw_data)
        return nested_list or []

    def _normalize_fallback_advanced(self, raw_data: Any) -> List:
        """Fallback پیشرفته"""
        # استراتژی‌های مختلف fallback
        if isinstance(raw_data, dict):
            # استراتژی 1: بزرگترین لیست را پیدا کن
            lists = [v for v in raw_data.values() if isinstance(v, list)]
            if lists:
                return max(lists, key=len)
            
            # استراتژی 2: اولین مقدار لیست را پیدا کن
            for value in raw_data.values():
                if isinstance(value, list):
                    return value
            
            # استراتژی 3: دیکشنری را به لیست تبدیل کن
            return [raw_data]
        
        elif isinstance(raw_data, list):
            return raw_data
        
        else:
            return [raw_data] if raw_data is not None else []

    # ========================== متدهای موجود (با بهبود) ==========================

    def _normalize_direct_list(self, raw_data: List) -> List:
        return raw_data

    def _normalize_dict_with_data(self, raw_data: Dict) -> List:
        return raw_data.get('data', [])

    def _normalize_dict_with_result(self, raw_data: Dict) -> List:
        return raw_data.get('result', [])

    def _normalize_dict_with_items(self, raw_data: Dict) -> List:
        return raw_data.get('items', [])

    def _normalize_dict_with_coins(self, raw_data: Dict) -> List:
        return raw_data.get('coins', [])

    def _extract_data_from_complex_structure(self, raw_data: Any) -> List:
        if isinstance(raw_data, dict):
            lists_in_dict = [v for v in raw_data.values() if isinstance(v, list)]
            if lists_in_dict:
                return max(lists_in_dict, key=len)
        return []

    def _extract_metadata_advanced(self, raw_data: Any, structure_type: StructureType) -> Dict[str, Any]:
        """استخراج متادیتا - نسخه پیشرفته"""
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
        """محاسبه امتیاز کیفیت - نسخه پیشرفته"""
        if not normalized_data:
            return 0.0
            
        score = 0.0
        
        # امتیاز بر اساس حجم داده
        data_count = len(normalized_data)
        if data_count > 0:
            score += min(data_count / 50, 0.3)  # حداکثر 30%
            
        # امتیاز بر اساس ساختار (ساختارهای جدید امتیاز بالاتر)
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
                'raw_data_samples': [],
                'last_detected': None,
                'pattern_stability': 0.0
            }
            
        pattern = self.metrics['endpoint_patterns'][endpoint]
        pattern['total_requests'] += 1
        pattern['structure_counts'][structure_type.value] = pattern['structure_counts'].get(structure_type.value, 0) + 1
        pattern['confidence_history'].append(confidence)
        pattern['last_detected'] = datetime.now().isoformat()
        
        # ذخیره نمونه داده برای آنالیز
        if pattern['total_requests'] <= 5:  # فقط 5 نمونه اول
            pattern['raw_data_samples'].append({
                'timestamp': datetime.now().isoformat(),
                'structure': structure_type.value,
                'data_preview': str(raw_data)[:200] + "..." if len(str(raw_data)) > 200 else str(raw_data)
            })
        
        # محاسبه پایداری الگو
        if pattern['total_requests'] > 1:
            main_structure_count = max(pattern['structure_counts'].values())
            pattern['pattern_stability'] = main_structure_count / pattern['total_requests']

    # ========================== متدهای عمومی جدید ==========================

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

    # بقیه متدها مانند قبل...
    def get_deep_analysis(self, raw_data: Any = None, endpoint: str = None) -> Dict[str, Any]:
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "total_requests": self.metrics['total_processed'],
                "success_rate": self.get_health_metrics().success_rate,
                "most_common_structure": max(
                    self.metrics['structure_counts'].items(), 
                    key=lambda x: x[1],
                    default=('unknown', 0)
                ),
                "avg_confidence": f"{sum(self.metrics['confidence_scores']) / len(self.metrics['confidence_scores']):.1f}%" if self.metrics['confidence_scores'] else "0%"
            },
            "endpoint_intelligence": self.get_endpoint_intelligence(),
            "structure_analysis": self.metrics['structure_counts'],
            "performance_analysis": {
                "avg_processing_time": f"{sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) * 1000:.2f}ms" if self.metrics['processing_times'] else "0ms",
                "pattern_efficiency": f"{(self.metrics['pattern_matches'] / self.metrics['total_processed'] * 100) if self.metrics['total_processed'] > 0 else 0:.1f}%"
            },
            "known_patterns": {k: v.value for k, v in self.known_patterns.items()},
            "alerts_and_warnings": self.metrics['alerts'][-20:],
            "recommendations": self._generate_recommendations_advanced()
        }
        
        if raw_data is not None:
            analysis["specific_data_analysis"] = self._analyze_specific_data_advanced(raw_data, endpoint)
            
        return analysis

    def _generate_recommendations_advanced(self) -> List[str]:
        recommendations = []
        metrics = self.get_health_metrics()
        
        if metrics.success_rate < 95:
            recommendations.append("🔄 نرخ موفقیت نرمال‌سازی پایین است. الگوهای جدید را بررسی کنید.")
            
        if metrics.performance_metrics['pattern_efficiency'] < '80%':
            recommendations.append("🎯 کارایی الگوها پایین است. endpointهای جدید را به known patterns اضافه کنید.")
            
        if metrics.data_quality['avg_quality_score'] < 80:
            recommendations.append("📊 کیفیت داده‌ها نیاز به بهبود دارد.")
            
        if not recommendations:
            recommendations.append("✅ سیستم در وضعیت مطلوب قرار دارد.")
            
        return recommendations

    def _analyze_specific_data_advanced(self, raw_data: Any, endpoint: str = None) -> Dict[str, Any]:
        structure_type, confidence, pattern_used = self._detect_structure_advanced(raw_data, endpoint or "analysis")
        
        return {
            "detected_structure": structure_type.value,
            "confidence": confidence,
            "pattern_used": pattern_used,
            "data_type": type(raw_data).__name__,
            "data_size": len(raw_data) if hasattr(raw_data, '__len__') else 'unknown',
            "structure_complexity": self._calculate_structure_complexity(raw_data),
            "sample_preview": str(raw_data)[:200] + "..." if len(str(raw_data)) > 200 else str(raw_data),
            "endpoint_context": endpoint,
        }

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

    def clear_cache(self, cache_type: str = None):
        if cache_type == 'structure' or cache_type is None:
            self.structure_cache.clear()
        if cache_type == 'health' or cache_type is None:
            self.health_cache.clear() 
        if cache_type == 'analysis' or cache_type is None:
            self.analysis_cache.clear()
        if cache_type == 'patterns' or cache_type is None:
            self.pattern_cache.clear()
            
        logger.info("🧹 Data Normalizer cache cleared")

    def reset_metrics(self):
        self._reset_metrics()
        logger.info("🔄 Data Normalizer metrics reset")

# نمونه گلوبال
data_normalizer = DataNormalizer()

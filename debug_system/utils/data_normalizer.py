"""
🤖 Data Normalizer - سیستم هوشمند استانداردسازی داده‌های API
ویژگی‌ها:
- تشخیص خودکار ساختار داده‌های ورودی
- تبدیل به فرمت استاندارد یکپارچه  
- حفظ داده‌های خام برای دیباگ
- ارائه متریک برای مانیتورینگ سلامت
- آنالیز عمیق برای دیباگ سیستم
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StructureType(Enum):
    """انواع ساختارهای شناسایی شده"""
    DIRECT_LIST = "direct_list"
    DICT_WITH_DATA = "dict_with_data"
    DICT_WITH_RESULT = "dict_with_result" 
    DICT_WITH_ITEMS = "dict_with_items"
    DICT_WITH_COINS = "dict_with_coins"
    CUSTOM_STRUCTURE = "custom_structure"
    UNKNOWN = "unknown"

class NormalizationStrategy(Enum):
    """استراتژی‌های نرمال‌سازی"""
    SMART = "smart"
    STRICT = "strict"
    LENIENT = "lenient"

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

class DataNormalizer:
    """
    سیستم هوشمند نرمال‌سازی داده‌های API
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._setup_logging()
        self._initialize_cache()
        self._reset_metrics()
        
        # استراتژی پیش‌فرض
        self.default_strategy = NormalizationStrategy.SMART
        
        # ساختارهای پشتیبانی شده
        self.supported_structures = {
            StructureType.DIRECT_LIST: self._normalize_direct_list,
            StructureType.DICT_WITH_DATA: self._normalize_dict_with_data,
            StructureType.DICT_WITH_RESULT: self._normalize_dict_with_result,
            StructureType.DICT_WITH_ITEMS: self._normalize_dict_with_items,
            StructureType.DICT_WITH_COINS: self._normalize_dict_with_coins,
        }
        
        logger.info("✅ Data Normalizer Initialized - Smart Mode Active")

    def _setup_logging(self):
        """تنظیمات لاگ‌گیری"""
        self.logger = logging.getLogger(__name__)

    def _initialize_cache(self):
        """راه‌اندازی کش"""
        self.structure_cache = {}  # کش ساختارهای کشف شده
        self.health_cache = {}     # کش متریک‌های سلامت
        self.analysis_cache = {}   # کش آنالیزها
        
        # تنظیمات عمر کش
        self.cache_ttl = {
            'structure': timedelta(days=7),      # 7 روز برای ساختارها
            'health': timedelta(hours=1),        # 1 ساعت برای سلامت
            'analysis': timedelta(minutes=30),   # 30 دقیقه برای آنالیز
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
            'alerts': []
        }

    def normalize(self, raw_data: Any, endpoint: str = "unknown", 
                 strategy: NormalizationStrategy = None) -> NormalizationResult:
        """
        نرمال‌سازی هوشمند داده‌های ورودی
        
        Args:
            raw_data: داده خام از API
            endpoint: نام endpoint برای الگوگیری
            strategy: استراتژی نرمال‌سازی
            
        Returns:
            NormalizationResult: نتیجه استاندارد شده
        """
        start_time = time.time()
        self.metrics['total_processed'] += 1
        
        try:
            # تشخیص ساختار داده
            structure_type, confidence = self._detect_structure(raw_data)
            self.metrics['structure_counts'][structure_type.value] += 1
            
            # نرمال‌سازی بر اساس ساختار تشخیص داده شده
            normalization_func = self.supported_structures.get(
                structure_type, 
                self._normalize_fallback
            )
            
            normalized_data = normalization_func(raw_data)
            
            # محاسبه کیفیت داده
            quality_score = self._calculate_quality_score(normalized_data, structure_type)
            self.metrics['quality_scores'].append(quality_score)
            
            # ذخیره الگوی endpoint
            self._update_endpoint_pattern(endpoint, structure_type, confidence)
            
            # ثبت زمان پردازش
            processing_time = time.time() - start_time
            self.metrics['processing_times'].append(processing_time)
            
            self.metrics['total_success'] += 1
            
            result = NormalizationResult(
                status="success",
                data=normalized_data,
                metadata=self._extract_metadata(raw_data, structure_type),
                raw_data=raw_data,
                normalization_info={
                    "detected_structure": structure_type.value,
                    "confidence": confidence,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "endpoint": endpoint,
                    "strategy": (strategy or self.default_strategy).value,
                    "timestamp": datetime.now().isoformat()
                },
                quality_score=quality_score
            )
            
            logger.info(f"✅ Normalized {endpoint} - Structure: {structure_type.value} - Quality: {quality_score}%")
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
                    "timestamp": datetime.now().isoformat()
                },
                quality_score=0.0
            )

    def _detect_structure(self, raw_data: Any) -> tuple[StructureType, float]:
        """
        تشخیص هوشمند ساختار داده
        
        Returns:
            tuple: (نوع ساختار, میزان اطمینان)
        """
        if isinstance(raw_data, list):
            return StructureType.DIRECT_LIST, 0.95
            
        elif isinstance(raw_data, dict):
            # بررسی کلیدهای مختلف
            if 'data' in raw_data and isinstance(raw_data['data'], list):
                return StructureType.DICT_WITH_DATA, 0.90
            elif 'result' in raw_data and isinstance(raw_data['result'], list):
                return StructureType.DICT_WITH_RESULT, 0.85
            elif 'items' in raw_data and isinstance(raw_data['items'], list):
                return StructureType.DICT_WITH_ITEMS, 0.80
            elif 'coins' in raw_data and isinstance(raw_data['coins'], list):
                return StructureType.DICT_WITH_COINS, 0.75
            else:
                # آنالیز عمیق‌تر برای ساختارهای سفارسی
                return self._analyze_custom_structure(raw_data)
        else:
            return StructureType.UNKNOWN, 0.1

    def _analyze_custom_structure(self, raw_data: Dict) -> tuple[StructureType, float]:
        """آنالیز ساختارهای سفارسی"""
        # جستجو برای لیست در سطوح مختلف
        for key, value in raw_data.items():
            if isinstance(value, list) and len(value) > 0:
                # بررسی اگر آیتم‌های لیست دیکشنری هستند (داده ساختاریافته)
                if all(isinstance(item, dict) for item in value):
                    return StructureType.CUSTOM_STRUCTURE, 0.7
                    
        return StructureType.UNKNOWN, 0.3

    def _normalize_direct_list(self, raw_data: List) -> List:
        """نرمال‌سازی لیست مستقیم"""
        return raw_data

    def _normalize_dict_with_data(self, raw_data: Dict) -> List:
        """نرمال‌سازی دیکشنری با کلید data"""
        return raw_data.get('data', [])

    def _normalize_dict_with_result(self, raw_data: Dict) -> List:
        """نرمال‌سازی دیکشنری با کلید result"""
        return raw_data.get('result', [])

    def _normalize_dict_with_items(self, raw_data: Dict) -> List:
        """نرمال‌سازی دیکشنری با کلید items"""
        return raw_data.get('items', [])

    def _normalize_dict_with_coins(self, raw_data: Dict) -> List:
        """نرمال‌سازی دیکشنری با کلید coins"""
        return raw_data.get('coins', [])

    def _normalize_fallback(self, raw_data: Any) -> List:
        """نرمال‌سازی fallback برای ساختارهای ناشناخته"""
        if isinstance(raw_data, (list, dict)):
            # تلاش برای استخراج داده از ساختارهای پیچیده
            return self._extract_data_from_complex_structure(raw_data)
        else:
            # تبدیل به لیست
            return [raw_data] if raw_data is not None else []

    def _extract_data_from_complex_structure(self, raw_data: Any) -> List:
        """استخراج داده از ساختارهای پیچیده"""
        if isinstance(raw_data, dict):
            # جستجو برای هر کلیدی که لیست باشد
            lists_in_dict = [v for v in raw_data.values() if isinstance(v, list)]
            if lists_in_dict:
                # بازگوردن بزرگترین لیست
                return max(lists_in_dict, key=len)
        return []

    def _extract_metadata(self, raw_data: Any, structure_type: StructureType) -> Dict[str, Any]:
        """استخراج متادیتا از داده خام"""
        metadata = {
            "structure_type": structure_type.value,
            "extracted_at": datetime.now().isoformat(),
            "data_source": "coinstats_api"
        }
        
        if isinstance(raw_data, dict):
            # استخراج متادیتاهای رایج
            common_meta_keys = ['meta', 'metadata', 'pagination', 'info', 'total', 'count']
            for key in common_meta_keys:
                if key in raw_data:
                    metadata[key] = raw_data[key]
                    
        return metadata

    def _calculate_quality_score(self, normalized_data: List, structure_type: StructureType) -> float:
        """محاسبه امتیاز کیفیت داده"""
        if not normalized_data:
            return 0.0
            
        score = 0.0
        
        # امتیاز بر اساس تعداد داده
        data_count = len(normalized_data)
        if data_count > 0:
            score += min(data_count / 100, 0.3)  # حداکثر 30% برای حجم داده
            
        # امتیاز بر اساس ساختار
        structure_scores = {
            StructureType.DIRECT_LIST: 0.2,
            StructureType.DICT_WITH_DATA: 0.25,
            StructureType.DICT_WITH_RESULT: 0.25,
            StructureType.DICT_WITH_ITEMS: 0.2,
            StructureType.DICT_WITH_COINS: 0.2,
            StructureType.CUSTOM_STRUCTURE: 0.15,
            StructureType.UNKNOWN: 0.1
        }
        score += structure_scores.get(structure_type, 0.1)
        
        # امتیاز بر اساس یکنواختی داده
        if data_count > 1:
            uniformity_score = self._calculate_uniformity_score(normalized_data)
            score += uniformity_score * 0.5
            
        return min(score * 100, 100.0)  # تبدیل به درصد

    def _calculate_uniformity_score(self, data: List) -> float:
        """محاسبه امتیاز یکنواختی داده‌ها"""
        if not data or len(data) < 2:
            return 0.5
            
        try:
            # بررسی یکنواختی کلیدها در آیتم‌ها (اگر دیکشنری هستند)
            if all(isinstance(item, dict) for item in data):
                first_keys = set(data[0].keys())
                common_keys = first_keys.intersection(*(set(item.keys()) for item in data[1:]))
                return len(common_keys) / len(first_keys) if first_keys else 0.5
        except:
            pass
            
        return 0.5

    def _update_endpoint_pattern(self, endpoint: str, structure_type: StructureType, confidence: float):
        """به روزرسانی الگوهای endpoint"""
        if endpoint not in self.metrics['endpoint_patterns']:
            self.metrics['endpoint_patterns'][endpoint] = {
                'total_requests': 0,
                'structure_counts': {},
                'last_detected': None,
                'confidence_avg': 0.0
            }
            
        pattern = self.metrics['endpoint_patterns'][endpoint]
        pattern['total_requests'] += 1
        pattern['structure_counts'][structure_type.value] = pattern['structure_counts'].get(structure_type.value, 0) + 1
        pattern['last_detected'] = datetime.now().isoformat()
        
        # محاسبه میانگین اطمینان
        current_avg = pattern['confidence_avg']
        total_reqs = pattern['total_requests']
        pattern['confidence_avg'] = (current_avg * (total_reqs - 1) + confidence) / total_reqs

    # ========================== PUBLIC METHODS FOR EXTERNAL USE ==========================

    def get_health_metrics(self) -> HealthMetrics:
        """دریافت متریک‌های سلامت برای سیستم مانیتورینگ"""
        total_processed = self.metrics['total_processed']
        success_rate = (self.metrics['total_success'] / total_processed * 100) if total_processed > 0 else 0
        
        # محاسبه متریک‌های عملکرد
        processing_times = self.metrics['processing_times']
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # محاسبه کیفیت داده
        quality_scores = self.metrics['quality_scores']
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return HealthMetrics(
            success_rate=round(success_rate, 2),
            total_processed=total_processed,
            total_success=self.metrics['total_success'],
            total_errors=self.metrics['total_errors'],
            common_structures=self.metrics['structure_counts'],
            performance_metrics={
                'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
                'total_processing_time_ms': round(sum(processing_times) * 1000, 2),
                'requests_per_second': round(total_processed / (sum(processing_times) or 1), 2)
            },
            alerts=self.metrics['alerts'][-10:],  # 10 هشدار آخر
            data_quality={
                'avg_quality_score': round(avg_quality, 2),
                'completeness_score': round(success_rate, 2),
                'consistency_score': round(self._calculate_consistency_score(), 2)
            }
        )

    def get_deep_analysis(self, raw_data: Any = None, endpoint: str = None) -> Dict[str, Any]:
        """
        آنالیز عمیق برای سیستم دیباگ
        
        Args:
            raw_data: داده برای آنالیز (اختیاری)
            endpoint: endpoint برای آنالیز (اختیاری)
            
        Returns:
            Dict: گزارش تحلیل کامل
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "total_requests": self.metrics['total_processed'],
                "success_rate": self.get_health_metrics().success_rate,
                "most_common_structure": max(
                    self.metrics['structure_counts'].items(), 
                    key=lambda x: x[1],
                    default=('unknown', 0)
                )
            },
            "endpoint_patterns": self.metrics['endpoint_patterns'],
            "structure_analysis": self.metrics['structure_counts'],
            "performance_analysis": {
                "avg_processing_time": f"{sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) * 1000:.2f}ms" 
                if self.metrics['processing_times'] else "0ms",
                "total_processing_time": f"{sum(self.metrics['processing_times']) * 1000:.2f}ms",
                "fastest_processing": f"{min(self.metrics['processing_times']) * 1000:.2f}ms" 
                if self.metrics['processing_times'] else "0ms",
                "slowest_processing": f"{max(self.metrics['processing_times']) * 1000:.2f}ms" 
                if self.metrics['processing_times'] else "0ms"
            },
            "quality_analysis": {
                "avg_quality_score": f"{sum(self.metrics['quality_scores']) / len(self.metrics['quality_scores']):.2f}%"
                if self.metrics['quality_scores'] else "0%",
                "quality_trend": "stable" if len(self.metrics['quality_scores']) < 2 else
                "improving" if self.metrics['quality_scores'][-1] > self.metrics['quality_scores'][0] else "declining"
            },
            "alerts_and_warnings": self.metrics['alerts'][-20:],  # 20 هشدار آخر
            "recommendations": self._generate_recommendations()
        }
        
        # آنالیز داده خاص اگر ارائه شده
        if raw_data is not None:
            analysis["specific_data_analysis"] = self._analyze_specific_data(raw_data, endpoint)
            
        return analysis

    def _calculate_consistency_score(self) -> float:
        """محاسبه امتیاز ثبات"""
        endpoint_patterns = self.metrics['endpoint_patterns']
        if not endpoint_patterns:
            return 0.0
            
        consistency_scores = []
        for endpoint, pattern in endpoint_patterns.items():
            if pattern['total_requests'] > 1:
                # هرچه الگوی ساختاری ثابت‌تر باشد، امتیاز更高
                main_structure = max(pattern['structure_counts'].items(), key=lambda x: x[1])
                consistency = main_structure[1] / pattern['total_requests']
                consistency_scores.append(consistency)
                
        return sum(consistency_scores) / len(consistency_scores) * 100 if consistency_scores else 0.0

    def _analyze_specific_data(self, raw_data: Any, endpoint: str = None) -> Dict[str, Any]:
        """آنالیز داده خاص"""
        structure_type, confidence = self._detect_structure(raw_data)
        
        return {
            "detected_structure": structure_type.value,
            "confidence": confidence,
            "data_type": type(raw_data).__name__,
            "data_size": len(raw_data) if hasattr(raw_data, '__len__') else 'unknown',
            "sample_preview": str(raw_data)[:200] + "..." if len(str(raw_data)) > 200 else str(raw_data),
            "endpoint_context": endpoint,
            "normalization_preview": self.normalize(raw_data, endpoint or "analysis").normalization_info
        }

    def _generate_recommendations(self) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        metrics = self.get_health_metrics()
        
        if metrics.success_rate < 95:
            recommendations.append("🔄 نرخ موفقیت نرمال‌سازی پایین است. ساختارهای جدید را بررسی کنید.")
            
        if metrics.total_errors > 10:
            recommendations.append("🐛 خطاهای نرمال‌سازی افزایش یافته. لاگ‌ها را بررسی کنید.")
            
        if metrics.data_quality['avg_quality_score'] < 80:
            recommendations.append("📊 کیفیت داده‌ها نیاز به بهبود دارد. الگوهای داده را آنالیز کنید.")
            
        if not recommendations:
            recommendations.append("✅ سیستم در وضعیت مطلوب قرار دارد.")
            
        return recommendations

    def clear_cache(self, cache_type: str = None):
        """پاک‌سازی کش"""
        if cache_type == 'structure' or cache_type is None:
            self.structure_cache.clear()
        if cache_type == 'health' or cache_type is None:
            self.health_cache.clear() 
        if cache_type == 'analysis' or cache_type is None:
            self.analysis_cache.clear()
            
        logger.info("🧹 Data Normalizer cache cleared")

    def reset_metrics(self):
        """بازنشانی متریک‌ها (برای تست و توسعه)"""
        self._reset_metrics()
        logger.info("🔄 Data Normalizer metrics reset")

# نمونه گلوبال برای استفاده آسان
data_normalizer = DataNormalizer()

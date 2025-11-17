import re
import time
import logging
from typing import Dict, List, Any, Set, Tuple
from collections import Counter, defaultdict
import heapq

logger = logging.getLogger(__name__)

class KnowledgeCompressor:
    """فشرده‌ساز هوشمند دانش برای بهینه‌سازی فضای حافظه"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_threshold = config.get('compression_threshold', 0.8)
        self.min_importance_to_keep = config.get('min_importance_to_keep', 0.3)
        
        # الگوهای فشرده‌سازی
        self.concept_patterns = defaultdict(int)
        self.redundant_data_cache = set()
        
        # آمار فشرده‌سازی
        self.compression_stats = {
            'total_compressions': 0,
            'space_saved_mb': 0.0,
            'last_compression_time': 0
        }
        
        logger.info("🚀 فشرده‌ساز دانش راه‌اندازی شد")
    
    def compress_knowledge(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """فشرده‌سازی داده‌های دانش"""
        if not knowledge_data:
            return {}
        
        original_size = self._calculate_data_size(knowledge_data)
        
        # فشرده‌سازی بر اساس نوع داده
        compressed_data = {}
        
        for key, value in knowledge_data.items():
            if isinstance(value, str):
                compressed_data[key] = self._compress_text(value)
            elif isinstance(value, dict):
                compressed_data[key] = self._compress_dict(value)
            elif isinstance(value, list):
                compressed_data[key] = self._compress_list(value)
            else:
                compressed_data[key] = value
        
        # حذف داده‌های کم اهمیت
        compressed_data = self._remove_low_importance_data(compressed_data)
        
        # به‌روزرسانی آمار
        compressed_size = self._calculate_data_size(compressed_data)
        space_saved = original_size - compressed_size
        
        if space_saved > 0:
            self.compression_stats['total_compressions'] += 1
            self.compression_stats['space_saved_mb'] += space_saved / (1024 * 1024)
            self.compression_stats['last_compression_time'] = time.time()
            
            logger.info(f"📦 فشرده‌سازی دانش: {original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB")
        
        return compressed_data
    
    def _compress_text(self, text: str) -> str:
        """فشرده‌سازی متن"""
        if len(text) < 100:
            return text
        
        # حذف فضاهای اضافی
        text = re.sub(r'\s+', ' ', text.strip())
        
        # شناسایی و جایگزینی الگوهای تکراری
        text = self._replace_patterns(text)
        
        # کوتاه کردن متن‌های بسیار طولانی
        if len(text) > 500:
            sentences = text.split('.')
            if len(sentences) > 3:
                # حفظ ۳ جمله اول و آخر
                compressed = '.'.join(sentences[:2] + ['...'] + sentences[-2:])
                return compressed
        
        return text
    
    def _compress_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """فشرده‌سازی دیکشنری"""
        compressed = {}
        
        for key, value in data_dict.items():
            # حفظ کلیدهای مهم
            if self._is_important_key(key):
                compressed[key] = value
            elif isinstance(value, (str, dict, list)):
                # فشرده‌سازی مقادیر پیچیده
                compressed_val = self.compress_knowledge({key: value}).get(key, value)
                if self._should_keep_data(key, compressed_val):
                    compressed[key] = compressed_val
            else:
                # حفظ مقادیر ساده
                compressed[key] = value
        
        return compressed
    
    def _compress_list(self, data_list: List[Any]) -> List[Any]:
        """فشرده‌سازی لیست"""
        if not data_list:
            return []
        
        # برای لیست‌های کوچک، فشرده‌سازی لازم نیست
        if len(data_list) <= 10:
            return data_list
        
        # فشرده‌سازی آیتم‌های لیست
        compressed_list = []
        for item in data_list:
            if isinstance(item, (dict, list)):
                compressed_item = self.compress_knowledge({'item': item}).get('item', item)
                compressed_list.append(compressed_item)
            else:
                compressed_list.append(item)
        
        # اگر هنوز لیست بزرگ است، نمونه‌گیری انجام بده
        if len(compressed_list) > 20:
            # حفظ اولین، آخرین و چند آیتم میانی
            sampled_list = (
                compressed_list[:5] + 
                [f"...({len(compressed_list)-10} موارد)"] + 
                compressed_list[-5:]
            )
            return sampled_list
        
        return compressed_list
    
    def _replace_patterns(self, text: str) -> str:
        """جایگزینی الگوهای تکراری در متن"""
        # الگوهای رایج در سوالات کاربر
        patterns = {
            r'قیمت\s+(بیتکوین|اتریوم|bitcoin|ethereum)': 'قیمت_ارز',
            r'لیست\s+(\d+)\s+ارز': 'لیست_ارز',
            r'وضعیت\s+سیستم': 'سلامت_سیستم',
            r'اخبار\s+(جدید|تازه|آخرین)': 'اخبار_جدید',
        }
        
        for pattern, replacement in patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _is_important_key(self, key: str) -> bool:
        """بررسی اهمیت کلید"""
        important_keys = {
            'intent', 'concept', 'pattern', 'essential', 'core', 'mastery',
            'timestamp', 'type', 'user_id', 'success', 'confidence'
        }
        
        return key in important_keys or any(imp in key.lower() for imp in important_keys)
    
    def _should_keep_data(self, key: str, value: Any) -> bool:
        """تصمیم‌گیری برای نگهداری یا حذف داده"""
        # محاسبه اهمیت داده
        importance_score = self._calculate_importance(key, value)
        
        # بررسی آستانه نگهداری
        return importance_score >= self.min_importance_to_keep
    
    def _calculate_importance(self, key: str, value: Any) -> float:
        """محاسبه اهمیت داده"""
        importance = 0.0
        
        # اهمیت بر اساس نوع کلید
        if self._is_important_key(key):
            importance += 0.5
        
        # اهمیت بر اساس نوع مقدار
        if isinstance(value, str):
            if len(value) > 50:  # متن‌های طولانی اهمیت بیشتری دارند
                importance += 0.2
        elif isinstance(value, (int, float)):
            importance += 0.1  # اعداد اهمیت پایین‌تری دارند
        elif isinstance(value, (dict, list)):
            if len(str(value)) > 100:  # ساختارهای پیچیده
                importance += 0.3
        
        # اهمیت بر اساس فرکانس استفاده (اگر موجود باشد)
        if hasattr(value, 'get') and callable(getattr(value, 'get')):
            access_count = value.get('access_count', 0)
            importance += min(access_count * 0.1, 0.5)
        
        return min(importance, 1.0)
    
    def _remove_low_importance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """حذف داده‌های کم اهمیت"""
        important_data = {}
        
        for key, value in data.items():
            if self._should_keep_data(key, value):
                important_data[key] = value
            else:
                logger.debug(f"🗑️ حذف داده کم اهمیت: {key}")
        
        removed_count = len(data) - len(important_data)
        if removed_count > 0:
            logger.info(f"🧹 حذف {removed_count} داده کم اهمیت")
        
        return important_data
    
    def extract_core_concepts(self, knowledge_data: Dict[str, Any]) -> Set[str]:
        """استخراج مفاهیم اصلی از داده‌های دانش"""
        concepts = set()
        
        for key, value in knowledge_data.items():
            # استخراج از کلیدها
            key_concepts = self._extract_concepts_from_text(str(key))
            concepts.update(key_concepts)
            
            # استخراج از مقادیر متنی
            if isinstance(value, str):
                value_concepts = self._extract_concepts_from_text(value)
                concepts.update(value_concepts)
            
            # استخراج بازگشتی از ساختارهای تو در تو
            elif isinstance(value, dict):
                nested_concepts = self.extract_core_concepts(value)
                concepts.update(nested_concepts)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested_concepts = self.extract_core_concepts(item)
                        concepts.update(nested_concepts)
        
        return concepts
    
    def _extract_concepts_from_text(self, text: str) -> Set[str]:
        """استخراج مفاهیم از متن"""
        if not isinstance(text, str):
            return set()
        
        # حذف علائم نگارشی و تبدیل به حروف کوچک
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # فیلتر کلمات کوتاه و عمومی
        concepts = set()
        for word in words:
            if (len(word) >= 3 and 
                not word.isdigit() and 
                word not in self._get_common_words()):
                concepts.add(word)
        
        return concepts
    
    def _get_common_words(self) -> Set[str]:
        """کلمات عمومی و کم اهمیت"""
        return {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'and', 'or', 'but', 'not', 'this',
            'that', 'these', 'those', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'
        }
    
    def _calculate_data_size(self, data: Any) -> int:
        """محاسبه اندازه داده به بایت"""
        if data is None:
            return 0
        
        import sys
        return sys.getsizeof(str(data))
    
    def optimize_memory_layout(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """بهینه‌سازی چیدمان حافظه برای دسترسی سریع‌تر"""
        if not memory_data:
            return {}
        
        # مرتب‌سازی بر اساس فرکانس دسترسی (اگر موجود باشد)
        sorted_data = {}
        
        for key, value in memory_data.items():
            # محاسبه امتیاز دسترسی
            access_score = value.get('access_count', 0)
            importance_score = value.get('importance', 0.1)
            total_score = access_score + (importance_score * 10)
            
            sorted_data[key] = (total_score, value)
        
        # مرتب‌سازی نزولی بر اساس امتیاز
        sorted_items = sorted(sorted_data.items(), key=lambda x: x[1][0], reverse=True)
        
        # ایجاد دیکشنری بهینه‌شده
        optimized_data = {}
        for key, (score, value) in sorted_items:
            optimized_data[key] = value
        
        logger.debug(f"🔧 بهینه‌سازی چیدمان {len(optimized_data)} آیتم")
        return optimized_data
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """آمار فشرده‌سازی"""
        return {
            'total_compressions': self.compression_stats['total_compressions'],
            'total_space_saved_mb': round(self.compression_stats['space_saved_mb'], 2),
            'last_compression_time': self.compression_stats['last_compression_time'],
            'compression_threshold': self.compression_threshold,
            'min_importance_to_keep': self.min_importance_to_keep
        }

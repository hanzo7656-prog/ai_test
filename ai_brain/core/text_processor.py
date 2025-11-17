import re
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import string

logger = logging.getLogger(__name__)

class TextProcessor:
    """پردازشگر متن چندزبانه ساده برای شبکه عصبی"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.max_vocab_size = config.get('max_vocab_size', 2000)
        
        # کلمات توقف پایه (فارسی + انگلیسی)
        self.stop_words = self._initialize_stop_words()
        
        # الگوهای تشخیص intent پایه
        self.intent_patterns = self._initialize_intent_patterns()
        
        logger.info("🚀 پردازشگر متن چندزبانه راه‌اندازی شد")
    
    def _initialize_stop_words(self) -> set:
        """مقداردهی اولیه کلمات توقف"""
        persian_stop_words = {
            'در', 'با', 'به', 'از', 'که', 'را', 'این', 'آن', 'و', 'برای',
            'تا', 'است', 'بود', 'شد', 'های', 'ترین', 'تر', 'میشود', 'شود',
            'هایش', 'اند', 'کرد', 'کردن', 'کنید', 'گیری', 'گیریی', 'ها'
        }
        
        english_stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'not'
        }
        
        return persian_stop_words.union(english_stop_words)
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """الگوهای تشخیص نیات پایه"""
        return {
            'health_check': [
                r'سلامت', r'وضعیت', r'status', r'health', r'چطوره', r'کار میکنه',
                r'سیستم', r'system'
            ],
            'price_request': [
                r'قیمت', r'نرخ', r'price', r'value', r'cost', r'چنده',
                r'چقدره', r'بیتکوین', r'اتریوم', r'bitcoin', r'ethereum'
            ],
            'news_request': [
                r'اخبار', r'خبر', r'news', r'تازه', r'جدید', r'latest',
                r'آپدیت', r'update'
            ],
            'list_request': [
                r'لیست', r'list', r'نمایش', r'show', r'همه', r'all',
                r'ارزها', r'coins', r'نمادها'
            ],
            'cache_status': [
                r'کش', r'cache', r'حافظه', r'memory', r'ذخیره', r'storage'
            ],
            'fear_greed': [
                r'ترس', r'طمع', r'fear', r'greed', r'شاخص', r'index',
                r'احساسات', r'sentiment'
            ]
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """پیش‌پردازش متن و استخراج توکن‌های معنادار"""
        if not text or not isinstance(text, str):
            return []
        
        # نرمال‌سازی متن
        text = self._normalize_text(text)
        
        # تجزیه به کلمات
        words = self._tokenize(text)
        
        # فیلتر کردن کلمات توقف و کوتاه
        filtered_words = [
            word for word in words 
            if (word not in self.stop_words and 
                len(word) > 1 and 
                not word.isdigit())
        ]
        
        logger.debug(f"🔤 پردازش متن: '{text}' → {len(filtered_words)} توکن")
        return filtered_words
    
    def _normalize_text(self, text: str) -> str:
        """نرمال‌سازی متن"""
        # حذف علائم نگارشی غیرضروری
        text = re.sub(r'[!؟?،,;؛]', ' ', text)
        
        # نرمال‌سازی فاصله‌ها
        text = re.sub(r'\s+', ' ', text)
        
        # تبدیل به حروف کوچک (برای انگلیسی)
        text = text.lower()
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """توکنایز کردن متن با پشتیبانی از فارسی و انگلیسی"""
        # الگوی ساده برای جدا کردن کلمات فارسی و انگلیسی
        tokens = re.findall(r'[a-zA-Z]+|[\u0600-\u06FF]+|[0-9]+', text)
        return tokens
    
    def text_to_vector(self, tokens: List[str], vector_size: int = 1000) -> np.ndarray:
        """تبدیل توکن‌ها به بردار برای شبکه عصبی"""
        vector = np.zeros(vector_size)
        
        if not tokens:
            return vector
        
        # ایجاد/به‌روزرسانی دایره واژگان
        self._update_vocab(tokens)
        
        # توزیع یکنواخت توکن‌ها در فضای بردار
        for token in tokens:
            # هش ساده برای توزیع یکنواخت
            hash_val = hash(token) % vector_size
            vector[hash_val] += 1
        
        # نرمال‌سازی
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector
    
    def _update_vocab(self, tokens: List[str]):
        """به‌روزرسانی دایره واژگان"""
        for token in tokens:
            if token not in self.vocab and self.vocab_size < self.max_vocab_size:
                self.vocab[token] = self.vocab_size
                self.reverse_vocab[self.vocab_size] = token
                self.vocab_size += 1
    
    def detect_intent(self, text: str) -> Tuple[str, float]:
        """تشخیص نیات از متن"""
        tokens = self.preprocess_text(text)
        text_lower = text.lower()
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            
            # امتیاز بر اساس تعداد توکن‌های مرتبط
            for token in tokens:
                if any(pattern in token for pattern in patterns if len(pattern) > 2):
                    score += 0.5
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1] / 5.0, 1.0)  # نرمال‌سازی به 0-1
            return best_intent[0], confidence
        
        return 'unknown', 0.0
    
    def extract_parameters(self, text: str, intent: str) -> Dict[str, Any]:
        """استخراج پارامترها از متن"""
        params = {}
        tokens = self.preprocess_text(text)
        text_lower = text.lower()
        
        # استخراج اعداد
        numbers = re.findall(r'\d+', text)
        if numbers:
            params['limit'] = int(numbers[0])
        
        # تشخیص نوع مرتب‌سازی
        if any(word in text_lower for word in ['قیمت', 'price', 'نرخ']):
            params['sort_by'] = 'price'
        elif any(word in text_lower for word in ['حجم', 'volume']):
            params['sort_by'] = 'volume'
        elif any(word in text_lower for word in ['ارزش', 'market', 'مارکت']):
            params['sort_by'] = 'marketCap'
        
        # تشخیص جهت مرتب‌سازی
        if any(word in text_lower for word in ['نزولی', 'desc', 'کم']):
            params['sort_dir'] = 'desc'
        else:
            params['sort_dir'] = 'asc'
        
        logger.debug(f"🎯 پارامترهای استخراج شده: {params}")
        return params
    
    def estimate_complexity(self, text: str) -> int:
        """تخمین پیچیدگی متن برای بررسی ظرفیت پردازش"""
        tokens = self.preprocess_text(text)
        
        # پیچیدگی بر اساس تعداد توکن‌های منحصر به فرد
        unique_tokens = len(set(tokens))
        
        # جریمه برای متن‌های طولانی
        length_penalty = max(0, len(tokens) - 10) * 0.5
        
        complexity = unique_tokens + length_penalty
        
        logger.debug(f"📊 پیچیدگی متن: {complexity} (توکن‌ها: {len(tokens)})")
        return int(complexity)
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """آمار پردازشگر"""
        return {
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'known_intents': len(self.intent_patterns),
            'stop_words_count': len(self.stop_words)
        }

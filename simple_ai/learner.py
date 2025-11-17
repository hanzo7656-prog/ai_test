import numpy as np
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import hashlib

class AILearningEngine:
    """موتور یادگیری و پردازش متن هوش مصنوعی"""
    
    def __init__(self):
        self.processed_files = 0
        self.vocabulary = set()
        self.patterns_learned = {}
        self.learning_stats = {
            'total_words_processed': 0,
            'unique_words_found': 0,
            'patterns_detected': 0,
            'last_processed': None
        }
    
    def process_text_file(self, file_content: str) -> np.ndarray:
        """پردازش فایل متنی و تبدیل به بردار عددی"""
        try:
            # پیش‌پردازش متن
            cleaned_text = self._clean_text(file_content)
            
            # استخراج کلمات و الگوها
            words = self._extract_words(cleaned_text)
            patterns = self._detect_patterns(cleaned_text)
            
            # به‌روزرسانی آمار
            self._update_learning_stats(words, patterns)
            
            # تبدیل به بردار عددی
            vector = self._text_to_vector(cleaned_text)
            
            self.processed_files += 1
            self.learning_stats['last_processed'] = datetime.now().isoformat()
            
            print(f"✅ Processed file #{self.processed_files} - {len(words)} words, {len(patterns)} patterns")
            
            return vector
            
        except Exception as e:
            print(f"❌ Error processing text file: {e}")
            return np.zeros(1000)  # بردار پیش‌فرض
    
    def _clean_text(self, text: str) -> str:
        """پاک‌سازی و نرمال‌سازی متن"""
        # حذف کاراکترهای خاص
        text = re.sub(r'[^\w\s\.\,\!\\?]', '', text)
        
        # تبدیل به حروف کوچک
        text = text.lower()
        
        # حذف فاصله‌های اضافی
        text = ' '.join(text.split())
        
        return text
    
    def _extract_words(self, text: str) -> List[str]:
        """استخراج کلمات از متن"""
        words = re.findall(r'\b[\w+]+\b', text)
        
        # به‌روزرسانی دایره واژگان
        new_words = set(words) - self.vocabulary
        self.vocabulary.update(new_words)
        
        return words
    
    def _detect_patterns(self, text: str) -> Dict[str, int]:
        """تشخیص الگوهای تکرارشونده در متن"""
        patterns = {}
        
        # الگوهای ساده - کلمات پرتکرار
        words = text.split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # فقط کلمات با طول مناسب
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # فیلتر کلمات پرتکرار
        for word, count in word_freq.items():
            if count >= 3:  # حداقل ۳ بار تکرار
                patterns[word] = count
        
        # به‌روزرسانی الگوهای یادگرفته شده
        for pattern, count in patterns.items():
            if pattern in self.patterns_learned:
                self.patterns_learned[pattern] += count
            else:
                self.patterns_learned[pattern] = count
                self.learning_stats['patterns_detected'] += 1
        
        return patterns
    
    def _text_to_vector(self, text: str, vector_size: int = 1000) -> np.ndarray:
        """تبدیل متن به بردار عددی"""
        vector = np.zeros(vector_size)
        
        # استفاده از هش برای توزیع یکنواخت
        for i, char in enumerate(text[:vector_size]):
            hash_val = int(hashlib.md5(char.encode()).hexdigest(), 16)
            vector[i] = (hash_val % 1000) / 1000.0  # نرمالایز کردن
        
        # اضافه کردن ویژگی‌های مبتنی بر الگوها
        if self.patterns_learned:
            pattern_score = sum(self.patterns_learned.values()) / 100.0
            vector[0] = min(pattern_score, 1.0)  # قرار دادن در اولین المان
        
        return vector
    
    def _update_learning_stats(self, words: List[str], patterns: Dict[str, int]):
        """به‌روزرسانی آمار یادگیری"""
        self.learning_stats['total_words_processed'] += len(words)
        self.learning_stats['unique_words_found'] = len(self.vocabulary)
    
    def extract_patterns(self, text: str) -> Dict[str, Any]:
        """استخراج الگوها و کلمات کلیدی از متن"""
        cleaned_text = self._clean_text(text)
        patterns = self._detect_patterns(cleaned_text)
        words = self._extract_words(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text[:500] + '...' if len(cleaned_text) > 500 else cleaned_text,
            'word_count': len(words),
            'unique_words': len(set(words)),
            'patterns_detected': patterns,
            'vocabulary_size': len(self.vocabulary),
            'top_patterns': dict(sorted(
                self.patterns_learned.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # 10 الگوی برتر
        }
    
    def generate_training_pairs(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """تولید جفت‌های آموزشی (ورودی-هدف)"""
        # در این نسخه ساده، ورودی و هدف یکسان هستند
        # در نسخه‌های پیشرفته می‌تواند متفاوت باشد
        input_vector = self.process_text_file(text)
        target_vector = input_vector.copy()  # هدف مشابه ورودی
        
        return input_vector, target_vector
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """دریافت آمار کامل یادگیری"""
        return {
            'processed_files': self.processed_files,
            'vocabulary_size': len(self.vocabulary),
            'learning_stats': self.learning_stats,
            'top_patterns': dict(sorted(
                self.patterns_learned.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]),
            'system_health': {
                'memory_usage_mb': (len(str(self.vocabulary)) + len(str(self.patterns_learned))) / (1024 * 1024),
                'processing_efficiency': self.processed_files / max(1, self.learning_stats['total_words_processed']),
                'pattern_detection_rate': self.learning_stats['patterns_detected'] / max(1, self.processed_files)
            }
        }

# نمونه گلوبال
ai_learner = AILearningEngine()

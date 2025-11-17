import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class LearningEngine:
    """موتور یادگیری خودآموز برای شبکه عصبی"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.01)
        self.min_learning_threshold = config.get('min_learning_threshold', 0.1)
        
        # تاریخچه یادگیری
        self.learning_history = []
        self.concept_mastery = defaultdict(float)
        self.interaction_patterns = defaultdict(list)
        
        # آستانه‌های یادگیری
        self.mastery_threshold = 0.7
        self.forgetting_factor = 0.99  # کاهش تدریجی تسلط
        
        # لاگ‌های یادگیری
        self.learning_events = []
        
        logger.info("🚀 موتور یادگیری خودآموز راه‌اندازی شد")
    
    def process_interaction(self, 
                          user_input: str,
                          activated_neurons: List[int],
                          api_response: Dict[str, Any],
                          success: bool) -> Dict[str, Any]:
        """پردازش یک تعامل برای یادگیری"""
        
        learning_result = {
            'timestamp': time.time(),
            'user_input': user_input,
            'activated_neurons_count': len(activated_neurons),
            'success': success,
            'learned_concepts': [],
            'strengthened_patterns': [],
            'complexity': self._calculate_complexity(user_input)
        }
        
        if success and activated_neurons:
            # یادگیری از تعامل موفق
            self._learn_from_success(activated_neurons, user_input, api_response)
            learning_result['learned_concepts'] = self._extract_new_concepts(user_input, api_response)
            learning_result['strengthened_patterns'] = self._identify_patterns(activated_neurons)
        
        # به‌روزرسانی تسلط مفاهیم
        self._update_concept_mastery(activated_neurons, success)
        
        # ذخیره در تاریخچه
        self.learning_history.append(learning_result)
        
        # حذف تاریخچه قدیمی
        self._prune_old_history()
        
        logger.debug(f"📚 یادگیری از تعامل: {len(activated_neurons)} نورون فعال")
        return learning_result
    
    def _learn_from_success(self, 
                          activated_neurons: List[int],
                          user_input: str,
                          api_response: Dict[str, Any]):
        """یادگیری از یک تعامل موفق"""
        
        # تقویت الگوهای فعال
        pattern_key = self._create_pattern_key(activated_neurons)
        self.interaction_patterns[pattern_key].append({
            'timestamp': time.time(),
            'input': user_input,
            'response_type': api_response.get('type', 'unknown')
        })
        
        # استخراج مفاهیم جدید از پاسخ API
        if 'data' in api_response:
            self._extract_concepts_from_data(api_response['data'], activated_neurons)
        
        # ثبت رویداد یادگیری
        self.learning_events.append({
            'type': 'successful_interaction',
            'neurons': activated_neurons[:10],  # 10 نورون اول
            'timestamp': time.time(),
            'input_sample': user_input[:50]  # نمونه کوتاه
        })
    
    def _extract_new_concepts(self, user_input: str, api_response: Dict[str, Any]) -> List[str]:
        """استخراج مفاهیم جدید از تعامل"""
        concepts = []
        
        # استخراج از ورودی کاربر
        words = user_input.lower().split()
        key_terms = [word for word in words if len(word) > 3 and word not in self.concept_mastery]
        
        # استخراج از پاسخ API
        if 'data' in api_response:
            data = api_response['data']
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (str, int, float)) and str(value) not in self.concept_mastery:
                        concepts.append(f"{key}_{value}")
        
        return concepts[:5]  # حداکثر 5 مفهوم جدید
    
    def _identify_patterns(self, activated_neurons: List[int]) -> List[str]:
        """شناسایی الگوهای تکراری در فعال‌سازی نورون‌ها"""
        if len(activated_neurons) < 3:
            return []
        
        # گروه‌بندی نورون‌های فعال
        neuron_groups = []
        current_group = []
        
        for neuron in sorted(activated_neurons):
            if not current_group or neuron == current_group[-1] + 1:
                current_group.append(neuron)
            else:
                if len(current_group) >= 2:
                    neuron_groups.append(current_group)
                current_group = [neuron]
        
        if len(current_group) >= 2:
            neuron_groups.append(current_group)
        
        return [f"group_{min(group)}_{max(group)}" for group in neuron_groups]
    
    def _update_concept_mastery(self, activated_neurons: List[int], success: bool):
        """به‌روزرسانی سطح تسلط مفاهیم"""
        for neuron in activated_neurons:
            concept_key = f"neuron_{neuron}"
            
            if success:
                # افزایش تسلط برای تعامل موفق
                self.concept_mastery[concept_key] = min(
                    1.0, 
                    self.concept_mastery.get(concept_key, 0) + self.learning_rate
                )
            else:
                # کاهش جزئی برای تعامل ناموفق
                self.concept_mastery[concept_key] = max(
                    0.0,
                    self.concept_mastery.get(concept_key, 0) - (self.learning_rate * 0.5)
                )
        
        # اعمال فراموشی تدریجی
        self._apply_forgetting()
    
    def _apply_forgetting(self):
        """اعمال فراموشی تدریجی بر مفاهیم کم‌استفاده"""
        current_time = time.time()
        forget_threshold = current_time - (30 * 24 * 3600)  # 30 روز قبل
        
        # حذف الگوهای قدیمی
        for pattern_key in list(self.interaction_patterns.keys()):
            recent_interactions = [
                interaction for interaction in self.interaction_patterns[pattern_key]
                if interaction['timestamp'] > forget_threshold
            ]
            
            if not recent_interactions:
                del self.interaction_patterns[pattern_key]
            else:
                self.interaction_patterns[pattern_key] = recent_interactions
        
        # کاهش تدریجی تسلط مفاهیم کم‌استفاده
        for concept in list(self.concept_mastery.keys()):
            self.concept_mastery[concept] *= self.forgetting_factor
            if self.concept_mastery[concept] < 0.01:
                del self.concept_mastery[concept]
    
    def _create_pattern_key(self, activated_neurons: List[int]) -> str:
        """ایجاد کلید یکتا برای الگوی فعال‌سازی"""
        if not activated_neurons:
            return "empty"
        
        # نرمال‌سازی و مرتب‌سازی
        sorted_neurons = sorted(activated_neurons)
        
        # ایجاد الگوی فشرده
        if len(sorted_neurons) <= 5:
            return f"exact_{'_'.join(map(str, sorted_neurons))}"
        else:
            # برای الگوهای بزرگ، از محدوده استفاده می‌کنیم
            ranges = []
            start = end = sorted_neurons[0]
            
            for neuron in sorted_neurons[1:]:
                if neuron == end + 1:
                    end = neuron
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = end = neuron
            
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            
            return f"range_{'_'.join(ranges)}"
    
    def _calculate_complexity(self, user_input: str) -> int:
        """محاسبه پیچیدگی ورودی کاربر"""
        words = user_input.split()
        unique_words = len(set(words))
        length_factor = min(len(words) / 10, 2.0)  # نرمال‌سازی طول
        
        return int(unique_words * length_factor)
    
    def _prune_old_history(self):
        """حذف تاریخچه قدیمی"""
        current_time = time.time()
        max_history_age = self.config.get('max_history_age_days', 30) * 24 * 3600
        
        self.learning_history = [
            record for record in self.learning_history
            if current_time - record['timestamp'] <= max_history_age
        ]
        
        self.learning_events = [
            event for event in self.learning_events
            if current_time - event['timestamp'] <= max_history_age
        ]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """آمار یادگیری"""
        recent_interactions = [
            interaction for interaction in self.learning_history
            if time.time() - interaction['timestamp'] <= (24 * 3600)  # 24 ساعت گذشته
        ]
        
        return {
            'total_interactions': len(self.learning_history),
            'recent_interactions_24h': len(recent_interactions),
            'mastered_concepts': len([c for c, m in self.concept_mastery.items() if m > self.mastery_threshold]),
            'active_patterns': len(self.interaction_patterns),
            'success_rate': self._calculate_success_rate(),
            'avg_complexity': np.mean([r['complexity'] for r in recent_interactions]) if recent_interactions else 0
        }
    
    def _calculate_success_rate(self) -> float:
        """محاسبه نرخ موفقیت"""
        if not self.learning_history:
            return 0.0
        
        successful = sum(1 for record in self.learning_history if record['success'])
        return successful / len(self.learning_history)
    
    def can_learn_more(self, current_memory_usage: float) -> bool:
        """بررسی امکان یادگیری بیشتر با توجه به حافظه"""
        max_memory_mb = self.config.get('max_learning_memory_mb', 50)
        return current_memory_usage < max_memory_mb
    
    def get_learning_insights(self) -> List[str]:
        """دریافت بینش‌های یادگیری"""
        insights = []
        
        # تحلیل الگوهای موفق
        successful_patterns = [
            pattern for pattern, interactions in self.interaction_patterns.items()
            if len(interactions) >= 3
        ]
        
        if successful_patterns:
            insights.append(f"🔍 {len(successful_patterns)} الگوی موفق شناسایی شده")
        
        # تحلیل مفاهیم مسلط
        mastered_concepts = [c for c, m in self.concept_mastery.items() if m > self.mastery_threshold]
        if mastered_concepts:
            insights.append(f"🎯 {len(mastered_concepts)} مفهوم تسلط یافته")
        
        # تحلیل پیچیدگی
        recent_complexities = [r['complexity'] for r in self.learning_history[-10:]]
        if recent_complexities:
            avg_complexity = np.mean(recent_complexities)
            if avg_complexity > 5:
                insights.append("📈 در حال پردازش سوالات پیچیده‌تر")
        
        return insights

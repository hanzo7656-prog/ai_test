import numpy as np
import json
import time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SparseNeuralNetwork:
    """شبکه عصبی اسپارس 1000 نورونی با یادگیری خودآموز"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_neurons = 1000
        self.sparsity = 0.1  # 10% اتصالات فعال
        
        # ماتریس وزن‌ها - فقط اتصالات فعال ذخیره می‌شوند
        self.weights = {}
        self.neuron_states = np.zeros(self.num_neurons)
        self.learning_rate = 0.01
        
        # نگاشت مفاهیم به نورون‌ها
        self.concept_neurons = {}
        self.neuron_concepts = {}
        
        # لاگ فعالیت
        self.activation_history = []
        
        self._initialize_network()
    
    def _initialize_network(self):
        """مقداردهی اولیه شبکه با اتصالات اسپارس"""
        logger.info(f"🚀 راه‌اندازی شبکه عصبی اسپارس با {self.num_neurons} نورون")
        
        # ایجاد اتصالات تصادفی اسپارس
        num_connections = int(self.num_neurons * self.num_neurons * self.sparsity)
        for _ in range(num_connections):
            i, j = np.random.randint(0, self.num_neurons, 2)
            if i != j:
                self.weights[(i, j)] = np.random.normal(0, 0.1)
        
        logger.info(f"✅ شبکه با {len(self.weights)} اتصال اسپارس راه‌اندازی شد")
    
    def process_input(self, input_vector: np.ndarray) -> np.ndarray:
        """پردازش ورودی و انتشار در شبکه"""
        if len(input_vector) != self.num_neurons:
            raise ValueError(f"ورودی باید بعد {self.num_neurons} داشته باشد")
        
        # ریست حالت نورون‌ها
        self.neuron_states = np.zeros(self.num_neurons)
        
        # فعال‌سازی نورون‌های ورودی
        input_indices = np.where(input_vector > 0)[0]
        for idx in input_indices:
            self.neuron_states[idx] = input_vector[idx]
        
        # انتشار در شبکه (یک پاس)
        new_states = self.neuron_states.copy()
        
        for (i, j), weight in self.weights.items():
            if self.neuron_states[i] > 0:  # فقط اگر نورون مبدأ فعال باشد
                new_states[j] += self.neuron_states[i] * weight
        
        # تابع فعال‌سازی
        self.neuron_states = np.tanh(new_states)
        
        # لاگ فعال‌سازی
        active_neurons = np.sum(self.neuron_states > 0.1)
        self.activation_history.append({
            'timestamp': time.time(),
            'active_neurons': active_neurons,
            'max_activation': np.max(self.neuron_states)
        })
        
        return self.neuron_states
    
    def hebbian_learn(self, active_neurons: List[int]):
        """یادگیری هبیان برای نورون‌های فعال"""
        if not active_neurons:
            return
        
        # تقویت اتصالات بین نورون‌های فعال همزمان
        for i in active_neurons:
            for j in active_neurons:
                if i != j and (i, j) in self.weights:
                    # قانون هبیان: سلول‌هایی که با هم فعال می‌شوند، با هم ارتباط برقرار می‌کنند
                    self.weights[(i, j)] += self.learning_rate * self.neuron_states[i] * self.neuron_states[j]
        
        logger.debug(f"📚 یادگیری هبیان برای {len(active_neurons)} نورون فعال")
    
    def learn_concept(self, concept: str, activated_neurons: List[int]):
        """یادگیری یک مفهوم جدید و نگاشت به نورون‌ها"""
        if not activated_neurons:
            return
        
        # نورون‌های اصلی مرتبط با مفهوم
        core_neurons = activated_neurons[:10]  # 10 نورون اول به عنوان هسته
        
        if concept not in self.concept_neurons:
            self.concept_neurons[concept] = set(core_neurons)
            
            # برعکس نگاشت برای جستجوی سریع
            for neuron in core_neurons:
                if neuron not in self.neuron_concepts:
                    self.neuron_concepts[neuron] = set()
                self.neuron_concepts[neuron].add(concept)
            
            logger.info(f"🎯 مفهوم '{concept}' به {len(core_neurons)} نورون نگاشت شد")
        else:
            # به‌روزرسانی مفهوم موجود
            self.concept_neurons[concept].update(core_neurons)
    
    def find_related_concepts(self, activated_neurons: List[int]) -> List[str]:
        """پیدا کردن مفاهیم مرتبط بر اساس نورون‌های فعال"""
        concept_scores = {}
        
        for neuron in activated_neurons:
            if neuron in self.neuron_concepts:
                for concept in self.neuron_concepts[neuron]:
                    concept_scores[concept] = concept_scores.get(concept, 0) + 1
        
        # مرتب‌سازی بر اساس امتیاز
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, score in sorted_concepts[:5]]  # 5 مفهوم برتر
    
    def get_network_stats(self) -> Dict[str, Any]:
        """آمار وضعیت شبکه"""
        active_weights = len(self.weights)
        total_possible = self.num_neurons * self.num_neurons
        actual_sparsity = active_weights / total_possible if total_possible > 0 else 0
        
        return {
            'total_neurons': self.num_neurons,
            'active_connections': active_weights,
            'actual_sparsity': round(actual_sparsity, 4),
            'learned_concepts': len(self.concept_neurons),
            'avg_activation': np.mean(self.neuron_states) if len(self.neuron_states) > 0 else 0,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """تخمین استفاده از حافظه"""
        weights_size = len(self.weights) * 12  # هر اتصال ≈12 بایت
        concepts_size = sum(len(neurons) * 20 for neurons in self.concept_neurons.values())  # هر مفهوم ≈20 بایت
        total_bytes = weights_size + concepts_size + self.num_neurons * 8  # حالت نورون‌ها
        
        return round(total_bytes / (1024 * 1024), 2)  # به مگابایت
    
    def can_process_complexity(self, input_complexity: int) -> bool:
        """بررسی توانایی پردازش پیچیدگی ورودی"""
        max_complexity = self.config.get('max_complexity', 50)
        
        if input_complexity > max_complexity:
            logger.warning(f"⚠️ پیچیدگی ورودی ({input_complexity}) از حد مجاز ({max_complexity}) بیشتر است")
            return False
        return True
    
    def save_state(self, filepath: str):
        """ذخیره حالت شبکه"""
        state = {
            'weights': {f"{i}_{j}": weight for (i, j), weight in self.weights.items()},
            'concept_neurons': {concept: list(neurons) for concept, neurons in self.concept_neurons.items()},
            'neuron_concepts': {neuron: list(concepts) for neuron, concepts in self.neuron_concepts.items()},
            'config': self.config,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 حالت شبکه در {filepath} ذخیره شد")
    
    def load_state(self, filepath: str):
        """بارگذاری حالت شبکه"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # بازیابی وزن‌ها
            self.weights = {}
            for key, weight in state['weights'].items():
                i, j = map(int, key.split('_'))
                self.weights[(i, j)] = weight
            
            # بازیابی مفاهیم
            self.concept_neurons = {concept: set(neurons) for concept, neurons in state['concept_neurons'].items()}
            self.neuron_concepts = {int(neuron): set(concepts) for neuron, concepts in state['neuron_concepts'].items()}
            
            logger.info(f"📂 حالت شبکه از {filepath} بارگذاری شد")
            
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری حالت: {e}")
            # در صورت خطا، شبکه جدید راه‌اندازی می‌شود
            self._initialize_network()

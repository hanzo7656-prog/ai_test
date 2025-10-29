# sparse_technical_analyzer.py - تحلیل‌گر تکنیکال اسپارس با داده‌های خام

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import math
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SparseConfig:
    """پیکربندی معماری اسپارس"""
    total_neurons: int = 2500
    connections_per_neuron: int = 50
    temporal_sequence: int = 60
    input_features: int = 5
    hidden_size: int = 128
    specialty_groups: Dict = None
    
    def __post_init__(self):
        if self.specialty_groups is None:
            self.specialty_groups = {
                "support_resistance": 800,
                "trend_detection": 700,
                "pattern_recognition": 600,
                "volume_analysis": 400
            }

class SparseTechnicalNeuron(nn.Module):
    """نورون اسپارس تحلیل تکنیکال"""
    def __init__(self, neuron_id: int, specialty: str, config: SparseConfig):
        super().__init__()
        self.neuron_id = neuron_id
        self.specialty = specialty
        self.config = config

        # پارامترهای تخصصی سبک
        self.sensitivity = nn.Parameter(torch.randn(1) * 0.1 + 1.0)
        self.threshold = nn.Parameter(torch.randn(1) * 0.1 + 0.6)

        # اتصالات اسپارس - فقط وزن‌های ضروری
        self.connection_weights = nn.Parameter(
            torch.randn(config.connections_per_neuron) * 0.1
        )
        self.connection_indices = None  # بعدا مقداردهی می‌شود

    def set_connections(self, indices: torch.Tensor):
        """تنظیم اتصالات اسپارس"""
        self.connection_indices = indices

    def forward(self, x: torch.Tensor, all_activations: torch.Tensor) -> torch.Tensor:
        """پردازش با اتصالات اسپارس"""
        if self.connection_indices is None:
            return torch.tensor(0.0)

        # جمع آوری فعالیت نورون های متصل
        connected_activations = all_activations[:, self.connection_indices]

        # محاسبه وزن دار
        weighted_input = torch.sum(connected_activations * self.connection_weights, dim=1)

        # فعال سازی تخصصی
        if self.specialty == "support_resistance":
            output = torch.sigmoid(weighted_input * self.sensitivity - self.threshold)
        elif self.specialty == "trend_detection":
            output = torch.tanh(weighted_input * self.sensitivity)
        elif self.specialty == "pattern_recognition":
            output = torch.relu(weighted_input * self.sensitivity - self.threshold)
        else:  # volume_analysis
            output = torch.sigmoid(weighted_input * self.sensitivity)

        return output

class SparseTechnicalNetwork(nn.Module):
    """شبکه اسپارس تحلیل تکنیکال با داده‌های خام"""
    
    def __init__(self, config: SparseConfig):
        super().__init__()
        self.config = config

        # لایه پردازش زمانی
        self.temporal_processor = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )

        # پروجکشن ویژگی های زمانی
        self.feature_projection = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        # ایجاد نورون‌های اسپارس
        self.neurons = nn.ModuleList()
        self._initialize_sparse_neurons()
        self._initialize_sparse_connections()

        # لایه ادغام هوشمند
        self.integrator = nn.Sequential(
            nn.Linear(config.total_neurons, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        # سرهای خروجی تخصصی
        self.output_heads = nn.ModuleDict({
            'trend_strength': nn.Linear(32, 3),  # صعودی، نزولی، خنثی
            'pattern_signals': nn.Linear(32, 6),  # الگوهای اصلی
            'key_levels': nn.Linear(32, 4),  # سطوح کلیدی
            'market_volatility': nn.Linear(32, 1),  # نوسان بازار
            'signal_confidence': nn.Linear(32, 1)  # اطمینان سیگنال
        })

        logger.info(f"🧠 شبکه اسپارس با {config.total_neurons} نورون ایجاد شد")
        logger.info(f"🔗 اتصالات اسپارس: {config.connections_per_neuron} per neuron")
        logger.info("📊 حالت داده خام فعال")

    def _initialize_sparse_neurons(self):
        """ایجاد نورون های اسپارس"""
        neuron_id = 0
        for specialty, count in self.config.specialty_groups.items():
            for i in range(count):
                self.neurons.append(
                    SparseTechnicalNeuron(neuron_id, specialty, self.config)
                )
                neuron_id += 1

    def _initialize_sparse_connections(self):
        """مقداردهی اتصالات اسپارس هوشمند"""
        total_neurons = self.config.total_neurons

        for i, neuron in enumerate(self.neurons):
            # اتصالات هوشمند بر اساس گروه‌های تخصصی
            connections = self._get_smart_connections(i, neuron.specialty)
            neuron.set_connections(connections)

    def _get_smart_connections(self, neuron_idx: int, specialty: str) -> torch.Tensor:
        """اتصالات هوشمند بر اساس تخصص و موقعیت"""
        connections = set()
        total_neurons = self.config.total_neurons

        # 1. اتصالات درون‌گروهی (60%)
        specialty_start, specialty_count = self._get_specialty_range(specialty)
        in_group_count = int(self.config.connections_per_neuron * 0.6)

        in_group_indices = torch.randperm(specialty_count)[:in_group_count] + specialty_start
        connections.update(in_group_indices.tolist())

        # 2. اتصالات بین‌گروهی (40%)
        cross_group_count = self.config.connections_per_neuron - len(connections)

        if specialty == "support_resistance":
            target_specialty = "trend_detection"
        elif specialty == "trend_detection":
            target_specialty = "pattern_recognition"
        elif specialty == "pattern_recognition":
            target_specialty = "volume_analysis"
        else:  # volume_analysis
            target_specialty = "support_resistance"

        target_start, target_count = self._get_specialty_range(target_specialty)
        cross_indices = torch.randperm(target_count)[:cross_group_count] + target_start

        connections.update(cross_indices.tolist())

        # تبدیل به تانسور
        return torch.tensor(list(connections)[:self.config.connections_per_neuron])

    def _get_specialty_range(self, specialty: str) -> Tuple[int, int]:
        """محدوده نورون‌های یک تخصص"""
        start_idx = 0
        for spec, count in self.config.specialty_groups.items():
            if spec == specialty:
                return start_idx, count
            start_idx += count
        return 0, 0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """پردازش کامل تحلیل تکنیکال با داده‌های خام"""
        batch_size = x.shape[0]

        # 1. پردازش زمانی با LSTM
        temporal_out, (hidden, cell) = self.temporal_processor(x)
        temporal_features = hidden[-1]  # آخرین hidden state

        # 2. استخراج ویژگی‌های پایه از داده خام
        base_features = self.feature_projection(temporal_features)

        # 3. پردازش دسته‌ای نورون‌های اسپارس
        all_activations = base_features.unsqueeze(1).repeat(1, self.config.total_neurons, 1)
        neuron_outputs = []

        # پردازش در دسته‌های 100 تایی برای بهینه‌سازی سرعت
        batch_size_neurons = 100
        for i in range(0, len(self.neurons), batch_size_neurons):
            batch_neurons = self.neurons[i:i+batch_size_neurons]
            batch_outputs = []

            for neuron in batch_neurons:
                neuron_out = neuron(base_features, all_activations)
                batch_outputs.append(neuron_out.unsqueeze(1))

            neuron_outputs.extend(batch_outputs)

        # ترکیب خروجی نورون‌ها
        all_neuron_outputs = torch.cat(neuron_outputs, dim=1)

        # 4. ادغام هوشمند تصمیم‌ها
        integrated = self.integrator(all_neuron_outputs)

        # 5. تولید خروجی‌های نهایی
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(integrated)

        # محاسبه اطمینان کلی
        outputs['overall_confidence'] = torch.sigmoid(
            outputs['signal_confidence']
        ).squeeze(-1)

        # فعالیت گروه‌های تخصصی
        outputs['specialty_activities'] = self._calculate_specialty_activities(
            all_neuron_outputs
        )

        # اطلاعات داده خام
        outputs['raw_data_metrics'] = {
            'input_shape': x.shape,
            'processed_sequences': batch_size,
            'timestamp': datetime.now().isoformat()
        }

        return outputs

    def _calculate_specialty_activities(self, neuron_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """محاسبه فعالیت میانگین هر گروه تخصصی"""
        activities = {}
        start_idx = 0

        for specialty, count in self.config.specialty_groups.items():
            end_idx = start_idx + count
            specialty_outputs = neuron_outputs[:, start_idx:end_idx]
            activities[specialty] = specialty_outputs.mean(dim=1)
            start_idx = end_idx

        return activities

    def analyze_raw_market_data(self, raw_data: Dict) -> Dict[str, Any]:
        """تحلیل داده‌های بازار خام"""
        try:
            # استخراج و پیش‌پردازش داده‌های خام
            processed_data = self._preprocess_raw_data(raw_data)
            
            if processed_data is None:
                return {
                    'error': 'داده‌های خام نامعتبر',
                    'confidence': 0.0,
                    'raw_data_quality': 'poor'
                }

            # تبدیل به تانسور
            input_tensor = torch.FloatTensor(processed_data).unsqueeze(0)
            
            # تحلیل با شبکه اسپارس
            with torch.no_grad():
                analysis_results = self.forward(input_tensor)
            
            # تفسیر نتایج
            interpreted_results = self._interpret_analysis(analysis_results, raw_data)
            
            return {
                **interpreted_results,
                'raw_data_used': True,
                'processing_timestamp': datetime.now().isoformat(),
                'model_confidence': analysis_results['overall_confidence'].item()
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل داده‌های خام: {e}")
            return {
                'error': str(e),
                'confidence': 0.0,
                'raw_data_quality': 'error'
            }

    def _preprocess_raw_data(self, raw_data: Dict) -> Optional[np.ndarray]:
        """پیش‌پردازش داده‌های خام برای شبکه"""
        try:
            # استخراج داده‌های قیمتی خام
            price_data = raw_data.get('prices', [])
            volume_data = raw_data.get('volumes', [])
            
            if len(price_data) < self.config.temporal_sequence:
                logger.warning("❌ داده‌های خام ناکافی برای تحلیل")
                return None
            
            # نرمال‌سازی داده‌ها
            normalized_prices = self._normalize_data(price_data)
            normalized_volumes = self._normalize_data(volume_data) if volume_data else np.zeros_like(normalized_prices)
            
            # ایجاد دنباله‌های زمانی
            sequences = []
            for i in range(len(normalized_prices) - self.config.temporal_sequence + 1):
                sequence = np.column_stack([
                    normalized_prices[i:i+self.config.temporal_sequence],
                    normalized_volumes[i:i+self.config.temporal_sequence] if len(volume_data) > 0 else np.zeros(self.config.temporal_sequence)
                ])
                sequences.append(sequence)
            
            return np.array(sequences)
            
        except Exception as e:
            logger.error(f"❌ خطا در پیش‌پردازش داده‌های خام: {e}")
            return None

    def _normalize_data(self, data: List[float]) -> np.ndarray:
        """نرمال‌سازی داده‌های خام"""
        data_array = np.array(data)
        if len(data_array) == 0:
            return np.array([])
        
        # نرمال‌سازی min-max
        if np.max(data_array) != np.min(data_array):
            normalized = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
        else:
            normalized = np.ones_like(data_array) * 0.5
            
        return normalized

    def _interpret_analysis(self, analysis_results: Dict, raw_data: Dict) -> Dict[str, Any]:
        """تفسیر نتایج تحلیل شبکه"""
        try:
            # تفسیر روند
            trend_probs = torch.softmax(analysis_results['trend_strength'], dim=-1)
            trend_labels = ['صعودی', 'نزولی', 'خنثی']
            dominant_trend = trend_labels[torch.argmax(trend_probs).item()]
            
            # تفسیر الگوها
            pattern_probs = torch.softmax(analysis_results['pattern_signals'], dim=-1)
            pattern_labels = ['سر و شانه', 'دو قله', 'دو دره', 'مثلث', 'کنج', 'کانال']
            dominant_pattern = pattern_labels[torch.argmax(pattern_probs).item()]
            
            # تفسیر سطوح کلیدی
            key_levels = analysis_results['key_levels'].squeeze().tolist()
            
            return {
                'trend_analysis': {
                    'direction': dominant_trend,
                    'confidence': trend_probs.max().item(),
                    'probabilities': trend_probs.squeeze().tolist()
                },
                'pattern_analysis': {
                    'detected_pattern': dominant_pattern,
                    'confidence': pattern_probs.max().item(),
                    'all_patterns': dict(zip(pattern_labels, pattern_probs.squeeze().tolist()))
                },
                'key_levels': {
                    'support': key_levels[0] if len(key_levels) > 0 else 0,
                    'resistance': key_levels[1] if len(key_levels) > 1 else 0,
                    'breakout_level': key_levels[2] if len(key_levels) > 2 else 0,
                    'rejection_level': key_levels[3] if len(key_levels) > 3 else 0
                },
                'market_metrics': {
                    'volatility': analysis_results['market_volatility'].item(),
                    'signal_confidence': analysis_results['signal_confidence'].item(),
                    'overall_confidence': analysis_results['overall_confidence'].item()
                },
                'specialty_activities': {
                    k: v.item() for k, v in analysis_results['specialty_activities'].items()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ خطا در تفسیر نتایج: {e}")
            return {
                'error': 'خطا در تفسیر نتایج',
                'trend_analysis': {'direction': 'نامشخص', 'confidence': 0.0},
                'pattern_analysis': {'detected_pattern': 'هیچ', 'confidence': 0.0}
            }

class TechnicalAnalysisTrainer:
    """آموزش دهنده تحلیل تکنیکال با داده‌های خام"""
    
    def __init__(self, config: SparseConfig):
        self.config = config
        self.model = SparseTechnicalNetwork(config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("🎯 آموزش دهنده تحلیل تکنیکال راه‌اندازی شد")

    def train_on_historical_data(self, symbols: List[str], epochs: int = 50):
        """آموزش روی داده های تاریخی خام"""
        # پیاده سازی آموزش با داده های واقعی
        logger.info(f"📚 آموزش روی {len(symbols)} نماد برای {epochs} دوره")
        # TODO: پیاده‌سازی کامل آموزش
        
    def analyze_market(self, market_data: torch.Tensor) -> Dict:
        """آنالیز بازار در زمان واقعی با داده‌های خام"""
        self.model.eval()
        with torch.no_grad():
            return self.model(market_data)

# تست عملکرد
def test_sparse_architecture():
    """تست کامل معماری اسپارس"""
    config = SparseConfig()
    model = SparseTechnicalNetwork(config)

    # تست با داده نمونه
    sample_data = torch.randn(32, 60, 5)  # 32 نمونه، 60 کندل، 5 ویژگی
    outputs = model(sample_data)

    print("✅ معماری اسپارس ترکیبی فعال شد!")
    print(f"🧠 نورون ها: {config.total_neurons}")
    print(f"🔗 اتصالات اسپارس: {config.connections_per_neuron} per neuron")
    print(f"📊 پارامترهای کل: {sum(p.numel() for p in model.parameters())}")

    print("\n📈 خروجی های تولید شده:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} گروه تخصصی")

    # تست سرعت
    import time
    start_time = time.time()
    
    for _ in range(100):
        _ = model(sample_data)
        
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000  # میلی ثانیه

    print(f"\n⚡ سرعت متوسط: {avg_time:.1f}ms per تحلیل")
    print(f"📊 {1000/avg_time:.0f} تحلیل بر ثانيه")

    return model

if __name__ == "__main__":
    analyzer = test_sparse_architecture()

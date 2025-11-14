import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class SparseNeuralNetwork:
    """شبکه عصبی اسپارس 1000 نورونی"""
    
    def __init__(self):
        self.neuron_count = 1000
        self.connection_sparsity = 0.1  # 10% اتصالات فعال
        self.learning_rate = 0.01
        
        # ماتریس وزن‌ها (اسپارس)
        self.weights = self._initialize_sparse_weights()
        
        # نورون‌ها
        self.neurons = np.zeros(self.neuron_count)
        self.bias = np.random.normal(0, 0.1, self.neuron_count)
        
        # تاریخچه یادگیری
        self.learning_history = []
        self.performance_metrics = {
            'training_samples': 0,
            'successful_predictions': 0,
            'accuracy_trend': [],
            'last_training_time': None
        }
    
    def _initialize_sparse_weights(self) -> np.ndarray:
        """مقداردهی اولیه وزن‌های اسپارس"""
        weights = np.zeros((self.neuron_count, self.neuron_count))
        
        # ایجاد اتصالات تصادفی اسپارس
        connections_per_neuron = int(self.neuron_count * self.connection_sparsity)
        
        for i in range(self.neuron_count):
            # انتخاب نورون‌های متصل به صورت تصادفی
            connected_neurons = np.random.choice(
                self.neuron_count, 
                connections_per_neuron, 
                replace=False
            )
            # مقداردهی وزن‌های تصادفی
            weights[i, connected_neurons] = np.random.normal(
                0, 0.1, connections_per_neuron
            )
        
        return weights
    
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """فعال‌سازی شبکه"""
        if len(inputs) != self.neuron_count:
            # تطبیق اندازه ورودی
            padded_inputs = np.zeros(self.neuron_count)
            min_len = min(len(inputs), self.neuron_count)
            padded_inputs[:min_len] = inputs[:min_len]
            inputs = padded_inputs
        
        # محاسبه خروجی شبکه
        self.neurons = np.tanh(
            np.dot(self.weights, inputs) + self.bias
        )
        
        return self.neurons
    
    def learn(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float = None):
        """یادگیری از داده‌های جدید"""
        lr = learning_rate or self.learning_rate
        
        # فعال‌سازی شبکه
        outputs = self.activate(inputs)
        
        # محاسبه خطا
        error = targets - outputs
        
        # به‌روزرسانی وزن‌ها (فقط اتصالات فعال)
        for i in range(self.neuron_count):
            active_connections = np.where(self.weights[i] != 0)[0]
            for j in active_connections:
                self.weights[i, j] += lr * error[i] * inputs[j]
        
        # به‌روزرسانی بایاس
        self.bias += lr * error
        
        # ذخیره تاریخچه یادگیری
        self.performance_metrics['training_samples'] += 1
        accuracy = 1.0 - np.mean(np.abs(error))
        self.performance_metrics['accuracy_trend'].append(accuracy)
        
        # حفظ اندازه لیست دقت
        if len(self.performance_metrics['accuracy_trend']) > 100:
            self.performance_metrics['accuracy_trend'].pop(0)
        
        self.performance_metrics['last_training_time'] = datetime.now().isoformat()
        
        return accuracy
    
    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """پیش‌بینی با آستانه"""
        outputs = self.activate(inputs)
        return (outputs > threshold).astype(int)
    
    def get_network_health(self) -> Dict[str, Any]:
        """گزارش سلامت شبکه"""
        active_neurons = np.sum(self.neurons != 0)
        active_connections = np.sum(self.weights != 0)
        total_possible_connections = self.neuron_count ** 2
        
        return {
            'neuron_count': self.neuron_count,
            'active_neurons': int(active_neurons),
            'active_connections': int(active_connections),
            'connection_sparsity': f"{self.connection_sparsity * 100}%",
            'actual_sparsity': f"{(active_connections / total_possible_connections) * 100:.2f}%",
            'average_weight': float(np.mean(np.abs(self.weights[self.weights != 0]))),
            'bias_range': {
                'min': float(np.min(self.bias)),
                'max': float(np.max(self.bias)),
                'mean': float(np.mean(self.bias))
            },
            'performance': {
                'training_samples': self.performance_metrics['training_samples'],
                'current_accuracy': self.performance_metrics['accuracy_trend'][-1] if self.performance_metrics['accuracy_trend'] else 0,
                'accuracy_trend_10': np.mean(self.performance_metrics['accuracy_trend'][-10:]) if len(self.performance_metrics['accuracy_trend']) >= 10 else 0,
                'last_training': self.performance_metrics['last_training_time']
            },
            'memory_usage_mb': (self.weights.nbytes + self.neurons.nbytes + self.bias.nbytes) / (1024 * 1024)
        }
    
    def optimize_architecture(self):
        """بهینه‌سازی خودکار معماری"""
        # تحلیل عملکرد
        recent_accuracy = self.performance_metrics['accuracy_trend'][-10:] if self.performance_metrics['accuracy_trend'] else [0]
        avg_accuracy = np.mean(recent_accuracy)
        
        # تنظیم نرخ یادگیری بر اساس عملکرد
        if avg_accuracy < 0.7:
            self.learning_rate = min(0.1, self.learning_rate * 1.1)
        elif avg_accuracy > 0.9:
            self.learning_rate = max(0.001, self.learning_rate * 0.9)
        
        # هرس اتصالات ضعیف
        weight_threshold = np.percentile(np.abs(self.weights[self.weights != 0]), 10)
        self.weights[np.abs(self.weights) < weight_threshold] = 0
        
        print(f"🔄 Architecture optimized - LR: {self.learning_rate:.4f}")

# نمونه گلوبال
ai_brain = SparseNeuralNetwork()

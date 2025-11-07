# شبکه عصبی اسپارس 100 نورونی برای تحلیل بازار
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SparseNeuralNetwork:
    """شبکه عصبی اسپارس 100 نورونی برای تحلیل بازارهای مالی"""
    
    def __init__(self, input_size=20, hidden_size=100, output_size=5, sparsity=0.8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sparsity = sparsity  # 80% اسپارس
        
        # وزن‌های شبکه
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden = None
        self.bias_output = None
        
        # تاریخچه آموزش
        self.training_history = []
        self.is_trained = False
        
        self.initialize_weights()
        logger.info(f"✅ Sparse Neural Network initialized: {hidden_size} neurons, {sparsity*100}% sparsity")
    
    def initialize_weights(self):
        """مقداردهی اولیه وزن‌ها با اسپارسیتی"""
        # لایه پنهان - ایجاد اسپارسیتی
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.1
        
        # اعمال اسپارسیتی - 80% وزن‌ها صفر میشوند
        mask = np.random.random((self.input_size, self.hidden_size)) > self.sparsity
        self.weights_input_hidden *= mask
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    
    def relu(self, x):
        """تابع فعال‌ساز ReLU"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """تابع softmax برای خروجی"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """پاس رو به جلو"""
        # لایه پنهان
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        
        # لایه خروجی
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.output_input)
        
        return self.output
    
    def predict(self, features):
        """پیش‌بینی بر اساس ویژگی‌های ورودی"""
        try:
            if not self.is_trained:
                return self._random_prediction()
            
            # نرمال‌سازی ویژگی‌ها
            normalized_features = self._normalize_features(features)
            
            # پیش‌بینی
            prediction = self.forward(normalized_features)
            
            # تفسیر نتایج
            return self._interpret_prediction(prediction[0])
            
        except Exception as e:
            logger.error(f"خطا در پیش‌بینی شبکه عصبی: {e}")
            return self._random_prediction()
    
    def _normalize_features(self, features):
        """نرمال‌سازی ویژگی‌های ورودی"""
        # تبدیل به آرایه numpy
        feature_array = np.array([features])
        
        # نرمال‌سازی ساده
        normalized = (feature_array - np.mean(feature_array)) / (np.std(feature_array) + 1e-8)
        return normalized
    
    def _interpret_prediction(self, prediction):
        """تفسیر خروجی شبکه عصبی"""
        class_labels = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        return {
            'signal': class_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                label: float(prob) for label, prob in zip(class_labels, prediction)
            },
            'neural_network_used': True,
            'hidden_neurons_activated': int(np.sum(self.hidden_output > 0)),
            'timestamp': datetime.now().isoformat()
        }
    
    def _random_prediction(self):
        """پیش‌بینی تصادفی برای زمانی که مدل آموزش ندیده"""
        signals = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
        random_signal = np.random.choice(signals)
        
        return {
            'signal': random_signal,
            'confidence': 0.3 + np.random.random() * 0.3,
            'probabilities': {sig: 0.2 for sig in signals},
            'neural_network_used': False,
            'hidden_neurons_activated': 0,
            'timestamp': datetime.now().isoformat(),
            'note': 'مدل آموزش ندیده - استفاده از پیش‌بینی پایه'
        }
    
    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        """آموزش شبکه عصبی"""
        try:
            logger.info(f"🚀 شروع آموزش شبکه عصبی برای {epochs} دوره")
            
            for epoch in range(epochs):
                # پاس رو به جلو
                output = self.forward(X_train)
                
                # محاسبه خطا
                loss = self._compute_loss(output, y_train)
                
                # پس‌انتشار (backpropagation) ساده
                error = output - y_train
                
                # آپدیت وزن‌ها
                d_weights_hidden_output = np.dot(self.hidden_output.T, error)
                d_bias_output = np.sum(error, axis=0, keepdims=True)
                
                error_hidden = np.dot(error, self.weights_hidden_output.T)
                error_hidden[self.hidden_output <= 0] = 0  # ReLU derivative
                
                d_weights_input_hidden = np.dot(X_train.T, error_hidden)
                d_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True)
                
                # اعمال آپدیت‌ها
                self.weights_hidden_output -= learning_rate * d_weights_hidden_output
                self.bias_output -= learning_rate * d_bias_output
                self.weights_input_hidden -= learning_rate * d_weights_input_hidden
                self.bias_hidden -= learning_rate * d_bias_hidden
                
                # ذخیره تاریخچه
                if epoch % 10 == 0:
                    accuracy = self._compute_accuracy(output, y_train)
                    self.training_history.append({
                        'epoch': epoch,
                        'loss': float(loss),
                        'accuracy': float(accuracy)
                    })
                    
                    logger.info(f"📊 دوره {epoch}: خطا={loss:.4f}, دقت={accuracy:.2f}")
            
            self.is_trained = True
            logger.info("✅ آموزش شبکه عصبی کامل شد")
            
        except Exception as e:
            logger.error(f"خطا در آموزش شبکه عصبی: {e}")
    
    def _compute_loss(self, output, y_true):
        """محاسبه خطا"""
        return -np.sum(y_true * np.log(output + 1e-8)) / len(y_true)
    
    def _compute_accuracy(self, output, y_true):
        """محاسبه دقت"""
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)
    
    def get_network_info(self):
        """دریافت اطلاعات شبکه"""
        active_weights = np.sum(self.weights_input_hidden != 0)
        total_weights = self.weights_input_hidden.size
        sparsity_ratio = 1 - (active_weights / total_weights)
        
        return {
            'input_neurons': self.input_size,
            'hidden_neurons': self.hidden_size,
            'output_neurons': self.output_size,
            'sparsity': f"{sparsity_ratio*100:.1f}%",
            'active_weights': int(active_weights),
            'total_weights': total_weights,
            'is_trained': self.is_trained,
            'training_samples': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        }
    
    def save_model(self, filepath="models/sparse_network.npy"):
        """ذخیره مدل"""
        try:
            Path("models").mkdir(exist_ok=True)
            
            model_data = {
                'weights_input_hidden': self.weights_input_hidden,
                'weights_hidden_output': self.weights_hidden_output,
                'bias_hidden': self.bias_hidden,
                'bias_output': self.bias_output,
                'training_history': self.training_history,
                'is_trained': self.is_trained,
                'config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'output_size': self.output_size,
                    'sparsity': self.sparsity
                }
            }
            
            np.save(filepath, model_data, allow_pickle=True)
            logger.info(f"💾 مدل در {filepath} ذخیره شد")
            
        except Exception as e:
            logger.error(f"خطا در ذخیره مدل: {e}")
    
    def load_model(self, filepath="models/sparse_network.npy"):
        """بارگذاری مدل"""
        try:
            model_data = np.load(filepath, allow_pickle=True).item()
            
            self.weights_input_hidden = model_data['weights_input_hidden']
            self.weights_hidden_output = model_data['weights_hidden_output']
            self.bias_hidden = model_data['bias_hidden']
            self.bias_output = model_data['bias_output']
            self.training_history = model_data['training_history']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"📂 مدل از {filepath} بارگذاری شد")
            
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل: {e}")

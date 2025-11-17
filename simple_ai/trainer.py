import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import time

class AITrainingManager:
    """مدیریت آموزش و بهینه‌سازی هوش مصنوعی"""
    
    def __init__(self, brain, learner, memory):
        self.brain = brain
        self.learner = learner
        self.memory = memory
        
        self.training_history = []
        self.performance_metrics = {
            'total_training_sessions': 0,
            'total_training_samples': 0,
            'average_accuracy': 0.0,
            'best_accuracy': 0.0,
            'training_time_seconds': 0.0
        }
    
    def train_batch(self, training_data: List[str]) -> Dict[str, float]:
        """آموزش دسته‌ای روی داده‌های متنی"""
        start_time = time.time()
        accuracies = []
        
        try:
            for i, text_data in enumerate(training_data):
                # تولید جفت‌های آموزشی
                inputs, targets = self.learner.generate_training_pairs(text_data)
                
                # آموزش شبکه
                accuracy = self.brain.learn(inputs, targets)
                accuracies.append(accuracy)
                
                # ذخیره دانش آموخته شده
                knowledge_key = f"training_sample_{self.performance_metrics['total_training_samples'] + i + 1}"
                self.memory.save_knowledge(
                    key=knowledge_key,
                    knowledge={
                        'text_sample': text_data[:200] + '...' if len(text_data) > 200 else text_data,
                        'processed_patterns': self.learner.extract_patterns(text_data),
                        'training_accuracy': accuracy,
                        'timestamp': datetime.now().isoformat()
                    },
                    category="training_data"
                )
            
            # محاسبه میانگین دقت
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            max_accuracy = np.max(accuracies) if accuracies else 0.0
            
            # به‌روزرسانی آمار
            training_time = time.time() - start_time
            self._update_training_stats(
                samples_count=len(training_data),
                avg_accuracy=avg_accuracy,
                max_accuracy=max_accuracy,
                training_time=training_time
            )
            
            # ذخیره تاریخچه آموزش
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'samples_count': len(training_data),
                'average_accuracy': avg_accuracy,
                'max_accuracy': max_accuracy,
                'training_time_seconds': training_time,
                'learning_rate': self.brain.learning_rate
            })
            
            print(f"🎯 Training completed: {len(training_data)} samples, Avg Accuracy: {avg_accuracy:.3f}")
            
            return {
                'samples_trained': len(training_data),
                'average_accuracy': avg_accuracy,
                'max_accuracy': max_accuracy,
                'training_time_seconds': training_time,
                'learning_rate': self.brain.learning_rate
            }
            
        except Exception as e:
            print(f"❌ Batch training failed: {e}")
            return {
                'samples_trained': 0,
                'average_accuracy': 0.0,
                'max_accuracy': 0.0,
                'training_time_seconds': 0.0,
                'error': str(e)
            }
    
    def _update_training_stats(self, samples_count: int, avg_accuracy: float, 
                             max_accuracy: float, training_time: float):
        """به‌روزرسانی آمار آموزش"""
        self.performance_metrics['total_training_sessions'] += 1
        self.performance_metrics['total_training_samples'] += samples_count
        
        # به‌روزرسانی میانگین دقت
        total_accuracy = self.performance_metrics['average_accuracy'] * (self.performance_metrics['total_training_sessions'] - 1)
        self.performance_metrics['average_accuracy'] = (total_accuracy + avg_accuracy) / self.performance_metrics['total_training_sessions']
        
        # به‌روزرسانی بهترین دقت
        if max_accuracy > self.performance_metrics['best_accuracy']:
            self.performance_metrics['best_accuracy'] = max_accuracy
        
        self.performance_metrics['training_time_seconds'] += training_time
    
    def validate_performance(self, validation_data: List[str] = None) -> Dict[str, float]:
        """ارزیابی عملکرد روی داده‌های validation"""
        if not validation_data:
            # استفاده از آخرین داده‌های آموزشی برای validation ساده
            return {
                'validation_accuracy': self.performance_metrics['average_accuracy'],
                'samples_validated': 0,
                'note': 'Using training metrics as validation'
            }
        
        accuracies = []
        
        for text_data in validation_data:
            try:
                inputs, _ = self.learner.generate_training_pairs(text_data)
                outputs = self.brain.activate(inputs)
                
                # محاسبه دقت ساده (میانگین فعال‌سازی)
                accuracy = np.mean(outputs)
                accuracies.append(accuracy)
                
            except Exception as e:
                print(f"⚠️ Validation sample failed: {e}")
                continue
        
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        return {
            'validation_accuracy': avg_accuracy,
            'samples_validated': len(accuracies),
            'accuracy_std': np.std(accuracies) if accuracies else 0.0,
            'accuracy_range': [np.min(accuracies), np.max(accuracies)] if accuracies else [0, 0]
        }
    
    def adjust_hyperparameters(self) -> Dict[str, Any]:
        """تنظیم خودکار پارامترها بر اساس عملکرد"""
        try:
            # تحلیل تاریخچه آموزش
            recent_performance = self.training_history[-10:] if len(self.training_history) >= 10 else self.training_history
            
            if not recent_performance:
                return {'status': 'no_data', 'message': 'Insufficient training data'}
            
            # محاسبه روند دقت
            recent_accuracies = [session['average_accuracy'] for session in recent_performance]
            accuracy_trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
            
            # تنظیم نرخ یادگیری بر اساس روند
            old_learning_rate = self.brain.learning_rate
            
            if accuracy_trend > 0.01:  # روند بهبودی
                # کاهش نرخ یادگیری برای پایداری
                self.brain.learning_rate = max(0.001, self.brain.learning_rate * 0.9)
            elif accuracy_trend < -0.01:  # روند کاهشی
                # افزایش نرخ یادگیری برای خروج از مینیمم محلی
                self.brain.learning_rate = min(0.1, self.brain.learning_rate * 1.1)
            
            # فعال‌سازی بهینه‌ساز معماری اگر دقت پایین باشد
            current_accuracy = self.performance_metrics['average_accuracy']
            if current_accuracy < 0.6 and len(self.training_history) > 5:
                self.brain.optimize_architecture()
                architecture_optimized = True
            else:
                architecture_optimized = False
            
            return {
                'status': 'adjusted',
                'old_learning_rate': old_learning_rate,
                'new_learning_rate': self.brain.learning_rate,
                'accuracy_trend': accuracy_trend,
                'architecture_optimized': architecture_optimized,
                'current_accuracy': current_accuracy,
                'adjustment_reason': 'accuracy_trend_analysis'
            }
            
        except Exception as e:
            print(f"❌ Hyperparameter adjustment failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_training_report(self) -> Dict[str, Any]:
        """گزارش کامل آموزش"""
        validation_results = self.validate_performance()
        
        return {
            'performance_metrics': self.performance_metrics,
            'validation_results': validation_results,
            'training_history_summary': {
                'total_sessions': len(self.training_history),
                'recent_performance': self.training_history[-5:] if self.training_history else [],
                'performance_trend': 'improving' if len(self.training_history) >= 2 and 
                self.training_history[-1]['average_accuracy'] > self.training_history[0]['average_accuracy'] else 'stable'
            },
            'system_status': {
                'learning_rate': self.brain.learning_rate,
                'network_health': self.brain.get_network_health(),
                'learning_stats': self.learner.get_learning_stats(),
                'memory_stats': self.memory.get_knowledge_base_stats()
            },
            'recommendations': self._generate_training_recommendations()
        }
    
    def _generate_training_recommendations(self) -> List[str]:
        """تولید توصیه‌های آموزشی"""
        recommendations = []
        
        current_accuracy = self.performance_metrics['average_accuracy']
        total_samples = self.performance_metrics['total_training_samples']
        
        if current_accuracy < 0.5:
            recommendations.append("🔴 دقت پایین است - حجم داده‌های آموزشی را افزایش دهید")
        elif current_accuracy < 0.7:
            recommendations.append("🟡 دقت متوسط است - تنوع داده‌ها را افزایش دهید")
        
        if total_samples < 100:
            recommendations.append("📊 داده‌های آموزشی کم هستند - نمونه‌های بیشتری اضافه کنید")
        
        if self.brain.learning_rate > 0.05:
            recommendations.append("🎯 نرخ یادگیری بالا است - ممکن است ناپایدار باشد")
        
        if not recommendations:
            recommendations.append("✅ سیستم در وضعیت مطلوب قرار دارد")
        
        return recommendations

# تابع ایجاد نمونه
def create_training_manager(brain, learner, memory):
    """ایجاد نمونه Training Manager"""
    return AITrainingManager(brain, learner, memory)

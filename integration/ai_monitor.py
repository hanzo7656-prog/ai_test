import time
from datetime import datetime
from typing import Dict, Any

# ایمپورت از سیستم موجود
try:
    from debug_system.core.metrics_collector import metrics_collector
    from debug_system.utils.logger import get_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from debug_system.core.metrics_collector import metrics_collector
    from debug_system.utils.logger import get_logger

class AIMonitor:
    """مانیتورینگ هوش مصنوعی - متصل به سیستم مادر"""
    
    def __init__(self):
        self.logger = get_logger("ai_monitor")
        self.performance_history = []
        
    def collect_ai_metrics(self) -> Dict[str, Any]:
        """جمع‌آوری متریک‌های هوش مصنوعی"""
        try:
            from simple_ai.brain import ai_brain
            
            health_data = ai_brain.get_network_health()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'component': 'ai_brain',
                'resource_usage': {
                    'memory_mb': health_data['memory_usage_mb'],
                    'active_neurons': health_data['active_neurons'],
                    'training_samples': health_data['performance']['training_samples']
                },
                'performance': {
                    'current_accuracy': health_data['performance']['current_accuracy'],
                    'trend_accuracy': health_data['performance']['accuracy_trend_10'],
                    'learning_rate': ai_brain.learning_rate
                },
                'architecture_health': {
                    'connection_sparsity': health_data['actual_sparsity'],
                    'weight_stability': health_data['average_weight'],
                    'bias_balance': health_data['bias_range']['mean']
                }
            }
            
            # ذخیره در تاریخچه
            self.performance_history.append(metrics)
            if len(self.performance_history) > 1000:
                self.performance_history.pop(0)
            
            self.logger.info(f"AI metrics collected - Accuracy: {metrics['performance']['current_accuracy']:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting AI metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'component': 'ai_brain',
                'error': str(e),
                'status': 'unhealthy'
            }
    
    def get_ai_health_report(self) -> Dict[str, Any]:
        """گزارش سلامت هوش مصنوعی برای سیستم مادر"""
        ai_metrics = self.collect_ai_metrics()
        system_metrics = metrics_collector.get_current_metrics()
        
        # ارزیابی سلامت کلی
        accuracy = ai_metrics.get('performance', {}).get('current_accuracy', 0)
        memory_usage = ai_metrics.get('resource_usage', {}).get('memory_mb', 0)
        
        health_status = "healthy"
        if accuracy < 0.5:
            health_status = "degraded"
        elif memory_usage > 50:  # 50MB
            health_status = "high_memory"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'component': 'ai_brain',
            'status': health_status,
            'ai_metrics': ai_metrics,
            'system_integration': {
                'cpu_usage': system_metrics.get('cpu', {}).get('percent', 0),
                'memory_usage': system_metrics.get('memory', {}).get('percent', 0),
                'normalization_health': system_metrics.get('data_normalization', {}).get('success_rate', 0)
            },
            'recommendations': self._generate_recommendations(ai_metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """تولید توصیه‌های بهینه‌سازی"""
        recommendations = []
        
        accuracy = metrics.get('performance', {}).get('current_accuracy', 0)
        memory_usage = metrics.get('resource_usage', {}).get('memory_mb', 0)
        
        if accuracy < 0.6:
            recommendations.append("Consider increasing training data diversity")
        if accuracy > 0.9:
            recommendations.append("High accuracy - consider reducing learning rate")
        if memory_usage > 30:
            recommendations.append("Monitor memory usage - consider optimization")
        
        return recommendations

# نمونه گلوبال
ai_monitor = AIMonitor()

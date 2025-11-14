# ml_core/model_manager.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MLModelManager:
    """مدیریت متمرکز مدل‌های هوش مصنوعی"""
    
    def __init__(self):
        self.active_models = {}
        self.model_versions = {}
        self.performance_metrics = {}
        
        # ارتباط با سیستم کش موجود
        from debug_system.storage.cache_debugger import cache_debugger
        self.cache_manager = cache_debugger
        
        logger.info("🧠 ML Model Manager initialized")

    def register_model(self, model_name: str, model: nn.Module, version: str = "1.0.0"):
        """ثبت مدل جدید در سیستم"""
        self.active_models[model_name] = {
            'model': model,
            'version': version,
            'created_at': datetime.now(),
            'last_used': datetime.now(),
            'performance': {}
        }
        
        # ذخیره metadata در کش UTA
        model_metadata = {
            'name': model_name,
            'version': version,
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': str(model.__class__.__name__),
            'registered_at': datetime.now().isoformat()
        }
        
        self.cache_manager.set_data("uta", f"model_meta:{model_name}", model_metadata, expire=86400)
        logger.info(f"✅ Model registered: {model_name} v{version}")

    async def predict(self, model_name: str, input_data: torch.Tensor) -> Dict[str, Any]:
        """انجام پیش‌بینی با مدل مشخص"""
        if model_name not in self.active_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.active_models[model_name]
        model = model_info['model']
        
        # بررسی کش برای نتیجه مشابه
        cache_key = f"prediction:{model_name}:{self._tensor_hash(input_data)}"
        cached_result = self.cache_manager.get_data("uta", cache_key)
        
        if cached_result is not None:
            logger.info(f"✅ Prediction cache HIT for {model_name}")
            return cached_result
        
        # اجرای پیش‌بینی
        model.eval()
        with torch.no_grad():
            start_time = datetime.now()
            output = model(input_data)
            inference_time = (datetime.now() - start_time).total_seconds()
        
        # پردازش خروجی
        result = self._process_model_output(model_name, output)
        result['inference_time'] = inference_time
        result['model_version'] = model_info['version']
        result['timestamp'] = datetime.now().isoformat()
        
        # ذخیره در کش
        self.cache_manager.set_data("uta", cache_key, result, expire=300)
        
        # به‌روزرسانی آمار
        self._update_performance_metrics(model_name, inference_time, True)
        
        return result

    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """ایجاد هش از تانسور برای کلید کش"""
        import hashlib
        tensor_str = str(tensor.shape) + str(tensor.sum().item())
        return hashlib.md5(tensor_str.encode()).hexdigest()

    def _process_model_output(self, model_name: str, output: Any) -> Dict[str, Any]:
        """پردازش خروجی مدل بر اساس نوع"""
        if isinstance(output, dict):
            return output
        elif isinstance(output, torch.Tensor):
            return {
                'predictions': output.cpu().numpy().tolist(),
                'confidence': torch.max(torch.softmax(output, dim=-1)).item()
            }
        else:
            return {'raw_output': str(output)}

    def _update_performance_metrics(self, model_name: str, inference_time: float, success: bool):
        """به‌روزرسانی متریک‌های عملکرد"""
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = {
                'total_predictions': 0,
                'successful_predictions': 0,
                'total_inference_time': 0,
                'average_inference_time': 0
            }
        
        metrics = self.performance_metrics[model_name]
        metrics['total_predictions'] += 1
        metrics['total_inference_time'] += inference_time
        metrics['average_inference_time'] = metrics['total_inference_time'] / metrics['total_predictions']
        
        if success:
            metrics['successful_predictions'] += 1
        
        # ذخیره در کش برای مانیتورینگ
        self.cache_manager.set_data("uta", f"metrics:{model_name}", metrics, expire=3600)

    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """دریافت وضعیت سلامت مدل"""
        if model_name not in self.active_models:
            return {'status': 'not_found', 'health': 'unknown'}
        
        model_info = self.active_models[model_name]
        metrics = self.performance_metrics.get(model_name, {})
        
        return {
            'status': 'active',
            'health': 'healthy' if metrics.get('success_rate', 1) > 0.95 else 'degraded',
            'version': model_info['version'],
            'uptime': (datetime.now() - model_info['created_at']).total_seconds(),
            'performance_metrics': metrics,
            'last_used': model_info['last_used'].isoformat()
        }

    async def batch_predict(self, model_name: str, batch_data: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """پیش‌بینی دسته‌ای"""
        results = []
        for data in batch_data:
            result = await self.predict(model_name, data)
            results.append(result)
        return results

# نمونه global برای استفاده در سراسر سیستم
ml_model_manager = MLModelManager()

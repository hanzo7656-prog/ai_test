# 📁 src/core/optimization/memory_optimizer.py

import gc
import psutil
import logging
from typing import Dict, List
import torch

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """بهینه‌ساز حافظه برای مدل‌های اسپارس"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.memory_history: List[Dict] = []
    
    def optimize_model_memory(self, model):
        """بهینه‌سازی حافظه مدل"""
        try:
            # فشرده‌سازی وزن‌ها
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    if param.data.is_sparse:
                        param.data = param.data.coalesce()
            
            # پاک‌سازی cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # جمع‌آوری زباله
            gc.collect()
            
            logger.info("✅ Model memory optimized")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    def get_memory_usage(self) -> Dict:
        """دریافت استفاده حافظه"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def should_cleanup(self) -> bool:
        """بررسی نیاز به پاک‌سازی حافظه"""
        memory_usage = self.get_memory_usage()
        return memory_usage['rss_mb'] > self.max_memory_mb * 0.8  # 80% از حداکثر
